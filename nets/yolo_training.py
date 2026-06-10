import math
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def wh2xy(x):
    """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
    assert x.shape[-1] == 4, f"expected 4 but input shape is {x.shape}"
    if isinstance(x, torch.Tensor):
        y = torch.empty_like(x, dtype=torch.float32)
    else:
        y = torch.empty_like(x, dtype=torch.float32)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y

def make_anchors(feats, strides, offset=0.5):
    """Generate anchor points from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (
            int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def bbox_iou(box1, box2, eps=1e-7):
    """Calculate standard IoU between two boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    x = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
    y = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    inter = x * y
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    return iou


def bbox_ciou(box1, box2, eps=1e-7):
    """Calculate Complete IoU (CIoU) for adverse weather detection.
    
    CIoU considers:
    - IoU: Intersection over Union
    - Distance: Center point distance penalty
    - Aspect ratio: Shape consistency penalty
    
    Better than standard IoU for blurry/occluded objects in bad weather.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    
    # Standard IoU
    x = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
    y = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    inter = x * y
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # CIoU penalty terms
    # 1. Center distance penalty
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw.pow(2) + ch.pow(2) + eps  # diagonal length squared
    a = (b2_x1 + b2_x2 - b1_x1 - b1_x2)
    b = (b2_y1 + b2_y2 - b1_y1 - b1_y2)
    rho2 = (a.pow(2) + b.pow(2)) / 4  # center distance squared

    # 2. Aspect ratio penalty
    v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    
    # CIoU = IoU - (center_distance_penalty + aspect_ratio_penalty)
    return iou - (rho2 / c2 + v * alpha)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in adverse weather.
    
    Reduces the loss contribution from easy negatives (clear background)
    and focuses training on hard positives (blurry/occluded objects).
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative samples
        gamma: Focusing parameter >= 0. Higher gamma = more focus on hard samples
               gamma=0 -> standard BCE, gamma=2 is typical
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (before sigmoid), shape: [N, ...]
            target: Ground truth labels (0 or 1), shape: [N, ...]
        """
        # Standard BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Compute probability
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        # Focal weight: (1 - p_t)^gamma
        # For well-classified samples (p_t close to 1), weight is small
        # For misclassified samples (p_t close to 0), weight is large
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting for class balance
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Final focal loss
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DFLoss(nn.Module):
    """Distribution Focal Loss."""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl, tr = target.long(), target.long() + 1
        wl, wr = tr - target, 1 - (tr - target)
        loss_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none")
        loss_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none")
        loss = (loss_l.view(tl.shape) * wl + loss_r.view(tl.shape) * wr)
        return loss.mean(-1, keepdim=True)

class BoxLoss(nn.Module):
    """Box loss for YOLO v11 with CIoU for adverse weather."""
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max)

    def forward(self, p_dist, p_box, anchors, gt_box, scores, scores_sum, mask):
        reg_max = self.dfl_loss.reg_max
        weight = scores.sum(-1)[mask].unsqueeze(-1)
        # Use CIoU instead of standard IoU for better handling of blurry boundaries
        ciou = bbox_ciou(p_box[mask], gt_box[mask])
        loss_box = ((1.0 - ciou) * weight).sum() / scores_sum

        a, b = gt_box.chunk(2, -1)
        distance = torch.cat((anchors - a, b - anchors), -1)
        target = distance.clamp_(0, (reg_max - 1) - 0.01)
        pred = p_dist[mask].view(-1, reg_max)
        loss_dfl = self.dfl_loss(pred, target[mask])
        loss_dfl = (loss_dfl * weight).sum() / scores_sum

        return loss_box, loss_dfl


class Assigner(nn.Module):
    """Task-aligned assigner for YOLO v11."""
    def __init__(self, top_k=10, nc=80, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.nc = nc
        self.eps = eps
        self.beta = beta
        self.top_k = top_k
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, score, p_box, anchors, gt_labels, gt_box, mask):
        bs = score.shape[0]
        na = p_box.shape[-2]
        n_max_boxes = gt_box.shape[1]

        if n_max_boxes == 0:
            return (
                torch.full_like(score[..., 0], self.nc),
                torch.zeros_like(p_box),
                torch.zeros_like(score),
                torch.zeros_like(score[..., 0]),
                torch.zeros_like(score[..., 0]))

        lt, rb = gt_box.view(-1, 1, 4).chunk(2, 2)
        box_delta = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = box_delta.view(gt_box.shape[0], gt_box.shape[1],
                                     anchors.shape[0], -1)
        mask_in_gts = mask_in_gts.amin(3).gt_(1e-9)
        mask_gts = (mask_in_gts * mask).bool()
        overlaps = torch.zeros([bs, n_max_boxes, na], dtype=p_box.dtype,
                               device=p_box.device)
        bbox_scores = torch.zeros([bs, n_max_boxes, na], dtype=score.dtype,
                                  device=score.device)

        ind = torch.zeros([2, bs, n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=bs).view(-1, 1).expand(-1, n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gts] = score[ind[0], :, ind[1]][mask_gts]
        pd_boxes = p_box.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask_gts]
        gt_boxes = gt_box.unsqueeze(2).expand(-1, -1, na, -1)[mask_gts]
        overlaps[mask_gts] = bbox_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        top_mask = mask.expand(-1, -1, self.top_k).bool()
        top_metrics, top_id = torch.topk(metric, self.top_k, dim=-1,
                                         largest=True)
        if top_mask is None:
            top_mask = (top_metrics.max(-1, keepdim=True)[
                            0] > self.eps).expand_as(top_id)
        top_id.masked_fill_(~top_mask, 0)

        count_tensor = torch.zeros(metric.shape, dtype=torch.int8,
                                   device=top_id.device)
        ones = torch.ones_like(top_id[:, :, :1], dtype=torch.int8,
                               device=top_id.device)
        for k in range(self.top_k):
            count_tensor.scatter_add_(-1, top_id[:, :, k: k + 1], ones)

        count_tensor.masked_fill_(count_tensor > 1, 0)
        mask_pos = count_tensor.to(metric.dtype) * mask_in_gts * mask
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes,
                                                               -1)

            max_over = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype,
                                   device=mask_pos.device)
            max_over.scatter_(1, overlaps.argmax(1).unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, max_over, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        gt_idx = mask_pos.argmax(-2)

        batch_ind = \
            torch.arange(end=bs, dtype=torch.int64, device=gt_labels.device)[
                ..., None]
        gt_idx = gt_idx + batch_ind * n_max_boxes
        target_labels = gt_labels.long().flatten()[gt_idx]

        target_bboxes = gt_box.view(-1, gt_box.shape[-1])[gt_idx]
        target_labels.clamp_(0)
        sc = (target_labels.shape[0], target_labels.shape[1], self.nc)
        target_scores = torch.zeros(sc, dtype=torch.int64,
                                    device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(scores_mask > 0, target_scores, 0)

        # Normalize
        metric *= mask_pos
        pos_metrics = metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_metric = (metric * pos_overlaps / (pos_metrics + self.eps))
        target_scores = target_scores * (norm_metric.amax(-2).unsqueeze(-1))
        return target_bboxes, target_scores, fg_mask.bool()


class DetectionLoss(nn.Module):
    """Detection loss for YOLO v11 (anchor-free) with Focal Loss for adverse weather."""
    def __init__(self, model, num_classes):
        super().__init__()
        device = next(model.parameters()).device

        m = model.detect
        self.nc = num_classes
        self.device = device
        self.stride = m.stride.to(device)
        self.reg_max = m.reg_max
        self.no = m.nc + m.reg_max * 4

        self.assigner = Assigner(nc=self.nc)
        self.bbox_loss = BoxLoss(m.reg_max).to(device)
        # Replace BCE with Focal Loss for better handling of hard samples in bad weather
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, gt, bs, scale):
        """Preprocess ground truth."""
        device = gt.device
        nl, ne = gt.shape
        if nl == 0:
            out = torch.zeros(bs, 0, ne - 1, device=device)
        else:
            i = gt[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(bs, counts.max(), ne - 1, device=device)
            for j in range(bs):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = gt[matches, 1:]
            # Input format is already [x1, y1, x2, y2], no conversion needed
            # Scale if scale is not 1
            if scale.max() > 1.1 or scale.min() < 0.9:
                out[..., 1:5] = out[..., 1:5].mul_(scale)
        return out

    def bbox_decode(self, anchor, pred_dist):
        """Decode bounding boxes from predictions."""
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        # Ensure proj is on the same device as pred_dist
        proj = self.proj.to(pred_dist.device).type(pred_dist.dtype)
        pred_dist = pred_dist.matmul(proj)
        lt, rb = pred_dist.chunk(2, -1)
        x1y1, x2y2 = anchor - lt, anchor + rb
        return torch.cat((x1y1, x2y2), -1)

    def forward(self, pred, targets, input_shape):
        """Forward pass for loss computation.
        
        Args:
            pred: Model predictions (list of tensors or tuple)
            targets: Ground truth targets (list of tensors, format: [batch_idx, class, x, y, w, h])
            input_shape: Input image shape [H, W]
        """
        loss = torch.zeros(3, device=self.device)
        # Handle both training mode (list of feature maps) and eval mode (tuple of (output, feature_maps))
        # Training mode: pred = list of 3 feature maps, each [B, no, H, W]
        # Eval mode: pred = (output_tensor, list of 3 feature maps)
        if isinstance(pred, tuple) and len(pred) == 2:
            # Check if second element is a list (eval mode returns (output, [feat1, feat2, feat3]))
            if isinstance(pred[1], (list, tuple)):
                feats = list(pred[1])
            else:
                # Fallback: treat as list of tensors
                feats = list(pred)
        elif isinstance(pred, (list, tuple)):
            # Training mode: pred is a list of feature maps
            feats = list(pred)
        else:
            raise ValueError(f"Unexpected pred type: {type(pred)}")
        
        # Concatenate all feature maps
        x = torch.cat([f.view(feats[0].shape[0], self.no, -1) for f in feats], 2)
        pred_distri, pred_scores = x.split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype, bs = pred_scores.dtype, pred_scores.shape[0]
        # Ensure stride is on the same device as predictions
        stride = self.stride.to(pred_scores.device)
        img_size = torch.tensor(input_shape[::-1], device=pred_scores.device, dtype=dtype)  # [W, H]
        img_size = img_size * stride[0]
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        # Process targets: convert from list format to tensor format
        pred_device = pred_scores.device
        if isinstance(targets, list):
            # Convert list of targets to tensor format
            all_targets = []
            for batch_idx, target in enumerate(targets):
                if target.numel() > 0 and target.shape[0] > 0:
                    # Ensure target is on the same device as predictions
                    target = target.to(pred_device)
                    # target shape: [N, 5] where N is number of objects, format: [class, x, y, w, h]
                    batch_indices = torch.full((target.shape[0], 1), batch_idx, 
                                              dtype=target.dtype, device=pred_device)
                    # Concatenate: [batch_idx, class, x, y, w, h]
                    all_targets.append(torch.cat([batch_indices, target], dim=1))
            
            if len(all_targets) > 0:
                targets_tensor = torch.cat(all_targets, dim=0)
            else:
                targets_tensor = torch.zeros(0, 6, device=pred_device)  # [idx, cls, x, y, w, h]
        else:
            targets_tensor = targets.to(pred_device) if isinstance(targets, torch.Tensor) else targets

        if targets_tensor.numel() == 0:
            # No targets: return in-graph zero so backward() works
            return pred_scores.sum() * 0, loss.detach()

        idx, cls, box = targets_tensor[:, 0:1], targets_tensor[:, 1:2], targets_tensor[:, 2:6]

        # IMPORTANT: Coordinates arriving here are ALREADY in absolute pixels [0, 416]
        # They were converted in utils_fit.py from normalized to absolute
        # So we DON'T need to scale them again!
        
        # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
        # box is already in absolute pixel coordinates
        box_xyxy = torch.zeros_like(box)
        box_xyxy[:, 0] = box[:, 0] - box[:, 2] / 2  # x1 = x_center - w/2
        box_xyxy[:, 1] = box[:, 1] - box[:, 3] / 2  # y1 = y_center - h/2
        box_xyxy[:, 2] = box[:, 0] + box[:, 2] / 2  # x2 = x_center + w/2
        box_xyxy[:, 3] = box[:, 1] + box[:, 3] / 2  # y2 = y_center + h/2
        
        targets_processed = torch.cat((idx, cls, box_xyxy), 1).to(pred_device)
        # No additional scaling needed, coordinates are in absolute pixels
        targets_processed = self.preprocess(targets_processed, bs, torch.ones(4, device=pred_device))

        gt_labels, gt_bboxes = targets_processed.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        scores_sum = target_scores.sum()
        fg_sum = fg_mask.sum()
        
        # Use Focal Loss instead of BCE for classification
        _loss = self.focal_loss(pred_scores, target_scores.to(dtype))
        if scores_sum > 0:
            loss[1] = _loss.sum() / scores_sum
        else:
            # If no positive samples, use average loss of all predictions (with small weight)
            loss[1] = _loss.mean() * 0.01  # Reduce weight to avoid excessive loss when no targets

        # Bbox loss
        if fg_mask.sum() > 0:
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri,
                                              pred_bboxes,
                                              anchor_points,
                                              target_bboxes,
                                              target_scores,
                                              max(scores_sum, 1),
                                              fg_mask)
        else:
            loss[0] = pred_distri.sum() * 0
            loss[2] = pred_distri.sum() * 0

        loss[0] *= 5.0  # bbox loss
        loss[1] *= 5.0  # cls loss
        loss[2] *= 1.5  # dfl loss
        
        total_loss = loss.sum() * bs
        return total_loss, loss.detach()



class DetectionLossYOLO26(nn.Module):
    """
    Detection loss for YOLO26n (reg_max=1, no DFL).
    
    YOLO26n uses direct box regression instead of Distribution Focal Loss,
    so we use a simplified loss computation.
    """
    def __init__(self, model, num_classes):
        super().__init__()
        device = next(model.parameters()).device

        m = model.detect
        self.nc = num_classes
        self.device = device
        self.stride = m.stride.to(device) if hasattr(m.stride, 'to') else torch.tensor([8., 16., 32.], device=device)
        self.reg_max = m.reg_max  # Should be 1 for YOLO26n, 16 for YOLO11n
        self.no = m.nc + m.reg_max * 4

        # DFL projection for reg_max > 1
        if self.reg_max > 1:
            self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        else:
            self.proj = None

        self.assigner = Assigner(nc=self.nc)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')

    def preprocess(self, gt, bs, scale):
        """Preprocess ground truth."""
        device = gt.device
        nl, ne = gt.shape
        if nl == 0:
            out = torch.zeros(bs, 0, ne - 1, device=device)
        else:
            i = gt[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(bs, counts.max(), ne - 1, device=device)
            for j in range(bs):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = gt[matches, 1:]
            if scale.max() > 1.1 or scale.min() < 0.9:
                out[..., 1:5] = out[..., 1:5].mul_(scale)
        return out

    def bbox_decode(self, anchor, pred_dist):
        """
        Decode bounding boxes from predictions.
        
        For YOLO26n with reg_max=1: pred_dist is [B, num_anchors, 4] (direct ltrb offsets)
        For YOLO11n with reg_max=16: pred_dist is [B, num_anchors, 64] (DFL distribution)
        """
        if self.reg_max > 1:
            # DFL (Distribution Focal Loss) decoding for YOLO11n
            # pred_dist shape: [B, num_anchors, reg_max*4]
            b, a, c = pred_dist.shape
            # Reshape to [B, num_anchors, 4, reg_max]
            pred_dist = pred_dist.view(b, a, 4, c // 4)
            # Apply softmax and weighted sum
            # Move proj to same device as pred_dist
            proj = self.proj.to(pred_dist.device)
            pred_dist = pred_dist.softmax(3).matmul(proj.type(pred_dist.dtype))
            # Now pred_dist is [B, num_anchors, 4]
        
        # pred_dist shape: [B, num_anchors, 4] (ltrb offsets)
        lt, rb = pred_dist.chunk(2, -1)
        x1y1, x2y2 = anchor.unsqueeze(0) - lt, anchor.unsqueeze(0) + rb
        return torch.cat((x1y1, x2y2), -1)

    def forward(self, pred, targets, input_shape):
        """Forward pass for loss computation."""
        loss = torch.zeros(3, device=self.device)
        
        # Handle prediction format
        # Model returns [B, no, total_anchors] tensor (DataParallel safe)
        if isinstance(pred, torch.Tensor) and pred.dim() == 3:
            x = pred
        elif isinstance(pred, tuple) and len(pred) == 2:
            if isinstance(pred[1], (list, tuple)):
                x = torch.cat([f.view(f.shape[0], self.no, -1) for f in pred[1]], 2)
            else:
                x = torch.cat([f.view(f.shape[0], self.no, -1) for f in pred], 2)
        elif isinstance(pred, (list, tuple)):
            x = torch.cat([f.view(f.shape[0], self.no, -1) for f in pred], 2)
        else:
            raise ValueError(f"Unexpected pred type: {type(pred)}")

        pred_distri, pred_scores = x.split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype, bs = pred_scores.dtype, pred_scores.shape[0]
        stride = self.stride.to(pred_scores.device)
        # Build anchors from input_shape and stride directly
        anchor_points_list, stride_tensor_list = [], []
        for s in stride:
            fh, fw = int(input_shape[0] / s), int(input_shape[1] / s)
            sx = torch.arange(fw, device=pred_scores.device, dtype=dtype) + 0.5
            sy = torch.arange(fh, device=pred_scores.device, dtype=dtype) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points_list.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor_list.append(torch.full((fh * fw, 1), s, dtype=dtype, device=pred_scores.device))
        anchor_points = torch.cat(anchor_points_list)
        stride_tensor = torch.cat(stride_tensor_list)

        # Process targets
        pred_device = pred_scores.device
        if isinstance(targets, list):
            all_targets = []
            for batch_idx, target in enumerate(targets):
                if target.numel() > 0 and target.shape[0] > 0:
                    target = target.to(pred_device)
                    batch_indices = torch.full((target.shape[0], 1), batch_idx, 
                                              dtype=target.dtype, device=pred_device)
                    all_targets.append(torch.cat([batch_indices, target], dim=1))
            
            if len(all_targets) > 0:
                targets_tensor = torch.cat(all_targets, dim=0)
            else:
                targets_tensor = torch.zeros(0, 6, device=pred_device)
        else:
            targets_tensor = targets.to(pred_device) if isinstance(targets, torch.Tensor) else targets

        if targets_tensor.numel() == 0:
            return pred_scores.sum() * 0, loss.detach()

        idx, cls, box = targets_tensor[:, 0:1], targets_tensor[:, 1:2], targets_tensor[:, 2:6]

        # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
        box_xyxy = torch.zeros_like(box)
        box_xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
        box_xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
        box_xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
        box_xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
        
        targets_processed = torch.cat((idx, cls, box_xyxy), 1).to(pred_device)
        targets_processed = self.preprocess(targets_processed, bs, torch.ones(4, device=pred_device))

        gt_labels, gt_bboxes = targets_processed.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        scores_sum = target_scores.sum()
        fg_sum = fg_mask.sum()
        
        # Classification loss (Focal Loss)
        _loss = self.focal_loss(pred_scores, target_scores.to(dtype))
        if scores_sum > 0:
            loss[1] = _loss.sum() / scores_sum
        else:
            loss[1] = _loss.mean() * 0.01

        # Bbox loss (CIoU, no DFL)
        if fg_mask.sum() > 0:
            target_bboxes /= stride_tensor
            ciou = bbox_ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            loss[0] = ((1.0 - ciou) * weight).sum() / max(scores_sum, 1)
            loss[2] = pred_distri.sum() * 0  # No DFL loss
        else:
            loss[0] = pred_distri.sum() * 0
            loss[2] = pred_distri.sum() * 0

        loss[0] *= 5.0  # bbox loss
        loss[1] *= 5.0  # cls loss

        return loss.sum() * bs, loss.detach()
