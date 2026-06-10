"""
YOLO26n Wrapper for TSDA-Debias Framework

This module wraps the Ultralytics YOLO26n model to be compatible with
the custom PyTorch framework used in TSDA-Debias.

Usage:
    from nets.yolo26_wrapper import YOLO26Wrapper, yolo_v26_n

    # Create model
    model = yolo_v26_n(num_cls=5, pretrained='model_data/yolo26n_Ultralytics.pt')

    # Forward pass (compatible with existing framework)
    predictions = model(images)  # inference
    predictions, features = model(images, return_features=True)  # with features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def load_ultralytics_model(pretrained_path):
    """
    Load Ultralytics model and extract the pure PyTorch model.
    This avoids the Ultralytics wrapper's train/eval hooks.
    """
    from ultralytics import YOLO as UltralyticsYOLO

    # Load the model
    ultralytics_model = UltralyticsYOLO(pretrained_path)

    # Extract the pure PyTorch model (DetectionModel)
    pytorch_model = ultralytics_model.model

    # Deep copy to detach from Ultralytics wrapper
    pytorch_model = copy.deepcopy(pytorch_model)

    return pytorch_model


class YOLO26Wrapper(nn.Module):
    """
    Wrapper class that adapts Ultralytics YOLO26n to the TSDA-Debias framework.

    This wrapper:
    1. Loads the Ultralytics model and extracts pure PyTorch model
    2. Provides compatible forward() interface with return_features option
    3. Extracts multi-scale features for CDAPT training
    4. Handles training/inference mode switching properly
    5. Exposes detect layer attributes for DetectionLoss compatibility
    """

    def __init__(self, num_cls, pretrained=None):
        super().__init__()
        self.num_classes = num_cls

        # Load Ultralytics model and extract PyTorch model
        if pretrained:
            print(f"[YOLO26Wrapper] Loading Ultralytics model from {pretrained}")
            self.model = load_ultralytics_model(pretrained)
        else:
            raise ValueError("YOLO26Wrapper requires pretrained weights path")

        # Modify detection head for custom number of classes if needed
        if num_cls != 80:  # COCO has 80 classes
            self._modify_num_classes(num_cls)

        # Store stride information for compatibility
        self.stride = self.model.stride if hasattr(self.model, 'stride') else torch.tensor([8., 16., 32.])

        # Create a detect attribute that exposes necessary properties for DetectionLoss
        # This makes YOLO26Wrapper compatible with DetectionLoss(model, num_classes)
        self.detect = self._create_detect_proxy()

        # Feature extraction indices
        # YOLO26n structure:
        # Backbone: 0-10 (P3@4, P4@6, P5@10)
        # Neck outputs: 16 (64ch), 19 (128ch), 22 (256ch)
        self.backbone_feature_indices = [4, 6, 10]
        self.neck_output_indices = [16, 19, 22]

        print(f"[YOLO26Wrapper] Model loaded with {num_cls} classes")
        print(f"[YOLO26Wrapper] Stride: {self.stride}")
        print(f"[YOLO26Wrapper] reg_max: {self.detect.reg_max}, no: {self.detect.no}")

    def _create_detect_proxy(self):
        """Create a proxy object that exposes detect layer attributes."""
        detect_layer = self.model.model[-1]

        class DetectProxy:
            def __init__(self, detect):
                self.nc = detect.nc
                self.reg_max = detect.reg_max
                self.no = detect.no
                self.stride = detect.stride if hasattr(detect, 'stride') else torch.tensor([8., 16., 32.])

        return DetectProxy(detect_layer)

    def _modify_num_classes(self, num_cls):
        """
        Modify the detection head for custom number of classes.

        Strategy:
        - Backbone + Neck: Keep pretrained weights (fine-tune during training)
        - cv2 (box regression): Keep pretrained weights (output = reg_max*4, class-independent)
        - cv3 (classification): Reinitialize for new class count
        """
        detect = self.model.model[-1]  # Last layer is Detect

        if hasattr(detect, 'nc'):
            old_nc = detect.nc

            if old_nc == num_cls:
                print(f"[YOLO26Wrapper] Classes already match ({num_cls}), no modification needed")
                return

            detect.nc = num_cls
            detect.no = num_cls + detect.reg_max * 4

            # cv2 (box regression heads): output = reg_max * 4, independent of class count.
            # Keep pretrained weights — only reinitialize if output channels actually changed.
            for i, cv2 in enumerate(detect.cv2):
                last_conv = cv2[-1]
                if isinstance(last_conv, nn.Conv2d):
                    expected_out = detect.reg_max * 4
                    if last_conv.out_channels != expected_out:
                        in_ch = last_conv.in_channels
                        new_conv = nn.Conv2d(in_ch, expected_out, kernel_size=1, stride=1)
                        nn.init.normal_(new_conv.weight.data, mean=0.0, std=0.01)
                        if new_conv.bias is not None:
                            nn.init.constant_(new_conv.bias.data, 0.0)
                        cv2[-1] = new_conv
                        print(f"[YOLO26Wrapper] cv2[{i}] reinitialized: {last_conv.out_channels} -> {expected_out}")

            # cv3 (classification heads): output = nc, must reinitialize for new class count.
            for i, cv3 in enumerate(detect.cv3):
                last_conv = cv3[-1]
                if isinstance(last_conv, nn.Conv2d):
                    in_ch = last_conv.in_channels
                    new_conv = nn.Conv2d(in_ch, num_cls, kernel_size=1, stride=1)
                    nn.init.normal_(new_conv.weight.data, mean=0.0, std=0.01)
                    if new_conv.bias is not None:
                        # Initialize bias for target initial probability
                        # For 5 classes: use 3% per class (higher than COCO's ~1.5% for 80 classes)
                        # bias = -log((1-p)/p) where p is target probability
                        import math
                        target_prob = 0.03 if num_cls < 20 else 0.01
                        bias_init = -math.log((1 - target_prob) / target_prob)
                        nn.init.constant_(new_conv.bias.data, bias_init)
                    cv3[-1] = new_conv

            print(f"[YOLO26Wrapper] Modified classes: {old_nc} -> {num_cls}")
            print(f"[YOLO26Wrapper] - Backbone + Neck: Pretrained (fine-tune)")
            print(f"[YOLO26Wrapper] - Box Regression (cv2): Pretrained PRESERVED (reg_max*4={detect.reg_max*4}, unchanged)")
            print(f"[YOLO26Wrapper] - Classification (cv3): Reinitialized ({old_nc} -> {num_cls})")

    def forward(self, x, return_features=False, raw_output=False):
        """
        Forward pass compatible with TSDA-Debias framework.

        Training mode:
            Returns list of per-scale feature maps [B, no, H, W] for loss computation.
        Eval mode:
            Returns [B, 4+nc, num_anchors] with decoded boxes (xywh) and sigmoid cls.
        If return_features=True, also returns (P3, P4, P5) feature tuple.
        If raw_output=True, always returns raw feature maps regardless of mode.
        """
        y = []
        features = {}
        detect_layer = self.model.model[-1]
        num_layers = len(self.model.model)

        # Forward through all layers EXCEPT the Detect layer
        for i, m in enumerate(self.model.model):
            if i == num_layers - 1:
                break

            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]

            x = m(x)
            y.append(x)

            if i in self.neck_output_indices:
                features[i] = x

        # Gather Detect layer inputs
        if isinstance(detect_layer.f, (list, tuple)):
            detect_inputs = [y[j] for j in detect_layer.f]
        elif detect_layer.f == -1:
            detect_inputs = [x]
        else:
            detect_inputs = [y[detect_layer.f]]

        # Manually apply cv2/cv3 to produce per-scale feature maps
        feat_maps = []
        for i, xi in enumerate(detect_inputs):
            box_out = detect_layer.cv2[i](xi)   # [B, reg_max*4, H, W]
            cls_out = detect_layer.cv3[i](xi)   # [B, nc, H, W]
            feat_maps.append(torch.cat([box_out, cls_out], dim=1))  # [B, no, H, W]

        if self.training or raw_output:
            predictions = torch.cat([f.view(f.shape[0], f.shape[1], -1) for f in feat_maps], 2)
        else:
            # Eval: decode boxes and sigmoid cls for inference
            predictions = self._decode_predictions(feat_maps)

        if return_features:
            feat_list = [features.get(idx) for idx in self.neck_output_indices]
            return predictions, tuple(feat_list)

        return predictions

    def _decode_predictions(self, feat_maps):
        """Decode raw feature maps into [B, 4+nc, num_anchors] for inference."""
        device = feat_maps[0].device
        dtype = feat_maps[0].dtype
        bs = feat_maps[0].shape[0]
        no = self.detect.no
        nc = self.detect.nc
        reg_max = self.detect.reg_max
        stride = self.stride.to(device)

        # Generate anchor points
        anchor_points, stride_tensor = [], []
        for i, fm in enumerate(feat_maps):
            h, w = fm.shape[2], fm.shape[3]
            sx = torch.arange(w, device=device, dtype=dtype) + 0.5
            sy = torch.arange(h, device=device, dtype=dtype) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride[i], dtype=dtype, device=device))
        anchor_points = torch.cat(anchor_points)  # [num_anchors, 2]
        stride_tensor = torch.cat(stride_tensor)   # [num_anchors, 1]

        # Concat all scales: [B, no, total_anchors]
        x = torch.cat([fm.view(bs, no, -1) for fm in feat_maps], 2)
        box_raw, cls_raw = x.split((reg_max * 4, nc), 1)

        # Decode boxes
        box_raw = box_raw.permute(0, 2, 1).contiguous()  # [B, anchors, reg_max*4]
        if reg_max > 1:
            proj = torch.arange(reg_max, dtype=dtype, device=device)
            box_raw = box_raw.view(bs, -1, 4, reg_max).softmax(3).matmul(proj)  # [B, anchors, 4]
        else:
            box_raw = box_raw.view(bs, -1, 4)

        # dist2bbox: ltrb -> xywh, scaled by stride
        lt, rb = box_raw.chunk(2, -1)
        x1y1 = anchor_points.unsqueeze(0) - lt
        x2y2 = anchor_points.unsqueeze(0) + rb
        boxes = torch.cat((
            (x1y1 + x2y2) / 2,  # cx, cy
            x2y2 - x1y1          # w, h
        ), -1) * stride_tensor.unsqueeze(0)  # [B, anchors, 4]

        cls_scores = cls_raw.sigmoid()  # [B, nc, anchors]

        # Output: [B, 4+nc, num_anchors]
        return torch.cat([boxes.permute(0, 2, 1), cls_scores], dim=1)

    def forward_backbone_and_neck(self, x):
        """Run backbone + neck and return both backbone taps and neck outputs."""
        y = []
        features = {}
        num_layers = len(self.model.model)

        for i, m in enumerate(self.model.model):
            if i == num_layers - 1:
                break
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if i in self.backbone_feature_indices or i in self.neck_output_indices:
                features[i] = x

        backbone_feats = [features[idx] for idx in self.backbone_feature_indices]  # [C3, C4, C5]
        neck_feats = [features[idx] for idx in self.neck_output_indices]  # [P3, P4, P5]
        return backbone_feats, neck_feats

    def forward_backbone_neck(self, x):
        """Run backbone + neck only, return P3/P4/P5 neck outputs."""
        _, neck_feats = self.forward_backbone_and_neck(x)
        return neck_feats

    def infer_dasm_channels(self, img_size=640, device=None):
        """Infer C3/C4/C5/P4 channels with a cheap dummy forward."""
        training = self.training
        device = device or next(self.parameters()).device
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        with torch.no_grad():
            self.eval()
            backbone_feats, neck_feats = self.forward_backbone_and_neck(dummy)
        if training:
            self.train()
        return {
            'c3_channels': backbone_feats[0].shape[1],
            'c4_channels': backbone_feats[1].shape[1],
            'c5_channels': backbone_feats[2].shape[1],
            'p4_channels': neck_feats[1].shape[1],
        }

    def forward_from_features(self, neck_features, raw_output=False):
        """Forward from neck features (for feature forwarding)."""
        detect_layer = self.model.model[-1]

        feat_maps = []
        for i, xi in enumerate(neck_features):
            box_out = detect_layer.cv2[i](xi)
            cls_out = detect_layer.cv3[i](xi)
            feat_maps.append(torch.cat([box_out, cls_out], dim=1))

        if self.training or raw_output:
            predictions = torch.cat([f.view(f.shape[0], f.shape[1], -1) for f in feat_maps], 2)
        else:
            predictions = self._decode_predictions(feat_maps)

        return predictions

    def fuse(self):
        """Fuse Conv and BatchNorm layers for faster inference."""
        if hasattr(self.model, 'fuse'):
            self.model.fuse()
        return self


def yolo_v26(num_cls=80, pretrained=None, variant='n'):
    """
    Create YOLO26 model compatible with TSDA-Debias framework.

    Args:
        num_cls: Number of classes
        pretrained: Path to Ultralytics .pt file
        variant: Model variant ('n', 's', 'm', 'l', 'x'). Auto-detected from pretrained path.

    Returns:
        YOLO26Wrapper model
    """
    # Auto-detect variant from pretrained path
    if pretrained and variant == 'n':
        import re
        match = re.search(r'yolo26([nsmxl])', pretrained.lower())
        if match:
            variant = match.group(1)

    return YOLO26Wrapper(num_cls, pretrained=pretrained)
