"""
Ultralytics YOLO wrapper for the TSDA-Debias/CFD-MAE training framework.

This module wraps Ultralytics YOLOv10, YOLO11, YOLO12, and YOLO26 models so they
can share the custom training/evaluation code used in this repository.

Usage:
    from nets.ultralytics.yolo11_wrapper import YOLO11Wrapper, yolo_v11_ultralytics

    # Create model (auto-detects variant from pretrained path)
    model = yolo_v11_ultralytics(num_cls=5, pretrained='model_data/yolo11n.pt')
    model = yolo_v11_ultralytics(num_cls=5, pretrained='model_data/yolo12n.pt')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import re


def load_ultralytics_model(pretrained_path):
    """Load Ultralytics model and extract the pure PyTorch model."""
    from ultralytics import YOLO as UltralyticsYOLO
    
    ultralytics_model = UltralyticsYOLO(pretrained_path)
    pytorch_model = copy.deepcopy(ultralytics_model.model)
    
    return pytorch_model


class YOLO11Wrapper(nn.Module):
    """
    Wrapper class that adapts Ultralytics YOLO detectors to the CFD-MAE framework.

    Supports YOLOv10, YOLO11, YOLO12 and YOLO26 checkpoints with n/s/m/l/x
    variants where the checkpoint is available.
    """

    def __init__(self, num_cls, pretrained=None, variant='n', family=None):
        super().__init__()
        self.num_classes = num_cls
        self.variant = variant
        self.family = family or self._infer_family(pretrained)

        if pretrained:
            print(f"[YOLOUltralyticsWrapper] Loading Ultralytics {self.family}{variant} from {pretrained}")
            self.model = load_ultralytics_model(pretrained)
        else:
            raise ValueError("YOLOUltralyticsWrapper requires pretrained weights path")
        
        # Modify detection head for custom number of classes
        if num_cls != 80:
            self._modify_num_classes(num_cls)
        
        self.stride = self.model.stride if hasattr(self.model, 'stride') else torch.tensor([8., 16., 32.])
        self.detect = self._create_detect_proxy()
        
        # Get neck output indices from Detect layer (these are the inputs to Detect).
        detect_layer = self.model.model[-1]
        self.neck_output_indices = list(detect_layer.f) if hasattr(detect_layer, 'f') else [16, 19, 22]
        self.backbone_feature_indices = self._infer_backbone_feature_indices()

        print(f"[YOLOUltralyticsWrapper] {self.family}{variant} loaded with {num_cls} classes")
        print(f"[YOLOUltralyticsWrapper] Stride: {self.stride}")
        print(f"[YOLOUltralyticsWrapper] reg_max: {self.detect.reg_max}, no: {self.detect.no}")
        print(f"[YOLOUltralyticsWrapper] Backbone feature indices: {self.backbone_feature_indices}")
        print(f"[YOLOUltralyticsWrapper] Neck output indices: {self.neck_output_indices}")

    @staticmethod
    def _infer_family(pretrained):
        name = os.path.basename(pretrained or '').lower()
        if re.search(r'yolov?10', name):
            return 'yolov10'
        if 'yolo12' in name:
            return 'yolo12'
        if 'yolo26' in name:
            return 'yolo26'
        return 'yolo11'

    def _infer_backbone_feature_indices(self):
        # YOLO12 has a shorter neck; layer 8 is the final backbone P5 feature.
        if self.family == 'yolo12':
            return [4, 6, 8]
        return [4, 6, 10]
    
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
        
        Strategy (aligned with Custom PyTorch):
        - Backbone + Neck: Keep pretrained weights (fine-tune during training)
        - Detection Head: Reinitialize from scratch (train from scratch)
        """
        detect = self.model.model[-1]
        
        if hasattr(detect, 'nc'):
            old_nc = detect.nc
            
            if old_nc == num_cls:
                print(f"[YOLOUltralyticsWrapper] Classes already match ({num_cls}), no modification needed")
                return
            
            detect.nc = num_cls
            detect.no = num_cls + detect.reg_max * 4
            
            def reset_box_heads(heads):
                for cv2 in heads:
                    last_conv = cv2[-1]
                    if isinstance(last_conv, nn.Conv2d):
                        in_ch = last_conv.in_channels
                        out_ch = detect.reg_max * 4
                        new_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
                        nn.init.normal_(new_conv.weight.data, mean=0.0, std=0.01)
                        if new_conv.bias is not None:
                            nn.init.constant_(new_conv.bias.data, 0.0)
                        cv2[-1] = new_conv

            def reset_cls_heads(heads):
                for cv3 in heads:
                    last_conv = cv3[-1]
                    if isinstance(last_conv, nn.Conv2d):
                        in_ch = last_conv.in_channels
                        new_conv = nn.Conv2d(in_ch, num_cls, kernel_size=1, stride=1)
                        nn.init.normal_(new_conv.weight.data, mean=0.0, std=0.01)
                        if new_conv.bias is not None:
                            import math
                            target_prob = 0.03 if num_cls < 20 else 0.01
                            bias_init = -math.log((1 - target_prob) / target_prob)
                            nn.init.constant_(new_conv.bias.data, bias_init)
                        cv3[-1] = new_conv

            reset_box_heads(detect.cv2)
            reset_cls_heads(detect.cv3)
            if hasattr(detect, 'one2one_cv2'):
                reset_box_heads(detect.one2one_cv2)
            if hasattr(detect, 'one2one_cv3'):
                reset_cls_heads(detect.one2one_cv3)

            print(f"[YOLOUltralyticsWrapper] Modified classes: {old_nc} -> {num_cls}")
            print(f"[YOLOUltralyticsWrapper] Strategy: Backbone+Neck pretrained, Detection Head from scratch")
            print(f"[YOLOUltralyticsWrapper] - Backbone: Pretrained (fine-tune)")
            print(f"[YOLOUltralyticsWrapper] - Neck: Pretrained (fine-tune)")
            print(f"[YOLOUltralyticsWrapper] - Box Regression (cv2): Reinitialized (train from scratch)")
            print(f"[YOLOUltralyticsWrapper] - Classification (cv3): Reinitialized (train from scratch)")

    def _forward_layers_until_detect(self, x, collect_indices=None):
        y = []
        features = {}
        collect_indices = set(collect_indices or [])
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
            if i in collect_indices:
                features[i] = x

        return x, y, features

    def _get_detect_inputs(self, y, x):
        detect_layer = self.model.model[-1]
        if isinstance(detect_layer.f, (list, tuple)):
            return [y[j] for j in detect_layer.f]
        if detect_layer.f == -1:
            return [x]
        return [y[detect_layer.f]]

    def _head_forward(self, detect_inputs):
        detect_layer = self.model.model[-1]
        feat_maps = []
        for i, xi in enumerate(detect_inputs):
            box_out = detect_layer.cv2[i](xi)
            cls_out = detect_layer.cv3[i](xi)
            feat_maps.append(torch.cat([box_out, cls_out], dim=1))
        return feat_maps

    def forward_backbone_and_neck(self, x):
        """Run backbone + neck and return CFD-MAE DASM taps plus Detect inputs."""
        collect = self.backbone_feature_indices + self.neck_output_indices
        last_x, y, features = self._forward_layers_until_detect(x, collect)
        backbone_feats = [features[idx] for idx in self.backbone_feature_indices]
        neck_feats = [features[idx] for idx in self.neck_output_indices]
        return backbone_feats, neck_feats

    def forward_backbone_neck(self, x):
        """Run backbone + neck only and return P3/P4/P5 neck outputs."""
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
    
    def forward(self, x, return_features=False, raw_output=False):
        """
        Forward pass compatible with TSDA-Debias framework.

        Args:
            x: Input tensor [B, 3, H, W]
            return_features: If True, also return multi-scale features
            raw_output: If True, always return raw feature maps regardless of mode

        Returns:
            In training mode: list of feature maps [B, no, H, W] for each scale
            In eval mode: processed predictions
        """
        x, y, features = self._forward_layers_until_detect(x, self.neck_output_indices)
        feat_maps = self._head_forward(self._get_detect_inputs(y, x))

        if self.training or raw_output:
            # Concat into single tensor [B, no, total_anchors] so DataParallel won't break the format
            predictions = torch.cat([f.view(f.shape[0], f.shape[1], -1) for f in feat_maps], 2)
        else:
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
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)

        x = torch.cat([fm.view(bs, no, -1) for fm in feat_maps], 2)
        box_raw, cls_raw = x.split((reg_max * 4, nc), 1)

        box_raw = box_raw.permute(0, 2, 1).contiguous()
        if reg_max > 1:
            proj = torch.arange(reg_max, dtype=dtype, device=device)
            box_raw = box_raw.view(bs, -1, 4, reg_max).softmax(3).matmul(proj)
        else:
            box_raw = box_raw.view(bs, -1, 4)

        lt, rb = box_raw.chunk(2, -1)
        x1y1 = anchor_points.unsqueeze(0) - lt
        x2y2 = anchor_points.unsqueeze(0) + rb
        boxes = torch.cat((
            (x1y1 + x2y2) / 2,
            x2y2 - x1y1
        ), -1) * stride_tensor.unsqueeze(0)

        cls_scores = cls_raw.sigmoid()

        return torch.cat([boxes.permute(0, 2, 1), cls_scores], dim=1)

    def forward_from_features(self, neck_features, raw_output=False):
        """
        Forward from neck features (for feature forwarding).

        Args:
            neck_features: tuple of 3 feature maps [P3, P4, P5] from neck

        Returns:
            predictions: same format as forward()
        """
        feat_maps = self._head_forward(neck_features)

        if self.training or raw_output:
            predictions = torch.cat([f.view(f.shape[0], f.shape[1], -1) for f in feat_maps], 2)
        else:
            predictions = self._decode_predictions(feat_maps)

        return predictions

    def fuse(self):
        """Fuse Conv and BatchNorm layers."""
        if hasattr(self.model, 'fuse'):
            self.model.fuse()
        return self


def yolo_v11_ultralytics(num_cls=80, pretrained=None, variant='n'):
    """
    Create YOLO11 Ultralytics model compatible with TSDA-Debias framework.

    Args:
        num_cls: Number of classes
        pretrained: Path to Ultralytics .pt file (e.g., 'yolo11n.pt', 'yolo11x.pt')
        variant: Model variant ('n', 's', 'm', 'l', 'x'). Auto-detected from pretrained path if not specified.

    Returns:
        YOLO11Wrapper model
    """
    family = YOLO11Wrapper._infer_family(pretrained)
    if pretrained and variant == 'n':
        match = re.search(r'(?:yolov?10|yolo11|yolo12|yolo26)([nsmbxl])', pretrained.lower())
        if match:
            variant = match.group(1)

    return YOLO11Wrapper(num_cls, pretrained=pretrained, variant=variant, family=family)


def yolo_ultralytics(num_cls=80, pretrained=None, variant='n'):
    """Alias with a neutral name for YOLOv10/11/12/26 baselines."""
    return yolo_v11_ultralytics(num_cls=num_cls, pretrained=pretrained, variant=variant)
