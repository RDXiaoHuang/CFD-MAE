import math
import torch
import torch.nn as nn
from .backbone import Backbone
from .head import Detect
from .neck import Neck
from .Common import fuse_conv, Conv


def initialize_weights(model):
    """Initialize model weights."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


class YOLO(torch.nn.Module):
    def __init__(self, num_cls, width, depth, csp, pretrained=None):
        super().__init__()
        self.backbone = Backbone(width, depth, csp, pretrained=pretrained)
        self.neck = Neck(width, depth, csp, pretrained=pretrained)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.detect = Detect(num_cls, (width[3], width[4], width[5]))
        # Initialize stride after forward pass
        with torch.no_grad():
            backbone_out = self.backbone(img_dummy)
            neck_out = self.neck(backbone_out)
            self.detect.stride = torch.tensor(
                [256 / x.shape[-2] for x in neck_out], 
                dtype=torch.float32)
        self.stride = self.detect.stride
        self.detect.bias_init()
        initialize_weights(self)

    def forward(self, x, return_features=False):
        backbone_out = self.backbone(x)
        p3, p4, p5 = backbone_out

        x_neck = self.neck((p3, p4, p5))
        predictions = self.detect(list(x_neck))

        if return_features:
            return predictions, x_neck
        return predictions

    def forward_from_features(self, neck_features):
        """Forward from neck features (for feature forwarding)"""
        predictions = self.detect(list(neck_features))
        return predictions

    def fuse(self):
        """Fuse Conv and BatchNorm layers for faster inference."""
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.forward_fuse
                delattr(m, 'norm')
        return self


def yolo_v11_n(num_cls=80, pretrained=None):
    """YOLO v11 nano model."""
    csp = [False, True]
    depth = [1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(num_cls, width, depth, csp, pretrained=pretrained)


def yolo_v11_s(num_cls=80, pretrained=None):
    """YOLO v11 small model."""
    csp = [False, True]
    depth = [1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(num_cls, width, depth, csp, pretrained=pretrained)


def yolo_v11_m(num_cls=80, pretrained=None):
    """YOLO v11 medium model."""
    csp = [True, True]
    depth = [1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(num_cls, width, depth, csp, pretrained=pretrained)


def yolo_v11_l(num_cls=80, pretrained=None):
    """YOLO v11 large model."""
    csp = [True, True]
    depth = [2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(num_cls, width, depth, csp, pretrained=pretrained)


def yolo_v11_x(num_cls=80, pretrained=None):
    """YOLO v11 xlarge model."""
    csp = [True, True]
    depth = [2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(num_cls, width, depth, csp, pretrained=pretrained)
