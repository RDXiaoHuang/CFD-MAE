import os
import math
import time
import copy
import random
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False


def wh2xy(x):
    assert x.shape[-1] == 4, f"expected 4 but input shape is {x.shape}"
    if isinstance(x, torch.Tensor):
        y = torch.empty_like(x, dtype=torch.float32)
    else:
        y = np.empty_like(x, dtype=np.float32)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def make_anchors(feats, strides, offset=0.5):
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
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    x = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
    y = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    inter = x * y
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    a = (b2_x1 + b2_x2 - b1_x1 - b1_x2)
    b = (b2_y1 + b2_y2 - b1_y1 - b1_y2)
    rho2 = (a.pow(2) + b.pow(2)) / 4

    v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def non_max_suppression(pred, conf_th=0.001, iou_th=0.7):
    import torchvision
    max_det = 300
    max_wh = 7680
    max_nms = 30000

    pred = pred[0] if isinstance(pred, (list, tuple)) else pred

    bs = pred.shape[0]  # batch size
    nc = pred.shape[1] - 4  # number of classes
    xc = pred[:, 4:(4 + nc)].amax(1) > conf_th

    start_time = time.time()
    time_limit = 2.0 + 0.05 * bs

    pred = pred.transpose(-1, -2)
    pred[..., :4] = wh2xy(pred[..., :4])

    output = [torch.zeros((0, 6), device=pred.device)] * bs
    for xi, x in enumerate(pred):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)

        if nc > 1:
            i, j = torch.where(cls > conf_th)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_th]

        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]

        idx = torchvision.ops.nms(boxes, scores, iou_th)  # NMS
        idx = idx[:max_det]

        output[xi] = x[idx]
        if (time.time() - start_time) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    return output


class Colors:
    def __init__(self):
        hexs = (
            "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68", "FF6FDD",
            "FF444F",
            "CCED00", "00F344", "BD00FF", "00B4FF", "DD00BA", "00FFFF",
            "26C000",
            "01FFB3", "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B"
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
            [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
            [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
            [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
            [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
        ], dtype=np.uint8)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return c[::-1] if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def draw_box(im, box, index, label=""):
    import cv2
    color = Colors()(index, True)
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    if label:
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        h += 3
        outside = y1 < h + 3
        if x1 + w > im.shape[1]:
            x1 = im.shape[1] - w
        y2_label = y1 + h if outside else y1 - h
        cv2.rectangle(im, (x1, y1), (x1 + w, y2_label), color, -1)

        # Draw corner markers
        corners = [
            ((x1, y1), (x1 + 15, y1), (x1, y1 + 15)),  # Top-left
            ((x2, y2), (x2 - 15, y2), (x2, y2 - 15)),  # Bottom-right
            ((x2, y1), (x2 - 15, y1), (x2, y1 + 15)),  # Top-right
            ((x1, y2), (x1, y2 - 15), (x1 + 15, y2)),  # Bottom-left
        ]
        for center, *lines in corners:
            for pt in lines:
                cv2.line(im, center, pt, (0, 255, 255), 3)

        cv2.putText(im, label, (x1, y1 + h - 3 if outside else y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return im


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

    def decode_box(self, inputs):
        return inputs

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                    np.maximum(inter_rect_y2 - inter_rect_y1, 0)
                    
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        iou = inter_area / np.maximum(b1_area + b2_area - inter_area, 1e-6)
        return iou

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        return non_max_suppression(prediction, conf_thres, nms_thres)

class DecodeBoxNP():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBoxNP, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def decode_box(self, inputs):
        return inputs
    
    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                    np.maximum(inter_rect_y2 - inter_rect_y1, 0)
                    
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        iou = inter_area / np.maximum(b1_area + b2_area - inter_area, 1e-6)
        return iou

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        return non_max_suppression(prediction, conf_thres, nms_thres)
