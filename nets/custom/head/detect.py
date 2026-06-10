import math
import torch
import torch.nn as nn
from ..Common import Conv, DWConv, DFL
from utils import util


class Detect(nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.nc = nc
        self.reg_max = 16
        self.nl = len(filters)
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        box = max((filters[0] // 4, 64))
        cls = max(filters[0], min(self.nc, 100))

        self.box = nn.ModuleList(
            nn.Sequential(Conv(x, box, 3), Conv(box, box, 3),
                          nn.Conv2d(box, 4 * self.reg_max, 1)) for x in
            filters)

        self.cls = nn.ModuleList(
            nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, cls, 1)),
                          nn.Sequential(DWConv(cls, cls, 3),
                                        Conv(cls, cls, 1)),
                          nn.Conv2d(cls, self.nc, 1), ) for x in filters)
        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        for i in range(self.nl):
            box_out = self.box[i](x[i])
            cls_out = self.cls[i](x[i])
            x[i] = torch.cat((box_out, cls_out), 1)

        if self.training:
            return x

        bs = x[0].shape
        x_cat = torch.cat([xi.view(bs[0], self.no, -1) for xi in x], 2)
        anchor_points, stride_tensor = util.make_anchors(x, self.stride.tolist())
        self.anchors = anchor_points.transpose(0, 1)
        self.strides = stride_tensor.transpose(0, 1)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        lt, rb = self.dfl(box).chunk(2, 1)
        x1y1 = self.anchors.unsqueeze(0) - lt
        x2y2 = self.anchors.unsqueeze(0) + rb
        c_xy, wh = (x1y1 + x2y2) / 2, x2y2 - x1y1
        d_box = torch.cat((c_xy, wh), 1)

        output = torch.cat((d_box * self.strides, cls.sigmoid()), 1)
        return output, x

    def bias_init(self):
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

