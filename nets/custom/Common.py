import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.init import xavier_uniform_, constant_, kaiming_uniform_


def fuse_conv(conv, norm):
    """Fuse convolution and batch normalization layers."""
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(
        conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(
        norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(
        torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0),
                         device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(
        torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(
        torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


# CBS Block
class Conv(nn.Module):
    def __init__(self, inp, oup, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, self._pad(k, p), d, g, False)
        self.norm = nn.BatchNorm2d(oup)
        self.act = nn.SiLU(inplace=True) if act is True else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @staticmethod
    def _pad(k, p=None):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p


# Bottleneck Block
class Residual(nn.Module):
    def __init__(self, inp, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.conv1 = Conv(inp, int(inp * e), k[0], 1)
        self.conv2 = Conv(int(inp * e), inp, k[1], 1, g=g)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


# C3k Block
class CSPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(2 * (out_ch // 2), out_ch)
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))
 
    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


# C3k2 Block
class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r=2):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r))
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch)

        if not csp:
            self.res_m = torch.nn.ModuleList(
                Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(
                CSPBlock(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


# SPPF Block
class SPP(nn.Module):
    def __init__(self, inp, k=5):
        super().__init__()
        self.conv1 = Conv(inp, inp // 2, 1, 1)
        self.conv2 = Conv(inp // 2 * 4, inp, 1, 1)
        self.m = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.conv2(torch.cat(y, 1))


class Attention(nn.Module):
    def __init__(self, dim, num_head=8):
        super().__init__()
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.key_dim = self.head_dim // 2
        self.scale = self.key_dim ** -0.5
        h = dim + self.key_dim * num_head * 2

        # Convolution for query, key, and value
        self.qkv_conv = Conv(dim, h, 1, act=False)

        # Projection and Positional encoding convolution
        self.proj_conv = Conv(dim, dim, 1, act=False)
        self.pe_conv = Conv(dim, dim, 3, g=dim, act=False)

    def forward(self, x):
        b, ch, h, w = x.shape

        qkv = self.qkv_conv(x)
        qkv = qkv.view(b, self.num_head, self.key_dim * 2 + self.head_dim,
                       h * w)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(b, ch, h, w)

        return self.proj_conv(out + self.pe_conv(v.reshape(b, ch, h, w)))


class PSABlock(nn.Module):
    def __init__(self, inp, num_head=4):
        super().__init__()
        self.att = Attention(inp, num_head)
        self.ffn = nn.Sequential(Conv(inp, inp * 2, 1),
                                 Conv(inp * 2, inp, 1, act=False))

    def forward(self, x):
        x = x + self.att(x)
        return x + self.ffn(x)


# C2PSA Block
class PSA(nn.Module):
    def __init__(self, inp, oup, n=1):
        super().__init__()
        assert inp == oup
        self.conv1 = Conv(inp, 2 * (inp // 2))
        self.conv2 = Conv(2 * (inp // 2), inp)

        self.m = nn.Sequential(
            *(PSABlock(inp // 2, inp // 128) for _ in range(n)))

    def forward(self, x):
        a, b = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat((a, self.m(b)), 1))



class DFL(nn.Module):
    def __init__(self, inp=16):
        super().__init__()
        self.inp = inp
        self.conv = nn.Conv2d(inp, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(inp, dtype=torch.float).view(1, inp, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)

    def forward(self, x):
        b, _, a = x.shape
        out = x.view(b, 4, self.inp, a).transpose(2, 1)
        return self.conv(out.softmax(1)).view(b, 4, a)


class DWConv(Conv):
    def __init__(self, inp, oup, k=1, s=1, d=1, act=True):
        super().__init__(inp, oup, k, s, g=math.gcd(inp, oup), d=d, act=act)
