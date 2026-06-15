"""
cfdmae.py: CFD-MAE downstream detection components.
Pretraining components (LaplacianPyramid, FreqMAE, CFDMAE, etc.) are in mask.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from nets.mask import LaplacianPyramid, FreqMAE, CFDMAE


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class PSA(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out


# ==============================================================
# LFGHE: LF-Guided HF Enhancement
# ==============================================================
class LFGHE(nn.Module):
    """LF-Guided HF Enhancement (LFGHE): use LF priors to modulate HF with residual scale-shift conditioning."""
    def __init__(self, in_ch=3, hidden_ch=16, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hf_encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size, padding=padding),
            nn.SiLU(inplace=True),
        )
        self.scale_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size, padding=padding)
        self.shift_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size, padding=padding)
        self.decoder = nn.Conv2d(hidden_ch, in_ch, kernel_size, padding=padding)
        self._init_zero()

    def _init_zero(self):
        """Initialize scale/shift/decoder to zero so modulation starts as identity."""
        for m in (self.scale_conv, self.shift_conv, self.decoder):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, hf_map, lf_map):
        hf_feat = self.hf_encoder(hf_map)
        scale = self.scale_conv(lf_map)
        shift = self.shift_conv(lf_map)
        modulated = hf_feat + (hf_feat * scale + shift)
        return hf_map + self.decoder(modulated)


class LKA(nn.Module):
    """Large Kernel Attention (LKA): depthwise large-kernel attention for long-range spatial context."""
    def __init__(self, channels):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw2 = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        attn = self.dw1(x)
        attn = self.dw2(attn)
        attn = self.bn(self.pw(attn))
        return x * torch.sigmoid(attn)


class SEAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class CBAMAttention(nn.Module):
    """Lightweight channel-spatial attention used as a PSA replacement."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_attn = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_attn = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        x = x * torch.sigmoid(avg_attn + max_attn)
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        return x * self.spatial(torch.cat([avg_map, max_map], dim=1))


class DilatedContext(nn.Module):
    """Dilated depthwise context block used as an LKA replacement."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.block(x))


def build_local_attention(mode, channels):
    if mode == 'psa':
        return PSA(channels, channels)
    if mode == 'cbam':
        return CBAMAttention(channels)
    if mode == 'se':
        return SEAttention(channels)
    if mode == 'none':
        return nn.Identity()
    raise ValueError(f"Unsupported DASM local attention: {mode}")


def build_long_attention(mode, channels):
    if mode == 'lka':
        return LKA(channels)
    if mode == 'dilated':
        return DilatedContext(channels)
    if mode == 'none':
        return nn.Identity()
    raise ValueError(f"Unsupported DASM long attention: {mode}")


class DASM(nn.Module):
    """Degradation-Adaptive Suppression Module (DASM): suppress degradation artifacts in neck features using multi-scale backbone priors."""
    def __init__(self, c3_channels=128, c4_channels=128, c5_channels=256, p4_channels=128,
                 hidden_dim=64, alpha=0.02, min_keep=0.95,
                 local_attention='psa', long_attention='lka'):
        super().__init__()
        hidden_dim = max(hidden_dim, 32)
        if local_attention not in {'psa', 'cbam', 'se', 'none'}:
            raise ValueError(f"local_attention must be psa/cbam/se/none, got: {local_attention}")
        if long_attention not in {'lka', 'dilated', 'none'}:
            raise ValueError(f"long_attention must be lka/dilated/none, got: {long_attention}")
        self.local_attention = local_attention
        self.long_attention = long_attention
        self.min_keep = min_keep
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(0.05))
        self.hf_detail_scale = nn.Parameter(torch.tensor(0.5))
        self.c3_proj = nn.Sequential(
            nn.Conv2d(c3_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.c4_proj = nn.Sequential(
            nn.Conv2d(c4_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.c5_proj = nn.Sequential(
            nn.Conv2d(c5_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.prior_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            build_local_attention(local_attention, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        self.long_range_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.GELU(),
            build_long_attention(long_attention, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        self.out_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(hidden_dim, p4_channels, 1, bias=False),
            nn.BatchNorm2d(p4_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, backbone_feats, p4, hf_prior=None):
        c3, c4, c5 = backbone_feats
        target_size = p4.shape[2:]
        c3_feat = self.c3_proj(F.interpolate(c3, size=target_size, mode='bilinear', align_corners=False))
        c4_feat = self.c4_proj(c4)
        c5_feat = self.c5_proj(F.interpolate(c5, size=target_size, mode='bilinear', align_corners=False))
        prior = self.prior_fuse(torch.cat([c3_feat, c4_feat, c5_feat], dim=1))
        local_feat = self.local_branch(prior)
        long_feat = self.long_range_branch(prior)
        prior = self.out_fuse(torch.cat([local_feat, long_feat], dim=1))
        spatial_prior = self.spatial_head(prior)
        if hf_prior is not None:
            hf_prior = F.interpolate(hf_prior, size=target_size, mode='bilinear', align_corners=False)
            spatial_prior = spatial_prior * (1.0 - self.hf_detail_scale * hf_prior)
        refine = self.refine_head(prior)
        keep_weight = 1.0 - self.alpha * spatial_prior
        keep_weight = torch.clamp(keep_weight, min=self.min_keep, max=1.0)
        return p4 * keep_weight + self.beta * refine


class ASPPGateSuppression(nn.Module):
    """ASPP context gate used as a whole-module replacement for DASM + HF prior."""
    def __init__(self, p4_channels=128, hidden_dim=64, alpha=0.02, min_keep=0.95):
        super().__init__()
        hidden_dim = max(hidden_dim, 32)
        self.min_keep = min_keep
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(0.05))
        self.input_proj = nn.Sequential(
            nn.Conv2d(p4_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, dilation=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=3, dilation=3, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=5, dilation=5, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ),
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 5, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(hidden_dim, p4_channels, 1, bias=False),
            nn.BatchNorm2d(p4_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, p4):
        x = self.input_proj(p4)
        pooled = F.interpolate(self.image_pool(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        context = self.fuse(torch.cat([branch(x) for branch in self.branches] + [pooled], dim=1))
        spatial_gate = self.gate_head(context)
        keep_weight = 1.0 - self.alpha * spatial_gate
        keep_weight = torch.clamp(keep_weight, min=self.min_keep, max=1.0)
        return p4 * keep_weight + self.beta * self.refine_head(context)


class CoordGateSuppression(nn.Module):
    """Coordinate-attention gate used as a whole-module replacement for DASM + HF prior."""
    def __init__(self, p4_channels=128, hidden_dim=64, alpha=0.02, min_keep=0.95):
        super().__init__()
        hidden_dim = max(hidden_dim, 32)
        coord_hidden = max(hidden_dim // 2, 16)
        self.min_keep = min_keep
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(0.05))
        self.input_proj = nn.Sequential(
            nn.Conv2d(p4_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.coord_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, coord_hidden, 1, bias=False),
            nn.BatchNorm2d(coord_hidden),
            nn.SiLU(inplace=True),
        )
        self.attn_h = nn.Conv2d(coord_hidden, hidden_dim, 1)
        self.attn_w = nn.Conv2d(coord_hidden, hidden_dim, 1)
        self.context = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(hidden_dim, p4_channels, 1, bias=False),
            nn.BatchNorm2d(p4_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, p4):
        x = self.input_proj(p4)
        b, c, h, w = x.shape
        pooled_h = x.mean(dim=3, keepdim=True)
        pooled_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        coord = self.coord_fuse(torch.cat([pooled_h, pooled_w], dim=2))
        attn_h, attn_w = torch.split(coord, [h, w], dim=2)
        attn_w = attn_w.permute(0, 1, 3, 2)
        x = x * torch.sigmoid(self.attn_h(attn_h)) * torch.sigmoid(self.attn_w(attn_w))
        context = self.context(x)
        spatial_gate = self.gate_head(context)
        keep_weight = 1.0 - self.alpha * spatial_gate
        keep_weight = torch.clamp(keep_weight, min=self.min_keep, max=1.0)
        return p4 * keep_weight + self.beta * self.refine_head(context)


class LSKGateSuppression(nn.Module):
    """Large selective-kernel gate used as a whole-module replacement for DASM + HF prior."""
    def __init__(self, p4_channels=128, hidden_dim=64, alpha=0.02, min_keep=0.95):
        super().__init__()
        hidden_dim = max(hidden_dim, 32)
        self.min_keep = min_keep
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(0.05))
        self.input_proj = nn.Sequential(
            nn.Conv2d(p4_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.small_kernel = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.large_kernel = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 7, padding=9, dilation=3, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.select = nn.Sequential(
            nn.Conv2d(2, 2, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(hidden_dim, p4_channels, 1, bias=False),
            nn.BatchNorm2d(p4_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, p4):
        x = self.input_proj(p4)
        small = self.small_kernel(x)
        large = self.large_kernel(small)
        pooled = torch.cat([
            torch.mean(small + large, dim=1, keepdim=True),
            torch.amax(small + large, dim=1, keepdim=True),
        ], dim=1)
        weights = self.select(pooled)
        context = self.fuse(small * weights[:, 0:1] + large * weights[:, 1:2])
        spatial_gate = self.gate_head(context)
        keep_weight = 1.0 - self.alpha * spatial_gate
        keep_weight = torch.clamp(keep_weight, min=self.min_keep, max=1.0)
        return p4 * keep_weight + self.beta * self.refine_head(context)


class BAMGateSuppression(nn.Module):
    """BAM-style bottleneck attention gate used as a whole-module replacement for DASM + HF prior."""
    def __init__(self, p4_channels=128, hidden_dim=64, alpha=0.02, min_keep=0.95):
        super().__init__()
        hidden_dim = max(hidden_dim, 32)
        reduction_dim = max(hidden_dim // 4, 16)
        self.min_keep = min_keep
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(0.05))
        self.input_proj = nn.Sequential(
            nn.Conv2d(p4_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduction_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduction_dim, reduction_dim, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduction_dim, reduction_dim, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduction_dim, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(hidden_dim, p4_channels, 1, bias=False),
            nn.BatchNorm2d(p4_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, p4):
        x = self.input_proj(p4)
        channel_gate = self.channel_gate(x)
        spatial_gate = self.spatial_gate(x)
        bam_gate = torch.sigmoid(channel_gate + spatial_gate)
        context = self.context(x * (1.0 + bam_gate))
        spatial_gate = self.gate_head(context)
        keep_weight = 1.0 - self.alpha * spatial_gate
        keep_weight = torch.clamp(keep_weight, min=self.min_keep, max=1.0)
        return p4 * keep_weight + self.beta * self.refine_head(context)
# ==============================================================
# CFDMAEDetector: Downstream detection model
# Pixel Enhancement + Frequency-Guided Noise Suppression
# ==============================================================
class CFDMAEDetector(nn.Module):
    """Downstream detection model with LF enhancement and HF-guided suppression.

    The detector uses asymmetric frequency roles:
      - LF adapters inject stable structural cues before the backbone
      - HF is converted into a detail-preserving prior for DASM rather than fused into image content
      - Optional LF-guided HF modulation regularizes the HF prior before suppression guidance

    ablation_mode:
        'full'    — LF enhancement + HF-guided DASM
        'no_lf'   — remove LF enhancement (HF prior may still guide DASM)
        'no_hf'   — remove HF prior guidance (LF-only enhancement)
        'no_dasm' — disable DASM only
    """

    def __init__(self, num_classes, pretrained_cfdmae_path=None,
                 yolo_pretrained_path=None, img_size=640, patch_size=16,
                 embed_dim=256, encoder_depth=6, num_heads=8, num_levels=2,
                 ablation_mode='none', use_dasm=True,
                 dasm_hidden=64, dasm_alpha=0.02,
                 dasm_min_keep=0.95, dasm_local_attention='psa',
                 dasm_long_attention='lka', dasm_replacement='dasm',
                 diag_mode='normal',
                 reconstruction_mode='cross',
                 **kwargs):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        if reconstruction_mode not in {'cross', 'same'}:
            raise ValueError(f"reconstruction_mode must be 'cross' or 'same', got: {reconstruction_mode}")
        self.reconstruction_mode = reconstruction_mode
        legacy_ablation_aliases = {
            'none': 'full',
            'no_lf_mae': 'no_lf',
            'no_hf_mae': 'no_hf',
        }
        self.ablation_mode = legacy_ablation_aliases.get(ablation_mode, ablation_mode)
        self.lap_pyramid = LaplacianPyramid(num_levels=num_levels)
        self.use_lf = 'no_lf' not in self.ablation_mode
        self.use_hf = 'no_hf' not in self.ablation_mode
        self.use_dasm = 'no_dasm' not in self.ablation_mode
        self.dasm_requested = use_dasm
        self.apply_dasm = use_dasm and self.use_dasm
        if dasm_replacement not in {'dasm', 'aspp_gate', 'coord_gate', 'lsk_gate', 'bam_gate'}:
            raise ValueError(f"dasm_replacement must be dasm/aspp_gate/coord_gate/lsk_gate/bam_gate, got: {dasm_replacement}")
        self.dasm_replacement = dasm_replacement
        self.use_hf_prior = self.use_hf and self.dasm_replacement == 'dasm'
        self.use_lfghe = self.use_lf and self.use_hf_prior and 'no_lfghe' not in self.ablation_mode
        self.diag_mode = diag_mode

        # ---- Frozen ViT encoders ----
        if self.use_lf:
            self.lf_encoder = FreqMAE(img_size=img_size, patch_size=patch_size,
                                      embed_dim=embed_dim, encoder_depth=encoder_depth,
                                      num_heads=num_heads)
        if self.use_hf_prior:
            self.hf_encoder = FreqMAE(img_size=img_size, patch_size=patch_size,
                                      embed_dim=embed_dim, encoder_depth=encoder_depth,
                                      num_heads=num_heads)

        if pretrained_cfdmae_path:
            self._load_and_freeze(pretrained_cfdmae_path)

        # ---- Pre-backbone LF enhancement + HF prior extraction ----
        if self.use_lf:
            self.lf_adapter = nn.Sequential(
                nn.Linear(embed_dim, 3 * patch_size * patch_size),
                nn.BatchNorm2d(3),
                nn.SiLU(inplace=True),
            )
        if self.use_hf_prior:
            self.hf_adapter = nn.Sequential(
                nn.Linear(embed_dim, 3 * patch_size * patch_size),
                nn.BatchNorm2d(3),
                nn.SiLU(inplace=True),
            )
            hf_prior_in_ch = 3 + 3 * int(self.use_lf)
            self.hf_prior_head = nn.Sequential(
                nn.Conv2d(hf_prior_in_ch, 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.SiLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        if self.use_lfghe:
            self.lf_guided_hf = LFGHE(in_ch=3, hidden_ch=16, kernel_size=3)

        fusion_in_ch = 3 + 3 * int(self.use_lf)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_ch, 3, 1),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True),
        )
        if self.diag_mode == 'identity_fusion':
            self._init_identity_fusion(fusion_in_ch)
        else:
            self._init_zero_residual_fusion()

        # ---- YOLO detector ----
        from nets.yolo_training import DetectionLossYOLO26

        if yolo_pretrained_path and 'yolo26' in yolo_pretrained_path.lower():
            from nets.ultralytics.yolo26_wrapper import yolo_v26
            self.detector = yolo_v26(num_cls=num_classes, pretrained=yolo_pretrained_path)
        else:
            from nets.ultralytics.yolo11_wrapper import yolo_ultralytics
            self.detector = yolo_ultralytics(num_cls=num_classes, pretrained=yolo_pretrained_path)
        self.loss_fn = DetectionLossYOLO26(self.detector, num_classes)
        if self.dasm_requested and not self.apply_dasm:
            print(f'[CFD-MAE] DASM disabled by ablation mode: {self.ablation_mode}.')
        if self.apply_dasm:
            dasm_channels = self.detector.infer_dasm_channels(img_size=img_size)
            if self.dasm_replacement == 'dasm':
                self.dasm = DASM(
                    c3_channels=dasm_channels['c3_channels'],
                    c4_channels=dasm_channels['c4_channels'],
                    c5_channels=dasm_channels['c5_channels'],
                    p4_channels=dasm_channels['p4_channels'],
                    hidden_dim=dasm_hidden,
                    alpha=dasm_alpha,
                    min_keep=dasm_min_keep,
                    local_attention=dasm_local_attention,
                    long_attention=dasm_long_attention,
                )
            elif self.dasm_replacement == 'aspp_gate':
                self.dasm = ASPPGateSuppression(
                    p4_channels=dasm_channels['p4_channels'],
                    hidden_dim=dasm_hidden,
                    alpha=dasm_alpha,
                    min_keep=dasm_min_keep,
                )
            elif self.dasm_replacement == 'coord_gate':
                self.dasm = CoordGateSuppression(
                    p4_channels=dasm_channels['p4_channels'],
                    hidden_dim=dasm_hidden,
                    alpha=dasm_alpha,
                    min_keep=dasm_min_keep,
                )
            elif self.dasm_replacement == 'lsk_gate':
                self.dasm = LSKGateSuppression(
                    p4_channels=dasm_channels['p4_channels'],
                    hidden_dim=dasm_hidden,
                    alpha=dasm_alpha,
                    min_keep=dasm_min_keep,
                )
            elif self.dasm_replacement == 'bam_gate':
                self.dasm = BAMGateSuppression(
                    p4_channels=dasm_channels['p4_channels'],
                    hidden_dim=dasm_hidden,
                    alpha=dasm_alpha,
                    min_keep=dasm_min_keep,
                )
            print(f'[CFD-MAE] DASM channels inferred from detector: {dasm_channels}')
            print(f'[CFD-MAE] DASM replacement: {self.dasm_replacement}')
            if self.dasm_replacement == 'dasm':
                print(f'[CFD-MAE] DASM attention: local={dasm_local_attention}, long={dasm_long_attention}')

    def _init_identity_fusion(self, fusion_in_ch):
        conv = self.fusion[0]
        bn = self.fusion[1]
        with torch.no_grad():
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()
            for c in range(min(3, fusion_in_ch)):
                conv.weight[c, c, 0, 0] = 1.0
            bn.weight.fill_(1.0)
            bn.bias.zero_()
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
        print('[CFD-MAE] Initialized fusion to near-identity on image channels.')

    def _init_zero_residual_fusion(self):
        conv = self.fusion[0]
        bn = self.fusion[1]
        with torch.no_grad():
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()
            bn.weight.fill_(1.0)
            bn.bias.zero_()
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
        print('[CFD-MAE] Initialized fusion as zero residual branch.')

    def _load_and_freeze(self, ckpt_path):
        """Load pretrained CFD-MAE and freeze encoders."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt['model']

        lf_state = {}
        hf_state = {}
        for k, v in state.items():
            if k.startswith('lf_mae.'):
                lf_state[k.replace('lf_mae.', '')] = v
            elif k.startswith('hf_mae.'):
                hf_state[k.replace('hf_mae.', '')] = v

        if self.use_lf:
            self.lf_encoder.load_state_dict(lf_state, strict=False)
            for p in self.lf_encoder.parameters():
                p.requires_grad = False
            print(f"[CFD-MAE] LF encoder loaded and frozen")

        if self.use_hf_prior:
            self.hf_encoder.load_state_dict(hf_state, strict=False)
            for p in self.hf_encoder.parameters():
                p.requires_grad = False
            print(f"[CFD-MAE] HF encoder loaded and frozen")

        print(f"[CFD-MAE] Loaded from {ckpt_path} (ablation={self.ablation_mode})")

    def _adapt_features(self, feat, adapter):
        """Convert ViT [B, N+1, D] features to spatial [B, 3, H, W]."""
        feat = feat[:, 1:, :]  # Remove cls token [B, N, D]
        B, N, D = feat.shape
        h = w = int(N ** 0.5)

        feat = adapter[0](feat)  # Linear: [B, N, 3*P*P]
        feat = feat.reshape(B, h, w, 3, self.patch_size, self.patch_size)
        feat = feat.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, h * self.patch_size, w * self.patch_size)

        feat = adapter[1](feat)  # BN
        feat = adapter[2](feat)  # SiLU
        return feat

    def _build_hf_prior(self, hf_map, lf_map=None):
        prior_inputs = [hf_map]
        if lf_map is not None:
            prior_inputs.append(lf_map)
        hf_prior = self.hf_prior_head(torch.cat(prior_inputs, dim=1))
        return hf_prior

    def forward(self, images, targets=None):
        """Forward pass.
        Args:
            images: [B, 3, H, W]
            targets: list of [N, 5] for training, None for inference
        """
        with torch.no_grad():
            pyramid = self.lap_pyramid.decompose(images)
            high_freq = pyramid[0]
            low_freq = pyramid[-1]
            low_freq_up = F.interpolate(low_freq, size=images.shape[2:],
                                        mode='bilinear', align_corners=False)

        lf_map = None
        hf_map = None
        hf_prior = None

        # Ablation: keep the detector input consistent with the active enhancement sources.
        feat_list = [images]

        if self.diag_mode != 'detector_only':
            if self.use_lf:
                with torch.no_grad():
                    lf_input = low_freq_up if self.reconstruction_mode == 'same' else high_freq
                    lf_map = self._adapt_features(self.lf_encoder.forward_features(lf_input), self.lf_adapter)
                feat_list.append(lf_map)

            if self.use_hf_prior:
                with torch.no_grad():
                    hf_input = high_freq if self.reconstruction_mode == 'same' else low_freq_up
                    hf_raw = self._adapt_features(self.hf_encoder.forward_features(hf_input), self.hf_adapter)
                hf_map = hf_raw
                if self.use_lfghe and lf_map is not None:
                    hf_map = self.lf_guided_hf(hf_raw, lf_map)
                hf_prior = self._build_hf_prior(hf_map, lf_map)

        if self.diag_mode in {'detector_only', 'images_only'}:
            enhanced = images
        else:
            fusion_out = self.fusion(torch.cat(feat_list, dim=1))
            if self.diag_mode == 'identity_fusion':
                enhanced = fusion_out
            else:
                enhanced = images + fusion_out
        backbone_feats, neck_feats = self.detector.forward_backbone_and_neck(enhanced)
        if self.apply_dasm:
            if self.dasm_replacement == 'dasm':
                neck_feats[1] = self.dasm(backbone_feats, neck_feats[1], hf_prior=hf_prior)
            else:
                neck_feats[1] = self.dasm(neck_feats[1])

        if targets is not None:
            outputs = self.detector.forward_from_features(neck_feats, raw_output=True)
            det_loss, _ = self.loss_fn(outputs, targets, images.shape[2:])
            result = {'loss': det_loss, 'loss_det': det_loss.detach(), 'predictions': outputs}
            if hf_prior is not None:
                result['hf_prior_mean'] = hf_prior.detach().mean()
            return result

        outputs = self.detector.forward_from_features(neck_feats)
        result = {'predictions': outputs}
        if hf_prior is not None:
            result['hf_prior_mean'] = hf_prior.detach().mean()
        return result


# ==============================================================
# CFDMAEAdapter: for baseline training framework
# ==============================================================
class CFDMAEAdapter(nn.Module):
    """Adapter for baseline training framework compatibility."""

    def __init__(self, num_classes, cfdmae_pretrained_path=None,
                 yolo_pretrained_path=None, **kwargs):
        super().__init__()
        self.detector = CFDMAEDetector(
            num_classes=num_classes,
            pretrained_cfdmae_path=cfdmae_pretrained_path,
            yolo_pretrained_path=yolo_pretrained_path,
            **kwargs,
        )

    def forward(self, images, targets=None):
        return self.detector(images, targets)
