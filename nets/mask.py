"""
mask.py: CFD-MAE pretraining components.
  - LaplacianPyramid: frequency decomposition
  - PatchEmbed: image to patch embedding
  - get_2d_sincos_pos_embed: sinusoidal positional embedding
  - TransformerBlock: standard ViT block
  - FreqMAE: single-branch frequency masked autoencoder
  - CFDMAE: full frequency pretraining model
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Laplacian Pyramid
# ==============================================================
class LaplacianPyramid(nn.Module):
    """Fixed Laplacian Pyramid for frequency decomposition."""

    def __init__(self, num_levels=2, kernel_size=5, channels=3):
        super().__init__()
        self.num_levels = num_levels
        self.register_buffer('kernel', self._gauss_kernel(kernel_size, channels))

    def _gauss_kernel(self, kernel_size, channels):
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * (kernel_size / 6.)**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        return kernel

    def _conv_gauss(self, x):
        return F.conv2d(x, self.kernel, padding=self.kernel.shape[-1] // 2, groups=x.shape[1])

    def _downsample(self, x):
        return x[:, :, ::2, ::2]

    def _upsample(self, x, target_size):
        up = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self._conv_gauss(up)

    def decompose(self, x):
        """Decompose image into [high_freq_0, high_freq_1, ..., low_freq]."""
        current = x
        pyramid = []
        for _ in range(self.num_levels):
            down = self._downsample(self._conv_gauss(current))
            up = self._upsample(down, current.shape[2:])
            high = current - up
            pyramid.append(high)
            current = down
        pyramid.append(current)
        return pyramid

    def reconstruct(self, pyramid):
        """Reconstruct image from pyramid [high_0, high_1, ..., low_freq]."""
        image = pyramid[-1]
        for high in reversed(pyramid[:-1]):
            image = self._upsample(image, high.shape[2:]) + high
        return image


# ==============================================================
# Patch Embedding
# ==============================================================
class PatchEmbed(nn.Module):
    """Image to Patch Embedding via Conv2d."""

    def __init__(self, img_size=640, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                    # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)    # [B, N, D]
        x = self.norm(x)
        return x


# ==============================================================
# Positional Embedding (sinusoidal)
# ==============================================================
def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, dtype=torch.float32):
    """Generate 2D sinusoidal positional embedding."""
    grid_h_arr = torch.arange(grid_h, dtype=dtype)
    grid_w_arr = torch.arange(grid_w, dtype=dtype)
    grid = torch.stack(torch.meshgrid(grid_h_arr, grid_w_arr, indexing='ij'))  # [2, H, W]
    grid = grid.reshape(2, -1)  # [2, H*W]

    half_dim = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(0, half_dim, 2, dtype=dtype) / half_dim))

    out_h = grid[0:1].T @ omega.unsqueeze(0)
    out_w = grid[1:2].T @ omega.unsqueeze(0)

    pos = torch.cat([torch.sin(out_h), torch.cos(out_h),
                     torch.sin(out_w), torch.cos(out_w)], dim=-1)

    if pos.shape[1] > embed_dim:
        pos = pos[:, :embed_dim]
    elif pos.shape[1] < embed_dim:
        pos = F.pad(pos, (0, embed_dim - pos.shape[1]))

    return pos.unsqueeze(0)  # [1, H*W, D]


# ==============================================================
# Transformer Block
# ==============================================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================
# FreqMAE: Single-branch Frequency MAE
# ==============================================================
class FreqMAE(nn.Module):
    """Single-branch Frequency MAE.
    - Encoder: takes visible patches of input freq component
    - Decoder: reconstructs the target freq component
    """

    def __init__(self, img_size=640, patch_size=16, in_channels=3,
                 embed_dim=256, encoder_depth=6,
                 decoder_embed_dim=128, decoder_depth=2,
                 num_heads=8, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        grid_size = img_size // patch_size
        out_channels = in_channels * patch_size ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(embed_dim, grid_size, grid_size),
            requires_grad=False
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, grid_size),
            requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def random_masking(self, x):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def encode(self, x, mask=True):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        if mask:
            x, mask_out, ids_restore = self.random_masking(x)
        else:
            mask_out = ids_restore = None
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x), mask_out, ids_restore

    def decode(self, x, ids_restore):
        x = self.decoder_embed(x)
        cls_token, x = x[:, :1, :], x[:, 1:, :]
        N_full = self.pos_embed.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], N_full - x.shape[1], 1)
        x_full = torch.gather(
            torch.cat([x, mask_tokens], dim=1), 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x_full = torch.cat([cls_token, x_full], dim=1)[:, 1:, :] + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        return self.decoder_pred(self.decoder_norm(x_full))

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        return imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)

    def unpatchify(self, x):
        p = self.patch_size
        C = self.in_channels
        h = w = int(x.shape[1] ** 0.5)
        return x.reshape(x.shape[0], h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], C, h * p, w * p)

    def forward_pretrain(self, input_freq, target_freq, active_mask=None, loss_config=None):
        latent, mask, ids_restore = self.encode(input_freq, mask=True)
        preds = self.decode(latent, ids_restore)
        target = self.patchify(target_freq)
        per_patch = ((preds - target) ** 2).mean(dim=-1)
        if active_mask is not None:
            loss = (per_patch * active_mask).sum() / (active_mask.sum() + 1e-6)
        else:
            loss = per_patch.mean()
        fft_weight = (loss_config or {}).get('fft_weight', 0.0)
        if fft_weight > 0:
            fft_pred = torch.fft.rfft2(self.unpatchify(preds).float(), norm='ortho')
            fft_tgt = torch.fft.rfft2(self.unpatchify(target).float(), norm='ortho')
            loss = loss + fft_weight * F.l1_loss(fft_pred.abs(), fft_tgt.abs())
        return loss, preds, mask

    def forward_features(self, x):
        feat, _, _ = self.encode(x, mask=False)
        return feat

    def forward(self, x):
        return self.forward_features(x)


# ==============================================================
# CFDMAE: Full pretraining model
# ==============================================================
class CFDMAE(nn.Module):
    """CFD-MAE pretraining model.
    - cross mode: LFMAE input=high_freq target=low_freq; HFMAE input=low_freq target=high_freq
    - same mode: LFMAE input=low_freq target=low_freq; HFMAE input=high_freq target=high_freq
    """

    def __init__(self, img_size=640, patch_size=16, embed_dim=256,
                 encoder_depth=6, decoder_embed_dim=128, decoder_depth=2,
                 num_heads=8, mask_ratio=0.75, num_levels=2,
                 self_recon_weight=0.1, hf_loss_weight=1.0, pretrain_loss_config=None,
                 reconstruction_mode='cross'):
        super().__init__()
        self.self_recon_weight = self_recon_weight
        self.hf_loss_weight = hf_loss_weight
        if reconstruction_mode not in {'cross', 'same'}:
            raise ValueError(f"reconstruction_mode must be 'cross' or 'same', got: {reconstruction_mode}")
        self.reconstruction_mode = reconstruction_mode
        self.pretrain_loss_config = copy.deepcopy(pretrain_loss_config or {
            'lf': {'use_mask': False, 'fft_weight': 0.0},
            'hf': {'use_mask': True, 'fft_weight': 1.0},
        })
        self.lap_pyramid = LaplacianPyramid(num_levels=num_levels)
        mae_kwargs = dict(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            encoder_depth=encoder_depth, decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth, num_heads=num_heads, mask_ratio=mask_ratio,
        )
        self.lf_mae = FreqMAE(**mae_kwargs)
        self.hf_mae = FreqMAE(**mae_kwargs)

    def forward(self, images):
        pyramid = self.lap_pyramid.decompose(images)
        high_freq = pyramid[0]
        low_freq_up = F.interpolate(pyramid[-1], size=images.shape[2:], mode='bilinear', align_corners=False)

        hf_energy = (self.lf_mae.patchify(high_freq) ** 2).mean(dim=-1)
        hf_active = (hf_energy > hf_energy.mean(dim=1, keepdim=True)).float()

        lf_cfg = self.pretrain_loss_config.get('lf', {})
        hf_cfg = self.pretrain_loss_config.get('hf', {})

        if self.reconstruction_mode == 'same':
            lf_input, lf_target = low_freq_up, low_freq_up
            hf_input, hf_target = high_freq, high_freq
            aux_self = None
        else:
            lf_input, lf_target = high_freq, low_freq_up
            hf_input, hf_target = low_freq_up, high_freq
            loss_self_lf, _, _ = self.lf_mae.forward_pretrain(low_freq_up, low_freq_up)
            loss_self_hf, _, _ = self.hf_mae.forward_pretrain(high_freq, high_freq, active_mask=hf_active)
            aux_self = loss_self_lf + loss_self_hf

        loss_lf, _, _ = self.lf_mae.forward_pretrain(lf_input, lf_target,
            active_mask=hf_active if lf_cfg.get('use_mask') else None, loss_config=lf_cfg)
        loss_hf, _, _ = self.hf_mae.forward_pretrain(hf_input, hf_target,
            active_mask=hf_active if hf_cfg.get('use_mask') else None, loss_config=hf_cfg)
        loss_self = torch.zeros((), device=images.device) if aux_self is None else self.self_recon_weight * aux_self
        total_loss = loss_lf + self.hf_loss_weight * loss_hf + loss_self
        return {
            'loss': total_loss,
            'loss_lf': loss_lf.item(),
            'loss_hf': loss_hf.item(),
            'loss_self': loss_self.item(),
        }
