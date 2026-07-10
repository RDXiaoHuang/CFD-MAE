"""
CFD-MAE Pretraining Script
Self-supervised pretraining with configurable frequency reconstruction.

Usage:
    python train_cfdmae_pretrain.py
    python train_cfdmae_pretrain.py --data rtts
    python train_cfdmae_pretrain.py --data all
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import math
import glob
import random
import numpy as np

from nets.cfdmae import CFDMAE
from utils.checkpoint_cleanup import cleanup_periodic_checkpoints, should_save_periodic_checkpoint
import config_cfdmae_pretrain as config


# ============================================================
# Self-supervised Dataset (no labels needed)
# ============================================================
class PretrainImageDataset(Dataset):
    """Load images for self-supervised pretraining. No labels needed."""

    def __init__(self, image_paths, img_size=640, data_name='rtts'):
        self.image_paths = image_paths
        self.data_name = data_name

        # Weather-specific augmentation strategies
        if data_name == 'snow':
            # Snow: Preserve brightness, minimal color jitter
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),  # Reduced
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif data_name == 'exdark':
            # Low-light: augment brightness/contrast to improve diversity
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1, hue=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Default for fog/rain
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


def collect_image_paths():
    """Collect image paths from configured directories or annotation files."""
    paths = []

    # Try image directories first
    for d in config.train_image_dirs:
        if os.path.isdir(d):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                paths.extend(glob.glob(os.path.join(d, ext)))
                paths.extend(glob.glob(os.path.join(d, ext.upper())))

    # Fallback: parse annotation file
    if len(paths) == 0 and os.path.exists(config.train_annotation_path):
        with open(config.train_annotation_path) as f:
            for line in f:
                img_path = line.strip().split()[0]
                if os.path.exists(img_path):
                    paths.append(img_path)

    paths = list(set(paths))
    random.shuffle(paths)
    print(f"[CFD-MAE Pretrain] Collected {len(paths)} images")
    return paths


# ============================================================
# LR Scheduler with warmup + cosine decay
# ============================================================
def get_cosine_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr):
    """Cosine LR schedule with linear warmup."""
    if epoch < warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if config.reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Main Training Loop
# ============================================================
def train():
    set_seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    checkpoint_path = os.path.join(config.save_dir, 'last.pth')
    best_path = os.path.join(config.save_dir, 'best.pth')

    # Collect images
    image_paths = collect_image_paths()
    if len(image_paths) == 0:
        raise RuntimeError("No images found. Check config.train_image_dirs or train_annotation_path.")

    dataset = PretrainImageDataset(image_paths, img_size=config.img_size, data_name=config.data_name)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )

    # Build model
    model = CFDMAE(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        encoder_depth=config.encoder_depth,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        num_heads=config.num_heads,
        mask_ratio=config.mask_ratio,
        num_levels=config.num_levels,
        self_recon_weight=config.self_recon_weight,
        hf_loss_weight=config.hf_loss_weight,
        pretrain_loss_config=config.pretrain_loss_config,
        reconstruction_mode=config.pretrain_reconstruction_mode,
    )

    if config.Cuda:
        model = model.cuda()

    # Optimizer
    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                     if 'bias' not in n and 'norm' not in n],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                     if 'bias' in n or 'norm' in n],
         'weight_decay': 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))

    # AMP scaler
    scaler = GradScaler(enabled=config.fp16)

    start_epoch = 0
    best_loss = float('inf')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler_state = checkpoint.get('scaler', checkpoint.get('scaler_state_dict'))
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))
        print(f"[CFD-MAE Pretrain] Resume from epoch {start_epoch}")
    elif os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))
        print(f"[CFD-MAE Pretrain] Warm start from {best_path}")

    # Print info
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'='*60}")
    print(f"CFD-MAE Pretraining")
    print(f"{'='*60}")
    print(f"Images: {len(image_paths)}, Batch: {config.batch_size}")
    print(f"Epochs: {config.epochs}, LR: {config.lr}")
    print(f"Mask ratio: {config.mask_ratio}")
    print(f"Reconstruction mode: {config.pretrain_reconstruction_mode}")
    print(f"Embed dim: {config.embed_dim}, Encoder depth: {config.encoder_depth}")
    print(f"Patch size: {config.patch_size}, Pyramid levels: {config.num_levels}")
    print(f"Parameters: {total_params:.1f}M")
    print(f"Save dir: {config.save_dir}")
    print(f"{'='*60}\n")

    # Training log file
    log_path = os.path.join(config.save_dir, 'train_log.txt')
    if start_epoch == 0 or not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write(f"CFD-MAE Pretraining Log - {config.data_name}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Images: {len(image_paths)}, Batch: {config.batch_size}, Epochs: {config.epochs}\n")
            f.write(f"LR: {config.lr}, Embed dim: {config.embed_dim}, Depth: {config.encoder_depth}\n")
            f.write(f"Reconstruction mode: {config.pretrain_reconstruction_mode}\n")
            f.write(f"Patch size: {config.patch_size}, Mask ratio: {config.mask_ratio}, Levels: {config.num_levels}\n")
            f.write(f"Parameters: {total_params:.1f}M\n")
            f.write(f"{'='*70}\n")
            f.write(f"{'Epoch':>6} {'Loss':>10} {'LF':>10} {'HF':>10} {'Self':>10} {'LR':>12} {'Best':>6}\n")
            f.write(f"{'-'*70}\n")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        lr = get_cosine_lr(epoch, config.epochs, config.warmup_epochs,
                           config.lr, config.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        total_loss = 0
        total_lf = 0
        total_hf = 0
        total_self = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for i, images in enumerate(pbar):
            if config.Cuda:
                images = images.cuda(non_blocking=True)

            with autocast(enabled=config.fp16):
                result = model(images)
                loss = result['loss']

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_lf += result['loss_lf']
            total_hf += result['loss_hf']
            total_self += result['loss_self']

            if (i + 1) % config.log_interval == 0 or i == len(dataloader) - 1:
                avg = total_loss / (i + 1)
                pbar.set_postfix(loss=f'{avg:.4f}', lr=f'{lr:.2e}',
                                 lf=f'{total_lf/(i+1):.4f}',
                                 hf=f'{total_hf/(i+1):.4f}')

        avg_loss = total_loss / len(dataloader)
        avg_lf = total_lf / len(dataloader)
        avg_hf = total_hf / len(dataloader)
        avg_self = total_self / len(dataloader)
        is_best = avg_loss < best_loss

        print(f'Epoch {epoch+1}/{config.epochs} | '
              f'Loss: {avg_loss:.4f} | '
              f'LF: {avg_lf:.4f} | '
              f'HF: {avg_hf:.4f} | '
              f'Self: {avg_self:.4f} | '
              f'LR: {lr:.2e}')

        # Write to log file
        with open(log_path, 'a') as f:
            best_mark = '*' if is_best else ''
            f.write(f"{epoch+1:>6} {avg_loss:>10.4f} {avg_lf:>10.4f} {avg_hf:>10.4f} {avg_self:>10.4f} {lr:>12.2e} {best_mark:>6}\n")

        # Save checkpoint
        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'loss': avg_loss,
            'best_loss': min(best_loss, avg_loss),
            'config': {
                'img_size': config.img_size,
                'patch_size': config.patch_size,
                'embed_dim': config.embed_dim,
                'encoder_depth': config.encoder_depth,
                'num_levels': config.num_levels,
                'reconstruction_mode': config.pretrain_reconstruction_mode,
            },
        }

        if should_save_periodic_checkpoint() and (epoch + 1) % config.save_period == 0:
            torch.save(ckpt, f'{config.save_dir}/ep{epoch+1:03d}_loss{avg_loss:.4f}.pth')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, f'{config.save_dir}/best.pth')
            print(f'  -> Best model saved (loss={best_loss:.4f})')

        torch.save(ckpt, f'{config.save_dir}/last.pth')

    print(f"\nCFD-MAE pretraining completed! Best loss: {best_loss:.4f}")
    cleanup_periodic_checkpoints(config.save_dir)

    # Write summary to log
    with open(log_path, 'a') as f:
        f.write(f"{'-'*70}\n")
        f.write(f"Training completed. Best loss: {best_loss:.4f}\n")


if __name__ == '__main__':
    train()
