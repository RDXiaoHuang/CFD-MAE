"""
CFD-MAE Downstream Detection Training Script
Frozen CFD-MAE encoders + YOLOv26n for adverse weather object detection.

Usage:
    python train_cfdmae_detect.py
    python train_cfdmae_detect.py --data rtts
    python train_cfdmae_detect.py --data rain
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import re
import sys
from functools import partial

from nets.cfdmae import CFDMAEDetector
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, get_anchors, get_lr, seed_everything, worker_init_fn
from utils.callbacks import EvalCallback, LossHistory
from utils.checkpoint_cleanup import cleanup_periodic_checkpoints, should_save_periodic_checkpoint
from nets.yolo_training import get_lr_scheduler, set_optimizer_lr
import config_cfdmae_detect as config


def find_resume_epoch(save_dir):
    loss_file = os.path.join(save_dir, 'epoch_loss.txt')
    if not os.path.exists(loss_file):
        return 0
    with open(loss_file) as f:
        return sum(1 for line in f if line.strip())


def find_latest_epoch_weight(save_dir):
    latest_epoch = 0
    latest_path = None
    pattern = re.compile(r'^ep(\d+)-loss.*\.pth$')
    for name in os.listdir(save_dir):
        match = pattern.match(name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = os.path.join(save_dir, name)
    return latest_epoch, latest_path


def validate_annotation_lines(lines, annotation_path, split_name):
    total = len(lines)
    labeled = sum(1 for line in lines if len(line.split()) > 1)
    if total == 0:
        raise RuntimeError(f"{split_name} annotation file is empty: {annotation_path}")
    if labeled == 0:
        raise RuntimeError(
            f"{split_name} annotation file contains no bounding boxes: {annotation_path}. "
            f"Check whether the split txt is missing, or whether remapping removed all target boxes. "
            f"Inspect {annotation_path}.bak and preview with tools/remap_annotations.py --dry first; "
            f"only rerun tools/voc_annotation.py when the original split txt itself needs regeneration."
        )
    print(f"[Dataset] {split_name}: {labeled}/{total} images with boxes from {annotation_path}")


def train():
    seed_everything(config.seed)

    class_names, num_classes = get_classes(config.classes_path)
    anchors, _ = get_anchors(config.anchors_path)

    # Build model
    model = CFDMAEDetector(
        num_classes=num_classes,
        pretrained_cfdmae_path=config.cfdmae_pretrained_path,
        yolo_pretrained_path=config.yolo_pretrained_path,
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        encoder_depth=config.encoder_depth,
        num_heads=config.num_heads,
        num_levels=config.num_levels,
        ablation_mode=config.ablation_mode,
        use_dasm=config.use_dasm,
        dasm_hidden=config.dasm_hidden,
        dasm_alpha=config.dasm_alpha,
        dasm_min_keep=config.dasm_min_keep,
        dasm_local_attention=config.dasm_local_attention,
        dasm_long_attention=config.dasm_long_attention,
        dasm_replacement=config.dasm_replacement,
        diag_mode=config.cfdmae_diag_mode,
        reconstruction_mode=config.cfdmae_reconstruction_mode,
        clip_adapt_mode=config.clip_adapt_mode,
        clip_model_name=config.clip_model_name,
        clip_prompts=config.clip_prompts,
        clip_negative_prompt_index=config.clip_negative_prompt_index,
        clip_strength=config.clip_strength,
        clip_device=config.clip_device,
        clip_cache_dir=config.clip_cache_dir,
        clip_weights_path=config.clip_weights_path,
    )

    if config.Cuda:
        model = model.cuda()

    os.makedirs(config.save_dir, exist_ok=True)
    resume_epoch = find_resume_epoch(config.save_dir)
    latest_epoch, latest_weight = find_latest_epoch_weight(config.save_dir)
    if latest_weight and latest_epoch >= resume_epoch:
        state_dict = torch.load(latest_weight, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        resume_epoch = latest_epoch
        print(f"[CFD-MAE Detect] Resume from epoch {resume_epoch}: {latest_weight}")
    elif os.path.exists(os.path.join(config.save_dir, 'last.pth')):
        last_path = os.path.join(config.save_dir, 'last.pth')
        state_dict = torch.load(last_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"[CFD-MAE Detect] Loaded last weights: {last_path}")

    # Only optimize non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    train_params = sum(p.numel() for p in trainable_params)
    print(f"[CFD-MAE] Trainable: {train_params/1e6:.1f}M, Frozen: {frozen_params/1e6:.1f}M")

    # Optimizer (same as baseline for fair comparison)
    nbs = 64
    lr_limit_max = 5e-2
    lr_limit_min = 5e-4
    Init_lr_fit = min(max(config.batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(config.batch_size / nbs * config.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter) and v.bias.requires_grad:
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            if v.weight is not None and v.weight.requires_grad:
                pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter) and v.weight.requires_grad:
            pg1.append(v.weight)

    optimizer = optim.SGD(pg0, Init_lr_fit, momentum=config.momentum, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": config.weight_decay})
    optimizer.add_param_group({"params": pg2})

    lr_scheduler_func = get_lr_scheduler(config.lr_decay_type, Init_lr_fit, Min_lr_fit, config.UnFreeze_Epoch)

    # Datasets
    with open(config.train_annotation_path) as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(config.val_annotation_path) as f:
        val_lines = [line.strip() for line in f if line.strip()]

    validate_annotation_lines(train_lines, config.train_annotation_path, 'train')
    validate_annotation_lines(val_lines, config.val_annotation_path, 'val')

    train_dataset = YoloDataset(train_lines, config.input_shape, num_classes,
                                epoch_length=config.UnFreeze_Epoch, train=True, data_name=config.data_name)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=yolo_dataset_collate,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=config.seed),
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )

    val_dataset = YoloDataset(val_lines, config.input_shape, num_classes,
                              epoch_length=config.UnFreeze_Epoch, train=False, data_name=config.data_name)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=yolo_dataset_collate,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=config.seed),
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )

    run_stop_epoch = min(config.Run_Stop_Epoch, config.UnFreeze_Epoch)

    # Logging & eval
    log_dir = config.save_dir
    os.makedirs(log_dir, exist_ok=True)

    # Clean old tensorboard logs
    for f in os.listdir(log_dir):
        if f.startswith('events.out.tfevents'):
            try:
                os.remove(os.path.join(log_dir, f))
            except:
                pass

    loss_history = LossHistory(log_dir, model, config.input_shape)
    eval_callback = EvalCallback(model, config.input_shape, anchors, config.anchors_mask,
                                 class_names, num_classes, val_lines, log_dir, True,
                                 map_out_path=os.path.join(log_dir, '.temp_map_out'),
                                 eval_flag=config.eval_flag, period=config.eval_period)

    if resume_epoch > 0:
        with open(os.path.join(log_dir, 'epoch_loss.txt')) as f:
            loss_history.losses = [float(line.strip()) for line in f if line.strip()][:resume_epoch]
        with open(os.path.join(log_dir, 'epoch_val_loss.txt')) as f:
            loss_history.val_loss = [float(line.strip()) for line in f if line.strip()][:resume_epoch]
        map_path = os.path.join(log_dir, 'epoch_map.txt')
        if os.path.exists(map_path):
            with open(map_path) as f:
                eval_callback.maps = [float(line.strip()) for line in f if line.strip()]
            eval_callback.epoches = list(range(config.eval_period, config.eval_period * len(eval_callback.maps) + 1, config.eval_period))
            if eval_callback.maps:
                eval_callback.best_map = max(eval_callback.maps)
                eval_callback.best_map_epoch = eval_callback.epoches[eval_callback.maps.index(eval_callback.best_map)]

    print(f'\n{"="*60}')
    print(f'CFD-MAE Downstream Detection Training')
    print(f'{"="*60}')
    print(f'Dataset: {config.data_name}, Classes: {num_classes}')
    print(f'Planned epochs: {config.UnFreeze_Epoch}, Run stop epoch: {run_stop_epoch}, Batch: {config.batch_size}')
    print(f'Optimizer: SGD, Init LR: {Init_lr_fit:.6f}, Min LR: {Min_lr_fit:.6f}')
    print(f'Ablation mode: {config.ablation_mode}')
    print(f'Diag mode: {config.cfdmae_diag_mode}')
    print(f'Reconstruction mode: {config.cfdmae_reconstruction_mode}')
    print(f'CLIP adapt mode: {config.clip_adapt_mode}')
    print(f'DASM requested: {config.use_dasm}')
    print(f'DASM effective: {model.apply_dasm}')
    print(f'DASM replacement: {config.dasm_replacement}')
    print(f'DASM local attention: {config.dasm_local_attention}')
    print(f'DASM long attention: {config.dasm_long_attention}')
    print(f'CFD-MAE pretrained: {config.cfdmae_pretrained_path}')
    print(f'YOLO pretrained: {config.yolo_pretrained_path}')
    print(f'Save dir: {log_dir}')
    print(f'{"="*60}\n')

    if resume_epoch >= run_stop_epoch:
        print(f'[CFD-MAE Detect] Save dir already reached target stop epoch {run_stop_epoch}. Nothing to run.')
        return

    # Training loop
    for epoch in range(resume_epoch, run_stop_epoch):
        if hasattr(train_dataset, 'epoch_now'):
            train_dataset.epoch_now = epoch
        if hasattr(val_dataset, 'epoch_now'):
            val_dataset.epoch_now = epoch

        model.train()
        total_loss = 0
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        print('Start Train')
        pbar = tqdm(total=len(train_loader),
                    desc=f'Epoch {epoch+1}/{run_stop_epoch}',
                    postfix=dict, mininterval=0.3)

        for iteration, batch in enumerate(train_loader):
            images, targets = batch[0].cuda(), batch[1]
            img_h, img_w = images.shape[2], images.shape[3]
            targets_list = []
            for t in targets:
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, dtype=torch.float32)
                t = t.cuda()
                if t.numel() > 0:
                    cls = t[:, 4:5]
                    x_pix = t[:, 0:1] * img_w
                    y_pix = t[:, 1:2] * img_h
                    w_pix = t[:, 2:3] * img_w
                    h_pix = t[:, 3:4] * img_h
                    targets_list.append(torch.cat([cls, x_pix, y_pix, w_pix, h_pix], dim=1))
                else:
                    targets_list.append(torch.zeros(0, 5, device=images.device))

            optimizer.zero_grad()
            result = model(images, targets_list)
            loss = result['loss']

            if torch.isnan(loss):
                print(f"WARNING: NaN loss at iteration {iteration}, skipping...")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            postfix = {'total_loss': total_loss / (iteration + 1), 'lr': get_lr(optimizer)}
            if 'loss_hf_detail' in result:
                postfix['hf_d'] = float(result['loss_hf_detail'].detach().item())
            if 'loss_hf_noise' in result:
                postfix['hf_n'] = float(result['loss_hf_noise'].detach().item())
            if 'loss_lf_consistency' in result:
                postfix['lf_c'] = float(result['loss_lf_consistency'].detach().item())
            if 'hf_alpha_mean' in result:
                postfix['hf_a'] = float(result['hf_alpha_mean'].detach().item())
            if 'clip_adverse_mean' in result:
                postfix['clip_w'] = float(result['clip_adverse_mean'].detach().item())
            pbar.set_postfix(**postfix)
            pbar.update(1)

        pbar.close()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        print('Finish Train')
        print('Start Validation')
        model.eval()
        val_loss = 0
        pbar = tqdm(total=len(val_loader),
                    desc=f'Epoch {epoch+1}/{run_stop_epoch}',
                    postfix=dict, mininterval=0.3)

        with torch.no_grad():
            for iteration, batch in enumerate(val_loader):
                images, targets = batch[0].cuda(), batch[1]
                img_h, img_w = images.shape[2], images.shape[3]
                targets_list = []
                for t in targets:
                    if not isinstance(t, torch.Tensor):
                        t = torch.tensor(t, dtype=torch.float32)
                    t = t.cuda()
                    if t.numel() > 0:
                        cls = t[:, 4:5]
                        x_pix = t[:, 0:1] * img_w
                        y_pix = t[:, 1:2] * img_h
                        w_pix = t[:, 2:3] * img_w
                        h_pix = t[:, 3:4] * img_h
                        targets_list.append(torch.cat([cls, x_pix, y_pix, w_pix, h_pix], dim=1))
                    else:
                        targets_list.append(torch.zeros(0, 5, device=images.device))

                result = model(images, targets_list)
                loss = result['loss']
                val_loss += loss.item()
                postfix = {'val_loss': val_loss / (iteration + 1)}
                if 'loss_hf_detail' in result:
                    postfix['hf_d'] = float(result['loss_hf_detail'].detach().item())
                if 'loss_hf_noise' in result:
                    postfix['hf_n'] = float(result['loss_hf_noise'].detach().item())
                if 'loss_lf_consistency' in result:
                    postfix['lf_c'] = float(result['loss_lf_consistency'].detach().item())
                if 'hf_alpha_mean' in result:
                    postfix['hf_a'] = float(result['hf_alpha_mean'].detach().item())
                if 'clip_adverse_mean' in result:
                    postfix['clip_w'] = float(result['clip_adverse_mean'].detach().item())
                pbar.set_postfix(**postfix)
                pbar.update(1)

        pbar.close()
        avg_val_loss = val_loss / len(val_loader)

        print('Finish Validation')
        loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
        eval_callback.on_epoch_end(epoch + 1, model)
        print(f'Epoch: {epoch+1}/{run_stop_epoch} (plan {config.UnFreeze_Epoch})')
        print(f'Total Loss: {avg_train_loss:.3f} || Val Loss: {avg_val_loss:.3f}')

        # Save checkpoint
        if should_save_periodic_checkpoint() and (
            (epoch + 1) % config.save_period == 0 or epoch + 1 == run_stop_epoch
        ):
            torch.save(model.state_dict(),
                       f'{log_dir}/ep{epoch+1:03d}-loss{avg_train_loss:.3f}-val_loss{avg_val_loss:.3f}.pth')

        if len(loss_history.val_loss) <= 1 or avg_val_loss <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), f'{log_dir}/best_epoch_weights.pth')

        torch.save(model.state_dict(), f'{log_dir}/last.pth')

    cleanup_periodic_checkpoints(log_dir)
    print(f"CFD-MAE detection training completed!")


if __name__ == '__main__':
    train()
