"""YOLOv26 baseline training script used for CFD-MAE comparison."""
import os
import re
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config_ultralytics as config
from config_ultralytics import (
    Cuda, seed, reproducible, distributed, sync_bn, fp16,
    data_name, model_name, classes_path, anchors_path, anchors_mask,
    pretrained, pretrained_path, input_shape, Init_Epoch, UnFreeze_Epoch,
    batch_size, Init_lr, Min_lr, optimizer_type, momentum, weight_decay,
    lr_decay_type, save_period, eval_flag, eval_period, num_workers,
    train_annotation_path, val_annotation_path, save_dir
)

from nets.ultralytics.yolo26_wrapper import yolo_v26
from nets.yolo_training import get_lr_scheduler, set_optimizer_lr, DetectionLossYOLO26
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


run_stop_epoch = min(getattr(config, 'Run_Stop_Epoch', UnFreeze_Epoch), UnFreeze_Epoch)


def find_resume_epoch(target_dir):
    loss_file = os.path.join(target_dir, 'epoch_loss.txt')
    if not os.path.exists(loss_file):
        return 0
    with open(loss_file) as f:
        return sum(1 for line in f if line.strip())


def find_latest_epoch_weight(target_dir):
    latest_epoch = 0
    latest_path = None
    pattern = re.compile(r'^ep(\d+)-loss.*\.pth$')
    if not os.path.isdir(target_dir):
        return latest_epoch, latest_path

    for name in os.listdir(target_dir):
        match = pattern.match(name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = os.path.join(target_dir, name)
    return latest_epoch, latest_path


if __name__ == "__main__":
    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    class_names, num_classes = get_classes(classes_path)
    anchors, _ = get_anchors(anchors_path)

    if "yolo26" not in model_name.lower() and pretrained and "yolo26" not in pretrained_path.lower():
        raise ValueError(
            "This reproducibility release keeps the YOLOv26 baseline only. "
            "Set MODEL_NAME=yolo26n and PRETRAINED_PATH=model_data/yolo26n_Ultralytics.pt."
        )
    model = yolo_v26(num_cls=num_classes, pretrained=pretrained_path if pretrained else None)

    yolo_loss = DetectionLossYOLO26(model, num_classes)

    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            model = model.cuda()
            model_train = torch.nn.DataParallel(model)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = [line.strip() for line in f if line.strip()]
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            model_name=model_name, classes_path=classes_path, anchors_path=anchors_path, anchors_mask=anchors_mask,
            input_shape=input_shape, Init_Epoch=Init_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            batch_size=batch_size, Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir, num_workers=num_workers,
            num_train=num_train, num_val=num_val
        )

    runtime_batch_size = batch_size

    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(runtime_batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(runtime_batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = {
        'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
        'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
    }[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, train=True)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, train=False)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if local_rank == 0:
        print(f'Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}')

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("The dataset is too small for training. Please expand the dataset.")

    resume_epoch = find_resume_epoch(save_dir)
    latest_epoch, latest_weight = find_latest_epoch_weight(save_dir)
    if latest_weight and latest_epoch >= resume_epoch:
        checkpoint = torch.load(latest_weight, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=False)
        resume_epoch = latest_epoch
        if local_rank == 0:
            print(f"[Baseline Detect] Resume from epoch {resume_epoch}: {latest_weight}")
    elif os.path.exists(os.path.join(save_dir, 'last.pth')):
        last_path = os.path.join(save_dir, 'last.pth')
        checkpoint = torch.load(last_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            model.load_state_dict(checkpoint, strict=False)
        if local_rank == 0:
            print(f"[Baseline Detect] Loaded last weights: {last_path}")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    gen_val = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    log_dir = save_dir
    if local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        for file_name in os.listdir(log_dir):
            if file_name.startswith('events.out.tfevents'):
                try:
                    os.remove(os.path.join(log_dir, file_name))
                except OSError:
                    pass

        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
        eval_callback = EvalCallback(
            model,
            input_shape,
            anchors,
            anchors_mask,
            class_names,
            num_classes,
            val_lines,
            log_dir,
            Cuda,
            map_out_path=os.path.join(log_dir, '.temp_map_out'),
            eval_flag=eval_flag,
            period=eval_period,
        )

        if resume_epoch > 0:
            with open(os.path.join(log_dir, 'epoch_loss.txt')) as f:
                loss_history.losses = [float(line.strip()) for line in f if line.strip()][:resume_epoch]
            with open(os.path.join(log_dir, 'epoch_val_loss.txt')) as f:
                loss_history.val_loss = [float(line.strip()) for line in f if line.strip()][:resume_epoch]
            map_path = os.path.join(log_dir, 'epoch_map.txt')
            if os.path.exists(map_path):
                with open(map_path) as f:
                    eval_callback.maps = [float(line.strip()) for line in f if line.strip()]
                eval_callback.epoches = list(range(eval_period, eval_period * len(eval_callback.maps) + 1, eval_period))
                if eval_callback.maps:
                    eval_callback.best_map = max(eval_callback.maps)
                    eval_callback.best_map_epoch = eval_callback.epoches[eval_callback.maps.index(eval_callback.best_map)]
    else:
        loss_history = None
        eval_callback = None

    if local_rank == 0:
        print(f'\n{"="*60}')
        print('Baseline YOLO Downstream Detection Training')
        print(f'{"="*60}')
        print(f'Dataset: {data_name}, Classes: {num_classes}')
        print(f'Planned epochs: {UnFreeze_Epoch}, Run stop epoch: {run_stop_epoch}, Batch: {runtime_batch_size}')
        print(f'Optimizer: {optimizer_type.upper()}, Init LR: {Init_lr_fit:.6f}, Min LR: {Min_lr_fit:.6f}')
        print(f'Model: {model_name}')
        print(f'YOLO pretrained: {pretrained_path}')
        print(f'Save dir: {log_dir}')
        print(f'{"="*60}\n')

    if resume_epoch >= run_stop_epoch:
        if local_rank == 0:
            print(f'[Baseline Detect] Save dir already reached target stop epoch {run_stop_epoch}. Nothing to run.')
    else:
        for epoch in range(resume_epoch, run_stop_epoch):
            if hasattr(gen.dataset, 'epoch_now'):
                gen.dataset.epoch_now = epoch

            if distributed:
                train_sampler.set_epoch(epoch)
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(
                model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                epoch_step_val, gen, gen_val, run_stop_epoch, Cuda, fp16, None, save_period, save_dir,
                local_rank, input_shape=input_shape
            )

            if distributed:
                dist.barrier()

    if local_rank == 0:
        print('\nTraining completed!')
        print(f'Best model saved to: {save_dir}')
