import os
import re

import torch
import numpy as np
from tqdm import tqdm

from utils.checkpoint_cleanup import cleanup_periodic_checkpoints, should_save_periodic_checkpoint
from utils.utils import get_lr


def fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0, time_str=None, input_shape=None):
    """Standard YOLO training for one epoch"""
    loss = 0
    val_loss = 0

    if input_shape is None:
        input_shape = [640, 640]

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
                images = images.cuda(local_rank)

                targets_list = []
                for ann in targets:
                    if isinstance(ann, np.ndarray):
                        ann = torch.from_numpy(ann).type(torch.FloatTensor)
                    targets_list.append(ann.cuda(local_rank))
                targets = targets_list

        targets_converted = []
        for target in targets:
            if target.numel() > 0:
                x_center, y_center, w, h, cls = target[:, 0], target[:, 1], target[:, 2], target[:, 3], target[:, 4]
                img_h, img_w = images.shape[2], images.shape[3]
                target_reordered = torch.stack([
                    cls, x_center * img_w, y_center * img_h, w * img_w, h * img_h
                ], dim=1)
                targets_converted.append(target_reordered)
            else:
                targets_converted.append(torch.zeros(0, 5, device=images.device))

        optimizer.zero_grad()

        if not fp16:
            outputs = model_train(images, return_features=False)
            loss_value, _ = yolo_loss(outputs, targets_converted, input_shape)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images, return_features=False)
                loss_value, _ = yolo_loss(outputs, targets_converted, input_shape)
            scaler.scale(loss_value).backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': loss / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
                images = images.cuda(local_rank)

                targets_list = []
                for ann in targets:
                    if isinstance(ann, np.ndarray):
                        ann = torch.from_numpy(ann).type(torch.FloatTensor)
                    elif not isinstance(ann, torch.Tensor):
                        ann = torch.tensor(ann).type(torch.FloatTensor)
                    targets_list.append(ann.cuda(local_rank))
                targets = targets_list

            targets_converted = []
            for target in targets:
                if target.numel() > 0:
                    x_center, y_center, w, h, cls = target[:, 0], target[:, 1], target[:, 2], target[:, 3], target[:, 4]
                    img_h, img_w = images.shape[2], images.shape[3]
                    target_reordered = torch.stack([
                        cls, x_center * img_w, y_center * img_h, w * img_w, h * img_h
                    ], dim=1)
                    targets_converted.append(target_reordered)
                else:
                    targets_converted.append(torch.zeros(0, 5, device=images.device))

            outputs = model_train(images, raw_output=True)
            loss_value, _ = yolo_loss(outputs, targets_converted, input_shape)

        val_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print(f'Epoch: {epoch + 1}/{Epoch}')
        print(f'Total Loss: {loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        if should_save_periodic_checkpoint() and ((epoch + 1) % save_period == 0 or epoch + 1 == Epoch):
            torch.save(model.state_dict(), os.path.join(save_dir, f"ep{epoch+1:03d}-loss{loss/epoch_step:.3f}-val_loss{val_loss/epoch_step_val:.3f}.pth"))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))
        if epoch + 1 == Epoch:
            cleanup_periodic_checkpoints(save_dir)


def fit_one_epoch_baseline(model, optimizer, epoch, train_loader, val_loader, device, save_dir, eval_flag=False):
    """Simplified training function for baseline models"""
    model.train()
    total_loss = 0

    print(f'Epoch {epoch + 1} - Training')
    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}', postfix=dict, mininterval=0.3)

    for iteration, batch in enumerate(train_loader):
        images, targets = batch[0], batch[1]

        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        images = images.to(device)

        targets_list = []
        for ann in targets:
            if isinstance(ann, np.ndarray):
                ann = torch.from_numpy(ann).type(torch.FloatTensor)
            targets_list.append(ann.to(device))
        targets = targets_list

        optimizer.zero_grad()
        loss = model(images, targets)

        if isinstance(loss, dict):
            loss = sum(loss.values())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(**{'loss': total_loss / (iteration + 1)})
        pbar.update(1)

    pbar.close()
    avg_train_loss = total_loss / len(train_loader)

    if eval_flag:
        model.eval()
        val_loss = 0

        print('Validation')
        pbar = tqdm(total=len(val_loader), desc='Validation', postfix=dict, mininterval=0.3)

        with torch.no_grad():
            for iteration, batch in enumerate(val_loader):
                images, targets = batch[0], batch[1]

                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
                images = images.to(device)

                targets_list = []
                for ann in targets:
                    if isinstance(ann, np.ndarray):
                        ann = torch.from_numpy(ann).type(torch.FloatTensor)
                    targets_list.append(ann.to(device))
                targets = targets_list

                loss = model(images, targets)

                if isinstance(loss, dict):
                    loss = sum(loss.values())

                val_loss += loss.item()

                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

        pbar.close()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    else:
        print(f'Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}')

    return avg_train_loss
