from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


def open_image(path):
    image_open = Image.open
    if getattr(image_open, "__module__", "") == "ultralytics.utils.patches":
        from ultralytics.utils.patches import _image_open
        return _image_open(path)
    return image_open(path)


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, train, mosaic_prob=0.6, mosaic_end_epoch=160, data_name='rtts'):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.train              = train
        self.mosaic_prob        = mosaic_prob
        self.mosaic_end_epoch   = mosaic_end_epoch
        self.data_name          = data_name

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)

        # Weather-specific augmentation parameters
        if data_name == 'snow':
            self.aug_params = {'hue': 0.02, 'sat': 0.3, 'val': 0.2}
        elif data_name == 'exdark':
            self.aug_params = {'hue': 0.0, 'sat': 0.0, 'val': 0.2}
        else:
            self.aug_params = {'hue': 0.1, 'sat': 0.7, 'val': 0.4}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        use_mosaic = (self.train and np.random.rand() < self.mosaic_prob and self.epoch_now < self.mosaic_end_epoch)

        if use_mosaic:
            image, box = self.get_mosaic_data(index)
            image_array = np.array(image, dtype=np.float32)
        else:
            image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape,
                                               hue=self.aug_params['hue'],
                                               sat=self.aug_params['sat'],
                                               val=self.aug_params['val'],
                                               random=self.train)
            image_array = np.array(image, dtype=np.float32)

        # Apply preprocessing and transpose to (C, H, W) format
        image = np.transpose(preprocess_input(image_array), (2, 0, 1))

        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        image   = open_image(line[0])
        image   = cvtColor(image)
        iw, ih  = image.size
        h, w    = input_shape
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]

            return image_data, box

        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)

        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype

        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox


    def get_mosaic_data(self, index):
        h, w = self.input_shape
        center_x = int(np.random.uniform(w * 0.3, w * 0.7))
        center_y = int(np.random.uniform(h * 0.3, h * 0.7))

        indices = [index] + [np.random.randint(0, self.length) for _ in range(3)]
        mosaic_img = np.full((h, w, 3), 114, dtype=np.uint8)
        mosaic_boxes = []

        for i, idx in enumerate(indices):
            img, box = self.get_random_data(self.annotation_lines[idx], self.input_shape, random=True)
            img = np.array(img, dtype=np.uint8)

            if i == 0:
                x1, y1, x2, y2 = max(0, center_x - w), max(0, center_y - h), center_x, center_y
                crop_x1, crop_y1, crop_x2, crop_y2 = w - (x2 - x1), h - (y2 - y1), w, h
            elif i == 1:
                x1, y1, x2, y2 = center_x, max(0, center_y - h), min(w, center_x + w), center_y
                crop_x1, crop_y1, crop_x2, crop_y2 = 0, h - (y2 - y1), x2 - x1, h
            elif i == 2:
                x1, y1, x2, y2 = max(0, center_x - w), center_y, center_x, min(h, center_y + h)
                crop_x1, crop_y1, crop_x2, crop_y2 = w - (x2 - x1), 0, w, y2 - y1
            else:
                x1, y1, x2, y2 = center_x, center_y, min(w, center_x + w), min(h, center_y + h)
                crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, x2 - x1, y2 - y1

            mosaic_img[y1:y2, x1:x2] = img[crop_y1:crop_y2, crop_x1:crop_x2]

            if len(box) > 0:
                box = np.array(box)
                box[:, [0, 2]] = box[:, [0, 2]] - crop_x1 + x1
                box[:, [1, 3]] = box[:, [1, 3]] - crop_y1 + y1
                box[:, [0, 2]] = np.clip(box[:, [0, 2]], x1, x2)
                box[:, [1, 3]] = np.clip(box[:, [1, 3]], y1, y2)
                box_w, box_h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 4, box_h > 4)]
                if len(box) > 0:
                    mosaic_boxes.append(box)

        mosaic_boxes = np.concatenate(mosaic_boxes, axis=0) if mosaic_boxes else np.array([])
        return mosaic_img, mosaic_boxes

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes
