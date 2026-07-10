import os

import scipy.signal
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox
from .utils_map import get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer     = SummaryWriter(self.log_dir)

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.005, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        # Note: Confidence threshold lowered to 0.005 to capture more predictions during early training for evaluation
        super(EvalCallback, self).__init__()

        self.net                = net
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period

        self.maps               = []
        self.epoches            = []
        self.best_map           = 0.0  # Track best mAP
        self.best_map_epoch     = 0    # Epoch when best mAP was achieved
        self.no_improve_count   = 0    # Consecutive eval cycles without mAP improvement

        if self.eval_flag:
            pass

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            # Set model to evaluation mode
            self.net.eval()
            
            # YOLOv11 anchor-free inference logic
            # Check if it's a teacher-student model
            if hasattr(self.net, 'module'):
                # DataParallel wrapper
                actual_model = self.net.module
            else:
                actual_model = self.net
            
            # Forward pass
            model_output = self.net(images)
            if isinstance(model_output, dict):
                preds = model_output['predictions']
            elif isinstance(model_output, tuple):
                preds, _ = model_output
            else:
                preds = model_output
            
            # preds shape: [batch_size, 4+num_classes, num_anchors]
            # First 4 channels are bbox coordinates (already scaled by stride)
            # Remaining channels are class probabilities (already sigmoid)
            
            preds = preds[0]  # First batch: [4+num_classes, num_anchors]
            
            # Transpose to [num_anchors, 4+num_classes]
            preds = preds.transpose(0, 1)
            
            # Extract bounding boxes and class scores
            boxes_xywh = preds[:, :4]  # [num_anchors, 4] - coordinates in input image scale
            cls_scores = preds[:, 4:]  # [num_anchors, num_classes] - already sigmoid
            
            # Calculate confidence
            cls_conf, cls_ids = torch.max(cls_scores, dim=1)
            
            # Apply confidence threshold
            mask = cls_conf > self.confidence
            
            # Debug: Print statistics for first few images during early epochs
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
            
            if self._debug_count <= 3:  # Only for first 3 images
                print(f'\n[DEBUG] Image {image_id}:')
                print(f'  Total anchors: {cls_conf.shape[0]}')
                print(f'  Max confidence: {cls_conf.max().item():.6f}')
                print(f'  Mean confidence: {cls_conf.mean().item():.6f}')
                print(f'  Predictions > {self.confidence}: {mask.sum().item()}')
            
            if mask.sum() == 0:
                f.close()
                return 
                
            boxes_xywh = boxes_xywh[mask]
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]
            
            # Boxes are already in input image coordinates (scaled by stride in detect head)
            # Need to scale to original image size
            if self.letterbox_image:
                # For letterbox, need to account for padding
                scale = min(self.input_shape[1] / image_shape[1], self.input_shape[0] / image_shape[0])
                pad_w = (self.input_shape[1] - image_shape[1] * scale) / 2
                pad_h = (self.input_shape[0] - image_shape[0] * scale) / 2
                
                # Convert center coordinates
                boxes_xywh[:, 0] = (boxes_xywh[:, 0] - pad_w) / scale
                boxes_xywh[:, 1] = (boxes_xywh[:, 1] - pad_h) / scale
                boxes_xywh[:, 2] = boxes_xywh[:, 2] / scale
                boxes_xywh[:, 3] = boxes_xywh[:, 3] / scale
            else:
                # Direct scaling
                scale_x = image_shape[1] / self.input_shape[1]
                scale_y = image_shape[0] / self.input_shape[0]
                boxes_xywh[:, 0] = boxes_xywh[:, 0] * scale_x
                boxes_xywh[:, 1] = boxes_xywh[:, 1] * scale_y
                boxes_xywh[:, 2] = boxes_xywh[:, 2] * scale_x
                boxes_xywh[:, 3] = boxes_xywh[:, 3] * scale_y
            
            # Convert to xyxy format
            xyxy = torch.zeros_like(boxes_xywh)
            xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
            xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
            xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
            xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
            
            # NMS
            from torchvision.ops import nms
            keep = nms(xyxy, cls_conf, self.nms_iou)
            
            if len(keep) == 0:
                f.close()
                return 
                
            top_boxes = xyxy[keep].cpu().numpy()
            top_conf = cls_conf[keep].cpu().numpy()
            top_label = cls_ids[keep].cpu().numpy()
            
            # Sort by confidence and take top detections
            conf_indices = np.argsort(top_conf)[::-1][:self.max_boxes]
            
            for idx in conf_indices:
                x1, y1, x2, y2 = top_boxes[idx]
                class_id = int(top_label[idx])
                score = float(top_conf[idx])
                
                f.write("%s %s %s %s %s %s\n" % (self.class_names[class_id], str(score),
                                           str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2))))

        f.close()

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))

            for annotation_line in self.val_lines:
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                image       = Image.open(line[0])
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                # Try to read difficult information from XML
                xml_path = self._get_xml_path(line[0])
                difficult_dict = self._parse_difficult_from_xml(xml_path) if xml_path else {}
                
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for idx, box in enumerate(gt_boxes):
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        # Add difficult flag if available
                        is_difficult = difficult_dict.get(idx, False)
                        if is_difficult:
                            new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                        else:
                            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            
            results_path = os.path.join(self.map_out_path, 'results')
            if not os.path.exists(results_path):
                os.makedirs(results_path, exist_ok=True)
            
            # Calculate mAP but don't save results.txt yet
            temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            # Only keep results.txt if this is the best mAP so far
            if temp_map > self.best_map:
                self.best_map = temp_map
                self.best_map_epoch = epoch
                self.no_improve_count = 0
                if hasattr(model_eval, "module"):
                    torch.save(model_eval.module.state_dict(), os.path.join(self.log_dir, "best_map_weights.pth"))
                else:
                    torch.save(model_eval.state_dict(), os.path.join(self.log_dir, "best_map_weights.pth"))
                print(f'Epoch: {epoch} || mAP: {temp_map:.4f} || Best mAP: {self.best_map:.4f} (epoch {self.best_map_epoch}) *NEW BEST*')
            else:
                # Remove results.txt since it's not the best
                results_file = os.path.join(results_path, 'results.txt')
                if os.path.exists(results_file):
                    os.remove(results_file)
                self.no_improve_count += 1
                print(f'Epoch: {epoch} || mAP: {temp_map:.4f} || Best mAP: {self.best_map:.4f} (epoch {self.best_map_epoch})')

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

    def _get_xml_path(self, image_path):
        """
        Get XML annotation path from image path.
        Supports RTTS dataset structure.
        """
        # RTTS structure: dataset/RTTS/VOC2007/JPEGImages/xxx.jpg
        #              -> dataset/RTTS/VOC2007/Annotations/xxx.xml
        if 'RTTS' in image_path and 'JPEGImages' in image_path:
            xml_path = image_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml').replace('.png', '.xml')
            if os.path.exists(xml_path):
                return xml_path
        
        # Try other common structures
        base_dir = os.path.dirname(os.path.dirname(image_path))
        image_name = os.path.basename(image_path).rsplit('.', 1)[0]
        
        possible_paths = [
            os.path.join(base_dir, 'Annotations', image_name + '.xml'),
            os.path.join(base_dir, 'annotations', image_name + '.xml'),
            os.path.join(os.path.dirname(image_path), '..', 'Annotations', image_name + '.xml'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _parse_difficult_from_xml(self, xml_path):
        """
        Parse difficult flags from VOC XML annotation.
        Returns dict: {object_index: is_difficult}
        """
        if not xml_path or not os.path.exists(xml_path):
            return {}
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            difficult_dict = {}
            for idx, obj in enumerate(root.findall('object')):
                difficult = obj.find('difficult')
                if difficult is not None:
                    difficult_dict[idx] = (int(difficult.text) == 1)
                else:
                    difficult_dict[idx] = False
            
            return difficult_dict
        except Exception as e:
            print(f"Warning: Failed to parse XML {xml_path}: {e}")
            return {}
