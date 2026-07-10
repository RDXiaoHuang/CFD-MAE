Cuda = True
seed = 114514
reproducible = True
distributed = False
sync_bn = False
fp16 = False

import os

data_name = os.getenv('DATA_NAME', 'rtts')
_dataset_aliases = {
    'foggy': 'voc_foggy',
    'rain': 'voc_rain',
}
data_name = _dataset_aliases.get(data_name, data_name)
model_name = os.getenv('MODEL_NAME', 'yolo26n_ultra')
classes_path = os.getenv('CLASSES_PATH', 'model_data/unified_classes.txt')
anchors_path = os.getenv('ANCHORS_PATH', 'model_data/yolo_anchors.txt')
anchors_mask = [[3, 4, 5], [1, 2, 3]]

pretrained = os.getenv('PRETRAINED', '1').lower() in {'1', 'true', 'yes', 'on'}
pretrained_path = os.getenv('PRETRAINED_PATH', 'model_data/yolo26n_Ultralytics.pt')

training_mode = 'standard'

input_shape = [640, 640]
Init_Epoch = 0
UnFreeze_Epoch = 200
Run_Stop_Epoch = int(os.getenv('RUN_STOP_EPOCH', str(UnFreeze_Epoch)))
batch_size = int(os.getenv('BATCH_SIZE', '16'))
Init_lr = 1e-3
Min_lr = Init_lr * 0.01
optimizer_type = "sgd"
momentum = 0.937
weight_decay = 5e-4
lr_decay_type = "cos"
grad_clip = 10.0

save_period = 10
eval_flag = True
eval_period = 1
num_workers = int(os.getenv('NUM_WORKERS', '8'))

train_annotation_path = os.getenv('TRAIN_ANNOTATION_PATH', f'dataset_split/train_{data_name}.txt')
val_annotation_path = os.getenv('VAL_ANNOTATION_PATH', f'dataset_split/test_{data_name}.txt')
save_dir = os.getenv('SAVE_DIR', f'logs/cfdmae/cfdmae_{model_name}_baseline-{data_name}')
