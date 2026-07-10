import copy
import os

# =====================================
# CFD-MAE Pretraining Configuration
# Cross-Frequency Decoupled Masked Autoencoder
# =====================================

Cuda = True
seed = 114514
reproducible = True
fp16 = True

# =====================================
# Data Settings (single-domain pretraining)
# =====================================
# Supported: 'rtts', 'voc_foggy', 'voc_rain', 'exdark'
data_name = os.getenv('DATA_NAME', 'exdark')
_dataset_aliases = {
    'foggy': 'voc_foggy',
    'rain': 'voc_rain',
}
data_name = _dataset_aliases.get(data_name, data_name)

# Per-domain image directories
_domain_image_dirs = {
    'rtts': ['dataset/RTTS/VOC2007/JPEGImages'],
    'voc_foggy': ['dataset/VOC_Foggy/VOC2007/JPEGImages'],
    'voc_rain': ['dataset/VOC_Rain/VOC2007/JPEGImages'],
    'exdark': [
        'dataset/Exdark/VOC2007/JPEGImages',
        'dataset/ExDark/VOC2007/JPEGImages',
    ],
}
train_image_dirs = _domain_image_dirs.get(data_name, _domain_image_dirs['rtts'])

# Fallback: annotation file
train_annotation_path = f'dataset_split/train_{data_name}.txt'

# =====================================
# Model Architecture
# =====================================
img_size = 640
patch_size = 16
embed_dim = 256
encoder_depth = 6
decoder_embed_dim = 128
decoder_depth = 2
num_heads = 8
mask_ratio = 0.60       # Reduced for low-light (less info per patch)
num_levels = 2          # Laplacian pyramid levels
self_recon_weight = 0.1 # Auxiliary self-reconstruction loss weight
hf_loss_weight = 1.5    # Upweighted to compensate for small HF signal amplitude
pretrain_reconstruction_mode = os.getenv('PRETRAIN_RECON_MODE', 'cross').lower()

_PRETRAIN_LOSS_CONFIGS = {
    'rtts': {
        'lf': {'use_mask': False},
        'hf': {'use_mask': True, 'fft_weight': 1.0},
    },
    'voc_foggy': {
        'lf': {'use_mask': False},
        'hf': {'use_mask': True, 'fft_weight': 1.0},
    },
    'voc_rain': {
        'lf': {'use_mask': False},
        'hf': {'use_mask': True, 'fft_weight': 1.0},
    },
    'exdark': {
        'lf': {'use_mask': False},
        'hf': {'use_mask': False, 'fft_weight': 2.0},  # increased fft_weight to drive HF learning
    },
}


def get_pretrain_loss_config(dataset_name):
    dataset_name = _dataset_aliases.get(dataset_name, dataset_name)
    return copy.deepcopy(_PRETRAIN_LOSS_CONFIGS.get(dataset_name, _PRETRAIN_LOSS_CONFIGS['rtts']))


pretrain_loss_config = get_pretrain_loss_config(data_name)
if pretrain_reconstruction_mode not in {'cross', 'same'}:
    raise ValueError(f"PRETRAIN_RECON_MODE must be 'cross' or 'same', got: {pretrain_reconstruction_mode}")

# =====================================
# Training Settings
# =====================================
epochs = int(os.getenv('PRETRAIN_EPOCHS', '200'))
batch_size = int(os.getenv('PRETRAIN_BATCH_SIZE', '16'))
lr = 1.5e-4
min_lr = 1e-6
weight_decay = 0.05
warmup_epochs = 10
optimizer_type = 'adamw'
lr_decay_type = 'cos'

# =====================================
# Logging & Saving
# =====================================
save_period = 20
log_interval = 50
_default_save_root = 'logs/cfdmae_pretrain_same' if pretrain_reconstruction_mode == 'same' else 'logs/cfdmae_pretrain'
save_dir = os.getenv('SAVE_DIR', f'{_default_save_root}/{data_name}')
num_workers = int(os.getenv('NUM_WORKERS', '8'))
