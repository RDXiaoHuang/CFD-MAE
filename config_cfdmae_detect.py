import os

# =====================================
# CFD-MAE Downstream Detection Configuration
# Frozen CFD-MAE encoders + configurable Ultralytics YOLO detector
# =====================================
# IMPORTANT: Training hyperparameters MUST match baseline config for fair comparison


def _env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ('1', 'true', 'yes', 'on')


def _default_cfdmae_pretrain_path(dataset):
    release_path = f'checkpoint/cfdmae_pretrain_{dataset}.pth'
    if os.path.exists(release_path):
        return release_path
    return f'logs/cfdmae_pretrain/{dataset}/best.pth'


Cuda = True
seed = 114514
reproducible = True
distributed = False
sync_bn = False
fp16 = False

data_name = os.getenv('DATA_NAME', 'rtts')
_dataset_aliases = {
    'foggy': 'voc_foggy',
    'rain': 'voc_rain',
}
data_name = _dataset_aliases.get(data_name, data_name)
model_name = os.getenv('MODEL_NAME', 'cfdmae_yolo26n')
classes_path = 'model_data/unified_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
anchors_mask = [[3, 4, 5], [1, 2, 3]]

# =====================================
# Pretrained Weights
# =====================================
# CFD-MAE pretrained encoder weights
cfdmae_pretrained_path = os.getenv('CFDMAE_PRETRAINED_PATH', _default_cfdmae_pretrain_path(data_name))
# YOLO pretrained weights (COCO)
yolo_pretrained_path = os.getenv('YOLO_PRETRAINED_PATH', 'model_data/yolo26n_Ultralytics.pt')

# =====================================
# CFD-MAE Architecture (must match pretrain config)
# =====================================
img_size = 640
patch_size = 16
embed_dim = 256
encoder_depth = 6
num_heads = 8
num_levels = 2

# Ablation mode:
#   'full'      — LF enhancement + HF-guided DASM
#   'no_hf'     — remove HF prior guidance (LF enhancement remains)
#   'no_lf'     — remove LF enhancement (HF prior may still guide DASM)
#   'no_lfghe'  — disable LF-guided HF prior regularization
#   'no_dasm'   — disable DASM only
# Backward-compatible aliases:
#   'none' -> 'full'
#   'no_hf_mae' -> 'no_hf'
#   'no_lf_mae' -> 'no_lf'
ablation_mode = os.getenv('ABLATION_MODE', 'full')


use_dasm = _env_flag(
    'USE_DASM',
    _env_flag('USE_DEGRADATION_SUPPRESSION', _env_flag('USE_FGR_PSA_NECK', True))
)
dasm_hidden = int(os.getenv('DASM_HIDDEN', os.getenv('DEGRADATION_SUPPRESSION_HIDDEN', os.getenv('FGR_PSA_HIDDEN', '64'))))
dasm_alpha = float(os.getenv('DASM_ALPHA', os.getenv('DEGRADATION_SUPPRESSION_ALPHA', os.getenv('FGR_PSA_ALPHA', '0.02'))))
dasm_min_keep = float(os.getenv('DASM_MIN_KEEP', os.getenv('DEGRADATION_SUPPRESSION_MIN_KEEP', '0.95')))
dasm_local_attention = os.getenv('DASM_LOCAL_ATTENTION', 'psa').lower()
dasm_long_attention = os.getenv('DASM_LONG_ATTENTION', 'lka').lower()
dasm_replacement = 'dasm'
hf_fusion_mode = os.getenv('HF_FUSION_MODE', 'prior').lower()
hf_fusion_gate = os.getenv('HF_FUSION_GATE', 'spatial').lower()
hf_fusion_hidden = int(os.getenv('HF_FUSION_HIDDEN', '16'))

# Backward-compatible aliases
use_degradation_suppression = use_dasm
use_fgr_psa_neck = use_dasm
degradation_suppression_hidden = dasm_hidden
degradation_suppression_alpha = dasm_alpha
degradation_suppression_min_keep = dasm_min_keep
fgr_psa_hidden = dasm_hidden
fgr_psa_alpha = dasm_alpha

cfdmae_diag_mode = os.getenv('CFDMAE_DIAG_MODE', 'normal').lower()
cfdmae_reconstruction_mode = os.getenv('CFDMAE_RECON_MODE', os.getenv('PRETRAIN_RECON_MODE', 'cross')).lower()


# =====================================
# Training Settings (EXACTLY same as baseline for fair comparison)
# =====================================
input_shape = [640, 640]
Init_Epoch = int(os.getenv('INIT_EPOCH', '0'))
UnFreeze_Epoch = int(os.getenv('UNFREEZE_EPOCH', '200'))
Run_Stop_Epoch = int(os.getenv('RUN_STOP_EPOCH', str(UnFreeze_Epoch)))
batch_size = int(os.getenv('BATCH_SIZE', '16'))
Init_lr = 1e-3
Min_lr = Init_lr * 0.01
optimizer_type = "sgd"
momentum = 0.937
weight_decay = 5e-4
lr_decay_type = "cos"
save_period = 10
eval_flag = True
eval_period = 1
num_workers = int(os.getenv('NUM_WORKERS', '8'))
grad_clip = 10.0

# =====================================
# Data Paths
# =====================================
train_annotation_path = f'dataset_split/train_{data_name}.txt'
val_annotation_path = f'dataset_split/test_{data_name}.txt'


def _normalize_ablation_mode(mode):
    aliases = {
        'none': 'full',
        'no_hf_mae': 'no_hf',
        'no_lf_mae': 'no_lf',
        'no_esm': 'no_dasm',
    }
    return aliases.get(mode, mode)


def _validate_choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got: {value}")
    return value


ablation_mode = _normalize_ablation_mode(ablation_mode)
cfdmae_diag_mode = _validate_choice('CFDMAE_DIAG_MODE', cfdmae_diag_mode, {'normal', 'detector_only', 'identity_fusion', 'images_only'})
cfdmae_reconstruction_mode = _validate_choice('CFDMAE_RECON_MODE', cfdmae_reconstruction_mode, {'cross', 'same'})
dasm_local_attention = _validate_choice('DASM_LOCAL_ATTENTION', dasm_local_attention, {'psa', 'cbam', 'se', 'none'})
dasm_long_attention = _validate_choice('DASM_LONG_ATTENTION', dasm_long_attention, {'lka', 'dilated', 'none'})
hf_fusion_mode = _validate_choice('HF_FUSION_MODE', hf_fusion_mode, {'prior', 'replace', 'adaptive', 'direct'})
hf_fusion_gate = _validate_choice('HF_FUSION_GATE', hf_fusion_gate, {'spatial', 'global'})


def _derive_variant_name():
    manual_name = os.getenv('VARIANT_NAME')
    if manual_name:
        return manual_name
    return ablation_mode


variant_name = _derive_variant_name()

# Save directory
save_dir = os.getenv('SAVE_DIR_OVERRIDE', f'logs/cfdmae/{model_name}_{variant_name}-{data_name}')
