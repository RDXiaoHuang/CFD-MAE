# CFD-MAE

This is a compact GitHub release of **CFD-MAE** for object detection under degraded visual conditions, including haze/fog, low-light, and rain scenes.

## Graphical Abstract
src="https://github.com/RDXiaoHuang/CFD-MAE/blob/master/figs/Graphical_abstract.jpg?raw=true">

The release keeps the main reproducible pipeline only:

- cross-frequency masked autoencoder pretraining;
- LF image enhancement and HF-prior-guided DASM;
- YOLOv26n baseline training;
- CFD-MAE + YOLOv26n training and VOC AP@50 testing.

Extra ablation runners, external detector baselines, temporary logs, and paper-table scripts are intentionally removed.

## Environment

```bash
conda create -n cfdmae python=3.9
conda activate cfdmae
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default `pip` resolver is not suitable.

## Data Layout

Place datasets under the repository root:

```text
CFD-MAE/
├── dataset/
│   ├── RTTS/VOC2007/JPEGImages/
│   ├── Exdark/VOC2007/JPEGImages/
│   └── VOC_Rain/VOC2007/JPEGImages/
└── dataset_split/
    ├── train_rtts.txt
    ├── test_rtts.txt
    ├── train_exdark.txt
    ├── test_exdark.txt
    ├── train_voc_rain.txt
    └── test_voc_rain.txt
```

Each split line uses:

```text
image_path x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...
```

The released split files use five classes:

```text
0 person
1 bicycle
2 car
3 motorbike
4 bus
```

## Checkpoints

For local use, this prepared copy contains the following files:

```text
checkpoint/cfdmae_rtts.pth
checkpoint/cfdmae_exdark.pth
checkpoint/cfdmae_voc_rain.pth
checkpoint/cfdmae_pretrain_rtts.pth
checkpoint/cfdmae_pretrain_exdark.pth
checkpoint/cfdmae_pretrain_voc_rain.pth
model_data/yolo26n_Ultralytics.pt
```

For GitHub publication, keep large `*.pth`/`*.pt` files out of git and upload them as release assets or provide an external download link. The `.gitignore` already excludes these large files. Checksums are listed in `checkpoint/SHA256SUMS`.

## Pretraining

Run CFD-MAE pretraining for each dataset:

```bash
DATA_NAME=rtts python train_cfdmae_pretrain.py
DATA_NAME=exdark python train_cfdmae_pretrain.py
DATA_NAME=voc_rain python train_cfdmae_pretrain.py
```

Useful overrides:

```bash
DATA_NAME=rtts PRETRAIN_EPOCHS=200 PRETRAIN_BATCH_SIZE=16 NUM_WORKERS=8 python train_cfdmae_pretrain.py
```

Outputs:

```text
logs/cfdmae_pretrain/<dataset>/best.pth
logs/cfdmae_pretrain/<dataset>/last.pth
```

## YOLOv26n Baseline

```bash
DATA_NAME=rtts python train_ultralytics.py
DATA_NAME=exdark python train_ultralytics.py
DATA_NAME=voc_rain python train_ultralytics.py
```

Useful overrides:

```bash
DATA_NAME=rtts \
PRETRAINED_PATH=model_data/yolo26n_Ultralytics.pt \
RUN_STOP_EPOCH=200 \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
python train_ultralytics.py
```

## CFD-MAE Training

Detection training uses the released pretraining weights in `checkpoint/` by default when they exist:

```bash
DATA_NAME=rtts python train_cfdmae_detect.py
DATA_NAME=exdark python train_cfdmae_detect.py
DATA_NAME=voc_rain python train_cfdmae_detect.py
```

Explicit form:

```bash
DATA_NAME=rtts \
CFDMAE_PRETRAINED_PATH=checkpoint/cfdmae_pretrain_rtts.pth \
YOLO_PRETRAINED_PATH=model_data/yolo26n_Ultralytics.pt \
RUN_STOP_EPOCH=200 \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
python train_cfdmae_detect.py
```

Outputs:

```text
logs/cfdmae/cfdmae_yolo26n_full-<dataset>/
```

## Testing

Evaluate the released CFD-MAE detector checkpoints:

```bash
python test.py --data rtts
python test.py --data exdark
python test.py --data voc_rain
```

Useful smoke test:

```bash
python test.py --data rtts --max-images 16
```

By default, `test.py` reads `checkpoint/cfdmae_<dataset>.pth`, uses `dataset_split/test_<dataset>.txt`, and writes VOC-format evaluation files to `runs/test/<dataset>/`.

## Main Files

```text
config_cfdmae_pretrain.py
config_cfdmae_detect.py
config_ultralytics.py
train_cfdmae_pretrain.py
train_cfdmae_detect.py
train_ultralytics.py
test.py
nets/
utils/
dataset_split/
checkpoint/
model_data/
```

## Citation

The manuscript is currently under review. Citation information will be updated after publication.

## License

This repository is released for academic research and reproducibility.
