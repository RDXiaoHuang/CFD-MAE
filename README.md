# CFD-MAE

This repository contains the official implementation of **CFD-MAE**, a deep learning framework for robust object detection under complex degraded visual conditions.

The corresponding manuscript has been submitted to **IEEE Transactions on Instrumentation and Measurement (IEEE TIM)** and is currently under review.

## Overview

CFD-MAE is designed to improve object detection performance in challenging scenes such as low-light, foggy, and rainy environments. The project includes training scripts, model components, dataset split files, pretrained checkpoints, and visualization results.

## Framework

We will provide the overall architecture of CFD-MAE as soon as possible.

## Results

Detection results are shown below.

![CFD-MAE Results](figs/result.png)

## Checkpoints

The pretrained weight files for the three datasets are listed below.

| Dataset | Checkpoint |
| --- | --- |
| ExDark | `checkpoint/cfdmae_exdark.pth` |
| RTTS | `checkpoint/cfdmae_rtts.pth` |
| VOC-Rain | `checkpoint/cfdmae_voc_rain.pth` |

## Repository Structure

```text
CFD-MAE/
├── checkpoint/
│   ├── cfdmae_exdark.pth
│   ├── cfdmae_rtts.pth
│   └── cfdmae_voc_rain.pth
├── dataset_split/
├── figs/
│   ├── cfd-mae.png
│   └── result.png
├── model_data/
├── nets/
├── utils/
├── train_cfdmae_detect.py
└── train_cfdmae_pretrain.py
```

## Training

Pretraining:

```bash
python train_cfdmae_pretrain.py
```

Detection training:

```bash
python train_cfdmae_detect.py
```

## Citation

The paper is currently under review. Citation information will be updated after publication.

## Notice

This repository is released for academic research and reproducibility. More details will be updated after the review process is completed.
