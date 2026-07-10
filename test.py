#!/usr/bin/env python3
"""Evaluate released CFD-MAE detector checkpoints with VOC AP@50."""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config_cfdmae_detect as config
from nets.cfdmae import CFDMAEDetector
from utils.callbacks import EvalCallback
from utils.utils import get_anchors, get_classes, seed_everything
from utils.utils_map import get_map


DATA_ALIASES = {
    "foggy": "voc_foggy",
    "rain": "voc_rain",
}


def normalize_data_name(name):
    return DATA_ALIASES.get(name, name)


def default_detector_weight(data_name):
    return Path("checkpoint") / f"cfdmae_{data_name}.pth"


def default_pretrain_weight(data_name):
    return Path("checkpoint") / f"cfdmae_pretrain_{data_name}.pth"


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state = checkpoint.state_dict()
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def read_annotation_lines(path):
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise RuntimeError(f"empty annotation file: {path}")
    return lines


def write_ground_truth(evaluator, image_path, boxes, output_path):
    xml_path = evaluator._get_xml_path(image_path)
    difficult = evaluator._parse_difficult_from_xml(xml_path) if xml_path else {}
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, box in enumerate(boxes):
            left, top, right, bottom, obj = box
            suffix = " difficult" if difficult.get(index, False) else ""
            handle.write(
                f"{evaluator.class_names[int(obj)]} "
                f"{int(left)} {int(top)} {int(right)} {int(bottom)}{suffix}\n"
            )


def build_model(args, num_classes):
    pretrain_path = args.pretrained_cfdmae
    if pretrain_path and not Path(pretrain_path).exists():
        print(f"[test] pretrain init skipped; file not found: {pretrain_path}")
        pretrain_path = None
    if not Path(args.yolo_pretrained).exists():
        raise FileNotFoundError(
            f"YOLOv26 initialization weight not found: {args.yolo_pretrained}. "
            "Place yolo26n_Ultralytics.pt under model_data/ or pass --yolo-pretrained."
        )

    model = CFDMAEDetector(
        num_classes=num_classes,
        pretrained_cfdmae_path=pretrain_path,
        yolo_pretrained_path=args.yolo_pretrained,
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
    )

    state = load_state_dict(args.weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[test] missing keys: {len(missing)}")
    if unexpected:
        print(f"[test] unexpected keys: {len(unexpected)}")
    return model


def evaluate(args):
    data_name = normalize_data_name(args.data)
    args.weights = args.weights or str(default_detector_weight(data_name))
    args.pretrained_cfdmae = args.pretrained_cfdmae or str(default_pretrain_weight(data_name))
    args.annotation = args.annotation or f"dataset_split/test_{data_name}.txt"
    args.map_out = args.map_out or str(Path("runs") / "test" / data_name)

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"detector checkpoint not found: {args.weights}")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[test] CUDA requested but unavailable; using CPU.")

    seed_everything(config.seed)
    class_names, num_classes = get_classes(config.classes_path)
    anchors, _ = get_anchors(config.anchors_path)
    val_lines = read_annotation_lines(args.annotation)
    if args.max_images > 0:
        val_lines = val_lines[: args.max_images]

    model = build_model(args, num_classes)
    if use_cuda:
        model = model.cuda()
    model.eval()

    map_out = Path(args.map_out)
    if map_out.exists():
        shutil.rmtree(map_out)
    (map_out / "ground-truth").mkdir(parents=True, exist_ok=True)
    (map_out / "detection-results").mkdir(parents=True, exist_ok=True)

    evaluator = EvalCallback(
        model,
        config.input_shape,
        anchors,
        config.anchors_mask,
        class_names,
        num_classes,
        val_lines,
        str(map_out),
        use_cuda,
        map_out_path=str(map_out),
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        letterbox_image=args.letterbox_image,
        MINOVERLAP=args.min_overlap,
        eval_flag=False,
    )
    evaluator._debug_count = 3

    for line in tqdm(val_lines, desc=f"Evaluate {data_name}"):
        parts = line.split()
        image_path = parts[0]
        image_id = Path(image_path).stem
        image = Image.open(image_path).convert("RGB")
        boxes = np.array([list(map(int, item.split(","))) for item in parts[1:]], dtype=np.int64)
        if boxes.size == 0:
            boxes = boxes.reshape(0, 5)
        evaluator.get_map_txt(image_id, image, class_names, str(map_out))
        write_ground_truth(evaluator, image_path, boxes, map_out / "ground-truth" / f"{image_id}.txt")

    map50 = get_map(args.min_overlap, False, path=str(map_out))
    print(f"[test] dataset={data_name} images={len(val_lines)} mAP@{args.min_overlap:.2f}={map50:.4f} ({map50 * 100:.2f}%)")
    print(f"[test] outputs saved to: {map_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CFD-MAE detector checkpoints.")
    parser.add_argument("--data", default="rtts", choices=["rtts", "exdark", "voc_rain", "rain"])
    parser.add_argument("--weights", default="", help="Detector checkpoint. Defaults to checkpoint/cfdmae_<data>.pth.")
    parser.add_argument("--pretrained-cfdmae", default="", help="Optional CFD-MAE pretrain init weight.")
    parser.add_argument("--yolo-pretrained", default="model_data/yolo26n_Ultralytics.pt")
    parser.add_argument("--annotation", default="", help="Evaluation split txt. Defaults to dataset_split/test_<data>.txt.")
    parser.add_argument("--map-out", default="", help="Output directory for VOC-format evaluation files.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--confidence", type=float, default=0.005)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for a smoke test.")
    parser.add_argument("--letterbox-image", action="store_true", default=True)
    parser.add_argument("--no-letterbox-image", dest="letterbox_image", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
