#!/usr/bin/env python3
"""Offline object detection CLI using torchvision's Faster R-CNN ResNet50 FPN v2 (COCO).

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image path/to/image.jpg --annotate
    python infer.py --folder path/to/images/ --threshold 0.6
    python infer.py --folder path/to/images/ --recursive --annotate

Expects the weights file to already be on disk (see the download snippet
using torch.hub.load_state_dict_from_url). No network access is needed.

Output (stdout) is JSON. For --image:
    {
      "objects": ["cup", "laptop"],
      "detections": [
        {"label": "cup", "score": 0.9731, "box": [x_min, y_min, x_max, y_max]},
        ...
      ]
    }
For --folder, the output is an object mapping each image path to the above.
Boxes are in pixel coordinates of the (EXIF-upright) image.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

WEIGHTS = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
DEFAULT_WEIGHTS_PATH = Path("./fasterrcnn_resnet50_fpn_v2_coco.pth")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
ANNOTATED_SUFFIX = "_annotated"
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#008080", "#9a6324", "#800000", "#808000", "#000075",
]


def parse_args():
    p = argparse.ArgumentParser(description="Detect objects with a local Faster R-CNN.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="path to a single image file")
    group.add_argument(
        "--folder", type=Path, help="path to a folder; every image in it is processed"
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="with --folder, also search subfolders",
    )
    p.add_argument(
        "--weights-path",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"path to the .pth weights file (default: {DEFAULT_WEIGHTS_PATH})",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="minimum confidence score to keep a detection (default: 0.5)",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help=f"save a copy of each image with boxes drawn, suffixed '{ANNOTATED_SUFFIX}', "
        "next to the original",
    )
    return p.parse_args()


def collect_images(folder: Path, recursive: bool):
    """Find images, skipping '*_annotated' files this script produced earlier."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        f for f in folder.glob(pattern)
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.stem.endswith(ANNOTATED_SUFFIX)
    )


def load_model(weights_path: Path, device: str):
    # weights_backbone=None stops torchvision from trying to download the
    # ImageNet backbone weights, which would fail offline.
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def detect(path: Path, model, preprocess, categories, device, threshold):
    """Run inference once on one image. Returns (PIL image, result dict)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    with torch.inference_mode():
        out = model([preprocess(img).to(device)])[0]

    detections = []
    for box, label_id, score in zip(
        out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()
    ):
        label = categories[label_id]
        if score < threshold or label in ("N/A", "__background__"):
            continue
        detections.append(
            {
                "label": label,
                "score": round(score, 4),
                "box": [round(c, 1) for c in box],
            }
        )

    # Unique labels, ordered by each label's highest-confidence detection
    # (the model returns detections sorted by score).
    objects = list(dict.fromkeys(d["label"] for d in detections))
    return img, {"objects": objects, "detections": detections}


def save_annotated(img: Image.Image, detections, src_path: Path) -> Path:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    longest = max(img.size)
    line_width = max(2, round(longest / 300))
    font_size = max(12, round(longest / 50))
    try:
        font = ImageFont.load_default(size=font_size)  # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()

    pad = 2
    for d in detections:
        color = PALETTE[sum(map(ord, d["label"])) % len(PALETTE)]
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        text = f'{d["label"]} {d["score"]:.2f}'
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        ty = y1 - th - 2 * pad
        if ty < 0:  # no room above the box; put the tag inside its top edge
            ty = y1
        draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color)
        draw.text((x1 + pad - l, ty + pad - t), text, fill="white", font=font)

    out_path = src_path.with_name(f"{src_path.stem}{ANNOTATED_SUFFIX}{src_path.suffix}")
    img.save(out_path)
    return out_path


def main():
    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        sys.exit("error: --threshold must be between 0 and 1")
    if not args.weights_path.is_file():
        sys.exit(f"error: weights file not found: {args.weights_path}")

    if args.image is not None:
        if not args.image.is_file():
            sys.exit(f"error: image not found: {args.image}")
        paths = [args.image]
    else:
        if not args.folder.is_dir():
            sys.exit(f"error: folder not found: {args.folder}")
        paths = collect_images(args.folder, args.recursive)
        if not paths:
            sys.exit(f"error: no images found in {args.folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights_path, device)
    preprocess = WEIGHTS.transforms()  # PIL -> float tensor in [0, 1]
    categories = WEIGHTS.meta["categories"]  # bundled in torchvision, works offline

    results = {}
    failures = 0

    # Images are processed one at a time; inference runs once per image and
    # both the JSON and the annotated copy are built from that same result.
    for path in paths:
        try:
            img, result = detect(
                path, model, preprocess, categories, device, args.threshold
            )
            if args.annotate:
                save_annotated(img, result["detections"], path)
        except Exception as e:
            print(f"error: {path}: {e}", file=sys.stderr)
            failures += 1
            continue
        results[str(path)] = result

    if args.image is not None:
        if results:
            print(json.dumps(results[str(args.image)], indent=2))
    else:
        print(json.dumps(results, indent=2))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()