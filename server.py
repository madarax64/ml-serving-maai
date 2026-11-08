#!/usr/bin/env python3
"""LitServe service: object detection with Faster R-CNN ResNet50 FPN v2 (COCO).

Run:
    DETECTOR_WEIGHTS=./fasterrcnn_resnet50_fpn_v2_coco.pth python server.py

Request (POST /predict, JSON body):
    {
      "input": "<base64-encoded image>",   # required
      "annotate": false,                   # optional, default false
      "threshold": 0.5                     # optional, default 0.5 (range 0-1)
    }

Response (JSON):
    {
      "objects": ["cup", "laptop"],
      "detections": [
        {"label": "cup", "score": 0.9731, "box": [x_min, y_min, x_max, y_max]},
        ...
      ],
      "annotated_image": "<base64 JPG>"    # only present when annotate is true
    }

Invalid input (bad base64, undecodable image, bad parameters) returns HTTP 400.
"""
import base64
import binascii
import io
import os
from pathlib import Path

import litserve as ls
import torch
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from fastapi.middleware.cors import CORSMiddleware

WEIGHTS = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
DEFAULT_WEIGHTS_PATH = "./fasterrcnn_resnet50_fpn_v2_coco.pth"
DEFAULT_THRESHOLD = 0.5
PORT = 24242
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#008080", "#9a6324", "#800000", "#808000", "#000075",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def decode_image(b64: str) -> Image.Image:
    """Base64 string -> upright RGB PIL image. Raises HTTP 400 on bad input."""
    if not isinstance(b64, str):
        raise bad_request("'input' must be a base64-encoded string")
    if b64.startswith("data:") and "," in b64:  # tolerate data URIs from browsers
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise bad_request("'input' is not valid base64")
    try:
        img = Image.open(io.BytesIO(raw))
        # Phone photos often store rotation in EXIF; apply it so that boxes
        # line up with the image as people see it.
        return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise bad_request("'input' could not be decoded as an image")


def draw_boxes(img: Image.Image, detections: list) -> Image.Image:
    """Return a copy of `img` with a labelled box drawn for each detection."""
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
    return img


def to_base64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# The service
# --------------------------------------------------------------------------
class DetectorAPI(ls.LitAPI):
    """Request lifecycle: decode_request -> predict -> encode_response."""

    def setup(self, device):
        """Runs once per worker, with the device LitServe assigned to it."""
        weights_path = Path(os.environ.get("DETECTOR_WEIGHTS", DEFAULT_WEIGHTS_PATH))
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path} "
                "(set the DETECTOR_WEIGHTS environment variable)"
            )

        # weights_backbone=None stops torchvision from trying to download the
        # ImageNet backbone weights, which would fail offline.
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        self.device = device
        self.model = model.eval().to(device)
        self.preprocess = WEIGHTS.transforms()  # PIL -> float tensor in [0, 1]
        self.categories = WEIGHTS.meta["categories"]  # bundled, works offline

    def decode_request(self, request):
        """Validate the JSON body and turn it into what predict() needs."""
        if not isinstance(request, dict) or "input" not in request:
            raise bad_request("request body must be a JSON object with an 'input' key")

        annotate = request.get("annotate", False)
        if not isinstance(annotate, bool):
            raise bad_request("'annotate' must be true or false")

        threshold = request.get("threshold", DEFAULT_THRESHOLD)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise bad_request("'threshold' must be a number")
        if not 0.0 <= threshold <= 1.0:
            raise bad_request("'threshold' must be between 0 and 1")

        return {
            "image": decode_image(request["input"]),
            "annotate": annotate,
            "threshold": float(threshold),
        }

    def predict(self, request):
        """Run the model once and build the detection result."""
        image = request["image"]
        with torch.inference_mode():
            output = self.model([self.preprocess(image).to(self.device)])[0]

        detections = []
        for box, label_id, score in zip(
            output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()
        ):
            label = self.categories[label_id]
            if score < request["threshold"] or label in ("N/A", "__background__"):
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

        return {
            "image": image,
            "annotate": request["annotate"],
            "result": {"objects": objects, "detections": detections},
        }

    def encode_response(self, output):
        """Attach the annotated image if requested, and return the JSON body."""
        response = output["result"]
        if output["annotate"]:
            annotated = draw_boxes(output["image"], response["detections"])
            response = {**response, "annotated_image": to_base64_jpeg(annotated)}
        return response


if __name__ == "__main__":
    # server = ls.LitServer(DetectorAPI(), accelerator="auto", workers_per_device=5,
    #     middlewares=[(CORSMiddleware, {
    #     "allow_origins": ["*"],
    #     "allow_methods": ["POST", "OPTIONS"],
    #     "allow_headers": ["Content-Type"],
    # })])
    server = ls.LitServer(DetectorAPI(), accelerator="auto", workers_per_device=2, timeout=120)
    server.run(port=PORT)
