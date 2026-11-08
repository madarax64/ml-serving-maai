#!/usr/bin/env python3
"""CLI client for the detection service (server.py).

Usage:
    python 2_api_client.py --image desk.jpg
    python 2_api_client.py --image desk.jpg --threshold 0.7
    python 2_api_client.py --image desk.jpg --annotate
    python 2_api_client.py --image desk.jpg --url http://other-host:24242/predict

Prints the server's JSON (objects and detections) to stdout. With --annotate,
the annotated image is saved as api_response.png next to this script; status
messages and errors go to stderr, so stdout stays clean, parseable JSON.

Requires: pip install requests
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:24242/predict"
OUTPUT_PATH = Path(__file__).resolve().parent / "api_response.jpg"
TIMEOUT_SECONDS = 120  # generous: the first request can be slow while the GPU warms up


def parse_args():
    p = argparse.ArgumentParser(description="Send an image to the detection service.")
    p.add_argument("--image", type=Path, required=True, help="path to the image file")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="minimum confidence score to keep a detection (default: 0.5)",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help=f"ask the server to draw boxes; the image is saved to {OUTPUT_PATH.name}",
    )
    p.add_argument(
        "--url", default=DEFAULT_URL, help=f"service endpoint (default: {DEFAULT_URL})"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.image.is_file():
        sys.exit(f"error: image not found: {args.image}")

    payload = {
        "input": base64.b64encode(args.image.read_bytes()).decode("ascii"),
        "annotate": args.annotate,
        "threshold": args.threshold,
    }

    try:
        response = requests.post(args.url, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError:
        sys.exit(f"error: could not connect to {args.url} (is the server running?)")
    except requests.exceptions.Timeout:
        sys.exit(f"error: request timed out after {TIMEOUT_SECONDS}s")

    if not response.ok:
        # The server reports problems as {"detail": "..."}.
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        sys.exit(f"error: server returned HTTP {response.status_code}: {detail}")

    result = response.json()

    # The annotated image is a huge base64 string: save it, don't print it.
    annotated = result.pop("annotated_image", None)
    if args.annotate:
        if annotated:
            OUTPUT_PATH.write_bytes(base64.b64decode(annotated))
            print(f"saved annotated image to {OUTPUT_PATH}", file=sys.stderr)
        else:
            print("warning: server returned no annotated image", file=sys.stderr)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
