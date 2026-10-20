"""Level 1: apply image effects and manually crop a Region of Interest.

Run from the repository root with ``python level_codes/level1.py --image PATH``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "level1"
EFFECTS = ("Grayscale", "CLAHE", "Threshold")
ROI_ENLARGEMENT = 5


def apply_effect(image: np.ndarray, effect: str, clahe_clip_limit=2.0,
                 threshold_value=128) -> np.ndarray:
    """Apply the selected OpenCV effect without modifying the source image."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("The image must be grayscale, BGR, or BGRA.")

    source = image.copy()
    if source.ndim == 3 and source.shape[2] == 4:
        source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
    gray = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    if effect == "Grayscale":
        result = gray.copy()
    elif effect == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(8, 8))
        result = clahe.apply(gray)
    elif effect == "Threshold":
        threshold = int(threshold_value)
        if not 0 <= threshold <= 255:
            raise ValueError("Threshold must be between 0 and 255.")
        _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown image effect: {effect}")

    if result.size == 0:
        raise RuntimeError("Image processing returned an empty result.")
    return result


def process_roi(image: np.ndarray, effect: str, region,
                clahe_clip_limit=2.0, threshold_value=128):
    """Crop a manually selected (x, y, width, height) ROI and enlarge both views.

    Coordinates refer to the original, full-resolution source image. The returned
    tuple contains the enlarged processed view and validated ROI coordinates.
    Nearest-neighbour scaling keeps small digital display segments crisp while
    students inspect them.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("The image must be grayscale, BGR, or BGRA.")
    if region is None:
        raise ValueError("Enter X, Y, width, and height for the ROI.")
    try:
        coordinates = tuple(region)
    except TypeError as exc:
        raise ValueError("Enter X, Y, width, and height for the ROI.") from exc
    if len(coordinates) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in coordinates
    ):
        raise ValueError("ROI coordinates and dimensions must be four whole numbers.")
    x, y, roi_width, roi_height = (int(value) for value in coordinates)
    zoom_factor = ROI_ENLARGEMENT

    height, width = image.shape[:2]
    if x < 0 or y < 0:
        raise ValueError("ROI X and Y must be zero or greater.")
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI width and height must be greater than zero.")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError(
            f"ROI is outside the {width} × {height} image. Keep X + W ≤ {width} "
            f"and Y + H ≤ {height}."
        )
    detail_width, detail_height = roi_width * zoom_factor, roi_height * zoom_factor
    if max(detail_width, detail_height) > 16_384 or detail_width * detail_height > 16_000_000:
        raise ValueError("That ROI is too large to enlarge. Reduce its width or height.")

    # This is the sole crop implementation used by the UI and command-line tool.
    crop = image[y:y + roi_height, x:x + roi_width]
    if crop.size == 0:
        raise ValueError("The selected ROI is empty.")
    processed_crop = apply_effect(crop, effect, clahe_clip_limit, threshold_value)
    interpolation = cv2.INTER_NEAREST
    processed_detail = cv2.resize(
        processed_crop, None, fx=zoom_factor, fy=zoom_factor, interpolation=interpolation
    )
    return processed_detail, (x, y, roi_width, roi_height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Path to evidence image")
    parser.add_argument("--effect", choices=EFFECTS, default="CLAHE")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE contrast clip limit")
    parser.add_argument("--threshold", type=int, default=128, help="Binary threshold value (0-255)")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Optional manual ROI coordinates: X Y WIDTH HEIGHT")
    args = parser.parse_args()

    image_path = args.image.expanduser()
    if not image_path.is_absolute() and not image_path.is_file():
        image_path = ROOT / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}. Provide a valid image file with --image PATH.")

    height, width = image.shape[:2]
    print(f"Image dimensions: {width} × {height}; ROI coordinates use this full-resolution image.")
    processed = apply_effect(image, args.effect, args.clip_limit, args.threshold)
    details = None
    if args.roi is not None:
        details = process_roi(
            image, args.effect, args.roi, args.clip_limit, args.threshold
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / f"{args.effect.lower()}_full.png"), processed)
    cv2.imshow("Original evidence", image)
    cv2.imshow(f"{args.effect} enhanced evidence", processed)
    if details is not None:
        processed_detail, region = details
        cv2.imwrite(str(OUTPUT_DIR / "roi_detail_enhanced.png"), processed_detail)
        cv2.imshow(f"Processed ROI {region}", processed_detail)
    else:
        print("No ROI selected. Use --roi X Y WIDTH HEIGHT to inspect a region.")
    print(f"Saved outputs in {OUTPUT_DIR}. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
