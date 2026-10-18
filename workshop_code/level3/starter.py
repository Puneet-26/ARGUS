"""Level 3 starter: generate red candidate regions from the supplied image."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workshop_code.vision import ASSETS, OUTPUTS, read_image, write_image, save_json


def process(image):
    # TODO 1: convert BGR to HSV and combine red masks for H=0..10 and 170..179.
    hsv = None
    mask = None
    if mask is None:
        raise NotImplementedError("Complete TODO 1 to create the wrapped red HSV mask.")

    # TODO 2: use a small elliptical kernel for opening then closing.
    cleaned = None
    if cleaned is None:
        raise NotImplementedError("Complete TODO 2 to reduce mask noise.")

    # TODO 3: find external contours; keep areas from 100 px² through 75% of
    # the image area. These are red candidates, not automatically a backpack.
    contours = None
    if contours is None:
        raise NotImplementedError("Complete TODO 3 to find and filter candidate contours.")

    annotated = image.copy()
    candidates = []
    for index, contour in enumerate(contours, 1):
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 220, 255), 2)
        cv2.putText(annotated, f"red candidate {index}", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        candidates.append({"area": area, "box": [x, y, width, height]})
    isolated = cv2.bitwise_and(image, image, mask=cleaned)
    return cleaned, annotated, isolated, candidates


def main():
    source = ASSETS / "level3_corridor_entry.png"
    image = read_image(source)
    mask, annotated, isolated, candidates = process(image)
    output = OUTPUTS / "level3"
    write_image(output / "red_mask.png", mask)
    write_image(output / "red_regions.png", annotated)
    write_image(output / "red_isolated.png", isolated)
    save_json(output / "regions.json", {"source": str(source.resolve()), "candidates": candidates})
    print(f"Saved the actual red mask and {len(candidates)} area-filtered candidates from {source.name}.")
    print("Colour candidates are not bag identities; inspect distractors and the patch separately.")


if __name__ == "__main__":
    main()
