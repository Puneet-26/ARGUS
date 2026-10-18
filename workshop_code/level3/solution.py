"""Instructor reference: red segmentation with hue wrap-around and contours."""
import cv2
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workshop_code.vision import ASSETS, OUTPUTS, read_image, red_regions, write_image, save_json


def process(image):
    mask, regions = red_regions(image, minimum_area=100, maximum_area_ratio=0.75)
    annotated = image.copy()
    for index, region in enumerate(regions, 1):
        x, y, w, h = region["box"]
        area = region["area"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 255), 2)
        cv2.putText(annotated, f"red candidate {index}", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    return mask, annotated, regions


def main():
    source = ASSETS / "level3_corridor_entry.png"
    mask, annotated, regions = process(read_image(source))
    out = OUTPUTS / "level3"
    write_image(out / "red_mask.png", mask)
    write_image(out / "red_regions.png", annotated)
    save_json(out / "regions.json", {"regions": regions, "limitation": "Candidates are red regions after cleanup and area filtering; they are not classified as a backpack or owner."})
    print(f"Found {len(regions)} red candidate region(s). Inspect each against the actual image and separate records.")


if __name__ == "__main__":
    main()
