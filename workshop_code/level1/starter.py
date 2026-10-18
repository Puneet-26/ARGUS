"""Level 1 starter: recover visible detail from the supplied dark frame."""
from pathlib import Path
import sys
import cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workshop_code.vision import ASSETS, OUTPUTS, read_image, write_image


def main():
    source = ASSETS / "level1_cctv_dark.png"
    image = read_image(source)
    # TODO 1: convert image from BGR to grayscale.
    gray = None
    # TODO 2: create and apply CLAHE (clipLimit=2.0, tileGridSize=(8, 8)).
    enhanced = None
    # TODO 3: threshold enhanced grayscale at 150, then save the binary result.
    result = None
    output = write_image(OUTPUTS / "level1" / "timestamp_threshold.png", result)
    print(f"Processed {source.name} ({image.shape[1]} x {image.shape[0]}). Output: {output}")


if __name__ == "__main__":
    main()
