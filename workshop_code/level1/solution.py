"""Instructor reference: actual grayscale, CLAHE, and threshold processing."""
import sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workshop_code.vision import ASSETS, OUTPUTS, read_image, write_image


def process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, result = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
    return gray, enhanced, result


def main():
    source = ASSETS / "level1_cctv_dark.png"
    gray, enhanced, result = process(read_image(source))
    output_dir = OUTPUTS / "level1"
    write_image(output_dir / "grayscale.png", gray)
    write_image(output_dir / "clahe.png", enhanced)
    output = write_image(output_dir / "timestamp_threshold.png", result)
    print(f"Processed supplied evidence. Inspect {output}; enhancement does not guarantee legibility.")


if __name__ == "__main__":
    main()
