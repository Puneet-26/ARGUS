"""Helpers for saving genuine OpenCV outputs with notebook references."""

from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_image(level: int, filename: str, image: np.ndarray) -> str:
    """Write an actual processed image under outputs/ and return its relative path."""
    if level not in (1, 2, 3, 4):
        raise ValueError("Level must be between 1 and 4.")
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("There is no processed image to save.")
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("Output filename is empty.")
    destination = PROJECT_ROOT / "outputs" / f"level{level}" / safe_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"OpenCV could not save the processed output to {destination}.")
    return destination.relative_to(PROJECT_ROOT).as_posix()
