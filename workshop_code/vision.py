"""Small OpenCV routines shared by the standalone workshop scripts."""
from pathlib import Path
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "evidence"
OUTPUTS = ROOT / "outputs"


def read_image(path):
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Evidence image is missing or unreadable: {path}")
    return image


def enhance(image, threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(contrast, threshold, 255, cv2.THRESH_BINARY)
    return {"gray": gray, "clahe": contrast, "threshold": binary}


def red_mask(image, minimum_saturation=70, minimum_value=40):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, np.array([0, minimum_saturation, minimum_value]), np.array([10, 255, 255]))
    high = cv2.inRange(hsv, np.array([170, minimum_saturation, minimum_value]), np.array([179, 255, 255]))
    return cv2.bitwise_or(low, high)


def red_regions(image, minimum_area=100, maximum_area_ratio=0.75):
    mask = red_mask(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area or area > image.shape[0] * image.shape[1] * maximum_area_ratio:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append({"area": area, "box": (x, y, w, h), "centroid": (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))})
    return mask, regions


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save output: {path}")
    return path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path
