"""Level 1: enhance a CCTV image and inspect its clock crop.

Run from the repository root with ``python level_codes/level1.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "assets" / "evidence"
OUTPUT_DIR = ROOT / "outputs" / "level1"
DEFAULT_IMAGE = EVIDENCE_DIR / "EVIDENCE-01.png"
EFFECTS = ("Grayscale", "CLAHE", "Threshold")
CLOCK_ZOOM = 5
LEVEL1_CLOCK_EVIDENCE = "EVIDENCE-01"
LEVEL1_CLOCK_EVIDENCE_ALIASES = {"evidence-01", "level1_cctv_dark"}
# Clock housing in the supplied 2752x1536 workshop evidence. The ROI is checked
# against the expected asset dimensions and its dim red LED pixels at runtime.
# No clock digits or interpreted time are stored here.
LEVEL1_CLOCK_IMAGE_SIZE = (2752, 1536)
LEVEL1_CLOCK_ROI = (1755, 230, 185, 75)


def detect_clock_region(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find a compact clock-like line of bright text in the complete CCTV frame.

    Detection is content based so arbitrary uploads never inherit coordinates from
    the Level 1 asset. The candidate must contain several similarly sized glyphs
    grouped on one baseline and be near a frame corner, where CCTV clock overlays
    are normally placed. Ambiguous images are rejected instead of mis-cropped.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("The image must be grayscale, BGR, or BGRA.")

    height, width = gray.shape[:2]
    min_char_height = max(3, int(height * 0.003))
    max_char_height = max(min_char_height + 1, int(height * 0.065))
    block = min(51, max(15, (min(height, width) // 20) | 1))
    if block >= min(height, width):
        block = max(3, (min(height, width) - 1) | 1)
    masks = [
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, -5,
        ),
        cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1],
    ]

    candidates = []
    for mask in masks:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        glyphs = []
        for x, y, w, h, area in stats[1:count]:
            if not min_char_height <= h <= max_char_height:
                continue
            if w > max_char_height or area < max(3, int(h * 0.12)):
                continue
            glyphs.append((int(x), int(y), int(w), int(h)))
        glyphs.sort(key=lambda item: (item[1] + item[3] / 2, item[0]))

        # Form baseline groups using glyph-center distance, then split on large
        # horizontal gaps so unrelated text elsewhere cannot enlarge the crop.
        groups = []
        for glyph in glyphs:
            gx, gy, gw, gh = glyph
            center_y = gy + gh / 2
            matching = next(
                (group for group in groups if abs(group["center"] - center_y) <= max(2, gh * 0.45)),
                None,
            )
            if matching is None:
                groups.append({"center": center_y, "items": [glyph]})
            else:
                matching["items"].append(glyph)
                matching["center"] = float(np.median([g[1] + g[3] / 2 for g in matching["items"]]))

        for group in groups:
            row = sorted(group["items"], key=lambda item: item[0])
            if len(row) < 4:
                continue
            med_h = float(np.median([g[3] for g in row]))
            runs = [[row[0]]]
            for glyph in row[1:]:
                previous = runs[-1][-1]
                gap = glyph[0] - (previous[0] + previous[2])
                if gap > max(2, med_h * 1.35):
                    runs.append([glyph])
                else:
                    runs[-1].append(glyph)
            for run in runs:
                if len(run) < 4:
                    continue
                x0 = min(g[0] for g in run)
                y0 = min(g[1] for g in run)
                x1 = max(g[0] + g[2] for g in run)
                y1 = max(g[1] + g[3] for g in run)
                run_w, run_h = x1 - x0, y1 - y0
                # Require a colon-shaped separator (two small dots stacked
                # vertically) so ordinary corner captions are not mistaken
                # for a clock or timestamp.
                small = [
                    g for g in glyphs
                    if g[3] <= med_h * 0.55 and g[2] <= med_h * 0.55
                    and x0 - med_h * 0.5 <= g[0] + g[2] / 2 <= x1 + med_h * 0.5
                    and y0 - med_h * 0.4 <= g[1] + g[3] / 2 <= y1 + med_h * 1.3
                ]
                has_colon = any(
                    abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)) <= med_h * 0.55
                    and med_h * 0.2 <= (b[1] + b[3] / 2) - (a[1] + a[3] / 2) <= med_h * 0.8
                    for i, a in enumerate(small)
                    for b in small[i + 1:]
                    if b[1] + b[3] / 2 > a[1] + a[3] / 2
                )
                if not has_colon:
                    continue
                if run_w < med_h * 1.6 or run_w > width * 0.30:
                    continue
                if not 1.4 <= run_w / max(1, run_h) <= 16:
                    continue
                # Give the automatic crop a small border without including a
                # large part of the scene, and prefer candidates near corners.
                pad_x = max(3, int(med_h * 0.55))
                pad_y = max(3, int(med_h * 0.7))
                left, top = max(0, x0 - pad_x), max(0, y0 - pad_y)
                right, bottom = min(width, x1 + pad_x), min(height, y1 + pad_y)
                corner_distance = min(
                    (x0 / width) ** 2 + (y0 / height) ** 2,
                    ((width - x1) / width) ** 2 + (y0 / height) ** 2,
                    (x0 / width) ** 2 + ((height - y1) / height) ** 2,
                    ((width - x1) / width) ** 2 + ((height - y1) / height) ** 2,
                )
                score = len(run) + min(run_w / max(1, run_h), 8) - corner_distance * 10
                candidates.append((score, (left, top, right - left, bottom - top)))

    if not candidates:
        raise ValueError(
            "No clear clock or timestamp region was found in this image. "
            "Upload the Level 1 CCTV frame or an image with a visible clock."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, region = candidates[0]
    if best_score < 5:
        raise ValueError("A clock region could not be validated. Choose a clearer image.")
    if len(candidates) > 1 and candidates[1][0] >= best_score * 0.96:
        first, second = candidates[0][1], candidates[1][1]
        overlap_x = max(0, min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0]))
        overlap_y = max(0, min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1]))
        overlap = overlap_x * overlap_y / max(1, min(first[2] * first[3], second[2] * second[3]))
        if overlap < 0.5:
            raise ValueError("More than one possible clock region was found. Choose a clearer image.")
    return region


def configured_clock_roi(image: np.ndarray, evidence_name: str):
    """Resolve the supplied Level 1 asset's clock ROI, never for arbitrary uploads.

    The separately supplied workshop image is not bundled in the repository, so
    the exact ROI is validated from the named asset's clock pixels at runtime
    rather than guessing and baking coordinates from a missing file. Once the
    asset changes, this function is the single place to update that policy.
    """
    evidence_stem = Path(evidence_name or "").stem.casefold()
    if evidence_stem not in LEVEL1_CLOCK_EVIDENCE_ALIASES:
        return None
    height, width = image.shape[:2]
    if (width, height) != LEVEL1_CLOCK_IMAGE_SIZE:
        try:
            # A resized/re-encoded copy may still be safely used when the clock
            # detector validates a region from its own pixels.
            return detect_clock_region(image)
        except ValueError as exc:
            raise ValueError(
                f"{LEVEL1_CLOCK_EVIDENCE} has different dimensions from the supplied "
                "workshop image, and no clock region could be validated automatically."
            ) from exc
    if LEVEL1_CLOCK_ROI is not None:
        x, y, roi_width, roi_height = LEVEL1_CLOCK_ROI
        if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0 or x + roi_width > width or y + roi_height > height:
            raise ValueError(
                "The configured Level 1 clock crop is outside this image. "
                "Check that the supplied workshop evidence is loaded."
            )
        crop = image[y:y + roi_height, x:x + roi_width]
        if crop.ndim == 3 and crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("The supplied Level 1 image does not contain a valid colour clock region.")
        blue, green, red = cv2.split(crop)
        red_led_pixels = (
            (red >= 16)
            & (red.astype(np.uint16) >= green.astype(np.uint16) * 1.2)
            & (red.astype(np.uint16) >= blue.astype(np.uint16) * 1.1)
        )
        if int(np.count_nonzero(red_led_pixels)) < 40:
            raise ValueError(
                "The configured clock area does not contain the expected red LED signal. "
                "Choose the supplied Level 1 CCTV evidence."
            )
        return LEVEL1_CLOCK_ROI
    return detect_clock_region(image)


def process_clock_region(image: np.ndarray, effect: str, region=None, clahe_clip_limit=2.0, threshold_value=128):
    """Return an enlarged processed view of the selected clock crop."""
    x, y, width, height = region or detect_clock_region(image)
    crop = image[y:y + height, x:x + width]
    if crop.size == 0:
        raise ValueError("The detected clock region is empty.")
    if crop.ndim == 2:
        gray = crop
    elif crop.shape[2] == 4:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if effect == "Grayscale":
        result = gray
    elif effect == "CLAHE":
        result = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8)).apply(gray)
    elif effect == "Threshold":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown image effect: {effect}")
    if result.size == 0:
        raise RuntimeError("Image processing returned an empty result.")
    zoomed_result = cv2.resize(
        result, None, fx=CLOCK_ZOOM, fy=CLOCK_ZOOM, interpolation=cv2.INTER_CUBIC
    )
    return zoomed_result, (x, y, width, height)


def apply_effect(image: np.ndarray, effect: str, clahe_clip_limit=2.0, threshold_value=128) -> np.ndarray:
    """Return a processed image without modifying the uploaded source array."""
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
        result = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8)).apply(gray)
    elif effect == "Threshold":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown image effect: {effect}")

    if result is None or result.size == 0:
        raise RuntimeError("Image processing returned an empty result.")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Evidence image; defaults to assets/evidence/EVIDENCE-01.*")
    parser.add_argument("--effect", choices=EFFECTS, default="CLAHE")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--threshold", type=int, default=128)
    args = parser.parse_args()
    image_path = args.image
    if image_path is not None:
        image_path = image_path.expanduser()
        if not image_path.is_absolute() and not image_path.is_file():
            image_path = ROOT / image_path
    if image_path is None:
        matches = sorted(EVIDENCE_DIR.glob("EVIDENCE-01.*"))
        image_path = matches[0] if matches else DEFAULT_IMAGE
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}. Add EVIDENCE-01 to assets/evidence or pass --image.")
    height, width = image.shape[:2]
    print(f"Image dimensions: {width} x {height}; OpenCV reads color channels as BGR.")
    enhanced = apply_effect(image, args.effect, args.clip_limit, args.threshold)
    region = configured_clock_roi(image, image_path.name)
    detail = None
    if region is not None:
        detail, region = process_clock_region(image, args.effect, region, args.clip_limit, args.threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / f"{args.effect.lower()}_full.png"), enhanced)
    if detail is not None:
        cv2.imwrite(str(OUTPUT_DIR / "clock_detail_enhanced.png"), detail)
    cv2.imshow("Original evidence", image)
    cv2.imshow(f"{args.effect} enhanced evidence", enhanced)
    if detail is not None:
        cv2.imshow("Enhanced clock detail", detail)
    else:
        print("No validated clock ROI for this image; showing the full-frame enhancement only.")
    print(f"Saved outputs in {OUTPUT_DIR}. Press any key to close.")
    print("Enhancement may improve visibility but cannot recover detail absent from the source.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
