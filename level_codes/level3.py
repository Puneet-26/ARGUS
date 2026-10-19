"""Level 3: segment red regions and inspect geometric candidates.

The boxes identify red regions only; they do not confirm an object identity.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "assets" / "evidence" / "level3_corridor_entry.png"
OUTPUT_DIR = ROOT / "outputs" / "level3"

def convert_to_hsv(image):
    """
    Convert BGR image to HSV color space.

    Args:
        image: Input BGR image

    Returns:
        HSV image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def create_color_mask(hsv_image, lower_hsv, upper_hsv):
    """
    Create a binary mask for pixels within the HSV range.

    Args:
        hsv_image: Input image in HSV color space
        lower_hsv: Lower bound as tuple (H, S, V)
        upper_hsv: Upper bound as tuple (H, S, V)

    Returns:
        Binary mask (white where color is in range, black elsewhere)
    """
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower, upper)
    return mask


def apply_morphology(mask, kernel_size=5):
    """
    Clean up mask using morphological operations.

    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel

    Returns:
        Cleaned mask
    """
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Opening first removes isolated specks. Closing then reconnects small gaps
    # without starting from a mask that has already been expanded.
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def find_candidate_contours(mask, min_area=500, max_area_ratio=0.50,
                            min_aspect_ratio=0.12, max_aspect_ratio=8.0):
    """Return geometrically plausible red regions, not semantic object labels.

    A small frame-relative minimum side rejects thin specks that survive
    thresholding. The broad aspect-ratio range retains varied red objects such
    as backpacks and fire extinguishers while filtering lines and strips.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    image_area = max(1, height * width)
    min_dimension = max(5, int(round(min(height, width) * 0.005)))
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > image_area * max_area_ratio:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < min_dimension or box_height < min_dimension:
            continue
        aspect = box_width / max(1, box_height)
        if not min_aspect_ratio <= aspect <= max_aspect_ratio:
            continue
        candidates.append(contour)
    # Stable labels make it easier to compare candidates as sliders change.
    return sorted(candidates, key=cv2.contourArea, reverse=True)


def draw_candidate_overlay(image, contours):
    """Draw candidate boxes and place small labels without label collisions."""
    output = image.copy()
    height, width = output.shape[:2]
    used_labels = []
    box_color = (0, 210, 255)
    text_color = (255, 245, 220)
    background_color = (35, 35, 35)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, contour in enumerate(contours, start=1):
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        cv2.rectangle(output, (x, y), (x + box_width, y + box_height), box_color, 2)

        text = f"C{index} {area:.0f}px2"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.42, 1)
        label_width = text_width + 8
        label_height = text_height + baseline + 6
        label_x = min(max(0, x), max(0, width - label_width))
        preferred_y = y - label_height - 2
        alternatives = [preferred_y, y + box_height + 2]
        step = label_height + 2
        for offset in range(step, height + step, step):
            alternatives.extend((preferred_y - offset, preferred_y + offset))
        label_y = next((
            pos for pos in alternatives
            if 0 <= pos <= height - label_height
            and all(
                label_x + label_width <= ox
                or ox + ow <= label_x
                or pos + label_height <= oy
                or oy + oh <= pos
                for ox, oy, ow, oh in used_labels
            )
        ), None)
        if label_y is None:
            continue
        cv2.rectangle(output, (label_x, label_y), (label_x + label_width, label_y + label_height), background_color, -1)
        cv2.putText(output, text, (label_x + 4, label_y + text_height + 2), font, 0.42, text_color, 1, cv2.LINE_AA)
        used_labels.append((label_x, label_y, label_width, label_height))
    return output


def process_color_segmentation(image, h_min, h_max, s_min, s_max, v_min, v_max,
                                morph_kernel=3, min_area=500, max_area_ratio=0.50):
    """
    Complete color segmentation pipeline.

    Args:
        image: Input BGR image
        h_min, h_max: Hue range (0-179)
        s_min, s_max: Saturation range (0-255)
        v_min, v_max: Value range (0-255)
        morph_kernel: Morphological kernel size

    Returns:
        Dictionary includes HSV, raw/clean masks, candidate regions, and annotated result
    """
    results = {}

    # Step 1: Convert to HSV
    results['hsv'] = convert_to_hsv(image)

    # Step 2: Create the two red hue bands. With h_min > h_max, red is split
    # across OpenCV's hue wraparound at 0/179 and combined from both ranges.
    lower_bound = (h_min, s_min, v_min)
    upper_bound = (h_max, s_max, v_max)
    if h_min <= h_max:
        results['mask'] = create_color_mask(results['hsv'], lower_bound, upper_bound)
    else:
        # OpenCV hue wraps at 179; red can occupy both ends of the hue axis.
        low_red = create_color_mask(results['hsv'], (0, s_min, v_min), upper_bound)
        high_red = create_color_mask(results['hsv'], lower_bound, (179, s_max, v_max))
        results['mask'] = cv2.bitwise_or(low_red, high_red)

    # Step 3: Clean up mask
    results['mask_cleaned'] = apply_morphology(results['mask'], morph_kernel)

    # Step 4: Find and filter external red-region contours.
    contours = find_candidate_contours(
        results['mask_cleaned'], min_area=min_area, max_area_ratio=max_area_ratio
    )
    results['candidate_regions'] = [
        {"area": float(cv2.contourArea(contour)), "box": cv2.boundingRect(contour)}
        for contour in contours
    ]

    # Step 5: Draw every filtered red region as a candidate. No contour is
    # called a backpack: semantic identification remains a human comparison.
    results['result'] = draw_candidate_overlay(image, contours)

    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to actual corridor evidence image")
    parser.add_argument("--hue-tolerance", type=int, default=10, help="Red hue band width at both ends of OpenCV's hue scale")
    parser.add_argument("--min-saturation", type=int, default=120, help="Minimum saturation for red pixels (0-255)")
    parser.add_argument("--min-value", type=int, default=25, help="Minimum brightness, kept low for dark red evidence")
    parser.add_argument("--kernel-size", type=int, choices=range(3, 12, 2), default=3, help="Odd morphology kernel size; opening then closing")
    parser.add_argument("--min-area", type=float, default=500, help="Minimum contour area in source-frame pixels")
    args = parser.parse_args()
    if not 1 <= args.hue_tolerance <= 30:
        raise SystemExit("--hue-tolerance must be between 1 and 30")
    if not 0 <= args.min_saturation <= 255 or not 0 <= args.min_value <= 255:
        raise SystemExit("--min-saturation and --min-value must be between 0 and 255")
    if args.min_area < 0:
        raise SystemExit("--min-area cannot be negative")

    image_path = args.image.expanduser()
    if not image_path.is_absolute() and not image_path.is_file():
        image_path = ROOT / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}\nSupply the real evidence image with --image PATH.")

    # OpenCV hue is 0-179, so these endpoints define two actual red bands.
    results = process_color_segmentation(
        image,
        180 - args.hue_tolerance,
        args.hue_tolerance,
        args.min_saturation,
        255,
        args.min_value,
        255,
        morph_kernel=args.kernel_size,
        min_area=args.min_area,
    )
    candidates = results["candidate_regions"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "red_mask.png"), results["mask_cleaned"])
    cv2.imwrite(str(OUTPUT_DIR / "red_candidates.png"), results["result"])
    cv2.imshow("Original corridor evidence", image)
    cv2.imshow("Cleaned red mask", results["mask_cleaned"])
    cv2.imshow("Filtered red-region candidates", results["result"])
    print(
        f"Found {len(candidates)} filtered red-region candidate(s) with "
        f"S>={args.min_saturation}, V>={args.min_value}, "
        f"{args.kernel_size}x{args.kernel_size} open/close, area>={args.min_area:g}px2."
    )
    print("Candidates are red regions, not confirmed backpacks. Inspect each against the separate evidence.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
