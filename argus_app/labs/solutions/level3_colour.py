"""
Level 3: Colour Segmentation Lab - REFERENCE SOLUTION

Complete implementation of HSV color segmentation.
"""
import cv2
import numpy as np


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Close to fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Open to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def isolate_colored_object(image, mask):
    """
    Extract the colored object from the original image using the mask.

    Args:
        image: Original BGR image
        mask: Binary mask

    Returns:
        Image with only the masked region visible (rest black)
    """
    isolated = cv2.bitwise_and(image, image, mask=mask)
    return isolated


def find_largest_contour(mask):
    """
    Find the largest contour in the mask.

    Args:
        mask: Binary mask

    Returns:
        Largest contour or None if no contours found
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Find largest by area
    largest = max(contours, key=cv2.contourArea)

    # Filter out very small contours
    if cv2.contourArea(largest) < 100:
        return None

    return largest


def find_candidate_contours(mask, min_area=100, max_area_ratio=0.75):
    """Return external red regions within explicit image-relative area limits."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = max(1, mask.shape[0] * mask.shape[1])
    return [
        contour for contour in contours
        if min_area <= cv2.contourArea(contour) <= image_area * max_area_ratio
    ]


def draw_bounding_box(image, contour):
    """
    Draw a bounding box around a contour.

    Args:
        image: Image to draw on (will be copied)
        contour: Contour to draw around

    Returns:
        Image with bounding box drawn
    """
    output = image.copy()

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Draw rectangle
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate and draw centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

    return output


def process_color_segmentation(image, h_min, h_max, s_min, s_max, v_min, v_max,
                                morph_kernel=5):
    """
    Complete color segmentation pipeline.

    Args:
        image: Input BGR image
        h_min, h_max: Hue range (0-179)
        s_min, s_max: Saturation range (0-255)
        v_min, v_max: Value range (0-255)
        morph_kernel: Morphological kernel size

    Returns:
        Dictionary with keys: 'hsv', 'mask', 'mask_cleaned', 'isolated',
        'largest_contour', 'result'
    """
    results = {}

    # Step 1: Convert to HSV
    results['hsv'] = convert_to_hsv(image)

    # Step 2: Create color mask
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

    # Step 4: Isolate object
    results['isolated'] = isolate_colored_object(image, results['mask_cleaned'])

    # Step 5: Find largest contour
    results['candidate_contours'] = find_candidate_contours(results['mask_cleaned'])
    results['largest_contour'] = max(
        results['candidate_contours'], key=cv2.contourArea, default=None
    )

    # Step 6: Draw bounding box if contour found
    if results['largest_contour'] is not None:
        results['result'] = image.copy()
        for contour in results['candidate_contours']:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            cv2.rectangle(results['result'], (x, y), (x + w, y + h), (0, 220, 255), 2)
            cv2.putText(results['result'], f"candidate {area:.0f}px2", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    else:
        results['result'] = image.copy()

    return results
