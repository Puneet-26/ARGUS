"""
Level 3: Colour Segmentation Lab

Students implement HSV-based color filtering to isolate objects by color.
Works great with live webcam for interactive color picking.
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

    TODO: Implement BGR to HSV conversion using cv2.cvtColor
    Hint: Use cv2.COLOR_BGR2HSV
    """
    # TODO: Your code here
    pass


def create_color_mask(hsv_image, lower_hsv, upper_hsv):
    """
    Create a binary mask for pixels within the HSV range.

    Args:
        hsv_image: Input image in HSV color space
        lower_hsv: Lower bound as tuple (H, S, V)
        upper_hsv: Upper bound as tuple (H, S, V)

    Returns:
        Binary mask (white where color is in range, black elsewhere)

    TODO: Implement color masking using cv2.inRange
    Hint: Convert tuples to numpy arrays first
    """
    # TODO: Your code here
    pass


def apply_morphology(mask, kernel_size=5):
    """
    Clean up mask using morphological operations.

    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel

    Returns:
        Cleaned mask

    TODO: Implement morphological closing (fill holes) then opening (remove noise)
    Hints:
    - Create kernel with cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    - Use cv2.morphologyEx with cv2.MORPH_CLOSE then cv2.MORPH_OPEN
    """
    # TODO: Your code here
    pass


def isolate_colored_object(image, mask):
    """
    Extract the colored object from the original image using the mask.

    Args:
        image: Original BGR image
        mask: Binary mask

    Returns:
        Image with only the masked region visible (rest black)

    TODO: Implement masking using cv2.bitwise_and
    Hint: Use the mask parameter in cv2.bitwise_and
    """
    # TODO: Your code here
    pass


def find_largest_contour(mask):
    """
    Find the largest contour in the mask.

    Args:
        mask: Binary mask

    Returns:
        Largest contour or None if no contours found

    TODO: Implement contour finding and selection
    Hints:
    - Use cv2.findContours with cv2.RETR_EXTERNAL and cv2.CHAIN_APPROX_SIMPLE
    - Use cv2.contourArea to find the largest
    """
    # TODO: Your code here
    pass


def draw_bounding_box(image, contour):
    """
    Draw a bounding box around a contour.

    Args:
        image: Image to draw on (will be copied)
        contour: Contour to draw around

    Returns:
        Image with bounding box drawn

    TODO: Implement bounding box drawing
    Hints:
    - Use cv2.boundingRect to get (x, y, w, h)
    - Use cv2.rectangle to draw the box
    - Copy the image first!
    """
    # TODO: Your code here
    pass


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
    if results['hsv'] is None:
        return None

    # Step 2: Create color mask
    lower_bound = (h_min, s_min, v_min)
    upper_bound = (h_max, s_max, v_max)
    results['mask'] = create_color_mask(results['hsv'], lower_bound, upper_bound)
    if results['mask'] is None:
        return None

    # Step 3: Clean up mask
    results['mask_cleaned'] = apply_morphology(results['mask'], morph_kernel)
    if results['mask_cleaned'] is None:
        results['mask_cleaned'] = results['mask']  # Use uncleaned if morphology fails

    # Step 4: Isolate object
    results['isolated'] = isolate_colored_object(image, results['mask_cleaned'])
    if results['isolated'] is None:
        return None

    # Step 5: Find largest contour
    results['largest_contour'] = find_largest_contour(results['mask_cleaned'])

    # Step 6: Draw bounding box if contour found
    if results['largest_contour'] is not None:
        results['result'] = draw_bounding_box(image, results['largest_contour'])
    else:
        results['result'] = image.copy()

    return results
