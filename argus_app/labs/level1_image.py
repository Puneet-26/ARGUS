"""
Level 1: Image Processing Lab

Students implement basic image processing functions:
- Grayscale conversion
- Gaussian blur
- Binary threshold
- Canny edge detection
- Contour detection

Each function should process the input and return the result.
"""
import cv2
import numpy as np


def enhance_evidence(image):
    """
    Inspect dark CCTV evidence from the VESIT Robotics Lab case.

    Args:
        image: Input BGR CCTV frame (numpy array)

    Returns:
        tuple (enhanced, threshold)
    """
    # Step 1: Convert BGR to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # TODO: Improve local contrast using CLAHE or contrast scaling
    # Hint: clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #       enhanced = clahe.apply(gray)
    enhanced = gray

    # TODO: Reduce sensor noise with Gaussian Blur
    # Hint: denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    denoised = enhanced

    # TODO: Threshold the relevant region to isolate text/features
    # Hint: _, threshold = cv2.threshold(denoised, 90, 255, cv2.THRESH_BINARY)
    threshold = denoised

    return enhanced, threshold


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """
    Adjust brightness and contrast of an image.

    Args:
        image: Input image (BGR or Grayscale)
        brightness: Brightness offset (-100 to 100)
        contrast: Contrast multiplier (0.1 to 3.0)

    Returns:
        Adjusted image
    """
    # TODO: Your code here
    pass


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Input grayscale or BGR image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization (width, height)

    Returns:
        CLAHE enhanced image
    """
    # TODO: Your code here
    pass


def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.

    Args:
        image: Input BGR image (numpy array)

    Returns:
        Grayscale image (numpy array)

    TODO: Implement grayscale conversion using cv2.cvtColor
    Hint: Use cv2.COLOR_BGR2GRAY
    """
    # TODO: Your code here
    pass


def apply_gaussian_blur(image, kernel_size=5):
    """
    Apply Gaussian blur to reduce noise.

    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel (must be odd)

    Returns:
        Blurred image

    TODO: Implement Gaussian blur using cv2.GaussianBlur
    Hint: kernel_size should be (kernel_size, kernel_size)
    """
    # TODO: Your code here
    pass


def apply_threshold(image, threshold_value=127):
    """
    Apply binary threshold to create a binary image.

    Args:
        image: Input grayscale image
        threshold_value: Threshold value (0-255)

    Returns:
        Binary thresholded image

    TODO: Implement binary thresholding using cv2.threshold
    Hint: Use cv2.THRESH_BINARY as the threshold type
    Return only the thresholded image (second value from cv2.threshold)
    """
    # TODO: Your code here
    pass


def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Detect edges using Canny edge detector.

    Args:
        image: Input grayscale image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection

    Returns:
        Edge map (binary image)

    TODO: Implement Canny edge detection using cv2.Canny
    """
    # TODO: Your code here
    pass


def find_contours(binary_image):
    """
    Find contours in a binary image.

    Args:
        binary_image: Input binary image (from threshold or Canny)

    Returns:
        List of contours

    TODO: Implement contour detection using cv2.findContours
    Hint: Use cv2.RETR_EXTERNAL and cv2.CHAIN_APPROX_SIMPLE
    Return only the contours (may be second or first value depending on OpenCV version)
    """
    # TODO: Your code here
    pass


def draw_contours_on_image(image, contours):
    """
    Draw detected contours on an image.

    Args:
        image: Input BGR image (will be copied)
        contours: List of contours from find_contours

    Returns:
        Image with contours drawn

    TODO: Implement contour drawing using cv2.drawContours
    Hint: Draw all contours (-1) in green color (0, 255, 0) with thickness 2
    Make sure to copy the input image first!
    """
    # TODO: Your code here
    pass


def process_pipeline(image, brightness=0, contrast=1.0, use_clahe=False, clahe_clip=2.0,
                     blur_kernel=5, threshold_value=127, canny_low=50, canny_high=150):
    """
    Complete processing pipeline that returns all intermediate steps.

    Args:
        image: Input BGR image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast factor (0.1 to 3.0)
        use_clahe: Whether to apply CLAHE enhancement
        clahe_clip: CLAHE clip limit
        blur_kernel: Kernel size for Gaussian blur
        threshold_value: Threshold value for binary threshold
        canny_low: Lower threshold for Canny
        canny_high: Upper threshold for Canny

    Returns:
        Dictionary with intermediate & output steps
    """
    results = {}

    # Step 0: Brightness / Contrast
    enhanced = adjust_brightness_contrast(image, brightness, contrast)
    if enhanced is None:
        enhanced = image
    results['enhanced'] = enhanced

    # Step 1: Grayscale
    results['gray'] = convert_to_grayscale(enhanced)
    if results['gray'] is None:
        return None

    # Step 1b: CLAHE
    if use_clahe:
        clahe_res = apply_clahe(results['gray'], clahe_clip)
        if clahe_res is not None:
            results['gray'] = clahe_res
    results['clahe'] = results['gray']

    # Step 2: Apply blur
    results['blur'] = apply_gaussian_blur(results['gray'], blur_kernel)
    if results['blur'] is None:
        return None

    # Step 3: Apply threshold
    results['threshold'] = apply_threshold(results['blur'], threshold_value)
    if results['threshold'] is None:
        return None

    # Step 4: Detect edges
    results['edges'] = detect_edges(results['blur'], canny_low, canny_high)
    if results['edges'] is None:
        return None

    # Step 5: Find contours (using edges)
    results['contours'] = find_contours(results['edges'])
    if results['contours'] is None:
        return None

    # Step 6: Draw contours on original
    results['contours_drawn'] = draw_contours_on_image(enhanced, results['contours'])
    if results['contours_drawn'] is None:
        return None

    return results
