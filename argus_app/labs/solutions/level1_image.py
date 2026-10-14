"""
Level 1: Image Processing Lab - REFERENCE SOLUTION

Complete implementation of all image processing functions.
"""
import cv2
import numpy as np


def enhance_evidence(image):
    """
    Reference solution: Recover dark CCTV evidence frame from Lab 4B corridor.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, threshold = cv2.threshold(denoised, 90, 255, cv2.THRESH_BINARY)
    return enhanced, threshold


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """
    Adjust brightness and contrast of an image.

    Args:
        image: Input BGR image
        brightness: Offset (-100 to 100)
        contrast: Scale factor (0.1 to 3.0)

    Returns:
        Adjusted image
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return clahe.apply(image)


def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.

    Args:
        image: Input BGR image (numpy array)

    Returns:
        Grayscale image (numpy array)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image, kernel_size=5):
    """
    Apply Gaussian blur to reduce noise.

    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel (must be odd)

    Returns:
        Blurred image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_threshold(image, threshold_value=127):
    """
    Apply binary threshold to create a binary image.

    Args:
        image: Input grayscale image
        threshold_value: Threshold value (0-255)

    Returns:
        Binary thresholded image
    """
    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh


def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Detect edges using Canny edge detector.

    Args:
        image: Input grayscale image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection

    Returns:
        Edge map (binary image)
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def find_contours(binary_image):
    """
    Find contours in a binary image.

    Args:
        binary_image: Input binary image (from threshold or Canny)

    Returns:
        List of contours
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours_on_image(image, contours):
    """
    Draw detected contours on an image.

    Args:
        image: Input BGR image (will be copied)
        contours: List of contours from find_contours

    Returns:
        Image with contours drawn
    """
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


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
        Dictionary with all intermediate steps
    """
    results = {}

    # Step 0: Brightness / Contrast
    enhanced = adjust_brightness_contrast(image, brightness, contrast)
    results['enhanced'] = enhanced

    # Step 1: Grayscale
    gray = convert_to_grayscale(enhanced)

    # Step 1b: CLAHE
    if use_clahe:
        gray = apply_clahe(gray, clahe_clip)
    results['gray'] = gray

    # Step 2: Apply blur
    results['blur'] = apply_gaussian_blur(gray, blur_kernel)

    # Step 3: Apply threshold
    results['threshold'] = apply_threshold(results['blur'], threshold_value)

    # Step 4: Detect edges
    results['edges'] = detect_edges(results['blur'], canny_low, canny_high)

    # Step 5: Find contours (using edges)
    results['contours'] = find_contours(results['edges'])

    # Step 6: Draw contours on original enhanced image
    results['contours_drawn'] = draw_contours_on_image(enhanced, results['contours'])

    return results
