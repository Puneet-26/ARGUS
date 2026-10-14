"""
Level 4: Motion Analysis Lab - REFERENCE SOLUTION

Complete implementation of motion detection and tracking.
"""
import cv2
import numpy as np


def create_background_subtractor():
    """
    Create a background subtractor object.

    Returns:
        Background subtractor object
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )


def apply_background_subtraction(bg_subtractor, frame):
    """
    Apply background subtraction to detect motion.

    Args:
        bg_subtractor: Background subtractor object
        frame: Input BGR frame

    Returns:
        Binary foreground mask
    """
    return bg_subtractor.apply(frame)


def clean_motion_mask(mask, kernel_size=5):
    """
    Clean up motion mask using morphological operations.

    Args:
        mask: Binary motion mask
        kernel_size: Size of morphological kernel

    Returns:
        Cleaned mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Opening removes noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing fills holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_motion_contours(mask, min_area=500):
    """
    Find contours in motion mask and filter by area.

    Args:
        mask: Binary motion mask
        min_area: Minimum contour area to keep

    Returns:
        List of valid contours
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    return valid_contours


def compute_contour_centroids(contours):
    """
    Compute centroids for a list of contours.

    Args:
        contours: List of contours

    Returns:
        List of (x, y) centroid tuples
    """
    centroids = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    return centroids


def draw_tracked_objects(image, tracked_objects, trails=None):
    """
    Draw tracked objects with IDs and optional trails.

    Args:
        image: Image to draw on (will be copied)
        tracked_objects: Dict of {object_id: (x, y)}
        trails: Optional dict of {object_id: list of points}

    Returns:
        Image with tracking visualization
    """
    output = image.copy()

    # Draw trails first (under objects)
    if trails:
        for obj_id, trail in trails.items():
            if len(trail) > 1:
                # Convert to numpy array for polylines
                points = np.array(trail, dtype=np.int32)
                cv2.polylines(output, [points], False, (0, 255, 0), 2)

    # Draw tracked objects
    for obj_id, (x, y) in tracked_objects.items():
        # Draw circle at centroid
        cv2.circle(output, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(output, (x, y), 10, (255, 255, 255), 2)

        # Draw ID text
        text = f"ID:{obj_id}"
        cv2.putText(output, text, (x + 15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output


def draw_exit_line(image, line_start, line_end, color=(0, 255, 255), thickness=3):
    """
    Draw exit/crossing line on image.

    Args:
        image: Image to draw on (modified in place)
        line_start: (x, y) tuple
        line_end: (x, y) tuple
        color: Line color (B, G, R)
        thickness: Line thickness

    Returns:
        Image with line drawn
    """
    output = image.copy()
    cv2.line(output, line_start, line_end, color, thickness)

    # Draw endpoints
    cv2.circle(output, line_start, 6, color, -1)
    cv2.circle(output, line_end, 6, color, -1)

    return output


def check_line_crossing(prev_pos, curr_pos, line_start, line_end):
    """
    Check if movement from prev_pos to curr_pos crosses the line.

    Args:
        prev_pos: Previous position (x, y)
        curr_pos: Current position (x, y)
        line_start: Line start point (x, y)
        line_end: Line end point (x, y)

    Returns:
        True if crossed, False otherwise
    """
    # Convert to numpy arrays
    p1 = np.array(prev_pos, dtype=np.float32)
    p2 = np.array(curr_pos, dtype=np.float32)
    p3 = np.array(line_start, dtype=np.float32)
    p4 = np.array(line_end, dtype=np.float32)

    # Line segment intersection using parametric equations
    denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])

    if abs(denom) < 1e-6:
        return False  # Parallel lines

    ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom
    ub = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / denom

    # Check if intersection occurs within both segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return True

    return False


def process_motion_analysis(frame, bg_subtractor, tracker, line_crossing,
                            exit_line_start, exit_line_end, min_area=500):
    """
    Complete motion analysis pipeline.

    Args:
        frame: Input BGR frame
        bg_subtractor: Background subtractor object (persistent)
        tracker: CentroidTracker object (persistent)
        line_crossing: LineCrossing object (persistent)
        exit_line_start: Exit line start point
        exit_line_end: Exit line end point
        min_area: Minimum contour area

    Returns:
        Dictionary with keys: 'mask', 'mask_cleaned', 'tracked_objects',
        'trails', 'result', 'cross_count'
    """
    results = {}

    # Step 1: Apply background subtraction
    mask = apply_background_subtraction(bg_subtractor, frame)
    results['mask'] = mask

    # Step 2: Clean mask
    mask_cleaned = clean_motion_mask(mask)
    results['mask_cleaned'] = mask_cleaned

    # Step 3: Find contours
    contours = find_motion_contours(mask_cleaned, min_area)
    results['contours'] = contours

    # Step 4: Compute centroids
    centroids = compute_contour_centroids(contours)

    # Step 5: Update tracker
    tracked_objects = tracker.update(centroids)
    results['tracked_objects'] = tracked_objects

    # Step 6: Update line crossing detector
    line_crossing.update_line(exit_line_start, exit_line_end)

    # Check line crossings
    old_crossed = set(line_crossing.crossed_ids)
    for obj_id, position in tracked_objects.items():
        line_crossing.check_crossing(obj_id, position)
    results['crossed_ids'] = sorted(line_crossing.crossed_ids - old_crossed)

    results['cross_count'] = line_crossing.get_count()

    # Step 7: Get trails
    trails = {}
    for obj_id in tracked_objects.keys():
        trails[obj_id] = tracker.get_trail(obj_id)
    results['trails'] = trails

    # Step 8: Draw visualization
    result = frame.copy()

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + width, y + height), (60, 200, 255), 2)

    # Draw tracked objects and trails
    result = draw_tracked_objects(result, tracked_objects, trails)

    # Draw exit line
    result = draw_exit_line(result, exit_line_start, exit_line_end)

    # Add crossing count text
    cv2.putText(result, f"Crossings: {results['cross_count']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    results['result'] = result

    return results
