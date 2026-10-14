"""
Simple centroid-based object tracking utilities.
Pure Python - zero Qt imports.
"""
import numpy as np
from collections import OrderedDict, deque


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


class CentroidTracker:
    """
    Simple centroid-based tracker that assigns IDs to objects.
    Uses nearest-neighbor matching.
    """

    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Args:
            max_disappeared: Maximum frames an object can disappear before being deregistered
            max_distance: Maximum distance for matching (pixels)
        """
        self.next_object_id = 1
        self.objects = OrderedDict()  # {id: centroid}
        self.disappeared = OrderedDict()  # {id: frame_count}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Trail history
        self.trails = OrderedDict()  # {id: deque of points}
        self.max_trail_length = 30

    def register(self, centroid):
        """Register a new object."""
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.trails[object_id] = deque(maxlen=self.max_trail_length)
        self.trails[object_id].append(centroid)
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        """Remove an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trails[object_id]

    def update(self, input_centroids):
        """
        Update tracker with new centroids.

        Args:
            input_centroids: List of (x, y) tuples

        Returns:
            OrderedDict of {id: centroid}
        """
        # No detections
        if len(input_centroids) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Deregister if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # No existing objects - register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        else:
            # Match existing objects to new centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance matrix
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    D[i, j] = euclidean_distance(obj_centroid, input_centroid)

            # Find best matches (greedy nearest neighbor)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Check distance threshold
                if D[row, col] > self.max_distance:
                    continue

                # Update existing object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trails[object_id].append(input_centroids[col])

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new detections
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

    def get_trail(self, object_id):
        """Get trail points for an object."""
        return list(self.trails.get(object_id, []))

    def reset(self):
        """Reset tracker."""
        self.objects.clear()
        self.disappeared.clear()
        self.trails.clear()
        self.next_object_id = 1


class LineCrossing:
    """
    Track objects crossing a line.
    """

    def __init__(self, line_start, line_end):
        """
        Args:
            line_start: (x, y) tuple
            line_end: (x, y) tuple
        """
        self.line_start = np.array(line_start, dtype=np.float32)
        self.line_end = np.array(line_end, dtype=np.float32)
        self.crossed_ids = set()
        self.cross_count = 0
        self.previous_positions = {}  # {id: position}

    def update_line(self, line_start, line_end):
        """Update line position."""
        self.line_start = np.array(line_start, dtype=np.float32)
        self.line_end = np.array(line_end, dtype=np.float32)

    def check_crossing(self, object_id, current_pos):
        """
        Check if object crossed the line.

        Args:
            object_id: Object ID
            current_pos: Current position (x, y)

        Returns:
            True if crossed this frame
        """
        if object_id not in self.previous_positions:
            self.previous_positions[object_id] = current_pos
            return False

        prev_pos = self.previous_positions[object_id]
        self.previous_positions[object_id] = current_pos

        # Check if object already crossed
        if object_id in self.crossed_ids:
            return False

        # Check line crossing using cross product
        crossed = self._line_segment_intersection(prev_pos, current_pos)

        if crossed:
            self.crossed_ids.add(object_id)
            self.cross_count += 1
            return True

        return False

    def _line_segment_intersection(self, p1, p2):
        """Check if line segment p1-p2 intersects with the exit line."""
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)

        p3 = self.line_start
        p4 = self.line_end

        # Parametric line equations
        denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])

        if abs(denom) < 1e-6:
            return False

        ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom
        ub = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return True

        return False

    def reset(self):
        """Reset crossing counter."""
        self.crossed_ids.clear()
        self.previous_positions.clear()
        self.cross_count = 0

    def get_count(self):
        """Get total crossing count."""
        return self.cross_count
