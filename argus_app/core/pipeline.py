"""Small, shared utilities for the desktop frame-processing loop."""

import time

import cv2


def downscale_frame(frame, max_dimension=1280):
    """Limit the largest frame dimension while preserving its aspect ratio."""
    height, width = frame.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return frame
    scale = max_dimension / largest
    return cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


class FPSCounter:
    """Calculate a short rolling estimate of the frame-processing rate."""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []
        self.fps = 0.0

    def tick(self):
        self.timestamps.append(time.monotonic())
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        if len(self.timestamps) > 1:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self.fps = (len(self.timestamps) - 1) / elapsed

    def get_fps(self):
        return self.fps

    def reset(self):
        self.timestamps.clear()
        self.fps = 0.0
