"""Read selected still images and looping video files with OpenCV."""

from enum import Enum
from pathlib import Path

import cv2
import numpy as np


def read_image_file(filepath):
    """Read image files through OpenCV while supporting Unicode Windows paths."""
    try:
        encoded = np.fromfile(str(filepath), dtype=np.uint8)
        if encoded.size:
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is not None:
                return image
    except (OSError, ValueError):
        pass
    return cv2.imread(str(filepath), cv2.IMREAD_COLOR)


class SourceType(Enum):
    VIDEO = "video"
    IMAGE = "image"


class InputSource:
    def __init__(self):
        self.source_type = None
        self.capture = None
        self.image = None
        self.is_active = False
        self.frame_count = 1
        self.frame_position = 0
        self.fps = 0.0

    def open_video(self, filepath):
        self.release()
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.release()
            raise RuntimeError(f"OpenCV could not open video file: {path}")
        self.source_type = SourceType.VIDEO
        self.frame_count = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_position = 0
        self.is_active = True

    def open_image(self, filepath):
        self.release()
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        self.image = read_image_file(path)
        if self.image is None:
            raise RuntimeError(f"OpenCV could not read image file: {path}")
        self.source_type = SourceType.IMAGE
        self.frame_count = 1
        self.frame_position = 0
        self.fps = 0.0
        self.is_active = True

    def read(self):
        if not self.is_active:
            return False, None
        if self.source_type == SourceType.IMAGE:
            return True, self.image.copy()

        ok, frame = self.capture.read()
        if not ok:
            return False, None
        self.frame_position = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        return ok, frame

    def seek_frame(self, frame_number):
        if self.source_type != SourceType.VIDEO or self.capture is None:
            return False, None
        target = max(0, min(int(frame_number), self.frame_count - 1))
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, target):
            raise RuntimeError("The video reader could not seek to that frame.")
        return self.read()

    def release(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.image = None
        self.source_type = None
        self.is_active = False
        self.frame_count = 1
        self.frame_position = 0
        self.fps = 0.0

    def is_opened(self):
        return self.is_active
