"""Shared base classes and display helpers for the level workspaces."""

import importlib

import cv2
import numpy as np

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class LabWidget(QWidget):
    """Base class for lab widgets."""

    log_message = Signal(str, str)

    def __init__(self, lab_name, parent=None):
        super().__init__(parent)
        self.lab_name = lab_name
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI - override in subclasses."""
        layout = QVBoxLayout(self)

        self.content_widget = QWidget()
        layout.addWidget(self.content_widget, 1)

    @staticmethod
    def load_level_implementation(level_number, required_names=()):
        """Load one level on demand and describe only expected availability issues."""
        module_name = f"level_codes.level{int(level_number)}"
        source_file = f"level_codes/level{int(level_number)}.py"
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name in {module_name, "level_codes"}:
                return None, (
                    f"Level implementation not found: {source_file}. "
                    "Add the corresponding level code file and try again."
                )
            return None, (
                f"Level {level_number} cannot load because '{exc.name}' is missing. "
                "Install the required dependency or add the missing level module."
            )
        except ImportError as exc:
            if getattr(exc, "name", None) in {module_name, "level_codes"}:
                return None, (
                    f"Level implementation not found: {source_file}. "
                    "Add the corresponding level code file and try again."
                )
            if getattr(exc, "name", None):
                return None, (
                    f"Level {level_number} cannot load because '{exc.name}' is unavailable. "
                    "Install the required dependency to use this level."
                )
            raise

        missing = [name for name in required_names if not hasattr(module, name)]
        if missing:
            names = ", ".join(name if name.isupper() else f"{name}()" for name in missing)
            return None, f"Level implementation incomplete: {source_file} is missing {names}."
        return module, ""

    @staticmethod
    def cv_to_qimage(cv_img):
        """Convert OpenCV image to QImage."""
        if cv_img is None:
            return None

        # Ensure array is contiguous in memory
        cv_img = np.ascontiguousarray(cv_img)

        # Convert to RGB if needed
        if len(cv_img.shape) == 2:
            # Grayscale
            height, width = cv_img.shape
            bytes_per_line = width
            qimg = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # BGR to RGB
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            rgb_img = np.ascontiguousarray(rgb_img)
            height, width, channel = rgb_img.shape
            bytes_per_line = channel * width
            qimg = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        return qimg.copy()

    @staticmethod
    def cv_to_pixmap(cv_img, max_width=None, max_height=None):
        """Convert OpenCV image to QPixmap with optional scaling."""
        qimg = LabWidget.cv_to_qimage(cv_img)
        if qimg is None:
            return None

        pixmap = QPixmap.fromImage(qimg)

        if max_width is not None or max_height is not None:
            w = max(1, max_width or pixmap.width())
            h = max(1, max_height or pixmap.height())
            pixmap = pixmap.scaled(
                w,
                h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

        return pixmap


class MediaTimelineControls(QFrame):
    """Shared native playback and frame-seeking controls for evidence videos."""

    play_pause_requested = Signal()
    seek_requested = Signal(int)
    restart_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.source_type = ""
        self.frame_count = 1
        self.fps = 0.0
        self.frame_number = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        self.filename_label = QLabel("No evidence selected")
        self.filename_label.setObjectName("secondaryLabel")
        self.filename_label.setMinimumWidth(80)
        self.filename_label.setWordWrap(True)
        layout.addWidget(self.filename_label)

        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_pause_requested.emit)
        layout.addWidget(self.play_button)

        self.restart_button = QPushButton("Restart")
        self.restart_button.setEnabled(False)
        self.restart_button.clicked.connect(self.restart_requested.emit)
        layout.addWidget(self.restart_button)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderReleased.connect(
            lambda: self.seek_requested.emit(self.seek_slider.value())
        )
        layout.addWidget(self.seek_slider, 1)

        self.position_label = QLabel("Open a video to select a frame.")
        self.position_label.setObjectName("secondaryLabel")
        self.position_label.setMinimumWidth(130)
        layout.addWidget(self.position_label)

    def set_media_info(self, filename, source_type, frame_count=1, fps=0.0):
        self.source_type = getattr(source_type, "value", source_type)
        self.frame_count = max(1, int(frame_count or 1))
        self.fps = float(fps or 0)
        self.filename_label.setText(str(filename or "No evidence selected"))
        is_video = self.source_type == "video"
        self.play_button.setEnabled(is_video)
        self.restart_button.setEnabled(is_video)
        self.seek_slider.setEnabled(is_video and self.frame_count > 1)
        with QSignalBlocker(self.seek_slider):
            self.seek_slider.setRange(0, max(0, self.frame_count - 1))
            self.seek_slider.setValue(0)
        self.frame_number = 0
        self.set_playing(is_video)
        if is_video:
            self.set_frame_position(0, self.frame_count)
        else:
            self.position_label.setText("Still image · no timeline")

    def set_playing(self, playing):
        if self.source_type == "video":
            self.play_button.setText("Pause" if playing else "Play")

    def set_frame_position(self, frame_number, frame_count=None):
        if self.source_type != "video":
            return
        if frame_count is not None:
            self.frame_count = max(1, int(frame_count))
        self.frame_number = max(0, min(int(frame_number), self.frame_count - 1))
        with QSignalBlocker(self.seek_slider):
            self.seek_slider.setRange(0, max(0, self.frame_count - 1))
            self.seek_slider.setValue(self.frame_number)
        if self.fps > 0:
            current = self.frame_number / self.fps
            total = self.frame_count / self.fps
            self.position_label.setText(f"{self._format_time(current)} / {self._format_time(total)}")
        else:
            self.position_label.setText(f"Frame {self.frame_number + 1:,} / {self.frame_count:,}")

    def reset(self):
        self.source_type = ""
        self.frame_count = 1
        self.fps = 0.0
        self.frame_number = 0
        self.filename_label.setText("No evidence selected")
        self.play_button.setText("Play")
        self.play_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.position_label.setText("Open a video to select a frame.")

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"


class ImageDisplayLabel(QLabel):
    """Label for displaying images with sizing."""

    upload_requested = Signal()
    file_dropped = Signal(str)

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(140, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid #404040; background-color: #1a1a1a;")
        self._source_pixmap = None
        self._upload_enabled = False
        self._upload_media_kind = "image"
        self.setAcceptDrops(True)
        self.setText(title if title else "No Image")

    def set_upload_enabled(self, enabled=True, media_kind="image"):
        self._upload_enabled = bool(enabled)
        self._upload_media_kind = media_kind if media_kind in {"video", "image or video"} else "image"
        self.setCursor(Qt.PointingHandCursor if self._upload_enabled else Qt.ArrowCursor)
        if self._source_pixmap is None:
            self._show_empty_state()

    def _show_empty_state(self):
        if self._upload_enabled:
            upload_noun = {
                "video": "a video",
                "image or video": "an image or a video",
            }.get(self._upload_media_kind, "an image")
            self.setText(
                '<div style="text-align:center;">'
                '<span style="font-size:17px; font-weight:600; color:#E2DED5;">Upload Evidence</span>'
                f'<br><span style="font-size:13px; color:#A8ABA5;">Click here or drag and drop {upload_noun}</span>'
                '</div>'
            )
        else:
            self.setText(self.title if self.title else "No Image")

    def mousePressEvent(self, event):
        if (
            self._upload_enabled
            and self._source_pixmap is None
            and event.button() == Qt.LeftButton
        ):
            self.upload_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if self._upload_enabled and event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        if not self._upload_enabled:
            event.ignore()
            return
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.file_dropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
        event.ignore()

    def set_image(self, pixmap):
        """Set image pixmap."""
        if pixmap and not pixmap.isNull():
            self._source_pixmap = pixmap
            self.setCursor(Qt.ArrowCursor)
            self._update_scaled_pixmap()
        else:
            self._source_pixmap = None
            self.setCursor(Qt.PointingHandCursor if self._upload_enabled else Qt.ArrowCursor)
            self.clear()
            self._show_empty_state()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._source_pixmap and not self._source_pixmap.isNull():
            self.setText("")
            self.setPixmap(self._source_pixmap.scaled(
                max(1, self.width() - 4),
                max(1, self.height() - 4),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            ))
