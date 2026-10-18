"""Simple two-panel Level 1 CCTV image processing tool."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import LabWidget
from evidence.artifacts import save_image
from evidence.registry import EvidenceEntry, registry


EFFECTS = ("Grayscale", "CLAHE", "Threshold")
CLOCK_ZOOM = 5


def detect_clock_region(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find a compact clock-like line of bright text in the complete CCTV frame.

    Detection is content based so arbitrary uploads never inherit coordinates from
    the Level 1 asset. The candidate must contain several similarly sized glyphs
    grouped on one baseline and be near a frame corner, where CCTV clock overlays
    are normally placed. Ambiguous images are rejected instead of mis-cropped.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("The image must be grayscale, BGR, or BGRA.")

    height, width = gray.shape[:2]
    min_char_height = max(3, int(height * 0.003))
    max_char_height = max(min_char_height + 1, int(height * 0.065))
    block = min(51, max(15, (min(height, width) // 20) | 1))
    if block >= min(height, width):
        block = max(3, (min(height, width) - 1) | 1)
    masks = [
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, -5,
        ),
        cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1],
    ]

    candidates = []
    for mask in masks:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        glyphs = []
        for x, y, w, h, area in stats[1:count]:
            if not min_char_height <= h <= max_char_height:
                continue
            if w > max_char_height or area < max(3, int(h * 0.12)):
                continue
            glyphs.append((int(x), int(y), int(w), int(h)))
        glyphs.sort(key=lambda item: (item[1] + item[3] / 2, item[0]))

        # Form baseline groups using glyph-center distance, then split on large
        # horizontal gaps so unrelated text elsewhere cannot enlarge the crop.
        groups = []
        for glyph in glyphs:
            gx, gy, gw, gh = glyph
            center_y = gy + gh / 2
            matching = next(
                (group for group in groups if abs(group["center"] - center_y) <= max(2, gh * 0.45)),
                None,
            )
            if matching is None:
                groups.append({"center": center_y, "items": [glyph]})
            else:
                matching["items"].append(glyph)
                matching["center"] = float(np.median([g[1] + g[3] / 2 for g in matching["items"]]))

        for group in groups:
            row = sorted(group["items"], key=lambda item: item[0])
            if len(row) < 4:
                continue
            med_h = float(np.median([g[3] for g in row]))
            runs = [[row[0]]]
            for glyph in row[1:]:
                previous = runs[-1][-1]
                gap = glyph[0] - (previous[0] + previous[2])
                if gap > max(2, med_h * 1.35):
                    runs.append([glyph])
                else:
                    runs[-1].append(glyph)
            for run in runs:
                if len(run) < 4:
                    continue
                x0 = min(g[0] for g in run)
                y0 = min(g[1] for g in run)
                x1 = max(g[0] + g[2] for g in run)
                y1 = max(g[1] + g[3] for g in run)
                run_w, run_h = x1 - x0, y1 - y0
                # Require a colon-shaped separator (two small dots stacked
                # vertically) so ordinary corner captions are not mistaken
                # for a clock or timestamp.
                small = [
                    g for g in glyphs
                    if g[3] <= med_h * 0.55 and g[2] <= med_h * 0.55
                    and x0 - med_h * 0.5 <= g[0] + g[2] / 2 <= x1 + med_h * 0.5
                    and y0 - med_h * 0.4 <= g[1] + g[3] / 2 <= y1 + med_h * 1.3
                ]
                has_colon = any(
                    abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)) <= med_h * 0.55
                    and med_h * 0.2 <= (b[1] + b[3] / 2) - (a[1] + a[3] / 2) <= med_h * 0.8
                    for i, a in enumerate(small)
                    for b in small[i + 1:]
                    if b[1] + b[3] / 2 > a[1] + a[3] / 2
                )
                if not has_colon:
                    continue
                if run_w < med_h * 1.6 or run_w > width * 0.30:
                    continue
                if not 1.4 <= run_w / max(1, run_h) <= 16:
                    continue
                # Give the automatic crop a small border without including a
                # large part of the scene, and prefer candidates near corners.
                pad_x = max(3, int(med_h * 0.55))
                pad_y = max(3, int(med_h * 0.7))
                left, top = max(0, x0 - pad_x), max(0, y0 - pad_y)
                right, bottom = min(width, x1 + pad_x), min(height, y1 + pad_y)
                corner_distance = min(
                    (x0 / width) ** 2 + (y0 / height) ** 2,
                    ((width - x1) / width) ** 2 + (y0 / height) ** 2,
                    (x0 / width) ** 2 + ((height - y1) / height) ** 2,
                    ((width - x1) / width) ** 2 + ((height - y1) / height) ** 2,
                )
                score = len(run) + min(run_w / max(1, run_h), 8) - corner_distance * 10
                candidates.append((score, (left, top, right - left, bottom - top)))

    if not candidates:
        raise ValueError(
            "No clear clock or timestamp region was found in this image. "
            "Upload the Level 1 CCTV frame or an image with a visible clock."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, region = candidates[0]
    if best_score < 5:
        raise ValueError("A clock region could not be validated. Choose a clearer image.")
    if len(candidates) > 1 and candidates[1][0] >= best_score * 0.96:
        first, second = candidates[0][1], candidates[1][1]
        overlap_x = max(0, min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0]))
        overlap_y = max(0, min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1]))
        overlap = overlap_x * overlap_y / max(1, min(first[2] * first[3], second[2] * second[3]))
        if overlap < 0.5:
            raise ValueError("More than one possible clock region was found. Choose a clearer image.")
    return region


def process_clock_region(image: np.ndarray, effect: str):
    """Return a zoomed, unenhanced source crop and its selected OpenCV result."""
    x, y, width, height = detect_clock_region(image)
    crop = image[y:y + height, x:x + width]
    if crop.size == 0:
        raise ValueError("The detected clock region is empty.")
    zoomed_source = cv2.resize(
        crop, None, fx=CLOCK_ZOOM, fy=CLOCK_ZOOM, interpolation=cv2.INTER_CUBIC
    )
    if zoomed_source.ndim == 2:
        gray = zoomed_source
    elif zoomed_source.shape[2] == 4:
        gray = cv2.cvtColor(zoomed_source, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(zoomed_source, cv2.COLOR_BGR2GRAY)
    if effect == "Grayscale":
        result = gray
    elif effect == "CLAHE":
        result = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    elif effect == "Threshold":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unknown image effect: {effect}")
    if result.size == 0:
        raise RuntimeError("Image processing returned an empty result.")
    return zoomed_source, result, (x, y, width, height)


def apply_effect(image: np.ndarray, effect: str) -> np.ndarray:
    """Return a processed image without modifying the uploaded source array."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("The image must be grayscale, BGR, or BGRA.")

    source = image.copy()
    if source.ndim == 3 and source.shape[2] == 4:
        source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
    gray = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    if effect == "Grayscale":
        result = gray.copy()
    elif effect == "CLAHE":
        result = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    elif effect == "Threshold":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unknown image effect: {effect}")

    if result is None or result.size == 0:
        raise RuntimeError("Image processing returned an empty result.")
    return result


class ImagePanel(QLabel):
    """Aspect preserving image view with optional drag and drop support."""

    file_dropped = Signal(str)
    upload_requested = Signal()

    def __init__(self, placeholder: str, accept_files: bool = False, parent=None):
        super().__init__(placeholder, parent)
        self.placeholder = placeholder
        self.accept_files = accept_files
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAcceptDrops(accept_files)
        self.setWordWrap(True)
        self.setStyleSheet(
            f"QLabel {{ background: {COLORS['canvas_bg']}; color: {COLORS['text_muted']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: {RADIUS['md']}; "
            f"padding: 12px; font-size: {FONTS['size_sm']}; }}"
        )

    def set_cv_image(self, image):
        self._source_pixmap = LabWidget.cv_to_pixmap(image) if image is not None else None
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self.clear()
            self.setText(self.placeholder)
        else:
            self._fit_pixmap()

    def setText(self, text):
        self._source_pixmap = None
        super().setPixmap(QPixmap())
        super().setText(text)

    def _fit_pixmap(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        super().setText("")
        super().setPixmap(
            self._source_pixmap.scaled(
                max(1, self.width() - 24),
                max(1, self.height() - 24),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self.accept_files and event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if self.accept_files:
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    self.file_dropped.emit(url.toLocalFile())
                    event.acceptProposedAction()
                    return
        event.ignore()


class Lab1ImageWidget(LabWidget):
    """Small image processing interface: upload, select an effect, reset."""

    level_completed = Signal(str)
    reset_requested = Signal()

    def __init__(self, parent=None):
        self.raw_evidence_frame = None
        self.clock_region = None
        self.zoomed_source = None
        self.processed_frame = None
        self.enhanced_frame = None  # retained as an alias for existing app integrations
        self.evidence_name = ""
        super().__init__("Sight Repair", parent)

    def setup_ui(self):
        super().setup_ui()
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.upload_button = QPushButton("Upload CCTV Image")
        self.upload_button.setObjectName("primaryButton")
        self.upload_button.setMinimumHeight(42)
        self.upload_button.clicked.connect(self._request_upload)
        controls.addWidget(self.upload_button)

        self.effect_selector = QComboBox()
        self.effect_selector.addItems(EFFECTS)
        self.effect_selector.setMinimumHeight(42)
        self.effect_selector.setMinimumWidth(170)
        self.effect_selector.setToolTip("Choose the image effect to apply to the uploaded image.")
        self.effect_selector.currentTextChanged.connect(self._process_selected_effect)
        controls.addWidget(self.effect_selector)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setMinimumHeight(42)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        controls.addWidget(self.reset_button)
        self.record_button = QPushButton("Record result")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.record_result)
        controls.addWidget(self.record_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.panels_container = QWidget()
        self.panels_layout = QBoxLayout(QBoxLayout.LeftToRight, self.panels_container)
        self.panels_layout.setContentsMargins(0, 0, 0, 0)
        self.panels_layout.setSpacing(12)
        self.original_panel = self._make_panel("Original Image", "Upload a CCTV image to begin.")
        self.processed_panel = self._make_panel("Processed Image", "Select an effect to see the processed image.")
        self.canvas_orig = self.original_panel[1]
        self.canvas_proc = self.processed_panel[1]
        self.panels_layout.addWidget(self.original_panel[0], 1)
        self.panels_layout.addWidget(self.processed_panel[0], 1)
        layout.addWidget(self.panels_container, 1)

        self.status_label = QLabel("Ready to upload an image.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {FONTS['size_sm']};")
        layout.addWidget(self.status_label)
        self.effect_selector.setEnabled(False)

    def _make_panel(self, title: str, placeholder: str):
        panel = QFrame()
        panel.setObjectName("lab1ImagePanel")
        panel.setStyleSheet(
            f"QFrame#lab1ImagePanel {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: {RADIUS['md']}; }}"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)
        heading = QLabel(title)
        heading.setStyleSheet(f"color: {COLORS['text_main']}; font-size: {FONTS['size_md']}; font-weight: 600;")
        image = ImagePanel(placeholder, accept_files=(title == "Original Image"))
        panel_layout.addWidget(heading)
        panel_layout.addWidget(image, 1)
        return panel, image

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "panels_layout"):
            direction = QBoxLayout.LeftToRight if event.size().width() >= 760 else QBoxLayout.TopToBottom
            self.panels_layout.setDirection(direction)

    def _request_upload(self):
        self.canvas_orig.upload_requested.emit()

    def set_evidence_name(self, name):
        self.evidence_name = str(name or "")

    def _process_image(self, frame, effect):
        if Path(self.evidence_name).name.lower() == "level1_cctv_dark.png":
            source, result, region = process_clock_region(frame, effect)
            return source, result, region
        return frame.copy(), apply_effect(frame, effect), None

    def load_frame(self, frame):
        """Load and immediately process a valid decoded image."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            self.status_label.setText("Could not load the image. Choose a valid, non-empty image file.")
            return False
        try:
            # Validate the format before changing the existing display/state.
            preview, result, region = self._process_image(frame, self.effect_selector.currentText())
        except Exception as exc:
            self.status_label.setText(f"Could not process the image: {exc}")
            self.log_message.emit(f"Could not load/process the image: {exc}", "error")
            return False
        self.raw_evidence_frame = frame.copy()
        self.clock_region = region
        self.zoomed_source = preview
        self.processed_frame = result
        self.enhanced_frame = result
        self.canvas_orig.set_cv_image(preview)
        self.canvas_proc.set_cv_image(result)
        self.effect_selector.setEnabled(True)
        self.record_button.setEnabled(True)
        if region:
            x, y, width, height = region
            self.status_label.setText(f"Clock region detected at ({x}, {y}) · {width} × {height} px · enlarged {CLOCK_ZOOM}×")
        else:
            height, width = frame.shape[:2]
            self.status_label.setText(f"Full image · {width} × {height} px")
        return True

    def process_frame(self, frame):
        try:
            source, output, region = self._process_image(frame, self.effect_selector.currentText())
            return {"output": output, "preview": source, "region": region, "success": True, "effect": self.effect_selector.currentText()}
        except Exception as exc:
            return {"output": None, "success": False, "error": str(exc)}

    def _process_selected_effect(self, effect):
        if self.raw_evidence_frame is None:
            return
        try:
            source, result, region = self._process_image(self.raw_evidence_frame, effect)
            self.clock_region = region
            self.zoomed_source = source
            self.processed_frame = result
            self.enhanced_frame = result
            self.canvas_proc.set_cv_image(result)
            self.record_button.setEnabled(True)
            if region:
                x, y, width, height = region
                self.status_label.setText(f"Clock region detected at ({x}, {y}) · {width} × {height} px · enlarged {CLOCK_ZOOM}×")
            else:
                height, width = self.raw_evidence_frame.shape[:2]
                self.status_label.setText(f"Full image · {width} × {height} px")
            self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: {FONTS['size_sm']};")
        except Exception as exc:
            self.processed_frame = None
            self.enhanced_frame = None
            self.record_button.setEnabled(False)
            self.canvas_proc.setText("Processing failed. Choose another effect or image.")
            self.status_label.setText(f"Could not process the image: {exc}")
            self.status_label.setStyleSheet(f"color: {COLORS['alert']}; font-size: {FONTS['size_sm']};")
            self.log_message.emit(f"Could not process the image: {exc}", "error")

    def record_result(self):
        if self.raw_evidence_frame is None or self.processed_frame is None:
            self.log_message.emit("Load and process an image before recording a result.", "warning")
            return
        try:
            output_path = save_image(1, "processed_timestamp_view.png", self.processed_frame)
        except (OSError, ValueError) as exc:
            self.log_message.emit(f"Could not save the processed result: {exc}", "error")
            return
        region = f" Clock crop {self.clock_region} enlarged {CLOCK_ZOOM}×." if self.clock_region else " Full image processed."
        effect = self.effect_selector.currentText()
        registry.add(EvidenceEntry(
            module="Level 1: Sight Repair", kind="image_processing",
            value={"effect": effect, "clock_region": self.clock_region},
            technique=f"OpenCV {effect}",
            observation=f"Applied {effect} to the selected evidence.{region}",
            interpretation="Read a timestamp only if its digits are legible in the source and processed view. The image timestamp is not independently verified incident time.",
            uncertainty="Medium", source_filename=Path(self.evidence_name).name,
            source_path=self.evidence_name, output_path=output_path, evidence_id="E-01",
            unresolved_question="Which digits are supported by the actual image?",
            limitation="Enhancement can improve visibility but cannot restore detail absent from the source; no timestamp is inferred by the application.",
        ))
        self.level_completed.emit("Eyes Online")
        self.log_message.emit(f"Level 1 result recorded · {output_path}", "success")

    def reset(self):
        self.raw_evidence_frame = None
        self.clock_region = None
        self.zoomed_source = None
        self.processed_frame = None
        self.enhanced_frame = None
        self.evidence_name = ""
        self.canvas_orig.setText("Upload a CCTV image to begin.")
        self.canvas_proc.setText("Select an effect to see the processed image.")
        self.effect_selector.blockSignals(True)
        self.effect_selector.setCurrentIndex(0)
        self.effect_selector.blockSignals(False)
        self.effect_selector.setEnabled(False)
        self.record_button.setEnabled(False)
        self.status_label.setText("Ready to upload an image.")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {FONTS['size_sm']};")
