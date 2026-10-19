"""Level 1 CCTV filter preview with an enlarged automatic clock detail."""

from __future__ import annotations

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
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import LabWidget
from evidence.artifacts import save_image
from evidence.registry import EvidenceEntry, registry


from level_codes.level1 import (
    CLOCK_ZOOM, EFFECTS, LEVEL1_CLOCK_EVIDENCE, apply_effect,
    configured_clock_roi, process_clock_region,
)


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
        self.clock_processed_frame = None
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

        parameter_row = QHBoxLayout()
        self.clahe_parameter_label = QLabel("CLAHE clip limit: 2.0")
        self.clahe_parameter_slider = QSlider(Qt.Horizontal)
        self.clahe_parameter_slider.setRange(5, 50)
        self.clahe_parameter_slider.setValue(20)
        self.clahe_parameter_slider.setToolTip("Higher clip limits produce stronger local contrast and can also amplify noise.")
        self.threshold_parameter_label = QLabel("Binary threshold: 128")
        self.threshold_parameter_slider = QSlider(Qt.Horizontal)
        self.threshold_parameter_slider.setRange(0, 255)
        self.threshold_parameter_slider.setValue(128)
        self.threshold_parameter_slider.setToolTip("Pixels above this grayscale value become white; lower values become black.")
        parameter_row.addWidget(self.clahe_parameter_label)
        parameter_row.addWidget(self.clahe_parameter_slider, 1)
        parameter_row.addWidget(self.threshold_parameter_label)
        parameter_row.addWidget(self.threshold_parameter_slider, 1)
        layout.addLayout(parameter_row)
        self.clahe_parameter_slider.valueChanged.connect(self._parameter_changed)
        self.threshold_parameter_slider.valueChanged.connect(self._parameter_changed)
        self._update_parameter_visibility()

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

        self.clock_section = QWidget()
        clock_layout = QVBoxLayout(self.clock_section)
        clock_layout.setContentsMargins(0, 0, 0, 0)
        clock_layout.setSpacing(8)
        self.clock_heading = QLabel("Clock Detail Analysis")
        self.clock_heading.setStyleSheet(
            f"color: {COLORS['text_main']}; font-size: {FONTS['size_md']}; font-weight: 600;"
        )
        clock_layout.addWidget(self.clock_heading)
        self.clock_panels_container = QWidget()
        self.clock_processed_panel = self._make_panel(
            "Processed Clock Detail", "Upload the supplied Level 1 evidence to inspect its enhanced clock crop."
        )
        self.clock_processed_canvas = self.clock_processed_panel[1]
        self.clock_panels_container_layout = QVBoxLayout(self.clock_panels_container)
        self.clock_panels_container_layout.setContentsMargins(0, 0, 0, 0)
        self.clock_panels_container_layout.addWidget(self.clock_processed_panel[0])
        clock_layout.addWidget(self.clock_panels_container, 1)
        layout.addWidget(self.clock_section, 1)

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

    def _update_parameter_visibility(self):
        effect = self.effect_selector.currentText()
        has_image = self.raw_evidence_frame is not None
        self.clahe_parameter_label.setText(f"CLAHE clip limit: {self.clahe_parameter_slider.value() / 10.0:.1f}")
        self.threshold_parameter_label.setText(f"Binary threshold: {self.threshold_parameter_slider.value()}")
        self.clahe_parameter_label.setVisible(effect == "CLAHE")
        self.clahe_parameter_slider.setVisible(effect == "CLAHE")
        self.clahe_parameter_slider.setEnabled(has_image and effect == "CLAHE")
        self.threshold_parameter_label.setVisible(effect == "Threshold")
        self.threshold_parameter_slider.setVisible(effect == "Threshold")
        self.threshold_parameter_slider.setEnabled(has_image and effect == "Threshold")

    def _parameter_changed(self, _value):
        self._update_parameter_visibility()
        if self.raw_evidence_frame is not None:
            self._process_selected_effect(self.effect_selector.currentText())

    def _process_image(self, frame, effect):
        """Process the full frame and, only for the named case asset, its clock ROI."""
        clip_limit = self.clahe_parameter_slider.value() / 10.0
        threshold_value = self.threshold_parameter_slider.value()
        filtered_full_frame = apply_effect(frame, effect, clip_limit, threshold_value)
        region = configured_clock_roi(frame, self.evidence_name)
        if region is None:
            return frame.copy(), filtered_full_frame, None, None
        detail, region = process_clock_region(frame, effect, region, clip_limit, threshold_value)
        return frame.copy(), filtered_full_frame, region, detail

    def load_frame(self, frame):
        """Load and immediately process a valid decoded image."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            self.status_label.setText("Could not load the image. Choose a valid, non-empty image file.")
            return False
        try:
            # Validate the format before changing the existing display/state.
            original, result, region, detail = self._process_image(
                frame, self.effect_selector.currentText()
            )
        except Exception as exc:
            self.status_label.setText(f"Could not process the image: {exc}")
            self.log_message.emit(f"Could not load/process the image: {exc}", "error")
            return False
        self.raw_evidence_frame = frame.copy()
        self.clock_region = region
        self.clock_processed_frame = detail
        self.processed_frame = result
        self.enhanced_frame = self.processed_frame
        self.canvas_orig.set_cv_image(original)
        self.canvas_proc.set_cv_image(result)
        if detail is not None:
            self.clock_processed_canvas.set_cv_image(detail)
        else:
            message = (
                f"Automatic clock cropping is configured for {LEVEL1_CLOCK_EVIDENCE}. "
                "This upload is shown above without a clock crop."
            )
            self.clock_processed_canvas.setText(message)
        self.effect_selector.setEnabled(True)
        self._update_parameter_visibility()
        self.record_button.setEnabled(True)
        if region:
            _, _, width, height = region
            self.status_label.setText(
                f"Clock ROI validated · {width} × {height} px source crop · enlarged {CLOCK_ZOOM}×"
            )
        else:
            self.status_label.setText(
                f"Full image processed. Clock cropping is configured for {LEVEL1_CLOCK_EVIDENCE}."
            )
        return True

    def process_frame(self, frame):
        try:
            original, output, region, detail = self._process_image(
                frame, self.effect_selector.currentText()
            )
            return {
                "output": output,
                "preview": original,
                "detail_source": None,
                "detail_output": detail,
                "region": region,
                "success": True,
                "effect": self.effect_selector.currentText(),
            }
        except Exception as exc:
            return {"output": None, "success": False, "error": str(exc)}

    def _process_selected_effect(self, effect):
        self._update_parameter_visibility()
        if self.raw_evidence_frame is None:
            return
        try:
            original, result, region, detail = self._process_image(
                self.raw_evidence_frame, effect
            )
            self.clock_region = region
            self.clock_processed_frame = detail
            self.processed_frame = result
            self.enhanced_frame = self.processed_frame
            self.canvas_proc.set_cv_image(result)
            if detail is not None:
                self.clock_processed_canvas.set_cv_image(detail)
            self.record_button.setEnabled(True)
            self._update_parameter_visibility()
            if region:
                _, _, width, height = region
                self.status_label.setText(f"Clock ROI processed · {width} × {height} px source crop · enlarged {CLOCK_ZOOM}×")
            else:
                self.status_label.setText(
                    f"Full image processed. Clock cropping is configured for {LEVEL1_CLOCK_EVIDENCE}."
                )
            self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: {FONTS['size_sm']};")
        except Exception as exc:
            self.processed_frame = None
            self.enhanced_frame = None
            self.record_button.setEnabled(False)
            self.canvas_proc.setText("Processing failed. Choose another effect or image.")
            self.clock_processed_canvas.setText("Processing failed. Choose another effect or image.")
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
        region = (
            f" Clock crop {self.clock_region} enlarged {CLOCK_ZOOM}×."
            if self.clock_region
            else " No configured clock crop was applied to this general upload."
        )
        effect = self.effect_selector.currentText()
        parameters = {
            "clahe_clip_limit": self.clahe_parameter_slider.value() / 10.0,
            "threshold_value": self.threshold_parameter_slider.value(),
        }
        registry.add(EvidenceEntry(
            module="Level 1: Sight Repair", kind="image_processing",
            value={"effect": effect, "clock_region": self.clock_region, **parameters},
            technique=f"OpenCV {effect}" + (f" (clip limit {parameters['clahe_clip_limit']:.1f})" if effect == "CLAHE" else f" (threshold {parameters['threshold_value']})" if effect == "Threshold" else ""),
            observation=f"Applied {effect} to the selected evidence using the displayed parameter values.{region}",
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
        self.clock_processed_frame = None
        self.processed_frame = None
        self.enhanced_frame = None
        self.evidence_name = ""
        self.canvas_orig.setText("Upload a CCTV image to begin.")
        self.canvas_proc.setText("Select an effect to see the processed image.")
        self.clock_processed_canvas.setText("Upload the supplied Level 1 evidence to inspect its enhanced clock crop.")
        self.effect_selector.blockSignals(True)
        self.effect_selector.setCurrentIndex(0)
        self.effect_selector.blockSignals(False)
        self.effect_selector.setEnabled(False)
        self.clahe_parameter_slider.blockSignals(True)
        self.clahe_parameter_slider.setValue(20)
        self.clahe_parameter_slider.blockSignals(False)
        self.threshold_parameter_slider.blockSignals(True)
        self.threshold_parameter_slider.setValue(128)
        self.threshold_parameter_slider.blockSignals(False)
        self._update_parameter_visibility()
        self.record_button.setEnabled(False)
        self.status_label.setText("Ready to upload an image.")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {FONTS['size_sm']};")
