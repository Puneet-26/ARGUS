"""Level 1 CCTV filter preview with an enlarged automatic clock detail."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import LabWidget
from evidence.artifacts import save_image
from evidence.registry import EvidenceEntry, registry


from level_codes.level1 import (
    CLOCK_ZOOM, EFFECTS, LEVEL1_CLOCK_EVIDENCE, apply_effect,
    configured_clock_roi, get_default_roi, process_roi, process_clock_region,
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

        roi_row = QHBoxLayout()
        roi_row.setSpacing(8)
        self.roi_enable_check = QCheckBox("Enable ROI Crop")
        self.roi_enable_check.setChecked(True)
        self.roi_enable_check.toggled.connect(self._on_roi_toggled)
        roi_row.addWidget(self.roi_enable_check)

        roi_row.addWidget(QLabel("Preset:"))
        self.roi_preset_combo = QComboBox()
        self.roi_preset_combo.addItems([
            "Auto / Detect",
            "Center (25%)",
            "Top-Left",
            "Top-Right",
            "Bottom-Left",
            "Bottom-Right",
            "Full Frame",
            "Custom",
        ])
        self.roi_preset_combo.currentTextChanged.connect(self._on_roi_preset_changed)
        roi_row.addWidget(self.roi_preset_combo)

        roi_row.addWidget(QLabel("X:"))
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 10000)
        self.roi_x_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        roi_row.addWidget(self.roi_x_spin)

        roi_row.addWidget(QLabel("Y:"))
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 10000)
        self.roi_y_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        roi_row.addWidget(self.roi_y_spin)

        roi_row.addWidget(QLabel("W:"))
        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(1, 10000)
        self.roi_w_spin.setValue(100)
        self.roi_w_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        roi_row.addWidget(self.roi_w_spin)

        roi_row.addWidget(QLabel("H:"))
        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(1, 10000)
        self.roi_h_spin.setValue(100)
        self.roi_h_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        roi_row.addWidget(self.roi_h_spin)

        roi_row.addWidget(QLabel("Zoom:"))
        self.roi_zoom_spin = QSpinBox()
        self.roi_zoom_spin.setRange(1, 10)
        self.roi_zoom_spin.setValue(CLOCK_ZOOM)
        self.roi_zoom_spin.valueChanged.connect(self._on_roi_spinbox_changed)
        roi_row.addWidget(self.roi_zoom_spin)

        roi_row.addStretch(1)
        layout.addLayout(roi_row)

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
        self.clock_heading = QLabel("Region of Interest (ROI) Analysis")
        self.clock_heading.setStyleSheet(
            f"color: {COLORS['text_main']}; font-size: {FONTS['size_md']}; font-weight: 600;"
        )
        clock_layout.addWidget(self.clock_heading)
        self.clock_panels_container = QWidget()
        self.clock_processed_panel = self._make_panel(
            "Processed ROI Detail", "Upload an image and adjust ROI controls to inspect enhanced details."
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

    def _on_roi_toggled(self, checked):
        self.roi_preset_combo.setEnabled(checked)
        self.roi_x_spin.setEnabled(checked)
        self.roi_y_spin.setEnabled(checked)
        self.roi_w_spin.setEnabled(checked)
        self.roi_h_spin.setEnabled(checked)
        self.roi_zoom_spin.setEnabled(checked)
        if self.raw_evidence_frame is not None:
            self._process_selected_effect(self.effect_selector.currentText())

    def _on_roi_preset_changed(self, text):
        if self.raw_evidence_frame is None or not self.roi_enable_check.isChecked():
            return
        height, width = self.raw_evidence_frame.shape[:2]
        if text == "Auto / Detect":
            roi = get_default_roi(self.raw_evidence_frame, self.evidence_name)
        elif text == "Center (25%)":
            w = max(10, int(width * 0.25))
            h = max(10, int(height * 0.25))
            roi = ((width - w) // 2, (height - h) // 2, w, h)
        elif text == "Top-Left":
            w = max(10, int(width * 0.25))
            h = max(10, int(height * 0.25))
            roi = (0, 0, w, h)
        elif text == "Top-Right":
            w = max(10, int(width * 0.25))
            h = max(10, int(height * 0.25))
            roi = (width - w, 0, w, h)
        elif text == "Bottom-Left":
            w = max(10, int(width * 0.25))
            h = max(10, int(height * 0.25))
            roi = (0, height - h, w, h)
        elif text == "Bottom-Right":
            w = max(10, int(width * 0.25))
            h = max(10, int(height * 0.25))
            roi = (width - w, height - h, w, h)
        elif text == "Full Frame":
            roi = (0, 0, width, height)
        else:
            return

        if roi:
            x, y, w, h = roi
            self._set_spinboxes_blocked(x, y, w, h)
            self._process_selected_effect(self.effect_selector.currentText())

    def _on_roi_spinbox_changed(self):
        if self.raw_evidence_frame is None:
            return
        if self.roi_preset_combo.currentText() != "Custom":
            self.roi_preset_combo.blockSignals(True)
            self.roi_preset_combo.setCurrentText("Custom")
            self.roi_preset_combo.blockSignals(False)
        self._process_selected_effect(self.effect_selector.currentText())

    def _set_spinboxes_blocked(self, x, y, w, h):
        for spin, val in ((self.roi_x_spin, x), (self.roi_y_spin, y), (self.roi_w_spin, w), (self.roi_h_spin, h)):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    def _parameter_changed(self, _value):
        self._update_parameter_visibility()
        if self.raw_evidence_frame is not None:
            self._process_selected_effect(self.effect_selector.currentText())

    def _process_image(self, frame, effect):
        """Process the full frame and, if enabled, the configured ROI."""
        clip_limit = self.clahe_parameter_slider.value() / 10.0
        threshold_value = self.threshold_parameter_slider.value()
        filtered_full_frame = apply_effect(frame, effect, clip_limit, threshold_value)
        preview_frame = frame.copy()

        if not self.roi_enable_check.isChecked():
            return preview_frame, filtered_full_frame, None, None

        height, width = frame.shape[:2]
        x = max(0, min(self.roi_x_spin.value(), width - 1))
        y = max(0, min(self.roi_y_spin.value(), height - 1))
        w = max(1, min(self.roi_w_spin.value(), width - x))
        h = max(1, min(self.roi_h_spin.value(), height - y))
        roi = (x, y, w, h)
        zoom = self.roi_zoom_spin.value()

        line_w = max(2, int(min(height, width) / 350))
        cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 220, 255), line_w)

        try:
            detail, region = process_roi(
                frame, effect, region=roi, zoom=zoom,
                clahe_clip_limit=clip_limit, threshold_value=threshold_value,
            )
        except Exception:
            detail, region = None, None

        return preview_frame, filtered_full_frame, region, detail

    def load_frame(self, frame):
        """Load and immediately process a valid decoded image."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            self.status_label.setText("Could not load the image. Choose a valid, non-empty image file.")
            return False

        height, width = frame.shape[:2]
        self.roi_x_spin.blockSignals(True)
        self.roi_y_spin.blockSignals(True)
        self.roi_w_spin.blockSignals(True)
        self.roi_h_spin.blockSignals(True)
        self.roi_x_spin.setMaximum(max(0, width - 1))
        self.roi_y_spin.setMaximum(max(0, height - 1))
        self.roi_w_spin.setMaximum(width)
        self.roi_h_spin.setMaximum(height)
        self.roi_x_spin.blockSignals(False)
        self.roi_y_spin.blockSignals(False)
        self.roi_w_spin.blockSignals(False)
        self.roi_h_spin.blockSignals(False)

        default_roi = get_default_roi(frame, self.evidence_name)
        if default_roi:
            dx, dy, dw, dh = default_roi
            self._set_spinboxes_blocked(dx, dy, dw, dh)
            self.roi_preset_combo.blockSignals(True)
            self.roi_preset_combo.setCurrentText("Auto / Detect")
            self.roi_preset_combo.blockSignals(False)

        try:
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
            self.clock_processed_canvas.setText("ROI crop is disabled or not set. Full processed image is shown above.")
        self.effect_selector.setEnabled(True)
        self._update_parameter_visibility()
        self.record_button.setEnabled(True)
        if region:
            _, _, width_box, height_box = region
            zoom = self.roi_zoom_spin.value()
            self.status_label.setText(
                f"ROI ({region[0]}, {region[1]}, {width_box}, {height_box}) · enlarged {zoom}×"
            )
        else:
            self.status_label.setText("Full image processed without ROI cropping.")
        self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: {FONTS['size_sm']};")
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
            self.canvas_orig.set_cv_image(original)
            self.canvas_proc.set_cv_image(result)
            if detail is not None:
                self.clock_processed_canvas.set_cv_image(detail)
            else:
                self.clock_processed_canvas.setText("ROI crop is disabled. Full processed image is shown above.")
            self.record_button.setEnabled(True)
            self._update_parameter_visibility()
            if region:
                _, _, width_box, height_box = region
                zoom = self.roi_zoom_spin.value()
                self.status_label.setText(f"ROI ({region[0]}, {region[1]}, {width_box}, {height_box}) · enlarged {zoom}×")
            else:
                self.status_label.setText("Full image processed without ROI cropping.")
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
        zoom = self.roi_zoom_spin.value()
        region_desc = (
            f" ROI crop {self.clock_region} enlarged {zoom}×."
            if self.clock_region
            else " Full image processed without ROI cropping."
        )
        effect = self.effect_selector.currentText()
        parameters = {
            "clahe_clip_limit": self.clahe_parameter_slider.value() / 10.0,
            "threshold_value": self.threshold_parameter_slider.value(),
        }
        registry.add(EvidenceEntry(
            module="Level 1: Sight Repair", kind="image_processing",
            value={"effect": effect, "roi_region": self.clock_region, **parameters},
            technique=f"OpenCV {effect}" + (f" (clip limit {parameters['clahe_clip_limit']:.1f})" if effect == "CLAHE" else f" (threshold {parameters['threshold_value']})" if effect == "Threshold" else ""),
            observation=f"Applied {effect} to the selected evidence using the displayed parameter values.{region_desc}",
            interpretation="Read details only if legible in the source and processed view. Enhancement improves visibility but does not fabricate absent detail.",
            uncertainty="Medium", source_filename=Path(self.evidence_name).name,
            source_path=self.evidence_name, output_path=output_path, evidence_id="E-01",
            unresolved_question="Which details/digits are supported by the actual image?",
            limitation="Enhancement can improve visibility but cannot restore detail absent from the source; no details are inferred by the application.",
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
        self.clock_processed_canvas.setText("Upload an image and adjust ROI controls to inspect enhanced details.")
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
