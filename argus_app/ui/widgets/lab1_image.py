"""Level 1 image effects with a manually selected, enlarged ROI."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
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
        self.roi_region = None
        self.roi_processed_frame = None
        self.processed_frame = None
        self.enhanced_frame = None  # retained as an alias for existing app integrations
        self.evidence_name = ""
        self.level_module = None
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
        self.effect_selector.addItems(("Grayscale", "CLAHE", "Threshold"))
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
        self.roi_enable_check.setChecked(False)
        self.roi_enable_check.toggled.connect(self._on_roi_toggled)
        roi_row.addWidget(self.roi_enable_check)

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

        self._on_roi_toggled(False)

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

        self.roi_section = QWidget()
        roi_layout = QVBoxLayout(self.roi_section)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(8)
        self.roi_heading = QLabel("Region of Interest (ROI) Detail")
        self.roi_heading.setStyleSheet(
            f"color: {COLORS['text_main']}; font-size: {FONTS['size_md']}; font-weight: 600;"
        )
        roi_layout.addWidget(self.roi_heading)
        self.roi_panels_container = QWidget()
        self.roi_panels_layout = QBoxLayout(QBoxLayout.LeftToRight, self.roi_panels_container)
        self.roi_panels_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_panels_layout.setSpacing(12)
        self.roi_processed_panel = self._make_panel(
            "Processed ROI", "Enable ROI Crop and enter X, Y, width, and height."
        )
        self.roi_processed_canvas = self.roi_processed_panel[1]
        self.roi_panels_layout.addWidget(self.roi_processed_panel[0], 1)
        roi_layout.addWidget(self.roi_panels_container, 1)
        layout.addWidget(self.roi_section, 1)

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
            if hasattr(self, "roi_panels_layout"):
                self.roi_panels_layout.setDirection(direction)

    def _request_upload(self):
        self.canvas_orig.upload_requested.emit()

    def set_evidence_name(self, name):
        self.evidence_name = str(name or "")

    def ensure_implementation(self):
        if self.level_module is not None:
            return True
        module, message = self.load_level_implementation(
            1, ("apply_effect", "process_roi", "ROI_ENLARGEMENT")
        )
        if module is None:
            self.status_label.setText(message)
            self.status_label.setStyleSheet(
                f"color: {COLORS['alert']}; font-size: {FONTS['size_sm']};"
            )
            return False
        self.level_module = module
        if self.status_label.text().startswith("Level "):
            self.status_label.setText("Ready to upload an image.")
            self.status_label.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: {FONTS['size_sm']};"
            )
        return True

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
        self.roi_x_spin.setEnabled(checked)
        self.roi_y_spin.setEnabled(checked)
        self.roi_w_spin.setEnabled(checked)
        self.roi_h_spin.setEnabled(checked)
        if self.raw_evidence_frame is not None:
            self._process_selected_effect(self.effect_selector.currentText())

    def _on_roi_spinbox_changed(self):
        if self.raw_evidence_frame is not None and self.roi_enable_check.isChecked():
            self._process_selected_effect(self.effect_selector.currentText())

    def _set_roi_values(self, x, y, w, h):
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
        if not self.ensure_implementation():
            raise RuntimeError(self.status_label.text())
        filtered_full_frame = self.level_module.apply_effect(frame, effect, clip_limit, threshold_value)
        original_frame = frame.copy()

        if not self.roi_enable_check.isChecked():
            return original_frame, filtered_full_frame, None, None

        roi = (
            self.roi_x_spin.value(), self.roi_y_spin.value(),
            self.roi_w_spin.value(), self.roi_h_spin.value(),
        )
        processed_detail, region = self.level_module.process_roi(
            frame, effect, region=roi,
            clahe_clip_limit=clip_limit, threshold_value=threshold_value,
        )
        return original_frame, filtered_full_frame, region, processed_detail

    def load_frame(self, frame):
        """Load and immediately process a valid decoded image."""
        if not self.ensure_implementation():
            return False
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

        self._set_roi_values(0, 0, min(100, width), min(100, height))

        try:
            original, result, region, processed_detail = self._process_image(
                frame, self.effect_selector.currentText()
            )
        except Exception as exc:
            self.status_label.setText(f"Could not process the image: {exc}")
            self.log_message.emit(f"Could not load/process the image: {exc}", "error")
            return False

        self.raw_evidence_frame = frame.copy()
        self.roi_region = region
        self.roi_processed_frame = processed_detail
        self.processed_frame = result
        self.enhanced_frame = self.processed_frame
        self.canvas_orig.set_cv_image(original)
        self.canvas_proc.set_cv_image(result)
        if processed_detail is not None:
            self.roi_processed_canvas.set_cv_image(processed_detail)
        else:
            self.roi_processed_canvas.setText("Enable ROI Crop and enter X, Y, width, and height.")
        self.effect_selector.setEnabled(True)
        self._update_parameter_visibility()
        self.record_button.setEnabled(True)
        if region:
            _, _, width_box, height_box = region
            self.status_label.setText(
                f"ROI ({region[0]}, {region[1]}, {width_box}, {height_box}) · enlarged {self.level_module.ROI_ENLARGEMENT}× · "
                f"source image {width} × {height} px"
            )
        else:
            self.status_label.setText("Full image processed without ROI cropping.")
        self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: {FONTS['size_sm']};")
        return True

    def process_frame(self, frame):
        try:
            original, output, region, processed_detail = self._process_image(
                frame, self.effect_selector.currentText()
            )
            return {
                "output": output,
                "preview": original,
                "detail_source": None,
                "detail_output": processed_detail,
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
            original, result, region, processed_detail = self._process_image(
                self.raw_evidence_frame, effect
            )
            self.roi_region = region
            self.roi_processed_frame = processed_detail
            self.processed_frame = result
            self.enhanced_frame = self.processed_frame
            self.canvas_orig.set_cv_image(original)
            self.canvas_proc.set_cv_image(result)
            if processed_detail is not None:
                self.roi_processed_canvas.set_cv_image(processed_detail)
            else:
                self.roi_processed_canvas.setText("Enable ROI Crop and enter X, Y, width, and height.")
            self.record_button.setEnabled(True)
            self._update_parameter_visibility()
            if region:
                _, _, width_box, height_box = region
                self.status_label.setText(
                    f"ROI ({region[0]}, {region[1]}, {width_box}, {height_box}) · enlarged {self.level_module.ROI_ENLARGEMENT}×"
                )
            else:
                self.status_label.setText("Full image processed without ROI cropping.")
            self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: {FONTS['size_sm']};")
        except Exception as exc:
            self.processed_frame = None
            self.enhanced_frame = None
            self.roi_region = None
            self.roi_processed_frame = None
            self.record_button.setEnabled(False)
            self.canvas_proc.setText("Processing failed. Choose another effect or image.")
            self.roi_processed_canvas.setText("ROI processing failed. Check its coordinates.")
            self.status_label.setText(f"Could not process the image: {exc}")
            self.status_label.setStyleSheet(f"color: {COLORS['alert']}; font-size: {FONTS['size_sm']};")
            self.log_message.emit(f"Could not process the image: {exc}", "error")

    def record_result(self):
        if not self.ensure_implementation():
            return
        if self.raw_evidence_frame is None or self.processed_frame is None:
            self.log_message.emit("Load and process an image before recording a result.", "warning")
            return
        try:
            output_path = save_image(1, "processed_timestamp_view.png", self.processed_frame)
        except (OSError, ValueError) as exc:
            self.log_message.emit(f"Could not save the processed result: {exc}", "error")
            return
        region_desc = (
            f" ROI crop {self.roi_region} enlarged {self.level_module.ROI_ENLARGEMENT}×."
            if self.roi_region
            else " Full image processed without ROI cropping."
        )
        effect = self.effect_selector.currentText()
        parameters = {
            "clahe_clip_limit": self.clahe_parameter_slider.value() / 10.0,
            "threshold_value": self.threshold_parameter_slider.value(),
        }
        registry.add(EvidenceEntry(
            module="Level 1: Sight Repair", kind="image_processing",
            value={"effect": effect, "roi_region": self.roi_region, **parameters},
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
        self.roi_region = None
        self.roi_processed_frame = None
        self.processed_frame = None
        self.enhanced_frame = None
        self.evidence_name = ""
        self.canvas_orig.setText("Upload a CCTV image to begin.")
        self.canvas_proc.setText("Select an effect to see the processed image.")
        self.roi_processed_canvas.setText("Enable ROI Crop and enter X, Y, width, and height.")
        self.roi_enable_check.blockSignals(True)
        self.roi_enable_check.setChecked(False)
        self.roi_enable_check.blockSignals(False)
        self._set_roi_values(0, 0, 100, 100)
        self._on_roi_toggled(self.roi_enable_check.isChecked())
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
