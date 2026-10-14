"""Level 3: explore colour regions with OpenCV HSV masks."""

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from evidence.registry import EvidenceEntry, registry
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget


class Lab3ColourWidget(LabWidget):
    """Colour segmentation workspace with useful presets and optional fine controls."""

    level_completed = Signal(str)

    PRESETS = {
        "Red": (170, 10, 90, 255, 35, 255),
        "Orange": (7, 24, 90, 255, 45, 255),
        "Yellow": (22, 38, 80, 255, 60, 255),
        "Green": (36, 85, 55, 255, 35, 255),
        "Blue": (90, 135, 55, 255, 35, 255),
        "Purple": (135, 170, 45, 255, 35, 255),
    }

    def __init__(self, parent=None):
        self.h_min, self.h_max, self.s_min, self.s_max, self.v_min, self.v_max = self.PRESETS["Red"]
        self.morph_kernel = 5
        self.last_results = None
        self.last_frame = None
        self.evidence_name = ""
        self.solution_module = None
        super().__init__("The Red Bag", parent)
        self.load_modules()

    def load_modules(self):
        try:
            from labs.solutions import level3_colour
            self.solution_module = level3_colour
        except ImportError as exc:
            self.log_message.emit(f"Colour analysis is unavailable: {exc}", "error")

    def setup_ui(self):
        super().setup_ui()
        root = QHBoxLayout(self.content_widget)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        viewers = QWidget()
        grid = QGridLayout(viewers)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.original_panel, self.original_display = self._image_panel(
            "ORIGINAL EVIDENCE", "Upload an image to begin."
        )
        self.mask_panel, self.mask_display = self._image_panel(
            "COLOUR MASK", "Run the colour mask to see matching pixels."
        )
        self.isolated_panel, self.isolated_display = self._image_panel(
            "ISOLATED REGION", "Matching pixels will appear here."
        )
        self.result_panel, self.result_display = self._image_panel(
            "REGION OVERLAY", "The mask overlay will appear here."
        )
        grid.addWidget(self.original_panel, 0, 0)
        grid.addWidget(self.mask_panel, 0, 1)
        grid.addWidget(self.isolated_panel, 1, 0)
        grid.addWidget(self.result_panel, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        viewer_layout = QVBoxLayout()
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)
        viewer_layout.addWidget(viewers, 1)
        viewer_container = QWidget()
        viewer_container.setLayout(viewer_layout)
        splitter.addWidget(viewer_container)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(285)
        controls_scroll.setWidget(self._controls_panel())
        splitter.addWidget(controls_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 320])

    def _image_panel(self, title, empty_text):
        panel = QFrame()
        panel.setObjectName("evidencePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        label = QLabel(title)
        label.setObjectName("eyebrowLabel")
        layout.addWidget(label)
        display = ImageDisplayLabel(empty_text)
        if title == "ORIGINAL EVIDENCE":
            display.set_upload_enabled(True, "image")
        layout.addWidget(display, 1)
        return panel, display

    def _controls_panel(self):
        panel = QFrame()
        panel.setObjectName("controlPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel("Colour analysis")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        self.source_label = QLabel("No evidence selected")
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("secondaryLabel")
        layout.addWidget(self.source_label)

        description = QLabel(
            "Choose a colour to highlight pixels in an HSV range. This is colour segmentation: it groups pixels by hue, saturation, and brightness, rather than recognizing an object’s meaning. Lighting and compression affect the mask; a colour match cannot identify a person or prove ownership."
        )
        description.setWordWrap(True)
        description.setObjectName("pageSummary")
        layout.addWidget(description)

        preset_label = QLabel("Colour preset")
        preset_label.setObjectName("sectionHeading")
        layout.addWidget(preset_label)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([*self.PRESETS, "Custom"])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        layout.addWidget(self.preset_combo)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced HSV ranges  ▸")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.advanced_toggle.toggled.connect(self.toggle_advanced)
        layout.addWidget(self.advanced_toggle)

        self.advanced_body = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_body)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(5)
        self.range_sliders = {}
        for key, label, minimum, maximum, value in (
            ("h_min", "Hue minimum", 0, 179, self.h_min),
            ("h_max", "Hue maximum", 0, 179, self.h_max),
            ("s_min", "Saturation minimum", 0, 255, self.s_min),
            ("s_max", "Saturation maximum", 0, 255, self.s_max),
            ("v_min", "Brightness minimum", 0, 255, self.v_min),
            ("v_max", "Brightness maximum", 0, 255, self.v_max),
        ):
            self.range_sliders[key] = self._range_row(advanced_layout, label, minimum, maximum, value)
        self.advanced_body.setVisible(False)
        layout.addWidget(self.advanced_body)

        self.contour_info = QLabel("No colour regions analyzed yet.")
        self.contour_info.setWordWrap(True)
        self.contour_info.setObjectName("statusCard")
        layout.addWidget(self.contour_info)

        self.apply_button = QPushButton("Run colour mask")
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.clicked.connect(self.refresh_current_frame)
        self.apply_button.setEnabled(False)
        layout.addWidget(self.apply_button)

        self.log_button = QPushButton("Record result in notebook")
        self.log_button.clicked.connect(self.log_to_evidence)
        self.log_button.setEnabled(False)
        layout.addWidget(self.log_button)
        layout.addStretch(1)
        return panel

    def _range_row(self, layout, title, minimum, maximum, value):
        row = QHBoxLayout()
        label = QLabel(title)
        label.setMinimumWidth(126)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        value_label = QLabel(str(value))
        value_label.setMinimumWidth(28)
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        layout.addLayout(row)
        slider.valueChanged.connect(lambda v, key=title: value_label.setText(str(v)))
        slider.valueChanged.connect(self.on_custom_range_changed)
        return slider

    def toggle_advanced(self, expanded):
        self.advanced_body.setVisible(expanded)
        self.advanced_toggle.setText("Advanced HSV ranges  ▾" if expanded else "Advanced HSV ranges  ▸")

    def apply_preset(self, name):
        if name not in self.PRESETS:
            return
        values = self.PRESETS[name]
        for key, value in zip(("h_min", "h_max", "s_min", "s_max", "v_min", "v_max"), values):
            with QSignalBlocker(self.range_sliders[key]):
                self.range_sliders[key].setValue(value)
        self._store_ranges()
        self._mark_analysis_stale()

    def on_custom_range_changed(self, _value):
        if self.preset_combo.currentText() != "Custom":
            self.preset_combo.setCurrentText("Custom")
        self._store_ranges()
        self._mark_analysis_stale()

    def _mark_analysis_stale(self):
        if self.last_frame is not None and self.last_results is not None:
            self.last_results = None
            self._clear_analysis_views()
            self.contour_info.setText("Settings changed. Run the mask to refresh the analysis.")
            self.log_button.setEnabled(False)

    def _store_ranges(self):
        for key, slider in self.range_sliders.items():
            setattr(self, key, slider.value())

    def set_evidence_name(self, filename):
        self.evidence_name = str(filename)
        if self.last_frame is None:
            self.source_label.setText(self.evidence_name)

    @property
    def has_evidence(self):
        return self.last_frame is not None

    def reset(self):
        self.last_results = None
        self.last_frame = None
        self.evidence_name = ""
        self.source_label.setText("No evidence selected")
        self.original_display.set_image(None)
        self._clear_analysis_views()
        self.contour_info.setText("Choose a colour preset, then run the mask.")
        self.apply_button.setEnabled(False)
        self.log_button.setEnabled(False)

    def load_evidence_frame(self, frame):
        """Store actual BGR evidence and show it without running analysis."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.reset()
            self.log_message.emit("The selected evidence contains no readable image frame.", "error")
            return False
        if frame.ndim not in (2, 3):
            self.reset()
            self.log_message.emit("The selected evidence has an unsupported image format.", "error")
            return False

        self.last_frame = np.ascontiguousarray(frame).copy()
        self.last_results = None
        self.original_display.set_image(self.cv_to_pixmap(self.last_frame))
        self._clear_analysis_views()
        height, width = self.last_frame.shape[:2]
        name = self.evidence_name or "Selected frame"
        self.source_label.setText(f"{name} · {width} × {height}")
        self.contour_info.setText("Evidence loaded. Choose a colour and run the mask.")
        self.apply_button.setEnabled(True)
        self.log_button.setEnabled(False)
        return True

    def _clear_analysis_views(self):
        self.mask_display.set_image(None)
        self.isolated_display.set_image(None)
        self.result_display.set_image(None)

    def refresh_current_frame(self):
        self.run_colour_mask()

    def run_colour_mask(self):
        if self.last_frame is None:
            self.log_message.emit("Upload an image before running the colour mask.", "warning")
            return {"output": None, "error": "No evidence selected", "success": False}
        if self.solution_module is None:
            self.log_message.emit("Colour analysis is unavailable because its OpenCV module did not load.", "error")
            return {"output": self.last_frame, "error": "Colour analysis unavailable", "success": False}
        try:
            self._store_ranges()
            frame = self.last_frame.copy()
            results = self.solution_module.process_color_segmentation(
                frame,
                h_min=self.h_min,
                h_max=self.h_max,
                s_min=self.s_min,
                s_max=self.s_max,
                v_min=self.v_min,
                v_max=self.v_max,
                morph_kernel=self.morph_kernel,
            )
            if results is None:
                self._clear_analysis_views()
                self.last_results = None
                self.log_button.setEnabled(False)
                return {"output": frame, "error": "Segmentation failed", "success": False}
            self.last_results = results
            self.original_display.set_image(self.cv_to_pixmap(frame))
            self.mask_display.set_image(self.cv_to_pixmap(results.get("mask_cleaned")))
            self.isolated_display.set_image(self.cv_to_pixmap(results.get("isolated")))
            self.result_display.set_image(self.cv_to_pixmap(results.get("result")))
            mask = results.get("mask_cleaned")
            matched_pixels = int(np.count_nonzero(mask)) if mask is not None else 0
            image_pixels = int(mask.size) if mask is not None else int(frame.shape[0] * frame.shape[1])
            coverage = (matched_pixels / image_pixels * 100.0) if image_pixels else 0.0
            contour = results.get("largest_contour")
            if contour is None:
                self.contour_info.setText(
                    f"{matched_pixels:,} matching pixels · {coverage:.2f}% of image · no significant region found."
                )
            else:
                area = cv2.contourArea(contour)
                moments = cv2.moments(contour)
                center_x = int(moments["m10"] / moments["m00"]) if moments["m00"] else 0
                center_y = int(moments["m01"] / moments["m00"]) if moments["m00"] else 0
                self.contour_info.setText(
                    f"{matched_pixels:,} matching pixels · {coverage:.2f}% of image\n"
                    f"Largest region · {area:,.0f} px² · center ({center_x}, {center_y})"
                )
            self.apply_button.setEnabled(True)
            self.log_button.setEnabled(True)
            return {"output": results.get("result", frame), "success": True}
        except Exception as exc:
            self._clear_analysis_views()
            self.last_results = None
            self.log_button.setEnabled(False)
            self.log_message.emit(f"Colour analysis failed: {exc}", "error")
            return {"output": self.last_frame, "error": str(exc), "success": False}

    def log_to_evidence(self):
        if self.last_results is None:
            self.log_message.emit("Upload an image and run the colour mask before recording it.", "warning")
            return
        hsv = f"H {self.h_min}–{self.h_max}, S {self.s_min}–{self.s_max}, V {self.v_min}–{self.v_max}"
        registry.add(EvidenceEntry(
            module="Level 3: Color Identification",
            kind="color_segmentation",
            value=f"Color mask ({hsv})",
            technique=f"OpenCV HSV range and morphology (kernel {self.morph_kernel})",
            observation=self.contour_info.text(),
            interpretation="The mask identifies pixels matching the selected colour range; it cannot identify an owner or person.",
            uncertainty="Medium",
            source_filename=self.evidence_name,
            source_timestamp="",
            frame_number=None,
        ))
        self.level_completed.emit("The Red Bag")
        self.log_message.emit("Colour-mask observation recorded in the Evidence Notebook.", "success")
