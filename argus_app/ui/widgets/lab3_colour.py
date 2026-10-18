"""Level 3: inspect actual HSV masks and red-region candidates."""

import cv2
import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout

from evidence.artifacts import save_image
from evidence.registry import EvidenceEntry, registry
from ui.style import COLORS, FONTS
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget


class Lab3ColourWidget(LabWidget):
    """Run the fixed red segmentation pipeline and let students inspect outputs."""

    level_completed = Signal(str)
    RED_RANGE = (170, 10, 90, 255, 35, 255)

    def __init__(self, parent=None):
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
            self.solution_module = None
            self.log_message.emit(f"Colour analysis is unavailable: {exc}", "error")

    def setup_ui(self):
        super().setup_ui()
        root = QVBoxLayout(self.content_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        views = QSplitter(Qt.Horizontal)
        views.setChildrenCollapsible(False)
        self.original_display = ImageDisplayLabel("Open the supplied corridor image to begin.")
        self.original_display.set_upload_enabled(True, "image")
        self.result_display = ImageDisplayLabel("Run segmentation to inspect the mask or candidates.")
        views.addWidget(self._panel("SOURCE IMAGE", self.original_display))
        result_panel = self._panel("PROCESSING OUTPUT", self.result_display)
        self.result_mode = QComboBox()
        self.result_mode.addItems(["Candidate regions", "Cleaned red mask"])
        self.result_mode.currentTextChanged.connect(self._show_result_mode)
        result_panel.layout().insertWidget(1, self.result_mode)
        views.addWidget(result_panel)
        views.setStretchFactor(0, 1)
        views.setStretchFactor(1, 1)
        views.setSizes([1, 1])
        root.addWidget(views, 1)

        controls = QHBoxLayout()
        self.contour_info = QLabel("HSV → wrapped red ranges → cleaned mask → area-filtered candidates.")
        self.contour_info.setWordWrap(True)
        self.contour_info.setObjectName("pageSummary")
        controls.addWidget(self.contour_info, 1)
        self.apply_button = QPushButton("Segment red regions")
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.run_colour_mask)
        controls.addWidget(self.apply_button)
        self.log_button = QPushButton("Record result")
        self.log_button.setEnabled(False)
        self.log_button.clicked.connect(self.log_to_evidence)
        controls.addWidget(self.log_button)
        root.addLayout(controls)
        self.result_mode.setEnabled(False)

    @staticmethod
    def _panel(title, display):
        panel = QFrame()
        panel.setObjectName("evidencePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        label = QLabel(title)
        label.setStyleSheet(f"color: {COLORS['text_muted']}; font-family: {FONTS['family_mono']}; font-size: {FONTS['size_xs']};")
        layout.addWidget(label)
        layout.addWidget(display, 1)
        return panel

    def set_evidence_name(self, filename):
        self.evidence_name = str(filename or "")

    @property
    def has_evidence(self):
        return self.last_frame is not None

    def reset(self):
        self.last_results = None
        self.last_frame = None
        self.evidence_name = ""
        self.original_display.set_image(None)
        self.result_display.set_image(None)
        self.contour_info.setText("HSV → wrapped red ranges → cleaned mask → area-filtered candidates.")
        self.apply_button.setEnabled(False)
        self.log_button.setEnabled(False)
        self.result_mode.setEnabled(False)

    def load_evidence_frame(self, frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            self.reset()
            self.log_message.emit("Choose a readable BGR colour image for Level 3.", "error")
            return False
        self.last_frame = np.ascontiguousarray(frame).copy()
        self.last_results = None
        self.original_display.set_image(self.cv_to_pixmap(self.last_frame))
        self.result_display.set_image(None)
        height, width = self.last_frame.shape[:2]
        self.contour_info.setText(f"{self.evidence_name or 'Image'} · {width} × {height} · select Segment red regions.")
        self.apply_button.setEnabled(True)
        self.log_button.setEnabled(False)
        self.result_mode.setEnabled(False)
        return True

    def run_colour_mask(self):
        if self.last_frame is None:
            self.log_message.emit("Open the Level 3 evidence image before segmenting.", "warning")
            return {"output": None, "error": "No evidence selected", "success": False}
        if self.solution_module is None:
            self.contour_info.setText("OpenCV segmentation code is unavailable.")
            self.log_message.emit("Colour analysis is unavailable because its OpenCV module did not load.", "error")
            return {"output": None, "error": "Segmentation unavailable", "success": False}
        try:
            results = self.solution_module.process_color_segmentation(
                self.last_frame.copy(), *self.RED_RANGE, morph_kernel=5
            )
            if not results or results.get("mask_cleaned") is None:
                raise RuntimeError("The segmentation pipeline returned no mask.")
            self.last_results = results
            results["isolated_candidates"] = results["isolated"].copy()
            candidates = results.get("candidate_contours", [])
            areas = sorted((cv2.contourArea(item) for item in candidates), reverse=True)
            matched = int(np.count_nonzero(results["mask_cleaned"]))
            coverage = matched / max(1, results["mask_cleaned"].size) * 100
            area_summary = ", ".join(f"{area:,.0f}" for area in areas[:6]) or "none"
            self.contour_info.setText(
                f"{len(candidates)} red candidate(s) · {matched:,} cleaned pixels ({coverage:.2f}%) · candidate areas: {area_summary} px²."
            )
            self._show_result_mode(self.result_mode.currentText())
            self.result_mode.setEnabled(True)
            self.apply_button.setEnabled(True)
            self.log_button.setEnabled(True)
            return {"output": results["result"], "success": True}
        except Exception as exc:
            self.last_results = None
            self.result_display.setText("Segmentation failed. Check the selected image.")
            self.contour_info.setText(f"Could not segment the image: {exc}")
            self.log_button.setEnabled(False)
            self.log_message.emit(f"Colour analysis failed: {exc}", "error")
            return {"output": self.last_frame, "error": str(exc), "success": False}

    def _show_result_mode(self, mode):
        if not self.last_results:
            return
        image = self.last_results.get("mask_cleaned") if mode == "Cleaned red mask" else self.last_results.get("result")
        if image is not None:
            self.result_display.set_image(self.cv_to_pixmap(image))

    def log_to_evidence(self):
        if self.last_results is None:
            return
        try:
            candidate_output = save_image(3, "candidate_regions.png", self.last_results["result"])
            mask_output = save_image(3, "cleaned_red_mask.png", self.last_results["mask_cleaned"])
        except (OSError, ValueError) as exc:
            self.log_message.emit(f"Could not save segmentation outputs: {exc}", "error")
            return
        candidate_data = []
        for contour in self.last_results.get("candidate_contours", []):
            x, y, width, height = cv2.boundingRect(contour)
            candidate_data.append({"area": float(cv2.contourArea(contour)), "box": (x, y, width, height)})
        registry.add(EvidenceEntry(
            module="Level 3: Red Bag", kind="color_segmentation",
            value={"candidate_regions": candidate_data, "outputs": [candidate_output, mask_output]},
            technique="BGR-to-HSV, wrapped red hue ranges, morphological cleanup, area-filtered contours, and candidate overlay",
            observation=self.contour_info.text(),
            interpretation="The mask marks red pixels and the boxes mark area-filtered candidates. Compare them with the source image and any supplied inventory record; the algorithm does not recognize bag identity or its pale patch.",
            uncertainty="Medium", source_filename=Path(self.evidence_name).name,
            source_path=self.evidence_name, output_path=candidate_output, evidence_id="E-03",
            unresolved_question="Which candidate, if any, matches the bag based on separate visual or documentary evidence?",
            limitation="Other red objects may be candidates. HSV segmentation does not establish that a region is the backpack, who owns it, or who carries it.",
        ))
        self.level_completed.emit("The Red Bag")
        self.log_message.emit(f"Red-region result recorded · {candidate_output} and {mask_output}", "success")
