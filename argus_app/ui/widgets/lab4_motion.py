"""Level 4: review a video timeline and inspect motion line crossings."""

import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evidence.registry import EvidenceEntry, registry
from evidence.artifacts import save_image
from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget, MediaTimelineControls
from level_codes.level4 import process_tracking_frame


class Lab4MotionWidget(LabWidget):
    """Video tracking workspace with playback, seeking, and an event list."""

    level_completed = Signal(str)
    play_pause_requested = Signal()
    seek_requested = Signal(int)

    def __init__(self, parent=None):
        self.min_area = 500
        self.max_step_fraction = 0.18
        configured_line = os.environ.get("ARGUS_LEVEL4_EXIT_LINE_X_FRACTION", "").strip()
        try:
            self.exit_line_fraction = float(configured_line) if configured_line else None
        except ValueError:
            self.exit_line_fraction = None
        if self.exit_line_fraction is not None and not 0 <= self.exit_line_fraction <= 1:
            self.exit_line_fraction = None
        self.last_frame = None
        self.last_results = None
        self.frame_number = 0
        self.last_processed_frame_number = None
        self.frame_count = 1
        self.video_fps = 0.0
        self.source_kind = ""
        self.source_path = ""
        self.previous_cross_count = 0
        self.crossing_recorded = False
        self.previous_red_centroid = None
        self.red_trail = []
        self.red_segments = []
        self.track_note = ""
        super().__init__("Level 4: Track the Red Bag", parent)
        self.reset()

    def setup_ui(self):
        super().setup_ui()
        root = QVBoxLayout(self.content_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(9)

        self.media_controls = MediaTimelineControls()
        self.media_controls.play_pause_requested.connect(self.play_pause_requested.emit)
        self.media_controls.seek_requested.connect(self.seek_requested.emit)
        root.addWidget(self.media_controls)

        tracking_controls = QGridLayout()
        self.area_slider = QSlider(Qt.Horizontal)
        self.area_slider.setRange(0, 50)
        self.area_slider.setValue(5)
        self.area_slider.setToolTip("Ignore red contours smaller than this area.")
        self.jump_slider = QSlider(Qt.Horizontal)
        self.jump_slider.setRange(5, 40)
        self.jump_slider.setValue(18)
        self.jump_slider.setToolTip("Maximum allowed centroid movement between consecutive frames, as a fraction of frame diagonal.")
        self.area_label = QLabel()
        self.jump_label = QLabel()
        self.entrance_line_checkbox = QCheckBox("Evaluate doorway line crossings")
        self.entrance_line_checkbox.setChecked(self.exit_line_fraction is not None)
        self.line_slider = QSlider(Qt.Horizontal)
        self.line_slider.setRange(0, 1000)
        self.line_slider.setValue(round(self.exit_line_fraction * 1000) if self.exit_line_fraction is not None else 600)
        self.line_slider.setEnabled(self.exit_line_fraction is not None)
        self.line_slider.setToolTip("Horizontal line position as a fraction of frame width. Set this from the actual Lab 4B doorway.")
        self.line_label = QLabel()
        tracking_controls.addWidget(self.area_label, 0, 0)
        tracking_controls.addWidget(self.area_slider, 0, 1)
        tracking_controls.addWidget(self.jump_label, 1, 0)
        tracking_controls.addWidget(self.jump_slider, 1, 1)
        tracking_controls.addWidget(self.entrance_line_checkbox, 2, 0)
        tracking_controls.addWidget(self.line_slider, 2, 1)
        tracking_controls.addWidget(self.line_label, 3, 0, 1, 2)
        root.addLayout(tracking_controls)
        self.area_slider.valueChanged.connect(self._tracking_parameters_changed)
        self.jump_slider.valueChanged.connect(self._tracking_parameters_changed)
        self.line_slider.valueChanged.connect(self._tracking_parameters_changed)
        self.entrance_line_checkbox.toggled.connect(self._tracking_parameters_changed)
        self._tracking_parameters_changed()

        views = QGridLayout()
        views.setContentsMargins(0, 0, 0, 0)
        views.setHorizontalSpacing(10)
        views.setVerticalSpacing(6)
        views.addWidget(self._viewer("VIDEO FRAME", "Upload a video to begin.", "original_display"), 0, 0)
        views.addWidget(self._viewer("MOTION TRACKING", "Tracked result appears here.", "result_display"), 0, 1)
        views.setColumnStretch(0, 1)
        views.setColumnStretch(1, 1)
        views.setRowStretch(0, 1)

        workspace = QFrame()
        workspace.setObjectName("evidencePanel")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(8, 8, 8, 8)
        workspace_layout.setSpacing(6)
        workspace_layout.addLayout(views, 1)
        root.addWidget(workspace, 1)

        actions = QHBoxLayout()
        self.analysis_status = QLabel("Adjust the tracking sliders below. Enable the doorway line only after locating Lab 4B in the supplied video.")
        self.analysis_status.setWordWrap(True)
        self.analysis_status.setObjectName("pageSummary")
        actions.addWidget(self.analysis_status, 1)
        self.notebook_button = QPushButton("Record finding")
        self.notebook_button.setObjectName("primaryButton")
        self.notebook_button.clicked.connect(self.log_to_evidence)
        self.notebook_button.setEnabled(False)
        actions.addWidget(self.notebook_button)
        root.addLayout(actions)

        self.events_table = QTableWidget(0, 3)
        self.events_table.setHorizontalHeaderLabels(["Video time", "Event", "Details"])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.setMaximumHeight(126)
        self.events_table.setAlternatingRowColors(True)
        root.addWidget(self.events_table)

    def _viewer(self, title, placeholder, attribute):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        title_label = QLabel(title)
        title_label.setObjectName("eyebrowLabel")
        layout.addWidget(title_label)
        display = ImageDisplayLabel(placeholder)
        if attribute == "original_display":
            display.set_upload_enabled(True, "video")
        setattr(self, attribute, display)
        layout.addWidget(display, 1)
        return panel

    def set_media_info(self, filename, source_type, frame_count=1, fps=0.0):
        self.source_kind = getattr(source_type, "value", source_type)
        if self.source_kind != "video":
            self.clear_media()
            self.log_message.emit("Level 4 accepts video files only. Movement and line crossing need multiple frames.", "warning")
            return
        self.frame_count = max(1, int(frame_count))
        self.video_fps = float(fps or 0)
        self.frame_number = 0
        self.media_controls.set_media_info(filename, self.source_kind, self.frame_count, self.video_fps)
        is_video = self.source_kind == "video"
        self.original_display.set_image(None)
        self.result_display.set_image(None)
        self.last_frame = None
        self.last_results = None
        self.notebook_button.setEnabled(False)
        self.reset()
        self.set_playing(is_video)
        if is_video:
            self.set_frame_position(0, self.frame_count)

    def set_playing(self, playing):
        self.media_controls.set_playing(playing)

    def _tracking_parameters_changed(self, *_args):
        self.min_area = self.area_slider.value() * 100
        self.max_step_fraction = self.jump_slider.value() / 100.0
        use_line = self.entrance_line_checkbox.isChecked()
        self.exit_line_fraction = self.line_slider.value() / 1000.0 if use_line else None
        self.area_label.setText(f"Minimum red contour area: {self.min_area} px²")
        self.jump_label.setText(f"Maximum centroid jump: {self.max_step_fraction:.0%} of frame diagonal")
        self.line_label.setText(
            f"Doorway line at {self.exit_line_fraction:.1%} of frame width" if use_line
            else "Doorway line off · inspect the supplied footage before enabling crossings"
        )
        self.line_slider.setEnabled(use_line)
        if self.last_frame is not None and self.source_kind == "video":
            # A parameter change starts a fresh segment so it cannot create a
            # crossing by comparing this frame against a result from old settings.
            if self.red_trail:
                self.red_segments.append(self.red_trail)
                self.red_trail = []
            self.previous_red_centroid = None
            self.last_processed_frame_number = self.frame_number - 1
            self.process_frame(self.last_frame.copy())

    def set_source_path(self, path):
        self.source_path = str(path or "")

    def set_frame_position(self, frame_number, frame_count):
        self.frame_number = max(0, int(frame_number))
        self.frame_count = max(1, int(frame_count))
        self.media_controls.set_frame_position(self.frame_number, self.frame_count)

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def refresh_current_frame(self, *_args):
        if self.last_frame is None or self.last_results is None:
            return
        # The teaching pipeline owns all tracking annotations; the UI only shows them.
        self.result_display.set_image(self.cv_to_pixmap(self.last_results["result"]))

    def reset(self):
        try:
            self.previous_cross_count = 0
            self.crossing_recorded = False
            self.previous_red_centroid = None
            self.red_trail = []
            self.red_segments = []
            self.track_note = ""
            self.last_results = None
            self.last_processed_frame_number = None
            if hasattr(self, "events_table"):
                self.events_table.setRowCount(0)
            if hasattr(self, "notebook_button"):
                self.notebook_button.setEnabled(False)
        except Exception as exc:
            self.log_message.emit(f"Could not reset motion tracking: {exc}", "error")

    def process_frame(self, frame):
        if self.source_kind != "video":
            self.original_display.set_image(self.cv_to_pixmap(frame))
            self.result_display.setText("Tracking requires video frames.")
            return {"output": frame, "success": False, "error": "Video required"}
        try:
            self.track_note = ""
            if self.last_processed_frame_number is not None and self.frame_number != self.last_processed_frame_number + 1:
                self.previous_red_centroid = None
                if self.red_trail:
                    self.red_segments.append(self.red_trail)
                    self.red_trail = []
                self.track_note = "Frame gap or seek; centroid continuity was reset."
            self.last_processed_frame_number = self.frame_number
            self.last_frame = frame.copy()
            height, width = frame.shape[:2]
            x = round(width * self.exit_line_fraction) if self.exit_line_fraction is not None else None
            pipeline = process_tracking_frame(
                frame,
                previous_centroid=self.previous_red_centroid,
                trail=self.red_trail,
                segments=self.red_segments,
                line_x=x,
                min_area=self.min_area,
                max_step_fraction=self.max_step_fraction,
            )
            self.previous_red_centroid = pipeline["previous_centroid"]
            self.red_trail = pipeline["trail"]
            self.red_segments = pipeline["segments"]
            continuity_note = self.track_note
            self.track_note = pipeline["note"]
            if continuity_note:
                self.track_note = continuity_note + " " + self.track_note
            centroid = pipeline["centroid"]
            crossed = pipeline["crossed"]
            candidates = pipeline["candidates"]
            newly_crossed = crossed and not self.crossing_recorded
            cross_count = self.previous_cross_count + int(newly_crossed)
            segments = self.red_segments + ([self.red_trail] if self.red_trail else [])
            results = {
                "result": pipeline["result"],
                "mask": pipeline["mask"],
                "tracked_objects": ({1: {"centroid": centroid}} if centroid else {}),
                "cross_count": cross_count,
                "crossed_ids": ([1] if newly_crossed else []),
                "candidates": pipeline["candidate_count"],
                "centroid": centroid,
                "red_segments": segments,
            }
            self.last_results = results
            self.original_display.set_image(self.cv_to_pixmap(frame))
            self.result_display.set_image(self.cv_to_pixmap(results.get("result", frame)))
            self.notebook_button.setEnabled(True)
            object_count = len(results.get("tracked_objects", {}))
            cross_count = int(results.get("cross_count", 0))
            if x is None:
                status = f"{pipeline["candidate_count"]} red candidate(s) · entrance line not configured; crossings are not evaluated."
            else:
                status = f"{pipeline["candidate_count"]} red candidate(s) · {cross_count} supported exit-line crossing(s)."
            if centroid is None:
                status += " No centroid is tracked in this frame."
            if self.track_note:
                status += " " + self.track_note
            if x is None:
                status += " Inspect the real video and set ARGUS_LEVEL4_EXIT_LINE_X_FRACTION to the Lab 4B doorway position."
            self.analysis_status.setText(status + " Red colour does not identify its carrier.")
            if newly_crossed:
                self.crossing_recorded = True
                for object_id in results.get("crossed_ids", []):
                    self._append_event(object_id, object_count)
            self.previous_cross_count = cross_count
            return {"output": results.get("result", frame), "success": True, "cross_count": cross_count}
        except Exception as exc:
            self.log_message.emit(f"Motion analysis failed: {exc}", "error")
            return {"output": frame, "error": str(exc), "success": False}

    def _append_event(self, object_id, object_count):
        row = self.events_table.rowCount()
        self.events_table.insertRow(row)
        seconds = self.frame_number / self.video_fps if self.video_fps > 0 else 0
        values = [self._format_time(seconds), "Red-candidate exit crossing", "A continuity-selected red candidate crossed the configured line; identity remains unverified."]
        for column, value in enumerate(values):
            self.events_table.setItem(row, column, QTableWidgetItem(value))
        self.events_table.scrollToBottom()

    def log_to_evidence(self):
        if self.last_results is None:
            self.log_message.emit("Open a video and inspect tracking results before recording a finding.", "warning")
            return
        cross_count = int(self.last_results.get("cross_count", 0))
        try:
            output_path = save_image(4, f"frame_{self.frame_number:06d}_tracked.png", self.last_results["result"])
        except (OSError, ValueError) as exc:
            self.log_message.emit(f"Could not save the tracking output: {exc}", "error")
            return
        centroid = self.last_results.get("centroid")
        registry.add(EvidenceEntry(
            module="Level 4: Track the Red Bag",
            kind="video_tracking",
            value={"candidate_count": self.last_results.get("candidates", 0), "centroid": centroid, "red_segments": self.last_results.get("red_segments", []), "line_crossings": cross_count, "entrance_line_x_fraction": self.exit_line_fraction, "minimum_contour_area": self.min_area, "max_centroid_jump_fraction": self.max_step_fraction},
            technique="OpenCV wrapped red HSV mask, morphology, area-filtered contours, nearest-centroid continuity" + (f" (configured line at {self.exit_line_fraction:.0%} frame width)" if self.exit_line_fraction is not None else " (entrance line not configured; crossings not evaluated)"),
            observation=self.analysis_status.text(),
            interpretation="The selected contour is a red candidate followed by centroid proximity between adjacent readable frames. A supported line crossing describes clip-relative movement only; it does not identify the bag or its carrier.",
            uncertainty="High" if self.last_results.get("candidates", 0) != 1 or centroid is None else "Medium",
            source_filename=self.media_controls.filename_label.text(),
            source_path=self.source_path,
            output_path=output_path,
            source_timestamp=self._format_time(self.frame_number / self.video_fps if self.video_fps > 0 else 0),
            frame_number=self.frame_number,
            evidence_id="E-04",
            unresolved_question="Does the available clip and supporting documentary evidence establish that the same physical bag exited?",
            limitation="HSV candidates can be distractors; a skipped or occluded detection breaks centroid continuity, and clip time is not wall-clock time without a verified offset.",
        ))
        self.level_completed.emit("The Final Clue")
        self.log_message.emit("Tracking observation recorded in the Evidence Notebook.", "success")

    def clear_media(self):
        self.source_kind = ""
        self.source_path = ""
        self.last_frame = None
        self.last_results = None
        self.frame_number = 0
        self.frame_count = 1
        self.video_fps = 0.0
        self.media_controls.reset()
        self.original_display.set_image(None)
        self.result_display.set_image(None)
        self.analysis_status.setText("Configure the entrance line from the actual video before evaluating crossings. Red colour does not identify a person.")
        self.events_table.setRowCount(0)
        self.notebook_button.setEnabled(False)
        self.reset()
