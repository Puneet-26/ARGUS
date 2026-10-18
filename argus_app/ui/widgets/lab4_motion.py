"""Level 4: review a video timeline and inspect motion line crossings."""

import cv2
import math
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evidence.registry import EvidenceEntry, registry
from evidence.artifacts import save_image
from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget, MediaTimelineControls


class Lab4MotionWidget(LabWidget):
    """Video tracking workspace with playback, seeking, and an event list."""

    level_completed = Signal(str)
    play_pause_requested = Signal()
    seek_requested = Signal(int)

    def __init__(self, parent=None):
        self.min_area = 500
        self.exit_line_fraction = 0.60
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
        super().__init__("The Final Clue", parent)
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
        self.analysis_status = QLabel("The line marks the configured exit. Red colour and a crossing do not identify a person.")
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
        height, width = self.last_frame.shape[:2]
        x = int(width * self.exit_line_fraction)
        line_start, line_end = (x, 0), (x, height - 1)
        output = self.last_frame.copy()
        cv2.line(output, line_start, line_end, (255, 180, 0), 2)
        for segment in self.red_segments + ([self.red_trail] if self.red_trail else []):
            for start, end in zip(segment[-120:], segment[-119:]):
                cv2.line(output, start, end, (0, 255, 0), 2)
        tracked = self.last_results.get("tracked_objects", {})
        if tracked and tracked.get(1, {}).get("centroid"):
            cv2.circle(output, tracked[1]["centroid"], 6, (0, 255, 0), -1)
        self.last_results["result"] = output
        self.result_display.set_image(self.cv_to_pixmap(output))

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
            x = int(width * self.exit_line_fraction)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_low = cv2.inRange(hsv, (0, 70, 40), (10, 255, 255))
            red_high = cv2.inRange(hsv, (170, 70, 40), (179, 255, 255))
            mask = cv2.bitwise_or(red_low, red_high)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_area = max(1, width * height)
            contours = [contour for contour in contours if self.min_area <= cv2.contourArea(contour) <= frame_area * 0.75]
            result = frame.copy()
            cv2.line(result, (x, 0), (x, height - 1), (255, 180, 0), 2)
            crossed = False
            centroid = None
            if contours:
                candidates = []
                for candidate in contours:
                    moments = cv2.moments(candidate)
                    if moments["m00"]:
                        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                        candidates.append((candidate, center))
                selected = None
                if self.previous_red_centroid is not None and candidates:
                    max_step = max(40.0, math.hypot(width, height) * 0.18)
                    selected = min(candidates, key=lambda item: math.dist(self.previous_red_centroid, item[1]))
                    if math.dist(self.previous_red_centroid, selected[1]) > max_step:
                        selected = None
                        self.track_note = "Candidate jump exceeded the continuity limit; tracking is ambiguous."
                elif candidates:
                    selected = max(candidates, key=lambda item: cv2.contourArea(item[0]))
                    self.track_note = "Largest red candidate seeded the track; its physical identity is unverified."
                if selected is None:
                    self.previous_red_centroid = None
                    if self.red_trail:
                        self.red_segments.append(self.red_trail)
                        self.red_trail = []
                else:
                    contour, centroid = selected
                    for candidate, center in candidates:
                        bx, by, bw, bh = cv2.boundingRect(candidate)
                        cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (120, 120, 120), 1)
                    if len(candidates) > 1 and self.previous_red_centroid is None:
                        self.track_note = f"{len(candidates)} candidates; largest area was selected. Identity is ambiguous."
                if selected is not None:
                    contour = selected[0]
                    centroid = selected[1]
                    if self.previous_red_centroid is not None:
                        old_x = self.previous_red_centroid[0]
                        crossed = (old_x < x <= centroid[0]) or (old_x > x >= centroid[0])
                    self.previous_red_centroid = centroid
                    self.red_trail.append(centroid)
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (0, 220, 255), 2)
                    cv2.circle(result, centroid, 6, (0, 255, 0), -1)
            else:
                self.previous_red_centroid = None
                self.track_note = "No valid red candidate in this frame; track continuity is interrupted."
                if self.red_trail:
                    self.red_segments.append(self.red_trail)
                    self.red_trail = []
            segments = self.red_segments + ([self.red_trail] if self.red_trail else [])
            for segment in segments:
                for start, end in zip(segment[-120:], segment[-119:]):
                    cv2.line(result, start, end, (0, 255, 0), 2)
            newly_crossed = crossed and not self.crossing_recorded
            cross_count = self.previous_cross_count + int(newly_crossed)
            results = {"result": result, "tracked_objects": ({1: {"centroid": centroid}} if centroid else {}), "cross_count": cross_count, "crossed_ids": ([1] if newly_crossed else []), "candidates": len(contours), "centroid": centroid, "red_segments": segments}
            self.last_results = results
            self.original_display.set_image(self.cv_to_pixmap(frame))
            self.result_display.set_image(self.cv_to_pixmap(results.get("result", frame)))
            self.notebook_button.setEnabled(True)
            object_count = len(results.get("tracked_objects", {}))
            cross_count = int(results.get("cross_count", 0))
            status = f"{len(contours)} red candidate(s) · {cross_count} supported exit-line crossing(s)."
            if centroid is None:
                status += " No centroid is tracked in this frame."
            if self.track_note:
                status += " " + self.track_note
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
            module="Level 4: Red Bag Tracking",
            kind="video_tracking",
            value={"candidate_count": self.last_results.get("candidates", 0), "centroid": centroid, "red_segments": self.last_results.get("red_segments", []), "line_crossings": cross_count},
            technique=f"OpenCV wrapped red HSV mask, morphology, area-filtered contours, nearest-centroid continuity (fixed exit line at {self.exit_line_fraction:.0%} frame width)",
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
        self.analysis_status.setText("The line marks the configured exit. Red colour and a crossing do not identify a person.")
        self.events_table.setRowCount(0)
        self.notebook_button.setEnabled(False)
        self.reset()
