"""Level 4: review a video timeline and inspect motion line crossings."""

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.tracking import CentroidTracker, LineCrossing
from evidence.registry import EvidenceEntry, registry
from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget, MediaTimelineControls


class Lab4MotionWidget(LabWidget):
    """Video tracking workspace with playback, seeking, and an event list."""

    level_completed = Signal(str)
    play_pause_requested = Signal()
    seek_requested = Signal(int)

    def __init__(self, parent=None):
        self.min_area = 500
        self.last_frame = None
        self.last_results = None
        self.bg_subtractor = None
        self.tracker = None
        self.line_crossing = None
        self.frame_number = 0
        self.frame_count = 1
        self.video_fps = 0.0
        self.source_kind = ""
        self.previous_cross_count = 0
        self.load_modules()
        super().__init__("The Final Clue", parent)
        self.reset()

    def load_modules(self):
        try:
            from labs.solutions import level4_motion
            self.solution_module = level4_motion
        except ImportError as exc:
            self.solution_module = None
            print(f"Level 4 tracking module is unavailable: {exc}")

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
        self.analysis_status = QLabel("Open a video to inspect movement. Centroid tracking links motion regions across frames; a line crossing is an event in this clip, not identity.")
        self.analysis_status.setWordWrap(True)
        self.analysis_status.setObjectName("pageSummary")
        actions.addWidget(self.analysis_status, 1)
        actions.addWidget(QLabel("Minimum region area"))
        self.area_spin = QSpinBox()
        self.area_spin.setRange(100, 10000)
        self.area_spin.setSingleStep(100)
        self.area_spin.setValue(self.min_area)
        self.area_spin.valueChanged.connect(self.on_area_changed)
        actions.addWidget(self.area_spin)
        actions.addWidget(QLabel("Exit line"))
        self.exit_line_slider = QSlider(Qt.Horizontal)
        self.exit_line_slider.setRange(5, 95)
        self.exit_line_slider.setValue(55)
        self.exit_line_slider.setMinimumWidth(100)
        self.exit_line_slider.valueChanged.connect(self.refresh_current_frame)
        actions.addWidget(self.exit_line_slider)
        self.reset_button = QPushButton("Reset tracking")
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setEnabled(False)
        actions.addWidget(self.reset_button)
        self.save_button = QPushButton("Save frame")
        self.save_button.clicked.connect(self.save_annotated_frame)
        self.save_button.setEnabled(False)
        actions.addWidget(self.save_button)
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
        self.save_button.setEnabled(False)
        self.notebook_button.setEnabled(False)
        self.reset_button.setEnabled(is_video)
        self.set_playing(is_video)
        if is_video:
            self.set_frame_position(0, self.frame_count)

    def set_playing(self, playing):
        self.media_controls.set_playing(playing)

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
        module = self.solution_module
        height, width = self.last_frame.shape[:2]
        x = int(width * self.exit_line_slider.value() / 100)
        line_start, line_end = (x, 0), (x, height - 1)
        output = self.last_frame.copy()
        for contour in self.last_results.get("contours", []):
            cx, cy, cw, ch = cv2.boundingRect(contour)
            cv2.rectangle(output, (cx, cy), (cx + cw, cy + ch), (60, 200, 255), 2)
        output = module.draw_tracked_objects(
            output, self.last_results.get("tracked_objects", {}), self.last_results.get("trails", {})
        )
        output = module.draw_exit_line(output, line_start, line_end)
        self.last_results["result"] = output
        self.result_display.set_image(self.cv_to_pixmap(output))

    def on_area_changed(self, value):
        self.min_area = value
        self.analysis_status.setText("Minimum region area updated. The next video frame will use this setting.")

    def reset(self):
        module = self.solution_module
        if module is None:
            return
        try:
            self.bg_subtractor = module.create_background_subtractor()
            self.tracker = CentroidTracker(max_disappeared=30, max_distance=50)
            width = self.last_frame.shape[1] if self.last_frame is not None else 1280
            height = self.last_frame.shape[0] if self.last_frame is not None else 720
            x = int(width * self.exit_line_slider.value() / 100) if hasattr(self, "exit_line_slider") else int(width * 0.55)
            self.line_crossing = LineCrossing((x, 0), (x, height))
            self.previous_cross_count = 0
            self.last_results = None
            if hasattr(self, "events_table"):
                self.events_table.setRowCount(0)
            if hasattr(self, "save_button"):
                self.save_button.setEnabled(False)
                self.notebook_button.setEnabled(False)
        except Exception as exc:
            self.log_message.emit(f"Could not reset motion tracking: {exc}", "error")

    def process_frame(self, frame):
        if self.source_kind != "video":
            self.original_display.set_image(self.cv_to_pixmap(frame))
            self.result_display.setText("Tracking requires video frames.")
            return {"output": frame, "success": False, "error": "Video required"}
        module = self.solution_module
        if module is None:
            return {"output": frame, "error": "Motion tracking is unavailable", "success": False}
        try:
            self.last_frame = frame.copy()
            if self.bg_subtractor is None or self.tracker is None:
                self.reset()
            height, width = frame.shape[:2]
            x = int(width * self.exit_line_slider.value() / 100)
            line_start, line_end = (x, 0), (x, height - 1)
            self.line_crossing.update_line(line_start, line_end)
            results = module.process_motion_analysis(
                frame,
                bg_subtractor=self.bg_subtractor,
                tracker=self.tracker,
                line_crossing=self.line_crossing,
                exit_line_start=line_start,
                exit_line_end=line_end,
                min_area=self.min_area,
            )
            if results is None:
                return {"output": frame, "error": "Motion analysis failed", "success": False}
            self.last_results = results
            self.original_display.set_image(self.cv_to_pixmap(frame))
            self.result_display.set_image(self.cv_to_pixmap(results.get("result", frame)))
            self.save_button.setEnabled(True)
            self.notebook_button.setEnabled(True)
            object_count = len(results.get("tracked_objects", {}))
            cross_count = int(results.get("cross_count", 0))
            self.analysis_status.setText(
                f"{object_count} moving region(s) · {cross_count} line crossing(s). Tracking associates detections across frames; it does not identify a person."
            )
            if cross_count > self.previous_cross_count:
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
        values = [self._format_time(seconds), "Exit line crossing", f"Track ID {object_id}; {object_count} region(s) currently tracked"]
        for column, value in enumerate(values):
            self.events_table.setItem(row, column, QTableWidgetItem(value))
        self.events_table.scrollToBottom()

    def save_annotated_frame(self):
        if self.last_results is None:
            self.log_message.emit("Open evidence and run tracking before saving a frame.", "warning")
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Tracking Frame", "tracking_frame.png", "Images (*.png *.jpg *.jpeg)")
        if path:
            cv2.imwrite(path, self.last_results.get("result", self.last_frame))
            self.log_message.emit(f"Saved tracking frame to {path}", "success")

    def log_to_evidence(self):
        if self.last_results is None:
            self.log_message.emit("Open a video and inspect tracking results before recording a finding.", "warning")
            return
        object_count = len(self.last_results.get("tracked_objects", {}))
        cross_count = int(self.last_results.get("cross_count", 0))
        registry.add(EvidenceEntry(
            module="Level 4: Timeline and Tracking",
            kind="video_tracking",
            value=f"Tracked {object_count} region(s); {cross_count} line crossing(s)",
            technique=f"OpenCV MOG2 and centroid tracking (minimum region {self.min_area} px²)",
            observation=self.analysis_status.text(),
            interpretation="A line crossing describes movement in this video and does not establish a person's identity or intent.",
            uncertainty="Medium",
            source_filename=self.media_controls.filename_label.text(),
            source_timestamp=self._format_time(self.frame_number / self.video_fps if self.video_fps > 0 else 0),
            frame_number=self.frame_number,
        ))
        self.level_completed.emit("The Final Clue")
        self.log_message.emit("Tracking observation recorded in the Evidence Notebook.", "success")

    def clear_media(self):
        self.source_kind = ""
        self.last_frame = None
        self.last_results = None
        self.frame_number = 0
        self.frame_count = 1
        self.video_fps = 0.0
        self.media_controls.reset()
        self.original_display.set_image(None)
        self.result_display.set_image(None)
        self.analysis_status.setText("Upload video evidence to inspect movement and crossings.")
        self.events_table.setRowCount(0)
        self.reset_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.notebook_button.setEnabled(False)
        self.reset()
