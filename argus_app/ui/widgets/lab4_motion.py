"""Level 4: review a video timeline and inspect motion line crossings."""

import os
from pathlib import Path
from PySide6.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from level_codes.level3 import COLOR_PRESETS


class TrackingJobSignals(QObject):
    finished = Signal(object, int, int, int, object)


class TrackingJob(QRunnable):
    """Run the authoritative Level 4 pipeline away from the interface thread."""

    def __init__(self, frame, frame_number, source_generation, epoch, revision, parameters):
        super().__init__()
        self.frame = frame.copy()
        self.frame_number = int(frame_number)
        self.source_generation = int(source_generation)
        self.epoch = int(epoch)
        self.revision = int(revision)
        self.parameters = parameters
        self.signals = TrackingJobSignals()

    def run(self):
        try:
            pipeline = process_tracking_frame(self.frame, **self.parameters)
            error = ""
        except Exception as exc:
            pipeline = None
            error = str(exc)
        self.signals.finished.emit(
            self.frame, self.frame_number, self.source_generation, self.epoch, (self.revision, pipeline, error)
        )


class Lab4MotionWidget(LabWidget):
    """Video tracking workspace with playback, seeking, and an event list."""

    level_completed = Signal(str)
    play_pause_requested = Signal()
    seek_requested = Signal(int)
    restart_requested = Signal()

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
        self.track_lost = False
        self.crossing_direction = "left_to_right"
        self.target_color = "Red"
        preset = COLOR_PRESETS["Red"]
        self.target_h_min = preset["h_min"]
        self.target_h_max = preset["h_max"]
        self.target_s_min = preset["s_min"]
        self.target_s_max = 255
        self.target_v_min = preset["v_min"]
        self.target_v_max = 255
        self._tracking_pool = None
        self._tracking_busy = False
        self._tracking_pending = None
        self._tracking_source_generation = None
        self._tracking_epoch = 0
        self._tracking_revision = 0
        super().__init__("Level 4: Track the Red Bag", parent)
        self._tracking_pool = QThreadPool(self)
        self.reset()

    def setup_ui(self):
        super().setup_ui()
        root = QVBoxLayout(self.content_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(9)

        self.media_controls = MediaTimelineControls()
        self.media_controls.play_pause_requested.connect(self.play_pause_requested.emit)
        self.media_controls.seek_requested.connect(self.seek_requested.emit)
        self.media_controls.restart_requested.connect(self.restart_requested.emit)
        root.addWidget(self.media_controls)

        tracking_controls = QGridLayout()
        self.target_color_combo = QComboBox()
        self.target_color_combo.addItems(["Red", "Blue", "Green", "Yellow", "Orange"])
        self.target_color_combo.currentTextChanged.connect(self._target_color_changed)
        self.area_slider = QSlider(Qt.Horizontal)
        self.area_slider.setRange(0, 50)
        self.area_slider.setValue(5)
        self.area_slider.setToolTip("Ignore target contours smaller than this area.")
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
        self.direction_combo = QComboBox()
        self.direction_combo.addItem("Left → right crossing", "left_to_right")
        self.direction_combo.addItem("Right → left crossing", "right_to_left")
        self.direction_combo.currentIndexChanged.connect(self._tracking_parameters_changed)
        tracking_controls.addWidget(QLabel("Target colour:"), 0, 0)
        tracking_controls.addWidget(self.target_color_combo, 0, 1)
        tracking_controls.addWidget(self.area_label, 1, 0)
        tracking_controls.addWidget(self.area_slider, 1, 1)
        tracking_controls.addWidget(self.jump_label, 2, 0)
        tracking_controls.addWidget(self.jump_slider, 2, 1)
        tracking_controls.addWidget(self.entrance_line_checkbox, 3, 0)
        tracking_controls.addWidget(self.line_slider, 3, 1)
        tracking_controls.addWidget(self.line_label, 4, 0, 1, 2)
        tracking_controls.addWidget(QLabel("Crossing direction:"), 5, 0)
        tracking_controls.addWidget(self.direction_combo, 5, 1)
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

    def _target_color_changed(self, color_name):
        self.target_color = color_name
        preset = COLOR_PRESETS.get(color_name, COLOR_PRESETS["Red"])
        self.target_h_min = preset["h_min"]
        self.target_h_max = preset["h_max"]
        self.target_s_min = preset["s_min"]
        self.target_s_max = 255
        self.target_v_min = preset["v_min"]
        self.target_v_max = 255
        self._tracking_parameters_changed()

    def _tracking_parameters_changed(self, *_args):
        self._tracking_revision += 1
        self.min_area = self.area_slider.value() * 100
        self.max_step_fraction = self.jump_slider.value() / 100.0
        use_line = self.entrance_line_checkbox.isChecked()
        self.exit_line_fraction = self.line_slider.value() / 1000.0 if use_line else None
        self.crossing_direction = (
            self.direction_combo.currentData()
            if hasattr(self, "direction_combo")
            else "left_to_right"
        )
        color = getattr(self, "target_color", "Red")
        self.area_label.setText(f"Minimum {color.lower()} contour area: {self.min_area} px²")
        self.jump_label.setText(f"Maximum centroid jump: {self.max_step_fraction:.0%} of frame diagonal")
        self.line_label.setText(
            f"Doorway line at {self.exit_line_fraction:.1%} of frame width" if use_line
            else "Doorway line off · inspect video before enabling crossings"
        )
        self.line_slider.setEnabled(use_line)
        if self.last_frame is not None and self.source_kind == "video":
            if self.red_trail:
                self.red_segments.append(self.red_trail)
                self.red_trail = []
            self.previous_red_centroid = None
            self.track_lost = False
            self.last_processed_frame_number = self.frame_number - 1
            if self._tracking_source_generation is not None:
                self.submit_video_frame(
                    self.last_frame.copy(),
                    self.frame_number,
                    self.frame_count,
                    self._tracking_source_generation,
                )

    def set_source_path(self, path):
        self.source_path = str(path or "")

    def set_frame_position(self, frame_number, frame_count):
        self.frame_number = max(0, int(frame_number))
        self.frame_count = max(1, int(frame_count))
        self.media_controls.set_frame_position(self.frame_number, self.frame_count)

    def submit_video_frame(self, frame, frame_number, frame_count, source_generation):
        """Queue only the newest frame while the tracking worker is occupied."""
        if frame is None or self.source_kind != "video":
            return
        generation = int(source_generation)
        if self._tracking_source_generation != generation:
            self._tracking_source_generation = generation
            self._tracking_epoch += 1
            self.previous_red_centroid = None
            self.track_lost = False
            self.red_trail = []
            self.last_processed_frame_number = None
        request = (
            frame.copy(),
            int(frame_number),
            int(frame_count),
            generation,
            self._tracking_epoch,
        )
        if self._tracking_busy:
            self._tracking_pending = request
            return
        self._launch_tracking_job(request)

    def _launch_tracking_job(self, request):
        frame, frame_number, frame_count, source_generation, epoch = request
        if self.last_processed_frame_number is not None and frame_number != self.last_processed_frame_number + 1:
            had_track = self.previous_red_centroid is not None or self.track_lost
            self.previous_red_centroid = None
            self.track_lost = had_track
            if self.red_trail:
                self.red_segments.append(self.red_trail)
                self.red_trail = []
            self.track_note = (
                "Frame gap; the previously followed candidate is marked lost. Seek to begin a new segment."
                if had_track else "Frame gap or seek; centroid continuity was reset."
            )
        else:
            self.track_note = ""
        self.frame_count = max(1, frame_count)
        _, width = frame.shape[:2]
        line_x = round(width * self.exit_line_fraction) if self.exit_line_fraction is not None else None
        job = TrackingJob(
            frame,
            frame_number,
            source_generation,
            epoch,
            self._tracking_revision,
            {
                "previous_centroid": self.previous_red_centroid,
                "trail": list(self.red_trail),
                "segments": [list(segment) for segment in self.red_segments],
                "line_x": line_x,
                "min_area": self.min_area,
                "max_step_fraction": self.max_step_fraction,
                "track_lost": self.track_lost,
                "crossing_direction": self.crossing_direction,
                "h_min": self.target_h_min,
                "h_max": self.target_h_max,
                "s_min": self.target_s_min,
                "s_max": self.target_s_max,
                "v_min": self.target_v_min,
                "v_max": self.target_v_max,
                "target_name": self.target_color,
            },
        )
        job.signals.finished.connect(self._on_tracking_job_finished)
        self._tracking_busy = True
        self._tracking_pool.start(job)

    @Slot(object, int, int, int, object)
    def _on_tracking_job_finished(self, frame, frame_number, source_generation, epoch, payload):
        revision, pipeline, error = payload
        self._tracking_busy = False
        is_current = (
            source_generation == self._tracking_source_generation
            and epoch == self._tracking_epoch
            and revision == self._tracking_revision
        )
        if is_current:
            if error:
                self.analysis_status.setText(f"Tracking could not process frame {frame_number:,}: {error}")
                self.last_processed_frame_number = int(frame_number)
            else:
                self.frame_number = int(frame_number)
                self.media_controls.set_frame_position(self.frame_number, self.frame_count)
                self._show_tracking_result(frame, pipeline)
        request = self._tracking_pending
        self._tracking_pending = None
        if request is not None and request[3] == self._tracking_source_generation:
            self._launch_tracking_job(request)

    def invalidate_live_processing(self):
        self._tracking_epoch += 1
        self._tracking_pending = None

    def stop_tracking_worker(self):
        self.invalidate_live_processing()
        if self._tracking_pool is not None:
            self._tracking_pool.waitForDone(3000)

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def reset(self):
        try:
            self.previous_cross_count = 0
            self.crossing_recorded = False
            self.previous_red_centroid = None
            self.track_lost = False
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

    def _show_tracking_result(self, frame, pipeline):
        """Update the interface from a result produced by the Level 4 worker."""
        if pipeline is None:
            return

        try:
            self.last_frame = frame.copy()
            _, width = frame.shape[:2]
            line_x = (
                round(width * self.exit_line_fraction)
                if self.exit_line_fraction is not None
                else None
            )
            self.last_processed_frame_number = self.frame_number
            self.previous_red_centroid = pipeline["previous_centroid"]
            self.track_lost = pipeline["track_lost"]
            self.red_trail = pipeline["trail"]
            self.red_segments = pipeline["segments"]
            continuity_note = self.track_note
            self.track_note = pipeline["note"]
            if continuity_note:
                self.track_note = continuity_note + " " + self.track_note
            centroid = pipeline["centroid"]
            crossed = pipeline["crossed"]
            newly_crossed = crossed and not self.crossing_recorded
            cross_count = self.previous_cross_count + int(newly_crossed)
            segments = self.red_segments + ([self.red_trail] if self.red_trail else [])
            results = {
                "result": pipeline["result"],
                "cross_count": cross_count,
                "candidates": pipeline["candidate_count"],
                "centroid": centroid,
                "red_segments": segments,
            }
            self.last_results = results
            self.original_display.set_image(self.cv_to_pixmap(frame))
            self.result_display.set_image(self.cv_to_pixmap(results.get("result", frame)))
            self.notebook_button.setEnabled(True)
            cross_count = int(results.get("cross_count", 0))
            color = getattr(self, "target_color", "Red")
            if line_x is None:
                status = (
                    f"{pipeline['candidate_count']} {color.lower()} candidate(s) · entrance line not configured; "
                    "crossings are not evaluated."
                )
            else:
                direction = self.crossing_direction.replace("_", " ")
                status = (
                    f"{pipeline['candidate_count']} {color.lower()} candidate(s) · "
                    f"{cross_count} supported {direction} crossing(s)."
                )
            if centroid is None:
                status += " No centroid is tracked in this frame."
            if self.track_note:
                status += " " + self.track_note
            if line_x is None:
                status += " Inspect the video and configure the doorway line position."
            self.analysis_status.setText(status + f" {color} colour does not identify its carrier.")
            if newly_crossed:
                self.crossing_recorded = True
                self._append_event()
            self.previous_cross_count = cross_count
        except Exception as exc:
            self.log_message.emit(f"Motion analysis failed: {exc}", "error")

    def _append_event(self):
        row = self.events_table.rowCount()
        self.events_table.insertRow(row)
        seconds = self.frame_number / self.video_fps if self.video_fps > 0 else 0
        color = self.target_color
        values = [
            f"Frame {self.frame_number:,}",
            self._format_time(seconds),
            f"{color}-candidate line crossing",
            f"A continuity-selected {color.lower()} candidate crossed the configured line; identity remains unverified.",
        ]
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
        color = self.target_color
        registry.add(EvidenceEntry(
            module="Level 4: Track the Red Bag",
            kind="video_tracking",
            value={"candidate_count": self.last_results.get("candidates", 0), "centroid": centroid, "segments": self.last_results.get("red_segments", []), "line_crossings": cross_count, "entrance_line_x_fraction": self.exit_line_fraction, "crossing_direction": self.crossing_direction, "minimum_contour_area": self.min_area, "max_centroid_jump_fraction": self.max_step_fraction, "target_color": color},
            technique=f"OpenCV {color} HSV segmentation, morphology, area-filtered contours, nearest-centroid continuity" + (f" (configured line at {self.exit_line_fraction:.0%} frame width, direction {self.crossing_direction.replace('_', ' ')})" if self.exit_line_fraction is not None else " (entrance line not configured; crossings not evaluated)"),
            observation=self.analysis_status.text(),
            interpretation=f"The selected contour is a {color.lower()} candidate followed by centroid proximity between adjacent readable frames. A supported line crossing describes clip-relative movement only; it does not identify the object or its carrier.",
            uncertainty="High" if self.last_results.get("candidates", 0) != 1 or centroid is None else "Medium",
            source_filename=self.media_controls.filename_label.text(),
            source_path=self.source_path,
            output_path=output_path,
            source_timestamp=self._format_time(self.frame_number / self.video_fps if self.video_fps > 0 else 0),
            frame_number=self.frame_number,
            evidence_id="E-04",
            unresolved_question="Does the available clip and supporting evidence establish that the same physical object crossed?",
            limitation="HSV candidates can be distractors; a skipped or occluded detection breaks centroid continuity, and clip time is not wall-clock time without a verified offset.",
        ))
        self.level_completed.emit("The Final Clue")
        self.log_message.emit(f"{color} tracking observation recorded in the Evidence Notebook.", "success")

    def clear_media(self):
        self.invalidate_live_processing()
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
