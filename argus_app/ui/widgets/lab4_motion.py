"""Level 4: review a video timeline and inspect motion line crossings."""

import os
from PySide6.QtCore import QRunnable, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
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
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget, MediaTimelineControls


class TrackingJobSignals(QObject):
    finished = Signal(object, int, int, int, object)


class TrackingJob(QRunnable):
    """Run the authoritative Level 4 pipeline away from the interface thread."""

    def __init__(self, module, frame, frame_number, source_generation, epoch, revision, parameters):
        super().__init__()
        self.module = module
        self.frame = frame.copy()
        self.frame_number = int(frame_number)
        self.source_generation = int(source_generation)
        self.epoch = int(epoch)
        self.revision = int(revision)
        self.parameters = parameters
        self.signals = TrackingJobSignals()

    def run(self):
        try:
            pipeline = self.module.process_tracking_frame(self.frame, **self.parameters)
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
        self.level_module = None
        self.min_area = None
        self.max_step_fraction = None
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
        self.crossing_direction = "either"
        self.target_color = "Red"
        self.target_h_min = None
        self.target_h_max = None
        self.target_s_min = None
        self.target_s_max = None
        self.target_v_min = None
        self.target_v_max = None
        self._tracking_pool = None
        self._tracking_busy = False
        self._tracking_pending = None
        self._tracking_source_generation = None
        self._tracking_epoch = 0
        self._tracking_revision = 0
        super().__init__("Level 4: Track the Red Bag", parent)
        self._tracking_pool = QThreadPool(self)
        self.reset()

    def ensure_implementation(self):
        if self.level_module is not None:
            return True
        module, message = self.load_level_implementation(
            4,
            ("process_tracking_frame", "TRACKING_MIN_AREA", "TRACKING_MAX_STEP_FRACTION", "COLOR_PRESETS"),
        )
        if module is None:
            self.analysis_status.setText(message)
            return False
        self.level_module = module
        self.min_area = module.TRACKING_MIN_AREA
        self.max_step_fraction = module.TRACKING_MAX_STEP_FRACTION
        self._set_target_colour_parameters(self.target_color)
        if self.analysis_status.text().startswith("Level "):
            self.analysis_status.setText(
                "Upload a video to track colour candidates. No doorway crossing is reported unless a reliable line is configured."
            )
        return True

    def _set_target_colour_parameters(self, color_name):
        preset = self.level_module.COLOR_PRESETS.get(color_name)
        if preset is None:
            raise ValueError(f"Unsupported tracking colour: {color_name}")
        self.target_h_min = preset["h_min"]
        self.target_h_max = preset["h_max"]
        self.target_s_min = preset["s_min"]
        self.target_s_max = preset["s_max"]
        self.target_v_min = preset["v_min"]
        self.target_v_max = preset["v_max"]

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

        tracking_controls = QHBoxLayout()
        self.target_color_combo = QComboBox()
        self.target_color_combo.addItems(["Red", "Green", "Blue"])
        self.target_color_combo.currentTextChanged.connect(self._target_color_changed)
        tracking_controls.addWidget(QLabel("Target colour:"))
        tracking_controls.addWidget(self.target_color_combo)
        tracking_controls.addStretch(1)
        root.addLayout(tracking_controls)

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
        self.analysis_status = QLabel("Upload a video to track colour candidates. No doorway crossing is reported unless a reliable line is configured.")
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
        if not self.ensure_implementation():
            return
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
        if self.level_module is None:
            return
        self._set_target_colour_parameters(color_name)
        self._restart_tracking()

    def _restart_tracking(self):
        self._tracking_revision += 1
        if self.last_frame is not None and self.source_kind == "video":
            self.red_trail = []
            self.red_segments = []
            self.previous_red_centroid = None
            self.track_lost = False
            self.previous_cross_count = 0
            self.crossing_recorded = False
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
        if not self.ensure_implementation() or frame is None or self.source_kind != "video":
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
            self.level_module,
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
            if centroid is not None:
                status = f"Target detected · {pipeline['candidate_count']} {color.lower()} candidate(s)."
            elif pipeline["track_lost"]:
                status = f"Target temporarily lost · {pipeline['candidate_count']} {color.lower()} candidate(s)."
            else:
                status = f"Target not detected · {pipeline['candidate_count']} {color.lower()} candidate(s)."
            if self.track_note:
                status += " " + self.track_note
            if line_x is None:
                status += " Trajectory only; doorway crossing is not evaluated without a reliable line."
            else:
                status += f" {cross_count} candidate crossing(s) of the configured line."
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
        saved = registry.add(EvidenceEntry(
            module="Level 4: Track the Red Bag",
            kind="video_tracking",
            value={"candidate_count": self.last_results.get("candidates", 0), "centroid": centroid, "segments": self.last_results.get("red_segments", []), "line_crossings": cross_count, "entrance_line_x_fraction": self.exit_line_fraction, "crossing_direction": self.crossing_direction, "minimum_contour_area": self.min_area, "max_centroid_jump_fraction": self.max_step_fraction, "target_color": color},
            technique=f"OpenCV {color} HSV segmentation, morphology, area-filtered contours, nearest-centroid continuity" + (f" (configured line at {self.exit_line_fraction:.0%} frame width, either crossing direction)" if self.exit_line_fraction is not None else " (trajectory shown; doorway crossings not evaluated without a reliable line)"),
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
        if not saved:
            self.log_message.emit(registry.persistence_error or "Could not save tracking to the Evidence Notebook.", "error")
            return
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
        self.analysis_status.setText("Upload a video to track colour candidates. Doorway crossings need a reliable configured line; colour does not identify a person.")
        self.events_table.setRowCount(0)
        self.notebook_button.setEnabled(False)
        self.reset()
