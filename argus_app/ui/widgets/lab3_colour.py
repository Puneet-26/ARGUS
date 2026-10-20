"""Level 3: inspect actual HSV masks and red-region candidates."""

import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt, QRunnable, QThread, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QSplitter, QVBoxLayout

from evidence.artifacts import save_image
from evidence.registry import EvidenceEntry, registry
from ui.style import COLORS, FONTS
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget
from ui.widgets.lab2_detection import VIDEO_SUFFIXES, VideoPlaybackWorker


class SegmentationJobSignals(QObject):
    finished = Signal(object, int, int, int, object, str)


class SegmentationJob(QRunnable):
    """Run the authoritative Level 3 segmentation on a worker thread."""

    def __init__(self, module, frame, frame_number, generation, revision, parameters):
        super().__init__()
        self.module = module
        self.frame = frame.copy()
        self.frame_number = int(frame_number)
        self.generation = int(generation)
        self.revision = int(revision)
        self.parameters = parameters
        self.signals = SegmentationJobSignals()

    def run(self):
        try:
            result = self.module.process_color_preset(self.frame, self.parameters)
            error = ""
        except Exception as exc:
            result = None
            error = str(exc)
        self.signals.finished.emit(
            self.frame, self.frame_number, self.generation, self.revision, result, error
        )


class Lab3ColourWidget(LabWidget):
    """Run the fixed red segmentation pipeline and let students inspect outputs."""

    level_completed = Signal(str)
    open_video_requested = Signal(str, int)
    set_playing_requested = Signal(bool)
    seek_requested = Signal(int, int)
    close_video_requested = Signal()
    shutdown_requested = Signal()

    def __init__(self, parent=None):
        self.last_results = None
        self.last_frame = None
        self.evidence_name = ""
        self.media_kind = ""
        self.current_frame_number = None
        self.last_result_frame_number = None
        self.video_fps = 0.0
        self.video_frame_count = 0
        self._video_generation = 0
        self._segmentation_pool = None
        self._segmentation_busy = False
        self._segmentation_pending = None
        self._segmentation_revision = 0
        self.solution_module = None
        super().__init__("Level 3: Identify the Red Bag", parent)
        self._segmentation_pool = QThreadPool(self)
        self.playback_thread = QThread(self)
        self.playback_worker = VideoPlaybackWorker()
        self.playback_worker.moveToThread(self.playback_thread)
        self.playback_thread.started.connect(self.playback_worker.initialize)
        self.open_video_requested.connect(self.playback_worker.open_video)
        self.set_playing_requested.connect(self.playback_worker.set_playing)
        self.seek_requested.connect(self.playback_worker.seek)
        self.close_video_requested.connect(self.playback_worker.close_video)
        self.shutdown_requested.connect(self.playback_worker.shutdown)
        self.playback_worker.opened.connect(self.on_video_opened)
        self.playback_worker.frame_ready.connect(self.on_video_frame)
        self.playback_worker.failed.connect(self.on_video_failed)
        self.playback_worker.ended.connect(self.on_video_ended)
        self.playback_worker.closed.connect(self.playback_thread.quit, Qt.DirectConnection)
        self.playback_thread.finished.connect(self.playback_worker.deleteLater)
        self.playback_thread.start()

    def ensure_implementation(self):
        if self.solution_module is not None:
            return True
        module, message = self.load_level_implementation(3, ("process_color_preset",))
        if module is None:
            self.contour_info.setText(message)
            return False
        self.solution_module = module
        if self.contour_info.text().startswith("Level "):
            self.contour_info.setText("Choose a colour and select Detect Colour.")
        return True

    def setup_ui(self):
        super().setup_ui()
        root = QVBoxLayout(self.content_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(QLabel("Target colour:"))
        self.color_preset_combo = QComboBox()
        self.color_preset_combo.addItems(["Red", "Green", "Blue"])
        self.color_preset_combo.currentTextChanged.connect(self._on_color_preset_changed)
        controls.addWidget(self.color_preset_combo)
        self.apply_button = QPushButton("Detect Colour")
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.run_colour_mask)
        controls.addWidget(self.apply_button)
        controls.addWidget(QLabel("View:"))
        self.result_mode = QComboBox()
        self.result_mode.addItems(["Candidate regions", "Cleaned mask"])
        self.result_mode.currentTextChanged.connect(self._show_result_mode)
        self.result_mode.setEnabled(False)
        controls.addWidget(self.result_mode)
        self.log_button = QPushButton("Record result")
        self.log_button.setEnabled(False)
        self.log_button.clicked.connect(self.log_to_evidence)
        controls.addWidget(self.log_button)
        controls.addStretch(1)
        root.addLayout(controls)

        views = QSplitter(Qt.Horizontal)
        views.setChildrenCollapsible(False)
        self.original_display = ImageDisplayLabel("Open corridor image or video evidence to begin.")
        self.original_display.set_upload_enabled(True, "image or video")
        self.result_display = ImageDisplayLabel("Run segmentation to inspect the mask or candidates.")
        views.addWidget(self._panel("ORIGINAL IMAGE / VIDEO", self.original_display))
        result_panel = self._panel("PROCESSED RESULT", self.result_display)
        views.addWidget(result_panel)
        views.setStretchFactor(0, 1)
        views.setStretchFactor(1, 1)
        root.addWidget(views, 1)

        self.video_navigation = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_playback)
        self.video_navigation.addWidget(self.play_button)
        self.restart_button = QPushButton("Restart")
        self.restart_button.setEnabled(False)
        self.restart_button.clicked.connect(self.restart_video)
        self.video_navigation.addWidget(self.restart_button)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.sliderReleased.connect(self.seek_selected_frame)
        self.video_navigation.addWidget(self.frame_slider, 1)
        self.frame_position = QLabel("Image mode · video controls activate for a video.")
        self.video_navigation.addWidget(self.frame_position)
        root.addLayout(self.video_navigation)

        self.contour_info = QLabel("Choose a colour and select Detect Colour.")
        self.contour_info.setWordWrap(True)
        self.contour_info.setObjectName("pageSummary")
        root.addWidget(self.contour_info)

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
        self._video_generation += 1
        self._segmentation_revision += 1
        self._segmentation_pending = None
        self.set_playing_requested.emit(False)
        self.close_video_requested.emit()
        self.last_results = None
        self.last_result_frame_number = None
        self.last_frame = None
        self.evidence_name = ""
        self.media_kind = ""
        self.current_frame_number = None
        self.video_fps = 0.0
        self.video_frame_count = 0
        self.original_display.set_image(None)
        self.original_display.setText("Open corridor image or video evidence to begin.")
        self.result_display.set_image(None)
        self.result_display.setText("Run segmentation to inspect the mask or candidates.")
        self.contour_info.setText("Choose a colour and select Detect Colour.")
        self.apply_button.setText("Detect Colour")
        self.apply_button.setEnabled(False)
        self.log_button.setEnabled(False)
        self.result_mode.setEnabled(False)
        self.play_button.setText("Play")
        self.play_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_position.setText("Image mode · video controls activate for a video.")

    def load_evidence_frame(self, frame, frame_number=None, preserve_results=False):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            self.log_message.emit("Choose a readable BGR colour frame for Level 3.", "error")
            return False
        if self.media_kind != "video":
            self.media_kind = "image"
        self.current_frame_number = int(frame_number) if frame_number is not None else None
        self.last_frame = np.ascontiguousarray(frame).copy()
        if not preserve_results:
            self.last_results = None
            self.last_result_frame_number = None
            self.original_display.set_image(self.cv_to_pixmap(self.last_frame))
            self.result_display.set_image(None)
        height, width = self.last_frame.shape[:2]
        if self.media_kind == "video" and self.current_frame_number is not None:
            self._update_video_position(self.current_frame_number)
            if not preserve_results:
                self.contour_info.setText(f"{Path(self.evidence_name).name} · frame {self.current_frame_number:,} · select Detect Colour.")
        else:
            self.frame_position.setText(f"Image · {width} × {height}")
            self.contour_info.setText(f"{Path(self.evidence_name).name if self.evidence_name else 'Image'} · {width} × {height} · select Detect Colour.")
        self.apply_button.setEnabled(True)
        if not preserve_results:
            self.log_button.setEnabled(False)
            self.result_mode.setEnabled(False)
        return True

    def load_evidence_video(self, path):
        video = Path(path)
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            self.log_message.emit("Choose an existing supported video file for Level 3.", "error")
            return False
        self._video_generation += 1
        self._segmentation_revision += 1
        self._segmentation_pending = None
        self.set_playing_requested.emit(False)
        self.close_video_requested.emit()
        self.media_kind = "video"
        self.current_frame_number = None
        self.video_fps = 0.0
        self.video_frame_count = 0
        self.last_frame = None
        self.last_results = None
        self.last_result_frame_number = None
        self.original_display.set_image(None)
        self.original_display.setText("Opening video and decoding its first frame…")
        self.result_display.set_image(None)
        self.result_display.setText("Waiting for the first video frame…")
        self.contour_info.setText(f"{video.name} · opening video…")
        self.apply_button.setEnabled(False)
        self.log_button.setEnabled(False)
        self.result_mode.setEnabled(False)
        self.play_button.setText("Play")
        self.play_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_position.setText("Opening video…")
        self.open_video_requested.emit(str(video.resolve()), self._video_generation)
        return True

    @Slot(object, int, float, int, int)
    def on_video_opened(self, frame, frame_number, fps, frame_count, generation):
        if self.media_kind != "video" or generation != self._video_generation:
            return
        self.video_fps = float(fps or 0.0)
        self.video_frame_count = int(frame_count or 0)
        self.frame_slider.setRange(0, max(0, self.video_frame_count - 1))
        self.frame_slider.setEnabled(self.video_frame_count > 1)
        self.play_button.setEnabled(True)
        self.restart_button.setEnabled(True)
        self.load_evidence_frame(frame, frame_number)
        self.run_colour_mask()

    @Slot(object, int, int)
    def on_video_frame(self, frame, frame_number, generation):
        if self.media_kind == "video" and generation == self._video_generation:
            self.load_evidence_frame(frame, frame_number, preserve_results=True)
            self._schedule_segmentation(frame, frame_number)

    @Slot(str, int)
    def on_video_failed(self, message, generation):
        if self.media_kind != "video" or generation != self._video_generation:
            return
        self.last_frame = None
        self.apply_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.original_display.set_image(None)
        self.original_display.setText(f"Could not open video: {message}")
        self.contour_info.setText(f"Could not load {Path(self.evidence_name).name}: {message}")
        self.log_message.emit(f"Could not open Level 3 video: {message}", "error")

    @Slot(int)
    def on_video_ended(self, generation):
        if generation == self._video_generation:
            self.play_button.setText("Play")

    def restart_video(self):
        if self.media_kind == "video":
            self.set_playing_requested.emit(False)
            self.play_button.setText("Play")
            self._segmentation_revision += 1
            self._segmentation_pending = None
            self.seek_requested.emit(0, self._video_generation)

    def pause_playback(self):
        self.set_playing_requested.emit(False)
        self.play_button.setText("Play")
        self._segmentation_revision += 1
        self._segmentation_pending = None

    def toggle_playback(self):
        if self.media_kind != "video" or self.last_frame is None:
            return
        playing = self.play_button.text() != "Pause"
        self.play_button.setText("Pause" if playing else "Play")
        self.set_playing_requested.emit(playing)

    def seek_selected_frame(self):
        self.set_playing_requested.emit(False)
        self.play_button.setText("Play")
        self._segmentation_revision += 1
        self._segmentation_pending = None
        self.seek_requested.emit(self.frame_slider.value(), self._video_generation)

    def _update_video_position(self, frame_number):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(int(frame_number))
        self.frame_slider.blockSignals(False)
        timestamp = frame_number / self.video_fps if self.video_fps > 0 else None
        if timestamp is not None:
            self.frame_position.setText(f"Frame {frame_number:,} · {timestamp:.2f}s")
        else:
            self.frame_position.setText(f"Frame {frame_number:,}")

    def shutdown_media(self):
        if self.playback_thread.isRunning():
            self.shutdown_requested.emit()
            self.playback_thread.wait(3000)
        if self._segmentation_pool is not None:
            self._segmentation_pool.waitForDone(3000)

    def run_colour_mask(self):
        if not self.ensure_implementation():
            return {"output": None, "error": self.contour_info.text(), "success": False}
        if self.last_frame is None:
            self.log_message.emit("Open Level 3 image or video evidence before segmenting.", "warning")
            return {"output": None, "error": "No evidence selected", "success": False}
        if self.media_kind == "video":
            self._segmentation_revision += 1
            self._schedule_segmentation(self.last_frame, self.current_frame_number)
            return {"output": None, "success": True, "queued": True}
        try:
            results = self._process_segmentation(self.last_frame, self.color_preset_combo.currentText())
            self._apply_segmentation_result(self.last_frame, self.current_frame_number, results)
            return {"output": results["result"], "success": True}
        except Exception as exc:
            self.last_results = None
            self.result_display.setText("Segmentation failed. Check the selected image or frame.")
            self.contour_info.setText(f"Could not segment the image: {exc}")
            self.log_button.setEnabled(False)
            self.log_message.emit(f"Colour analysis failed: {exc}", "error")
            return {"output": self.last_frame, "error": str(exc), "success": False}

    def _on_color_preset_changed(self, color_name):
        if self.last_frame is not None:
            self.run_colour_mask()

    def _process_segmentation(self, frame, color_name):
        results = self.solution_module.process_color_preset(frame.copy(), color_name)
        if not results or results.get("mask_cleaned") is None:
            raise RuntimeError("The segmentation pipeline returned no mask.")
        return results

    def _schedule_segmentation(self, frame, frame_number):
        if frame is None or self.solution_module is None:
            return
        request = (frame.copy(), int(frame_number or 0), self._video_generation, self._segmentation_revision,
                   self.color_preset_combo.currentText())
        if self._segmentation_busy:
            self._segmentation_pending = request
            return
        self._launch_segmentation_job(request)

    def _launch_segmentation_job(self, request):
        frame, frame_number, generation, revision, parameters = request
        job = SegmentationJob(self.solution_module, frame, frame_number, generation, revision, parameters)
        job.signals.finished.connect(self._on_segmentation_finished)
        self._segmentation_busy = True
        self._segmentation_pool.start(job)

    @Slot(object, int, int, int, object, str)
    def _on_segmentation_finished(self, frame, frame_number, generation, revision, results, error):
        self._segmentation_busy = False
        if generation == self._video_generation and revision == self._segmentation_revision:
            if error:
                self.contour_info.setText(f"Could not segment frame {frame_number:,}: {error}")
                self.log_button.setEnabled(False)
            else:
                self.current_frame_number = frame_number
                self.last_frame = frame.copy()
                self.original_display.set_image(self.cv_to_pixmap(frame))
                self._update_video_position(frame_number)
                self._apply_segmentation_result(frame, frame_number, results)
        pending = self._segmentation_pending
        self._segmentation_pending = None
        if pending is not None and pending[2] == self._video_generation:
            self._launch_segmentation_job(pending)

    def _apply_segmentation_result(self, frame, frame_number, results):
        color_name = self.color_preset_combo.currentText()
        self.last_results = results
        self.last_result_frame_number = frame_number
        candidate_data = results.get("candidate_regions", [])
        frame_info = f" · frame {frame_number:,}" if frame_number is not None else ""
        if frame_number is not None and self.video_fps > 0:
            frame_info += f" · {frame_number / self.video_fps:.2f}s"
        self.contour_info.setText(
            f"{len(candidate_data)} {color_name.lower()} candidate region(s){frame_info}. "
            "Colour regions are not confirmed object identities."
        )
        self._show_result_mode(self.result_mode.currentText())
        self.result_mode.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.log_button.setEnabled(True)

    def _show_result_mode(self, mode):
        if not self.last_results:
            return
        image = self.last_results.get("mask_cleaned") if "mask" in mode.lower() else self.last_results.get("result")
        if image is not None:
            self.result_display.set_image(self.cv_to_pixmap(image))

    def log_to_evidence(self):
        if self.last_results is None:
            return
        result_frame_number = self.last_result_frame_number
        frame_suffix = f"_frame_{result_frame_number}" if self.media_kind == "video" and result_frame_number is not None else ""
        file_stem = Path(self.evidence_name).stem or "level3_evidence"
        frame_timestamp = result_frame_number / self.video_fps if result_frame_number is not None and self.video_fps > 0 else None
        source_timestamp = f"{frame_timestamp:.2f}s" if frame_timestamp is not None else ""
        try:
            candidate_output = save_image(3, f"{file_stem}{frame_suffix}_candidate_regions.png", self.last_results["result"])
            mask_output = save_image(3, f"{file_stem}{frame_suffix}_cleaned_mask.png", self.last_results["mask_cleaned"])
        except (OSError, ValueError) as exc:
            self.log_message.emit(f"Could not save segmentation outputs: {exc}", "error")
            return
        candidate_data = self.last_results.get("candidate_regions", [])
        color_name = self.color_preset_combo.currentText()
        saved = registry.add(EvidenceEntry(
            module="Level 3: Identify the Red Bag", kind="color_segmentation",
            value={"candidate_regions": candidate_data, "outputs": [candidate_output, mask_output], "parameters": self.last_results.get("parameters", {}), "media_kind": self.media_kind, "frame_number": result_frame_number, "fps": self.video_fps or None, "timestamp_seconds": frame_timestamp},
            technique=f"BGR-to-HSV, {color_name} color range segmentation, morphology, area-filtered contours, and candidate overlay",
            observation=self.contour_info.text(),
            interpretation=f"The mask marks {color_name.lower()} pixels and the boxes mark area-filtered candidates. Compare them with the source image; the algorithm does not recognize semantic object identity.",
            uncertainty="Medium", source_filename=Path(self.evidence_name).name,
            source_path=self.evidence_name, output_path=candidate_output, source_timestamp=source_timestamp, frame_number=result_frame_number, evidence_id="E-03",
            unresolved_question="Which candidate, if any, matches the object based on separate visual evidence?",
            limitation=f"Other {color_name.lower()} objects may be candidates. HSV segmentation does not establish object identity or ownership.",
        ))
        if not saved:
            self.log_message.emit(registry.persistence_error or "Could not save segmentation to the Evidence Notebook.", "error")
            return
        self.level_completed.emit("The Red Bag")
        self.log_message.emit(f"{color_name} region result recorded · {candidate_output} and {mask_output}", "success")
