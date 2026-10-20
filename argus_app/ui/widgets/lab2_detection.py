"""Level 2: inspect images or video frames with actual local YOLO inference."""

import threading
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.input_source import read_image_file
from evidence.artifacts import save_image
from evidence.registry import EvidenceEntry, registry
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".dib", ".webp", ".tif", ".tiff", ".ppm", ".pgm", ".pnm", ".jp2"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv", ".3gp", ".3g2", ".flv", ".ts", ".mts", ".m2ts", ".ogv"}
IMAGE_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.jfif *.bmp *.dib *.webp *.tif *.tiff *.ppm *.pgm *.pnm *.jp2)"
VIDEO_FILE_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.mpeg *.mpg *.wmv)"


class ImageLoadWorker(QThread):
    """Decode a still image outside the Qt interface thread."""

    loaded = Signal(object, str, int)
    failed = Signal(str, int)

    def __init__(self, path, generation, parent=None):
        super().__init__(parent)
        self.path = str(path)
        self.generation = int(generation)

    def run(self):
        frame = read_image_file(self.path)
        if self.isInterruptionRequested():
            return
        if frame is None:
            self.failed.emit(f"OpenCV could not decode {Path(self.path).name}. Choose another supported image file.", self.generation)
            return
        self.loaded.emit(frame, self.path, self.generation)


class VideoPlaybackWorker(QObject):
    """Own an OpenCV video handle in a worker thread and emit decoded frames."""

    opened = Signal(object, int, float, int, int)
    frame_ready = Signal(object, int, int)
    failed = Signal(str, int)
    ended = Signal(int)
    closed = Signal()

    def __init__(self):
        super().__init__()
        self.capture = None
        self.fps = 0.0
        self.frame_count = 0
        self.timer = None
        self.generation = 0

    @Slot()
    def initialize(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.read_next)

    @Slot(str, int)
    def open_video(self, path, generation):
        self.generation = int(generation)
        self._release_capture()
        media = Path(path)
        if not media.is_file():
            self.failed.emit(f"The selected video does not exist: {media}", self.generation)
            return
        capture = cv2.VideoCapture(str(media))
        if not capture.isOpened():
            capture.release()
            self.failed.emit(f"OpenCV could not open {media.name}. Check its codec and file integrity.", self.generation)
            return
        self.capture = capture
        self.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not self.fps or self.fps <= 0:
            self.fps = 25.0
        self.frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        ok, frame = capture.read()
        if not ok or frame is None:
            self._release_capture()
            self.failed.emit(f"OpenCV opened {media.name} but could not decode a frame.", self.generation)
            return
        self.opened.emit(frame.copy(), 0, self.fps, self.frame_count, self.generation)

    @Slot(bool)
    def set_playing(self, playing):
        if self.timer is None or self.capture is None:
            return
        if not playing:
            self.timer.stop()
            return
        position = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        if self.frame_count and position >= self.frame_count:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._read_and_emit()
        interval = max(1, min(250, round(1000.0 / self.fps)))
        self.timer.start(interval)

    @Slot(int, int)
    def seek(self, frame_number, generation):
        if self.capture is None or int(generation) != self.generation:
            return
        maximum = max(0, self.frame_count - 1)
        target = max(0, min(int(frame_number), maximum))
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, target):
            self.failed.emit("The video reader could not seek to that frame.", self.generation)
            return
        self._read_and_emit()

    @Slot()
    def read_next(self):
        self._read_and_emit()

    def _read_and_emit(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            if self.timer is not None:
                self.timer.stop()
            self.ended.emit(self.generation)
            return
        frame_number = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.frame_ready.emit(frame.copy(), frame_number, self.generation)

    def _release_capture(self):
        if self.timer is not None:
            self.timer.stop()
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    @Slot()
    def close_video(self):
        self._release_capture()

    @Slot()
    def shutdown(self):
        self._release_capture()
        self.closed.emit()


class DetectionWorker(QThread):
    """Run CPU YOLO inference off the UI thread and reuse loaded checkpoints."""

    _models = {}
    _model_lock = threading.Lock()
    progress = Signal(str)
    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(self, media_path: str, frame, media_kind="image", frame_number=None, fps=0.0, confidence=0.25, parent=None):
        super().__init__(parent)
        self.media_path = str(media_path)
        self.frame = frame.copy() if frame is not None else None
        self.media_kind = str(media_kind)
        self.frame_number = frame_number
        self.fps = float(fps or 0.0)
        self.confidence = float(confidence)

    def cancel(self):
        self.requestInterruption()

    @staticmethod
    def _weights_path() -> str:
        import os

        configured = os.environ.get("ARGUS_YOLO_WEIGHTS", "").strip()
        if configured:
            return str(Path(configured).expanduser())
        project_root = Path(__file__).resolve().parents[3]
        custom = project_root / "assets" / "models" / "custom" / "best.pt"
        bundled = project_root / "assets" / "models" / "yolo11n.pt"
        if custom.is_file():
            return str(custom)
        if bundled.is_file():
            return str(bundled)
        return ""

    def run(self):
        try:
            media = Path(self.media_path)
            if media.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
                raise RuntimeError("Level 2 accepts supported image and video files only.")
            if not media.is_file():
                raise RuntimeError(f"The selected evidence no longer exists: {media}")
            if self.frame is None:
                raise RuntimeError("There is no decoded frame to inspect. Select or seek to a readable frame first.")
            weights = self._weights_path()
            if not weights:
                raise RuntimeError('YOLO weights are missing. From the project root run: python -c "from ultralytics import YOLO; YOLO(\'yolo11n.pt\')". Or set ARGUS_YOLO_WEIGHTS to a local .pt file.')
            if not Path(weights).expanduser().is_file():
                raise RuntimeError(f"YOLO weights were not found: {weights}")
            checkpoint = str(Path(weights).expanduser().resolve())
            self.progress.emit(f"Loading YOLO model {Path(checkpoint).name}…")
            with self._model_lock:
                model = self._models.get(checkpoint)
                if model is None:
                    try:
                        from ultralytics import YOLO
                    except ImportError as exc:
                        raise RuntimeError("Ultralytics is not installed. From the project folder run `python -m pip install -r requirements.txt`.") from exc
                    model = YOLO(checkpoint)
                    self._models[checkpoint] = model
            if self.isInterruptionRequested():
                return
            self.progress.emit("Running CPU YOLO inference on the selected frame…")
            if self.isInterruptionRequested():
                return
            from level_codes.level2 import infer_frame
            with self._model_lock:
                annotated, records = infer_frame(model, self.frame, self.confidence, 640)
            if self.isInterruptionRequested():
                return
            frame_number = self.frame_number
            timestamp = frame_number / self.fps if frame_number is not None and self.fps > 0 else None
            if self.media_kind == "video":
                where = f"frame {frame_number:,}" if frame_number is not None else "selected frame"
                if timestamp is not None:
                    where += f" at {timestamp:.2f}s"
                summary = f"Video frame inference · {where} · {len(records)} supported detection(s). Unlisted objects and object identity could not be verified."
            else:
                summary = f"Image inference · {len(records)} supported detection(s). Unlisted objects could not be verified by this model."
            self.completed.emit([{
                "frame_number": frame_number,
                "fps": self.fps,
                "original": self.frame,
                "annotated": annotated,
                "records": records,
            }], summary)
        except Exception as exc:
            self.failed.emit(str(exc))


class Lab2DetectionWidget(LabWidget):
    """Image and video frame detector with synchronized source and YOLO views."""

    level_completed = Signal(str)
    clear_requested = Signal()
    evidence_requested = Signal(str)
    open_video_requested = Signal(str, int)
    set_playing_requested = Signal(bool)
    seek_requested = Signal(int, int)
    close_video_requested = Signal()
    shutdown_requested = Signal()

    def __init__(self, parent=None):
        self.media_path = ""
        self.media_kind = ""
        self.worker = None
        self.last_records = None
        self.last_summary = ""
        self.frame_results = []
        self._infer_after_seek = False
        self.current_frame = None
        self.current_frame_number = None
        self.video_fps = 0.0
        self.video_frame_count = 0
        self._infer_after_seek = False
        self._video_generation = 0
        self._media_generation = 0
        self._image_readers = set()
        self.image_reader = None
        self.auto_detect_after_image_load = False
        self._focus_annotated = False
        self._live_pending = None
        super().__init__("Level 2: The Missing Phone", parent)
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

    def setup_ui(self):
        super().setup_ui()
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        controls = QHBoxLayout()
        self.open_button = QPushButton("Open Evidence…")
        self.open_button.clicked.connect(self.choose_media)
        controls.addWidget(self.open_button)
        self.run_button = QPushButton("Run YOLO Detection")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_detection)
        controls.addWidget(self.run_button)
        self.live_checkbox = QCheckBox("Live YOLO during playback")
        self.live_checkbox.setChecked(True)
        self.live_checkbox.setToolTip("Runs YOLO off the interface thread. Intermediate frames are dropped when inference is busy.")
        self.live_checkbox.toggled.connect(self._toggle_live_detection)
        controls.addWidget(self.live_checkbox)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        self.focus_button = QPushButton("Focus annotated view")
        self.focus_button.setToolTip("Expand the annotated evidence to the full viewing area.")
        self.focus_button.clicked.connect(self.toggle_annotated_focus)
        controls.addWidget(self.focus_button)
        layout.addLayout(controls)

        confidence_row = QHBoxLayout()
        self.confidence_label = QLabel("Minimum confidence: 25%")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(5, 95)
        self.confidence_slider.setValue(25)
        self.confidence_slider.setToolTip("Move the threshold and release to rerun YOLO on the selected frame. Higher thresholds hide weaker predictions.")
        self.confidence_slider.valueChanged.connect(lambda value: self.confidence_label.setText(f"Minimum confidence: {value}%"))
        self.confidence_slider.sliderReleased.connect(self._rerun_for_confidence)
        confidence_row.addWidget(self.confidence_label)
        confidence_row.addWidget(self.confidence_slider, 1)
        layout.addLayout(confidence_row)

        self.views = QSplitter(Qt.Horizontal)
        self.views.setChildrenCollapsible(False)
        self.original_display = ImageDisplayLabel("Select image or video evidence.")
        self.original_display.set_upload_enabled(True, "image or video")
        self.result_display = ImageDisplayLabel("Actual model predictions appear here.")
        self.original_panel = self._panel("ORIGINAL EVIDENCE", self.original_display)
        self.annotated_panel = self._panel("ANNOTATED EVIDENCE", self.result_display)
        self.original_display.setMinimumHeight(280)
        self.result_display.setMinimumHeight(280)
        self.views.addWidget(self.original_panel)
        self.views.addWidget(self.annotated_panel)
        self.views.setStretchFactor(0, 1)
        self.views.setStretchFactor(1, 1)
        self.views.setSizes([1, 1])
        layout.addWidget(self.views, 1)

        detected_panel = QFrame()
        detected_panel.setObjectName("evidencePanel")
        detected_layout = QVBoxLayout(detected_panel)
        detected_layout.setContentsMargins(8, 5, 8, 5)
        detected_layout.setSpacing(4)
        detected_heading = QLabel("DETECTED OBJECTS · ORIGINAL-FRAME COORDINATES")
        detected_heading.setObjectName("eyebrowLabel")
        detected_layout.addWidget(detected_heading)
        self.detected_objects_scroll = QScrollArea()
        self.detected_objects_scroll.setWidgetResizable(True)
        self.detected_objects_scroll.setMinimumHeight(82)
        self.detected_objects_scroll.setMaximumHeight(142)
        self.detected_objects_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.detected_objects_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.detected_objects_content = QWidget()
        self.detected_objects_layout = QHBoxLayout(self.detected_objects_content)
        self.detected_objects_layout.setContentsMargins(2, 2, 2, 2)
        self.detected_objects_layout.setSpacing(8)
        self.detected_objects_scroll.setWidget(self.detected_objects_content)
        self._populate_detection_list([])
        detected_layout.addWidget(self.detected_objects_scroll)
        detected_panel.setMaximumHeight(160)
        layout.addWidget(detected_panel)

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
        self.frame_position = QLabel("Select a video to use playback controls.")
        self.video_navigation.addWidget(self.frame_position)
        layout.addLayout(self.video_navigation)

        self.status = QLabel("Choose an image or video. Pause or seek to a frame before running YOLO.")
        self.status.setWordWrap(True)
        self.status.setObjectName("pageSummary")
        layout.addWidget(self.status)
        self.results_summary = QLabel("Predictions cover model-supported classes only; detections do not identify people or prove an event.")
        self.results_summary.setWordWrap(True)
        self.results_summary.setObjectName("mutedLabel")
        feedback = QHBoxLayout()
        feedback.setSpacing(8)
        feedback.addWidget(self.results_summary, 1)
        self.record_button = QPushButton("Record detections in notebook")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.record_finding)
        feedback.addWidget(self.record_button)
        layout.addLayout(feedback)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "views") and not self._focus_annotated:
            orientation = Qt.Horizontal if event.size().width() >= 1050 else Qt.Vertical
            if self.views.orientation() != orientation:
                self.views.setOrientation(orientation)

    def toggle_annotated_focus(self):
        self._focus_annotated = not self._focus_annotated
        self.original_panel.setVisible(not self._focus_annotated)
        self.focus_button.setText("Show both views" if self._focus_annotated else "Focus annotated view")
        self.focus_button.setToolTip(
            "Show the source and annotated frames side by side."
            if self._focus_annotated
            else "Expand the annotated evidence to the full viewing area."
        )

    @staticmethod
    def _panel(title, display):
        panel = QFrame()
        panel.setObjectName("evidencePanel")
        box = QVBoxLayout(panel)
        box.setContentsMargins(8, 8, 8, 8)
        label = QLabel(title)
        label.setObjectName("eyebrowLabel")
        box.addWidget(label)
        box.addWidget(display, 1)
        return panel

    def _populate_detection_list(self, records, frame_number=None, empty_message=None):
        while self.detected_objects_layout.count():
            item = self.detected_objects_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not records:
            if empty_message:
                message = empty_message
            elif frame_number is not None or self.last_records is not None:
                message = "No supported objects detected in this frame."
            else:
                message = "Run YOLO to list actual model detections."
            empty = QLabel(message)
            empty.setWordWrap(True)
            empty.setObjectName("mutedLabel")
            self.detected_objects_layout.addWidget(empty)
            return

        for detection in records:
            card = QFrame()
            card.setObjectName("detectionCard")
            card.setStyleSheet("QFrame#detectionCard { background: #20252b; border: 1px solid #46515d; border-radius: 7px; }")
            card.setMinimumWidth(230)
            card.setMaximumWidth(330)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 7, 10, 7)
            card_layout.setSpacing(3)
            class_label = QLabel(str(detection.get("label", "unknown")))
            class_label.setObjectName("detectionClassLabel")
            class_label.setStyleSheet("font-size: 15px; font-weight: 700;")
            confidence = float(detection.get("confidence", 0.0))
            score_label = QLabel(f"Confidence: {confidence:.2f}")
            score_label.setObjectName("mutedLabel")
            score_label.setStyleSheet("font-size: 13px;")
            coordinates = detection.get("box", detection.get("xyxy", []))
            if len(coordinates) == 4:
                box_label = QLabel(
                    "Bounding box (x1, y1, x2, y2): "
                    f"({int(coordinates[0])}, {int(coordinates[1])}, "
                    f"{int(coordinates[2])}, {int(coordinates[3])})"
                )
            else:
                box_label = QLabel("Bounding box: unavailable")
            box_label.setObjectName("mutedLabel")
            box_label.setWordWrap(True)
            box_label.setStyleSheet("font-size: 12px;")
            card_layout.addWidget(class_label)
            card_layout.addWidget(score_label)
            card_layout.addWidget(box_label)
            if frame_number is not None:
                frame_label = QLabel(f"Frame: {frame_number:,}")
                frame_label.setObjectName("mutedLabel")
                frame_label.setStyleSheet("font-size: 12px;")
                card_layout.addWidget(frame_label)
            self.detected_objects_layout.addWidget(card)

    def choose_media(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image or video evidence",
            "",
            f"Images and videos ({' '.join('*' + suffix for suffix in sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES))});;{IMAGE_FILE_FILTER};;{VIDEO_FILE_FILTER};;All files (*)",
        )
        if path:
            self.evidence_requested.emit(path)

    def load_media(self, path: str, auto_detect=False):
        media = Path(path)
        suffix = media.suffix.lower()
        if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            self.status.setText("Level 2 accepts supported image and video files only.")
            return False
        if not media.is_file():
            self.status.setText(f"The selected evidence does not exist: {media}")
            return False

        self.set_playing_requested.emit(False)
        self._video_generation += 1
        self._media_generation += 1
        generation = self._media_generation
        self.auto_detect_after_image_load = False
        self.media_path = str(media.resolve())
        self.media_kind = "video" if suffix in VIDEO_SUFFIXES else "image"
        self._live_pending = None
        self.last_records = None
        self.last_summary = ""
        self.frame_results = []
        self.current_frame = None
        self.current_frame_number = None
        self.video_fps = 0.0
        self.video_frame_count = 0
        self.result_display.set_image(None)
        self.result_display.setText("Run YOLO on the selected frame to inspect predictions.")
        self.results_summary.setText("Predictions cover model-supported classes only; detections do not identify people or prove an event.")
        self._populate_detection_list([])
        self.run_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.clear_button.setEnabled(True)
        self.open_button.setEnabled(True)

        if self.media_kind == "image":
            self.close_video_requested.emit()
            self.original_display.set_image(None)
            self.original_display.setText("Loading image evidence…")
            self.frame_position.setText("Loading image…")
            self.play_button.setEnabled(False)
            self.frame_slider.setEnabled(False)
            self.frame_slider.setRange(0, 0)
            self.status.setText(f"{media.name} · decoding image…")
            self.auto_detect_after_image_load = bool(auto_detect)
            reader = ImageLoadWorker(str(media.resolve()), generation, self)
            self.image_reader = reader
            self._image_readers.add(reader)
            reader.loaded.connect(self.on_image_loaded)
            reader.failed.connect(self.on_image_failed)
            reader.finished.connect(lambda worker=reader: self._image_readers.discard(worker))
            reader.finished.connect(reader.deleteLater)
            reader.start()
            return True
        else:
            self.original_display.set_image(None)
            self.original_display.setText("Opening video and decoding its first frame…")
            self.frame_position.setText("Opening video…")
            self.play_button.setText("Play")
            self.play_button.setEnabled(False)
            self.frame_slider.setEnabled(False)
            self.frame_slider.setRange(0, 0)
            self.status.setText(f"{media.name} · opening video…")
            self.open_video_requested.emit(str(media.resolve()), self._video_generation)
        return True

    @Slot(object, str, int)
    def on_image_loaded(self, frame, path, generation):
        if self.media_kind != "image" or generation != self._media_generation or path != self.media_path:
            return
        self.current_frame = frame.copy()
        self.original_display.set_image(self.cv_to_pixmap(frame))
        height, width = frame.shape[:2]
        self.frame_position.setText(f"Image · {width} × {height}")
        self.run_button.setEnabled(True)
        self.status.setText(f"{Path(path).name} · image · {width} × {height} · ready for YOLO")
        if self.auto_detect_after_image_load:
            self.auto_detect_after_image_load = False
            self.run_detection()

    @Slot(str, int)
    def on_image_failed(self, message, generation):
        if self.media_kind != "image" or generation != self._media_generation:
            return
        self.current_frame = None
        self.original_display.set_image(None)
        self.original_display.setText(message)
        self.run_button.setEnabled(False)
        self.status.setText(message)
        self.log_message.emit(message, "error")

    @Slot(object, int, float, int, int)
    def on_video_opened(self, frame, frame_number, fps, frame_count, generation):
        if self.media_kind != "video" or generation != self._video_generation:
            return
        self.video_fps = float(fps or 0.0)
        self.video_frame_count = int(frame_count or 0)
        self.restart_button.setEnabled(True)
        self.frame_slider.setRange(0, max(0, self.video_frame_count - 1))
        self.frame_slider.setEnabled(self.video_frame_count > 1)
        self.play_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self._set_video_frame(frame, frame_number)
        if self.live_checkbox.isChecked():
            self._queue_live_detection(frame, frame_number)
        self.status.setText(f"{Path(self.media_path).name} · video · {self.video_frame_count or 'unknown'} frames · {self.video_fps:.2f} FPS")

    @Slot(object, int, int)
    def on_video_frame(self, frame, frame_number, generation):
        if self.media_kind == "video" and generation == self._video_generation:
            self._set_video_frame(frame, frame_number)
            if self.live_checkbox.isChecked():
                self._queue_live_detection(frame, frame_number)
            if self._infer_after_seek:
                self._infer_after_seek = False
                QTimer.singleShot(0, self.run_detection)

    def _set_video_frame(self, frame, frame_number):
        self.current_frame = frame.copy()
        self.current_frame_number = int(frame_number)
        if not self.live_checkbox.isChecked():
            self.original_display.set_image(self.cv_to_pixmap(frame))
        self._update_video_position(self.current_frame_number)
        if not self.live_checkbox.isChecked():
            self._clear_stale_result("Frame changed. Run YOLO to inspect this frame.")

    def _update_video_position(self, frame_number):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(int(frame_number))
        self.frame_slider.blockSignals(False)
        duration = self.video_frame_count / self.video_fps if self.video_frame_count and self.video_fps > 0 else None
        timestamp = frame_number / self.video_fps if self.video_fps > 0 else None
        if timestamp is not None and duration is not None:
            self.frame_position.setText(f"Frame {frame_number:,} · {timestamp:.2f}s / {duration:.2f}s")
        elif timestamp is not None:
            self.frame_position.setText(f"Frame {frame_number:,} · {timestamp:.2f}s")
        else:
            self.frame_position.setText(f"Frame {frame_number:,}")

    @Slot(str, int)
    def on_video_failed(self, message, generation):
        if self.media_kind == "video" and generation == self._video_generation:
            self._infer_after_seek = False
            if "seek" in message.lower():
                self.status.setText(message)
                self.log_message.emit(message, "error")
                return
            self.set_playing_requested.emit(False)
            self.current_frame = None
            self.run_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.frame_slider.setEnabled(False)
            self.restart_button.setEnabled(False)
            self.original_display.set_image(None)
            self.original_display.setText(message)
            self.status.setText(f"Video could not be loaded: {message}")
            self.log_message.emit(message, "error")

    @Slot(int)
    def on_video_ended(self, generation):
        if generation == self._video_generation:
            self.play_button.setText("Play")

    def restart_video(self):
        if self.media_kind == "video":
            self.set_playing_requested.emit(False)
            self.play_button.setText("Play")
            self._media_generation += 1
            self._live_pending = None
            self._clear_stale_result("Restarted. Waiting for live YOLO results." if self.live_checkbox.isChecked() else "Restarted. Run YOLO to inspect a frame.")
            self.seek_requested.emit(0, self._video_generation)

    def pause_playback(self):
        self.set_playing_requested.emit(False)
        self.play_button.setText("Play")
        self._live_pending = None
        if self.worker is not None and self.worker.isRunning() and getattr(self.worker, "live_request", False):
            self._media_generation += 1

    def _toggle_live_detection(self, enabled):
        if self.media_kind != "video":
            return
        if self.worker is None or not self.worker.isRunning() or getattr(self.worker, "live_request", False):
            self._media_generation += 1
        self._live_pending = None
        if enabled and self.current_frame is not None:
            self._queue_live_detection(self.current_frame, self.current_frame_number)
        elif not enabled:
            self._live_pending = None
            if self.worker is None or not self.worker.isRunning():
                self._clear_stale_result("Live YOLO is off. Run YOLO on a selected frame.")

    def _queue_live_detection(self, frame, frame_number):
        if not self.live_checkbox.isChecked() or self.media_kind != "video":
            return
        request = (frame.copy(), int(frame_number), self._video_generation, self._media_generation)
        if self.worker is not None and self.worker.isRunning():
            self._live_pending = request  # one latest frame only; stale intermediates are dropped
            return
        self._start_detection(request[0], request[1], live=True, generations=request[2:])

    def toggle_playback(self):
        if self.media_kind != "video" or self.current_frame is None:
            return
        playing = self.play_button.text() != "Pause"
        self.play_button.setText("Pause" if playing else "Play")
        self.set_playing_requested.emit(playing)

    def seek_selected_frame(self):
        self.set_playing_requested.emit(False)
        self.play_button.setText("Play")
        self._media_generation += 1
        self._live_pending = None
        self._infer_after_seek = self.last_records is not None
        self.seek_requested.emit(self.frame_slider.value(), self._video_generation)

    def _clear_stale_result(self, message):
        self.frame_results = []
        self.last_records = None
        self.last_summary = ""
        self.result_display.set_image(None)
        self.result_display.setText(message)
        self._populate_detection_list([], empty_message=message)
        self.record_button.setEnabled(False)
        self.results_summary.setText("Run YOLO on the selected frame. No result is recorded until inference succeeds.")

    def clear_selection(self):
        inference_worker = self.worker
        if inference_worker and inference_worker.isRunning():
            inference_worker.cancel()
            inference_worker.wait()
        self.worker = None
        if inference_worker:
            inference_worker.deleteLater()
        self._media_generation += 1
        self._video_generation += 1
        for reader in tuple(self._image_readers):
            if reader.isRunning():
                reader.requestInterruption()
                reader.wait()
        self._image_readers.clear()
        self.image_reader = None
        self.set_playing_requested.emit(False)
        self.close_video_requested.emit()
        self._video_generation += 1
        self.media_path = ""
        self.media_kind = ""
        self._infer_after_seek = False
        self.last_records = None
        self.last_summary = ""
        self.frame_results = []
        self.current_frame = None
        self.current_frame_number = None
        self.video_fps = 0.0
        self.video_frame_count = 0
        self.original_display.set_image(None)
        self.original_display.set_upload_enabled(True, "image or video")
        self.result_display.set_image(None)
        self.result_display.setText("Actual model predictions appear here.")
        self._populate_detection_list([])
        self.results_summary.setText("Predictions cover model-supported classes only; detections do not identify people or prove an event.")
        self.run_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.play_button.setText("Play")
        self.play_button.setEnabled(False)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_position.setText("Select a video to use playback controls.")
        self.open_button.setEnabled(True)
        self.status.setText("Choose an image or video. Pause or seek to a frame before running YOLO.")

    def run_detection(self):
        if not self.media_path or self.current_frame is None:
            self.status.setText("Choose a decodable image or video frame before running YOLO.")
            return
        if self.worker and self.worker.isRunning():
            return
        self._start_detection(self.current_frame, self.current_frame_number, live=False)

    def _start_detection(self, frame, frame_number, live=False, generations=None):
        if not self.media_path or frame is None or (self.worker and self.worker.isRunning()):
            return
        if live:
            if generations != (self._video_generation, self._media_generation) or not self.live_checkbox.isChecked():
                return
        else:
            self.set_playing_requested.emit(False)
            self.play_button.setText("Play")
            self.open_button.setEnabled(False)
            self.run_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self.record_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.frame_slider.setEnabled(False)
            self.restart_button.setEnabled(False)
            self.confidence_slider.setEnabled(False)
        where = f"frame {frame_number:,}" if self.media_kind == "video" and frame_number is not None else "image"
        if not live:
            self.status.setText(f"{Path(self.media_path).name} · running YOLO on {where}…")
            self.results_summary.setText("Running actual local-model inference… Previous successful output is retained until this run succeeds.")
        self.worker = DetectionWorker(
            self.media_path,
            frame,
            media_kind=self.media_kind,
            frame_number=frame_number,
            fps=self.video_fps,
            confidence=self.confidence_slider.value() / 100.0,
            parent=self,
        )
        self.worker.progress.connect(self.status.setText)
        self.worker.completed.connect(self.on_detection_complete)
        self.worker.failed.connect(self.on_detection_failed)
        self.worker.live_request = bool(live)
        self.worker.request_generations = generations or (self._video_generation, self._media_generation)
        self.worker.start()

    def on_detection_complete(self, frames, note):
        inference_worker = self.sender()
        if not self.media_path or inference_worker is not self.worker:
            return
        if not frames:
            self.on_detection_failed("YOLO completed without returning a result frame.")
            return
        self.frame_results = frames
        self.last_summary = note
        selected = frames[0]
        live_request = bool(getattr(inference_worker, "live_request", False))
        if (getattr(inference_worker, "request_generations", None) != (self._video_generation, self._media_generation)
                or (live_request and not self.live_checkbox.isChecked())):
            self.worker = None
            inference_worker.deleteLater()
            self._start_next_live_request()
            return
        self.last_records = selected["records"]
        if not live_request:
            self.current_frame = selected["original"].copy()
            self.current_frame_number = selected.get("frame_number")
            if self.media_kind == "video" and self.current_frame_number is not None:
                self._update_video_position(self.current_frame_number)
        self.original_display.set_image(self.cv_to_pixmap(selected["original"]))
        self.result_display.set_image(self.cv_to_pixmap(selected["annotated"]))
        self._populate_detection_list(self.last_records, selected.get("frame_number"))
        frame_no = selected.get("frame_number")
        if live_request and frame_no is not None:
            timestamp = frame_no / self.video_fps if self.video_fps > 0 else None
            stamp = f" · {timestamp:.2f}s" if timestamp is not None else ""
            self.results_summary.setText(f"Live result · frame {frame_no:,}{stamp} · {note}")
        else:
            self.results_summary.setText(note)
        self.status.setText(f"{Path(self.media_path).name} · {'live result' if live_request else 'inference complete'} · frame {selected.get('frame_number')}")
        self.open_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.play_button.setEnabled(self.media_kind == "video" and self.current_frame is not None)
        self.restart_button.setEnabled(self.media_kind == "video" and self.current_frame is not None)
        self.frame_slider.setEnabled(self.media_kind == "video" and self.video_frame_count > 1)
        self.confidence_slider.setEnabled(True)
        self.worker = None
        inference_worker.deleteLater()
        self._start_next_live_request()

    def _start_next_live_request(self):
        pending = self._live_pending
        self._live_pending = None
        if pending is not None and self.live_checkbox.isChecked():
            frame, number, video_generation, media_generation = pending
            self._start_detection(frame, number, live=True, generations=(video_generation, media_generation))

    def on_detection_failed(self, message):
        inference_worker = self.sender()
        if not self.media_path or inference_worker is not self.worker:
            return
        live_request = bool(getattr(inference_worker, "live_request", False))
        if getattr(inference_worker, "request_generations", None) != (self._video_generation, self._media_generation):
            self.worker = None
            inference_worker.deleteLater()
            self._start_next_live_request()
            return
        if not live_request:
            self.status.setText(f"Detection could not run: {message}")
        self.open_button.setEnabled(True)
        self.run_button.setEnabled(self.current_frame is not None)
        self.clear_button.setEnabled(True)
        self.record_button.setEnabled(bool(self.frame_results))
        self.play_button.setEnabled(self.media_kind == "video" and self.current_frame is not None)
        self.restart_button.setEnabled(self.media_kind == "video" and self.current_frame is not None)
        self.frame_slider.setEnabled(self.media_kind == "video" and self.video_frame_count > 1)
        self.confidence_slider.setEnabled(True)
        self.results_summary.setText("The previous successful result, if any, is retained. Check the selected file, model weights, and installed dependencies.")
        self.worker = None
        inference_worker.deleteLater()
        if not live_request:
            self.log_message.emit(message, "error")
        self._start_next_live_request()

    def _rerun_for_confidence(self):
        if self.media_path and self.current_frame is not None and not (self.worker and self.worker.isRunning()):
            self.run_detection()

    def record_finding(self):
        if not self.frame_results or self.last_records is None or not self.media_path:
            return
        selected = self.frame_results[0]
        frame_number = selected.get("frame_number")
        fps = float(selected.get("fps", 0.0))
        timestamp_seconds = frame_number / fps if frame_number is not None and fps > 0 else None
        clip_time = f"{timestamp_seconds:.2f}s" if timestamp_seconds is not None else ""
        labels = ", ".join(f"{item['label']} ({item['confidence']:.0%})" for item in self.last_records)
        if not labels:
            labels = "No supported objects detected in the selected frame"
        suffix = f"_frame_{frame_number}" if frame_number is not None else "_image"
        try:
            output_path = save_image(2, f"{Path(self.media_path).stem}{suffix}_yolo.png", selected["annotated"])
        except (OSError, ValueError) as exc:
            self.log_message.emit(f"Could not save YOLO output: {exc}", "error")
            return
        registry.add(EvidenceEntry(
            module="Level 2: The Missing Phone",
            kind="detection",
            value={
                "file": Path(self.media_path).name,
                "media_kind": self.media_kind,
                "frame_number": frame_number,
                "timestamp_seconds": timestamp_seconds,
                "fps": fps or None,
                "confidence_threshold": self.confidence_slider.value() / 100.0,
                "detections": self.last_records,
                "video_summary": self.last_summary,
            },
            description=f"YOLO result for {Path(self.media_path).name}: {labels}",
            technique=f"Pretrained Ultralytics YOLO object detection at confidence ≥{self.confidence_slider.value()}%",
            observation=f"{self.frame_position.text()} · {labels}. {self.last_summary}",
            interpretation="Model predictions describe supported visual classes; they do not identify a person or establish ownership.",
            uncertainty="Medium",
            source_filename=Path(self.media_path).name,
            source_path=str(Path(self.media_path).resolve()),
            output_path=output_path,
            source_timestamp=clip_time,
            frame_number=frame_number,
            evidence_id="E-02",
            unresolved_question="Which visible objects does this model miss or label inconsistently across the selected frames?",
            limitation="Predictions cover model-supported classes only. A missed detection does not prove absence; boxes do not track object identity across frames or identify people.",
        ))
        self.level_completed.emit("Object Detection")
        self.log_message.emit(f"Actual YOLO result recorded · {output_path}", "success")

    def shutdown_media(self):
        for reader in tuple(self._image_readers):
            if reader.isRunning():
                reader.requestInterruption()
                reader.wait()
        self._image_readers.clear()
        if self.playback_thread.isRunning():
            self.shutdown_requested.emit()
            self.playback_thread.wait(3000)

    def reset(self):
        self.clear_selection()
