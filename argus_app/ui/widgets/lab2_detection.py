"""Level 2: actual pretrained YOLO inference on one image or video."""

from pathlib import Path
from threading import Lock

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QSplitter, QVBoxLayout

from evidence.artifacts import save_image
from core.input_source import read_image_file
from evidence.registry import EvidenceEntry, registry
from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".dib", ".webp", ".tif", ".tiff", ".ppm", ".pgm", ".pnm", ".jp2"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"}
_MODEL_LOCK = Lock()
_MODEL_CACHE = {}


class DetectionWorker(QThread):
    """Run CPU YOLO inference away from the GUI thread."""

    progress = Signal(str)
    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(self, media_path: str, confidence=0.25, parent=None):
        super().__init__(parent)
        self.media_path = str(media_path)
        self.confidence = float(confidence)

    @staticmethod
    def _weights_path() -> str:
        import os

        configured = os.environ.get("ARGUS_YOLO_WEIGHTS", "").strip()
        if configured:
            return str(Path(configured).expanduser())
        project_root = Path(__file__).resolve().parents[3]
        custom = project_root / "models" / "custom" / "best.pt"
        bundled = project_root / "yolo11n.pt"
        if custom.is_file():
            return str(custom)
        if bundled.is_file():
            return str(bundled)
        return ""

    def run(self):
        try:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics is not installed. Run setup.ps1 or install requirements.txt.") from exc

            media = Path(self.media_path)
            suffix = media.suffix.lower()
            if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
                raise RuntimeError("Level 2 accepts supported image and video files only.")
            if not media.is_file():
                raise RuntimeError(f"The selected evidence no longer exists: {media}")
            weights = self._weights_path()
            if not weights:
                raise RuntimeError("YOLO weights are missing. Add yolo11n.pt or set ARGUS_YOLO_WEIGHTS to a local .pt checkpoint.")
            if not Path(weights).expanduser().is_file():
                raise RuntimeError(f"YOLO weights were not found: {weights}")

            cache_key = str(Path(weights).expanduser().resolve())
            self.progress.emit(f"Loading {Path(weights).name}…")
            with _MODEL_LOCK:
                model = _MODEL_CACHE.get(cache_key)
                if model is None:
                    model = YOLO(weights)
                    _MODEL_CACHE[cache_key] = model

            if suffix in IMAGE_SUFFIXES:
                frame = read_image_file(media)
                if frame is None:
                    raise RuntimeError("OpenCV could not decode this image.")
                self.progress.emit("Running YOLO on the selected image…")
                result = model.predict(frame, imgsz=640, conf=self.confidence, device="cpu", verbose=False)[0]
                records = self._records(result)
                frames = [{"frame_number": None, "original": frame, "annotated": result.plot(), "records": records}]
                summary = f"Image inference · {len(records)} supported detection(s). Unlisted objects could not be verified by this model."
                self.completed.emit(frames, summary)
                return

            self.progress.emit("Running YOLO on video frames…")
            capture = cv2.VideoCapture(str(media))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError("OpenCV could not open this video.")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            stride = max(1, int(round(fps * 0.5)) if fps > 0 else 15)
            if declared_count > 0:
                stride = max(stride, (declared_count + 35) // 36)
            frames, class_frame_hits = [], {}
            evaluated = detection_total = 0
            while True:
                if self.isInterruptionRequested():
                    capture.release()
                    return
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number = evaluated
                result = model.predict(frame, imgsz=640, conf=self.confidence, device="cpu", verbose=False)[0]
                records = self._records(result)
                evaluated += 1
                detection_total += len(records)
                for label in {record["label"] for record in records}:
                    class_frame_hits[label] = class_frame_hits.get(label, 0) + 1
                if frame_number % stride == 0 and len(frames) < 35:
                    frames.append({"frame_number": frame_number, "original": frame.copy(), "annotated": result.plot(), "records": records})
                if evaluated % 30 == 0:
                    self.progress.emit(f"YOLO evaluated {evaluated:,} of {declared_count or 'unknown'} frames…")
            capture.release()
            if evaluated == 0:
                raise RuntimeError("YOLO/OpenCV could not decode any frames from this video.")
            # Preserve the final actual frame as a useful endpoint when it was
            # not already sampled; memory remains bounded by the stride above.
            if frame_number % stride != 0 and frames and len(frames) < 36:
                frames.append({"frame_number": frame_number, "original": frame.copy(), "annotated": result.plot(), "records": self._records(result)})
            class_notes = []
            for name, hits in sorted(class_frame_hits.items()):
                consistency = "in every frame" if hits == evaluated else f"intermittently ({hits}/{evaluated} frames)"
                class_notes.append(f"{name}: {consistency}")
            detected = "; ".join(class_notes) if class_notes else "no supported class was detected"
            summary = (
                f"Video inference · {evaluated:,} frames evaluated · {detection_total:,} detections. "
                f"Detected classes: {detected}. Unlisted classes and object identity could not be verified."
            )
            frames = frames[:36]
            for item in frames:
                item["fps"] = fps
                item["evaluated_frames"] = evaluated
                item["class_frame_hits"] = class_frame_hits
            self.completed.emit(frames, summary)
        except Exception as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def _records(result):
        names = result.names
        records = []
        if result.boxes is None:
            return records
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            records.append({"label": str(label), "confidence": confidence, "box": (x1, y1, x2, y2)})
        return records


class Lab2DetectionWidget(LabWidget):
    """Focused image/video detector with original and actual annotated output."""

    level_completed = Signal(str)
    clear_requested = Signal()
    evidence_requested = Signal(str)

    def __init__(self, parent=None):
        self.media_path = ""
        self.media_kind = ""
        self.worker = None
        self.last_records = None
        self.last_summary = ""
        self.frame_results = []
        self.current_frame_index = 0
        super().__init__("Object Detection", parent)

    def setup_ui(self):
        super().setup_ui()
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.open_button = QPushButton("Open image or video…")
        self.open_button.clicked.connect(self.choose_media)
        controls.addWidget(self.open_button)
        self.run_button = QPushButton("Run YOLO")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_detection)
        controls.addWidget(self.run_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        views = QSplitter(Qt.Horizontal)
        views.setChildrenCollapsible(False)
        self.original_display = ImageDisplayLabel("Select an image or video to begin.")
        self.original_display.set_upload_enabled(True, "image")
        self.result_display = ImageDisplayLabel("Actual model predictions appear here.")
        views.addWidget(self._panel("SOURCE FRAME", self.original_display))
        views.addWidget(self._panel("YOLO OUTPUT", self.result_display))
        views.setStretchFactor(0, 1)
        views.setStretchFactor(1, 1)
        views.setSizes([1, 1])
        layout.addWidget(views, 1)

        self.frame_navigation = QHBoxLayout()
        self.previous_frame = QPushButton("Previous frame")
        self.previous_frame.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() - 1))
        self.frame_navigation.addWidget(self.previous_frame)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.show_selected_frame)
        self.frame_navigation.addWidget(self.frame_slider, 1)
        self.frame_position = QLabel("No inference frames")
        self.frame_navigation.addWidget(self.frame_position)
        self.next_frame = QPushButton("Next frame")
        self.next_frame.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() + 1))
        self.frame_navigation.addWidget(self.next_frame)
        layout.addLayout(self.frame_navigation)
        self._set_frame_navigation_enabled(False)

        self.status = QLabel("Choose one image or video, then run the local pretrained model.")
        self.status.setWordWrap(True)
        self.status.setObjectName("pageSummary")
        layout.addWidget(self.status)
        self.results_summary = QLabel("Boxes and labels come from actual model inference; unsupported objects may be missed.")
        self.results_summary.setWordWrap(True)
        self.results_summary.setObjectName("mutedLabel")
        layout.addWidget(self.results_summary)
        self.record_button = QPushButton("Record result in notebook")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.record_finding)
        layout.addWidget(self.record_button, 0, Qt.AlignRight)

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

    def choose_media(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image or video evidence",
            "",
            "Images and videos (*.png *.jpg *.jpeg *.jfif *.bmp *.dib *.webp *.tif *.tiff *.ppm *.pgm *.pnm *.jp2 *.mp4 *.mov *.avi *.mkv *.webm *.m4v *.mpeg *.mpg *.wmv)",
        )
        if path:
            self.evidence_requested.emit(path)

    def load_media(self, path: str, auto_detect=True):
        if self.worker and self.worker.isRunning():
            self.status.setText("Wait for the current YOLO run to finish before replacing evidence.")
            return False
        media = Path(path)
        suffix = media.suffix.lower()
        if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            self.status.setText("Level 2 accepts supported image and video files only.")
            return False
        if suffix in IMAGE_SUFFIXES:
            frame = read_image_file(media)
        else:
            capture = cv2.VideoCapture(str(media))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                frame = None
        if frame is None:
            self.status.setText("OpenCV could not decode the selected evidence. Choose another image or video.")
            return False

        self.media_path = str(media)
        self.media_kind = "video" if suffix in VIDEO_SUFFIXES else "image"
        self.last_records = None
        self.last_summary = ""
        self.frame_results = []
        self.current_frame_index = 0
        height, width = frame.shape[:2]
        self.original_display.set_image(self.cv_to_pixmap(frame))
        self.original_display.set_upload_enabled(True, "image")
        self.result_display.set_image(None)
        self.result_display.setText("Running YOLO…" if auto_detect else "Run YOLO to inspect model predictions.")
        self.results_summary.setText("Inference uses the selected evidence and local model weights.")
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.record_button.setEnabled(False)
        self._set_frame_navigation_enabled(False)
        self.frame_position.setText("No inference frames")
        self.status.setText(f"{media.name} · {width} × {height} · {self.media_kind} selected")
        if auto_detect:
            self.run_detection()
        return True

    def clear_selection(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        self.media_path = ""
        self.media_kind = ""
        self.last_records = None
        self.last_summary = ""
        self.frame_results = []
        self.current_frame_index = 0
        self.original_display.set_image(None)
        self.original_display.set_upload_enabled(True, "image")
        self.result_display.set_image(None)
        self.result_display.setText("Actual model predictions appear here.")
        self.results_summary.setText("Boxes and labels come from actual model inference; unsupported objects may be missed.")
        self.run_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self._set_frame_navigation_enabled(False)
        self.open_button.setEnabled(True)
        self.status.setText("Choose one image or video, then run the local pretrained model.")

    def run_detection(self):
        if not self.media_path:
            self.status.setText("Choose an image or video before running YOLO.")
            return
        if self.worker and self.worker.isRunning():
            return
        self.frame_results = []
        self.last_records = None
        self._set_frame_navigation_enabled(False)
        self.frame_position.setText("No inference frames")
        self.results_summary.setText("Running actual model inference…")
        self.result_display.set_image(None)
        self.result_display.setText("Running pretrained YOLO…")
        self.open_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self._set_frame_navigation_enabled(False)
        self.frame_position.setText("No inference frames")
        self.status.setText(f"{Path(self.media_path).name} · CPU inference…")
        self.worker = DetectionWorker(self.media_path, parent=self)
        self.worker.progress.connect(self.status.setText)
        self.worker.completed.connect(self.on_detection_complete)
        self.worker.failed.connect(self.on_detection_failed)
        self.worker.start()

    def _set_frame_navigation_enabled(self, enabled):
        for control in (self.frame_slider, self.previous_frame, self.next_frame):
            control.setEnabled(bool(enabled))

    def on_detection_complete(self, frames, note):
        if not self.media_path or self.sender() is not self.worker:
            return
        self.frame_results = frames or []
        self.last_summary = note
        self.current_frame_index = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, len(self.frame_results) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._set_frame_navigation_enabled(len(self.frame_results) > 1)
        self.show_selected_frame(0)
        self.status.setText(f"{Path(self.media_path).name} · inference complete")
        self.open_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.record_button.setEnabled(bool(self.frame_results))

    def show_selected_frame(self, index):
        if not self.frame_results:
            self.last_records = None
            return
        self.current_frame_index = max(0, min(int(index), len(self.frame_results) - 1))
        item = self.frame_results[self.current_frame_index]
        self.last_records = item["records"]
        self.original_display.set_image(self.cv_to_pixmap(item["original"]))
        self.result_display.set_image(self.cv_to_pixmap(item["annotated"]))
        if item.get("frame_number") is None:
            frame_note = "Image"
        else:
            fps = float(item.get("fps", 0.0))
            seconds = item["frame_number"] / fps if fps > 0 else None
            timestamp = f" · clip {seconds:.2f}s" if seconds is not None else ""
            frame_note = f"Frame {item['frame_number']:,}{timestamp} · view {self.current_frame_index + 1}/{len(self.frame_results)}"
        self.frame_position.setText(frame_note)
        labels = ", ".join(f"{entry['label']} {entry['confidence']:.0%}" for entry in self.last_records[:8])
        visible = f"Displayed frame: {labels or 'no supported detections'}"
        self.results_summary.setText(f"{self.last_summary}\n{visible}")

    def on_detection_failed(self, message):
        if not self.media_path or self.sender() is not self.worker:
            return
        self.last_records = None
        self.frame_results = []
        self.status.setText(f"Detection could not run: {message}")
        self.result_display.setText("Detection unavailable. The original evidence remains selected.")
        self.open_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.record_button.setEnabled(False)
        self.results_summary.setText("Check the selected file, model weights, and installed dependencies.")
        self.log_message.emit(message, "error")

    def record_finding(self):
        if self.last_records is None or not self.media_path:
            return
        labels = ", ".join(f"{item['label']} ({item['confidence']:.0%})" for item in self.last_records)
        if not labels:
            labels = "No supported objects detected in the displayed frame"
        selected = self.frame_results[self.current_frame_index]
        frame_number = selected.get("frame_number")
        try:
            output_path = save_image(2, f"{Path(self.media_path).stem}_frame_{frame_number if frame_number is not None else 'image'}_yolo.png", selected["annotated"])
        except (OSError, ValueError) as exc:
            self.log_message.emit(f"Could not save YOLO output: {exc}", "error")
            return
        fps = float(selected.get("fps", 0.0))
        clip_time = f"{frame_number / fps:.2f}s" if frame_number is not None and fps > 0 else ""
        registry.add(EvidenceEntry(
            module="Level 2: Object Detection",
            kind="detection",
            value={"file": Path(self.media_path).name, "media_kind": self.media_kind, "frame_number": frame_number, "detections": self.last_records, "video_summary": self.last_summary},
            description=f"YOLO result for {Path(self.media_path).name}: {labels or 'no supported detections in this frame'}",
            technique="Pretrained Ultralytics YOLO object detection",
            observation=f"{self.frame_position.text()} · {labels or 'no supported detections in this frame'}. {self.last_summary}",
            interpretation="Model predictions describe supported visual classes; they do not identify a person or establish ownership.",
            uncertainty="Medium",
            source_filename=Path(self.media_path).name,
            source_path=str(Path(self.media_path).resolve()), output_path=output_path,
            source_timestamp=clip_time, frame_number=frame_number,
            evidence_id="E-02",
            unresolved_question="Which visible objects does this model miss or label inconsistently across the selected frames?",
            limitation="Predictions cover model-supported classes only. A missed detection does not prove absence; boxes do not track object identity across frames or identify people.",
        ))
        self.level_completed.emit("Object Detection")
        self.log_message.emit(f"YOLO finding and actual annotated frame recorded · {output_path}", "success")

    def reset(self):
        self.clear_selection()
