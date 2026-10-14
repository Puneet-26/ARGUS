"""Level 2: pretrained YOLO object detection for one selected image."""

from pathlib import Path
from threading import Lock

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.input_source import read_image_file
from evidence.registry import EvidenceEntry, registry
from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".dib", ".webp", ".tif", ".tiff", ".ppm", ".pgm", ".pnm", ".jp2"}
COMMON_CLASSES = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "backpack", "umbrella", "handbag", "suitcase", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "toothbrush",
}
_MODEL_LOCK = Lock()
_MODEL_CACHE = {}


class DetectionWorker(QThread):
    """Run CPU inference away from the GUI thread."""

    progress = Signal(str)
    completed = Signal(object, object, object, str)
    failed = Signal(str)

    def __init__(self, image_path: str, confidence=0.25, class_filter="All classes", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.confidence = float(confidence)
        self.class_filter = class_filter

    @staticmethod
    def _weights_path() -> str:
        import os

        configured = os.environ.get("ARGUS_YOLO_WEIGHTS", "").strip()
        if configured:
            return str(Path(configured).expanduser())
        project_root = Path(__file__).resolve().parents[3]
        custom = project_root / "models" / "custom" / "best.pt"
        if custom.is_file():
            return str(custom)
        bundled = project_root / "yolo11n.pt"
        if bundled.is_file():
            return str(bundled)
        return "yolo11n.pt"

    def run(self):
        try:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is not installed. Run setup.ps1 or install requirements.txt."
                ) from exc

            image_path = Path(self.image_path)
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise RuntimeError("Level 2 accepts images only. Choose PNG, JPEG, BMP, WebP, TIFF, or another supported image.")
            if not image_path.is_file():
                raise RuntimeError(f"The selected image no longer exists: {image_path}")

            weights = self._weights_path()
            if weights != "yolo11n.pt" and not Path(weights).expanduser().is_file():
                raise RuntimeError(
                    f"Configured YOLO weights were not found: {weights}. Set ARGUS_YOLO_WEIGHTS to an existing .pt file, "
                    "or place fine-tuned weights at models/custom/best.pt."
                )
            self.progress.emit(f"Loading pretrained weights ({Path(weights).name})…")
            cache_key = str(Path(weights).expanduser().resolve()) if weights != "yolo11n.pt" else weights
            with _MODEL_LOCK:
                model = _MODEL_CACHE.get(cache_key)
                if model is None:
                    try:
                        model = YOLO(weights)
                    except Exception as exc:
                        raise RuntimeError(f"Could not load or download YOLO weights '{weights}': {exc}") from exc
                    _MODEL_CACHE[cache_key] = model

            frame = read_image_file(image_path)
            if frame is None:
                raise RuntimeError("OpenCV could not decode this image. Try PNG, JPEG, BMP, or WebP.")
            class_ids = self._class_ids(model.names, self.class_filter)
            if class_ids == []:
                raise RuntimeError(f"The loaded model does not include classes for the '{self.class_filter}' filter.")
            self.progress.emit("Detecting objects in the selected image…")
            result = model.predict(
                frame,
                imgsz=640,
                conf=self.confidence,
                classes=class_ids,
                device="cpu",
                verbose=False,
            )[0]
            records = self._records(result)
            self.completed.emit(frame, result.plot(), records, f"{len(records)} object(s) detected in the image.")
        except Exception as exc:
            message = str(exc)
            if "yolo11n.pt" in message or "best.pt" in message or "download" in message.lower():
                message = f"Could not load or download YOLO weights. Check internet access for first use or ARGUS_YOLO_WEIGHTS. Details: {message}"
            self.failed.emit(message)

    def _class_ids(self, names, class_filter):
        if class_filter == "All classes":
            return None
        entries = names.items() if isinstance(names, dict) else enumerate(names)
        ids = []
        for class_id, label in entries:
            normalized = str(label).strip().lower()
            if class_filter == "People" and normalized in {"person", "people", "human"}:
                ids.append(int(class_id))
            elif class_filter == "Common objects" and normalized in COMMON_CLASSES:
                ids.append(int(class_id))
        return ids

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
    """Single-image object detection workspace; a selected image is analyzed immediately."""

    level_completed = Signal(str)
    clear_requested = Signal()
    evidence_requested = Signal(str)

    def __init__(self, parent=None):
        self.media_path = ""
        self.worker = None
        self.last_records = None
        self._rerun_pending = False
        super().__init__("Object Detection", parent)

    def setup_ui(self):
        super().setup_ui()
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.open_button = QPushButton("Open Image…")
        self.open_button.clicked.connect(self.choose_image)
        controls.addWidget(self.open_button)
        self.run_button = QPushButton("Run YOLO Detection")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_detection)
        controls.addWidget(self.run_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        controls.addWidget(self.clear_button)

        controls.addSpacing(12)
        controls.addWidget(QLabel("Confidence"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 90)
        self.confidence_slider.setValue(25)
        self.confidence_slider.setFixedWidth(130)
        controls.addWidget(self.confidence_slider)
        self.confidence_label = QLabel("25%")
        self.confidence_label.setMinimumWidth(38)
        controls.addWidget(self.confidence_label)

        controls.addWidget(QLabel("Classes"))
        self.class_filter = QComboBox()
        self.class_filter.addItems(["All classes", "People", "Common objects"])
        controls.addWidget(self.class_filter)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(350)
        self._settings_timer.timeout.connect(self.run_detection)
        self.confidence_slider.valueChanged.connect(self._on_detection_setting_changed)
        self.class_filter.currentTextChanged.connect(self._on_detection_setting_changed)
        self.confidence_slider.valueChanged.connect(lambda value: self.confidence_label.setText(f"{value}%"))

        briefing = QLabel(
            "ARGUS has recovered CCTV evidence from Lab 4B. Object detection predicts boxes around visible objects; each box shows the model's class and confidence score. "
            "Pretrained weights already include common classes such as people, cups, and backpacks. Results can miss small or unclear objects, and a confidence score is not certainty. "
            "A custom class needs reviewed bounding-box annotations and optional fine-tuning. YOLO class labels do not identify a real person."
        )
        briefing.setWordWrap(True)
        briefing.setObjectName("infoCard")
        layout.addWidget(briefing)

        views = QSplitter(Qt.Horizontal)
        views.setChildrenCollapsible(False)
        self.original_display = ImageDisplayLabel("Upload an image to begin.")
        self.original_display.set_upload_enabled(True, "image")
        self.result_display = ImageDisplayLabel("Annotated detections appear here.")
        views.addWidget(self._panel("ORIGINAL EVIDENCE", self.original_display))
        views.addWidget(self._panel("YOLO DETECTIONS", self.result_display))
        views.setStretchFactor(0, 1)
        views.setStretchFactor(1, 1)
        layout.addWidget(views, 1)

        self.status = QLabel("Upload an image to run pretrained YOLO.")
        self.status.setWordWrap(True)
        self.status.setObjectName("pageSummary")
        layout.addWidget(self.status)

        self.results_summary = QLabel("No results yet.")
        self.results_summary.setObjectName("mutedLabel")
        layout.addWidget(self.results_summary)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(["Class", "Confidence", "x₁", "y₁", "x₂", "y₂"])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.setMaximumHeight(170)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setVisible(False)
        layout.addWidget(self.results_table)

        self.record_button = QPushButton("Record detections in notebook")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.record_finding)
        layout.addWidget(self.record_button, 0, Qt.AlignRight)

    @staticmethod
    def _panel(title, display):
        panel = QFrame()
        panel.setStyleSheet(f"QFrame {{ background-color: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: {RADIUS['sm']}; }}")
        box = QVBoxLayout(panel)
        box.setContentsMargins(8, 8, 8, 8)
        label = QLabel(title)
        label.setStyleSheet(f"color: {COLORS['text_muted']}; font-family: {FONTS['family_mono']}; font-size: {FONTS['size_xs']};")
        box.addWidget(label)
        box.addWidget(display, 1)
        return panel

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CCTV image",
            "",
            "Images (*.png *.jpg *.jpeg *.jfif *.bmp *.dib *.webp *.tif *.tiff *.ppm *.pgm *.pnm *.jp2)",
        )
        if path:
            self.evidence_requested.emit(path)

    def load_media(self, path: str, auto_detect=True):
        if self.worker and self.worker.isRunning():
            self.status.setText("Wait for detection to finish before replacing the selected image.")
            return False
        image_path = Path(path)
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            self.status.setText("Level 2 accepts images only. Choose a supported image file.")
            return False
        frame = read_image_file(image_path)
        if frame is None:
            self.status.setText("OpenCV could not decode the image. Choose a readable PNG, JPEG, BMP, WebP, TIFF, or PNM file.")
            return False

        self.media_path = str(image_path)
        self.last_records = None
        height, width = frame.shape[:2]
        self.original_display.set_image(self.cv_to_pixmap(frame))
        self.original_display.set_upload_enabled(True, "image")
        self.result_display.set_image(None)
        self.result_display.setText("Running detection…" if auto_detect else "Run detection to see the model output.")
        self.results_summary.setText("No results yet.")
        self.results_table.setRowCount(0)
        self.results_table.setVisible(False)
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.record_button.setEnabled(False)
        self.status.setText(f"{image_path.name} · {width} × {height} · ready to detect")
        if auto_detect:
            self.run_detection()
        return True

    def clear_selection(self):
        self._settings_timer.stop()
        self._rerun_pending = False
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        self.media_path = ""
        self.last_records = None
        self.original_display.set_image(None)
        self.original_display.set_upload_enabled(True, "image")
        self.result_display.set_image(None)
        self.result_display.setText("Annotated detections appear here.")
        self.results_table.setRowCount(0)
        self.results_table.setVisible(False)
        self.results_summary.setText("No results yet.")
        self.run_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.open_button.setEnabled(True)
        self.status.setText("Upload an image to run pretrained YOLO.")

    def _on_detection_setting_changed(self, *_args):
        self.last_records = None
        self.record_button.setEnabled(False)
        self.result_display.set_image(None)
        if self.media_path:
            self.status.setText("Detection settings changed · rerunning YOLO…")
            self._settings_timer.start()

    def run_detection(self):
        if not self.media_path:
            self.status.setText("Upload one image before running detection.")
            return
        if self.worker and self.worker.isRunning():
            self._rerun_pending = True
            return
        self.results_table.setRowCount(0)
        self.results_table.setVisible(False)
        self.results_summary.setText("Running detection…")
        self.results_summary.setVisible(True)
        self.result_display.set_image(None)
        self.result_display.setText("Running pretrained YOLO…")
        self.open_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.status.setText(f"{Path(self.media_path).name} · running CPU inference…")
        self.worker = DetectionWorker(
            self.media_path,
            self.confidence_slider.value() / 100.0,
            self.class_filter.currentText(),
            self,
        )
        self.worker.progress.connect(self.status.setText)
        self.worker.completed.connect(self.on_detection_complete)
        self.worker.failed.connect(self.on_detection_failed)
        self.worker.finished.connect(self._start_pending_rerun)
        self.worker.start()

    def _start_pending_rerun(self):
        if self._rerun_pending and self.media_path:
            self._rerun_pending = False
            QTimer.singleShot(0, self.run_detection)

    def _show_records(self, records):
        self.results_table.setRowCount(len(records))
        has_results = bool(records)
        self.results_table.setVisible(has_results)
        self.results_summary.setVisible(not has_results)
        self.results_summary.setText(
            f"{len(records)} detection(s) in this image." if has_results
            else "No supported objects detected in this image."
        )
        for row, record in enumerate(records):
            values = [record["label"], f"{record['confidence']:.1%}", *map(str, record["box"])]
            for column, value in enumerate(values):
                self.results_table.setItem(row, column, QTableWidgetItem(value))

    def on_detection_complete(self, original, annotated, records, note):
        if not self.media_path or self.sender() is not self.worker:
            return
        self.last_records = records
        self.original_display.set_image(self.cv_to_pixmap(original))
        self.result_display.set_image(self.cv_to_pixmap(annotated))
        self._show_records(records)
        self.status.setText(f"{Path(self.media_path).name} · {note}")
        self.open_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.record_button.setEnabled(True)

    def on_detection_failed(self, message):
        if not self.media_path or self.sender() is not self.worker:
            return
        self.last_records = None
        self.status.setText(f"Detection could not run: {message}")
        self.result_display.setText("Detection unavailable")
        self.open_button.setEnabled(True)
        self.run_button.setEnabled(bool(self.media_path))
        self.clear_button.setEnabled(bool(self.media_path))
        self.record_button.setEnabled(False)
        self.results_summary.setVisible(True)
        self.results_summary.setText("Detection unavailable. The original evidence is still selected.")
        self.log_message.emit(message, "error")

    def record_finding(self):
        if self.last_records is None or not self.media_path:
            return
        labels = ", ".join(f"{item['label']} ({item['confidence']:.0%})" for item in self.last_records)
        if not labels:
            labels = "No supported objects detected"
        registry.add(EvidenceEntry(
            module="Level 2: Object Detection",
            kind="detection",
            value={"file": Path(self.media_path).name, "detections": self.last_records},
            description=f"YOLO result for {Path(self.media_path).name}: {labels}.",
            technique="Pretrained Ultralytics YOLO object detection",
            observation=f"{len(self.last_records)} object(s) detected: {labels}.",
            interpretation="Model detections cover supported visual classes; they do not identify a person or establish ownership.",
            uncertainty="Medium",
            source_filename=Path(self.media_path).name,
        ))
        self.level_completed.emit("Object Detection")
        self.log_message.emit("YOLO result recorded in the Evidence Notebook.", "success")

    def reset(self):
        self.clear_selection()
