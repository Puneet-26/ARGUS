"""Native PySide6 shell for the ARGUS investigation application."""

from pathlib import Path
from threading import Lock

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core import FPSCounter, InputSource, downscale_frame, read_image_file
from ui.style import COLORS, FONTS, MAIN_STYLE
from ui.widgets.dashboard import DashboardWidget
from ui.widgets.evidence_notebook import EvidenceNotebookWidget
from ui.widgets.final_report import FinalReportWidget
from ui.widgets.lab1_image import Lab1ImageWidget
from ui.widgets.lab2_detection import Lab2DetectionWidget
from ui.widgets.lab3_colour import Lab3ColourWidget
from ui.widgets.lab4_motion import Lab4MotionWidget
from ui.widgets.settings import SettingsWidget


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".dib", ".webp", ".tif", ".tiff", ".ppm", ".pgm", ".pnm", ".jp2"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"}


class ProcessingWorker(QThread):
    """Read Level 4 video frames without blocking the interface."""

    frame_ready = Signal(object, float, int, int, int)
    error_occurred = Signal(str)
    playback_finished = Signal()

    def __init__(self):
        super().__init__()
        self.input_source = InputSource()
        self.fps_counter = FPSCounter()
        self.running = False
        self.processing_enabled = True
        self.max_dimension = 1280
        self._source_lock = Lock()
        self._at_end = False
        self._current_frame = None
        self._current_frame_number = 0
        self._current_frame_count = 1
        self._source_generation = 0

    def open_source(self, source_type, source_value):
        try:
            with self._source_lock:
                self._source_generation += 1
                if source_type != "video":
                    raise ValueError("The playback worker accepts video evidence only.")
                self.input_source.open_video(source_value)
            self.fps_counter.reset()
            self.processing_enabled = True
            self._at_end = False
            self._current_frame = None
            self._current_frame_number = 0
            self._current_frame_count = 1
            return True
        except Exception as exc:
            self.error_occurred.emit(f"Could not open evidence: {exc}")
            return False

    def run(self):
        while self.running:
            if not self.processing_enabled:
                self.msleep(50)
                continue
            try:
                with self._source_lock:
                    if not self.input_source.is_opened():
                        ok, frame, source_kind = False, None, None
                        frame_number, frame_count = 0, 1
                    else:
                        source_kind = self.input_source.source_type
                        ok, frame = self.input_source.read()
                        frame_number = self.input_source.frame_position
                        frame_count = self.input_source.frame_count
                    source_generation = self._source_generation
                if not ok or frame is None:
                    if (
                        source_kind is not None
                        and source_kind.value == "video"
                        and not self._at_end
                        and source_generation == self._source_generation
                    ):
                        self._at_end = True
                        self.processing_enabled = False
                        self.playback_finished.emit()
                    self.msleep(30)
                    continue
                frame = downscale_frame(frame, self.max_dimension)
                with self._source_lock:
                    if source_generation != self._source_generation:
                        continue
                    self._current_frame = frame.copy()
                    self._current_frame_number = frame_number
                    self._current_frame_count = frame_count
                self.fps_counter.tick()
                self.frame_ready.emit(frame, self.fps_counter.get_fps(), frame_number, frame_count, source_generation)
                source_fps = self.input_source.fps
                delay = max(1, min(200, round(1000 / source_fps))) if source_fps > 0 else 33
                self.msleep(delay)
            except Exception as exc:
                self.error_occurred.emit(f"Frame processing failed: {exc}")
                self.msleep(100)

    def set_playing(self, playing):
        if playing and self._at_end:
            self.seek_frame(0)
            self._at_end = False
        self.processing_enabled = bool(playing)

    def seek_frame(self, frame_number):
        try:
            with self._source_lock:
                if not self.input_source.is_opened() or self.input_source.source_type.value != "video":
                    return
                ok, frame = self.input_source.seek_frame(frame_number)
                current = self.input_source.frame_position
                total = self.input_source.frame_count
                source_generation = self._source_generation
            if ok and frame is not None:
                frame = downscale_frame(frame, self.max_dimension)
                self.frame_ready.emit(frame, self.fps_counter.get_fps(), current, total, source_generation)
                self._at_end = False
        except Exception as exc:
            self.error_occurred.emit(f"Could not seek video: {exc}")

    def get_current_frame(self):
        with self._source_lock:
            if self._current_frame is None:
                return None, 0, 1
            return self._current_frame.copy(), self._current_frame_number, self._current_frame_count

    def stop(self):
        self.running = False
        with self._source_lock:
            self.input_source.release()


class MainWindow(QMainWindow):
    """Single-window casefile with responsive navigation and evidence workspaces."""

    PAGE_LABELS = [
        "Case",
        "Eyes Online",
        "Detection",
        "Colour ID",
        "Timeline",
        "Notebook",
        "Conclusion",
        "Settings",
    ]
    PAGE_TITLES = {
        1: ("LEVEL 01", "Eyes Online", "Enhance one selected image and inspect details that may be difficult to see."),
        2: ("LEVEL 02", "Object Detection", "Run pretrained YOLO on one selected image."),
        3: ("LEVEL 03", "Colour Identification", "Use HSV masks to inspect colour regions in one selected image."),
        4: ("LEVEL 04", "Timeline and Tracking", "Inspect movement in one selected video and record line-crossing events."),
    }

    def __init__(self):
        super().__init__()
        self.current_lab = None
        self.worker = None
        self.current_evidence_path = None
        self.evidence_paths = {1: None, 2: None, 3: None, 4: None}
        self.level_upload_buttons = {}
        self.level_clear_buttons = {}
        self.lab_widgets = {
            "Eyes Online": Lab1ImageWidget(),
            "Object Detection": Lab2DetectionWidget(),
            "The Red Bag": Lab3ColourWidget(),
            "The Final Clue": Lab4MotionWidget(),
        }
        self.page_widgets = {}
        self.nav_buttons = []
        self.setup_ui()
        self.select_page(0)
        self.setup_worker()

        for lab in self.lab_widgets.values():
            if hasattr(lab, "level_completed"):
                lab.level_completed.connect(self.on_level_completed)
        self.lab_widgets["The Final Clue"].play_pause_requested.connect(self.toggle_playback)
        self.lab_widgets["The Final Clue"].seek_requested.connect(self.worker.seek_frame)
        self.lab_widgets["Object Detection"].clear_requested.connect(lambda: self.clear_evidence(2))
        self.lab_widgets["Object Detection"].evidence_requested.connect(lambda path: self.open_evidence_path(path, 2))
        self.notebook_widget.entries_changed.connect(self.on_notebook_changed)
        for level, name in enumerate(("Eyes Online", "Object Detection", "The Red Bag", "The Final Clue"), start=1):
            lab = self.lab_widgets[name]
            display = getattr(lab, "canvas_orig", None) if level == 1 else getattr(lab, "original_display", None)
            if display is not None and hasattr(display, "upload_requested"):
                display.upload_requested.connect(lambda level=level: self.choose_evidence(level))
                display.file_dropped.connect(lambda path, level=level: self.open_evidence_path(path, level))
        for widget in [*self.lab_widgets.values(), self.notebook_widget, self.report_widget, self.settings_widget]:
            widget.log_message.connect(self.show_status_message)

    def setup_ui(self):
        self.setWindowTitle("ARGUS · The Lab 4B Incident")
        self.setMinimumSize(1000, 680)
        self.resize(1360, 860)
        self.setAcceptDrops(True)
        self.setStyleSheet(MAIN_STYLE)

        central = QWidget()
        central.setAcceptDrops(True)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.create_top_bar())

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.dashboard_scroll = QScrollArea()
        self.dashboard_scroll.setWidgetResizable(True)
        self.dashboard_scroll.setFrameShape(QFrame.NoFrame)
        self.dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.dashboard = DashboardWidget()
        self.dashboard_scroll.setWidget(self._page_container(self.dashboard))
        self.dashboard.level_selected.connect(self.on_level_selected_from_dashboard)
        self.dashboard.notebook_requested.connect(lambda: self.select_page(5))
        self.stacked_widget.addWidget(self.dashboard_scroll)

        names = ["Eyes Online", "Object Detection", "The Red Bag", "The Final Clue"]
        for level, name in enumerate(names, start=1):
            lab = self.lab_widgets[name]
            page = self.create_level_page(level, lab)
            self.page_widgets[name] = page
            self.stacked_widget.addWidget(page)

        self.notebook_widget = EvidenceNotebookWidget()
        self.stacked_widget.addWidget(self._simple_page(self.notebook_widget))
        self.report_widget = FinalReportWidget()
        self.stacked_widget.addWidget(self._simple_page(self.report_widget))
        self.settings_widget = SettingsWidget()
        self.stacked_widget.addWidget(self._simple_page(self.settings_widget))

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(20, 12, 20, 8)
        workspace_layout.setSpacing(0)
        workspace_layout.addWidget(self.stacked_widget, 1)
        root.addWidget(workspace, 1)

        self.status_line = QLabel("Ready · Open evidence or choose an investigation level.")
        self.status_line.setObjectName("statusLine")
        self.status_line.setWordWrap(True)
        self.status_line.setMinimumHeight(30)
        self.status_line.setContentsMargins(20, 4, 20, 6)
        root.addWidget(self.status_line)

    @staticmethod
    def _page_container(widget):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget, 1)
        return container

    def _simple_page(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._page_container(widget))
        return scroll

    def create_level_page(self, level: int, lab_widget: QWidget) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page = QWidget()
        page.setMinimumWidth(0)
        scroll.setWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        index, title, summary = self.PAGE_TITLES[level]
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 2, 0, 8)
        header_layout.setSpacing(3)
        index_label = QLabel(index)
        index_label.setStyleSheet(
            f"color: {COLORS['amber']}; font-family: {FONTS['family_mono']}; font-size: 12px; letter-spacing: 1px;"
        )
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        summary_label = QLabel(summary)
        summary_label.setObjectName("pageSummary")
        summary_label.setWordWrap(True)
        header_layout.addWidget(index_label)
        header_layout.addWidget(title_label)
        header_layout.addWidget(summary_label)
        layout.addWidget(header)
        if level != 2:
            actions = QHBoxLayout()
            self.level_upload_buttons[level] = QPushButton(
                "Open Video Evidence…" if level == 4 else "Open Image Evidence…"
            )
            self.level_upload_buttons[level].setObjectName("primaryButton")
            self.level_upload_buttons[level].clicked.connect(lambda _checked=False, item=level: self.choose_evidence(item))
            actions.addWidget(self.level_upload_buttons[level])
            self.level_clear_buttons[level] = QPushButton("Clear Evidence")
            self.level_clear_buttons[level].clicked.connect(lambda _checked=False, item=level: self.clear_evidence(item))
            self.level_clear_buttons[level].setEnabled(False)
            actions.addWidget(self.level_clear_buttons[level])
            actions.addStretch(1)
            layout.addLayout(actions)
        layout.addWidget(lab_widget, 1)
        return scroll

    def create_top_bar(self):
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar.setMinimumHeight(58)
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand_name = QLabel("ARGUS")
        brand_name.setObjectName("brandName")
        brand_meta = QLabel("LAB 4B CASEFILE")
        brand_meta.setObjectName("brandMeta")
        brand.addWidget(brand_name)
        brand.addWidget(brand_meta)
        layout.addLayout(brand)

        nav = QWidget()
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(1)
        for index, label in enumerate(self.PAGE_LABELS):
            button = QPushButton(label)
            button.setObjectName("caseNavButton")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            button.clicked.connect(lambda _checked=False, page=index: self.select_page(page))
            nav_layout.addWidget(button, 1)
            self.nav_buttons.append(button)
        self.nav_buttons[0].setChecked(True)
        layout.addWidget(nav, 1)

        return top_bar

    def setup_worker(self):
        self.worker = ProcessingWorker()
        self.worker.max_dimension = self.settings_widget.dim_spin.value()
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.error_occurred.connect(lambda message: self.show_status_message(message, "error"))
        self.worker.playback_finished.connect(self.on_playback_finished)
        self.settings_widget.frame_dimension_changed.connect(self.set_frame_dimension)
        self.settings_widget.case_reset.connect(self.on_case_reset)
        self.worker.running = True
        self.worker.start()

    def set_frame_dimension(self, dimension):
        if self.worker:
            self.worker.max_dimension = int(dimension)

    def select_page(self, index: int):
        if not 0 <= index < len(self.PAGE_LABELS):
            return
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        self.on_sidebar_changed(index)

    def on_sidebar_changed(self, index: int):
        self.stacked_widget.setCurrentIndex(index)
        if 1 <= index <= 4:
            name = ["Eyes Online", "Object Detection", "The Red Bag", "The Final Clue"][index - 1]
            self.current_lab = self.lab_widgets[name]
            self.dashboard.set_active_stage(name)
            self.dashboard.set_level_status(name, "in_progress")
            if name == "The Final Clue" and self.worker and self.worker.input_source.is_opened():
                self.worker.set_playing(True)
                self.lab_widgets[name].set_playing(True)
            elif self.worker and self.worker.input_source.source_type is not None:
                self.worker.set_playing(False)
                self.lab_widgets["The Final Clue"].set_playing(False)
        else:
            self.current_lab = None
            if self.worker and self.worker.input_source.source_type is not None:
                self.worker.set_playing(False)
                self.lab_widgets["The Final Clue"].set_playing(False)
            if index == 0:
                self.dashboard.refresh_notebook_summary()
            elif index == 5:
                filename = Path(self.current_evidence_path).name if self.current_evidence_path else ""
                source_time = ""
                if self.worker and self.worker.input_source.source_type and self.worker.input_source.source_type.value == "video":
                    _, frame_number, _ = self.worker.get_current_frame()
                    fps = self.worker.input_source.fps
                    if fps > 0:
                        seconds = int(frame_number / fps)
                        source_time = f"{seconds // 60:02d}:{seconds % 60:02d}"
                self.notebook_widget.set_source_metadata(filename, source_time)
                self.notebook_widget.refresh_table()
                self.dashboard.refresh_notebook_summary()
            elif index == 6:
                self.report_widget.generate_report()

    def choose_evidence(self, level=None):
        if level is None:
            index = self.stacked_widget.currentIndex()
            level = index if 1 <= index <= 4 else None
        if level not in (1, 2, 3, 4):
            self.show_status_message("Open an investigation level before selecting evidence.", "warning")
            return
        if level == 4:
            title = "Select video evidence"
            file_filter = "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.mpeg *.mpg *.wmv)"
        else:
            title = "Select image evidence"
            file_filter = "Images (*.png *.jpg *.jpeg *.jfif *.bmp *.dib *.webp *.tif *.tiff *.ppm *.pgm *.pnm *.jp2)"
        path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if path:
            self.open_evidence_path(path, level)

    def open_evidence_path(self, path, level=None):
        if level is None:
            index = self.stacked_widget.currentIndex()
            level = index if 1 <= index <= 4 else None
        if level not in (1, 2, 3, 4):
            self.show_status_message("Choose an investigation level, then open its evidence.", "warning")
            return False

        suffix = Path(path).suffix.lower()
        required_suffixes = VIDEO_SUFFIXES if level == 4 else IMAGE_SUFFIXES
        kind = "video" if level == 4 else "image"
        if suffix not in required_suffixes:
            expected = "video" if level == 4 else "image"
            self.show_status_message(f"Level {level} accepts {expected} files only. Choose a supported {expected} format.", "error")
            return False

        if level == 1:
            frame = read_image_file(path)
            if frame is None:
                self.show_status_message("OpenCV could not decode this image. Choose another image file.", "error")
                return False
            lab = self.lab_widgets["Eyes Online"]
            lab.reset()
            lab.set_evidence_name(Path(path).name)
            loaded = lab.load_frame(frame)
        elif level == 2:
            lab = self.lab_widgets["Object Detection"]
            loaded = lab.load_media(path, auto_detect=True)
        elif level == 3:
            frame = read_image_file(path)
            if frame is None:
                self.show_status_message("OpenCV could not decode this image. Choose another image file.", "error")
                return False
            lab = self.lab_widgets["The Red Bag"]
            lab.reset()
            lab.set_evidence_name(Path(path).name)
            loaded = lab.load_evidence_frame(frame)
        else:
            candidate = InputSource()
            try:
                candidate.open_video(path)
                readable, _first_frame = candidate.read()
                if not readable:
                    raise RuntimeError("The selected video contains no readable frames.")
            except Exception as exc:
                candidate.release()
                self.show_status_message(f"Could not decode {Path(path).name}: {exc}", "error")
                return False
            candidate.release()
            lab = self.lab_widgets["The Final Clue"]
            lab.clear_media()
            if not self.worker.open_source(kind, path):
                self.show_status_message(f"Could not open video {Path(path).name}. Check its codec and file integrity.", "error")
                return False
            source = self.worker.input_source
            lab.set_media_info(Path(path).name, kind, source.frame_count, source.fps)
            loaded = True

        if not loaded:
            self.show_status_message(f"Could not load {Path(path).name}. Check that it is a valid {kind} file.", "error")
            return False
        self.evidence_paths[level] = str(path)
        self.current_evidence_path = str(path)
        self.notebook_widget.set_source_metadata(Path(path).name)
        if level in self.level_clear_buttons:
            self.level_clear_buttons[level].setEnabled(True)
        self.show_status_message(f"Loaded {Path(path).name} for Level {level}.", "success")
        return True

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        local_files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if local_files:
            index = self.stacked_widget.currentIndex()
            level = index if 1 <= index <= 4 else None
            self.open_evidence_path(local_files[0], level)
            event.acceptProposedAction()
        else:
            event.ignore()

    def on_level_selected_from_dashboard(self, level_name: str):
        level_map = {"Eyes Online": 1, "Object Detection": 2, "The Red Bag": 3, "The Final Clue": 4}
        if level_name in level_map:
            self.select_page(level_map[level_name])

    def on_level_completed(self, level_name: str):
        self.dashboard.sync_progress_from_registry()
        self.notebook_widget.refresh_table()
        self.show_status_message(f"{level_name} finding recorded in the Evidence Notebook.", "success")

    def on_notebook_changed(self):
        self.dashboard.sync_progress_from_registry()
        if self.stacked_widget.currentIndex() == 6:
            self.report_widget.generate_report()

    def on_case_reset(self):
        self.dashboard.reset_progress()
        self.notebook_widget.refresh_table()
        self.report_widget.generate_report()

    def on_frame_ready(self, frame, fps, frame_number, frame_count, source_generation):
        if self.worker:
            with self.worker._source_lock:
                if source_generation != self.worker._source_generation:
                    return
                self.worker._current_frame = frame.copy()
                self.worker._current_frame_number = frame_number
                self.worker._current_frame_count = frame_count
        if self.current_lab is self.lab_widgets["The Final Clue"] and frame is not None:
            try:
                kind = self.worker.input_source.source_type.value if self.worker and self.worker.input_source.source_type else ""
                if kind != "video":
                    return
                if hasattr(self.current_lab, "set_frame_position"):
                    self.current_lab.set_frame_position(frame_number, frame_count)
                self.current_lab.process_frame(frame)
            except Exception as exc:
                self.show_status_message(f"Image processing error: {exc}", "error")

    def toggle_playback(self):
        if self.current_lab is not self.lab_widgets["The Final Clue"] or not self.worker or not self.worker.input_source.is_opened():
            self.show_status_message("Open video evidence in Level 4 before using playback controls.", "warning")
            return
        playing = not self.worker.processing_enabled
        self.worker.set_playing(playing)
        self.lab_widgets["The Final Clue"].set_playing(playing)

    def on_playback_finished(self):
        self.lab_widgets["The Final Clue"].set_playing(False)
        self.show_status_message("Video reached the end.", "info")

    def clear_evidence(self, level=None):
        if level is None:
            index = self.stacked_widget.currentIndex()
            level = index if 1 <= index <= 4 else None
        if level not in (1, 2, 3, 4):
            return
        if level == 1:
            self.lab_widgets["Eyes Online"].reset()
        elif level == 2:
            self.lab_widgets["Object Detection"].clear_selection()
        elif level == 3:
            self.lab_widgets["The Red Bag"].reset()
        else:
            if self.worker:
                self.worker.set_playing(False)
                with self.worker._source_lock:
                    self.worker._source_generation += 1
                    self.worker.input_source.release()
                    self.worker._current_frame = None
                    self.worker._current_frame_number = 0
                    self.worker._current_frame_count = 1
            self.lab_widgets["The Final Clue"].clear_media()
        self.evidence_paths[level] = None
        self.current_evidence_path = next((path for path in reversed(list(self.evidence_paths.values())) if path), None)
        if level in self.level_clear_buttons:
            self.level_clear_buttons[level].setEnabled(False)
        if self.current_evidence_path:
            self.notebook_widget.set_source_metadata(Path(self.current_evidence_path).name)
        else:
            self.notebook_widget.set_source_metadata()
        self.show_status_message(f"Level {level} evidence cleared.", "info")

    def show_status_message(self, message, level="info"):
        color = {
            "info": COLORS["text_secondary"],
            "success": COLORS["success"],
            "warning": COLORS["amber"],
            "error": COLORS["alert"],
        }.get(level, COLORS["text_secondary"])
        self.status_line.setStyleSheet(
            f"color: {color}; padding: 4px 20px 6px 20px; border-top: 1px solid {COLORS['border']};"
        )
        self.status_line.setText(str(message))

    def closeEvent(self, event):
        for lab in self.lab_widgets.values():
            inference_worker = getattr(lab, "worker", None)
            if inference_worker and inference_worker.isRunning():
                inference_worker.requestInterruption()
                inference_worker.wait()
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()
