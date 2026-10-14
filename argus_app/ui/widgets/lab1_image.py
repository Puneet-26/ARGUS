"""Level 1: inspect CCTV evidence and enhance it with direct OpenCV controls."""

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QBoxLayout,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from evidence.registry import EvidenceEntry, registry
from ui.style import COLORS, FONTS, RADIUS
from ui.widgets.lab_base import ImageDisplayLabel, LabWidget


class _ForensicImageLabel(ImageDisplayLabel):
    """Image label that supports fit-to-view and exact-size rendering."""

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.fit_to_view = True
        self.canvas_owner = None
        self.setMinimumSize(1, 1)
        self.setStyleSheet(
            f"QLabel {{ background: {COLORS['canvas_bg']}; border: none; "
            f"color: {COLORS['text_muted']}; font-family: {FONTS['family_mono']}; "
            f"font-size: {FONTS['size_sm']}; }}"
        )

    def _update_scaled_pixmap(self):
        if not self._source_pixmap or self._source_pixmap.isNull():
            return
        self.setText("")
        if self.fit_to_view:
            self.setPixmap(
                self._source_pixmap.scaled(
                    max(1, self.width() - 8),
                    max(1, self.height() - 8),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self.setPixmap(self._source_pixmap)

    def mousePressEvent(self, event):
        if (
            self.canvas_owner
            and self.canvas_owner.scale_mode == "actual"
            and self._source_pixmap is not None
            and event.button() == Qt.LeftButton
        ):
            self.canvas_owner._begin_pan(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.canvas_owner and self.canvas_owner._pan_position is not None:
            self.canvas_owner._move_pan(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.canvas_owner and self.canvas_owner._pan_position is not None:
            self.canvas_owner._end_pan()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ForensicImageCanvas(QScrollArea):
    """Scrollable image viewport with hidden-bar panning at exact pixel size."""

    upload_requested = Signal()
    file_dropped = Signal(str)

    def __init__(self, title="No Evidence Loaded", parent=None, bordered=True):
        super().__init__(parent)
        self.title = title
        self.scale_mode = "fit"
        self.bordered = bordered
        self._pan_position = None
        self.setObjectName("forensicImageCanvas")
        self.setMinimumSize(140, 100)
        self.setFrameShape(QFrame.NoFrame)
        self.setAlignment(Qt.AlignCenter)
        self.setWidgetResizable(True)
        self.setAcceptDrops(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        border = f"1px solid {COLORS['border']}" if bordered else "none"
        self.setStyleSheet(
            f"QScrollArea#forensicImageCanvas {{ background: {COLORS['canvas_bg']}; "
            f"border: {border}; border-radius: {RADIUS['sm']}; }}"
        )
        self.viewport().setStyleSheet(f"background: {COLORS['canvas_bg']};")
        self.viewport().setAcceptDrops(True)
        self.display_label = _ForensicImageLabel(title)
        self.display_label.canvas_owner = self
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.upload_requested.connect(self.upload_requested.emit)
        self.display_label.file_dropped.connect(self.file_dropped.emit)
        self.setWidget(self.display_label)

    def set_upload_enabled(self, enabled=True, media_kind="image"):
        self.display_label.set_upload_enabled(enabled, media_kind)

    def set_cv_image(self, image):
        self.set_image(LabWidget.cv_to_pixmap(image))

    def set_image(self, pixmap):
        self.display_label.set_image(pixmap)
        self._sync_scale_mode()

    def setText(self, text):
        self.display_label.setText(text)

    def set_scale_mode(self, mode):
        self.scale_mode = "actual" if mode == "actual" else "fit"
        self.display_label.fit_to_view = self.scale_mode == "fit"
        self._sync_scale_mode()

    def _sync_scale_mode(self):
        source = self.display_label._source_pixmap
        if self.scale_mode == "actual" and source is not None and not source.isNull():
            self.setWidgetResizable(False)
            self.display_label.setMinimumSize(source.size())
            self.display_label.setMaximumSize(source.size())
            self.display_label.resize(source.size())
            self.display_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.display_label.setCursor(Qt.OpenHandCursor)
        else:
            self.setWidgetResizable(True)
            self.display_label.setMinimumSize(1, 1)
            self.display_label.setMaximumSize(16777215, 16777215)
            self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.display_label.resize(self.viewport().size())
            self.display_label.setCursor(
                Qt.PointingHandCursor
                if self.display_label._upload_enabled and source is None
                else Qt.ArrowCursor
            )
        self.display_label._update_scaled_pixmap()

    def _begin_pan(self, event):
        self._pan_position = event.globalPosition().toPoint()
        self.display_label.setCursor(Qt.ClosedHandCursor)

    def _move_pan(self, event):
        position = event.globalPosition().toPoint()
        delta = position - self._pan_position
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        self._pan_position = position

    def _end_pan(self):
        self._pan_position = None
        self.display_label.setCursor(Qt.OpenHandCursor)

    def dragEnterEvent(self, event):
        if (
            self.display_label._upload_enabled
            and event.mimeData().hasUrls()
            and any(url.isLocalFile() for url in event.mimeData().urls())
        ):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        if not self.display_label._upload_enabled:
            event.ignore()
            return
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.file_dropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
        event.ignore()


class ResponsiveFindingRow(QWidget):
    """Keep the short observation form horizontal until its available width is tight."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("lab1FindingRow")
        self.setStyleSheet("QWidget#lab1FindingRow { background: transparent; border: none; }")
        self.row_layout = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.row_layout.setContentsMargins(0, 0, 0, 0)
        self.row_layout.setSpacing(8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        direction = (
            QBoxLayout.LeftToRight
            if event.size().width() >= 350
            else QBoxLayout.TopToBottom
        )
        if self.row_layout.direction() != direction:
            self.row_layout.setDirection(direction)


class Lab1ImageWidget(LabWidget):
    """Non-code image enhancement workflow for recovered Lab 4B footage."""

    level_completed = Signal(str)

    def __init__(self, parent=None):
        self.raw_evidence_frame = None
        self.enhanced_frame = None
        self.threshold_frame = None
        self.pipeline_images = {}
        self.finding_recorded = False
        self.evidence_name = ""
        self._is_processing = False
        self._last_saved_observation = ""
        super().__init__("Eyes Online", parent)

    def setup_ui(self):
        super().setup_ui()
        page = QVBoxLayout(self.content_widget)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(10)

        workspace = QSplitter(Qt.Horizontal)
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(6)
        evidence_panel = self._create_evidence_panel()
        controls_panel = self._create_investigation_panel()
        evidence_panel.setMinimumWidth(500)
        controls_panel.setMinimumWidth(300)
        workspace.addWidget(evidence_panel)
        workspace.addWidget(controls_panel)
        workspace.setStretchFactor(0, 7)
        workspace.setStretchFactor(1, 3)
        workspace.setSizes([700, 300])
        page.addWidget(workspace, 1)

    def _panel(self, margins=(12, 10, 12, 10)):
        panel = QFrame()
        panel.setObjectName("lab1Panel")
        panel.setStyleSheet(
            f"QFrame#lab1Panel {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: {RADIUS['md']}; }}"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(*margins)
        layout.setSpacing(9)
        return panel, layout

    @staticmethod
    def _section_label(text):
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {COLORS['amber']}; font-family: {FONTS['family_mono']}; "
            f"font-size: {FONTS['size_xs']}; letter-spacing: 1px;"
        )
        return label

    def _create_evidence_panel(self):
        panel, layout = self._panel((10, 8, 10, 8))
        header = QHBoxLayout()
        header.addWidget(self._section_label("EVIDENCE VIEWER · LAB 4B"))
        header.addStretch(1)
        self.scale_button_group = QButtonGroup(self)
        self.scale_button_group.setExclusive(True)
        self.fit_view_button = QPushButton("Fit to View")
        self.fit_view_button.setCheckable(True)
        self.fit_view_button.setChecked(True)
        self.fit_view_button.setToolTip("Scale each image proportionally to fit its viewer.")
        self.actual_size_button = QPushButton("100%")
        self.actual_size_button.setCheckable(True)
        self.actual_size_button.setToolTip(
            "Show original pixels. Drag an image to pan when it is larger than its viewer."
        )
        self.scale_button_group.addButton(self.fit_view_button)
        self.scale_button_group.addButton(self.actual_size_button)
        for button in (self.fit_view_button, self.actual_size_button):
            button.setMinimumHeight(30)
            button.setStyleSheet(
                f"QPushButton {{ background: {COLORS['elevated']}; color: {COLORS['text_secondary']}; "
                f"border: 1px solid {COLORS['border']}; border-radius: {RADIUS['md']}; padding: 4px 9px; }}"
                f"QPushButton:hover {{ color: {COLORS['text_main']}; border-color: {COLORS['primary']}; }}"
                f"QPushButton:checked {{ background: {COLORS['surface']}; color: {COLORS['text_main']}; "
                f"border-color: {COLORS['primary']}; font-weight: 600; }}"
            )
        self.fit_view_button.clicked.connect(lambda: self._set_view_scale("fit"))
        self.actual_size_button.clicked.connect(lambda: self._set_view_scale("actual"))
        header.addWidget(self.fit_view_button)
        header.addWidget(self.actual_size_button)
        layout.addLayout(header)

        viewer_splitter = QSplitter(Qt.Vertical)
        viewer_splitter.setChildrenCollapsible(False)
        viewer_splitter.setHandleWidth(4)
        primary_viewers = QWidget()
        primary_viewers.setObjectName("lab1PrimaryViewers")
        primary_viewers.setStyleSheet(
            "QWidget#lab1PrimaryViewers { background: transparent; border: none; }"
        )
        displays = QHBoxLayout()
        displays.setContentsMargins(0, 0, 0, 0)
        displays.setSpacing(8)
        self.canvas_orig, self.original_caption = self._labeled_canvas(
            displays, "Original Evidence", "Upload an image to begin."
        )
        self.canvas_proc, self.result_caption = self._labeled_canvas(
            displays, "Enhanced Result", "Apply enhancement to view the result."
        )
        primary_viewers.setLayout(displays)
        primary_viewers.setMinimumHeight(190)
        viewer_splitter.addWidget(primary_viewers)

        stage_section = QWidget()
        stage_section.setObjectName("lab1StageSection")
        stage_section.setStyleSheet(
            "QWidget#lab1StageSection { background: transparent; border: none; }"
        )
        stage_layout = QVBoxLayout(stage_section)
        stage_layout.setContentsMargins(0, 4, 0, 0)
        stage_layout.setSpacing(6)
        stage_layout.addWidget(self._section_label("PROCESSING STAGES"))
        self.stage_tabs = QTabWidget()
        self.stage_tabs.setDocumentMode(True)
        self.stage_tabs.setMinimumHeight(125)
        self.stage_canvases = {}
        for name in ("Grayscale", "CLAHE", "Denoised", "Threshold"):
            canvas = ForensicImageCanvas(f"Apply enhancement to generate the {name} stage.", bordered=False)
            canvas.setMinimumSize(140, 90)
            self.stage_canvases[name] = canvas
            self.stage_tabs.addTab(canvas, name)
        stage_layout.addWidget(self.stage_tabs, 1)
        viewer_splitter.addWidget(stage_section)
        viewer_splitter.setStretchFactor(0, 3)
        viewer_splitter.setStretchFactor(1, 2)
        viewer_splitter.setSizes([380, 220])
        viewer_splitter.setMinimumHeight(340)
        layout.addWidget(viewer_splitter, 1)

        self._set_view_scale("fit")

        actions = QHBoxLayout()
        self.save_button = QPushButton("Save Enhanced…")
        self.save_button.clicked.connect(self.save_enhanced_image)
        self.save_button.setEnabled(False)
        actions.addWidget(self.save_button)
        crop_button = QPushButton("Export Region Crop…")
        crop_button.clicked.connect(self.save_region_crop)
        crop_button.setEnabled(False)
        self.crop_button = crop_button
        actions.addWidget(crop_button)
        actions.addStretch(1)
        footer = QVBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(4)
        self.meta_summary_label = QLabel("No evidence selected")
        self.meta_summary_label.setWordWrap(True)
        self.meta_summary_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-family: {FONTS['family_mono']}; "
            f"font-size: {FONTS['size_xs']};"
        )
        footer.addWidget(self.meta_summary_label)
        footer.addLayout(actions)
        layout.addLayout(footer)
        return panel

    def _labeled_canvas(self, layout, title, empty_text):
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(5)
        caption = QLabel(title)
        caption.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {FONTS['size_xs']};")
        column.addWidget(caption)
        canvas = ForensicImageCanvas(empty_text)
        canvas.setMinimumSize(180, 150)
        if title.lower() == "original evidence":
            canvas.set_upload_enabled(True, "image")
        column.addWidget(canvas, 1)
        layout.addLayout(column, 1)
        return canvas, caption

    def _set_view_scale(self, mode):
        self.view_scale_mode = "actual" if mode == "actual" else "fit"
        for canvas in (
            getattr(self, "canvas_orig", None),
            getattr(self, "canvas_proc", None),
            *getattr(self, "stage_canvases", {}).values(),
        ):
            if canvas is not None:
                canvas.set_scale_mode(self.view_scale_mode)

    def _create_investigation_panel(self):
        panel, panel_layout = self._panel((10, 9, 10, 9))
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)

        directive = QVBoxLayout()
        directive.setContentsMargins(2, 1, 2, 1)
        directive.setSpacing(5)
        directive.addWidget(self._section_label("INVESTIGATION DIRECTIVE"))
        story = QLabel(
            "Enhance the selected CCTV image to reveal details that may be difficult to see in the original footage. "
            "Processing improves visibility, but does not create new evidence or identify a person."
        )
        story.setWordWrap(True)
        story.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_sm']};"
        )
        directive.addWidget(story)
        layout.addLayout(directive)
        layout.addWidget(self._section_separator())

        controls = QFrame()
        controls.setObjectName("lab1EnhancementControls")
        controls.setStyleSheet(
            f"QFrame#lab1EnhancementControls {{ background: {COLORS['elevated']}; "
            f"border: 1px solid {COLORS['border']}; "
            f"border-radius: {RADIUS['md']}; }}"
        )
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 10, 12, 11)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(self._section_label("IMAGE ENHANCEMENT"))
        instruction = QLabel(
            "Choose operations, adjust active settings, apply the enhancement, then inspect the result and stages."
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_sm']};")
        controls_layout.addWidget(instruction)

        self.grayscale_check = QCheckBox("Grayscale")
        self.grayscale_check.setChecked(True)
        self.grayscale_check.setToolTip("Convert pixel brightness to grayscale for easier inspection.")
        self.clahe_check = QCheckBox("CLAHE contrast")
        self.clahe_check.setChecked(True)
        self.clahe_check.setToolTip("Improve local contrast in small image regions.")
        self.denoise_check = QCheckBox("Reduce noise")
        self.denoise_check.setChecked(True)
        self.denoise_check.setToolTip("Use a light blur to reduce fine sensor noise; small details may soften.")
        self.threshold_check = QCheckBox("Threshold detail")
        self.threshold_check.setToolTip("Create a black-and-white mask using the threshold level below.")
        option_layout = QVBoxLayout()
        option_layout.setContentsMargins(0, 1, 0, 1)
        option_layout.setSpacing(2)
        for checkbox in (self.grayscale_check, self.clahe_check, self.denoise_check, self.threshold_check):
            checkbox.setMinimumHeight(27)
            checkbox.setStyleSheet(
                f"QCheckBox {{ color: {COLORS['text_main']}; background: transparent; spacing: 9px; }}"
                f"QCheckBox:hover {{ color: {COLORS['text_main']}; background: {COLORS['surface']}; }}"
                f"QCheckBox:focus {{ color: {COLORS['text_main']}; border: 1px solid {COLORS['primary']}; "
                f"border-radius: {RADIUS['sm']}; }}"
                f"QCheckBox:disabled {{ color: {COLORS['text_muted']}; }}"
            )
            checkbox.toggled.connect(self._mark_enhancement_pending)
            option_layout.addWidget(checkbox)
        controls_layout.addLayout(option_layout)

        self.clip_slider, self.clip_value, self.clahe_parameter_row = self._slider_row(
            controls_layout,
            "CLAHE strength",
            "Higher values increase local contrast.",
            10,
            60,
            30,
            lambda value: f"{value / 10:.1f}",
        )
        self.threshold_slider, self.threshold_value, self.threshold_parameter_row = self._slider_row(
            controls_layout,
            "Threshold level",
            "Pixels brighter than this level appear white.",
            0,
            255,
            90,
            str,
        )
        self.clip_slider.valueChanged.connect(self._mark_enhancement_pending)
        self.threshold_slider.valueChanged.connect(self._mark_enhancement_pending)
        self.clip_slider.valueChanged.connect(lambda v: self.clip_value.setText(f"{v / 10:.1f}"))
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_value.setText(str(v)))

        self.enhance_button = QPushButton("Apply Enhancement")
        self.enhance_button.setObjectName("primaryButton")
        self.enhance_button.setMinimumHeight(40)
        self.enhance_button.setStyleSheet(
            f"QPushButton {{ background: {COLORS['primary']}; color: #111416; "
            f"border: 1px solid {COLORS['primary']}; border-radius: {RADIUS['md']}; "
            f"padding: 8px 14px; font-weight: 700; }}"
            f"QPushButton:hover {{ background: {COLORS['primary_hover']}; "
            f"border-color: {COLORS['primary_hover']}; }}"
            f"QPushButton:pressed {{ background: {COLORS['primary_pressed']}; }}"
            f"QPushButton:disabled {{ background: {COLORS['surface']}; color: {COLORS['text_muted']}; "
            f"border-color: {COLORS['border']}; }}"
        )
        self.enhance_button.clicked.connect(self.run_enhancement)
        controls_layout.addWidget(self.enhance_button)
        self.enhancement_status_label = QLabel("Upload evidence to begin.")
        self.enhancement_status_label.setWordWrap(True)
        self.enhancement_status_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: {FONTS['size_xs']};"
        )
        controls_layout.addWidget(self.enhancement_status_label)
        layout.addWidget(controls)

        layout.addWidget(self._section_separator())
        finding_layout = QVBoxLayout()
        finding_layout.setContentsMargins(2, 1, 2, 0)
        finding_layout.setSpacing(6)
        finding_layout.addWidget(self._section_label("RECORD YOUR FINDING"))
        question = QLabel("What visible detail did you recover?")
        question.setWordWrap(True)
        question.setStyleSheet(f"color: {COLORS['text_main']}; font-size: {FONTS['size_sm']};")
        finding_layout.addWidget(question)
        entry_row = ResponsiveFindingRow()
        self.finding_form_layout = entry_row.row_layout
        self.finding_input = QLineEdit()
        self.finding_input.setPlaceholderText("e.g. Visible timestamp: 23:41")
        self.finding_input.setMinimumWidth(80)
        self.finding_input.setMinimumHeight(36)
        input_palette = self.finding_input.palette()
        input_palette.setColor(QPalette.PlaceholderText, QColor(COLORS["text_secondary"]))
        self.finding_input.setPalette(input_palette)
        self.finding_input.returnPressed.connect(self.record_finding)
        self.finding_input.textChanged.connect(self._update_observation_controls)
        entry_row.row_layout.addWidget(self.finding_input, 1)
        self.record_button = QPushButton("Record Observation")
        self.record_button.setObjectName("secondaryButton")
        self.record_button.setMinimumHeight(36)
        self.record_button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {COLORS['text_secondary']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: {RADIUS['md']}; padding: 6px 12px; }}"
            f"QPushButton:hover {{ background: {COLORS['surface']}; color: {COLORS['text_main']}; "
            f"border-color: {COLORS['primary']}; }}"
            f"QPushButton:disabled {{ background: {COLORS['surface']}; color: {COLORS['text_muted']}; "
            f"border-color: {COLORS['border']}; }}"
        )
        self.record_button.clicked.connect(self.record_finding)
        entry_row.row_layout.addWidget(self.record_button)
        finding_layout.addWidget(entry_row)
        self.verification_status_label = QLabel("Load evidence and apply enhancement to record an observation.")
        self.verification_status_label.setWordWrap(True)
        self.verification_status_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: {FONTS['size_xs']};"
        )
        finding_layout.addWidget(self.verification_status_label)
        self.saved_observation_label = QLabel()
        self.saved_observation_label.setWordWrap(True)
        self.saved_observation_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_sm']};"
        )
        self.saved_observation_label.hide()
        finding_layout.addWidget(self.saved_observation_label)
        layout.addLayout(finding_layout)

        layout.addStretch(1)
        panel_layout.addLayout(layout, 1)
        self._sync_control_states()
        self._update_observation_controls()
        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("lab1ControlsScroll")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setWidget(panel)
        controls_scroll.setMinimumWidth(300)
        return controls_scroll

    @staticmethod
    def _section_separator():
        separator = QFrame()
        separator.setObjectName("lab1SectionSeparator")
        separator.setFixedHeight(1)
        separator.setStyleSheet(
            f"QFrame#lab1SectionSeparator {{ background: {COLORS['border']}; border: none; }}"
        )
        return separator

    def _slider_row(self, layout, label_text, help_text, minimum, maximum, initial, formatter):
        container = QWidget()
        container.setObjectName("lab1SliderRow")
        container.setStyleSheet("QWidget#lab1SliderRow { background: transparent; border: none; }")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 1, 0, 1)
        container_layout.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        label = QLabel(label_text)
        label.setMinimumWidth(104)
        value_label = QLabel(formatter(initial))
        value_label.setMinimumWidth(38)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setStyleSheet(
            f"color: {COLORS['text_main']}; font-family: {FONTS['family_mono']}; "
            f"font-size: {FONTS['size_sm']};"
        )
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(initial)
        slider.setMinimumWidth(90)
        slider.setMinimumHeight(24)
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        container_layout.addLayout(row)
        help_label = QLabel(help_text)
        help_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {FONTS['size_xs']};")
        container_layout.addWidget(help_label)
        layout.addWidget(container)
        return slider, value_label, container

    def set_evidence_name(self, filename):
        name = str(filename)
        if name != self.evidence_name:
            self._last_saved_observation = ""
            if hasattr(self, "saved_observation_label"):
                self.saved_observation_label.clear()
                self.saved_observation_label.hide()
        self.evidence_name = name

    def _mark_enhancement_pending(self, *_args):
        if self._is_processing:
            return
        if self.raw_evidence_frame is not None:
            self.save_button.setEnabled(False)
            self.crop_button.setEnabled(False)
            self.canvas_proc.set_image(None)
            self.canvas_proc.setText("Settings changed · apply enhancement to refresh.")
            self.enhanced_frame = None
            self.threshold_frame = None
            self.pipeline_images = {}
            for canvas in self.stage_canvases.values():
                canvas.set_image(None)
        self._sync_control_states()
        self._update_observation_controls()

    def _sync_control_states(self):
        if not hasattr(self, "enhance_button"):
            return
        has_frame = self.raw_evidence_frame is not None
        editable = has_frame and not self._is_processing
        for control in (self.grayscale_check, self.clahe_check, self.denoise_check, self.threshold_check):
            control.setEnabled(editable)
        self.clahe_parameter_row.setEnabled(editable and self.clahe_check.isChecked())
        self.threshold_parameter_row.setEnabled(editable and self.threshold_check.isChecked())
        self.enhance_button.setEnabled(editable)

        if self._is_processing:
            status = "Processing selected frame…"
            color = COLORS["amber"]
        elif not has_frame:
            status = "Upload evidence to begin."
            color = COLORS["text_muted"]
        elif self.enhanced_frame is None:
            status = "Ready. Apply enhancement to view the result and processing stages."
            color = COLORS["text_secondary"]
        else:
            status = "Enhancement applied. Inspect the result and processing stages."
            color = COLORS["success"]
        self.enhancement_status_label.setText(status)
        self.enhancement_status_label.setStyleSheet(
            f"color: {color}; font-size: {FONTS['size_xs']};"
        )

    def _update_observation_controls(self, *_args):
        if not hasattr(self, "record_button"):
            return
        has_frame = self.raw_evidence_frame is not None
        has_text = bool(self.finding_input.text().strip())
        can_record = (
            has_frame
            and self.enhanced_frame is not None
            and has_text
            and not self._is_processing
        )
        self.record_button.setEnabled(can_record)

        if self._is_processing:
            text = "Wait for enhancement to finish before recording."
            color = COLORS["text_muted"]
        elif not has_frame:
            text = "Load evidence to record an observation."
            color = COLORS["text_muted"]
        elif self.enhanced_frame is None:
            text = "Apply enhancement before recording an observation."
            color = COLORS["text_muted"]
        elif not has_text and self._last_saved_observation:
            text = "Observation saved to the Evidence Notebook."
            color = COLORS["success"]
        elif not has_text:
            text = "Enter a visible detail to enable recording."
            color = COLORS["text_muted"]
        else:
            text = "Ready to save this observation to the Evidence Notebook."
            color = COLORS["text_secondary"]
        self.verification_status_label.setText(text)
        self.verification_status_label.setStyleSheet(
            f"color: {color}; font-size: {FONTS['size_xs']};"
        )

    def load_frame(self, frame):
        """Display one decoded source image without modifying or processing it."""
        if self._is_processing or frame is None or getattr(frame, "size", 0) == 0:
            return False
        self.raw_evidence_frame = frame.copy()
        self.enhanced_frame = None
        self.threshold_frame = None
        self.pipeline_images = {}
        height, width = frame.shape[:2]
        self.canvas_orig.set_cv_image(frame)
        self.canvas_proc.set_image(None)
        self.canvas_proc.setText("Apply Enhancement to view the result.")
        for canvas in self.stage_canvases.values():
            canvas.set_image(None)
        self.save_button.setEnabled(False)
        self.crop_button.setEnabled(False)
        self.meta_summary_label.setText(f"{self.evidence_name or 'Selected image'} · {width} × {height}")
        self._sync_control_states()
        self._update_observation_controls()
        return True

    def run_enhancement(self, *_args):
        if self.raw_evidence_frame is None or self._is_processing:
            return False

        frame = self.raw_evidence_frame.copy()
        use_grayscale = self.grayscale_check.isChecked()
        use_clahe = self.clahe_check.isChecked()
        use_denoise = self.denoise_check.isChecked()
        use_threshold = self.threshold_check.isChecked()
        clip_limit = self.clip_slider.value() / 10.0
        threshold_level = self.threshold_slider.value()

        self._is_processing = True
        self.enhance_button.setText("Processing…")
        self._sync_control_states()
        self._update_observation_controls()
        self.enhance_button.repaint()
        self.enhancement_status_label.repaint()

        try:
            gray_stage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            working = gray_stage.copy() if use_grayscale else frame.copy()

            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                if working.ndim == 2:
                    working = clahe.apply(working)
                else:
                    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    working = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            contrast_stage = working.copy()

            if use_denoise:
                working = cv2.GaussianBlur(working, (5, 5), 0)
            denoised_stage = working.copy()
            gray_for_threshold = (
                working if working.ndim == 2 else cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
            )
            _, threshold_frame = cv2.threshold(
                gray_for_threshold, threshold_level, 255, cv2.THRESH_BINARY
            )
            enhanced_frame = threshold_frame if use_threshold else working
            pipeline_images = {
                "Grayscale": gray_stage,
                "CLAHE": contrast_stage,
                "Denoised": denoised_stage,
                "Threshold": threshold_frame,
            }

            self.threshold_frame = threshold_frame
            self.enhanced_frame = enhanced_frame
            self.pipeline_images = pipeline_images
            for name, image in pipeline_images.items():
                self.stage_canvases[name].set_cv_image(image)
            self.save_button.setEnabled(True)
            self.crop_button.setEnabled(True)
            self.update_view_display()
            succeeded = True
        except Exception as exc:
            succeeded = False
            self.enhanced_frame = None
            self.threshold_frame = None
            self.pipeline_images = {}
            self.save_button.setEnabled(False)
            self.crop_button.setEnabled(False)
            self.canvas_proc.set_image(None)
            self.canvas_proc.setText("Enhancement failed · check the selected evidence.")
            for canvas in self.stage_canvases.values():
                canvas.set_image(None)
            self.log_message.emit(f"Could not enhance this image: {exc}", "error")
        finally:
            self._is_processing = False
            self.enhance_button.setText("Apply Enhancement")
            self._sync_control_states()
            self._update_observation_controls()

        if succeeded:
            self.enhancement_status_label.setText("Enhancement applied. Inspect the result and processing stages.")
            self.enhancement_status_label.setStyleSheet(
                f"color: {COLORS['success']}; font-size: {FONTS['size_xs']};"
            )
            return True

        self.enhancement_status_label.setText("Enhancement failed. Check the image or choose another file.")
        self.enhancement_status_label.setStyleSheet(
            f"color: {COLORS['alert']}; font-size: {FONTS['size_xs']};"
        )
        return False

    def update_view_display(self, *_args):
        if self.raw_evidence_frame is None:
            self.canvas_orig.set_image(None)
            self.canvas_proc.set_image(None)
            return
        self.canvas_orig.set_cv_image(self.raw_evidence_frame)
        if self.enhanced_frame is not None:
            self.canvas_proc.set_cv_image(self.enhanced_frame)
        else:
            self.canvas_proc.set_image(None)
            self.canvas_proc.setText("Apply enhancement to view the result.")

    def record_finding(self):
        if self.raw_evidence_frame is None or self.enhanced_frame is None or self._is_processing:
            self._update_observation_controls()
            return False
        raw_value = self.finding_input.text().strip()
        if not raw_value:
            self._update_observation_controls()
            return False

        height, width = self.raw_evidence_frame.shape[:2]
        dimensions = f" from {self.evidence_name or 'selected image'} ({width} × {height})"
        try:
            registry.add(
                EvidenceEntry(
                    module="Level 1: Eyes Online",
                    kind="image_observation",
                    value=raw_value,
                    technique="OpenCV grayscale, contrast enhancement, denoising, and threshold views",
                    observation=f"Participant-recorded visible detail{dimensions}: {raw_value}",
                    interpretation="This is a recorded visual observation. Confirm the time or detail against the original evidence and independent records.",
                    uncertainty="Medium",
                    source_filename=self.evidence_name,
                    source_timestamp="",
                    frame_number=None,
                )
            )
        except Exception as exc:
            self.verification_status_label.setText("Observation could not be saved. Please try again.")
            self.verification_status_label.setStyleSheet(
                f"color: {COLORS['alert']}; font-size: {FONTS['size_xs']};"
            )
            self.log_message.emit(f"Could not save the observation: {exc}", "error")
            return False

        self.finding_recorded = True
        self.level_completed.emit("Eyes Online")
        self.log_message.emit("Image observation recorded in the Evidence Notebook.", "success")
        self._last_saved_observation = raw_value
        self.saved_observation_label.setText(f"Saved observation: {raw_value}")
        self.saved_observation_label.show()
        self.finding_input.clear()
        self._update_observation_controls()
        return True

    def save_enhanced_image(self):
        if self.enhanced_frame is None:
            self.log_message.emit("Apply enhancement before saving an output image.", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Enhanced Evidence", "lab4b_corridor_enhanced.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self.enhanced_frame)
            self.log_message.emit(f"Saved enhanced evidence to {path}", "success")

    def save_region_crop(self):
        target = self.enhanced_frame if self.enhanced_frame is not None else self.raw_evidence_frame
        if target is None:
            return
        height, width = target.shape[:2]
        crop = target[max(0, height - max(1, height // 9)) :, max(0, width - max(1, width // 3)) :]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image Region", "image_region_crop.png", "PNG (*.png)"
        )
        if path:
            cv2.imwrite(path, crop)
            self.log_message.emit(f"Saved image region to {path}", "success")

    def reset(self):
        self.finding_recorded = False
        self.raw_evidence_frame = None
        self.enhanced_frame = None
        self.threshold_frame = None
        self.pipeline_images = {}
        self.evidence_name = ""
        self._is_processing = False
        self._last_saved_observation = ""
        if hasattr(self, "finding_input"):
            self.finding_input.clear()
        if hasattr(self, "verification_status_label"):
            self.saved_observation_label.clear()
            self.saved_observation_label.hide()
            self.enhance_button.setText("Apply Enhancement")
            self.meta_summary_label.setText("No evidence selected")
            self.canvas_orig.set_image(None)
            self.canvas_proc.set_image(None)
            for canvas in self.stage_canvases.values():
                canvas.set_image(None)
            self.save_button.setEnabled(False)
            self.crop_button.setEnabled(False)
            self._sync_control_states()
            self._update_observation_controls()
