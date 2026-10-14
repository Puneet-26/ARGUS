"""Compact home screen for the four-level ARGUS case investigation."""

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from evidence.registry import registry
from ui.style import COLORS, FONTS, RADIUS


class StageCard(QFrame):
    clicked = Signal(str)

    def __init__(self, number, stage_id, title, objective, parent=None):
        super().__init__(parent)
        self.stage_id = stage_id
        self.setObjectName("stageCard")
        self.setStyleSheet(
            f"QFrame#stageCard {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: {RADIUS['md']}; }}"
            f"QFrame#stageCard:hover {{ background: {COLORS['elevated']}; border-color: {COLORS['primary']}; }}"
        )
        self.setMinimumHeight(128)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        header = QHBoxLayout()
        number_label = QLabel(f"LEVEL {number:02d}")
        number_label.setStyleSheet(f"color: {COLORS['amber']}; font-family: {FONTS['family_mono']}; font-size: 11px;")
        self.status_label = QLabel("NOT STARTED")
        self.status_label.setObjectName("stageStatus")
        header.addWidget(number_label)
        header.addStretch(1)
        header.addWidget(self.status_label)
        layout.addLayout(header)

        name = QLabel(title)
        name.installEventFilter(self)
        name.setStyleSheet("font-size: 18px; font-weight: 650; color: #FFFFFF;")
        layout.addWidget(name)
        description = QLabel(objective)
        description.installEventFilter(self)
        description.setWordWrap(True)
        description.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_sm']};")
        layout.addWidget(description, 1)

        button = QPushButton("Open level  ›")
        button.setObjectName("secondaryButton")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: self.clicked.emit(self.stage_id))
        layout.addWidget(button, 0, Qt.AlignRight)
        self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.stage_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            self.clicked.emit(self.stage_id)
            return True
        return super().eventFilter(watched, event)

    def set_status(self, status):
        text = "COMPLETED" if status in ("complete", "verified") else status.replace("_", " ").upper()
        self.status_label.setText(text)
        if text == "COMPLETED":
            self.status_label.setStyleSheet(
                f"color: {COLORS['success']}; background: {COLORS['elevated']}; border-radius: 3px; padding: 3px 7px; font-size: 10px;"
            )
        else:
            self.status_label.setStyleSheet(
                f"color: {COLORS['text_muted']}; background: {COLORS['elevated']}; border-radius: 3px; padding: 3px 7px; font-size: 10px;"
            )


class DashboardWidget(QWidget):
    """Case overview focused on the next useful action."""

    level_selected = Signal(str)
    notebook_requested = Signal()

    STAGES = [
        (1, "Eyes Online", "Eyes Online", "Enhance one image and inspect visible details."),
        (2, "Object Detection", "Object Detection", "Run pretrained YOLO on one image."),
        (3, "Color Identification", "The Red Bag", "Inspect colour regions in one image with HSV masks."),
        (4, "Timeline and Tracking", "The Final Clue", "Track movement through video and inspect exit crossings."),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage_cards = {}
        self.completed_stages = set()
        self.active_stage = None
        self.setup_ui()
        self.refresh_notebook_summary()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        hero = QFrame()
        hero.setObjectName("caseHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(20, 15, 20, 15)
        hero_layout.setSpacing(5)
        eyebrow = QLabel("ACTIVE CASE · ARGUS-04")
        eyebrow.setStyleSheet(f"color: {COLORS['amber']}; font-family: {FONTS['family_mono']}; font-size: 11px; letter-spacing: 1px;")
        title = QLabel("The Lab 4B Incident")
        title.setStyleSheet("color: #FFFFFF; font-size: 26px; font-weight: 700;")
        description = QLabel(
            "Investigate recovered CCTV and independent records to reconstruct what happened at Lab 4B. "
            "Open any level in any order; treat each result according to its limits."
        )
        description.setWordWrap(True)
        description.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_base']};")
        hero_layout.addWidget(eyebrow)
        hero_layout.addWidget(title)
        hero_layout.addWidget(description)
        root.addWidget(hero)

        progress = QFrame()
        progress.setObjectName("progressCard")
        progress_layout = QHBoxLayout(progress)
        progress_layout.setContentsMargins(16, 9, 16, 9)
        progress_layout.setSpacing(12)
        progress_title = QVBoxLayout()
        caption = QLabel("INVESTIGATION PROGRESS")
        caption.setStyleSheet(f"color: {COLORS['text_muted']}; font-family: {FONTS['family_mono']}; font-size: 10px; letter-spacing: 1px;")
        self.progress_count = QLabel("0 / 4 complete")
        self.progress_count.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 13px; font-weight: 600;")
        progress_title.addWidget(caption)
        progress_title.addWidget(self.progress_count)
        progress_layout.addLayout(progress_title)
        self.progress_segments = []
        segments = QHBoxLayout()
        segments.setSpacing(6)
        for _ in range(4):
            segment = QFrame()
            segment.setMinimumHeight(6)
            segment.setMaximumHeight(6)
            segment.setStyleSheet(f"background: {COLORS['border']}; border-radius: 3px;")
            self.progress_segments.append(segment)
            segments.addWidget(segment, 1)
        progress_layout.addLayout(segments, 1)
        self.continue_button = QPushButton("Continue investigation  ›")
        self.continue_button.setObjectName("primaryButton")
        self.continue_button.clicked.connect(self.continue_investigation)
        progress_layout.addWidget(self.continue_button)
        root.addWidget(progress)

        stage_heading = QLabel("INVESTIGATION LEVELS")
        stage_heading.setObjectName("sectionHeading")
        root.addWidget(stage_heading)

        stages = QGridLayout()
        stages.setContentsMargins(0, 0, 0, 0)
        stages.setHorizontalSpacing(12)
        stages.setVerticalSpacing(10)
        for index, (number, title, stage_id, objective) in enumerate(self.STAGES):
            card = StageCard(number, stage_id, title, objective)
            card.clicked.connect(self.level_selected.emit)
            self.stage_cards[stage_id] = card
            stages.addWidget(card, index // 2, index % 2)
        root.addLayout(stages)

        notebook = QFrame()
        notebook.setObjectName("summaryCard")
        notebook_layout = QHBoxLayout(notebook)
        notebook_layout.setContentsMargins(14, 10, 14, 10)
        notebook_layout.setSpacing(14)
        summary_copy = QVBoxLayout()
        summary_title = QLabel("CASE NOTEBOOK")
        summary_title.setObjectName("sectionHeading")
        self.notebook_summary = QLabel()
        self.notebook_summary.setWordWrap(True)
        self.notebook_summary.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_sm']};")
        summary_copy.addWidget(summary_title)
        summary_copy.addWidget(self.notebook_summary)
        notebook_layout.addLayout(summary_copy, 1)
        notebook_button = QPushButton("Open notebook  ›")
        notebook_button.clicked.connect(self.notebook_requested.emit)
        notebook_layout.addWidget(notebook_button)
        root.addWidget(notebook)
        root.addStretch(1)

    def refresh_notebook_summary(self):
        entries = registry.all()
        count = len(entries)
        if entries:
            recent = entries[-1]
            note = recent.observation or recent.description or str(recent.value)
            self.notebook_summary.setText(f"{count} clue(s) recorded. Latest: {note}")
        else:
            self.notebook_summary.setText("No clues recorded yet. Record a finding from any level to keep it here.")

    def continue_investigation(self):
        if self.active_stage and self.active_stage not in self.completed_stages:
            self.level_selected.emit(self.active_stage)
            return
        next_stage = next((stage for stage in self.STAGES if stage[2] not in self.completed_stages), self.STAGES[-1])
        self.level_selected.emit(next_stage[2])

    def set_active_stage(self, level_name):
        if level_name in self.stage_cards:
            self.active_stage = level_name

    def set_level_status(self, level_name: str, status: str):
        if level_name in self.stage_cards:
            if status in ("complete", "verified"):
                self.completed_stages.add(level_name)
            effective_status = "complete" if level_name in self.completed_stages else status
            self.stage_cards[level_name].set_status(effective_status)
        count = len(self.completed_stages)
        self.progress_count.setText(f"{count} / 4 complete")
        for index, segment in enumerate(self.progress_segments):
            stage_name = self.STAGES[index][2]
            color = COLORS["primary"] if stage_name in self.completed_stages else COLORS["border"]
            segment.setStyleSheet(f"background: {color}; border-radius: 3px;")

    def reset_progress(self):
        self.completed_stages.clear()
        self.active_stage = None
        for card in self.stage_cards.values():
            card.set_status("not_started")
        self.progress_count.setText("0 / 4 complete")
        for segment in self.progress_segments:
            segment.setStyleSheet(f"background: {COLORS['border']}; border-radius: 3px;")
        self.refresh_notebook_summary()

    def sync_progress_from_registry(self):
        entries = registry.all()
        completed = set()
        for entry in entries:
            module = entry.module.lower()
            if "level 1" in module or "eyes online" in module:
                completed.add("Eyes Online")
            elif "level 2" in module or "object detection" in module:
                completed.add("Object Detection")
            elif "level 3" in module or "red bag" in module or "color identification" in module:
                completed.add("The Red Bag")
            elif "level 4" in module or "final clue" in module or "timeline and tracking" in module:
                completed.add("The Final Clue")
        self.completed_stages = completed
        for stage in self.STAGES:
            stage_id = stage[2]
            status = "complete" if stage_id in completed else (
                "in_progress" if stage_id == self.active_stage else "not_started"
            )
            self.stage_cards[stage_id].set_status(status)
        self.progress_count.setText(f"{len(completed)} / 4 complete")
        for index, segment in enumerate(self.progress_segments):
            color = COLORS["primary"] if self.STAGES[index][2] in completed else COLORS["border"]
            segment.setStyleSheet(f"background: {color}; border-radius: 3px;")
        self.refresh_notebook_summary()
