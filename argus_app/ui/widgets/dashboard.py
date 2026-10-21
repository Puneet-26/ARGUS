"""Minimal case landing page and four-level progress overview."""

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from evidence.registry import registry
from ui.style import COLORS, FONTS, RADIUS


class StageCard(QFrame):
    clicked = Signal(str)

    def __init__(self, number, stage_id, title, _objective="", parent=None):
        super().__init__(parent)
        self.stage_id = stage_id
        self.setObjectName("stageCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame#stageCard {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: {RADIUS['md']}; }}"
            f"QFrame#stageCard:hover {{ background: {COLORS['elevated']}; border-color: {COLORS['primary']}; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)
        number_label = QLabel(f"{number:02d}")
        number_label.setFixedWidth(34)
        number_label.setStyleSheet(f"color: {COLORS['amber']}; font-family: {FONTS['family_mono']}; font-size: 16px;")
        layout.addWidget(number_label)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLORS['text_main']}; font-size: {FONTS['size_base']}; font-weight: 600;")
        layout.addWidget(title_label, 1)
        self.status_label = QLabel("NOT STARTED")
        self.status_label.setMinimumWidth(190)
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.status_label)
        self.setMinimumHeight(58)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.stage_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def set_status(self, status):
        labels = {
            "not_started": "NOT STARTED",
            "in_progress": "IN PROGRESS",
            "complete": "COMPLETED",
            "verified": "COMPLETED",
            "completed_independently": "COMPLETED",
            "checkpoint": "CHECKPOINT USED",
        }
        text = labels.get(status, status.replace("_", " ").upper())
        self.status_label.setText(text)
        color = COLORS["success"] if text == "COMPLETED" else COLORS["text_muted"]
        self.status_label.setStyleSheet(f"color: {color}; font-size: {FONTS['size_xs']}; font-weight: 600;")


class DashboardWidget(QWidget):
    """Home page focused on progress and continuing the investigation."""

    level_selected = Signal(str)

    STAGES = [
        (1, "Recover the timestamp", "Eyes Online", ""),
        (2, "The Missing Phone", "Object Detection", ""),
        (3, "Identify the Red Bag", "The Red Bag", ""),
        (4, "Track the Red Bag", "The Final Clue", ""),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage_cards = {}
        self.completed_stages = set()
        self.checkpoint_stages = set()
        self.active_stage = None
        self.setup_ui()
        self.sync_progress_from_registry()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(14)

        title = QLabel("ARGUS · ROHAN MEHTA CASE")
        title.setStyleSheet(f"color: {COLORS['text_main']}; font-size: {FONTS['size_2xl']}; font-weight: 700;")
        root.addWidget(title)
        description = QLabel("Investigate the surviving evidence from the Robotics Lab and record what it can—and cannot—show.")
        description.setWordWrap(True)
        description.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_base']};")
        root.addWidget(description)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(12)
        self.progress_count = QLabel("0 / 4 levels complete")
        self.progress_count.setStyleSheet(f"color: {COLORS['text_main']}; font-size: {FONTS['size_base']}; font-weight: 600;")
        progress_row.addWidget(self.progress_count)
        progress_row.addStretch(1)
        self.continue_button = QPushButton("Continue investigation")
        self.continue_button.setObjectName("primaryButton")
        self.continue_button.setMinimumHeight(40)
        self.continue_button.clicked.connect(self.continue_investigation)
        progress_row.addWidget(self.continue_button)
        root.addLayout(progress_row)

        levels_heading = QLabel("INVESTIGATION LEVELS")
        levels_heading.setObjectName("sectionHeading")
        root.addWidget(levels_heading)
        for number, title_text, stage_id, objective in self.STAGES:
            card = StageCard(number, stage_id, title_text, objective)
            card.clicked.connect(self.level_selected.emit)
            self.stage_cards[stage_id] = card
            root.addWidget(card)
        root.addStretch(1)

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
        if level_name not in self.stage_cards:
            return
        if status in ("complete", "verified", "completed_independently"):
            self.completed_stages.add(level_name)
            self.checkpoint_stages.discard(level_name)
        elif status == "checkpoint":
            self.completed_stages.add(level_name)
            self.checkpoint_stages.add(level_name)
        effective = "checkpoint" if level_name in self.checkpoint_stages else (
            "completed_independently" if level_name in self.completed_stages else status
        )
        self.stage_cards[level_name].set_status(effective)
        self.progress_count.setText(f"{len(self.completed_stages)} / 4 levels complete")

    def reset_progress(self):
        self.completed_stages.clear()
        self.checkpoint_stages.clear()
        self.active_stage = None
        for card in self.stage_cards.values():
            card.set_status("not_started")
        self.progress_count.setText("0 / 4 levels complete")

    def sync_progress_from_registry(self):
        entries = registry.all()
        completed, checkpoints, independent = set(), set(), set()
        stage_ids = {
            1: "Eyes Online",
            2: "Object Detection",
            3: "The Red Bag",
            4: "The Final Clue",
        }
        legacy_ids = {
            "eyes online": "Eyes Online",
            "object detection": "Object Detection",
            "the missing phone": "Object Detection",
            "the red bag": "The Red Bag",
            "the final clue": "The Final Clue",
            "follow the trail": "The Final Clue",
        }
        for entry in entries:
            module = entry.module.lower()
            match = re.match(r"^(?:checkpoint\s+)?level\s+([1-4])\b", module)
            stage_id = stage_ids.get(int(match.group(1))) if match else legacy_ids.get(module.strip())
            if stage_id:
                completed.add(stage_id)
                (checkpoints if entry.completion_mode == "checkpoint" else independent).add(stage_id)
        checkpoints.difference_update(independent)
        self.completed_stages = completed
        self.checkpoint_stages = checkpoints
        for _, _, stage_id, _ in self.STAGES:
            status = "checkpoint" if stage_id in checkpoints else "completed_independently" if stage_id in completed else (
                "in_progress" if stage_id == self.active_stage else "not_started"
            )
            self.stage_cards[stage_id].set_status(status)
        self.progress_count.setText(f"{len(completed)} / 4 levels complete")
