"""Small set of settings that affect the active investigation workflow."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

from evidence.registry import registry
from ui.style import COLORS


class SettingsWidget(QWidget):
    log_message = Signal(str, str)
    frame_dimension_changed = Signal(int)
    case_reset = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Adjust video frame sizing and clear saved case data.")
        subtitle.setObjectName("pageSummary")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        processing = QGroupBox("Video processing")
        processing_layout = QVBoxLayout(processing)
        description = QLabel(
            "Level 4 video frames are scaled before OpenCV processing and playback. Image levels retain the selected image resolution. The source evidence file is not changed."
        )
        description.setWordWrap(True)
        description.setStyleSheet(f"color: {COLORS['text_secondary']};")
        processing_layout.addWidget(description)
        row = QHBoxLayout()
        row.addWidget(QLabel("Maximum video frame dimension"))
        self.dim_spin = QSpinBox()
        self.dim_spin.setRange(320, 1920)
        self.dim_spin.setSingleStep(160)
        self.dim_spin.setValue(1280)
        self.dim_spin.setSuffix(" px")
        self.dim_spin.valueChanged.connect(self.frame_dimension_changed.emit)
        row.addWidget(self.dim_spin)
        row.addStretch(1)
        processing_layout.addLayout(row)
        layout.addWidget(processing)

        case_data = QGroupBox("Case data")
        case_layout = QVBoxLayout(case_data)
        reset_info = QLabel("Clear notebook entries and reset the dashboard progress indicators.")
        reset_info.setWordWrap(True)
        reset_info.setStyleSheet(f"color: {COLORS['text_secondary']};")
        case_layout.addWidget(reset_info)
        reset_row = QHBoxLayout()
        reset_button = QPushButton("Reset case data")
        reset_button.setObjectName("dangerButton")
        reset_button.clicked.connect(self.reset_case_data)
        reset_row.addWidget(reset_button)
        reset_row.addStretch(1)
        case_layout.addLayout(reset_row)
        layout.addWidget(case_data)
        layout.addStretch(1)

    def reset_case_data(self):
        answer = QMessageBox.question(
            self, "Reset case data",
            "Clear all notebook entries and dashboard progress? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        if not registry.clear():
            self.log_message.emit(registry.persistence_error or "Could not clear case data.", "error")
            return
        self.case_reset.emit()
        self.log_message.emit("Case data cleared. Notebook and dashboard progress were reset.", "warning")
