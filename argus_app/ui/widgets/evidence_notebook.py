"""
Evidence Notebook Widget - Forensic Investigation Log
Participants log technique, observation, interpretation, and uncertainty for each piece of evidence.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QFrame, QPushButton, QTableWidget, QTableWidgetItem,
                                QHeaderView, QTextEdit, QComboBox, QGroupBox, QFileDialog,
                                QGridLayout, QLineEdit)
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Qt, Signal
from evidence.registry import registry, EvidenceEntry
from ui.style import COLORS, FONTS, SPACING, RADIUS
from pathlib import Path


class EvidenceNotebookWidget(QWidget):
    """Forensic Evidence Notebook view."""

    log_message = Signal(str, str)
    entries_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.refresh_table()

    def setup_ui(self):
        """Set up UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(12)

        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            background-color: {COLORS['bg_panel']};
            border: 1px solid {COLORS['border']};
            border-radius: {RADIUS['md']};
            padding: {SPACING['md']};
        """)
        header_layout = QHBoxLayout(header_frame)

        title_layout = QVBoxLayout()
        title = QLabel("EVIDENCE NOTEBOOK")
        title.setStyleSheet(f"font-size: {FONTS['size_xl']}; font-weight: 700; color: {COLORS['accent_teal']};")
        title_layout.addWidget(title)

        subtitle = QLabel("Case: The Lab 4B Incident | Investigation Record")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_sm']};")
        title_layout.addWidget(subtitle)
        self.storage_status = QLabel()
        self.storage_status.setObjectName("secondaryLabel")
        self.storage_status.setWordWrap(True)
        self.storage_status.setToolTip(str(registry.storage_path))
        title_layout.addWidget(self.storage_status)
        header_layout.addLayout(title_layout)

        header_layout.addStretch()

        export_btn = QPushButton("Export Notebook (JSON)...")
        export_btn.clicked.connect(self.export_notebook)
        header_layout.addWidget(export_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setObjectName("dangerButton")
        clear_btn.clicked.connect(self.clear_all)
        header_layout.addWidget(clear_btn)

        main_layout.addWidget(header_frame)

        # Add new note group
        add_group = QGroupBox("Add / Edit Investigation Note")
        add_layout = QVBoxLayout(add_group)

        inputs_layout = QGridLayout()
        inputs_layout.setHorizontalSpacing(10)
        inputs_layout.setVerticalSpacing(8)
        inputs_layout.addWidget(QLabel("Module"), 0, 0)
        self.module_combo = QComboBox()
        self.module_combo.addItems(["Level 1: Eyes Online", "Level 2: Object Detection", "Level 3: The Red Bag", "Level 4: The Final Clue", "General Evidence"])
        inputs_layout.addWidget(self.module_combo, 0, 1)
        inputs_layout.addWidget(QLabel("Uncertainty"), 0, 2)
        self.uncertainty_combo = QComboBox()
        self.uncertainty_combo.addItems(["Low", "Medium", "High"])
        inputs_layout.addWidget(self.uncertainty_combo, 0, 3)

        inputs_layout.addWidget(QLabel("Technique"), 1, 0)
        self.technique_edit = QLineEdit()
        self.technique_edit.setPlaceholderText("e.g., CLAHE + Thresholding")
        inputs_layout.addWidget(self.technique_edit, 1, 1, 1, 3)

        add_layout.addLayout(inputs_layout)

        details_layout = QGridLayout()
        details_layout.setHorizontalSpacing(10)
        details_layout.setVerticalSpacing(6)
        details_layout.addWidget(QLabel("Direct observation"), 0, 0)
        details_layout.addWidget(QLabel("Interpretation"), 0, 1)
        self.obs_edit = QTextEdit()
        self.obs_edit.setPlaceholderText("Direct computer vision or physical observation...")
        self.obs_edit.setMinimumHeight(100)
        self.interp_edit = QTextEdit()
        self.interp_edit.setPlaceholderText("Investigator interpretation...")
        self.interp_edit.setMinimumHeight(100)
        details_layout.addWidget(self.obs_edit, 1, 0)
        details_layout.addWidget(self.interp_edit, 1, 1)
        add_layout.addLayout(details_layout)
        self.source_label = QLabel("Source: no evidence selected")
        self.source_label.setObjectName("secondaryLabel")
        add_layout.addWidget(self.source_label)
        unresolved_row = QHBoxLayout()
        unresolved_row.addWidget(QLabel("Unresolved question"))
        self.question_edit = QLineEdit()
        self.question_edit.setPlaceholderText("Optional question or evidence gap for the report")
        unresolved_row.addWidget(self.question_edit, 1)
        add_layout.addLayout(unresolved_row)

        add_btn_layout = QHBoxLayout()
        add_btn_layout.addStretch()
        add_note_btn = QPushButton("Save Entry to Notebook")
        add_note_btn.setObjectName("primaryButton")
        add_note_btn.clicked.connect(self.add_entry_from_form)
        add_btn_layout.addWidget(add_note_btn)
        add_layout.addLayout(add_btn_layout)

        main_layout.addWidget(add_group)

        # Table of evidence
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Module", "Technique", "Direct Observation",
            "Interpretation", "Source", "Source time", "Unresolved Question", "Uncertainty", "Action"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        self.table.setWordWrap(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_panel']};
                gridline-color: {COLORS['border']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['md']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['accent_teal']};
                font-weight: 600;
                padding: 6px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        main_layout.addWidget(self.table, 1)

    def set_source_metadata(self, filename="", source_timestamp=""):
        self.source_filename = str(filename or "")
        self.source_timestamp = str(source_timestamp or "")
        suffix = f" · {self.source_timestamp}" if self.source_timestamp else ""
        self.source_label.setText(f"Source: {self.source_filename or 'no evidence selected'}{suffix}")

    def refresh_table(self):
        """Refresh evidence table from registry."""
        entries = registry.all()
        if registry.persistence_error:
            self.storage_status.setText(f"Notebook save warning: {registry.persistence_error}")
            self.storage_status.setStyleSheet(f"color: {COLORS['alert']};")
        else:
            self.storage_status.setText(f"Saved automatically · {registry.storage_path}")
            self.storage_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.table.setRowCount(len(entries))

        for i, entry in enumerate(entries):
            self.table.setItem(i, 0, QTableWidgetItem(entry.timestamp))
            self.table.setItem(i, 1, QTableWidgetItem(entry.module))
            self.table.setItem(i, 2, QTableWidgetItem(entry.technique or entry.kind))
            self.table.setItem(i, 3, QTableWidgetItem(entry.observation or entry.description))
            self.table.setItem(i, 4, QTableWidgetItem(entry.interpretation or "Pending interpretation"))
            self.table.setItem(i, 5, QTableWidgetItem(entry.source_filename))
            source_time = entry.source_timestamp or (f"Frame {entry.frame_number + 1}" if entry.frame_number is not None else "")
            self.table.setItem(i, 6, QTableWidgetItem(source_time))
            self.table.setItem(i, 7, QTableWidgetItem(entry.unresolved_question))

            unc_item = QTableWidgetItem(entry.uncertainty)
            if entry.uncertainty == "High":
                unc_item.setForeground(Qt.red)
            elif entry.uncertainty == "Medium":
                unc_item.setForeground(Qt.yellow)
            else:
                unc_item.setForeground(Qt.green)
            self.table.setItem(i, 8, unc_item)

            del_btn = QPushButton("Delete")
            del_btn.setMaximumWidth(70)
            del_btn.clicked.connect(lambda _, idx=i: self.delete_entry(idx))
            self.table.setCellWidget(i, 9, del_btn)

    def add_entry_from_form(self):
        """Add new entry from form fields."""
        technique = self.technique_edit.text().strip()
        observation = self.obs_edit.toPlainText().strip()
        interpretation = self.interp_edit.toPlainText().strip()

        if not observation or not technique:
            self.log_message.emit("Enter both a direct observation and the technique used before saving.", "warning")
            return

        entry = EvidenceEntry(
            module=self.module_combo.currentText(),
            kind="manual_note",
            value=observation,
            technique=technique,
            observation=observation,
            interpretation=interpretation or "Under review",
            uncertainty=self.uncertainty_combo.currentText(),
            source_filename=getattr(self, "source_filename", ""),
            source_timestamp=getattr(self, "source_timestamp", ""),
            unresolved_question=self.question_edit.text().strip(),
        )
        registry.add(entry)
        self.refresh_table()
        self.entries_changed.emit()

        self.technique_edit.clear()
        self.obs_edit.clear()
        self.interp_edit.clear()
        self.question_edit.clear()
        self.log_message.emit("Added entry to Evidence Notebook", "success")

    def delete_entry(self, index):
        """Delete evidence entry."""
        registry.remove(index)
        self.refresh_table()
        self.entries_changed.emit()
        self.log_message.emit(f"Deleted evidence entry #{index+1}", "info")

    def clear_all(self):
        """Clear all entries."""
        answer = QMessageBox.question(
            self, "Clear Evidence Notebook",
            "Delete every recorded evidence entry? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        registry.clear()
        self.refresh_table()
        self.entries_changed.emit()
        self.log_message.emit("Cleared Evidence Notebook", "info")

    def export_notebook(self):
        """Export notebook entries to JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Evidence Notebook", "evidence_notebook.json", "JSON Files (*.json)"
        )
        if file_path:
            try:
                registry.save(Path(file_path))
                self.log_message.emit(f"Exported Evidence Notebook to {file_path}", "success")
            except Exception as e:
                self.log_message.emit(f"Export failed: {e}", "error")
