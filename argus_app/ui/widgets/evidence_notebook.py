"""Evidence list, record details, and investigator note form."""

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from evidence.labels import display_module_label
from evidence.registry import EvidenceEntry, registry
from ui.style import COLORS


class EvidenceNotebookWidget(QWidget):
    """A concise evidence list with a full detail view for the selected record."""

    log_message = Signal(str, str)
    entries_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_filename = ""
        self.source_timestamp = ""
        self.setup_ui()
        self.refresh_table()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Evidence Notebook")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        self.storage_status = QLabel("Saved automatically")
        self.storage_status.setObjectName("mutedLabel")
        self.storage_status.setToolTip(str(registry.storage_path))
        header.addWidget(self.storage_status, 1)
        root.addLayout(header)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.delete_button = QPushButton("Delete selected")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_selected)
        actions.addWidget(self.delete_button)
        export_button = QPushButton("Export JSON…")
        export_button.clicked.connect(self.export_notebook)
        actions.addWidget(export_button)
        clear_button = QPushButton("Clear notebook")
        clear_button.clicked.connect(self.clear_all)
        actions.addWidget(clear_button)
        root.addLayout(actions)

        self.add_note_toggle = QToolButton()
        self.add_note_toggle.setText("Add investigator note  ▸")
        self.add_note_toggle.setCheckable(True)
        self.add_note_toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.add_note_toggle.toggled.connect(self._toggle_note_form)
        root.addWidget(self.add_note_toggle, 0, Qt.AlignLeft)
        self._build_note_form(root)

        self.empty_state = QLabel(
            "No evidence recorded yet. Record a result from a level or add an investigator note."
        )
        self.empty_state.setWordWrap(True)
        self.empty_state.setObjectName("mutedLabel")
        root.addWidget(self.empty_state)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Level / source", "Finding", "Method", "Uncertainty"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        self.table.setColumnWidth(0, 58)
        self.table.setColumnWidth(1, 180)
        self.table.setColumnWidth(3, 190)
        self.table.setColumnWidth(4, 105)
        self.table.setWordWrap(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._show_selected_details)
        root.addWidget(self.table, 3)

        details_heading = QLabel("Selected evidence details")
        details_heading.setObjectName("sectionHeading")
        root.addWidget(details_heading)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select an entry to inspect its full observation and supporting details.")
        self.details.setMinimumHeight(125)
        root.addWidget(self.details, 2)

    def _build_note_form(self, root):
        self.note_form = QFrame()
        self.note_form.setObjectName("summaryCard")
        form = QGridLayout(self.note_form)
        form.setContentsMargins(10, 8, 10, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        self.module_combo = QComboBox()
        self.module_combo.addItems([
            "Level 1: Sight Repair", "Level 2: The Missing Phone", "Level 3: Identify the Red Bag",
            "Level 4: Track the Red Bag", "General Evidence",
        ])
        self.uncertainty_combo = QComboBox()
        self.uncertainty_combo.addItems(["Low", "Medium", "High"])
        self.technique_edit = QLineEdit()
        self.technique_edit.setPlaceholderText("Method used")
        self.obs_edit = QTextEdit()
        self.obs_edit.setPlaceholderText("Direct observation")
        self.obs_edit.setMaximumHeight(64)
        self.interp_edit = QTextEdit()
        self.interp_edit.setPlaceholderText("Interpretation (optional)")
        self.interp_edit.setMaximumHeight(64)
        self.limitation_edit = QLineEdit()
        self.limitation_edit.setPlaceholderText("What this result cannot establish")
        self.question_edit = QLineEdit()
        self.question_edit.setPlaceholderText("Unresolved question (optional)")

        form.addWidget(QLabel("Level"), 0, 0)
        form.addWidget(self.module_combo, 0, 1)
        form.addWidget(QLabel("Uncertainty"), 0, 2)
        form.addWidget(self.uncertainty_combo, 0, 3)
        form.addWidget(QLabel("Method"), 1, 0)
        form.addWidget(self.technique_edit, 1, 1, 1, 3)
        form.addWidget(self.obs_edit, 2, 0, 1, 2)
        form.addWidget(self.interp_edit, 2, 2, 1, 2)
        form.addWidget(self.limitation_edit, 3, 0, 1, 2)
        form.addWidget(self.question_edit, 3, 2, 1, 2)
        self.source_label = QLabel("Source: no evidence selected")
        self.source_label.setObjectName("mutedLabel")
        form.addWidget(self.source_label, 4, 0, 1, 3)
        save_button = QPushButton("Save note")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self.add_entry_from_form)
        form.addWidget(save_button, 4, 3, alignment=Qt.AlignRight)
        self.note_form.setVisible(False)
        root.addWidget(self.note_form)

    def _toggle_note_form(self, expanded):
        self.note_form.setVisible(expanded)
        self.add_note_toggle.setText("Add investigator note  ▾" if expanded else "Add investigator note  ▸")

    def set_source_metadata(self, filename="", source_timestamp=""):
        self.source_filename = str(filename or "")
        self.source_timestamp = str(source_timestamp or "")
        suffix = f" · {self.source_timestamp}" if self.source_timestamp else ""
        self.source_label.setText(f"Source: {self.source_filename or 'no evidence selected'}{suffix}")

    @staticmethod
    def _entry_finding(entry):
        return str(entry.observation or entry.description or entry.value or "No observation recorded")

    def refresh_table(self):
        entries = registry.all()
        if registry.persistence_error:
            self.storage_status.setText("Save warning")
            self.storage_status.setStyleSheet(f"color: {COLORS['alert']};")
            self.storage_status.setToolTip(registry.persistence_error)
        else:
            self.storage_status.setText(f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} · saved")
            self.storage_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.storage_status.setToolTip(str(registry.storage_path))

        self.table.setRowCount(len(entries))
        self.empty_state.setVisible(not entries)
        self.table.setVisible(bool(entries))
        for row, entry in enumerate(entries):
            source = entry.source_filename or display_module_label(entry.module) or entry.module
            finding = self._entry_finding(entry)
            method = entry.technique or entry.kind or "—"
            values = (entry.evidence_id or "—", source, finding, method, entry.uncertainty or "—")
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.table.setItem(row, column, item)
            self.table.setRowHeight(row, 42)

        if entries:
            self.table.selectRow(len(entries) - 1)
            self.table.scrollToBottom()
        else:
            self.delete_button.setEnabled(False)
            self.details.clear()

    def _show_selected_details(self):
        row = self.table.currentRow()
        entries = registry.all()
        self.delete_button.setEnabled(0 <= row < len(entries))
        if not 0 <= row < len(entries):
            self.details.clear()
            return
        entry = entries[row]
        artifact_path = Path(entry.output_path) if entry.output_path else None
        if artifact_path is not None and not artifact_path.is_absolute():
            artifact_path = Path(__file__).resolve().parents[3] / artifact_path
        artifact_status = (
            "Available" if artifact_path and artifact_path.is_file()
            else "Missing" if artifact_path else "Not recorded"
        )
        details = [
            f"{entry.evidence_id or 'Unnumbered'} · {display_module_label(entry.module) or entry.module}",
            f"Observation: {self._entry_finding(entry)}",
            f"Method: {entry.technique or entry.kind or 'Not recorded'}",
            f"Interpretation: {entry.interpretation or 'Not recorded'}",
            f"Uncertainty: {entry.uncertainty or 'Not recorded'}",
            f"Source file: {entry.source_filename or 'Not recorded'}",
            f"Source path: {entry.source_path or 'Not recorded'}",
            f"Source time: {entry.source_timestamp or 'Not recorded'}",
            f"Frame: {entry.frame_number if entry.frame_number is not None else 'Not recorded'}",
            f"Saved output: {entry.output_path or 'Not recorded'}",
            f"Output file status: {artifact_status}",
            f"Limitation: {entry.limitation or 'Not recorded'}",
            f"Unresolved question: {entry.unresolved_question or 'Not recorded'}",
            f"Completion: {entry.completion_mode or 'Not recorded'}",
            f"Recorded: {entry.timestamp or 'Not recorded'}",
            f"Value: {json.dumps(entry.value, ensure_ascii=False, indent=2, default=str)}",
        ]
        self.details.setPlainText("\n".join(details))

    def add_entry_from_form(self):
        technique = self.technique_edit.text().strip()
        observation = self.obs_edit.toPlainText().strip()
        interpretation = self.interp_edit.toPlainText().strip()
        limitation = self.limitation_edit.text().strip()
        if not observation or not technique or not limitation:
            self.log_message.emit("Enter a method, direct observation, and limitation before saving.", "warning")
            return

        module = self.module_combo.currentText()
        saved = registry.add(EvidenceEntry(
            module=module,
            kind="manual_note",
            value=observation,
            technique=technique,
            observation=observation,
            interpretation=interpretation or "Under review",
            uncertainty=self.uncertainty_combo.currentText(),
            source_filename=self.source_filename,
            source_timestamp=self.source_timestamp,
            unresolved_question=self.question_edit.text().strip(),
            limitation=limitation,
        ))
        if not saved:
            self.refresh_table()
            self.log_message.emit(registry.persistence_error or "Could not save investigator note.", "error")
            return

        self.refresh_table()
        self.entries_changed.emit()
        self.technique_edit.clear()
        self.obs_edit.clear()
        self.interp_edit.clear()
        self.question_edit.clear()
        self.limitation_edit.clear()
        self.add_note_toggle.setChecked(False)
        self.log_message.emit("Investigator note saved.", "success")

    def delete_selected(self):
        index = self.table.currentRow()
        entries = registry.all()
        if not 0 <= index < len(entries):
            return
        entry = entries[index]
        answer = QMessageBox.question(
            self, "Delete evidence entry",
            f"Permanently delete {entry.evidence_id or 'this evidence entry'}?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        if not registry.remove(index):
            self.refresh_table()
            self.log_message.emit(registry.persistence_error or "Could not delete the evidence entry.", "error")
            return
        self.refresh_table()
        self.entries_changed.emit()
        self.log_message.emit("Evidence entry deleted.", "info")

    def clear_all(self):
        if not registry.all():
            return
        answer = QMessageBox.question(
            self, "Clear Evidence Notebook", "Delete every recorded evidence entry? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        if not registry.clear():
            self.refresh_table()
            self.log_message.emit(registry.persistence_error or "Could not clear the notebook.", "error")
            return
        self.refresh_table()
        self.entries_changed.emit()
        self.log_message.emit("Evidence Notebook cleared.", "warning")

    def export_notebook(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Evidence Notebook", "evidence_notebook.json", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            registry.save(Path(path))
            self.log_message.emit(f"Notebook exported to {path}", "success")
        except (OSError, TypeError, ValueError) as exc:
            self.log_message.emit(f"Notebook export failed: {exc}", "error")
