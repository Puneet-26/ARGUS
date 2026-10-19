"""Compact evidence list and optional investigator note form."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
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

from evidence.registry import EvidenceEntry, registry
from evidence.artifacts import documentary_records, missing_manifest_assets
from evidence.labels import display_module_label
from ui.style import COLORS, FONTS, RADIUS


class EvidenceNotebookWidget(QWidget):
    """Show observations once and keep optional note entry out of the way."""

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
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Evidence Notebook")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        self.storage_status = QLabel("Saved automatically")
        self.storage_status.setObjectName("mutedLabel")
        self.storage_status.setToolTip(str(registry.storage_path))
        header.addWidget(self.storage_status, 1)
        export_btn = QPushButton("Export JSON…")
        export_btn.clicked.connect(self.export_notebook)
        header.addWidget(export_btn)
        clear_btn = QPushButton("Clear notebook")
        clear_btn.clicked.connect(self.clear_all)
        header.addWidget(clear_btn)
        root.addLayout(header)

        self.add_note_toggle = QToolButton()
        self.add_note_toggle.setText("Add investigator note  ▸")
        self.add_note_toggle.setCheckable(True)
        self.add_note_toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.add_note_toggle.toggled.connect(self._toggle_note_form)
        root.addWidget(self.add_note_toggle, 0, Qt.AlignLeft)

        self.note_form = QFrame()
        self.note_form.setObjectName("summaryCard")
        form = QGridLayout(self.note_form)
        form.setContentsMargins(10, 8, 10, 8)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        self.module_combo = QComboBox()
        self.module_combo.addItems([
            "Level 1: Sight Repair", "Level 2: The Missing Phone", "Level 3: Identify the Red Bag",
            "Level 4: Track the Red Bag", "E-05: Purge and Session Records", "General Evidence",
        ])
        self.uncertainty_combo = QComboBox()
        self.uncertainty_combo.addItems(["Low", "Medium", "High"])
        self.technique_edit = QLineEdit()
        self.technique_edit.setPlaceholderText("Method used")
        self.obs_edit = QTextEdit()
        self.obs_edit.setPlaceholderText("Direct observation")
        self.obs_edit.setMaximumHeight(72)
        self.interp_edit = QTextEdit()
        self.interp_edit.setPlaceholderText("Interpretation (optional)")
        self.interp_edit.setMaximumHeight(72)
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
        self.documentary_source = QComboBox()
        self.documentary_source.addItem("No documentary source selected", None)
        for record in documentary_records():
            self.documentary_source.addItem(record["filename"], record)
        form.addWidget(self.documentary_source, 5, 0, 1, 3)
        save = QPushButton("Save note")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.add_entry_from_form)
        form.addWidget(save, 5, 3, alignment=Qt.AlignRight)
        self.note_form.setVisible(False)
        root.addWidget(self.note_form)

        self.records_group = QGroupBox("Supplied documentary records")
        self.records_group.setCheckable(True)
        self.records_group.setChecked(False)
        records_layout = QVBoxLayout(self.records_group)
        self.record_selector = QComboBox()
        self.record_view = QTextEdit()
        self.record_view.setReadOnly(True)
        self.record_view.setMaximumHeight(130)
        self.available_records = documentary_records()
        for record in self.available_records:
            self.record_selector.addItem(f"{record['id']} · {record['filename']}", record)
        if not self.available_records:
            missing_records = [item for item in missing_manifest_assets() if item['filename'].lower().endswith('.txt')]
            self.record_selector.addItem("No supplied text records")
            self.record_view.setPlainText(
                "No documentary records are available.\n" + "\n".join(f"Missing: {item['filename']}" for item in missing_records)
            )
            self.record_selector.setEnabled(False)
        else:
            self.record_selector.currentIndexChanged.connect(self._show_documentary_record)
            self._show_documentary_record(0)
        records_layout.addWidget(self.record_selector)
        records_layout.addWidget(self.record_view)
        root.addWidget(self.records_group)

        self.empty_state = QLabel("No evidence recorded yet. Use a level's Record result button or add an investigator note.")
        self.empty_state.setWordWrap(True)
        self.empty_state.setObjectName("mutedLabel")
        root.addWidget(self.empty_state)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "Level / source", "Observation and method", "Interpretation", "Uncertainty", ""])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setWordWrap(True)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumWidth(0)
        root.addWidget(self.table, 1)

    def _toggle_note_form(self, expanded):
        self.note_form.setVisible(expanded)
        self.add_note_toggle.setText("Add investigator note  ▾" if expanded else "Add investigator note  ▸")

    def _show_documentary_record(self, index):
        if 0 <= index < len(self.available_records):
            record = self.available_records[index]
            self.record_view.setPlainText(f"{record['path']}\n\n{record['content']}")

    def set_source_metadata(self, filename="", source_timestamp=""):
        self.source_filename = str(filename or "")
        self.source_timestamp = str(source_timestamp or "")
        suffix = f" · {self.source_timestamp}" if self.source_timestamp else ""
        self.source_label.setText(f"Source: {self.source_filename or 'no evidence selected'}{suffix}")

    def refresh_table(self):
        entries = registry.all()
        if registry.persistence_error:
            self.storage_status.setText("Save warning")
            self.storage_status.setStyleSheet(f"color: {COLORS['alert']};")
            self.storage_status.setToolTip(registry.persistence_error)
        else:
            self.storage_status.setText(f"{len(entries)} saved entr{'y' if len(entries) == 1 else 'ies'}")
            self.storage_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self.storage_status.setToolTip(str(registry.storage_path))
        self.table.setRowCount(len(entries))
        self.empty_state.setVisible(not entries)
        self.table.setVisible(bool(entries))
        for row, entry in enumerate(entries):
            source = entry.source_filename or display_module_label(entry.module)
            method = entry.technique or entry.kind
            observed = entry.observation or entry.description or str(entry.value)
            values = [entry.evidence_id, source, f"{observed}\nMethod: {method}", entry.interpretation or "Pending interpretation", entry.uncertainty]
            details = f"{entry.timestamp}\n{display_module_label(entry.module)}\n{method}\nSource: {entry.source_path or entry.source_filename}\nSource time: {entry.source_timestamp}\nFrame: {entry.frame_number}\nOutput: {entry.output_path}\nLimitation: {entry.limitation}\nUnresolved: {entry.unresolved_question}\nCompletion: {entry.completion_mode}"
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(details)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, column, item)
            delete = QPushButton("Delete")
            delete.setMaximumWidth(64)
            delete.clicked.connect(lambda _checked=False, index=row: self.delete_entry(index))
            self.table.setCellWidget(row, 5, delete)
            self.table.setRowHeight(row, 54)

    def add_entry_from_form(self):
        technique = self.technique_edit.text().strip()
        observation = self.obs_edit.toPlainText().strip()
        interpretation = self.interp_edit.toPlainText().strip()
        limitation = self.limitation_edit.text().strip()
        if not observation or not technique or not limitation:
            self.log_message.emit("Enter a method, direct observation, and limitation before saving.", "warning")
            return
        module = self.module_combo.currentText()
        selected_record = self.documentary_source.currentData()
        source_filename = selected_record["filename"] if selected_record else self.source_filename
        source_path = selected_record["path"] if selected_record else ""
        registry.add(EvidenceEntry(
            module=module,
            kind="manual_note",
            value=observation,
            technique=technique,
            observation=observation,
            interpretation=interpretation or "Under review",
            uncertainty=self.uncertainty_combo.currentText(),
            source_filename=source_filename,
            source_path=source_path,
            source_timestamp=self.source_timestamp,
            unresolved_question=self.question_edit.text().strip(),
            evidence_id="E-05" if module.startswith("E-05") else "",
            limitation=limitation,
        ))
        self.refresh_table()
        self.entries_changed.emit()
        self.technique_edit.clear()
        self.obs_edit.clear()
        self.interp_edit.clear()
        self.question_edit.clear()
        self.limitation_edit.clear()
        self.add_note_toggle.setChecked(False)
        self.log_message.emit("Investigator note saved.", "success")

    def delete_entry(self, index):
        registry.remove(index)
        self.refresh_table()
        self.entries_changed.emit()
        self.log_message.emit("Evidence entry deleted.", "info")

    def clear_all(self):
        answer = QMessageBox.question(
            self, "Clear Evidence Notebook", "Delete every recorded evidence entry? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        registry.clear()
        self.refresh_table()
        self.entries_changed.emit()
        self.log_message.emit("Evidence Notebook cleared.", "warning")

    def export_notebook(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Evidence Notebook", "evidence_notebook.json", "JSON Files (*.json)")
        if path:
            try:
                registry.save(Path(path))
                self.log_message.emit(f"Notebook exported to {path}", "success")
            except Exception as exc:
                self.log_message.emit(f"Notebook export failed: {exc}", "error")
