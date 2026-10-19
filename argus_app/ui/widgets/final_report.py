"""Evidence summary and optional final deduction form."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from evidence.registry import EvidenceEntry, registry
from evidence.artifacts import documentary_records, missing_manifest_assets
from evidence.labels import display_module_label
from ui.style import COLORS, FONTS, RADIUS, SPACING


class FinalReportWidget(QWidget):
    log_message = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.generate_report()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("Final Report")
        title.setObjectName("pageTitle")
        header.addWidget(title, 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.generate_report)
        header.addWidget(refresh)
        export = QPushButton("Export Markdown…")
        export.setObjectName("primaryButton")
        export.clicked.connect(self.export_report)
        header.addWidget(export)
        root.addLayout(header)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet(
            f"QTextEdit {{ background: {COLORS['surface']}; color: {COLORS['text_main']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: {RADIUS['md']}; "
            f"padding: {SPACING['lg']}; font-family: {FONTS['family_ui']}; font-size: {FONTS['size_base']}; }}"
        )
        root.addWidget(self.report_text, 1)

        self.notes_group = QGroupBox("Investigator notes (optional)")
        self.notes_group.setCheckable(True)
        self.notes_group.setChecked(False)
        notes_layout = QVBoxLayout(self.notes_group)
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes included in the exported report.")
        self.notes_edit.setMaximumHeight(80)
        notes_layout.addWidget(self.notes_edit)
        notes_layout.setEnabled(False)
        self.notes_group.toggled.connect(notes_layout.setEnabled)
        root.addWidget(self.notes_group)

        self.deduction_group = QGroupBox("Complete final deduction (optional)")
        self.deduction_group.setCheckable(True)
        self.deduction_group.setChecked(False)
        deduction = QVBoxLayout(self.deduction_group)
        deduction.setSpacing(5)
        deduction.addWidget(QLabel("Working hypothesis, or state that the evidence is insufficient"))
        self.suspect_choice = QComboBox()
        self.suspect_choice.setEditable(True)
        self.suspect_choice.addItems(["", "Evidence is insufficient to identify a person"])
        self.suspect_choice.setCurrentIndex(0)
        self.suspect_choice.setEditText("")
        deduction.addWidget(self.suspect_choice)
        deduction.addWidget(QLabel("At least three clues from different levels or records, each with its source"))
        self.clues_edit = QTextEdit()
        self.clues_edit.setPlaceholderText("Separate observations from interpretations.")
        self.clues_edit.setMaximumHeight(72)
        deduction.addWidget(self.clues_edit)
        deduction.addWidget(QLabel("Access/system records (or state that they are missing)"))
        self.purge_edit = QTextEdit()
        self.purge_edit.setPlaceholderText("What a supplied record establishes, or explicitly note that no record was supplied.")
        self.purge_edit.setMaximumHeight(60)
        deduction.addWidget(self.purge_edit)
        deduction.addWidget(QLabel("What remains uncertain?"))
        self.uncertain_edit = QTextEdit()
        self.uncertain_edit.setPlaceholderText("Name unresolved identity or evidence gaps.")
        self.uncertain_edit.setMaximumHeight(60)
        deduction.addWidget(self.uncertain_edit)
        submit = QPushButton("Submit final deduction")
        submit.setObjectName("primaryButton")
        submit.clicked.connect(self.submit_deduction)
        deduction.addWidget(submit, 0, Qt.AlignRight)
        deduction.setEnabled(False)
        self.deduction_group.toggled.connect(deduction.setEnabled)
        root.addWidget(self.deduction_group)

    def generate_report(self):
        entries = registry.all()
        saved = next((entry for entry in reversed(entries) if entry.module == "Final Deduction" and isinstance(entry.value, dict)), None)
        if saved and not self.suspect_choice.currentText().strip() and not self.clues_edit.toPlainText().strip():
            value = saved.value
            self.suspect_choice.setEditText(str(value.get("working_hypothesis", value.get("leading_suspect", ""))))
            self.clues_edit.setPlainText(str(value.get("clues", "")))
            self.purge_edit.setPlainText(str(value.get("purge_job", "")))
            self.uncertain_edit.setPlainText(str(value.get("uncertainty", "")))

        direct = [entry for entry in entries if entry.kind in ("image_observation", "direct_observation")]
        manual = [entry for entry in entries if entry.kind == "manual_note"]
        computer = [entry for entry in entries if entry.kind in ("image_processing", "detection", "color_segmentation", "video_tracking")]
        documentary = [entry for entry in manual if "record" in (entry.source_path or entry.source_filename).lower() or entry.module.startswith("E-05")]
        report = "# ARGUS / Pixel Pursuit: Rohan Mehta Case\n\n"
        report += "Evidence is summarized from saved notebook results. Model outputs are measurements, not proof of identity or intent. Video times are clip-relative unless source documentation establishes a wall-clock offset.\n\n"
        report += "## Investigation timeline\n"
        level_order = {"Level 1:": 1, "Level 2:": 2, "Level 3:": 3, "Level 4:": 4}
        level_notes = [entry for entry in manual if any(entry.module.startswith(prefix) for prefix in level_order)]
        timeline = sorted([*computer, *level_notes], key=lambda entry: (next((n for prefix, n in level_order.items() if entry.module.startswith(prefix)), 9), entry.timestamp))
        if timeline:
            for entry in timeline:
                where = entry.source_timestamp or (f"frame {entry.frame_number}" if entry.frame_number is not None else "source time not established")
                if entry.kind in ("detection", "video_tracking") and entry.source_timestamp:
                    where = f"clip-relative {where}"
                note_prefix = "Investigator note · " if entry.kind == "manual_note" else ""
                module_label = display_module_label(entry.module)
                evidence_label = f"{entry.evidence_id} · {module_label}" if entry.evidence_id else module_label
                report += f"- **{note_prefix}{evidence_label} · {where}:** {entry.observation or entry.value}. Source: {entry.source_filename or 'not recorded'}.\n"
        else:
            report += "_No computer-vision observations have been recorded; no timeline facts are inferred._\n"

        report += "## Direct observations\n"
        if direct:
            for entry in direct:
                report += f"- **{entry.evidence_id + ' · ' if entry.evidence_id else ''}{display_module_label(entry.module)}:** {entry.observation or entry.description or entry.value}"
                if entry.source_filename:
                    report += f" _({entry.source_filename})_"
                if entry.limitation:
                    report += f" Limitation: {entry.limitation}"
                report += "\n"
        else:
            report += "_No direct observations have been recorded._\n"

        report += "\n## Supporting documentary evidence\n"
        if documentary:
            for entry in documentary:
                report += f"- **{entry.source_filename or entry.module}:** {entry.observation or entry.value}"
                if entry.source_path:
                    report += f" (source: `{entry.source_path}`)"
                report += "\n"
        else:
            report += "_No documentary record has been interpreted in the notebook._\n"
        other_notes = [entry for entry in manual if entry not in documentary]
        report += "\n## Investigator-entered notes\n"
        if other_notes:
            for entry in other_notes:
                report += f"- **{display_module_label(entry.module)}:** {entry.observation or entry.value}"
                if entry.source_filename:
                    report += f" (source: {entry.source_filename})"
                report += f" Interpretation: {entry.interpretation or 'not recorded'}. Uncertainty: {entry.uncertainty}.\n"
        else:
            report += "_No additional investigator notes have been recorded._\n"
        supplied_records = documentary_records()
        if supplied_records:
            report += "\n### Supplied record files\n"
            for record in supplied_records:
                report += f"- **{record['filename']}** (`{record['path']}`):\n\n  ```text\n  {record['content'].strip()}\n  ```\n"
        missing = missing_manifest_assets()
        if missing:
            report += "\n### Manifest files missing from this checkout\n"
            report += "The following evidence was not available for this report; its contents are not assumed:\n"
            report += "\n".join(f"- **{item['id']} · `{item['filename']}`:** {item['purpose']}" for item in missing) + "\n"

        report += "\n## Computer-vision results\n"
        if computer:
            for entry in computer:
                report += f"- **{entry.evidence_id + ' · ' if entry.evidence_id else ''}{display_module_label(entry.module)}:** {entry.observation or entry.value}. Method: {entry.technique or entry.kind}."
                if entry.source_filename:
                    report += f" Source: {entry.source_filename}."
                if entry.source_timestamp:
                    report += f" Source time/frame: {entry.source_timestamp}."
                elif entry.frame_number is not None:
                    report += f" Frame: {entry.frame_number}."
                if entry.output_path:
                    report += f" Saved output: `{entry.output_path}`."
                if entry.source_path:
                    report += f" Source file: `{entry.source_path}`."
                if entry.limitation:
                    report += f" Limitation: {entry.limitation}"
                report += "\n"
        else:
            report += "_No computer-vision results have been recorded._\n"

        report += "\n## Interpretations\n"
        interpretations = [entry for entry in entries if entry.interpretation and entry.module != "Final Deduction"]
        report += "\n".join(f"- **{entry.module}:** {entry.interpretation} ({entry.uncertainty} uncertainty)" for entry in interpretations) or "_No interpretations have been recorded._"
        report += "\n\n## Unresolved questions\n"
        questions = [entry for entry in entries if entry.unresolved_question.strip()]
        report += "\n".join(f"- **{entry.module}:** {entry.unresolved_question.strip()}" for entry in questions) or "_No unresolved questions have been recorded._"
        report += "\n\n## Participant's Final Deduction\n"
        report += f"- **Working hypothesis:** {self.suspect_choice.currentText().strip() or '_Not submitted._'}\n"
        report += f"- **Supporting clues:** {self.clues_edit.toPlainText().strip() or '_Not submitted._'}\n"
        report += f"- **Purge-job interpretation:** {self.purge_edit.toPlainText().strip() or '_Not submitted._'}\n"
        report += f"- **Unresolved:** {self.uncertain_edit.toPlainText().strip() or '_Not submitted._'}\n\n"
        report += "A working hypothesis is not proof of guilt. A detected object, colour region, or trajectory does not identify a person. An account named in a log does not by itself establish who physically operated the system.\n"
        self.report_text.setMarkdown(report)

    def submit_deduction(self):
        suspect = self.suspect_choice.currentText().strip()
        clues = self.clues_edit.toPlainText().strip()
        purge = self.purge_edit.toPlainText().strip()
        unresolved = self.uncertain_edit.toPlainText().strip()
        clue_lines = [line for line in clues.splitlines() if line.strip()]
        sourced_lines = [line for line in clue_lines if ":" in line]
        sources = {line.split(":", 1)[0].strip().lower() for line in sourced_lines}
        if not suspect or len(sourced_lines) < 3 or len(sources) < 3 or not purge or not unresolved:
            self.log_message.emit("Enter a working hypothesis, three clues with distinct source labels, a record assessment (or say missing), and an unresolved question.", "warning")
            return
        registry.add(EvidenceEntry(
            module="Final Deduction", kind="final_report",
            value={"working_hypothesis": suspect, "clues": clues, "purge_job": purge, "uncertainty": unresolved},
            technique="Participant synthesis of notebook entries and supplied records",
            observation=f"Working hypothesis: {suspect}. Supporting clues: {clues}",
            interpretation=f"Purge-job assessment: {purge}", uncertainty="High",
            unresolved_question=unresolved, completion_mode="independent",
            limitation="An evidence-based hypothesis is not proof of guilt; a model result or account attribution alone does not establish a person's actions.",
        ))
        self.generate_report()
        self.log_message.emit("Final synthesis recorded as a working hypothesis, not proof of guilt.", "success")

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Final Report", "last_chai_case_report.md", "Markdown Files (*.md);;Text Files (*.txt)")
        if not path:
            return
        content = self.report_text.toMarkdown().rstrip()
        notes = self.notes_edit.toPlainText().strip()
        if notes:
            content += "\n\n## Investigator notes\n" + notes
        try:
            Path(path).write_text(content + "\n", encoding="utf-8")
            self.log_message.emit(f"Report exported to {path}", "success")
        except OSError as exc:
            self.log_message.emit(f"Report export failed: {exc}", "error")
