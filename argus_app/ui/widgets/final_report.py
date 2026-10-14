"""
Final Report Widget - Case: THE LAST CHAI
Distinguishes directly observed evidence, computer vision results, participant interpretations, and unresolved questions.
Does NOT automatically declare guilt.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QFrame, QPushButton, QTextEdit, QScrollArea, QFileDialog)
from PySide6.QtCore import Qt, Signal
from evidence.registry import registry
from ui.style import COLORS, FONTS, SPACING, RADIUS
from pathlib import Path


class FinalReportWidget(QWidget):
    """Final Forensic Report view."""

    log_message = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.generate_report()

    def setup_ui(self):
        """Set up UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

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
        title = QLabel("FORENSIC INVESTIGATION REPORT")
        title.setStyleSheet(f"font-size: {FONTS['size_xl']}; font-weight: 700; color: {COLORS['accent_teal']};")
        title_layout.addWidget(title)

        subtitle = QLabel("Case: The Lab 4B Incident | ARGUS Investigation Record")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_sm']};")
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)

        header_layout.addStretch()

        refresh_btn = QPushButton("Refresh Report")
        refresh_btn.clicked.connect(self.generate_report)
        header_layout.addWidget(refresh_btn)

        export_btn = QPushButton("Export Report (Markdown)...")
        export_btn.setObjectName("primaryButton")
        export_btn.clicked.connect(self.export_report)
        header_layout.addWidget(export_btn)

        main_layout.addWidget(header_frame)

        # Disclaimer Banner (Red / Warning)
        disclaimer = QLabel(
            "⚠ FORENSIC NOTICE: Computer vision outputs and automated video tracking provide "
            "investigative support only. They do not automatically establish suspect guilt or legal proof."
        )
        disclaimer.setStyleSheet(f"""
            background-color: #3f1212;
            color: #fca5a5;
            border: 1px solid {COLORS['error']};
            border-radius: {RADIUS['md']};
            padding: {SPACING['md']};
            font-weight: 600;
            font-size: {FONTS['size_base']};
        """)
        disclaimer.setWordWrap(True)
        main_layout.addWidget(disclaimer)

        # Report display (scrollable / rich text)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS['md']};
                padding: {SPACING['lg']};
                font-family: {FONTS['family_ui']};
                font-size: {FONTS['size_base']};
                line-height: 1.6;
            }}
        """)
        main_layout.addWidget(self.report_text, 1)

        notes_title = QLabel("Investigator Notes")
        notes_title.setObjectName("sectionHeading")
        main_layout.addWidget(notes_title)
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Add investigator notes. Refreshing the report will keep these notes.")
        self.notes_edit.setMaximumHeight(150)
        self.notes_edit.setStyleSheet(f"QTextEdit {{ background-color: {COLORS['bg_panel']}; color: {COLORS['text_primary']}; border: 1px solid {COLORS['border']}; border-radius: {RADIUS['md']}; padding: {SPACING['md']}; }}")
        main_layout.addWidget(self.notes_edit)

    def generate_report(self):
        """Generate structured markdown report from registered evidence."""
        entries = registry.all()

        observed = [e for e in entries if e.kind in ("image_observation", "manual_note", "direct_observation")]
        cv_results = [e for e in entries if e.kind not in ("image_observation", "manual_note", "direct_observation")]

        report_status = "Evidence review in progress" if not entries else "Evidence summary compiled"
        report_md = f"""# FORENSIC INVESTIGATION REPORT
**Case:** The Lab 4B Incident
**System:** ARGUS desktop application
**Status:** {report_status}

---

## 1. Directly Observed Evidence
"""
        if observed:
            for idx, e in enumerate(observed, 1):
                report_md += f"### Evidence #{idx}: [{e.module}]\n"
                report_md += f"- **Technique Used:** {e.technique or e.kind}\n"
                report_md += f"- **Direct Observation:** {e.observation or e.description or e.value}\n"
                if e.source_filename:
                    source_time = f" · {e.source_timestamp}" if e.source_timestamp else ""
                    report_md += f"- **Source:** {e.source_filename}{source_time}\n"
                report_md += f"- **Timestamp:** {e.timestamp}\n\n"
        else:
            report_md += "_No direct observations have been recorded._\n\n"

        report_md += """## 2. Computer Vision Analysis Results
"""
        if cv_results:
            for e in cv_results:
                report_md += f"- **[{e.module}]** {e.technique}: {e.observation or e.value} (Uncertainty: {e.uncertainty})\n"
                if e.source_filename:
                    source_time = f" · {e.source_timestamp}" if e.source_timestamp else ""
                    report_md += f"  - Source: {e.source_filename}{source_time}\n"
        else:
            report_md += "_No computer vision results have been recorded._\n"

        report_md += """

## 3. Investigator Interpretations
"""
        interps = [e for e in entries if e.interpretation and e.interpretation != "Under review"]
        if interps:
            for e in interps:
                report_md += f"- **[{e.module}]**: {e.interpretation} (Uncertainty: {e.uncertainty})\n"
        else:
            report_md += "- _No investigator interpretations have been recorded._\n"

        report_md += "\n\n## 4. Unresolved Questions & Investigation Gaps\n"
        questions = [e for e in entries if e.unresolved_question.strip()]
        if questions:
            for e in questions:
                report_md += f"- **[{e.module}]** {e.unresolved_question.strip()}\n"
        else:
            report_md += "_No unresolved questions have been recorded._\n"
        self.report_text.setMarkdown(report_md)
        self.log_message.emit("Final Report refreshed from Evidence Notebook", "info")

    def export_report(self):
        """Export report to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Forensic Report", "lab4b_case_report.md", "Markdown Files (*.md);;Text Files (*.txt)"
        )
        if file_path:
            try:
                content = self.report_text.toMarkdown().rstrip()
                notes = self.notes_edit.toPlainText().strip()
                content += "\n\n## 5. Investigator Notes\n"
                content += notes if notes else "_No investigator notes have been recorded._"
                content += "\n"
                Path(file_path).write_text(content, encoding="utf-8")
                self.log_message.emit(f"Exported Forensic Report to {file_path}", "success")
            except Exception as e:
                self.log_message.emit(f"Export failed: {e}", "error")
