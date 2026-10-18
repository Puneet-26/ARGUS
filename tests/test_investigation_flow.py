import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "argus_app"))

from PySide6.QtWidgets import QApplication
from evidence.registry import EvidenceEntry, registry
from ui.widgets.dashboard import DashboardWidget
from ui.widgets.final_report import FinalReportWidget


class InvestigationFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.saved_entries = registry._entries
        self.saved_path = registry.storage_path
        registry._entries = []
        registry.storage_path = ROOT / ".test-tmp" / "flow-notebook.json"

    def tearDown(self):
        registry._entries = self.saved_entries
        registry.storage_path = self.saved_path
        (ROOT / ".test-tmp" / "flow-notebook.json").unlink(missing_ok=True)

    def test_checkpoint_then_independent_completion_labels(self):
        dashboard = DashboardWidget()
        dashboard.sync_progress_from_registry()
        self.assertEqual(dashboard.stage_cards["Eyes Online"].status_label.text(), "NOT STARTED")
        registry._entries.append(EvidenceEntry(module="Checkpoint Level 1: Sight Repair", kind="checkpoint", value="used", completion_mode="checkpoint"))
        dashboard.sync_progress_from_registry()
        self.assertEqual(dashboard.stage_cards["Eyes Online"].status_label.text(), "CHECKPOINT USED")
        registry._entries.append(EvidenceEntry(module="Level 1: Eyes Online", kind="image_observation", value="independent result recorded", completion_mode="independent"))
        dashboard.sync_progress_from_registry()
        self.assertEqual(dashboard.stage_cards["Eyes Online"].status_label.text(), "COMPLETED")
        dashboard.close()

    def test_final_deduction_requires_three_clues_and_is_reported(self):
        report = FinalReportWidget()
        report.suspect_choice.setEditText("A working hypothesis based on the combined evidence")
        report.clues_edit.setPlainText("E-01: timestamp view\nE-03: red-region candidates\nE-04: clip-relative route")
        report.purge_edit.setPlainText("No access or session records were supplied in this checkout.")
        report.uncertain_edit.setPlainText("The records do not establish who physically operated the terminal.")
        report.submit_deduction()
        saved = next(entry for entry in registry.all() if entry.module == "Final Deduction")
        self.assertEqual(saved.value["working_hypothesis"], "A working hypothesis based on the combined evidence")
        self.assertIn("account attribution", saved.limitation)
        report.generate_report()
        markdown = report.report_text.toMarkdown()
        self.assertIn("Participant's Final Deduction", markdown)
        self.assertIn("not proof of guilt", markdown)
        self.assertIn("Manifest files missing from this checkout", markdown)
        report.close()


if __name__ == "__main__":
    unittest.main()
