import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from workshop_code.vision import ROOT, enhance, read_image, red_mask, red_regions, write_image
from workshop_code.level4.solution import crossed_line
from workshop_code.level2.solution import run_detector
sys.path.insert(0, str(ROOT / "argus_app"))
from labs.solutions.level3_colour import process_color_segmentation
from argus_app.evidence.registry import EvidenceEntry, EvidenceRegistry

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".test-tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 120, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (20, 25), (55, 70), (0, 0, 255), -1)

    def test_enhance_returns_real_stages(self):
        stages = enhance(self.image)
        self.assertEqual(set(stages), {"gray", "clahe", "threshold"})
        self.assertEqual(stages["gray"].shape, (100, 120))
        self.assertTrue(set(np.unique(stages["threshold"])).issubset({0, 255}))
        self.assertGreaterEqual(len(np.unique(stages["threshold"])), 1)

    def test_red_mask_covers_both_hue_ends(self):
        hsv = np.zeros((20, 40, 3), dtype=np.uint8)
        hsv[:, :20] = (2, 220, 220)
        hsv[:, 20:] = (176, 220, 220)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mask = red_mask(bgr)
        self.assertEqual(cv2.countNonZero(mask), mask.size)

    def test_level3_returns_clean_mask_and_all_area_filtered_candidates(self):
        results = process_color_segmentation(self.image, 170, 10, 70, 255, 40, 255, morph_kernel=3)
        self.assertGreater(cv2.countNonZero(results["mask_cleaned"]), 0)
        self.assertEqual(len(results["candidate_contours"]), 1)
        self.assertIsNotNone(results["result"])

    def test_region_centroid_and_crossing_logic(self):
        mask, regions = red_regions(self.image, minimum_area=20)
        self.assertGreater(cv2.countNonZero(mask), 0)
        self.assertEqual(len(regions), 1)
        self.assertTrue(35 <= regions[0]["centroid"][0] <= 40)
        self.assertTrue(crossed_line(40, 60, 50))
        self.assertFalse(crossed_line(40, 45, 50))

    def test_missing_asset_and_output_handling(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp:
            missing = Path(temp) / "not_supplied.png"
            with self.assertRaises(FileNotFoundError):
                read_image(missing)
            output = write_image(Path(temp) / "result.png", self.image)
            self.assertTrue(output.is_file())
            self.assertIsNotNone(cv2.imread(str(output)))

    def test_detector_reports_missing_model(self):
        with patch.dict(os.environ, {"ARGUS_YOLO_WEIGHTS": "missing-test-model.pt"}, clear=True):
            prediction, message = run_detector(self.image, "unused.png")
        self.assertIsNone(prediction)
        self.assertIn("unavailable", message.lower())

    def test_local_yolo_inference_when_model_is_available(self):
        weights = ROOT / "yolo11n.pt"
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            self.skipTest("Ultralytics is not installed")
        if not weights.is_file():
            self.skipTest("No local YOLO weights were supplied")
        with patch.dict(os.environ, {}, clear=True):
            prediction, note = run_detector(self.image, ROOT / "in_memory_test.png")
        self.assertIsNotNone(prediction, note)
        annotated, detections = prediction
        self.assertEqual(annotated.shape, self.image.shape)
        self.assertIsInstance(detections, list)

    def test_notebook_persistence_keeps_id_and_checkpoint_mode(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temp:
            first = EvidenceRegistry()
            first.storage_path = Path(temp) / "notebook.json"
            first.add(EvidenceEntry(module="Level 1", kind="checkpoint", value="used", evidence_id="E-01", completion_mode="checkpoint", limitation="Assisted.", source_path="assets/evidence/a.png", output_path="outputs/level1/result.png"))
            second = EvidenceRegistry()
            second.load(first.storage_path)
            self.assertEqual(second.all()[0].evidence_id, "E-01")
            self.assertEqual(second.all()[0].completion_mode, "checkpoint")
            self.assertEqual(second.all()[0].limitation, "Assisted.")
            self.assertEqual(second.all()[0].source_path, "assets/evidence/a.png")
            self.assertEqual(second.all()[0].output_path, "outputs/level1/result.png")


if __name__ == "__main__":
    unittest.main()
