import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "argus_app"))

from PySide6.QtWidgets import QApplication, QTabWidget
from ui.widgets.lab1_image import (
    CLOCK_ZOOM,
    EFFECTS,
    ImagePanel,
    Lab1ImageWidget,
    apply_effect,
    detect_clock_region,
    process_clock_region,
)


class Level1ImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widget = Lab1ImageWidget()

    def tearDown(self):
        self.widget.close()

    @staticmethod
    def sample():
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:] = (24, 31, 34)
        cv2.putText(image, "12:34", (1100, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        return image

    def test_three_effects_are_real_processing_and_keep_source_unchanged(self):
        image = self.sample()
        source_copy = image.copy()
        gray = apply_effect(image, "Grayscale")
        clahe = apply_effect(image, "CLAHE")
        threshold = apply_effect(image, "Threshold")
        self.assertEqual(gray.shape, (720, 1280))
        self.assertEqual(clahe.shape, (720, 1280))
        self.assertEqual(threshold.shape, (720, 1280))
        self.assertEqual(set(np.unique(threshold)), {0, 255})
        self.assertFalse(np.array_equal(gray, clahe))
        np.testing.assert_array_equal(image, source_copy)

    def test_clock_region_is_detected_from_full_frame_and_zoomed(self):
        image = self.sample()
        x, y, width, height = detect_clock_region(image)
        self.assertGreater(x, 1000)
        self.assertGreater(y, 600)
        self.assertGreater(width, 30)
        self.assertGreater(height, 10)
        self.assertLess(x + width, image.shape[1])
        self.assertLess(y + height, image.shape[0])

    def test_image_without_clock_region_is_rejected_clearly(self):
        with self.assertRaisesRegex(ValueError, "No clear clock"):
            detect_clock_region(np.zeros((720, 1280, 3), dtype=np.uint8))
        caption = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(caption, "EXIT", (1100, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        with self.assertRaisesRegex(ValueError, "No clear clock"):
            detect_clock_region(caption)
        self.widget.set_evidence_name("level1_cctv_dark.png")
        self.assertFalse(self.widget.load_frame(np.zeros((720, 1280, 3), dtype=np.uint8)))
        self.assertIn("clock", self.widget.status_label.text().lower())

    def test_general_upload_uses_full_image_without_clock_crop(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        self.assertTrue(self.widget.load_frame(image))
        self.assertIsNone(self.widget.clock_region)
        self.assertEqual(self.widget.canvas_orig._source_pixmap.size().width(), 120)
        self.assertEqual(self.widget.canvas_proc._source_pixmap.size().width(), 120)

    def test_interface_has_only_two_image_panels_and_three_effects(self):
        self.assertEqual(tuple(self.widget.effect_selector.itemText(i) for i in range(self.widget.effect_selector.count())), EFFECTS)
        self.assertEqual(len(self.widget.findChildren(ImagePanel)), 2)
        self.assertEqual(len(self.widget.findChildren(QTabWidget)), 0)
        self.assertEqual(self.widget.canvas_orig.objectName(), "")
        self.assertEqual(self.widget.canvas_proc.objectName(), "")

    def test_upload_and_effect_change_update_processed_image_only(self):
        source = self.sample()
        self.widget.set_evidence_name("level1_cctv_dark.png")
        self.assertTrue(self.widget.load_frame(source))
        original_before = self.widget.raw_evidence_frame.copy()
        processed_gray = self.widget.processed_frame.copy()
        self.widget.effect_selector.setCurrentText("Threshold")
        np.testing.assert_array_equal(self.widget.processed_frame, process_clock_region(source, "Threshold")[1])
        self.assertFalse(np.array_equal(processed_gray, self.widget.processed_frame))
        np.testing.assert_array_equal(self.widget.raw_evidence_frame, original_before)
        x, y, width, height = self.widget.clock_region
        self.assertEqual(self.widget.canvas_orig._source_pixmap.size().width(), width * CLOCK_ZOOM)
        self.assertEqual(self.widget.canvas_proc._source_pixmap.size().height(), height * CLOCK_ZOOM)
        self.assertEqual(self.widget.canvas_orig._source_pixmap.size(), self.widget.canvas_proc._source_pixmap.size())

    def test_invalid_input_shows_clear_error_without_crashing(self):
        self.assertFalse(self.widget.load_frame(np.array([], dtype=np.uint8)))
        result = self.widget.process_frame(np.array([], dtype=np.uint8))
        self.assertFalse(result["success"])
        self.assertIn("valid", self.widget.status_label.text().lower())

    def test_reset_clears_both_panels_and_returns_to_grayscale(self):
        self.widget.load_frame(self.sample())
        self.widget.effect_selector.setCurrentText("CLAHE")
        self.widget.reset()
        self.assertIsNone(self.widget.raw_evidence_frame)
        self.assertIsNone(self.widget.processed_frame)
        self.assertEqual(self.widget.effect_selector.currentText(), "Grayscale")
        self.assertFalse(self.widget.effect_selector.isEnabled())
        self.assertIsNone(self.widget.canvas_orig._source_pixmap)
        self.assertIsNone(self.widget.canvas_proc._source_pixmap)


if __name__ == "__main__":
    unittest.main()
