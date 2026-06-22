import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.losses.logit_adjustment import adjust_logits, log_prior_offsets, margin_distortion


class LogitAdjustmentTest(unittest.TestCase):
    def test_offsets_follow_paper_sign(self):
        offsets = log_prior_offsets([0.25, 0.75], tau=1.0)
        self.assertAlmostEqual(offsets[0], -math.log(0.25))
        self.assertAlmostEqual(offsets[1], -math.log(0.75))

    def test_vector_adjustment(self):
        adjusted = adjust_logits([0.0, 0.0], [0.25, 0.75])
        self.assertGreater(adjusted[0], adjusted[1])

    def test_matrix_adjustment(self):
        adjusted = adjust_logits([[0.0, 0.0]], [0.25, 0.75])
        self.assertEqual(len(adjusted), 1)
        self.assertGreater(adjusted[0][0], adjusted[0][1])

    def test_margin_distortion(self):
        value = margin_distortion([0.8, 0.2], [0.5, 0.5], 0, 1, tau=1.0)
        self.assertAlmostEqual(value, -math.log((0.5 / 0.5) / (0.8 / 0.2)))


if __name__ == "__main__":
    unittest.main()
