from array import array
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundlayer.repair.repair_metrics import (
    action_compatibility,
    detect_subtype,
    diagnose_audio,
)


class RepairMetricsTest(unittest.TestCase):
    def test_silence_edges_and_event_energy_are_reported(self):
        samples = array("h", [0] * 100 + [2000] * 100 + [0] * 100)
        metrics = diagnose_audio(
            samples, 100, 1, {"start_sec": 1.0, "end_sec": 2.0}
        )
        self.assertAlmostEqual(metrics["leading_silence_ms"], 1000.0)
        self.assertGreater(metrics["event_window_rms"], 0)

    def test_stereo_geometry_uses_frames(self):
        samples = array("h", [0, 0] * 20 + [1000, -1000] * 10)
        metrics = diagnose_audio(samples, 10, 2)
        self.assertAlmostEqual(metrics["duration_sec"], 3.0)

    def test_peak_without_run_is_near_ceiling(self):
        metrics = diagnose_audio(array("h", [0, 32700, 0]), 10, 1)
        self.assertEqual(detect_subtype("clipping", metrics), "peak_near_ceiling")

    def test_long_flat_top_is_blocked(self):
        metrics = diagnose_audio(array("h", [32767] * 20), 10, 1)
        subtype = detect_subtype("clipping", metrics)
        compatible, reason = action_compatibility("clipping", subtype, "attenuate_limit")
        self.assertEqual(subtype, "long_flat_top")
        self.assertFalse(compatible)
        self.assertIn("cannot", reason)


if __name__ == "__main__":
    unittest.main()
