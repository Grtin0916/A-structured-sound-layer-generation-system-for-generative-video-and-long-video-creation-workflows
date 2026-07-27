from array import array
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundlayer.repair.signal_repair import (
    adaptive_headroom,
    conservative_micro_declip,
    silence_aware_trim,
    smooth_region_gain,
)


class SignalRepairTest(unittest.TestCase):
    def test_headroom_is_noop_below_ceiling(self):
        source = array("h", [0, 1000, -1000])
        output, meta = adaptive_headroom(source, 0.95)
        self.assertEqual(output, source)
        self.assertFalse(meta["recovered_clipped_waveform"])

    def test_headroom_reduces_peak_without_nonfinite_values(self):
        output, _ = adaptive_headroom(array("h", [32767, -32768]), 0.90)
        self.assertLessEqual(max(map(abs, output)), round(0.90 * 32768))
        self.assertTrue(all(math.isfinite(value) for value in output))

    def test_micro_declip_changes_short_run(self):
        source = array("h", [1000, 32767, 32767, 2000])
        output, meta = conservative_micro_declip(source)
        self.assertNotEqual(output, source)
        self.assertGreater(meta["changed_sample_ratio"], 0)

    def test_micro_declip_blocks_long_run(self):
        source = array("h", [1000] + [32767] * 8 + [2000])
        output, meta = conservative_micro_declip(source, max_run=6)
        self.assertEqual(output, source)
        self.assertEqual(meta["blocked_long_runs"], 1)

    def test_trim_preserves_all_silent_input(self):
        source = array("h", [0] * 100)
        output, meta = silence_aware_trim(source, 100, 1)
        self.assertEqual(output, source)
        self.assertTrue(meta["all_silent_preserved"])

    def test_trim_removes_edges_and_preserves_channels(self):
        source = array("h", [0, 0] * 20 + [2000, -2000] * 10 + [0, 0] * 20)
        output, _ = silence_aware_trim(source, 100, 2, padding_ms=0)
        self.assertEqual(len(output), 20)

    def test_region_gain_does_not_change_outside_window(self):
        source = array("h", [1000] * 100)
        output, _ = smooth_region_gain(source, 100, 1, 0.2, 0.8, 0.5, 10)
        self.assertEqual(output[:20], source[:20])
        self.assertEqual(output[80:], source[80:])
        self.assertLess(output[50], source[50])

    def test_region_gain_clamps_boundary_window(self):
        source = array("h", [1000] * 10)
        output, _ = smooth_region_gain(source, 10, 1, -1, 5, 1.2, 100)
        self.assertEqual(len(output), len(source))
        self.assertTrue(all(-32768 <= value <= 32767 for value in output))


if __name__ == "__main__":
    unittest.main()
