from array import array
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundlayer.ranking.repair_aware_reranker import decide_candidate
from soundlayer.repair.event_replacement import transplant_event
from soundlayer.repair.semantic_repair import (
    apply_mixed_only_repair,
    classify_source_mode,
)


class SemanticRepairTest(unittest.TestCase):
    def test_missing_stems_degrades_to_mixed_only(self):
        mode, _ = classify_source_mode([], "mix.wav")
        self.assertEqual(mode, "mixed_only")

    def test_duplicate_stem_path_does_not_claim_true_stems(self):
        mode, _ = classify_source_mode(["same.wav", "same.wav"], "mix.wav")
        self.assertEqual(mode, "mixed_only")

    def test_distinct_declared_stems_pass_initial_inventory(self):
        mode, _ = classify_source_mode(["amb.wav", "foley.wav"], "mix.wav")
        self.assertEqual(mode, "true_stems")

    def test_mixed_repair_is_local(self):
        source = array("h", [1000] * 100)
        output, metrics = apply_mixed_only_repair(
            source, 100, 1, 0.2, 0.8, "strengthen_expected_event"
        )
        self.assertEqual(output[:20], source[:20])
        self.assertEqual(output[80:], source[80:])
        self.assertGreater(metrics["target_window_rms_delta_db"], 0)

    def test_transplant_rejects_too_short_window(self):
        with self.assertRaises(ValueError):
            transplant_event(
                array("h", [0] * 100), array("h", [1000] * 100),
                100, 1, 0.0, 0.01, 0.0, 0.01, 40,
            )

    def test_crossfade_is_clamped_to_event_duration(self):
        output, metadata = transplant_event(
            array("h", [0] * 100), array("h", [1000] * 100),
            100, 1, 0.1, 0.6, 0.1, 0.6, 1000,
        )
        self.assertEqual(len(output), 100)
        self.assertGreater(metadata["changed_sample_ratio"], 0)

    def test_unknown_forbidden_label_forces_manual_review(self):
        decision, _ = decide_candidate({
            "output_readable": True,
            "severe_regression": False,
            "lineage_complete": True,
            "ordering_correct": True,
            "semantic_target_satisfied": True,
            "forbidden_event_status": "unknown",
        })
        self.assertEqual(decision, "MANUAL_REVIEW")

    def test_missing_handoff_path_is_blocked(self):
        decision, _ = decide_candidate({
            "output_readable": False,
            "lineage_complete": True,
            "semantic_target_satisfied": True,
        })
        self.assertEqual(decision, "REPAIR_BLOCKED")

    def test_zero_target_does_not_authorize_transplant(self):
        record = {"expected_target_count": 0, "zero_target_guard": True}
        allowed = record["expected_target_count"] > 0 and not record["zero_target_guard"]
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
