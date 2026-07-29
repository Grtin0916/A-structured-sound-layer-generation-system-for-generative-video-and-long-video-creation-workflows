import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.active_review import (
    acquisition_score,
    canonical_key,
    existing_content_keys,
    unlabeled_pairs,
)
from soundlayer.ranking.ranker_ablation import (
    DSS_INTERACTION_NAMES,
    dss_interactions,
    hard_guard,
    rule_score,
    selective_hybrid,
)


def features(**overrides):
    values = {
        "event_coverage": 0.8,
        "priority_weighted_coverage": 0.9,
        "onset_error_ms": 50.0,
        "tolerance_violation_count": 0.0,
        "event_window_energy": 0.4,
        "outside_window_energy": 0.1,
        "clip_ratio": 0.0,
        "silence_ratio": 0.1,
        "peak_abs": 0.8,
        "duration_error_ms": 50.0,
        "changed_sample_ratio": 0.0,
        "repair_action_count": 0.0,
        "source_reliability": 1.0,
    }
    values.update(overrides)
    return values


class DssInteractionTests(unittest.TestCase):
    def test_only_four_allowlisted_interactions(self):
        result = dss_interactions(features())
        self.assertEqual(tuple(result), DSS_INTERACTION_NAMES)

    def test_interactions_contain_no_identity_names(self):
        text = " ".join(DSS_INTERACTION_NAMES)
        for token in ("case", "strategy", "path", "digest", "model"):
            self.assertNotIn(token, text)

    def test_inverse_onset_is_finite(self):
        value = dss_interactions(features(onset_error_ms=0.0))[
            "priority_coverage_x_inverse_onset"
        ]
        self.assertGreater(value, 0.0)

    def test_rule_score_rewards_coverage(self):
        self.assertGreater(
            rule_score(features(event_coverage=1.0)),
            rule_score(features(event_coverage=0.0)),
        )


class SelectiveHybridTests(unittest.TestCase):
    def test_rejected_publish_decision_blocks(self):
        guard = hard_guard(features(), "REPAIR_REJECTED")
        self.assertFalse(guard["passed"])
        self.assertIn("PUBLISH_DECISION_BLOCKED", guard["reasons"])

    def test_clip_ratio_blocks(self):
        guard = hard_guard(features(clip_ratio=0.01))
        self.assertIn("CLIP_RATIO", guard["reasons"])

    def test_duration_mismatch_blocks(self):
        guard = hard_guard(features(duration_error_ms=2000))
        self.assertIn("DURATION_MISMATCH", guard["reasons"])

    def test_guard_has_priority_over_model(self):
        decision = selective_hybrid("a", "a", 0.9, hard_guard(features(clip_ratio=0.1)))
        self.assertEqual(decision["recommendation_status"], "PUBLISH_BLOCKED")

    def test_missing_model_fails_closed(self):
        decision = selective_hybrid(
            "a", "a", 0.9, hard_guard(features()), model_available=False
        )
        self.assertEqual(decision["recommendation_status"], "ABLATION_DATA_BLOCKED")

    def test_low_margin_defers(self):
        decision = selective_hybrid("a", "a", 0.05, hard_guard(features()))
        self.assertEqual(decision["recommendation_status"], "NEEDS_HUMAN_REVIEW")

    def test_ood_defers(self):
        decision = selective_hybrid("a", "a", 0.9, hard_guard(features()), ood=True)
        self.assertEqual(decision["defer_reason"], "OOD")

    def test_disagreement_uses_rule_fallback(self):
        decision = selective_hybrid("rule", "learned", 0.9, hard_guard(features()))
        self.assertEqual(decision["recommendation_status"], "RULE_FALLBACK")
        self.assertEqual(decision["selected_candidate"], "rule")

    def test_agreement_can_recommend(self):
        decision = selective_hybrid("a", "a", 0.9, hard_guard(features()))
        self.assertEqual(decision["recommendation_status"], "RANKER_RECOMMENDED")


class ActiveReviewTests(unittest.TestCase):
    def test_canonical_key_is_unordered(self):
        self.assertEqual(canonical_key("c", "a", "b"), canonical_key("c", "b", "a"))

    def test_existing_keys_include_hidden_and_audit_content(self):
        rows = [{"case_id": "c", "left_digest": "a", "right_digest": "b"}]
        self.assertIn(canonical_key("c", "a", "b"), existing_content_keys(rows))

    def test_unlabeled_pairs_stay_within_case(self):
        candidates = [
            {"case_id": "a", "artifact_digest": "1", "artifact_path": "1.wav"},
            {"case_id": "a", "artifact_digest": "2", "artifact_path": "2.wav"},
            {"case_id": "b", "artifact_digest": "3", "artifact_path": "3.wav"},
        ]
        pairs = unlabeled_pairs(candidates, set())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["case_id"], "a")

    def test_same_digest_pair_is_rejected(self):
        candidates = [
            {"case_id": "a", "artifact_digest": "1", "artifact_path": "1.wav"},
            {"case_id": "a", "artifact_digest": "1", "artifact_path": "copy.wav"},
        ]
        self.assertEqual(unlabeled_pairs(candidates, set()), [])

    def test_existing_pair_is_rejected(self):
        candidates = [
            {"case_id": "a", "artifact_digest": "1", "artifact_path": "1.wav"},
            {"case_id": "a", "artifact_digest": "2", "artifact_path": "2.wav"},
        ]
        existing = {canonical_key("a", "1", "2")}
        self.assertEqual(unlabeled_pairs(candidates, existing), [])

    def test_uncertainty_is_highest_at_half(self):
        self.assertGreater(acquisition_score(0.5), acquisition_score(0.9))

    def test_disagreement_and_closure_raise_score(self):
        base = acquisition_score(0.5)
        self.assertEqual(acquisition_score(0.5, True, True), base + 2.0)


if __name__ == "__main__":
    unittest.main()
