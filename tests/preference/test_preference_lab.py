import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.preference.candidate_graph import build_pair_graph
from soundlayer.preference.pair_sampler import sample_pairs
from soundlayer.preference.review_summary import summarize
from soundlayer.preference.schema import (
    LABEL_FIELDS,
    PUBLIC_ALLOWED_FIELDS,
    parse_bool,
    validate_judgment,
)


def candidate(case, strategy, digest=None):
    return {
        "case_id": case,
        "source_case_id": f"source-{case}",
        "strategy_id": strategy,
        "artifact_path": f"audio/{case}-{strategy}.wav",
        "artifact_digest": digest or f"sha256:{case}-{strategy}",
        "video_path": f"video/{case}.mp4",
    }


def complete_inventory():
    return {
        "candidates": [
            candidate(f"case-{index:02d}", strategy)
            for index in range(12)
            for strategy in "ABCD"
        ]
    }


class SchemaTests(unittest.TestCase):
    def test_label_contract_has_pair_id(self):
        self.assertIn("pair_id", LABEL_FIELDS)

    def test_bool_parser(self):
        self.assertTrue(parse_bool("YES"))
        self.assertFalse(parse_bool("false"))

    def test_valid_non_tie_requires_reason(self):
        row = {"submitted": True, "overall_preference": "LEFT", "confidence": "4"}
        self.assertIn("reason_codes_missing", validate_judgment(row))

    def test_tie_does_not_require_reason(self):
        row = {"submitted": True, "overall_preference": "TIE", "confidence": "3"}
        self.assertEqual(validate_judgment(row), [])

    def test_invalid_confidence_is_rejected(self):
        row = {"submitted": True, "overall_preference": "TIE", "confidence": "8"}
        self.assertIn("confidence_out_of_range", validate_judgment(row))


class GraphTests(unittest.TestCase):
    def test_complete_graph_has_36_edges(self):
        graph = build_pair_graph(complete_inventory())
        self.assertEqual(graph["summary"]["edgeCount"], 36)

    def test_all_complete_cases_connected(self):
        graph = build_pair_graph(complete_inventory())
        self.assertEqual(graph["summary"]["connectedCaseGraphCount"], 12)

    def test_same_digest_edge_is_rejected(self):
        inventory = {"candidates": [candidate("c", s) for s in "ABCD"]}
        inventory["candidates"][1]["artifact_digest"] = inventory["candidates"][0][
            "artifact_digest"
        ]
        graph = build_pair_graph(inventory)
        self.assertIn("A-B", graph["cases"][0]["same_digest_edges_rejected"])
        self.assertEqual(graph["summary"]["sameDigestPairsIncluded"], 0)

    def test_missing_strategies_mark_insufficient(self):
        graph = build_pair_graph({"candidates": [candidate("c", "A"), candidate("c", "B")]})
        self.assertEqual(graph["cases"][0]["status"], "PAIR_INSUFFICIENT_VARIATION")

    def test_disconnected_case_is_not_counted(self):
        graph = build_pair_graph({"candidates": [candidate("c", "A"), candidate("c", "B")]})
        self.assertEqual(graph["summary"]["connectedCaseGraphCount"], 0)


class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = complete_inventory()
        cls.graph = build_pair_graph(cls.inventory)
        cls.private, cls.public, cls.summary = sample_pairs(
            cls.inventory, cls.graph, 36, 8, 4, 20260727
        )

    def test_expected_judgment_count(self):
        self.assertEqual(self.summary["judgmentCount"], 48)

    def test_expected_pair_roles(self):
        self.assertEqual(self.summary["uniquePairCount"], 36)
        self.assertEqual(self.summary["hiddenRepeatCount"], 8)
        self.assertEqual(self.summary["auditPairCount"], 4)

    def test_all_cases_covered(self):
        self.assertEqual(self.summary["caseCoverage"], 12)

    def test_public_fields_are_allowlisted(self):
        self.assertTrue(all(set(row) == PUBLIC_ALLOWED_FIELDS for row in self.public))

    def test_public_manifest_has_no_strategy_words(self):
        text = str(self.public).lower()
        self.assertNotIn("strategy", text)
        self.assertNotIn("repair", text)

    def test_left_right_balancing_is_reproducible(self):
        again = sample_pairs(self.inventory, self.graph, 36, 8, 4, 20260727)
        self.assertEqual(self.private, again[0])

    def test_hidden_repeats_swap_digests(self):
        originals = {
            row["repeat_group_id"]: row
            for row in self.private
            if row["kind"] == "UNIQUE"
        }
        for repeat in (row for row in self.private if row["kind"] == "HIDDEN_REPEAT"):
            original = originals[repeat["repeat_group_id"]]
            self.assertEqual(repeat["left_digest"], original["right_digest"])
            self.assertEqual(repeat["right_digest"], original["left_digest"])

    def test_hidden_repeats_are_in_final_block(self):
        self.assertTrue(
            all(
                row["block_id"] == "block-4"
                for row in self.private
                if row["kind"] == "HIDDEN_REPEAT"
            )
        )

    def test_same_digest_pair_never_emitted(self):
        self.assertEqual(self.summary["sameDigestPairCount"], 0)


class QualityGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inventory = complete_inventory()
        graph = build_pair_graph(inventory)
        private, public, _ = sample_pairs(inventory, graph, 36, 8, 4, 20260727)
        labels = []
        for row in private:
            choice = "RIGHT" if row["kind"] == "HIDDEN_REPEAT" else "LEFT"
            labels.append(
                {
                    **row,
                    "overall_preference": choice,
                    "confidence": "4",
                    "reason_codes": "BETTER_SYNC",
                    "free_text_reason": "Event onset is easier to follow.",
                    "submitted": "true",
                }
            )
        cls.gate = summarize(private, public, labels, graph["summary"])
        cls.private = private
        cls.public = public
        cls.graph = graph

    def test_complete_data_is_training_eligible(self):
        self.assertEqual(self.gate["status"], "TRAINING_ELIGIBLE")

    def test_repeat_consistency_uses_digest_semantics(self):
        self.assertEqual(self.gate["metrics"]["hiddenRepeatConsistency"], 1.0)

    def test_gate_never_claims_inter_rater_agreement(self):
        self.assertFalse(self.gate["claimBoundary"]["interRaterAgreementClaimed"])

    def test_empty_labels_are_blocked(self):
        gate = summarize(self.private, self.public, [], self.graph["summary"])
        self.assertEqual(gate["status"], "REVIEW_QUALITY_BLOCKED")
        self.assertEqual(gate["metrics"]["judgmentCount"], 0)

    def test_final_selected_mutation_is_zero(self):
        self.assertEqual(self.gate["metrics"]["finalSelectedMutationCount"], 0)

    def test_public_blind_leak_count_is_zero(self):
        self.assertEqual(self.gate["metrics"]["blindLeakCount"], 0)


if __name__ == "__main__":
    unittest.main()
