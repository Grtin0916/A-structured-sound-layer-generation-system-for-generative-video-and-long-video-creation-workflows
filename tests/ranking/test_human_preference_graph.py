import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.human_preference_graph import build_human_graph


def pair(pair_id, left, right, *, hidden=False, audit=False):
    return {
        "pair_id": pair_id,
        "case_id": "case-1",
        "left_digest": left,
        "right_digest": right,
        "is_hidden_repeat": hidden,
        "is_audit_pair": audit,
    }


def label(pair_id, preference, submitted=True):
    return {
        "pair_id": pair_id,
        "overall_preference": preference,
        "submitted": submitted,
    }


class HumanPreferenceGraphTests(unittest.TestCase):
    def test_empty_labels_are_insufficient(self):
        graph = build_human_graph([], [pair("p1", "a", "b")])
        self.assertEqual(graph["cases"][0]["reference_status"], "INSUFFICIENT_COMPARISON")

    def test_left_preference_creates_direction(self):
        private = [pair("p1", "a", "b")]
        graph = build_human_graph([label("p1", "LEFT")], private)
        self.assertEqual(graph["cases"][0]["edges"], [{"preferred": "a", "not_preferred": "b"}])

    def test_right_preference_reverses_direction(self):
        private = [pair("p1", "a", "b")]
        graph = build_human_graph([label("p1", "RIGHT")], private)
        self.assertEqual(graph["cases"][0]["edges"], [{"preferred": "b", "not_preferred": "a"}])

    def test_tie_does_not_create_direction(self):
        private = [pair("p1", "a", "b")]
        graph = build_human_graph([label("p1", "TIE")], private)
        self.assertEqual(graph["cases"][0]["tie_count"], 1)
        self.assertEqual(graph["cases"][0]["decisive_edge_count"], 0)

    def test_unjudgeable_does_not_create_direction(self):
        private = [pair("p1", "a", "b")]
        graph = build_human_graph([label("p1", "UNJUDGEABLE")], private)
        self.assertEqual(graph["cases"][0]["unjudgeable_count"], 1)

    def test_hidden_repeat_is_not_a_reference_edge(self):
        private = [pair("p1", "a", "b", hidden=True)]
        graph = build_human_graph([label("p1", "LEFT")], private)
        self.assertEqual(graph["cases"][0]["observed_edge_count"], 0)

    def test_audit_pair_is_not_a_reference_edge(self):
        private = [pair("p1", "a", "b", audit=True)]
        graph = build_human_graph([label("p1", "LEFT")], private)
        self.assertEqual(graph["cases"][0]["observed_edge_count"], 0)

    def test_unique_top_requires_reachability(self):
        private = [pair("p1", "a", "b"), pair("p2", "a", "c")]
        labels = [label("p1", "LEFT"), label("p2", "LEFT")]
        graph = build_human_graph(labels, private)
        self.assertEqual(graph["cases"][0]["reference_status"], "UNIQUE_TOP")
        self.assertEqual(graph["cases"][0]["human_top_candidates"], ["a"])

    def test_partial_order_does_not_invent_single_winner(self):
        private = [pair("p1", "a", "b"), pair("p2", "a", "c")]
        graph = build_human_graph([label("p1", "LEFT")], private)
        self.assertEqual(graph["cases"][0]["reference_status"], "PARTIAL_ORDER")
        self.assertGreater(len(graph["cases"][0]["human_top_candidates"]), 1)

    def test_cycle_is_preserved(self):
        private = [
            pair("p1", "a", "b"),
            pair("p2", "b", "c"),
            pair("p3", "c", "a"),
        ]
        labels = [label("p1", "LEFT"), label("p2", "LEFT"), label("p3", "LEFT")]
        graph = build_human_graph(labels, private)
        self.assertEqual(graph["cases"][0]["reference_status"], "PREFERENCE_CYCLE")
        self.assertEqual(graph["cases"][0]["cycle_count"], 1)

    def test_duplicate_content_judgment_is_counted_once(self):
        private = [pair("p1", "a", "b"), pair("p2", "b", "a")]
        labels = [label("p1", "LEFT"), label("p2", "RIGHT")]
        graph = build_human_graph(labels, private)
        self.assertEqual(graph["cases"][0]["observed_edge_count"], 1)

    def test_summary_never_reports_fabricated_winner(self):
        graph = build_human_graph([], [pair("p1", "a", "b")])
        self.assertEqual(graph["summary"]["humanWinnerFabricationCount"], 0)
