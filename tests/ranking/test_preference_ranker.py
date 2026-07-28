import csv
import json
import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from soundlayer.ranking.leakage_guard import (
    assert_feature_safe,
    group_leakage,
    identity_features,
)
from soundlayer.ranking.pairwise_features import (
    FEATURE_NAMES,
    difference,
    extract_features,
)
from soundlayer.ranking.pairwise_logistic import (
    cluster_bootstrap,
    fit_logistic,
    fit_scaler,
    leave_one_case_out,
    predict,
    reverse_augment,
    scale,
    sigmoid,
    symmetry_error,
)
from soundlayer.ranking.preference_dataset import (
    build_dataset,
    canonical_pair_key,
    hydrate_labels,
)


def private_pair(kind="UNIQUE", submitted=True):
    return {
        "pair_id": "p1",
        "case_id": "case-1",
        "kind": kind,
        "left_digest": "sha256:b",
        "right_digest": "sha256:a",
        "left_strategy": "B",
        "right_strategy": "A",
        "is_hidden_repeat": kind == "HIDDEN_REPEAT",
        "is_audit_pair": kind == "AUDIT",
        "overall_preference": "LEFT",
        "submitted": submitted,
    }


def snapshot():
    base = {name: 0.0 for name in FEATURE_NAMES}
    return [
        {
            "case_id": "case-1",
            "artifact_digest": "sha256:a",
            **base,
            "event_coverage": 0.2,
        },
        {
            "case_id": "case-1",
            "artifact_digest": "sha256:b",
            **base,
            "event_coverage": 0.8,
        },
    ]


class LeakageTests(unittest.TestCase):
    def test_identity_names_detected(self):
        self.assertEqual(identity_features(["peak_abs", "case_id"]), ["case_id"])

    def test_safe_features_pass(self):
        assert_feature_safe(FEATURE_NAMES)

    def test_identity_features_rejected(self):
        with self.assertRaises(ValueError):
            assert_feature_safe(["peak_abs", "strategy_id"])

    def test_group_overlap_detected(self):
        self.assertEqual(
            group_leakage([{"case_id": "x"}], [{"case_id": "x"}]), ["x"]
        )

    def test_disjoint_groups_pass(self):
        self.assertEqual(
            group_leakage([{"case_id": "x"}], [{"case_id": "y"}]), []
        )


class DatasetTests(unittest.TestCase):
    def test_canonical_key_is_unordered(self):
        a = canonical_pair_key("c", "z", "a")
        b = canonical_pair_key("c", "a", "z")
        self.assertEqual(a, b)

    def test_hydration_restores_private_truth(self):
        hydrated = hydrate_labels(
            [{"pair_id": "p1", "overall_preference": "RIGHT"}], [private_pair()]
        )
        self.assertEqual(hydrated[0]["left_digest"], "sha256:b")
        self.assertEqual(hydrated[0]["overall_preference"], "RIGHT")

    def test_quality_gate_blocks_all_training_rows(self):
        pair = private_pair()
        rows, report = build_dataset(
            [pair], [pair], snapshot(), {"status": "REVIEW_QUALITY_BLOCKED"}
        )
        self.assertEqual(rows, [])
        self.assertEqual(report["status"], "DATA_BLOCKED")

    def test_training_eligible_builds_canonical_row(self):
        pair = private_pair()
        rows, _ = build_dataset(
            [pair], [pair], snapshot(), {"status": "TRAINING_ELIGIBLE"}
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], 0)

    def test_digest_orientation_controls_feature_sign(self):
        pair = private_pair()
        rows, _ = build_dataset(
            [pair], [pair], snapshot(), {"status": "TRAINING_ELIGIBLE"}
        )
        self.assertAlmostEqual(rows[0]["event_coverage"], -0.6)

    def test_hidden_repeat_is_excluded(self):
        pair = private_pair("HIDDEN_REPEAT")
        rows, report = build_dataset(
            [pair], [pair], snapshot(), {"status": "TRAINING_ELIGIBLE"}
        )
        self.assertEqual(rows, [])
        self.assertEqual(report["exclusionCounts"]["hidden_repeat"], 1)

    def test_audit_is_excluded(self):
        pair = private_pair("AUDIT")
        rows, report = build_dataset(
            [pair], [pair], snapshot(), {"status": "TRAINING_ELIGIBLE"}
        )
        self.assertEqual(rows, [])
        self.assertEqual(report["exclusionCounts"]["audit"], 1)

    def test_tie_is_excluded(self):
        pair = private_pair()
        pair["overall_preference"] = "TIE"
        rows, report = build_dataset(
            [pair], [pair], snapshot(), {"status": "TRAINING_ELIGIBLE"}
        )
        self.assertEqual(rows, [])
        self.assertEqual(report["exclusionCounts"]["tie_or_unjudgeable"], 1)

    def test_canonical_duplicate_is_removed(self):
        first = private_pair()
        second = {**private_pair(), "pair_id": "p2"}
        rows, report = build_dataset(
            [first, second], [first, second], snapshot(), {"status": "TRAINING_ELIGIBLE"}
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["canonicalDuplicateCount"], 1)


class FeatureTests(unittest.TestCase):
    def test_difference_uses_left_minus_right(self):
        left = {name: 2.0 for name in FEATURE_NAMES}
        right = {name: 0.5 for name in FEATURE_NAMES}
        self.assertTrue(all(value == 1.5 for value in difference(left, right).values()))

    def test_audio_feature_extraction_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "tone.wav"
            dss = root / "dss.json"
            rate = 8000
            samples = [
                int(10000 * math.sin(2 * math.pi * 440 * index / rate))
                for index in range(rate)
            ]
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in samples))
            dss.write_text(
                json.dumps(
                    {
                        "video": {"duration_s": 1.0},
                        "events": [
                            {
                                "time_s": 0.2,
                                "duration_s": 0.2,
                                "priority": 3,
                                "tolerance_ms": 100,
                                "layer_role": "foley",
                            }
                        ],
                    }
                )
            )
            features = extract_features(
                audio,
                dss,
                {
                    "candidate_role": "ABLATION_STRATEGY_A",
                    "strategy_id": "A",
                    "ablation_materialized": True,
                },
            )
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertAlmostEqual(features["duration_error_ms"], 0.0)
        self.assertEqual(features["source_reliability"], 1.0)


class LogisticTests(unittest.TestCase):
    def test_sigmoid_zero(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_sigmoid_extremes_are_stable(self):
        self.assertGreater(sigmoid(1000), 0.999)
        self.assertLess(sigmoid(-1000), 0.001)

    def test_scaler_is_train_data_only_primitive(self):
        means, scales = fit_scaler([[0.0], [2.0]])
        self.assertEqual(means, [1.0])
        self.assertEqual(scales, [1.0])
        self.assertEqual(scale([3.0], means, scales), [2.0])

    def test_reverse_augmentation(self):
        vectors, labels = reverse_augment([[1.0, -2.0]], [1])
        self.assertEqual(vectors, [[1.0, -2.0], [-1.0, 2.0]])
        self.assertEqual(labels, [1, 0])

    def test_logistic_learns_separable_data(self):
        weights, intercept = fit_logistic([[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1])
        self.assertLess(predict([-1.0], weights, intercept), 0.5)
        self.assertGreater(predict([1.0], weights, intercept), 0.5)

    def test_logistic_rejects_one_class(self):
        with self.assertRaises(ValueError):
            fit_logistic([[0.0], [1.0]], [1, 1])

    def test_symmetry_with_zero_intercept(self):
        self.assertLess(symmetry_error([0.4, -0.2], [1.2, 0.5], 0.0), 1e-12)

    def test_leave_one_case_out_emits_original_rows_only(self):
        rows = []
        for case in range(4):
            for label in (0, 1):
                row = {
                    "case_id": f"c{case}",
                    "canonical_pair_key": f"p{case}-{label}",
                    "label": label,
                    **{name: float(label * 2 - 1) for name in FEATURE_NAMES},
                }
                rows.append(row)
        output = leave_one_case_out(rows, FEATURE_NAMES)
        self.assertEqual(len(output), len(rows))
        self.assertEqual({row["case_id"] for row in output}, {f"c{i}" for i in range(4)})

    def test_cluster_bootstrap_is_deterministic(self):
        values = {"a": [1.0, 0.0], "b": [1.0], "c": [0.0]}
        self.assertEqual(
            cluster_bootstrap(values, 100, 7), cluster_bootstrap(values, 100, 7)
        )

    def test_cluster_bootstrap_empty(self):
        self.assertIsNone(cluster_bootstrap({}, 100, 7))


if __name__ == "__main__":
    unittest.main()
