import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "scripts/export_ranker_delivery_bundle.py"
SPEC = importlib.util.spec_from_file_location("ranker_delivery", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RankerDeliveryTest(unittest.TestCase):
    def test_blocked_contract_accepts_absent_learned_artifacts(self):
        MODULE.validate_status("DATA_BLOCKED", False, False, 0)

    def test_blocked_contract_rejects_model(self):
        with self.assertRaisesRegex(ValueError, "forbids"):
            MODULE.validate_status("DATA_BLOCKED", True, False, 0)

    def test_blocked_contract_rejects_oof(self):
        with self.assertRaisesRegex(ValueError, "forbids"):
            MODULE.validate_status("DATA_BLOCKED", False, True, 0)

    def test_blocked_contract_rejects_recommendations(self):
        with self.assertRaisesRegex(ValueError, "forbids"):
            MODULE.validate_status("DATA_BLOCKED", False, False, 1)

    def test_candidate_requires_model_and_oof(self):
        with self.assertRaisesRegex(ValueError, "requires model"):
            MODULE.validate_status("CANDIDATE", False, False, 12)

    def test_candidate_requires_recommendation(self):
        with self.assertRaisesRegex(ValueError, "recommendation"):
            MODULE.validate_status("CANDIDATE", True, True, 0)

    def test_artifact_ref_uses_relative_name_and_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "card.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            ref = MODULE.artifact_ref(path)
            self.assertEqual("card.json", ref["relativePath"])
            self.assertEqual(64, len(ref["sha256"]))
            self.assertTrue(ref["requiredForStatus"])


if __name__ == "__main__":
    unittest.main()
