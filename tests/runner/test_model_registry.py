import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.runner.model_registry import ModelRegistry,probe_capabilities
ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/"configs/demo/candidate_matrix_12cases.yaml").read_text())
class RegistryTest(unittest.TestCase):
    def test_duplicate_adapter_rejected(self):
        class A:adapter_id="a"
        r=ModelRegistry();r.register(A())
        with self.assertRaisesRegex(ValueError,"duplicate"):r.register(A())
    def test_unknown_adapter_rejected(self):
        with self.assertRaises(KeyError):ModelRegistry().get("missing")
    def test_control_ready(self):
        caps={x.adapter_id:x for x in probe_capabilities(CONFIG)}
        self.assertEqual(caps["control"].status,"READY")
    def test_capability_reason_preserved(self):
        caps={x.adapter_id:x for x in probe_capabilities(CONFIG)}
        self.assertTrue(caps["mmaudio"].reason and caps["mmaudio"].failure_code)
    def test_missing_repository_explicit(self):
        caps={x.adapter_id:x for x in probe_capabilities(CONFIG)}
        self.assertEqual(caps["foleycrafter"].status,"REPOSITORY_MISSING")
