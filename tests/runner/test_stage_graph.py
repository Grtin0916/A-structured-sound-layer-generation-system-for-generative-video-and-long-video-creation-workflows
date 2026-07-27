import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.runner.stage_graph import StageGraph
class StageGraphTest(unittest.TestCase):
    def test_acyclic(self): self.assertEqual(StageGraph([{"stage_id":"a"},{"stage_id":"b","dependencies":["a"]}]).order,["a","b"])
    def test_cycle(self):
        with self.assertRaisesRegex(ValueError,"cycle"): StageGraph([{"stage_id":"a","dependencies":["b"]},{"stage_id":"b","dependencies":["a"]}])
    def test_unknown_dependency(self):
        with self.assertRaisesRegex(ValueError,"unknown"): StageGraph([{"stage_id":"a","dependencies":["missing"]}])
    def test_duplicate_output(self):
        with self.assertRaisesRegex(ValueError,"duplicate exclusive"): StageGraph([{"stage_id":"a","exclusive_outputs":["x"]},{"stage_id":"b","exclusive_outputs":["x"]}])
