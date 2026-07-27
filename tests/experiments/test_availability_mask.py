import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.experiments.availability_mask import available,available_values
class AvailabilityTest(unittest.TestCase):
    def test_blocked_excluded(self):self.assertFalse(available({"status":"BLOCKED"},".")[0])
    def test_failed_excluded(self):self.assertFalse(available({"status":"FAILED"},".")[0])
    def test_missing_artifact_excluded(self):self.assertEqual(available({"status":"SUCCEEDED","output_path":"missing"},".")[1],"ARTIFACT_MISSING")
    def test_rejected_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,"x").write_text("x");self.assertEqual(available({"status":"SUCCEEDED","output_path":"x","repair_decision":"REPAIR_REJECTED"},d)[1],"REPAIR_REJECTED")
    def test_available(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,"x").write_text("x");self.assertTrue(available({"status":"SUCCEEDED","output_path":"x","repair_decision":"MANUAL_REVIEW"},d)[0])
    def test_missing_metric_not_zero(self):self.assertEqual(available_values([{"metric_available":False,"x":0},{"metric_available":True,"x":2}],"x"),[2])
