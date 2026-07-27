import json,shutil,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.runner.demo_runner import DemoRunner
ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/"configs/demo/runner.yaml").read_text())
class DemoRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.runner=DemoRunner(ROOT,CONFIG)
    def test_inventory_selects_real_complete_case(self):
        x=self.runner.choose();self.assertTrue(all(x["checks"].values()))
    def test_plan_has_eleven_stages(self): self.assertEqual(len(self.runner.plan()["stage_order"]),11)
    def test_manual_review_is_provisional(self): self.assertEqual(self.runner.plan()["publish_decision"],"provisional")
    def test_rejected_cannot_be_chosen(self): self.assertNotEqual(self.runner.choose()["decision"],"REPAIR_REJECTED")
    def test_run_key_is_stable(self):
        c=self.runner.choose();self.assertEqual(self.runner.identity(c)[0],self.runner.identity(c)[0])
    def test_fail_resume_and_idempotency(self):
        failed=self.runner.run(fail_after="evaluate"); rerun=None
        try:
            self.assertEqual(failed["status"],"FAILED")
            resumed=self.runner.run(resume_run_id=failed["run_id"]);self.assertEqual(resumed["status"],"SUCCEEDED")
            rerun=self.runner.run();self.assertEqual(rerun["status"],"SUCCEEDED")
            self.assertGreaterEqual(rerun["reusedStageCount"],7);self.assertEqual(rerun["duplicateArtifactCount"],0)
        finally:
            for run in filter(None,(failed,rerun)):
                shutil.rmtree(ROOT/"artifacts/runs"/run["run_id"],ignore_errors=True)
                for p in (ROOT/"reports").glob(f"demo_run_{run['run_id']}.*"): p.unlink()
