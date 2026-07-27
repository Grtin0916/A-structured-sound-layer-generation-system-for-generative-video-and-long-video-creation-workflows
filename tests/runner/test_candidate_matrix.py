import json,sys,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.runner.candidate_matrix import CandidateMatrix,SLOT_ORDER
ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/"configs/demo/candidate_matrix_12cases.yaml").read_text())
class MatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.matrix=CandidateMatrix(ROOT,CONFIG);cls.plan=cls.matrix.plan()
    def test_48_slots(self):self.assertEqual(self.plan["planned_count"],48)
    def test_12_cases(self):self.assertEqual(len(self.plan["case_order"]),12)
    def test_four_slots_each(self):
        self.assertEqual(set(Counter(x["matrix_case_id"] for x in self.plan["records"]).values()),{4})
    def test_no_output_collision(self):self.assertEqual(self.plan["collision_count"],0)
    def test_order_stable(self):
        pairs=[(x["case_order"],x["slot_order"]) for x in self.plan["records"]]
        self.assertEqual(pairs,sorted(pairs))
    def test_plan_digest_stable(self):self.assertEqual(self.plan["manifest_digest"],self.matrix.plan()["manifest_digest"])
    def test_slot_key_stable(self):self.assertEqual(self.plan["records"][0]["slot_key"],self.matrix.plan()["records"][0]["slot_key"])
    def test_seed_changes_slot_key(self):
        other=json.loads(json.dumps(CONFIG));other["seed_base"]+=1
        self.assertNotEqual(self.plan["records"][0]["slot_key"],CandidateMatrix(ROOT,other).plan()["records"][0]["slot_key"])
    def test_publish_gate(self):
        self.assertTrue(all(x["allowed_publish_type"]!="FINAL" for x in self.plan["records"]))
    def test_rejected_cases_block_publish(self):
        rejected=[x for x in self.plan["records"] if x["repair_decision"]=="REPAIR_REJECTED"]
        self.assertEqual(len({x["matrix_case_id"] for x in rejected}),2)
        self.assertTrue(all(x["allowed_publish_type"]=="BLOCKED" for x in rejected))
