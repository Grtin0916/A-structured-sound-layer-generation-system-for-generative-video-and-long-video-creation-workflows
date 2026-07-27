import sys,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.experiments.ablation_contract import build_contracts
CASES=[{"matrix_case_id":f"c{i}","repair_decision":"MANUAL_REVIEW"} for i in range(12)]
class ContractTest(unittest.TestCase):
    def test_48_records(self):self.assertEqual(len(build_contracts(CASES,"x")),48)
    def test_four_each(self):self.assertEqual(set(Counter(x["case_id"] for x in build_contracts(CASES,"x")).values()),{4})
    def test_stable_order(self):
        x=build_contracts(CASES,"x");self.assertEqual([(r["case_order"],r["strategy_order"]) for r in x],sorted((r["case_order"],r["strategy_order"]) for r in x))
    def test_stable_key(self):self.assertEqual(build_contracts(CASES,"x")[0]["contract_key"],build_contracts(CASES,"x")[0]["contract_key"])
    def test_no_final(self):self.assertTrue(all(x["publish_decision"]!="FINAL_SELECTED" for x in build_contracts(CASES,"x")))
