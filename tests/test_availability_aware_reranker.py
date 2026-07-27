import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from soundlayer.ranking.availability_aware_reranker import select
class RerankerTest(unittest.TestCase):
    def candidate(self,path,score,decision="MANUAL_REVIEW"):
        return {"status":"SUCCEEDED","output_path":path,"repair_decision":decision,"proxy_score":score,"estimated_edit_cost":0}
    def test_higher_proxy_selected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,"a").write_text("a");Path(d,"b").write_text("b");x,_=select([self.candidate("a",1),self.candidate("b",2)],d);self.assertEqual(x["output_path"],"b")
    def test_rejected_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,"a").write_text("a");x,why=select([self.candidate("a",2,"REPAIR_REJECTED")],d);self.assertIsNone(x)
    def test_missing_blocked(self):self.assertIsNone(select([self.candidate("missing",2)],".")[0])
    def test_edit_cost_tiebreak(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,"a").write_text("a");Path(d,"b").write_text("b");a=self.candidate("a",1);b=self.candidate("b",1);a["estimated_edit_cost"]=2
            self.assertEqual(select([a,b],d)[0]["output_path"],"b")
