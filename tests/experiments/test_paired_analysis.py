import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.experiments.paired_analysis import bootstrap_ci,holm,sign_permutation_p,signed_rank_p,summarize
class PairedTest(unittest.TestCase):
    def test_bootstrap_reproducible(self):self.assertEqual(bootstrap_ci([1,2,3],100,7),bootstrap_ci([1,2,3],100,7))
    def test_empty_unavailable(self):self.assertFalse(summarize([])["available"])
    def test_direction(self):self.assertEqual(summarize([1,2,3],100,1)["effect_direction"],"POSITIVE")
    def test_counts(self):
        x=summarize([-1,0,2],100,1);self.assertEqual((x["positive_case_count"],x["negative_case_count"],x["zero_case_count"]),(1,1,1))
    def test_all_zero_p(self):self.assertEqual(sign_permutation_p([0,0]),1)
    def test_signed_rank_all_zero(self):self.assertEqual(signed_rank_p([0,0]),1)
    def test_holm_monotone(self):
        x=holm([.01,.02,.5]);self.assertTrue(all(0<=v<=1 for v in x))
