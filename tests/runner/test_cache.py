import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.runner.cache import StageCache
from soundlayer.runner.contracts import cache_key,digest_file
class CacheTest(unittest.TestCase):
    def test_input_changes_key(self):
        a=cache_key("x","i","c",{"a":"1"},"VERIFY"); b=cache_key("x","i","c",{"a":"2"},"VERIFY");self.assertNotEqual(a,b)
    def test_config_changes_key(self): self.assertNotEqual(cache_key("x","i","a",{},"VERIFY"),cache_key("x","i","b",{},"VERIFY"))
    def test_code_changes_key(self): self.assertNotEqual(cache_key("x","a","c",{},"VERIFY"),cache_key("x","b","c",{},"VERIFY"))
    def test_missing_output_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            c=StageCache(Path(d)/"cache");c.store("sha256:k",{"output_digests":{"gone":"sha256:x"}})
            self.assertEqual(c.lookup("sha256:k",d)[1],"OUTPUT_MISSING")
    def test_wrong_digest_is_stale(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,"x").write_text("x");c=StageCache(Path(d)/"cache");c.store("sha256:k",{"output_digests":{"x":"sha256:bad"}})
            self.assertEqual(c.lookup("sha256:k",d)[1],"OUTPUT_DIGEST_MISMATCH")
    def test_valid_output_reused(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,"x").write_text("x");c=StageCache(Path(d)/"cache");c.store("sha256:k",{"output_digests":{"x":digest_file(Path(d,"x"))}})
            self.assertTrue(c.lookup("sha256:k",d)[0])
