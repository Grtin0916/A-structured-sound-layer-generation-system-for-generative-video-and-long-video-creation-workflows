import json,struct,sys,tempfile,unittest,wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from soundlayer.dss.control_compiler import compile_control
class CompilerTest(unittest.TestCase):
    def fixture(self,d):
        src=Path(d,"x.wav");out=Path(d,"y.wav");dss=Path(d,"dss.json")
        with wave.open(str(src),"wb") as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(1000);w.writeframes(struct.pack("<"+"h"*1000,*([1000]*1000)))
        dss.write_text(json.dumps({"events":[{"event_id":"e","time_s":-.2,"duration_s":.5,"priority":5,"avoid":["speech"]}]}))
        return src,out,dss
    def test_duration_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,j=self.fixture(d);self.assertTrue(compile_control(s,j,o,{str(i):1.1 for i in range(6)})["duration_match"])
    def test_out_of_bounds_safe(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,j=self.fixture(d);self.assertEqual(compile_control(s,j,o,{str(i):1.1 for i in range(6)})["windows"][0]["start_frame"],0)
    def test_priority_gain_applied(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,j=self.fixture(d);x=compile_control(s,j,o,{str(i):(1.5 if i==5 else 1) for i in range(6)});self.assertGreater(x["rms_after"],x["rms_before"])
    def test_no_clip_regression(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,j=self.fixture(d);self.assertEqual(compile_control(s,j,o,{str(i):20 for i in range(6)})["clip_ratio"],0)
    def test_avoid_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,j=self.fixture(d);self.assertEqual(compile_control(s,j,o,{str(i):1 for i in range(6)})["avoid_constraints"],["speech"])
