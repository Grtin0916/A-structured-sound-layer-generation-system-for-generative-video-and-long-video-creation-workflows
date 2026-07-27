import sys,threading,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from soundlayer.runner.model_adapter import CandidateResult
from soundlayer.runner.resource_scheduler import ResourceScheduler
def result(name):return CandidateResult(name,"s","a","SUCCEEDED","CONTROL")
class SchedulerTest(unittest.TestCase):
    def test_gpu_capacity_one(self):
        active=peak=0;lock=threading.Lock()
        def work():
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            time.sleep(.01)
            with lock:active-=1
            return result("x")
        ResourceScheduler(gpu_capacity=1,cpu_capacity=1,io_capacity=1).run([{"resource_class":"gpu","call":work} for _ in range(3)])
        self.assertEqual(peak,1)
    def test_failure_isolated(self):
        def bad():raise RuntimeError("boom")
        out=ResourceScheduler(1,1,1).run([{"resource_class":"cpu","call":bad},{"resource_class":"cpu","call":lambda:result("ok")}])
        self.assertEqual([x["status"] if isinstance(x,dict) else x.status for x in out],["FAILED","SUCCEEDED"])
    def test_timeout_classified(self):
        def bad():raise TimeoutError("late")
        self.assertEqual(ResourceScheduler(1,1,1).run([{"resource_class":"io","call":bad}])[0]["failure_code"],"TIMEOUT")
    def test_queue_wait_recorded(self):
        x=ResourceScheduler(1,1,1).run([{"resource_class":"cpu","call":lambda:result("ok")}])[0]
        self.assertGreaterEqual(x.queue_wait_ms,0)
