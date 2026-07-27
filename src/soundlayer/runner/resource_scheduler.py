"""Bounded resource scheduler with slot-level exception isolation."""
import concurrent.futures,threading,time
class ResourceScheduler:
    def __init__(self,gpu_capacity=1,cpu_capacity=4,io_capacity=8):
        self.capacities={"gpu":gpu_capacity,"cpu":cpu_capacity,"io":io_capacity}
        self.semaphores={k:threading.Semaphore(v) for k,v in self.capacities.items()}
    def run(self,tasks):
        results=[]
        def guarded(task):
            queued=time.perf_counter_ns()
            with self.semaphores[task["resource_class"]]:
                wait=(time.perf_counter_ns()-queued)//1_000_000
                try:
                    value=task["call"](); value.queue_wait_ms=wait; return value
                except TimeoutError as e:return {"status":"FAILED","failure_code":"TIMEOUT","failure_reason":str(e)}
                except Exception as e:return {"status":"FAILED","failure_code":"EXECUTION_FAILED","failure_reason":str(e)}
        workers=max(1,sum(self.capacities.values()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future=[pool.submit(guarded,x) for x in tasks]
            for f in future:results.append(f.result())
        return results
