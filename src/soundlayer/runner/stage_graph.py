"""Validated DAG used by the demo runner."""
from collections import deque
class StageGraph:
    def __init__(self,specs):
        self.specs={x["stage_id"]:x for x in specs}
        if len(self.specs)!=len(specs): raise ValueError("duplicate stage_id")
        outputs={}
        for s in specs:
            for d in s.get("dependencies",[]):
                if d not in self.specs: raise ValueError(f"unknown dependency: {d}")
            for o in s.get("exclusive_outputs",[]):
                if o in outputs: raise ValueError(f"duplicate exclusive output: {o}")
                outputs[o]=s["stage_id"]
        self.order=self._topological()
    def _topological(self):
        indeg={x:0 for x in self.specs}; edges={x:[] for x in self.specs}
        for sid,s in self.specs.items():
            for d in s.get("dependencies",[]): edges[d].append(sid); indeg[sid]+=1
        q=deque(x for x,d in indeg.items() if d==0); out=[]
        while q:
            u=q.popleft(); out.append(u)
            for v in edges[u]:
                indeg[v]-=1
                if indeg[v]==0:q.append(v)
        if len(out)!=len(self.specs): raise ValueError("stage graph contains cycle")
        return out
