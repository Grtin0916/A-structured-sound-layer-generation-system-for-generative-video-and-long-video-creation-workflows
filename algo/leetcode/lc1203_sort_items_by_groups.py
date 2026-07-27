#!/usr/bin/env python3
"""LeetCode 1203: two-level topological sort, O(n + e)."""
from collections import deque

def sort_items(n, m, group, before_items):
    group=group[:]
    for i in range(n):
        if group[i] == -1: group[i]=m; m+=1
    item_graph=[[] for _ in range(n)]; item_deg=[0]*n
    group_graph=[set() for _ in range(m)]; group_deg=[0]*m
    for item, deps in enumerate(before_items):
        for dep in deps:
            item_graph[dep].append(item); item_deg[item]+=1
            if group[dep] != group[item] and group[item] not in group_graph[group[dep]]:
                group_graph[group[dep]].add(group[item]); group_deg[group[item]]+=1
    def topo(graph, degree):
        q=deque(i for i,d in enumerate(degree) if d==0); out=[]
        while q:
            u=q.popleft(); out.append(u)
            for v in graph[u]:
                degree[v]-=1
                if degree[v]==0:q.append(v)
        return out if len(out)==len(graph) else []
    go=topo(group_graph,group_deg); io=topo(item_graph,item_deg)
    if not go or not io:return []
    buckets=[[] for _ in range(m)]
    for i in io:buckets[group[i]].append(i)
    return [i for g in go for i in buckets[g]]

def valid(result,n,group,before):
    if len(result)!=n:return False
    pos={x:i for i,x in enumerate(result)}
    if any(pos[d]>pos[i] for i,deps in enumerate(before) for d in deps):return False
    spans={}
    for i,x in enumerate(result):
        if group[x]!=-1:spans.setdefault(group[x],[]).append(i)
    return all(max(v)-min(v)+1==len(v) for v in spans.values())

if __name__=="__main__":
    cases=[
      (8,2,[-1,-1,1,0,0,1,0,-1],[[],[6],[5],[6],[3,6],[],[],[]],True),
      (8,2,[-1,-1,1,0,0,1,0,-1],[[],[6],[5],[6],[3],[6],[4],[]],False),
      (1,0,[-1],[[]],True),(3,1,[0,0,0],[[],[0],[1]],True),(2,1,[0,0],[[1],[0]],False),
      (4,2,[0,0,1,1],[[],[0],[1],[2]],True),(4,2,[0,1,0,1],[[],[],[],[]],True),
      (5,0,[-1]*5,[[],[0],[0],[1,2],[3]],True),(3,3,[0,1,2],[[],[0],[1]],True),
    ]
    passed=0
    for n,m,g,b,possible in cases:
        r=sort_items(n,m,g,b)
        ok=valid(r,n,g,b) if possible else r==[]
        assert ok; passed+=1
    print(f"LC1203 tests passed: {passed}/9")
