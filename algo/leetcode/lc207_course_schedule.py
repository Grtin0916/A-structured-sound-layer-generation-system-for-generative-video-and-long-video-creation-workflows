#!/usr/bin/env python3
"""LeetCode 207 Course Schedule, O(V + E)."""
from collections import deque
def can_finish(n, prerequisites):
    graph=[[] for _ in range(n)]; degree=[0]*n
    for course,pre in prerequisites: graph[pre].append(course);degree[course]+=1
    q=deque(i for i,d in enumerate(degree) if d==0);seen=0
    while q:
        u=q.popleft();seen+=1
        for v in graph[u]:
            degree[v]-=1
            if degree[v]==0:q.append(v)
    return seen==n
if __name__=="__main__":
    cases=[(0,[],True),(1,[],True),(2,[[1,0]],True),(4,[[1,0],[2,1],[3,2]],True),
           (4,[[1,0],[2,0],[3,1],[3,2]],True),(5,[[1,0],[4,3]],True),
           (1,[[0,0]],False),(3,[[1,0],[2,1],[0,2]],False)]
    for n,e,want in cases: assert can_finish(n,e)==want
    print("LC207 tests passed: 8/8")
