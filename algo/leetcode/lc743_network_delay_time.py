#!/usr/bin/env python3
"""LeetCode 743 Network Delay Time, O((V + E) log V)."""
import heapq
def network_delay_time(times,n,k):
    graph=[[] for _ in range(n+1)]
    for u,v,w in times:graph[u].append((v,w))
    dist=[float("inf")]*(n+1);dist[k]=0;q=[(0,k)]
    while q:
        d,u=heapq.heappop(q)
        if d!=dist[u]:continue
        for v,w in graph[u]:
            nd=d+w
            if nd<dist[v]:dist[v]=nd;heapq.heappush(q,(nd,v))
    answer=max(dist[1:])
    return -1 if answer==float("inf") else answer
if __name__=="__main__":
    cases=[([],1,1,0),([[1,2,3]],2,1,3),([[1,2,1],[2,3,2]],3,1,3),
      ([[1,2,1]],3,1,-1),([[1,2,5],[1,3,1],[3,2,1]],3,1,2),
      ([[1,2,5],[1,2,2]],2,1,2),([[1,2,1],[1,3,2],[1,4,3]],4,1,3),
      ([[1,2,1],[2,3,1],[3,4,1]],4,1,3),([[1,2,1],[2,3,1],[3,1,1]],3,1,2)]
    for edges,n,k,want in cases:assert network_delay_time(edges,n,k)==want
    print("LC743 tests passed: 9/9")
