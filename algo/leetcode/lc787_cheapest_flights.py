#!/usr/bin/env python3
"""LeetCode 787 bounded Bellman-Ford, O(K*E), O(V)."""
def find_cheapest_price(n,flights,src,dst,k):
    dist=[float("inf")]*n;dist[src]=0
    for _ in range(k+1):
        nxt=dist[:]
        for u,v,w in flights:
            if dist[u]!=float("inf"):nxt[v]=min(nxt[v],dist[u]+w)
        dist=nxt
    return -1 if dist[dst]==float("inf") else dist[dst]
if __name__=="__main__":
    cases=[(3,[],0,2,1,-1),(2,[[0,1,5]],0,1,0,5),
      (3,[[0,2,10],[0,1,2],[1,2,2]],0,2,1,4),
      (4,[[0,1,1],[1,2,1],[2,3,1],[0,3,9]],0,3,1,9),
      (3,[[0,1,1],[1,0,1],[1,2,1]],0,2,2,2),
      (2,[[0,1,5],[0,1,3]],0,1,0,3),(3,[[0,1,2],[1,2,2]],0,2,0,-1),
      (4,[[0,1,2],[0,2,2],[1,3,2],[2,3,2]],0,3,1,4),
      (5,[[0,4,20],[0,1,2],[1,2,2],[2,3,2],[3,4,2]],0,4,3,8)]
    for args in cases:assert find_cheapest_price(*args[:-1])==args[-1]
    print("LC787 tests passed: 9/9")
