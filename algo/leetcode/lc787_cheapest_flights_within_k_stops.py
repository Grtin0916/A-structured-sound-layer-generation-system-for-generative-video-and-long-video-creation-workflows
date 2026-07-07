"""
LC787. Cheapest Flights Within K Stops

Bellman-Ford with bounded edge relaxations.
- n: number of cities
- flights: [from, to, price]
- src/dst: source and destination
- k: at most k stops, so at most k + 1 edges

Complexity:
- Time: O((k + 1) * E)
- Space: O(n)
"""

from __future__ import annotations

from typing import List


class Solution:
    def findCheapestPrice(
        self,
        n: int,
        flights: List[List[int]],
        src: int,
        dst: int,
        k: int,
    ) -> int:
        if n <= 0:
            return -1
        if not (0 <= src < n and 0 <= dst < n):
            return -1
        if src == dst:
            return 0

        inf = 10**18
        dist = [inf] * n
        dist[src] = 0

        # k stops means at most k + 1 flight edges.
        for _ in range(k + 1):
            nxt = dist[:]
            for u, v, price in flights:
                if 0 <= u < n and 0 <= v < n and dist[u] != inf:
                    cand = dist[u] + price
                    if cand < nxt[v]:
                        nxt[v] = cand
            dist = nxt

        return -1 if dist[dst] == inf else dist[dst]


def _run_tests() -> None:
    s = Solution()

    assert s.findCheapestPrice(
        4,
        [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]],
        0,
        3,
        1,
    ) == 700

    assert s.findCheapestPrice(
        3,
        [[0, 1, 100], [1, 2, 100], [0, 2, 500]],
        0,
        2,
        1,
    ) == 200

    assert s.findCheapestPrice(
        3,
        [[0, 1, 100], [1, 2, 100], [0, 2, 500]],
        0,
        2,
        0,
    ) == 500

    assert s.findCheapestPrice(1, [], 0, 0, 0) == 0
    assert s.findCheapestPrice(3, [], 0, 2, 1) == -1
    assert s.findCheapestPrice(0, [], 0, 0, 0) == -1

    print("LC787 tests passed")


if __name__ == "__main__":
    _run_tests()
