from __future__ import annotations

from collections import deque
from typing import List


class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        n = len(graph)
        reverse = [[] for _ in range(n)]
        out_degree = [0] * n

        for u, nbrs in enumerate(graph):
            out_degree[u] = len(nbrs)
            for v in nbrs:
                if v < 0 or v >= n:
                    raise ValueError(f"edge out of range: {u}->{v}")
                reverse[v].append(u)

        q = deque([i for i, d in enumerate(out_degree) if d == 0])
        safe = [False] * n

        while q:
            v = q.popleft()
            safe[v] = True
            for pre in reverse[v]:
                out_degree[pre] -= 1
                if out_degree[pre] == 0:
                    q.append(pre)

        return [i for i, ok in enumerate(safe) if ok]


def _run_tests() -> None:
    s = Solution()
    assert s.eventualSafeNodes([[1, 2], [2, 3], [5], [0], [5], [], []]) == [2, 4, 5, 6]
    assert s.eventualSafeNodes([[1, 2, 3, 4], [1, 2], [3, 4], [0, 4], []]) == [4]
    assert s.eventualSafeNodes([[], [], []]) == [0, 1, 2]
    assert s.eventualSafeNodes([[1], [2], [0]]) == []
    assert s.eventualSafeNodes([[1], []]) == [0, 1]
    print("LC802 tests passed")


if __name__ == "__main__":
    _run_tests()
