#!/usr/bin/env python3
"""LeetCode 802: eventual safe states in O(V + E) time and space."""

from collections import deque


class Solution:
    def eventualSafeNodes(self, graph: list[list[int]]) -> list[int]:
        reverse = [[] for _ in graph]
        out_degree = [len(edges) for edges in graph]
        for source, targets in enumerate(graph):
            for target in targets:
                reverse[target].append(source)
        queue = deque(index for index, degree in enumerate(out_degree) if degree == 0)
        safe = []
        while queue:
            node = queue.popleft()
            safe.append(node)
            for predecessor in reverse[node]:
                out_degree[predecessor] -= 1
                if out_degree[predecessor] == 0:
                    queue.append(predecessor)
        return sorted(safe)


def self_test() -> None:
    solution = Solution()
    cases = [
        ([], []),
        ([[]], [0]),
        ([[1], [2], []], [0, 1, 2]),
        ([[1], [0]], []),
        ([[1], [2], [1], [2]], []),
        ([[1, 2], [2, 3], [5], [0], [5], [], []], [2, 4, 5, 6]),
        ([[1], [], [3], []], [0, 1, 2, 3]),
        ([[0]], []),
        ([[1], [2], [], [4], [3], [2, 6], []], [0, 1, 2, 5, 6]),
    ]
    for graph, expected in cases:
        assert solution.eventualSafeNodes(graph) == expected
    print(f"LC802 tests passed: {len(cases)}/{len(cases)}")


if __name__ == "__main__":
    self_test()
