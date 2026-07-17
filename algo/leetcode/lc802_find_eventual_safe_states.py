#!/usr/bin/env python3
"""LeetCode 802: find eventual safe states using reverse topological order."""

from collections import deque


def eventual_safe_nodes(graph: list[list[int]]) -> list[int]:
    reverse = [[] for _ in graph]
    out_degree = [len(edges) for edges in graph]
    for source, edges in enumerate(graph):
        for target in edges:
            reverse[target].append(source)

    queue = deque(index for index, degree in enumerate(out_degree) if degree == 0)
    safe: list[int] = []
    while queue:
        node = queue.popleft()
        safe.append(node)
        for parent in reverse[node]:
            out_degree[parent] -= 1
            if out_degree[parent] == 0:
                queue.append(parent)
    return sorted(safe)


def main() -> None:
    cases = [
        ([[1, 2], [2, 3], [5], [0], [5], [], []], [2, 4, 5, 6]),
        ([[1], [2], [0], []], [3]),
        ([[], [], []], [0, 1, 2]),
        (([[]]), [0]),
        (([]), []),
    ]
    for graph, expected in cases:
        actual = eventual_safe_nodes(graph)
        assert actual == expected, (graph, expected, actual)
    print("LC802 tests passed: cycle, terminal, isolated, all-safe, empty")


if __name__ == "__main__":
    main()
