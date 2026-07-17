#!/usr/bin/env python3
"""LeetCode 56: merge intervals, used by repair-window planning."""

from __future__ import annotations


def merge(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged = [ordered[0][:]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def main() -> None:
    cases = [
        ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
        ([[1, 4], [4, 5]], [[1, 5]]),
        ([], []),
        ([[1, 2]], [[1, 2]]),
        ([[5, 7], [1, 2], [2, 4]], [[1, 4], [5, 7]]),
        ([[1, 10], [2, 3], [4, 8]], [[1, 10]]),
        ([[-3, -1], [-2, 2], [3, 4]], [[-3, 2], [3, 4]]),
    ]
    for index, (inputs, expected) in enumerate(cases, 1):
        actual = merge(inputs)
        assert actual == expected, (inputs, actual, expected)
        print(f"case_{index}: PASS input={inputs} output={actual}")
    print("summary: 7/7 PASS")


if __name__ == "__main__":
    main()
