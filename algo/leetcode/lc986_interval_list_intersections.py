"""LeetCode 986: intersections of two sorted closed interval lists."""


def interval_intersection(first: list[list[int]], second: list[list[int]]) -> list[list[int]]:
    result: list[list[int]] = []
    left = right = 0
    while left < len(first) and right < len(second):
        start = max(first[left][0], second[right][0])
        end = min(first[left][1], second[right][1])
        if start <= end:
            result.append([start, end])
        if first[left][1] < second[right][1]:
            left += 1
        else:
            right += 1
    return result


def _self_test() -> None:
    cases = [
        ([], [], []),
        ([[1, 2]], [], []),
        ([[1, 5]], [[1, 5]], [[1, 5]]),
        ([[1, 3]], [[2, 4]], [[2, 3]]),
        ([[1, 2]], [[2, 3]], [[2, 2]]),
        ([[1, 10]], [[3, 4], [6, 8]], [[3, 4], [6, 8]]),
        ([[0, 2], [5, 10]], [[1, 5], [8, 12]], [[1, 2], [5, 5], [8, 10]]),
        ([[0, 1], [4, 5]], [[2, 3], [6, 7]], []),
    ]
    for index, (first, second, expected) in enumerate(cases, 1):
        actual = interval_intersection(first, second)
        assert actual == expected, (index, actual, expected)
        print(f"PASS case_{index}: {actual}")
    print(f"PASS {len(cases)} tests")


if __name__ == "__main__":
    _self_test()
