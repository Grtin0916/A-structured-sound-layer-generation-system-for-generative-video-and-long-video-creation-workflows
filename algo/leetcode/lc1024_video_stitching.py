"""LeetCode 1024: minimum greedy coverage for repair intervals."""


def video_stitching(clips: list[list[int]], time: int) -> int:
    """Return the minimum clips covering [0, time], or -1.

    Invariant: before each outer iteration, ``covered`` is reachable using
    ``used`` clips; among every clip starting at or before that boundary we
    extend to the farthest end. Sorting dominates at O(n log n).
    """
    if time == 0:
        return 0
    clips = sorted(clips, key=lambda clip: (clip[0], -clip[1]))
    used = index = covered = 0
    while covered < time:
        next_end = covered
        while index < len(clips) and clips[index][0] <= covered:
            next_end = max(next_end, clips[index][1])
            index += 1
        if next_end == covered:
            return -1
        covered = next_end
        used += 1
    return used


def _self_test() -> None:
    cases = [
        ([], 0, 0),
        ([], 1, -1),
        ([[0, 5]], 5, 1),
        ([[0, 2], [1, 5]], 5, 2),
        ([[0, 1], [2, 4]], 4, -1),
        ([[0, 2], [0, 4], [3, 5]], 5, 2),
        ([[0, 1], [0, 3], [1, 4], [3, 7], [4, 8]], 8, 3),
        ([[0, 4], [2, 8], [1, 5], [5, 10]], 10, 3),
    ]
    for index, (clips, time, expected) in enumerate(cases, 1):
        actual = video_stitching(clips, time)
        assert actual == expected, (index, actual, expected)
        print(f"PASS case_{index}: {actual}")
    print(f"PASS {len(cases)} tests")


if __name__ == "__main__":
    _self_test()
