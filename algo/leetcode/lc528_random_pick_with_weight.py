"""LeetCode 528: reproducible weighted sampling with prefix sums."""

from bisect import bisect_left
import random


class Solution:
    """O(n) initialization, O(log n) sampling, and O(n) storage."""

    def __init__(self, w, seed=None):
        if not w or any(weight <= 0 for weight in w):
            raise ValueError("weights must be a non-empty list of positive integers")
        self.prefix = []
        total = 0
        for weight in w:
            total += weight
            self.prefix.append(total)
        self.total = total
        self.random = random.Random(seed)

    def pickIndex(self):
        target = self.random.randint(1, self.total)
        return bisect_left(self.prefix, target)


def _self_test():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    check("single_weight", {Solution([7], 1).pickIndex() for _ in range(20)} == {0})
    picker = Solution([1, 3], 20260727)
    samples = [picker.pickIndex() for _ in range(4000)]
    check("indices_in_range", set(samples) <= {0, 1})
    check("heavier_bucket_wins", samples.count(1) > samples.count(0) * 2)
    a = Solution([2, 5, 3], 42)
    b = Solution([2, 5, 3], 42)
    check("seed_reproducible", [a.pickIndex() for _ in range(50)] == [b.pickIndex() for _ in range(50)])
    check("prefix_sum", Solution([2, 5, 3], 1).prefix == [2, 7, 10])
    check("total_weight", Solution([2, 5, 3], 1).total == 10)
    try:
        Solution([], 1)
    except ValueError:
        checks.append("reject_empty")
    try:
        Solution([1, 0], 1)
    except ValueError:
        checks.append("reject_zero")
    try:
        Solution([1, -1], 1)
    except ValueError:
        checks.append("reject_negative")
    return checks


if __name__ == "__main__":
    passed = _self_test()
    print(f"LC528 PASS {len(passed)}/9")
    for name in passed:
        print(f"PASS {name}")
