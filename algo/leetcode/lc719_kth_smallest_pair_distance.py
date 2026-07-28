"""LeetCode 719: binary search the k-th smallest pair distance."""


class Solution:
    def smallestDistancePair(self, nums, k):
        if len(nums) < 2:
            raise ValueError("at least two values are required")
        pair_count = len(nums) * (len(nums) - 1) // 2
        if not 1 <= k <= pair_count:
            raise ValueError("k is outside the pair count")
        values = sorted(nums)

        def count_at_most(distance):
            total = 0
            left = 0
            for right, value in enumerate(values):
                while value - values[left] > distance:
                    left += 1
                total += right - left
            return total

        low, high = 0, values[-1] - values[0]
        while low < high:
            middle = (low + high) // 2
            if count_at_most(middle) >= k:
                high = middle
            else:
                low = middle + 1
        return low


def _self_test():
    solution = Solution()
    cases = [
        ("all_equal", [1, 1, 1], 2, 0),
        ("duplicate", [1, 3, 1], 1, 0),
        ("k_first", [1, 6, 2], 1, 1),
        ("k_last", [1, 6, 2], 3, 5),
        ("unsorted", [9, 2, 6, 4], 3, 3),
        ("large_gap", [0, 100, 300], 2, 200),
        ("two_values", [8, 3], 1, 5),
        ("negative_values", [-10, -4, 2], 2, 6),
        ("repeated_cluster", [1, 1, 2, 2], 4, 1),
    ]
    for name, nums, k, expected in cases:
        actual = solution.smallestDistancePair(nums, k)
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
        print(f"PASS {name}")
    print(f"LC719 PASS {len(cases)}/9")


if __name__ == "__main__":
    _self_test()
