"""LeetCode 215: in-place Quickselect for top-k acquisition thresholds."""


class Solution:
    def findKthLargest(self, nums, k):
        if not nums or not 1 <= k <= len(nums):
            raise ValueError("k must address an element")
        values = list(nums)
        target = len(values) - k
        left, right = 0, len(values) - 1
        while left <= right:
            pivot = values[(left + right) // 2]
            low, index, high = left, left, right
            while index <= high:
                if values[index] < pivot:
                    values[low], values[index] = values[index], values[low]
                    low += 1
                    index += 1
                elif values[index] > pivot:
                    values[index], values[high] = values[high], values[index]
                    high -= 1
                else:
                    index += 1
            if target < low:
                right = low - 1
            elif target > high:
                left = high + 1
            else:
                return values[target]
        raise AssertionError("quickselect did not converge")


def _self_test():
    solution = Solution()
    cases = [
        ("single", [7], 1, 7),
        ("two", [2, 9], 2, 2),
        ("duplicates", [3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 4),
        ("all_equal", [4, 4, 4, 4], 3, 4),
        ("k_one", [3, 1, 8, 2], 1, 8),
        ("k_n", [3, 1, 8, 2], 4, 1),
        ("negative", [-5, -1, -9, -3], 2, -3),
        ("unsorted", [10, 2, 7, 4, 9, 1], 3, 7),
        ("larger", list(range(100, 0, -1)), 37, 64),
    ]
    for name, nums, k, expected in cases:
        actual = solution.findKthLargest(nums, k)
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
        print(f"PASS {name}")
    print(f"LC215 PASS {len(cases)}/9")


if __name__ == "__main__":
    _self_test()
