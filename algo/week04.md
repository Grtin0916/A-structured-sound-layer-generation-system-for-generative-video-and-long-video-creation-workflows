# Week 04 - Two Pointers / Sliding Window

## 2026-03-30

### 283. Move Zeroes
- Link: https://leetcode.com/problems/move-zeroes/
- Status: AC
- 指针定义：
  - `fast` 扫描整个数组；
  - `slow` 指向下一个该放非零元素的位置。
- 循环不变量：
  - 任意时刻，`nums[0:slow]` 都是已经整理好的非零元素，且相对顺序不变。
- 边界：
  - 空数组
  - 单元素
  - 全 0
  - 全非 0
- Python 代码：
~~~python
class Solution(object):
    def moveZeroes(self, nums):
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow] = nums[fast]
                slow += 1
        for i in range(slow, len(nums)):
            nums[i] = 0
~~~
- Complexity:
  - Time: `O(n)`
  - Space: `O(1)`
- One-line template:
  - “快指针找非零，慢指针负责落位，最后把尾部补 0。”

### 392. Is Subsequence
- Link: https://leetcode.com/problems/is-subsequence/
- Status: AC
- 指针定义：
  - `i` 指向 `s` 当前要匹配的位置；
  - `j` 指向 `t` 当前扫描到的位置。
- 循环不变量：
  - 任意时刻，`s[0:i]` 都已经在 `t[0:j]` 中按顺序匹配完成。
- 边界：
  - `s` 为空串
  - `t` 为空串
  - `s` 比 `t` 长
  - 匹配直到 `t` 末尾才完成
- Python 代码：
~~~python
class Solution(object):
    def isSubsequence(self, s, t):
        i = 0
        j = 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)
~~~
- Complexity:
  - Time: `O(len(t))`
  - Space: `O(1)`
- One-line template:
  - “一个指针盯目标串，一个指针扫母串，匹配上就推进目标指针。”

## 今日小结
- 283：练的是原地写入型双指针。
- 392：练的是单调扫描型双指针。
- 今天共同点：
  - 都是两个指针单调前进；
  - 都依赖清晰的不变量；
  - 都属于后面滑动窗口题的前置手感。

## 2026-03-31

### 167. Two Sum II - Input Array Is Sorted
- Link: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
- Status: AC
- 题意：
  - 在一个非递减数组中，找出两个数之和等于 target，返回 1-based 下标。
- 核心思路：
  - 用左右双指针；和小了左指针右移，和大了右指针左移，利用“数组有序”这个条件单调收缩。
- Complexity:
  - Time: `O(n)`
  - Space: `O(1)`
- 最容易写崩点：
  - 返回值是 **1-based**，不是 0-based。
- Python 代码：
~~~python
class Solution(object):
    def twoSum(self, numbers, target):
        l = 0
        r = len(numbers) - 1
        while l < r:
            s = numbers[l] + numbers[r]
            if s == target:
                return [l + 1, r + 1]
            elif s < target:
                l += 1
            else:
                r -= 1
~~~

### 11. Container With Most Water
- Link: https://leetcode.com/problems/container-with-most-water/
- Status: AC
- 题意：
  - 选两根柱子和 x 轴组成容器，求最多能装多少水。
- 核心思路：
  - 仍然是左右双指针；每次只移动短板，因为面积由短板决定，移动长板不可能让当前宽度损失后获得更优解。
- Complexity:
  - Time: `O(n)`
  - Space: `O(1)`
- 最容易写崩点：
  - 面积公式是 `min(height[l], height[r]) * (r - l)`，不是 `max(...)`。
- Python 代码：
~~~python
class Solution(object):
    def maxArea(self, height):
        l = 0
        r = len(height) - 1
        ans = 0
        while l < r:
            ans = max(ans, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return ans
~~~

## 今日补充小结（2026-03-31）
- 167：练的是“有序数组上的相向双指针”。
- 11：练的是“短板收缩”这一类贪心式双指针。
- 今天共同点：
  - 都依赖双指针单调移动；
  - 都要先想清楚“为什么某一侧可以安全移动”；
  - 都是后面滑动窗口与边界收缩题的基础手感。
