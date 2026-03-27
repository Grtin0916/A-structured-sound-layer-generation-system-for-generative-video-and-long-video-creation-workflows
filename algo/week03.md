# Week 03 Algorithm Notes

## LeetCode 20 - Valid Parentheses

### 题意
给定一个只包含 `()[]{}` 的字符串，判断括号是否合法匹配。

### 核心思路
使用栈：

- 遇到左括号，入栈
- 遇到右括号，检查栈顶是否为对应左括号
- 不匹配直接返回 `False`
- 最后栈为空才合法

### Python 代码

~~~python
class Solution(object):
    def isValid(self, s):
        mapping = {
            ')': '(',
            ']': '[',
            '}': '{'
        }
        stack = []

        for ch in s:
            if ch in mapping.values():
                stack.append(ch)
            else:
                if not stack or stack[-1] != mapping.get(ch):
                    return False
                stack.pop()

        return len(stack) == 0
~~~

### 复杂度

- 时间复杂度：`O(n)`
- 空间复杂度：`O(n)`

### 易错点

- 栈空时不能直接访问 `stack[-1]`
- 右括号出现时要先判空
- 最后必须检查栈是否为空

---

## LeetCode 232 - Implement Queue using Stacks

### 题意
仅用两个栈实现队列，支持以下操作：

- `push(x)`
- `pop()`
- `peek()`
- `empty()`

### 核心思路
使用两个栈：

- `in_stack`：负责入队
- `out_stack`：负责出队和取队头
- 当 `out_stack` 为空时，把 `in_stack` 中所有元素倒过去

这样可以保证先进先出。

### Python 代码

~~~python
class MyQueue(object):

    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def _move(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        self._move()
        return self.out_stack.pop()

    def peek(self):
        self._move()
        return self.out_stack[-1]

    def empty(self):
        return not self.in_stack and not self.out_stack
~~~

### 复杂度

- `push`：均摊 `O(1)`
- `pop`：均摊 `O(1)`
- `peek`：均摊 `O(1)`
- `empty`：`O(1)`
- 空间复杂度：`O(n)`

### 易错点

- 不是每次 `pop()` / `peek()` 都倒栈，只有 `out_stack` 为空时才倒
- `peek()` 不需要弹出再放回
- `empty()` 要同时判断两个栈

---

## 今日小结

今天算法部分完成：

- 20：括号匹配，标准栈模板题
- 232：双栈模拟队列，核心是“延迟搬运”

这两题都是后续栈、队列、设计类题目的基础模板。

---

## LeetCode 155 - Min Stack

### 题意
设计一个支持以下操作、并且 `getMin()` 时间复杂度为 `O(1)` 的栈：

- `push(val)`
- `pop()`
- `top()`
- `getMin()`

LeetCode 155 的官方题名就是 **Min Stack**。:contentReference[oaicite:1]{index=1}

### 核心思路
这题的关键不是“栈能不能存数”，而是**最小值如何同步维护**。

最稳的写法是两个栈：

- `stack`：正常存所有元素
- `min_stack`：存“到当前位置为止的最小值”

规则：

- `push(x)`：
  - `stack` 一定压入 `x`
  - `min_stack` 压入 `min(x, 当前最小值)`
- `pop()`：
  - 两个栈一起弹出
- `top()`：
  - 看 `stack[-1]`
- `getMin()`：
  - 看 `min_stack[-1]`

这样每一步都能在 `O(1)` 拿到当前最小值。

### Python 代码

~~~python
class MinStack(object):

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self):
        self.min_stack.pop()
        return self.stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]
~~~

### 复杂度

- `push`：`O(1)`
- `pop`：`O(1)`
- `top`：`O(1)`
- `getMin`：`O(1)`
- 空间复杂度：`O(n)`

### 边界情况

- 连续压入递减序列，例如 `5, 4, 3, 2`
- 连续压入相同最小值，例如 `2, 2, 2`
- 弹出后最小值发生回退，例如 `push(3), push(1), pop()`

### 易错点

- 只在“更小”时才压 `min_stack`，会导致重复最小值弹出后出错
- `pop()` 时忘记让两个栈同步弹出
- 把 `getMin()` 写成每次遍历主栈，复杂度直接寄了

---

## LeetCode 225 - Implement Stack using Queues

### 题意
只用队列实现栈，支持：

- `push(x)`
- `pop()`
- `top()`
- `empty()`

LeetCode 225 的官方题名就是 **Implement Stack using Queues**。官方题面写的是“只用队列实现后进先出栈”，并支持普通栈的主要操作。:contentReference[oaicite:2]{index=2}

### 核心思路
这题和昨天的 232 是镜像题：  
232 是“用栈实现队列”，225 是“用队列实现栈”。

最常见有两种思路：

- 两个队列
- 一个队列

今天先记**一个队列**版本，更省代码，也更适合口述。

做法：

- `push(x)`：先把 `x` 入队
- 然后把前面已有的元素，依次出队再入队
- 这样新元素就被旋转到队头
- 后面 `pop()` / `top()` 直接操作队头即可

本质上就是：  
**把 push 变贵，换来 pop / top 变简单。**

### Python 代码

~~~python
from collections import deque


class MyStack(object):

    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0
~~~

### 复杂度

- `push`：`O(n)`
- `pop`：`O(1)`
- `top`：`O(1)`
- `empty`：`O(1)`
- 空间复杂度：`O(n)`

### 边界情况

- 只有一个元素时连续执行 `top()`、`pop()`
- 多次 `push()` 后检查是否真的是后进先出
- 弹空后再调用 `empty()`

### 易错点

- 忘了在 `push()` 后做旋转，队列就还是 FIFO，不是栈
- 把旋转次数写成 `len(self.q)`，会多转一轮
- `top()` 和 `pop()` 的队头位置判断写反

---

## 周二补充小结

今天新增两题：

- 155：辅助栈，核心是“最小值同步维护”
- 225：队列模拟栈，核心是“把代价集中到 push”

到这里，Week 03 前两天的栈 / 队列题已经形成两组镜像模板：

- 20：括号匹配栈
- 232：双栈模拟队列
- 155：辅助栈维护最小值
- 225：单队列模拟栈


---

## LeetCode 933 - Number of Recent Calls

### 题意
设计一个类 `RecentCounter`，每次调用 `ping(t)` 时，返回最近 `3000` 毫秒内发生的请求数。

也就是统计区间：

- `[t - 3000, t]`

内一共有多少次请求。

### 核心思路
这题本质就是：

**用队列维护一个滑动时间窗口。**

每次新请求 `t` 到来时：

- 先把 `t` 放进队列
- 然后不断弹出队头，直到队头时间 `>= t - 3000`
- 剩下的元素就是合法窗口内的请求

为什么可以一直弹？

因为题目里的 `t` 是严格递增的。  
所以一旦某个旧时间已经小于 `t - 3000`，它以后也不可能再合法，直接永久出队即可。

### Python 代码

~~~python
from collections import deque


class RecentCounter(object):

    def __init__(self):
        self.q = deque()

    def ping(self, t):
        self.q.append(t)

        while self.q and self.q[0] < t - 3000:
            self.q.popleft()

        return len(self.q)
~~~

### 复杂度

- 单次 `ping`：均摊 `O(1)`
- 空间复杂度：`O(n)`

说明：

虽然 `while` 看起来可能弹很多次，但每个元素最多只会入队一次、出队一次，所以总摊还是 `O(1)`。

### 边界情况

- 连续请求都落在窗口内，例如 `1, 100, 3001`
- 请求刚好落在边界，例如 `t - 3000`
- 队列里旧元素很多，需要一次性连续弹出

### 易错点

- 条件写成 `<= t - 3000` 会把边界点错误弹掉
- 不理解“为什么可以一直弹”，本质上是因为 `t` 单调递增
- 用列表 `pop(0)` 会退化，队列应用 `deque`

---

## LeetCode 2390 - Removing Stars From a String

### 题意
给定一个字符串，遇到普通字符就保留，遇到 `*` 时删除它左边最近的一个非星号字符，返回最终结果。

### 核心思路
这题就是标准的：

**字符串栈回退。**

做法：

- 普通字符：入栈
- 遇到 `*`：把栈顶弹出
- 最后把栈里的字符拼起来

为什么最终 `''.join(stack)` 就是答案？

因为每个 `*` 都已经在扫描过程中完成了“回退删除”动作。  
扫描结束后，栈中剩下的字符，顺序正好就是最终字符串。

### Python 代码

~~~python
class Solution(object):
    def removeStars(self, s):
        stack = []

        for ch in s:
            if ch == '*':
                stack.pop()
            else:
                stack.append(ch)

        return ''.join(stack)
~~~

### 复杂度

- 时间复杂度：`O(n)`
- 空间复杂度：`O(n)`

### 边界情况

- 全是普通字符
- 星号连续出现
- 字符与星号交替出现
- 最终结果为空字符串

### 易错点

- 把它想复杂，去真的“删除左边子串”
- 忘了这是后进先出，应该直接弹栈
- 最后返回 `stack` 而不是 `''.join(stack)`

---

## 周三补充小结

今天新增两题：

- 933：队列维护滑动时间窗口
- 2390：字符串栈回退

到这里，Week 03 前三天的栈 / 队列模板又补了两类：

- 栈：括号匹配、最小栈、字符串回退
- 队列：双栈模拟队列、时间窗口队列、单队列模拟栈

---

## LeetCode 394 - Decode String

### 题意
给定一个经过编码的字符串，按规则将其解码：

- `k[encoded_string]`
- 表示方括号内部的字符串正好重复 `k` 次

例如：

- `3[a]` -> `aaa`
- `3[a2[c]]` -> `accaccacc`

### 核心思路
这题本质是：

**遇到 `[` 时进入子问题，遇到 `]` 时回到上一层。**

最稳的写法是用两个栈：

- `num_stack`：存每一层的重复次数
- `str_stack`：存进入当前 `[` 之前已经拼好的字符串

扫描过程：

- 如果是数字，就累积成当前数字 `num`
- 如果是普通字母，就直接拼到当前字符串 `cur`
- 如果遇到 `[`：
  - 把当前数字压入 `num_stack`
  - 把当前字符串压入 `str_stack`
  - 然后重置 `num = 0`，`cur = ""`
- 如果遇到 `]`：
  - 取出重复次数
  - 取出上一层字符串
  - 执行拼接：`上一层 + 当前串 * 次数`

### Python 代码

~~~python
class Solution(object):
    def decodeString(self, s):
        num_stack = []
        str_stack = []
        num = 0
        cur = ""

        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == '[':
                num_stack.append(num)
                str_stack.append(cur)
                num = 0
                cur = ""
            elif ch == ']':
                repeat = num_stack.pop()
                prev = str_stack.pop()
                cur = prev + cur * repeat
            else:
                cur += ch

        return cur
~~~

### 复杂度

- 时间复杂度：`O(n)`
- 空间复杂度：`O(n)`

### 边界情况

- 多位数字，例如 `12[a]`
- 多层嵌套，例如 `3[a2[c]]`
- 普通字符与编码片段混合，例如 `2[abc]3[cd]ef`

### 易错点

- 数字要累积，不能只取单个字符
- 遇到 `[` 时要同时保存“当前数字”和“当前已拼接字符串”
- 遇到 `]` 时拼接顺序不能反，必须是 `prev + cur * repeat`

---

## LeetCode 739 - Daily Temperatures

### 题意
给定一个温度数组 `temperatures`，对于每一天，求还要等几天才会出现更高温度；如果之后都不会升高，就返回 `0`。

### 核心思路
这题是典型的：

**单调栈求下一个更大元素。**

栈里不直接存温度值，而是存**下标**，这样后面才能直接计算天数差。

维护规则：

- 栈中下标对应的温度保持**单调递减**
- 当前温度如果比栈顶温度高，说明栈顶那一天等到了更高温度
- 于是不断弹栈，并计算：
  - `ans[idx] = i - idx`

为什么用下标而不是值？

因为题目要的不是“更大的温度是多少”，而是“还要等几天”。

### Python 代码

~~~python
class Solution(object):
    def dailyTemperatures(self, temperatures):
        n = len(temperatures)
        ans = [0] * n
        stack = []

        for i, t in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < t:
                idx = stack.pop()
                ans[idx] = i - idx
            stack.append(i)

        return ans
~~~

### 复杂度

- 时间复杂度：`O(n)`
- 空间复杂度：`O(n)`

### 边界情况

- 全部递增，例如 `[30, 31, 32, 33]`
- 全部递减，例如 `[33, 32, 31, 30]`
- 有重复值，例如 `[30, 30, 31]`

### 易错点

- 栈里要存下标，不要只存温度
- 比较条件是“严格更高”，所以用 `<`，不是 `<=`
- 最后栈中剩下的下标默认答案就是 `0`

---

## 周四补充小结

今天新增两题：

- 394：栈处理嵌套编码字符串
- 739：单调栈处理下一个更大元素

到这里，Week 03 前四天的栈 / 队列 / 单调栈模板进一步补齐：

- 普通栈：20，155，2390，394
- 队列：232，933，225
- 单调栈：739


---

## LeetCode 496 - Next Greater Element I

### 题意
给定两个数组 `nums1` 和 `nums2`，其中 `nums1` 是 `nums2` 的子集。  
对于 `nums1` 中每个元素，找到它在 `nums2` 中右侧第一个比它大的元素；如果不存在，返回 `-1`。

### 核心思路
这题和 739 属于同一类：

**单调栈 + 哈希映射。**

因为题目最终问的是 `nums1` 中的元素，但“下一个更大元素”的关系发生在 `nums2` 里，所以做法是：

1. 先遍历 `nums2`
2. 用单调递减栈维护“还没找到更大元素”的值
3. 一旦当前值比栈顶大，就说明当前值是栈顶元素的 next greater
4. 把这个关系记到字典里
5. 最后按 `nums1` 顺序取答案

### Python 代码

~~~python
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        stack = []
        next_greater = {}

        for x in nums2:
            while stack and stack[-1] < x:
                next_greater[stack.pop()] = x
            stack.append(x)

        while stack:
            next_greater[stack.pop()] = -1

        return [next_greater[x] for x in nums1]
~~~

### 复杂度

- 时间复杂度：`O(m + n)`
- 空间复杂度：`O(n)`

其中：
- `m = len(nums1)`
- `n = len(nums2)`

### 边界情况

- 所有元素都递减
- 所有元素都递增
- `nums1` 只有一个元素
- 某些元素没有更大值，结果应为 `-1`

### 易错点

- 栈里存的是元素值，不是下标
- 这是 739 的“值映射版”，不是原样抄下标模板
- 最后别忘了把栈里剩余元素统一映射成 `-1`

---

## LeetCode 1047 - Remove All Adjacent Duplicates In String

### 题意
给定一个字符串，反复删除所有相邻且相同的字符，直到不能再删，返回最终结果。

### 核心思路
这题是标准的：

**字符串栈消除。**

做法：

- 如果当前字符和栈顶相同，说明形成一对相邻重复字符，直接弹栈
- 否则就把当前字符压栈
- 最后把栈里的字符拼起来

本质上是“边扫描边消除”，不需要真的多轮反复遍历原字符串。

### Python 代码

~~~python
class Solution(object):
    def removeDuplicates(self, s):
        stack = []

        for ch in s:
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                stack.append(ch)

        return ''.join(stack)
~~~

### 复杂度

- 时间复杂度：`O(n)`
- 空间复杂度：`O(n)`

### 边界情况

- 空字符串
- 全部字符都相同
- 删除一轮后又形成新的相邻重复
- 最终结果为空字符串

### 易错点

- 不要把它想成“循环删除子串”，那样容易写复杂
- 关键是“当前字符与栈顶比较”
- 最后要 `''.join(stack)`，不是直接返回列表

---

## 周五二刷口述记录

### 20 - Valid Parentheses
- 模板类型：基础栈匹配
- 关键点：右括号时先判空，再看是否和栈顶匹配
- 最后必须检查栈是否为空
- 易错点：把不匹配和空栈情况分开写漏掉

### 155 - Min Stack
- 模板类型：辅助栈同步最小值
- 关键点：`min_stack` 每一层都记录“当前位置为止最小值”
- `push / pop` 两个栈必须同步
- 易错点：重复最小值时不能只在“更小”时更新

### 232 - Implement Queue using Stacks
- 模板类型：双栈模拟队列
- 关键点：`out_stack` 空时才把 `in_stack` 倒过去
- `push` 便宜，`pop / peek` 均摊 `O(1)`
- 易错点：每次都倒栈会把思路写乱

### 739 - Daily Temperatures
- 模板类型：单调栈
- 关键点：栈里存下标，不存值
- 当前温度更高时，不断弹出并计算等待天数
- 易错点：比较条件是严格更高，所以用 `<` 而不是 `<=`

---

## 周五补充小结

今天新增两题：

- 496：单调栈 + 哈希映射
- 1047：字符串栈消除

并对这周核心模板做了二刷口述：

- 20
- 155
- 232
- 739

到这里，Week 03 的栈 / 队列 / 单调栈闭环已经比较完整。
