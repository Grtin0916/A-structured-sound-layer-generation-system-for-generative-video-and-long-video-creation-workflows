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
