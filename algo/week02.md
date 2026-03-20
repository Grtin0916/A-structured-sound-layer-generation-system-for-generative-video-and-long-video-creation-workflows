# Week 02 - Linked List

## 2026-03-16

### 206. Reverse Linked List
- Link: https://leetcode.com/problems/reverse-linked-list/
- Status: AC
- Core idea:
  - 迭代三指针：prev / curr / nxt
  - 每次把 curr.next 指回 prev，然后整体向前推进
- Complexity:
  - Time: O(n)
  - Space: O(1)
- Boundary cases:
  - 空链表
  - 单节点
  - 两节点
- Easiest-to-break points:
  - 还没保存 next，就先改 curr.next，导致后面链断掉
  - 循环结束后返回了 head，而不是 prev
- One-line template:
  - “先存后继，再反转指针，再整体前移，最后返回 prev。”

### 876. Middle of the Linked List
- Link: https://leetcode.com/problems/middle-of-the-linked-list/
- Status: AC
- Core idea:
  - 快慢指针：fast 一次走 2 步，slow 一次走 1 步
  - 当 fast 到尾部时，slow 正好在中点
  - 如果有两个中点，题目要求返回第二个中点
- Complexity:
  - Time: O(n)
  - Space: O(1)
- Boundary cases:
  - 空链表
  - 单节点
  - 偶数长度链表
  - 奇数长度链表
- Easiest-to-break points:
  - while 条件写错，导致 fast.next 访问空指针
  - 偶数长度时返回了第一个中点，而不是第二个
- One-line template:
  - “快两步慢一步，fast 到尾时，slow 就在题目要的中点。”

# LeetCode 83 - 删除排序链表中的重复元素

## 题意

给定一个**已排序**链表，删除所有重复元素，使得每个元素只出现一次，并返回处理后的链表。

例如：

- 输入：`1 -> 1 -> 2`
- 输出：`1 -> 2`

---

## 核心思路

这题的关键前提是：**链表已经排序**。  
所以重复元素一定是**相邻出现**的，不需要哈希表，不需要双重遍历。

做法很直接：

1. 用一个指针 `cur` 从头节点开始遍历；
2. 只要 `cur` 和 `cur.next` 都存在，就比较它们的值：
   - 如果 `cur.val == cur.next.val`，说明当前和下一个重复；
     - 直接执行 `cur.next = cur.next.next`
     - 含义是：**跳过重复节点**
   - 否则，说明当前值和下一个值不同；
     - 才让 `cur = cur.next`，继续往后走
3. 最后返回 `head`

这里最容易写错的点是：  
**删除重复节点后，`cur` 先不要后移。**  
因为删掉一个重复节点以后，新的 `cur.next` 还有可能继续重复，必须继续检查。

---

## Python 代码

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        cur = head

        while cur and cur.next:
            if cur.val == cur.next.val:
                # 跳过重复节点
                cur.next = cur.next.next
            else:
                # 只有不重复时才后移
                cur = cur.next

        return head
```

---

## 复杂度

- 时间复杂度：`O(n)`
  - 每个节点最多被访问一次
- 空间复杂度：`O(1)`
  - 只使用了一个指针变量

---

## 易错点

### 1. 删除后立刻后移指针
错误写法思路：

```python
if cur.val == cur.next.val:
    cur.next = cur.next.next
    cur = cur.next
```

这样会漏掉连续重复，例如：

- 输入：`1 -> 1 -> 1 -> 2`

第一次删掉一个 `1` 后，如果立刻后移，就可能漏检查剩下那个重复 `1`。

---

### 2. 忘记判空
循环条件必须写成：

```python
while cur and cur.next:
```

因为你要访问 `cur.next.val`，所以 `cur.next` 不能是 `None`。

---

### 3. 误以为这题要删除“所有重复值”
这题不是 LeetCode 82。  
83 的要求是：

- 重复元素**保留一个**

例如：

- `1 -> 1 -> 2 -> 3 -> 3`
- 输出是：`1 -> 2 -> 3`

不是把 `1` 和 `3` 全部删光。

---

## 边界情况

### 情况 1：空链表
- 输入：`[]`
- 输出：`[]`

### 情况 2：只有一个节点
- 输入：`[1]`
- 输出：`[1]`

### 情况 3：全部重复
- 输入：`[1,1,1,1]`
- 输出：`[1]`

### 情况 4：完全没有重复
- 输入：`[1,2,3,4]`
- 输出：`[1,2,3,4]`

---

## 一句话总结

这题本质就是：

> **利用“有序”这个条件，只处理相邻重复节点；重复就删，不重复才前进。**

---

# LeetCode 141 - 环形链表

## 题意

给你一个链表的头节点 `head`，判断链表中是否有环。

如果链表中某个节点可以通过不断沿着 `next` 指针再次回到自己，就说明有环。

返回：

- `True`：有环
- `False`：无环

---

## 核心思路

这题最经典的做法是：**快慢指针**。

定义两个指针：

- `slow`：每次走 1 步
- `fast`：每次走 2 步

然后分两种情况：

### 情况 1：链表无环
`fast` 一定会先走到 `None`，因为它跑得更快，最后冲出链表。

### 情况 2：链表有环
`fast` 虽然起初更快，但因为环会“绕圈”，最终一定会在环里追上 `slow`。  
这和操场跑步一个逻辑：跑得快的人迟早套圈。

所以流程就是：

1. 初始化 `slow = head`，`fast = head`
2. 当 `fast` 和 `fast.next` 存在时继续循环
3. 每次：
   - `slow = slow.next`
   - `fast = fast.next.next`
4. 如果某一时刻 `slow == fast`，说明有环
5. 如果循环结束还没相遇，说明无环

---

## Python 代码

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: bool
        """
        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False
```

---

## 复杂度

- 时间复杂度：`O(n)`
  - 无环时，最多遍历到链表末尾
  - 有环时，快慢指针会在环内相遇
- 空间复杂度：`O(1)`
  - 只用了两个指针

---

## 为什么快慢指针一定能判断有环

这是这题最重要的理解点。

假设：

- 慢指针每轮走 1 步
- 快指针每轮走 2 步

那么每轮快指针相对慢指针会多走 1 步。  
如果链表有环，这个“相对距离”会不断缩小，最终一定在环内相遇。

你可以理解成：

- 慢指针在环上匀速走
- 快指针在同一个环上更快走
- 快指针迟早会从后面追上慢指针

这就是为什么**相遇 = 有环**。

---

## 易错点

### 1. 循环条件写错
很多人会写成：

```python
while fast:
```

这不够，因为循环体里有：

```python
fast = fast.next.next
```

所以必须保证 `fast.next` 也存在。  
正确写法是：

```python
while fast and fast.next:
```

---

### 2. 把节点值相等当成相遇
判断相遇应该写：

```python
if slow == fast:
```

不要写成：

```python
if slow.val == fast.val:
```

因为不同节点可能值一样，但并不表示它们是同一个节点。

---

### 3. 用哈希表做也可以，但不是最优
另一种做法是把访问过的节点放进 `set`，如果再次访问到同一个节点就说明有环：

```python
seen = set()
while head:
    if head in seen:
        return True
    seen.add(head)
    head = head.next
return False
```

这个也能过，但空间复杂度是 `O(n)`。  
快慢指针更优，属于这题标准解。

---

## 边界情况

### 情况 1：空链表
- 输入：`head = None`
- 输出：`False`

### 情况 2：只有一个节点，且没有环
- `1 -> None`
- 输出：`False`

### 情况 3：只有一个节点，但它指向自己
- `1 -> 1`
- 输出：`True`

### 情况 4：多个节点，尾部连回中间节点
- `3 -> 2 -> 0 -> -4`
- `-4.next = 2`
- 输出：`True`

---

## 一句话总结

这题本质就是：

> **无环时快指针先出界；有环时快指针一定会追上慢指针。**

---

## 今天这两题的共同点

这两题都属于**链表指针控制**题，核心不是“想法难”，而是：

- 你是否清楚指针什么时候该移动
- 你是否清楚修改 `next` 后链表结构发生了什么
- 你是否能守住边界条件，不把 `None` 解引用

今天这两题吃透，链表基础就不是纸糊的了。


## 2026-03-20

### 24. Swap Nodes in Pairs
- Link: https://leetcode.com/problems/swap-nodes-in-pairs/
- Status: Review
- Core idea:
  - 用 dummy 节点统一处理头节点交换
  - 每轮锁定 `first` 和 `second` 两个节点做局部交换
  - 交换完成后把 `prev` 移到这一轮交换后的尾部，继续处理下一对
- Complexity:
  - Time: O(n)
  - Space: O(1)
- Boundary cases:
  - 空链表
  - 单节点
  - 奇数长度链表
- Easiest-to-break points:
  - 没有 dummy，导致头两个节点交换时很别扭
  - 交换顺序写错，先改丢链
  - `prev` 没移到新尾巴，下一轮接不上
- One-line template:
  - “dummy 护头，按对交换，交换完让 prev 挂到这一对的新尾部。”

### 328. Odd Even Linked List
- Link: https://leetcode.com/problems/odd-even-linked-list/
- Status: Review
- Core idea:
  - 按“节点位置奇偶”拆成两条链，不是按节点值奇偶
  - `odd` 串起第 1、3、5... 个节点
  - `even` 串起第 2、4、6... 个节点
  - 最后把 odd 链尾接到 even 头
- Complexity:
  - Time: O(n)
  - Space: O(1)
- Boundary cases:
  - 空链表
  - 只有 1 个节点
  - 只有 2 个节点
- Easiest-to-break points:
  - 把题意看成值的奇偶，不是位置的奇偶
  - 忘了保存 `even_head`，最后接不回去
  - 循环条件写错，访问空指针
- One-line template:
  - “奇链走奇位，偶链走偶位，最后奇尾接偶头。”

### 2. Add Two Numbers
- Link: https://leetcode.com/problems/add-two-numbers/
- Status: Review
- Core idea:
  - 两条链表同步往后走，逐位相加
  - 用 `carry` 处理进位
  - 用 dummy 节点串起结果链表
  - 任一链表没了就补 0，最后若 `carry` 还在，再补一个节点
- Complexity:
  - Time: O(max(m, n))
  - Space: O(max(m, n))
- Boundary cases:
  - 两条链表长度不同
  - 最后一位相加后还要进位
  - 某一条链表为空
- Easiest-to-break points:
  - 忘了处理最后一个 `carry`
  - 只在两个链表都存在时循环，漏掉较长链表剩余部分
  - 结果链表没有 dummy，头节点处理容易乱
- One-line template:
  - “逐位相加带进位，dummy 串结果，最后别漏 carry。”

### 86. Partition List
- Link: https://leetcode.com/problems/partition-list/
- Status: Review
- Core idea:
  - 建两条链：
    - `< x` 的 small 链
    - `>= x` 的 large 链
  - 原链表遍历一遍，按条件分别挂到两条链后面
  - 遍历结束后，先把 `large` 尾置空，再把 `small` 尾接到 `large` 头
- Complexity:
  - Time: O(n)
  - Space: O(1)
- Boundary cases:
  - 所有节点都 `< x`
  - 所有节点都 `>= x`
  - 链表为空
- Easiest-to-break points:
  - 没把 `large_tail.next = None`，容易形成脏链接或环
  - 直接在原链上乱改，导致顺序被破坏
  - 合并两条链时忘了处理其中一条为空
- One-line template:
  - “小的挂 small，大的挂 large，最后 small 接 large，记得 large 先断尾。”

