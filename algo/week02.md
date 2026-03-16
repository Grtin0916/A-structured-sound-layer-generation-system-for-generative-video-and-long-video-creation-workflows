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
