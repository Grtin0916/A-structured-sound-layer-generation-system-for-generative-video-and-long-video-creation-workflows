"""LeetCode 146: LRU Cache with O(1) get/put operations."""

from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.values: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.values:
            return -1
        self.values.move_to_end(key)
        return self.values[key]

    def put(self, key: int, value: int) -> None:
        if key in self.values:
            self.values.move_to_end(key)
        self.values[key] = value
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)


def self_test() -> None:
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)
    assert cache.get(2) == -1
    cache.put(4, 4)
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

    single = LRUCache(1)
    single.put(-1, 7)
    single.put(-1, 8)
    assert single.get(-1) == 8
    single.put(2, 9)
    assert single.get(-1) == -1
    assert single.get(2) == 9
    print("LC146 9/9 passed")


if __name__ == "__main__":
    self_test()
