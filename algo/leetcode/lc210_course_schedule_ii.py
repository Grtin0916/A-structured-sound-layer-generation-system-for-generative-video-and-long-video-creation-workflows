"""LeetCode 210: deterministic topological ordering for the release DAG."""

from heapq import heapify, heappop, heappush


def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    if num_courses < 0:
        raise ValueError("num_courses must be non-negative")
    graph = [set() for _ in range(num_courses)]
    indegree = [0] * num_courses
    for course, prerequisite in prerequisites:
        if not 0 <= course < num_courses or not 0 <= prerequisite < num_courses:
            raise ValueError("course index outside graph")
        if course not in graph[prerequisite]:
            graph[prerequisite].add(course)
            indegree[course] += 1
    ready = [course for course, count in enumerate(indegree) if count == 0]
    heapify(ready)
    order = []
    while ready:
        prerequisite = heappop(ready)
        order.append(prerequisite)
        for course in sorted(graph[prerequisite]):
            indegree[course] -= 1
            if indegree[course] == 0:
                heappush(ready, course)
    return order if len(order) == num_courses else []


def self_test() -> None:
    assert find_order(0, []) == []
    assert find_order(1, []) == [0]
    assert find_order(4, [[1, 0], [2, 1], [3, 2]]) == [0, 1, 2, 3]
    assert find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]) == [0, 1, 2, 3]
    assert find_order(5, [[1, 0], [4, 3]]) == [0, 1, 2, 3, 4]
    assert find_order(2, [[0, 1], [1, 0]]) == []
    assert find_order(1, [[0, 0]]) == []
    assert find_order(2, [[1, 0], [1, 0]]) == [0, 1]
    assert find_order(8, [[1, 0], [2, 0], [3, 1], [3, 2], [4, 3], [5, 3], [6, 4], [6, 5], [7, 6]]) == list(range(8))
    try:
        find_order(2, [[2, 0]])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid course index was accepted")
    print("LC210 10/10 passed")


if __name__ == "__main__":
    self_test()
