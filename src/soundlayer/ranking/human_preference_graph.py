"""Build honest per-case partial orders from submitted pairwise judgments."""

from __future__ import annotations

from collections import defaultdict


def _submitted(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _bool(value):
    return str(value).strip().lower() == "true"


def hydrate(labels, private_pairs):
    truth = {row["pair_id"]: row for row in private_pairs}
    return [{**truth.get(row.get("pair_id", ""), {}), **row} for row in labels]


def _has_cycle(nodes, edges):
    adjacency = defaultdict(set)
    for winner, loser in edges:
        adjacency[winner].add(loser)
    visiting, visited = set(), set()

    def visit(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in adjacency[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in nodes)


def _reachable(start, edges):
    adjacency = defaultdict(set)
    for winner, loser in edges:
        adjacency[winner].add(loser)
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for child in adjacency[node]:
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def build_human_graph(labels, private_pairs):
    labels = hydrate(labels, private_pairs)
    candidates = defaultdict(set)
    for row in private_pairs:
        candidates[row["case_id"]].update((row["left_digest"], row["right_digest"]))
    by_case = defaultdict(list)
    for row in labels:
        by_case[row.get("case_id", "")].append(row)
    results = []
    for case_id in sorted(candidates):
        decisive, ties, unjudgeable = [], 0, 0
        observed_keys = set()
        for row in by_case[case_id]:
            if not _submitted(row.get("submitted")):
                continue
            if _bool(row.get("is_hidden_repeat")) or _bool(row.get("is_audit_pair")):
                continue
            key = tuple(sorted((row["left_digest"], row["right_digest"])))
            if key in observed_keys:
                continue
            observed_keys.add(key)
            choice = row.get("overall_preference", "").upper()
            if choice == "LEFT":
                decisive.append((row["left_digest"], row["right_digest"]))
            elif choice == "RIGHT":
                decisive.append((row["right_digest"], row["left_digest"]))
            elif choice == "TIE":
                ties += 1
            else:
                unjudgeable += 1
        nodes = sorted(candidates[case_id])
        cycle = _has_cycle(nodes, decisive)
        indegree = {node: 0 for node in nodes}
        for _, loser in decisive:
            indegree[loser] += 1
        tops = [node for node in nodes if indegree[node] == 0 and decisive]
        unique_top = (
            len(tops) == 1
            and len(_reachable(tops[0], decisive)) == len(nodes) - 1
            and not cycle
        )
        if not observed_keys:
            status = "INSUFFICIENT_COMPARISON"
            tops = []
        elif cycle:
            status = "PREFERENCE_CYCLE"
            tops = []
        elif unique_top:
            status = "UNIQUE_TOP"
        else:
            status = "PARTIAL_ORDER"
        results.append(
            {
                "case_id": case_id,
                "candidate_count": len(nodes),
                "observed_edge_count": len(observed_keys),
                "decisive_edge_count": len(decisive),
                "tie_count": ties,
                "unjudgeable_count": unjudgeable,
                "cycle_count": int(cycle),
                "transitivity_violation_count": int(cycle),
                "reference_status": status,
                "human_top_candidates": tops,
                "edges": [
                    {"preferred": winner, "not_preferred": loser}
                    for winner, loser in decisive
                ],
            }
        )
    return {
        "schemaVersion": "human-preference-graph/v1",
        "summary": {
            "caseCount": len(results),
            "observedEdgeCount": sum(row["observed_edge_count"] for row in results),
            "cycleCaseCount": sum(row["cycle_count"] > 0 for row in results),
            "uniqueTopCaseCount": sum(
                row["reference_status"] == "UNIQUE_TOP" for row in results
            ),
            "insufficientComparisonCaseCount": sum(
                row["reference_status"] == "INSUFFICIENT_COMPARISON"
                for row in results
            ),
            "humanWinnerFabricationCount": 0,
        },
        "cases": results,
    }
