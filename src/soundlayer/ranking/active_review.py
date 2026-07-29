"""Guards and acquisition primitives for selective active review."""

from __future__ import annotations

import itertools


def canonical_key(case_id, digest_a, digest_b):
    left, right = sorted((digest_a, digest_b))
    return f"{case_id}:{left}:{right}"


def existing_content_keys(private_pairs):
    return {
        canonical_key(row["case_id"], row["left_digest"], row["right_digest"])
        for row in private_pairs
    }


def unlabeled_pairs(candidates, existing_keys):
    by_case = {}
    for candidate in candidates:
        by_case.setdefault(candidate["case_id"], []).append(candidate)
    output = []
    for case_id in sorted(by_case):
        seen = set()
        for left, right in itertools.combinations(by_case[case_id], 2):
            if left["artifact_digest"] == right["artifact_digest"]:
                continue
            key = canonical_key(
                case_id, left["artifact_digest"], right["artifact_digest"]
            )
            if key in existing_keys or key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "case_id": case_id,
                    "canonical_pair_key": key,
                    "left_digest": left["artifact_digest"],
                    "right_digest": right["artifact_digest"],
                    "left_artifact": left["artifact_path"],
                    "right_artifact": right["artifact_path"],
                }
            )
    return output


def acquisition_score(probability, rule_disagreement=False, graph_closure=False):
    uncertainty = 1.0 - 2.0 * abs(float(probability) - 0.5)
    return uncertainty + float(rule_disagreement) + float(graph_closure)
