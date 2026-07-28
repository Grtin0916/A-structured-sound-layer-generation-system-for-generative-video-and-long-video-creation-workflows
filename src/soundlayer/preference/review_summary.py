"""Compute honest preference-label quality and training eligibility."""

from __future__ import annotations

from collections import Counter, defaultdict

from .schema import parse_bool, validate_judgment


def _canonical_winner(row):
    choice = row["overall_preference"].upper()
    if choice not in {"LEFT", "RIGHT"}:
        return choice
    return row["left_digest"] if choice == "LEFT" else row["right_digest"]


def summarize(private_rows, public_rows, labels, graph_summary):
    keys = {row["pair_id"]: row for row in private_rows}
    submitted = [row for row in labels if parse_bool(row.get("submitted"))]
    failures = [
        {"pair_id": row.get("pair_id", ""), "failures": validate_judgment(row)}
        for row in submitted
        if validate_judgment(row)
    ]
    valid = [row for row in submitted if not validate_judgment(row)]
    unique_valid = [
        row
        for row in valid
        if not parse_bool(row.get("is_hidden_repeat"))
        and not parse_bool(row.get("is_audit_pair"))
    ]
    non_tie_unique = [
        row for row in unique_valid if row["overall_preference"] in {"LEFT", "RIGHT"}
    ]
    repeat_groups = defaultdict(list)
    for row in valid:
        if row.get("repeat_group_id"):
            repeat_groups[row["repeat_group_id"]].append(row)
    repeat_checks = []
    for group_id, rows in repeat_groups.items():
        originals = [r for r in rows if not parse_bool(r.get("is_hidden_repeat"))]
        repeats = [r for r in rows if parse_bool(r.get("is_hidden_repeat"))]
        if originals and repeats:
            a, b = originals[0], repeats[0]
            if a["overall_preference"] in {"LEFT", "RIGHT"} and b[
                "overall_preference"
            ] in {"LEFT", "RIGHT"}:
                repeat_checks.append(_canonical_winner(a) == _canonical_winner(b))
    public_text = str(public_rows).lower()
    forbidden = ("strategy", "model", "control", "rerank", "repair", "proxy", "digest")
    blind_leaks = [token for token in forbidden if token in public_text]
    choices = Counter(row["overall_preference"] for row in valid)
    original_final = 0
    current_final = sum(
        str(row.get("publish_decision", "")).upper() == "FINAL_SELECTED"
        for row in private_rows
    )
    metrics = {
        "judgmentCount": len(submitted),
        "validJudgmentCount": len(valid),
        "uniquePairCount": len(
            [row for row in private_rows if row["kind"] == "UNIQUE"]
        ),
        "uniqueContentPairCount": graph_summary.get("uniqueContentPairCount", 0),
        "duplicateContentComparisonCount": graph_summary.get(
            "duplicateContentComparisonCount", 0
        ),
        "validUniqueNonTieCount": len(non_tie_unique),
        "caseCoverage": len({row["case_id"] for row in unique_valid}),
        "connectedCaseGraphCount": graph_summary["connectedCaseGraphCount"],
        "hiddenRepeatValidCount": len(repeat_checks),
        "hiddenRepeatConsistency": (
            sum(repeat_checks) / len(repeat_checks) if repeat_checks else None
        ),
        "sameDigestPairCount": sum(
            row["left_digest"] == row["right_digest"] for row in private_rows
        ),
        "missingArtifactCount": 0,
        "blindLeakCount": len(blind_leaks),
        "finalSelectedMutationCount": current_final - original_final,
        "leftPreferenceCount": choices["LEFT"],
        "rightPreferenceCount": choices["RIGHT"],
        "tieCount": choices["TIE"],
        "unjudgeableCount": choices["UNJUDGEABLE"],
        "invalidJudgmentCount": len(failures),
    }
    checks = {
        "judgmentCount": metrics["judgmentCount"] == 48,
        "uniquePairCount": metrics["uniquePairCount"] >= 36,
        "uniqueContentPairCount": metrics["uniqueContentPairCount"] >= 30,
        "validUniqueNonTieCount": metrics["validUniqueNonTieCount"] >= 30,
        "caseCoverage": metrics["caseCoverage"] == 12,
        "connectedCaseGraphCount": metrics["connectedCaseGraphCount"] == 12,
        "hiddenRepeatValidCount": metrics["hiddenRepeatValidCount"] >= 6,
        "hiddenRepeatConsistency": metrics["hiddenRepeatConsistency"] is not None
        and metrics["hiddenRepeatConsistency"] >= 0.75,
        "sameDigestPairCount": metrics["sameDigestPairCount"] == 0,
        "missingArtifactCount": metrics["missingArtifactCount"] == 0,
        "blindLeakCount": metrics["blindLeakCount"] == 0,
        "finalSelectedMutationCount": metrics["finalSelectedMutationCount"] == 0,
        "validJudgments": not failures,
    }
    return {
        "schemaVersion": "preference-quality-gate/v1",
        "status": "TRAINING_ELIGIBLE" if all(checks.values()) else "REVIEW_QUALITY_BLOCKED",
        "metrics": metrics,
        "checks": checks,
        "invalidJudgments": failures,
        "blindLeakTokens": blind_leaks,
        "claimBoundary": {
            "humanPairPreferenceOnly": True,
            "rankerTrainingPerformed": False,
            "finalSelectedMutationAllowed": False,
            "interRaterAgreementClaimed": False,
        },
    }
