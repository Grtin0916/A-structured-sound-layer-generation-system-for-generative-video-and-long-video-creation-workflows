#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def norm01(values: Dict[str, float], reverse: bool = False) -> Dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if abs(hi - lo) < 1e-12:
        return {k: 0.5 for k in values}
    out = {}
    for k, v in values.items():
        z = (v - lo) / (hi - lo)
        out[k] = 1.0 - z if reverse else z
    return out


def find_numeric(row: Dict[str, Any], needles: List[str]) -> Optional[float]:
    best = None
    best_key = ""
    for k, v in row.items():
        lk = k.lower()
        if any(n in lk for n in needles):
            fv = to_float(v)
            if fv is not None:
                # prefer selector/metric already-normalized score-like fields
                if best is None or ("score" in lk and "score" not in best_key):
                    best = fv
                    best_key = lk
    return best


def variant_family(v: str) -> str:
    v = v.lower()
    if v.startswith("dss"):
        return "dss"
    if v.startswith("naive"):
        return "naive"
    return "other"


def raw_features(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    keys = [r["candidate_key"] for r in rows]
    selector_score = {}
    event_coverage = {}
    forbidden = {}
    onset_penalty = {}
    clip_penalty = {}
    silence_penalty = {}

    for r in rows:
        k = r["candidate_key"]

        selector_score[k] = (
            to_float(r.get("selector_score_v2_probe"))
            or find_numeric(r, ["selector_score", "total_score", "composite_score", "final_score", "score"])
            or 0.0
        )

        event_coverage[k] = (
            find_numeric(r, ["event_coverage", "coverage"])
            or find_numeric(r, ["matched_event"])
            or 0.0
        )

        forbidden[k] = (
            find_numeric(r, ["forbidden", "leakage", "speech_leak"])
            or 0.0
        )

        onset_penalty[k] = (
            find_numeric(r, ["onset_error", "onset_abs", "late", "early"])
            or 0.0
        )

        clip_penalty[k] = (
            find_numeric(r, ["clip", "clipping"])
            or 0.0
        )

        silence_penalty[k] = (
            find_numeric(r, ["silence_ratio", "silent"])
            or 0.0
        )

    return {
        "selector_score": norm01(selector_score),
        "event_coverage": norm01(event_coverage),
        "forbidden_penalty": norm01(forbidden, reverse=True),
        "onset_quality": norm01(onset_penalty, reverse=True),
        "clip_quality": norm01(clip_penalty, reverse=True),
        "silence_quality": norm01(silence_penalty, reverse=True),
    }


def compute_selector_v2(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    feats = raw_features(rows)
    out = []

    for r in rows:
        k = r["candidate_key"]
        family = variant_family(r.get("variant", ""))
        source = r.get("source", "")

        source_reliability = 0.78 if "inferred" in source else 0.65
        dss_bonus = 0.06 if family == "dss" else 0.0
        naive_penalty = -0.03 if family == "naive" else 0.0
        repairability = 0.72 if r.get("is_repair_seed") == "true" else 0.45

        metric_confidence = (
            0.30 * feats["event_coverage"].get(k, 0.5)
            + 0.25 * feats["onset_quality"].get(k, 0.5)
            + 0.20 * feats["forbidden_penalty"].get(k, 0.5)
            + 0.15 * feats["clip_quality"].get(k, 0.5)
            + 0.10 * feats["silence_quality"].get(k, 0.5)
        )

        selector_base = feats["selector_score"].get(k, 0.5)

        score_v2 = (
            0.38 * selector_base
            + 0.30 * metric_confidence
            + 0.14 * source_reliability
            + 0.10 * repairability
            + dss_bonus
            + naive_penalty
        )

        row = dict(r)
        row.update({
            "variant_family": family,
            "selector_base_norm": f"{selector_base:.6f}",
            "metric_confidence": f"{metric_confidence:.6f}",
            "source_reliability": f"{source_reliability:.6f}",
            "repairability": f"{repairability:.6f}",
            "dss_bonus": f"{dss_bonus:.6f}",
            "naive_penalty": f"{naive_penalty:.6f}",
            "selector_v2_score": f"{score_v2:.6f}",
        })
        out.append(row)

    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in out:
        by_case.setdefault(r["case_id"], []).append(r)

    for case, group in by_case.items():
        group.sort(key=lambda x: float(x["selector_v2_score"]), reverse=True)
        for i, r in enumerate(group, 1):
            r["selector_v2_rank"] = i
            r["selector_v2_decision"] = "winner" if i == 1 else ("runner_up" if i == 2 else "repair_or_reject")
            r["selector_v2_rejection_reason"] = "" if i == 1 else explain_reject(r)

    return sorted(out, key=lambda x: (x["case_id"], int(x["selector_v2_rank"])))


def explain_reject(r: Dict[str, Any]) -> str:
    reasons = []
    if float(r["metric_confidence"]) < 0.45:
        reasons.append("low_metric_confidence")
    if float(r["repairability"]) >= 0.70:
        reasons.append("repair_candidate")
    if r["variant_family"] == "naive":
        reasons.append("naive_less_controllable")
    if not reasons:
        reasons.append("lower_selector_v2_score")
    return ";".join(reasons)


def build_pairs(scored: List[Dict[str, Any]], max_pairs: int) -> List[Dict[str, Any]]:
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in scored:
        by_case.setdefault(r["case_id"], []).append(r)

    pairs: List[Dict[str, Any]] = []

    def add_pair(case_id: str, a: Dict[str, Any], b: Dict[str, Any], reason: str) -> None:
        a_score = float(a["selector_v2_score"])
        b_score = float(b["selector_v2_score"])
        preferred = a["candidate_key"] if a_score >= b_score else b["candidate_key"]
        loser = b["candidate_key"] if preferred == a["candidate_key"] else a["candidate_key"]
        margin = abs(a_score - b_score)
        pairs.append({
            "pair_id": f"hp_{len(pairs)+1:03d}",
            "case_id": case_id,
            "candidate_a": a["candidate_key"],
            "candidate_b": b["candidate_key"],
            "variant_a": a["variant"],
            "variant_b": b["variant"],
            "audio_a": a.get("audio_path", ""),
            "audio_b": b.get("audio_path", ""),
            "bootstrap_preferred": preferred,
            "bootstrap_rejected": loser,
            "bootstrap_margin": f"{margin:.6f}",
            "comparison_reason": reason,
            "needs_manual_review": "true",
            "manual_preferred": "",
            "manual_notes": "",
        })

    for case_id, group in by_case.items():
        group = sorted(group, key=lambda x: float(x["selector_v2_score"]), reverse=True)
        winner = group[0]
        runner = group[1] if len(group) > 1 else None
        if runner:
            add_pair(case_id, winner, runner, "winner_vs_runner_up")

        # winner vs best naive/non-winner, or winner vs lowest score
        naive = [g for g in group if variant_family(g.get("variant", "")) == "naive" and g["candidate_key"] != winner["candidate_key"]]
        target = naive[0] if naive else group[-1]
        if target and target["candidate_key"] != winner["candidate_key"]:
            add_pair(case_id, winner, target, "winner_vs_naive_or_lowest")

    # add hardest close-margin pairs
    all_by_case = list(by_case.items())
    for case_id, group in all_by_case:
        group = sorted(group, key=lambda x: float(x["selector_v2_score"]), reverse=True)
        for i in range(len(group) - 1):
            if len(pairs) >= max_pairs:
                break
            a, b = group[i], group[i + 1]
            key = {a["candidate_key"], b["candidate_key"]}
            exists = any({p["candidate_a"], p["candidate_b"]} == key for p in pairs)
            if not exists:
                add_pair(case_id, a, b, "close_margin_audit")
        if len(pairs) >= max_pairs:
            break

    return pairs[:max_pairs]


def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    n = len(labels_a)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    labels = sorted(set(labels_a) | set(labels_b))
    pe = 0.0
    for lab in labels:
        pa = sum(a == lab for a in labels_a) / n
        pb = sum(b == lab for b in labels_b) / n
        pe += pa * pb
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def build_agreement(scored: List[Dict[str, Any]], pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    score = {r["candidate_key"]: float(r["selector_v2_score"]) for r in scored}

    selector_labels = []
    bootstrap_labels = []
    details = []

    for p in pairs:
        a, b = p["candidate_a"], p["candidate_b"]
        selector_preferred = a if score.get(a, -1) >= score.get(b, -1) else b
        bootstrap_preferred = p["bootstrap_preferred"]
        selector_labels.append(selector_preferred)
        bootstrap_labels.append(bootstrap_preferred)
        details.append({
            "pair_id": p["pair_id"],
            "case_id": p["case_id"],
            "selector_preferred": selector_preferred,
            "bootstrap_preferred": bootstrap_preferred,
            "agree": selector_preferred == bootstrap_preferred,
        })

    agreement_rate = sum(d["agree"] for d in details) / len(details) if details else 0.0

    return {
        "pair_count": len(pairs),
        "agreement_rate": agreement_rate,
        "cohen_kappa": cohen_kappa(selector_labels, bootstrap_labels),
        "manual_review_required": True,
        "note": "bootstrap labels are generated from selector_v2 score and must be replaced by real listening labels later",
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out-scored", required=True)
    ap.add_argument("--out-human-seed", required=True)
    ap.add_argument("--out-agreement", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--max-pairs", type=int, default=12)
    args = ap.parse_args()

    inv = read_csv(Path(args.inventory))
    scored = compute_selector_v2(inv)
    pairs = build_pairs(scored, args.max_pairs)
    agreement = build_agreement(scored, pairs)

    scored_fields = list(scored[0].keys()) if scored else []
    pair_fields = [
        "pair_id", "case_id", "candidate_a", "candidate_b",
        "variant_a", "variant_b", "audio_a", "audio_b",
        "bootstrap_preferred", "bootstrap_rejected", "bootstrap_margin",
        "comparison_reason", "needs_manual_review",
        "manual_preferred", "manual_notes",
    ]

    write_csv(Path(args.out_scored), scored, scored_fields)
    write_csv(Path(args.out_human_seed), pairs, pair_fields)
    Path(args.out_agreement).write_text(json.dumps(agreement, indent=2, ensure_ascii=False), encoding="utf-8")

    winner_count = sum(r["selector_v2_decision"] == "winner" for r in scored)
    repair_priority_count = sum(
        r["selector_v2_decision"] != "winner" and float(r["repairability"]) >= 0.70
        for r in scored
    )
    cases_with_pair = len({p["case_id"] for p in pairs})

    summary = {
        "task": "w18_selector_v2_with_human_seed",
        "input_inventory": args.inventory,
        "scored": len(scored),
        "case_count": len({r["case_id"] for r in scored}),
        "winner_count": winner_count,
        "repair_priority_count": repair_priority_count,
        "human_pair_count": len(pairs),
        "cases_with_human_pair": cases_with_pair,
        "agreement_rate": agreement["agreement_rate"],
        "cohen_kappa": agreement["cohen_kappa"],
        "manual_review_required": True,
        "dod": {
            "scored_ge_30": len(scored) >= 30,
            "winner_count_eq_6": winner_count == 6,
            "repair_priority_ge_8": repair_priority_count >= 8,
            "human_pair_count_ge_10": len(pairs) >= 10,
            "each_case_has_pair": cases_with_pair == 6,
            "agreement_json_exists": True,
        },
        "outputs": {
            "scored_csv": args.out_scored,
            "human_seed_csv": args.out_human_seed,
            "agreement_json": args.out_agreement,
            "summary_json": args.out_summary,
        },
    }

    Path(args.out_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
