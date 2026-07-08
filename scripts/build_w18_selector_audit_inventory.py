#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def first_existing(row: Dict[str, Any], names: List[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
    return default


def norm_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    candidate_id = first_existing(row, ["candidate_id", "candidate", "audio_id", "id", "wav_id"])
    case_id = first_existing(row, ["case_id", "case", "scene_id"])
    variant = first_existing(row, ["variant", "prompt_variant", "prompt_type", "condition"])
    source = first_existing(row, ["source", "model_source", "model", "generator", "route"])
    audio_path = first_existing(row, ["audio_path", "wav_path", "path", "candidate_path"])
    if not candidate_id:
        candidate_id = "|".join([case_id, variant, source, audio_path]).strip("|")
    return candidate_id, case_id, variant, source


def flatten_repair_ids(obj: Any) -> set:
    ids = set()

    def visit(x: Any):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in {"candidate_id", "candidate", "audio_id", "id"} and isinstance(v, (str, int, float)):
                    ids.add(str(v))
                else:
                    visit(v)
        elif isinstance(x, list):
            for item in x:
                visit(item)

    visit(obj)
    return ids


def prefix_row(prefix: str, row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in row.items():
        if k:
            out[f"{prefix}_{k}"] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--pairwise-csv", required=True)
    ap.add_argument("--selector-csv", required=True)
    ap.add_argument("--repair-seed", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--failures-json", required=True)
    args = ap.parse_args()

    metrics_path = Path(args.metrics_csv)
    pairwise_path = Path(args.pairwise_csv)
    selector_path = Path(args.selector_csv)
    repair_path = Path(args.repair_seed)

    metrics = read_csv(metrics_path)
    pairwise = read_csv(pairwise_path)
    selector = read_csv(selector_path)
    repair_obj = read_json(repair_path)
    repair_ids = flatten_repair_ids(repair_obj)

    inventory: Dict[str, Dict[str, Any]] = {}

    def ensure(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        cid, case_id, variant, source = norm_key(row)
        if not cid:
            cid = f"unknown_{len(inventory)+1:04d}"
        if cid not in inventory:
            inventory[cid] = {
                "candidate_id": cid,
                "case_id": case_id,
                "variant": variant,
                "source": source,
            }
        else:
            if case_id and not inventory[cid].get("case_id"):
                inventory[cid]["case_id"] = case_id
            if variant and not inventory[cid].get("variant"):
                inventory[cid]["variant"] = variant
            if source and not inventory[cid].get("source"):
                inventory[cid]["source"] = source
        return cid, inventory[cid]

    for row in metrics:
        cid, item = ensure(row)
        item.update(prefix_row("metric", row))
        item["has_metrics"] = "true"

    for row in selector:
        cid, item = ensure(row)
        item.update(prefix_row("selector", row))
        item["has_selector"] = "true"

    pairwise_by_case: Dict[str, int] = {}
    for row in pairwise:
        case_id = first_existing(row, ["case_id", "case", "scene_id"])
        if case_id:
            pairwise_by_case[case_id] = pairwise_by_case.get(case_id, 0) + 1

        # Pairwise rows may describe two candidates. Attach case-level coverage first.
        for key in ["candidate_id", "winner_candidate_id", "loser_candidate_id", "candidate_a", "candidate_b"]:
            if key in row and row[key]:
                pseudo = dict(row)
                pseudo["candidate_id"] = row[key]
                cid, item = ensure(pseudo)
                item.update(prefix_row("pairwise", row))
                item["has_pairwise"] = "true"

    for item in inventory.values():
        cid = item["candidate_id"]
        item["is_repair_seed"] = "true" if cid in repair_ids else "false"

        case_id = item.get("case_id", "")
        item["pairwise_case_count"] = str(pairwise_by_case.get(case_id, 0))

        selector_blob = " ".join(str(v).lower() for k, v in item.items() if k.startswith("selector_"))
        metric_blob = " ".join(str(v).lower() for k, v in item.items() if k.startswith("metric_"))

        is_winner = any(token in selector_blob for token in ["winner", "selected", "rank=1", "rank_1", "true"])
        item["is_selector_winner_proxy"] = "true" if is_winner else "false"

        has_reason = any(
            ("reason" in k.lower() or "why" in k.lower() or "decision" in k.lower())
            and str(v).strip()
            for k, v in item.items()
        )
        has_metric = item.get("has_metrics") == "true" or bool(metric_blob.strip())
        item["has_reason_or_metric"] = "true" if (has_reason or has_metric) else "false"

        missing = []
        for required in ["case_id", "variant", "source"]:
            if not item.get(required):
                missing.append(required)
        if item.get("has_metrics") != "true":
            missing.append("metrics")
        if item.get("has_selector") != "true":
            missing.append("selector")
        item["missing_evidence"] = ",".join(missing)

    rows = list(inventory.values())
    rows.sort(key=lambda r: (r.get("case_id", ""), r.get("variant", ""), r.get("source", ""), r.get("candidate_id", "")))

    all_fields = []
    seen = set()
    preferred = [
        "candidate_id", "case_id", "variant", "source",
        "has_metrics", "has_selector", "has_pairwise",
        "pairwise_case_count", "is_selector_winner_proxy",
        "is_repair_seed", "has_reason_or_metric", "missing_evidence",
    ]
    for f in preferred:
        all_fields.append(f)
        seen.add(f)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                all_fields.append(k)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    case_count = len({r.get("case_id", "") for r in rows if r.get("case_id")})
    variant_count = len({r.get("variant", "") for r in rows if r.get("variant")})
    winner_count = sum(1 for r in rows if r.get("is_selector_winner_proxy") == "true")
    repair_seed_count = sum(1 for r in rows if r.get("is_repair_seed") == "true")
    all_candidates_have_reason_or_metric = all(r.get("has_reason_or_metric") == "true" for r in rows) if rows else False

    failures = []
    for r in rows:
        if r.get("missing_evidence"):
            failures.append({
                "candidate_id": r.get("candidate_id"),
                "case_id": r.get("case_id"),
                "variant": r.get("variant"),
                "missing_evidence": r.get("missing_evidence"),
            })

    summary = {
        "task": "w18_selector_audit_inventory",
        "inputs": {
            "metrics_csv": str(metrics_path),
            "pairwise_csv": str(pairwise_path),
            "selector_csv": str(selector_path),
            "repair_seed": str(repair_path),
        },
        "candidate_count": len(rows),
        "case_count": case_count,
        "variant_count": variant_count,
        "winner_count_proxy": winner_count,
        "repair_seed_count": repair_seed_count,
        "all_candidates_have_reason_or_metric": all_candidates_have_reason_or_metric,
        "failure_count": len(failures),
        "outputs": {
            "inventory_csv": str(out_csv),
            "summary_json": args.out_summary,
            "failures_json": args.failures_json,
        },
        "dod": {
            "candidate_count_ge_30": len(rows) >= 30,
            "case_count_eq_6": case_count == 6,
            "winner_count_ge_6": winner_count >= 6,
            "repair_seed_count_ge_8": repair_seed_count >= 8,
            "all_candidates_have_reason_or_metric": all_candidates_have_reason_or_metric,
        },
    }

    Path(args.out_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.failures_json).write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
