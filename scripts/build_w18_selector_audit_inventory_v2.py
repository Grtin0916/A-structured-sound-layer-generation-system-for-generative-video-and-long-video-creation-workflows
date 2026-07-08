#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CASE_KEYS = ["case_id", "case", "scene_id"]
VARIANT_KEYS = ["variant", "prompt_variant", "prompt_type", "condition"]
SOURCE_KEYS = ["source", "model_source", "model", "generator", "route"]
PATH_KEYS = ["audio_path", "wav_path", "path", "candidate_path", "output_path"]
ID_KEYS = ["candidate_id", "candidate", "audio_id", "id", "wav_id"]


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


def pick(row: Dict[str, Any], keys: Iterable[str]) -> str:
    lower = {k.lower(): k for k in row.keys()}
    for key in keys:
        real = lower.get(key.lower())
        if real and row.get(real) not in (None, ""):
            return str(row[real]).strip()
    for k, v in row.items():
        lk = k.lower()
        if any(key.lower() in lk for key in keys) and v not in (None, ""):
            return str(v).strip()
    return ""


def infer_from_pipe_id(s: str) -> Tuple[str, str, str]:
    if not s:
        return "", "", ""
    path = ""
    core = s
    if "||" in s:
        core, path = s.split("||", 1)
    parts = core.split("|")
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip(), path.strip()
    return "", "", path.strip()


def infer_from_path(s: str) -> Tuple[str, str]:
    if not s:
        return "", ""
    parts = Path(s).parts
    if "w18_dss_ablation" in parts:
        i = parts.index("w18_dss_ablation")
        if len(parts) > i + 2:
            return parts[i + 1], parts[i + 2]
    # fallback: find known case-like token and following dss/naive token
    toks = re.split(r"[\\/]", s)
    for i, t in enumerate(toks[:-1]):
        if re.search(r"_\d{3}$", t) and i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt.startswith("dss_") or nxt in {"naive", "baseline", "control"}:
                return t, nxt
    return "", ""


def canonical_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    cid = pick(row, ID_KEYS)
    case_id, variant, pipe_path = infer_from_pipe_id(cid)

    if not case_id:
        case_id = pick(row, CASE_KEYS)
    if not variant:
        variant = pick(row, VARIANT_KEYS)

    path = pick(row, PATH_KEYS) or pipe_path
    if (not case_id or not variant) and path:
        p_case, p_variant = infer_from_path(path)
        case_id = case_id or p_case
        variant = variant or p_variant

    if not case_id or not variant:
        return "", case_id, variant

    return f"{case_id}|{variant}", case_id, variant


def prefix(prefix_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix_name}_{k}": v for k, v in row.items() if k}


def to_float(x: Any) -> Optional[float]:
    try:
        if x in (None, ""):
            return None
        return float(str(x).strip())
    except Exception:
        return None


def truthy(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "winner", "selected", "accept", "accepted", "pass"}


def detect_explicit_winner(item: Dict[str, Any]) -> Tuple[bool, str]:
    for k, v in item.items():
        lk = k.lower()
        sv = str(v).strip().lower()
        if not lk.startswith("selector_"):
            continue
        if any(t in lk for t in ["winner", "selected", "chosen", "is_best"]) and truthy(v):
            return True, f"explicit:{k}"
        if "rank" in lk:
            fv = to_float(v)
            if fv is not None and abs(fv - 1.0) < 1e-9:
                return True, f"rank1:{k}"
        if any(t in lk for t in ["decision", "status", "label", "reason"]) and any(
            t in sv for t in ["winner", "selected", "accepted", "top"]
        ):
            return True, f"decision:{k}"
    return False, ""


def selector_score(item: Dict[str, Any]) -> Tuple[Optional[float], str]:
    priority_names = ["score", "total", "composite", "final", "utility", "ranker"]
    candidates: List[Tuple[float, str]] = []

    for k, v in item.items():
        lk = k.lower()
        if not lk.startswith("selector_"):
            continue
        fv = to_float(v)
        if fv is None:
            continue
        if any(name in lk for name in priority_names):
            candidates.append((fv, k))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0]

    # final fallback: any numeric selector field except obvious rank/count/id
    for k, v in item.items():
        lk = k.lower()
        if not lk.startswith("selector_"):
            continue
        if any(bad in lk for bad in ["rank", "count", "id", "index"]):
            continue
        fv = to_float(v)
        if fv is not None:
            candidates.append((fv, k))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0]

    return None, ""


def compact_units(obj: Any) -> List[str]:
    units: List[str] = []

    def visit(x: Any):
        if isinstance(x, dict):
            txt = json.dumps(x, ensure_ascii=False).lower()
            if len(txt) <= 2500:
                units.append(txt)
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
        elif isinstance(x, str):
            if len(x) <= 1000:
                units.append(x.lower())

    visit(obj)
    return units


def repair_keys_from_json(obj: Any, known_keys: Iterable[str]) -> Dict[str, str]:
    known = list(known_keys)
    units = compact_units(obj)
    out: Dict[str, str] = {}

    for key in known:
        if "|" not in key:
            continue
        case_id, variant = key.split("|", 1)
        for u in units:
            if case_id.lower() in u and variant.lower() in u:
                out[key] = "case_variant_in_repair_seed"
                break

    # exact candidate strings fallback
    raw = json.dumps(obj, ensure_ascii=False).lower()
    for key in known:
        if key.lower() in raw and key not in out:
            out[key] = "exact_key_in_repair_seed"

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--pairwise-csv", required=True)
    ap.add_argument("--selector-csv", required=True)
    ap.add_argument("--repair-seed", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-issues", required=True)
    args = ap.parse_args()

    metrics_rows = read_csv(Path(args.metrics_csv))
    pairwise_rows = read_csv(Path(args.pairwise_csv))
    selector_rows = read_csv(Path(args.selector_csv))
    repair_obj = read_json(Path(args.repair_seed))

    inv: Dict[str, Dict[str, Any]] = {}

    def ensure(row: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        key, case_id, variant = canonical_key(row)
        if not key:
            return "", None
        item = inv.setdefault(key, {
            "candidate_key": key,
            "case_id": case_id,
            "variant": variant,
            "source": "",
            "audio_path": "",
            "has_metrics": "false",
            "has_selector": "false",
            "has_pairwise": "false",
        })

        src = pick(row, SOURCE_KEYS)
        path = pick(row, PATH_KEYS)
        cid = pick(row, ID_KEYS)
        _, _, pipe_path = infer_from_pipe_id(cid)
        path = path or pipe_path

        if src and not item["source"]:
            item["source"] = src
        if path and not item["audio_path"]:
            item["audio_path"] = path
        if not item["source"] and (path or pipe_path):
            item["source"] = "w18_dss_ablation_inferred"

        return key, item

    for r in metrics_rows:
        _, item = ensure(r)
        if item is None:
            continue
        item.update(prefix("metric", r))
        item["has_metrics"] = "true"

    for r in selector_rows:
        _, item = ensure(r)
        if item is None:
            continue
        item.update(prefix("selector", r))
        item["has_selector"] = "true"

    pairwise_case_count: Dict[str, int] = {}
    for r in pairwise_rows:
        case_id = pick(r, CASE_KEYS)
        if case_id:
            pairwise_case_count[case_id] = pairwise_case_count.get(case_id, 0) + 1

    for item in inv.values():
        item["pairwise_case_count"] = str(pairwise_case_count.get(item["case_id"], 0))
        item["has_pairwise"] = "true" if pairwise_case_count.get(item["case_id"], 0) > 0 else "false"

    repair_match = repair_keys_from_json(repair_obj, inv.keys())
    for key, item in inv.items():
        item["is_repair_seed"] = "true" if key in repair_match else "false"
        item["repair_seed_match_rule"] = repair_match.get(key, "")

    # explicit winner first
    for item in inv.values():
        is_win, rule = detect_explicit_winner(item)
        item["is_selector_winner"] = "true" if is_win else "false"
        item["winner_rule"] = rule
        score, score_field = selector_score(item)
        item["selector_score_v2_probe"] = "" if score is None else f"{score:.8g}"
        item["selector_score_field_v2_probe"] = score_field

    # if no explicit winners, infer one top-scored winner per case from selector score
    explicit_count = sum(1 for x in inv.values() if x["is_selector_winner"] == "true")
    if explicit_count == 0:
        by_case: Dict[str, List[Dict[str, Any]]] = {}
        for item in inv.values():
            by_case.setdefault(item["case_id"], []).append(item)
        for case_id, rows in by_case.items():
            scored = []
            for item in rows:
                fv = to_float(item.get("selector_score_v2_probe"))
                if fv is not None and item.get("has_selector") == "true":
                    scored.append((fv, item))
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                scored[0][1]["is_selector_winner"] = "true"
                scored[0][1]["winner_rule"] = "case_top_selector_score_fallback"

    issues = []
    for item in inv.values():
        missing = []
        if item["has_metrics"] != "true":
            missing.append("metrics")
        if item["has_selector"] != "true":
            missing.append("selector")
        if item["has_pairwise"] != "true":
            missing.append("pairwise")
        if not item.get("source"):
            missing.append("source")
        item["missing_evidence"] = ",".join(missing)

        if missing:
            issues.append({
                "candidate_key": item["candidate_key"],
                "case_id": item["case_id"],
                "variant": item["variant"],
                "missing_evidence": item["missing_evidence"],
            })

    rows = sorted(inv.values(), key=lambda x: (x["case_id"], x["variant"], x["candidate_key"]))

    fields = []
    preferred = [
        "candidate_key", "case_id", "variant", "source", "audio_path",
        "has_metrics", "has_selector", "has_pairwise", "pairwise_case_count",
        "is_selector_winner", "winner_rule",
        "selector_score_v2_probe", "selector_score_field_v2_probe",
        "is_repair_seed", "repair_seed_match_rule", "missing_evidence",
    ]
    seen = set()
    for f in preferred:
        fields.append(f)
        seen.add(f)
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    case_count = len({r["case_id"] for r in rows})
    variant_count = len({r["variant"] for r in rows})
    winner_count = sum(r["is_selector_winner"] == "true" for r in rows)
    repair_seed_count = sum(r["is_repair_seed"] == "true" for r in rows)

    summary = {
        "task": "w18_selector_audit_inventory_v2",
        "fix": "canonicalize metrics/selector/repair join by case_id|variant",
        "candidate_count": len(rows),
        "case_count": case_count,
        "variant_count": variant_count,
        "winner_count": winner_count,
        "repair_seed_count": repair_seed_count,
        "issue_count": len(issues),
        "dod": {
            "candidate_count_eq_30": len(rows) == 30,
            "case_count_eq_6": case_count == 6,
            "winner_count_ge_6": winner_count >= 6,
            "repair_seed_count_ge_8": repair_seed_count >= 8,
            "all_rows_have_metrics": all(r["has_metrics"] == "true" for r in rows),
            "all_rows_have_selector": all(r["has_selector"] == "true" for r in rows),
        },
        "outputs": {
            "inventory_csv": args.out_csv,
            "summary_json": args.out_summary,
            "issues_json": args.out_issues,
        },
    }

    Path(args.out_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.out_issues).write_text(json.dumps(issues, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
