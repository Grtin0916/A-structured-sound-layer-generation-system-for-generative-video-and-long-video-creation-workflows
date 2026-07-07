#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from soundlayer.ranking.dss_aware_selector import select_case


def load_expected_event_count(case_id: str) -> int:
    p = Path("cases") / case_id / "expected_events.csv"
    if not p.exists():
        return 0
    try:
        return len(list(csv.DictReader(p.open(encoding="utf-8"))))
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default="reports/w18_audio_metrics_eval_20260707.csv")
    ap.add_argument("--out-json", default="reports/w18_dss_aware_selector_eval_20260707.json")
    ap.add_argument("--out-csv", default="reports/w18_dss_aware_selector_eval_20260707.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.metrics_csv).open(encoding="utf-8")))
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)

    case_results = []
    flat_rows = []

    for case_id, case_rows in sorted(by_case.items()):
        expected_count = load_expected_event_count(case_id)
        result = select_case(case_id, case_rows, expected_count)
        case_results.append(result)

        for c in result["all_candidates"]:
            flat_rows.append({
                "case_id": case_id,
                "variant": c["variant"],
                "selector_score": c["selector_score"],
                "acoustic_score": c["acoustic_score"],
                "decision": c["decision"],
                "reasons": "|".join(c["reasons"]),
                "audio_path": c["audio_path"],
                "winner_variant": result["winner"]["variant"],
                "runner_up_variant": result["runner_up"]["variant"] if result["runner_up"] else "",
                "best_dss_variant": result["best_dss_variant"],
                "best_baseline_variant": result["best_baseline_variant"],
                "dss_delta_vs_best_baseline": result["dss_delta_vs_best_baseline"],
                "case_classification": result["case_classification"],
            })

    winner_counts = defaultdict(int)
    decision_counts = defaultdict(int)
    classification_counts = defaultdict(int)
    repair_count = 0
    listen_review_count = 0

    for c in case_results:
        winner_counts[c["winner"]["variant"]] += 1
        classification_counts[c["case_classification"]] += 1
        repair_count += len(c["repair_queue"])
        listen_review_count += len(c["listen_review"])
        for item in c["all_candidates"]:
            decision_counts[item["decision"]] += 1

    dss_winner_count = sum(v for k, v in winner_counts.items() if k.startswith("dss_"))

    invalid_all_rejected = decision_counts.get("rejected", 0) == len(flat_rows)

    summary = {
        "date": "2026-07-07",
        "scope": "w18_dss_aware_selector_eval",
        "status": "success" if len(case_results) == 6 and len(flat_rows) == 30 and not invalid_all_rejected else "review_required",
        "case_count": len(case_results),
        "candidate_count": len(flat_rows),
        "winner_counts": dict(winner_counts),
        "dss_winner_count": dss_winner_count,
        "classification_counts": dict(classification_counts),
        "decision_counts": dict(decision_counts),
        "repair_queue_count": repair_count,
        "listen_review_count": listen_review_count,
        "invalid_all_rejected": invalid_all_rejected,
        "case_results": case_results,
        "claim_boundary": [
            "Selector v0 uses acoustic sanity + small DSS prior.",
            "It is not a human preference model.",
            "It produces winner/runner-up/listen-review/repair queue for W18 candidates.",
            "Onset proxy is a weak signal and should be upgraded before semantic claims.",
        ],
        "outputs": {
            "json": args.out_json,
            "csv": args.out_csv,
        },
    }

    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps({
        "status": summary["status"],
        "case_count": summary["case_count"],
        "candidate_count": summary["candidate_count"],
        "winner_counts": summary["winner_counts"],
        "dss_winner_count": summary["dss_winner_count"],
        "classification_counts": summary["classification_counts"],
        "decision_counts": summary["decision_counts"],
        "repair_queue_count": summary["repair_queue_count"],
        "listen_review_count": summary["listen_review_count"],
        "invalid_all_rejected": summary["invalid_all_rejected"],
        "outputs": summary["outputs"],
    }, ensure_ascii=False, indent=2))

    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
