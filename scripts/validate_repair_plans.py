#!/usr/bin/env python3
"""Static safety gate for compiled repair plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def safe_parameters(action: str, params: dict) -> bool:
    if action == "attenuate_limit":
        return 0.0 < float(params.get("gain", 0.0)) <= 1.0 and 0.0 < float(params.get("peak_ceiling", 0.0)) <= 1.0
    if action == "trim":
        return 0.0 <= float(params.get("silence_threshold", -1.0)) <= 0.1 and 0 <= int(params.get("padding_ms", -1)) <= 500
    if action == "mixed_region_attenuation":
        return 0.0 < float(params.get("gain", 0.0)) <= 1.0 and 0 <= int(params.get("fade_ms", -1)) <= 500
    if action in {"shift_left", "delay_or_pad"}:
        value = params.get("max_shift_ms", params.get("max_delay_ms", -1))
        return 0 <= int(value) <= 500
    return action == "candidate_replace"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-bank", type=Path)
    parser.add_argument("--plans", "--repair-plans", dest="plans", type=Path, required=True)
    parser.add_argument("--out-validation", "--out-json", dest="out_validation", type=Path, required=True)
    parser.add_argument(
        "--out-manifest", "--execution-manifest", dest="out_manifest", type=Path, required=True
    )
    parser.add_argument("--min-ready", type=int, default=10)
    args = parser.parse_args()

    plans = [
        json.loads(line) for line in args.plans.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    missing_source = 0
    invalid_interval = 0
    unsafe_parameter = 0
    inapplicable_blocked = 0
    ready: list[dict] = []
    details: list[dict] = []
    for plan in plans:
        errors: list[str] = []
        if not Path(plan["source_audio"]).is_file():
            missing_source += 1
            errors.append("missing_source")
        start, end, duration = plan["window"]["start_sec"], plan["window"]["end_sec"], plan["duration_sec"]
        if not (0.0 <= start < end <= duration + 1.0e-3):
            invalid_interval += 1
            errors.append("invalid_interval")
        if not safe_parameters(plan["action"], plan["parameters"]):
            unsafe_parameter += 1
            errors.append("unsafe_parameters")
        if plan["action"] == "candidate_replace" and not plan["execution_ready"]:
            inapplicable_blocked += 1
        if plan["execution_ready"] and not errors:
            ready.append(plan)
        details.append({"failure_id": plan["failure_id"], "errors": errors})

    gate = (
        missing_source == 0
        and invalid_interval == 0
        and unsafe_parameter == 0
        and len(ready) >= args.min_ready
    )
    validation = {
        "planCount": len(plans),
        "missingSourceCount": missing_source,
        "invalidIntervalCount": invalid_interval,
        "unsafeParameterCount": unsafe_parameter,
        "inapplicableActionBlockedCount": inapplicable_blocked,
        "executionReadyCount": len(ready),
        "minimumExecutionReady": args.min_ready,
        "details": details,
        "gateStatus": "PASS" if gate else "FAIL",
    }
    args.out_validation.parent.mkdir(parents=True, exist_ok=True)
    args.out_validation.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_manifest.write_text(
        "".join(json.dumps(plan, sort_keys=True) + "\n" for plan in ready), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in validation.items() if key != "details"}, sort_keys=True))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
