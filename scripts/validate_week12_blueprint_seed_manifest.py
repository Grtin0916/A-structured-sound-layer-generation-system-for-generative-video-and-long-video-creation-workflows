#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

SCHEMA_PATH = ROOT / "schemas/soundlayer_blueprint_seed_v0.schema.json"
MANIFEST_PATH = ROOT / "artifacts/manifests/week12_blueprint_seed_manifest.json"
OUT_REPORT = ROOT / "artifacts/manifests/week12_blueprint_seed_validation_report.json"
LOG_DIR = ROOT / "artifacts/logs"


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def in_range(x: Any, lo: float, hi: float, strict_lo: bool = False) -> bool:
    if not is_number(x):
        return False
    if strict_lo:
        return lo < float(x) <= hi
    return lo <= float(x) <= hi


def nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def fail_row(seed_id: str, message: str) -> dict[str, str]:
    return {
        "seed_id": seed_id,
        "message": message,
    }


def validate_seed(seed: dict[str, Any], idx: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    seed_id = str(seed.get("blueprint_id") or f"seed_{idx}")

    if not nonempty_str(seed.get("blueprint_id")) or not str(seed.get("blueprint_id")).startswith("bp_"):
        failures.append(fail_row(seed_id, "blueprint_id must be non-empty and start with bp_"))

    if not nonempty_str(seed.get("source_case_id")):
        failures.append(fail_row(seed_id, "source_case_id is required"))

    if seed.get("schema_version") != "soundlayer_blueprint_seed_v0":
        failures.append(fail_row(seed_id, "seed schema_version must be soundlayer_blueprint_seed_v0"))

    source = seed.get("source")
    if not isinstance(source, dict):
        failures.append(fail_row(seed_id, "source must be object"))
        source = {}

    required_source_fields = [
        "artifactUri",
        "evalSummaryUri",
        "mainbaseEvalSummaryUri",
        "cloudGateSummaryUri",
        "qualityGateStatus",
    ]
    for field in required_source_fields:
        if not nonempty_str(source.get(field)):
            failures.append(fail_row(seed_id, f"source.{field} must be non-empty"))

    for local_field in ["artifactUri", "mainbaseEvalSummaryUri", "evalSummaryUri"]:
        value = source.get(local_field)
        if nonempty_str(value):
            path = ROOT / value
            if not path.exists():
                failures.append(fail_row(seed_id, f"source.{local_field} points to missing local path: {value}"))

    scores = seed.get("scores_proxy")
    if not isinstance(scores, dict):
        failures.append(fail_row(seed_id, "scores_proxy must be object"))
        scores = {}

    for field in ["semantic", "temporal"]:
        if not in_range(scores.get(field), 0.0, 1.0):
            failures.append(fail_row(seed_id, f"scores_proxy.{field} must be number in [0, 1]"))

    if not in_range(scores.get("quality"), 0.0, 1.0, strict_lo=True):
        failures.append(fail_row(seed_id, "scores_proxy.quality must be number in (0, 1]"))

    if not nonempty_str(scores.get("quality_boundary")):
        failures.append(fail_row(seed_id, "scores_proxy.quality_boundary is required"))

    events = seed.get("events")
    if not isinstance(events, list) or not events:
        failures.append(fail_row(seed_id, "events must be non-empty list"))
        events = []

    event_layers: set[str] = set()
    for event_idx, event in enumerate(events, 1):
        if not isinstance(event, dict):
            failures.append(fail_row(seed_id, f"events[{event_idx}] must be object"))
            continue

        for field in ["event_id", "layer", "sound_intent", "control_source"]:
            if not nonempty_str(event.get(field)):
                failures.append(fail_row(seed_id, f"events[{event_idx}].{field} must be non-empty"))

        layer = event.get("layer")
        if layer not in {"ambience", "foley", "music", "dialogue", "effect"}:
            failures.append(fail_row(seed_id, f"events[{event_idx}].layer has unsupported value: {layer}"))
        else:
            event_layers.add(str(layer))

        start = event.get("time_start_sec")
        end = event.get("time_end_sec")
        if not is_number(start) or float(start) < 0:
            failures.append(fail_row(seed_id, f"events[{event_idx}].time_start_sec must be non-negative number"))
        if not is_number(end) or float(end) < 0:
            failures.append(fail_row(seed_id, f"events[{event_idx}].time_end_sec must be non-negative number"))
        if is_number(start) and is_number(end) and float(end) < float(start):
            failures.append(fail_row(seed_id, f"events[{event_idx}].time_end_sec must be >= time_start_sec"))

        if "confidence_proxy" in event and not in_range(event.get("confidence_proxy"), 0.0, 1.0):
            failures.append(fail_row(seed_id, f"events[{event_idx}].confidence_proxy must be in [0, 1]"))

    layers_required = seed.get("layers_required")
    if not isinstance(layers_required, list) or not all(nonempty_str(x) for x in layers_required):
        failures.append(fail_row(seed_id, "layers_required must be non-empty string list"))
        layers_required_set = set()
    else:
        layers_required_set = set(str(x) for x in layers_required)

    if event_layers and layers_required_set and not event_layers.issubset(layers_required_set):
        failures.append(fail_row(seed_id, "layers_required must cover all event layers"))

    contract = seed.get("w12_consumption_contract")
    if not isinstance(contract, dict):
        failures.append(fail_row(seed_id, "w12_consumption_contract must be object"))
        contract = {}

    for field in [
        "artifact_pointer_ready",
        "eval_summary_ready",
        "cloud_gate_summary_ready",
        "quality_proxy_ready",
    ]:
        if contract.get(field) is not True:
            failures.append(fail_row(seed_id, f"w12_consumption_contract.{field} must be true"))

    consumers = contract.get("expected_consumer")
    if not isinstance(consumers, list) or not consumers:
        failures.append(fail_row(seed_id, "w12_consumption_contract.expected_consumer must be non-empty list"))

    non_goals = seed.get("non_goals")
    if not isinstance(non_goals, list) or not non_goals:
        failures.append(fail_row(seed_id, "non_goals must be non-empty list"))

    if source.get("qualityGateStatus") == "CREATED":
        warnings.append(fail_row(seed_id, "qualityGateStatus is CREATED; interpret as task/evidence state, not perceptual quality"))

    return failures, warnings


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    hard_failures: list[str] = []
    warnings: list[str] = []

    if not SCHEMA_PATH.exists():
        hard_failures.append(f"missing {SCHEMA_PATH.relative_to(ROOT)}")
    if not MANIFEST_PATH.exists():
        hard_failures.append(f"missing {MANIFEST_PATH.relative_to(ROOT)}")

    if hard_failures:
        for item in hard_failures:
            print(f"HARD_FAILURE: {item}")
        raise SystemExit(1)

    schema = load_json(SCHEMA_PATH)
    manifest = load_json(MANIFEST_PATH)

    if schema.get("title") != "SoundLayer Blueprint Seed Manifest V0":
        hard_failures.append("schema title mismatch")

    if manifest.get("schema_version") != "week12_blueprint_seed_manifest_v0":
        hard_failures.append("manifest schema_version mismatch")

    summary = manifest.get("summary")
    seeds = manifest.get("blueprint_seeds")

    if not isinstance(summary, dict):
        hard_failures.append("manifest.summary must be object")
        summary = {}

    if not isinstance(seeds, list) or not seeds:
        hard_failures.append("manifest.blueprint_seeds must be non-empty list")
        seeds = []

    seed_failures: list[dict[str, str]] = []
    seed_warnings: list[dict[str, str]] = []

    for idx, seed in enumerate(seeds, 1):
        if not isinstance(seed, dict):
            seed_failures.append(fail_row(f"seed_{idx}", "seed must be object"))
            continue
        failures, local_warnings = validate_seed(seed, idx)
        seed_failures.extend(failures)
        seed_warnings.extend(local_warnings)

    seed_count = len(seeds)
    if summary.get("seed_count") != seed_count:
        seed_failures.append(fail_row("manifest", f"summary.seed_count={summary.get('seed_count')} does not equal actual seed count={seed_count}"))

    if seed_count < 5:
        warnings.append("seed_count below 5; W12 seed coverage is weaker than current Week11 case set")

    layers_observed = summary.get("layers_observed")
    if not isinstance(layers_observed, list) or not {"ambience", "foley"}.issubset(set(layers_observed)):
        seed_failures.append(fail_row("manifest", "summary.layers_observed must include ambience and foley"))

    for flag in [
        "has_artifact_links",
        "has_mainbase_eval_summary_links",
        "has_cloud_gate_summary_links",
    ]:
        if summary.get(flag) is not True:
            seed_failures.append(fail_row("manifest", f"summary.{flag} must be true"))

    qmin = summary.get("quality_proxy_min")
    qmax = summary.get("quality_proxy_max")
    if not in_range(qmin, 0.0, 1.0, strict_lo=True):
        seed_failures.append(fail_row("manifest", "summary.quality_proxy_min must be in (0, 1]"))
    if not in_range(qmax, 0.0, 1.0, strict_lo=True):
        seed_failures.append(fail_row("manifest", "summary.quality_proxy_max must be in (0, 1]"))
    if is_number(qmin) and is_number(qmax) and float(qmin) > float(qmax):
        seed_failures.append(fail_row("manifest", "quality_proxy_min must be <= quality_proxy_max"))

    status = "PASS"
    if hard_failures or seed_failures:
        status = "FAIL"
    elif warnings or seed_warnings:
        status = "PASS_WITH_WARNINGS"

    report = {
        "schema_version": "week12_blueprint_seed_validation_report_v0",
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "repo": {
            "git_status": git("status", "-sb"),
            "head": git("rev-parse", "--short", "HEAD"),
            "branch": git("branch", "--show-current"),
        },
        "inputs": {
            "schema": str(SCHEMA_PATH.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        },
        "summary": {
            "status": status,
            "seed_count": seed_count,
            "failure_count": len(hard_failures) + len(seed_failures),
            "warning_count": len(warnings) + len(seed_warnings),
            "quality_proxy_min": qmin,
            "quality_proxy_max": qmax,
            "layers_observed": layers_observed,
        },
        "hard_failures": hard_failures,
        "seed_failures": seed_failures,
        "warnings": warnings,
        "seed_warnings": seed_warnings,
        "boundary": [
            "This validates W12 blueprint seed structure and local Mainbase artifact pointers.",
            "It does not validate generated audio perceptual quality.",
            "It does not claim production artifact registry or production SLO.",
            "cloudGateSummaryUri is treated as local Cloud evidence pointer, not as a required Mainbase local file."
        ],
    }

    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"week12_blueprint_seed_validation_{stamp}.log"
    log_path.write_text(
        "\n".join(
            [
                "===== Week12 Blueprint Seed Validation =====",
                f"status={status}",
                f"schema={SCHEMA_PATH.relative_to(ROOT)}",
                f"manifest={MANIFEST_PATH.relative_to(ROOT)}",
                f"report={OUT_REPORT.relative_to(ROOT)}",
                f"seed_count={seed_count}",
                f"failure_count={len(hard_failures) + len(seed_failures)}",
                f"warning_count={len(warnings) + len(seed_warnings)}",
                f"quality_proxy_min={qmin}",
                f"quality_proxy_max={qmax}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[{status}] Week12 blueprint seed validation")
    print(f"schema={SCHEMA_PATH.relative_to(ROOT)}")
    print(f"manifest={MANIFEST_PATH.relative_to(ROOT)}")
    print(f"report={OUT_REPORT.relative_to(ROOT)}")
    print(f"log={log_path.relative_to(ROOT)}")
    print(f"seed_count={seed_count}")
    print(f"failure_count={len(hard_failures) + len(seed_failures)}")
    print(f"warning_count={len(warnings) + len(seed_warnings)}")
    print(f"quality_proxy_min={qmin}")
    print(f"quality_proxy_max={qmax}")

    if status == "FAIL":
        for item in hard_failures:
            print(f"HARD_FAILURE: {item}")
        for item in seed_failures[:20]:
            print(f"SEED_FAILURE: {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()