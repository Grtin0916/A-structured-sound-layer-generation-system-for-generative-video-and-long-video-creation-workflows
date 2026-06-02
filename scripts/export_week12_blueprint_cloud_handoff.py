#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    return max(0, len(rows) - 1)


def git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def copy_named(src: Path, dst_dir: Path, role: str) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(f"missing required artifact for {role}: {src}")
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return {
        "role": role,
        "sourcePath": str(src),
        "bundlePath": str(dst.relative_to(dst_dir.parent)),
        "filename": dst.name,
        "sizeBytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def main() -> int:
    root = Path.cwd()
    if not (root / "schemas").exists() or not (root / "artifacts").exists():
        raise SystemExit("Run this script from Mainbase repo root.")

    summary_path = root / "artifacts/manifests/week12_blueprint_v1_final_summary.json"
    manifest_path = root / "artifacts/manifests/week12_blueprint_v1_manifest.json"
    timeline_jsonl_path = root / "artifacts/manifests/week12_event_timeline.jsonl"
    timeline_csv_path = root / "artifacts/manifests/week12_event_timeline.csv"
    semantic_report_path = root / "artifacts/manifests/week12_blueprint_v1_semantic_report.json"
    validation_report_path = root / "artifacts/manifests/week12_blueprint_v1_validation_report.json"
    contact_png_path = root / "artifacts/visuals/week12_event_timeline_contact_sheet.png"
    contact_svg_path = root / "artifacts/visuals/week12_event_timeline_contact_sheet.svg"
    schema_path = root / "schemas/soundlayer_blueprint_v1.schema.json"

    required = {
        "summary": summary_path,
        "manifest": manifest_path,
        "timeline_jsonl": timeline_jsonl_path,
        "timeline_csv": timeline_csv_path,
        "semantic_report": semantic_report_path,
        "validation_report": validation_report_path,
        "contact_sheet_png": contact_png_path,
        "schema": schema_path,
    }

    optional = {
        "contact_sheet_svg": contact_svg_path,
    }

    summary = read_json(summary_path)
    timeline_jsonl_rows = count_jsonl(timeline_jsonl_path)
    timeline_csv_rows = count_csv_data_rows(timeline_csv_path)

    gate_pass = (
        summary.get("status") == "PASS"
        and summary.get("blueprint_count") == 5
        and summary.get("event_count") == 10
        and summary.get("error_count") == 0
        and timeline_jsonl_rows == 10
        and timeline_csv_rows == 10
        and all(path.exists() for path in required.values())
    )

    if not gate_pass:
        raise SystemExit(
            "MAINBASE_BLUEPRINT_GATE_FAIL: "
            f"summary_status={summary.get('status')} "
            f"blueprint_count={summary.get('blueprint_count')} "
            f"event_count={summary.get('event_count')} "
            f"error_count={summary.get('error_count')} "
            f"timeline_jsonl_rows={timeline_jsonl_rows} "
            f"timeline_csv_rows={timeline_csv_rows}"
        )

    export_root = root / "artifacts/exports"
    export_root.mkdir(parents=True, exist_ok=True)

    bundle_name = "week12_soundlayer_blueprint_v1_cloud_handoff"
    bundle_dir = export_root / bundle_name

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifests").mkdir()
    (bundle_dir / "visuals").mkdir()
    (bundle_dir / "schema").mkdir()
    (bundle_dir / "metadata").mkdir()

    copied: dict[str, Any] = {}
    copied["summary"] = copy_named(summary_path, bundle_dir / "metadata", "summary")
    copied["manifest"] = copy_named(manifest_path, bundle_dir / "manifests", "manifest")
    copied["timeline_jsonl"] = copy_named(timeline_jsonl_path, bundle_dir / "manifests", "timeline_jsonl")
    copied["timeline_csv"] = copy_named(timeline_csv_path, bundle_dir / "manifests", "timeline_csv")
    copied["semantic_report"] = copy_named(semantic_report_path, bundle_dir / "metadata", "semantic_report")
    copied["validation_report"] = copy_named(validation_report_path, bundle_dir / "metadata", "validation_report")
    copied["contact_sheet_png"] = copy_named(contact_png_path, bundle_dir / "visuals", "contact_sheet_png")
    copied["schema"] = copy_named(schema_path, bundle_dir / "schema", "schema")

    for role, path in optional.items():
        if path.exists():
            copied[role] = copy_named(path, bundle_dir / "visuals", role)

    env_sample = "\n".join(
        [
            "# Week12 SoundLayer Blueprint V1 handoff environment",
            "# Set SOUNDLAYER_HANDOFF_ROOT to the extracted bundle root.",
            "SOUNDLAYER_HANDOFF_ROOT=/mnt/soundlayer-artifacts/week12_soundlayer_blueprint_v1_cloud_handoff",
            "SOUNDLAYER_BLUEPRINT_MANIFEST=${SOUNDLAYER_HANDOFF_ROOT}/manifests/week12_blueprint_v1_manifest.json",
            "SOUNDLAYER_EVENT_TIMELINE_JSONL=${SOUNDLAYER_HANDOFF_ROOT}/manifests/week12_event_timeline.jsonl",
            "SOUNDLAYER_EVENT_TIMELINE_CSV=${SOUNDLAYER_HANDOFF_ROOT}/manifests/week12_event_timeline.csv",
            "SOUNDLAYER_CONTACT_SHEET=${SOUNDLAYER_HANDOFF_ROOT}/visuals/week12_event_timeline_contact_sheet.png",
            "SOUNDLAYER_BLUEPRINT_SCHEMA=${SOUNDLAYER_HANDOFF_ROOT}/schema/soundlayer_blueprint_v1.schema.json",
            "",
        ]
    )
    env_path = bundle_dir / "handoff.env.example"
    env_path.write_text(env_sample, encoding="utf-8")

    handoff_manifest = {
        "schemaVersion": "week12.soundlayer-blueprint-cloud-handoff.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "sourceRepo": "mainbase",
        "sourceHead": git_head(root),
        "bundleName": bundle_name,
        "bundleDir": str(bundle_dir),
        "mainbaseGate": {
            "status": summary.get("status"),
            "blueprintCount": summary.get("blueprint_count"),
            "eventCount": summary.get("event_count"),
            "errorCount": summary.get("error_count"),
            "timelineJsonlRows": timeline_jsonl_rows,
            "timelineCsvRows": timeline_csv_rows,
        },
        "files": copied,
        "envSample": str(env_path.relative_to(bundle_dir)),
        "cloudMountSuggestion": {
            "extractRoot": "/mnt/soundlayer-artifacts",
            "handoffRoot": "/mnt/soundlayer-artifacts/week12_soundlayer_blueprint_v1_cloud_handoff",
            "volumeUse": "local/kind demo or worker handoff only; does not claim production object storage.",
        },
        "doesNotClaim": [
            "audio generation",
            "Java runtime HTTP success",
            "Cloud live Grafana import",
            "production Kubernetes volume",
            "remote artifact registry",
        ],
    }

    manifest_out = bundle_dir / "handoff_manifest.json"
    manifest_out.write_text(json.dumps(handoff_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    checksums = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file():
            checksums.append(f"{sha256_file(path)}  {path.relative_to(bundle_dir)}")
    checksum_path = bundle_dir / "SHA256SUMS.txt"
    checksum_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")

    tar_path = export_root / f"{bundle_name}.tar.gz"
    if tar_path.exists():
        tar_path.unlink()

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_name)

    tar_sha256_path = export_root / f"{bundle_name}.tar.gz.sha256"
    tar_sha256_path.write_text(f"{sha256_file(tar_path)}  {tar_path.name}\n", encoding="utf-8")

    log_dir = root / "artifacts/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"week12_blueprint_cloud_handoff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text(
        "\n".join(
            [
                "status=PASS",
                f"bundle_dir={bundle_dir}",
                f"tar_path={tar_path}",
                f"tar_sha256={sha256_file(tar_path)}",
                f"blueprint_count={summary.get('blueprint_count')}",
                f"event_count={summary.get('event_count')}",
                f"timeline_jsonl_rows={timeline_jsonl_rows}",
                f"timeline_csv_rows={timeline_csv_rows}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("status=PASS")
    print(f"bundle_dir={bundle_dir}")
    print(f"handoff_manifest={manifest_out}")
    print(f"checksum_path={checksum_path}")
    print(f"tar_path={tar_path}")
    print(f"tar_sha256_path={tar_sha256_path}")
    print(f"log_path={log_path}")
    print(f"blueprint_count={summary.get('blueprint_count')}")
    print(f"event_count={summary.get('event_count')}")
    print(f"timeline_jsonl_rows={timeline_jsonl_rows}")
    print(f"timeline_csv_rows={timeline_csv_rows}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())