#!/usr/bin/env python3
"""Build a browsable W18 experiment release from real cross-repo artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_into(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mainbase-root", type=Path, required=True)
    parser.add_argument("--java-root", type=Path, required=True)
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--claim-boundary", type=Path, required=True)
    parser.add_argument("--walkthrough", type=Path, required=True)
    parser.add_argument("--weekly-summary", type=Path, required=True)
    args = parser.parse_args()

    root = args.mainbase_root.resolve()
    java = args.java_root.resolve()
    cloud = args.cloud_root.resolve()
    out = args.out_dir.resolve()
    audio_dir = out / "audio"
    plot_dir = out / "plots"
    report_dir = out / "reports"
    for directory in (audio_dir, plot_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prompt = read_json(root / "reports/w18_prompt_compiler_summary_20260706.json")
    selector_summary = read_json(root / "reports/w18_selector_v2_summary_20260708.json")
    failure_summary = read_json(root / "reports/w18_failure_bank_summary_20260708.json")
    repair_summary = read_json(root / "reports/w18_micro_repair_probe_summary_20260708.json")
    lifecycle = read_json(java / "artifacts/manifests/w18_task_lifecycle_report_20260712.json")
    aggregation = read_json(cloud / "loadtest/reports/w18_experiment_aggregation_20260710.json")
    selector_rows = read_csv(root / "reports/w18_selector_v2_scores_20260708.csv")
    failure_rows = read_csv(root / "reports/w18_failure_bank_20260708.csv")
    repair_rows = read_csv(root / "reports/w18_micro_repair_probe_20260708.csv")
    winners = [row for row in selector_rows if row.get("selector_v2_decision") == "winner"]

    copied_audio: list[dict[str, str]] = []
    copied_plots: list[str] = []
    html_rows: list[str] = []

    for row in winners:
        source = root / row["audio_path"]
        name = f'winner__{row["case_id"]}__{row["variant"]}{source.suffix}'
        destination = audio_dir / name
        copy_into(source, destination)
        relative = destination.relative_to(out).as_posix()
        copied_audio.append({"role": "winner", "caseId": row["case_id"], "path": relative})
        html_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{:.4f}</td>"
            '<td><audio controls preload="none" src="{}"></audio></td></tr>'.format(
                html.escape(row["case_id"]),
                html.escape(row["variant"]),
                float(row["selector_v2_score"]),
                html.escape(relative),
            )
        )

    for row in repair_rows:
        for role, field in (("repair_before", "before_audio_path"), ("repair_after", "after_audio_path")):
            source = root / row[field]
            name = f'{role}__{row["probe_id"]}__{row["case_id"]}{source.suffix}'
            destination = audio_dir / name
            copy_into(source, destination)
            copied_audio.append(
                {"role": role, "caseId": row["case_id"], "probeId": row["probe_id"],
                 "path": destination.relative_to(out).as_posix()}
            )
        source_plot = root / row["plot_path"]
        destination_plot = plot_dir / source_plot.name
        copy_into(source_plot, destination_plot)
        copied_plots.append(destination_plot.relative_to(out).as_posix())

    source_reports = {
        "prompt_summary.json": root / "reports/w18_prompt_compiler_summary_20260706.json",
        "generation_summary.json": root / "reports/w18_full_30job_generation_summary_20260706.json",
        "selector_summary.json": root / "reports/w18_selector_v2_summary_20260708.json",
        "failure_summary.json": root / "reports/w18_failure_bank_summary_20260708.json",
        "repair_summary.json": root / "reports/w18_micro_repair_probe_summary_20260708.json",
        "java_lifecycle.json": java / "artifacts/manifests/w18_task_lifecycle_report_20260712.json",
        "cloud_aggregation.json": cloud / "loadtest/reports/w18_experiment_aggregation_20260710.json",
        "cloud_dashboard.json": cloud / "observability/grafana/dashboards/w18_experiment_dashboard.json",
        "cloud_provider.yml": cloud / "observability/grafana/provisioning/dashboards/w18-experiment.yml",
        "cloud_alert_summary.json": cloud / "artifacts/demo/week18_live_alert_gate/summary.json",
    }
    copied_reports = []
    for name, source in source_reports.items():
        destination = report_dir / name
        copy_into(source, destination)
        copied_reports.append(destination.relative_to(out).as_posix())

    claim_boundary = {
        "trueMmaudioBatchVerified": False,
        "subjectiveRepairQualityVerified": False,
        "productionPrometheusVerified": False,
        "liveGrafanaImportVerified": False,
        "alertmanagerConfigured": False,
        "productionAlertingVerified": False,
        "dockerDesktopEngineHealthy": False,
        "notes": [
            "The 30-candidate ablation is the recorded local experiment set.",
            "Micro repair improvements are objective proxies, not listening-test claims.",
            "Dashboard and provisioning files are versioned artifacts, not a live Grafana claim.",
        ],
    }
    args.claim_boundary.parent.mkdir(parents=True, exist_ok=True)
    args.claim_boundary.write_text(
        json.dumps(claim_boundary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    copy_into(args.claim_boundary.resolve(), report_dir / "claim_boundary.json")

    counts = {
        "caseCount": int(prompt["case_count"]),
        "candidateCount": int(selector_summary["scored"]),
        "winnerCount": len(winners),
        "failureCount": len(failure_rows),
        "repairBeforeAfterCount": len(repair_rows),
        "javaTaskCount": int(lifecycle["taskCount"]),
        "javaGaugeCount": int(aggregation["counts"]["javaGaugeCount"]),
        "alertRuleCount": int(aggregation["counts"]["alertRuleCount"]),
        "dashboardPanelCount": int(aggregation["counts"]["dashboardPanelCount"]),
    }
    manifest = {
        "schemaVersion": "w18-experiment-pack-v1",
        "counts": counts,
        "promptVariants": prompt["variants"],
        "failureCategories": failure_summary["category_counts"],
        "repairProxyImproveCount": repair_summary["proxy_improve_count"],
        "audio": copied_audio,
        "plots": copied_plots,
        "reports": copied_reports + ["reports/claim_boundary.json"],
        "entrypoint": "index.html",
        "claimBoundary": claim_boundary,
    }

    index = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>W18 SoundLayer Experiment</title>
<style>body{{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.5rem}}
code{{background:#eee;padding:.1rem .3rem}}</style></head><body>
<h1>W18 SoundLayer Experiment Release</h1>
<p>Six cases, 30 candidates, six selector winners, 12 failures and six micro-repair pairs.
Java binds 12 results; Cloud aggregates six gauges and four local alert rules.</p>
<h2>Selector winners</h2><table><thead><tr><th>Case</th><th>Variant</th>
<th>Score</th><th>Audio</th></tr></thead><tbody>{}</tbody></table>
<h2>Interpretation boundary</h2>
<p>Micro-repair gains are proxy metrics. The dashboard is provisionable but was not imported
into live Grafana. Local Prometheus verification is not production monitoring.</p>
<p>See <code>reports/claim_boundary.json</code> and the copied source reports.</p>
</body></html>
""".format("\n".join(html_rows))
    (out / "index.html").write_text(index, encoding="utf-8")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copy_into(args.manifest.resolve(), report_dir / "manifest.json")

    args.walkthrough.parent.mkdir(parents=True, exist_ok=True)
    args.walkthrough.write_text(
        "# W18 experiment walkthrough\n\n"
        "Open `artifacts/demo/w18_experiment_pack_20260710/index.html` and listen to the "
        "six selector winners. Compare the six before/after pairs with the failure and "
        "repair summaries, then inspect the Java lifecycle and Cloud dashboard reports. "
        "Treat repair gains as signal proxies, not subjective quality scores.\n",
        encoding="utf-8",
    )
    args.weekly_summary.parent.mkdir(parents=True, exist_ok=True)
    args.weekly_summary.write_text(
        "# W18 experiment summary — 2026-07-10\n\n"
        "W18 closed with 6 cases, 30 generated candidates, 6 selector winners, "
        "12 categorized failures and 6 micro-repair pairs. Java orchestrated 12 "
        "artifact-backed tasks and exposed 6 lifecycle gauges. Cloud loaded 4 local "
        "alert rules and produced a 9-panel provisionable dashboard. Docker Desktop "
        "remains unhealthy; no production Prometheus or live Grafana claim is made.\n",
        encoding="utf-8",
    )

    args.out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out.parent).as_posix())

    print(json.dumps({"status": "PASS", **counts, "zip": str(args.out_zip)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
