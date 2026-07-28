"""Inventory frozen W20 A/B/C/D audio and build content-safe pair graphs."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import wave
from pathlib import Path

PREFERRED_EDGES = (("A", "B"), ("B", "C"), ("C", "D"))
FALLBACK_EDGES = (("B", "D"), ("A", "C"), ("A", "D"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def audio_probe(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "duration_sec": frames / rate if rate else 0.0,
            "sample_rate": rate,
            "channels": handle.getnchannels(),
        }


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def git_last_change(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return subprocess.check_output(
        ["git", "-C", str(root), "log", "-1", "--format=%h", "--", str(relative)],
        text=True,
    ).strip()


def load_inventory(
    root: Path,
    candidate_matrix: Path,
    ablation: Path,
    handoff: Path,
    repair_handoff: Path | None = None,
):
    matrix_report = json.loads(candidate_matrix.with_suffix(".json").read_text())
    ablation_report = json.loads(ablation.with_suffix(".json").read_text())
    handoff_report = json.loads(handoff.read_text())
    repair_report = (
        json.loads(repair_handoff.read_text())
        if repair_handoff is not None and repair_handoff.is_file()
        else {"records": []}
    )
    repair_by_id = {row["repair_id"]: row for row in repair_report["records"]}
    matrix_by_case = {}
    for row in matrix_report["records"]:
        matrix_by_case.setdefault(row["matrix_case_id"], row)

    candidates = []
    case_summaries = []
    missing = []
    digest_mismatches = []
    current_head = git_head(root)
    feature_origin = git_last_change(root, ablation.with_suffix(".json"))
    repair_origin = (
        git_last_change(root, repair_handoff)
        if repair_handoff is not None and repair_handoff.is_file()
        else ""
    )
    diagnostic_count = 0
    for case in ablation_report["cases"]:
        matrix = matrix_by_case[case["case_id"]]
        case_candidates = []
        for strategy in "ABCD":
            rel = case.get(f"strategy_{strategy.lower()}", "")
            if not rel:
                continue
            path = root / rel
            if not path.is_file():
                missing.append(rel)
                continue
            digest = sha256_file(path)
            metrics = case["metrics"].get(strategy) or {}
            item = {
                "case_id": case["case_id"],
                "source_case_id": case["source_case_id"],
                "strategy_id": strategy,
                "artifact_path": rel,
                "artifact_digest": digest,
                "video_path": matrix["video_path"],
                **audio_probe(path),
                "proxy_metrics": {
                    key: metrics.get(key)
                    for key in ("proxy_score", "peak_abs", "clip_ratio", "silence_ratio")
                    if key in metrics
                },
                "publish_decision": case["publish_decision"],
                "repair_decision": case["repair_decision"],
                "git_head": current_head,
                "audio_feature_origin": feature_origin,
                "artifact_origin": handoff_report["sourceCommit"],
                "candidate_role": f"ABLATION_STRATEGY_{strategy}",
                "ablation_materialized": True,
            }
            case_candidates.append(item)
            candidates.append(item)
        # The W20 ablation deliberately left C/D empty for rejected variants.
        # Reuse the already-frozen v1/v2 repair handoff as diagnostic candidates;
        # no audio is regenerated and the case publish decision remains BLOCKED.
        failure_id = "_".join(case["case_id"].split("_")[:2])
        for strategy, suffix in (("C", "v1"), ("D", "v2")):
            if any(item["strategy_id"] == strategy for item in case_candidates):
                continue
            repair = repair_by_id.get(f"{failure_id}_transplant_{suffix}")
            if not repair:
                continue
            rel = repair["after_artifact"]
            path = root / rel
            if not path.is_file():
                missing.append(rel)
                continue
            metrics = repair.get("metrics", {})
            item = {
                "case_id": case["case_id"],
                "source_case_id": case["source_case_id"],
                "strategy_id": strategy,
                "artifact_path": rel,
                "artifact_digest": sha256_file(path),
                "video_path": matrix["video_path"],
                **audio_probe(path),
                "proxy_metrics": {
                    key: float(metrics[key])
                    for key in ("target_delta", "edit_cost", "outside_window_delta_db")
                    if key in metrics
                },
                "publish_decision": case["publish_decision"],
                "repair_decision": repair["decision"],
                "git_head": current_head,
                "audio_feature_origin": repair_origin,
                "artifact_origin": repair_origin,
                "candidate_role": f"FROZEN_REPAIR_DIAGNOSTIC_{suffix.upper()}",
                "ablation_materialized": False,
            }
            case_candidates.append(item)
            candidates.append(item)
            diagnostic_count += 1
        case_summaries.append(
            {
                "case_id": case["case_id"],
                "candidate_count": len(case_candidates),
                "unique_digest_count": len(
                    {item["artifact_digest"] for item in case_candidates}
                ),
            }
        )
    summary = {
        "caseCount": len(case_summaries),
        "candidateCount": len(candidates),
        "ablationMaterializedCandidateCount": len(candidates) - diagnostic_count,
        "frozenDiagnosticCandidateCount": diagnostic_count,
        "uniqueDigestCount": len({item["artifact_digest"] for item in candidates}),
        "missingArtifactCount": len(missing),
        "digestMismatchCount": len(digest_mismatches),
        "gitHead": current_head,
        "audioFeatureOrigin": feature_origin,
        "artifactOrigin": handoff_report["sourceCommit"],
        "diagnosticArtifactOrigin": repair_origin,
        "finalSelectedCount": handoff_report["finalSelectedCount"],
    }
    return {
        "schemaVersion": "preference-candidate-inventory/v1",
        "summary": summary,
        "cases": case_summaries,
        "candidates": candidates,
        "missingArtifacts": missing,
    }


def _connected(nodes, edges):
    if not nodes:
        return False
    seen = {nodes[0]}
    while True:
        expanded = seen | {
            endpoint
            for edge in edges
            if edge["left_strategy"] in seen or edge["right_strategy"] in seen
            for endpoint in (edge["left_strategy"], edge["right_strategy"])
        }
        if expanded == seen:
            return len(seen) == len(nodes)
        seen = expanded


def build_pair_graph(inventory: dict, edges_per_case: int = 3) -> dict:
    by_case = {}
    for item in inventory["candidates"]:
        by_case.setdefault(item["case_id"], {})[item["strategy_id"]] = item
    cases = []
    all_edges = []
    for case_id, strategies in by_case.items():
        valid_edges = []
        same_digest_rejected = []
        for left, right in PREFERRED_EDGES + FALLBACK_EDGES:
            if left not in strategies or right not in strategies:
                continue
            a, b = strategies[left], strategies[right]
            if a["artifact_digest"] == b["artifact_digest"]:
                same_digest_rejected.append(f"{left}-{right}")
                continue
            edge = {
                "edge_id": f"{case_id}:{left}-{right}",
                "case_id": case_id,
                "left_strategy": left,
                "right_strategy": right,
                "left_digest": a["artifact_digest"],
                "right_digest": b["artifact_digest"],
            }
            valid_edges.append(edge)
        present = sorted(strategies)
        edges = valid_edges[:edges_per_case]
        if len(present) == 4:
            connected_choices = [
                choice
                for choice in itertools.combinations(valid_edges, edges_per_case)
                if _connected(present, choice)
            ]
            if connected_choices:
                rank = {edge["edge_id"]: index for index, edge in enumerate(valid_edges)}
                def choice_score(choice):
                    content_pairs = {
                        tuple(sorted((edge["left_digest"], edge["right_digest"])))
                        for edge in choice
                    }
                    return (
                        -len(content_pairs),
                        sum(rank[edge["edge_id"]] for edge in choice),
                    )
                edges = list(
                    min(
                        connected_choices,
                        key=choice_score,
                    )
                )
        connected = _connected(present, edges)
        content_pairs = [
            tuple(sorted((edge["left_digest"], edge["right_digest"])))
            for edge in edges
        ]
        duplicate_content_comparisons = len(content_pairs) - len(set(content_pairs))
        sufficient = len(edges) == edges_per_case and connected and len(present) == 4
        cases.append(
            {
                "case_id": case_id,
                "present_strategies": present,
                "unique_digest_count": len(
                    {item["artifact_digest"] for item in strategies.values()}
                ),
                "edges": edges,
                "same_digest_edges_rejected": same_digest_rejected,
                "duplicate_content_comparison_count": duplicate_content_comparisons,
                "comparison_graph_connected": connected and len(present) == 4,
                "status": "PAIR_READY" if sufficient else "PAIR_INSUFFICIENT_VARIATION",
            }
        )
        all_edges.extend(edges)
    return {
        "schemaVersion": "preference-pair-graph/v1",
        "summary": {
            "caseCount": len(cases),
            "edgeCount": len(all_edges),
            "readyCaseCount": sum(c["status"] == "PAIR_READY" for c in cases),
            "connectedCaseGraphCount": sum(
                c["comparison_graph_connected"] for c in cases
            ),
            "insufficientVariationCaseCount": sum(
                c["status"] != "PAIR_READY" for c in cases
            ),
            "sameDigestPairsIncluded": 0,
            "duplicateContentComparisonCount": sum(
                c["duplicate_content_comparison_count"] for c in cases
            ),
            "uniqueContentPairCount": len(all_edges)
            - sum(c["duplicate_content_comparison_count"] for c in cases),
        },
        "cases": cases,
        "edges": all_edges,
    }
