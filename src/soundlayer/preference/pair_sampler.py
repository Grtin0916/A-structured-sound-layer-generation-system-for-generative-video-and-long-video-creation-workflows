"""Create private/public pair manifests without leaking strategy semantics."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from .session_planner import plan_blocks


def _opaque(seed, text, prefix):
    digest = hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def sample_pairs(inventory, graph, unique_pairs=36, repeat_pairs=8, audit_pairs=4, seed=0):
    candidates = {
        (item["case_id"], item["strategy_id"]): item
        for item in inventory["candidates"]
    }
    rng = random.Random(seed)
    available_edges = list(graph["edges"])
    selected = available_edges[:unique_pairs]
    records = []
    for index, edge in enumerate(selected):
        records.append(_make_record(edge, candidates, seed, index, "UNIQUE", rng))

    repeat_sources = records[:]
    rng.shuffle(repeat_sources)
    for index, source in enumerate(repeat_sources[:repeat_pairs]):
        repeated = dict(source)
        repeated["pair_id"] = _opaque(seed, f"repeat:{index}:{source['pair_id']}", "pair")
        repeated["kind"] = "HIDDEN_REPEAT"
        repeated["is_hidden_repeat"] = True
        repeated["repeat_group_id"] = source["repeat_group_id"]
        repeated["left_strategy"], repeated["right_strategy"] = (
            source["right_strategy"],
            source["left_strategy"],
        )
        repeated["left_artifact"], repeated["right_artifact"] = (
            source["right_artifact"],
            source["left_artifact"],
        )
        repeated["left_digest"], repeated["right_digest"] = (
            source["right_digest"],
            source["left_digest"],
        )
        repeated["presentation_order"] = "SWAPPED"
        records.append(repeated)

    audit_sources = [r for r in records if r["kind"] == "UNIQUE"]
    rng.shuffle(audit_sources)
    for index, source in enumerate(audit_sources[:audit_pairs]):
        audited = dict(source)
        audited["pair_id"] = _opaque(seed, f"audit:{index}:{source['pair_id']}", "pair")
        audited["kind"] = "AUDIT"
        audited["is_audit_pair"] = True
        audited["repeat_group_id"] = ""
        if rng.random() < 0.5:
            audited["left_strategy"], audited["right_strategy"] = (
                source["right_strategy"],
                source["left_strategy"],
            )
            audited["left_artifact"], audited["right_artifact"] = (
                source["right_artifact"],
                source["left_artifact"],
            )
            audited["left_digest"], audited["right_digest"] = (
                source["right_digest"],
                source["left_digest"],
            )
            audited["presentation_order"] = "SWAPPED"
        records.append(audited)

    planned = plan_blocks(records, seed)
    private = []
    public = []
    for record in planned:
        pair_id = record["pair_id"]
        left_ext = Path(record["left_artifact"]).suffix
        right_ext = Path(record["right_artifact"]).suffix
        video_ext = Path(record["video_path"]).suffix
        left_media = f"media/{_opaque(seed, pair_id + ':left', 'a')}{left_ext}"
        right_media = f"media/{_opaque(seed, pair_id + ':right', 'b')}{right_ext}"
        video_media = f"media/{_opaque(seed, record['case_id'] + ':video', 'v')}{video_ext}"
        record.update(
            left_media=left_media, right_media=right_media, video_media=video_media
        )
        private.append(record)
        public.append(
            {
                "opaque_pair_id": pair_id,
                "left_media": left_media,
                "right_media": right_media,
                "video_media": video_media,
                "block_id": record["block_id"],
                "display_index": record["display_index"],
            }
        )
    summary = {
        "requestedUniquePairCount": unique_pairs,
        "uniquePairCount": len(selected),
        "uniquePairShortfall": max(0, unique_pairs - len(selected)),
        "hiddenRepeatCount": sum(r["is_hidden_repeat"] for r in private),
        "auditPairCount": sum(r["is_audit_pair"] for r in private),
        "judgmentCount": len(private),
        "caseCoverage": len({r["case_id"] for r in private}),
        "sameDigestPairCount": sum(
            r["left_digest"] == r["right_digest"] for r in private
        ),
    }
    return private, public, summary


def _make_record(edge, candidates, seed, index, kind, rng):
    left = candidates[(edge["case_id"], edge["left_strategy"])]
    right = candidates[(edge["case_id"], edge["right_strategy"])]
    swapped = rng.random() < 0.5
    if swapped:
        left, right = right, left
    pair_id = _opaque(seed, f"unique:{index}:{edge['edge_id']}", "pair")
    return {
        "pair_id": pair_id,
        "case_id": edge["case_id"],
        "kind": kind,
        "protocol_version": "preference-v1",
        "review_session_id": f"review-{seed}",
        "left_strategy": left["strategy_id"],
        "right_strategy": right["strategy_id"],
        "left_artifact": left["artifact_path"],
        "right_artifact": right["artifact_path"],
        "left_digest": left["artifact_digest"],
        "right_digest": right["artifact_digest"],
        "video_path": left["video_path"],
        "presentation_order": "SWAPPED" if swapped else "AS_SAMPLED",
        "is_hidden_repeat": False,
        "repeat_group_id": _opaque(seed, f"group:{edge['edge_id']}", "repeat"),
        "is_audit_pair": False,
    }
