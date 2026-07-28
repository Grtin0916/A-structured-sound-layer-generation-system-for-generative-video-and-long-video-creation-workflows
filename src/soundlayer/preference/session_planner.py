"""Deterministic block planning for unique, repeat, and audit judgments."""

from __future__ import annotations

import random


def plan_blocks(records, seed, block_count=4):
    rng = random.Random(seed)
    unique = [r for r in records if r["kind"] == "UNIQUE"]
    repeat = [r for r in records if r["kind"] == "HIDDEN_REPEAT"]
    audit = [r for r in records if r["kind"] == "AUDIT"]
    rng.shuffle(unique)
    rng.shuffle(repeat)
    rng.shuffle(audit)
    blocks = [[] for _ in range(block_count)]
    for index, record in enumerate(unique):
        blocks[index % (block_count - 1)].append(record)
    # The final block contains only hidden repeats and audits, so no repeat is
    # adjacent to its source and the reviewer cannot infer it from ordering.
    blocks[-1].extend(repeat)
    blocks[-1].extend(audit)
    ordered = []
    for block_index, block in enumerate(blocks, 1):
        rng.shuffle(block)
        for record in block:
            record["block_id"] = f"block-{block_index}"
            ordered.append(record)
    for index, record in enumerate(ordered, 1):
        record["display_index"] = index
    return ordered
