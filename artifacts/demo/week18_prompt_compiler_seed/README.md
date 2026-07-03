# W18 Prompt Compiler Seed

This directory contains a machine-readable prompt task queue generated from the Week17 true-aware demo release.

## Files

- `week18_prompt_tasks_20260703.jsonl`: 12 prompt tasks.
- `week18_prompt_task_manifest_20260703.json`: source and boundary manifest.

## Task design

Each W17 case has two prompt variants:

1. `naive_prompt`: normal text prompt baseline.
2. `dss_prompt`: DirectorSound Script controlled prompt.

## Boundary

The true MMAudio record is a positive anchor only. It is not batch success.
