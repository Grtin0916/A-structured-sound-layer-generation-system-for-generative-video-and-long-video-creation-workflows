# W18 Prompt Compiler Seed

## Seed

`/home/GRT/work/audio_engineering_repo_skeleton_v1/reports/week18_seed_from_week17_demo_release_20260703.json`

## Core decision

W17 produced a claim-safe demo release candidate. W18 should not restart from generic prompts. It should use W17 case records as controlled inputs.

## Positive anchor

- Case: `glass_drop_room_001`
- Artifact: `artifacts/demo/week17_true_aware_demo_release/audio/glass_drop_room_001__mmaudio__true_replacement_v0.wav`
- Meaning: one true MMAudio replacement is traceable.
- Boundary: this is not batch success.

## W18 experiment design

1. Build DSS v1 fields: scene, events, layer roles, avoid list, timing tolerance.
2. Generate naive prompt and DSS prompt for each case.
3. Evaluate event coverage, onset alignment proxy, forbidden leakage, loudness, silence ratio.
4. Promote successes into selector examples.
5. Convert failures into repair targets.

## Hard boundary

Do not claim:

- true MMAudio batch success
- full candidate ranking
- production SLO
- k6 threshold pass
- live Grafana import
