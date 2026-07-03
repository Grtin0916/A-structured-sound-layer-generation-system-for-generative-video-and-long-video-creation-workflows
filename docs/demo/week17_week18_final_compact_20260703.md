# Week17/Week18 Final Compact｜2026-07-03

## Current state

Today produced two closed chains.

### W17 demo release chain

Mainbase packaged the true-aware demo release candidate with ZIP, index page, WAV fallback, manifest, and claim boundary. Java exposed the release handoff API and passed RANDOM_PORT integration test. Cloud aggregated the release gate with dashboard-ready, Prometheus sample, alert rules, and runbook.

### W18 seed chain

Mainbase generated 12 prompt tasks from six W17 cases: six naive prompts and six DSS prompts. Java exposed the prompt task seed API and passed RANDOM_PORT integration test. Cloud aggregates this seed as the next runnable experiment gate.

## Honest claim boundary

Safe claims:

- One true MMAudio replacement is traceable.
- W17 demo release is portable and browser-previewable.
- Java exposes artifact-backed APIs.
- Cloud produces dashboard-ready and Prometheus-sample artifacts.
- W18 has a machine-readable DSS-vs-naive prompt task queue.

Blocked claims:

- No true MMAudio batch success.
- No full candidate ranking.
- No production SLO verification.
- No k6 threshold pass.
- No live Grafana import.
- No model-quality gain claim for W18 yet.

## Interview bullets

- Built a Director-guided Video-to-Audio SoundLayer workflow from model artifact to platform handoff and cloud gate.
- Converted partial true-model success into a claim-safe demo release instead of overclaiming batch success.
- Designed W18 DSS-vs-naive prompt ablation inputs from real W17 cases.
- Preserved failure boundaries as first-class fields for later repair workflow.
- Verified Java handoff behavior with RANDOM_PORT integration tests.
- Exported Cloud metrics and dashboard-ready JSON without pretending production SLO or live Grafana import.

## Next Monday entry point

Start with `reports/week18_prompt_tasks_20260703.jsonl`.

Run each case with two prompt variants:

1. naive prompt
2. DSS prompt

Evaluate:

- event coverage
- onset alignment proxy
- forbidden leakage
- RMS / peak / silence ratio
- true-anchor consistency for `glass_drop_room_001`

Then route failures into repair bank.
