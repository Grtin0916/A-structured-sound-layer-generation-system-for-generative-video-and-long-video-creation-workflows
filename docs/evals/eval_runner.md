# Week11 Eval Runner V0

Scope: Mainbase provides the SoundLayer eval root for the Week11 cross-repo demo.

Artifacts:
- `scripts/run_eval.py`: produces the Week11 proxy eval outputs.
- `artifacts/evals/week11_eval_v0.json`: machine-readable eval payload.
- `artifacts/evals/week11_eval_v0.csv`: tabular eval rows.
- `artifacts/manifests/week11_crossrepo_task_bridge.json`: bridge between Mainbase eval evidence and downstream Java/Cloud consumption.
- `artifacts/manifests/week11_e2e_demo_index.json`: demo index for task -> eval -> API evidence link -> k6 consumer gate.

Boundary:
This V0 eval is a SoundLayer Blueprint / Eval-as-Product evidence check. It does not claim generated audio perceptual quality, production artifact registry support, production SLO, or real cloud load testing.
