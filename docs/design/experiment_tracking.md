# Week10 Experiment / Artifact Tracking Design

Date: 2026-05-11
Scope: local evidence only. No remote artifact store, no production tracking server.

## 1. Why now

Week09 produced ORT benchmark, runtime matrix, and profiling artifacts. These artifacts are useful, but they are still isolated CSV, JSON, log, and trace files unless the project defines how benchmark inputs, commands, outputs, and evidence boundaries are tracked.

Week10 introduces experiment and artifact tracking to make benchmark and export evidence reproducible, comparable, and reviewable.

## 2. Tracking decision

Default route: DVC-first.

Reason:

- DVC fits the current repository style because Week09 evidence is already organized as scripts, CSV / JSON artifacts, logs, and benchmark documents.
- DVC can describe reproducible stages in dvc.yaml without requiring a tracking server.
- The current goal is artifact provenance and reproducible local evidence, not a team-level experiment registry.

Fallback route: MLflow local tracking.

Use MLflow only if DVC conflicts with already Git-tracked artifacts or makes the local evidence chain unnecessarily noisy.

## 3. Objects to track

| Type | Path | Meaning | Boundary |
|---|---|---|---|
| benchmark_csv | artifacts/benchmarks/ort_week09.csv | ORT benchmark matrix result | local CPU evidence only |
| runtime_matrix_csv | artifacts/benchmarks/ort_week09_runtime_matrix.csv | ORT runtime configuration comparison | no final runtime selection |
| profile_json | artifacts/benchmarks/ort_week09_profile_20260507.json | ORT profiling summary metrics | CPUExecutionProvider only |
| profile_trace | artifacts/profiles/ort_week09_profile_20260507_2026-05-07_09-53-52.json | ORT profiling trace | local trace artifact only |
| benchmark_script | scripts/bench_ort_matrix.sh | benchmark reproduction entry | local shell command |
| profile_script | scripts/bench_ort_profile.py | profiling reproduction entry | conservative CPU probe |

## 4. Candidate DVC stage

The first DVC stage should be conservative. It should register the benchmark / profiling command and known outputs, but should not move large datasets or already stable evidence into remote storage.

Candidate stage name:

- week10_ort_evidence

Candidate dependencies:

- scripts/bench_ort_matrix.sh
- scripts/bench_ort_profile.py
- artifacts/onnx/baseline_v1.onnx

Candidate outputs or metrics:

- artifacts/benchmarks/ort_week09.csv
- artifacts/benchmarks/ort_week09_runtime_matrix.csv
- artifacts/benchmarks/ort_week09_profile_20260507.json

## 5. Verification commands

Run these commands from the repository root:

    python -m json.tool artifacts/benchmarks/ort_week09_profile_20260507.json >/dev/null
    head -n 5 artifacts/benchmarks/ort_week09.csv
    head -n 5 artifacts/benchmarks/ort_week09_runtime_matrix.csv
    test -f scripts/bench_ort_matrix.sh
    test -f scripts/bench_ort_profile.py

Expected result:

- The profiling JSON is valid.
- The benchmark CSV exists and prints its header / first rows.
- The runtime matrix CSV exists and prints its header / first rows.
- The benchmark and profiling scripts exist.

## 6. Not yet verified

- No remote DVC storage.
- No MLflow tracking server.
- No team experiment registry.
- No CUDAExecutionProvider benchmark.
- No TensorRT / vLLM / SGLang runtime comparison.
- No production runtime selection.
- No claim that DVC or MLflow is fully integrated into CI.

## 7. Week10 minimum DoD

Week10 minimum DoD for this branch:

- docs/design/experiment_tracking.md exists and defines tracking scope.
- README top section points to Week10 experiment / artifact tracking.
- A DVC or MLflow local tracking path is selected based on actual repository friction.
- At least one existing Week09 benchmark / profile artifact is registered or logged through the selected route.
- Evidence remains local and reproducible.
