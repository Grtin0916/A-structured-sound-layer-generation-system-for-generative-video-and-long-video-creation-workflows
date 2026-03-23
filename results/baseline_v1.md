# A Structured Sound-Layer Generation System for Generative Video and Long-Video Creation Workflows

## Overview

This repository is an engineering-first project for building a structured sound-layer generation system for generative video and long-video creation workflows.

At the current stage, the repository does **not** aim to present a final large-scale audio generation system. Instead, it focuses on building a **reproducible, inspectable, and extensible engineering baseline** that can support future work on:

- layered sound generation
- environment and Foley generation
- audio separation and preprocessing
- video-to-audio workflow integration
- deployment-oriented inference and evaluation

The current supported baseline is `baseline_v1`, a minimal mel-spectrogram reconstruction model used to validate the repository's end-to-end engineering spine:
data -> training -> checkpoint -> ONNX export -> ONNX Runtime inference -> benchmark -> Docker replay -> CI smoke

---

## Current Status

The repository has completed a Week 1 to Week 2 transition from a "minimal training demo" to a **v0.2 engineering baseline**.

What is already working:

- minimal dataset preparation and manifest-based loading
- baseline training and validation
- checkpoint saving and experiment logging
- ONNX export from the trained checkpoint
- PyTorch vs ONNX Runtime parity check
- ONNX Runtime CPU inference and benchmarking
- multi-sample parity and benchmark evidence
- Docker-based replay of the CPU inference path
- GitHub Actions smoke workflow
- media probing utility as a neutral tool layer for future video/audio pipelines

What is **not** claimed yet:

- final production-quality audio generation
- dynamic-shape-safe deployment for arbitrary inputs
- integrated AudioCraft / AudioLDM / Spleeter generation pipeline
- full service-level performance characterization

In plain English: this repo is already a solid base, but it is still the base.

---

## Main Supported Path

The current main supported path is the `baseline_v1` demo spine.

Recommended entry:

`bash scripts/run_demo.sh demo`

This path is intended to validate the smallest stable pipeline:

1. export the trained model to ONNX
2. run PyTorch vs ONNX Runtime parity check
3. run ONNX Runtime inference
4. run ONNX Runtime CPU benchmark

If you want to replay the broader pipeline from earlier stages, use:

`bash scripts/run_demo.sh full`

---

## Why This Repository Exists

A lot of research repos die in the exact same way: interesting ideas, messy execution.

This repository takes the opposite route.

The immediate goal is to make sure the project has:

- a stable directory structure
- a traceable experiment path
- reproducible inference assets
- benchmark evidence that can be compared over time
- clean hooks for future model and system upgrades

That makes it suitable as a long-horizon engineering base for future work in structured audio generation for video workflows.

---

## Repository Structure

Top-level directories are organized by responsibility.

- `configs/`  
  configuration files for training and runtime

- `data/`  
  processed datasets, manifests, and related data assets

- `src/`  
  core training and model code

- `scripts/`  
  runnable project entrypoints such as demo, benchmark, and audit scripts

- `artifacts/`  
  generated outputs such as checkpoints, ONNX files, logs, metrics, and experiment assets

- `results/`  
  model cards and result summaries

- `docs/`  
  weekly summaries, benchmark notes, design docs, ADRs, and postmortems

- `tools/`  
  utility modules decoupled from the main training/inference path

- `docker/`  
  container build assets for reproducible replay

- `.github/workflows/`  
  CI smoke workflows

---

## Quick Start

### 1. Environment

Use a Python environment that already contains the dependencies required by your current setup for:

- PyTorch
- NumPy
- ONNX
- ONNX Runtime
- audio preprocessing libraries used by this repository

If you are continuing from the existing weekly workflow, reuse your prepared environment rather than rebuilding everything from zero.

### 2. Minimal Demo

Run the main supported demo path:

`bash scripts/run_demo.sh demo`

### 3. Full Replay

Run the broader replay path:

`bash scripts/run_demo.sh full`

### 4. Training Entry

The baseline training entry is:

`python src/engine/train.py --config configs/train/baseline.yaml --device cuda`

### 5. Main Example Input

A representative example audio file used in the current baseline path is:

`data/processed/esc10_miniset/valid/5-177957-C-40.wav`

---

## Data

The current baseline uses a minimal ESC-10 subset derived from ESC-50.

Current dataset organization includes:

- processed train split
- processed valid split
- manifest files
- per-sample sidecar metadata

Key characteristics of the current data path:

- mono audio
- resampled to 16 kHz
- fixed clip duration
- manifest-driven loading
- dynamic log-mel computation during dataset reading

This setup is intentionally simple and stable so that the engineering path remains easy to debug and extend.

---

## Baseline Model

The current baseline model is `baseline_v1`.

Role of `baseline_v1`:

- the main smoke baseline for the repository
- the anchor point for ONNX and ONNX Runtime validation
- the reference model for CPU benchmark tracking
- the current center of README, result card, benchmark docs, and weekly summaries

It is a **minimal mel autoencoder-style baseline**, not the final target system.

Its purpose is not to be flashy.
Its purpose is to keep the main path stable, reproducible, and extensible.

---

## Key Outputs

Important generated assets include:

- trained checkpoints under `artifacts/experiments/baseline_v1/`
- exported ONNX model under `artifacts/onnx/`
- parity logs under `artifacts/logs/`
- benchmark CSVs under `artifacts/logs/`
- result summary under `results/baseline_v1.md`

Representative evidence files:

- `artifacts/logs/onnx_parity.md`
- `artifacts/logs/parity_multi_sample.csv`
- `artifacts/logs/bench_ort_cpu.csv`
- `artifacts/logs/bench_ort_multi_sample.csv`
- `artifacts/logs/bench_ort_multi_sample_docker.csv`

---

## Documentation Map

If you want the shortest path, start here:

- [Main result card](results/baseline_v1.md)
- [Week 2 summary](docs/weekly/2026-03-21_week02_summary.md)

If you want benchmarking details:

- [ORT CPU baseline](docs/benchmarks/ort_cpu_baseline.md)
- [ORT CPU multi-sample](docs/benchmarks/ort_cpu_multi_sample.md)
- [ORT CPU multi-sample in Docker](docs/benchmarks/ort_cpu_multi_sample_docker.md)
- [Host vs Docker comparison](docs/benchmarks/ort_docker_cpu_compare.md)

If you want audit and postmortem records:

- [Week 2 repro audit](docs/postmortems/2026-03-16_week02_repro_audit.md)
- [Week 2 shape audit](docs/postmortems/2026-03-18_week02_shape_audit.md)

If you want architecture and long-horizon project context:

- [Week 2 bridge ADR](docs/adr/0002-week02-bridge.md)
- [Audio generation stack design](docs/design/audio_generation_stack_design.md)

If you want the media utility layer:

- [Media probe README](tools/media_probe/README.md)

---

## CI and Reproducibility

This repository has already started moving from manual replay toward continuous reproducibility.

Current reproducibility-related assets include:

- Docker replay path for the CPU inference chain
- benchmark comparison between host and Docker
- GitHub Actions smoke workflow in `.github/workflows/smoke.yml`

The project standard is simple:

- if it cannot be rerun, it does not count
- if it cannot be compared, it is hard to improve
- if it cannot be explained, it will eventually rot

---

## Tool Layer and Future Integration

The repository already contains a neutral utility layer for future audio/video workflow integration:

- `tools/media_probe/`

`media_probe` is currently treated as a **tool layer**, not part of the core `baseline_v1` runtime.

Planned future directions include controlled integration of:

- AudioCraft-style generation backbones
- AudioLDM-style environment or Foley generation paths
- Spleeter-style source separation utilities
- video-to-audio structured workflow assembly

Important boundary:

these modules are currently **planned or staged**, not presented as already integrated production paths.

---

## Current Limitations

The current baseline is intentionally narrow.

Known limitations include:

- the main ONNX / ORT path is still centered on fixed-shape validation
- current benchmark evidence is smoke-level engineering evidence, not full deployment profiling
- the main validated runtime path is CPUExecutionProvider-centered
- the repo currently emphasizes engineering stability over generation sophistication

That tradeoff is deliberate.
Right now, correctness and reproducibility beat glamour.

---

## Roadmap

Near-term priorities:

1. keep the `baseline_v1` path stable and fully documented
2. continue cleaning Docker and log asset placement
3. add one concrete smoke fact for a future generation or separation branch
4. gradually extend from the current baseline to a richer structured sound-layer workflow

Longer-term priorities:

- layered audio generation for video workflows
- stronger model backbones
- cleaner deployment contracts
- broader runtime and evaluation support
- better linkage between media analysis, generation, and final workflow assembly

---

## Project Philosophy

This repository is built with an engineering-first mindset.

The goal is not to pretend the big system is already done.
The goal is to make sure each step is:

- real
- rerunnable
- documented
- comparable
- extensible

That is less glamorous than a giant one-shot repo drop.
It is also how real systems survive contact with reality.

---

## Current Recommended Reading Order

If you are new to this repository, read in this order:

1. `README.md`
2. `results/baseline_v1.md`
3. `docs/weekly/2026-03-21_week02_summary.md`
4. `docs/benchmarks/ort_docker_cpu_compare.md`
5. `docs/adr/0002-week02-bridge.md`

That gives you the shortest path from project intent to current evidence to future direction.

---

## Status Snapshot

Current project state:

- baseline path: working
- ONNX export: working
- ORT CPU inference: working
- multi-sample parity evidence: available
- multi-sample benchmark evidence: available
- Docker replay: available
- CI smoke: available
- long-term generation stack integration: staged, not yet the default path

In one sentence:

this repository has moved beyond a toy demo, but it is still intentionally building the floor before building the skyscraper.