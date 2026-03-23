# audio_generation_stack_design

## 1. Purpose

This document defines the long-horizon audio generation stack for this repository.

The goal is **not** to claim that the full video-to-audio generation system is already implemented.
The goal is to make the repository structurally ready for that system by clearly separating:

- the current stable baseline path
- the neutral media tool layer
- future generation backbones
- future separation and preprocessing utilities
- future workflow orchestration and deployment paths

At the end of Week 2, the repository already has a stable engineering baseline centered on `baseline_v1`.
This design document explains how that baseline connects to the future structured sound-layer generation system without polluting the current supported runtime.

---

## 2. Design Goals

The stack is designed around the following goals:

1. Keep one stable and reproducible baseline path.
2. Make future audio generation modules easy to attach.
3. Separate tool-layer infrastructure from the main runtime spine.
4. Support future video-conditioned or workflow-conditioned audio generation.
5. Make deployment, benchmarking, and later service integration easier.
6. Keep repository structure explainable and maintainable over multiple weeks.

In plain terms:

- the current repo must stay runnable,
- future growth must stay possible,
- and the boundaries must stay clear.

---

## 3. Non-Goals for the Current Stage

This document does **not** assume the repository already supports:

- full video-to-audio generation
- production-grade long-form soundtrack synthesis
- tightly integrated AudioCraft / AudioLDM / Spleeter runtime
- end-to-end serving with Java / gRPC / Kubernetes
- final deployment-scale performance optimization

Those are future stages.

Week 2 only defines where these things belong and how they should connect later.

---

## 4. System View

The long-term target system can be understood as a layered stack.

### Layer A: Stable baseline spine

Current owner:

- `baseline_v1`

Responsibilities:

- dataset loading
- training
- checkpoint saving
- ONNX export
- PyTorch vs ONNX Runtime parity
- ONNX Runtime inference
- CPU benchmark
- Docker replay
- CI smoke

This is the current supported engineering path.

### Layer B: Neutral media tool layer

Current owner:

- `tools/media_probe/`

Responsibilities:

- inspect audio/video files
- extract metadata
- provide input contracts for later workflow assembly
- reduce ambiguity before generation or evaluation stages

This layer is infrastructure, not the main model path.

### Layer C: Future generation backbones

Planned candidates:

- AudioCraft / MusicGen / AudioGen
- AudioLDM family

Responsibilities:

- text-to-audio generation
- background / ambience generation
- Foley-like sound generation
- controllable audio content generation

These modules are staged, not default.

### Layer D: Future separation / preprocessing utilities

Planned candidate:

- Spleeter

Responsibilities:

- source separation
- stem extraction
- preprocessing for layered generation workflows
- possible support for evaluation and remix-style workflows

This module is also staged, not default.

### Layer E: Workflow orchestration and serving

Future scope:

- service wrapper
- workflow chaining
- deployment contracts
- observability
- performance regression tracking

This layer is not active yet, but the current repository should be built so that this layer can grow on top of it later.

---

## 5. Why the Stack Is Split This Way

If all modules are merged into one runtime too early, the repository becomes hard to debug, hard to reproduce, and hard to explain.

Typical failure mode:

- a baseline exists
- a probe tool exists
- a generation library exists
- a separation tool exists
- all of them get mixed into one "demo"
- nobody can tell which part is actually stable

This design intentionally avoids that trap.

The current principle is:

- baseline spine stays small and reliable
- tools stay neutral
- future modules get explicit landing zones
- integration happens only after boundaries are written down

That is slower in appearance, but much safer in engineering terms.

---

## 6. Current Module Roles

### 6.1 `baseline_v1`

Role:

- current engineering anchor
- current smoke baseline
- reference point for ONNX / ORT export and inference validation
- benchmark center for Week 1–2

Why it exists:

- to provide a stable and reproducible path before the repository grows more complex

What it is not:

- not the final audio generation model
- not the final multimodal system
- not the final deployment target

### 6.2 `tools/media_probe/`

Role:

- neutral media inspection utility

Expected inputs:

- audio file
- video file
- future containerized media assets

Expected outputs:

- path
- duration
- stream types
- sample rate
- channels
- fps
- width
- height
- codec / container metadata where available

Why it matters:

future video-to-audio workflows should not guess media properties.
They should read them first.

### 6.3 AudioCraft

Role in this repository:

- primary future generation backbone candidate

Why this role is reasonable:

AudioCraft is an audio generation research library and includes models such as MusicGen and AudioGen, so it naturally fits the "main generation backbone" slot for later staged integration. :contentReference[oaicite:4]{index=4}

Current status in this repo:

- design-level role defined
- optional environment/smoke record can be added
- not part of current baseline runtime

### 6.4 AudioLDM

Role in this repository:

- alternative generation backbone
- especially useful as a backup or specialized path for environment sound / Foley-like generation experiments

Why this role is reasonable:

the official repository frames AudioLDM as supporting speech, sound effects, music, and broader audio generation, which makes it a good staged candidate for environment or prompt-driven audio generation tasks. :contentReference[oaicite:5]{index=5}

Current status in this repo:

- design-level role defined
- not in the default runtime path

### 6.5 Spleeter

Role in this repository:

- source separation utility candidate

Why this role is reasonable:

Spleeter is designed for source separation with pretrained models and is therefore better treated as a preprocessing / decomposition tool than as a main generative backbone. :contentReference[oaicite:6]{index=6}

Current status in this repo:

- staged utility candidate
- can later support stem extraction, remix workflows, and evaluation preparation
- not part of the default baseline runtime

### 6.6 ffprobe / FFmpeg-family utilities

Role in this repository:

- low-level media inspection and conversion substrate

Why this role is reasonable:

`ffprobe` is built to gather container and stream information in human- and machine-readable form, so it is the correct foundation for metadata-level probing before model inference or generation. :contentReference[oaicite:7]{index=7}

Current status in this repo:

- conceptual dependency for media probing
- may be called by tool-layer scripts
- not part of `baseline_v1` logic

### 6.7 librosa / OpenCV

Repository role:

- Python-side helper stack for lightweight signal or frame-level inspection

Expected future usage:

- librosa for waveform/audio metadata and lightweight feature-side checks
- OpenCV for frame count / fps / resolution / basic video-side utility checks

These libraries belong to the tool layer, not to the core baseline runtime.

---

## 7. Proposed Long-Term Workflow

The future workflow can be described as:

### Stage 1: Input probing

Input:

- raw video
- raw audio
- extracted clips

Tooling:

- `media_probe`
- ffprobe
- lightweight Python-side helpers

Output:

- normalized metadata summary
- basic input contract checks

### Stage 2: Optional decomposition / preparation

Input:

- raw or probed media assets

Tooling:

- Spleeter or related preprocessing utilities
- future audio segmentation helpers
- future alignment helpers

Output:

- stems
- cleaned reference tracks
- separated components
- task-specific preparation assets

### Stage 3: Generation

Input:

- prompts
- optional video cues
- optional reference audio
- optional decomposed stems

Tooling candidates:

- AudioCraft
- AudioLDM
- future custom models

Output:

- generated sound layers
- ambience
- Foley-like assets
- music or soundscape candidates

### Stage 4: Layer composition / alignment

Input:

- generated layers
- original video timing
- optional separated stems
- metadata contracts from Stage 1

Future responsibilities:

- temporal alignment
- layer mixing
- gain and routing policy
- structured output packaging

### Stage 5: Runtime evaluation and deployment

Input:

- model outputs
- packaged artifacts
- benchmark targets

Future responsibilities:

- ONNX / runtime export where relevant
- service wrapping
- observability
- regression tracking
- deployment-ready packaging

---

## 8. Why `media_probe` Is a Tool Layer Instead of Main Runtime Logic

This is one of the most important design decisions in the repository.

`media_probe` is a tool layer because:

1. it serves many future workflows, not only one model
2. it is useful before generation, after generation, and during evaluation
3. it should remain independently testable
4. it should not make the baseline runtime more fragile
5. it belongs to infrastructure, not to model identity

So the correct relationship is:

- `baseline_v1` remains the supported model path
- `media_probe` supports future workflow assembly
- generation backbones consume or benefit from probed metadata later

This keeps the repository clean.

---

## 9. Repository Mapping

The stack maps to the repository as follows.

### 9.1 Stable baseline spine

- `src/`
- `scripts/`
- `configs/`
- `artifacts/experiments/`
- `artifacts/onnx/`
- `artifacts/logs/`
- `results/baseline_v1.md`

### 9.2 Tool layer

- `tools/media_probe/`

### 9.3 Architecture and design memory

- `docs/adr/0002-week02-bridge.md`
- `docs/design/audio_generation_stack_design.md`

### 9.4 Weekly and evidence narrative

- `docs/weekly/`
- `docs/benchmarks/`
- `docs/postmortems/`

This separation matches the repository rule that design docs belong to `docs/design/`, architecture decisions belong to `docs/adr/`, experiment cards belong to `results/`, and raw logs belong to `artifacts/logs/`.

---

## 10. Current Supported Path vs Future Path

### 10.1 Current supported path

Today, the repository officially supports:

- `baseline_v1`
- export -> parity -> inference -> benchmark
- Docker replay
- CI smoke
- media probing as a separate utility

### 10.2 Future staged path

Later, the repository should support:

- video-conditioned or workflow-conditioned audio generation
- optional separation before generation
- structured layer generation
- stronger generation backbones
- richer evaluation and deployment flows

The key point is:

the future path must grow **on top of** the stable current path,
not replace it prematurely with a giant unstable demo.

---

## 11. Integration Policy

Before any future generation/separation module becomes part of the default runtime, it should satisfy at least the following:

1. role is clearly written in ADR/design docs
2. environment assumptions are recorded
3. one smoke fact exists
4. repository entrypoint is explicit
5. failure mode does not break the baseline spine
6. logs or evidence can be stored in the existing structure

This keeps the project from turning into a folder full of vibes and broken imports.

---

## 12. Immediate Next Steps

Near-term next steps should be conservative.

### Priority 1

Keep the `baseline_v1` spine stable and documented.

### Priority 2

Preserve `media_probe` as an independent utility and improve its README / sample outputs when needed.

### Priority 3

Add exactly one traceable smoke fact for either:

- AudioCraft environment check
- Spleeter Docker help / minimal invocation

### Priority 4

Only after the above, consider a small integration prototype that uses:

- metadata from `media_probe`
- one staged generation or separation module
- a narrow, testable workflow target

---

## 13. Risks

### 13.1 Over-integration risk

Trying to force all future modules into the baseline runtime too early will destroy clarity.

### 13.2 Narrative drift risk

If design, ADR, weekly summary, and result card all start repeating different versions of the same story, the repo becomes harder to maintain.

### 13.3 Tool/runtime boundary drift

If utility-layer scripts silently become baseline dependencies, debugging cost will rise fast.

### 13.4 Architecture vanity risk

It is easy to write a cool-looking design for a giant system.
It is much harder to keep a small system reproducible.
This repository should prefer the second.

---

## 14. One-Sentence Design Rule

Build the repository as a stable baseline spine plus a neutral media tool layer plus explicitly staged future generation/separation branches, so that the long-term structured sound-layer generation system can grow without breaking the current reproducible path.