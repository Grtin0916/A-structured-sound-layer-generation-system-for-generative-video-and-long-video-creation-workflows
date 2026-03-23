# 0002-week02-bridge

- Status: accepted
- Date: 2026-03-21

## Context

This repository is evolving from a Week 1 minimal engineering baseline into a longer-horizon project for structured sound-layer generation in generative video and long-video workflows.

By the end of Week 1, the repository already had a usable baseline path centered on `baseline_v1`:

- minimal dataset preparation
- manifest-based loading
- training and validation
- checkpoint saving
- ONNX export
- PyTorch vs ONNX Runtime parity
- ONNX Runtime CPU inference
- benchmark logging

During Week 2, the project scope widened in two different directions at the same time:

1. **Engineering hardening of the current baseline**
   - multi-sample parity and benchmark
   - export stability / shape audit
   - Docker replay
   - GitHub Actions smoke workflow
   - result card / weekly summary / benchmark docs

2. **Bridge work toward the long-term structured audio generation system**
   - `tools/media_probe/`
   - `docs/design/audio_generation_stack_design.md`
   - staged understanding of AudioCraft / AudioLDM / Spleeter roles

This created a structural risk:

- if every new idea is mixed directly into the main runtime path,
  the repository becomes harder to explain, harder to test, and harder to reproduce;
- if all bridge work remains only in notes and discussion,
  the long-term project never becomes a real engineering trajectory inside the repo.

Therefore, Week 2 requires an explicit architecture decision:
**how should the current baseline, the neutral media utility layer, and future generation / separation branches relate to each other inside this repository?**

---

## Decision Drivers

The decision is driven by the following needs:

1. Keep one stable and supported main runtime path.
2. Preserve reproducibility and CI friendliness.
3. Avoid mixing tool-layer code with the baseline inference path.
4. Give the long-term project real repository landing zones instead of abstract future ideas.
5. Make later integration of generation, separation, and media I/O possible without rewriting Week 2 assets.
6. Keep the repository explainable for future self-review, interviews, and incremental system growth.

---

## Considered Options

### Option A: Merge tool layer and future generation modules directly into the baseline runtime now

Examples:

- make `media_probe` part of `scripts/run_demo.sh`
- force AudioCraft / AudioLDM / Spleeter into the current smoke path
- treat Week 2 as the start of an integrated end-to-end system

#### Pros

- looks ambitious
- gives an illusion of faster system integration
- fewer conceptual boundaries on paper

#### Cons

- high coupling
- harder debugging
- more fragile CI and Docker replay
- harder to know whether failures come from baseline, runtime, or future bridge modules
- Week 2 evidence becomes noisy and difficult to compare over time

### Option B: Keep `baseline_v1` as the only supported main runtime spine, treat `media_probe` as a neutral tool layer, and stage AudioCraft / AudioLDM / Spleeter as future branches with explicit but non-default repository positions

#### Pros

- stable main path
- lower cognitive load
- easier Docker / CI / benchmark maintenance
- easier repository storytelling
- bridge modules still become real repository assets instead of vague ideas

#### Cons

- long-term system integration appears slower
- current repository does not yet look like a full audio generation platform
- some future-facing modules remain staged rather than fully active

---

## Decision

We choose **Option B**.

### 1. `baseline_v1` remains the single supported main runtime baseline

`baseline_v1` is the current engineering anchor of the repository.

It remains the only baseline that is:

- expected to pass the main demo path
- tied to ONNX export and ORT validation
- used in Docker replay
- referenced by the Week 2 result card, benchmark docs, postmortems, and weekly summary
- suitable for the current CI smoke workflow

This means the repository continues to have one clear answer to:
**what is the current supported path?**

The answer is:
`baseline_v1` and its demo / export / parity / infer / benchmark spine.

### 2. `tools/media_probe/` is defined as a neutral tool layer, not part of the baseline main runtime

`media_probe` is intentionally placed under `tools/` rather than `scripts/` or `src/engine/`.

Its role is to provide reusable media inspection capability for future pipelines, including:

- duration
- sample rate
- channels
- fps
- container / codec / stream metadata
- basic input contract checking

It is **not** currently part of the `baseline_v1` smoke path.

Reason:

- it serves future video/audio workflow integration,
- but it should not make the current main runtime path more fragile.

In short:

`media_probe` is infrastructure for future workflow assembly, not a replacement for the current baseline spine.

### 3. AudioCraft, AudioLDM, and Spleeter are staged as future branches, not mixed into the default runtime

Their current repository roles are defined as follows:

- **AudioCraft**: primary future generation backbone candidate
- **AudioLDM**: alternative text-to-audio / environment / Foley generation candidate
- **Spleeter**: source separation utility candidate

At Week 2, they are allowed to appear in:

- ADR discussion
- design documents
- environment checks
- optional smoke records
- future branch planning

They are **not** allowed to silently redefine the current default supported runtime.

This boundary is deliberate.

The repository should not pretend that a staged module is already part of the production-like main path.

### 4. Design documents, ADRs, benchmark reports, weekly summaries, and result cards each keep separate responsibilities

This bridge decision also fixes documentation boundaries:

- `docs/adr/` records architecture decisions
- `docs/design/` records long-horizon system structure and module roles
- `docs/benchmarks/` records human-readable performance summaries
- `docs/postmortems/` records audits and one-off investigations
- `docs/weekly/` records weekly summaries
- `results/` records model cards / experiment cards / result summaries
- `artifacts/logs/` stores runtime logs, CSVs, and generated evidence

This prevents Week 2 bridge work from degenerating into scattered notes.

### 5. The repository is treated as a long-term engineering base, not as a one-week demo folder

This decision formally defines the repository as:

- an engineering base that already has a stable baseline,
- plus explicitly staged extension points for future audio generation system growth.

That is the key bridge.

The long-term project is no longer "outside the repo".
It is now structurally represented inside the repo, but without destabilizing the supported baseline.

---

## Decision Outcome

The repository architecture is now interpreted in three layers:

### Layer 1: Stable main runtime spine

Current owner:

- `baseline_v1`

Current responsibilities:

- train
- export
- parity
- inference
- benchmark
- Docker replay
- CI smoke

This is the path that should stay boring, reliable, and easy to rerun.

### Layer 2: Neutral tool layer

Current owner:

- `tools/media_probe/`

Current responsibilities:

- media inspection
- metadata extraction
- input contract support
- future workflow assistance

This layer supports future system assembly but does not redefine the baseline path.

### Layer 3: Future generation / separation branches

Current staged candidates:

- AudioCraft
- AudioLDM
- Spleeter

Current responsibilities:

- role clarification
- optional environment or smoke checks
- later integration planning

These branches are real repository concerns, but they are not the current default runtime.

---

## Consequences

### Positive Consequences

1. The repository keeps one clean supported path.
2. Week 2 documents now have a stable narrative center.
3. Future modules can be added without rewriting the baseline story.
4. CI and Docker stay focused on a small, reproducible target.
5. The long-term project now has explicit landing zones inside the repo.
6. Interviews, reviews, and future self-maintenance become easier because the architecture is explainable.

### Negative Consequences

1. Some future-facing modules remain only staged rather than fully exercised.
2. The repository may look less ambitious than a fully integrated prototype.
3. More discipline is required to avoid sneaking future branch logic into the baseline path.

### Neutral but Important Consequence

This decision intentionally prioritizes:

- reproducibility over integration speed
- structure over appearance
- sustained project growth over short-term "big system" optics

That tradeoff is accepted.

---

## What This ADR Explicitly Forbids

The following behaviors are considered architecture drift and should be avoided unless a later ADR changes the policy:

1. Treating `media_probe` as if it were already part of the default baseline runtime.
2. Adding AudioCraft / AudioLDM / Spleeter as mandatory dependencies for the main smoke path without a new architecture decision.
3. Moving bridge notes into random weekly or log documents instead of maintaining ADR / design boundaries.
4. Splitting the same evolving bridge topic into many date-based note files.
5. Presenting staged future modules as already supported default production paths.

---

## What This ADR Enables Next

This decision makes the following next steps straightforward:

1. Keep `baseline_v1` stable while refining docs, results, Docker, and CI.
2. Add one optional, traceable smoke fact for AudioCraft or Spleeter without polluting the main path.
3. Continue expanding `docs/design/audio_generation_stack_design.md`.
4. Later introduce a dedicated integration branch or a new ADR for generation-chain onboarding.
5. Build future video-to-audio workflow assembly on top of:
   - a stable baseline spine
   - a neutral media tool layer
   - explicitly staged generation / separation modules

---

## Status Check After Week 2

At the end of Week 2, this ADR means:

- the baseline path is real
- the bridge is real
- the boundaries are real
- the full integrated system is not yet the default path

That is acceptable.

Week 2 is about making the repository credible, not pretending it is already complete.

---

## Related Files

- `results/baseline_v1.md`
- `docs/weekly/2026-03-21_week02_summary.md`
- `docs/design/audio_generation_stack_design.md`
- `docs/postmortems/2026-03-16_week02_repro_audit.md`
- `docs/postmortems/2026-03-18_week02_shape_audit.md`
- `tools/media_probe/README.md`
- `.github/workflows/smoke.yml`

---

## One-Sentence Summary

Keep `baseline_v1` as the only supported main runtime spine, define `media_probe` as a neutral tool layer, stage AudioCraft / AudioLDM / Spleeter as future branches with explicit repository roles, and use this boundary to turn the repo into a real long-term engineering base instead of a pile of loosely connected experiments.  