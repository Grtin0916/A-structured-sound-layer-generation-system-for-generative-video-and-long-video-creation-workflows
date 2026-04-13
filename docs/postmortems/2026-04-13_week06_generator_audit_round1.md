# Week06 Generator Audit Round 1

## Scope
- target_model: facebook/musicgen-small
- task: text-to-music generator audit round 1
- prompt: minimal ambient electronic texture with soft pads

## Environment
- python: 3.9.23
- torch: 2.1.0+cu118
- torchaudio: 2.1.0+cu118
- audiocraft: 1.3.0
- GPU: NVIDIA A100-PCIE-40GB
- runtime fix: prioritize `$CONDA_PREFIX/lib` in `LD_LIBRARY_PATH` to avoid `libstdc++.so.6` / `CXXABI_1.3.15` import failure

## Run 1
- duration_sec: 2
- output_path: artifacts/outputs/week06/musicgen_small_smoke_2s_001.wav
- sample_rate: 32000
- elapsed_sec: 19.75
- peak_allocated_mib: 1712.06
- peak_reserved_mib: 1790.00

## Run 2
- duration_sec: 5
- output_path: artifacts/outputs/week06/musicgen_small_followup_5s_001.wav
- sample_rate: 32000
- elapsed_sec: 8.75
- peak_allocated_mib: 1783.96
- peak_reserved_mib: 2078.00

## Interpretation
- `facebook/musicgen-small` is runnable in the current local environment.
- The earlier blocker was a dynamic library resolution issue, not a GPU insufficiency issue.
- The 5-second follow-up also succeeded, and the observed peak memory remained low enough for continued short-sequence audit work.
- The shorter elapsed time in Run 2 should not be overinterpreted as a strict performance improvement; model warm-up / cache state / runtime state may have affected it.

## Today conclusion
Today's minimum main-repo target for Week06 generator audit is completed:
- README aligned
- audit template created
- environment probe recorded
- first successful small-model generation recorded
- follow-up run recorded

## Next step
- switch to Java repo and freeze Week06 auth strategy in `docs/adr/0002-auth-strategy.md`
- later this week, converge this round into the formal Week06 audit file
