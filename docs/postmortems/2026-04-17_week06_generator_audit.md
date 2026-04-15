# 2026-04-17 Week06 Generator Audit

## Environment Summary
- conda env: audio-mini
- python_version: 3.9.23
- torch_version: 2.1.0+cu118
- torchaudio_version: 2.1.0+cu118
- audiocraft_version: 1.3.0
- ffmpeg_version: 6.1.2
- gpu: NVIDIA A100-PCIE-40GB
- cuda_available: True

## Execution Path
- step1: completed environment probe and verified current conda runtime
- step2: verified audiocraft import under current conda env and LD_LIBRARY_PATH
- step3: loaded pretrained model `facebook/musicgen-small`
- step4: completed run_001 text-to-music smoke generation with duration=4 sec
- step5: completed run_002 controlled text-to-music generation with duration=8 sec
- step6: saved artifacts under `artifacts/generated/week06_generator_audit/`

## Result Judgment
- reproducibility: reproducible_for_local_smoke_scope
- status: local MusicGen small smoke path is reproducible under the current environment
- validated_scope:
  - model loading via `MusicGen.get_pretrained("facebook/musicgen-small")`
  - GPU generation on A100 40GB
  - wav artifact export via `audio_write`
- evidence:
  - run_001 output: `artifacts/generated/week06_generator_audit/musicgen_small_try001.wav`
  - run_002 output: `artifacts/generated/week06_generator_audit/musicgen_small_try002_8s.wav`
  - run_001 status: success
  - run_002 status: success
  - run_002 elapsed_sec: 8.56
  - run_002 peak_allocated_mib: 1817.75
  - run_002 peak_reserved_mib: 2144.0

## Limitation
- current conclusion only covers local smoke inference scope for `facebook/musicgen-small`
- it does not yet cover medium / melody / larger models
- it does not yet establish a formal benchmark methodology
- run_001 vs run_002 elapsed time should not be treated as a strict runtime comparison because first-run load / warmup effects are not isolated

## Root Cause / Failure Notes
- earlier failure was related to libstdc++ / CXXABI mismatch triggered during av import
- current environment no longer reproduces that import failure under the active conda env and current library path
- this means the environment issue is currently mitigated for the validated local smoke path, but should still be documented as an environment-sensitive risk

## Next Step
1. sync README Verified Scope / Not Yet Verified / Next Hard Milestone with current Week06 evidence
2. optionally add a stable wrapper script such as `scripts/run_generator_audit.sh`
3. leave medium / melody models and richer evals to the next audit stage instead of over-expanding Week06
