#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

VARIANT_ORDER = {
    "naive": 0,
    "naive_rich": 1,
    "dss_global": 2,
    "dss_event_timeline": 3,
    "dss_layer_avoid": 4,
}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--queue-jsonl", default="artifacts/model_runs/w18_dss_ablation/generation_queue_duration_prompt_aligned_20260706.jsonl")
    ap.add_argument("--out-runner", required=True)
    args = ap.parse_args()

    queue = Path(args.queue_jsonl)
    jobs = [json.loads(x) for x in queue.read_text(encoding="utf-8").splitlines() if x.strip()]
    selected = [j for j in jobs if j["case_id"] == args.case_id]
    selected = sorted(selected, key=lambda j: VARIANT_ORDER.get(j["variant"], 99))

    if len(selected) != 5:
        raise SystemExit(f"expected 5 jobs for {args.case_id}, got {len(selected)}")

    out = Path(args.out_runner)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'MAINBASE="${MAINBASE:-$HOME/work/audio_engineering_repo_skeleton_v1}"',
        'MMAUDIO_ROOT="${MMAUDIO_ROOT:-/home/GRT/work/_external/MMAudio}"',
        'HF_HOME="${HF_HOME:-$MMAUDIO_ROOT/.hf_cache}"',
        'HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"',
        "export MAINBASE MMAUDIO_ROOT HF_HOME HUGGINGFACE_HUB_CACHE",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
        "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy",
        f'echo "CASE_ID={args.case_id}"',
        "",
    ]

    for j in selected:
        duration = int(round(float(j["duration_sec"])))
        job_dir = Path(j["expected_output_flac"]).parent
        raw_out = job_dir / "mmaudio_small_44k_raw_output"

        lines += [
            f"echo '### RUN {j['job_id']} duration={duration}s variant={j['variant']}'",
            f"mkdir -p \"$MAINBASE/{job_dir}\" \"$MAINBASE/{raw_out}\"",
            'cd "$MMAUDIO_ROOT"',
            "test -f weights/mmaudio_small_44k.pth",
            "test -f ext_weights/synchformer_state_dict.pth",
            "test -f ext_weights/v1-44.pth",
            f"if [ -f \"$MAINBASE/{j['expected_output_flac']}\" ] || [ -f \"$MAINBASE/{j['expected_output_wav']}\" ]; then echo 'SKIP existing output {j['job_id']}'; else",
            (
                "  python demo.py "
                f"--variant small_44k "
                f"--duration={duration} "
                f"--video=\"$MAINBASE/{j['video_path']}\" "
                f"--prompt {json.dumps(j['prompt'])} "
                f"--output=\"$MAINBASE/{raw_out}\" "
                "--seed=42"
            ),
            f"  NEW_AUDIO=$(find \"$MAINBASE/{raw_out}\" -type f \\( -name '*.flac' -o -name '*.wav' \\) -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)",
            '  if [ -z "$NEW_AUDIO" ]; then echo "NO_MMAUDIO_OUTPUT_FOUND"; exit 2; fi',
            f"  cp \"$NEW_AUDIO\" \"$MAINBASE/{j['expected_output_flac']}\"",
            f"  if command -v ffmpeg >/dev/null 2>&1; then ffmpeg -y -i \"$MAINBASE/{j['expected_output_flac']}\" \"$MAINBASE/{j['expected_output_wav']}\" >/dev/null 2>&1 || true; fi",
            "fi",
            f"ls -lh \"$MAINBASE/{j['expected_output_flac']}\" \"$MAINBASE/{j['expected_output_wav']}\" 2>/dev/null || true",
            "",
        ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.chmod(0o755)
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
