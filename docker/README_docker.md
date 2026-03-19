# Docker CPU reproduction (Week 02)

## Purpose
This image is used to reproduce the current ONNX Runtime CPU inference path in a standardized container environment.

## Dockerfile location
docker/ort_cpu.Dockerfile

## Build context
Build from the repository root so that `src/`, `configs/`, and `artifacts/onnx/` are available to the image.

## Build

Without proxy:
    docker build -f docker/ort_cpu.Dockerfile -t audio-ort-cpu:week02 .

With proxy:
    docker build \
      --build-arg http_proxy=http://host.docker.internal:7890 \
      --build-arg https_proxy=http://host.docker.internal:7890 \
      --build-arg HTTP_PROXY=http://host.docker.internal:7890 \
      --build-arg HTTPS_PROXY=http://host.docker.internal:7890 \
      -f docker/ort_cpu.Dockerfile \
      -t audio-ort-cpu:week02 .

## Smoke run
The default CMD runs ORT inference with random input:
    docker run --rm audio-ort-cpu:week02

## Run benchmark inside container
Mount the repo so that output files are written back to the host:
    docker run --rm \
      -v "$PWD":/workspace \
      -w /workspace \
      audio-ort-cpu:week02 \
      python src/infer/bench_ort.py \
        --config configs/train/baseline.yaml \
        --onnx artifacts/onnx/baseline_v1.onnx \
        --provider CPUExecutionProvider \
        --input-mode random \
        --input-shape 1,1,64,313 \
        --warmup 20 \
        --runs 200 \
        --csv artifacts/logs/bench_ort_docker.csv

## Expected behavior
- Image builds successfully.
- Default container command completes ORT inference without missing dependency errors.
- Benchmark command writes `artifacts/logs/bench_ort_docker.csv` to the host workspace.

## Known boundary
The current ONNX path is validated on a fixed input shape:
    (1, 1, 64, 313)

Dynamic-shape support is not yet the goal of this step.

## Common pitfalls
1. Building from the wrong directory causes missing `src/` or `configs/`.
2. Over-aggressive `.dockerignore` rules can exclude required runtime files.
3. If network access is restricted, build may require proxy args.
