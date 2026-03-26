FROM python:3.10-slim

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    bash \
    ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    pip install \
      fastapi \
      uvicorn \
      numpy \
      pyyaml \
      soundfile \
      librosa \
      tqdm \
      tensorboard \
      onnx \
      onnxruntime \
      onnxscript \
      torch \
      torchaudio \
      torchcodec

COPY configs /workspace/configs
COPY scripts /workspace/scripts
COPY src /workspace/src
COPY data /workspace/data
COPY results /workspace/results
COPY docs /workspace/docs
COPY README.md /workspace/README.md
COPY artifacts/experiments /workspace/artifacts/experiments
COPY artifacts/onnx /workspace/artifacts/onnx

RUN mkdir -p \
    /workspace/artifacts/logs \
    /workspace/docs/benchmarks

CMD ["bash", "scripts/run_demo.sh", "demo"]