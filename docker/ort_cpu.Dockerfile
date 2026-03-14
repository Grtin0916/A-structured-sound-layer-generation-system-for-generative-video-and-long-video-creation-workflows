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
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

RUN pip install \
 numpy \
 pyyaml \
 soundfile \
 librosa \
 tqdm \
 tensorboard \
 onnx \
 onnxruntime \
 onnxscript  \
 torch \
 torchaudio \
 torchcodec

COPY . /workspace
ENV PYTHONPATH=/workspace

CMD ["python", "src/infer/infer_ort.py", \
     "--config", "configs/train/baseline.yaml", \
     "--onnx", "artifacts/onnx/baseline_v1.onnx", \
     "--provider", "CPUExecutionProvider", \
     "--input-mode", "random"]