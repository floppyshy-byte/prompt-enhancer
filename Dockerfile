# =============================================================================
# Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Uses llama.cpp's llama-server binary (pre-built with CUDA from ai-dock) for
# inference instead of llama-cpp-python.
#
# Deploy to RunPod Serverless with the image.
# =============================================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps — lightweight, no build tools needed
RUN apt-get update -y --fix-missing \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps — just runpod + cryptography
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /requirements.txt

# ---------------------------------------------------------------------------
# llama.cpp pre-built binary (ai-dock/llama.cpp-cuda)
# CUDA 12.8, supports SM 7.5–12.0 (includes RTX 4090 / SM 8.9)
# ---------------------------------------------------------------------------
ARG LLAMA_CPP_TAG=b9484
RUN wget -q "https://github.com/ai-dock/llama.cpp-cuda/releases/download/${LLAMA_CPP_TAG}/llama.cpp-${LLAMA_CPP_TAG}-cuda-12.8-amd64.tar.gz" \
        -O /tmp/llama.cpp.tar.gz \
    && tar -xzf /tmp/llama.cpp.tar.gz -C /tmp \
    && cp /tmp/cuda-12.8/bin/llama-server /usr/local/bin/llama-server \
    && rm -rf /tmp/llama.cpp.tar.gz /tmp/cuda-12.8 \
    && chmod +x /usr/local/bin/llama-server \
    && llama-server --version

# App
WORKDIR /app
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
