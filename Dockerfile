# =============================================================================
# Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Uses llama.cpp's llama-server binary (from the official llama.cpp CUDA image)
# for inference instead of llama-cpp-python.
#
# Deploy to RunPod Serverless with the image.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Extract llama-server from the official llama.cpp CUDA image
# ---------------------------------------------------------------------------
FROM ghcr.io/ggml-org/llama.cpp:server-cuda AS llama

# ---------------------------------------------------------------------------
# Stage 2: Python runtime with CUDA (Ubuntu 22.04 — no PEP 668)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update -y --fix-missing \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — just runpod + cryptography
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /requirements.txt

# Copy llama-server binary and shared libraries from the official image
COPY --from=llama /app/llama-server /usr/local/bin/llama-server
COPY --from=llama /app/*.so* /usr/local/lib/

# Update linker cache so the binary finds its libs
RUN ldconfig

# Sanity check: binary exists, libs linked (--version may need GPU; don't fail)
RUN llama-server --version 2>&1 || true

# App
WORKDIR /app
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
