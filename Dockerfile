# =============================================================================
# Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Uses llama.cpp's llama-server binary (from the official llama.cpp CUDA image)
# for inference instead of llama-cpp-python.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Reference the official llama.cpp CUDA image
# ---------------------------------------------------------------------------
FROM ghcr.io/ggml-org/llama.cpp:server-cuda AS llama

# ---------------------------------------------------------------------------
# Stage 2: Python runtime with CUDA (Ubuntu 24.04 — matches server-cuda GLIBC)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps (libgomp1 required by llama-server)
RUN apt-get update -y --fix-missing \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — --break-system-packages needed on Ubuntu 24.04 (PEP 668)
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel --break-system-packages \
    && python3 -m pip install -r /requirements.txt --break-system-packages

# Extract llama-server binary and shared libraries from the official image
# (verified: the server-cuda image puts everything in /app/)
COPY --from=llama /app/llama-server /usr/local/bin/llama-server
COPY --from=llama /app/*.so* /usr/local/lib/

# Rebuild runtime dynamic linker cache
RUN ldconfig

# Sanity check: GLIBC 2.38 + libgomp1 are now satisfied
RUN llama-server --version

# App
WORKDIR /app
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
