# =============================================================================
# Sulphur Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Standalone endpoint for prompt enhancement.
# Loads the Qwen3.5-based 9B GGUF model + mmproj vision projection from
# RunPod Model Cache (HF repo: Floppyshy/prompt-enhancer).
#
# Deploy to RunPod Serverless with the image.
# =============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_PREFER_BINARY=1

# ---------------------------------------------------------------------------
# System deps + Python 3 (default 3.10 in Ubuntu 22.04)
# ---------------------------------------------------------------------------
RUN apt-get update -y --fix-missing \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        libssl-dev \
        git \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python deps
# ---------------------------------------------------------------------------
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /requirements.txt

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
WORKDIR /app
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
