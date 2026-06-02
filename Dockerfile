# =============================================================================
# Sulphur Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Standalone endpoint for prompt enhancement.
# Loads the Qwen3.5-based 9B GGUF model + mmproj vision projection from
# RunPod Model Cache (HF repo: Floppyshy/sulphur-2-runpod).
#
# Deploy to RunPod Serverless with the image.
# =============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# System deps + Python 3.11
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    build-essential \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ---------------------------------------------------------------------------
# Python deps
# ---------------------------------------------------------------------------
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
WORKDIR /app
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
