# =============================================================================
# Sulphur Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Lightweight standalone endpoint for prompt enhancement.
# Loads the Qwen3.5-based 9B GGUF model + mmproj vision projection.
#
# Models (~6.5 GB total) are downloaded at build time from HuggingFace.
# Build with:
#   docker build -t prompt-enhancer .
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
# Download models at build time
# ---------------------------------------------------------------------------
ARG HF_REPO=Floppyshy/sulphur-2-runpod
ARG MODEL_SUBDIR=prompt_enhancer
ARG MODEL_FILE=sulphur_prompt_enhancer-Q4_K_M-imatrix.gguf
ARG MMPROJ_FILE=sulphur_prompt_enhancer-mmproj-BF16.gguf

RUN mkdir -p /models/prompt_enhancer \
    && python3 -c "
import os
from huggingface_hub import hf_hub_download
repo = os.environ['HF_REPO']
path = os.environ['MODEL_SUBDIR']
files = [os.environ['MODEL_FILE'], os.environ['MMPROJ_FILE']]
for f in files:
    print(f'Downloading {f}...')
    hf_hub_download(repo_id=repo, filename=f'{path}/{f}', local_dir='/models', local_dir_use_symlinks=False)
    print(f'Done: {f}')
" \
    && ls -lah /models/prompt_enhancer/

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
WORKDIR /app
COPY handler.py /app/handler.py

ENV MODEL_PATH=/models/prompt_enhancer/${MODEL_FILE}
ENV MMPROJ_PATH=/models/prompt_enhancer/${MMPROJ_FILE}

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
