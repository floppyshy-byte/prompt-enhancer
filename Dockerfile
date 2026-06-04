# =============================================================================
# Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Uses the official llama.cpp server-cuda Docker image for the llama-server
# binary instead of compiling from source or using llama-cpp-python.
#
# Deploy to RunPod Serverless with the image.
# =============================================================================

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Add Python on top of the official llama.cpp CUDA image
RUN apt-get update -y --fix-missing \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Python deps — just runpod + cryptography
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /requirements.txt

# App
WORKDIR /app
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
# Reset ENTRYPOINT inherited from the server-cuda image (which defaults to llama-server)
ENTRYPOINT []
CMD ["python3", "-u", "handler.py"]
