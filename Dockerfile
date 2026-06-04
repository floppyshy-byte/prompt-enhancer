# =============================================================================
# Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Built directly on the official llama.cpp CUDA server image. No file copying,
# no GLIBC mismatches — just add Python and the handler on top.
# =============================================================================

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install Python on top of the official llama.cpp image
RUN apt-get update -y --fix-missing \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /requirements.txt

# Symlink so handler finds llama-server on PATH
RUN ln -s /app/llama-server /usr/local/bin/llama-server

# Sanity check
RUN llama-server --version

# App
WORKDIR /app
COPY handler.py /app/handler.py

# Reset inherited ENTRYPOINT so our CMD runs
ENTRYPOINT []
CMD ["python3", "-u", "handler.py"]
