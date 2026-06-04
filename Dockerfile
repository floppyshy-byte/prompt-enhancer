# =============================================================================
# Prompt Enhancer — RunPod Serverless Worker
# =============================================================================
# Uses llama.cpp's llama-server binary (built from source with CUDA) for
# inference instead of llama-cpp-python.
#
# Deploy to RunPod Serverless with the image.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build llama.cpp with CUDA support
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS llama-builder

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ccache \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin to a specific release tag for reproducibility
ARG LLAMA_CPP_VERSION=b9496
RUN git clone --depth 1 --branch ${LLAMA_CPP_VERSION} \
    https://github.com/ggml-org/llama.cpp.git /llama.cpp

WORKDIR /llama.cpp
RUN --mount=type=cache,target=/root/.ccache \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && cmake --build build --config Release -j$(nproc) --target llama-server \
    && ccache -s

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps — much leaner: no cmake, ninja, or build-essential needed
RUN apt-get update -y --fix-missing \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps — just runpod + cryptography (no llama-cpp-python)
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /requirements.txt

# Copy llama-server binary from build stage
COPY --from=llama-builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Verify the binary works
RUN llama-server --version

# App
WORKDIR /app
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
