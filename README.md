# Prompt Enhancer — RunPod Serverless

A generic serverless endpoint for prompt enhancement. Takes a text prompt
(and optionally an image) and returns an enhanced, detailed prompt. The
system prompt is fully configurable — use any instruct/chat GGUF model.

## Architecture

- **Inference:** llama.cpp `llama-server` binary (compiled from source with CUDA in the Docker build)
- **Model:** Any GGUF model (text-only or with mmproj vision projection)
- **GPU target:** RTX 4090 (24 GB) recommended for 9B+ models
- **Models:** Discovered from RunPod Model Cache or specified via env vars

The handler spawns `llama-server` as a subprocess and communicates via its
OpenAI-compatible HTTP API (`/v1/chat/completions`). No `llama-cpp-python`
dependency.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: compiles llama.cpp with CUDA, then installs Python deps |
| `handler.py` | RunPod serverless handler (text + vision + Fernet encryption) |
| `requirements.txt` | Python dependencies (`runpod`, `cryptography`) |
| `.github/workflows/docker-build.yml` | CI — validates Dockerfile builds on push/PR |

## Quick Start

### 1. Build the image

```bash
docker build -t prompt-enhancer:latest .
```

### 2. Configure

Set these env vars on your RunPod serverless endpoint:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SYSTEM_PROMPT` | no | generic enhancer prompt | Default system prompt for all requests |
| `MODEL_PATH` | no* | auto-discovered | Path to GGUF model file |
| `MMPROJ_PATH` | no | auto-discovered | Path to mmproj for vision (optional) |
| `HF_REPO_ID` | no | `Floppyshy/prompt-enhancer` | HF repo for model cache discovery |
| `MODEL_FILE` | no | `sulphur_prompt_enhancer-Q4_K_M-imatrix.gguf` | Model filename in cache |
| `MMPROJ_FILE` | no | `sulphur_prompt_enhancer-mmproj-BF16.gguf` | mmproj filename in cache |
| `ENCRYPTION_KEY` | no | — | Fernet key for encrypt/decrypt |
| `LLAMA_SERVER_PORT` | no | `8081` | Internal port for llama-server |

\* Required if models aren't in RunPod Model Cache.

### 3. Deploy to RunPod Serverless

1. Push the Docker image to a registry (Docker Hub, GHCR, etc.)
2. In RunPod Console → Serverless → New Endpoint:
   - **Container Image:** your registry image URL
   - **GPU:** RTX 4090 or better (for 9B+ models)
   - **Workers:** configure min/max as needed
   - **Environment Variables:** set as needed (see above)
3. Optionally configure **Model Cache** to pull models from a HuggingFace repo

## Handler API

### Request — text only

```json
{
  "input": {
    "prompt": "a basketball player doing a cool maneuver"
  }
}
```

### Request — with custom system prompt

```json
{
  "input": {
    "prompt": "a cozy cabin in the woods",
    "system_prompt": "You are a Stable Diffusion prompt enhancer. Add camera details, lighting, and artistic style. Output ONLY the prompt."
  }
}
```

### Request — text + image

```json
{
  "input": {
    "prompt": "a basketball player doing a cool maneuver",
    "image": "iVBORw0KGgo..."
  }
}
```

The `image` field can be a raw base64 string or a full data URI (`data:image/png;base64,...`).

### Encrypted request (Fernet)

Set `ENCRYPTION_KEY` env var on the endpoint. Generate one with:

```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

```json
{
  "input": {
    "encrypted_prompt": "gAAAAAB...",
    "encrypt_output": true
  }
}
```

### Optional parameters

| Param | Default | Description |
|-------|---------|-------------|
| `system_prompt` | `$SYSTEM_PROMPT` env var | Override the system instruction per-request |
| `max_tokens` | 512 | Max generation length |
| `temperature` | 0.7 | Sampling temperature |
| `top_p` | 0.9 | Nucleus sampling |
| `top_k` | 40 | Top-k sampling |
| `repeat_penalty` | 1.1 | Repetition penalty |
| `n_gpu_layers` | -1 | GPU layers (-1 = all) |
| `encrypt_output` | false | Encrypt the response fields |

### Response

```json
{
  "output": {
    "enhanced_prompt": "A professional basketball player executes a stunning crossover dribble on a sunlit outdoor court...",
    "raw_response": "A professional basketball player...",
    "input_prompt": "a basketball player doing a cool maneuver",
    "image_used": false
  }
}
```

### Encrypted response

```json
{
  "output": {
    "enhanced_prompt": "gAAAAAB...",
    "raw_response": "gAAAAAB...",
    "input_prompt": "a basketball player doing a cool maneuver",
    "image_used": false,
    "encrypted": true
  }
}
```

## How It Works

1. Container starts → handler.py imports → waits for first request
2. On first request, handler spawns `llama-server` as a subprocess with the GGUF model
3. Handler polls llama-server's `/health` endpoint until ready
4. Each RunPod job is translated to an OpenAI-format chat completion request
5. Request is POSTed to llama-server's `/v1/chat/completions`
6. Response is cleaned (thinking tags stripped), optionally encrypted, and returned

llama-server stays loaded between requests on a warm worker. Only the first
invocation pays the model-load cost.

## Using Your Own Model

Set these env vars to point to any GGUF model:

```bash
MODEL_PATH=/runpod-volume/models/my-model.gguf      # or let it be discovered from cache
HF_REPO_ID=my-org/my-model-repo
MODEL_FILE=my-model-Q4_K_M.gguf
MMPROJ_FILE=my-model-mmproj.gguf                     # optional, for vision
SYSTEM_PROMPT=You are a prompt enhancer for flux...
```

The handler will use your model and system prompt for all requests. Per-request
`system_prompt` overrides the default.

## Notes

- llama.cpp is compiled from source with CUDA in the Docker build stage — no pre-built binary dependency.
- If the mmproj file is missing, vision mode is unavailable and the handler falls back to text-only.
- Thinking tags (`<think>...</think>`) are automatically stripped from the output.
- The GitHub Action validates that the Dockerfile builds on every push to `main`.
