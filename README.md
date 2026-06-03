# Sulphur Prompt Enhancer — RunPod Serverless

Standalone serverless endpoint for the Sulphur-2 prompt enhancer.
Takes a text prompt (and optionally an image) and returns an enhanced, detailed prompt suitable for video generation.

## Architecture

- **Base model:** Qwen3.5-based 9B parameter VLM (GGUF Q4_K_M)
- **Vision projection:** mmproj BF16 (922 MB)
- **Inference:** `llama-cpp-python` with CUDA offloading
- **GPU target:** RTX 4090 (24 GB) — loads both models comfortably
- **Models:** Loaded from RunPod Model Cache (HF repo `Floppyshy/prompt-enhancer`)

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | CUDA 12.4 runtime image (no build-time model downloads) |
| `handler.py` | RunPod serverless handler (text + vision + encryption) |
| `requirements.txt` | Python dependencies for the handler |
| `server.py` | Optional FastAPI proxy server with polling + SQLite history |
| `requirements-server.txt` | Python dependencies for the proxy server |
| `.github/workflows/docker-build.yml` | CI — validates Dockerfile builds on push/PR |

## Handler Build

```bash
docker build -t prompt-enhancer:latest .
```

Models are **not** baked into the image. They are discovered at runtime from RunPod Model Cache or from env vars (`MODEL_PATH`, `MMPROJ_PATH`).

## Handler API

### Request — text only

```json
{
  "input": {
    "prompt": "a basketball player doing a cool maneuver"
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

The `image` field can be:
- Raw base64 string
- Full data URI (`data:image/png;base64,...`)

### Encrypted request (Fernet)

Set `ENCRYPTION_KEY` env var on the endpoint (a Fernet key).

```json
{
  "input": {
    "encrypted_prompt": "gAAAAAB...",
    "encrypt_output": true
  }
}
```

The handler decrypts the prompt, processes it, and encrypts the output if requested.

### Optional parameters

| Param | Default | Description |
|-------|---------|-------------|
| `system_prompt` | Sulphur-2 enhancer system prompt | Override the system instruction |
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

## Deploy on RunPod Serverless

1. Build and push the Docker image to a registry (Docker Hub, GHCR, etc.).
2. In RunPod Console → Serverless → New Endpoint:
   - **Container Image:** your registry image URL
   - **GPU:** RTX 4090 or better
   - **Workers:** configure min/max as needed
   - **Environment Variables**:
     - `ENCRYPTION_KEY` — Fernet key for encryption (optional)
     - `MODEL_PATH` — override default model path (optional)
     - `MMPROJ_PATH` — override default mmproj path (optional)
3. Configure **Model Cache** to pull from `Floppyshy/prompt-enhancer`.

## Proxy Server (`server.py`)

A FastAPI proxy that sits between your client and RunPod. It handles:

- **Encryption** — encrypts prompts before sending to RunPod
- **Server-side polling** — polls RunPod for completion so your client doesn't have to
- **Persistent history** — stores all requests/responses in SQLite (`history.db`)

### Setup

```bash
pip install -r requirements-server.txt
cp .env.example .env
# Edit .env with your RunPod credentials and optional Fernet key
python server.py
```

### Proxy API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/enhance` | POST | Submit a job. Returns `{job_id, status}` immediately |
| `/jobs/{job_id}` | GET | Get job status + result |
| `/history` | GET | List all jobs (newest first). Query: `?limit=50&offset=0` |
| `/history/{job_id}` | DELETE | Delete a job from local history |

### Example

```bash
curl -X POST http://localhost:8000/enhance \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat in a spacesuit"}'

# Returns: {"job_id": "...", "status": "queued"}

# Poll locally for result
curl http://localhost:8000/jobs/{job_id}
```

## Notes

- The handler pre-loads the model at import time. First invocation pays the VRAM load cost; subsequent invocations on a warm worker are fast.
- If the mmproj file is missing, vision mode is gracefully unavailable and the handler falls back to text-only enhancement.
- Thinking tags (`<think>...</think>`) are automatically stripped from the output.
- The GitHub Action validates that the Dockerfile builds cleanly on every push to `main`.
