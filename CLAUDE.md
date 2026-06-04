# CLAUDE.md — Prompt Enhancer

## Project Overview

A RunPod serverless endpoint for prompt enhancement. Takes a text prompt (and optionally an image) and uses a local LLM via llama.cpp's `llama-server` to expand it into a rich, detailed description. Built for image-generation workflows (Stable Diffusion, Flux, etc.) but generic enough for any prompt-expansion use case.

## Architecture

- **Runtime:** Python handler inside a Docker container deployed to RunPod Serverless
- **Inference:** llama.cpp `llama-server` binary spawned as a subprocess on first request
- **Protocol:** OpenAI-compatible HTTP API (`/v1/chat/completions`) — no Python SDK, uses `urllib.request`
- **Model format:** GGUF (any instruct/chat model, optionally with mmproj for vision)
- **GPU target:** RTX 4090 (24 GB) recommended for 9B+ parameter models

The handler spawns `llama-server`, polls `/health` until ready, then proxies each RunPod job as a chat completion request. The server stays warm between requests on the same worker.

## Key Files

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless handler — model discovery, server lifecycle, inference, encryption |
| `Dockerfile` | Multi-stage build: compiles llama.cpp with CUDA, installs Python deps |
| `requirements.txt` | `runpod`, `cryptography` |
| `.github/workflows/docker-build.yml` | CI — validates Dockerfile builds on push/PR |
| `test_handler.py` | Unit tests for handler logic (mocked HTTP, no real LLM) |

## Environment Variables

Set on the RunPod endpoint. The handler auto-discovers models from RunPod Model Cache if paths are not specified.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SYSTEM_PROMPT` | no | generic enhancer prompt | Default system prompt for all requests |
| `MODEL_PATH` | no* | auto-discovered | Exact path to GGUF model file |
| `MMPROJ_PATH` | no | auto-discovered | Path to mmproj for vision (optional) |
| `HF_REPO_ID` | no | — (scans all cache) | HF repo to search in cache |
| `MODEL_FILE` | no | — (auto-discovered) | Specific model filename in cache |
| `MMPROJ_FILE` | no | — (auto-discovered) | Specific mmproj filename in cache |
| `ENCRYPTION_KEY` | no | — | Fernet key for encrypt/decrypt |
| `LLAMA_SERVER_PORT` | no | `8081` | Internal port for llama-server |
| `MAX_TOKENS` | no | `5000` | Default max generation length (tokens) |

\* Required only if no models are in RunPod Model Cache.

## Model Auto-Discovery

When model paths are not specified, the handler scans `/runpod-volume/models/` (or `HF_REPO_ID` subdir) for `.gguf` files:

| Files found | Behavior |
|---|---|
| **0** | Error — no model available |
| **1** | Text-only mode |
| **2** | Vision mode — larger file = model, smaller = mmproj |
| **3+** | Error — ambiguous, set `MODEL_FILE`/`MMPROJ_FILE` |

## Request/Response API

### Request (text only)

```json
{
  "input": {
    "prompt": "a basketball player doing a cool maneuver"
  }
}
```

### Request (with options)

```json
{
  "input": {
    "prompt": "a cozy cabin in the woods",
    "system_prompt": "You are a Stable Diffusion prompt enhancer...",
    "max_tokens": 3000,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repeat_penalty": 1.15,
    "n_gpu_layers": -1
  }
}
```

### Request (with image)

```json
{
  "input": {
    "prompt": "describe this scene in detail",
    "image": "iVBORw0KGgo..."
  }
}
```

The `image` field accepts raw base64 or a full data URI.

### Optional Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `system_prompt` | `$SYSTEM_PROMPT` env var | Override system instruction per-request |
| `max_tokens` | `$MAX_TOKENS` env var (default `5000`) | Max generation length |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.9` | Nucleus sampling |
| `top_k` | `40` | Top-k sampling |
| `repeat_penalty` | `1.1` | Repetition penalty |
| `n_gpu_layers` | `-1` | GPU layers (-1 = all) |
| `encrypt_output` | `false` | Encrypt response fields |

### Response

```json
{
  "output": {
    "enhanced_prompt": "A professional basketball player...",
    "raw_response": "A professional basketball player...",
    "thinking": "The user wants a sports scene, so I'll add dynamic lighting...",
    "input_prompt": "a basketball player doing a cool maneuver",
    "image_used": false
  }
}
```

## Inference Flow (`handler.py`)

1. `enhance_prompt(prompt, image_b64, options)` called per request
2. Resolve config: `system_prompt`, `max_tokens`, `temperature`, etc.
   - `max_tokens` priority: per-request `options` > `$MAX_TOKENS` env var > default `5000`
3. `_start_llama_server(n_gpu_layers)` — spawns `llama-server` if not already running
4. Build OpenAI chat-format messages (system + user, optionally with image content)
5. POST to `llama-server:PORT/v1/chat/completions` via `urllib.request`
6. Parse response: extract `content`, `reasoning_content` from `response["choices"][0]["message"]`
7. Extract `thinking`: `reasoning_content` takes precedence, fall back to `<think>...</think>` tags in content via `_extract_thinking_tags()`
8. `raw_response` combines reasoning + content (full text)
9. `enhanced_prompt` = content with `<think>` tags stripped via `_strip_thinking_tags()` (reasoning never included)
10. Return `enhanced_prompt`, `raw_response`, `thinking`, `input_prompt`, `image_used`

## Token Limits

- `max_tokens` (default `5000`) caps the **output** length only
- If the model hits `max_tokens` before naturally completing, `finish_reason` will be `"length"` and the response is truncated mid-generation
- The input/prompt tokens do NOT count against `max_tokens`
- No explicit check is done on `finish_reason`; the response is returned as-is even if truncated

## Encryption

Optional Fernet encryption for both requests and responses. Set `ENCRYPTION_KEY` env var.

Generate a key:
```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Encrypted request:
```json
{
  "input": {
    "encrypted_prompt": "gAAAAAB...",
    "encrypt_output": true
  }
}
```

## Testing

Run unit tests (no LLM required — HTTP is mocked):

```bash
python -m pytest test_handler.py -v
```

## Development Notes

- llama.cpp is compiled from source in the Docker build — no pre-built binary
- CUDA is enabled during compilation (`LLAMA_CUDA=1`)
- The handler uses `urllib.request` directly, not the OpenAI SDK — response is a plain dict: `response["choices"][0]`
- `choices` always contains exactly 1 choice because the request never sets `n > 1`
- `reasoning_content` (used by Qwen3 models) is extracted into the `thinking` field and excluded from `enhanced_prompt`
- Thinking from `<think>...</think>` tags is similarly extracted and excluded from `enhanced_prompt`
- Debug logging: a single `print()` logs `finish_reason`, content length, and reasoning length per request
