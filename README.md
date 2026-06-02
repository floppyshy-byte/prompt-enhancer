# Sulphur Prompt Enhancer — RunPod Serverless

Standalone serverless endpoint for the Sulphur-2 prompt enhancer.  
Takes a text prompt (and optionally an image) and returns an enhanced, detailed prompt suitable for video generation.

## Architecture

- **Base model:** Qwen3.5-based 9B parameter VLM (GGUF Q4_K_M)
- **Vision projection:** mmproj BF16 (922 MB)
- **Inference:** `llama-cpp-python` with CUDA offloading
- **GPU target:** RTX 4090 (24 GB) — loads both models comfortably

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | CUDA 12.4 runtime image, downloads models at build time |
| `handler.py` | RunPod serverless handler (text + vision paths) |
| `requirements.txt` | Python dependencies |

## Build

```bash
docker build -t prompt-enhancer:latest .
```

Models (~6.5 GB) are baked into the image at build time so cold start is just model loading into VRAM (~5–10 s).

## API

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

## Deploy on RunPod Serverless

1. Push the Docker image to a registry (Docker Hub, GHCR, etc.).
2. In RunPod Console → Serverless → New Endpoint:
   - **Container Image:** your registry image URL
   - **GPU:** RTX 4090 or better
   - **Workers:** configure min/max as needed
   - **Environment Variables** (optional):
     - `MODEL_PATH` — override default model path
     - `MMPROJ_PATH` — override default mmproj path

## Notes

- The handler pre-loads both the text-only and vision models at import time.  
  First invocation pays the VRAM load cost; subsequent invocations on a warm worker are fast.
- If the mmproj file is missing, vision mode is gracefully unavailable and the handler falls back to text-only enhancement.
- Thinking tags (`<think>...</think>`) are automatically stripped from the output.
