"""
RunPod Serverless Handler — Sulphur Prompt Enhancer

A lightweight standalone endpoint that loads the Sulphur-2 prompt enhancer
GGUF model (Qwen3.5-based, 9B) and its mmproj vision projection from
RunPod Model Cache.

Input:
{
  "input": {
    "prompt": "a basketball player doing a cool maneuver",
    "image": "base64...",          // optional
    "system_prompt": "...",        // optional
    "max_tokens": 512,             // optional, default 512
    "temperature": 0.7,            // optional, default 0.7
    "top_p": 0.9,                  // optional, default 0.9
    "top_k": 40,                   // optional, default 40
    "repeat_penalty": 1.1,         // optional, default 1.1
    "n_gpu_layers": -1             // optional, default -1 (all layers on GPU)
  }
}

Output:
{
  "output": {
    "enhanced_prompt": "...",
    "input_prompt": "...",
    "image_used": true/false
  }
}
"""

import os
import re
import runpod
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_REPO_ID = "Floppyshy/sulphur-2-runpod"
MODEL_SUBDIR = "prompt_enhancer"
MODEL_FILE = "sulphur_prompt_enhancer-Q4_K_M-imatrix.gguf"
MMPROJ_FILE = "sulphur_prompt_enhancer-mmproj-BF16.gguf"

DEFAULT_SYSTEM_PROMPT = (
    "You are the Sulphur-2 Prompt Enhancer. Expand the user input into a rich, "
    "detailed video generation description. Output ONLY the finalized prompt paragraph string. "
    "No chat, no filler."
)

# ---------------------------------------------------------------------------
# Model discovery from RunPod Model Cache / HF cache
# ---------------------------------------------------------------------------
def _find_hf_cached_file(repo_id: str, filename: str) -> str | None:
    """Search HuggingFace hub cache for a specific file."""
    repo_dir_name = repo_id.replace("/", "--")
    cache_bases = [
        f"/runpod-volume/huggingface-cache/hub/models--{repo_dir_name}",
        os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_dir_name}"),
    ]

    for base in cache_bases:
        snapshots = os.path.join(base, "snapshots")
        if not os.path.isdir(snapshots):
            continue
        for snapshot in os.listdir(snapshots):
            candidate = os.path.join(snapshots, snapshot, filename)
            if os.path.isfile(candidate):
                return candidate
    return None


def _resolve_model_path() -> tuple[str, str | None]:
    """Return (model_path, mmproj_path) from env, cache, or bail."""
    model_path = os.environ.get("MODEL_PATH")
    mmproj_path = os.environ.get("MMPROJ_PATH")

    if model_path and os.path.isfile(model_path):
        print(f"[enhancer] Using MODEL_PATH from env: {model_path}")
    else:
        model_path = _find_hf_cached_file(HF_REPO_ID, f"{MODEL_SUBDIR}/{MODEL_FILE}")
        if not model_path:
            raise FileNotFoundError(
                f"Model file not found: {MODEL_SUBDIR}/{MODEL_FILE} "
                f"in HF cache for {HF_REPO_ID}. "
                f"Ensure RunPod Model Cache is configured or set MODEL_PATH."
            )
        print(f"[enhancer] Found model in HF cache: {model_path}")

    if mmproj_path and os.path.isfile(mmproj_path):
        print(f"[enhancer] Using MMPROJ_PATH from env: {mmproj_path}")
    else:
        mmproj_path = _find_hf_cached_file(HF_REPO_ID, f"{MODEL_SUBDIR}/{MMPROJ_FILE}")
        if mmproj_path:
            print(f"[enhancer] Found mmproj in HF cache: {mmproj_path}")
        else:
            print("[enhancer] mmproj not found in cache; vision will be disabled.")
            mmproj_path = None

    return model_path, mmproj_path


# ---------------------------------------------------------------------------
# Globals (loaded once at cold start)
# ---------------------------------------------------------------------------
_llm = None
_model_path = None
_mmproj_path = None


def _load_llm(n_gpu_layers=-1):
    """Load the LLM once. Uses vision handler if mmproj is available."""
    global _llm, _model_path, _mmproj_path
    if _llm is not None:
        return _llm

    _model_path, _mmproj_path = _resolve_model_path()

    if _mmproj_path and os.path.exists(_mmproj_path):
        print("[enhancer] Loading vision-capable LLM + mmproj...")
        chat_handler = Qwen25VLChatHandler(
            clip_model_path=_mmproj_path,
            verbose=False,
        )
        _llm = Llama(
            model_path=_model_path,
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print("[enhancer] Vision LLM loaded.")
    else:
        print("[enhancer] Loading text-only LLM (mmproj not found)...")
        _llm = Llama(
            model_path=_model_path,
            chat_format="chatml",
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print("[enhancer] Text-only LLM loaded.")

    return _llm


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> tags if present."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    return text


def _build_data_uri(image_b64: str) -> str:
    """Validate and return a proper data URI from base64 input."""
    clean_b64 = image_b64.strip()
    if clean_b64.startswith("data:"):
        return clean_b64
    return f"data:image/png;base64,{clean_b64}"


def enhance_prompt(prompt: str, image_b64: str = None, options: dict = None) -> dict:
    """Run the prompt enhancer and return the enhanced text."""
    options = options or {}
    system_prompt = options.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    max_tokens = int(options.get("max_tokens", 512))
    temperature = float(options.get("temperature", 0.7))
    top_p = float(options.get("top_p", 0.9))
    top_k = int(options.get("top_k", 40))
    repeat_penalty = float(options.get("repeat_penalty", 1.1))
    n_gpu_layers = int(options.get("n_gpu_layers", -1))

    llm = _load_llm(n_gpu_layers)

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    if image_b64:
        data_uri = _build_data_uri(image_b64)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt.strip()})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
    )

    raw_text = response["choices"][0]["message"]["content"]
    enhanced = _strip_thinking_tags(raw_text)

    return {
        "enhanced_prompt": enhanced,
        "raw_response": raw_text,
        "input_prompt": prompt,
        "image_used": bool(image_b64),
    }


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {"error": "Missing required field: 'prompt'"}

    image_b64 = job_input.get("image") or None

    # Pass through any extra options
    options = {
        k: v
        for k, v in job_input.items()
        if k not in ("prompt", "image")
    }

    try:
        result = enhance_prompt(prompt, image_b64=image_b64, options=options)
        return {"output": result}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Pre-load model at import time so cold-start is just inference
# ---------------------------------------------------------------------------
print("[enhancer] Cold start — locating model in cache/VRAM...")
_load_llm()

runpod.serverless.start({"handler": handler})
