"""
RunPod Serverless Handler — Generic Prompt Enhancer

A lightweight serverless endpoint that loads any GGUF model (with optional
mmproj vision projection) from RunPod Model Cache and uses llama.cpp's
llama-server binary for inference.

The system prompt is fully configurable — pass it per-request or set a
default via env var. Works with any instruct/chat model.

Uses llama.cpp's llama-server binary as a subprocess for inference
instead of llama-cpp-python.

Input (plain text):
{
  "input": {
    "prompt": "a basketball player doing a cool maneuver",
    "system_prompt": "...",        // optional, overrides default
    "image": "base64...",          // optional
    "max_tokens": 512,             // optional, default 512
    "temperature": 0.7,            // optional, default 0.7
    "top_p": 0.9,                  // optional, default 0.9
    "top_k": 40,                   // optional, default 40
    "repeat_penalty": 1.1,         // optional, default 1.1
    "n_gpu_layers": -1             // optional, default -1 (all layers on GPU)
  }
}

Input (encrypted payload):
{
  "input": {
    "encrypted_payload": "gAAAAAB..."  // Fernet-encrypted JSON
  }
}

Decrypting the payload yields the full input dict, e.g.:
{
  "prompt": "a basketball player...",
  "encrypt_output": true,
  "max_tokens": 512,
  ...
}

Output (plain):
{
  "output": {
    "enhanced_prompt": "...",
    "raw_response": "...",
    "thinking": "...",
    "input_prompt": "...",
    "image_used": true/false
  }
}

Output (encrypted):
{
  "output": {
    "encrypted_payload": "gAAAAAB...",  // full result JSON as one encrypted blob
    "encrypted": true
  }
}

Decrypting the payload yields the full result dict.
"""

import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
import urllib.error

import runpod
from cryptography.fernet import Fernet, InvalidToken

# ---------------------------------------------------------------------------
# Config — all model/discovery settings are overridable via env vars
# ---------------------------------------------------------------------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
MODEL_FILE = os.environ.get("MODEL_FILE", "")
MMPROJ_FILE = os.environ.get("MMPROJ_FILE", "")

LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8081"))
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"

DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a prompt enhancer. Expand the user input into a rich, detailed "
    "description. Output ONLY the finalized prompt. No chat, no filler.",
)

# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------
_fernet = None


def _get_fernet() -> Fernet | None:
    """Lazy-load Fernet from ENCRYPTION_KEY env var."""
    global _fernet
    if _fernet is not None:
        return _fernet

    key = os.environ.get("ENCRYPTION_KEY")
    if not key:
        return None

    # Fernet keys are 32 bytes urlsafe base64 encoded (43 chars + optional padding)
    # Accept raw base64 or already-urlsafe-base64
    key_b = key.encode()
    try:
        # If the user gave a standard base64 string, convert padding
        if len(key_b) == 44 and key_b.endswith(b"="):
            pass
        elif len(key_b) == 43:
            key_b += b"="
        _fernet = Fernet(key_b)
    except Exception as exc:
        raise ValueError(f"Invalid ENCRYPTION_KEY format: {exc}")

    return _fernet


def _decrypt_text(token: str) -> str:
    """Decrypt a Fernet token. Returns plain text."""
    f = _get_fernet()
    if f is None:
        raise ValueError("encrypted_prompt provided but ENCRYPTION_KEY is not set.")
    try:
        return f.decrypt(token.encode()).decode("utf-8")
    except InvalidToken:
        raise ValueError("Invalid encrypted_prompt token (bad key or corrupted data).")


def _encrypt_text(plaintext: str) -> str:
    """Encrypt plain text with Fernet. Returns base64 token string."""
    f = _get_fernet()
    if f is None:
        raise ValueError("encrypt_output requested but ENCRYPTION_KEY is not set.")
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def _encrypt_json(data: dict) -> str:
    """Encrypt a dict as a JSON string. Returns base64 token string."""
    return _encrypt_text(json.dumps(data, ensure_ascii=False))


def _decrypt_json(token: str) -> dict:
    """Decrypt a Fernet token and parse as JSON. Returns a dict."""
    plaintext = _decrypt_text(token)
    try:
        return json.loads(plaintext)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Decrypted payload is not valid JSON: {exc}")


# ---------------------------------------------------------------------------
# Model discovery from RunPod Model Cache / HF cache
# ---------------------------------------------------------------------------
def _get_cache_snapshot_dirs(repo_id: str) -> list[str]:
    """Return paths to snapshot directories in HF cache.

    If repo_id is given, searches only that repo.
    If empty, scans ALL repos under the cache hub for .gguf files.
    """
    cache_hubs = [
        "/runpod-volume/huggingface-cache/hub",
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    dirs = []

    for hub in cache_hubs:
        if not os.path.isdir(hub):
            continue

        if repo_id:
            # Specific repo — look for models--{repo_id}/snapshots/*
            repo_dir_name = repo_id.replace("/", "--").lower()
            model_dir = os.path.join(hub, f"models--{repo_dir_name}")
            snapshots = os.path.join(model_dir, "snapshots")
            if os.path.isdir(snapshots):
                for snapshot in os.listdir(snapshots):
                    snapshot_dir = os.path.join(snapshots, snapshot)
                    if os.path.isdir(snapshot_dir):
                        dirs.append(snapshot_dir)
        else:
            # No repo specified — scan all models--* directories
            try:
                for entry in os.listdir(hub):
                    if not entry.startswith("models--"):
                        continue
                    snapshots = os.path.join(hub, entry, "snapshots")
                    if not os.path.isdir(snapshots):
                        continue
                    for snapshot in os.listdir(snapshots):
                        snapshot_dir = os.path.join(snapshots, snapshot)
                        if os.path.isdir(snapshot_dir):
                            dirs.append(snapshot_dir)
            except OSError:
                pass

    return dirs


def _find_hf_cached_file(repo_id: str, filename: str) -> str | None:
    """Search HuggingFace hub cache for a specific file."""
    if not filename:
        return None
    for snapshot_dir in _get_cache_snapshot_dirs(repo_id):
        candidate = os.path.join(snapshot_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def _autodiscover_gguf_files(repo_id: str) -> list[tuple[int, str]]:
    """Find all .gguf files in cache, sorted by size (largest first)."""
    files: dict[str, int] = {}  # path -> size (dedupe across snapshots)
    for snapshot_dir in _get_cache_snapshot_dirs(repo_id):
        try:
            for f in os.listdir(snapshot_dir):
                if f.endswith(".gguf"):
                    path = os.path.join(snapshot_dir, f)
                    files[path] = os.path.getsize(path)
        except OSError:
            continue
    # Sort by size descending: (size, path)
    return sorted(
        [(size, path) for path, size in files.items()],
        reverse=True,
    )


def _resolve_model_path() -> tuple[str, str | None]:
    """Return (model_path, mmproj_path) from env, cache, or bail.

    Resolution order:
      1. MODEL_PATH / MMPROJ_PATH env vars (exact paths)
      2. MODEL_FILE / MMPROJ_FILE env vars (specific filenames in cache)
      3. Auto-discover all .gguf files in cache:
         - 0 files → error
         - 1 file  → text-only model (no mmproj)
         - 2 files → larger = model, smaller = mmproj (vision)
         - 3+ files → error (ambiguous)
    """
    model_path = os.environ.get("MODEL_PATH")
    mmproj_path = os.environ.get("MMPROJ_PATH")

    # --- resolve model ---
    if model_path and os.path.isfile(model_path):
        print(f"[enhancer] Using MODEL_PATH from env: {model_path}")
    else:
        model_path = _find_hf_cached_file(HF_REPO_ID, MODEL_FILE) if MODEL_FILE else None
        if model_path:
            print(f"[enhancer] Found model by name in cache: {model_path}")

    # --- resolve mmproj ---
    if mmproj_path and os.path.isfile(mmproj_path):
        print(f"[enhancer] Using MMPROJ_PATH from env: {mmproj_path}")
    else:
        mmproj_path = _find_hf_cached_file(HF_REPO_ID, MMPROJ_FILE) if MMPROJ_FILE else None
        if mmproj_path:
            print(f"[enhancer] Found mmproj by name in cache: {mmproj_path}")

    # --- auto-discover if either is still missing ---
    if not model_path or (not mmproj_path and model_path):
        gguf_files = _autodiscover_gguf_files(HF_REPO_ID)
        count = len(gguf_files)

        if count == 0:
            if not model_path:
                raise FileNotFoundError(
                    "No .gguf files found in cache. "
                    "Set MODEL_PATH, MODEL_FILE, or configure RunPod Model Cache."
                )

        elif count == 1:
            if not model_path:
                model_path = gguf_files[0][1]
                print(f"[enhancer] Auto-discovered model (sole .gguf, text-only): {model_path}")
            # mmproj stays None — single file means text-only

        elif count == 2:
            # Larger = model, smaller = mmproj
            larger = gguf_files[0][1]
            smaller = gguf_files[1][1]
            if not model_path:
                model_path = larger
                print(f"[enhancer] Auto-discovered model (larger .gguf): {model_path}")
            if model_path == larger and not mmproj_path:
                mmproj_path = smaller
                print(f"[enhancer] Auto-discovered mmproj (smaller .gguf): {mmproj_path}")
            elif model_path == smaller and not mmproj_path:
                mmproj_path = larger
                print(f"[enhancer] Auto-discovered mmproj (larger .gguf): {mmproj_path}")

        else:  # count >= 3
            if not model_path:
                # Can't determine anything — need explicit config
                files_list = "\n".join(f"  {size:>15d}  {path}" for size, path in gguf_files)
                raise FileNotFoundError(
                    f"Found {count} .gguf files in cache — don't know which is the model "
                    f"and which is the mmproj. Set MODEL_FILE and MMPROJ_FILE to be explicit.\n"
                    f"Files found:\n{files_list}"
                )
            # model_path already known from env — just skip mmproj discovery

    if not mmproj_path:
        print("[enhancer] No mmproj found; vision will be disabled.")

    return model_path, mmproj_path


# ---------------------------------------------------------------------------
# llama-server subprocess management
# ---------------------------------------------------------------------------
_server_process: subprocess.Popen | None = None
_discovered_model: str = ""
_discovered_mmproj: str = ""


def _find_llama_server() -> str:
    """Locate the llama-server binary. Checks env var first, then PATH."""
    binary = os.environ.get("LLAMA_SERVER_BINARY")
    if binary and os.path.isfile(binary):
        return binary

    # Check common locations
    candidates = [
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
        "/app/llama-server",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Check PATH
    found = shutil.which("llama-server")
    if found:
        return found

    raise FileNotFoundError(
        "llama-server binary not found. "
        "Set LLAMA_SERVER_BINARY env var or install llama.cpp."
    )


def _start_llama_server(n_gpu_layers: int = -1) -> None:
    """Spawn llama-server as a subprocess and wait until it's ready."""
    global _server_process

    if _server_process is not None:
        return

    model_path, mmproj_path = _resolve_model_path()
    global _discovered_model, _discovered_mmproj
    _discovered_model = model_path
    _discovered_mmproj = mmproj_path or ""
    llama_binary = _find_llama_server()

    cmd = [
        llama_binary,
        "-m", model_path,
        "--host", LLAMA_SERVER_HOST,
        "--port", str(LLAMA_SERVER_PORT),
        "-ngl", str(n_gpu_layers),
        "--ctx-size", "4096",
    ]

    if mmproj_path and os.path.isfile(mmproj_path):
        print("[enhancer] Vision enabled — loading with mmproj...")
        cmd.extend(["--mmproj", mmproj_path])
    else:
        print("[enhancer] No mmproj — text-only mode.")

    print(f"[enhancer] Starting llama-server: {' '.join(cmd)}")
    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Register cleanup
    atexit.register(_stop_llama_server)
    signal.signal(signal.SIGTERM, lambda *_: _stop_llama_server())
    signal.signal(signal.SIGINT, lambda *_: _stop_llama_server())

    # Wait for llama-server to be ready
    _wait_for_server(60)


def _stop_llama_server() -> None:
    """Terminate the llama-server subprocess."""
    global _server_process
    if _server_process is None:
        return
    print("[enhancer] Stopping llama-server...")
    _server_process.terminate()
    try:
        _server_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _server_process.kill()
        _server_process.wait()
    _server_process = None
    print("[enhancer] llama-server stopped.")


def _wait_for_server(timeout_sec: int = 60) -> None:
    """Poll llama-server /health until it responds 200 OK."""
    start = time.monotonic()
    url = f"{LLAMA_SERVER_URL}/health"
    last_error = None

    while time.monotonic() - start < timeout_sec:
        # Check if the process died
        if _server_process is not None and _server_process.poll() is not None:
            stderr = _server_process.stderr.read() if _server_process.stderr else ""
            raise RuntimeError(
                f"llama-server exited unexpectedly (code {_server_process.returncode}).\n"
                f"stderr: {stderr[:2000]}"
            )

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[enhancer] llama-server ready on port {LLAMA_SERVER_PORT}")
                    return
        except (urllib.error.URLError, OSError) as e:
            last_error = e

        time.sleep(0.5)

    raise TimeoutError(
        f"llama-server did not become ready within {timeout_sec}s. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Inference via llama-server HTTP API
# ---------------------------------------------------------------------------
def _strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> tags if present."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    return text


def _extract_thinking_tags(text: str) -> str:
    """Extract content inside <think>...</think> tags if present."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _build_data_uri(image_b64: str) -> str:
    """Validate and return a proper data URI from base64 input."""
    clean_b64 = image_b64.strip()
    if clean_b64.startswith("data:"):
        return clean_b64
    return f"data:image/png;base64,{clean_b64}"


def enhance_prompt(prompt: str, image_b64: str = None, options: dict = None) -> dict:
    """Run the prompt enhancer via llama-server and return the enhanced text."""
    options = options or {}
    system_prompt = options.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    max_tokens = int(
        options.get("max_tokens", os.environ.get("MAX_TOKENS", 5000))
    )
    temperature = float(options.get("temperature", 0.7))
    top_p = float(options.get("top_p", 0.9))
    top_k = int(options.get("top_k", 40))
    repeat_penalty = float(options.get("repeat_penalty", 1.1))
    n_gpu_layers = int(options.get("n_gpu_layers", -1))

    # Ensure llama-server is running (first call starts it)
    _start_llama_server(n_gpu_layers)

    # Build messages in OpenAI chat format
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

    # Build OpenAI-compatible request body
    request_body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "stream": False,
    }

    data = json.dumps(request_body).encode("utf-8")
    url = f"{LLAMA_SERVER_URL}/v1/chat/completions"

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(
            f"llama-server returned HTTP {e.code}: {body[:1000]}"
        )
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach llama-server: {e}")

    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content", "") or ""

    # Qwen3 models use reasoning_content for their thinking process.
    reasoning = message.get("reasoning_content", "") or ""

    # Extract thinking from <think> tags in content
    thinking_from_tags = _extract_thinking_tags(content)

    # thinking field: reasoning_content takes precedence, fall back to tags
    thinking = reasoning or thinking_from_tags

    # Build raw_response (full text including reasoning)
    if content and reasoning:
        raw_text = reasoning + "\n\n" + content
    elif content:
        raw_text = content
    else:
        raw_text = reasoning

    print(f"[enhancer] finish_reason={choice.get('finish_reason')!r}  "
          f"content_len={len(content)}  reasoning_len={len(reasoning)}",
          flush=True)

    # enhanced_prompt: content with <think> tags stripped, no reasoning
    enhanced = _strip_thinking_tags(content)

    return {
        "enhanced_prompt": enhanced.strip(),
        "raw_response": raw_text.strip(),
        "thinking": thinking,
        "input_prompt": prompt,
        "image_used": bool(image_b64),
        "model": os.path.basename(_discovered_model),
        "mmproj": os.path.basename(_discovered_mmproj) if _discovered_mmproj else None,
    }


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job):
    job_input = job.get("input", {})

    # --- payload decryption -----------------------------------------------
    encrypted_payload = job_input.get("encrypted_payload")
    if encrypted_payload:
        try:
            decrypted = _decrypt_json(encrypted_payload)
            if not isinstance(decrypted, dict):
                return {"error": "Decrypted payload must be a JSON object"}
            # Merge decrypted payload with outer job_input (outer wins on conflict)
            merged = {**decrypted, **job_input}
            merged.pop("encrypted_payload", None)
            job_input = merged
        except ValueError as exc:
            return {"error": str(exc)}

    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {"error": "Missing required field: 'prompt'"}

    image_b64 = job_input.get("image") or None
    encrypt_output = bool(job_input.get("encrypt_output", False))

    # Pass through any extra options
    options = {
        k: v
        for k, v in job_input.items()
        if k not in ("prompt", "image", "encrypt_output")
    }

    try:
        result = enhance_prompt(prompt, image_b64=image_b64, options=options)

        # --- output encryption ---------------------------------------------
        if encrypt_output:
            try:
                return {"output": {
                    "encrypted_payload": _encrypt_json(result),
                    "encrypted": True,
                }}
            except ValueError as exc:
                return {"error": str(exc)}

        return {"output": result}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Start llama-server at import time so cold-start latency is just the server
# coming up + first inference. The first call to enhance_prompt() will trigger
# _start_llama_server() which starts llama-server and waits for readiness.
# ---------------------------------------------------------------------------
print("[enhancer] Handler loaded. llama-server will start on first request.")

runpod.serverless.start({"handler": handler})
