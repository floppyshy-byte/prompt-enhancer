"""
Unit tests for handler.py functions.

Run with: pytest test_handler.py -v
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock runpod before importing handler (not installed in test environments)
# ---------------------------------------------------------------------------
sys.modules["runpod"] = MagicMock()

sys.path.insert(0, os.path.dirname(__file__))
import handler


# =============================================================================
# Helpers
# =============================================================================

def _make_cache(tmp_path, repo_id, files: dict[str, int]):
    """Create a fake HF cache with the given files and sizes in bytes."""
    repo_dir_name = repo_id.replace("/", "--").lower()
    cache_dir = tmp_path / "hub" / f"models--{repo_dir_name}" / "snapshots" / "abc123"
    cache_dir.mkdir(parents=True)
    for filename, size in files.items():
        (cache_dir / filename).write_bytes(b"x" * size)
    return tmp_path


def _mock_expanduser(hub_path):
    """Return a mock for os.path.expanduser that redirects ~/.cache/huggingface/hub
    to the given hub_path, preserving the path suffix."""
    prefix = "~/.cache/huggingface/hub"
    def _fn(p):
        if p.startswith(prefix):
            return hub_path + p[len(prefix):]
        return p
    return _fn


# =============================================================================
# _strip_thinking_tags
# =============================================================================
class TestStripThinkingTags:
    def test_strips_thinking_tags(self):
        text = "<think>hmm let me think about this...</think>A beautiful sunset."
        assert handler._strip_thinking_tags(text) == "A beautiful sunset."

    def test_strips_multiple_thinking_tags(self):
        text = "<think>first</think>Keep this.<think>second</think>More here."
        assert handler._strip_thinking_tags(text) == "Keep this.More here."

    def test_strips_multiline_thinking(self):
        text = "<think>\nline one\nline two\n</think>\nFinal prompt here."
        assert handler._strip_thinking_tags(text) == "Final prompt here."

    def test_no_thinking_tags(self):
        assert handler._strip_thinking_tags("A simple prompt.") == "A simple prompt."

    def test_empty_string(self):
        assert handler._strip_thinking_tags("") == ""

    def test_only_thinking_tags(self):
        assert handler._strip_thinking_tags("<think>just thinking</think>") == ""


# =============================================================================
# _build_data_uri
# =============================================================================
class TestBuildDataUri:
    def test_raw_base64(self):
        result = handler._build_data_uri("iVBORw0KGgo=")
        assert result.startswith("data:image/png;base64,")

    def test_already_data_uri(self):
        uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        assert handler._build_data_uri(uri) == uri

    def test_strips_whitespace(self):
        result = handler._build_data_uri("  abc123  ")
        assert result == "data:image/png;base64,abc123"

    def test_svg_data_uri(self):
        uri = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0i"
        assert handler._build_data_uri(uri) == uri


# =============================================================================
# _get_cache_snapshot_dirs
# =============================================================================
class TestGetCacheSnapshotDirs:
    def test_empty_repo_id_scans_all_repos(self, tmp_path, monkeypatch):
        """Without repo_id, scan all models--* dirs under the hub."""
        _make_cache(tmp_path, "myorg/myrepo", {"model.gguf": 1000})
        _make_cache(tmp_path, "otherorg/otherrepo", {"other.gguf": 500})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        dirs = handler._get_cache_snapshot_dirs("")
        assert len(dirs) == 2  # one snapshot per repo

    def test_finds_snapshots(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo", {"model.gguf": 1000})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        dirs = handler._get_cache_snapshot_dirs("myorg/myrepo")
        assert len(dirs) == 1
        assert "abc123" in dirs[0]

    def test_ignores_non_directories(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "hub" / "models--myorg--myrepo" / "snapshots"
        repo_dir.mkdir(parents=True)
        (repo_dir / "not_a_dir").write_text("i am a file")
        snapshot = repo_dir / "real_snapshot"
        snapshot.mkdir()
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        dirs = handler._get_cache_snapshot_dirs("myorg/myrepo")
        assert len(dirs) == 1
        assert "real_snapshot" in dirs[0]

    def test_no_snapshots_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        dirs = handler._get_cache_snapshot_dirs("nonexistent/repo")
        assert dirs == []


# =============================================================================
# _find_hf_cached_file
# =============================================================================
class TestFindHfCachedFile:
    def test_empty_filename(self):
        assert handler._find_hf_cached_file("org/repo", "") is None

    def test_finds_file(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo", {"model.gguf": 1000})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        result = handler._find_hf_cached_file("myorg/myrepo", "model.gguf")
        assert result is not None
        assert result.endswith("model.gguf")

    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo", {"other.gguf": 1000})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        assert handler._find_hf_cached_file("myorg/myrepo", "missing.gguf") is None


# =============================================================================
# _autodiscover_gguf_files
# =============================================================================
class TestAutodiscoverGgufFiles:
    def test_no_files(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo", {})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        assert handler._autodiscover_gguf_files("myorg/myrepo") == []

    def test_one_gguf(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo", {"model.gguf": 1000})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        result = handler._autodiscover_gguf_files("myorg/myrepo")
        assert len(result) == 1
        assert result[0][0] == 1000
        assert result[0][1].endswith("model.gguf")

    def test_two_ggufs_sorted_by_size(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo",
                    {"small.gguf": 100, "large.gguf": 5000})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        result = handler._autodiscover_gguf_files("myorg/myrepo")
        assert len(result) == 2
        assert result[0][0] == 5000
        assert result[0][1].endswith("large.gguf")
        assert result[1][0] == 100
        assert result[1][1].endswith("small.gguf")

    def test_three_ggufs(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo",
                    {"a.gguf": 100, "b.gguf": 200, "c.gguf": 300})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        result = handler._autodiscover_gguf_files("myorg/myrepo")
        assert len(result) == 3
        assert result[0][0] == 300

    def test_ignores_non_gguf_files(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "hub" / "models--myorg--myrepo" / "snapshots" / "abc123"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.gguf").write_bytes(b"x" * 1000)
        (cache_dir / "readme.md").write_text("docs")
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        result = handler._autodiscover_gguf_files("myorg/myrepo")
        assert len(result) == 1
        assert result[0][1].endswith("model.gguf")

    def test_dedupes_across_snapshots(self, tmp_path, monkeypatch):
        """Two snapshots, same filename → both appear as distinct entries."""
        base = tmp_path / "hub" / "models--myorg--myrepo" / "snapshots"
        for snap_id, size in [("abc123", 1000), ("def456", 2000)]:
            snap_dir = base / snap_id
            snap_dir.mkdir(parents=True)
            (snap_dir / "model.gguf").write_bytes(b"x" * size)
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        result = handler._autodiscover_gguf_files("myorg/myrepo")
        assert len(result) == 2
        assert result[0][0] == 2000


# =============================================================================
# _resolve_model_path
# =============================================================================
class TestResolveModelPath:
    def test_env_model_path_takes_priority(self, tmp_path, monkeypatch):
        model = tmp_path / "explicit.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setenv("MODEL_PATH", str(model))
        monkeypatch.delenv("MMPROJ_PATH", raising=False)
        monkeypatch.delenv("MODEL_FILE", raising=False)
        monkeypatch.delenv("MMPROJ_FILE", raising=False)
        monkeypatch.setattr(handler, "HF_REPO_ID", "")
        m, mm = handler._resolve_model_path()
        assert m == str(model)
        assert mm is None

    def test_model_file_by_name(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo", {"my-model.gguf": 1000})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("MMPROJ_PATH", raising=False)
        monkeypatch.setenv("MODEL_FILE", "my-model.gguf")
        monkeypatch.delenv("MMPROJ_FILE", raising=False)
        monkeypatch.setattr(handler, "HF_REPO_ID", "myorg/myrepo")
        m, mm = handler._resolve_model_path()
        assert m.endswith("my-model.gguf")
        assert mm is None

    def test_autodiscover_zero_files(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo", {})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("MMPROJ_PATH", raising=False)
        monkeypatch.delenv("MODEL_FILE", raising=False)
        monkeypatch.delenv("MMPROJ_FILE", raising=False)
        monkeypatch.setattr(handler, "HF_REPO_ID", "myorg/myrepo")
        with pytest.raises(FileNotFoundError, match="No .gguf files found"):
            handler._resolve_model_path()

    def test_autodiscover_one_file_text_only(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo", {"model.gguf": 5000})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("MMPROJ_PATH", raising=False)
        monkeypatch.delenv("MODEL_FILE", raising=False)
        monkeypatch.delenv("MMPROJ_FILE", raising=False)
        monkeypatch.setattr(handler, "HF_REPO_ID", "myorg/myrepo")
        m, mm = handler._resolve_model_path()
        assert m.endswith("model.gguf")
        assert mm is None

    def test_autodiscover_two_files_vision(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo",
                    {"big-model.gguf": 9000, "mmproj.gguf": 900})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("MMPROJ_PATH", raising=False)
        monkeypatch.delenv("MODEL_FILE", raising=False)
        monkeypatch.delenv("MMPROJ_FILE", raising=False)
        monkeypatch.setattr(handler, "HF_REPO_ID", "myorg/myrepo")
        m, mm = handler._resolve_model_path()
        assert m.endswith("big-model.gguf")
        assert mm is not None
        assert mm.endswith("mmproj.gguf")

    def test_autodiscover_three_files_raises(self, tmp_path, monkeypatch):
        _make_cache(tmp_path, "myorg/myrepo",
                    {"a.gguf": 100, "b.gguf": 200, "c.gguf": 300})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("MMPROJ_PATH", raising=False)
        monkeypatch.delenv("MODEL_FILE", raising=False)
        monkeypatch.delenv("MMPROJ_FILE", raising=False)
        monkeypatch.setattr(handler, "HF_REPO_ID", "myorg/myrepo")
        with pytest.raises(FileNotFoundError, match="Found 3 .gguf files"):
            handler._resolve_model_path()

    def test_model_path_bypasses_autodiscover(self, tmp_path, monkeypatch):
        """Explicit MODEL_PATH avoids ambiguity error even with 3 ggufs in cache."""
        _make_cache(tmp_path, "myorg/myrepo",
                    {"a.gguf": 100, "b.gguf": 200, "c.gguf": 300})
        explicit_model = tmp_path / "my-model.gguf"
        explicit_model.write_bytes(b"x" * 5000)
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        monkeypatch.setenv("MODEL_PATH", str(explicit_model))
        monkeypatch.delenv("MMPROJ_PATH", raising=False)
        monkeypatch.delenv("MODEL_FILE", raising=False)
        monkeypatch.delenv("MMPROJ_FILE", raising=False)
        monkeypatch.setattr(handler, "HF_REPO_ID", "myorg/myrepo")
        m, mm = handler._resolve_model_path()
        assert m == str(explicit_model)
        # model_path was explicit, so the 3-file ambiguity only affects mmproj,
        # which gets skipped (None) rather than raising an error.
        assert mm is None

    def test_autodiscover_without_hf_repo_id(self, tmp_path, monkeypatch):
        """Auto-discovery works even when HF_REPO_ID is empty (scans all repos)."""
        _make_cache(tmp_path, "myorg/myrepo", {"model.gguf": 5000})
        monkeypatch.setattr(handler.os.path, "expanduser",
                            _mock_expanduser(str(tmp_path / "hub")))
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("MMPROJ_PATH", raising=False)
        monkeypatch.delenv("MODEL_FILE", raising=False)
        monkeypatch.delenv("MMPROJ_FILE", raising=False)
        monkeypatch.setattr(handler, "HF_REPO_ID", "")
        m, mm = handler._resolve_model_path()
        assert m.endswith("model.gguf")
        assert mm is None  # only 1 file → text-only


# =============================================================================
# Fernet encryption helpers
# =============================================================================
class TestFernet:
    def test_get_fernet_no_key(self, monkeypatch):
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        handler._fernet = None
        assert handler._get_fernet() is None

    def test_get_fernet_valid_key(self, monkeypatch):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        handler._fernet = None
        assert handler._get_fernet() is not None

    def test_encrypt_decrypt_roundtrip(self, monkeypatch):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        handler._fernet = None
        original = "a secret prompt"
        encrypted = handler._encrypt_text(original)
        assert encrypted != original
        assert handler._decrypt_text(encrypted) == original

    def test_encrypt_without_key(self, monkeypatch):
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        handler._fernet = None
        with pytest.raises(ValueError, match="ENCRYPTION_KEY is not set"):
            handler._encrypt_text("test")

    def test_decrypt_without_key(self, monkeypatch):
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        handler._fernet = None
        with pytest.raises(ValueError, match="ENCRYPTION_KEY is not set"):
            handler._decrypt_text("gAAAAAB...")

    def test_decrypt_invalid_token(self, monkeypatch):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        handler._fernet = None
        with pytest.raises(ValueError, match="Invalid encrypted_prompt token"):
            handler._decrypt_text("not-a-valid-fernet-token")

    def test_cache_reuse(self, monkeypatch):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        handler._fernet = None
        f1 = handler._get_fernet()
        f2 = handler._get_fernet()
        assert f1 is f2


# =============================================================================
# _find_llama_server
# =============================================================================
class TestFindLlamaServer:
    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\necho fake")
        binary.chmod(0o755)
        monkeypatch.setenv("LLAMA_SERVER_BINARY", str(binary))
        assert handler._find_llama_server() == str(binary)

    def test_env_var_file_not_found_raises(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVER_BINARY", "/nonexistent/llama-server")
        with patch.object(handler.os.path, "isfile", return_value=False), \
             patch.object(handler.shutil, "which", return_value=None):
            with pytest.raises(FileNotFoundError, match="llama-server binary not found"):
                handler._find_llama_server()


# =============================================================================
# enhance_prompt — integration-style test with mocked llama-server
# =============================================================================
def _make_mock_response(content):
    """Create a mock urllib response with the given content string."""
    data = json.dumps(
        {"choices": [{"message": {"content": content}}]}
    ).encode("utf-8")
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = data
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _make_mock_httperror(code, body_bytes):
    """Create a fake HTTPError that mimics urllib.error.HTTPError."""
    fp = MagicMock()
    fp.read.return_value = body_bytes
    err = handler.urllib.error.HTTPError(
        "http://url", code, "Error", {}, fp
    )
    return err


class TestEnhancePrompt:
    def test_text_only_request(self, monkeypatch):
        mock_resp = _make_mock_response(
            "<think>let me enhance this</think>A stunning golden sunset."
        )

        def mock_urlopen(request, timeout=None):
            body = json.loads(request.data)
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["content"] == "a sunset"
            assert body["stream"] is False
            return mock_resp

        monkeypatch.setattr(handler, "_start_llama_server", lambda n_gpu_layers=-1: None)
        monkeypatch.setattr(handler, "LLAMA_SERVER_URL", "http://127.0.0.1:18081")

        with patch.object(handler.urllib.request, "urlopen", mock_urlopen):
            result = handler.enhance_prompt(
                "a sunset",
                options={"system_prompt": "You are a prompt enhancer."}
            )

        assert result["input_prompt"] == "a sunset"
        assert result["image_used"] is False
        assert result["enhanced_prompt"] == "A stunning golden sunset."
        assert "think" in result["raw_response"]

    def test_image_request(self, monkeypatch):
        mock_resp = _make_mock_response("Enhanced image prompt.")

        def mock_urlopen(request, timeout=None):
            body = json.loads(request.data)
            user_msg = body["messages"][1]
            assert isinstance(user_msg["content"], list)
            assert user_msg["content"][0]["type"] == "text"
            assert user_msg["content"][1]["type"] == "image_url"
            return mock_resp

        monkeypatch.setattr(handler, "_start_llama_server", lambda n_gpu_layers=-1: None)
        monkeypatch.setattr(handler, "LLAMA_SERVER_URL", "http://127.0.0.1:18081")

        with patch.object(handler.urllib.request, "urlopen", mock_urlopen):
            result = handler.enhance_prompt(
                "describe this image",
                image_b64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk",
            )

        assert result["image_used"] is True

    def test_llama_server_http_error(self, monkeypatch):
        monkeypatch.setattr(handler, "_start_llama_server", lambda n_gpu_layers=-1: None)
        monkeypatch.setattr(handler, "LLAMA_SERVER_URL", "http://127.0.0.1:18081")

        def mock_urlopen(request, timeout=None):
            raise _make_mock_httperror(500, b"Internal Server Error")

        with patch.object(handler.urllib.request, "urlopen", mock_urlopen):
            with pytest.raises(RuntimeError, match="llama-server returned HTTP 500"):
                handler.enhance_prompt("test")

    def test_passes_all_options(self, monkeypatch):
        captured_body = {}

        def mock_urlopen(request, timeout=None):
            nonlocal captured_body
            captured_body = json.loads(request.data)
            return _make_mock_response("ok")

        monkeypatch.setattr(handler, "_start_llama_server", lambda n_gpu_layers=-1: None)
        monkeypatch.setattr(handler, "LLAMA_SERVER_URL", "http://127.0.0.1:18081")

        with patch.object(handler.urllib.request, "urlopen", mock_urlopen):
            handler.enhance_prompt("test", options={
                "max_tokens": 256, "temperature": 0.5,
                "top_p": 0.8, "top_k": 50, "repeat_penalty": 1.2,
            })

        assert captured_body["max_tokens"] == 256
        assert captured_body["temperature"] == 0.5
        assert captured_body["top_p"] == 0.8
        assert captured_body["top_k"] == 50
        assert captured_body["repeat_penalty"] == 1.2


# =============================================================================
# RunPod handler function
# =============================================================================
class TestHandler:
    def test_basic_text_request(self, monkeypatch):
        def mock_enhance(prompt, image_b64=None, options=None):
            return {
                "enhanced_prompt": "Enhanced: " + prompt,
                "raw_response": "Raw: " + prompt,
                "input_prompt": prompt,
                "image_used": False,
            }
        monkeypatch.setattr(handler, "enhance_prompt", mock_enhance)

        result = handler.handler({"input": {"prompt": "a cat"}})
        assert result["output"]["enhanced_prompt"] == "Enhanced: a cat"

    def test_missing_prompt(self, monkeypatch):
        result = handler.handler({"input": {}})
        assert "error" in result
        err = result["error"].lower()
        assert "prompt" in err or "encrypted_prompt" in err

    def test_encrypted_prompt(self, monkeypatch):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        handler._fernet = None
        fernet = handler._get_fernet()
        encrypted = fernet.encrypt(b"secret prompt").decode()

        def mock_enhance(prompt, image_b64=None, options=None):
            return {
                "enhanced_prompt": "Enhanced: " + prompt,
                "raw_response": "Raw: " + prompt,
                "input_prompt": prompt,
                "image_used": False,
            }
        monkeypatch.setattr(handler, "enhance_prompt", mock_enhance)

        result = handler.handler({"input": {"encrypted_prompt": encrypted}})
        assert result["output"]["input_prompt"] == "secret prompt"

    def test_encrypt_output(self, monkeypatch):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        handler._fernet = None

        def mock_enhance(prompt, image_b64=None, options=None):
            return {
                "enhanced_prompt": "enhanced text",
                "raw_response": "raw text",
                "input_prompt": prompt,
                "image_used": False,
            }
        monkeypatch.setattr(handler, "enhance_prompt", mock_enhance)

        result = handler.handler(
            {"input": {"prompt": "test", "encrypt_output": True}}
        )
        assert result["output"]["encrypted"] is True
        assert result["output"]["enhanced_prompt"] != "enhanced text"
        decrypted = handler._decrypt_text(result["output"]["enhanced_prompt"])
        assert decrypted == "enhanced text"

    def test_passes_extra_options(self, monkeypatch):
        captured_options = {}

        def mock_enhance(prompt, image_b64=None, options=None):
            nonlocal captured_options
            captured_options = options or {}
            return {
                "enhanced_prompt": "ok", "raw_response": "ok",
                "input_prompt": prompt, "image_used": False,
            }
        monkeypatch.setattr(handler, "enhance_prompt", mock_enhance)

        handler.handler({"input": {
            "prompt": "test", "max_tokens": 100,
            "temperature": 0.3, "top_k": 60,
        }})
        assert captured_options["max_tokens"] == 100
        assert captured_options["temperature"] == 0.3
        assert "prompt" not in captured_options

    def test_handler_returns_error_on_exception(self, monkeypatch):
        def mock_enhance(prompt, image_b64=None, options=None):
            raise RuntimeError("GPU out of memory")
        monkeypatch.setattr(handler, "enhance_prompt", mock_enhance)

        result = handler.handler({"input": {"prompt": "test"}})
        assert result["error"] == "GPU out of memory"
