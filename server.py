"""
Prompt Enhancer Proxy Server
==============================
A lightweight FastAPI server that sits between your client and the RunPod
serverless endpoint.  It handles:

  * Fernet encryption of prompts (same scheme as the handler)
  * Async submission to RunPod
  * Server-side polling for completion
  * Persistent SQLite storage of all requests / responses
  * REST API for querying job status and history

Environment variables
---------------------
RUNPOD_API_KEY      – Your RunPod API key (required)
RUNPOD_ENDPOINT_ID  – RunPod serverless endpoint ID (required)
ENCRYPTION_KEY      – Fernet key for encrypting prompts / outputs (optional)
DATABASE_PATH       – Path to SQLite DB (default: ./history.db)
POLL_INTERVAL_SEC   – How often to poll RunPod (default: 2)
POLL_TIMEOUT_SEC    – Max time to poll before giving up (default: 300)

API
---
POST /enhance
  Submit a prompt enhancement job.
  Body: {
    "prompt": "a cat in space",
    "image": "base64...",          // optional
    "encrypt_output": false,       // optional
    "options": { ... }             // optional (max_tokens, temperature, etc.)
  }
  Returns: { "job_id": "<uuid>", "status": "queued" }

GET  /jobs/{job_id}
  Get a single job's status + result.

GET  /history
  List all jobs (newest first).  Query params: limit, offset.

DELETE /history/{job_id}
  Delete a job from the local DB.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
DATABASE_PATH = os.environ.get("DATABASE_PATH", "history.db")
POLL_INTERVAL_SEC = float(os.environ.get("POLL_INTERVAL_SEC", "2"))
POLL_TIMEOUT_SEC = float(os.environ.get("POLL_TIMEOUT_SEC", "300"))

ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
_fernet: Fernet | None = None
if ENCRYPTION_KEY:
    key_b = ENCRYPTION_KEY.encode()
    if len(key_b) == 43:
        key_b += b"="
    _fernet = Fernet(key_b)

RUNPOD_RUN_URL = (
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    if RUNPOD_ENDPOINT_ID
    else None
)
RUNPOD_STATUS_URL = (
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"
    if RUNPOD_ENDPOINT_ID
    else None
)

HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}"} if RUNPOD_API_KEY else {}


# ---------------------------------------------------------------------------
# Encryption helpers (same scheme as handler.py)
# ---------------------------------------------------------------------------
def _encrypt_text(plaintext: str) -> str:
    if _fernet is None:
        raise ValueError("ENCRYPTION_KEY not configured")
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def _decrypt_text(token: str) -> str:
    if _fernet is None:
        raise ValueError("ENCRYPTION_KEY not configured")
    try:
        return _fernet.decrypt(token.encode()).decode("utf-8")
    except InvalidToken:
        raise ValueError("Invalid token (bad key or corrupted data)")


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------
def _init_db() -> None:
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id              TEXT PRIMARY KEY,
                runpod_job_id   TEXT,
                status          TEXT NOT NULL DEFAULT 'queued',
                input           TEXT,
                output          TEXT,
                error           TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at    TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at
            ON jobs (created_at DESC)
            """
        )
        conn.commit()


def _insert_job(
    job_id: str,
    runpod_job_id: str | None,
    status: str,
    input_json: dict,
) -> None:
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, runpod_job_id, status, input)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, runpod_job_id, status, json.dumps(input_json)),
        )
        conn.commit()


def _update_job(
    job_id: str,
    status: str,
    output_json: dict | None = None,
    error: str | None = None,
) -> None:
    completed_at = datetime.now(timezone.utc).isoformat() if status in ("completed", "failed") else None
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                output = COALESCE(?, output),
                error  = COALESCE(?, error),
                completed_at = COALESCE(?, completed_at)
            WHERE id = ?
            """,
            (
                status,
                json.dumps(output_json) if output_json is not None else None,
                error,
                completed_at,
                job_id,
            ),
        )
        conn.commit()


def _get_job(job_id: str) -> dict | None:
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def _list_jobs(limit: int = 50, offset: int = 0) -> list[dict]:
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image: str | None = None
    encrypt_output: bool = False
    options: dict = Field(default_factory=dict)


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobDetail(BaseModel):
    id: str
    runpod_job_id: str | None
    status: str
    input: dict | None
    output: dict | None
    error: str | None
    created_at: str | None
    completed_at: str | None


# ---------------------------------------------------------------------------
# Background polling
# ---------------------------------------------------------------------------
async def _poll_runpod(job_id: str, runpod_job_id: str) -> None:
    """Poll RunPod until the job finishes, then update the DB."""
    if RUNPOD_STATUS_URL is None:
        _update_job(job_id, "failed", error="RUNPOD_ENDPOINT_ID not configured")
        return

    deadline = asyncio.get_event_loop().time() + POLL_TIMEOUT_SEC
    url = f"{RUNPOD_STATUS_URL}/{runpod_job_id}"

    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                _update_job(job_id, "failed", error=f"RunPod poll error: {exc}")
                return

            status = data.get("status")

            if status in ("COMPLETED", "FAILED"):
                output = data.get("output")
                error = data.get("error")

                # Decrypt output if it was encrypted by the handler
                if output and isinstance(output, dict):
                    if output.get("encrypted") and _fernet is not None:
                        try:
                            if "enhanced_prompt" in output:
                                output["enhanced_prompt"] = _decrypt_text(
                                    output["enhanced_prompt"]
                                )
                            if "raw_response" in output:
                                output["raw_response"] = _decrypt_text(
                                    output["raw_response"]
                                )
                        except ValueError as exc:
                            _update_job(
                                job_id,
                                "failed",
                                error=f"Output decryption failed: {exc}",
                            )
                            return

                _update_job(
                    job_id,
                    "completed" if status == "COMPLETED" else "failed",
                    output_json=output,
                    error=error,
                )
                return

            await asyncio.sleep(POLL_INTERVAL_SEC)

    # Timeout
    _update_job(job_id, "failed", error="Polling timed out")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _init_db()
    yield


app = FastAPI(title="Prompt Enhancer Proxy", lifespan=lifespan)


@app.post("/enhance", response_model=JobResponse)
async def enhance(req: EnhanceRequest) -> JobResponse:
    if RUNPOD_RUN_URL is None:
        raise HTTPException(500, "RUNPOD_ENDPOINT_ID not configured")
    if RUNPOD_API_KEY is None:
        raise HTTPException(500, "RUNPOD_API_KEY not configured")

    job_id = str(uuid.uuid4())

    # Build RunPod input
    runpod_input: dict = {
        "input": {
            "prompt": req.prompt,
            "encrypt_output": req.encrypt_output,
            **req.options,
        }
    }
    if req.image:
        runpod_input["input"]["image"] = req.image

    # Encrypt prompt if key is available
    if _fernet is not None:
        runpod_input["input"]["encrypted_prompt"] = _encrypt_text(req.prompt)
        del runpod_input["input"]["prompt"]

    # Submit to RunPod
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                RUNPOD_RUN_URL,
                headers={**HEADERS, "Content-Type": "application/json"},
                json=runpod_input,
                timeout=30,
            )
            resp.raise_for_status()
            runpod_resp = resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                502, f"RunPod returned {exc.response.status_code}: {exc.response.text}"
            )
        except Exception as exc:
            raise HTTPException(502, f"RunPod request failed: {exc}")

    runpod_job_id = runpod_resp.get("id")
    if not runpod_job_id:
        raise HTTPException(502, "RunPod did not return a job ID")

    # Persist locally
    _insert_job(
        job_id=job_id,
        runpod_job_id=runpod_job_id,
        status="queued",
        input_json=runpod_input,
    )

    # Start background polling
    asyncio.create_task(_poll_runpod(job_id, runpod_job_id))

    return JobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobDetail)
async def get_job(job_id: str) -> JobDetail:
    row = _get_job(job_id)
    if row is None:
        raise HTTPException(404, "Job not found")
    return JobDetail(
        id=row["id"],
        runpod_job_id=row["runpod_job_id"],
        status=row["status"],
        input=json.loads(row["input"]) if row["input"] else None,
        output=json.loads(row["output"]) if row["output"] else None,
        error=row["error"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


@app.get("/history")
async def list_history(limit: int = 50, offset: int = 0) -> list[JobDetail]:
    rows = _list_jobs(limit, offset)
    return [
        JobDetail(
            id=r["id"],
            runpod_job_id=r["runpod_job_id"],
            status=r["status"],
            input=json.loads(r["input"]) if r["input"] else None,
            output=json.loads(r["output"]) if r["output"] else None,
            error=r["error"],
            created_at=r["created_at"],
            completed_at=r["completed_at"],
        )
        for r in rows
    ]


@app.delete("/history/{job_id}")
async def delete_job(job_id: str) -> dict:
    with sqlite3.connect(DATABASE_PATH) as conn:
        cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Job not found")
    return {"deleted": True}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, reload=False)
