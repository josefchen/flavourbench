from __future__ import annotations

import asyncio
import hmac
import json
import os
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response, status

MANIFEST_PATH = Path(
    os.environ.get(
        "FLAVOURBENCH_DEPLOYMENT_MANIFEST", "/opt/flavourbench/deployment-manifest.json"
    )
)
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
MANIFEST_SHA256 = str(MANIFEST["spec_sha256"])
EXPECTED_MODEL = str(MANIFEST["model"]["served_model_name"])
DECODING = MANIFEST["decoding"]
INTERNAL_PORT = int(os.environ.get("FLAVOURBENCH_INTERNAL_VLLM_PORT", "8001"))
GATEWAY_PORT = int(os.environ.get("FLAVOURBENCH_GATEWAY_PORT", "8000"))
GATEWAY_HOST = os.environ.get("FLAVOURBENCH_GATEWAY_HOST", "0.0.0.0")  # noqa: S104
INTERNAL_URL = f"http://127.0.0.1:{INTERNAL_PORT}"
GATEWAY_API_KEY = os.environ.get("FLAVOURBENCH_GATEWAY_API_KEY", "")
INTERNAL_API_KEY = os.environ.get("FLAVOURBENCH_INTERNAL_VLLM_API_KEY", "")
VLLM_ARGS = json.loads(os.environ["FLAVOURBENCH_VLLM_ARGS_JSON"])
if not isinstance(VLLM_ARGS, list) or not all(isinstance(item, str) for item in VLLM_ARGS):
    raise RuntimeError("FLAVOURBENCH_VLLM_ARGS_JSON must encode a string array")

HOP_BY_HOP = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _runtime_attestation() -> dict[str, Any]:
    gpu = "unavailable"
    driver = "unavailable"
    try:
        result = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
        gpu = "; ".join(rows)
        if rows and "," in rows[0]:
            driver = rows[0].rsplit(",", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "manifest_sha256": MANIFEST_SHA256,
        "backend": MANIFEST["backend"]["kind"],
        "container_image": MANIFEST["runtime"]["container_image"],
        "gpu": gpu,
        "driver_version": driver,
        "modal_cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_region": os.environ.get("MODAL_REGION"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "lambda_instance_id": os.environ.get("FLAVOURBENCH_LAMBDA_INSTANCE_ID"),
    }


def _authorize(authorization: str | None) -> None:
    if not GATEWAY_API_KEY:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "gateway key is not configured")
    supplied = ""
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(supplied, GATEWAY_API_KEY):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid gateway credentials")


def _normalize_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("model") != EXPECTED_MODEL:
        raise HTTPException(status.HTTP_409_CONFLICT, "requested model is not frozen here")
    if payload.get("stream") not in {None, False}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "streaming is disabled")

    normalized = dict(payload)
    for field in ("temperature", "top_p", "max_tokens", "seed"):
        expected = DECODING[field]
        supplied = normalized.get(field, expected)
        if supplied != expected:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"decoding parameter {field} differs from the frozen manifest",
            )
        normalized[field] = expected
    normalized["parallel_tool_calls"] = False
    normalized["stream"] = False
    return normalized


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    command = list(VLLM_ARGS)
    if INTERNAL_API_KEY:
        command.extend(["--api-key", INTERNAL_API_KEY])
    # Every command token is derived from a hash-validated deployment manifest.
    process = subprocess.Popen(command, start_new_session=True)  # noqa: S603
    _app.state.vllm = process
    _app.state.client = httpx.AsyncClient(
        base_url=INTERNAL_URL,
        timeout=httpx.Timeout(600, connect=10),
    )
    try:
        yield
    finally:
        await _app.state.client.aclose()
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 20)
        except subprocess.TimeoutExpired:
            process.kill()


app = FastAPI(
    title="FlavourBench immutable inference gateway",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> Response:
    process = app.state.vllm
    if process.poll() is not None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "vLLM process exited")
    try:
        response = await app.state.client.get("/health")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "vLLM is warming") from exc
    return Response(
        content=json.dumps({"status": "ready", "manifest_sha256": MANIFEST_SHA256}),
        media_type="application/json",
        headers={"X-FlavourBench-Deployment-SHA256": MANIFEST_SHA256},
    )


@app.get("/flavourbench/manifest")
async def deployment_manifest(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    return {"spec": MANIFEST, "runtime": _runtime_attestation()}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_vllm(
    path: str,
    request: Request,
    authorization: str | None = Header(default=None),
    generation_id: str | None = Header(default=None, alias="X-FlavourBench-Generation-ID"),
) -> Response:
    _authorize(authorization)
    if path not in {"models", "chat/completions"}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "endpoint is not exposed")
    if path == "chat/completions" and not generation_id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "X-FlavourBench-Generation-ID is required",
        )

    body = await request.body()
    if request.method == "POST":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "request must be an object")
        if path == "chat/completions":
            payload = _normalize_chat_request(payload)
        body = json.dumps(payload, separators=(",", ":")).encode()

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP and key.lower() != "authorization"
    }
    if INTERNAL_API_KEY:
        headers["Authorization"] = f"Bearer {INTERNAL_API_KEY}"
    if generation_id:
        headers["X-Request-Id"] = generation_id
    try:
        upstream = await app.state.client.request(
            request.method,
            f"/{path}",
            content=body or None,
            headers=headers,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "inference unavailable") from exc

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in HOP_BY_HOP
    }
    response_headers["X-FlavourBench-Deployment-SHA256"] = MANIFEST_SHA256
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


if __name__ == "__main__":
    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT, access_log=False, server_header=False)
