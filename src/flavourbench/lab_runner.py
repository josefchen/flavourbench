"""Provider-neutral model runners for the public FlavourBench lab kit."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import httpx

from .lab import LAB_RESPONSE_SCHEMA_VERSION, LabValidationError


def _chat_completions_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def _message_content(document: Mapping[str, Any]) -> str:
    if not isinstance(document, Mapping):
        raise LabValidationError("endpoint response is not a JSON object")
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise LabValidationError("endpoint response has no choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise LabValidationError("endpoint response has no choices[0].message")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    raise LabValidationError("endpoint response contains no answer text")


async def run_openai_compatible(
    tasks: Sequence[Mapping[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key_env: str = "OPENAI_API_KEY",
    concurrency: int = 8,
    timeout_seconds: float = 180.0,
    max_tokens: int = 256,
    temperature: float | None = 0.0,
    max_attempts: int = 3,
    extra_body: Mapping[str, Any] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate tasks through any OpenAI-compatible chat-completions endpoint.

    Credentials are read from an environment variable and are never returned or written into an
    artifact. Failed calls remain explicit; a report is comparable only after all cells succeed.
    """

    key = os.environ.get(api_key_env)
    if not key:
        raise LabValidationError(
            f"required credential environment variable is unset: {api_key_env}"
        )
    parsed_url = urlparse(base_url)
    if not model.strip() or parsed_url.scheme not in {"https", "http"} or not parsed_url.hostname:
        raise LabValidationError("model and an HTTP(S) base URL are required")
    if parsed_url.scheme == "http" and parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise LabValidationError("plain HTTP is allowed only for a local endpoint")
    if concurrency <= 0 or max_attempts <= 0:
        raise LabValidationError("concurrency and max_attempts must be positive")
    endpoint = _chat_completions_url(base_url)
    semaphore = asyncio.Semaphore(concurrency)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "flavourbench-lab/1",
    }

    async with httpx.AsyncClient(timeout=timeout_seconds, headers=headers) as client:

        async def one(task: Mapping[str, Any]) -> dict[str, Any]:
            task_id = str(task["task_id"])
            payload: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": str(task["prompt"])}],
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                payload["temperature"] = temperature
            if extra_body:
                protected = {"model", "messages"}
                if protected & set(extra_body):
                    raise LabValidationError("extra_body cannot replace model or messages")
                payload.update(dict(extra_body))
            started = time.perf_counter()
            errors: list[str] = []
            async with semaphore:
                for attempt in range(1, max_attempts + 1):
                    try:
                        response = await client.post(endpoint, json=payload)
                        if (
                            response.status_code in {408, 409, 425, 429}
                            or response.status_code >= 500
                        ):
                            raise httpx.HTTPStatusError(
                                f"retryable HTTP {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        response.raise_for_status()
                        document = response.json()
                        answer = _message_content(document)
                        usage = document.get("usage") if isinstance(document, Mapping) else None
                        return {
                            "schema_version": LAB_RESPONSE_SCHEMA_VERSION,
                            "task_id": task_id,
                            "status": "completed",
                            "response": answer,
                            "model": model,
                            "backend": "openai_compatible",
                            "attempts": attempt,
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                            "usage": dict(usage) if isinstance(usage, Mapping) else None,
                        }
                    except (httpx.HTTPError, ValueError, LabValidationError) as error:
                        errors.append(f"{type(error).__name__}: {str(error)[:240]}")
                        if attempt < max_attempts:
                            await asyncio.sleep(min(8.0, 0.75 * 2 ** (attempt - 1)))
            return {
                "schema_version": LAB_RESPONSE_SCHEMA_VERSION,
                "task_id": task_id,
                "status": "failed",
                "response": None,
                "model": model,
                "backend": "openai_compatible",
                "attempts": max_attempts,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "error": errors[-1] if errors else "unknown endpoint failure",
            }

        order = {str(task["task_id"]): index for index, task in enumerate(tasks)}
        rows: list[dict[str, Any]] = []
        for future in asyncio.as_completed([one(task) for task in tasks]):
            row = await future
            rows.append(row)
            if on_result is not None:
                on_result(row)
        return sorted(rows, key=lambda row: order[str(row["task_id"])])


def run_transformers(
    tasks: Sequence[Mapping[str, Any]],
    *,
    model: str,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    trust_remote_code: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a local or Hub checkpoint with the Transformers text-generation pipeline."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise LabValidationError(
            "local evaluation requires `pip install 'epicure-flavourbench[transformers]'`"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    checkpoint = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        dtype="auto",
        trust_remote_code=trust_remote_code,
    )
    generator = pipeline("text-generation", model=checkpoint, tokenizer=tokenizer)

    prompts: list[str] = []
    for task in tasks:
        prompt = str(task["prompt"])
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)
    started = time.perf_counter()
    outputs = generator(
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    rows: list[dict[str, Any]] = []
    for task, output in zip(tasks, outputs, strict=True):
        item = output[0] if isinstance(output, list) and output else output
        text = item.get("generated_text") if isinstance(item, Mapping) else None
        rows.append(
            {
                "schema_version": LAB_RESPONSE_SCHEMA_VERSION,
                "task_id": str(task["task_id"]),
                "status": "completed" if isinstance(text, str) and text.strip() else "failed",
                "response": text if isinstance(text, str) else None,
                "model": model,
                "backend": "transformers",
                "latency_ms_total_run": round((time.perf_counter() - started) * 1000),
            }
        )
    return rows
