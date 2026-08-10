"""Independent Bedrock curation for the real-user Season 0 task bank.

The curation models never see accepted answers. Their role is limited to
screening real, licensed human questions for benchmark-family fit and the
specialist risks excluded from the general FlavourBench track.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import httpx

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .bedrock_provider import structured_output_config
from .execution_policy import assert_legacy_paid_cli_allowed
from .real_task_bank import (
    FAMILY_TAGS,
    SelectionPolicy,
    StackExchangeClient,
    _eligible_question,
    _family_match_score,
    _owner,
    html_to_text,
    sha256_json,
    sha256_text,
    utc_iso,
)

PREPARE_SCHEMA = "flavourbench-real-task-candidate-pool-v1"
CURATION_SCHEMA = "flavourbench-bedrock-task-curation-batch-v1"
CURATION_CONFIRMATION = "RUN_REAL_BEDROCK_TASK_CURATION_V1"
ALLOWED_FAMILIES = frozenset({"substitution", "composition", "cookability", "evidence"})
ALLOWED_RISKS = frozenset(
    {"none", "nutrition", "allergen", "food_safety", "cultural_authenticity", "other"}
)

CURATION_SYSTEM_PROMPT = """You are an independent benchmark task curator.
Judge only whether each REAL human-authored Cooking Stack Exchange question is suitable for
FlavourBench. Never answer the culinary question. The accepted human answer is hidden from you.

General-track families:
- substitution: ingredient substitution under explicit culinary constraints. Exclude equipment or
  method replacement unless the central problem is an ingredient substitution.
- composition: multi-ingredient selection, flavour balancing, bridge ingredients, or
  recipe-component interaction. Exclude simple product, ingredient, or utensil identification.
- cookability: actionable recipe execution, practical technique, quantities, timing, or equipment
  constraints where a cook could act on the answer.
- evidence: interpretation of culinary mechanisms or evidence, especially where causal reasoning,
  competing explanations, or calibrated uncertainty matter.

Reject tasks dominated by formal nutrition, allergens/intolerances, food safety, or cultural
authenticity; trivial identification; shopping/location lookup; broken context; or questions that
cannot support a substantive culinary answer. Vegan/gluten-free constraints are allowed only when
they are culinary formulation constraints rather than allergy or medical advice.

For every input question return exactly one JSON object with:
question_id (integer), include (boolean), family (one of substitution, composition, cookability,
evidence, reject), family_fit (0-4), difficulty (0-4), specificity (0-4),
epicure_relevance (0-4), specialist_risk (one of none, nutrition, allergen, food_safety,
cultural_authenticity, other), self_contained (boolean), requires_multi_step (boolean), and
rationale (at most 24 words). A task can be included only when family is not reject,
specialist_risk is none, and self_contained is true. Return a JSON array only, in the same order as
the inputs."""


@dataclass(frozen=True)
class CuratorSpec:
    curator_id: str
    backend: Literal["bedrock", "openrouter"]
    provider: str
    model_id: str
    provider_slug: str | None
    conservative_input_usd_per_million: Decimal
    conservative_output_usd_per_million: Decimal
    batch_size: int
    structured_output: bool


CURATORS = (
    CuratorSpec(
        curator_id="anthropic-sonnet-4-6",
        backend="bedrock",
        provider="Anthropic via Amazon Bedrock",
        model_id="global.anthropic.claude-sonnet-4-6",
        provider_slug=None,
        conservative_input_usd_per_million=Decimal("5"),
        conservative_output_usd_per_million=Decimal("25"),
        batch_size=8,
        structured_output=False,
    ),
    CuratorSpec(
        curator_id="google-gemini-3-1-flash-lite",
        backend="openrouter",
        provider="Google Vertex via OpenRouter",
        model_id="google/gemini-3.1-flash-lite",
        provider_slug="google-vertex/global/flex",
        conservative_input_usd_per_million=Decimal("1"),
        conservative_output_usd_per_million=Decimal("3"),
        batch_size=8,
        structured_output=False,
    ),
)


class TaskCurationError(RuntimeError):
    """The real-task curation contract was not satisfied."""


def _atomic_content_addressed_write(
    directory: Path, prefix: str, payload: Mapping[str, Any]
) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    rendered_bytes = rendered.encode("utf-8") + b"\n"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != rendered_bytes:
            raise TaskCurationError(f"content-address conflict at {destination}")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _candidate_record(
    question: Mapping[str, Any], answer: Mapping[str, Any], policy: SelectionPolicy
) -> dict[str, Any]:
    question_id = int(question["question_id"])
    answer_id = int(answer["answer_id"])
    title = html.unescape(str(question.get("title") or "")).strip()
    question_text = html_to_text(str(question.get("body") or ""))
    answer_text = html_to_text(str(answer.get("body") or ""))
    prompt = f"{title}\n\n{question_text}".strip()
    record: dict[str, Any] = {
        "candidate_id": f"fb-s0-source-{question_id}",
        "question_id": question_id,
        "title": title,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "source": {
            "corpus": "Seasoned Advice (Stack Exchange)",
            "human_origin": True,
            "question_id": question_id,
            "url": str(question["link"]),
            "created_utc": utc_iso(int(question["creation_date"])),
            "last_activity_utc": utc_iso(int(question["last_activity_date"])),
            "license": str(question["content_license"]),
            "author": _owner(question),
            "score_at_import": int(question.get("score") or 0),
            "answer_count_at_import": int(question.get("answer_count") or 0),
            "tags": sorted(str(tag) for tag in question.get("tags", [])),
        },
        "human_reference": {
            "answer_id": answer_id,
            "accepted": True,
            "text": answer_text,
            "text_sha256": sha256_text(answer_text),
            "url": f"https://cooking.stackexchange.com/a/{answer_id}",
            "created_utc": utc_iso(int(answer["creation_date"])),
            "last_activity_utc": utc_iso(
                int(answer.get("last_activity_date") or answer["creation_date"])
            ),
            "license": str(answer["content_license"]),
            "author": _owner(answer),
            "score_at_import": int(answer.get("score") or 0),
            "use": "hidden_reference_not_automatic_ground_truth",
        },
        "heuristics": {
            "family_match_scores": {
                family: _family_match_score(question, family) for family in sorted(FAMILY_TAGS)
            },
            "recent": int(question["creation_date"]) >= policy.recent_fromdate,
            "specialist_regex_screen_passed": True,
        },
    }
    record["record_sha256"] = sha256_json(record)
    return record


def build_candidate_pool(*, policy: SelectionPolicy) -> dict[str, Any]:
    with StackExchangeClient() as client:
        questions = client.fetch_questions(fromdate=policy.fromdate)
        answer_ids = sorted(
            {
                int(question["accepted_answer_id"])
                for question in questions
                if isinstance(question.get("accepted_answer_id"), int)
            }
        )
        answers = client.fetch_answers(answer_ids)

    candidates: list[dict[str, Any]] = []
    for question in questions:
        answer_id = question.get("accepted_answer_id")
        if not isinstance(answer_id, int):
            continue
        answer = answers.get(answer_id)
        if not _eligible_question(question, answer, policy):
            continue
        assert answer is not None
        candidates.append(_candidate_record(question, answer, policy))
    candidates.sort(key=lambda item: int(item["question_id"]))
    pool: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "benchmark": "FlavourBench Season 0",
        "source_class": "real_human_authored_public_questions",
        "synthetic_tasks": 0,
        "curation_answers_visible_to_models": False,
        "source": {
            "api": "https://api.stackexchange.com/2.3",
            "site": "cooking",
            "fromdate_utc": utc_iso(policy.fromdate),
            "recent_fromdate_utc": utc_iso(policy.recent_fromdate),
            "observed_through_utc": utc_iso(policy.observed_through),
            "retrieved_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "questions_retrieved": len(questions),
            "accepted_answers_retrieved": len(answers),
        },
        "counts": {
            "eligible_candidates": len(candidates),
            "recent_candidates": sum(item["heuristics"]["recent"] for item in candidates),
            "accepted_human_references": len(candidates),
        },
        "candidates": candidates,
    }
    pool["content_sha256"] = sha256_json(
        {
            "schema_version": PREPARE_SCHEMA,
            "source": {
                key: pool["source"][key]
                for key in ("site", "fromdate_utc", "recent_fromdate_utc", "observed_through_utc")
            },
            "candidates": candidates,
        }
    )
    return pool


def write_candidate_pool(pool: Mapping[str, Any], output_dir: Path) -> Path:
    payload = dict(pool)
    digest = str(payload["content_sha256"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"real-task-candidate-pool-{digest}.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    if path.exists() and path.read_bytes() != rendered:
        raise TaskCurationError("candidate-pool content-address conflict")
    path.write_bytes(rendered)
    return path


def load_candidate_pool(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("schema_version") != PREPARE_SCHEMA:
        raise TaskCurationError("candidate pool has the wrong schema")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TaskCurationError("candidate pool is empty")
    expected = sha256_json(
        {
            "schema_version": PREPARE_SCHEMA,
            "source": {
                key: value["source"][key]
                for key in ("site", "fromdate_utc", "recent_fromdate_utc", "observed_through_utc")
            },
            "candidates": candidates,
        }
    )
    if value.get("content_sha256") != expected:
        raise TaskCurationError("candidate pool content digest does not verify")
    return value


def _curation_input(candidate: Mapping[str, Any]) -> dict[str, Any]:
    source = candidate["source"]
    return {
        "question_id": candidate["question_id"],
        "title": candidate["title"],
        "question": candidate["prompt"],
        "tags": source["tags"],
        "created_utc": source["created_utc"],
    }


def build_batch_prompt(candidates: Sequence[Mapping[str, Any]]) -> str:
    payload = [_curation_input(candidate) for candidate in candidates]
    return (
        "Evaluate every question below under the frozen rubric. Return JSON array only.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def curation_output_schema(min_items: int) -> dict[str, Any]:
    if min_items <= 0:
        raise TaskCurationError("curation output schema requires a positive item count")
    score = {"type": "integer", "enum": [0, 1, 2, 3, 4]}
    properties: dict[str, Any] = {
        "question_id": {"type": "integer"},
        "include": {"type": "boolean"},
        "family": {
            "type": "string",
            "enum": ["substitution", "composition", "cookability", "evidence", "reject"],
        },
        "family_fit": score,
        "difficulty": score,
        "specificity": score,
        "epicure_relevance": score,
        "specialist_risk": {"type": "string", "enum": sorted(ALLOWED_RISKS)},
        "self_contained": {"type": "boolean"},
        "requires_multi_step": {"type": "boolean"},
        "rationale": {"type": "string"},
    }
    return {
        "type": "array",
        # Bedrock's supported JSON-Schema subset accepts minItems only as 0 or 1.
        # Exact batch cardinality is enforced by parse_judgments after generation.
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def _schema_for_curator(spec: CuratorSpec, batch_length: int) -> dict[str, Any]:
    schema = curation_output_schema(batch_length)
    if spec.backend == "openrouter":
        judgments = {**schema, "minItems": batch_length, "maxItems": batch_length}
        schema = {
            "type": "object",
            "properties": {"judgments": judgments},
            "required": ["judgments"],
            "additionalProperties": False,
        }
    return schema


def _curation_input_sha(spec: CuratorSpec, prompt: str, batch_length: int) -> str:
    payload: dict[str, Any] = {
        "system_prompt": CURATION_SYSTEM_PROMPT,
        "prompt": prompt,
        "model_id": spec.model_id,
    }
    if spec.structured_output:
        payload["structured_output_schema_sha256"] = sha256_json(
            _schema_for_curator(spec, batch_length)
        )
    if spec.backend == "openrouter":
        payload["backend"] = spec.backend
        payload["provider_slug"] = spec.provider_slug
    return sha256_json(payload)


def parse_judgments(raw_text: str, expected_question_ids: Sequence[int]) -> list[dict[str, Any]]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise TaskCurationError("curator did not return a JSON array")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise TaskCurationError("curator returned invalid JSON") from error
    if not isinstance(value, list) or len(value) != len(expected_question_ids):
        raise TaskCurationError("curator returned the wrong number of judgments")
    normalized: list[dict[str, Any]] = []
    for expected_id, item in zip(expected_question_ids, value, strict=True):
        if not isinstance(item, Mapping) or item.get("question_id") != expected_id:
            raise TaskCurationError("curator changed question order or identity")
        family = str(item.get("family") or "")
        risk = str(item.get("specialist_risk") or "")
        if family not in ALLOWED_FAMILIES | {"reject"} or risk not in ALLOWED_RISKS:
            raise TaskCurationError("curator returned an invalid categorical label")
        scores: dict[str, int] = {}
        for key in ("family_fit", "difficulty", "specificity", "epicure_relevance"):
            score = item.get(key)
            if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
                raise TaskCurationError(f"curator returned an invalid {key}")
            scores[key] = score
        include = item.get("include")
        self_contained = item.get("self_contained")
        multi_step = item.get("requires_multi_step")
        if not all(isinstance(value, bool) for value in (include, self_contained, multi_step)):
            raise TaskCurationError("curator returned an invalid boolean")
        logically_includable = family != "reject" and risk == "none" and self_contained
        rationale = " ".join(str(item.get("rationale") or "").split())
        if not rationale or len(rationale.split()) > 32:
            raise TaskCurationError("curator rationale is empty or too long")
        normalized.append(
            {
                "question_id": expected_id,
                "include": logically_includable,
                "include_reported": bool(include),
                "include_consistent": bool(include) == logically_includable,
                "family": family,
                **scores,
                "specialist_risk": risk,
                "self_contained": bool(self_contained),
                "requires_multi_step": bool(multi_step),
                "rationale": rationale,
            }
        )
    return normalized


def _text_from_converse(response: Mapping[str, Any]) -> str:
    try:
        blocks = response["output"]["message"]["content"]
    except (KeyError, TypeError) as error:
        raise TaskCurationError("Bedrock response has no output message") from error
    if not isinstance(blocks, list):
        raise TaskCurationError("Bedrock response content is invalid")
    text = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, Mapping))
    if not text:
        raise TaskCurationError("Bedrock response contains no text")
    return text


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(raw.get("inputTokens") or 0),
        "output_tokens": int(raw.get("outputTokens") or 0),
        "total_tokens": int(raw.get("totalTokens") or 0),
    }


def _estimated_cost(spec: CuratorSpec, usage: Mapping[str, int]) -> Decimal:
    return (
        Decimal(usage["input_tokens"]) * spec.conservative_input_usd_per_million
        + Decimal(usage["output_tokens"]) * spec.conservative_output_usd_per_million
    ) / Decimal(1_000_000)


def _reservation_cost(spec: CuratorSpec, prompt: str, max_tokens: int) -> Decimal:
    conservative_input_tokens = math.ceil(len(prompt) / 3)
    return (
        Decimal(conservative_input_tokens) * spec.conservative_input_usd_per_million
        + Decimal(max_tokens) * spec.conservative_output_usd_per_million
    ) / Decimal(1_000_000)


def _curate_batch(
    *,
    runtime: Any | None,
    openrouter_client: httpx.Client | None,
    spec: CuratorSpec,
    candidates: Sequence[Mapping[str, Any]],
    batch_index: int,
    candidate_pool_sha256: str,
    max_tokens: int,
    attempts: int,
    allow_singleton_repair: bool = True,
) -> dict[str, Any]:
    prompt = build_batch_prompt(candidates)
    expected_ids = [int(candidate["question_id"]) for candidate in candidates]
    input_sha256 = _curation_input_sha(spec, prompt, len(candidates))
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            returned_model_id: str | None = None
            returned_provider: str | None = None
            generation_id: str | None = None
            request_id: str | None = None
            if spec.backend == "bedrock":
                if runtime is None:
                    raise TaskCurationError("Bedrock curator has no runtime client")
                request: dict[str, Any] = {
                    "modelId": spec.model_id,
                    "system": [{"text": CURATION_SYSTEM_PROMPT}],
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
                    "requestMetadata": {
                        "flavourbench_phase": "task_curation",
                        "flavourbench_batch": hashlib.sha256(input_sha256.encode()).hexdigest(),
                    },
                }
                if spec.structured_output:
                    request["outputConfig"] = structured_output_config(
                        _schema_for_curator(spec, len(candidates))
                    )
                response = runtime.converse(**request)
                raw_text = _text_from_converse(response)
                usage = _usage(response)
                stop_reason = str(response.get("stopReason") or "unknown")
                response_metadata = response.get("ResponseMetadata")
                request_id = (
                    str(response_metadata.get("RequestId") or "") or None
                    if isinstance(response_metadata, Mapping)
                    else None
                )
            else:
                if openrouter_client is None or not spec.provider_slug:
                    raise TaskCurationError("OpenRouter curator has no fixed client or provider")
                openrouter_payload: dict[str, Any] = {
                    "model": spec.model_id,
                    "messages": [
                        {"role": "system", "content": CURATION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "provider": {
                        "only": [spec.provider_slug],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                        "data_collection": "deny",
                    },
                    "usage": {"include": True},
                }
                if spec.structured_output:
                    openrouter_payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "flavourbench_task_curation",
                            "strict": True,
                            "schema": _schema_for_curator(spec, len(candidates)),
                        },
                    }
                response_http = openrouter_client.post(
                    "chat/completions",
                    json=openrouter_payload,
                    headers={"Idempotency-Key": input_sha256},
                )
                response_http.raise_for_status()
                response = response_http.json()
                if not isinstance(response, Mapping):
                    raise TaskCurationError("OpenRouter returned a non-object response")
                choices = response.get("choices")
                if (
                    not isinstance(choices, list)
                    or not choices
                    or not isinstance(choices[0], Mapping)
                ):
                    raw_error = response.get("error")
                    safe_error = (
                        str(raw_error.get("message") or "unknown")[:300]
                        if isinstance(raw_error, Mapping)
                        else "unknown"
                    )
                    raise TaskCurationError(f"OpenRouter returned no curator choice: {safe_error}")
                choice = choices[0]
                message = choice.get("message")
                if not isinstance(message, Mapping):
                    raise TaskCurationError("OpenRouter curator choice has no message")
                parsed = message.get("parsed")
                raw_text = (
                    json.dumps(parsed, ensure_ascii=False)
                    if isinstance(parsed, (list, Mapping))
                    else str(message.get("content") or "")
                )
                raw_usage = response.get("usage")
                if not isinstance(raw_usage, Mapping):
                    raw_usage = {}
                usage = {
                    "input_tokens": int(raw_usage.get("prompt_tokens") or 0),
                    "output_tokens": int(raw_usage.get("completion_tokens") or 0),
                    "total_tokens": int(raw_usage.get("total_tokens") or 0),
                }
                stop_reason = str(choice.get("finish_reason") or "unknown")
                request_id = response_http.headers.get("x-request-id")
                returned_model_id = str(response.get("model") or "") or None
                returned_provider = str(response.get("provider") or "") or None
                generation_id = str(response.get("id") or "") or None
            elapsed_ms = round((time.monotonic() - started) * 1000)
            judgments = parse_judgments(raw_text, expected_ids)
            return {
                "schema_version": CURATION_SCHEMA,
                "candidate_pool_sha256": candidate_pool_sha256,
                "curator": {
                    "curator_id": spec.curator_id,
                    "provider": spec.provider,
                    "requested_model_id": spec.model_id,
                    "returned_model_id": returned_model_id,
                    "returned_provider": returned_provider,
                    "identity_note": (
                        "Bedrock Converse did not return a destination-model attestation"
                        if spec.backend == "bedrock"
                        else (
                            "OpenRouter response identity recorded; "
                            "accounting reconciliation pending"
                        )
                    ),
                },
                "batch_index": batch_index,
                "question_ids": expected_ids,
                "input_sha256": input_sha256,
                "system_prompt_sha256": sha256_text(CURATION_SYSTEM_PROMPT),
                "rubric_version": "flavourbench-real-task-curation-v1",
                "decoding": {
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "structured_output": spec.structured_output,
                },
                "attempt": attempt,
                "prior_errors": errors,
                "stop_reason": stop_reason,
                "usage": usage,
                "estimated_cost_usd": format(_estimated_cost(spec, usage), "f"),
                "cost_status": "conservative_rate_estimate_not_billing_reconciled",
                "latency_ms": elapsed_ms,
                "provider_request_id_sha256": (
                    sha256_text(str(request_id)) if request_id else None
                ),
                "generation_id": generation_id,
                "raw_response": raw_text,
                "raw_response_sha256": sha256_text(raw_text),
                "judgments": judgments,
            }
        except Exception as error:  # noqa: BLE001 - error is redacted before persistence
            message = re.sub(r"(?<!\d)\d{12}(?!\d)", "<account-redacted>", str(error))
            errors.append(f"{type(error).__name__}: {message[:500]}")
    if allow_singleton_repair and spec.backend == "openrouter" and len(candidates) > 1:
        subcalls = [
            _curate_batch(
                runtime=runtime,
                openrouter_client=openrouter_client,
                spec=spec,
                candidates=[candidate],
                batch_index=batch_index,
                candidate_pool_sha256=candidate_pool_sha256,
                max_tokens=max_tokens,
                attempts=attempts,
                allow_singleton_repair=False,
            )
            for candidate in candidates
        ]
        judgments = [subcall["judgments"][0] for subcall in subcalls]
        if [judgment["question_id"] for judgment in judgments] != expected_ids:
            raise TaskCurationError("singleton repair changed question order or identity")
        usage = {
            key: sum(int(subcall["usage"][key]) for subcall in subcalls)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        raw_responses = [str(subcall["raw_response"]) for subcall in subcalls]
        raw_response = json.dumps(raw_responses, ensure_ascii=False, separators=(",", ":"))
        generation_ids = [
            str(subcall["generation_id"]) for subcall in subcalls if subcall.get("generation_id")
        ]
        returned_model_ids = sorted(
            {
                str(subcall["curator"]["returned_model_id"])
                for subcall in subcalls
                if subcall["curator"].get("returned_model_id")
            }
        )
        returned_providers = sorted(
            {
                str(subcall["curator"]["returned_provider"])
                for subcall in subcalls
                if subcall["curator"].get("returned_provider")
            }
        )
        return {
            "schema_version": CURATION_SCHEMA,
            "candidate_pool_sha256": candidate_pool_sha256,
            "curator": {
                "curator_id": spec.curator_id,
                "provider": spec.provider,
                "requested_model_id": spec.model_id,
                "returned_model_id": (
                    returned_model_ids[0] if len(returned_model_ids) == 1 else None
                ),
                "returned_provider": (
                    returned_providers[0] if len(returned_providers) == 1 else None
                ),
                "identity_note": (
                    "OpenRouter singleton repair response identities recorded; "
                    "accounting reconciliation pending"
                ),
            },
            "batch_index": batch_index,
            "question_ids": expected_ids,
            "input_sha256": input_sha256,
            "system_prompt_sha256": sha256_text(CURATION_SYSTEM_PROMPT),
            "rubric_version": "flavourbench-real-task-curation-v1",
            "decoding": {
                "temperature": 0,
                "max_tokens": max_tokens,
                "structured_output": spec.structured_output,
            },
            "attempt": attempts,
            "prior_errors": errors,
            "repair_strategy": "one_real_provider_call_per_question_after_batch_parse_failure",
            "stop_reason": "singleton_repair_complete",
            "usage": usage,
            "estimated_cost_usd": format(_estimated_cost(spec, usage), "f"),
            "cost_status": "conservative_rate_estimate_not_billing_reconciled",
            "latency_ms": sum(int(subcall["latency_ms"]) for subcall in subcalls),
            "provider_request_id_sha256": None,
            "provider_request_id_sha256s": [
                subcall["provider_request_id_sha256"]
                for subcall in subcalls
                if subcall.get("provider_request_id_sha256")
            ],
            "generation_id": None,
            "generation_ids": generation_ids,
            "raw_response": raw_response,
            "raw_response_sha256": sha256_text(raw_response),
            "raw_response_sha256s": [str(subcall["raw_response_sha256"]) for subcall in subcalls],
            "judgments": judgments,
        }
    raise TaskCurationError(
        f"curator {spec.curator_id} batch {batch_index} failed after {attempts} attempts: "
        + " | ".join(errors)
    )


def run_curation(
    *,
    pool: Mapping[str, Any],
    output_dir: Path,
    runtime: Any | None,
    openrouter_client: httpx.Client | None = None,
    curators: Sequence[CuratorSpec] = CURATORS,
    batch_size: int | None = None,
    max_workers: int = 4,
    max_tokens: int = 3_000,
    attempts: int = 2,
    cap_usd: Decimal = Decimal("30"),
) -> dict[str, Any]:
    if batch_size is not None and batch_size <= 0:
        raise TaskCurationError("a curation batch-size override must be positive")
    if max_workers <= 0 or max_tokens <= 0 or attempts <= 0:
        raise TaskCurationError("curation execution limits must be positive")
    candidates = pool["candidates"]
    jobs: list[tuple[CuratorSpec, int, Sequence[Mapping[str, Any]], str, Decimal]] = []
    total_reservation = Decimal(0)
    batches_by_curator: dict[str, int] = {}
    for spec in curators:
        effective_batch_size = batch_size or spec.batch_size
        batches = [
            candidates[start : start + effective_batch_size]
            for start in range(0, len(candidates), effective_batch_size)
        ]
        batches_by_curator[spec.curator_id] = len(batches)
        for batch_index, batch in enumerate(batches):
            prompt = build_batch_prompt(batch)
            input_sha = _curation_input_sha(spec, prompt, len(batch))
            reservation = _reservation_cost(spec, prompt, max_tokens) * attempts
            if spec.backend == "openrouter" and len(batch) > 1:
                reservation += sum(
                    (
                        _reservation_cost(spec, build_batch_prompt([candidate]), max_tokens)
                        * attempts
                        for candidate in batch
                    ),
                    Decimal(0),
                )
            total_reservation += reservation
            jobs.append((spec, batch_index, batch, input_sha, reservation))
    if total_reservation > cap_usd:
        raise TaskCurationError(
            f"worst-case curation reservation {total_reservation} exceeds cap {cap_usd}"
        )

    existing_by_key: dict[tuple[str, int, str], Path] = {}
    for path in output_dir.glob("curation-*.json") if output_dir.exists() else ():
        try:
            value = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            continue
        curator = value.get("curator")
        if not isinstance(curator, Mapping):
            continue
        key = (
            str(curator.get("curator_id") or ""),
            int(value.get("batch_index") or 0),
            str(value.get("input_sha256") or ""),
        )
        existing_by_key[key] = path

    artifacts: list[Path] = []
    pending: list[tuple[CuratorSpec, int, Sequence[Mapping[str, Any]]]] = []
    for spec, batch_index, batch, input_sha, _reservation in jobs:
        existing = existing_by_key.get((spec.curator_id, batch_index, input_sha))
        if existing is not None:
            artifacts.append(existing)
        else:
            pending.append((spec, batch_index, batch))

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _curate_batch,
                runtime=runtime,
                openrouter_client=openrouter_client,
                spec=spec,
                candidates=batch,
                batch_index=batch_index,
                candidate_pool_sha256=str(pool["content_sha256"]),
                max_tokens=max_tokens,
                attempts=attempts,
            ): (spec, batch_index)
            for spec, batch_index, batch in pending
        }
        for future in as_completed(futures):
            spec, batch_index = futures[future]
            try:
                payload = future.result()
            except Exception as error:  # noqa: BLE001 - aggregate after preserving successes
                failures.append(
                    f"{spec.curator_id} batch {batch_index}: {type(error).__name__}: {error}"
                )
                continue
            path = _atomic_content_addressed_write(
                output_dir, f"curation-{spec.curator_id}-{batch_index:03d}", payload
            )
            artifacts.append(path)

    if failures:
        raise TaskCurationError(
            f"{len(failures)} curation batches failed; successful artifacts were retained: "
            + " | ".join(failures[:5])
        )

    documents = [json.loads(path.read_bytes()) for path in artifacts]
    documents.sort(key=lambda value: (value["curator"]["curator_id"], value["batch_index"]))
    expected_artifacts = sum(batches_by_curator.values())
    if len(documents) != expected_artifacts:
        raise TaskCurationError("curation did not produce one artifact per curator batch")
    usage = {
        key: sum(int(document["usage"][key]) for document in documents)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    estimated_cost = sum(
        (Decimal(str(document["estimated_cost_usd"])) for document in documents), Decimal(0)
    )
    summary: dict[str, Any] = {
        "schema_version": "flavourbench-bedrock-task-curation-summary-v1",
        "candidate_pool_sha256": pool["content_sha256"],
        "synthetic_tasks": 0,
        "curation_models": [
            {
                "curator_id": spec.curator_id,
                "backend": spec.backend,
                "provider": spec.provider,
                "model_id": spec.model_id,
                "provider_slug": spec.provider_slug,
                "batch_size": batch_size or spec.batch_size,
                "structured_output": spec.structured_output,
            }
            for spec in curators
        ],
        "counts": {
            "candidates": len(candidates),
            "batches_by_curator": batches_by_curator,
            "artifacts": len(documents),
            "judgments": sum(len(document["judgments"]) for document in documents),
        },
        "usage": usage,
        "estimated_cost_usd": format(estimated_cost, "f"),
        "reservation_cap_usd": format(cap_usd, "f"),
        "reservation_worst_case_usd": format(total_reservation, "f"),
        "curation_system_prompt": CURATION_SYSTEM_PROMPT,
        "curation_system_prompt_sha256": sha256_text(CURATION_SYSTEM_PROMPT),
        "artifact_sha256s": sorted(document["artifact_sha256"] for document in documents),
    }
    summary_path = _atomic_content_addressed_write(output_dir, "curation-summary", summary)
    return {**summary, "summary_path": str(summary_path)}


def load_curation_judgments(
    directory: Path,
    *,
    candidate_pool_sha256: str,
    allowed_artifact_sha256s: frozenset[str] | None = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    indexed: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted(directory.glob("curation-*.json")):
        value = json.loads(path.read_bytes())
        if value.get("schema_version") != CURATION_SCHEMA:
            continue
        artifact_sha = value.pop("artifact_sha256", None)
        if artifact_sha != sha256_json(value):
            raise TaskCurationError(f"curation artifact digest does not verify: {path.name}")
        if allowed_artifact_sha256s is not None and artifact_sha not in allowed_artifact_sha256s:
            continue
        if value.get("candidate_pool_sha256") != candidate_pool_sha256:
            continue
        curator = value.get("curator")
        judgments = value.get("judgments")
        if not isinstance(curator, Mapping) or not isinstance(judgments, list):
            raise TaskCurationError(f"curation artifact is malformed: {path.name}")
        curator_id = str(curator.get("curator_id") or "")
        if not curator_id:
            raise TaskCurationError("curation artifact has no curator identity")
        for judgment in judgments:
            if not isinstance(judgment, dict) or not isinstance(judgment.get("question_id"), int):
                raise TaskCurationError("curation artifact has an invalid judgment")
            question_id = int(judgment["question_id"])
            if curator_id in indexed.setdefault(question_id, {}):
                raise TaskCurationError("duplicate curator judgment for a question")
            indexed[question_id][curator_id] = judgment
    return indexed


def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels
    )
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def summarize_curation(
    *,
    pool: Mapping[str, Any],
    judgment_directory: Path,
    curation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if curation_summary.get("candidate_pool_sha256") != pool.get("content_sha256"):
        raise TaskCurationError("curation summary and candidate pool differ")
    artifact_sha256s = curation_summary.get("artifact_sha256s")
    if not isinstance(artifact_sha256s, list) or not artifact_sha256s:
        raise TaskCurationError("curation summary has no artifact manifest")
    indexed = load_curation_judgments(
        judgment_directory,
        candidate_pool_sha256=str(pool["content_sha256"]),
        allowed_artifact_sha256s=frozenset(str(value) for value in artifact_sha256s),
    )
    expected_curators = [spec.curator_id for spec in CURATORS]
    candidate_ids = [int(candidate["question_id"]) for candidate in pool["candidates"]]
    missing = [
        question_id
        for question_id in candidate_ids
        if set(indexed.get(question_id, {})) != set(expected_curators)
    ]
    if missing:
        raise TaskCurationError(f"{len(missing)} candidates lack both independent judgments")

    left_labels: list[str] = []
    right_labels: list[str] = []
    exact_family_agreement = 0
    inclusion_agreement = 0
    strict_consensus: dict[str, list[dict[str, Any]]] = {
        family: [] for family in sorted(ALLOWED_FAMILIES)
    }
    candidate_by_id = {int(candidate["question_id"]): candidate for candidate in pool["candidates"]}
    for question_id in candidate_ids:
        left = indexed[question_id][expected_curators[0]]
        right = indexed[question_id][expected_curators[1]]
        left_labels.append(str(left["family"]))
        right_labels.append(str(right["family"]))
        exact_family_agreement += left["family"] == right["family"]
        inclusion_agreement += left["include"] == right["include"]
        family = str(left["family"])
        strict = (
            left["include"]
            and right["include"]
            and family == right["family"]
            and family in ALLOWED_FAMILIES
            and left["specialist_risk"] == right["specialist_risk"] == "none"
            and left["self_contained"]
            and right["self_contained"]
            and min(left["family_fit"], right["family_fit"]) >= 3
            and min(left["difficulty"], right["difficulty"]) >= 2
            and min(left["specificity"], right["specificity"]) >= 2
            and (left["epicure_relevance"] + right["epicure_relevance"]) / 2 >= 2
        )
        if strict:
            candidate = candidate_by_id[question_id]
            mean_score = (
                sum(
                    left[key] + right[key]
                    for key in ("family_fit", "difficulty", "specificity", "epicure_relevance")
                )
                / 2
            )
            strict_consensus[family].append(
                {
                    "question_id": question_id,
                    "title": candidate["title"],
                    "recent": candidate["heuristics"]["recent"],
                    "mean_curation_score": mean_score,
                    "judgments": {
                        expected_curators[0]: left,
                        expected_curators[1]: right,
                    },
                }
            )
    for values in strict_consensus.values():
        values.sort(
            key=lambda item: (
                item["mean_curation_score"],
                item["recent"],
                item["question_id"],
            ),
            reverse=True,
        )
    total = len(candidate_ids)
    return {
        "schema_version": "flavourbench-real-task-curation-audit-v1",
        "candidate_pool_sha256": pool["content_sha256"],
        "synthetic_tasks": 0,
        "counts": {
            "candidates": total,
            "independent_judgments": total * len(expected_curators),
            "strict_consensus_by_family": {
                family: len(values) for family, values in strict_consensus.items()
            },
        },
        "agreement": {
            "exact_family_rate": exact_family_agreement / total,
            "inclusion_rate": inclusion_agreement / total,
            "family_cohen_kappa": _cohen_kappa(left_labels, right_labels),
        },
        "curator_ids": expected_curators,
        "strict_consensus": strict_consensus,
    }


def _prepare(args: argparse.Namespace) -> None:
    pool = build_candidate_pool(policy=SelectionPolicy())
    path = write_candidate_pool(pool, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(path),
                "content_sha256": pool["content_sha256"],
                "counts": pool["counts"],
                "synthetic_tasks": 0,
            },
            indent=2,
        )
    )


def _curate(args: argparse.Namespace) -> None:
    if args.confirmation != CURATION_CONFIRMATION:
        raise TaskCurationError(f"paid curation requires --confirmation {CURATION_CONFIRMATION}")
    settings = BedrockLaneSettings.from_environ()
    if not settings.enabled or not settings.live_authorized:
        raise TaskCurationError("Bedrock curation requires enabled and live-authorized settings")
    cap = Decimal(str(args.cap_usd))
    if cap <= 0 or cap > settings.hard_cap_usd:
        raise TaskCurationError(
            "curation cap must be positive and within the authorized Bedrock cap"
        )
    pool = load_candidate_pool(args.pool)
    clients = create_boto3_clients(settings)
    openrouter_client: httpx.Client | None = None
    if any(spec.backend == "openrouter" for spec in CURATORS):
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
            "FLAVOURBENCH_OPENROUTER_API_KEY"
        )
        if not api_key:
            raise TaskCurationError("OpenRouter fallback curator requires an API key")
        base_url = (
            os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("FLAVOURBENCH_OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://epicure.kaikaku.ai/flavourbench",
            "X-Title": "Epicure FlavourBench task curation",
        }
        if "gateway.ai.cloudflare.com" in base_url:
            gateway_token = os.environ.get("CLOUDFLARE_AI_GATEWAY_TOKEN") or ""
            if not gateway_token:
                raise TaskCurationError("Cloudflare-routed curation requires its gateway token")
            headers.update(
                {
                    "cf-aig-authorization": f"Bearer {gateway_token}",
                    "cf-aig-skip-cache": "true",
                    "cf-aig-collect-log-payload": "false",
                }
            )
        openrouter_client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=180,
        )
    try:
        summary = run_curation(
            pool=pool,
            output_dir=args.output_dir,
            runtime=clients.runtime,
            openrouter_client=openrouter_client,
            batch_size=args.batch_size or None,
            max_workers=args.max_workers,
            cap_usd=cap,
        )
    finally:
        if openrouter_client is not None:
            openrouter_client.close()
    print(json.dumps(summary, indent=2))


def _summarize(args: argparse.Namespace) -> None:
    pool = load_candidate_pool(args.pool)
    summary = json.loads(args.summary.read_bytes())
    audit = summarize_curation(
        pool=pool,
        judgment_directory=args.curation_dir,
        curation_summary=summary,
    )
    path = _atomic_content_addressed_write(args.output_dir, "curation-audit", audit)
    print(
        json.dumps(
            {
                "output": str(path),
                "counts": audit["counts"],
                "agreement": audit["agreement"],
            },
            indent=2,
        )
    )


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-curate-real-tasks")
    parser = argparse.ArgumentParser(description="Curate real human FlavourBench tasks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="download the licensed human candidate pool")
    prepare.add_argument("--output-dir", type=Path, default=Path("data/season0/source"))
    prepare.set_defaults(handler=_prepare)

    curate = subparsers.add_parser("curate", help="run two independent real Bedrock curators")
    curate.add_argument("--pool", type=Path, required=True)
    curate.add_argument("--output-dir", type=Path, default=Path("data/season0/curation"))
    curate.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="positive override for all curators; 0 uses frozen per-curator sizes",
    )
    curate.add_argument("--max-workers", type=int, default=4)
    curate.add_argument("--cap-usd", type=str, default="30")
    curate.add_argument("--confirmation", required=True)
    curate.set_defaults(handler=_curate)

    summarize = subparsers.add_parser("summarize", help="audit inter-curator agreement")
    summarize.add_argument("--pool", type=Path, required=True)
    summarize.add_argument("--curation-dir", type=Path, required=True)
    summarize.add_argument("--summary", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, default=Path("data/season0/curation"))
    summarize.set_defaults(handler=_summarize)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    run()
