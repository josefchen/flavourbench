"""Source-reconstructing v4 qualification for one OpenRouter + Epicure pair.

The qualification is deliberately narrow: one fresh, permanently unranked
Epicure-off/on pair may establish that the current response-envelope adapter can
execute the frozen low-effort protocol on one exact fixed route. It is never a
quality observation and it does not replace the prespecified sensitivity study.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .config import get_settings
from .execution_policy import ExecutionPolicy
from .live_smoke import (
    CONFIRMATION,
    REQUIRED_ENDPOINT_PARAMETERS,
    endpoint_execution_contract,
    endpoint_execution_contract_sha256,
    live_smoke,
)
from .reasoning_effort_route_recovery import (
    V3_AUDIT_SCHEMA_VERSION,
    verify_v3_route_closure,
    verify_v3_route_plan,
)
from .run_journal import load_run_journal

CATALOG_SCHEMA_VERSION = "flavourbench-response-envelope-v4-catalog-snapshot-v1"
PLAN_SCHEMA_VERSION = "flavourbench-response-envelope-route-v4-plan-v1"
RECEIPT_SCHEMA_VERSION = "flavourbench-response-envelope-route-v4-execution-v1"
AUDIT_SCHEMA_VERSION = "flavourbench-response-envelope-route-v4-audit-v1"
CLOSURE_SCHEMA_VERSION = "flavourbench-response-envelope-route-v4-closure-v1"

MODEL_ID = "deepseek/deepseek-v4-flash-0731"
CANONICAL_MODEL_SLUG = "deepseek/deepseek-v4-flash-20260731"
PROVIDER_TAG = "deepinfra/fp4"
PROVIDER_NAME = "DeepInfra"
TASK_ID = "fb-s0-substitution-003"
PROMPT_SHA256 = "4eb9cf16da129b1379d978518a5b28d50f3524eb3ef0b987489b559b563cc03f"
EXPECTED_PROVIDER_CONTROLS = {
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
    "only": [PROVIDER_TAG],
    "max_price": {"prompt": 0.09, "completion": 0.18},
}
V4_NAMESPACE = uuid.UUID("9ea50f27-1034-4819-9ab7-19c6579035a6")
SOURCE_FILES = (
    "flavourbench/src/flavourbench/provider.py",
    "flavourbench/src/flavourbench/live_smoke.py",
    "flavourbench/src/flavourbench/response_envelope_route_v4.py",
    "flavourbench/src/flavourbench/frontier_coverage_repair_executor.py",
    "flavourbench/src/flavourbench/run_journal.py",
    "flavourbench/src/flavourbench/execution_policy.py",
    "flavourbench/src/flavourbench/mcp_client.py",
    "flavourbench/src/flavourbench/protocol_contract.py",
    "flavourbench/requirements.lock",
    "flavourbench/Dockerfile",
)
POST_EXECUTION_VERIFIER_FILES = frozenset(
    {"flavourbench/src/flavourbench/response_envelope_route_v4.py"}
)


class V4RouteError(RuntimeError):
    """A v4 qualification input, execution, or receipt fails closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise V4RouteError(f"{field} is not a decimal") from error
    if not result.is_finite() or result < 0:
        raise V4RouteError(f"{field} must be finite and non-negative")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise V4RouteError(f"input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V4RouteError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise V4RouteError(f"expected a JSON object: {path}")
    return value


def _artifact_verifies(document: object, schema_version: str) -> bool:
    if not isinstance(document, Mapping) or document.get("schema_version") != schema_version:
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return _is_sha256(digest) and _sha256(unhashed) == digest


def _live_artifact_verifies(document: object) -> bool:
    if not isinstance(document, Mapping) or document.get("schema_version") != (
        "flavourbench-live-smoke-v1"
    ):
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    # live_smoke predates this verifier and hashes JSON with the default
    # ensure_ascii=True. Reproduce that historical serializer exactly.
    live_digest = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _is_sha256(digest) and live_digest == digest


def _source_document_digest(document: Mapping[str, Any], path: Path) -> str:
    artifact_digest = document.get("artifact_sha256")
    if _is_sha256(artifact_digest):
        unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
        if _sha256(unhashed) != artifact_digest:
            raise V4RouteError(f"source artifact content address failed: {path}")
        return str(artifact_digest)
    address = document.get("content_address")
    if isinstance(address, Mapping) and _is_sha256(address.get("digest")):
        unhashed = {key: value for key, value in document.items() if key != "content_address"}
        if (
            address.get("algorithm") != "sha256"
            or address.get("uri") != f"sha256:{address['digest']}"
            or _sha256(unhashed) != address["digest"]
        ):
            raise V4RouteError(f"source manifest content address failed: {path}")
        return str(address["digest"])
    return _file_sha256(path)


def _write_artifact(output_dir: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise V4RouteError("content-addressed output conflict")
        return path
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved_root = repo_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise V4RouteError(f"artifact path escapes the evaluation repository: {value}") from error
    return resolved


def _relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _source_bundle(repo_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = _repo_path(repo_root, relative)
        if path.is_symlink() or not path.is_file():
            raise V4RouteError(f"source binding is not a regular file: {relative}")
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        )
    return {"files": records, "bundle_sha256": _sha256(records)}


def _policy_from_manifest(manifest: Mapping[str, Any]) -> ExecutionPolicy:
    document = ((manifest.get("run_design") or {}).get("execution_policy") or {})
    limits = document.get("limits") or {}
    decoding = document.get("decoding") or {}
    reasoning = document.get("reasoning") or {}
    forecast = document.get("cost_forecast") or {}
    policy = ExecutionPolicy(
        max_output_tokens=int(limits["max_output_tokens"]),
        max_tool_rounds=int(limits["max_tool_rounds"]),
        max_tool_result_bytes=int(limits["max_tool_result_bytes"]),
        max_cumulative_tool_result_bytes=int(limits["max_cumulative_tool_result_bytes"]),
        max_tool_calls_per_round=int(limits["max_tool_calls_per_round"]),
        max_tool_calls_total=int(limits["max_tool_calls_total"]),
        max_provider_attempts=int(limits["max_provider_attempts"]),
        decoding_temperature=float(decoding["temperature"]),
        decoding_top_p=float(decoding["top_p"]),
        decoding_seed=int(decoding["seed"]),
        tool_argument_repair_turns=int(limits["tool_argument_repair_turns"]),
        approximate_non_user_prompt_bytes=int(
            forecast["approximate_non_user_prompt_bytes"]
        ),
        conservative_bytes_per_token=int(forecast["conservative_bytes_per_token"]),
        pair_arm_scheduling="sequential",
        final_response_mode=str(document["final_response_mode"]),
        max_intermediate_tokens=int(limits["max_intermediate_tokens"]),
        required_tool_contract_max_intermediate_tokens=int(
            limits["required_tool_contract_max_intermediate_tokens"]
        ),
        matched_planning=bool(document["matched_planning"]),
        evidence_protocol=str(document["evidence_protocol"]),
        intermediate_reasoning_effort=reasoning.get("intermediate_effort"),
        final_reasoning_effort=reasoning.get("final_effort"),
        required_tool_contract_protocol=str(document["required_tool_contract_protocol"]),
        tool_catalog_bytes_bound=int(forecast["tool_catalog_bytes_bound"]),
        epicure_on_tool_required=bool(document["epicure_on_tool_required"]),
    )
    policy.validate()
    return policy


def _model_record(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    records = manifest.get("models")
    if not isinstance(records, list):
        raise V4RouteError("source manifest has no model records")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and (record.get("model") or {}).get("id") == MODEL_ID
    ]
    if len(matches) != 1:
        raise V4RouteError("source manifest lacks one exact DeepSeek V4 Flash route")
    return matches[0]


def _attempt_slots(run_id: str, route_cell_id: str, freeze_nonce: str) -> list[dict[str, Any]]:
    arms = {
        condition: f"{run_id}:{condition}"
        for condition in ("epicure_off", "epicure_on")
    }
    coordinates: list[tuple[str, str, int]] = []
    for phase in ("planning", "evidence_decision", "final"):
        for attempt in range(2):
            coordinates.append((arms["epicure_off"], phase, attempt))
    for phase in ("planning", "tool_round_0", "tool_round_1", "tool_round_2", "final"):
        for attempt in range(2):
            coordinates.append((arms["epicure_on"], phase, attempt))
    coordinates.append((arms["epicure_on"], "mcp_session", 0))
    for round_index in range(3):
        for call_index in range(6):
            coordinates.append(
                (arms["epicure_on"], f"mcp_tool_{round_index}_{call_index}", 0)
            )
    return [
        {
            "arm_id": arm_id,
            "phase": phase,
            "attempt_index": attempt_index,
            "attempt_id": str(
                uuid.uuid5(
                    V4_NAMESPACE,
                    (
                        f"{freeze_nonce}:{route_cell_id}:{arm_id}:{phase}:"
                        f"{attempt_index}"
                    ),
                )
            ),
        }
        for arm_id, phase, attempt_index in coordinates
    ]


def _prior_inventory(
    *,
    v3_plan: Mapping[str, Any],
    v3_audit: Mapping[str, Any],
    v3_closure: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    closed = v3_plan["closed_identifiers"]
    work_ids = {
        str(value)
        for revision in ("v1", "v2")
        for value in closed[revision]["work_item_ids"]
    }
    attempt_ids = {
        str(value)
        for revision in ("v1", "v2")
        for value in closed[revision]["attempt_ids"]
    }
    generation_ids = {
        str(value)
        for revision in ("v1", "v2")
        for value in closed[revision]["generation_ids"]
        if value
    }
    v3_ids = v3_closure["closed_identifiers"]
    work_ids.update(str(value) for value in v3_ids["work_item_ids"])
    attempt_ids.update(str(value) for value in v3_ids["attempt_ids"])
    generation_ids.update(str(value) for value in v3_ids["generation_ids"] if value)
    run_ids: set[str] = set()
    arm_ids = {str(value) for value in v3_ids.get("arm_ids") or []}
    request_key_hashes: set[str] = set()
    for source in v3_audit.get("source_artifacts") or []:
        if not isinstance(source, Mapping):
            continue
        matching = [
            item
            for item in v3_audit.get("variant_audits") or []
            if isinstance(item, Mapping)
            and item.get("source_artifact_sha256") == source.get("source_artifact_sha256")
            and item.get("source_path")
        ]
        for item in matching:
            path = _repo_path(repo_root, str(item["source_path"]))
            document = _regular_json(path)
            if document.get("artifact_sha256") != source.get("source_artifact_sha256"):
                raise V4RouteError("v3 source artifact differs from its audit")
            run_id = str(document.get("run_id") or "")
            if run_id:
                run_ids.add(run_id)
            for event in document.get("provider_attempt_events") or []:
                if not isinstance(event, Mapping):
                    continue
                arm_id = str(event.get("arm_id") or "")
                request_key = str(event.get("request_key_sha256") or "")
                if arm_id:
                    arm_ids.add(arm_id)
                if _is_sha256(request_key):
                    request_key_hashes.add(request_key)
    inventory = {
        "work_item_ids": sorted(work_ids),
        "arm_ids": sorted(arm_ids),
        "run_ids": sorted(run_ids),
        "attempt_ids": sorted(attempt_ids),
        "generation_ids": sorted(generation_ids),
        "request_key_sha256s": sorted(request_key_hashes),
    }
    return {**inventory, "inventory_sha256": _sha256(inventory)}


def build_catalog_snapshot(
    *,
    model_document: Mapping[str, Any],
    endpoint_document: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    model = model_document.get("data")
    endpoint_data = endpoint_document.get("data")
    if not isinstance(model, Mapping) or not isinstance(endpoint_data, Mapping):
        raise V4RouteError("OpenRouter catalog responses lack data objects")
    endpoints = endpoint_data.get("endpoints")
    matches = [
        item
        for item in endpoints or []
        if isinstance(item, Mapping) and item.get("tag") == PROVIDER_TAG
    ]
    if len(matches) != 1:
        raise V4RouteError("fresh catalog lacks one exact DeepInfra fp4 endpoint")
    endpoint = matches[0]
    model_parameters = sorted(str(value) for value in model.get("supported_parameters") or [])
    endpoint_parameters = sorted(
        str(value) for value in endpoint.get("supported_parameters") or []
    )
    required = sorted({*REQUIRED_ENDPOINT_PARAMETERS, "reasoning"})
    reasoning = model.get("reasoning") if isinstance(model.get("reasoning"), Mapping) else {}
    efforts = sorted(str(value) for value in reasoning.get("supported_efforts") or [])
    if (
        model.get("id") != MODEL_ID
        or model.get("canonical_slug") != CANONICAL_MODEL_SLUG
        or endpoint.get("model_id") != MODEL_ID
        or endpoint.get("provider_name") != PROVIDER_NAME
        or any(value not in endpoint_parameters for value in required)
        or "low" not in efforts
    ):
        raise V4RouteError("fresh catalog route does not satisfy the v4 contract")
    safe_endpoint = endpoint_execution_contract(dict(endpoint))
    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "observed_at": observed_at,
        "official_sources": {
            "model_url": (
                "https://openrouter.ai/api/v1/model/deepseek/"
                "deepseek-v4-flash-0731"
            ),
            "endpoint_url": (
                "https://openrouter.ai/api/v1/models/deepseek/"
                "deepseek-v4-flash-0731/endpoints"
            ),
            "model_document_sha256": _sha256(model_document),
            "endpoint_document_sha256": _sha256(endpoint_document),
        },
        "selected_route": {
            "model_id": MODEL_ID,
            "canonical_model_slug": CANONICAL_MODEL_SLUG,
            "provider_tag": PROVIDER_TAG,
            "provider_name": PROVIDER_NAME,
            "model_supported_parameters": model_parameters,
            "endpoint_supported_parameters": endpoint_parameters,
            "required_parameters": required,
            "supported_reasoning_efforts": efforts,
            "default_reasoning_effort": reasoning.get("default_effort"),
            "endpoint_execution_contract": safe_endpoint,
            "endpoint_execution_contract_sha256": endpoint_execution_contract_sha256(
                dict(endpoint)
            ),
            "pricing": endpoint.get("pricing") or {},
            "status": endpoint.get("status"),
            "uptime_last_1d": endpoint.get("uptime_last_1d"),
        },
        "provider_calls_made": False,
        "epicure_calls_made": False,
        "quality_observations": 0,
        "rank_eligible": False,
    }
    return payload


def verify_catalog_snapshot(document: object) -> bool:
    if not _artifact_verifies(document, CATALOG_SCHEMA_VERSION):
        return False
    assert isinstance(document, Mapping)
    route = document.get("selected_route")
    if not isinstance(route, Mapping):
        return False
    required = route.get("required_parameters")
    endpoint_parameters = route.get("endpoint_supported_parameters")
    return bool(
        route.get("model_id") == MODEL_ID
        and route.get("canonical_model_slug") == CANONICAL_MODEL_SLUG
        and route.get("provider_tag") == PROVIDER_TAG
        and route.get("provider_name") == PROVIDER_NAME
        and isinstance(required, list)
        and isinstance(endpoint_parameters, list)
        and set(required) <= set(endpoint_parameters)
        and "low" in (route.get("supported_reasoning_efforts") or [])
        and _is_sha256(route.get("endpoint_execution_contract_sha256"))
        and document.get("provider_calls_made") is False
        and document.get("epicure_calls_made") is False
        and document.get("quality_observations") == 0
        and document.get("rank_eligible") is False
    )


async def refresh_catalog() -> dict[str, Any]:
    base = "https://openrouter.ai/api/v1"
    headers = {"Accept": "application/json"}
    settings = get_settings()
    if settings.openrouter_api_key:
        headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=60) as client:
        author, slug = MODEL_ID.split("/", 1)
        model_response, endpoint_response = await asyncio.gather(
            client.get(f"model/{quote(author)}/{quote(slug, safe=':')}"),
            client.get(f"models/{quote(author)}/{quote(slug, safe=':')}/endpoints"),
        )
        model_response.raise_for_status()
        endpoint_response.raise_for_status()
        model_document = model_response.json()
        endpoint_document = endpoint_response.json()
    if not isinstance(model_document, dict) or not isinstance(endpoint_document, dict):
        raise V4RouteError("OpenRouter returned a non-object catalog response")
    return build_catalog_snapshot(
        model_document=model_document,
        endpoint_document=endpoint_document,
        observed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def build_v4_plan(
    *,
    repo_root: Path,
    v3_plan_path: Path,
    v3_audit_path: Path,
    v3_closure_path: Path,
    route_registry_path: Path,
    catalog_audit_path: Path,
    fresh_catalog_path: Path,
    high_resource_manifest_path: Path,
    task_dossier_path: Path,
    epicure_attestation_path: Path,
    coverage_execution_plan_path: Path,
    freeze_nonce: str | None = None,
) -> dict[str, Any]:
    inputs = {
        name: _regular_json(path)
        for name, path in {
            "v3_plan": v3_plan_path,
            "v3_audit": v3_audit_path,
            "v3_closure": v3_closure_path,
            "route_registry": route_registry_path,
            "catalog_audit": catalog_audit_path,
            "fresh_catalog": fresh_catalog_path,
            "manifest": high_resource_manifest_path,
            "task_dossier": task_dossier_path,
            "epicure": epicure_attestation_path,
            "coverage_execution": coverage_execution_plan_path,
        }.items()
    }
    if not verify_v3_route_plan(inputs["v3_plan"]):
        raise V4RouteError("v3 plan does not verify")
    if not _artifact_verifies(inputs["v3_audit"], V3_AUDIT_SCHEMA_VERSION):
        raise V4RouteError("v3 audit does not verify")
    if not verify_v3_route_closure(inputs["v3_closure"]):
        raise V4RouteError("v3 closure does not verify")
    if not verify_catalog_snapshot(inputs["fresh_catalog"]):
        raise V4RouteError("fresh route snapshot does not verify")
    if inputs["v3_closure"].get("v3_route_plan_sha256") != inputs["v3_plan"].get(
        "artifact_sha256"
    ):
        raise V4RouteError("v3 closure is not bound to the v3 plan")
    registry_matches = [
        item
        for item in inputs["route_registry"].get("models") or []
        if isinstance(item, Mapping)
        and item.get("canonical_model_slug") == CANONICAL_MODEL_SLUG
    ]
    if (
        len(registry_matches) != 1
        or registry_matches[0].get("provider_endpoint") != PROVIDER_TAG
        or registry_matches[0].get("contract_status") != "passed_unranked"
        or registry_matches[0].get("quality_observations") != 0
    ):
        raise V4RouteError("route registry does not contain the exact unranked route")
    catalog_matches = [
        item
        for item in inputs["catalog_audit"].get("models") or []
        if isinstance(item, Mapping) and item.get("model_id") == MODEL_ID
    ]
    if len(catalog_matches) != 1 or catalog_matches[0].get("status") != (
        "freshness_contract_passed"
    ):
        raise V4RouteError("catalog audit does not pass the selected route")
    model_record = _model_record(inputs["manifest"])
    model = model_record["model"]
    endpoint = model_record["endpoint"]
    if (
        model.get("canonical_slug") != CANONICAL_MODEL_SLUG
        or endpoint.get("tag") != PROVIDER_TAG
        or endpoint.get("provider_name") != PROVIDER_NAME
        or "low" not in ((model.get("reasoning") or {}).get("supported_efforts") or [])
    ):
        raise V4RouteError("high-resource manifest route differs from v4 selection")
    task_matches = [
        item
        for item in inputs["task_dossier"].get("tasks") or []
        if isinstance(item, Mapping) and item.get("task_id") == TASK_ID
    ]
    if (
        len(task_matches) != 1
        or task_matches[0].get("prompt_sha256") != PROMPT_SHA256
        or inputs["task_dossier"].get("counts", {}).get("synthetic_tasks") != 0
    ):
        raise V4RouteError("task dossier does not contain the frozen human-authored task")
    task = task_matches[0]
    epicure = inputs["epicure"]
    if any(
        not _is_sha256(epicure.get(field))
        for field in ("bundle_sha256", "application_sha256")
    ) or not epicure.get("release_id"):
        raise V4RouteError("Epicure runtime attestation is incomplete")
    tool_sha = "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd"
    policy = _policy_from_manifest(inputs["manifest"])
    if (
        policy.final_response_mode != "plain_text"
        or policy.evidence_protocol != "matched_evidence_v2"
        or not policy.matched_planning
        or policy.intermediate_reasoning_effort != "low"
        or policy.final_reasoning_effort != "low"
        or not policy.epicure_on_tool_required
        or policy.max_provider_attempts != 2
    ):
        raise V4RouteError("derived diagnostic policy differs from the frozen low envelope")
    prior = _prior_inventory(
        v3_plan=inputs["v3_plan"],
        v3_audit=inputs["v3_audit"],
        v3_closure=inputs["v3_closure"],
        repo_root=repo_root,
    )
    nonce = freeze_nonce or str(uuid.uuid4())
    try:
        uuid.UUID(nonce)
    except ValueError as error:
        raise V4RouteError("freeze nonce must be a UUID") from error
    source_bundle = _source_bundle(repo_root)
    source_digests = {
        name: _source_document_digest(document, path)
        for name, (document, path) in {
            "v3_plan": (inputs["v3_plan"], v3_plan_path),
            "v3_audit": (inputs["v3_audit"], v3_audit_path),
            "v3_closure": (inputs["v3_closure"], v3_closure_path),
            "route_registry": (inputs["route_registry"], route_registry_path),
            "catalog_audit": (inputs["catalog_audit"], catalog_audit_path),
            "fresh_catalog": (inputs["fresh_catalog"], fresh_catalog_path),
            "high_resource_manifest": (inputs["manifest"], high_resource_manifest_path),
            "task_dossier": (inputs["task_dossier"], task_dossier_path),
            "epicure_attestation": (inputs["epicure"], epicure_attestation_path),
            "coverage_execution": (
                inputs["coverage_execution"],
                coverage_execution_plan_path,
            ),
        }.items()
    }
    if any(not _is_sha256(value) for value in source_digests.values()):
        raise V4RouteError("one source artifact lacks a content digest")
    route_coordinate = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "freeze_nonce": nonce,
        "model_id": MODEL_ID,
        "canonical_model_slug": CANONICAL_MODEL_SLUG,
        "provider_tag": PROVIDER_TAG,
        "provider_name": PROVIDER_NAME,
        "task_id": TASK_ID,
        "prompt_sha256": PROMPT_SHA256,
        "execution_policy_sha256": policy.sha256,
        "epicure_release_id": epicure["release_id"],
        "epicure_bundle_sha256": epicure["bundle_sha256"],
        "epicure_application_sha256": epicure["application_sha256"],
        "epicure_tool_schema_sha256": tool_sha,
        "source_digests": source_digests,
        "source_bundle_sha256": source_bundle["bundle_sha256"],
        "prior_inventory_sha256": prior["inventory_sha256"],
    }
    route_cell_id = _sha256(route_coordinate)
    work_item_id = _sha256({"route_cell_id": route_cell_id, "role": "v4-diagnostic-pair"})
    run_id = str(uuid.uuid5(V4_NAMESPACE, f"run:{nonce}:{route_cell_id}:{work_item_id}"))
    arms = [f"{run_id}:epicure_off", f"{run_id}:epicure_on"]
    slots = _attempt_slots(run_id, route_cell_id, nonce)
    planned_attempt_ids = {str(item["attempt_id"]) for item in slots}
    if (
        work_item_id in set(prior["work_item_ids"])
        or run_id in set(prior["run_ids"])
        or set(arms) & set(prior["arm_ids"])
        or planned_attempt_ids & set(prior["attempt_ids"])
        or len(planned_attempt_ids) != len(slots)
    ):
        raise V4RouteError("v4 identifier pool overlaps or duplicates a closed identifier")
    forecast = _decimal(
        model_record["forecast"]["model_block_worst_case_usd"], field="forecast"
    ) / Decimal(str(model_record["forecast"]["pairs"]))
    reserve = Decimal("0.05")
    if forecast > reserve:
        raise V4RouteError("route forecast exceeds the fixed v4 reserve")
    coverage_budget = inputs["coverage_execution"].get("budget") or {}
    current_exposure = _decimal(
        coverage_budget.get("current_total_exposure_usd"), field="current exposure"
    )
    admission_ceiling = _decimal(
        coverage_budget.get("admission_ceiling_usd"), field="admission ceiling"
    )
    projected = current_exposure + reserve
    paths = {
        name: _relative(repo_root, path)
        for name, path in {
            "v3_plan": v3_plan_path,
            "v3_audit": v3_audit_path,
            "v3_closure": v3_closure_path,
            "route_registry": route_registry_path,
            "catalog_audit": catalog_audit_path,
            "fresh_catalog": fresh_catalog_path,
            "high_resource_manifest": high_resource_manifest_path,
            "task_dossier": task_dossier_path,
            "epicure_attestation": epicure_attestation_path,
            "coverage_execution": coverage_execution_plan_path,
        }.items()
    }
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "record_role": "one_pair_unranked_response_envelope_qualification",
        "freeze_nonce": nonce,
        "source_artifacts": {
            name: {"path": paths[name], "sha256": source_digests[name]}
            for name in sorted(paths)
        },
        "source_code": source_bundle,
        "route_coordinate": route_coordinate,
        "route_cell_id": route_cell_id,
        "route": {
            "model_id": MODEL_ID,
            "canonical_model_slug": CANONICAL_MODEL_SLUG,
            "provider_tag": PROVIDER_TAG,
            "provider_name": PROVIDER_NAME,
            "endpoint_execution_contract_sha256": inputs["fresh_catalog"][
                "selected_route"
            ]["endpoint_execution_contract_sha256"],
            "required_parameters": inputs["fresh_catalog"]["selected_route"][
                "required_parameters"
            ],
            "supported_reasoning_efforts": inputs["fresh_catalog"]["selected_route"][
                "supported_reasoning_efforts"
            ],
            "provider_controls": EXPECTED_PROVIDER_CONTROLS,
            "provider_controls_sha256": _sha256(EXPECTED_PROVIDER_CONTROLS),
            "zdr_requested": False,
        },
        "task": {
            "task_id": TASK_ID,
            "family": task["family"],
            "prompt": task["prompt"],
            "prompt_sha256": PROMPT_SHA256,
            "source_class": "licensed_real_human_authored_public_question",
            "synthetic": False,
            "rank_eligible": False,
        },
        "epicure": {
            "release_id": epicure["release_id"],
            "bundle_sha256": epicure["bundle_sha256"],
            "application_sha256": epicure["application_sha256"],
            "tool_schema_sha256": tool_sha,
            "public_reconstruction_complete": False,
            "qualification_requires_exact_private_runtime_attestation": True,
        },
        "execution_policy": policy.document(),
        "execution_policy_sha256": policy.sha256,
        "work": {
            "work_item_id": work_item_id,
            "run_id": run_id,
            "arm_ids": arms,
            "conditions": ["epicure_off", "epicure_on"],
            "attempt_slots": slots,
            "matched_pairs": 1,
            "response_arms": 2,
            "sequential_arms": True,
            "maximum_external_generation_attempts": sum(
                1 for item in slots if not str(item["phase"]).startswith("mcp_")
            ),
            "maximum_mcp_attempts": sum(
                1 for item in slots if str(item["phase"]).startswith("mcp_")
            ),
            "retry_limit_per_provider_phase": 2,
        },
        "prior_closed_identifiers": prior,
        "freshness": {
            "work_item_overlap": [],
            "run_id_overlap": [],
            "arm_id_overlap": [],
            "attempt_id_overlap": [],
            "generation_ids_prefrozen": False,
        },
        "budget": {
            "currency": "USD",
            "forecast_worst_case_usd": _decimal_text(forecast),
            "reserved_worst_case_usd": _decimal_text(reserve),
            "hard_v4_limit_usd": "5",
            "current_total_exposure_usd": _decimal_text(current_exposure),
            "projected_total_exposure_usd": _decimal_text(projected),
            "admission_ceiling_usd": _decimal_text(admission_ceiling),
            "admitted": projected <= admission_ceiling and reserve < Decimal("5"),
        },
        "acceptance": {
            "required_usable_pairs": 1,
            "required_usable_arms": 2,
            "minimum_successful_epicure_calls": 1,
            "minimum_final_characters_per_arm": 100,
            "minimum_final_words_per_arm": 20,
            "maximum_identity_mismatches": 0,
            "maximum_unreconciled_generations": 0,
            "maximum_unsafe_or_uncertain_attempts": 0,
            "diagnostic_outputs_reused_in_coverage": False,
            "all_predicates_required": True,
        },
        "preflight": {
            "decision": (
                "ready_for_one_v4_pair" if projected <= admission_ceiling else "blocked"
            ),
            "provider_calls_made": False,
            "epicure_calls_made": False,
        },
        "claim_boundary": {
            "route_smoke_only": True,
            "quality_observations": 0,
            "synthetic_or_model_judge_evidence": False,
            "official": False,
            "rank_eligible": False,
            "sensitivity_effect_estimable": False,
            "coverage_outputs_reused": False,
        },
    }


def _rederive_plan_identifiers(
    plan: Mapping[str, Any],
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    route_coordinate = plan.get("route_coordinate")
    if not isinstance(route_coordinate, Mapping):
        raise V4RouteError("plan route coordinate is missing")
    route_cell_id = _sha256(route_coordinate)
    work_item_id = _sha256({"route_cell_id": route_cell_id, "role": "v4-diagnostic-pair"})
    nonce = str(plan.get("freeze_nonce") or "")
    run_id = str(uuid.uuid5(V4_NAMESPACE, f"run:{nonce}:{route_cell_id}:{work_item_id}"))
    arms = [f"{run_id}:epicure_off", f"{run_id}:epicure_on"]
    return route_cell_id, work_item_id, arms, _attempt_slots(run_id, route_cell_id, nonce)


def verify_v4_plan(
    document: object,
    *,
    repo_root: Path | None = None,
    require_current_sources: bool = False,
) -> bool:
    if not _artifact_verifies(document, PLAN_SCHEMA_VERSION):
        return False
    assert isinstance(document, Mapping)
    try:
        route_cell_id, work_item_id, arms, slots = _rederive_plan_identifiers(document)
        budget = document["budget"]
        projected = _decimal(budget["projected_total_exposure_usd"], field="projected")
        ceiling = _decimal(budget["admission_ceiling_usd"], field="ceiling")
        reserve = _decimal(budget["reserved_worst_case_usd"], field="reserve")
    except (KeyError, TypeError, V4RouteError, ValueError):
        return False
    route = document.get("route") or {}
    work = document.get("work") or {}
    prior = document.get("prior_closed_identifiers") or {}
    claims = document.get("claim_boundary") or {}
    slot_ids = [str(item.get("attempt_id") or "") for item in slots]
    stored_slots = work.get("attempt_slots")
    valid = bool(
        route_cell_id == document.get("route_cell_id")
        and work_item_id == work.get("work_item_id")
        and arms == work.get("arm_ids")
        and slots == stored_slots
        and len(slot_ids) == len(set(slot_ids))
        and not set(slot_ids) & set(prior.get("attempt_ids") or [])
        and work_item_id not in set(prior.get("work_item_ids") or [])
        and work.get("run_id") not in set(prior.get("run_ids") or [])
        and not set(arms) & set(prior.get("arm_ids") or [])
        and route.get("model_id") == MODEL_ID
        and route.get("canonical_model_slug") == CANONICAL_MODEL_SLUG
        and route.get("provider_tag") == PROVIDER_TAG
        and route.get("provider_name") == PROVIDER_NAME
        and route.get("provider_controls") == EXPECTED_PROVIDER_CONTROLS
        and route.get("provider_controls_sha256") == _sha256(EXPECTED_PROVIDER_CONTROLS)
        and route.get("zdr_requested") is False
        and "low" in (route.get("supported_reasoning_efforts") or [])
        and document.get("task", {}).get("prompt_sha256") == PROMPT_SHA256
        and document.get("task", {}).get("synthetic") is False
        and document.get("epicure", {}).get("public_reconstruction_complete") is False
        and document.get("execution_policy", {}).get("reasoning", {}).get(
            "intermediate_effort"
        )
        == "low"
        and document.get("execution_policy", {}).get("reasoning", {}).get("final_effort")
        == "low"
        and document.get("execution_policy", {}).get("pair_arm_scheduling") == "sequential"
        and work.get("matched_pairs") == 1
        and work.get("response_arms") == 2
        and reserve < Decimal("5")
        and projected <= ceiling
        and budget.get("admitted") is True
        and document.get("preflight", {}).get("provider_calls_made") is False
        and document.get("preflight", {}).get("epicure_calls_made") is False
        and claims.get("route_smoke_only") is True
        and claims.get("quality_observations") == 0
        and claims.get("synthetic_or_model_judge_evidence") is False
        and claims.get("official") is False
        and claims.get("rank_eligible") is False
        and claims.get("coverage_outputs_reused") is False
    )
    if not valid or not require_current_sources:
        return valid
    if repo_root is None:
        return False
    try:
        current = _source_bundle(repo_root)
        for reference in (document.get("source_artifacts") or {}).values():
            if not isinstance(reference, Mapping):
                return False
            source_path = _repo_path(repo_root, str(reference.get("path") or ""))
            source_document = _regular_json(source_path)
            if _source_document_digest(source_document, source_path) != reference.get(
                "sha256"
            ):
                return False
    except V4RouteError:
        return False
    stored = document.get("source_code")
    if not isinstance(stored, Mapping):
        return False
    current_files = current.get("files")
    stored_files = stored.get("files")
    if not isinstance(current_files, list) or not isinstance(stored_files, list):
        return False
    current_by_path = {
        str(item.get("path") or ""): item
        for item in current_files
        if isinstance(item, Mapping)
    }
    stored_by_path = {
        str(item.get("path") or ""): item
        for item in stored_files
        if isinstance(item, Mapping)
    }
    return set(current_by_path) == set(stored_by_path) and all(
        current_by_path[path] == stored_by_path[path]
        for path in current_by_path
        if path not in POST_EXECUTION_VERIFIER_FILES
    )


def _live_args(plan: Mapping[str, Any], output_dir: Path) -> argparse.Namespace:
    route = plan["route"]
    task = plan["task"]
    epicure = plan["epicure"]
    work = plan["work"]
    return argparse.Namespace(
        confirm=CONFIRMATION,
        cap_usd=Decimal(plan["budget"]["reserved_worst_case_usd"]),
        model_id=route["model_id"],
        provider_slug=route["provider_tag"],
        prompt=task["prompt"],
        category=task["family"],
        skip_tool_contract=True,
        contract_only=False,
        condition=None,
        plain_text_final=True,
        tool_catalog_bytes_bound=plan["execution_policy"]["cost_forecast"][
            "tool_catalog_bytes_bound"
        ],
        require_epicure_call=True,
        evidence_protocol="matched_evidence_v2",
        intermediate_reasoning_effort="low",
        final_reasoning_effort="low",
        output_dir=str(output_dir),
        candidate_manifest_sha256=plan["source_artifacts"]["high_resource_manifest"][
            "sha256"
        ],
        sequential_arms=True,
        dataset_work_item_id=work["work_item_id"],
        dataset_task_id=task["task_id"],
        expected_canonical_model_slug=route["canonical_model_slug"],
        expected_endpoint_execution_sha256=route["endpoint_execution_contract_sha256"],
        expected_execution_policy_sha256=plan["execution_policy_sha256"],
        expected_epicure_release_id=epicure["release_id"],
        expected_epicure_bundle_sha256=epicure["bundle_sha256"],
        expected_epicure_application_sha256=epicure["application_sha256"],
        expected_epicure_tool_schema_sha256=epicure["tool_schema_sha256"],
        frozen_run_id=work["run_id"],
        frozen_attempt_slots=work["attempt_slots"],
    )


async def execute_v4_plan(
    *, plan_path: Path, repo_root: Path, output_dir: Path
) -> dict[str, Any]:
    plan = _regular_json(plan_path)
    if not verify_v4_plan(plan, repo_root=repo_root, require_current_sources=True):
        raise V4RouteError("v4 plan or current source binding does not verify")
    # Post-execution verifier corrections may be audited, but they may never
    # reopen the frozen plan for another external invocation.
    if _source_bundle(repo_root) != plan.get("source_code"):
        raise V4RouteError("v4 execution source changed after the frozen invocation")
    if plan["budget"]["admitted"] is not True:
        raise V4RouteError("v4 plan is not budget-admitted")
    summary: dict[str, Any] | None = None
    error_record: dict[str, str] | None = None
    try:
        summary = await live_smoke(_live_args(plan, output_dir))
    except Exception as error:  # preserve the only authorized invocation and close IDs
        error_record = {"type": type(error).__name__, "message": str(error)[:1000]}
    live_path = Path(str((summary or {}).get("artifact") or ""))
    live_sha = None
    if live_path.is_file() and not live_path.is_symlink():
        source = _regular_json(live_path)
        live_sha = source.get("artifact_sha256")
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "record_role": "single_invocation_v4_execution_receipt",
        "v4_plan_sha256": plan["artifact_sha256"],
        "work_item_id": plan["work"]["work_item_id"],
        "run_id": plan["work"]["run_id"],
        "invocation_count": 1,
        "status": (
            "source_artifact_available"
            if summary is not None and live_sha and summary.get("status") == "complete"
            else "failed_closed"
        ),
        "live_artifact": (
            {"path": _relative(repo_root, live_path), "sha256": live_sha}
            if live_sha
            else None
        ),
        "summary": summary,
        "error": error_record,
        "retry_outside_frozen_provider_phases": False,
        "second_route_attempted": False,
        "quality_observations": 0,
        "rank_eligible": False,
    }
    return payload


def _load_live_source(
    *, plan: Mapping[str, Any], receipt: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any] | None, Path | None, list[str]]:
    failures: list[str] = []
    receipt_document = {key: value for key, value in receipt.items() if key != "_path"}
    if not _artifact_verifies(receipt_document, RECEIPT_SCHEMA_VERSION):
        return None, None, ["execution_receipt_does_not_verify"]
    if (
        receipt.get("v4_plan_sha256") != plan.get("artifact_sha256")
        or receipt.get("work_item_id") != plan.get("work", {}).get("work_item_id")
        or receipt.get("run_id") != plan.get("work", {}).get("run_id")
        or receipt.get("invocation_count") != 1
        or receipt.get("second_route_attempted") is not False
    ):
        failures.append("execution_receipt_binding_mismatch")
    reference = receipt.get("live_artifact")
    if not isinstance(reference, Mapping):
        failures.append("live_source_artifact_missing")
        return None, None, failures
    try:
        path = _repo_path(repo_root, str(reference.get("path") or ""))
        source = _regular_json(path)
    except V4RouteError:
        failures.append("live_source_artifact_unreadable")
        return None, None, failures
    if (
        not _live_artifact_verifies(source)
        or source.get("artifact_sha256") != reference.get("sha256")
    ):
        failures.append("live_source_artifact_hash_mismatch")
    return source, path, failures


def _journal_evidence(
    source: Mapping[str, Any], source_path: Path
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    failures: list[str] = []
    descriptor = source.get("run_journal")
    if not isinstance(descriptor, Mapping):
        return [], ["run_journal_descriptor_missing"], None
    filename = str(descriptor.get("filename") or "")
    journal_path = source_path.parent / filename
    try:
        entries = load_run_journal(journal_path)
    except Exception:
        return [], ["run_journal_failed_hash_chain_verification"], None
    physical = _file_sha256(journal_path)
    if (
        descriptor.get("sha256") != physical
        or descriptor.get("entry_count") != len(entries)
        or descriptor.get("head_entry_sha256") != entries[-1].get("entry_sha256")
        or descriptor.get("run_id") != source.get("run_id")
        or descriptor.get("finalized") is not True
    ):
        failures.append("run_journal_descriptor_mismatch")
    journal_attempts = [
        dict(entry.get("payload") or {})
        for entry in entries
        if entry.get("event_type") == "provider_attempt"
    ]
    journal_tools = [
        dict(entry.get("payload") or {})
        for entry in entries
        if entry.get("event_type") == "mcp_trace"
    ]
    if journal_attempts != list(source.get("provider_attempt_events") or []):
        failures.append("journal_provider_events_differ_from_source")
    if journal_tools != list(source.get("mcp_trace_events") or []):
        failures.append("journal_mcp_events_differ_from_source")
    return entries, failures, physical


def derive_v4_audit(
    *,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    if not verify_v4_plan(plan, repo_root=repo_root, require_current_sources=True):
        failures.append("plan_or_current_source_binding_does_not_verify")
    source, source_path, source_failures = _load_live_source(
        plan=plan, receipt=receipt, repo_root=repo_root
    )
    failures.extend(source_failures)
    journal_sha: str | None = None
    provider_starts: list[Mapping[str, Any]] = []
    accepted: list[Mapping[str, Any]] = []
    observed_attempt_ids: list[str] = []
    observed_generation_ids: list[str] = []
    request_keys: list[str] = []
    successful_tools = 0
    actual_cost_micros = 0
    substantive: dict[str, dict[str, int | bool]] = {}
    if source is not None and source_path is not None:
        _, journal_failures, journal_sha = _journal_evidence(source, source_path)
        failures.extend(journal_failures)
        if source.get("status") != "complete" or source.get("errors") != {}:
            failures.append("live_pair_not_complete")
        if source.get("run_id") != plan["work"]["run_id"]:
            failures.append("run_id_mismatch")
        if source.get("requested_conditions") != ["epicure_off", "epicure_on"]:
            failures.append("condition_set_or_order_mismatch")
        if (
            source.get("requested_model_id") != MODEL_ID
            or source.get("requested_provider") != PROVIDER_TAG
            or source.get("model_contract", {}).get("canonical_slug")
            != CANONICAL_MODEL_SLUG
            or source.get("endpoint_contract", {}).get("provider_name") != PROVIDER_NAME
            or source.get("endpoint_execution_contract_sha256")
            != plan["route"]["endpoint_execution_contract_sha256"]
        ):
            failures.append("requested_or_catalog_route_identity_mismatch")
        if (
            source.get("provider_routing_controls") != EXPECTED_PROVIDER_CONTROLS
            or source.get("provider_routing_controls_sha256")
            != _sha256(EXPECTED_PROVIDER_CONTROLS)
        ):
            failures.append("fixed_provider_controls_mismatch")
        if source.get("execution_policy_sha256") != plan["execution_policy_sha256"]:
            failures.append("execution_policy_mismatch")
        epicure = source.get("epicure") or {}
        if any(
            epicure.get(field) != plan["epicure"][field]
            for field in ("release_id", "bundle_sha256", "application_sha256")
        ) or source.get("epicure_tool_schema_sha256") != plan["epicure"][
            "tool_schema_sha256"
        ]:
            failures.append("epicure_runtime_identity_mismatch")
        events = [
            item
            for item in source.get("provider_attempt_events") or []
            if isinstance(item, Mapping)
        ]
        provider_starts = [item for item in events if item.get("event_type") == "request_started"]
        accepted = [item for item in events if item.get("event_type") == "response_received"]
        if any(
            not isinstance(item.get("metadata"), Mapping)
            or item["metadata"].get("response_envelope", {}).get(
                "accepted_chat_completion"
            )
            is not True
            or item["metadata"].get("response_envelope", {}).get("classification")
            != "chat_completions"
            or str(item["metadata"].get("openrouter_cache_status") or "").upper()
            == "HIT"
            or str(item["metadata"].get("cloudflare_cache_status") or "").upper()
            == "HIT"
            for item in accepted
        ):
            failures.append("accepted_response_envelope_or_cache_attestation_invalid")
        unsafe_events = [
            item
            for item in events
            if item.get("event_type") in {"uncertain_delivery", "invalid_response"}
        ]
        if unsafe_events:
            failures.append("unsafe_or_uncertain_provider_event")
        external_starts = [
            item
            for item in events
            if item.get("event_type")
            in {"request_started", "mcp_session_started", "mcp_call_started"}
        ]
        observed_attempt_ids = [str(item.get("attempt_id") or "") for item in external_starts]
        planned_slots = {
            (
                str(item["arm_id"]),
                str(item["phase"]),
                int(item["attempt_index"]),
            ): str(item["attempt_id"])
            for item in plan["work"]["attempt_slots"]
        }
        for event in provider_starts:
            coordinate = (
                str(event.get("arm_id") or ""),
                str(event.get("phase") or ""),
                int(event.get("attempt_index", -1)),
            )
            if planned_slots.get(coordinate) != event.get("attempt_id"):
                failures.append("provider_attempt_outside_prefrozen_slot_pool")
            metadata = event.get("metadata")
            request = metadata.get("request_contract") if isinstance(metadata, Mapping) else None
            if not isinstance(request, Mapping) or metadata.get(
                "request_contract_sha256"
            ) != _sha256(request):
                failures.append("request_semantics_projection_missing_or_invalid")
                continue
            if request.get("model") != MODEL_ID or request.get("provider") != (
                EXPECTED_PROVIDER_CONTROLS
            ):
                failures.append("request_route_controls_differ")
            if request.get("reasoning") != {"effort": "low", "exclude": True}:
                failures.append("low_reasoning_request_not_explicit")
            phase = str(event.get("phase") or "")
            arm_id = str(event.get("arm_id") or "")
            if phase.startswith("tool_round_"):
                if request.get("tools_present") is not True or not request.get("tools"):
                    failures.append("tool_round_lacks_attested_tool_catalog")
                if phase == "tool_round_0" and request.get("tool_choice") != "required":
                    failures.append("first_epicure_tool_round_not_required")
            elif request.get("tools_present") is True:
                failures.append("non_tool_phase_exposed_tools")
            if arm_id.endswith(":epicure_off") and request.get("tools_present") is True:
                failures.append("epicure_off_received_tools")
        for event in external_starts:
            coordinate = (
                str(event.get("arm_id") or ""),
                str(event.get("phase") or ""),
                int(event.get("attempt_index", -1)),
            )
            if planned_slots.get(coordinate) != event.get("attempt_id"):
                failures.append("external_attempt_outside_prefrozen_slot_pool")
        if len(observed_attempt_ids) != len(set(observed_attempt_ids)):
            failures.append("attempt_id_reused_across_external_sends")
        if set(observed_attempt_ids) & set(plan["prior_closed_identifiers"]["attempt_ids"]):
            failures.append("attempt_id_replays_prior_revision")
        phase_keys: dict[tuple[str, str], set[str]] = {}
        for event in external_starts:
            key = (str(event.get("arm_id") or ""), str(event.get("phase") or ""))
            request_key = str(event.get("request_key_sha256") or "")
            request_keys.append(request_key)
            if not _is_sha256(request_key):
                failures.append("request_key_is_not_sha256")
            phase_keys.setdefault(key, set()).add(request_key)
        if any(len(values) != 1 for values in phase_keys.values()):
            failures.append("retry_chain_changed_idempotency_key")
        if len(set(request_keys)) != len(phase_keys):
            failures.append("request_key_reused_across_arm_or_phase")
        if set(request_keys) & set(
            plan["prior_closed_identifiers"]["request_key_sha256s"]
        ):
            failures.append("request_key_replays_prior_revision")
        accepted_generation_ids = [str(item.get("generation_id") or "") for item in accepted]
        results = source.get("results") or {}
        if set(results) != {"epicure_off", "epicure_on"}:
            failures.append("exact_two_arm_results_missing")
        metadata_records: list[Mapping[str, Any]] = []
        for condition in ("epicure_off", "epicure_on"):
            result = results.get(condition)
            if not isinstance(result, Mapping):
                continue
            if (
                result.get("actual_model_id") != CANONICAL_MODEL_SLUG
                or result.get("actual_provider") != PROVIDER_NAME
                or result.get("finish_reason") not in {"stop", "end_turn"}
                or result.get("final_response_mode") != "plain_text"
                or result.get("cost_reconciled") is not True
            ):
                failures.append(f"{condition}_identity_finish_or_accounting_invalid")
            answer = str(result.get("answer_markdown") or "").strip()
            words = len(answer.split())
            substantive[condition] = {
                "characters": len(answer),
                "words": words,
                "passed": len(answer) >= 100 and words >= 20,
            }
            if len(answer) < 100 or words < 20:
                failures.append(f"{condition}_final_answer_not_substantive")
            for intermediate in result.get("intermediate_outputs") or []:
                if isinstance(intermediate, Mapping) and intermediate.get("truncated") is True:
                    failures.append(f"{condition}_intermediate_truncated")
            traces = result.get("tool_trace") or []
            if condition == "epicure_off" and traces:
                failures.append("epicure_off_has_tool_trace")
            if condition == "epicure_on":
                successful_tools = sum(
                    1
                    for trace in traces
                    if isinstance(trace, Mapping) and trace.get("is_error") is False
                )
                if successful_tools < 1:
                    failures.append("epicure_on_has_no_successful_real_tool_call")
            generation_ids = [str(value) for value in result.get("generation_ids") or []]
            observed_generation_ids.extend(generation_ids)
            metadata_records.extend(
                item
                for item in result.get("generation_metadata") or []
                if isinstance(item, Mapping)
            )
        if (
            not observed_generation_ids
            or len(observed_generation_ids) != len(set(observed_generation_ids))
            or set(observed_generation_ids) != set(accepted_generation_ids)
        ):
            failures.append("accepted_generation_id_bijection_failed")
        if set(observed_generation_ids) & set(
            plan["prior_closed_identifiers"]["generation_ids"]
        ):
            failures.append("generation_id_replays_prior_revision")
        metadata_ids = [str(item.get("generation_id") or "") for item in metadata_records]
        if (
            len(metadata_ids) != len(set(metadata_ids))
            or set(metadata_ids) != set(observed_generation_ids)
            or any(
                item.get("reconciled") is not True
                or item.get("model") != CANONICAL_MODEL_SLUG
                or item.get("provider") != PROVIDER_NAME
                for item in metadata_records
            )
        ):
            failures.append("generation_metadata_cost_identity_bijection_failed")
        actual_cost_micros = sum(int(item.get("cost_micros") or 0) for item in metadata_records)
        budget = source.get("budget") or {}
        if (
            budget.get("all_generation_costs_reconciled") is not True
            or int(budget.get("actual_cost_micros") or 0) != actual_cost_micros
            or Decimal(actual_cost_micros) / Decimal(1_000_000)
            > _decimal(plan["budget"]["reserved_worst_case_usd"], field="reserve")
            or source.get("incomplete_generation_metadata") != []
        ):
            failures.append("cost_reconciliation_or_reserve_failed")
        mcp_events = [
            item
            for item in events
            if str(item.get("event_type") or "").startswith("mcp_")
        ]
        mcp_starts = {
            str(item.get("attempt_id") or "")
            for item in mcp_events
            if item.get("event_type") == "mcp_call_started"
        }
        mcp_completions = {
            str(item.get("attempt_id") or "")
            for item in mcp_events
            if item.get("event_type") == "mcp_call_completed"
        }
        if not mcp_starts or mcp_starts != mcp_completions:
            failures.append("mcp_call_start_completion_bijection_failed")
        if not any(item.get("event_type") == "mcp_session_started" for item in mcp_events):
            failures.append("mcp_session_start_missing")
        if not any(item.get("event_type") == "mcp_session_attested" for item in mcp_events):
            failures.append("mcp_session_attestation_missing")
        completed_result_hashes = sorted(
            str(item.get("payload_sha256") or "")
            for item in mcp_events
            if item.get("event_type") == "mcp_call_completed"
        )
        traced_result_hashes = sorted(
            str(item.get("result_sha256") or "")
            for item in source.get("mcp_trace_events") or []
            if isinstance(item, Mapping)
        )
        if (
            not traced_result_hashes
            or completed_result_hashes != traced_result_hashes
            or any(not _is_sha256(value) for value in traced_result_hashes)
        ):
            failures.append("mcp_result_hash_bijection_failed")
    current_source = _source_bundle(repo_root)
    planned_source = plan.get("source_code") or {}
    planned_files = {
        str(item.get("path") or ""): item
        for item in planned_source.get("files") or []
        if isinstance(item, Mapping)
    }
    current_files = {
        str(item.get("path") or ""): item
        for item in current_source.get("files") or []
        if isinstance(item, Mapping)
    }
    corrections = [
        {
            "path": path,
            "planned_sha256": planned_files[path].get("sha256"),
            "current_sha256": current_files[path].get("sha256"),
            "planned_bytes": planned_files[path].get("bytes"),
            "current_bytes": current_files[path].get("bytes"),
            "reason": "post_execution_source_reconstruction_correction_without_provider_replay",
        }
        for path in sorted(set(planned_files) & set(current_files))
        if planned_files[path] != current_files[path]
    ]
    generation_sources_unchanged = bool(
        set(planned_files) == set(current_files)
        and all(
            planned_files[path] == current_files[path]
            for path in planned_files
            if path not in POST_EXECUTION_VERIFIER_FILES
        )
        and all(
            correction["path"] in POST_EXECUTION_VERIFIER_FILES
            for correction in corrections
        )
    )
    if not generation_sources_unchanged:
        failures.append("generation_source_changed_after_frozen_invocation")
    unique_failures = sorted(set(failures))
    passed = not unique_failures
    source_reference = receipt.get("live_artifact") if isinstance(receipt, Mapping) else None
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "record_role": "source_reconstructed_v4_response_envelope_gate",
        "v4_plan_sha256": plan.get("artifact_sha256"),
        "execution_receipt_sha256": receipt.get("artifact_sha256"),
        "execution_receipt_path": receipt.get("_path"),
        "source_artifact": source_reference,
        "journal_sha256": journal_sha,
        "verification_source": {
            "planned_bundle_sha256": planned_source.get("bundle_sha256"),
            "current_bundle_sha256": current_source.get("bundle_sha256"),
            "current_files": current_source.get("files"),
            "post_execution_corrections": corrections,
            "permitted_correction_files": sorted(POST_EXECUTION_VERIFIER_FILES),
            "generation_source_files_unchanged": generation_sources_unchanged,
            "provider_calls_made_by_correction": False,
            "mcp_calls_made_by_correction": False,
        },
        "decision": "passed_all_predicates" if passed else "failed_one_or_more_predicates",
        "failures": unique_failures,
        "counts": {
            "attempted_pairs": 1 if source is not None else 0,
            "usable_pairs": 1 if passed else 0,
            "intended_arms": 2,
            "usable_arms": 2 if passed else 0,
            "provider_requests": len(provider_starts),
            "accepted_chat_completions": len(accepted),
            "successful_epicure_tool_calls": successful_tools,
            "epicure_off_tool_calls": (
                len((source or {}).get("results", {}).get("epicure_off", {}).get("tool_trace", []))
                if source is not None
                else 0
            ),
            "synthetic_arms": 0,
        },
        "identifier_audit": {
            "observed_attempt_ids": sorted(observed_attempt_ids),
            "observed_generation_ids": sorted(observed_generation_ids),
            "observed_request_key_sha256s": sorted(set(request_keys)),
            "attempt_id_overlap": sorted(
                set(observed_attempt_ids)
                & set(plan.get("prior_closed_identifiers", {}).get("attempt_ids") or [])
            ),
            "generation_id_overlap": sorted(
                set(observed_generation_ids)
                & set(plan.get("prior_closed_identifiers", {}).get("generation_ids") or [])
            ),
        },
        "substantive_integrity": substantive,
        "accounting": {
            "actual_cost_micros": actual_cost_micros,
            "actual_cost_usd": _decimal_text(
                Decimal(actual_cost_micros) / Decimal(1_000_000)
            ),
            "reserved_worst_case_usd": plan.get("budget", {}).get(
                "reserved_worst_case_usd"
            ),
            "reconciled": passed or "cost_reconciliation_or_reserve_failed" not in failures,
        },
        "coverage_admission": {
            "authorized": passed,
            "evidence_outputs_reused": False,
            "scope": "materialize_fresh_zero_call_frontier_coverage_plan_only",
        },
        "claim_boundary": {
            "route_smoke_only": True,
            "quality_observations": 0,
            "official": False,
            "rank_eligible": False,
            "synthetic_or_model_judge_evidence": False,
        },
    }


def build_v4_audit(
    *, plan_path: Path, receipt_path: Path, repo_root: Path
) -> dict[str, Any]:
    plan = _regular_json(plan_path)
    receipt = _regular_json(receipt_path)
    receipt["_path"] = _relative(repo_root, receipt_path)
    return derive_v4_audit(plan=plan, receipt=receipt, repo_root=repo_root)


def build_v4_closure(
    *,
    plan: Mapping[str, Any],
    audit: Mapping[str, Any],
    receipt_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if not verify_v4_plan(plan):
        raise V4RouteError("v4 plan does not verify for closure")
    if not _artifact_verifies(audit, AUDIT_SCHEMA_VERSION):
        raise V4RouteError("v4 audit does not verify for closure")
    observed = audit.get("identifier_audit") or {}
    planned_attempts = sorted(
        str(item["attempt_id"]) for item in plan["work"]["attempt_slots"]
    )
    used_attempts = sorted(str(value) for value in observed.get("observed_attempt_ids") or [])
    return {
        "schema_version": CLOSURE_SCHEMA_VERSION,
        "record_role": "permanent_v4_identifier_and_invocation_closure",
        "v4_plan_sha256": plan["artifact_sha256"],
        "v4_audit_sha256": audit["artifact_sha256"],
        "execution_receipt": {
            "path": _relative(repo_root, receipt_path),
            "sha256": audit["execution_receipt_sha256"],
        },
        "closed_identifiers": {
            "route_cell_ids": [plan["route_cell_id"]],
            "work_item_ids": [plan["work"]["work_item_id"]],
            "run_ids": [plan["work"]["run_id"]],
            "arm_ids": sorted(plan["work"]["arm_ids"]),
            "attempt_ids": planned_attempts,
            "used_attempt_ids": used_attempts,
            "unused_attempt_ids": sorted(set(planned_attempts) - set(used_attempts)),
            "generation_ids": sorted(
                str(value) for value in observed.get("observed_generation_ids") or []
            ),
            "request_key_sha256s": sorted(
                str(value) for value in observed.get("observed_request_key_sha256s") or []
            ),
            "replay_permitted": False,
        },
        "decision": {
            "route_qualified": audit.get("decision") == "passed_all_predicates",
            "coverage_materialization_permitted": (
                audit.get("coverage_admission", {}).get("authorized") is True
            ),
            "full_coverage_execution_performed": False,
            "second_route_attempted": False,
        },
        "cost": audit.get("accounting"),
        "claim_boundary": audit.get("claim_boundary"),
    }


def verify_v4_closure(
    document: object, *, plan: Mapping[str, Any], audit: Mapping[str, Any]
) -> bool:
    if not _artifact_verifies(document, CLOSURE_SCHEMA_VERSION):
        return False
    assert isinstance(document, Mapping)
    identifiers = document.get("closed_identifiers") or {}
    decision = document.get("decision") or {}
    expected_attempts = sorted(
        str(item["attempt_id"]) for item in plan.get("work", {}).get("attempt_slots") or []
    )
    used = identifiers.get("used_attempt_ids") or []
    return bool(
        document.get("v4_plan_sha256") == plan.get("artifact_sha256")
        and document.get("v4_audit_sha256") == audit.get("artifact_sha256")
        and identifiers.get("route_cell_ids") == [plan.get("route_cell_id")]
        and identifiers.get("work_item_ids") == [plan.get("work", {}).get("work_item_id")]
        and identifiers.get("run_ids") == [plan.get("work", {}).get("run_id")]
        and identifiers.get("arm_ids") == sorted(plan.get("work", {}).get("arm_ids") or [])
        and identifiers.get("attempt_ids") == expected_attempts
        and identifiers.get("unused_attempt_ids")
        == sorted(set(expected_attempts) - set(used))
        and identifiers.get("replay_permitted") is False
        and decision.get("route_qualified")
        is (audit.get("decision") == "passed_all_predicates")
        and decision.get("coverage_materialization_permitted")
        is (audit.get("coverage_admission", {}).get("authorized") is True)
        and decision.get("full_coverage_execution_performed") is False
        and decision.get("second_route_attempted") is False
    )


def verify_v4_route_acceptance_paths(
    *, plan_path: Path, audit_path: Path, closure_path: Path, repo_root: Path
) -> bool:
    """Open and rederive every v4 evidence layer; summaries alone never pass."""

    try:
        plan = _regular_json(plan_path)
        audit = _regular_json(audit_path)
        closure = _regular_json(closure_path)
        if not verify_v4_plan(plan, repo_root=repo_root, require_current_sources=True):
            return False
        if not _artifact_verifies(audit, AUDIT_SCHEMA_VERSION):
            return False
        receipt_reference = closure.get("execution_receipt")
        if not isinstance(receipt_reference, Mapping):
            return False
        receipt_path = _repo_path(repo_root, str(receipt_reference.get("path") or ""))
        receipt = _regular_json(receipt_path)
        if (
            not _artifact_verifies(receipt, RECEIPT_SCHEMA_VERSION)
            or receipt.get("artifact_sha256") != receipt_reference.get("sha256")
        ):
            return False
        receipt_for_derivation = dict(receipt)
        receipt_for_derivation["_path"] = _relative(repo_root, receipt_path)
        expected_audit = derive_v4_audit(
            plan=plan, receipt=receipt_for_derivation, repo_root=repo_root
        )
        expected_audit["artifact_sha256"] = _sha256(expected_audit)
        if dict(audit) != expected_audit or audit.get("decision") != "passed_all_predicates":
            return False
        expected_closure = build_v4_closure(
            plan=plan,
            audit=audit,
            receipt_path=receipt_path,
            repo_root=repo_root,
        )
        expected_closure["artifact_sha256"] = _sha256(expected_closure)
        return dict(closure) == expected_closure and verify_v4_closure(
            closure, plan=plan, audit=audit
        )
    except (OSError, ValueError, TypeError, KeyError, V4RouteError):
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    refresh = sub.add_parser("refresh-catalog")
    refresh.add_argument("--output-dir", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--v3-plan", type=Path, required=True)
    freeze.add_argument("--v3-audit", type=Path, required=True)
    freeze.add_argument("--v3-closure", type=Path, required=True)
    freeze.add_argument("--route-registry", type=Path, required=True)
    freeze.add_argument("--catalog-audit", type=Path, required=True)
    freeze.add_argument("--fresh-catalog", type=Path, required=True)
    freeze.add_argument("--high-resource-manifest", type=Path, required=True)
    freeze.add_argument("--task-dossier", type=Path, required=True)
    freeze.add_argument("--epicure-attestation", type=Path, required=True)
    freeze.add_argument("--coverage-execution-plan", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--receipt", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    close = sub.add_parser("close")
    close.add_argument("--plan", type=Path, required=True)
    close.add_argument("--audit", type=Path, required=True)
    close.add_argument("--receipt", type=Path, required=True)
    close.add_argument("--output-dir", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--audit", type=Path, required=True)
    verify.add_argument("--closure", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    repo_root = arguments.repo_root.resolve()
    if arguments.command == "refresh-catalog":
        payload = asyncio.run(refresh_catalog())
        path = _write_artifact(
            arguments.output_dir, "response-envelope-v4-catalog-snapshot", payload
        )
    elif arguments.command == "freeze":
        payload = build_v4_plan(
            repo_root=repo_root,
            v3_plan_path=arguments.v3_plan,
            v3_audit_path=arguments.v3_audit,
            v3_closure_path=arguments.v3_closure,
            route_registry_path=arguments.route_registry,
            catalog_audit_path=arguments.catalog_audit,
            fresh_catalog_path=arguments.fresh_catalog,
            high_resource_manifest_path=arguments.high_resource_manifest,
            task_dossier_path=arguments.task_dossier,
            epicure_attestation_path=arguments.epicure_attestation,
            coverage_execution_plan_path=arguments.coverage_execution_plan,
        )
        path = _write_artifact(
            arguments.output_dir, "response-envelope-route-v4-plan", payload
        )
    elif arguments.command == "execute":
        payload = asyncio.run(
            execute_v4_plan(
                plan_path=arguments.plan,
                repo_root=repo_root,
                output_dir=arguments.output_dir / "source",
            )
        )
        path = _write_artifact(
            arguments.output_dir, "response-envelope-route-v4-execution", payload
        )
    elif arguments.command == "audit":
        payload = build_v4_audit(
            plan_path=arguments.plan,
            receipt_path=arguments.receipt,
            repo_root=repo_root,
        )
        path = _write_artifact(
            arguments.output_dir, "response-envelope-route-v4-audit", payload
        )
    elif arguments.command == "close":
        payload = build_v4_closure(
            plan=_regular_json(arguments.plan),
            audit=_regular_json(arguments.audit),
            receipt_path=arguments.receipt,
            repo_root=repo_root,
        )
        path = _write_artifact(
            arguments.output_dir, "response-envelope-route-v4-closure", payload
        )
    else:
        passed = verify_v4_route_acceptance_paths(
            plan_path=arguments.plan,
            audit_path=arguments.audit,
            closure_path=arguments.closure,
            repo_root=repo_root,
        )
        print(json.dumps({"passed": passed}, indent=2, sort_keys=True))
        raise SystemExit(0 if passed else 1)
    document = _regular_json(path)
    print(
        json.dumps(
            {
                "output": str(path.resolve()),
                "artifact_sha256": document["artifact_sha256"],
                "decision": document.get("decision"),
                "provider_calls_made_by_builder": False,
                "epicure_calls_made_by_builder": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
