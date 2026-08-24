"""Endpoint-isolated successor to the failed reasoning-effort v4 route gate.

The v4 gate correctly stopped before generation when OpenRouter changed the
textual precision of one Gemini cache-price field.  This module keeps that
failure and its reservation closed, preserves the two source-verified
DeepSeek pairs, and gives Gemini and Sonnet independent fresh identifier
pools.  Endpoint identity is frozen twice: the raw catalog projection remains
available for forensic reconstruction, while admission uses a semantic
projection that Decimal-normalizes prices.  A provider/model/capability/price
change fails closed; an equivalent decimal spelling does not.

``snapshot`` performs catalog metadata GETs only. ``audit-v4`` and ``freeze``
are zero-call builders. Paid execution is endpoint-specific and never runs as
part of planning.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

SNAPSHOT_SCHEMA = "flavourbench-openrouter-endpoint-snapshot-v5"
V4_INCIDENT_AUDIT_SCHEMA = "flavourbench-reasoning-effort-v4-pre-request-audit-v1"
ROUTE_PLAN_SCHEMA = "flavourbench-reasoning-effort-route-gate-plan-v5"
ENDPOINT_RECEIPT_SCHEMA = "flavourbench-reasoning-effort-endpoint-receipt-v5"
ENDPOINT_AUDIT_SCHEMA = "flavourbench-reasoning-effort-endpoint-audit-v5"
ENDPOINT_CLOSURE_SCHEMA = "flavourbench-reasoning-effort-endpoint-closure-v5"
AGGREGATE_AUDIT_SCHEMA = "flavourbench-reasoning-effort-route-gate-audit-v5"
AGGREGATE_CLOSURE_SCHEMA = "flavourbench-reasoning-effort-route-gate-closure-v5"

V4_ROUTE_PLAN_SHA256 = "2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352"
V4_RECEIPT_SHA256 = "172f4a08003656371de69c0907975f83761597338b159031b16052417d575852"
V4_AUDIT_SHA256 = "c90617d7b6a8cab918bf0f50f7190f8ad8f49badb5ce036c7c9fa716d7d9a959"
V4_CLOSURE_SHA256 = "807aa054e7f0aaaa770630adae7696bba8fc24251d7ed2b08082b46a0edfde87"
V4_GEMINI_WORK_ITEM_ID = "189d76023f42d7b14912b61daa8b98fde587b86b05695908b522b45ba9175002"
V4_GEMINI_RUN_ID = "19125098-99b0-58af-b87b-a6260a9c5bd3"
V4_GEMINI_RESERVATION_SHA256 = "db19e86ac60a9fa9d0c34a7787b7b383e4aa2b3ec30eec4006628ffd7e8a4e26"
V4_GEMINI_INCIDENT_SHA256 = "86a99483395c09d57fcc1ada43bce5a8c4a2e5930f2a554ac099f47f02291e0c"
V4_ERROR_TEXT = "current endpoint execution contract differs from the frozen manifest"
V4_ERROR_SHA256 = "71cd44184907309cc160fb501e395865355a417f4907a1a6ec1e3a6fa3ef0e83"
V4_GEMINI_RAW_SHA256 = "6247ff4fae463bb8a056043d5cefb4188fcdde425dd65b5bfc459fcd9a19bc81"

FREEZE_NONCE = "effort-v5-2026-08-03-endpoint-isolated-semantic-contract"
NAMESPACE = uuid.UUID("ac0399af-bb2d-5a96-b23d-e41c2b1476ef")
PRICE_QUANTUM_USD_PER_RAW_UNIT = Decimal("0.000000000000001")
TOKEN_PRICE_QUANTUM_USD_PER_MTOK = Decimal("0.000000001")
ENDPOINTS = ("gemini", "sonnet")
ENDPOINT_MODELS = {
    "gemini": "google/gemini-3.6-flash",
    "sonnet": "anthropic/claude-sonnet-5",
}
CONFIRMATIONS = {
    "gemini": "RUN_EXACT_REASONING_EFFORT_V5_GEMINI_2_PAIRS",
    "sonnet": "RUN_EXACT_REASONING_EFFORT_V5_SONNET_2_PAIRS",
}
CONDITIONS = ("epicure_off", "epicure_on")
RAW_FIELDS = (
    "model_id",
    "provider_name",
    "tag",
    "quantization",
    "context_length",
    "max_completion_tokens",
    "pricing",
    "supported_parameters",
)


class RouteGateV5Error(RuntimeError):
    """A frozen v5 input or fail-closed predicate did not verify."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RouteGateV5Error(f"expected a regular non-symlink file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_verifies(document: object, schema: str) -> bool:
    if not isinstance(document, Mapping) or document.get("schema_version") != schema:
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return isinstance(digest, str) and digest == _sha256(unhashed)


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RouteGateV5Error(f"input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RouteGateV5Error(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise RouteGateV5Error(f"expected a JSON object: {path}")
    return value


def _write_artifact(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RouteGateV5Error(f"content-addressed output conflict: {path}")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError as error:
        raise RouteGateV5Error(f"path is outside the evaluation repository: {path}") from error


def _decimal_text(value: object) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise RouteGateV5Error(f"invalid decimal value: {value!r}") from error
    if not number.is_finite() or number < 0:
        raise RouteGateV5Error("decimal values must be finite and non-negative")
    rendered = format(number.normalize(), "f")
    return "0" if Decimal(rendered) == 0 else rendered


def _semantic_decimal_text(value: object) -> str:
    """Canonicalize catalog prices to a fixed absolute price quantum.

    OpenRouter emitted the same repeating cache price rounded at adjacent
    final digits. The 1e-15 USD/raw-unit quantum is 1e-9 USD/MTok for token
    prices. Its maximum rounding error is 5e-10 USD/MTok, far below the
    benchmark's micro-dollar generation accounting resolution.
    """

    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise RouteGateV5Error(f"invalid decimal value: {value!r}") from error
    if not number.is_finite() or number < 0:
        raise RouteGateV5Error("decimal values must be finite and non-negative")
    rounded = number.quantize(PRICE_QUANTUM_USD_PER_RAW_UNIT)
    rendered = format(rounded.normalize(), "f")
    return "0" if rounded == 0 else rendered


def raw_endpoint_contract(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    contract = {field: copy.deepcopy(endpoint.get(field)) for field in RAW_FIELDS}
    contract["supported_parameters"] = sorted(contract.get("supported_parameters") or [])
    return contract


def semantic_endpoint_contract(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize catalog spelling without weakening execution identity."""

    contract = raw_endpoint_contract(endpoint)
    pricing = contract.get("pricing")
    if not isinstance(pricing, Mapping):
        raise RouteGateV5Error("endpoint pricing is absent or malformed")
    normalized: dict[str, Any] = {}
    for key, value in sorted(pricing.items()):
        try:
            normalized[str(key)] = _semantic_decimal_text(value)
        except RouteGateV5Error:
            normalized[str(key)] = value
    contract["pricing"] = normalized
    contract["pricing_normalization"] = {
        "method": "decimal_fixed_absolute_quantization",
        "quantum_usd_per_raw_unit": _decimal_text(PRICE_QUANTUM_USD_PER_RAW_UNIT),
        "token_price_quantum_usd_per_mtok": _decimal_text(TOKEN_PRICE_QUANTUM_USD_PER_MTOK),
        "maximum_token_price_rounding_usd_per_mtok": "0.0000000005",
        "generation_accounting_resolution_usd": "0.000001",
    }
    return contract


def _normalise_provider(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


async def _catalog_endpoint(
    client: httpx.AsyncClient, model_id: str, provider_slug: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str | None]]:
    author, slug = model_id.split("/", 1)
    model_path = f"model/{quote(author)}/{quote(slug, safe=':')}"
    endpoint_path = f"models/{quote(author)}/{quote(slug, safe=':')}/endpoints"
    model_response = await client.get(model_path)
    model_response.raise_for_status()
    endpoint_response = await client.get(endpoint_path)
    endpoint_response.raise_for_status()
    model = (model_response.json() or {}).get("data") or {}
    endpoint_data = (endpoint_response.json() or {}).get("data") or {}
    endpoints = endpoint_data.get("endpoints") or []
    expected = _normalise_provider(provider_slug)
    matches = [
        item for item in endpoints if _normalise_provider(str(item.get("tag") or "")) == expected
    ]
    if len(matches) != 1:
        raise RouteGateV5Error(
            f"expected one exact endpoint for {model_id}@{provider_slug}; found {len(matches)}"
        )
    safe_model = {
        field: copy.deepcopy(model.get(field))
        for field in (
            "id",
            "canonical_slug",
            "name",
            "context_length",
            "pricing",
            "supported_parameters",
        )
    }
    dates = {
        "model_response_date": model_response.headers.get("date"),
        "endpoint_response_date": endpoint_response.headers.get("date"),
    }
    return safe_model, matches[0], dates


async def build_snapshot(*, api_base: str, api_key: str) -> dict[str, Any]:
    if not api_key:
        raise RouteGateV5Error("OpenRouter API key is absent")
    headers = {"Authorization": f"Bearer {api_key}"}
    records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=api_base.rstrip("/") + "/", headers=headers, timeout=30
    ) as client:
        for endpoint_id, model_id, provider in (
            ("gemini", "google/gemini-3.6-flash", "google-ai-studio/flex"),
            ("sonnet", "anthropic/claude-sonnet-5", "anthropic"),
        ):
            model, endpoint, dates = await _catalog_endpoint(client, model_id, provider)
            raw = raw_endpoint_contract(endpoint)
            semantic = semantic_endpoint_contract(endpoint)
            records.append(
                {
                    "endpoint_id": endpoint_id,
                    "requested_model_id": model_id,
                    "requested_provider_endpoint": provider,
                    "model": model,
                    "raw_execution_contract": raw,
                    "raw_execution_contract_sha256": _sha256(raw),
                    "semantic_execution_contract": semantic,
                    "semantic_execution_contract_sha256": _sha256(semantic),
                    "response_dates": dates,
                }
            )
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "record_role": "zero_generation_openrouter_endpoint_metadata_snapshot",
        "api_origin": api_base.rstrip("/"),
        "records": records,
        "counts": {
            "catalog_http_gets": 4,
            "provider_completion_requests": 0,
            "epicure_calls": 0,
        },
        "secrets_persisted": False,
    }


def _attempt_slots(run_id: str, route_cell_id: str) -> list[dict[str, Any]]:
    coordinates: list[tuple[str, str, int]] = []
    off = f"{run_id}:epicure_off"
    on = f"{run_id}:epicure_on"
    for phase in ("planning", "evidence_decision", "final"):
        coordinates.extend((off, phase, attempt) for attempt in (0, 1))
    for phase in ("planning", "tool_round_0", "tool_round_1", "tool_round_2", "final"):
        coordinates.extend((on, phase, attempt) for attempt in (0, 1))
    coordinates.append((on, "mcp_session", 0))
    for round_index in range(3):
        for call_index in range(6):
            coordinates.append((on, f"mcp_tool_{round_index}_{call_index}", 0))
    return [
        {
            "arm_id": arm_id,
            "phase": phase,
            "attempt_index": attempt_index,
            "attempt_id": str(
                uuid.uuid5(
                    NAMESPACE,
                    f"{FREEZE_NONCE}:{route_cell_id}:{arm_id}:{phase}:{attempt_index}",
                )
            ),
        }
        for arm_id, phase, attempt_index in coordinates
    ]


def _assert_artifact(path: Path, digest: str, schema: str | None = None) -> dict[str, Any]:
    document = _regular_json(path)
    if document.get("artifact_sha256") != digest:
        raise RouteGateV5Error(f"unexpected artifact identity: {path}")
    if schema is not None and not _artifact_verifies(document, schema):
        raise RouteGateV5Error(f"artifact content address or schema does not verify: {path}")
    return document


def _hash_chain_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RouteGateV5Error(f"JSONL input must be a regular file: {path}")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RouteGateV5Error(f"malformed JSONL object at {path}:{sequence}")
        claimed = value.get("entry_sha256")
        body = {key: item for key, item in value.items() if key != "entry_sha256"}
        if (
            value.get("sequence") != sequence
            or value.get("previous_entry_sha256") != previous
            or claimed != _sha256(body)
        ):
            raise RouteGateV5Error(f"JSONL hash chain failed at {path}:{sequence}")
        entries.append(value)
        previous = str(claimed)
    return entries


def _endpoint_record(snapshot: Mapping[str, Any], endpoint_id: str) -> dict[str, Any]:
    records = [
        item
        for item in snapshot.get("records") or []
        if isinstance(item, Mapping) and item.get("endpoint_id") == endpoint_id
    ]
    if len(records) != 1:
        raise RouteGateV5Error(f"snapshot lacks exactly one {endpoint_id} endpoint")
    record = dict(records[0])
    if (
        record.get("raw_execution_contract_sha256") != _sha256(record.get("raw_execution_contract"))
        or record.get("semantic_execution_contract_sha256")
        != _sha256(record.get("semantic_execution_contract"))
        or semantic_endpoint_contract(record.get("raw_execution_contract") or {})
        != record.get("semantic_execution_contract")
    ):
        raise RouteGateV5Error(f"snapshot {endpoint_id} raw/semantic projections do not rederive")
    return record


def build_v4_incident_audit(
    *,
    v4_route_plan_path: Path,
    v4_receipt_path: Path,
    v4_audit_path: Path,
    v4_closure_path: Path,
    v4_ledger_path: Path,
    v4_journal_path: Path,
    v4_source_directory: Path,
    endpoint_snapshot_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Reconstruct the exact v4 stop without replay or inference from a summary."""

    from .reasoning_effort_route_gate_v4 import (
        AUDIT_SCHEMA as V4_AUDIT_SCHEMA,
    )
    from .reasoning_effort_route_gate_v4 import (
        CLOSURE_SCHEMA as V4_CLOSURE_SCHEMA,
    )
    from .reasoning_effort_route_gate_v4 import (
        EXECUTION_RECEIPT_SCHEMA as V4_RECEIPT_SCHEMA,
    )
    from .reasoning_effort_sensitivity_v4 import ROUTE_PLAN_SCHEMA as V4_PLAN_SCHEMA

    route = _assert_artifact(v4_route_plan_path, V4_ROUTE_PLAN_SHA256, V4_PLAN_SCHEMA)
    receipt = _assert_artifact(v4_receipt_path, V4_RECEIPT_SHA256, V4_RECEIPT_SCHEMA)
    audit = _assert_artifact(v4_audit_path, V4_AUDIT_SHA256, V4_AUDIT_SCHEMA)
    closure = _assert_artifact(v4_closure_path, V4_CLOSURE_SHA256, V4_CLOSURE_SCHEMA)
    snapshot = _regular_json(endpoint_snapshot_path)
    if not _artifact_verifies(snapshot, SNAPSHOT_SCHEMA):
        raise RouteGateV5Error("endpoint snapshot does not verify")
    gemini_snapshot = _endpoint_record(snapshot, "gemini")
    ledger = _hash_chain_jsonl(v4_ledger_path)
    journal = _hash_chain_jsonl(v4_journal_path)

    reservation = [
        item
        for item in ledger
        if item.get("event_type") == "reservation_created"
        and item.get("work_item_id") == V4_GEMINI_WORK_ITEM_ID
    ]
    incident = [
        item
        for item in ledger
        if item.get("event_type") == "execution_incident"
        and item.get("work_item_id") == V4_GEMINI_WORK_ITEM_ID
    ]
    if (
        len(reservation) != 1
        or reservation[0].get("entry_sha256") != V4_GEMINI_RESERVATION_SHA256
        or len(incident) != 1
        or incident[0].get("entry_sha256") != V4_GEMINI_INCIDENT_SHA256
        or incident[0].get("error_type") != "RuntimeError"
        or incident[0].get("error_sha256") != V4_ERROR_SHA256
        or incident[0].get("replay_permitted") is not False
    ):
        raise RouteGateV5Error("v4 Gemini reservation/incident evidence differs")
    if hashlib.sha256(V4_ERROR_TEXT.encode()).hexdigest() != V4_ERROR_SHA256:
        raise RouteGateV5Error("v4 exception preimage does not match its immutable hash")
    if [item.get("event_type") for item in journal] != [
        "run_started",
        "openrouter_key_status",
    ]:
        raise RouteGateV5Error("v4 in-progress journal contains an unexpected event")
    start = journal[0].get("payload") or {}
    if (
        journal[0].get("run_id") != V4_GEMINI_RUN_ID
        or start.get("dataset_work_item_id") != V4_GEMINI_WORK_ITEM_ID
        or start.get("requested_model_id") != "google/gemini-3.6-flash"
        or start.get("requested_provider") != "google-ai-studio/flex"
    ):
        raise RouteGateV5Error("v4 in-progress journal is not the stopped Gemini run")
    source_names = {
        str(item.get("dataset_work_item_id") or "")
        for path in v4_source_directory.glob("*.json")
        for item in [_regular_json(path)]
    }
    if V4_GEMINI_WORK_ITEM_ID in source_names:
        raise RouteGateV5Error("v4 Gemini unexpectedly has a completed source artifact")
    closed = closure.get("closed_identifiers") or {}
    if (
        V4_GEMINI_WORK_ITEM_ID not in (closed.get("work_item_ids") or [])
        or V4_GEMINI_RUN_ID not in (closed.get("run_ids") or [])
        or closed.get("replay_permitted") is not False
    ):
        raise RouteGateV5Error("v4 closure does not permanently retire the incident identifiers")

    manifest_reference = route["source_artifacts"]["manifest_v29"]
    manifest_path = repo_root / str(manifest_reference["path"])
    manifest = _regular_json(manifest_path)
    old_endpoint = next(
        item["endpoint"]
        for item in manifest.get("models") or []
        if (item.get("model") or {}).get("id") == "google/gemini-3.6-flash"
    )
    old_raw = raw_endpoint_contract(old_endpoint)
    new_raw = gemini_snapshot["raw_execution_contract"]
    differences: list[dict[str, Any]] = []
    for field in RAW_FIELDS:
        if old_raw.get(field) != new_raw.get(field):
            if field == "pricing":
                keys = sorted(set(old_raw[field]) | set(new_raw[field]))
                for key in keys:
                    if old_raw[field].get(key) != new_raw[field].get(key):
                        old_decimal = Decimal(str(old_raw[field].get(key)))
                        new_decimal = Decimal(str(new_raw[field].get(key)))
                        delta = abs(old_decimal - new_decimal)
                        differences.append(
                            {
                                "field": f"pricing.{key}",
                                "frozen_raw": old_raw[field].get(key),
                                "observed_raw": new_raw[field].get(key),
                                "raw_decimal_equal": old_decimal == new_decimal,
                                "raw_delta_usd_per_unit": _decimal_text(delta),
                                "raw_delta_usd_per_mtok_if_token_price": _decimal_text(
                                    delta * Decimal(1_000_000)
                                ),
                                "within_frozen_semantic_quantization": (
                                    _semantic_decimal_text(old_raw[field].get(key))
                                    == _semantic_decimal_text(new_raw[field].get(key))
                                ),
                            }
                        )
            else:
                differences.append(
                    {
                        "field": field,
                        "frozen_raw": old_raw.get(field),
                        "observed_raw": new_raw.get(field),
                        "raw_decimal_equal": None,
                        "within_frozen_semantic_quantization": False,
                    }
                )
    old_semantic = semantic_endpoint_contract(old_raw)
    new_semantic = semantic_endpoint_contract(new_raw)
    if (
        _sha256(old_raw) != V4_GEMINI_RAW_SHA256
        or _sha256(new_raw) != gemini_snapshot["raw_execution_contract_sha256"]
        or old_semantic != new_semantic
        or differences
        != [
            {
                "field": "pricing.input_cache_write",
                "frozen_raw": "0.00000004166666666666667",
                "observed_raw": "0.0000000416666666666667",
                "raw_decimal_equal": False,
                "raw_delta_usd_per_unit": "0.00000000000000000000003",
                "raw_delta_usd_per_mtok_if_token_price": "0.00000000000000003",
                "within_frozen_semantic_quantization": True,
            }
        ]
    ):
        raise RouteGateV5Error("the exact raw-only Gemini drift no longer rederives")

    reserve = reservation[0].get("reserved_usd")
    return {
        "schema_version": V4_INCIDENT_AUDIT_SCHEMA,
        "record_role": "source_reconstructed_v4_pre_generation_endpoint_contract_stop",
        "v4_artifacts": {
            "route_plan_sha256": route["artifact_sha256"],
            "receipt_sha256": receipt["artifact_sha256"],
            "audit_sha256": audit["artifact_sha256"],
            "closure_sha256": closure["artifact_sha256"],
            "ledger_sha256": _file_sha256(v4_ledger_path),
            "journal_sha256": _file_sha256(v4_journal_path),
        },
        "incident": {
            "work_item_id": V4_GEMINI_WORK_ITEM_ID,
            "run_id": V4_GEMINI_RUN_ID,
            "reservation_entry_sha256": V4_GEMINI_RESERVATION_SHA256,
            "incident_entry_sha256": V4_GEMINI_INCIDENT_SHA256,
            "exception_type": "RuntimeError",
            "exception_text": V4_ERROR_TEXT,
            "exception_sha256": V4_ERROR_SHA256,
            "journal_event_types": [item["event_type"] for item in journal],
        },
        "cause": {
            "stage": "post_catalog_pre_epicure_pre_generation",
            "frozen_raw_execution_contract_sha256": _sha256(old_raw),
            "observed_raw_execution_contract_sha256": _sha256(new_raw),
            "semantic_execution_contract_sha256": _sha256(old_semantic),
            "raw_differences": differences,
            "semantic_price_policy": old_semantic["pricing_normalization"],
            "classification": ("raw_decimal_rounding_drift_below_frozen_semantic_quantization"),
        },
        "request_boundary": {
            "openrouter_account_or_catalog_gets": 3,
            "provider_completion_requests": 0,
            "mcp_sessions": 0,
            "mcp_tool_calls": 0,
            "provider_attempt_events": 0,
            "complete_source_artifacts": 0,
        },
        "accounting": {
            "actual_generation_cost_usd": "0",
            "reserved_usd_retained_as_conservative_exposure": reserve,
            "reservation_released": False,
        },
        "closure": {
            "old_identifiers_replay_permitted": False,
            "retroactive_v4_admission": False,
            "fresh_identifiers_required": True,
        },
        "endpoint_snapshot": {
            "path": _relative(repo_root, endpoint_snapshot_path),
            "artifact_sha256": snapshot["artifact_sha256"],
        },
    }


def _closed_v4_identifiers(closure: Mapping[str, Any]) -> dict[str, set[str]]:
    values = closure.get("closed_identifiers") or {}
    return {
        key: {str(item) for item in values.get(key) or []}
        for key in (
            "route_cell_ids",
            "work_item_ids",
            "run_ids",
            "arm_ids",
            "attempt_ids",
            "generation_ids",
            "request_key_sha256s",
        )
    }


def _source_reference(repo_root: Path, path: Path, digest: str) -> dict[str, Any]:
    return {
        "path": _relative(repo_root, path),
        "bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
        "semantic_sha256": digest,
    }


def build_route_plan(
    *,
    v4_route_plan_path: Path,
    v4_receipt_path: Path,
    v4_audit_path: Path,
    v4_closure_path: Path,
    v4_incident_audit_path: Path,
    endpoint_snapshot_path: Path,
    v4_source_directory: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Freeze four fresh pairs while carrying two verified DeepSeek pairs by reference."""

    from .reasoning_effort_route_gate_v4 import _audit_pair_source, _source_map

    v4_route = _assert_artifact(v4_route_plan_path, V4_ROUTE_PLAN_SHA256)
    v4_receipt = _assert_artifact(v4_receipt_path, V4_RECEIPT_SHA256)
    v4_audit = _assert_artifact(v4_audit_path, V4_AUDIT_SHA256)
    v4_closure = _assert_artifact(v4_closure_path, V4_CLOSURE_SHA256)
    incident = _regular_json(v4_incident_audit_path)
    snapshot = _regular_json(endpoint_snapshot_path)
    if not _artifact_verifies(incident, V4_INCIDENT_AUDIT_SCHEMA):
        raise RouteGateV5Error("v4 incident audit does not verify")
    if not _artifact_verifies(snapshot, SNAPSHOT_SCHEMA):
        raise RouteGateV5Error("endpoint snapshot does not verify")
    if (
        incident.get("v4_artifacts", {}).get("closure_sha256") != V4_CLOSURE_SHA256
        or incident.get("endpoint_snapshot", {}).get("artifact_sha256")
        != snapshot.get("artifact_sha256")
        or incident.get("closure", {}).get("old_identifiers_replay_permitted") is not False
    ):
        raise RouteGateV5Error("incident audit does not bind the exact v4 closure and snapshot")

    v4_items = {
        (
            str(item["route_coordinate"]["model_id"]),
            str(item["route_coordinate"]["variant_id"]),
        ): item
        for item in v4_route["work_items"]
    }
    v4_sources = _source_map(v4_source_directory)
    accepted_prior_pairs: list[dict[str, Any]] = []
    for variant_id in ("provider_default", "explicit_high"):
        item = v4_items[("deepseek/deepseek-v4-flash-0731", variant_id)]
        source_path, source, source_digest = v4_sources[item["work_item_id"]]
        pair_audit = _audit_pair_source(
            route_plan=v4_route,
            work_item=item,
            source_path=source_path,
            source=source,
            source_digest=source_digest,
            repo_root=repo_root,
        )
        if pair_audit.get("decision") != "passed_all_predicates":
            raise RouteGateV5Error(f"DeepSeek {variant_id} source no longer passes v4 predicates")
        accepted_prior_pairs.append(
            {
                "model_id": item["route_coordinate"]["model_id"],
                "variant_id": variant_id,
                "work_item_id": item["work_item_id"],
                "run_id": item["run_id"],
                "source": _source_reference(repo_root, source_path, source_digest),
                "pair_audit_sha256": _sha256(pair_audit),
                "actual_cost_usd": pair_audit["accounting"]["actual_cost_usd"],
                "accepted_without_replay": True,
            }
        )
    if v4_receipt.get("source_artifacts") != [
        {
            "artifact_sha256": pair["source"]["semantic_sha256"],
            "path": pair["source"]["path"],
            "work_item_id": pair["work_item_id"],
        }
        for pair in accepted_prior_pairs
    ]:
        raise RouteGateV5Error("v4 receipt DeepSeek source inventory differs")
    pair_audits = v4_audit.get("pair_audits") or []
    if len(pair_audits) != 2 or any(
        pair.get("decision") != "passed_all_predicates" for pair in pair_audits
    ):
        raise RouteGateV5Error("v4 audit no longer records exactly two passed DeepSeek pairs")

    variants = copy.deepcopy(v4_route["variants"])
    model_records: list[dict[str, Any]] = []
    for endpoint_id in ENDPOINTS:
        snapshot_record = _endpoint_record(snapshot, endpoint_id)
        model_id = ENDPOINT_MODELS[endpoint_id]
        frozen_model = next(item for item in v4_route["models"] if item["model_id"] == model_id)
        model = snapshot_record["model"]
        raw = snapshot_record["raw_execution_contract"]
        if (
            model.get("id") != model_id
            or model.get("canonical_slug") != frozen_model["canonical_model_slug"]
            or raw.get("tag") != frozen_model["provider_endpoint"]
            or raw.get("provider_name") != frozen_model["actual_provider_name"]
            or "reasoning_effort" not in (raw.get("supported_parameters") or [])
            or "tools" not in (raw.get("supported_parameters") or [])
            or "max_tokens" not in (raw.get("supported_parameters") or [])
        ):
            raise RouteGateV5Error(f"current {endpoint_id} model/route/capabilities differ")
        model_records.append(
            {
                "endpoint_id": endpoint_id,
                "model_id": model_id,
                "canonical_model_slug": model["canonical_slug"],
                "provider_endpoint": raw["tag"],
                "actual_provider_name": raw["provider_name"],
                "provider_controls": frozen_model["provider_controls"],
                "snapshot_raw_execution_contract_sha256": snapshot_record[
                    "raw_execution_contract_sha256"
                ],
                "semantic_execution_contract_sha256": snapshot_record[
                    "semantic_execution_contract_sha256"
                ],
                "supported_efforts": frozen_model["supported_efforts"],
                "provider_default_effort": frozen_model["provider_default_effort"],
                "provider_default_mandatory": frozen_model["provider_default_mandatory"],
            }
        )

    new_items: list[dict[str, Any]] = []
    for endpoint_id in ENDPOINTS:
        model = next(item for item in model_records if item["endpoint_id"] == endpoint_id)
        for variant in variants:
            old = v4_items[(model["model_id"], variant["variant_id"])]
            coordinate = {
                "schema_version": "flavourbench-reasoning-effort-route-coordinate-v5",
                "freeze_nonce": FREEZE_NONCE,
                "endpoint_snapshot_sha256": snapshot["artifact_sha256"],
                "endpoint_id": endpoint_id,
                "model_id": model["model_id"],
                "canonical_model_slug": model["canonical_model_slug"],
                "provider_endpoint": model["provider_endpoint"],
                "actual_provider_name": model["actual_provider_name"],
                "snapshot_raw_execution_contract_sha256": model[
                    "snapshot_raw_execution_contract_sha256"
                ],
                "endpoint_execution_contract_sha256": model[
                    "snapshot_raw_execution_contract_sha256"
                ],
                "semantic_execution_contract_sha256": model["semantic_execution_contract_sha256"],
                "provider_controls": model["provider_controls"],
                "task_id": v4_route["task"]["task_id"],
                "prompt_sha256": v4_route["task"]["prompt_sha256"],
                "variant_id": variant["variant_id"],
                "intermediate_reasoning_effort": variant["intermediate_reasoning_effort"],
                "final_reasoning_effort": variant["final_reasoning_effort"],
                "epicure_bundle_sha256": v4_route["epicure"]["bundle_sha256"],
                "epicure_application_sha256": v4_route["epicure"]["application_sha256"],
                "epicure_tool_schema_sha256": v4_route["epicure"]["tool_schema_sha256"],
            }
            route_cell_id = _sha256(coordinate)
            work_item_id = _sha256(
                {
                    "route_cell_id": route_cell_id,
                    "role": "effort-v5-endpoint-isolated-gate",
                }
            )
            run_id = str(uuid.uuid5(NAMESPACE, f"{route_cell_id}:{work_item_id}"))
            new_items.append(
                {
                    "endpoint_id": endpoint_id,
                    "route_coordinate": coordinate,
                    "route_cell_id": route_cell_id,
                    "work_item_id": work_item_id,
                    "run_id": run_id,
                    "arm_ids": [f"{run_id}:{condition}" for condition in CONDITIONS],
                    "attempt_slots": _attempt_slots(run_id, route_cell_id),
                    "worst_case_reserve_usd": old["worst_case_reserve_usd"],
                    "diagnostic_outputs_reused": False,
                }
            )

    prior = _closed_v4_identifiers(v4_closure)
    for key, values in (
        ("route_cell_ids", [item["route_cell_id"] for item in new_items]),
        ("work_item_ids", [item["work_item_id"] for item in new_items]),
        ("run_ids", [item["run_id"] for item in new_items]),
        ("arm_ids", [value for item in new_items for value in item["arm_ids"]]),
        (
            "attempt_ids",
            [slot["attempt_id"] for item in new_items for slot in item["attempt_slots"]],
        ),
    ):
        if len(values) != len(set(values)) or set(values) & prior[key]:
            raise RouteGateV5Error(f"fresh v5 {key} overlap sibling or closed v4 identifiers")

    new_reserve = sum(Decimal(item["worst_case_reserve_usd"]) for item in new_items)
    if new_reserve != Decimal("3.650511"):
        raise RouteGateV5Error("four-pair v5 reserve differs from the frozen envelope")
    module_path = Path(__file__).resolve()
    return {
        "schema_version": ROUTE_PLAN_SCHEMA,
        "record_role": "endpoint_isolated_reasoning_effort_default_high_route_gate",
        "freeze_nonce": FREEZE_NONCE,
        "task": v4_route["task"],
        "epicure": v4_route["epicure"],
        "variants": variants,
        "models": model_records,
        "accepted_prior_pairs": accepted_prior_pairs,
        "work_items": new_items,
        "execution_order_by_endpoint": {
            endpoint_id: [
                item["work_item_id"] for item in new_items if item["endpoint_id"] == endpoint_id
            ]
            for endpoint_id in ENDPOINTS
        },
        "confirmation_by_endpoint": dict(CONFIRMATIONS),
        "budget": {
            "currency": "USD",
            "v4_actual_cost_usd": "0.006154",
            "v4_orphan_reserve_retained_usd": "0.6765315",
            "new_four_pair_worst_case_usd": _decimal_text(new_reserve),
            "hard_cap_usd": "100",
            "admission_ceiling_usd": "85",
        },
        "source_artifacts": {
            **v4_route["source_artifacts"],
            "v4_route_plan": _source_reference(
                repo_root, v4_route_plan_path, v4_route["artifact_sha256"]
            ),
            "v4_receipt": _source_reference(
                repo_root, v4_receipt_path, v4_receipt["artifact_sha256"]
            ),
            "v4_audit": _source_reference(repo_root, v4_audit_path, v4_audit["artifact_sha256"]),
            "v4_closure": _source_reference(
                repo_root, v4_closure_path, v4_closure["artifact_sha256"]
            ),
            "v4_incident_audit": _source_reference(
                repo_root, v4_incident_audit_path, incident["artifact_sha256"]
            ),
            "endpoint_snapshot": _source_reference(
                repo_root, endpoint_snapshot_path, snapshot["artifact_sha256"]
            ),
        },
        "source_code": {
            "path": _relative(repo_root, module_path),
            "bytes": module_path.stat().st_size,
            "sha256": _file_sha256(module_path),
            "generation_core": v4_route["source_code"],
        },
        "counts": {
            "intended_pairs": 6,
            "accepted_prior_pairs": 2,
            "fresh_pairs": 4,
            "fresh_response_arms": 8,
            "models": 3,
            "effort_variants": 2,
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "isolation": {
            "separate_ledgers": True,
            "separate_source_directories": True,
            "separate_confirmation_tokens": True,
            "failure_closes_only_selected_endpoint": True,
            "aggregate_pass_requires_both_endpoint_passes": True,
        },
        "acceptance": {
            "raw_snapshot_retained": True,
            "semantic_contract_controls_admission": True,
            "decimal_equivalent_price_spelling_tolerated": True,
            "provider_model_capability_or_semantic_price_drift_tolerated": False,
            "old_v4_identifiers_replay_permitted": False,
            "new_endpoint_identifiers_replay_permitted": False,
            "all_generation_costs_reconciled": True,
            "complete_source_and_journal_reconstruction_required": True,
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "official": False,
            "rank_eligible": False,
            "enters_sensitivity_fit": False,
        },
    }


def validate_route_plan(plan: Mapping[str, Any], *, repo_root: Path) -> None:
    if not _artifact_verifies(plan, ROUTE_PLAN_SCHEMA):
        raise RouteGateV5Error("v5 route plan content address or schema does not verify")
    source = plan.get("source_code") or {}
    module_path = repo_root / str(source.get("path") or "")
    if (
        source.get("sha256") != _file_sha256(module_path)
        or source.get("bytes") != module_path.stat().st_size
    ):
        raise RouteGateV5Error("v5 executor source differs from the frozen route plan")
    if (
        plan.get("counts", {}).get("accepted_prior_pairs") != 2
        or len(plan.get("work_items") or []) != 4
    ):
        raise RouteGateV5Error("v5 route plan does not contain two prior and four fresh pairs")
    for name, reference in (plan.get("source_artifacts") or {}).items():
        if not isinstance(reference, Mapping) or "path" not in reference:
            continue
        path = repo_root / str(reference["path"])
        if _file_sha256(path) != reference.get(
            "file_sha256"
        ) or path.stat().st_size != reference.get("bytes"):
            raise RouteGateV5Error(f"v5 source artifact differs: {name}")


def _endpoint_items(plan: Mapping[str, Any], endpoint_id: str) -> list[dict[str, Any]]:
    if endpoint_id not in ENDPOINTS:
        raise RouteGateV5Error(f"unknown endpoint: {endpoint_id}")
    items = [
        dict(item)
        for item in plan.get("work_items") or []
        if item.get("endpoint_id") == endpoint_id
    ]
    if [item["work_item_id"] for item in items] != plan.get("execution_order_by_endpoint", {}).get(
        endpoint_id
    ):
        raise RouteGateV5Error(f"{endpoint_id} execution order differs from the freeze")
    if len(items) != 2:
        raise RouteGateV5Error(f"{endpoint_id} does not contain exactly two variants")
    return items


async def _one_endpoint_snapshot(
    *, api_base: str, api_key: str, endpoint_id: str
) -> dict[str, Any]:
    if not api_key:
        raise RouteGateV5Error("OpenRouter API key is absent")
    model_id = ENDPOINT_MODELS[endpoint_id]
    provider = "google-ai-studio/flex" if endpoint_id == "gemini" else "anthropic"
    async with httpx.AsyncClient(
        base_url=api_base.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    ) as client:
        model, endpoint, dates = await _catalog_endpoint(client, model_id, provider)
    raw = raw_endpoint_contract(endpoint)
    semantic = semantic_endpoint_contract(endpoint)
    return {
        "schema_version": "flavourbench-reasoning-effort-endpoint-admission-attestation-v5",
        "record_role": "pre_reservation_zero_generation_endpoint_attestation",
        "endpoint_id": endpoint_id,
        "model": model,
        "raw_execution_contract": raw,
        "raw_execution_contract_sha256": _sha256(raw),
        "semantic_execution_contract": semantic,
        "semantic_execution_contract_sha256": _sha256(semantic),
        "response_dates": dates,
        "counts": {
            "catalog_http_gets": 2,
            "provider_completion_requests": 0,
            "epicure_calls": 0,
        },
    }


def _source_map(source_directory: Path) -> dict[str, tuple[Path, dict[str, Any], str]]:
    from .frontier_contract_runner import _verify_live_artifact, scan_live_smoke_artifacts

    result: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for exposure in scan_live_smoke_artifacts(source_directory).artifacts:
        document, digest = _verify_live_artifact(exposure.path)
        work_item_id = str(document.get("dataset_work_item_id") or "")
        if not work_item_id or work_item_id in result:
            raise RouteGateV5Error("source directory has absent or duplicate work-item identity")
        result[work_item_id] = (exposure.path, document, digest)
    return result


def _v5_accounting(
    *,
    plan: Mapping[str, Any],
    v4_receipt: Mapping[str, Any],
    endpoint_roots: Mapping[str, Path],
    selected_endpoint: str,
) -> dict[str, Any]:
    from .frontier_coverage_repair_executor import SupplementalRun, _run_accounting
    from .real_dataset_runner import dataset_ledger_state, load_dataset_ledger

    if v4_receipt.get("artifact_sha256") != V4_RECEIPT_SHA256:
        raise RouteGateV5Error("budget baseline is not the exact v4 execution receipt")
    baseline = Decimal(v4_receipt["final_budget"]["current_total_exposure_usd"])
    current = baseline
    actual = Decimal(0)
    orphan = Decimal(0)
    endpoint_blockers: dict[str, list[dict[str, Any]]] = {}
    reserved: set[str] = set()
    for endpoint_id, root in endpoint_roots.items():
        accounting = _run_accounting(
            SupplementalRun(source_directory=root / "source", ledger_path=root / "ledger.jsonl"),
            label=f"reasoning_effort_route_gate_v5_{endpoint_id}",
        )
        current += accounting.exposure_usd + accounting.orphan_reservation_usd
        actual += accounting.actual_cost_usd
        orphan += accounting.orphan_reservation_usd
        endpoint_blockers[endpoint_id] = [dict(item) for item in accounting.blockers]
        reservations, _ = dataset_ledger_state(load_dataset_ledger(root / "ledger.jsonl"))
        reserved.update(reservations)
    outstanding = sum(
        (
            Decimal(str(item["worst_case_reserve_usd"]))
            for item in plan["work_items"]
            if item["work_item_id"] not in reserved
        ),
        Decimal(0),
    )
    projected = current + outstanding
    selected_blockers = endpoint_blockers.get(selected_endpoint, [])
    return {
        "currency": "USD",
        "v4_budget_baseline_receipt_sha256": V4_RECEIPT_SHA256,
        "v4_total_exposure_including_retained_orphan_usd": _decimal_text(baseline),
        "v5_actual_cost_usd": _decimal_text(actual),
        "v5_orphan_reservation_usd": _decimal_text(orphan),
        "current_total_exposure_usd": _decimal_text(current),
        "outstanding_v5_worst_case_usd": _decimal_text(outstanding),
        "projected_total_exposure_usd": _decimal_text(projected),
        "admission_ceiling_usd": "85",
        "hard_cap_usd": "100",
        "selected_endpoint": selected_endpoint,
        "selected_endpoint_blockers": selected_blockers,
        "other_endpoint_blockers_retained_but_not_propagated": {
            key: value for key, value in endpoint_blockers.items() if key != selected_endpoint
        },
        "admission_allowed": not selected_blockers and projected <= Decimal("85"),
    }


def _adapted_pair_audit(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    source_path: Path,
    source: Mapping[str, Any],
    digest: str,
    repo_root: Path,
) -> dict[str, Any]:
    from .reasoning_effort_route_gate_v4 import _audit_pair_source

    endpoint_contract = source.get("endpoint_contract") or {}
    raw = raw_endpoint_contract(endpoint_contract)
    raw_sha = _sha256(raw)
    semantic = semantic_endpoint_contract(raw)
    adapted_item = copy.deepcopy(dict(item))
    adapted_item["route_coordinate"]["endpoint_execution_contract_sha256"] = raw_sha
    pair = _audit_pair_source(
        route_plan=plan,
        work_item=adapted_item,
        source_path=source_path,
        source=source,
        source_digest=digest,
        repo_root=repo_root,
    )
    failures = list(pair.get("failures") or [])
    if source.get("endpoint_execution_contract_sha256") != raw_sha:
        failures.append("source_raw_endpoint_contract_does_not_rederive")
    if _sha256(semantic) != item["route_coordinate"]["semantic_execution_contract_sha256"]:
        failures.append("source_endpoint_semantic_contract_differs_from_v5_freeze")
    pair["endpoint_contract_v5"] = {
        "raw_execution_contract_sha256": raw_sha,
        "semantic_execution_contract_sha256": _sha256(semantic),
        "frozen_semantic_execution_contract_sha256": item["route_coordinate"][
            "semantic_execution_contract_sha256"
        ],
    }
    pair["failures"] = sorted(set(failures))
    pair["decision"] = "passed_all_predicates" if not pair["failures"] else "failed"
    return pair


def _journal_for_run(source_directory: Path, run_id: str) -> Path | None:
    matches = [
        path
        for path in source_directory.glob(f"*journal*{run_id}*.jsonl")
        if path.is_file() and not path.is_symlink()
    ]
    return matches[0] if len(matches) == 1 else None


async def execute_endpoint(
    *,
    plan: Mapping[str, Any],
    endpoint_id: str,
    endpoint_root: Path,
    peer_endpoint_root: Path,
    v4_receipt: Mapping[str, Any],
    repo_root: Path,
    global_budget_lock_path: Path,
    api_base: str,
    api_key: str,
) -> dict[str, Any]:
    from .config import get_settings
    from .frontier_contract_runner import AdmissionDenied, _exclusive_runner_lock
    from .live_smoke import live_smoke
    from .real_dataset_runner import (
        _dataset_ledger_lock,
        append_dataset_ledger_event,
        dataset_ledger_state,
        load_dataset_ledger,
    )
    from .reasoning_effort_route_gate_v4 import (
        _live_args,
        _policy_environment,
        _require_live_environment_before_reservation,
        _variant_policy,
    )

    _require_live_environment_before_reservation()
    source_directory = endpoint_root / "source"
    ledger_path = endpoint_root / "ledger.jsonl"
    output_directory = endpoint_root / "receipts"
    source_directory.mkdir(parents=True, exist_ok=True)
    endpoint_root.mkdir(parents=True, exist_ok=True)
    endpoint_roots = {
        endpoint_id: endpoint_root,
        next(value for value in ENDPOINTS if value != endpoint_id): peer_endpoint_root,
    }
    outcomes: list[dict[str, Any]] = []
    attestations: list[dict[str, Any]] = []
    new_invocations = 0
    with _exclusive_runner_lock(global_budget_lock_path):
        with _dataset_ledger_lock(ledger_path):
            for item in _endpoint_items(plan, endpoint_id):
                entries = load_dataset_ledger(ledger_path)
                reservations, finalizations = dataset_ledger_state(entries)
                sources = _source_map(source_directory)
                work_item_id = item["work_item_id"]
                if work_item_id in finalizations:
                    passed = finalizations[work_item_id].get("route_gate_pair_passed") is True
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "skip_finalized_pass"
                            if passed
                            else "stop_finalized_failure",
                        }
                    )
                    if not passed:
                        break
                    continue
                if work_item_id in reservations:
                    source_record = sources.get(work_item_id)
                    if source_record is None:
                        outcomes.append(
                            {
                                "work_item_id": work_item_id,
                                "decision": "stop_reserved_without_source_no_replay",
                            }
                        )
                        break
                    path, source, digest = source_record
                    pair = _adapted_pair_audit(
                        plan=plan,
                        item=item,
                        source_path=path,
                        source=source,
                        digest=digest,
                        repo_root=repo_root,
                    )
                    final = append_dataset_ledger_event(
                        ledger_path,
                        {
                            "event_type": "source_artifact_recorded",
                            "runner_run_id": f"reasoning-effort-v5-{endpoint_id}",
                            "work_item_id": work_item_id,
                            "reservation_entry_sha256": reservations[work_item_id]["entry_sha256"],
                            "source_artifact_sha256": digest,
                            "source_path": _relative(repo_root, path),
                            "route_gate_pair_passed": pair["decision"] == "passed_all_predicates",
                            "pair_audit_sha256": _sha256(pair),
                            "actual_cost_usd": pair["accounting"]["actual_cost_usd"],
                            "quality_observations": 0,
                            "rank_eligible": False,
                        },
                    )
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "recovered_source_without_provider_call",
                            "passed": pair["decision"] == "passed_all_predicates",
                            "ledger_entry_sha256": final["entry_sha256"],
                        }
                    )
                    if pair["decision"] != "passed_all_predicates":
                        break
                    continue
                if work_item_id in sources:
                    raise RouteGateV5Error("source exists without a prior reservation")

                attestation = await _one_endpoint_snapshot(
                    api_base=api_base, api_key=api_key, endpoint_id=endpoint_id
                )
                attestation_path = _write_artifact(
                    endpoint_root / "endpoint-attestations",
                    f"{endpoint_id}-pre-admission-{item['route_coordinate']['variant_id']}",
                    attestation,
                )
                attestation_document = _regular_json(attestation_path)
                attestations.append(
                    {
                        "work_item_id": work_item_id,
                        "path": _relative(repo_root, attestation_path),
                        "artifact_sha256": attestation_document["artifact_sha256"],
                    }
                )
                frozen_semantic = item["route_coordinate"]["semantic_execution_contract_sha256"]
                if attestation["semantic_execution_contract_sha256"] != frozen_semantic:
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "stop_semantic_endpoint_drift_before_reservation",
                            "frozen_semantic_sha256": frozen_semantic,
                            "observed_semantic_sha256": attestation[
                                "semantic_execution_contract_sha256"
                            ],
                        }
                    )
                    break
                budget = _v5_accounting(
                    plan=plan,
                    v4_receipt=v4_receipt,
                    endpoint_roots=endpoint_roots,
                    selected_endpoint=endpoint_id,
                )
                if budget["admission_allowed"] is not True:
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "stop_budget_before_reservation",
                            "budget": budget,
                        }
                    )
                    break
                coordinate = item["route_coordinate"]
                reservation = append_dataset_ledger_event(
                    ledger_path,
                    {
                        "event_type": "reservation_created",
                        "runner_run_id": f"reasoning-effort-v5-{endpoint_id}",
                        "work_item_id": work_item_id,
                        "route_plan_sha256": plan["artifact_sha256"],
                        "route_cell_id": item["route_cell_id"],
                        "run_id": item["run_id"],
                        "arm_ids": item["arm_ids"],
                        "model_id": coordinate["model_id"],
                        "canonical_model_slug": coordinate["canonical_model_slug"],
                        "provider_endpoint": coordinate["provider_endpoint"],
                        "actual_provider_name": coordinate["actual_provider_name"],
                        "endpoint_snapshot_sha256": coordinate["endpoint_snapshot_sha256"],
                        "raw_endpoint_execution_contract_sha256": attestation[
                            "raw_execution_contract_sha256"
                        ],
                        "semantic_endpoint_execution_contract_sha256": frozen_semantic,
                        "endpoint_attestation_sha256": attestation_document["artifact_sha256"],
                        "variant_id": coordinate["variant_id"],
                        "intermediate_reasoning_effort": coordinate[
                            "intermediate_reasoning_effort"
                        ],
                        "final_reasoning_effort": coordinate["final_reasoning_effort"],
                        "conditions": list(CONDITIONS),
                        "reserved_usd": item["worst_case_reserve_usd"],
                        "total_exposure_before_usd": budget["current_total_exposure_usd"],
                        "projected_all_remaining_usd": budget["projected_total_exposure_usd"],
                        "replay_permitted": False,
                        "quality_observations": 0,
                        "rank_eligible": False,
                    },
                )
                policy = _variant_policy(plan, item, repo_root)
                args = _live_args(
                    route_plan=plan,
                    work_item=item,
                    repo_root=repo_root,
                    source_directory=source_directory,
                )
                args.expected_endpoint_execution_sha256 = attestation[
                    "raw_execution_contract_sha256"
                ]
                try:
                    with _policy_environment(
                        policy=policy,
                        endpoint=attestation["raw_execution_contract"],
                    ):
                        settings = get_settings()
                        if settings.execution_mode != "live" or not settings.live_authorized:
                            raise AdmissionDenied("live authority changed after reservation")
                        new_invocations += 1
                        summary = await live_smoke(args)
                except Exception as error:
                    journal_path = _journal_for_run(source_directory, item["run_id"])
                    journal_descriptor: dict[str, Any] | None = None
                    if journal_path is not None:
                        journal_entries = _hash_chain_jsonl(journal_path)
                        journal_descriptor = {
                            "path": _relative(repo_root, journal_path),
                            "sha256": _file_sha256(journal_path),
                            "entry_count": len(journal_entries),
                            "event_types": [entry["event_type"] for entry in journal_entries],
                            "provider_attempt_events": sum(
                                entry["event_type"] == "provider_attempt"
                                for entry in journal_entries
                            ),
                            "mcp_trace_events": sum(
                                entry["event_type"] == "mcp_trace" for entry in journal_entries
                            ),
                        }
                    incident = append_dataset_ledger_event(
                        ledger_path,
                        {
                            "event_type": "execution_incident",
                            "runner_run_id": f"reasoning-effort-v5-{endpoint_id}",
                            "work_item_id": work_item_id,
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "endpoint_attestation_sha256": attestation_document["artifact_sha256"],
                            "incident": "reservation_retained_endpoint_closed_no_replay",
                            "error_type": type(error).__name__,
                            "error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                            "journal": journal_descriptor,
                            "replay_permitted": False,
                        },
                    )
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "execution_incident_endpoint_closed_no_replay",
                            "incident_entry_sha256": incident["entry_sha256"],
                        }
                    )
                    break
                artifact_path = Path(str((summary or {}).get("artifact") or ""))
                if (
                    not artifact_path.is_file()
                    or artifact_path.resolve().parent != source_directory.resolve()
                ):
                    raise RouteGateV5Error(
                        "live smoke returned no source inside the endpoint directory"
                    )
                source, digest = _source_map(source_directory)[work_item_id][1:]
                pair = _adapted_pair_audit(
                    plan=plan,
                    item=item,
                    source_path=artifact_path,
                    source=source,
                    digest=digest,
                    repo_root=repo_root,
                )
                passed = pair["decision"] == "passed_all_predicates"
                final = append_dataset_ledger_event(
                    ledger_path,
                    {
                        "event_type": "source_artifact_recorded",
                        "runner_run_id": f"reasoning-effort-v5-{endpoint_id}",
                        "work_item_id": work_item_id,
                        "reservation_entry_sha256": reservation["entry_sha256"],
                        "source_artifact_sha256": digest,
                        "source_path": _relative(repo_root, artifact_path),
                        "route_gate_pair_passed": passed,
                        "pair_audit_sha256": _sha256(pair),
                        "actual_cost_usd": pair["accounting"]["actual_cost_usd"],
                        "quality_observations": 0,
                        "rank_eligible": False,
                    },
                )
                outcomes.append(
                    {
                        "work_item_id": work_item_id,
                        "decision": "source_finalized_pass"
                        if passed
                        else "source_finalized_failure",
                        "source_artifact_sha256": digest,
                        "ledger_entry_sha256": final["entry_sha256"],
                        "failures": pair["failures"],
                    }
                )
                if not passed:
                    break
    final_budget = _v5_accounting(
        plan=plan,
        v4_receipt=v4_receipt,
        endpoint_roots=endpoint_roots,
        selected_endpoint=endpoint_id,
    )
    entries = load_dataset_ledger(ledger_path)
    _, finalizations = dataset_ledger_state(entries)
    sources = _source_map(source_directory)
    passed = len(finalizations) == 2 and all(
        item.get("route_gate_pair_passed") is True for item in finalizations.values()
    )
    receipt = {
        "schema_version": ENDPOINT_RECEIPT_SCHEMA,
        "record_role": "endpoint_isolated_two_pair_reasoning_effort_receipt",
        "route_plan_sha256": plan["artifact_sha256"],
        "endpoint_id": endpoint_id,
        "status": "two_pair_sources_available" if passed else "failed_or_incomplete_closed",
        "new_pair_invocations_this_command": new_invocations,
        "total_source_pairs": len(sources),
        "total_finalized_pairs": len(finalizations),
        "source_artifacts": [
            {
                "work_item_id": work_item_id,
                "path": _relative(repo_root, path),
                "artifact_sha256": digest,
            }
            for work_item_id, (path, _, digest) in sorted(sources.items())
        ],
        "endpoint_attestations": attestations,
        "ledger": {
            "path": _relative(repo_root, ledger_path),
            "sha256": _file_sha256(ledger_path)
            if ledger_path.exists()
            else hashlib.sha256(b"").hexdigest(),
            "entry_count": len(entries),
            "head_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        },
        "outcomes": outcomes,
        "final_budget": final_budget,
        "failed_other_endpoint": False,
        "other_endpoint_execution_blocked": False,
        "uncertain_delivery_replayed": False,
        "quality_observations": 0,
        "rank_eligible": False,
    }
    path = _write_artifact(output_directory, f"reasoning-effort-v5-{endpoint_id}-receipt", receipt)
    return {"path": str(path), "document": _regular_json(path)}


def build_endpoint_audit(
    *,
    plan: Mapping[str, Any],
    endpoint_id: str,
    receipt_path: Path,
    endpoint_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    from .real_dataset_runner import dataset_ledger_state, load_dataset_ledger

    receipt = _regular_json(receipt_path)
    ledger_path = endpoint_root / "ledger.jsonl"
    source_directory = endpoint_root / "source"
    failures: list[str] = []
    if not _artifact_verifies(receipt, ENDPOINT_RECEIPT_SCHEMA):
        failures.append("receipt_content_address_or_schema_failed")
    if (
        receipt.get("route_plan_sha256") != plan.get("artifact_sha256")
        or receipt.get("endpoint_id") != endpoint_id
        or receipt.get("uncertain_delivery_replayed") is not False
        or receipt.get("other_endpoint_execution_blocked") is not False
    ):
        failures.append("receipt_route_endpoint_or_isolation_binding_failed")
    entries = load_dataset_ledger(ledger_path)
    reservations, finalizations = dataset_ledger_state(entries)
    sources = _source_map(source_directory)
    known = {item["work_item_id"] for item in _endpoint_items(plan, endpoint_id)}
    if (set(reservations) | set(finalizations) | set(sources)) - known:
        failures.append("unknown_work_item_in_endpoint_evidence")
    pair_audits: list[dict[str, Any]] = []
    incident_audits: list[dict[str, Any]] = []
    all_attempts: set[str] = set()
    all_generations: set[str] = set()
    all_request_keys: set[str] = set()
    for item in _endpoint_items(plan, endpoint_id):
        work_item_id = item["work_item_id"]
        source_record = sources.get(work_item_id)
        finalization = finalizations.get(work_item_id)
        if source_record is not None:
            path, source, digest = source_record
            pair = _adapted_pair_audit(
                plan=plan,
                item=item,
                source_path=path,
                source=source,
                digest=digest,
                repo_root=repo_root,
            )
            pair_audits.append(pair)
            if (
                finalization is None
                or finalization.get("source_artifact_sha256") != digest
                or finalization.get("route_gate_pair_passed")
                is not (pair["decision"] == "passed_all_predicates")
            ):
                failures.append(f"ledger_finalization_mismatch:{work_item_id}")
            for name, target in (
                ("attempt_ids", all_attempts),
                ("generation_ids", all_generations),
                ("request_key_sha256s", all_request_keys),
            ):
                observed = set(pair["identifiers"][name])
                if target & observed:
                    failures.append(f"cross_pair_{name}_overlap")
                target.update(observed)
            if pair["decision"] != "passed_all_predicates":
                failures.append(f"pair_predicate_failure:{work_item_id}")
            continue
        incidents = [
            entry
            for entry in entries
            if entry.get("event_type") == "execution_incident"
            and entry.get("work_item_id") == work_item_id
        ]
        if incidents:
            incident = incidents[-1]
            journal = incident.get("journal")
            reconstructed = False
            provider_attempts: int | None = None
            mcp_traces: int | None = None
            if isinstance(journal, Mapping):
                journal_path = repo_root / str(journal.get("path") or "")
                try:
                    journal_entries = _hash_chain_jsonl(journal_path)
                    reconstructed = (
                        _file_sha256(journal_path) == journal.get("sha256")
                        and len(journal_entries) == journal.get("entry_count")
                        and [entry["event_type"] for entry in journal_entries]
                        == journal.get("event_types")
                    )
                    provider_attempts = sum(
                        entry["event_type"] == "provider_attempt" for entry in journal_entries
                    )
                    mcp_traces = sum(
                        entry["event_type"] == "mcp_trace" for entry in journal_entries
                    )
                except (OSError, ValueError, TypeError, RouteGateV5Error):
                    reconstructed = False
            incident_audits.append(
                {
                    "work_item_id": work_item_id,
                    "incident_entry_sha256": incident.get("entry_sha256"),
                    "journal_reconstructed": reconstructed,
                    "provider_attempt_events": provider_attempts,
                    "mcp_trace_events": mcp_traces,
                    "pre_generation_failure": reconstructed and provider_attempts == 0,
                    "reservation_retained": work_item_id in reservations,
                    "replay_permitted": False,
                }
            )
        failures.append(f"missing_complete_source:{work_item_id}")
    if len(pair_audits) != 2:
        failures.append("both_endpoint_pairs_required")
    planned_attempts = {
        slot["attempt_id"]
        for item in _endpoint_items(plan, endpoint_id)
        for slot in item["attempt_slots"]
    }
    if not all_attempts <= planned_attempts:
        failures.append("attempt_outside_frozen_endpoint_pool")
    unique_failures = sorted(set(failures))
    passed = not unique_failures
    actual_micros = sum(int(pair["accounting"]["actual_cost_micros"]) for pair in pair_audits)
    return {
        "schema_version": ENDPOINT_AUDIT_SCHEMA,
        "record_role": "source_reconstructed_endpoint_isolated_reasoning_effort_audit",
        "route_plan_sha256": plan["artifact_sha256"],
        "endpoint_id": endpoint_id,
        "receipt": {
            "path": _relative(repo_root, receipt_path),
            "artifact_sha256": receipt.get("artifact_sha256"),
        },
        "ledger": {
            "path": _relative(repo_root, ledger_path),
            "sha256": _file_sha256(ledger_path)
            if ledger_path.exists()
            else hashlib.sha256(b"").hexdigest(),
            "entry_count": len(entries),
            "head_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        },
        "decision": "passed_all_predicates" if passed else "failed_one_or_more_predicates",
        "failures": unique_failures,
        "pair_audits": pair_audits,
        "incident_audits": incident_audits,
        "identifier_audit": {
            "planned_attempt_ids": sorted(planned_attempts),
            "observed_attempt_ids": sorted(all_attempts),
            "observed_generation_ids": sorted(all_generations),
            "observed_request_key_sha256s": sorted(all_request_keys),
        },
        "counts": {
            "required_pairs": 2,
            "source_verified_pairs": len(pair_audits),
            "usable_pairs": 2 if passed else 0,
            "usable_arms": 4 if passed else 0,
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "accounting": {
            "actual_cost_micros": actual_micros,
            "actual_cost_usd": _decimal_text(Decimal(actual_micros) / Decimal(1_000_000)),
            "all_source_generation_costs_reconciled": all(
                pair["accounting"]["reconciled"] is True for pair in pair_audits
            ),
        },
        "isolation": {
            "other_endpoint_evidence_read": False,
            "other_endpoint_closed": False,
            "failure_scope": endpoint_id,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def build_endpoint_closure(
    *, plan: Mapping[str, Any], endpoint_id: str, audit: Mapping[str, Any]
) -> dict[str, Any]:
    if not _artifact_verifies(audit, ENDPOINT_AUDIT_SCHEMA):
        raise RouteGateV5Error("endpoint audit does not verify")
    items = _endpoint_items(plan, endpoint_id)
    identifiers = audit.get("identifier_audit") or {}
    return {
        "schema_version": ENDPOINT_CLOSURE_SCHEMA,
        "record_role": "permanent_endpoint_scoped_reasoning_effort_v5_closure",
        "route_plan_sha256": plan["artifact_sha256"],
        "endpoint_id": endpoint_id,
        "endpoint_audit_sha256": audit["artifact_sha256"],
        "closed_identifiers": {
            "route_cell_ids": sorted(item["route_cell_id"] for item in items),
            "work_item_ids": sorted(item["work_item_id"] for item in items),
            "run_ids": sorted(item["run_id"] for item in items),
            "arm_ids": sorted(value for item in items for value in item["arm_ids"]),
            "attempt_ids": sorted(identifiers.get("planned_attempt_ids") or []),
            "used_attempt_ids": sorted(identifiers.get("observed_attempt_ids") or []),
            "generation_ids": sorted(identifiers.get("observed_generation_ids") or []),
            "request_key_sha256s": sorted(identifiers.get("observed_request_key_sha256s") or []),
            "replay_permitted": False,
        },
        "decision": {
            "endpoint_qualified": audit.get("decision") == "passed_all_predicates",
            "endpoint_identifiers_permanently_closed": True,
            "other_endpoint_execution_blocked": False,
            "aggregate_gate_qualified": False,
        },
        "incident_classification": audit.get("incident_audits"),
        "cost": audit.get("accounting"),
        "claim_boundary": audit.get("claim_boundary"),
    }


def build_aggregate_audit(
    *,
    plan: Mapping[str, Any],
    gemini_audit_path: Path,
    gemini_closure_path: Path,
    sonnet_audit_path: Path,
    sonnet_closure_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    endpoint_inputs: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for endpoint_id, audit_path, closure_path in (
        ("gemini", gemini_audit_path, gemini_closure_path),
        ("sonnet", sonnet_audit_path, sonnet_closure_path),
    ):
        audit = _regular_json(audit_path)
        closure = _regular_json(closure_path)
        if not _artifact_verifies(audit, ENDPOINT_AUDIT_SCHEMA):
            failures.append(f"{endpoint_id}_audit_does_not_verify")
        if not _artifact_verifies(closure, ENDPOINT_CLOSURE_SCHEMA):
            failures.append(f"{endpoint_id}_closure_does_not_verify")
        if (
            audit.get("route_plan_sha256") != plan.get("artifact_sha256")
            or audit.get("endpoint_id") != endpoint_id
            or closure.get("endpoint_audit_sha256") != audit.get("artifact_sha256")
            or closure.get("endpoint_id") != endpoint_id
            or closure.get("closed_identifiers", {}).get("replay_permitted") is not False
        ):
            failures.append(f"{endpoint_id}_audit_closure_binding_failed")
        if audit.get("decision") != "passed_all_predicates":
            failures.append(f"{endpoint_id}_did_not_pass")
        endpoint_inputs[endpoint_id] = {
            "audit": {
                "path": _relative(repo_root, audit_path),
                "artifact_sha256": audit.get("artifact_sha256"),
            },
            "closure": {
                "path": _relative(repo_root, closure_path),
                "artifact_sha256": closure.get("artifact_sha256"),
            },
            "usable_pairs": audit.get("counts", {}).get("usable_pairs"),
            "actual_cost_usd": audit.get("accounting", {}).get("actual_cost_usd"),
        }
    if len(plan.get("accepted_prior_pairs") or []) != 2 or any(
        item.get("accepted_without_replay") is not True
        for item in plan.get("accepted_prior_pairs") or []
    ):
        failures.append("two_verified_deepseek_carryover_pairs_missing")
    unique = sorted(set(failures))
    passed = not unique
    endpoint_cost = sum(
        Decimal(str(value["actual_cost_usd"] or "0")) for value in endpoint_inputs.values()
    )
    return {
        "schema_version": AGGREGATE_AUDIT_SCHEMA,
        "record_role": "source_reconstructed_six_pair_reasoning_effort_route_gate_v5",
        "route_plan_sha256": plan["artifact_sha256"],
        "accepted_prior_pairs": plan["accepted_prior_pairs"],
        "endpoint_inputs": endpoint_inputs,
        "decision": "passed_all_predicates" if passed else "failed_one_or_more_predicates",
        "failures": unique,
        "counts": {
            "deepseek_prior_pairs": 2,
            "gemini_fresh_pairs": endpoint_inputs.get("gemini", {}).get("usable_pairs", 0),
            "sonnet_fresh_pairs": endpoint_inputs.get("sonnet", {}).get("usable_pairs", 0),
            "usable_pairs": 6 if passed else 0,
            "usable_arms": 12 if passed else 0,
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "accounting": {
            "deepseek_v4_actual_cost_usd": "0.006154",
            "fresh_endpoint_actual_cost_usd": _decimal_text(endpoint_cost),
            "v4_orphan_reserve_retained_usd": "0.6765315",
        },
        "study_admission": {
            "authorized": passed,
            "scope": "fresh_zero_call_full_sensitivity_study_preflight_only",
            "full_study_execution_performed": False,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def build_aggregate_closure(
    *, plan: Mapping[str, Any], aggregate_audit: Mapping[str, Any]
) -> dict[str, Any]:
    if not _artifact_verifies(aggregate_audit, AGGREGATE_AUDIT_SCHEMA):
        raise RouteGateV5Error("aggregate audit does not verify")
    endpoint_ids = {
        key: value["closure"]["artifact_sha256"]
        for key, value in aggregate_audit["endpoint_inputs"].items()
    }
    return {
        "schema_version": AGGREGATE_CLOSURE_SCHEMA,
        "record_role": "permanent_reasoning_effort_v5_aggregate_closure",
        "route_plan_sha256": plan["artifact_sha256"],
        "aggregate_audit_sha256": aggregate_audit["artifact_sha256"],
        "endpoint_closure_sha256s": endpoint_ids,
        "v4_closure_sha256": V4_CLOSURE_SHA256,
        "decision": {
            "route_gate_qualified": aggregate_audit.get("decision") == "passed_all_predicates",
            "full_study_zero_call_preflight_permitted": aggregate_audit.get(
                "study_admission", {}
            ).get("authorized")
            is True,
            "all_old_and_new_route_identifiers_closed": True,
            "replay_permitted": False,
        },
        "cost": aggregate_audit.get("accounting"),
        "claim_boundary": aggregate_audit.get("claim_boundary"),
    }


def build_endpoint_execution_plan(
    *,
    plan: Mapping[str, Any],
    endpoint_id: str,
    endpoint_root: Path,
    peer_endpoint_root: Path,
    v4_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    from .real_dataset_runner import dataset_ledger_state, load_dataset_ledger

    peer = next(value for value in ENDPOINTS if value != endpoint_id)
    roots = {endpoint_id: endpoint_root, peer: peer_endpoint_root}
    budget = _v5_accounting(
        plan=plan,
        v4_receipt=v4_receipt,
        endpoint_roots=roots,
        selected_endpoint=endpoint_id,
    )
    ledger_path = endpoint_root / "ledger.jsonl"
    reservations, finalizations = dataset_ledger_state(load_dataset_ledger(ledger_path))
    sources = _source_map(endpoint_root / "source")
    decisions: list[dict[str, Any]] = []
    for item in _endpoint_items(plan, endpoint_id):
        work_item_id = item["work_item_id"]
        if work_item_id in finalizations:
            decision = "skip_finalized"
        elif work_item_id in reservations and work_item_id in sources:
            decision = "recover_source_without_provider_call"
        elif work_item_id in reservations:
            decision = "closed_reserved_without_source_no_replay"
        elif budget["admission_allowed"]:
            decision = "fresh_catalog_attestation_then_single_pair_reservation"
        else:
            decision = "blocked_before_catalog_or_provider_call"
        decisions.append(
            {
                "work_item_id": work_item_id,
                "variant_id": item["route_coordinate"]["variant_id"],
                "worst_case_reserve_usd": item["worst_case_reserve_usd"],
                "decision": decision,
            }
        )
    return {
        "schema_version": "flavourbench-reasoning-effort-endpoint-execution-plan-v5",
        "record_role": "zero_call_endpoint_isolated_execution_plan",
        "route_plan_sha256": plan["artifact_sha256"],
        "endpoint_id": endpoint_id,
        "status": "admissible_dry_run" if budget["admission_allowed"] else "blocked_dry_run",
        "budget": budget,
        "decisions": decisions,
        "execution": {
            "confirmation": CONFIRMATIONS[endpoint_id],
            "catalog_gets_made_by_plan": 0,
            "provider_completion_requests_made_by_plan": 0,
            "epicure_calls_made_by_plan": 0,
            "peer_endpoint_mutated": False,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def _api_key() -> str:
    return str(
        os.environ.get("FLAVOURBENCH_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or ""
    )


def _add_v4_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--v4-route-plan", type=Path, required=True)
    parser.add_argument("--v4-receipt", type=Path, required=True)
    parser.add_argument("--v4-audit", type=Path, required=True)
    parser.add_argument("--v4-closure", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    snapshot.add_argument("--output-directory", type=Path, required=True)

    incident = sub.add_parser("audit-v4")
    _add_v4_paths(incident)
    incident.add_argument("--v4-ledger", type=Path, required=True)
    incident.add_argument("--v4-journal", type=Path, required=True)
    incident.add_argument("--v4-source-directory", type=Path, required=True)
    incident.add_argument("--endpoint-snapshot", type=Path, required=True)
    incident.add_argument("--output-directory", type=Path, required=True)

    freeze = sub.add_parser("freeze")
    _add_v4_paths(freeze)
    freeze.add_argument("--v4-incident-audit", type=Path, required=True)
    freeze.add_argument("--endpoint-snapshot", type=Path, required=True)
    freeze.add_argument("--v4-source-directory", type=Path, required=True)
    freeze.add_argument("--output-directory", type=Path, required=True)

    plan = sub.add_parser("plan-endpoint")
    plan.add_argument("--route-plan", type=Path, required=True)
    plan.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    plan.add_argument("--endpoint-root", type=Path, required=True)
    plan.add_argument("--peer-endpoint-root", type=Path, required=True)
    plan.add_argument("--v4-receipt", type=Path, required=True)
    plan.add_argument("--output-directory", type=Path, required=True)

    execute = sub.add_parser("execute-endpoint")
    execute.add_argument("--route-plan", type=Path, required=True)
    execute.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    execute.add_argument("--endpoint-root", type=Path, required=True)
    execute.add_argument("--peer-endpoint-root", type=Path, required=True)
    execute.add_argument("--v4-receipt", type=Path, required=True)
    execute.add_argument("--global-budget-lock", type=Path, required=True)
    execute.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    execute.add_argument("--confirm", required=True)

    audit = sub.add_parser("audit-endpoint")
    audit.add_argument("--route-plan", type=Path, required=True)
    audit.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    audit.add_argument("--endpoint-root", type=Path, required=True)
    audit.add_argument("--receipt", type=Path, required=True)
    audit.add_argument("--output-directory", type=Path, required=True)

    close = sub.add_parser("close-endpoint")
    close.add_argument("--route-plan", type=Path, required=True)
    close.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    close.add_argument("--audit", type=Path, required=True)
    close.add_argument("--output-directory", type=Path, required=True)

    aggregate = sub.add_parser("aggregate-audit")
    aggregate.add_argument("--route-plan", type=Path, required=True)
    aggregate.add_argument("--gemini-audit", type=Path, required=True)
    aggregate.add_argument("--gemini-closure", type=Path, required=True)
    aggregate.add_argument("--sonnet-audit", type=Path, required=True)
    aggregate.add_argument("--sonnet-closure", type=Path, required=True)
    aggregate.add_argument("--output-directory", type=Path, required=True)

    aggregate_close = sub.add_parser("aggregate-close")
    aggregate_close.add_argument("--route-plan", type=Path, required=True)
    aggregate_close.add_argument("--aggregate-audit", type=Path, required=True)
    aggregate_close.add_argument("--output-directory", type=Path, required=True)
    return parser


def _print_output(path: Path, document: Mapping[str, Any], **extra: object) -> None:
    print(
        json.dumps(
            {
                "output": str(path.resolve()),
                "artifact_sha256": document.get("artifact_sha256"),
                "status": document.get("status"),
                "decision": document.get("decision"),
                **extra,
            },
            indent=2,
            sort_keys=True,
        )
    )


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    repo_root = arguments.repo_root.resolve()
    if arguments.command == "snapshot":
        payload = asyncio.run(build_snapshot(api_base=arguments.api_base, api_key=_api_key()))
        path = _write_artifact(
            arguments.output_directory, "reasoning-effort-v5-endpoint-snapshot", payload
        )
        _print_output(
            path,
            _regular_json(path),
            catalog_http_gets=4,
            provider_completion_requests=0,
            epicure_calls=0,
        )
        return
    if arguments.command == "audit-v4":
        payload = build_v4_incident_audit(
            v4_route_plan_path=arguments.v4_route_plan,
            v4_receipt_path=arguments.v4_receipt,
            v4_audit_path=arguments.v4_audit,
            v4_closure_path=arguments.v4_closure,
            v4_ledger_path=arguments.v4_ledger,
            v4_journal_path=arguments.v4_journal,
            v4_source_directory=arguments.v4_source_directory,
            endpoint_snapshot_path=arguments.endpoint_snapshot,
            repo_root=repo_root,
        )
        path = _write_artifact(
            arguments.output_directory, "reasoning-effort-v4-pre-request-audit", payload
        )
        _print_output(path, _regular_json(path), provider_completion_requests=0, epicure_calls=0)
        return
    if arguments.command == "freeze":
        payload = build_route_plan(
            v4_route_plan_path=arguments.v4_route_plan,
            v4_receipt_path=arguments.v4_receipt,
            v4_audit_path=arguments.v4_audit,
            v4_closure_path=arguments.v4_closure,
            v4_incident_audit_path=arguments.v4_incident_audit,
            endpoint_snapshot_path=arguments.endpoint_snapshot,
            v4_source_directory=arguments.v4_source_directory,
            repo_root=repo_root,
        )
        path = _write_artifact(
            arguments.output_directory, "reasoning-effort-v5-route-gate-plan", payload
        )
        _print_output(path, _regular_json(path), provider_completion_requests=0, epicure_calls=0)
        return

    route_plan = _regular_json(arguments.route_plan)
    validate_route_plan(route_plan, repo_root=repo_root)
    if arguments.command == "plan-endpoint":
        v4_receipt = _assert_artifact(arguments.v4_receipt, V4_RECEIPT_SHA256)
        payload = build_endpoint_execution_plan(
            plan=route_plan,
            endpoint_id=arguments.endpoint,
            endpoint_root=arguments.endpoint_root,
            peer_endpoint_root=arguments.peer_endpoint_root,
            v4_receipt=v4_receipt,
        )
        path = _write_artifact(
            arguments.output_directory,
            f"reasoning-effort-v5-{arguments.endpoint}-execution-plan",
            payload,
        )
        _print_output(
            path,
            _regular_json(path),
            provider_completion_requests=0,
            epicure_calls=0,
        )
        return
    if arguments.command == "execute-endpoint":
        if arguments.confirm != CONFIRMATIONS[arguments.endpoint]:
            raise RouteGateV5Error(
                f"execution requires --confirm {CONFIRMATIONS[arguments.endpoint]}"
            )
        v4_receipt = _assert_artifact(arguments.v4_receipt, V4_RECEIPT_SHA256)
        result = asyncio.run(
            execute_endpoint(
                plan=route_plan,
                endpoint_id=arguments.endpoint,
                endpoint_root=arguments.endpoint_root,
                peer_endpoint_root=arguments.peer_endpoint_root,
                v4_receipt=v4_receipt,
                repo_root=repo_root,
                global_budget_lock_path=arguments.global_budget_lock,
                api_base=arguments.api_base,
                api_key=_api_key(),
            )
        )
        _print_output(
            Path(result["path"]),
            result["document"],
            endpoint=arguments.endpoint,
        )
        return
    if arguments.command == "audit-endpoint":
        payload = build_endpoint_audit(
            plan=route_plan,
            endpoint_id=arguments.endpoint,
            receipt_path=arguments.receipt,
            endpoint_root=arguments.endpoint_root,
            repo_root=repo_root,
        )
        path = _write_artifact(
            arguments.output_directory,
            f"reasoning-effort-v5-{arguments.endpoint}-audit",
            payload,
        )
        _print_output(path, _regular_json(path), endpoint=arguments.endpoint)
        return
    if arguments.command == "close-endpoint":
        audit = _regular_json(arguments.audit)
        payload = build_endpoint_closure(
            plan=route_plan, endpoint_id=arguments.endpoint, audit=audit
        )
        path = _write_artifact(
            arguments.output_directory,
            f"reasoning-effort-v5-{arguments.endpoint}-closure",
            payload,
        )
        _print_output(path, _regular_json(path), endpoint=arguments.endpoint)
        return
    if arguments.command == "aggregate-audit":
        payload = build_aggregate_audit(
            plan=route_plan,
            gemini_audit_path=arguments.gemini_audit,
            gemini_closure_path=arguments.gemini_closure,
            sonnet_audit_path=arguments.sonnet_audit,
            sonnet_closure_path=arguments.sonnet_closure,
            repo_root=repo_root,
        )
        path = _write_artifact(
            arguments.output_directory, "reasoning-effort-v5-aggregate-audit", payload
        )
        _print_output(path, _regular_json(path))
        return
    if arguments.command == "aggregate-close":
        audit = _regular_json(arguments.aggregate_audit)
        payload = build_aggregate_closure(plan=route_plan, aggregate_audit=audit)
        path = _write_artifact(
            arguments.output_directory, "reasoning-effort-v5-aggregate-closure", payload
        )
        _print_output(path, _regular_json(path))


if __name__ == "__main__":
    run()
