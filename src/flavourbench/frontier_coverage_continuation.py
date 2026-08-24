"""Audit and retire the stopped coverage run without contacting a provider.

This module is intentionally network-free.  It reconstructs the immutable
failure evidence, appends a no-replay retirement for the pre-request Cohere
orphan, and freezes a separate v2 continuation namespace.  The v1 ledger is
never rewritten and the v1 executor remains fail-closed on its orphan.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .frontier_contract_runner import AdmissionDenied, IntegrityError
from .real_dataset_runner import (
    _dataset_ledger_lock,
    append_dataset_ledger_event,
    load_dataset_ledger,
)
from .real_task_bank import sha256_json
from .run_journal import load_run_journal

AUDIT_SCHEMA_VERSION = "flavourbench-frontier-coverage-stopped-run-audit-v1"
CLOSURE_SCHEMA_VERSION = "flavourbench-frontier-coverage-orphan-closure-v1"
CONTINUATION_SCHEMA_VERSION = "flavourbench-frontier-coverage-continuation-v2"
REPLACEMENT_SCHEMA_VERSION = "flavourbench-frontier-coverage-replacement-v3"
CLOSURE_CONFIRMATION = "RETIRE_COHERE_ORPHAN_WITHOUT_REPLAY"
CONTINUATION_NAMESPACE = uuid.UUID("8bfdb8f7-ecbd-4f5e-8a79-8e5a4579245f")
REPLACEMENT_NAMESPACE = uuid.UUID("f1f802af-3c8c-41f7-95de-9d6cd8da7c10")

GLM_WORK_ITEM_ID = "1c19a9e76d80737e0f53c5ba31c576f29762f3948abc9ddf632e807cde057dc3"
GLM_SOURCE_SHA256 = "0f26ba12661e13e6d73d85e39b4d09e51ee13e5ff8bbca951cc5c21a5a4d249b"
GLM_JOURNAL_SHA256 = "924bb78df590200b8d14e57acd10f26ec2a57754ae812aafc82c8633b016fca5"
GLM_ERROR = "ProviderError: provider tool-call fan-out (10) exceeded the per-round cap (6)"

MISTRAL_WORK_ITEM_ID = "e255d245e00d55146edbb757316bff9f7fb73cd0843e01c9316c0a1bcbe6c78b"
MISTRAL_SOURCE_SHA256 = "d55b8562d761ca30f60c903a3dbd754c867f3c1bea6d7284701a078d1187662b"
MISTRAL_JOURNAL_SHA256 = "8271d3a4758596cc79f94cf32789eb68a2b1a9f809e19551efb586ed177d4971"
MISTRAL_ERROR = "ProviderError: provider evidence-decision turn did not finish normally"

COHERE_WORK_ITEM_ID = "63cf4b5c57e627ae17d150c6d0a37d30b7f59bee1c1f9a301a6c48c30b700a79"
COHERE_RESERVATION_SHA256 = "73c5132a4ff2715194a4b0eeb32794f5d386439c0c98088e8fdcd8980ac13ab4"
COHERE_INCIDENT_SHA256 = "6ff19c9151515c0946d203f0eeca5bc135b7c1567247537884518be403ac8d95"
COHERE_STDOUT_SHA256 = "edadf4887f68b98688935988c31184feb16611014cc51351a00ff0f009f8bd03"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

RETIRED_WORK_ITEM_IDS = frozenset(
    {GLM_WORK_ITEM_ID, MISTRAL_WORK_ITEM_ID, COHERE_WORK_ITEM_ID}
)

# The six v1 cells below had no reservation and no provider call.  They may be
# migrated to a separate ledger, but receive alternate development tasks and
# wholly new work/arm/attempt IDs so the continuation cannot accidentally
# become a replay of the stopped schedule.
MIGRATED_TASKS = {
    "c9f74db7c1b7e913d78bf19d8b9fc8d0bd53d1357840aba5da78ca4bc7659158": (
        "fb-s0-evidence-004",
        "18d90f1562128e11a52c421e7733c03cb7ca33d2d5457bd3aaf8775bd75b69d9",
    ),
    "8c3007aa026b1404f5926dc850de66d243f968910f703dde3d77cfa89071e14c": (
        "fb-s0-evidence-004",
        "18d90f1562128e11a52c421e7733c03cb7ca33d2d5457bd3aaf8775bd75b69d9",
    ),
    "a03ee0419caddb421f32b9b328ebab22efe64dfea491ccfed475a73f414e7736": (
        "fb-s0-evidence-004",
        "18d90f1562128e11a52c421e7733c03cb7ca33d2d5457bd3aaf8775bd75b69d9",
    ),
    "36cf0baadba5e0d4c244d240ac5231a345ce1b56ed1efc364dbd8bac21b1876d": (
        "fb-s0-substitution-005",
        "d563864beb5f9efdf90dc2a4e44c1a711728a0348cc36dc0b282dc06fcddba2c",
    ),
    "d4203520379ea9a18faf8884b305696c91721743d7adbcd0546f3cc7ff4c148c": (
        "fb-s0-substitution-005",
        "d563864beb5f9efdf90dc2a4e44c1a711728a0348cc36dc0b282dc06fcddba2c",
    ),
    "2fb77d3b4ce1bb7d0b32c796aad5602683121c4043954e15ec17409da3942089": (
        "fb-s0-substitution-005",
        "d563864beb5f9efdf90dc2a4e44c1a711728a0348cc36dc0b282dc06fcddba2c",
    ),
}

# These are alternate, surface-screened development tasks not previously used
# by the corresponding model in the frozen source inventory.  They are a
# separate post-failure stratum: success cannot erase the original reliability
# failure and the records remain ineligible for an official fit.
REPLACEMENT_TASKS = {
    GLM_WORK_ITEM_ID: (
        "fb-s0-composition-010",
        "4dd4fca9e8649b9505ba4561c2b6ff0ece06ce2c5feda14aff050ae26771b4c7",
    ),
    MISTRAL_WORK_ITEM_ID: (
        "fb-s0-cookability-012",
        "a981ee02a5edba2a929ef23acbe6224544ee23a4e897d62a1933c6b464909938",
    ),
    COHERE_WORK_ITEM_ID: (
        "fb-s0-cookability-012",
        "a981ee02a5edba2a929ef23acbe6224544ee23a4e897d62a1933c6b464909938",
    ),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"{label} must be a regular non-symlink file")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise IntegrityError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} is not a JSON object")
    return value


def _verify_content_addressed(
    path: Path, *, label: str, legacy_ascii_json: bool = False
) -> dict[str, Any]:
    value = _load_json(path, label=label)
    digest = value.get("artifact_sha256")
    unhashed = dict(value)
    unhashed.pop("artifact_sha256", None)
    observed = (
        _sha256_bytes(json.dumps(unhashed, separators=(",", ":"), sort_keys=True).encode())
        if legacy_ascii_json
        else sha256_json(unhashed)
    )
    if not isinstance(digest, str) or digest != observed:
        raise IntegrityError(f"{label} content digest does not verify")
    if digest not in path.name and digest[:12] not in path.name:
        raise IntegrityError(f"{label} filename does not contain its digest")
    return value


def _atomic_content_addressed_write(
    payload: Mapping[str, Any], *, output_directory: Path, prefix: str
) -> Path:
    document = dict(payload)
    document["artifact_sha256"] = sha256_json(document)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"{prefix}-{document['artifact_sha256']}.json"
    encoded = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if destination.exists():
        if _load_json(destination, label=prefix) != document:
            raise IntegrityError(f"existing {prefix} differs at its content address")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{prefix}-", suffix=".tmp", dir=output_directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _source_path(source_directory: Path, digest: str) -> Path:
    matches = sorted(source_directory.glob(f"*-{digest[:12]}.json"))
    if len(matches) != 1:
        raise IntegrityError(f"expected exactly one source artifact for {digest}")
    return matches[0]


def _journal_audit(source: Mapping[str, Any], source_directory: Path) -> dict[str, Any]:
    descriptor = source.get("run_journal")
    if not isinstance(descriptor, Mapping):
        raise IntegrityError("source has no run-journal descriptor")
    journal_path = source_directory / str(descriptor.get("filename") or "")
    _regular_file(journal_path, label="run journal")
    physical_digest = _sha256_bytes(journal_path.read_bytes())
    entries = load_run_journal(journal_path)
    if (
        physical_digest != descriptor.get("sha256")
        or len(entries) != descriptor.get("entry_count")
        or entries[-1].get("entry_sha256") != descriptor.get("head_entry_sha256")
        or entries[0].get("run_id") != descriptor.get("run_id")
        or entries[-1].get("event_type") != "run_finalized"
    ):
        raise IntegrityError("source run-journal descriptor does not verify")
    return {
        "filename": journal_path.name,
        "sha256": physical_digest,
        "entry_count": len(entries),
        "head_entry_sha256": entries[-1]["entry_sha256"],
        "finalized": True,
    }


def _accepted_generations(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = source.get("provider_attempt_events")
    if not isinstance(events, list):
        raise IntegrityError("source has no provider-attempt events")
    accepted = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event_type") == "response_received"
        and event.get("http_status") == 200
        and isinstance(event.get("metadata"), Mapping)
        and isinstance(event["metadata"].get("response_envelope"), Mapping)
        and event["metadata"]["response_envelope"].get("accepted_chat_completion") is True
    ]
    ids = [str(event.get("generation_id") or "") for event in accepted]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise IntegrityError("accepted generation IDs are absent or repeated")
    return accepted


def _generation_metadata(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    results = source.get("results")
    if not isinstance(results, Mapping):
        raise IntegrityError("source results are malformed")
    for result in results.values():
        if not isinstance(result, Mapping):
            raise IntegrityError("source result is malformed")
        metadata = result.get("generation_metadata")
        if not isinstance(metadata, list):
            raise IntegrityError("source result has no generation metadata")
        rows.extend(item for item in metadata if isinstance(item, Mapping))
    incomplete = source.get("incomplete_generation_metadata")
    if not isinstance(incomplete, list):
        raise IntegrityError("source incomplete-generation metadata is malformed")
    rows.extend(item for item in incomplete if isinstance(item, Mapping))
    if len(rows) != sum(
        len(result.get("generation_metadata") or [])
        for result in results.values()
        if isinstance(result, Mapping)
    ) + len(incomplete):
        raise IntegrityError("generation metadata contains a non-object entry")
    return rows


def _verify_failed_source(
    *,
    source_directory: Path,
    digest: str,
    work_item_id: str,
    model_id: str,
    canonical_model: str,
    provider_tag: str,
    actual_provider: str,
    error: str,
    cost_micros: int,
    accepted_count: int,
    mcp_count: int,
    required_finish_reason: tuple[str, str] | None,
) -> dict[str, Any]:
    path = _source_path(source_directory, digest)
    source = _verify_content_addressed(
        path, label="stopped-run source", legacy_ascii_json=True
    )
    if (
        source.get("dataset_work_item_id") != work_item_id
        or source.get("requested_model_id") != model_id
        or source.get("requested_provider") != provider_tag
        or source.get("status") != "failed_or_unreconciled"
        or source.get("errors") != {"epicure_on": error}
        or sorted((source.get("results") or {}).keys()) != ["epicure_off"]
        or (source.get("budget") or {}).get("actual_cost_micros") != cost_micros
        or (source.get("budget") or {}).get("all_generation_costs_reconciled") is not True
    ):
        raise IntegrityError("stopped-run source identity, outcome, or cost differs")
    accepted = _accepted_generations(source)
    metadata = _generation_metadata(source)
    accepted_ids = {str(item["generation_id"]) for item in accepted}
    metadata_ids = {str(item.get("generation_id") or "") for item in metadata}
    if (
        len(accepted) != accepted_count
        or accepted_ids != metadata_ids
        or sum(int(item.get("cost_micros") or -1) for item in metadata) != cost_micros
        or any(
            item.get("model") != canonical_model
            or item.get("provider") != actual_provider
            or item.get("reconciled") is not True
            for item in metadata
        )
        or any(
            event["metadata"].get("response_model") != model_id
            or event["metadata"]["response_envelope"].get("provider") != actual_provider
            or event["metadata"].get("cloudflare_cache_status") != "MISS"
            for event in accepted
        )
    ):
        raise IntegrityError("stopped-run generation identity or accounting differs")
    if required_finish_reason is not None:
        phase, finish_reason = required_finish_reason
        matches = [
            event
            for event in accepted
            if event.get("phase") == phase
            and event["metadata"].get("finish_reason") == finish_reason
            and event["metadata"].get("native_finish_reason") == finish_reason
        ]
        if len(matches) != 1:
            raise IntegrityError("required provider finish-reason evidence is absent")
    traces = source.get("mcp_trace_events")
    if not isinstance(traces, list) or len(traces) != mcp_count:
        raise IntegrityError("stopped-run MCP trace count differs")
    journal = _journal_audit(source, source_directory)
    return {
        "work_item_id": work_item_id,
        "source_artifact_filename": path.name,
        "source_artifact_sha256": digest,
        "journal": journal,
        "requested_model_id": model_id,
        "canonical_model_slug": canonical_model,
        "requested_provider": provider_tag,
        "actual_provider": actual_provider,
        "accepted_http_200_generations": len(accepted),
        "unique_generation_ids": len(accepted_ids),
        "all_generation_costs_reconciled": True,
        "actual_cost_usd": f"{cost_micros / 1_000_000:.6f}",
        "retained_conditions": ["epicure_off"],
        "failed_condition": "epicure_on",
        "error": error,
        "mcp_calls_executed": mcp_count,
        "safe_to_replay_original_work_item": False,
    }


def _function_body(tree: ast.AST, name: str) -> list[ast.stmt]:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise IntegrityError(f"cannot identify exactly one {name} function")
    return functions[0].body


def _cohere_source_reconstruction(code_directory: Path) -> dict[str, Any]:
    direct_pair = code_directory / "direct_kimi_pair.py"
    cohere_wrapper = code_directory / "direct_cohere_pair.py"
    config = code_directory / "config.py"
    for path in (direct_pair, cohere_wrapper, config):
        _regular_file(path, label="reconstructed source")
    direct_text = direct_pair.read_text(encoding="utf-8")
    body = _function_body(ast.parse(direct_text), "_run_direct_pair")
    guard_lines = [
        statement.lineno
        for statement in body
        if isinstance(statement, ast.If)
        and "credential_attribute" in ast.unparse(statement.test)
        and any(
            isinstance(node, ast.Raise)
            and "credential is not configured" in ast.unparse(node)
            for node in ast.walk(statement)
        )
    ]
    manifest_lines = [
        node.lineno
        for node in ast.walk(ast.parse(direct_text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_candidate_manifest"
    ]
    provider_lines = [
        node.lineno
        for node in ast.walk(ast.parse(direct_text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "provider_factory"
    ]
    if (
        len(guard_lines) != 1
        or len(manifest_lines) != 1
        or len(provider_lines) != 1
        or not guard_lines[0] < manifest_lines[0] < provider_lines[0]
    ):
        raise IntegrityError("Cohere credential guard is not before provider construction")
    config_text = config.read_text(encoding="utf-8")
    if 'env_prefix="FLAVOURBENCH_"' not in config_text or "cohere_api_key: str" not in config_text:
        raise IntegrityError("Cohere settings prefix contract differs")
    reconstructed = (
        json.dumps(
            {
                "status": "failed",
                "error": "RuntimeError: direct Cohere credential is not configured",
            }
        )
        + "\n"
    ).encode()
    if _sha256_bytes(reconstructed) != COHERE_STDOUT_SHA256:
        raise IntegrityError("Cohere incident stdout cannot be reconstructed")
    return {
        "classification": "verified_pre_request_credential_gate",
        "required_environment_variable": "FLAVOURBENCH_COHERE_API_KEY",
        "unprefixed_variable_is_not_accepted": True,
        "credential_guard_line": guard_lines[0],
        "manifest_load_line": manifest_lines[0],
        "provider_factory_line": provider_lines[0],
        "reconstructed_stdout_sha256": COHERE_STDOUT_SHA256,
        "source_sha256s": {
            path.name: _sha256_bytes(path.read_bytes())
            for path in (direct_pair, cohere_wrapper, config)
        },
    }


def _find_entry(ledger: Sequence[Mapping[str, Any]], digest: str) -> Mapping[str, Any]:
    matches = [entry for entry in ledger if entry.get("entry_sha256") == digest]
    if len(matches) != 1:
        raise IntegrityError(f"ledger entry {digest} is absent or repeated")
    return matches[0]


def build_stopped_run_audit(
    *, source_directory: Path, response_directory: Path, ledger_path: Path, code_directory: Path
) -> dict[str, Any]:
    ledger = load_dataset_ledger(ledger_path)
    reservation = _find_entry(ledger, COHERE_RESERVATION_SHA256)
    incident = _find_entry(ledger, COHERE_INCIDENT_SHA256)
    if (
        reservation.get("event_type") != "reservation_created"
        or reservation.get("work_item_id") != COHERE_WORK_ITEM_ID
        or reservation.get("reserved_usd") != "0"
        or incident.get("event_type") != "execution_incident"
        or incident.get("work_item_id") != COHERE_WORK_ITEM_ID
        or incident.get("reservation_entry_sha256") != COHERE_RESERVATION_SHA256
        or incident.get("incident")
        != "no_verifiable_artifact_reservation_retained_no_replay"
        or incident.get("subprocess_returncode") != 1
        or incident.get("stdout_sha256") != COHERE_STDOUT_SHA256
        or incident.get("stderr_sha256") != EMPTY_SHA256
    ):
        raise IntegrityError("Cohere reservation or incident evidence differs")
    sources = [
        _load_json(path, label="coverage source")
        for path in sorted(source_directory.glob("*.json"))
    ]
    if any(source.get("dataset_work_item_id") == COHERE_WORK_ITEM_ID for source in sources):
        raise IntegrityError("Cohere orphan unexpectedly has a source artifact")
    responses = [
        _load_json(path, label="coverage response")
        for path in sorted(response_directory.glob("*.json"))
    ]
    if any(response.get("dataset_work_item_id") == COHERE_WORK_ITEM_ID for response in responses):
        raise IntegrityError("Cohere orphan unexpectedly has a normalized response")

    glm = _verify_failed_source(
        source_directory=source_directory,
        digest=GLM_SOURCE_SHA256,
        work_item_id=GLM_WORK_ITEM_ID,
        model_id="z-ai/glm-5.2",
        canonical_model="z-ai/glm-5.2-20260616",
        provider_tag="deepinfra/fp4",
        actual_provider="DeepInfra",
        error=GLM_ERROR,
        cost_micros=18877,
        accepted_count=6,
        mcp_count=4,
        required_finish_reason=None,
    )
    glm["failure_class"] = "local_tool_fanout_safety_guard"
    glm["provider_transport_failure"] = False
    glm["identity_substitution"] = False

    mistral = _verify_failed_source(
        source_directory=source_directory,
        digest=MISTRAL_SOURCE_SHA256,
        work_item_id=MISTRAL_WORK_ITEM_ID,
        model_id="mistralai/mistral-medium-3-5",
        canonical_model="mistralai/mistral-medium-3.5-20260430",
        provider_tag="mistral",
        actual_provider="Mistral",
        error=MISTRAL_ERROR,
        cost_micros=128927,
        accepted_count=5,
        mcp_count=0,
        required_finish_reason=("tool_round_0", "length"),
    )
    mistral["failure_class"] = "provider_declared_length_stop_before_tool_execution"
    mistral["provider_transport_failure"] = False
    mistral["identity_substitution"] = False

    cohere_reconstruction = _cohere_source_reconstruction(code_directory)
    cohere = {
        "work_item_id": COHERE_WORK_ITEM_ID,
        "reservation_entry_sha256": COHERE_RESERVATION_SHA256,
        "incident_entry_sha256": COHERE_INCIDENT_SHA256,
        "reserved_usd": "0",
        "source_artifacts": 0,
        "normalized_responses": 0,
        "provider_calls_verified": 0,
        "epicure_calls_verified": 0,
        "failure_class": "pre_request_credential_gate",
        "delivery_status": "verified_not_sent_under_reconstructed_source",
        "safe_to_replay_original_work_item": False,
        "work_item_disposition": "retire_without_source",
        "source_reconstruction": cohere_reconstruction,
    }
    evidence = {
        "ledger_path": str(ledger_path),
        "ledger_head_entry_sha256": ledger[-1]["entry_sha256"],
        "ledger_entry_count": len(ledger),
        "source_directory": str(source_directory),
        "response_directory": str(response_directory),
        "failures": [glm, mistral, cohere],
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "pass_stopped_run_reconstructed",
        "provider_calls_made_by_audit": 0,
        "epicure_calls_made_by_audit": 0,
        "synthetic_arms": 0,
        "evidence": evidence,
        "evidence_sha256": sha256_json(evidence),
        "original_work_item_policy": {
            "retired_work_item_ids": sorted(RETIRED_WORK_ITEM_IDS),
            "same_work_replay_permitted": False,
            "same_arm_replay_permitted": False,
            "failed_observations_remain_in_reliability_metrics": True,
        },
        "statistical_disposition": {
            GLM_WORK_ITEM_ID: {
                "epicure_off": "retain_as_real_model_arena_arm_if_other_gates_pass",
                "epicure_on": "protocol_failure_no_response",
                "uplift_pair": "exclude_incomplete_pair",
                "coverage_gap": "composition_epicure_on_protocol_failure",
            },
            MISTRAL_WORK_ITEM_ID: {
                "epicure_off": "retain_as_real_model_arena_arm_if_other_gates_pass",
                "epicure_on": "length_failure_no_response",
                "uplift_pair": "exclude_incomplete_pair",
                "coverage_gap": "cookability_epicure_on_length_failure",
            },
            COHERE_WORK_ITEM_ID: {
                "epicure_off": "no_arm_created",
                "epicure_on": "no_arm_created",
                "uplift_pair": "absent",
                "coverage_gap": "cookability_pre_request_credential_failure",
            },
        },
    }


def write_audit(
    *,
    source_directory: Path,
    response_directory: Path,
    ledger_path: Path,
    code_directory: Path,
    output_directory: Path,
) -> Path:
    return _atomic_content_addressed_write(
        build_stopped_run_audit(
            source_directory=source_directory,
            response_directory=response_directory,
            ledger_path=ledger_path,
            code_directory=code_directory,
        ),
        output_directory=output_directory,
        prefix="frontier-coverage-stopped-run-audit",
    )


def append_orphan_closure(
    *,
    audit_path: Path,
    ledger_path: Path,
    source_directory: Path,
    response_directory: Path,
    code_directory: Path,
    output_directory: Path,
    confirmation: str,
) -> tuple[Path, Mapping[str, Any]]:
    if confirmation != CLOSURE_CONFIRMATION:
        raise AdmissionDenied(f"orphan retirement requires --confirm {CLOSURE_CONFIRMATION}")
    audit = _verify_content_addressed(audit_path, label="stopped-run audit")
    with _dataset_ledger_lock(ledger_path):
        ledger = load_dataset_ledger(ledger_path)
        existing = [
            entry
            for entry in ledger
            if entry.get("event_type") == "execution_incident"
            and entry.get("incident")
            == "verified_pre_request_credential_failure_no_delivery_work_retired"
            and entry.get("work_item_id") == COHERE_WORK_ITEM_ID
        ]
        if existing:
            if len(existing) != 1:
                raise IntegrityError("Cohere orphan has duplicate retirement events")
            closure_path = output_directory / str(existing[0].get("closure_filename") or "")
            closure = _verify_content_addressed(closure_path, label="orphan closure")
            if closure.get("ledger_event_sha256") not in {None, existing[0]["entry_sha256"]}:
                raise IntegrityError("existing orphan closure event differs")
            return closure_path, existing[0]
        rebuilt = build_stopped_run_audit(
            source_directory=source_directory,
            response_directory=response_directory,
            ledger_path=ledger_path,
            code_directory=code_directory,
        )
        if audit.get("evidence_sha256") != rebuilt.get("evidence_sha256"):
            raise IntegrityError("stopped-run evidence changed after the audit")
        if ledger[-1].get("entry_sha256") != COHERE_INCIDENT_SHA256:
            raise IntegrityError("ledger head moved after the audited Cohere incident")
        closure_payload = {
            "schema_version": CLOSURE_SCHEMA_VERSION,
            "status": "verified_closed_without_source",
            "audit_artifact_sha256": audit["artifact_sha256"],
            "audit_filename": audit_path.name,
            "work_item_id": COHERE_WORK_ITEM_ID,
            "reservation_entry_sha256": COHERE_RESERVATION_SHA256,
            "incident_entry_sha256": COHERE_INCIDENT_SHA256,
            "ledger_head_before_closure_sha256": ledger[-1]["entry_sha256"],
            "failure_class": "pre_request_credential_gate",
            "delivery_status": "verified_not_sent_under_reconstructed_source",
            "reserved_usd": "0",
            "provider_calls": 0,
            "epicure_calls": 0,
            "source_artifacts": 0,
            "safe_to_replay": False,
            "work_item_retired": True,
            "v1_ledger_remains_blocked_by_design": True,
            "continuation_requires_separate_ledger_and_fresh_identifiers": True,
        }
        closure_path = _atomic_content_addressed_write(
            closure_payload,
            output_directory=output_directory,
            prefix="frontier-coverage-orphan-closure",
        )
        closure = _verify_content_addressed(closure_path, label="orphan closure")
        event = append_dataset_ledger_event(
            ledger_path,
            {
                "event_type": "execution_incident",
                "runner_run_id": "frontier-coverage-continuation-v2-preflight",
                "work_item_id": COHERE_WORK_ITEM_ID,
                "reservation_entry_sha256": COHERE_RESERVATION_SHA256,
                "superseded_incident_entry_sha256": COHERE_INCIDENT_SHA256,
                "incident": (
                    "verified_pre_request_credential_failure_no_delivery_work_retired"
                ),
                "closure_filename": closure_path.name,
                "closure_artifact_sha256": closure["artifact_sha256"],
                "reserved_usd": "0",
                "provider_calls_verified": 0,
                "epicure_calls_verified": 0,
                "safe_to_replay": False,
                "work_item_retired": True,
            },
        )
    return closure_path, event


def require_prefixed_credential_before_reservation(
    execution_backend: str, environment: Mapping[str, str]
) -> None:
    """Fail before ledger reservation; unprefixed aliases are deliberately ignored."""

    if execution_backend == "cohere_direct" and not environment.get(
        "FLAVOURBENCH_COHERE_API_KEY"
    ):
        raise AdmissionDenied(
            "cohere_direct requires FLAVOURBENCH_COHERE_API_KEY before reservation"
        )


def append_guarded_continuation_reservation(
    *,
    ledger_path: Path,
    runner_run_id: str,
    cell: Mapping[str, Any],
    reserved_usd: str,
    environment: Mapping[str, str],
    additional_fields: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """The only v2/v3 reservation boundary; credential checks run first."""

    backend = str(cell.get("execution_backend") or "")
    work_item_id = str(cell.get("work_item_id") or "")
    if cell.get("schema_version") not in {
        CONTINUATION_SCHEMA_VERSION,
        REPLACEMENT_SCHEMA_VERSION,
    }:
        raise IntegrityError("guarded reservation received a non-continuation cell")
    if len(work_item_id) != 64 or work_item_id in RETIRED_WORK_ITEM_IDS:
        raise IntegrityError("guarded reservation received an invalid or retired work item")
    require_prefixed_credential_before_reservation(backend, environment)
    protected = {
        "event_type",
        "runner_run_id",
        "work_item_id",
        "reserved_usd",
        "execution_backend",
        "credential_preflight",
        "safe_to_replay_retired_work",
    }
    extra = dict(additional_fields or {})
    if protected.intersection(extra):
        raise IntegrityError("guarded reservation additional fields override protected fields")
    return append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "reservation_created",
            "runner_run_id": runner_run_id,
            "work_item_id": work_item_id,
            "reserved_usd": reserved_usd,
            "execution_backend": backend,
            "credential_preflight": (
                "prefixed_cohere_present_before_reservation"
                if backend == "cohere_direct"
                else "not_applicable"
            ),
            "safe_to_replay_retired_work": False,
            **extra,
        },
    )


def verify_orphan_closure(
    *, closure_path: Path, audit_path: Path, ledger_path: Path
) -> dict[str, Any]:
    closure = _verify_content_addressed(closure_path, label="orphan closure")
    audit = _verify_content_addressed(audit_path, label="stopped-run audit")
    if (
        closure.get("schema_version") != CLOSURE_SCHEMA_VERSION
        or closure.get("audit_artifact_sha256") != audit.get("artifact_sha256")
        or closure.get("work_item_id") != COHERE_WORK_ITEM_ID
        or closure.get("reservation_entry_sha256") != COHERE_RESERVATION_SHA256
        or closure.get("incident_entry_sha256") != COHERE_INCIDENT_SHA256
        or closure.get("provider_calls") != 0
        or closure.get("epicure_calls") != 0
        or closure.get("safe_to_replay") is not False
        or closure.get("work_item_retired") is not True
    ):
        raise IntegrityError("orphan closure policy or source links differ")
    ledger = load_dataset_ledger(ledger_path)
    linked = [
        entry
        for entry in ledger
        if entry.get("closure_artifact_sha256") == closure["artifact_sha256"]
        and entry.get("work_item_id") == COHERE_WORK_ITEM_ID
        and entry.get("reservation_entry_sha256") == COHERE_RESERVATION_SHA256
        and entry.get("superseded_incident_entry_sha256") == COHERE_INCIDENT_SHA256
        and entry.get("incident")
        == "verified_pre_request_credential_failure_no_delivery_work_retired"
        and entry.get("safe_to_replay") is False
    ]
    if len(linked) != 1:
        raise IntegrityError("orphan closure is not linked exactly once from the ledger")
    return closure


def _attempt_slots(
    run_id: str,
    cell_id: str,
    conditions: Sequence[str],
    *,
    namespace: uuid.UUID = CONTINUATION_NAMESPACE,
) -> list[dict[str, Any]]:
    coordinates: list[tuple[str, str, int]] = []
    for condition in conditions:
        arm_id = f"{run_id}:{condition}"
        phases = ["planning", "evidence_decision", "final"]
        if condition == "epicure_on":
            phases = ["planning", *[f"tool_round_{index}" for index in range(8)], "final"]
        for phase in phases:
            for attempt_index in range(2):
                coordinates.append((arm_id, phase, attempt_index))
        if condition == "epicure_on":
            coordinates.append((arm_id, "mcp_session", 0))
            for round_index in range(8):
                for call_index in range(6):
                    coordinates.append((arm_id, f"mcp_tool_{round_index}_{call_index}", 0))
    return [
        {
            "arm_id": arm_id,
            "phase": phase,
            "attempt_index": attempt_index,
            "attempt_id": str(
                uuid.uuid5(
                    namespace,
                    f"{run_id}:{cell_id}:{arm_id}:{phase}:{attempt_index}",
                )
            ),
        }
        for arm_id, phase, attempt_index in coordinates
    ]


def build_continuation_plan(
    *,
    materialization_path: Path,
    task_validity_path: Path,
    audit_path: Path,
    closure_path: Path,
    ledger_path: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    materialization = _verify_content_addressed(materialization_path, label="v1 materialization")
    tasks = _verify_content_addressed(task_validity_path, label="task-validity artifact")
    audit = _verify_content_addressed(audit_path, label="stopped-run audit")
    closure = verify_orphan_closure(
        closure_path=closure_path, audit_path=audit_path, ledger_path=ledger_path
    )
    ledger = load_dataset_ledger(ledger_path)
    retirement_events = [
        entry
        for entry in ledger
        if entry.get("closure_artifact_sha256") == closure["artifact_sha256"]
        and entry.get("incident")
        == "verified_pre_request_credential_failure_no_delivery_work_retired"
    ]
    if len(retirement_events) != 1:
        raise IntegrityError("orphan closure is not linked once from the append-only ledger")
    task_index = {
        str(task.get("task_id")): task
        for task in tasks.get("tasks") or []
        if isinstance(task, Mapping)
    }
    original_cells = {
        str(item.get("work_item", {}).get("work_item_id") or ""): item
        for item in materialization.get("work_items") or []
        if isinstance(item, Mapping) and isinstance(item.get("work_item"), Mapping)
    }
    cells: list[dict[str, Any]] = []
    all_old_arm_ids = {
        str(arm)
        for item in materialization.get("work_items") or []
        if isinstance(item, Mapping)
        for arm in item.get("arm_ids") or []
    }
    for ordinal, (old_work_id, (task_id, prompt_sha)) in enumerate(
        sorted(MIGRATED_TASKS.items()), start=1
    ):
        original = original_cells.get(old_work_id)
        task = task_index.get(task_id)
        if original is None or task is None:
            raise IntegrityError("continuation source cell or alternate task is absent")
        work = original["work_item"]
        if (
            task.get("prompt_sha256") != prompt_sha
            or task.get("family") != work.get("task_family")
            or (task.get("surface_dependency_screen") or {}).get("status") != "pass"
        ):
            raise IntegrityError("alternate continuation task is ineligible or drifted")
        conditions = tuple(original.get("required_new_conditions") or ())
        cell_basis = {
            "schema_version": CONTINUATION_SCHEMA_VERSION,
            "source_work_item_id": old_work_id,
            "model_id": work.get("model_id"),
            "provider_tag": work.get("provider_tag"),
            "execution_backend": work.get("execution_backend"),
            "task_id": task_id,
            "prompt_sha256": prompt_sha,
            "conditions": list(conditions),
            "continuation_ordinal": ordinal,
        }
        cell_id = sha256_json({**cell_basis, "kind": "continuation_cell"})
        work_item_id = sha256_json({**cell_basis, "kind": "continuation_work_item"})
        run_id = str(uuid.uuid5(CONTINUATION_NAMESPACE, f"run:{cell_id}"))
        arm_ids = {
            condition: sha256_json(
                {
                    "schema_version": CONTINUATION_SCHEMA_VERSION,
                    "work_item_id": work_item_id,
                    "condition": condition,
                }
            )
            for condition in conditions
        }
        if work_item_id in original_cells or set(arm_ids.values()) & all_old_arm_ids:
            raise IntegrityError("continuation identifier collides with v1")
        cells.append(
            {
                **cell_basis,
                "cell_id": cell_id,
                "work_item_id": work_item_id,
                "run_id": run_id,
                "arm_ids": arm_ids,
                "attempt_slots": _attempt_slots(run_id, cell_id, conditions),
                "execution_gate": "eligible_non_cohere_fresh_namespace",
                "source_v1_cell_had_reservation": False,
                "same_task_as_v1": False,
                "same_work_or_arm_id_as_v1": False,
            }
        )
    require_prefixed_credential_before_reservation_status = "pass"
    try:
        require_prefixed_credential_before_reservation("cohere_direct", environment)
    except AdmissionDenied:
        require_prefixed_credential_before_reservation_status = (
            "blocked_missing_prefixed_credential"
        )
    return {
        "schema_version": CONTINUATION_SCHEMA_VERSION,
        "status": "frozen_no_provider_calls",
        "provider_calls_made_by_plan": 0,
        "epicure_calls_made_by_plan": 0,
        "synthetic_arms": 0,
        "sources": {
            "v1_materialization_sha256": materialization["artifact_sha256"],
            "task_validity_sha256": tasks["artifact_sha256"],
            "stopped_run_audit_sha256": audit["artifact_sha256"],
            "orphan_closure_sha256": closure["artifact_sha256"],
            "v1_ledger_head_sha256": ledger[-1]["entry_sha256"],
        },
        "v1_disposition": {
            "ledger_mutated_or_deleted": False,
            "v1_executor_remains_fail_closed": True,
            "retired_work_item_ids": sorted(RETIRED_WORK_ITEM_IDS),
            "retired_work_items_replayed": 0,
        },
        "fresh_non_cohere_cells": cells,
        "counts": {
            "migrated_untouched_non_cohere_cells": len(cells),
            "new_work_item_ids": len({cell["work_item_id"] for cell in cells}),
            "new_arm_ids": sum(len(cell["arm_ids"]) for cell in cells),
            "new_attempt_ids": sum(len(cell["attempt_slots"]) for cell in cells),
            "cohere_cells_admitted": 0,
        },
        "credential_gate": {
            "cohere_direct": require_prefixed_credential_before_reservation_status,
            "required_variable": "FLAVOURBENCH_COHERE_API_KEY",
            "checked_before_reservation": True,
            "unprefixed_alias_accepted": False,
        },
        "replacement_and_gap_policy": {
            "glm_composition": {
                "failed_cell": "retained_as_protocol_failure",
                "replacement": "reuse_independently_verified_existing_pair_on_composition_024",
                "replacement_source_artifact_sha256": (
                    "39468babadcfb27553f037e3d046eb495a97c86252584b67115ad8828f5ea6a8"
                ),
                "new_provider_call_required": False,
            },
            "mistral_cookability": {
                "failed_cell": "retained_as_length_failure",
                "replacement": "not_adaptively_rerun_in_this_continuation",
                "reported_gap": "cookability_uplift_unavailable_or_provisional",
            },
            "cohere_cookability": {
                "failed_cell": "retired_pre_request_no_arm",
                "replacement": "separate_future_fresh_task_only_after_prefixed_credential_gate",
                "reported_gap": "cookability_anchor_absent",
            },
        },
        "statistical_use": {
            "run_class": "development_unranked",
            "official_fit_eligible": False,
            "failed_cells_retained_in_reliability_denominator": True,
            "incomplete_uplift_pairs_excluded_from_preference_fit": True,
            "partial_off_arms_not_treated_as_matched_uplift_pairs": True,
        },
        "execution_recommendation": {
            "safe_now": "materialize_and_review_fresh_non_cohere_cells_only",
            "do_not_run": [
                "any v1 work item",
                "any Cohere cell without FLAVOURBENCH_COHERE_API_KEY",
                "Mistral or GLM same-task retries intended to erase failures",
            ],
            "provider_execution_implemented_by_this_module": False,
        },
    }


def write_continuation_plan(**kwargs: Any) -> Path:
    output_directory = Path(kwargs.pop("output_directory"))
    return _atomic_content_addressed_write(
        build_continuation_plan(**kwargs),
        output_directory=output_directory,
        prefix="frontier-coverage-continuation-plan",
    )


def _source_model_task_inventory(current_quality_root: Path) -> dict[str, Any]:
    if current_quality_root.is_symlink() or not current_quality_root.is_dir():
        raise IntegrityError("current-quality root must be a regular directory")
    selected_models = {
        "z-ai/glm-5.2",
        "mistralai/mistral-medium-3-5",
        "cohere/command-a-reasoning-08-2025",
    }
    records: set[tuple[str, str, str]] = set()
    scanned: list[dict[str, str]] = []
    for path in sorted(current_quality_root.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        model_id = document.get("requested_model_id")
        task_id = document.get("dataset_task_id")
        if model_id not in selected_models or not isinstance(task_id, str):
            continue
        artifact = str(document.get("artifact_sha256") or _sha256_bytes(path.read_bytes()))
        records.add((str(model_id), task_id, artifact))
        scanned.append(
            {
                "path": str(path.relative_to(current_quality_root)),
                "physical_sha256": _sha256_bytes(path.read_bytes()),
            }
        )
    inventory = {
        "records": [
            {"model_id": model, "task_id": task, "artifact_sha256": artifact}
            for model, task, artifact in sorted(records)
        ],
        "source_file_bindings": scanned,
    }
    return {
        "record_count": len(records),
        "source_file_count": len(scanned),
        "inventory_sha256": sha256_json(inventory),
        "records": inventory["records"],
    }


def build_replacement_plan(
    *,
    materialization_path: Path,
    task_validity_path: Path,
    audit_path: Path,
    closure_path: Path,
    continuation_path: Path,
    ledger_path: Path,
    current_quality_root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    materialization = _verify_content_addressed(materialization_path, label="v1 materialization")
    tasks = _verify_content_addressed(task_validity_path, label="task-validity artifact")
    audit = _verify_content_addressed(audit_path, label="stopped-run audit")
    closure = verify_orphan_closure(
        closure_path=closure_path, audit_path=audit_path, ledger_path=ledger_path
    )
    continuation = _verify_content_addressed(continuation_path, label="v2 continuation")
    ledger = load_dataset_ledger(ledger_path)
    if not any(
        entry.get("closure_artifact_sha256") == closure["artifact_sha256"]
        and entry.get("incident")
        == "verified_pre_request_credential_failure_no_delivery_work_retired"
        for entry in ledger
    ):
        raise IntegrityError("replacement plan has no verified append-only orphan retirement")
    task_index = {
        str(task.get("task_id")): task
        for task in tasks.get("tasks") or []
        if isinstance(task, Mapping)
    }
    original_cells = {
        str(item.get("work_item", {}).get("work_item_id") or ""): item
        for item in materialization.get("work_items") or []
        if isinstance(item, Mapping) and isinstance(item.get("work_item"), Mapping)
    }
    inventory = _source_model_task_inventory(current_quality_root)
    observed_pairs = {
        (str(item["model_id"]), str(item["task_id"]))
        for item in inventory["records"]
    }
    prior_identifiers = {
        str(item.get("work_item", {}).get("work_item_id") or "")
        for item in materialization.get("work_items") or []
        if isinstance(item, Mapping)
    }
    prior_identifiers.update(
        str(arm)
        for item in materialization.get("work_items") or []
        if isinstance(item, Mapping)
        for arm in item.get("arm_ids") or []
    )
    for cell in continuation.get("fresh_non_cohere_cells") or []:
        if not isinstance(cell, Mapping):
            continue
        prior_identifiers.add(str(cell.get("work_item_id") or ""))
        prior_identifiers.update(str(value) for value in (cell.get("arm_ids") or {}).values())
        prior_identifiers.update(
            str(slot.get("attempt_id") or "")
            for slot in cell.get("attempt_slots") or []
            if isinstance(slot, Mapping)
        )
    cells: list[dict[str, Any]] = []
    for ordinal, (failed_work_id, (task_id, prompt_sha)) in enumerate(
        REPLACEMENT_TASKS.items(), start=1
    ):
        original = original_cells.get(failed_work_id)
        task = task_index.get(task_id)
        if original is None or task is None:
            raise IntegrityError("replacement source cell or alternate task is absent")
        work = original["work_item"]
        model_id = str(work.get("model_id") or "")
        if (
            task.get("prompt_sha256") != prompt_sha
            or task.get("family") != work.get("task_family")
            or (task.get("surface_dependency_screen") or {}).get("status") != "pass"
            or (model_id, task_id) in observed_pairs
        ):
            raise IntegrityError("replacement task is ineligible, drifted, or previously exposed")
        basis = {
            "schema_version": REPLACEMENT_SCHEMA_VERSION,
            "failed_work_item_id": failed_work_id,
            "model_id": model_id,
            "provider_tag": work.get("provider_tag"),
            "execution_backend": work.get("execution_backend"),
            "task_id": task_id,
            "prompt_sha256": prompt_sha,
            "conditions": ["epicure_off", "epicure_on"],
            "replacement_ordinal": ordinal,
            "analysis_stratum": "post_failure_replacement_development_only",
        }
        cell_id = sha256_json({**basis, "kind": "replacement_cell"})
        work_item_id = sha256_json({**basis, "kind": "replacement_work_item"})
        run_id = str(uuid.uuid5(REPLACEMENT_NAMESPACE, f"run:{cell_id}"))
        arms = {
            condition: sha256_json(
                {
                    "schema_version": REPLACEMENT_SCHEMA_VERSION,
                    "work_item_id": work_item_id,
                    "condition": condition,
                }
            )
            for condition in ("epicure_off", "epicure_on")
        }
        attempts = _attempt_slots(
            run_id,
            cell_id,
            ("epicure_off", "epicure_on"),
            namespace=REPLACEMENT_NAMESPACE,
        )
        new_identifiers = {
            work_item_id,
            *arms.values(),
            *(str(slot["attempt_id"]) for slot in attempts),
        }
        if prior_identifiers.intersection(new_identifiers):
            raise IntegrityError("replacement identifier collides with v1 or v2")
        prior_identifiers.update(new_identifiers)
        backend = str(work.get("execution_backend") or "")
        credential_gate = "not_applicable"
        if backend == "cohere_direct":
            try:
                require_prefixed_credential_before_reservation(backend, environment)
                credential_gate = "pass_prefixed_credential_present"
            except AdmissionDenied:
                credential_gate = "blocked_missing_prefixed_credential"
        cells.append(
            {
                **basis,
                "cell_id": cell_id,
                "work_item_id": work_item_id,
                "run_id": run_id,
                "arm_ids": arms,
                "attempt_slots": attempts,
                "alternate_task_not_previously_exposed_to_model": True,
                "fresh_identifiers_disjoint_from_v1_v2": True,
                "credential_gate": credential_gate,
                "execution_gate": "frozen_not_automatically_executable",
            }
        )
    cohere_gate = next(
        cell["credential_gate"]
        for cell in cells
        if cell["execution_backend"] == "cohere_direct"
    )
    return {
        "schema_version": REPLACEMENT_SCHEMA_VERSION,
        "status": (
            "blocked_missing_prefixed_cohere_credential"
            if cohere_gate == "blocked_missing_prefixed_credential"
            else "frozen_pending_independent_review"
        ),
        "provider_calls_made_by_plan": 0,
        "epicure_calls_made_by_plan": 0,
        "synthetic_arms": 0,
        "sources": {
            "v1_materialization_sha256": materialization["artifact_sha256"],
            "task_validity_sha256": tasks["artifact_sha256"],
            "stopped_run_audit_sha256": audit["artifact_sha256"],
            "orphan_closure_sha256": closure["artifact_sha256"],
            "v2_continuation_sha256": continuation["artifact_sha256"],
            "v1_ledger_head_sha256": ledger[-1]["entry_sha256"],
            "prior_model_task_inventory_sha256": inventory["inventory_sha256"],
            "prior_model_task_record_count": inventory["record_count"],
        },
        "replacement_cells": cells,
        "counts": {
            "replacement_cells": len(cells),
            "new_real_arms_planned": sum(len(cell["arm_ids"]) for cell in cells),
            "new_work_item_ids": len({cell["work_item_id"] for cell in cells}),
            "new_attempt_ids": sum(len(cell["attempt_slots"]) for cell in cells),
            "provider_calls_executed": 0,
        },
        "methodological_boundary": {
            "original_failures_remain_in_reliability_denominator": True,
            "replacement_success_cannot_supersede_original_failure": True,
            "replacement_pairs_are_not_missing_at_random": True,
            "official_preference_or_uplift_fit_eligible": False,
            "permitted_analysis": "post_failure_task_sensitivity_and_graph_diagnostics",
            "same_task_retry_permitted": False,
            "same_work_or_arm_replay_permitted": False,
        },
        "credential_gate": {
            "required_before_any_joint_replacement_execution": True,
            "cohere_direct": cohere_gate,
            "required_variable": "FLAVOURBENCH_COHERE_API_KEY",
            "unprefixed_alias_accepted": False,
            "reservation_must_follow_gate": True,
        },
        "execution_recommendation": {
            "automatic_execution": "prohibited",
            "next_action": (
                "configure prefixed Cohere credential, rerun this no-call freeze, then obtain "
                "independent plan review before implementing a separate v3 executor"
            ),
        },
    }


def write_replacement_plan(**kwargs: Any) -> Path:
    output_directory = Path(kwargs.pop("output_directory"))
    return _atomic_content_addressed_write(
        build_replacement_plan(**kwargs),
        output_directory=output_directory,
        prefix="frontier-coverage-replacement-plan",
    )


def _default_paths(project_root: Path) -> dict[str, Path]:
    current = project_root / "artifacts/season1/current-quality-run"
    execution = current / "frontier-coverage-repair-execution-v1"
    return {
        "source_directory": execution / "source",
        "response_directory": execution / "responses",
        "ledger_path": execution / "ledger.jsonl",
        "code_directory": project_root / "src/flavourbench",
        "current_quality_root": current,
        "output_directory": current / "frontier-coverage-continuation-v2",
        "materialization_path": execution
        / (
            "frontier-coverage-materialization-"
            "eb27d59a5ec474f3b7975ea4649217182054f92d4b40bd6efbf3f1e4567b029f.json"
        ),
        "task_validity_path": project_root
        / "artifacts/season1/task-validity/development-v2"
        / (
            "development-task-validity-v2-"
            "86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json"
        ),
    }


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("audit", "close-orphan", "freeze", "freeze-replacements")
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--closure", type=Path)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--confirm", default="")
    arguments = parser.parse_args()
    paths = _default_paths(arguments.project_root.resolve())
    if arguments.command == "audit":
        path = write_audit(
            source_directory=paths["source_directory"],
            response_directory=paths["response_directory"],
            ledger_path=paths["ledger_path"],
            code_directory=paths["code_directory"],
            output_directory=paths["output_directory"],
        )
        print(json.dumps({"status": "audited_no_calls", "artifact": str(path)}))
        return
    if arguments.audit is None:
        raise SystemExit("--audit is required")
    if arguments.command == "close-orphan":
        path, event = append_orphan_closure(
            audit_path=arguments.audit,
            ledger_path=paths["ledger_path"],
            source_directory=paths["source_directory"],
            response_directory=paths["response_directory"],
            code_directory=paths["code_directory"],
            output_directory=paths["output_directory"],
            confirmation=arguments.confirm,
        )
        print(
            json.dumps(
                {
                    "status": "retired_no_calls_no_replay",
                    "artifact": str(path),
                    "ledger_event_sha256": event["entry_sha256"],
                }
            )
        )
        return
    if arguments.closure is None:
        raise SystemExit("--closure is required")
    if arguments.command == "freeze-replacements":
        if arguments.continuation is None:
            raise SystemExit("--continuation is required")
        path = write_replacement_plan(
            materialization_path=paths["materialization_path"],
            task_validity_path=paths["task_validity_path"],
            audit_path=arguments.audit,
            closure_path=arguments.closure,
            continuation_path=arguments.continuation,
            ledger_path=paths["ledger_path"],
            current_quality_root=paths["current_quality_root"],
            environment=os.environ,
            output_directory=paths["output_directory"],
        )
        print(json.dumps({"status": "replacement_plan_frozen_no_calls", "artifact": str(path)}))
        return
    path = write_continuation_plan(
        materialization_path=paths["materialization_path"],
        task_validity_path=paths["task_validity_path"],
        audit_path=arguments.audit,
        closure_path=arguments.closure,
        ledger_path=paths["ledger_path"],
        environment=os.environ,
        output_directory=paths["output_directory"],
    )
    print(json.dumps({"status": "frozen_no_calls", "artifact": str(path)}))


if __name__ == "__main__":
    run()
