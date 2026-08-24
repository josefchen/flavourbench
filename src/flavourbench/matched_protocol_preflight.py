"""Freeze and verify exact matched-Epicure protocol preflights.

The preflight is deliberately separate from benchmark collection.  A base
candidate manifest freezes the routes and execution policy.  A content-
addressed plan then assigns one real natural on/off pair and one required-tool
diagnostic to every route.  Only a complete registry of verified live receipts
can promote the base manifest to an execution-ready development manifest.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

from .execution_policy import (
    DIRECT_TOOL_CONTRACT_PROTOCOL,
    GOVERNED_EPICURE_PROTOCOLS,
    MATCHED_EVIDENCE_PROTOCOLS,
    MATCHED_TOOL_ACCESS_PROTOCOL_V1,
    verify_policy_document,
)
from .frontier_contract_runner import (
    IntegrityError,
    load_candidate_manifest,
    select_candidates,
)
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import _worst_case_cost_usd, endpoint_execution_contract_sha256
from .run_journal import JournalIntegrityError, verify_journal_descriptor
from .tool_contract import required_tool_contract

PLAN_SCHEMA_VERSION = "flavourbench-matched-protocol-preflight-plan-v4"
REGISTRY_SCHEMA_VERSION = "flavourbench-matched-protocol-preflight-registry-v4"
PROMOTION_SCHEMA_VERSION = "flavourbench-matched-protocol-promotion-v4"
LIVE_SMOKE_SCHEMA_VERSION = "flavourbench-live-smoke-v1"
LIVE_PROTOCOL_SCHEMA_VERSION = "flavourbench-live-development-protocol-v9"
LEGACY_LIVE_PROTOCOL_SCHEMA_VERSION = "flavourbench-live-development-protocol-v8"
PREFLIGHT_TASK_ID = "matched-protocol-preflight-v4"
PREFLIGHT_PROMPT = (
    "Design a pear, white miso, and toasted buckwheat dish. Use any available culinary "
    "evidence to identify a coherent bridge ingredient, explain which evidence is suggestive "
    "rather than causal, and give a practical preparation."
)
PREFLIGHT_CATEGORY = "composition"
EXECUTION_CONFIRMATION = "RUN_MATCHED_PROTOCOL_PREFLIGHT"
LEDGER_SCHEMA_VERSION = "flavourbench-matched-protocol-preflight-ledger-v1"
PREFLIGHT_SOURCE_FILES = (
    "config.py",
    "current_development_manifest.py",
    "execution_policy.py",
    "live_smoke.py",
    "matched_protocol_preflight.py",
    "protocol_contract.py",
    "provider.py",
    "real_dataset_runner.py",
    "tool_contract.py",
)


class ProtocolPreflightError(RuntimeError):
    """A preflight plan, receipt, registry, or promotion failed closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _live_smoke_sha256(value: object) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolPreflightError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolPreflightError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise ProtocolPreflightError(f"expected a JSON object: {path}")
    return value


def _verified_artifact(path: Path, *, schema_version: str) -> dict[str, Any]:
    value = _regular_json(path)
    recorded = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if (
        value.get("schema_version") != schema_version
        or not isinstance(recorded, str)
        or len(recorded) != 64
        or _sha256(payload) != recorded
    ):
        raise ProtocolPreflightError(f"content address does not verify: {path}")
    return value


def _verified_live_smoke(path: Path) -> dict[str, Any]:
    value = _regular_json(path)
    recorded = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if (
        value.get("schema_version") != LIVE_SMOKE_SCHEMA_VERSION
        or not isinstance(recorded, str)
        or len(recorded) != 64
        or _live_smoke_sha256(payload) != recorded
    ):
        raise ProtocolPreflightError(f"live-smoke content address does not verify: {path}")
    try:
        verify_journal_descriptor(path.parent, value.get("run_journal") or {})
    except JournalIntegrityError as error:
        raise ProtocolPreflightError(f"live-smoke journal does not verify: {path}") from error
    return value


def _write_artifact(output_dir: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    unhashed = dict(payload)
    unhashed.pop("artifact_sha256", None)
    digest = _sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ProtocolPreflightError("content-addressed output conflict")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def build_plan(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    prompt: str = PREFLIGHT_PROMPT,
    category: str = PREFLIGHT_CATEGORY,
) -> dict[str, Any]:
    """Build a no-call preflight slate from a frozen candidate manifest."""

    try:
        manifest = load_candidate_manifest(
            manifest_path,
            expected_digest=expected_manifest_sha256,
        )
        candidates = select_candidates(manifest)
    except IntegrityError as error:
        raise ProtocolPreflightError(str(error)) from error
    if manifest.get("manifest_role") != "current_frontier_real_development_quality_run":
        raise ProtocolPreflightError("preflight accepts only the current development manifest")
    design = manifest.get("run_design")
    if not isinstance(design, Mapping):
        raise ProtocolPreflightError("candidate manifest has no run design")
    policy = design.get("execution_policy")
    protocol = design.get("generation_protocol")
    protocol_name = (
        str(protocol.get("evidence_protocol") or "") if isinstance(protocol, Mapping) else ""
    )
    protocol_schema = (
        str(protocol.get("schema_version") or "") if isinstance(protocol, Mapping) else ""
    )
    schema_is_accepted = (
        protocol_schema == LIVE_PROTOCOL_SCHEMA_VERSION
        if protocol_name == MATCHED_TOOL_ACCESS_PROTOCOL_V1
        else protocol_schema in {LEGACY_LIVE_PROTOCOL_SCHEMA_VERSION, LIVE_PROTOCOL_SCHEMA_VERSION}
    )
    if (
        not verify_policy_document(policy)
        or not isinstance(protocol, Mapping)
        or not schema_is_accepted
        or protocol.get("evidence_protocol") not in GOVERNED_EPICURE_PROTOCOLS
        or protocol.get("final_response_mode") != "plain_text"
        or protocol.get("matched_planning")
        is not (protocol.get("evidence_protocol") in MATCHED_EVIDENCE_PROTOCOLS)
        or protocol.get("required_tool_contract_protocol") != DIRECT_TOOL_CONTRACT_PROTOCOL
        or protocol.get("intermediate_reasoning_effort")
        not in {"minimal", "low", "medium", "high", "xhigh", "max"}
        or protocol.get("final_reasoning_effort")
        not in {"minimal", "low", "medium", "high", "xhigh", "max"}
    ):
        raise ProtocolPreflightError("candidate does not freeze the accepted matched protocol")
    policy_object = _policy_from_plan({"execution_policy": policy})
    frozen_tool_contract = required_tool_contract(policy_object)
    if (
        protocol.get("required_tool_contract") != frozen_tool_contract
        or protocol.get("required_tool_contract_sha256")
        != frozen_tool_contract["content_address"]["digest"]
        or protocol.get("required_tool_contract_max_intermediate_tokens")
        != frozen_tool_contract["limits"]["max_intermediate_tokens"]
    ):
        raise ProtocolPreflightError("candidate required-tool diagnostic is not content-bound")
    if len(candidates) != 14:
        raise ProtocolPreflightError("the exact current preflight slate must contain 14 routes")
    manifest_sha256 = str(manifest["content_address"]["digest"])
    policy_sha256 = str(design.get("execution_policy_sha256") or "")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    source_hashes = {
        filename: hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
        for filename in PREFLIGHT_SOURCE_FILES
    }
    source_manifest_sha256 = _sha256(source_hashes)
    entries: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        endpoint_execution_sha256 = endpoint_execution_contract_sha256(candidate.endpoint)
        work_item = {
            "schema_version": "flavourbench-matched-protocol-preflight-work-item-v4",
            "candidate_manifest_sha256": manifest_sha256,
            "execution_policy_sha256": policy_sha256,
            "orchestration_source_manifest_sha256": source_manifest_sha256,
            "prompt_sha256": prompt_sha256,
            "task_id": PREFLIGHT_TASK_ID,
            "category": category,
            "model_id": candidate.model_id,
            "canonical_model_slug": candidate.canonical_model_slug,
            "provider_tag": candidate.provider_tag,
            "actual_provider_name": candidate.provider_name,
            "endpoint_execution_sha256": endpoint_execution_sha256,
            "required_tool_contract_sha256": frozen_tool_contract["content_address"]["digest"],
            "required_lanes": [
                "natural_epicure_off_on_pair",
                "direct_tool_first_diagnostic",
            ],
        }
        entries.append(
            {
                "ordinal": ordinal,
                "work_item_id": _sha256(work_item),
                **work_item,
            }
        )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "frozen_before_generation",
        "official": False,
        "rank_eligible": False,
        "candidate_manifest_sha256": manifest_sha256,
        "execution_policy": policy,
        "execution_policy_sha256": policy_sha256,
        "generation_protocol": dict(protocol),
        "required_tool_contract": frozen_tool_contract,
        "orchestration_source_sha256": source_hashes,
        "orchestration_source_manifest_sha256": source_manifest_sha256,
        "task": {
            "task_id": PREFLIGHT_TASK_ID,
            "prompt": prompt,
            "prompt_sha256": prompt_sha256,
            "category": category,
            "quality_observations_used_for_selection": 0,
        },
        "acceptance": {
            "required_receipts": len(entries),
            "required_result_lanes": ["epicure_off", "epicure_on", "tool_contract"],
            "all_generation_costs_reconciled": True,
            "normal_finish_required": True,
            "exact_model_and_provider_identity_required": True,
            "required_tool_trace": "successful find_pairings in tool_contract",
            "required_tool_contract_protocol": DIRECT_TOOL_CONTRACT_PROTOCOL,
            "required_tool_contract_is_permanently_unranked": True,
            "natural_tool_adoption_is_a_reported_mediator_not_an_eligibility_gate": True,
            "failed_or_retried_receipts_may_not_be_silently_replaced": True,
        },
        "entries": entries,
    }


def _entry_by_work_item(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ProtocolPreflightError("preflight plan has no entries")
    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ProtocolPreflightError("preflight plan contains a non-object entry")
        work_item_id = str(entry.get("work_item_id") or "")
        if len(work_item_id) != 64 or work_item_id in indexed:
            raise ProtocolPreflightError("preflight work-item identity is absent or duplicated")
        indexed[work_item_id] = entry
    return indexed


def _validate_result(
    *,
    result: object,
    lane: str,
    canonical_model_slug: str,
    provider_name: str,
    evidence_protocol: str,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ProtocolPreflightError(f"preflight receipt lacks {lane}")
    if (
        result.get("actual_model_id") != canonical_model_slug
        or result.get("actual_provider") != provider_name
        or result.get("finish_reason") != "stop"
        or result.get("cost_reconciled") is not True
        or result.get("final_response_mode") != "plain_text"
        or result.get("structured_output_requested") is not False
        or not str(result.get("answer_markdown") or "").strip()
        or not result.get("generation_ids")
    ):
        raise ProtocolPreflightError(f"preflight {lane} failed identity/completion invariants")
    traces = result.get("tool_trace")
    if not isinstance(traces, list):
        raise ProtocolPreflightError(f"preflight {lane} has no tool-trace list")
    intermediate_outputs = result.get("intermediate_outputs")
    if not isinstance(intermediate_outputs, list) or any(
        not isinstance(output, Mapping)
        or output.get("truncated") is True
        or output.get("finish_reason") == "length"
        for output in intermediate_outputs
    ):
        raise ProtocolPreflightError(f"preflight {lane} has an invalid intermediate trace")
    phases = [str(output.get("phase") or "") for output in intermediate_outputs]
    if lane == "tool_contract":
        if (
            not phases
            or phases[0] != "tool_selection"
            or "planning" in phases
            or "evidence_decision" in phases
            or int(intermediate_outputs[0].get("tool_call_count") or 0) != 1
        ):
            raise ProtocolPreflightError("required-tool diagnostic is not a direct tool-first lane")
    elif evidence_protocol == MATCHED_TOOL_ACCESS_PROTOCOL_V1:
        if lane == "epicure_off" and phases:
            raise ProtocolPreflightError(
                "matched-tool-access off lane unexpectedly has an intermediate turn"
            )
        if lane == "epicure_on" and (
            not phases or any(phase != "tool_selection" for phase in phases)
        ):
            raise ProtocolPreflightError(
                "matched-tool-access on lane lacks its bounded tool-selection trace"
            )
    else:
        if not phases:
            raise ProtocolPreflightError(f"preflight {lane} lacks its staged intermediate trace")
        required_phase = "evidence_decision" if lane == "epicure_off" else "tool_selection"
        if phases[0] != "planning" or required_phase not in phases:
            raise ProtocolPreflightError(f"preflight {lane} lacks its required matched-stage trace")
    if lane == "epicure_off" and traces:
        raise ProtocolPreflightError("Epicure-off preflight unexpectedly contains tool traces")
    return {
        "generation_ids": list(result.get("generation_ids") or []),
        "finish_reason": result.get("finish_reason"),
        "cost_micros": int(result.get("cost_micros") or 0),
        "prompt_tokens": int(result.get("prompt_tokens") or 0),
        "completion_tokens": int(result.get("completion_tokens") or 0),
        "reasoning_tokens": int(result.get("reasoning_tokens") or 0),
        "tool_call_count": len(traces),
        "successful_tool_call_count": sum(
            1 for trace in traces if isinstance(trace, Mapping) and not trace.get("is_error")
        ),
    }


def validate_receipt(
    *,
    artifact_path: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify one real live-smoke receipt against its predeclared work item."""

    artifact = _verified_live_smoke(artifact_path)
    indexed = _entry_by_work_item(plan)
    work_item_id = str(artifact.get("dataset_work_item_id") or "")
    entry = indexed.get(work_item_id)
    if entry is None:
        raise ProtocolPreflightError("live receipt is absent from the frozen preflight slate")
    task = plan.get("task")
    if not isinstance(task, Mapping):
        raise ProtocolPreflightError("preflight plan has no task contract")
    expected_exact = {
        "status": "complete",
        "run_purpose": "epicure_on_off_pair",
        "candidate_manifest_sha256": plan.get("candidate_manifest_sha256"),
        "dataset_task_id": task.get("task_id"),
        "prompt_sha256": task.get("prompt_sha256"),
        "category": task.get("category"),
        "requested_model_id": entry.get("model_id"),
        "requested_provider": entry.get("provider_tag"),
        "endpoint_execution_contract_sha256": entry.get("endpoint_execution_sha256"),
    }
    if any(artifact.get(field) != value for field, value in expected_exact.items()):
        raise ProtocolPreflightError("live receipt differs from the frozen preflight work item")
    if artifact.get("official") is not False or artifact.get("rank_eligible") is not False:
        raise ProtocolPreflightError("preflight receipt must remain permanently unranked")
    if artifact.get("errors") != {}:
        raise ProtocolPreflightError("preflight receipt contains a failed lane")
    if artifact.get("execution_policy") != plan.get("execution_policy"):
        raise ProtocolPreflightError("preflight receipt execution policy differs from the plan")
    frozen = artifact.get("frozen_generation_contract")
    generation_protocol = plan.get("generation_protocol")
    if not isinstance(frozen, Mapping) or not isinstance(generation_protocol, Mapping):
        raise ProtocolPreflightError("preflight frozen generation contract does not match")
    if (
        frozen.get("evidence_protocol") != generation_protocol.get("evidence_protocol")
        or frozen.get("final_response_mode") != "plain_text"
        or frozen.get("matched_planning")
        is not (generation_protocol.get("evidence_protocol") in MATCHED_EVIDENCE_PROTOCOLS)
        or frozen.get("required_tool_contract_protocol") != DIRECT_TOOL_CONTRACT_PROTOCOL
        or frozen.get("required_tool_contract_max_intermediate_tokens")
        != generation_protocol.get("required_tool_contract_max_intermediate_tokens")
        or frozen.get("required_tool_contract_sha256")
        != generation_protocol.get("required_tool_contract_sha256")
        or frozen.get("intermediate_reasoning_effort")
        != generation_protocol.get("intermediate_reasoning_effort")
        or frozen.get("final_reasoning_effort") != generation_protocol.get("final_reasoning_effort")
        or frozen.get("expected_actual_model_id") != entry.get("canonical_model_slug")
        or frozen.get("expected_actual_provider_slug") != entry.get("actual_provider_name")
    ):
        raise ProtocolPreflightError("preflight frozen generation contract does not match")
    prompts = artifact.get("system_prompt_sha256")
    if (
        not isinstance(prompts, Mapping)
        or prompts.get("epicure_off") != prompts.get("epicure_on")
        or not prompts.get("epicure_off")
    ):
        raise ProtocolPreflightError("preflight arms do not share one system prompt")
    bundle = artifact.get("protocol_bundle")
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("schema_version") != LIVE_PROTOCOL_SCHEMA_VERSION
    ):
        raise ProtocolPreflightError("preflight receipt has the wrong live protocol schema")
    binding = bundle.get("run_binding")
    if not isinstance(binding, Mapping) or any(
        binding.get(field) != value
        for field, value in {
            "candidate_manifest_sha256": plan.get("candidate_manifest_sha256"),
            "dataset_work_item_id": work_item_id,
            "dataset_task_id": task.get("task_id"),
            "prompt_sha256": task.get("prompt_sha256"),
            "requested_model_id": entry.get("model_id"),
            "canonical_model_slug": entry.get("canonical_model_slug"),
            "provider_tag": entry.get("provider_tag"),
            "execution_policy_sha256": plan.get("execution_policy_sha256"),
            "evidence_protocol": generation_protocol.get("evidence_protocol"),
            "required_tool_contract_protocol": DIRECT_TOOL_CONTRACT_PROTOCOL,
            "required_tool_contract_max_intermediate_tokens": generation_protocol.get(
                "required_tool_contract_max_intermediate_tokens"
            ),
            "required_tool_contract_sha256": generation_protocol.get(
                "required_tool_contract_sha256"
            ),
            "intermediate_reasoning_effort": generation_protocol.get(
                "intermediate_reasoning_effort"
            ),
            "final_reasoning_effort": generation_protocol.get("final_reasoning_effort"),
        }.items()
    ):
        raise ProtocolPreflightError("preflight live protocol binding does not match the plan")
    core_bundle = bundle.get("core_protocol_bundle")
    implementation_hashes = (
        core_bundle.get("implementation_sha256") if isinstance(core_bundle, Mapping) else None
    )
    release_inputs = core_bundle.get("release_inputs") if isinstance(core_bundle, Mapping) else None
    if not isinstance(implementation_hashes, Mapping) or not isinstance(release_inputs, Mapping):
        raise ProtocolPreflightError("preflight receipt lacks implementation provenance")
    expected_sources = plan.get("orchestration_source_sha256")
    if not isinstance(expected_sources, Mapping) or any(
        implementation_hashes.get(filename) != digest
        for filename, digest in expected_sources.items()
    ):
        raise ProtocolPreflightError(
            "preflight receipt implementation differs from the frozen orchestration source"
        )
    budget = artifact.get("budget")
    if not isinstance(budget, Mapping) or budget.get("all_generation_costs_reconciled") is not True:
        raise ProtocolPreflightError("preflight receipt has unreconciled generation cost")
    if artifact.get("required_tool_contract") != plan.get("required_tool_contract"):
        raise ProtocolPreflightError("preflight receipt changed the required-tool contract")
    results = artifact.get("results")
    if not isinstance(results, Mapping) or set(results) != {
        "epicure_off",
        "epicure_on",
        "tool_contract",
    }:
        raise ProtocolPreflightError("preflight receipt does not contain exactly three lanes")
    lane_summaries = {
        lane: _validate_result(
            result=results[lane],
            lane=lane,
            canonical_model_slug=str(entry.get("canonical_model_slug") or ""),
            provider_name=str(entry.get("actual_provider_name") or ""),
            evidence_protocol=str(generation_protocol.get("evidence_protocol") or ""),
        )
        for lane in ("epicure_off", "epicure_on", "tool_contract")
    }
    tool_traces = results["tool_contract"].get("tool_trace") or []
    if not any(
        isinstance(trace, Mapping)
        and trace.get("name") == "find_pairings"
        and trace.get("is_error") is False
        for trace in tool_traces
    ):
        raise ProtocolPreflightError("required-tool lane has no successful find_pairings trace")
    first_tool_trace = tool_traces[0] if tool_traces else {}
    direct_tool_schema_sha256 = str(
        results["tool_contract"].get("backend_tool_schema_sha256") or ""
    )
    if len(direct_tool_schema_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in direct_tool_schema_sha256
    ):
        raise ProtocolPreflightError("required-tool lane lacks its singleton schema hash")
    epicure = artifact.get("epicure")
    if not isinstance(epicure, Mapping):
        raise ProtocolPreflightError("preflight receipt has no Epicure provenance")
    epicure_identity = {
        "release_id": str(epicure.get("release_id") or ""),
        "bundle_sha256": str(epicure.get("bundle_sha256") or ""),
        "application_sha256": str(epicure.get("application_sha256") or ""),
        "tool_schema_sha256": str(artifact.get("epicure_tool_schema_sha256") or ""),
    }
    if not epicure_identity["release_id"] or any(
        len(value) != 64 for key, value in epicure_identity.items() if key != "release_id"
    ):
        raise ProtocolPreflightError("preflight Epicure provenance is incomplete")
    receipt = {
        "work_item_id": work_item_id,
        "model_id": entry.get("model_id"),
        "canonical_model_slug": entry.get("canonical_model_slug"),
        "provider_tag": entry.get("provider_tag"),
        "actual_provider_name": entry.get("actual_provider_name"),
        "endpoint_execution_sha256": entry.get("endpoint_execution_sha256"),
        "artifact_filename": artifact_path.name,
        "artifact_sha256": artifact.get("artifact_sha256"),
        "journal_sha256": (artifact.get("run_journal") or {}).get("sha256"),
        "implementation_manifest_sha256": _sha256(implementation_hashes),
        "release_inputs_sha256": _sha256(release_inputs),
        "container_image_digest": release_inputs.get("container_image_digest"),
        "actual_cost_micros": int(budget.get("actual_cost_micros") or 0),
        "lanes": lane_summaries,
        "natural_epicure_tool_adopted": lane_summaries["epicure_on"]["tool_call_count"] > 0,
        "direct_tool_first_attempt_valid": bool(
            isinstance(first_tool_trace, Mapping)
            and first_tool_trace.get("name") == "find_pairings"
            and first_tool_trace.get("is_error") is False
        ),
        "direct_tool_schema_sha256": direct_tool_schema_sha256,
        "passed": True,
    }
    return receipt, epicure_identity


def build_registry(
    *,
    plan_path: Path,
    artifact_paths: Sequence[Path],
) -> dict[str, Any]:
    """Aggregate exactly one successful, predeclared receipt per route."""

    plan = _verified_artifact(plan_path, schema_version=PLAN_SCHEMA_VERSION)
    indexed = _entry_by_work_item(plan)
    if len(artifact_paths) != len(indexed):
        raise ProtocolPreflightError(
            f"expected {len(indexed)} explicitly selected receipts, got {len(artifact_paths)}"
        )
    receipts: dict[str, dict[str, Any]] = {}
    epicure_identities: set[tuple[str, str, str, str]] = set()
    implementation_identities: set[tuple[str, str, str]] = set()
    direct_tool_schema_identities: set[str] = set()
    for path in artifact_paths:
        receipt, epicure = validate_receipt(artifact_path=path, plan=plan)
        work_item_id = str(receipt["work_item_id"])
        if work_item_id in receipts:
            raise ProtocolPreflightError("two receipts were selected for one preflight work item")
        receipts[work_item_id] = receipt
        epicure_identities.add(
            (
                epicure["release_id"],
                epicure["bundle_sha256"],
                epicure["application_sha256"],
                epicure["tool_schema_sha256"],
            )
        )
        implementation_identities.add(
            (
                str(receipt["implementation_manifest_sha256"]),
                str(receipt["release_inputs_sha256"]),
                str(receipt["container_image_digest"]),
            )
        )
        direct_tool_schema_identities.add(str(receipt["direct_tool_schema_sha256"]))
    missing = set(indexed) - set(receipts)
    if missing:
        raise ProtocolPreflightError(f"preflight registry is missing {len(missing)} routes")
    if len(epicure_identities) != 1:
        raise ProtocolPreflightError("preflight receipts do not share one Epicure runtime")
    if len(implementation_identities) != 1:
        raise ProtocolPreflightError("preflight receipts do not share one implementation build")
    if len(direct_tool_schema_identities) != 1:
        raise ProtocolPreflightError(
            "preflight receipts do not share one singleton required-tool schema"
        )
    release_id, bundle_sha, application_sha, tool_sha = next(iter(epicure_identities))
    implementation_sha, release_inputs_sha, container_digest = next(iter(implementation_identities))
    ordered_receipts = [receipts[str(entry["work_item_id"])] for entry in indexed.values()]
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "passed",
        "official": False,
        "rank_eligible": False,
        "candidate_manifest_sha256": plan.get("candidate_manifest_sha256"),
        "preflight_plan_sha256": plan.get("artifact_sha256"),
        "execution_policy_sha256": plan.get("execution_policy_sha256"),
        "protocol_schema_version": LIVE_PROTOCOL_SCHEMA_VERSION,
        "required_tool_contract_sha256": plan.get("required_tool_contract", {})
        .get("content_address", {})
        .get("digest"),
        "model_count": len(ordered_receipts),
        "receipt_count": len(ordered_receipts),
        "all_required_routes_passed": True,
        "synthetic_receipts": 0,
        "direct_tool_first_attempt_valid_count": sum(
            bool(receipt.get("direct_tool_first_attempt_valid")) for receipt in ordered_receipts
        ),
        "direct_tool_schema_sha256": next(iter(direct_tool_schema_identities)),
        "epicure": {
            "release_id": release_id,
            "bundle_sha256": bundle_sha,
            "application_sha256": application_sha,
            "tool_schema_sha256": tool_sha,
        },
        "implementation": {
            "implementation_manifest_sha256": implementation_sha,
            "release_inputs_sha256": release_inputs_sha,
            "container_image_digest": container_digest,
            "official_container_identity_resolved": container_digest != "unresolved",
        },
        "receipts": ordered_receipts,
    }


def verify_registry_for_manifest(
    *, registry_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a registry against an already promoted collection manifest."""

    registry = _verified_artifact(registry_path, schema_version=REGISTRY_SCHEMA_VERSION)
    promotion = manifest.get("protocol_preflight")
    design = manifest.get("run_design")
    if not isinstance(promotion, Mapping) or not isinstance(design, Mapping):
        raise ProtocolPreflightError("collection manifest is not preflight-promoted")
    if (
        promotion.get("schema_version") != PROMOTION_SCHEMA_VERSION
        or promotion.get("status") != "passed"
        or promotion.get("registry_sha256") != registry.get("artifact_sha256")
        or promotion.get("plan_sha256") != registry.get("preflight_plan_sha256")
        or promotion.get("basis_manifest_sha256") != registry.get("candidate_manifest_sha256")
        or registry.get("execution_policy_sha256") != design.get("execution_policy_sha256")
        or registry.get("model_count") != len(manifest.get("models") or [])
        or registry.get("all_required_routes_passed") is not True
        or registry.get("synthetic_receipts") != 0
    ):
        raise ProtocolPreflightError("preflight registry does not authorize this manifest")
    expected_routes = {
        (
            str(entry.get("model", {}).get("id") or ""),
            str(entry.get("model", {}).get("canonical_slug") or ""),
            str(entry.get("endpoint", {}).get("tag") or ""),
            endpoint_execution_contract_sha256(dict(entry.get("endpoint") or {})),
        )
        for entry in manifest.get("models") or []
        if isinstance(entry, Mapping)
    }
    observed_routes = {
        (
            str(receipt.get("model_id") or ""),
            str(receipt.get("canonical_model_slug") or ""),
            str(receipt.get("provider_tag") or ""),
            str(receipt.get("endpoint_execution_sha256") or ""),
        )
        for receipt in registry.get("receipts") or []
        if isinstance(receipt, Mapping) and receipt.get("passed") is True
    }
    if expected_routes != observed_routes:
        raise ProtocolPreflightError("preflight registry route set differs from the manifest")
    return registry


def promote_manifest(
    *, manifest_path: Path, expected_manifest_sha256: str, registry_path: Path
) -> dict[str, Any]:
    """Bind a passing preflight registry into a new immutable manifest."""

    try:
        manifest = load_candidate_manifest(
            manifest_path,
            expected_digest=expected_manifest_sha256,
        )
    except IntegrityError as error:
        raise ProtocolPreflightError(str(error)) from error
    registry = _verified_artifact(registry_path, schema_version=REGISTRY_SCHEMA_VERSION)
    if (
        registry.get("status") != "passed"
        or registry.get("candidate_manifest_sha256") != expected_manifest_sha256
        or registry.get("execution_policy_sha256")
        != (manifest.get("run_design") or {}).get("execution_policy_sha256")
        or registry.get("model_count") != len(manifest.get("models") or [])
        or registry.get("all_required_routes_passed") is not True
    ):
        raise ProtocolPreflightError("passing registry does not match the base manifest")
    promoted = copy.deepcopy(manifest)
    promoted.pop("content_address", None)
    promoted["protocol_preflight"] = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "passed",
        "basis_manifest_sha256": expected_manifest_sha256,
        "plan_sha256": registry.get("preflight_plan_sha256"),
        "registry_sha256": registry.get("artifact_sha256"),
        "model_count": registry.get("model_count"),
        "epicure": registry.get("epicure"),
        "implementation": registry.get("implementation"),
    }
    governance = promoted.get("governance")
    if isinstance(governance, dict):
        governance["freeze_status"] = "exact_routes_workload_and_protocol_preflights_frozen"
    digest = _sha256(promoted)
    promoted["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest_content_address(promoted):
        raise ProtocolPreflightError("promoted manifest content address failed")
    return promoted


def _write_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    if not verify_manifest_content_address(manifest):
        raise ProtocolPreflightError("refusing an invalid promoted manifest")
    digest = str(manifest["content_address"]["digest"])
    path = output_dir / f"flavourbench-openrouter-unranked-{digest}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ProtocolPreflightError("promoted manifest output conflict")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ProtocolPreflightError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise ProtocolPreflightError(f"{field} must be finite and non-negative")
    return parsed


def _ledger_digest(entry: Mapping[str, Any]) -> str:
    payload = dict(entry)
    payload.pop("entry_sha256", None)
    return _sha256(payload)


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ProtocolPreflightError("preflight ledger must be a regular file")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProtocolPreflightError(f"invalid preflight ledger line {line_number}") from error
        if (
            not isinstance(entry, dict)
            or entry.get("schema_version") != LEDGER_SCHEMA_VERSION
            or entry.get("sequence") != line_number
            or entry.get("previous_entry_sha256") != previous
            or entry.get("entry_sha256") != _ledger_digest(entry)
        ):
            raise ProtocolPreflightError(f"preflight ledger hash chain fails at line {line_number}")
        entries.append(entry)
        previous = str(entry["entry_sha256"])
    return entries


def _append_ledger(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = {"api_key", "authorization", "mcp_token", "stderr", "stdout"}
    if any(str(key).lower() in forbidden_keys for key in event):
        raise ProtocolPreflightError("preflight ledger event contains a forbidden field")
    entries = _load_ledger(path)
    entry = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sequence": len(entries) + 1,
        "previous_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        **dict(event),
    }
    entry["entry_sha256"] = _ledger_digest(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _canonical(entry) + b"\n"
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        if os.write(descriptor, rendered) != len(rendered):
            raise OSError("short append while writing preflight ledger")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return entry


@contextmanager
def _ledger_lock(path: Path) -> Iterator[TextIO]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _policy_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    policy = plan.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise ProtocolPreflightError("preflight plan has no execution policy")
    limits = policy.get("limits")
    decoding = policy.get("decoding")
    if not isinstance(limits, Mapping) or not isinstance(decoding, Mapping):
        raise ProtocolPreflightError("preflight policy limits or decoding are absent")
    return {
        "FLAVOURBENCH_MAX_OUTPUT_TOKENS": str(limits["max_output_tokens"]),
        "FLAVOURBENCH_MAX_INTERMEDIATE_TOKENS": str(limits["max_intermediate_tokens"]),
        "FLAVOURBENCH_MAX_TOOL_ROUNDS": str(limits["max_tool_rounds"]),
        "FLAVOURBENCH_MAX_TOOL_RESULT_BYTES": str(limits["max_tool_result_bytes"]),
        "FLAVOURBENCH_MAX_CUMULATIVE_TOOL_RESULT_BYTES": str(
            limits["max_cumulative_tool_result_bytes"]
        ),
        "FLAVOURBENCH_MAX_TOOL_CALLS_PER_ROUND": str(limits["max_tool_calls_per_round"]),
        "FLAVOURBENCH_MAX_TOOL_CALLS_TOTAL": str(limits["max_tool_calls_total"]),
        "FLAVOURBENCH_MAX_PROVIDER_ATTEMPTS": str(limits["max_provider_attempts"]),
        "FLAVOURBENCH_DECODING_TEMPERATURE": str(decoding["temperature"]),
        "FLAVOURBENCH_DECODING_TOP_P": str(decoding["top_p"]),
        "FLAVOURBENCH_DECODING_SEED": str(decoding["seed"]),
    }


def _policy_from_plan(plan: Mapping[str, Any]):
    from .execution_policy import ExecutionPolicy

    policy = plan.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise ProtocolPreflightError("preflight plan has no policy")
    limits = policy.get("limits")
    decoding = policy.get("decoding")
    cost = policy.get("cost_forecast")
    reasoning = policy.get("reasoning")
    if not all(isinstance(value, Mapping) for value in (limits, decoding, cost, reasoning)):
        raise ProtocolPreflightError("preflight policy is incomplete")
    return ExecutionPolicy(
        max_output_tokens=int(limits["max_output_tokens"]),
        max_intermediate_tokens=int(limits["max_intermediate_tokens"]),
        required_tool_contract_max_intermediate_tokens=int(
            limits["required_tool_contract_max_intermediate_tokens"]
        ),
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
        approximate_non_user_prompt_bytes=int(cost["approximate_non_user_prompt_bytes"]),
        conservative_bytes_per_token=int(cost["conservative_bytes_per_token"]),
        pair_arm_scheduling=str(policy["pair_arm_scheduling"]),
        final_response_mode=str(policy["final_response_mode"]),
        matched_planning=bool(policy["matched_planning"]),
        evidence_protocol=str(policy["evidence_protocol"]),
        required_tool_contract_protocol=str(policy["required_tool_contract_protocol"]),
        intermediate_reasoning_effort=str(reasoning["intermediate_effort"]),
        final_reasoning_effort=str(reasoning["final_effort"]),
        tool_catalog_bytes_bound=int(cost["tool_catalog_bytes_bound"]),
    )


def _scan_selected_receipts(
    *, output_dir: Path, plan: Mapping[str, Any]
) -> dict[str, tuple[dict[str, Any], Path]]:
    indexed = _entry_by_work_item(plan)
    selected: dict[str, tuple[dict[str, Any], Path]] = {}
    if not output_dir.exists():
        return selected
    for path in sorted(output_dir.glob("*.json")):
        try:
            artifact = _verified_live_smoke(path)
        except ProtocolPreflightError:
            continue
        work_item_id = str(artifact.get("dataset_work_item_id") or "")
        if (
            artifact.get("candidate_manifest_sha256") != plan.get("candidate_manifest_sha256")
            or work_item_id not in indexed
        ):
            continue
        if work_item_id in selected:
            raise ProtocolPreflightError(
                "multiple real receipts exist for one preflight work item; selection is frozen"
            )
        try:
            receipt, _epicure = validate_receipt(artifact_path=path, plan=plan)
        except ProtocolPreflightError:
            continue
        selected[work_item_id] = (receipt, path)
    return selected


def execute_plan(
    *,
    plan_path: Path,
    manifest_path: Path,
    output_dir: Path,
    ledger_path: Path,
    cap_usd: Decimal,
    confirmation: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run the predeclared real slate sequentially with a hash-chained reserve."""

    if confirmation != EXECUTION_CONFIRMATION:
        raise ProtocolPreflightError(f"real preflight requires --confirm {EXECUTION_CONFIRMATION}")
    if cap_usd <= 0:
        raise ProtocolPreflightError("preflight cap must be positive")
    plan = _verified_artifact(plan_path, schema_version=PLAN_SCHEMA_VERSION)
    manifest_sha256 = str(plan.get("candidate_manifest_sha256") or "")
    try:
        manifest = load_candidate_manifest(
            manifest_path,
            expected_digest=manifest_sha256,
        )
        candidates = select_candidates(manifest)
    except IntegrityError as error:
        raise ProtocolPreflightError(str(error)) from error
    by_model = {candidate.model_id: candidate for candidate in candidates}
    policy = _policy_from_plan(plan)
    task = plan.get("task")
    if not isinstance(task, Mapping):
        raise ProtocolPreflightError("preflight task contract is absent")
    forecasts: dict[str, Decimal] = {}
    for entry in plan.get("entries") or []:
        if not isinstance(entry, Mapping):
            raise ProtocolPreflightError("preflight plan entry is malformed")
        candidate = by_model.get(str(entry.get("model_id") or ""))
        if candidate is None:
            raise ProtocolPreflightError("preflight entry is absent from the manifest")
        forecasts[str(entry["work_item_id"])] = _worst_case_cost_usd(
            candidate.endpoint,
            prompt=str(task["prompt"]),
            include_tool_contract=True,
            execution_policy=policy,
        )
    total_forecast = sum(forecasts.values(), Decimal(0))
    admission_ceiling = cap_usd * Decimal("0.85")
    if total_forecast > admission_ceiling:
        raise ProtocolPreflightError(
            f"preflight worst-case ${total_forecast} exceeds the 85% ceiling ${admission_ceiling}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with _ledger_lock(ledger_path):
        existing = _scan_selected_receipts(output_dir=output_dir, plan=plan)
        ledger = _load_ledger(ledger_path)
        reserved = {
            str(entry.get("work_item_id") or "")
            for entry in ledger
            if entry.get("event_type") == "reservation_created"
        }
        finalized = {
            str(entry.get("work_item_id") or "")
            for entry in ledger
            if entry.get("event_type") in {"receipt_recorded", "existing_receipt_adopted"}
        }
        unresolved = reserved - finalized - set(existing)
        if unresolved:
            raise ProtocolPreflightError(
                "an earlier paid preflight reservation has no verifiable receipt; refusing replay"
            )
        for work_item_id, (receipt, path) in existing.items():
            if work_item_id not in finalized:
                _append_ledger(
                    ledger_path,
                    {
                        "event_type": "existing_receipt_adopted",
                        "plan_sha256": plan["artifact_sha256"],
                        "work_item_id": work_item_id,
                        "artifact_filename": path.name,
                        "artifact_sha256": receipt["artifact_sha256"],
                        "actual_cost_micros": receipt["actual_cost_micros"],
                    },
                )
        environment = dict(os.environ)
        environment.update(_policy_environment(plan))
        failures: list[dict[str, Any]] = []
        for entry in plan.get("entries") or []:
            assert isinstance(entry, Mapping)
            work_item_id = str(entry["work_item_id"])
            if work_item_id in existing:
                continue
            forecast = forecasts[work_item_id]
            _append_ledger(
                ledger_path,
                {
                    "event_type": "reservation_created",
                    "plan_sha256": plan["artifact_sha256"],
                    "work_item_id": work_item_id,
                    "model_id": entry["model_id"],
                    "provider_tag": entry["provider_tag"],
                    "reserved_usd": format(forecast, "f"),
                },
            )
            command = [
                sys.executable,
                "-m",
                "flavourbench.live_smoke",
                "--confirm",
                "UNRANKED_REAL_SMOKE",
                "--cap-usd",
                format(forecast + Decimal("0.000001"), "f"),
                "--model-id",
                str(entry["model_id"]),
                "--provider-slug",
                str(entry["provider_tag"]),
                "--prompt",
                str(task["prompt"]),
                "--category",
                str(task["category"]),
                "--plain-text-final",
                "--tool-catalog-bytes-bound",
                str(policy.tool_catalog_bytes_bound),
                "--evidence-protocol",
                policy.evidence_protocol,
                "--intermediate-reasoning-effort",
                str(policy.intermediate_reasoning_effort),
                "--final-reasoning-effort",
                str(policy.final_reasoning_effort),
                "--output-dir",
                str(output_dir),
                "--candidate-manifest-sha256",
                manifest_sha256,
                "--dataset-work-item-id",
                work_item_id,
                "--dataset-task-id",
                str(task["task_id"]),
                "--expected-canonical-model-slug",
                str(entry["canonical_model_slug"]),
                "--expected-endpoint-execution-sha256",
                str(entry["endpoint_execution_sha256"]),
                "--expected-execution-policy-sha256",
                str(plan["execution_policy_sha256"]),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
            stdout_sha256 = hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest()
            stderr_sha256 = hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest()
            try:
                summary = json.loads(completed.stdout)
                artifact_path = Path(str(summary["artifact"]))
                receipt, _epicure = validate_receipt(
                    artifact_path=artifact_path,
                    plan=plan,
                )
            except Exception as error:
                failure_artifact: dict[str, Any] = {}
                try:
                    raw_artifact_path = Path(
                        str((json.loads(completed.stdout) or {}).get("artifact") or "")
                    )
                    raw_artifact = _verified_live_smoke(raw_artifact_path)
                    failure_artifact = {
                        "artifact_filename": raw_artifact_path.name,
                        "artifact_sha256": raw_artifact.get("artifact_sha256"),
                        "artifact_status": raw_artifact.get("status"),
                        "all_generation_costs_reconciled": (
                            (raw_artifact.get("budget") or {}).get(
                                "all_generation_costs_reconciled"
                            )
                        ),
                        "actual_cost_micros": int(
                            (raw_artifact.get("budget") or {}).get("actual_cost_micros") or 0
                        ),
                    }
                except Exception:
                    failure_artifact = {}
                _append_ledger(
                    ledger_path,
                    {
                        "event_type": "execution_failed",
                        "plan_sha256": plan["artifact_sha256"],
                        "work_item_id": work_item_id,
                        "returncode": completed.returncode,
                        "stdout_sha256": stdout_sha256,
                        "stderr_sha256": stderr_sha256,
                        "error_class": type(error).__name__,
                        "model_id": entry["model_id"],
                        **failure_artifact,
                    },
                )
                if (
                    not failure_artifact
                    or failure_artifact.get("all_generation_costs_reconciled") is not True
                ):
                    raise ProtocolPreflightError(
                        f"preflight accounting is unresolved for {entry['model_id']}; "
                        "stopping before another provider request"
                    ) from error
                failures.append(
                    {
                        "work_item_id": work_item_id,
                        "model_id": entry["model_id"],
                        "provider_tag": entry["provider_tag"],
                        "error_class": type(error).__name__,
                        **failure_artifact,
                    }
                )
                continue
            _append_ledger(
                ledger_path,
                {
                    "event_type": "receipt_recorded",
                    "plan_sha256": plan["artifact_sha256"],
                    "work_item_id": work_item_id,
                    "artifact_filename": artifact_path.name,
                    "artifact_sha256": receipt["artifact_sha256"],
                    "actual_cost_micros": receipt["actual_cost_micros"],
                    "returncode": completed.returncode,
                    "stdout_sha256": stdout_sha256,
                    "stderr_sha256": stderr_sha256,
                },
            )
            existing[work_item_id] = (receipt, artifact_path)
        final_receipts = _scan_selected_receipts(output_dir=output_dir, plan=plan)
        actual_cost_micros = sum(
            int(receipt["actual_cost_micros"]) for receipt, _path in final_receipts.values()
        ) + sum(int(failure.get("actual_cost_micros") or 0) for failure in failures)
        complete = not failures and len(final_receipts) == len(plan.get("entries") or [])
        return {
            "status": "complete" if complete else "failed",
            "official": False,
            "rank_eligible": False,
            "plan_sha256": plan["artifact_sha256"],
            "planned_model_count": len(plan.get("entries") or []),
            "passed_model_count": len(final_receipts),
            "failed_model_count": len(failures),
            "forecast_worst_case_usd": format(total_forecast, "f"),
            "cap_usd": format(cap_usd, "f"),
            "actual_cost_micros": actual_cost_micros,
            "artifacts": [
                str(final_receipts[str(entry["work_item_id"])][1])
                for entry in plan.get("entries") or []
                if isinstance(entry, Mapping) and str(entry["work_item_id"]) in final_receipts
            ],
            "failures": failures,
            "ledger": str(ledger_path),
        }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="freeze the no-call preflight slate")
    plan_parser.add_argument("--manifest", type=Path, required=True)
    plan_parser.add_argument("--expected-manifest-sha256", required=True)
    plan_parser.add_argument("--output-dir", type=Path, required=True)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="verify explicitly selected real preflight receipts"
    )
    aggregate_parser.add_argument("--plan", type=Path, required=True)
    aggregate_parser.add_argument("--artifact", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)

    execute_parser = subparsers.add_parser(
        "execute", help="run the complete real preflight slate sequentially"
    )
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--manifest", type=Path, required=True)
    execute_parser.add_argument("--output-dir", type=Path, required=True)
    execute_parser.add_argument("--ledger", type=Path, required=True)
    execute_parser.add_argument("--cap-usd", type=Decimal, default=Decimal("20"))
    execute_parser.add_argument("--confirm", required=True)
    execute_parser.add_argument("--timeout-seconds", type=int, default=600)

    promote_parser = subparsers.add_parser(
        "promote", help="bind a passing registry into an execution-ready manifest"
    )
    promote_parser.add_argument("--manifest", type=Path, required=True)
    promote_parser.add_argument("--expected-manifest-sha256", required=True)
    promote_parser.add_argument("--registry", type=Path, required=True)
    promote_parser.add_argument("--output-dir", type=Path, required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "plan":
        payload = build_plan(
            manifest_path=arguments.manifest,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
        )
        path = _write_artifact(arguments.output_dir, "matched-protocol-preflight-plan", payload)
        summary = {
            "status": "planned",
            "provider_calls_made": False,
            "output": str(path),
            "artifact_sha256": json.loads(path.read_text())["artifact_sha256"],
            "models": len(payload["entries"]),
        }
    elif arguments.command == "aggregate":
        payload = build_registry(
            plan_path=arguments.plan,
            artifact_paths=arguments.artifact,
        )
        path = _write_artifact(
            arguments.output_dir,
            "matched-protocol-preflight-registry",
            payload,
        )
        summary = {
            "status": "passed",
            "provider_calls_made": False,
            "output": str(path),
            "artifact_sha256": json.loads(path.read_text())["artifact_sha256"],
            "models": payload["model_count"],
        }
    elif arguments.command == "execute":
        summary = execute_plan(
            plan_path=arguments.plan,
            manifest_path=arguments.manifest,
            output_dir=arguments.output_dir,
            ledger_path=arguments.ledger,
            cap_usd=arguments.cap_usd,
            confirmation=arguments.confirm,
            timeout_seconds=arguments.timeout_seconds,
        )
    else:
        payload = promote_manifest(
            manifest_path=arguments.manifest,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
            registry_path=arguments.registry,
        )
        path = _write_manifest(arguments.output_dir, payload)
        summary = {
            "status": "promoted",
            "provider_calls_made": False,
            "output": str(path),
            "manifest_sha256": payload["content_address"]["digest"],
            "preflight_registry_sha256": payload["protocol_preflight"]["registry_sha256"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if arguments.command == "execute" and summary.get("status") != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    run()
