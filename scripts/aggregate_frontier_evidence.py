#!/usr/bin/env python3
"""Build the public, permanently unranked frontier-contract evidence snapshot.

This script is intentionally provider-free: it reads immutable ``live-smoke``
records, applies append-only cost corrections, verifies generation accounting,
validates the runner summary and hash-chained ledger (including no-artifact
reconciliation proofs), and writes one content-addressed aggregate.  The exact
same JSON bytes are copied into the paper and web application so the two
presentations cannot silently drift.

The aggregate describes systems compatibility only.  It never creates a vote,
preference score, Bradley--Terry rating, or Epicure-uplift estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "flavourbench-frontier-evidence-v1"
CONDITIONS = ("epicure_off", "epicure_on", "tool_contract")
FRONTIER_MODEL_ORDER = (
    "openai/gpt-5.6-sol-pro",
    "openai/gpt-5.6-luna",
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4.8",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite",
    "x-ai/grok-4.5",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "nvidia/nemotron-3-ultra-550b-a55b",
)
DISPLAY_NAMES = {
    "openai/gpt-5.6-sol-pro": "GPT-5.6 Sol Pro",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "anthropic/claude-fable-5": "Claude Fable 5",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "x-ai/grok-4.5": "Grok 4.5",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek/deepseek-r1-0528": "DeepSeek R1 0528",
    "qwen/qwen3.5-397b-a17b": "Qwen 3.5 397B",
    "z-ai/glm-5.2": "GLM 5.2",
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
}
VALID_STATES = {
    "valid_normalized",
    "valid_with_epicure_tool",
    "valid_no_tool_use",
    "valid_with_tool_error",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _artifact_digest(value: dict[str, Any]) -> str:
    recorded = value.get("artifact_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("live-smoke artifact is missing its content address")
    without_digest = dict(value)
    without_digest.pop("artifact_sha256", None)
    calculated = _sha256(without_digest)
    if calculated != recorded:
        raise ValueError(
            f"live-smoke content address mismatch: recorded={recorded}, calculated={calculated}"
        )
    return recorded


def _condition_events(artifact: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    suffix = f":{condition}"
    return [
        event
        for event in artifact.get("provider_attempt_events", [])
        if str(event.get("arm_id", "")).endswith(suffix)
    ]


def _classify_error(error: str, events: list[dict[str, Any]]) -> str:
    lowered = error.lower()
    if "requires a complete frozen endpoint contract" in lowered:
        return "local_preflight_rejection"
    if "parallel tool calls" in lowered:
        # These artifacts were rejected by the then-current FlavourBench client
        # limit.  The production engine later moved to a bounded parallel-call
        # policy, so this is historical contract evidence, not model incapacity.
        return "client_parallel_policy_rejection"
    if "invalid final json" in lowered:
        return "invalid_structured_output"
    rejected = [event for event in events if event.get("event_type") == "request_rejected"]
    if rejected:
        return "route_rejected"
    if "429 too many requests" in lowered and "/mcp" in lowered:
        return "mcp_transport_error"
    return "provider_error"


def _condition_observation(
    artifact: dict[str, Any],
    condition: str,
    accounted_generation_ids: set[str],
) -> dict[str, Any]:
    events = _condition_events(artifact, condition)
    results = artifact.get("results") or {}
    result = results.get(condition) if isinstance(results, dict) else None
    error = str((artifact.get("errors") or {}).get(condition, ""))
    response_ids = [
        str(event.get("generation_id"))
        for event in events
        if event.get("event_type") == "response_received" and event.get("generation_id")
    ]
    route_statuses = [
        int(event["http_status"])
        for event in events
        if event.get("event_type") == "request_rejected"
        and isinstance(event.get("http_status"), int)
    ]

    observation: dict[str, Any] = {
        "attempted": bool(events or error or isinstance(result, dict)),
        "request_count": sum(event.get("event_type") == "request_started" for event in events),
        "provider_generation_count": len(response_ids),
        "generation_ids": response_ids,
        "generation_costs_reconciled": all(
            generation_id in accounted_generation_ids for generation_id in response_ids
        ),
        "route_http_statuses": route_statuses,
    }

    if isinstance(result, dict):
        traces = result.get("tool_trace") or []
        tool_errors = sum(bool(trace.get("is_error")) for trace in traces)
        if traces and tool_errors:
            state = "valid_with_tool_error"
        elif traces:
            state = "valid_with_epicure_tool"
        elif condition == "epicure_on":
            state = "valid_no_tool_use"
        else:
            state = "valid_normalized"
        observation.update(
            {
                "state": state,
                "valid_normalized_response": True,
                "actual_model_id": result.get("actual_model_id"),
                "actual_provider": result.get("actual_provider"),
                "latency_ms": result.get("latency_ms"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "reasoning_tokens": result.get("reasoning_tokens"),
                "cost_micros": result.get("cost_micros"),
                "finish_reason": result.get("finish_reason"),
                "tool_calls": [
                    {
                        "name": trace.get("name"),
                        "round_index": trace.get("round_index"),
                        "is_error": bool(trace.get("is_error")),
                        "result_sha256": trace.get("result_sha256"),
                    }
                    for trace in traces
                ],
            }
        )
        return observation

    observation.update(
        {
            "state": _classify_error(error, events) if error else "not_attempted",
            "valid_normalized_response": False,
            "error": error,
        }
    )
    return observation


def _identity_records(
    artifacts: list[dict[str, Any]],
    correction_metadata: dict[str, list[dict[str, Any]]],
) -> list[dict[str, str]]:
    identities: set[tuple[str, str]] = set()
    for artifact in artifacts:
        for result in (artifact.get("results") or {}).values():
            if not isinstance(result, dict):
                continue
            model = result.get("actual_model_id")
            provider = result.get("actual_provider")
            if model and provider:
                identities.add((str(model), str(provider)))
        for event in artifact.get("provider_attempt_events", []):
            if event.get("event_type") != "accounting_reconciled":
                continue
            metadata = event.get("metadata") or {}
            model, provider = metadata.get("model"), metadata.get("provider")
            if model and provider:
                identities.add((str(model), str(provider)))
        for metadata in correction_metadata.get(str(artifact.get("artifact_sha256")), []):
            model, provider = metadata.get("model"), metadata.get("provider")
            if model and provider:
                identities.add((str(model), str(provider)))
    return [
        {"actual_model_id": model, "actual_provider": provider}
        for model, provider in sorted(identities)
    ]


def _best_observation(attempts: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    observations = [
        (attempt, attempt["conditions"][condition])
        for attempt in attempts
        if attempt["conditions"][condition]["attempted"]
    ]
    valid = [item for item in observations if item[1]["state"] in VALID_STATES]
    route_observations = [
        item
        for item in observations
        if item[1].get("request_count", 0) > 0
        or item[1].get("provider_generation_count", 0) > 0
        or item[1].get("route_http_statuses")
    ]
    candidates = valid or route_observations or observations
    attempt, observation = (
        candidates[-1]
        if candidates
        else (
            None,
            {
                "attempted": False,
                "state": "not_attempted",
                "valid_normalized_response": False,
            },
        )
    )
    selected = dict(observation)
    if attempt is not None:
        selected.update(
            {
                "artifact_sha256": attempt["artifact_sha256"],
                "artifact_filename": attempt["artifact_filename"],
                "provider_tag": attempt["provider_tag"],
                "endpoint_revision": attempt["endpoint_revision"],
                "attempt_index": attempt["attempt_index"],
            }
        )
    return selected


def _compatibility_class(best: dict[str, dict[str, Any]]) -> str:
    valid = [condition for condition in CONDITIONS if best[condition]["state"] in VALID_STATES]
    if len(valid) == 3:
        providers = {best[condition].get("provider_tag") for condition in valid}
        return (
            "all_conditions_one_endpoint"
            if len(providers) == 1
            else "all_conditions_across_endpoints"
        )
    if valid:
        return "partial_observed_compatibility"
    return "no_valid_normalized_condition"


def _model_record(
    model_id: str,
    artifacts: list[dict[str, Any]],
    manifest_model: dict[str, Any],
    corrected_costs: dict[str, int],
    correction_metadata: dict[str, list[dict[str, Any]]],
    accounted_generation_ids: set[str],
) -> dict[str, Any]:
    endpoint_revisions: list[dict[str, Any]] = []
    revision_by_tag: dict[str, int] = {}
    attempts: list[dict[str, Any]] = []

    for attempt_index, artifact in enumerate(
        sorted(
            artifacts,
            key=lambda value: (
                str(value.get("started_at")),
                str(value.get("run_id")),
            ),
        )
    ):
        provider_tag = str(artifact.get("requested_provider"))
        if provider_tag not in revision_by_tag:
            revision_by_tag[provider_tag] = len(revision_by_tag) + 1
            endpoint_revisions.append(
                {
                    "revision": revision_by_tag[provider_tag],
                    "provider_tag": provider_tag,
                    "first_observed_at": artifact.get("started_at"),
                    "candidate_manifest_sha256": artifact.get("candidate_manifest_sha256"),
                    "endpoint_contract_sha256": artifact.get("endpoint_contract_sha256"),
                }
            )
        digest = str(artifact["artifact_sha256"])
        conditions = {
            condition: _condition_observation(
                artifact,
                condition,
                accounted_generation_ids,
            )
            for condition in CONDITIONS
        }
        identities = _identity_records([artifact], correction_metadata)
        attempts.append(
            {
                "attempt_index": attempt_index + 1,
                "endpoint_revision": revision_by_tag[provider_tag],
                "provider_tag": provider_tag,
                "started_at": artifact.get("started_at"),
                "completed_at": artifact.get("completed_at"),
                "artifact_filename": artifact["_filename"],
                "artifact_sha256": digest,
                "artifact_status": artifact.get("status"),
                "candidate_manifest_sha256": artifact.get("candidate_manifest_sha256"),
                "endpoint_contract_sha256": artifact.get("endpoint_contract_sha256"),
                "corrected_exposure_micros": corrected_costs[digest],
                "actual_identities": identities,
                "conditions": conditions,
            }
        )

    best = {condition: _best_observation(attempts, condition) for condition in CONDITIONS}
    failure_counts: Counter[str] = Counter()
    rejected_statuses: Counter[str] = Counter()
    for attempt in attempts:
        for observation in attempt["conditions"].values():
            state = str(observation["state"])
            if state not in VALID_STATES and state != "not_attempted":
                failure_counts[state] += 1
            for status in observation.get("route_http_statuses", []):
                rejected_statuses[str(status)] += 1

    tool_calls = [
        call
        for attempt in attempts
        for observation in attempt["conditions"].values()
        for call in observation.get("tool_calls", [])
    ]
    valid_count = sum(best[condition]["state"] in VALID_STATES for condition in CONDITIONS)
    canonical = manifest_model.get("model", {}).get("canonical_slug")
    slot = manifest_model.get("slot", {})
    return {
        "display_name": DISPLAY_NAMES[model_id],
        "requested_model_id": model_id,
        "catalog_canonical_slug": canonical,
        "slot_id": slot.get("slot_id"),
        "cohort": slot.get("cohort"),
        "actual_identities": _identity_records(artifacts, correction_metadata),
        "endpoint_revisions": endpoint_revisions,
        "endpoint_attempts": attempts,
        "condition_best": best,
        "compatibility_class": _compatibility_class(best),
        "valid_condition_count": valid_count,
        "successful_epicure_tool_calls": sum(not call["is_error"] for call in tool_calls),
        "epicure_tool_errors": sum(call["is_error"] for call in tool_calls),
        "observed_exposure_micros": sum(
            corrected_costs[str(artifact["artifact_sha256"])] for artifact in artifacts
        ),
        "failure_event_counts": dict(sorted(failure_counts.items())),
        "route_rejection_http_status_counts": dict(sorted(rejected_statuses.items())),
    }


def _manifest_revisions(
    manifest_directory: Path,
    frontier_models: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    revisions: list[dict[str, Any]] = []
    model_metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(manifest_directory.glob("*.json")):
        manifest = _read_json(path)
        models = manifest.get("models") or []
        ids = {str(model.get("slot", {}).get("model_id")) for model in models}
        if ids != frontier_models:
            continue
        digest = str(manifest.get("content_address", {}).get("digest", ""))
        if len(digest) != 64:
            raise ValueError(f"manifest has no content address: {path}")
        revisions.append(
            {
                "manifest_sha256": digest,
                "filename": path.name,
                "observed_at": manifest.get("observed_at"),
                "endpoint_routes": [
                    {
                        "model_id": model.get("slot", {}).get("model_id"),
                        "canonical_slug": model.get("model", {}).get("canonical_slug"),
                        "provider_tag": model.get("endpoint", {}).get("tag"),
                        "provider_name": model.get("endpoint", {}).get("provider_name"),
                    }
                    for model in models
                ],
            }
        )
        for model in models:
            model_id = str(model.get("slot", {}).get("model_id"))
            model_metadata.setdefault(model_id, model)
    revisions.sort(key=lambda value: str(value.get("observed_at")))
    if set(model_metadata) != frontier_models:
        missing = sorted(frontier_models - set(model_metadata))
        raise ValueError(f"candidate manifests are missing frontier models: {missing}")
    return revisions, model_metadata


def _content_address(value: dict[str, Any], *, label: str) -> str:
    recorded = value.get("content_address", {}).get("digest")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(f"{label} is missing its content address")
    without_digest = dict(value)
    without_digest.pop("content_address", None)
    calculated = _sha256(without_digest)
    if calculated != recorded:
        raise ValueError(
            f"{label} content address mismatch: recorded={recorded}, calculated={calculated}"
        )
    return recorded


def _compact_decimal(value: Any) -> str:
    # Provider costs are reconciled to microdollars; admission forecasts in the
    # runner use one additional decimal place.  Trim arithmetic tail noise while
    # preserving that seventh decimal exactly.
    rendered = format(
        Decimal(str(value)).quantize(Decimal("0.0000001")),
        "f",
    )
    return rendered.rstrip("0").rstrip(".") or "0"


def _read_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError(f"frontier ledger line {line_number} is not an object")
        recorded = entry.get("entry_sha256")
        without_digest = dict(entry)
        without_digest.pop("entry_sha256", None)
        calculated = _sha256(without_digest)
        if recorded != calculated:
            raise ValueError(f"frontier ledger line {line_number} content address mismatch")
        if entry.get("previous_entry_sha256") != previous:
            raise ValueError(f"frontier ledger chain breaks at line {line_number}")
        previous = str(recorded)
        entries.append(entry)
    return entries


def _runner_summary(
    summary_directory: Path,
    ledger_path: Path,
) -> dict[str, Any] | None:
    summaries = []
    for path in summary_directory.glob("*.json"):
        summary = _read_json(path)
        _content_address(summary, label=f"frontier runner summary {path.name}")
        summaries.append(summary | {"_filename": path.name})
    if not summaries:
        return None
    summary = max(summaries, key=lambda value: str(value.get("completed_at")))
    ledger_entries = _read_ledger(ledger_path)
    ledger_head = ledger_entries[-1].get("entry_sha256") if ledger_entries else None
    reported_head = summary.get("ledger", {}).get("head_entry_sha256")
    reported_head_indexes = [
        index
        for index, entry in enumerate(ledger_entries)
        if entry.get("entry_sha256") == reported_head
    ]
    if len(reported_head_indexes) != 1:
        raise ValueError(
            "latest frontier runner summary head is not present exactly once in the ledger"
        )
    reported_head_index = reported_head_indexes[0]
    if summary.get("ledger", {}).get("entry_count") != reported_head_index + 1:
        raise ValueError("frontier runner summary ledger length does not match its head")
    no_artifact_outcomes = [
        outcome
        for outcome in summary.get("outcomes", [])
        if outcome.get("decision") == "no_artifact_reservation_retained"
    ]
    incident_entries = [
        entry
        for entry in ledger_entries
        if entry.get("event_type") == "execution_incident"
        and entry.get("incident") == "no_verifiable_artifact_reservation_retained"
    ]
    if len(no_artifact_outcomes) != len(incident_entries):
        raise ValueError("runner summary and execution ledger disagree on no-artifact incidents")
    resolution_entries = [
        entry
        for entry in ledger_entries
        if entry.get("event_type") == "no_artifact_reconciliation_recorded"
        and entry.get("decision") == "release_pre_generation_no_cost_reservation"
    ]
    reconciliation_directory = summary_directory.parent / "reconciliations"
    reconciliations: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(reconciliation_directory.glob("*.json")):
        reconciliation = _read_json(path)
        digest = _content_address(
            reconciliation,
            label=f"frontier no-artifact reconciliation {path.name}",
        )
        reconciliations[digest] = (path.name, reconciliation)

    incident_records: list[dict[str, Any]] = []
    for outcome, incident in zip(
        no_artifact_outcomes,
        incident_entries,
        strict=True,
    ):
        reservation_entry_sha256 = outcome.get("reservation_entry_sha256")
        matching_resolutions = [
            entry
            for entry in resolution_entries
            if entry.get("incident_entry_sha256") == incident.get("entry_sha256")
            and entry.get("reservation_entry_sha256") == reservation_entry_sha256
        ]
        if len(matching_resolutions) > 1:
            raise ValueError("a no-artifact incident has multiple release resolutions")
        reservation_usd = _compact_decimal(outcome.get("forecast", {}).get("forecast_usd", "0"))
        record: dict[str, Any] = {
            "reservation_entry_sha256": reservation_entry_sha256,
            "execution_incident_entry_sha256": incident.get("entry_sha256"),
            "reservation_usd": reservation_usd,
            "provider_call_evidenced": False,
            "artifact_evidenced": False,
            "actual_spend_assumed": False,
            "performance_attribution": False,
        }
        if not matching_resolutions:
            record.update(
                {
                    "incident": "no_verifiable_artifact_reservation_retained",
                    "status": "outstanding",
                    "outstanding_reservation_usd": reservation_usd,
                    "released_exposure_usd": "0",
                    "interpretation": (
                        "A subprocess ended without a verifiable artifact. The admission "
                        "reservation remains conservatively active, but is neither actual "
                        "spend nor evidence about any model's behavior."
                    ),
                }
            )
            incident_records.append(record)
            continue

        resolution = matching_resolutions[0]
        reconciliation_sha256 = str(resolution.get("reconciliation_sha256", ""))
        proof_entry = reconciliations.get(reconciliation_sha256)
        if proof_entry is None:
            raise ValueError("no-artifact resolution references an unknown proof artifact")
        proof_filename, proof = proof_entry
        conclusion = proof.get("conclusion", {})
        released_usd = _compact_decimal(resolution.get("released_exposure_usd", "0"))
        if released_usd != reservation_usd:
            raise ValueError("no-artifact resolution does not release the exact reservation")
        if (
            proof.get("provider_calls_made") is not False
            or conclusion.get("provider_generation_request_reached") is not False
            or _compact_decimal(conclusion.get("provider_generation_cost_usd", "0")) != "0"
            or conclusion.get("reservation_release_authorized") is not True
            or _compact_decimal(resolution.get("provider_generation_cost_usd", "0")) != "0"
        ):
            raise ValueError(
                "no-artifact proof does not establish a pre-provider zero-cost release"
            )
        record.update(
            {
                "incident": "reservation_resolved_no_provider_call",
                "status": "resolved",
                "outstanding_reservation_usd": "0",
                "released_exposure_usd": released_usd,
                "resolution_event_type": resolution.get("event_type"),
                "resolution_decision": resolution.get("decision"),
                "resolution_entry_sha256": resolution.get("entry_sha256"),
                "reconciliation_filename": proof_filename,
                "reconciliation_sha256": reconciliation_sha256,
                "provider_generation_request_reached": False,
                "provider_generation_cost_usd": "0",
                "interpretation": (
                    "Append-only proof bound the recorded stdout to a strict pre-provider "
                    "validation failure and reconciled the external account delta to the "
                    "preceding artifact. The conservative reservation was released with "
                    "zero provider-generation cost; this is not model-performance evidence."
                ),
            }
        )
        incident_records.append(record)

    reservations = {
        str(entry.get("entry_sha256")): Decimal(str(entry.get("reserved_usd", "0")))
        for entry in ledger_entries
        if entry.get("event_type") == "reservation_created"
    }
    resolved_reservations = {
        str(entry.get("reservation_entry_sha256"))
        for entry in ledger_entries
        if entry.get("event_type")
        in {
            "artifact_recorded",
            "no_artifact_reconciliation_recorded",
        }
        and entry.get("reservation_entry_sha256")
    }
    active_reservation = sum(
        (
            amount
            for reservation_sha256, amount in reservations.items()
            if reservation_sha256 not in resolved_reservations
        ),
        start=Decimal("0"),
    )

    budget = summary.get("budget", {})
    final_artifacts = budget.get("final_artifacts", {})
    actual_cost = Decimal(str(final_artifacts.get("actual_cost_usd", "0")))
    return {
        "filename": summary["_filename"],
        "summary_sha256": summary.get("content_address", {}).get("digest"),
        "completed_at": summary.get("completed_at"),
        "runner_run_id": summary.get("runner_run_id"),
        "reported_actual_cost_usd": _compact_decimal(final_artifacts.get("actual_cost_usd", "0")),
        "reported_artifact_count": final_artifacts.get("artifact_count"),
        "reported_failed_artifact_reserve_usd": _compact_decimal(
            final_artifacts.get("failed_or_unreconciled_reserve_usd", "0")
        ),
        "outstanding_conservative_reservation_usd": _compact_decimal(active_reservation),
        "conservative_total_exposure_usd": _compact_decimal(actual_cost + active_reservation),
        "ledger_entry_count": len(ledger_entries),
        "ledger_head_entry_sha256": ledger_head,
        "summary_attested_ledger_head_sha256": reported_head,
        "post_summary_ledger_entry_count": len(ledger_entries) - reported_head_index - 1,
        "no_artifact_incidents": incident_records,
        "outstanding_no_artifact_incident_count": sum(
            record["status"] == "outstanding" for record in incident_records
        ),
        "resolved_no_artifact_incident_count": sum(
            record["status"] == "resolved" for record in incident_records
        ),
    }


def _earlier_gemma_evidence(
    artifacts: list[dict[str, Any]],
    accounted_generation_ids: set[str],
    corrected_costs: dict[str, int],
) -> dict[str, Any]:
    gemma = [
        artifact
        for artifact in artifacts
        if str(artifact.get("requested_model_id", "")).startswith("google/gemma-4-26b-a4b-it")
    ]
    complete = [artifact for artifact in gemma if artifact.get("status") == "complete"]
    paired = next(
        artifact
        for artifact in complete
        if "epicure_off" in (artifact.get("results") or {})
        and "epicure_on" in (artifact.get("results") or {})
    )
    contract = next(
        artifact for artifact in complete if "tool_contract" in (artifact.get("results") or {})
    )
    selected = [paired, contract]
    conditions = {}
    for condition, artifact in (
        ("epicure_off", paired),
        ("epicure_on", paired),
        ("tool_contract", contract),
    ):
        conditions[condition] = _condition_observation(
            artifact,
            condition,
            accounted_generation_ids,
        )
    return {
        "label": "Earlier Gemma engineering evidence",
        "requested_model_id": paired.get("requested_model_id"),
        "actual_model_id": conditions["epicure_off"].get("actual_model_id"),
        "actual_provider": conditions["epicure_off"].get("actual_provider"),
        "artifact_count_all_attempts": len(gemma),
        "selected_complete_artifacts": [
            {
                "filename": artifact["_filename"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
            for artifact in selected
        ],
        "conditions": conditions,
        "selected_exposure_micros": sum(
            corrected_costs[str(artifact["artifact_sha256"])] for artifact in selected
        ),
        "interpretation": (
            "The automatic Epicure arm made no tool call. The explicit probe made two "
            "malformed calls that reached MCP and returned ingredient-resolution errors."
        ),
    }


def build_payload(evaluation_root: Path) -> dict[str, Any]:
    artifact_directory = evaluation_root / "flavourbench" / "artifacts" / "live-smoke"
    correction_directory = evaluation_root / "flavourbench" / "artifacts" / "corrections"
    manifest_directory = evaluation_root / "flavourbench" / "artifacts" / "manifests"
    summary_directory = (
        evaluation_root / "flavourbench" / "artifacts" / "frontier-contract" / "summaries"
    )
    ledger_path = (
        evaluation_root / "flavourbench" / "artifacts" / "frontier-contract" / "ledger.jsonl"
    )

    artifacts: list[dict[str, Any]] = []
    for path in sorted(artifact_directory.glob("*.json")):
        artifact = _read_json(path)
        _artifact_digest(artifact)
        artifact["_filename"] = path.name
        artifacts.append(artifact)
    if not artifacts:
        raise ValueError("no live-smoke artifacts found")

    corrected_costs = {
        str(artifact["artifact_sha256"]): int(
            artifact.get("budget", {}).get("actual_cost_micros", 0)
        )
        for artifact in artifacts
    }
    corrections: list[dict[str, Any]] = []
    correction_metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(correction_directory.glob("*.json")):
        correction = _read_json(path)
        source_digest = str(correction.get("source", {}).get("artifact_sha256"))
        if source_digest not in corrected_costs:
            raise ValueError(f"cost correction references an unknown artifact: {path}")
        corrected_costs[source_digest] = int(
            correction.get("cost", {}).get("corrected_total_cost_micros", 0)
        )
        correction_metadata[source_digest].extend(correction.get("generation_metadata") or [])
        corrections.append(
            {
                "filename": path.name,
                "correction_sha256": correction.get("artifact_sha256"),
                "source_artifact_sha256": source_digest,
                "corrected_total_cost_micros": corrected_costs[source_digest],
                "all_missing_generations_reconciled": correction.get(
                    "all_missing_generations_reconciled"
                ),
            }
        )

    response_generation_ids = {
        str(event["generation_id"])
        for artifact in artifacts
        for event in artifact.get("provider_attempt_events", [])
        if event.get("event_type") == "response_received" and event.get("generation_id")
    }
    accounted_generation_ids = {
        str(event["generation_id"])
        for artifact in artifacts
        for event in artifact.get("provider_attempt_events", [])
        if event.get("event_type") == "accounting_reconciled" and event.get("generation_id")
    }
    accounted_generation_ids.update(
        str(metadata["generation_id"])
        for values in correction_metadata.values()
        for metadata in values
        if metadata.get("generation_id")
    )
    missing_accounting = sorted(response_generation_ids - accounted_generation_ids)
    if missing_accounting:
        raise ValueError(f"unreconciled provider generations remain: {missing_accounting}")

    frontier_set = set(FRONTIER_MODEL_ORDER)
    manifest_revisions, manifest_models = _manifest_revisions(
        manifest_directory,
        frontier_set,
    )
    frontier_artifacts = [
        artifact for artifact in artifacts if artifact.get("requested_model_id") in frontier_set
    ]
    artifacts_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in frontier_artifacts:
        artifacts_by_model[str(artifact["requested_model_id"])].append(artifact)
    missing_models = [model for model in FRONTIER_MODEL_ORDER if not artifacts_by_model[model]]
    if missing_models:
        raise ValueError(f"frontier models have no real artifacts: {missing_models}")

    model_records = [
        _model_record(
            model_id,
            artifacts_by_model[model_id],
            manifest_models[model_id],
            corrected_costs,
            correction_metadata,
            accounted_generation_ids,
        )
        for model_id in FRONTIER_MODEL_ORDER
    ]

    frontier_response_ids = {
        str(event["generation_id"])
        for artifact in frontier_artifacts
        for event in artifact.get("provider_attempt_events", [])
        if event.get("event_type") == "response_received" and event.get("generation_id")
    }
    frontier_accounted = frontier_response_ids & accounted_generation_ids
    frontier_request_count = sum(
        event.get("event_type") == "request_started"
        for artifact in frontier_artifacts
        for event in artifact.get("provider_attempt_events", [])
    )
    frontier_route_rejections = [
        event
        for artifact in frontier_artifacts
        for event in artifact.get("provider_attempt_events", [])
        if event.get("event_type") == "request_rejected"
    ]
    failure_totals: Counter[str] = Counter()
    for model in model_records:
        failure_totals.update(model["failure_event_counts"])
    compatibility_totals = Counter(str(model["compatibility_class"]) for model in model_records)
    frontier_exposure_micros = sum(
        corrected_costs[str(artifact["artifact_sha256"])] for artifact in frontier_artifacts
    )
    all_exposure_micros = sum(corrected_costs.values())
    latest_runner = _runner_summary(summary_directory, ledger_path)
    runner_matches = False
    if latest_runner and latest_runner.get("reported_actual_cost_usd") is not None:
        runner_matches = (
            int(Decimal(latest_runner["reported_actual_cost_usd"]) * Decimal(1_000_000))
            == all_exposure_micros
        )
    outstanding_reservation_usd = (
        latest_runner.get("outstanding_conservative_reservation_usd", "0") if latest_runner else "0"
    )
    conservative_total_exposure_usd = (
        latest_runner.get("conservative_total_exposure_usd") if latest_runner else None
    )
    governance_incidents = latest_runner.get("no_artifact_incidents", []) if latest_runner else []
    outstanding_incident_count = sum(
        incident.get("status") == "outstanding" for incident in governance_incidents
    )
    resolved_incident_count = sum(
        incident.get("status") == "resolved" for incident in governance_incidents
    )
    released_reservation_usd = _compact_decimal(
        sum(
            (
                Decimal(str(incident.get("released_exposure_usd", "0")))
                for incident in governance_incidents
            ),
            start=Decimal("0"),
        )
    )

    provenance_values = {
        (
            artifact.get("epicure", {}).get("release_id"),
            artifact.get("epicure", {}).get("bundle_sha256"),
            artifact.get("epicure", {}).get("application_sha256"),
            artifact.get("epicure_tool_schema_sha256"),
            artifact.get("epicure", {}).get("ingredient_count"),
            artifact.get("epicure", {}).get("embedding_dimensions"),
        )
        for artifact in artifacts
    }
    if len(provenance_values) != 1:
        raise ValueError("live-smoke artifacts do not share one Epicure provenance tuple")
    release_id, bundle, application, tool_schema, ingredients, dimensions = next(
        iter(provenance_values)
    )

    observed_through = max(str(artifact.get("completed_at")) for artifact in artifacts)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "exploratory_unranked_contract_evidence",
        "observed_through": observed_through,
        "route": "OpenRouter via Cloudflare AI Gateway to private Epicure MCP",
        "scientific_status": {
            "rank_eligible": False,
            "official": False,
            "blinded_public_judgments": 0,
            "blinded_expert_judgments": 0,
            "bradley_terry_ratings": 0,
            "epicure_uplift_estimates": 0,
            "interpretation": (
                "Endpoint and tool-contract engineering evidence only; it supports no culinary "
                "quality ordering and no Epicure-uplift claim."
            ),
        },
        "budget_reconciliation": {
            "authorised_hard_cap_usd": "100.000000",
            "verified_exposure_micros": all_exposure_micros,
            "verified_exposure_usd": f"{all_exposure_micros / 1_000_000:.6f}",
            "frontier_panel_exposure_micros": frontier_exposure_micros,
            "frontier_panel_exposure_usd": f"{frontier_exposure_micros / 1_000_000:.6f}",
            "artifact_count": len(artifacts),
            "cost_correction_count": len(corrections),
            "provider_generation_count": len(response_generation_ids),
            "reconciled_provider_generation_count": len(
                response_generation_ids & accounted_generation_ids
            ),
            "all_provider_generations_reconciled": not missing_accounting,
            "latest_runner_summary_matches_independent_scan": runner_matches,
            "outstanding_conservative_reservation_usd": outstanding_reservation_usd,
            "conservative_total_exposure_usd": conservative_total_exposure_usd,
            "no_artifact_incident_count": len(governance_incidents),
            "outstanding_no_artifact_incident_count": outstanding_incident_count,
            "resolved_no_artifact_incident_count": resolved_incident_count,
            "released_pre_generation_reservation_usd": released_reservation_usd,
            "reservation_interpretation": (
                "Append-only reconciliation proved the no-artifact subprocess failed "
                "before provider instantiation at zero provider-generation cost. Its "
                "conservative reservation is released and is not model-performance evidence."
                if resolved_incident_count and not outstanding_incident_count
                else "An unresolved no-artifact subprocess retains a conservative cap hold."
            ),
        },
        "frontier_summary": {
            "model_count": len(model_records),
            "artifact_attempt_count": len(frontier_artifacts),
            "endpoint_request_artifact_count": sum(
                any(
                    event.get("event_type") == "request_started"
                    for event in artifact.get("provider_attempt_events", [])
                )
                for artifact in frontier_artifacts
            ),
            "endpoint_revision_count": sum(
                len(model["endpoint_revisions"]) for model in model_records
            ),
            "request_attempt_count": frontier_request_count,
            "provider_generation_count": len(frontier_response_ids),
            "reconciled_provider_generation_count": len(frontier_accounted),
            "normalized_condition_result_count": sum(
                sum(
                    attempt["conditions"][condition]["state"] in VALID_STATES
                    for condition in CONDITIONS
                )
                for model in model_records
                for attempt in model["endpoint_attempts"]
            ),
            "successful_epicure_tool_call_count": sum(
                int(model["successful_epicure_tool_calls"]) for model in model_records
            ),
            "epicure_tool_error_count": sum(
                int(model["epicure_tool_errors"]) for model in model_records
            ),
            "models_with_successful_epicure_tool_call": sum(
                int(model["successful_epicure_tool_calls"]) > 0 for model in model_records
            ),
            "pre_generation_route_rejection_count": len(frontier_route_rejections),
            "route_rejection_http_status_counts": dict(
                sorted(
                    Counter(
                        str(event.get("http_status")) for event in frontier_route_rejections
                    ).items()
                )
            ),
            "failure_condition_counts": dict(sorted(failure_totals.items())),
            "compatibility_class_counts": dict(sorted(compatibility_totals.items())),
        },
        "epicure_provenance": {
            "release_id": release_id,
            "release_status": "exploratory_unmatched_bundle",
            "bundle_sha256": bundle,
            "application_sha256": application,
            "tool_schema_sha256": tool_schema,
            "ingredient_count": ingredients,
            "embedding_dimensions": dimensions,
            "official_public_release_match": False,
        },
        "models": model_records,
        "earlier_engineering_evidence": _earlier_gemma_evidence(
            artifacts,
            accounted_generation_ids,
            corrected_costs,
        ),
        "source": {
            "manifest_revisions": manifest_revisions,
            "live_smoke_artifacts": [
                {
                    "filename": artifact["_filename"],
                    "artifact_sha256": artifact["artifact_sha256"],
                    "requested_model_id": artifact.get("requested_model_id"),
                    "requested_provider": artifact.get("requested_provider"),
                }
                for artifact in artifacts
            ],
            "cost_corrections": corrections,
            "latest_frontier_runner_summary": latest_runner,
        },
        "governance_incidents": governance_incidents,
        "limitations": [
            "The panel has zero blinded public or expert judgments and therefore no ranking.",
            "A valid contract response or tool call is not evidence of culinary quality.",
            "Historical parallel-call rejections reflect the then-current client limit; "
            "the engine later adopted bounded parallel Epicure calls.",
            "The Epicure bundle is unmatched to the published Cooc, Core, and Chem releases.",
            "Endpoint revisions were exploratory and are not a frozen Season 0 manifest.",
            "Costs include valid generations used by failed normalized conditions and "
            "pre-ranking diagnostics.",
            "One no-artifact runner incident was resolved by append-only proof of a "
            "pre-provider zero-cost failure; the released reservation is not model evidence.",
        ],
    }
    return payload


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _tex_state(state: str) -> str:
    values = {
        "valid_normalized": ("ValidBG", "valid"),
        "valid_with_epicure_tool": ("ToolBG", "tool"),
        "valid_no_tool_use": ("NeutralBG", "valid/no call"),
        "valid_with_tool_error": ("WarnBG", "tool error"),
        "client_parallel_policy_rejection": ("WarnBG", "prior policy"),
        "invalid_structured_output": ("FailBG", "invalid JSON"),
        "route_rejected": ("RouteBG", "route reject"),
        "mcp_transport_error": ("RouteBG", "MCP error"),
        "provider_error": ("FailBG", "provider error"),
        "local_preflight_rejection": ("NeutralBG", "local preflight"),
        "not_attempted": ("NeutralBG", "not run"),
    }
    color, label = values[state]
    return f"\\cellcolor{{{color}}}\\strut {_tex_escape(label)}"


def render_tex(payload: dict[str, Any]) -> str:
    rows = []
    for model in payload["models"]:
        endpoints = " $\\rightarrow$ ".join(
            f"\\texttt{{{_tex_escape(revision['provider_tag'])}}}"
            for revision in model["endpoint_revisions"]
        )
        identity = (
            model["actual_identities"][0]["actual_model_id"]
            if model["actual_identities"]
            else "unresolved"
        )
        row = (
            f"{_tex_escape(model['display_name'])}\\newline"
            f"{{\\tiny\\nolinkurl{{{identity}}}}} & "
            f"{endpoints} & "
            f"{_tex_state(model['condition_best']['epicure_off']['state'])} & "
            f"{_tex_state(model['condition_best']['epicure_on']['state'])} & "
            f"{_tex_state(model['condition_best']['tool_contract']['state'])} & "
            f"\\${model['observed_exposure_micros'] / 1_000_000:.6f} \\\\"
        )
        rows.append(row)
    return "\n".join(
        [
            "% Generated by scripts/aggregate_frontier_evidence.py; do not edit by hand.",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Exploratory frontier contract matrix through 15 July 2026. "
            "``Tool'' means a normalized response with at least one successful real "
            "Epicure call; ``valid'' means only that the response normalized. "
            "Prior-policy cells are historical client rejections of multiple tool "
            "calls, not model-quality failures. Endpoint arrows show tested route "
            "revisions. Exposure includes failed diagnostics. There are zero blinded "
            "judgments; no row is ranked.}",
            "\\label{tab:frontier-contract}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3.2pt}",
            "\\begin{tabularx}{\\textwidth}{@{}p{0.19\\textwidth}Xccc r@{}}",
            "\\toprule",
            "Model and resolved identity & Endpoint attempt sequence & Epicure off & "
            "Epicure available & Tool contract & Exposure \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabularx}",
            "\\end{table*}",
            "",
        ]
    )


def render_tex_macros(payload: dict[str, Any]) -> str:
    summary = payload["frontier_summary"]
    budget = payload["budget_reconciliation"]
    failures = summary["failure_condition_counts"]
    route_statuses = summary["route_rejection_http_status_counts"]
    compatibility = summary["compatibility_class_counts"]
    values = {
        "frontierModelCount": summary["model_count"],
        "frontierArtifactAttempts": summary["artifact_attempt_count"],
        "frontierEndpointRequestArtifacts": summary["endpoint_request_artifact_count"],
        "frontierEndpointRevisions": summary["endpoint_revision_count"],
        "frontierRequestAttempts": summary["request_attempt_count"],
        "frontierProviderGenerations": summary["provider_generation_count"],
        "frontierReconciledGenerations": summary["reconciled_provider_generation_count"],
        "allProviderGenerations": budget["provider_generation_count"],
        "allReconciledGenerations": budget["reconciled_provider_generation_count"],
        "frontierSuccessfulToolCalls": summary["successful_epicure_tool_call_count"],
        "frontierModelsWithToolCalls": summary["models_with_successful_epicure_tool_call"],
        "frontierInvalidJSONEvents": failures.get("invalid_structured_output", 0),
        "frontierParallelPolicyEvents": failures.get("client_parallel_policy_rejection", 0),
        "frontierLocalPreflightEvents": failures.get("local_preflight_rejection", 0),
        "frontierRouteRejections": summary["pre_generation_route_rejection_count"],
        "frontierRouteHTTPFourHundred": route_statuses.get("400", 0),
        "frontierRouteHTTPFourOhFour": route_statuses.get("404", 0),
        "frontierRouteHTTPFourTwentyNine": route_statuses.get("429", 0),
        "frontierAllConditionsOneEndpoint": compatibility.get("all_conditions_one_endpoint", 0),
        "frontierAllConditionsAcrossEndpoints": compatibility.get(
            "all_conditions_across_endpoints", 0
        ),
        "frontierCostCorrections": budget["cost_correction_count"],
        "verifiedExposureUSD": budget["verified_exposure_usd"],
        "outstandingReservationUSD": budget["outstanding_conservative_reservation_usd"],
        "conservativeTotalExposureUSD": budget["conservative_total_exposure_usd"],
        "noArtifactIncidentCount": budget["no_artifact_incident_count"],
        "outstandingNoArtifactIncidentCount": budget["outstanding_no_artifact_incident_count"],
        "resolvedNoArtifactIncidentCount": budget["resolved_no_artifact_incident_count"],
        "releasedReservationUSD": budget["released_pre_generation_reservation_usd"],
        "frontierAggregateDigest": payload["content_address"]["digest"],
    }
    lines = ["% Generated by scripts/aggregate_frontier_evidence.py; do not edit by hand."]
    lines.extend(f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in values.items())
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    evaluation_root = Path(__file__).resolve().parents[2]
    workspace_root = evaluation_root.parents[1] / "epicure"
    paper_root = evaluation_root / "paper" / "flavourbench"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=evaluation_root)
    parser.add_argument(
        "--aggregate-directory",
        type=Path,
        default=evaluation_root / "flavourbench" / "artifacts" / "aggregates",
    )
    parser.add_argument(
        "--paper-json",
        type=Path,
        default=paper_root / "data" / "frontier_contract_evidence.json",
    )
    parser.add_argument(
        "--paper-tex",
        type=Path,
        default=paper_root / "figures" / "frontier_contract_matrix.tex",
    )
    parser.add_argument(
        "--paper-macros",
        type=Path,
        default=paper_root / "figures" / "frontier_contract_macros.tex",
    )
    parser.add_argument(
        "--web-json",
        type=Path,
        default=workspace_root / "epicure-webapp" / "lib" / "flavourbench-frontier-evidence.json",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = build_payload(args.evaluation_root.resolve())
    digest = _sha256(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.aggregate_directory.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.aggregate_directory / f"frontier-contract-evidence-{digest}.json"
    aggregate_path.write_bytes(encoded)

    for destination in (args.paper_json, args.web_json):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
    args.paper_tex.parent.mkdir(parents=True, exist_ok=True)
    args.paper_tex.write_text(render_tex(payload), encoding="utf-8")
    args.paper_macros.parent.mkdir(parents=True, exist_ok=True)
    args.paper_macros.write_text(render_tex_macros(payload), encoding="utf-8")

    # A byte-for-byte postcondition makes the two product surfaces auditable.
    for destination in (args.paper_json, args.web_json):
        if destination.read_bytes() != aggregate_path.read_bytes():
            raise RuntimeError(f"generated evidence copy drifted: {destination}")

    print(
        json.dumps(
            {
                "aggregate": str(aggregate_path),
                "content_address": digest,
                "models": payload["frontier_summary"]["model_count"],
                "verified_exposure_usd": payload["budget_reconciliation"]["verified_exposure_usd"],
                "provider_generations": payload["budget_reconciliation"][
                    "provider_generation_count"
                ],
                "all_provider_generations_reconciled": payload["budget_reconciliation"][
                    "all_provider_generations_reconciled"
                ],
                "rank_eligible": False,
                "blinded_judgments": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
