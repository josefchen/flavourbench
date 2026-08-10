"""Read-only post-run verification for the real FlavourBench exploration.

The verifier deliberately has no provider or MCP client dependency.  It reads
the immutable manifest, summaries, source runs, response artifacts, journals,
and budget ledgers produced by :mod:`flavourbench.real_dataset_runner` and
checks that they form one exact, fully accounted record graph.

Two modes are supported:

``strict_final``
    Requires all 120 planned pairs to have a finalized source record, no live
    or orphaned reservation, no in-progress journal, and a content-addressed
    execute summary that exactly describes the final state.

``allow_incomplete``
    Applies the same integrity, identity, provenance, and accounting checks,
    but reports unfinished workload and active reservations as warnings.  This
    is useful for observing a paid sequential run without touching its locks or
    making any network calls.

Partial or failed model pairs are not silently discarded.  They are accepted
only when the immutable source proves every attempt is cost-accounted; they
remain visible as reliability outcomes and never become preference data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .execution_policy import ExecutionPolicy, verify_policy_document
from .frontier_contract_runner import (
    IntegrityError,
    load_candidate_manifest,
    select_candidates,
)
from .real_dataset_runner import (
    CONDITIONS,
    SOURCE_INCIDENT_RESOLUTION_EVENT_TYPE,
    SUMMARY_SCHEMA_VERSION,
    TASK_FAMILIES,
    DatasetState,
    WorkItem,
    _load_state,
    _summary_coverage,
    _validate_state_against_workload,
    build_balanced_work_items,
    load_dataset_ledger,
    select_balanced_tasks,
)
from .run_journal import JournalIntegrityError, scan_recovery_journals

VERIFIER_SCHEMA_VERSION = "flavourbench-real-exploratory-verification-v1"
EXPECTED_MODEL_COUNT = 12
EXPECTED_ASSIGNMENTS_PER_MODEL = 10
EXPECTED_PAIR_COUNT = 120
EXPECTED_RESPONSE_OPPORTUNITIES = 240
EXPECTED_TASK_POOL_PER_FAMILY = 3
EXPECTED_SELECTED_TASK_COUNT = 12
EXPECTED_FAMILY_PAIR_COUNT = 30
EXPLORATORY_EPICURE_RELEASE_ID = "exploratory-unmatched-1790-runtime"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} is not a decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{field} must be a non-negative finite decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IntegrityError(f"could not read JSON input: {path}") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON input is not an object: {path}")
    return value


def verify_summary_content_address(summary: Mapping[str, Any], path: Path | None = None) -> bool:
    """Return whether a summary and, optionally, its filename are content addressed."""

    address = summary.get("content_address")
    if not isinstance(address, Mapping):
        return False
    digest = address.get("digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or address.get("algorithm") != "sha256"
        or address.get("uri") != f"sha256:{digest}"
    ):
        return False
    unhashed = dict(summary)
    unhashed.pop("content_address", None)
    if _sha256(unhashed) != digest:
        return False
    return path is None or path.name == f"real-exploratory-summary-{digest}.json"


@dataclass(frozen=True)
class ExpectedPair:
    ordinal: int
    work_item_id: str
    manifest_sha256: str
    task_registry_sha256: str
    task_id: str
    task_family: str
    prompt: str
    prompt_sha256: str
    task_split: str
    task_review_status: str
    slot_id: str
    model_id: str
    canonical_model_slug: str
    provider_tag: str
    provider_name: str
    endpoint_manifest_sha256: str
    endpoint_execution_sha256: str
    execution_policy_sha256: str

    @classmethod
    def from_work_item(cls, item: WorkItem) -> ExpectedPair:
        return cls(
            ordinal=item.ordinal,
            work_item_id=item.work_item_id,
            manifest_sha256=item.manifest_sha256,
            task_registry_sha256=item.task_registry_sha256,
            task_id=item.task.public_id,
            task_family=item.task.family,
            prompt=item.task.prompt,
            prompt_sha256=item.task.prompt_sha256,
            task_split=item.task.split,
            task_review_status=item.task.review_status,
            slot_id=item.candidate.slot_id,
            model_id=item.candidate.model_id,
            canonical_model_slug=item.candidate.canonical_model_slug,
            provider_tag=item.candidate.provider_tag,
            provider_name=item.candidate.provider_name,
            endpoint_manifest_sha256=item.candidate.endpoint_sha256,
            endpoint_execution_sha256=item.endpoint_execution_sha256,
            execution_policy_sha256=item.execution_policy_sha256,
        )


@dataclass(frozen=True)
class Finding:
    check_id: str
    status: str
    severity: str
    message: str
    expected: Any | None = None
    observed: Any | None = None
    examples: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["examples"] = list(self.examples)
        return {key: item for key, item in value.items() if item is not None and item != ()}


class Audit:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(
        self,
        check_id: str,
        status: str,
        severity: str,
        message: str,
        *,
        expected: Any | None = None,
        observed: Any | None = None,
        examples: Sequence[str] = (),
    ) -> None:
        if status not in {"pass", "warn", "fail"}:
            raise ValueError(f"unsupported finding status: {status}")
        self.findings.append(
            Finding(
                check_id=check_id,
                status=status,
                severity=severity,
                message=message,
                expected=expected,
                observed=observed,
                examples=tuple(examples[:10]),
            )
        )

    @property
    def failures(self) -> list[Finding]:
        return [item for item in self.findings if item.status == "fail"]

    @property
    def warnings(self) -> list[Finding]:
        return [item for item in self.findings if item.status == "warn"]


def _policy_from_document(document: object) -> ExecutionPolicy:
    if not verify_policy_document(document) or not isinstance(document, Mapping):
        raise IntegrityError("summary has no valid content-addressed execution policy")
    limits = document.get("limits")
    decoding = document.get("decoding")
    forecast = document.get("cost_forecast")
    if not all(isinstance(value, Mapping) for value in (limits, decoding, forecast)):
        raise IntegrityError("execution policy lacks limits/decoding/cost forecast")
    try:
        policy = ExecutionPolicy(
            max_output_tokens=int(limits["max_output_tokens"]),
            max_tool_rounds=int(limits["max_tool_rounds"]),
            max_tool_result_bytes=int(limits["max_tool_result_bytes"]),
            max_cumulative_tool_result_bytes=int(limits["max_cumulative_tool_result_bytes"]),
            max_tool_calls_per_round=int(limits["max_tool_calls_per_round"]),
            max_tool_calls_total=int(limits["max_tool_calls_total"]),
            max_provider_attempts=int(limits["max_provider_attempts"]),
            tool_argument_repair_turns=int(limits["tool_argument_repair_turns"]),
            decoding_temperature=float(decoding["temperature"]),
            decoding_top_p=float(decoding["top_p"]),
            decoding_seed=int(decoding["seed"]),
            approximate_non_user_prompt_bytes=int(forecast["approximate_non_user_prompt_bytes"]),
            conservative_bytes_per_token=int(forecast["conservative_bytes_per_token"]),
            pair_arm_scheduling=str(document["pair_arm_scheduling"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("execution policy contains invalid fields") from error
    policy.validate()
    if policy.document() != document:
        raise IntegrityError("execution policy cannot be exactly reconstructed")
    return policy


def _summary_documents(
    summary_directory: Path,
    *,
    explicit_summary: Path | None,
    strict_final: bool,
    audit: Audit,
) -> tuple[dict[str, Any] | None, Path | None, list[tuple[Path, dict[str, Any]]]]:
    paths = sorted(summary_directory.glob("real-exploratory-summary-*.json"))
    if explicit_summary is not None and explicit_summary not in paths:
        paths.append(explicit_summary)
    documents: list[tuple[Path, dict[str, Any]]] = []
    invalid: list[str] = []
    seen_digests: set[str] = set()
    seen_runs: set[str] = set()
    for path in paths:
        try:
            document = _read_json(path)
        except IntegrityError as error:
            invalid.append(str(error))
            continue
        if document.get(
            "schema_version"
        ) != SUMMARY_SCHEMA_VERSION or not verify_summary_content_address(document, path):
            invalid.append(path.name)
            continue
        digest = str((document.get("content_address") or {}).get("digest") or "")
        run_id = str(document.get("runner_run_id") or "")
        if digest in seen_digests or not run_id or run_id in seen_runs:
            invalid.append(f"duplicate digest/run identity: {path.name}")
            continue
        seen_digests.add(digest)
        seen_runs.add(run_id)
        documents.append((path, document))
    audit.add(
        "summary_content_addresses",
        "fail" if invalid else "pass",
        "critical",
        (
            "One or more summary records fail schema, uniqueness, or content-address checks."
            if invalid
            else "Every discovered summary is immutable, unique, and content addressed."
        ),
        observed={"valid": len(documents), "invalid": len(invalid)},
        examples=invalid,
    )
    if explicit_summary is not None:
        selected = next((item for item in documents if item[0] == explicit_summary), None)
    else:
        candidates = [item for item in documents if item[1].get("mode") == "execute"]
        if not candidates and not strict_final:
            candidates = documents
        selected = max(
            candidates, key=lambda item: str(item[1].get("completed_at") or ""), default=None
        )
    if selected is None:
        fallback = max(
            documents,
            key=lambda item: str(item[1].get("completed_at") or ""),
            default=None,
        )
        if strict_final:
            audit.add(
                "final_execute_summary",
                "fail",
                "high",
                "No valid execute-mode summary exists for a strict final verification.",
                expected="content-addressed mode=execute summary",
                observed="none",
            )
        else:
            audit.add(
                "final_execute_summary",
                "warn",
                "info",
                "The run has not emitted a final execute summary yet.",
                expected="after collection completes",
                observed="dry-run workload summary only" if fallback else "none",
            )
        selected = fallback
    else:
        mode = selected[1].get("mode")
        status = "pass" if mode == "execute" else ("fail" if strict_final else "warn")
        audit.add(
            "final_execute_summary",
            status,
            "high" if status == "fail" else "info",
            (
                "A final execute-mode summary is available."
                if mode == "execute"
                else "Using a dry-run summary only to reconstruct the in-progress workload."
            ),
            expected="execute" if strict_final else "execute when collection completes",
            observed=mode,
        )
    return (
        selected[1] if selected else None,
        selected[0] if selected else None,
        documents,
    )


def _expected_workload(
    *,
    manifest_path: Path,
    summary: Mapping[str, Any],
    audit: Audit,
) -> tuple[list[WorkItem], ExecutionPolicy]:
    summary_manifest = summary.get("manifest")
    selection = summary.get("task_selection")
    workload = summary.get("workload")
    if not all(isinstance(value, Mapping) for value in (summary_manifest, selection, workload)):
        raise IntegrityError("summary lacks manifest/task-selection/workload objects")
    manifest_digest = str(summary_manifest.get("sha256") or "")
    manifest = load_candidate_manifest(manifest_path, expected_digest=manifest_digest)
    model_rows = summary_manifest.get("models")
    if not isinstance(model_rows, list) or not model_rows:
        raise IntegrityError("summary manifest has no selected model records")
    selectors = [
        str(item.get("model_id") or "") for item in model_rows if isinstance(item, Mapping)
    ]
    if len(selectors) != len(model_rows) or any(not value for value in selectors):
        raise IntegrityError("summary selected model records are malformed")
    candidates = select_candidates(manifest, selectors)
    try:
        task_pool_per_family = int(selection["task_pool_per_family"])
        assignments_per_model = int(selection["assignments_per_model"])
        seed = str(selection["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("summary task-selection settings are malformed") from error
    selected_tasks, registry_sha = select_balanced_tasks(
        tasks_per_family=task_pool_per_family,
        seed=seed,
    )
    if selection.get("candidate_registry_sha256") != registry_sha:
        raise IntegrityError("summary task registry digest differs from the current registry")
    policy = _policy_from_document(summary.get("execution_policy"))
    if summary.get("execution_policy_sha256") != policy.sha256:
        raise IntegrityError("summary execution-policy digest mismatch")
    work_items = build_balanced_work_items(
        manifest_sha256=manifest_digest,
        task_registry_digest=registry_sha,
        selected_tasks=selected_tasks,
        candidates=candidates,
        execution_policy=policy,
        assignments_per_model=assignments_per_model,
    )
    expected_public = [item.public_payload() for item in work_items]
    summary_public = workload.get("work_items")
    exact_summary = summary_public == expected_public
    audit.add(
        "frozen_workload_reconstruction",
        "pass" if exact_summary else "fail",
        "critical",
        (
            "The manifest, candidate registry, policy, and schedule reconstruct exactly."
            if exact_summary
            else "The summary workload does not reconstruct from its frozen inputs."
        ),
        expected={"work_items": len(expected_public)},
        observed={
            "work_items": len(summary_public) if isinstance(summary_public, list) else None,
            "exact_match": exact_summary,
        },
    )
    family_counts = Counter(item.task.family for item in work_items)
    model_counts = Counter(item.candidate.model_id for item in work_items)
    envelope_issues: list[str] = []
    if len(candidates) != EXPECTED_MODEL_COUNT:
        envelope_issues.append(f"model_count={len(candidates)}")
    if task_pool_per_family != EXPECTED_TASK_POOL_PER_FAMILY:
        envelope_issues.append(f"task_pool_per_family={task_pool_per_family}")
    if assignments_per_model != EXPECTED_ASSIGNMENTS_PER_MODEL:
        envelope_issues.append(f"assignments_per_model={assignments_per_model}")
    if len(selected_tasks) != EXPECTED_SELECTED_TASK_COUNT:
        envelope_issues.append(f"selected_task_count={len(selected_tasks)}")
    if len(work_items) != EXPECTED_PAIR_COUNT:
        envelope_issues.append(f"pair_count={len(work_items)}")
    if set(family_counts) != set(TASK_FAMILIES) or any(
        family_counts[family] != EXPECTED_FAMILY_PAIR_COUNT for family in TASK_FAMILIES
    ):
        envelope_issues.append(f"family_counts={dict(family_counts)}")
    if set(model_counts.values()) != {EXPECTED_ASSIGNMENTS_PER_MODEL}:
        envelope_issues.append(f"model_counts={dict(model_counts)}")
    audit.add(
        "benchmark_workload_envelope",
        "fail" if envelope_issues else "pass",
        "critical",
        (
            "The frozen workload is exactly 12 models × 10 paired tasks, balanced 30 per family."
            if not envelope_issues
            else "The frozen workload differs from the authorized 120-pair design."
        ),
        expected={
            "models": EXPECTED_MODEL_COUNT,
            "pairs": EXPECTED_PAIR_COUNT,
            "response_opportunities": EXPECTED_RESPONSE_OPPORTUNITIES,
            "pairs_per_family": EXPECTED_FAMILY_PAIR_COUNT,
        },
        observed={
            "models": len(candidates),
            "pairs": len(work_items),
            "response_opportunities": len(work_items) * len(CONDITIONS),
            "pairs_per_family": dict(family_counts),
        },
        examples=envelope_issues,
    )
    return work_items, policy


def _ledger_graph(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    list[Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    list[str],
]:
    reservations: dict[str, Mapping[str, Any]] = {}
    reservation_digests: dict[str, str] = {}
    finalizations: dict[str, Mapping[str, Any]] = {}
    incidents: list[Mapping[str, Any]] = []
    incidents_by_digest: dict[str, Mapping[str, Any]] = {}
    resolutions: dict[str, Mapping[str, Any]] = {}
    issues: list[str] = []
    for entry in entries:
        event_type = entry.get("event_type")
        work_item_id = str(entry.get("work_item_id") or "")
        if event_type == "reservation_created":
            if not work_item_id or work_item_id in reservations:
                issues.append(f"duplicate/absent reservation work item: {work_item_id}")
                continue
            reservations[work_item_id] = entry
            digest = str(entry.get("entry_sha256") or "")
            if digest:
                reservation_digests[digest] = work_item_id
        elif event_type == "source_artifact_recorded":
            if not work_item_id or work_item_id in finalizations:
                issues.append(f"duplicate/absent finalization work item: {work_item_id}")
                continue
            reservation_digest = str(entry.get("reservation_entry_sha256") or "")
            if reservation_digests.get(reservation_digest) != work_item_id:
                issues.append(f"invalid finalization reservation link: {work_item_id}")
            finalizations[work_item_id] = entry
        elif event_type == "execution_incident":
            incidents.append(entry)
            digest = str(entry.get("entry_sha256") or "")
            if digest:
                incidents_by_digest[digest] = entry
        elif event_type == SOURCE_INCIDENT_RESOLUTION_EVENT_TYPE:
            if not work_item_id or work_item_id in resolutions:
                issues.append(f"duplicate/absent incident-resolution work item: {work_item_id}")
                continue
            reservation_digest = str(entry.get("reservation_entry_sha256") or "")
            incident_digest = str(entry.get("incident_entry_sha256") or "")
            incident = incidents_by_digest.get(incident_digest)
            if reservation_digests.get(reservation_digest) != work_item_id:
                issues.append(f"invalid incident-resolution reservation link: {work_item_id}")
            if (
                incident is None
                or incident.get("work_item_id") != work_item_id
                or incident.get("reservation_entry_sha256") != reservation_digest
                or incident.get("incident") != "generation_cost_unreconciled_reservation_retained"
            ):
                issues.append(f"invalid incident-resolution incident link: {work_item_id}")
            if (
                entry.get("provider_cost_exact_for_unidentified_response") is not False
                or entry.get("safe_to_replay") is not False
                or entry.get("normalizable_conditions") != ["epicure_on"]
            ):
                issues.append(f"unsafe incident-resolution policy: {work_item_id}")
            resolutions[work_item_id] = entry
        else:
            issues.append(f"unsupported ledger event type: {event_type}")
    return reservations, finalizations, incidents, resolutions, issues


def _trace_without_arm_id(events: object) -> list[dict[str, Any]] | None:
    if not isinstance(events, list):
        return None
    normalized: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            return None
        value = dict(event)
        value.pop("arm_id", None)
        normalized.append(value)
    return normalized


def _find_forbidden_keys(value: object, *, prefix: str = "") -> list[str]:
    forbidden = {
        "api_key",
        "authorization",
        "client_ip",
        "cloudflare_ai_gateway_token",
        "cookie",
        "environment",
        "headers",
        "ip_address",
        "mcp_token",
        "openrouter_api_key",
        "password",
        "raw_ip",
        "raw_request",
        "request_payload",
        "response_body",
        "secret",
    }
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in forbidden:
                found.append(path)
            found.extend(_find_forbidden_keys(item, prefix=path))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, prefix=f"{prefix}[{index}]"))
    return found


def audit_record_graph(
    *,
    expected_pairs: Mapping[str, ExpectedPair],
    ledger: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    strict_final: bool,
    max_tool_rounds: int,
    max_tool_calls_total: int,
    source_actual_costs_usd: Mapping[str, Decimal] | None = None,
) -> tuple[list[Finding], dict[str, Any]]:
    """Audit an already parsed record graph; useful for both the CLI and tests."""

    audit = Audit()
    (
        reservations,
        finalizations,
        incidents,
        incident_resolutions,
        ledger_issues,
    ) = _ledger_graph(ledger)
    expected_ids = set(expected_pairs)
    source_ids = set(sources)
    response_ids = {key[0] for key in responses}
    ledger_ids = set(reservations) | set(finalizations) | set(incident_resolutions)
    unknown_ids = sorted((source_ids | response_ids | ledger_ids) - expected_ids)
    missing_sources = sorted(expected_ids - source_ids)
    missing_finalizations = sorted(expected_ids - set(finalizations))
    active_reservations = sorted(set(reservations) - set(finalizations))
    graph_issues = list(ledger_issues)
    graph_issues.extend(f"unknown work item: {item}" for item in unknown_ids)
    resolution_actual = Decimal(0)
    resolution_exposure = Decimal(0)
    for work_item_id, resolution in incident_resolutions.items():
        source = sources.get(work_item_id)
        if source is None:
            graph_issues.append(f"incident resolution has no source: {work_item_id}")
            continue
        if resolution.get("source_artifact_sha256") != source.get("artifact_sha256"):
            graph_issues.append(f"incident-resolution/source digest mismatch: {work_item_id}")
        resolution_digest = str(resolution.get("resolution_artifact_sha256") or "")
        resolution_event_digest = str(resolution.get("entry_sha256") or "")
        if len(resolution_digest) != 64 or len(resolution_event_digest) != 64:
            graph_issues.append(f"incident-resolution content address missing: {work_item_id}")
        finalization = finalizations.get(work_item_id)
        if finalization is not None and (
            finalization.get("source_incident_resolution_sha256") != resolution_digest
            or finalization.get("source_incident_resolution_ledger_entry_sha256")
            != resolution_event_digest
            or finalization.get("all_generation_costs_reconciled") is not False
            or finalization.get("provider_cost_exact") is not False
        ):
            graph_issues.append(f"incident-resolution/finalization policy mismatch: {work_item_id}")
        try:
            provider_actual = _decimal(
                resolution.get("provider_reconciled_actual_cost_usd"),
                field=f"{work_item_id} incident resolution provider actual",
            )
            conservative_exposure = _decimal(
                resolution.get("conservative_budget_exposure_usd"),
                field=f"{work_item_id} incident resolution conservative exposure",
            )
            reservation = reservations.get(work_item_id)
            reserved = _decimal(
                (reservation or {}).get("reserved_usd"),
                field=f"{work_item_id} incident resolution reservation",
            )
            if conservative_exposure != reserved or provider_actual > conservative_exposure:
                graph_issues.append(
                    f"incident-resolution conservative exposure mismatch: {work_item_id}"
                )
            resolution_actual += provider_actual
            resolution_exposure += conservative_exposure
        except ValueError as error:
            graph_issues.append(str(error))
    for work_item_id, finalization in finalizations.items():
        source = sources.get(work_item_id)
        if source is None:
            graph_issues.append(f"finalized work item has no source: {work_item_id}")
            continue
        if finalization.get("source_artifact_sha256") != source.get("artifact_sha256"):
            graph_issues.append(f"finalization/source digest mismatch: {work_item_id}")
        response_digests = {
            str(response.get("artifact_sha256") or "")
            for (response_work_item, _), response in responses.items()
            if response_work_item == work_item_id
        }
        if set(finalization.get("response_artifact_sha256s") or []) != response_digests:
            graph_issues.append(f"finalization/response digest mismatch: {work_item_id}")
    audit.add(
        "record_graph_links",
        "fail" if graph_issues else "pass",
        "critical",
        (
            "Every reservation, source, response, and finalization link agrees."
            if not graph_issues
            else "The reservation/source/response record graph is inconsistent."
        ),
        observed={
            "reservations": len(reservations),
            "sources": len(sources),
            "responses": len(responses),
            "finalizations": len(finalizations),
            "incidents": len(incidents),
            "incident_resolutions": len(incident_resolutions),
        },
        examples=graph_issues,
    )

    completion_issues = [
        *(f"missing source: {item}" for item in missing_sources[:5]),
        *(f"missing finalization: {item}" for item in missing_finalizations[:5]),
        *(f"active reservation: {item}" for item in active_reservations[:5]),
    ]
    if completion_issues:
        completion_status = "fail" if strict_final else "warn"
        completion_severity = "high" if strict_final else "info"
        completion_message = (
            "The strict final workload is incomplete or retains active reservations."
            if strict_final
            else "The sequential collection is structurally valid but still in progress."
        )
    else:
        completion_status = "pass"
        completion_severity = "info"
        completion_message = "All planned pairs have a source and finalized reservation."
    audit.add(
        "workload_completion",
        completion_status,
        completion_severity,
        completion_message,
        expected={
            "sources": len(expected_pairs),
            "finalizations": len(expected_pairs),
            "active": 0,
        },
        observed={
            "sources": len(sources),
            "finalizations": len(finalizations),
            "active": len(active_reservations),
        },
        examples=completion_issues,
    )

    audit.add(
        "conservative_no_id_incident_resolutions",
        "warn" if incident_resolutions else "pass",
        "high" if incident_resolutions else "info",
        (
            "HTTP-200/no-choice incidents remain unranked: no generation ID or exact "
            "provider cost was invented, and each full admitted allowance remains charged."
            if incident_resolutions
            else "No conservative no-generation-ID incident resolution is present."
        ),
        observed={
            "count": len(incident_resolutions),
            "provider_reconciled_actual_cost_usd": _decimal_text(resolution_actual),
            "conservative_budget_exposure_usd": _decimal_text(resolution_exposure),
            "rank_eligible": False,
            "research_release_eligible": False,
        },
    )

    identity_issues: list[str] = []
    accounting_issues: list[str] = []
    generation_issues: list[str] = []
    trace_issues: list[str] = []
    provenance_issues: list[str] = []
    governance_issues: list[str] = []
    privacy_issues: list[str] = []
    source_generation_owner: dict[str, str] = {}
    attempt_owner: dict[str, str] = {}
    epicure_identities: set[tuple[str, str, str, str]] = set()
    source_cost_micros_total = 0
    generation_count = 0
    provider_attempt_count = 0
    tool_call_count = 0
    tool_success_count = 0

    for work_item_id, source in sources.items():
        expected = expected_pairs.get(work_item_id)
        if expected is None:
            continue
        exact_source_fields = {
            "dataset_work_item_id": expected.work_item_id,
            "dataset_task_id": expected.task_id,
            "candidate_manifest_sha256": expected.manifest_sha256,
            "prompt": expected.prompt,
            "prompt_sha256": expected.prompt_sha256,
            "category": expected.task_family,
            "requested_model_id": expected.model_id,
            "requested_provider": expected.provider_tag,
            "run_purpose": "epicure_on_off_pair",
            "endpoint_execution_contract_sha256": expected.endpoint_execution_sha256,
            "execution_policy_sha256": expected.execution_policy_sha256,
        }
        for field, expected_value in exact_source_fields.items():
            if source.get(field) != expected_value:
                identity_issues.append(f"{work_item_id}: source {field} mismatch")
        model_contract = source.get("model_contract")
        frozen_generation = source.get("frozen_generation_contract")
        if not isinstance(model_contract, Mapping) or (
            model_contract.get("id") != expected.model_id
            or model_contract.get("canonical_slug") != expected.canonical_model_slug
        ):
            identity_issues.append(f"{work_item_id}: source model contract mismatch")
        if not isinstance(frozen_generation, Mapping) or (
            frozen_generation.get("expected_actual_model_id") != expected.canonical_model_slug
            or frozen_generation.get("expected_actual_provider_slug") != expected.provider_name
        ):
            identity_issues.append(f"{work_item_id}: frozen returned identity mismatch")
        for field in ("official", "rank_eligible", "research_result"):
            if source.get(field) is not False:
                governance_issues.append(f"{work_item_id}: source {field} is not false")
        privacy_issues.extend(
            f"{work_item_id}: source.{path}" for path in _find_forbidden_keys(source)
        )

        epicure = source.get("epicure")
        tool_schema_sha = str(source.get("epicure_tool_schema_sha256") or "")
        if not isinstance(epicure, Mapping):
            provenance_issues.append(f"{work_item_id}: missing Epicure provenance")
        else:
            release_id = str(epicure.get("release_id") or "")
            application_sha = str(epicure.get("application_sha256") or "")
            bundle_sha = str(epicure.get("bundle_sha256") or "")
            if not all(
                len(value) == 64 for value in (application_sha, bundle_sha, tool_schema_sha)
            ):
                provenance_issues.append(f"{work_item_id}: incomplete Epicure content hashes")
            if release_id != EXPLORATORY_EPICURE_RELEASE_ID:
                provenance_issues.append(f"{work_item_id}: unexpected Epicure release ID")
            epicure_identities.add((release_id, application_sha, bundle_sha, tool_schema_sha))

        results = source.get("results")
        source_errors = source.get("errors")
        if not isinstance(results, Mapping) or set(results) - set(CONDITIONS):
            accounting_issues.append(f"{work_item_id}: invalid source results object")
            results = {}
        if not isinstance(source_errors, Mapping):
            accounting_issues.append(f"{work_item_id}: invalid source errors object")
            source_errors = {}
        result_cost_micros = 0
        source_condition_generation_ids: set[str] = set()
        for condition, result in results.items():
            if not isinstance(result, Mapping):
                accounting_issues.append(f"{work_item_id}/{condition}: result is not an object")
                continue
            if (
                result.get("actual_model_id") != expected.canonical_model_slug
                or result.get("actual_provider") != expected.provider_name
            ):
                identity_issues.append(f"{work_item_id}/{condition}: actual identity mismatch")
            generation_ids = result.get("generation_ids")
            metadata = result.get("generation_metadata")
            cost_micros = result.get("cost_micros")
            if (
                not isinstance(generation_ids, list)
                or not isinstance(metadata, list)
                or not isinstance(cost_micros, int)
                or isinstance(cost_micros, bool)
                or cost_micros < 0
            ):
                accounting_issues.append(f"{work_item_id}/{condition}: malformed generation cost")
                continue
            result_cost_micros += cost_micros
            metadata_ids: list[str] = []
            metadata_cost = 0
            for generation in metadata:
                if not isinstance(generation, Mapping):
                    accounting_issues.append(
                        f"{work_item_id}/{condition}: malformed generation metadata"
                    )
                    continue
                generation_id = str(generation.get("generation_id") or "")
                generation_cost = generation.get("cost_micros")
                if (
                    not generation_id
                    or not isinstance(generation_cost, int)
                    or isinstance(generation_cost, bool)
                    or generation_cost < 0
                ):
                    accounting_issues.append(
                        f"{work_item_id}/{condition}: invalid generation ID/cost"
                    )
                    continue
                metadata_ids.append(generation_id)
                metadata_cost += generation_cost
                owner = source_generation_owner.setdefault(generation_id, work_item_id)
                if owner != work_item_id or generation_id in source_condition_generation_ids:
                    generation_issues.append(
                        f"duplicate generation {generation_id}: {owner} and {work_item_id}"
                    )
                source_condition_generation_ids.add(generation_id)
                generation_count += 1
                if (
                    result.get("cost_reconciled") is True
                    and generation.get("reconciled") is not True
                ):
                    accounting_issues.append(
                        f"{work_item_id}/{condition}: result claims reconciliation "
                        "but metadata does not"
                    )
            if (
                len(metadata_ids) != len(set(metadata_ids))
                or set(map(str, generation_ids)) != set(metadata_ids)
                or metadata_cost != cost_micros
                or str(result.get("generation_id") or "") not in set(metadata_ids)
            ):
                accounting_issues.append(
                    f"{work_item_id}/{condition}: generation IDs or cost sum mismatch"
                )
            traces = result.get("tool_trace")
            if not isinstance(traces, list):
                trace_issues.append(f"{work_item_id}/{condition}: tool trace is not a list")
                traces = []
            if condition == "epicure_off" and traces:
                trace_issues.append(f"{work_item_id}: Epicure-off contains a tool trace")
            if len(traces) > max_tool_calls_total:
                trace_issues.append(f"{work_item_id}/{condition}: total tool-call cap exceeded")
            for trace in traces:
                if not isinstance(trace, Mapping):
                    trace_issues.append(f"{work_item_id}/{condition}: malformed tool trace")
                    continue
                round_index = trace.get("round_index")
                if (
                    not isinstance(round_index, int)
                    or isinstance(round_index, bool)
                    or round_index < 0
                    or round_index >= max_tool_rounds
                ):
                    trace_issues.append(f"{work_item_id}/{condition}: tool round out of bounds")
                tool_call_count += 1
                if trace.get("is_error") is False:
                    tool_success_count += 1

        for condition in CONDITIONS:
            result = results.get(condition)
            condition_has_error = any(
                str(key) == condition or str(key).startswith(f"{condition}_")
                for key in source_errors
            )
            normalization_expected = (
                work_item_id in finalizations
                and isinstance(result, Mapping)
                and result.get("cost_reconciled") is True
                and not condition_has_error
            )
            response_exists = (work_item_id, condition) in responses
            if normalization_expected != response_exists:
                accounting_issues.append(
                    f"{work_item_id}/{condition}: normalized-response eligibility mismatch"
                )

        incomplete = source.get("incomplete_generation_metadata") or []
        incomplete_cost_micros = 0
        if not isinstance(incomplete, list):
            accounting_issues.append(f"{work_item_id}: incomplete metadata is not a list")
            incomplete = []
        for generation in incomplete:
            if not isinstance(generation, Mapping):
                accounting_issues.append(f"{work_item_id}: malformed incomplete metadata")
                continue
            generation_id = str(generation.get("generation_id") or "")
            generation_cost = generation.get("cost_micros")
            if (
                not generation_id
                or not isinstance(generation_cost, int)
                or isinstance(generation_cost, bool)
                or generation_cost < 0
            ):
                accounting_issues.append(f"{work_item_id}: invalid incomplete generation")
                continue
            incomplete_cost_micros += generation_cost
            owner = source_generation_owner.setdefault(generation_id, work_item_id)
            if owner != work_item_id or generation_id in source_condition_generation_ids:
                generation_issues.append(
                    f"duplicate generation {generation_id}: {owner} and {work_item_id}"
                )
            source_condition_generation_ids.add(generation_id)
            generation_count += 1
        budget = source.get("budget")
        actual_cost_micros = (
            budget.get("actual_cost_micros") if isinstance(budget, Mapping) else None
        )
        if (
            not isinstance(actual_cost_micros, int)
            or isinstance(actual_cost_micros, bool)
            or actual_cost_micros < 0
            or result_cost_micros + incomplete_cost_micros != actual_cost_micros
        ):
            accounting_issues.append(f"{work_item_id}: source budget cost does not reconcile")
        else:
            source_cost_micros_total += actual_cost_micros

        reservation = reservations.get(work_item_id)
        finalization = finalizations.get(work_item_id)
        if reservation is not None:
            try:
                reserved_usd = _decimal(
                    reservation.get("reserved_usd"), field=f"{work_item_id} reservation"
                )
                effective_actual = (
                    source_actual_costs_usd.get(work_item_id)
                    if source_actual_costs_usd is not None
                    else Decimal(actual_cost_micros or 0) / Decimal(1_000_000)
                )
                if effective_actual is None or effective_actual > reserved_usd:
                    accounting_issues.append(
                        f"{work_item_id}: actual cost exceeds transactional reservation"
                    )
            except ValueError as error:
                accounting_issues.append(str(error))
        if finalization is not None and source_actual_costs_usd is not None:
            try:
                recorded_actual = _decimal(
                    finalization.get("source_actual_cost_usd"),
                    field=f"{work_item_id} finalization actual cost",
                )
                if recorded_actual != source_actual_costs_usd.get(work_item_id):
                    accounting_issues.append(f"{work_item_id}: finalization actual cost mismatch")
            except ValueError as error:
                accounting_issues.append(str(error))

        attempts = source.get("provider_attempt_events") or []
        if not isinstance(attempts, list):
            generation_issues.append(f"{work_item_id}: provider attempts are not a list")
            attempts = []
        for event in attempts:
            if not isinstance(event, Mapping):
                generation_issues.append(f"{work_item_id}: malformed provider attempt")
                continue
            attempt_id = str(event.get("attempt_id") or "")
            if not attempt_id:
                generation_issues.append(f"{work_item_id}: provider attempt has no ID")
                continue
            owner = attempt_owner.setdefault(attempt_id, work_item_id)
            if owner != work_item_id:
                generation_issues.append(
                    f"duplicate provider attempt {attempt_id}: {owner} and {work_item_id}"
                )
            provider_attempt_count += 1

        mcp_events = source.get("mcp_trace_events") or []
        if not isinstance(mcp_events, list):
            trace_issues.append(f"{work_item_id}: MCP trace is not a list")
            mcp_events = []
        for event in mcp_events:
            if not isinstance(event, Mapping):
                trace_issues.append(f"{work_item_id}: malformed MCP trace event")
                continue
            arm_id = str(event.get("arm_id") or "")
            if not arm_id.endswith(":epicure_on"):
                trace_issues.append(f"{work_item_id}: MCP event leaked into Epicure-off")

    for key, response in responses.items():
        work_item_id, condition = key
        expected = expected_pairs.get(work_item_id)
        source = sources.get(work_item_id)
        if expected is None or source is None:
            continue
        if response.get("work_item_id") != work_item_id or response.get("condition") != condition:
            identity_issues.append(f"{work_item_id}/{condition}: response key mismatch")
        if condition not in CONDITIONS:
            identity_issues.append(f"{work_item_id}/{condition}: unsupported condition")
            continue
        for field in ("official", "rank_eligible", "research_result", "research_release_eligible"):
            if response.get(field) is not False:
                governance_issues.append(f"{work_item_id}/{condition}: {field} is not false")
        task = response.get("task")
        model = response.get("model")
        response_source = response.get("source")
        provenance = response.get("provenance")
        cost = response.get("cost")
        result = response.get("response")
        if not all(
            isinstance(value, Mapping)
            for value in (task, model, response_source, provenance, cost, result)
        ):
            identity_issues.append(f"{work_item_id}/{condition}: response sections missing")
            continue
        expected_task = {
            "public_id": expected.task_id,
            "family": expected.task_family,
            "prompt": expected.prompt,
            "prompt_sha256": expected.prompt_sha256,
            "split": expected.task_split,
            "review_status": expected.task_review_status,
        }
        if dict(task) != expected_task:
            identity_issues.append(f"{work_item_id}/{condition}: task identity mismatch")
        exact_model = {
            "slot_id": expected.slot_id,
            "requested_model_id": expected.model_id,
            "canonical_model_slug": expected.canonical_model_slug,
            "provider_tag": expected.provider_tag,
            "endpoint_manifest_sha256": expected.endpoint_manifest_sha256,
            "endpoint_execution_sha256": expected.endpoint_execution_sha256,
            "execution_policy_sha256": expected.execution_policy_sha256,
            "actual_model_id": expected.canonical_model_slug,
            "actual_provider": expected.provider_name,
        }
        for field, expected_value in exact_model.items():
            if model.get(field) != expected_value:
                identity_issues.append(f"{work_item_id}/{condition}: model.{field} mismatch")
        if (
            response.get("manifest_sha256") != expected.manifest_sha256
            or response.get("task_registry_sha256") != expected.task_registry_sha256
            or response.get("execution_policy_sha256") != expected.execution_policy_sha256
        ):
            identity_issues.append(f"{work_item_id}/{condition}: frozen input hash mismatch")
        if response_source.get("artifact_sha256") != source.get(
            "artifact_sha256"
        ) or response_source.get("run_id") != source.get("run_id"):
            identity_issues.append(f"{work_item_id}/{condition}: source link mismatch")
        source_result = (source.get("results") or {}).get(condition)
        if not isinstance(source_result, Mapping) or dict(result) != dict(source_result):
            accounting_issues.append(
                f"{work_item_id}/{condition}: normalized response differs from source"
            )
        else:
            if (
                cost.get("actual_cost_micros") != result.get("cost_micros")
                or cost.get("generation_ids") != result.get("generation_ids")
                or cost.get("generation_metadata") != result.get("generation_metadata")
                or cost.get("all_generation_costs_reconciled") is not True
                or result.get("cost_reconciled") is not True
            ):
                accounting_issues.append(
                    f"{work_item_id}/{condition}: normalized generation accounting mismatch"
                )
        expected_access = condition == "epicure_on"
        if provenance.get("epicure_access") is not expected_access:
            trace_issues.append(f"{work_item_id}/{condition}: Epicure access flag mismatch")
        if provenance.get("epicure") != source.get("epicure") or provenance.get(
            "epicure_tool_schema_sha256"
        ) != source.get("epicure_tool_schema_sha256"):
            provenance_issues.append(f"{work_item_id}/{condition}: Epicure provenance mismatch")
        source_events = [
            event
            for event in (source.get("mcp_trace_events") or [])
            if isinstance(event, Mapping)
            and str(event.get("arm_id") or "").endswith(f":{condition}")
        ]
        response_events = provenance.get("mcp_trace_events")
        result_traces = result.get("tool_trace")
        if (
            response_events != source_events
            or _trace_without_arm_id(source_events) != result_traces
        ):
            trace_issues.append(f"{work_item_id}/{condition}: MCP/result trace mismatch")
        if condition == "epicure_off" and (source_events or result_traces):
            trace_issues.append(f"{work_item_id}: Epicure-off condition used a tool")
        privacy_issues.extend(
            f"{work_item_id}/{condition}: response.{path}"
            for path in _find_forbidden_keys(response)
        )

    if len(epicure_identities) > 1:
        provenance_issues.append(
            f"multiple Epicure runtime identities observed: {len(epicure_identities)}"
        )
    audit.add(
        "model_provider_identity",
        "fail" if identity_issues else "pass",
        "critical",
        (
            "Every requested, canonical, returned-model, and provider identity is exact."
            if not identity_issues
            else "At least one source or response has model/provider/task identity drift."
        ),
        observed={"issues": len(identity_issues)},
        examples=identity_issues,
    )
    audit.add(
        "generation_accounting",
        "fail" if accounting_issues else "pass",
        "critical",
        (
            "Generation metadata, normalized responses, reservations, and costs reconcile."
            if not accounting_issues
            else "Generation or transactional cost accounting is inconsistent."
        ),
        observed={"issues": len(accounting_issues)},
        examples=accounting_issues,
    )
    audit.add(
        "generation_and_attempt_uniqueness",
        "fail" if generation_issues else "pass",
        "critical",
        (
            "Generation IDs and provider-attempt IDs are unique across source runs."
            if not generation_issues
            else "Duplicate or malformed paid-generation/attempt identities were found."
        ),
        observed={
            "unique_generation_ids": len(source_generation_owner),
            "unique_attempt_ids": len(attempt_owner),
            "issues": len(generation_issues),
        },
        examples=generation_issues,
    )
    audit.add(
        "epicure_condition_and_trace_integrity",
        "fail" if trace_issues else "pass",
        "critical",
        (
            "Epicure-off is isolated and Epicure-on MCP traces exactly match model tool traces."
            if not trace_issues
            else "Epicure condition isolation, tool bounds, or trace completeness failed."
        ),
        observed={
            "tool_calls": tool_call_count,
            "successful_tool_calls": tool_success_count,
            "issues": len(trace_issues),
        },
        examples=trace_issues,
    )
    audit.add(
        "epicure_runtime_provenance",
        "fail" if provenance_issues else "pass",
        "critical",
        (
            "Every record carries one exact Epicure application, data-bundle, "
            "and tool-schema identity."
            if not provenance_issues
            else "Epicure runtime provenance is incomplete or inconsistent."
        ),
        observed={
            "runtime_identity_count": len(epicure_identities),
            "identities": [
                {
                    "release_id": item[0],
                    "application_sha256": item[1],
                    "bundle_sha256": item[2],
                    "tool_schema_sha256": item[3],
                }
                for item in sorted(epicure_identities)
            ],
            "issues": len(provenance_issues),
        },
        examples=provenance_issues,
    )
    audit.add(
        "unranked_governance_flags",
        "fail" if governance_issues else "pass",
        "critical",
        (
            "All source and response records remain non-official, non-ranked, "
            "and non-release-eligible."
            if not governance_issues
            else "A record improperly claims official, ranked, or research-release status."
        ),
        observed={"issues": len(governance_issues)},
        examples=governance_issues,
    )
    audit.add(
        "secret_and_network_identifier_absence",
        "fail" if privacy_issues else "pass",
        "critical",
        (
            "No secret-bearing or raw network-identifier fields were found."
            if not privacy_issues
            else "Potential secrets or raw network identifiers occur in immutable records."
        ),
        observed={"issues": len(privacy_issues)},
        examples=privacy_issues,
    )

    complete_pairs = 0
    partial_pairs = 0
    failed_pairs = 0
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    execution_errors: Counter[str] = Counter()
    for work_item_id, expected in expected_pairs.items():
        finalized = work_item_id in finalizations
        off = (work_item_id, "epicure_off") in responses
        on = (work_item_id, "epicure_on") in responses
        source_errors = (sources.get(work_item_id) or {}).get("errors") or {}
        if isinstance(source_errors, Mapping):
            execution_errors.update(str(value) for value in source_errors.values())
        if finalized and off and on:
            complete_pairs += 1
            result = "complete"
        elif finalized and (off or on):
            partial_pairs += 1
            result = "partial"
        elif finalized:
            failed_pairs += 1
            result = "failed"
        else:
            result = "pending"
        by_family[expected.task_family][result] += 1
        by_model[expected.model_id][result] += 1
    reliability_failures = partial_pairs + failed_pairs
    audit.add(
        "paired_response_reliability",
        "warn" if reliability_failures else "pass",
        "medium" if reliability_failures else "info",
        (
            "Finalized partial/failed pairs remain excluded from preference fitting "
            "and are reported as reliability outcomes."
            if reliability_failures
            else "Every finalized pair currently has both normalized response arms."
        ),
        observed={
            "complete_pairs": complete_pairs,
            "partial_pairs": partial_pairs,
            "failed_pairs": failed_pairs,
        },
    )
    finalized_model_counts = {
        model: sum(values.get(status, 0) for status in ("complete", "partial", "failed"))
        for model, values in by_model.items()
    }
    comparability_issues = [
        (
            f"{model}: {values.get('complete', 0)}/{finalized_model_counts[model]} "
            "finalized pairs complete"
        )
        for model, values in sorted(by_model.items())
        if finalized_model_counts[model] >= 2
        and values.get("complete", 0) * 2 < finalized_model_counts[model]
    ]
    audit.add(
        "cross_model_comparability",
        "warn" if comparability_issues else "pass",
        "high" if comparability_issues else "info",
        (
            "Model-specific response attrition is too uneven for a fair cross-model ranking."
            if comparability_issues
            else "No severe model-specific complete-pair attrition is currently observed."
        ),
        observed={"models_below_50_percent_complete": len(comparability_issues)},
        examples=comparability_issues,
    )
    finalized_arm_opportunities = len(finalizations) * len(CONDITIONS)
    execution_error_count = sum(execution_errors.values())
    execution_error_rate = (
        Decimal(execution_error_count) / Decimal(finalized_arm_opportunities)
        if finalized_arm_opportunities
        else Decimal(0)
    )
    audit.add(
        "execution_failure_profile",
        "warn" if execution_errors else "pass",
        (
            "high"
            if execution_error_rate >= Decimal("0.25")
            else ("medium" if execution_errors else "info")
        ),
        (
            "Provider/structured-output/tool failures are retained as reliability evidence."
            if execution_errors
            else "No execution errors are recorded in finalized sources."
        ),
        observed={
            "error_arms": execution_error_count,
            "finalized_arm_opportunities": finalized_arm_opportunities,
            "error_rate": _decimal_text(execution_error_rate),
            "by_message": [
                {"message": message, "count": count}
                for message, count in execution_errors.most_common()
            ],
        },
    )
    real_generation_status = (
        "pass" if generation_count > 0 else ("fail" if strict_final else "warn")
    )
    audit.add(
        "real_provider_generation_evidence",
        real_generation_status,
        "critical" if real_generation_status == "fail" else "info",
        (
            "Immutable provider generation IDs and reconciled metadata prove real model calls."
            if generation_count > 0
            else "No provider generation evidence has been collected yet."
        ),
        expected=">0 reconciled provider generations",
        observed=generation_count,
    )
    metrics = {
        "expected_pairs": len(expected_pairs),
        "expected_response_opportunities": len(expected_pairs) * len(CONDITIONS),
        "reservations": len(reservations),
        "finalizations": len(finalizations),
        "active_reservations": len(active_reservations),
        "source_artifacts": len(sources),
        "response_artifacts": len(responses),
        "complete_pairs": complete_pairs,
        "partial_pairs": partial_pairs,
        "failed_pairs": failed_pairs,
        "pending_pairs": len(expected_pairs) - len(finalizations),
        "generation_ids": generation_count,
        "unique_generation_ids": len(source_generation_owner),
        "provider_attempt_events": provider_attempt_count,
        "unique_provider_attempt_ids": len(attempt_owner),
        "source_recorded_cost_usd": _decimal_text(
            Decimal(source_cost_micros_total) / Decimal(1_000_000)
        ),
        "tool_calls": tool_call_count,
        "successful_tool_calls": tool_success_count,
        "ledger_incidents": len(incidents),
        "conservative_incident_resolutions": len(incident_resolutions),
        "conservative_incident_provider_actual_usd": _decimal_text(resolution_actual),
        "conservative_incident_budget_exposure_usd": _decimal_text(resolution_exposure),
        "execution_error_arms": execution_error_count,
        "execution_error_rate": _decimal_text(execution_error_rate),
        "execution_errors_by_message": [
            {"message": message, "count": count}
            for message, count in execution_errors.most_common()
        ],
        "by_task_family": {family: dict(by_family[family]) for family in TASK_FAMILIES},
        "by_model": {model: dict(values) for model, values in sorted(by_model.items())},
    }
    return audit.findings, metrics


def _load_stable_state(
    *,
    prior_artifact_directory: Path,
    prior_corrections_directory: Path | None,
    frontier_ledger_path: Path,
    source_directory: Path,
    source_corrections_directory: Path | None,
    response_directory: Path,
    dataset_ledger_path: Path,
    attempts: int = 3,
) -> tuple[DatasetState, bool]:
    """Take a lock-free snapshot, retrying if the append-only ledger advances."""

    last_state: DatasetState | None = None
    for _ in range(attempts):
        before = load_dataset_ledger(dataset_ledger_path)
        before_head = before[-1]["entry_sha256"] if before else None
        state = _load_state(
            prior_artifact_directory=prior_artifact_directory,
            prior_corrections_directory=prior_corrections_directory,
            prior_reservation_ledger_path=frontier_ledger_path,
            source_directory=source_directory,
            source_corrections_directory=source_corrections_directory,
            response_directory=response_directory,
            ledger_path=dataset_ledger_path,
        )
        after = load_dataset_ledger(dataset_ledger_path)
        after_head = after[-1]["entry_sha256"] if after else None
        last_state = state
        if before_head == after_head and (
            not state.ledger or state.ledger[-1]["entry_sha256"] == after_head
        ):
            return state, True
    if last_state is None:
        raise IntegrityError("could not load exploratory dataset state")
    return last_state, False


def _summary_state_issues(
    summary: Mapping[str, Any],
    state: DatasetState,
    work_items: Sequence[WorkItem],
) -> list[str]:
    issues: list[str] = []
    ledger = summary.get("ledger")
    budget = summary.get("budget")
    if not isinstance(ledger, Mapping) or not isinstance(budget, Mapping):
        return ["summary lacks ledger/budget sections"]
    expected_ledger = {
        "filename": "ledger.jsonl",
        "entry_count": len(state.ledger),
        "head_entry_sha256": state.ledger[-1]["entry_sha256"] if state.ledger else None,
    }
    if dict(ledger) != expected_ledger:
        issues.append("final summary ledger head/count mismatch")
    expected_budget_fields = {
        "verified_prior_artifact_exposure_usd": state.prior_verified_exposure_usd,
        "effective_prior_exposure_usd": state.prior_effective_exposure_usd,
        "active_frontier_ledger_reservations_usd": state.prior_active_reservation_usd,
        "dataset_actual_cost_usd": state.dataset_actual_cost_usd,
        "dataset_source_exposure_usd": state.dataset_source_exposure_usd,
        "unresolved_dataset_source_reserve_usd": (state.unresolved_dataset_source_reserve_usd),
        "active_reservations_without_source_usd": state.orphan_reservation_usd,
        "final_total_exposure_usd": state.total_exposure_usd,
    }
    for field, expected in expected_budget_fields.items():
        try:
            observed = _decimal(budget.get(field), field=f"summary budget.{field}")
        except ValueError:
            issues.append(f"summary budget.{field} is invalid")
            continue
        if observed != expected:
            issues.append(f"summary budget.{field}={observed} but state={expected}")
    if summary.get("coverage_and_reliability") != _summary_coverage(work_items, state):
        issues.append("final summary coverage/reliability differs from immutable state")
    return issues


def _is_resolution_only_summary_transition(
    summary: Mapping[str, Any],
    state: DatasetState,
    issues: Sequence[str],
) -> bool:
    """Allow a verified no-network resolution suffix during an incomplete run."""

    summary_ledger = summary.get("ledger")
    if not isinstance(summary_ledger, Mapping):
        return False
    count = summary_ledger.get("entry_count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or count >= len(state.ledger)
        or state.ledger[count - 1].get("entry_sha256") != summary_ledger.get("head_entry_sha256")
    ):
        return False
    suffix = state.ledger[count:]
    if not suffix or any(
        entry.get("event_type") != SOURCE_INCIDENT_RESOLUTION_EVENT_TYPE for entry in suffix
    ):
        return False
    resolution_event_hashes = {
        resolution.ledger_event_sha256 for resolution in state.incident_resolutions.values()
    }
    if {str(entry.get("entry_sha256") or "") for entry in suffix} - resolution_event_hashes:
        return False
    allowed_issue_prefixes = (
        "final summary ledger head/count mismatch",
        "summary budget.unresolved_dataset_source_reserve_usd=",
    )
    return bool(issues) and all(issue.startswith(allowed_issue_prefixes) for issue in issues)


def verify_real_exploratory_run(
    *,
    manifest_path: str | Path,
    artifact_root: str | Path = "artifacts/real-exploratory",
    summary_path: str | Path | None = None,
    prior_artifact_directory: str | Path = "artifacts/live-smoke",
    prior_corrections_directory: str | Path | None = "artifacts/corrections",
    frontier_ledger_path: str | Path = "artifacts/frontier-contract/ledger.jsonl",
    strict_final: bool = True,
    external_account_delta_usd: Decimal | None = None,
) -> dict[str, Any]:
    """Verify one on-disk collection without calling OpenRouter or Epicure."""

    audit = Audit()
    root = Path(artifact_root)
    manifest = Path(manifest_path)
    source_root = root / "source-runs"
    response_root = root / "responses"
    summary_root = root / "summaries"
    dataset_ledger = root / "ledger.jsonl"
    source_corrections = root / "corrections"
    selected_summary, selected_summary_path, summary_documents = _summary_documents(
        summary_root,
        explicit_summary=Path(summary_path) if summary_path is not None else None,
        strict_final=strict_final,
        audit=audit,
    )
    work_items: list[WorkItem] = []
    policy: ExecutionPolicy | None = None
    state: DatasetState | None = None
    metrics: dict[str, Any] = {}
    stable_snapshot = False
    correction_count = 0
    journal_metrics: dict[str, Any] = {}

    if selected_summary is None:
        audit.add(
            "verification_inputs",
            "fail",
            "critical",
            "No valid summary exists from which to reconstruct the authorized workload.",
        )
    else:
        for field in ("official", "rank_eligible", "research_result"):
            if selected_summary.get(field) is not False:
                audit.add(
                    "summary_governance_flags",
                    "fail",
                    "critical",
                    f"Selected summary improperly sets {field}.",
                    observed=selected_summary.get(field),
                )
        try:
            work_items, policy = _expected_workload(
                manifest_path=manifest,
                summary=selected_summary,
                audit=audit,
            )
            audit.add(
                "manifest_content_address_and_routing",
                "pass",
                "critical",
                "The frozen manifest content address and exact-endpoint routing policy validate.",
                observed=(selected_summary.get("manifest") or {}).get("sha256"),
            )
        except (IntegrityError, KeyError, TypeError, ValueError) as error:
            audit.add(
                "manifest_content_address_and_routing",
                "fail",
                "critical",
                "The manifest/workload/policy inputs cannot be verified.",
                examples=[str(error)],
            )

    if work_items:
        try:
            state, stable_snapshot = _load_stable_state(
                prior_artifact_directory=Path(prior_artifact_directory),
                prior_corrections_directory=(
                    Path(prior_corrections_directory)
                    if prior_corrections_directory is not None
                    else None
                ),
                frontier_ledger_path=Path(frontier_ledger_path),
                source_directory=source_root,
                source_corrections_directory=(
                    source_corrections if source_corrections.exists() else None
                ),
                response_directory=response_root,
                dataset_ledger_path=dataset_ledger,
            )
            _validate_state_against_workload(state, work_items)
            audit.add(
                "immutable_artifact_and_ledger_integrity",
                "pass" if stable_snapshot else ("fail" if strict_final else "warn"),
                "critical" if strict_final and not stable_snapshot else "info",
                (
                    "Source/response content addresses, journals, ledgers, corrections, "
                    "and links validate on a stable snapshot."
                    if stable_snapshot
                    else (
                        "The append-only dataset ledger advanced during every lock-free "
                        "snapshot attempt."
                    )
                ),
                observed={"stable_snapshot": stable_snapshot},
            )
        except (IntegrityError, JournalIntegrityError, OSError, ValueError) as error:
            audit.add(
                "immutable_artifact_and_ledger_integrity",
                "fail",
                "critical",
                "Source/response content addresses, journals, ledgers, or links fail validation.",
                examples=[str(error)],
            )

    if state is not None and policy is not None:
        expected_pairs = {
            item.work_item_id: ExpectedPair.from_work_item(item) for item in work_items
        }
        source_records = {
            work_item_id: dict(source.artifact) for work_item_id, source in state.sources.items()
        }
        response_records = {
            key: _read_json(response.path) for key, response in state.responses.items()
        }
        source_actual_costs = {
            work_item_id: source.exposure.actual_cost_usd
            for work_item_id, source in state.sources.items()
        }
        graph_findings, metrics = audit_record_graph(
            expected_pairs=expected_pairs,
            ledger=state.ledger,
            sources=source_records,
            responses=response_records,
            strict_final=strict_final,
            max_tool_rounds=policy.max_tool_rounds,
            max_tool_calls_total=policy.max_tool_calls_total,
            source_actual_costs_usd=source_actual_costs,
        )
        audit.findings.extend(graph_findings)

        correction_count = sum(
            source.exposure.cost_correction_sha256 is not None for source in state.sources.values()
        )
        calculated_dataset_actual = sum(
            (source.exposure.actual_cost_usd for source in state.sources.values()),
            Decimal(0),
        )
        calculated_dataset_exposure = sum(
            (source.exposure.exposure_usd for source in state.sources.values()),
            Decimal(0),
        )
        budget_integrity_issues: list[str] = []
        if state.dataset_actual_cost_usd != calculated_dataset_actual:
            budget_integrity_issues.append("dataset state/source records actual-cost mismatch")
        if state.dataset_source_exposure_usd != calculated_dataset_exposure:
            budget_integrity_issues.append("dataset state/source records exposure mismatch")
        budget_outstanding: list[str] = []
        if state.prior_active_reservation_usd != 0:
            budget_outstanding.append("frontier ledger retains an active reservation")
        if state.orphan_reservation_usd != 0:
            budget_outstanding.append("dataset ledger retains a reservation without a source")
        if state.unresolved_dataset_source_reserve_usd != 0:
            budget_outstanding.append("dataset source exposure is not fully reconciled")
        if budget_integrity_issues or (strict_final and budget_outstanding):
            budget_status = "fail"
        elif budget_outstanding:
            budget_status = "warn"
        else:
            budget_status = "pass"
        audit.add(
            "budget_exposure_reconciliation",
            budget_status,
            "critical"
            if budget_status == "fail"
            else ("high" if budget_status == "warn" else "info"),
            (
                "Source costs, conservative exposure, prior exposure, and active "
                "reservations reconcile."
                if budget_status == "pass"
                else (
                    "The in-progress state retains conservatively charged exposure."
                    if budget_status == "warn"
                    else "The final budget/exposure state is not fully reconciled."
                )
            ),
            observed={
                "verified_prior_artifact_exposure_usd": _decimal_text(
                    state.prior_verified_exposure_usd
                ),
                "effective_prior_exposure_usd": _decimal_text(state.prior_effective_exposure_usd),
                "active_frontier_reservations_usd": _decimal_text(
                    state.prior_active_reservation_usd
                ),
                "dataset_actual_cost_usd": _decimal_text(state.dataset_actual_cost_usd),
                "dataset_source_exposure_usd": _decimal_text(state.dataset_source_exposure_usd),
                "unresolved_dataset_source_reserve_usd": _decimal_text(
                    state.unresolved_dataset_source_reserve_usd
                ),
                "orphan_dataset_reservation_usd": _decimal_text(state.orphan_reservation_usd),
                "total_exposure_usd": _decimal_text(state.total_exposure_usd),
                "cost_corrections": correction_count,
            },
            examples=[*budget_integrity_issues, *budget_outstanding],
        )

        if external_account_delta_usd is None:
            audit.add(
                "external_account_delta_reconciliation",
                "warn",
                "medium",
                (
                    "No independently captured OpenRouter account-usage delta was supplied; "
                    "per-generation accounting is verified, but account-level reconciliation "
                    "remains a governance follow-up."
                ),
                expected=_decimal_text(state.dataset_actual_cost_usd),
                observed="not supplied",
            )
        else:
            difference = abs(external_account_delta_usd - state.dataset_actual_cost_usd)
            audit.add(
                "external_account_delta_reconciliation",
                "pass" if difference <= Decimal("0.000001") else "fail",
                "high",
                (
                    "The independently supplied account delta matches reconciled dataset cost."
                    if difference <= Decimal("0.000001")
                    else (
                        "The independently supplied account delta differs from generation "
                        "accounting."
                    )
                ),
                expected=_decimal_text(state.dataset_actual_cost_usd),
                observed=_decimal_text(external_account_delta_usd),
            )

        try:
            journals = scan_recovery_journals(source_root)
            in_progress = [item for item in journals if not item.finalized]
            unresolved_journals = [
                item
                for item in journals
                if item.unreconciled_generation_ids or item.uncertain_attempt_ids
            ]
            unresolved_finalized = [item for item in unresolved_journals if item.finalized]
            source_journal_names = {
                str((source.artifact.get("run_journal") or {}).get("filename") or "")
                for source in state.sources.values()
            }
            final_journal_names = {item.path.name for item in journals if item.finalized}
            unlinked_final = sorted(final_journal_names - source_journal_names)
            journal_issues = [
                *(
                    f"unresolved finalized journal: {item.path.name}"
                    for item in unresolved_finalized
                ),
                *(f"unlinked final journal: {name}" for name in unlinked_final),
            ]
            if strict_final:
                journal_issues.extend(
                    f"in-progress journal: {item.path.name}" for item in in_progress
                )
            status = "fail" if journal_issues else ("warn" if in_progress else "pass")
            audit.add(
                "run_journal_recovery_state",
                status,
                "critical" if journal_issues else "info",
                (
                    "Every finalized journal is linked and no uncertain or unreconciled "
                    "attempt remains."
                    if status == "pass"
                    else (
                        "A live journal is present, as expected during an incomplete "
                        "sequential run."
                        if status == "warn"
                        else "Journal recovery evidence contains unresolved or unlinked state."
                    )
                ),
                observed={
                    "journals": len(journals),
                    "finalized": len(final_journal_names),
                    "in_progress": len(in_progress),
                    "unresolved": len(unresolved_journals),
                    "unresolved_finalized": len(unresolved_finalized),
                    "unlinked_final": len(unlinked_final),
                },
                examples=journal_issues,
            )
            journal_metrics = {
                "total": len(journals),
                "finalized": len(final_journal_names),
                "in_progress": len(in_progress),
                "unresolved": len(unresolved_journals),
                "unresolved_finalized": len(unresolved_finalized),
                "unlinked_final": len(unlinked_final),
            }
        except JournalIntegrityError as error:
            audit.add(
                "run_journal_recovery_state",
                "fail",
                "critical",
                "At least one run journal fails its hash chain or recovery-state validation.",
                examples=[str(error)],
            )

        if selected_summary is not None and selected_summary.get("mode") == "execute":
            summary_issues = _summary_state_issues(selected_summary, state, work_items)
            real_execute_history = any(
                document.get("mode") == "execute" and document.get("provider_calls_made") is True
                for _, document in summary_documents
            )
            if not real_execute_history:
                summary_issues.append("no execute summary records paid provider calls")
            resolution_transition = not strict_final and _is_resolution_only_summary_transition(
                selected_summary,
                state,
                summary_issues,
            )
            summary_status = (
                "warn" if resolution_transition else ("fail" if summary_issues else "pass")
            )
            audit.add(
                "final_summary_state_reproduction",
                summary_status,
                "high" if resolution_transition else "critical",
                (
                    "The final execute summary exactly reproduces the immutable ledger, "
                    "costs, and coverage."
                    if not summary_issues
                    else (
                        "A verified no-network incident-resolution suffix follows the latest "
                        "execute summary; a resumed runner must publish the next summary."
                        if resolution_transition
                        else (
                            "The final execute summary does not reproduce the immutable final "
                            "state."
                        )
                    )
                ),
                examples=summary_issues,
            )

    external_delta_reconciled = (
        external_account_delta_usd is not None
        and state is not None
        and abs(external_account_delta_usd - state.dataset_actual_cost_usd) <= Decimal("0.000001")
    )
    governance_holds = [
        {
            "hold_id": "epicure_model_lineage",
            "severity": "high",
            "status": "open",
            "reason": (
                "The runtime release is explicitly exploratory-unmatched, not a public "
                "Cooc/Core/Chem release. It is correctly versioned for exploration but "
                "cannot support an official leaderboard until the lineage gate is resolved."
            ),
        },
        {
            "hold_id": "candidate_task_review",
            "severity": "high",
            "status": "open",
            "reason": (
                "The selected task split is candidate/pilot, not a frozen confirmatory task set."
            ),
        },
        {
            "hold_id": "human_preference_collection",
            "severity": "high",
            "status": "open",
            "reason": (
                "No public or qualified-expert pairwise judgments exist, so no "
                "Bradley–Terry or uplift estimate is authorized."
            ),
        },
        {
            "hold_id": "research_release_consent",
            "severity": "high",
            "status": "open",
            "reason": (
                "Response artifacts explicitly set research_release_eligible=false pending "
                "consent and PII/identity review."
            ),
        },
        {
            "hold_id": "external_account_delta",
            "severity": "medium",
            "status": "resolved" if external_delta_reconciled else "open",
            "reason": (
                "Independent account-level usage evidence is optional for record integrity "
                "but required for full governance reconciliation."
            ),
        },
    ]
    failure_count = len(audit.failures)
    warning_count = len(audit.warnings)
    if failure_count:
        status = "fail"
    elif not strict_final and metrics.get("pending_pairs", 0):
        status = "in_progress"
    elif warning_count:
        status = "pass_with_warnings"
    else:
        status = "pass"
    report: dict[str, Any] = {
        "schema_version": VERIFIER_SCHEMA_VERSION,
        "verified_at": _utc_now(),
        "verification_mode": "strict_final" if strict_final else "allow_incomplete",
        "status": status,
        "provider_or_mcp_calls_made_by_verifier": False,
        "inputs": {
            "artifact_root": str(root),
            "manifest": str(manifest),
            "selected_summary": str(selected_summary_path) if selected_summary_path else None,
            "selected_summary_sha256": (
                (selected_summary.get("content_address") or {}).get("digest")
                if selected_summary
                else None
            ),
            "dataset_ledger": str(dataset_ledger),
            "frontier_ledger": str(frontier_ledger_path),
        },
        "expected": {
            "models": EXPECTED_MODEL_COUNT,
            "pairs": EXPECTED_PAIR_COUNT,
            "response_opportunities": EXPECTED_RESPONSE_OPPORTUNITIES,
            "pairs_per_family": EXPECTED_FAMILY_PAIR_COUNT,
        },
        "observed": {
            **metrics,
            "stable_snapshot": stable_snapshot,
            "journals": journal_metrics,
            "cost_corrections": correction_count,
            **(
                {
                    "dataset_actual_cost_usd": _decimal_text(state.dataset_actual_cost_usd),
                    "dataset_source_exposure_usd": _decimal_text(state.dataset_source_exposure_usd),
                    "prior_effective_exposure_usd": _decimal_text(
                        state.prior_effective_exposure_usd
                    ),
                    "active_frontier_reservations_usd": _decimal_text(
                        state.prior_active_reservation_usd
                    ),
                    "orphan_dataset_reservation_usd": _decimal_text(state.orphan_reservation_usd),
                    "total_exposure_usd": _decimal_text(state.total_exposure_usd),
                }
                if state is not None
                else {}
            ),
        },
        "findings": [item.payload() for item in audit.findings],
        "finding_counts": {
            "pass": sum(item.status == "pass" for item in audit.findings),
            "warn": warning_count,
            "fail": failure_count,
        },
        "governance": {
            "official_ranking_authorized": False,
            "preference_or_uplift_fitting_authorized": False,
            "raw_research_release_authorized": False,
            "publication_readiness": "blocked",
            "open_holds": governance_holds,
            "paired_order_assessment": (
                "The collection schedules Epicure-off/on concurrently; left/right blinding "
                "is not applicable because this runner collects no preference votes."
            ),
        },
    }
    unhashed = dict(report)
    report["report_sha256"] = _sha256(unhashed)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify real FlavourBench artifacts without any provider or MCP calls. "
            "Strict final mode is the default."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", default="artifacts/real-exploratory")
    parser.add_argument("--summary")
    parser.add_argument("--prior-artifact-directory", default="artifacts/live-smoke")
    parser.add_argument("--prior-corrections-directory", default="artifacts/corrections")
    parser.add_argument("--frontier-ledger", default="artifacts/frontier-contract/ledger.jsonl")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Warn rather than fail for unfinished pairs, active reservations, or live journals",
    )
    parser.add_argument(
        "--external-account-delta-usd",
        help="Optional independently captured account-usage delta; this verifier never fetches it",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional JSON output path; stdout is always emitted and no output is written "
            "by default"
        ),
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        external_delta = (
            _decimal(args.external_account_delta_usd, field="external account delta")
            if args.external_account_delta_usd is not None
            else None
        )
        report = verify_real_exploratory_run(
            manifest_path=args.manifest,
            artifact_root=args.artifact_root,
            summary_path=args.summary,
            prior_artifact_directory=args.prior_artifact_directory,
            prior_corrections_directory=(
                args.prior_corrections_directory
                if args.prior_corrections_directory.lower() != "none"
                else None
            ),
            frontier_ledger_path=args.frontier_ledger,
            strict_final=not args.allow_incomplete,
            external_account_delta_usd=external_delta,
        )
    except (IntegrityError, JournalIntegrityError, OSError, ValueError) as error:
        report = {
            "schema_version": VERIFIER_SCHEMA_VERSION,
            "verified_at": _utc_now(),
            "verification_mode": ("allow_incomplete" if args.allow_incomplete else "strict_final"),
            "status": "fail",
            "provider_or_mcp_calls_made_by_verifier": False,
            "fatal_error": f"{type(error).__name__}: {error}",
        }
        report["report_sha256"] = _sha256(report)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 1 if report.get("status") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(run())
