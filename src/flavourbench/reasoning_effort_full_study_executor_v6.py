"""Execute one independently authorised V6 reasoning-effort block.

The executor is restart-safe across the source, journal, endpoint ledger,
coordinator ledger, canonical reservation ledger, and receipt lifecycle.  The
module is inert without the exact V6 confirmation and a different independent
reviewer's content-addressed GO.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import frontier_contract_runner as frontier
from . import reasoning_effort_full_study_executor_v1 as local
from . import reasoning_effort_full_study_v6 as study

RECEIPT_SCHEMA = "flavourbench-reasoning-effort-family-block-receipt-v6"
ATTESTATION_SCHEMA = "flavourbench-reasoning-effort-block-attestations-v6"

FullStudyExecutionError = local.FullStudyExecutionError
_load_ledger = local._load_ledger
_append_ledger = local._append_ledger
_ledger_lock = local._ledger_lock
_item_map = local._item_map
_block_map = local._block_map
_item_wave_id = local._item_wave_id
_roots = local._roots
_policy = local._policy
_endpoint_state = local._endpoint_state
_journal_evidence = local._journal_evidence
_terminalize_block = local._terminalize_block

FailureInjector = Callable[[str, str | None], None]
AttestAdapter = Callable[..., Awaitable[list[dict[str, Any]]]]
InvokeAdapter = Callable[..., Awaitable[None]]
SourceClassifier = Callable[..., dict[str, Any]]


class SimulatedCrash(BaseException):
    """Test-only process-crash cut; deliberately bypasses exception handlers."""


class MissingCanonicalSource(FullStudyExecutionError):
    """A durable item start has no safely finalizable canonical source."""


@dataclass(frozen=True)
class ExecutionAdapters:
    """Explicit boundaries for offline tests; production defaults are live."""

    attest_all: AttestAdapter | None = None
    invoke_pair: InvokeAdapter | None = None
    classify_source: SourceClassifier | None = None
    require_live_environment: Callable[[], None] | None = None
    roots: tuple[Path, Mapping[str, Path]] | None = None
    global_ledger_path: Path | None = None
    source_root: Path | None = None


@dataclass(frozen=True)
class BoundRuntime:
    policy: Any
    args: argparse.Namespace
    endpoint_wrapper: Mapping[str, Any]
    raw_execution_contract: Mapping[str, Any]
    raw_execution_contract_sha256: str


@dataclass(frozen=True)
class CanonicalDisposition:
    status: str
    reservation: Mapping[str, Any]
    artifact_event: Mapping[str, Any] | None

    @property
    def retained(self) -> bool:
        return self.status == "active_reservation"


def _global_ledger_path(plan: Mapping[str, Any], repo_root: Path) -> Path:
    return repo_root / str(plan["execution_roots"]["canonical_global_reservation_ledger"])


def _canonical_source_root(plan: Mapping[str, Any], repo_root: Path) -> Path:
    return repo_root / str(plan["execution_roots"]["canonical_global_source"])


def _inject(
    failure_injector: FailureInjector | None, stage: str, item_id: str | None = None
) -> None:
    if failure_injector is not None:
        failure_injector(stage, item_id)


def _error_record(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
    }


def _live_args(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    repo_root: Path,
    source_root: Path,
    raw_endpoint_sha256: str,
) -> argparse.Namespace:
    from .live_smoke import CONFIRMATION as LIVE_SMOKE_CONFIRMATION

    coordinate = item["route_coordinate"]
    policy = _policy(plan, item, repo_root)
    return argparse.Namespace(
        confirm=LIVE_SMOKE_CONFIRMATION,
        cap_usd=Decimal(item["worst_case_reserve_usd"]),
        model_id=coordinate["model_id"],
        provider_slug=coordinate["provider_endpoint"],
        prompt=item["task"]["prompt"],
        category=item["task"]["family"],
        skip_tool_contract=True,
        contract_only=False,
        condition=None,
        plain_text_final=True,
        tool_catalog_bytes_bound=policy.tool_catalog_bytes_bound,
        require_epicure_call=True,
        evidence_protocol=policy.evidence_protocol,
        intermediate_reasoning_effort=coordinate["intermediate_reasoning_effort"],
        final_reasoning_effort=coordinate["final_reasoning_effort"],
        output_dir=str(source_root),
        candidate_manifest_sha256=item["manifest"]["semantic_sha256"],
        sequential_arms=False,
        dataset_work_item_id=item["work_item_id"],
        dataset_task_id=item["task"]["task_id"],
        expected_canonical_model_slug=coordinate["canonical_model_slug"],
        expected_endpoint_execution_sha256=raw_endpoint_sha256,
        expected_execution_policy_sha256=policy.sha256,
        expected_epicure_release_id=plan["epicure"]["release_id"],
        expected_epicure_bundle_sha256=plan["epicure"]["bundle_sha256"],
        expected_epicure_application_sha256=plan["epicure"]["application_sha256"],
        expected_epicure_tool_schema_sha256=plan["epicure"]["tool_schema_sha256"],
        frozen_run_id=item["run_id"],
        frozen_attempt_slots=item["attempt_slots"],
    )


def _validate_live_args(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    args: argparse.Namespace,
    policy: Any,
    source_root: Path,
) -> None:
    coordinate = item["route_coordinate"]
    if (
        args.dataset_work_item_id != item["work_item_id"]
        or args.frozen_run_id != item["run_id"]
        or args.frozen_attempt_slots != item["attempt_slots"]
        or args.model_id != coordinate["model_id"]
        or args.provider_slug != coordinate["provider_endpoint"]
        or args.expected_canonical_model_slug != coordinate["canonical_model_slug"]
        or args.expected_execution_policy_sha256 != coordinate["execution_policy_sha256"]
        or args.intermediate_reasoning_effort != coordinate["intermediate_reasoning_effort"]
        or args.final_reasoning_effort != coordinate["final_reasoning_effort"]
        or args.evidence_protocol != "matched_evidence_v2"
        or args.sequential_arms is not False
        or args.require_epicure_call is not True
        or args.plain_text_final is not True
        or args.cap_usd != Decimal(item["worst_case_reserve_usd"])
        or policy.pair_arm_scheduling != "concurrent"
        or policy.max_tool_calls_total != 13
        or policy.max_tool_calls_per_round != 13
        or policy.max_tool_rounds != 3
        or policy.max_provider_attempts != 2
        or len(item.get("attempt_slots") or []) != 56
        or not re.fullmatch(r"[0-9a-f]{64}", str(args.expected_endpoint_execution_sha256))
        or Path(args.output_dir).resolve() != source_root.resolve()
    ):
        raise FullStudyExecutionError(
            f"runtime arguments differ for work item {item['work_item_id']}"
        )
    if (
        args.expected_epicure_release_id != plan["epicure"]["release_id"]
        or args.expected_epicure_bundle_sha256 != plan["epicure"]["bundle_sha256"]
        or args.expected_epicure_application_sha256 != plan["epicure"]["application_sha256"]
        or args.expected_epicure_tool_schema_sha256 != plan["epicure"]["tool_schema_sha256"]
    ):
        raise FullStudyExecutionError("runtime Epicure identity differs")


def prepare_all_runtime_items(
    *, plan: Mapping[str, Any], repo_root: Path, source_root: Path | None = None
) -> dict[str, tuple[Any, argparse.Namespace]]:
    """Resolve every import, manifest, policy, and argument before side effects."""

    resolved_source = source_root or _canonical_source_root(plan, repo_root)
    prepared: dict[str, tuple[Any, argparse.Namespace]] = {}
    for item_id, item in sorted(_item_map(plan).items()):
        policy = _policy(plan, item, repo_root)
        args = _live_args(
            plan=plan,
            item=item,
            repo_root=repo_root,
            source_root=resolved_source,
            raw_endpoint_sha256="0" * 64,
        )
        _validate_live_args(
            plan=plan,
            item=item,
            args=args,
            policy=policy,
            source_root=resolved_source,
        )
        prepared[item_id] = (policy, args)
    if len(prepared) != 168:
        raise FullStudyExecutionError("runtime preparation did not cover 168 items")
    return prepared


async def _attest_all_endpoints(
    *, plan: Mapping[str, Any], api_base: str, api_key: str
) -> list[dict[str, Any]]:
    return await local._attest_all_endpoints(plan=plan, api_base=api_base, api_key=api_key)


def _validated_attestations(
    attestations: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any], str]]:
    by_endpoint: dict[str, tuple[Mapping[str, Any], Mapping[str, Any], str]] = {}
    for wrapper in attestations:
        endpoint_id = str(wrapper.get("endpoint_id") or "")
        raw = wrapper.get("raw_execution_contract")
        raw_sha = str(wrapper.get("raw_execution_contract_sha256") or "")
        if (
            endpoint_id in by_endpoint
            or endpoint_id not in study.ENDPOINTS
            or not isinstance(raw, Mapping)
            or not re.fullmatch(r"[0-9a-f]{64}", raw_sha)
            or study._sha256(raw) != raw_sha
        ):
            raise FullStudyExecutionError("endpoint attestation hash binding differs")
        by_endpoint[endpoint_id] = (dict(wrapper), dict(raw), raw_sha)
    if set(by_endpoint) != set(study.ENDPOINTS):
        raise FullStudyExecutionError("attestations do not cover exactly three endpoints")
    return by_endpoint


def _bind_block_runtime_after_attestation(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    attestations: Sequence[Mapping[str, Any]],
    prepared_all: Mapping[str, tuple[Any, argparse.Namespace]],
    repo_root: Path,
    source_root: Path,
) -> dict[str, BoundRuntime]:
    by_endpoint = _validated_attestations(attestations)
    items = _item_map(plan)
    rebound: dict[str, BoundRuntime] = {}
    for item_id in block["work_item_ids"]:
        item = items[item_id]
        policy = prepared_all[item_id][0]
        endpoint_id = str(item["route_coordinate"]["endpoint_id"])
        wrapper, raw, raw_sha = by_endpoint[endpoint_id]
        args = _live_args(
            plan=plan,
            item=item,
            repo_root=repo_root,
            source_root=source_root,
            raw_endpoint_sha256=raw_sha,
        )
        args.expected_execution_policy_sha256 = policy.sha256
        _validate_live_args(
            plan=plan,
            item=item,
            args=args,
            policy=policy,
            source_root=source_root,
        )
        rebound[item_id] = BoundRuntime(
            policy=policy,
            args=args,
            endpoint_wrapper=wrapper,
            raw_execution_contract=raw,
            raw_execution_contract_sha256=raw_sha,
        )
    if list(rebound) != list(block["work_item_ids"]):
        raise FullStudyExecutionError("full block runtime binding order differs")
    return rebound


def _require_live_environment_before_reservation() -> None:
    from .reasoning_effort_route_gate_v4 import (
        _require_live_environment_before_reservation as require,
    )

    require()


async def _invoke_live_pair(
    *, args: argparse.Namespace, policy: Any, raw_endpoint: Mapping[str, Any]
) -> None:
    from .config import get_settings
    from .frontier_contract_runner import AdmissionDenied
    from .live_smoke import live_smoke
    from .reasoning_effort_route_gate_v4 import _policy_environment

    # The attestation wrapper is deliberately not accepted here.  Pricing and
    # provider identity are read from the hash-checked raw contract itself.
    with _policy_environment(policy=policy, endpoint=raw_endpoint):
        settings = get_settings()
        if settings.execution_mode != "live" or not settings.live_authorized:
            raise AdmissionDenied("live authority changed after reservation")
        await live_smoke(args)


def _verified_source_index(
    source_root: Path,
    *,
    current_work_item_ids: Sequence[str] | set[str] | frozenset[str],
) -> dict[str, tuple[Path, dict[str, Any], str]]:
    """Verify every source while tolerating declared historical ID shapes.

    Historical live-smoke artifacts either omit ``dataset_work_item_id`` or
    declare it as null.  Both shapes are verified but not indexed.  Every
    non-null ID must be a canonical lowercase SHA-256 and globally unique.
    """

    if source_root.exists() and (source_root.is_symlink() or not source_root.is_dir()):
        raise FullStudyExecutionError("canonical source root is not a regular directory")
    current = set(map(str, current_work_item_ids))
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in current):
        raise FullStudyExecutionError("current work-item identity set is malformed")
    indexed: dict[str, tuple[Path, dict[str, Any], str]] = {}
    seen_current_paths: dict[str, Path] = {}
    paths = sorted(source_root.glob("*.json")) if source_root.exists() else []
    for path in paths:
        try:
            artifact, digest = frontier._verify_live_artifact(path)
        except Exception as error:
            raise FullStudyExecutionError(
                f"canonical source verification failed: {path.name}"
            ) from error
        if "dataset_work_item_id" not in artifact:
            continue
        value = artifact.get("dataset_work_item_id")
        if value is None:
            continue
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise FullStudyExecutionError(
                f"canonical source has malformed non-null work-item ID: {path.name}"
            )
        if value in indexed:
            label = "current-ID ambiguity" if value in current else "duplicate historical ID"
            raise FullStudyExecutionError(f"canonical source {label}: {value}")
        indexed[value] = (path, artifact, digest)
        if value in current:
            seen_current_paths[value] = path
    if len(seen_current_paths) != len(set(seen_current_paths)):
        raise FullStudyExecutionError("canonical source current-ID ambiguity")
    return indexed


def _canonical_source_for_item(
    source_root: Path,
    work_item_id: str,
    *,
    current_work_item_ids: Sequence[str] | set[str] | frozenset[str],
):
    return _verified_source_index(source_root, current_work_item_ids=current_work_item_ids).get(
        work_item_id
    )


def _coordinator_state(
    plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    state = local._coordinator_state(plan, entries)
    blocks = _block_map(plan)
    for block_id, reservation in state["reservations"].items():
        mapping = reservation.get("canonical_reservation_entry_sha256_by_work_item")
        shas = reservation.get("canonical_reservation_entry_sha256s")
        if (
            not isinstance(mapping, Mapping)
            or set(map(str, mapping)) != set(blocks[block_id]["work_item_ids"])
            or shas != [mapping[item_id] for item_id in blocks[block_id]["work_item_ids"]]
            or len(set(map(str, shas))) != 28
        ):
            raise FullStudyExecutionError("local block lacks exact canonical reservations")
    for terminal in state["terminals"].values():
        if not all(
            isinstance(terminal.get(field), str) and re.fullmatch(r"[0-9a-f]{64}", terminal[field])
            for field in (
                "canonical_reservation_entry_sha256",
                "canonical_artifact_record_entry_sha256",
                "source_artifact_sha256",
            )
        ):
            raise FullStudyExecutionError("local terminal lacks canonical finalization binding")
    return state


def _verify_global_anchor(*, plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> None:
    anchor = plan["canonical_global_ledger_anchor"]
    sequence = int(anchor["sequence"])
    if (
        len(entries) < sequence
        or entries[sequence - 1].get("entry_sha256") != anchor["head_entry_sha256"]
    ):
        raise FullStudyExecutionError("canonical global-ledger prefix differs")


def _canonical_reservation_identity(
    *, plan: Mapping[str, Any], block: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    coordinate = item["route_coordinate"]
    return {
        "event_type": "reservation_created",
        "runner_run_id": item["run_id"],
        "manifest_sha256": item["manifest"]["semantic_sha256"],
        "model_id": coordinate["model_id"],
        "provider_tag": coordinate["provider_endpoint"],
        "reserved_usd": study._decimal_text(Decimal(item["worst_case_reserve_usd"])),
        "campaign_id": study.STUDY_ID,
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "work_item_id": item["work_item_id"],
        "reservation_role": "reasoning_effort_work_item_pair",
        "response_arms": 2,
        "replay_permitted": False,
    }


def _campaign_global_reservations(
    *, plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    items = _item_map(plan)
    blocks = _block_map(plan)
    item_to_block = {
        item_id: block_id
        for block_id, block in blocks.items()
        for item_id in block["work_item_ids"]
    }
    found: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if raw.get("campaign_id") != study.STUDY_ID:
            continue
        if raw.get("event_type") != "reservation_created":
            continue
        item_id = str(raw.get("work_item_id") or "")
        if (
            raw.get("study_plan_sha256") != plan["artifact_sha256"]
            or item_id not in items
            or item_id in found
        ):
            raise FullStudyExecutionError("forged or duplicate canonical reservation")
        block_id = item_to_block[item_id]
        expected = _canonical_reservation_identity(
            plan=plan, block=blocks[block_id], item=items[item_id]
        )
        if any(raw.get(key) != value for key, value in expected.items()):
            raise FullStudyExecutionError("canonical reservation identity differs")
        found[item_id] = dict(raw)
    return found


def _global_accounting_locked(
    *,
    plan: Mapping[str, Any],
    repo_root: Path,
    global_ledger: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    entries = frontier.load_ledger(global_ledger)
    _verify_global_anchor(plan=plan, entries=entries)
    resolved_source_root = source_root or _canonical_source_root(plan, repo_root)
    artifacts_root = repo_root / "flavourbench/artifacts"
    scan = frontier.scan_live_smoke_artifacts(
        resolved_source_root,
        corrections_directory=artifacts_root / "corrections",
    )
    frontier.validate_ledger_artifact_links(
        entries,
        scan,
        reconciliation_directory=artifacts_root / "frontier-contract/reconciliations",
    )
    artifact_by_sha = {row.artifact_sha256: row for row in scan.artifacts}
    anchor_sequence = int(plan["canonical_global_ledger_anchor"]["sequence"])
    finalized_suffix = [
        row for row in entries[anchor_sequence:] if row.get("event_type") == "artifact_recorded"
    ]
    suffix_exposure = study._exact_sum(
        [
            artifact_by_sha[str(event.get("artifact_sha256") or "")].exposure_usd
            for event in finalized_suffix
            if str(event.get("artifact_sha256") or "") in artifact_by_sha
        ]
    )
    if len(finalized_suffix) != len(
        {str(event.get("artifact_sha256") or "") for event in finalized_suffix}
    ) or any(
        str(event.get("artifact_sha256") or "") not in artifact_by_sha for event in finalized_suffix
    ):
        raise FullStudyExecutionError("global suffix artifact exposure is ambiguous")
    active = frontier.active_ledger_reservations(entries)
    active_total = study._exact_sum(list(active.values()))
    current = study._exact_add(
        study.CURRENT_EXPOSURE_USD, study._exact_add(suffix_exposure, active_total)
    )
    active_incidents = [
        dict(row)
        for row in entries[anchor_sequence:]
        if row.get("event_type") == "execution_incident"
        and row.get("reservation_entry_sha256") in active
    ]
    return {
        "entries": entries,
        "active": active,
        "baseline_exposure_usd": study._decimal_text(study.CURRENT_EXPOSURE_USD),
        "post_anchor_finalized_exposure_usd": study._decimal_text(suffix_exposure),
        "canonical_active_reservation_usd": study._decimal_text(active_total),
        "current_total_exposure_usd": study._decimal_text(current),
        "active_incidents": active_incidents,
    }


def _ensure_canonical_reservations(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    repo_root: Path,
    global_ledger: Path,
    source_root: Path | None,
    failure_injector: FailureInjector | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accounting = _global_accounting_locked(
        plan=plan,
        repo_root=repo_root,
        global_ledger=global_ledger,
        source_root=source_root,
    )
    found = _campaign_global_reservations(plan=plan, entries=accounting["entries"])
    items = _item_map(plan)
    missing = [item_id for item_id in block["work_item_ids"] if item_id not in found]
    missing_reserve = study._exact_sum(
        [Decimal(items[item_id]["worst_case_reserve_usd"]) for item_id in missing]
    )
    projected = study._exact_add(Decimal(accounting["current_total_exposure_usd"]), missing_reserve)
    foreign_incidents = [
        row for row in accounting["active_incidents"] if row.get("campaign_id") != study.STUDY_ID
    ]
    if (
        foreign_incidents
        or projected > study.ADMISSION_CEILING_USD
        or projected > study.HARD_CAP_USD
    ):
        raise frontier.AdmissionDenied("canonical shared-ledger admission is blocked")
    for item_id in missing:
        _inject(failure_injector, "before_global_reservation", item_id)
        entry = frontier.append_ledger_event(
            global_ledger,
            _canonical_reservation_identity(plan=plan, block=block, item=items[item_id]),
        )
        found[item_id] = entry
        _inject(failure_injector, "after_global_reservation", item_id)
    refreshed = _global_accounting_locked(
        plan=plan,
        repo_root=repo_root,
        global_ledger=global_ledger,
        source_root=source_root,
    )
    found = _campaign_global_reservations(plan=plan, entries=refreshed["entries"])
    ordered = [found[item_id] for item_id in block["work_item_ids"]]
    if len(ordered) != 28 or len({row["entry_sha256"] for row in ordered}) != 28:
        raise FullStudyExecutionError("canonical block reservations are incomplete")
    return ordered, refreshed


def _verify_local_global_binding(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    global_entries: Sequence[Mapping[str, Any]],
) -> None:
    found = _campaign_global_reservations(plan=plan, entries=global_entries)
    mapping = local_reservation.get("canonical_reservation_entry_sha256_by_work_item")
    if not isinstance(mapping, Mapping):
        raise FullStudyExecutionError("local reservation has no global mapping")
    expected = {item_id: found[item_id]["entry_sha256"] for item_id in block["work_item_ids"]}
    if dict(mapping) != expected or local_reservation.get(
        "canonical_reservation_entry_sha256s"
    ) != list(expected.values()):
        raise FullStudyExecutionError("local/global reservation binding differs")


def _canonical_disposition(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    global_ledger: Path,
) -> CanonicalDisposition:
    entries = frontier.load_ledger(global_ledger)
    _verify_global_anchor(plan=plan, entries=entries)
    expected = _canonical_reservation_identity(plan=plan, block=block, item=item)
    if any(canonical_reservation.get(key) != value for key, value in expected.items()):
        raise FullStudyExecutionError("canonical reservation identity differs at recovery")
    reservations = [
        row
        for row in entries
        if row.get("entry_sha256") == canonical_reservation.get("entry_sha256")
    ]
    if len(reservations) != 1:
        raise FullStudyExecutionError("canonical reservation is absent or ambiguous")
    artifacts = [
        row
        for row in entries
        if row.get("event_type") == "artifact_recorded"
        and row.get("reservation_entry_sha256") == canonical_reservation.get("entry_sha256")
    ]
    if len(artifacts) > 1:
        raise FullStudyExecutionError("canonical artifact finalization is ambiguous")
    if artifacts:
        event = artifacts[0]
        if (
            event.get("campaign_id") != study.STUDY_ID
            or event.get("study_plan_sha256") != plan["artifact_sha256"]
            or event.get("admission_block_id") != block["admission_block_id"]
            or event.get("work_item_id") != item["work_item_id"]
        ):
            raise FullStudyExecutionError("canonical artifact finalization identity differs")
        return CanonicalDisposition("artifact_recorded", reservations[0], event)
    active = frontier.active_ledger_reservations(entries)
    if canonical_reservation["entry_sha256"] not in active:
        raise FullStudyExecutionError("canonical reservation disposition is unexplained")
    return CanonicalDisposition("active_reservation", reservations[0], None)


def _source_terminal_payload(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    source_record: tuple[Path, dict[str, Any], str],
    repo_root: Path,
    endpoint_root: Path,
) -> dict[str, Any]:
    path, artifact, digest = source_record
    verified, verified_digest = frontier._verify_live_artifact(path)
    coordinate = item["route_coordinate"]
    if (
        verified_digest != digest
        or verified != artifact
        or artifact.get("run_id") != item["run_id"]
        or artifact.get("dataset_work_item_id") != item["work_item_id"]
        or artifact.get("requested_model_id") != coordinate["model_id"]
        or artifact.get("requested_provider") != coordinate["provider_endpoint"]
        or artifact.get("candidate_manifest_sha256") != item["manifest"]["semantic_sha256"]
    ):
        raise FullStudyExecutionError("canonical source identity differs")
    pair = study.pair_audit(plan=plan, item=item, source_path=path, repo_root=repo_root)
    accounting = pair.get("accounting") or {}
    if accounting.get("reconciled") is not True:
        raise FullStudyExecutionError("source generation cost is not fully reconciled")
    audit_path = study._write_artifact(
        endpoint_root / "audits",
        f"reasoning-effort-v6-pair-audit-{item['work_item_id'][:12]}",
        pair,
    )
    usable = pair.get("decision") == "passed_all_predicates"
    return {
        "disposition": "source_usable" if usable else "source_reliability_failure",
        "source_path": study._relative(repo_root, path),
        "source_artifact_sha256": digest,
        "pair_audit_path": study._relative(repo_root, audit_path),
        "pair_audit_sha256": study._regular_json(audit_path)["artifact_sha256"],
        "actual_cost_usd": study._decimal_text(Decimal(str(accounting["actual_cost_usd"]))),
        "audit_failures": list(pair.get("failures") or []),
    }


def _record_canonical_artifact(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    source_record: tuple[Path, dict[str, Any], str],
    global_ledger: Path,
) -> dict[str, Any]:
    path, artifact, digest = source_record
    entries = frontier.load_ledger(global_ledger)
    _verify_global_anchor(plan=plan, entries=entries)
    matches = [
        row
        for row in entries
        if row.get("event_type") == "artifact_recorded"
        and row.get("reservation_entry_sha256") == canonical_reservation["entry_sha256"]
    ]
    expected = {
        "event_type": "artifact_recorded",
        "runner_run_id": item["run_id"],
        "reservation_entry_sha256": canonical_reservation["entry_sha256"],
        "manifest_sha256": item["manifest"]["semantic_sha256"],
        "model_id": item["route_coordinate"]["model_id"],
        "provider_tag": item["route_coordinate"]["provider_endpoint"],
        "artifact_filename": path.name,
        "artifact_sha256": digest,
        "campaign_id": study.STUDY_ID,
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "work_item_id": item["work_item_id"],
    }
    if matches:
        if len(matches) != 1 or any(
            matches[0].get(key) != value for key, value in expected.items()
        ):
            raise FullStudyExecutionError("canonical artifact finalization differs")
        return dict(matches[0])
    if artifact.get("artifact_sha256") != digest:
        raise FullStudyExecutionError("canonical source digest differs")
    return frontier.append_ledger_event(global_ledger, expected)


def _terminal_common(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    canonical_artifact_event: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
    return {
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "task_wave_id": _item_wave_id(plan, item_id),
        "work_item_id": item_id,
        "block_reservation_entry_sha256": local_reservation["entry_sha256"],
        "canonical_reservation_entry_sha256": canonical_reservation["entry_sha256"],
        "canonical_artifact_record_entry_sha256": canonical_artifact_event["entry_sha256"],
        **dict(payload),
        "replay_permitted": False,
        "rank_eligible": False,
    }


def _coordinator_terminal_from_endpoint(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    endpoint_terminal: Mapping[str, Any],
    coordinator_ledger: Path,
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
    if (
        endpoint_terminal.get("study_plan_sha256") != plan["artifact_sha256"]
        or endpoint_terminal.get("admission_block_id") != block["admission_block_id"]
        or endpoint_terminal.get("work_item_id") != item_id
        or endpoint_terminal.get("block_reservation_entry_sha256")
        != local_reservation["entry_sha256"]
        or endpoint_terminal.get("canonical_reservation_entry_sha256")
        != canonical_reservation["entry_sha256"]
        or endpoint_terminal.get("disposition") not in local.SOURCE_DISPOSITIONS
    ):
        raise FullStudyExecutionError("endpoint terminal recovery identity differs")
    protected = {
        "schema_version",
        "ledger_role",
        "sequence",
        "recorded_at",
        "previous_entry_sha256",
        "entry_sha256",
        "event_type",
    }
    common = {key: value for key, value in endpoint_terminal.items() if key not in protected}
    expected = {
        **common,
        "endpoint_id": item["route_coordinate"]["endpoint_id"],
        "variant_id": item["route_coordinate"]["variant_id"],
        "endpoint_terminal_entry_sha256": endpoint_terminal["entry_sha256"],
    }
    state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
    existing = state["terminals"].get(item_id)
    if existing is not None:
        if any(existing.get(key) != value for key, value in expected.items()):
            raise FullStudyExecutionError("coordinator terminal recovery binding differs")
        return dict(existing)
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={"event_type": "family_block_item_terminalized", **expected},
    )


def _append_local_terminal_idempotent(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    canonical_artifact_event: Mapping[str, Any],
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    payload: Mapping[str, Any],
    failure_injector: FailureInjector | None,
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
    common = _terminal_common(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical_reservation,
        canonical_artifact_event=canonical_artifact_event,
        payload=payload,
    )
    state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
    incident = state["incidents"].get(item_id)
    if incident is not None:
        raise FullStudyExecutionError("endpoint incident precludes terminal append")
    endpoint_terminal = state["terminals"].get(item_id)
    if endpoint_terminal is None:
        endpoint_terminal = _append_ledger(
            endpoint_ledger,
            role="endpoint",
            event={"event_type": "source_terminalized", **common},
        )
    elif any(endpoint_terminal.get(key) != value for key, value in common.items()):
        raise FullStudyExecutionError("endpoint terminal recovery binding differs")
    _inject(
        failure_injector,
        "after_endpoint_terminal_before_coordinator_terminal",
        item_id,
    )
    return _coordinator_terminal_from_endpoint(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical_reservation,
        endpoint_terminal=endpoint_terminal,
        coordinator_ledger=coordinator_ledger,
    )


_LEDGER_PROTECTED = {
    "schema_version",
    "ledger_role",
    "sequence",
    "recorded_at",
    "previous_entry_sha256",
    "entry_sha256",
    "event_type",
}


def _validate_endpoint_incident_identity(
    *,
    incident: Mapping[str, Any],
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
) -> None:
    if (
        incident.get("study_plan_sha256") != plan["artifact_sha256"]
        or incident.get("admission_block_id") != block["admission_block_id"]
        or incident.get("work_item_id") != item["work_item_id"]
        or incident.get("block_reservation_entry_sha256") != local_reservation["entry_sha256"]
        or incident.get("canonical_reservation_entry_sha256")
        != canonical_reservation["entry_sha256"]
        or incident.get("incident") != "durable_post_start_without_finalizable_canonical_source"
        or incident.get("replay_permitted") is not False
    ):
        raise FullStudyExecutionError("endpoint incident recovery identity differs")


def _replay_endpoint_incident_exactly(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    endpoint_incident: Mapping[str, Any],
    coordinator_ledger: Path,
) -> dict[str, Any]:
    _validate_endpoint_incident_identity(
        incident=endpoint_incident,
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical_reservation,
    )
    payload = {
        key: value for key, value in endpoint_incident.items() if key not in _LEDGER_PROTECTED
    }
    expected = {
        **payload,
        "endpoint_incident_entry_sha256": endpoint_incident["entry_sha256"],
    }
    state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
    matches = [
        row
        for row in state["incidents"].get(block["admission_block_id"], [])
        if row.get("work_item_id") == item["work_item_id"]
    ]
    if matches:
        if len(matches) != 1 or any(
            matches[0].get(key) != value for key, value in expected.items()
        ):
            raise FullStudyExecutionError("coordinator incident replay differs")
        return dict(matches[0])
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={"event_type": "family_block_execution_incident", **expected},
    )


def _append_incident_idempotent(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    global_ledger: Path,
    evidence: Mapping[str, Any],
    error: BaseException,
    failure_injector: FailureInjector | None = None,
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
    endpoint_state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
    endpoint_terminal = endpoint_state["terminals"].get(item_id)
    if endpoint_terminal is not None:
        # Terminal state wins.  Never corrupt the endpoint ledger by appending
        # an incident after it.
        return _coordinator_terminal_from_endpoint(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            endpoint_terminal=endpoint_terminal,
            coordinator_ledger=coordinator_ledger,
        )
    existing = endpoint_state["incidents"].get(item_id)
    if existing is not None:
        disposition = _canonical_disposition(
            plan=plan,
            block=block,
            item=item,
            canonical_reservation=canonical_reservation,
            global_ledger=global_ledger,
        )
        expected_artifact_event = (
            disposition.artifact_event.get("entry_sha256")
            if disposition.artifact_event is not None
            else None
        )
        if (
            existing.get("canonical_reservation_status") != disposition.status
            or existing.get("canonical_reservation_retained") is not disposition.retained
            or existing.get("canonical_artifact_record_entry_sha256") != expected_artifact_event
            or Decimal(str(existing.get("work_item_reserve_retained_usd") or "0"))
            != (Decimal(item["worst_case_reserve_usd"]) if disposition.retained else Decimal(0))
        ):
            raise FullStudyExecutionError(
                "endpoint incident canonical disposition no longer verifies"
            )
        return _replay_endpoint_incident_exactly(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            endpoint_incident=existing,
            coordinator_ledger=coordinator_ledger,
        )
    disposition = _canonical_disposition(
        plan=plan,
        block=block,
        item=item,
        canonical_reservation=canonical_reservation,
        global_ledger=global_ledger,
    )
    common = {
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "task_wave_id": _item_wave_id(plan, item_id),
        "work_item_id": item_id,
        "block_reservation_entry_sha256": local_reservation["entry_sha256"],
        "canonical_reservation_entry_sha256": canonical_reservation["entry_sha256"],
        "incident": "durable_post_start_without_finalizable_canonical_source",
        "journal_evidence": dict(evidence),
        **_error_record(error),
        "canonical_reservation_status": disposition.status,
        "canonical_artifact_record_entry_sha256": (
            disposition.artifact_event.get("entry_sha256")
            if disposition.artifact_event is not None
            else None
        ),
        "work_item_reserve_retained_usd": (
            item["worst_case_reserve_usd"] if disposition.retained else "0"
        ),
        "canonical_reservation_retained": disposition.retained,
        "replay_permitted": False,
    }
    endpoint_incident = _append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={"event_type": "uncertain_execution_incident", **common},
    )
    _inject(
        failure_injector,
        "after_endpoint_incident_before_coordinator_incident",
        item_id,
    )
    return _replay_endpoint_incident_exactly(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical_reservation,
        endpoint_incident=endpoint_incident,
        coordinator_ledger=coordinator_ledger,
    )


def _safe_journal_evidence(source_root: Path, item_id: str) -> dict[str, Any]:
    try:
        return _journal_evidence(source_root, item_id)
    except Exception as error:
        return {
            "journal_count": None,
            "request_started_count": None,
            "journals": [],
            "evidence_error": _error_record(error),
        }


def _outcome_from_terminal(terminal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "work_item_id": terminal["work_item_id"],
        "decision": terminal["disposition"],
        "terminal_entry_sha256": terminal["entry_sha256"],
    }


async def _classification_fence(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    prepared_block: Mapping[str, BoundRuntime],
    repo_root: Path,
    source_root: Path,
    endpoint_root: Path,
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    global_ledger: Path,
    adapters: ExecutionAdapters,
    failure_injector: FailureInjector | None,
) -> dict[str, Any]:
    """Classify every operation after a durable item start in one fence."""

    item_id = str(item["work_item_id"])
    current_ids = frozenset(_item_map(plan))
    classifier = adapters.classify_source or _source_terminal_payload
    try:
        endpoint_state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
        endpoint_terminal = endpoint_state["terminals"].get(item_id)
        if endpoint_terminal is not None:
            recovered = _coordinator_terminal_from_endpoint(
                plan=plan,
                block=block,
                item=item,
                local_reservation=local_reservation,
                canonical_reservation=canonical_reservation,
                endpoint_terminal=endpoint_terminal,
                coordinator_ledger=coordinator_ledger,
            )
            return _outcome_from_terminal(recovered)
        endpoint_incident = endpoint_state["incidents"].get(item_id)
        if endpoint_incident is not None:
            incident = _replay_endpoint_incident_exactly(
                plan=plan,
                block=block,
                item=item,
                local_reservation=local_reservation,
                canonical_reservation=canonical_reservation,
                endpoint_incident=endpoint_incident,
                coordinator_ledger=coordinator_ledger,
            )
            return {
                "work_item_id": item_id,
                "decision": "durable_incident_reservation_derived",
                "incident_entry_sha256": incident["entry_sha256"],
            }

        # These lookups are intentionally inside the classification fence.  A
        # missing dependency on restart therefore becomes a durable incident.
        runtime = prepared_block[item_id]
        if item_id not in endpoint_state["started"]:
            _inject(failure_injector, "before_item_start", item_id)
            _append_ledger(
                endpoint_ledger,
                role="endpoint",
                event={
                    "event_type": "item_execution_started",
                    "study_plan_sha256": plan["artifact_sha256"],
                    "admission_block_id": block["admission_block_id"],
                    "task_wave_id": _item_wave_id(plan, item_id),
                    "work_item_id": item_id,
                    "run_id": item["run_id"],
                    "endpoint_id": item["route_coordinate"]["endpoint_id"],
                    "variant_id": item["route_coordinate"]["variant_id"],
                    "block_reservation_entry_sha256": local_reservation["entry_sha256"],
                    "canonical_reservation_entry_sha256": canonical_reservation["entry_sha256"],
                    "raw_endpoint_execution_sha256": runtime.raw_execution_contract_sha256,
                    "replay_permitted": False,
                },
            )
            _inject(failure_injector, "after_item_start", item_id)

        _inject(failure_injector, "before_source_lookup", item_id)
        source = _canonical_source_for_item(source_root, item_id, current_work_item_ids=current_ids)
        if source is None:
            evidence = _safe_journal_evidence(source_root, item_id)
            if evidence["request_started_count"]:
                raise MissingCanonicalSource(
                    "provider request may have started without a canonical source"
                )
            _inject(failure_injector, "before_provider_invocation", item_id)
            if adapters.invoke_pair is None:
                await _invoke_live_pair(
                    args=runtime.args,
                    policy=runtime.policy,
                    raw_endpoint=runtime.raw_execution_contract,
                )
            else:
                await adapters.invoke_pair(
                    args=runtime.args,
                    policy=runtime.policy,
                    raw_endpoint=runtime.raw_execution_contract,
                )
            _inject(
                failure_injector,
                "after_provider_invocation_before_source_lookup",
                item_id,
            )
            source = _canonical_source_for_item(
                source_root, item_id, current_work_item_ids=current_ids
            )
        if source is None:
            raise MissingCanonicalSource("pair invocation produced no canonical source")
        _inject(failure_injector, "before_source_classification", item_id)
        payload = classifier(
            plan=plan,
            item=item,
            source_record=source,
            repo_root=repo_root,
            endpoint_root=endpoint_root,
        )
        _inject(failure_injector, "after_source_classification", item_id)
        _inject(failure_injector, "before_global_artifact_finalization", item_id)
        artifact_event = _record_canonical_artifact(
            plan=plan,
            block=block,
            item=item,
            canonical_reservation=canonical_reservation,
            source_record=source,
            global_ledger=global_ledger,
        )
        _inject(failure_injector, "after_global_artifact_finalization", item_id)
        _inject(failure_injector, "before_local_terminal", item_id)
        terminal = _append_local_terminal_idempotent(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            canonical_artifact_event=artifact_event,
            endpoint_ledger=endpoint_ledger,
            coordinator_ledger=coordinator_ledger,
            payload=payload,
            failure_injector=failure_injector,
        )
        _inject(failure_injector, "after_coordinator_terminal", item_id)
        return _outcome_from_terminal(terminal)
    except SimulatedCrash:
        raise
    except Exception as error:
        # First recover a terminal, if the failed operation actually committed
        # it.  This branch never appends an incident after an endpoint terminal.
        try:
            endpoint_state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
            endpoint_terminal = endpoint_state["terminals"].get(item_id)
            if endpoint_terminal is not None:
                recovered = _coordinator_terminal_from_endpoint(
                    plan=plan,
                    block=block,
                    item=item,
                    local_reservation=local_reservation,
                    canonical_reservation=canonical_reservation,
                    endpoint_terminal=endpoint_terminal,
                    coordinator_ledger=coordinator_ledger,
                )
                return _outcome_from_terminal(recovered)
            existing_incident = endpoint_state["incidents"].get(item_id)
            if existing_incident is not None:
                incident = _replay_endpoint_incident_exactly(
                    plan=plan,
                    block=block,
                    item=item,
                    local_reservation=local_reservation,
                    canonical_reservation=canonical_reservation,
                    endpoint_incident=existing_incident,
                    coordinator_ledger=coordinator_ledger,
                )
                return {
                    "work_item_id": item_id,
                    "decision": "durable_incident_reservation_derived",
                    "incident_entry_sha256": incident["entry_sha256"],
                }

            # A canonical artifact_recorded event finalizes the reserve.  If a
            # normal exception occurred immediately afterwards, complete the
            # local terminal from that exact lifecycle instead of claiming the
            # reservation remains active.
            disposition = _canonical_disposition(
                plan=plan,
                block=block,
                item=item,
                canonical_reservation=canonical_reservation,
                global_ledger=global_ledger,
            )
            if disposition.artifact_event is not None:
                source = _canonical_source_for_item(
                    source_root, item_id, current_work_item_ids=current_ids
                )
                if source is not None:
                    payload = classifier(
                        plan=plan,
                        item=item,
                        source_record=source,
                        repo_root=repo_root,
                        endpoint_root=endpoint_root,
                    )
                    terminal = _append_local_terminal_idempotent(
                        plan=plan,
                        block=block,
                        item=item,
                        local_reservation=local_reservation,
                        canonical_reservation=canonical_reservation,
                        canonical_artifact_event=disposition.artifact_event,
                        endpoint_ledger=endpoint_ledger,
                        coordinator_ledger=coordinator_ledger,
                        payload=payload,
                        failure_injector=None,
                    )
                    return _outcome_from_terminal(terminal)
        except SimulatedCrash:
            raise
        except Exception:
            # The incident below derives reserve state afresh from the ledger.
            pass
        incident = _append_incident_idempotent(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            endpoint_ledger=endpoint_ledger,
            coordinator_ledger=coordinator_ledger,
            global_ledger=global_ledger,
            evidence=_safe_journal_evidence(source_root, item_id),
            error=error,
            failure_injector=failure_injector,
        )
        if incident.get("event_type") == "family_block_item_terminalized":
            return _outcome_from_terminal(incident)
        return {
            "work_item_id": item_id,
            "decision": "durable_incident_reservation_derived",
            "incident_entry_sha256": incident["entry_sha256"],
        }


def _receipt_prefix(block: Mapping[str, Any]) -> str:
    return f"reasoning-effort-v6-block-{int(block['block_ordinal']):02d}-receipt"


def _receipt_document(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    governance_go: Mapping[str, Any],
    block: Mapping[str, Any],
    state: Mapping[str, Any],
    global_ledger: Path,
) -> dict[str, Any]:
    block_id = str(block["admission_block_id"])
    block_terminal = state["completed"].get(block_id)
    if block_terminal is None:
        raise FullStudyExecutionError("cannot write receipt for nonterminal block")
    terminals = [state["terminals"].get(item_id) for item_id in block["work_item_ids"]]
    if any(value is None for value in terminals):
        raise FullStudyExecutionError("receipt terminal set is incomplete")
    entries = frontier.load_ledger(global_ledger)
    _verify_global_anchor(plan=plan, entries=entries)
    reservations = _campaign_global_reservations(plan=plan, entries=entries)
    lifecycle: list[dict[str, Any]] = []
    for item_id, terminal in zip(block["work_item_ids"], terminals, strict=True):
        assert terminal is not None
        reservation = reservations[item_id]
        artifact_matches = [
            row
            for row in entries
            if row.get("event_type") == "artifact_recorded"
            and row.get("reservation_entry_sha256") == reservation["entry_sha256"]
        ]
        if len(artifact_matches) != 1 or artifact_matches[0].get("entry_sha256") != terminal.get(
            "canonical_artifact_record_entry_sha256"
        ):
            raise FullStudyExecutionError("receipt canonical lifecycle differs")
        lifecycle.append(
            {
                "work_item_id": item_id,
                "canonical_reservation_entry_sha256": reservation["entry_sha256"],
                "canonical_artifact_record_entry_sha256": artifact_matches[0]["entry_sha256"],
                "source_artifact_sha256": artifact_matches[0]["artifact_sha256"],
                "endpoint_terminal_entry_sha256": terminal["endpoint_terminal_entry_sha256"],
                "coordinator_terminal_entry_sha256": terminal["entry_sha256"],
            }
        )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "record_role": "recoverable_shared_ledger_family_block_execution",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "bound_preflight_sha256": bound_preflight["artifact_sha256"],
        "governance_go_sha256": governance_go["artifact_sha256"],
        "admission_block_id": block_id,
        "block_terminal_entry_sha256": block_terminal["entry_sha256"],
        "canonical_reservation_count": 28,
        "outcomes": [_outcome_from_terminal(value) for value in terminals if value],
        "canonical_lifecycle": lifecycle,
        "block_terminal": True,
        "durable_incident": False,
        "provider_substitution_performed": False,
        "rank_eligible": False,
    }


def _ensure_terminal_receipt(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    governance_go: Mapping[str, Any],
    block: Mapping[str, Any],
    state: Mapping[str, Any],
    coordinator_root: Path,
    global_ledger: Path,
) -> tuple[Path, dict[str, Any]]:
    document = _receipt_document(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        governance_go=governance_go,
        block=block,
        state=state,
        global_ledger=global_ledger,
    )
    expected_sha = study._sha256(document)
    existing = sorted((coordinator_root / "receipts").glob(f"{_receipt_prefix(block)}-*.json"))
    if existing:
        if len(existing) != 1:
            raise FullStudyExecutionError("terminal block receipt is ambiguous")
        observed = study._regular_json(existing[0])
        body = {key: value for key, value in observed.items() if key != "artifact_sha256"}
        if observed.get("artifact_sha256") != expected_sha or body != document:
            raise FullStudyExecutionError("terminal block receipt differs")
        return existing[0], observed
    path = study._write_artifact(coordinator_root / "receipts", _receipt_prefix(block), document)
    return path, study._regular_json(path)


async def execute_one_block(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    governance_go: Mapping[str, Any],
    repo_root: Path,
    api_base: str,
    api_key: str,
    failure_injector: FailureInjector | None = None,
    adapters: ExecutionAdapters | None = None,
) -> dict[str, Any]:
    study.validate_plan(plan, repo_root=repo_root)
    study.verify_bound_preflight(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
    )
    study.verify_governance_go(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        governance_go=governance_go,
    )
    boundary = adapters or ExecutionAdapters()
    if boundary.roots is None:
        coordinator_root, endpoint_roots = _roots(plan, repo_root)
    else:
        coordinator_root, raw_endpoint_roots = boundary.roots
        endpoint_roots = {str(key): Path(value) for key, value in raw_endpoint_roots.items()}
        if set(endpoint_roots) != set(study.ENDPOINTS):
            raise FullStudyExecutionError("adapter roots do not cover all endpoints")
    coordinator_ledger = coordinator_root / "ledger.jsonl"
    global_ledger = boundary.global_ledger_path or _global_ledger_path(plan, repo_root)
    source_root = boundary.source_root or _canonical_source_root(plan, repo_root)
    blocks = _block_map(plan)
    items = _item_map(plan)

    # Receipt recovery precedes block selection, endpoint attestation, source
    # scanning, reservation, and provider invocation.  A crash after the block
    # terminal event can therefore never advance to the next block.
    preliminary_state = _coordinator_state(
        plan, _load_ledger(coordinator_ledger, role="coordinator")
    )
    authorized_id = str(governance_go.get("authorized_admission_block_id") or "")
    if authorized_id in preliminary_state["completed"]:
        path, document = _ensure_terminal_receipt(
            plan=plan,
            human_protocol=human_protocol,
            bound_preflight=bound_preflight,
            governance_go=governance_go,
            block=blocks[authorized_id],
            state=preliminary_state,
            coordinator_root=coordinator_root,
            global_ledger=global_ledger,
        )
        return {
            "decision": "block_terminal_receipt_recovered",
            "document": document,
            "receipt_path": str(path),
            "outcomes": document["outcomes"],
        }
    if preliminary_state["active_block_id"]:
        target_id = str(preliminary_state["active_block_id"])
    else:
        completed = len(preliminary_state["completed"])
        if completed == 6:
            return {"decision": "all_blocks_terminal", "outcomes": []}
        target_id = str(plan["block_execution_order"][completed])
    block = blocks[target_id]
    if authorized_id != target_id:
        raise FullStudyExecutionError("independent GO does not authorize the target block")
    if preliminary_state["incidents"].get(target_id):
        return {
            "decision": "blocked_by_durable_incident",
            "admission_block_id": target_id,
            "outcomes": [],
        }

    # Resolve all production symbols, manifests, policies, and runtime argument
    # shapes before any item start or reservation.
    prepared_all = prepare_all_runtime_items(
        plan=plan, repo_root=repo_root, source_root=source_root
    )
    _inject(failure_injector, "after_all_runtime_validation_before_side_effect")
    require_environment = (
        boundary.require_live_environment or _require_live_environment_before_reservation
    )
    require_environment()
    _verified_source_index(source_root, current_work_item_ids=frozenset(items))

    _inject(failure_injector, "before_endpoint_attestations")
    attest = boundary.attest_all or _attest_all_endpoints
    attestations = await attest(plan=plan, api_base=api_base, api_key=api_key)
    _inject(failure_injector, "after_endpoint_attestations")
    prepared_block = _bind_block_runtime_after_attestation(
        plan=plan,
        block=block,
        attestations=attestations,
        prepared_all=prepared_all,
        repo_root=repo_root,
        source_root=source_root,
    )
    attestation_document = {
        "schema_version": ATTESTATION_SCHEMA,
        "record_role": "raw_contract_unwrapped_hash_bound_prestart_dependency_attestation",
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": target_id,
        "records": list(attestations),
        "record_sha256s": [study._sha256(record) for record in attestations],
        "raw_execution_contract_sha256_by_endpoint": {
            endpoint_id: runtime.raw_execution_contract_sha256
            for endpoint_id in study.ENDPOINTS
            for runtime in [
                next(
                    value
                    for item_id, value in prepared_block.items()
                    if items[item_id]["route_coordinate"]["endpoint_id"] == endpoint_id
                )
            ]
        },
        "bound_runtime_work_item_ids": list(prepared_block),
        "bound_runtime_count": 28,
        "catalog_http_gets": sum(
            int(record.get("catalog_http_gets") or 0) for record in attestations
        ),
        "provider_completion_requests": 0,
        "epicure_calls": 0,
    }
    attestation_path = study._write_artifact(
        coordinator_root / "endpoint-attestations",
        f"reasoning-effort-v6-block-{block['block_ordinal']:02d}-attestations",
        attestation_document,
    )
    attestation_ref = study._file_ref(repo_root, attestation_path)
    _inject(failure_injector, "after_attestation_binding_before_global_lock")

    outcomes: list[dict[str, Any]] = []
    receipt_path: Path | None = None
    receipt_document: dict[str, Any] | None = None
    with frontier._exclusive_runner_lock(global_ledger):
        _inject(failure_injector, "after_global_lock_before_reservations")
        with _ledger_lock(coordinator_ledger):
            state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
            if state["active_block_id"] not in {None, target_id}:
                raise FullStudyExecutionError("target block changed during attestation")
            existing_local = state["reservations"].get(target_id)
            if existing_local is not None:
                _verify_local_global_binding(
                    plan=plan,
                    block=block,
                    local_reservation=existing_local,
                    global_entries=frontier.load_ledger(global_ledger),
                )
            canonical_reservations, accounting = _ensure_canonical_reservations(
                plan=plan,
                block=block,
                repo_root=repo_root,
                global_ledger=global_ledger,
                source_root=source_root,
                failure_injector=failure_injector,
            )
            canonical_by_item = dict(
                zip(block["work_item_ids"], canonical_reservations, strict=True)
            )
            state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
            local_reservation = state["reservations"].get(target_id)
            if local_reservation is None:
                _inject(failure_injector, "before_local_block_reservation")
                local_reservation = _append_ledger(
                    coordinator_ledger,
                    role="coordinator",
                    event={
                        "event_type": "family_block_reservation_created",
                        "study_plan_sha256": plan["artifact_sha256"],
                        "admission_block_id": target_id,
                        "block_ordinal": block["block_ordinal"],
                        "wave_ids": block["wave_ids"],
                        "task_ids": block["task_ids"],
                        "task_families": block["task_families"],
                        "work_item_ids": block["work_item_ids"],
                        "reserved_usd": block["worst_case_reserve_usd"],
                        "canonical_reservation_entry_sha256_by_work_item": {
                            item_id: canonical_by_item[item_id]["entry_sha256"]
                            for item_id in block["work_item_ids"]
                        },
                        "canonical_reservation_entry_sha256s": [
                            canonical_by_item[item_id]["entry_sha256"]
                            for item_id in block["work_item_ids"]
                        ],
                        "endpoint_attestation": attestation_ref,
                        "global_accounting_at_admission": {
                            key: value
                            for key, value in accounting.items()
                            if key not in {"entries", "active"}
                        },
                        "replay_permitted": False,
                    },
                )
                _inject(failure_injector, "after_local_block_reservation")
            _verify_local_global_binding(
                plan=plan,
                block=block,
                local_reservation=local_reservation,
                global_entries=frontier.load_ledger(global_ledger),
            )

            for item_id in block["work_item_ids"]:
                state = _coordinator_state(
                    plan, _load_ledger(coordinator_ledger, role="coordinator")
                )
                if item_id in state["terminals"]:
                    continue
                if state["incidents"].get(target_id):
                    break
                item = items[item_id]
                endpoint_id = str(item["route_coordinate"]["endpoint_id"])
                endpoint_root = endpoint_roots[endpoint_id]
                outcome = await _classification_fence(
                    plan=plan,
                    block=block,
                    item=item,
                    local_reservation=local_reservation,
                    canonical_reservation=canonical_by_item[item_id],
                    prepared_block=prepared_block,
                    repo_root=repo_root,
                    source_root=source_root,
                    endpoint_root=endpoint_root,
                    endpoint_ledger=endpoint_root / "ledger.jsonl",
                    coordinator_ledger=coordinator_ledger,
                    global_ledger=global_ledger,
                    adapters=boundary,
                    failure_injector=failure_injector,
                )
                outcomes.append(outcome)
                if outcome["decision"] == "durable_incident_reservation_derived":
                    break
            state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
            if not state["incidents"].get(target_id) and all(
                item_id in state["terminals"] for item_id in block["work_item_ids"]
            ):
                _inject(failure_injector, "before_block_terminal")
                _terminalize_block(plan=plan, block=block, coordinator_ledger=coordinator_ledger)
                _inject(failure_injector, "after_block_terminal")
            final_state = _coordinator_state(
                plan, _load_ledger(coordinator_ledger, role="coordinator")
            )
            if target_id in final_state["completed"]:
                receipt_path, receipt_document = _ensure_terminal_receipt(
                    plan=plan,
                    human_protocol=human_protocol,
                    bound_preflight=bound_preflight,
                    governance_go=governance_go,
                    block=block,
                    state=final_state,
                    coordinator_root=coordinator_root,
                    global_ledger=global_ledger,
                )

    if receipt_document is not None and receipt_path is not None:
        return {
            "decision": "block_terminal",
            "document": receipt_document,
            "receipt_path": str(receipt_path),
            "outcomes": receipt_document["outcomes"],
        }
    return {
        "decision": "durable_incident_stop",
        "admission_block_id": target_id,
        "outcomes": outcomes,
    }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--bound-preflight", type=Path, required=True)
    parser.add_argument("--governance-go", type=Path, required=True)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--confirm")
    arguments = parser.parse_args(argv)
    if arguments.confirm != study.CONFIRMATION:
        raise SystemExit("exact V6 independently reviewed one-block confirmation is required")
    repo_root = arguments.repo_root.resolve()
    result = asyncio.run(
        execute_one_block(
            plan=study._regular_json(arguments.plan.resolve()),
            human_protocol=study._regular_json(arguments.human_protocol.resolve()),
            bound_preflight=study._regular_json(arguments.bound_preflight.resolve()),
            governance_go=study._regular_json(arguments.governance_go.resolve()),
            repo_root=repo_root,
            api_base=arguments.api_base,
            api_key=arguments.api_key,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
