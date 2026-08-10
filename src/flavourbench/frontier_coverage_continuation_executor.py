"""Fail-closed executor and source-reconstructing auditor for coverage continuation.

The command accepts only the exact v2 continuation and v3 post-failure
replacement plans.  Planning is deterministic and network-free.  Paid
execution is sequential by cell, uses separate append-only v2/v3 ledgers under
the shared frontier budget mutex, and permanently closes on the first
incomplete cell.  Every output remains development-only and may be used for
comparison-graph diagnostics, never an official preference or uplift fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .frontier_contract_runner import (
    AUTHORIZED_TOTAL_CAP_USD,
    DEFAULT_ADMISSION_FRACTION,
    AdmissionDenied,
    IntegrityError,
    _exclusive_runner_lock,
    _extract_artifact_path,
    _safe_process_hash,
    _verify_live_artifact,
    load_candidate_manifest,
    select_candidates,
)
from .frontier_coverage_continuation import (
    CONTINUATION_SCHEMA_VERSION,
    REPLACEMENT_SCHEMA_VERSION,
    RETIRED_WORK_ITEM_IDS,
    append_guarded_continuation_reservation,
    build_stopped_run_audit,
    require_prefixed_credential_before_reservation,
    verify_orphan_closure,
)
from .frontier_coverage_repair_executor import (
    HARD_POSTFLIGHT_EXEMPTIONS,
    SupplementalRun,
    _decimal,
    _decimal_text,
    _global_ledger_state,
    _policy_from_document,
    _run_accounting,
    _verify_budget_audit,
)
from .real_dataset_runner import (
    DatasetSource,
    WorkItem,
    _dataset_ledger_lock,
    _source_postflight_issues,
    _subprocess_command,
    append_dataset_ledger_event,
    dataset_ledger_state,
    derive_conditions_forecast,
    load_dataset_ledger,
    load_development_task_inventory,
    normalise_source_responses,
    scan_response_artifacts,
    task_registry_sha256,
)
from .real_task_bank import sha256_json
from .response_envelope_route_v4 import verify_v4_route_acceptance_paths
from .run_journal import scan_recovery_journals

MATERIALIZATION_SCHEMA_VERSION = (
    "flavourbench-frontier-coverage-continuation-materialization-v1"
)
PREFLIGHT_SCHEMA_VERSION = "flavourbench-frontier-coverage-continuation-preflight-v1"
RECEIPT_SCHEMA_VERSION = "flavourbench-frontier-coverage-continuation-receipt-v1"
CLOSURE_SCHEMA_VERSION = "flavourbench-frontier-coverage-continuation-closure-v1"
POSTRUN_AUDIT_SCHEMA_VERSION = (
    "flavourbench-frontier-coverage-continuation-postrun-audit-v1"
)
EXECUTION_CONFIRMATION = "RUN_EXACT_V2_V3_COVERAGE_CONTINUATION_18_REAL_ARMS"

V2_PLAN_SHA256 = "e9f4375f8976ec7468d436ff1ade21642d6746a6eca1722f4355cdd96be19646"
V3_PLAN_SHA256 = "3baff4ae405b0dbe4eb5168a5a088b29cb9438c86b01ce3d5a5be670839d14ee"
V1_MATERIALIZATION_SHA256 = (
    "eb27d59a5ec474f3b7975ea4649217182054f92d4b40bd6efbf3f1e4567b029f"
)
STOPPED_AUDIT_SHA256 = (
    "b0990b3b8869325771433cccd8a390a0e48038cf07637cac7ee244a39e9ca4d5"
)
ORPHAN_CLOSURE_SHA256 = (
    "3cb144abd1162447e3e64ba0b703ea09d9ead595d141e2dbf1ffb0103d27e370"
)
V4_ROUTE_PLAN_SHA256 = (
    "a3ef7434064415c93ab78fe818339e0466b100bee01e10e67cbdf1e4d848a4d6"
)
V4_ROUTE_AUDIT_SHA256 = (
    "70fb6f9389885059f0ddf9bb6868ffe846ebcd48df67644a34075b9043dd32c3"
)
V4_ROUTE_CLOSURE_SHA256 = (
    "dfb54062b304b31c52f69a9698d6ffeda39f38f7bdf749d60fc9554f0d15078c"
)
EXPECTED_V2_CELLS = 6
EXPECTED_V3_CELLS = 3
EXPECTED_ARMS = 18

SOURCE_FILES = (
    "src/flavourbench/frontier_coverage_continuation_executor.py",
    "src/flavourbench/continuation_openrouter_pair.py",
    "src/flavourbench/frontier_coverage_continuation.py",
    "src/flavourbench/frontier_coverage_repair_executor.py",
    "src/flavourbench/real_dataset_runner.py",
    "src/flavourbench/live_smoke.py",
    "src/flavourbench/direct_kimi_pair.py",
    "src/flavourbench/direct_cohere_pair.py",
    "src/flavourbench/provider.py",
    "src/flavourbench/service_cohere.py",
    "src/flavourbench/run_journal.py",
)


@dataclass(frozen=True)
class ContinuationCell:
    plan_kind: str
    plan_sha256: str
    cell_id: str
    run_id: str
    arm_ids: Mapping[str, str]
    attempt_slots: tuple[Mapping[str, Any], ...]
    work_item: WorkItem
    route_manifest_path: Path
    route_manifest_sha256: str
    forecast_usd: Decimal
    source_work_item_id: str

    @property
    def conditions(self) -> tuple[str, ...]:
        return ("epicure_off", "epicure_on")

    def public_payload(self) -> dict[str, Any]:
        return {
            "plan_kind": self.plan_kind,
            "plan_sha256": self.plan_sha256,
            "cell_id": self.cell_id,
            "run_id": self.run_id,
            "work_item": self.work_item.public_payload(),
            "route_manifest": {
                "path": str(self.route_manifest_path),
                "sha256": self.route_manifest_sha256,
            },
            "source_work_item_id": self.source_work_item_id,
            "conditions": list(self.conditions),
            "arm_ids": dict(self.arm_ids),
            "attempt_slot_count": len(self.attempt_slots),
            "attempt_slots_sha256": sha256_json(list(self.attempt_slots)),
            "reserved_worst_case_usd": _decimal_text(self.forecast_usd),
            "official_fit_eligible": False,
            "permitted_analysis": "comparison_graph_diagnostics_only",
        }


@dataclass(frozen=True)
class RuntimeBundle:
    document: Mapping[str, Any]
    cells: tuple[ContinuationCell, ...]
    epicure: Mapping[str, str]
    execution_policy: Any


@dataclass(frozen=True)
class RunPaths:
    root: Path
    source: Path
    responses: Path
    ledger: Path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} must be a JSON object")
    return value


def _load_addressed(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    value = _load_json(path, label=label)
    digest = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if (
        value.get("schema_version") != expected_schema
        or not _is_sha256(digest)
        or sha256_json(payload) != digest
        or str(digest) not in path.name
        or (expected_digest is not None and digest != expected_digest)
    ):
        raise IntegrityError(f"{label} schema or content address does not verify")
    return value


def _write_addressed(
    payload: Mapping[str, Any], *, directory: Path, prefix: str
) -> Path:
    document = {**dict(payload), "artifact_sha256": sha256_json(payload)}
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{document['artifact_sha256']}.json"
    rendered = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise IntegrityError(f"existing {prefix} conflicts at its content address")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=f".{prefix}-", delete=False
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def _source_bundle(project_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"execution source is missing or non-regular: {relative}")
        files.append(
            {
                "path": relative,
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {"files": files, "bundle_sha256": sha256_json(files)}


def _relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError as error:
        raise IntegrityError("bound path is outside the FlavourBench project root") from error


def _verify_plan_shapes(
    v2: Mapping[str, Any], v3: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    v2_cells = v2.get("fresh_non_cohere_cells")
    v3_cells = v3.get("replacement_cells")
    if (
        v2.get("status") != "frozen_no_provider_calls"
        or not isinstance(v2_cells, list)
        or len(v2_cells) != EXPECTED_V2_CELLS
        or v2.get("counts", {}).get("new_arm_ids") != 12
        or any(
            not isinstance(cell, Mapping)
            or cell.get("schema_version") != CONTINUATION_SCHEMA_VERSION
            or cell.get("execution_backend") == "cohere_direct"
            or cell.get("same_task_as_v1") is not False
            or cell.get("same_work_or_arm_id_as_v1") is not False
            for cell in v2_cells
        )
    ):
        raise IntegrityError("v2 continuation is not the exact six-cell fresh schedule")
    if (
        v3.get("status") != "blocked_missing_prefixed_cohere_credential"
        or not isinstance(v3_cells, list)
        or len(v3_cells) != EXPECTED_V3_CELLS
        or v3.get("counts", {}).get("new_real_arms_planned") != 6
        or v3.get("sources", {}).get("v2_continuation_sha256") != V2_PLAN_SHA256
        or v3.get("methodological_boundary", {}).get(
            "official_preference_or_uplift_fit_eligible"
        )
        is not False
        or v3.get("methodological_boundary", {}).get("permitted_analysis")
        != "post_failure_task_sensitivity_and_graph_diagnostics"
        or any(
            not isinstance(cell, Mapping)
            or cell.get("schema_version") != REPLACEMENT_SCHEMA_VERSION
            or cell.get("alternate_task_not_previously_exposed_to_model") is not True
            or cell.get("fresh_identifiers_disjoint_from_v1_v2") is not True
            for cell in v3_cells
        )
    ):
        raise IntegrityError("v3 replacement is not the exact three-cell development schedule")
    return list(v2_cells), list(v3_cells)


def build_runtime_bundle(
    *,
    project_root: Path,
    v2_plan_path: Path,
    v3_plan_path: Path,
    v1_materialization_path: Path,
    task_validity_path: Path,
    route_manifest_paths: Sequence[Path],
    stopped_audit_path: Path,
    orphan_closure_path: Path,
    v1_ledger_path: Path,
    v1_source_directory: Path,
    v1_response_directory: Path,
) -> RuntimeBundle:
    v2 = _load_addressed(
        v2_plan_path,
        label="v2 continuation plan",
        expected_schema=CONTINUATION_SCHEMA_VERSION,
        expected_digest=V2_PLAN_SHA256,
    )
    v3 = _load_addressed(
        v3_plan_path,
        label="v3 replacement plan",
        expected_schema=REPLACEMENT_SCHEMA_VERSION,
        expected_digest=V3_PLAN_SHA256,
    )
    v2_cells, v3_cells = _verify_plan_shapes(v2, v3)
    v1 = _load_addressed(
        v1_materialization_path,
        label="v1 materialization",
        expected_schema="flavourbench-frontier-coverage-materialization-v1",
        expected_digest=V1_MATERIALIZATION_SHA256,
    )
    stopped = _load_addressed(
        stopped_audit_path,
        label="stopped-run audit",
        expected_schema="flavourbench-frontier-coverage-stopped-run-audit-v1",
        expected_digest=STOPPED_AUDIT_SHA256,
    )
    closure = _load_addressed(
        orphan_closure_path,
        label="v1 orphan closure",
        expected_schema="flavourbench-frontier-coverage-orphan-closure-v1",
        expected_digest=ORPHAN_CLOSURE_SHA256,
    )
    verify_orphan_closure(
        closure_path=orphan_closure_path,
        audit_path=stopped_audit_path,
        ledger_path=v1_ledger_path,
    )
    rebuilt_stopped = build_stopped_run_audit(
        source_directory=v1_source_directory,
        response_directory=v1_response_directory,
        ledger_path=v1_ledger_path,
        code_directory=project_root / "src/flavourbench",
    )
    rebuilt_failures = {
        str(item.get("work_item_id") or ""): item
        for item in rebuilt_stopped.get("evidence", {}).get("failures") or []
        if isinstance(item, Mapping)
    }
    historical_failures = {
        str(item.get("work_item_id") or ""): item
        for item in stopped.get("evidence", {}).get("failures") or []
        if isinstance(item, Mapping)
    }
    if (
        set(rebuilt_failures) != set(RETIRED_WORK_ITEM_IDS)
        or set(historical_failures) != set(RETIRED_WORK_ITEM_IDS)
        or any(
            rebuilt_failures[work_id].get("failure_class")
            != historical_failures[work_id].get("failure_class")
            or rebuilt_failures[work_id].get("provider_calls_verified")
            != historical_failures[work_id].get("provider_calls_verified")
            or rebuilt_failures[work_id].get("safe_to_replay_original_work_item") is not False
            for work_id in RETIRED_WORK_ITEM_IDS
        )
    ):
        raise IntegrityError("v1 stopped-run causal evidence no longer reconstructs")
    if (
        v2.get("sources", {}).get("v1_materialization_sha256")
        != V1_MATERIALIZATION_SHA256
        or v2.get("sources", {}).get("stopped_run_audit_sha256")
        != STOPPED_AUDIT_SHA256
        or v2.get("sources", {}).get("orphan_closure_sha256")
        != ORPHAN_CLOSURE_SHA256
        or v3.get("sources", {}).get("v1_materialization_sha256")
        != V1_MATERIALIZATION_SHA256
        or v3.get("sources", {}).get("stopped_run_audit_sha256")
        != STOPPED_AUDIT_SHA256
        or v3.get("sources", {}).get("orphan_closure_sha256")
        != ORPHAN_CLOSURE_SHA256
        or closure.get("work_item_id") not in RETIRED_WORK_ITEM_IDS
    ):
        raise IntegrityError("continuation plans do not bind the immutable v1 disposition")

    tasks, task_source = load_development_task_inventory(task_validity_path)
    tasks_by_id = {task.public_id: task for task in tasks}
    registry_digest = task_registry_sha256(tasks)
    if (
        task_source.get("artifact_sha256")
        != v1.get("source", {}).get("task_validity_sha256")
        or registry_digest != v1.get("source", {}).get("task_registry_sha256")
    ):
        raise IntegrityError("task dossier differs from the frozen v1 source binding")
    policy = _policy_from_document(v1.get("execution_policy"))
    if policy.sha256 != v1.get("execution_policy_sha256"):
        raise IntegrityError("v1 execution policy digest does not reconstruct")
    epicure = v1.get("epicure")
    if not isinstance(epicure, Mapping) or any(
        not epicure.get(field)
        for field in ("release_id", "bundle_sha256", "application_sha256", "tool_schema_sha256")
    ):
        raise IntegrityError("v1 Epicure provenance is incomplete")

    candidates: dict[tuple[str, str, str], tuple[Any, Path, str]] = {}
    for manifest_path in route_manifest_paths:
        manifest = load_candidate_manifest(manifest_path, expected_digest="")
        digest = str(manifest.get("content_address", {}).get("digest") or "")
        if digest not in set(v1.get("source", {}).get("route_manifest_sha256s") or []):
            raise IntegrityError("route manifest is outside the frozen v1 set")
        for candidate in select_candidates(manifest):
            key = (
                candidate.model_id,
                candidate.provider_tag,
                candidate.execution_backend,
            )
            candidates[key] = (candidate, manifest_path, digest)
    v1_items = {
        str(item.get("work_item", {}).get("work_item_id") or ""): item
        for item in v1.get("work_items") or []
        if isinstance(item, Mapping) and isinstance(item.get("work_item"), Mapping)
    }
    runtime_cells: list[ContinuationCell] = []
    all_attempt_ids: set[str] = set()
    all_identifiers: set[str] = set()
    for plan_kind, plan_sha, raw_cells in (
        ("v2_continuation", V2_PLAN_SHA256, v2_cells),
        ("v3_replacement", V3_PLAN_SHA256, v3_cells),
    ):
        for ordinal, raw in enumerate(raw_cells, start=1):
            source_work_id = str(
                raw.get("source_work_item_id") or raw.get("failed_work_item_id") or ""
            )
            source_item = v1_items.get(source_work_id)
            if source_item is None:
                raise IntegrityError("continuation cell has no exact v1 source work item")
            source_work = source_item["work_item"]
            key = (
                str(raw.get("model_id") or ""),
                str(raw.get("provider_tag") or ""),
                str(raw.get("execution_backend") or ""),
            )
            selected = candidates.get(key)
            task = tasks_by_id.get(str(raw.get("task_id") or ""))
            if selected is None or task is None:
                raise IntegrityError("continuation route or task is absent")
            candidate, manifest_path, manifest_sha = selected
            if (
                task.prompt_sha256 != raw.get("prompt_sha256")
                or source_work.get("model_id") != candidate.model_id
                or source_work.get("provider_tag") != candidate.provider_tag
                or source_work.get("execution_backend") != candidate.execution_backend
                or source_item.get("route_manifest", {}).get("artifact_sha256") != manifest_sha
            ):
                raise IntegrityError("continuation cell drifts from task or v1 route source")
            work_item_id = str(raw.get("work_item_id") or "")
            cell_id = str(raw.get("cell_id") or "")
            run_id = str(raw.get("run_id") or "")
            arm_ids = raw.get("arm_ids")
            attempts = raw.get("attempt_slots")
            if (
                not _is_sha256(work_item_id)
                or not _is_sha256(cell_id)
                or not isinstance(arm_ids, Mapping)
                or set(arm_ids) != {"epicure_off", "epicure_on"}
                or not all(_is_sha256(value) for value in arm_ids.values())
                or not isinstance(attempts, list)
                or not attempts
                or any(not isinstance(slot, Mapping) for slot in attempts)
            ):
                raise IntegrityError("continuation identifiers or attempt slots are malformed")
            try:
                uuid.UUID(run_id)
            except ValueError as error:
                raise IntegrityError("continuation run ID is not a UUID") from error
            attempt_ids = {str(slot.get("attempt_id") or "") for slot in attempts}
            if len(attempt_ids) != len(attempts) or any(not value for value in attempt_ids):
                raise IntegrityError("continuation attempt IDs are duplicated or empty")
            new_ids = {work_item_id, cell_id, run_id, *map(str, arm_ids.values()), *attempt_ids}
            if all_identifiers.intersection(new_ids) or all_attempt_ids.intersection(attempt_ids):
                raise IntegrityError("v2/v3 continuation namespaces overlap")
            all_identifiers.update(new_ids)
            all_attempt_ids.update(attempt_ids)
            work_item = WorkItem(
                ordinal=(100 if plan_kind == "v3_replacement" else 0) + ordinal,
                work_item_id=work_item_id,
                manifest_sha256=manifest_sha,
                task_registry_sha256=registry_digest,
                task=task,
                candidate=candidate,
                endpoint_execution_sha256=candidate.endpoint_execution_sha256,
                execution_policy_sha256=policy.sha256,
                execution_policy=policy,
            )
            forecast = derive_conditions_forecast(
                work_item, policy=policy, conditions=("epicure_off", "epicure_on")
            )
            runtime_cells.append(
                ContinuationCell(
                    plan_kind=plan_kind,
                    plan_sha256=plan_sha,
                    cell_id=cell_id,
                    run_id=run_id,
                    arm_ids={key: str(value) for key, value in arm_ids.items()},
                    attempt_slots=tuple(dict(slot) for slot in attempts),
                    work_item=work_item,
                    route_manifest_path=manifest_path,
                    route_manifest_sha256=manifest_sha,
                    forecast_usd=forecast.forecast_usd,
                    source_work_item_id=source_work_id,
                )
            )
    if len(runtime_cells) != EXPECTED_V2_CELLS + EXPECTED_V3_CELLS:
        raise IntegrityError("continuation runtime cell count differs")
    total_arms = sum(len(cell.conditions) for cell in runtime_cells)
    if total_arms != EXPECTED_ARMS:
        raise IntegrityError("continuation runtime arm count differs")
    total_forecast = sum((cell.forecast_usd for cell in runtime_cells), Decimal(0))
    payload = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "frozen_zero_call_materialization",
        "sources": {
            "v2_plan_sha256": V2_PLAN_SHA256,
            "v3_plan_sha256": V3_PLAN_SHA256,
            "v1_materialization_sha256": V1_MATERIALIZATION_SHA256,
            "stopped_run_audit_sha256": STOPPED_AUDIT_SHA256,
            "orphan_closure_sha256": ORPHAN_CLOSURE_SHA256,
            "task_validity_sha256": task_source["artifact_sha256"],
            "task_registry_sha256": registry_digest,
            "route_manifest_sha256s": sorted(
                {cell.route_manifest_sha256 for cell in runtime_cells}
            ),
        },
        "execution_policy": policy.document(),
        "execution_policy_sha256": policy.sha256,
        "reasoning_effort_disclosure": {
            "intermediate": policy.intermediate_reasoning_effort,
            "final": policy.final_reasoning_effort,
        },
        "epicure": dict(epicure),
        "counts": {
            "v2_cells": EXPECTED_V2_CELLS,
            "v3_cells": EXPECTED_V3_CELLS,
            "real_arms": EXPECTED_ARMS,
            "synthetic_arms": 0,
            "provider_calls_by_materialization": 0,
            "epicure_calls_by_materialization": 0,
        },
        "worst_case_budget_usd": _decimal_text(total_forecast),
        "cells": [cell.public_payload() for cell in runtime_cells],
        "v1_disposition": {
            "failure_records_preserved": True,
            "reliability_denominator_preserved": True,
            "retired_work_item_ids": sorted(RETIRED_WORK_ITEM_IDS),
            "retired_work_items_replayed": 0,
        },
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "official_preference_or_uplift_fit_eligible": False,
            "permitted_analysis": "comparison_graph_diagnostics_only",
            "replacement_observations_are_not_missing_at_random": True,
            "quality_judgments": 0,
        },
    }
    document = {**payload, "artifact_sha256": sha256_json(payload)}
    return RuntimeBundle(
        document=document,
        cells=tuple(runtime_cells),
        epicure=dict(epicure),
        execution_policy=policy,
    )


def _run_paths(output_root: Path, plan_kind: str) -> RunPaths:
    name = "v2" if plan_kind == "v2_continuation" else "v3"
    root = output_root / name
    return RunPaths(
        root=root,
        source=root / "source",
        responses=root / "responses",
        ledger=root / "ledger.jsonl",
    )


def _state(
    cells: Sequence[ContinuationCell], paths: RunPaths, *, label: str
) -> tuple[Any, Mapping[tuple[str, str], Any]]:
    accounting = _run_accounting(
        SupplementalRun(paths.source, paths.ledger), label=label
    )
    allowed = {cell.work_item.work_item_id for cell in cells}
    unknown = (
        set(accounting.sources)
        | set(accounting.reservations)
        | set(accounting.finalizations)
    ) - allowed
    if unknown:
        raise IntegrityError(f"{label} contains unknown work-item IDs: {sorted(unknown)}")
    responses = scan_response_artifacts(paths.responses)
    if any(work_id not in allowed for work_id, _ in responses):
        raise IntegrityError(f"{label} contains an unknown normalized response")
    return accounting, responses


def _verify_v4_gate(
    *, project_root: Path, plan_path: Path, audit_path: Path, closure_path: Path
) -> dict[str, str]:
    plan = _load_addressed(
        plan_path,
        label="v4 route plan",
        expected_schema="flavourbench-response-envelope-route-v4-plan-v1",
        expected_digest=V4_ROUTE_PLAN_SHA256,
    )
    audit = _load_addressed(
        audit_path,
        label="v4 route audit",
        expected_schema="flavourbench-response-envelope-route-v4-audit-v1",
        expected_digest=V4_ROUTE_AUDIT_SHA256,
    )
    closure = _load_addressed(
        closure_path,
        label="v4 route closure",
        expected_schema="flavourbench-response-envelope-route-v4-closure-v1",
        expected_digest=V4_ROUTE_CLOSURE_SHA256,
    )
    if (
        audit.get("decision") != "passed_all_predicates"
        or closure.get("decision", {}).get("route_qualified") is not True
        or not verify_v4_route_acceptance_paths(
            plan_path=plan_path,
            audit_path=audit_path,
            closure_path=closure_path,
            repo_root=project_root.parent,
        )
    ):
        raise IntegrityError("v4 source-reconstructed route qualification does not verify")
    return {
        "plan_sha256": plan["artifact_sha256"],
        "audit_sha256": audit["artifact_sha256"],
        "closure_sha256": closure["artifact_sha256"],
    }


def build_preflight(
    *,
    bundle: RuntimeBundle,
    project_root: Path,
    output_root: Path,
    input_paths: Mapping[str, Any],
    budget_audit_path: Path,
    supplemental_runs: Sequence[SupplementalRun],
    v1_run: SupplementalRun,
    v1_orphan_closure_path: Path,
    global_ledger_path: Path,
    global_artifact_directory: Path,
    global_corrections_directory: Path | None,
    global_reconciliation_directory: Path | None,
    v4_route: Mapping[str, str],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    try:
        require_prefixed_credential_before_reservation("cohere_direct", environment)
        credential_status = "pass_prefixed_credential_present"
    except AdmissionDenied:
        credential_status = "blocked_missing_prefixed_credential"
        blockers.append(
            {
                "gate": "cohere_prefixed_credential_before_any_reservation",
                "required_variable": "FLAVOURBENCH_COHERE_API_KEY",
            }
        )
    budget_audit, seen = _verify_budget_audit(
        budget_audit_path,
        project_root=project_root.parent,
        cap_usd=AUTHORIZED_TOTAL_CAP_USD,
        admission_fraction=DEFAULT_ADMISSION_FRACTION,
    )
    global_active, global_blockers = _global_ledger_state(
        ledger_path=global_ledger_path,
        artifact_directory=global_artifact_directory,
        corrections_directory=global_corrections_directory,
        reconciliation_directory=global_reconciliation_directory,
    )
    blockers.extend(dict(item) for item in global_blockers)
    seen_digests = set(seen)
    supplemental_exposure = Decimal(0)
    supplemental_actual = Decimal(0)
    supplemental_orphans = Decimal(0)
    for index, run in enumerate(supplemental_runs, start=1):
        accounting = _run_accounting(run, label=f"supplemental_{index}")
        if seen_digests.intersection(accounting.artifact_sha256s):
            raise IntegrityError("supplemental accounting duplicates baseline artifacts")
        seen_digests.update(accounting.artifact_sha256s)
        supplemental_exposure += accounting.exposure_usd
        supplemental_actual += accounting.actual_cost_usd
        supplemental_orphans += accounting.orphan_reservation_usd
        blockers.extend(dict(item) for item in accounting.blockers)
    v1_accounting = _run_accounting(v1_run, label="stopped_v1_coverage")
    if seen_digests.intersection(v1_accounting.artifact_sha256s):
        raise IntegrityError("stopped v1 accounting duplicates baseline artifacts")
    ignored_v1_blockers: list[Mapping[str, Any]] = []
    for item in v1_accounting.blockers:
        if (
            item.get("work_item_id")
            == "63cf4b5c57e627ae17d150c6d0a37d30b7f59bee1c1f9a301a6c48c30b700a79"
            and _decimal(item.get("reserved_usd"), field="closed v1 reserve") == 0
        ):
            ignored_v1_blockers.append(item)
        else:
            blockers.append(dict(item))
    closure = _load_addressed(
        v1_orphan_closure_path,
        label="closed v1 orphan",
        expected_schema="flavourbench-frontier-coverage-orphan-closure-v1",
        expected_digest=ORPHAN_CLOSURE_SHA256,
    )
    if len(ignored_v1_blockers) != 1 or closure.get("work_item_retired") is not True:
        raise IntegrityError("v1 orphan exception is not exactly the verified zero-dollar closure")

    run_accounting: dict[str, Any] = {}
    continuation_exposure = Decimal(0)
    continuation_orphans = Decimal(0)
    outstanding = Decimal(0)
    decisions: list[dict[str, Any]] = []
    for kind in ("v2_continuation", "v3_replacement"):
        cells = [cell for cell in bundle.cells if cell.plan_kind == kind]
        paths = _run_paths(output_root, kind)
        accounting, _ = _state(cells, paths, label=kind)
        if seen_digests.intersection(accounting.artifact_sha256s):
            raise IntegrityError("continuation artifacts duplicate prior budget inputs")
        seen_digests.update(accounting.artifact_sha256s)
        continuation_exposure += accounting.exposure_usd
        continuation_orphans += accounting.orphan_reservation_usd
        blockers.extend(dict(item) for item in accounting.blockers)
        run_accounting[kind] = accounting
        for cell in cells:
            work_id = cell.work_item.work_item_id
            if work_id in accounting.finalizations:
                decision = "skip_permanently_finalized"
            elif work_id in accounting.sources:
                decision = "recover_source_without_provider_call"
            elif work_id in accounting.reservations:
                decision = "block_reserved_without_source_no_replay"
            else:
                decision = "admit_sequentially_after_guarded_reservation"
                outstanding += cell.forecast_usd
            decisions.append(
                {
                    "plan_kind": kind,
                    "cell_id": cell.cell_id,
                    "work_item_id": work_id,
                    "model_id": cell.work_item.candidate.model_id,
                    "task_id": cell.work_item.task.public_id,
                    "execution_backend": cell.work_item.candidate.execution_backend,
                    "forecast_usd": _decimal_text(cell.forecast_usd),
                    "decision": decision,
                }
            )
    baseline = _decimal(
        budget_audit.get("current_total_exposure_usd"), field="baseline exposure"
    )
    current = (
        baseline
        + global_active
        + supplemental_exposure
        + supplemental_orphans
        + v1_accounting.exposure_usd
        + v1_accounting.orphan_reservation_usd
        + continuation_exposure
        + continuation_orphans
    )
    projected = current + outstanding
    ceiling = AUTHORIZED_TOTAL_CAP_USD * DEFAULT_ADMISSION_FRACTION
    if projected > ceiling or projected > AUTHORIZED_TOTAL_CAP_USD:
        blockers.append(
            {
                "gate": "shared_budget_projection",
                "projected_total_exposure_usd": _decimal_text(projected),
            }
        )
    existing_closures = sorted(output_root.glob("frontier-coverage-continuation-closure-*.json"))
    if existing_closures:
        blockers.append(
            {
                "gate": "permanent_execution_closure_exists",
                "closure_files": [path.name for path in existing_closures],
            }
        )
    materialization_path = _write_addressed(
        {key: value for key, value in bundle.document.items() if key != "artifact_sha256"},
        directory=output_root,
        prefix="frontier-coverage-continuation-materialization",
    )
    source_bundle = _source_bundle(project_root)
    payload = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": (
            "admissible_zero_call_preflight"
            if not blockers
            else "blocked_zero_call_preflight"
        ),
        "materialization": {
            "path": _relative(project_root, materialization_path),
            "sha256": bundle.document["artifact_sha256"],
        },
        "exact_inputs": dict(input_paths),
        "source_code": source_bundle,
        "route_gate": dict(v4_route),
        "credential_gate": {
            "status": credential_status,
            "required_variable": "FLAVOURBENCH_COHERE_API_KEY",
            "checked_before_any_reservation": True,
            "unprefixed_alias_accepted": False,
        },
        "fresh_namespaces": {
            "v2_ledger": _relative(project_root, _run_paths(output_root, "v2_continuation").ledger),
            "v3_ledger": _relative(project_root, _run_paths(output_root, "v3_replacement").ledger),
            "v2_reservation_namespace_sha256": sha256_json(
                {"plan": V2_PLAN_SHA256, "kind": "v2_guarded_reservations"}
            ),
            "v3_reservation_namespace_sha256": sha256_json(
                {"plan": V3_PLAN_SHA256, "kind": "v3_guarded_reservations"}
            ),
            "separate_from_v1": True,
        },
        "budget": {
            "currency": "USD",
            "baseline_exposure_usd": _decimal_text(baseline),
            "global_active_reservation_usd": _decimal_text(global_active),
            "supplemental_actual_cost_usd": _decimal_text(supplemental_actual),
            "supplemental_exposure_usd": _decimal_text(supplemental_exposure),
            "stopped_v1_actual_cost_usd": _decimal_text(v1_accounting.actual_cost_usd),
            "stopped_v1_exposure_usd": _decimal_text(v1_accounting.exposure_usd),
            "continuation_exposure_usd": _decimal_text(continuation_exposure),
            "current_total_exposure_usd": _decimal_text(current),
            "outstanding_worst_case_usd": _decimal_text(outstanding),
            "projected_total_exposure_usd": _decimal_text(projected),
            "admission_ceiling_usd": _decimal_text(ceiling),
            "hard_cap_usd": _decimal_text(AUTHORIZED_TOTAL_CAP_USD),
            "admission_allowed": not blockers,
        },
        "decisions": decisions,
        "blockers": blockers,
        "execution": {
            "cells_are_sequential": True,
            "manual_cell_retry": False,
            "provider_attempts_per_phase_max": bundle.execution_policy.max_provider_attempts,
            "incomplete_cell_action": "permanent_stop_and_identifier_closure",
            "confirmation": EXECUTION_CONFIRMATION,
        },
        "calls": {"provider": 0, "epicure": 0},
        "synthetic_arms": 0,
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "official_fit_eligible": False,
            "permitted_analysis": "comparison_graph_diagnostics_only",
        },
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _reservation_fields(
    *, cell: ContinuationCell, bundle: RuntimeBundle, namespace_sha256: str
) -> dict[str, Any]:
    return {
        "plan_kind": cell.plan_kind,
        "plan_sha256": cell.plan_sha256,
        "reservation_namespace_sha256": namespace_sha256,
        "cell_id": cell.cell_id,
        "frozen_run_id": cell.run_id,
        "source_v1_work_item_id": cell.source_work_item_id,
        "manifest_sha256": cell.route_manifest_sha256,
        "task_registry_sha256": cell.work_item.task_registry_sha256,
        "task_id": cell.work_item.task.public_id,
        "task_family": cell.work_item.task.family,
        "prompt_sha256": cell.work_item.task.prompt_sha256,
        "model_id": cell.work_item.candidate.model_id,
        "canonical_model_slug": cell.work_item.candidate.canonical_model_slug,
        "provider_tag": cell.work_item.candidate.provider_tag,
        "endpoint_execution_sha256": cell.work_item.endpoint_execution_sha256,
        "execution_policy_sha256": cell.work_item.execution_policy_sha256,
        "conditions": list(cell.conditions),
        "attempt_slots_sha256": sha256_json(list(cell.attempt_slots)),
        "epicure": dict(bundle.epicure),
        "official_fit_eligible": False,
        "permitted_analysis": "comparison_graph_diagnostics_only",
    }


def _source_for_work_item(paths: RunPaths, work_item_id: str) -> DatasetSource | None:
    accounting = _run_accounting(
        SupplementalRun(paths.source, paths.ledger), label=f"source_scan_{work_item_id[:12]}"
    )
    return accounting.sources.get(work_item_id)


def _recovery_evidence(paths: RunPaths, work_item_id: str) -> dict[str, Any]:
    states = scan_recovery_journals(
        paths.source, dataset_work_item_id=work_item_id
    )
    rows = [
        {
            "filename": state.path.name,
            "journal_sha256": state.journal_sha256,
            "head_entry_sha256": state.head_entry_sha256,
            "entry_count": state.entry_count,
            "run_id": state.run_id,
            "finalized": state.finalized,
            "generation_ids": list(state.generation_ids),
            "unreconciled_generation_ids": list(state.unreconciled_generation_ids),
            "uncertain_attempt_ids": list(state.uncertain_attempt_ids),
            "recovery_action": state.recovery_action,
            "journal_classifier_safe_to_replay": state.safe_to_replay,
        }
        for state in states
    ]
    if not states:
        classification = "no_journal_delivery_state_unknown"
    elif any(
        state.uncertain_attempt_ids or state.unreconciled_generation_ids
        for state in states
    ):
        classification = "uncertain_delivery_or_unreconciled_generation"
    elif any(state.generation_ids for state in states):
        classification = "provider_generation_observed_source_missing"
    elif all(state.safe_to_replay for state in states):
        classification = "pre_request_or_safe_provider_rejection_no_generation"
    else:
        classification = "journal_present_delivery_state_requires_manual_reconciliation"
    # The continuation's parent policy closes the work item in every class;
    # journal-level replay safety is diagnostic only and never reopens it.
    return {
        "delivery_classification": classification,
        "journals": rows,
        "parent_policy_safe_to_replay": False,
    }


def _finalize_source(
    *,
    cell: ContinuationCell,
    bundle: RuntimeBundle,
    paths: RunPaths,
    reservation: Mapping[str, Any],
    runner_run_id: str,
    source: DatasetSource,
) -> tuple[Mapping[str, Any], bool, list[str]]:
    issues = _source_postflight_issues(
        source,
        cell.work_item,
        expected_conditions=cell.conditions,
        expected_epicure=bundle.epicure,
    )
    hard_issues = sorted(set(issues) - HARD_POSTFLIGHT_EXEMPTIONS)
    if hard_issues:
        raise IntegrityError(f"source route/protocol binding failed: {hard_issues}")
    responses, normalization_issues = normalise_source_responses(
        source,
        cell.work_item,
        response_directory=paths.responses,
        expected_conditions=cell.conditions,
        expected_epicure=bundle.epicure,
    )
    all_issues = sorted(set(issues + normalization_issues))
    response_conditions = sorted(response.condition for response in responses)
    complete = response_conditions == sorted(cell.conditions) and not all_issues
    event = append_dataset_ledger_event(
        paths.ledger,
        {
            "event_type": "source_artifact_recorded",
            "runner_run_id": runner_run_id,
            "work_item_id": cell.work_item.work_item_id,
            "reservation_entry_sha256": reservation["entry_sha256"],
            **_reservation_fields(
                cell=cell,
                bundle=bundle,
                namespace_sha256=str(reservation["reservation_namespace_sha256"]),
            ),
            "source_artifact_filename": source.path.name,
            "source_artifact_sha256": source.artifact_sha256,
            "source_status": source.exposure.status,
            "source_actual_cost_usd": _decimal_text(source.exposure.actual_cost_usd),
            "source_budget_exposure_usd": _decimal_text(source.exposure.exposure_usd),
            "response_artifact_sha256s": sorted(
                response.artifact_sha256 for response in responses
            ),
            "response_conditions": response_conditions,
            "normalization_issues": all_issues,
            "complete_required_conditions": complete,
            "safe_to_replay": False,
            "reliability_failure_retained": not complete,
        },
    )
    return event, complete, all_issues


def _find_closure(output_root: Path) -> Path | None:
    paths = sorted(output_root.glob("frontier-coverage-continuation-closure-*.json"))
    if len(paths) > 1:
        raise IntegrityError("more than one continuation closure exists")
    return paths[0] if paths else None


def execute_preflight(
    *,
    preflight_path: Path,
    project_root: Path,
    output_root: Path,
    confirmation: str,
    process_timeout_seconds: int,
) -> tuple[Path, Path]:
    if confirmation != EXECUTION_CONFIRMATION:
        raise AdmissionDenied(f"execution requires --confirm {EXECUTION_CONFIRMATION}")
    preflight = _load_addressed(
        preflight_path,
        label="continuation preflight",
        expected_schema=PREFLIGHT_SCHEMA_VERSION,
    )
    if preflight.get("status") != "admissible_zero_call_preflight":
        raise AdmissionDenied("continuation preflight is not admissible")
    if preflight.get("source_code") != _source_bundle(project_root):
        raise IntegrityError("execution source changed after the preflight freeze")
    if _find_closure(output_root) is not None:
        raise AdmissionDenied("continuation identifiers are permanently closed")
    require_prefixed_credential_before_reservation("cohere_direct", os.environ)
    exact = preflight.get("exact_inputs")
    if not isinstance(exact, Mapping):
        raise IntegrityError("preflight exact inputs are absent")

    def bound_path(name: str) -> Path:
        record = exact.get(name)
        if not isinstance(record, Mapping):
            raise IntegrityError(f"preflight input is absent: {name}")
        path = project_root / str(record.get("path") or "")
        if _file_sha256(path) != record.get("physical_sha256"):
            raise IntegrityError(f"preflight physical input changed: {name}")
        return path

    route_manifests = [project_root / str(item) for item in exact["route_manifests"]]
    bundle = build_runtime_bundle(
        project_root=project_root,
        v2_plan_path=bound_path("v2_plan"),
        v3_plan_path=bound_path("v3_plan"),
        v1_materialization_path=bound_path("v1_materialization"),
        task_validity_path=bound_path("task_validity"),
        route_manifest_paths=route_manifests,
        stopped_audit_path=bound_path("stopped_audit"),
        orphan_closure_path=bound_path("orphan_closure"),
        v1_ledger_path=bound_path("v1_ledger"),
        v1_source_directory=project_root / str(exact["v1_source_directory"]),
        v1_response_directory=project_root / str(exact["v1_response_directory"]),
    )
    if bundle.document["artifact_sha256"] != preflight.get("materialization", {}).get(
        "sha256"
    ):
        raise IntegrityError("preflight materialization does not reconstruct")
    namespaces = preflight.get("fresh_namespaces")
    if not isinstance(namespaces, Mapping):
        raise IntegrityError("preflight ledger namespaces are absent")

    runner_run_id = str(uuid.uuid4())
    outcomes: list[dict[str, Any]] = []
    subprocesses_started = 0
    stopped = False
    stop_reason = "all_cells_finalized"
    global_ledger = project_root / str(exact["global_ledger"])
    with _exclusive_runner_lock(global_ledger):
        for cell in bundle.cells:
            if stopped:
                break
            paths = _run_paths(output_root, cell.plan_kind)
            namespace = str(
                namespaces[
                    "v2_reservation_namespace_sha256"
                    if cell.plan_kind == "v2_continuation"
                    else "v3_reservation_namespace_sha256"
                ]
            )
            with _dataset_ledger_lock(paths.ledger):
                entries = load_dataset_ledger(paths.ledger)
                reservations, finalizations = dataset_ledger_state(entries)
                work_id = cell.work_item.work_item_id
                if work_id in finalizations:
                    outcomes.append(
                        {"work_item_id": work_id, "decision": "skip_permanently_finalized"}
                    )
                    continue
                reservation = reservations.get(work_id)
                source = _source_for_work_item(paths, work_id) if reservation else None
                if reservation is not None and source is None:
                    incident = append_dataset_ledger_event(
                        paths.ledger,
                        {
                            "event_type": "execution_incident",
                            "runner_run_id": runner_run_id,
                            "work_item_id": work_id,
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "incident": "reserved_without_source_permanent_no_replay_stop",
                            "safe_to_replay": False,
                        },
                    )
                    outcomes.append(
                        {
                            "work_item_id": work_id,
                            "decision": "permanent_stop_reserved_without_source",
                            "incident_entry_sha256": incident["entry_sha256"],
                        }
                    )
                    stopped = True
                    stop_reason = "reserved_without_source"
                    continue
                if reservation is not None and source is not None:
                    event, complete, issues = _finalize_source(
                        cell=cell,
                        bundle=bundle,
                        paths=paths,
                        reservation=reservation,
                        runner_run_id=runner_run_id,
                        source=source,
                    )
                    outcomes.append(
                        {
                            "work_item_id": work_id,
                            "decision": (
                                "recovered_source_without_provider_call"
                                if complete
                                else "recovered_incomplete_source_permanent_stop"
                            ),
                            "source_artifact_sha256": source.artifact_sha256,
                            "ledger_entry_sha256": event["entry_sha256"],
                            "issues": issues,
                        }
                    )
                    if not complete:
                        stopped = True
                        stop_reason = "incomplete_recovered_source"
                    continue

                # The prefixed gate is intentionally repeated while holding the
                # shared budget mutex and before this cell's reservation append.
                require_prefixed_credential_before_reservation("cohere_direct", os.environ)
                reservation = append_guarded_continuation_reservation(
                    ledger_path=paths.ledger,
                    runner_run_id=runner_run_id,
                    cell={
                        "schema_version": (
                            CONTINUATION_SCHEMA_VERSION
                            if cell.plan_kind == "v2_continuation"
                            else REPLACEMENT_SCHEMA_VERSION
                        ),
                        "execution_backend": cell.work_item.candidate.execution_backend,
                        "work_item_id": work_id,
                    },
                    reserved_usd=_decimal_text(cell.forecast_usd),
                    environment=os.environ,
                    additional_fields=_reservation_fields(
                        cell=cell, bundle=bundle, namespace_sha256=namespace
                    ),
                )
                paths.source.mkdir(parents=True, exist_ok=True)
                paths.responses.mkdir(parents=True, exist_ok=True)
                forecast = derive_conditions_forecast(
                    cell.work_item,
                    policy=bundle.execution_policy,
                    conditions=cell.conditions,
                )
                command = _subprocess_command(
                    cell.work_item,
                    forecast=forecast,
                    source_directory=paths.source,
                    manifest_path=cell.route_manifest_path,
                    conditions=cell.conditions,
                    expected_epicure=bundle.epicure,
                )
                if cell.work_item.candidate.execution_backend == "openrouter":
                    if command[1:3] != ["-m", "flavourbench.live_smoke"]:
                        raise IntegrityError(
                            "OpenRouter subprocess module is not the qualified runner"
                        )
                    command[2] = "flavourbench.continuation_openrouter_pair"
                command.extend(["--frozen-run-id", cell.run_id])
                command.extend(
                    [
                        "--frozen-attempt-slots-json",
                        json.dumps(
                            list(cell.attempt_slots),
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    ]
                )
                environment = os.environ.copy()
                environment.update(bundle.execution_policy.settings_environment())
                environment["FLAVOURBENCH_OPENROUTER_MAX_PROMPT_PRICE_PER_MTOK"] = (
                    _decimal_text(forecast.price_envelope.prompt_usd_per_mtok)
                )
                environment["FLAVOURBENCH_OPENROUTER_MAX_COMPLETION_PRICE_PER_MTOK"] = (
                    _decimal_text(forecast.price_envelope.completion_usd_per_mtok)
                )
                subprocesses_started += 1
                try:
                    completed = subprocess.run(
                        command,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=process_timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    recovery = _recovery_evidence(paths, work_id)
                    incident = append_dataset_ledger_event(
                        paths.ledger,
                        {
                            "event_type": "execution_incident",
                            "runner_run_id": runner_run_id,
                            "work_item_id": work_id,
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "incident": "subprocess_timeout_uncertain_delivery_permanent_stop",
                            "timeout_seconds": process_timeout_seconds,
                            "output_sha256": _safe_process_hash(str(error.output or "")),
                            "delivery_evidence": recovery,
                            "safe_to_replay": False,
                        },
                    )
                    outcomes.append(
                        {
                            "work_item_id": work_id,
                            "decision": "timeout_permanent_stop_no_replay",
                            "incident_entry_sha256": incident["entry_sha256"],
                        }
                    )
                    stopped = True
                    stop_reason = "subprocess_timeout"
                    continue
                artifact_path = _extract_artifact_path(completed.stdout, paths.source)
                if artifact_path is None or not artifact_path.exists():
                    recovery = _recovery_evidence(paths, work_id)
                    incident = append_dataset_ledger_event(
                        paths.ledger,
                        {
                            "event_type": "execution_incident",
                            "runner_run_id": runner_run_id,
                            "work_item_id": work_id,
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "incident": "no_verifiable_source_permanent_stop_no_replay",
                            "subprocess_returncode": completed.returncode,
                            "stdout_sha256": _safe_process_hash(completed.stdout),
                            "stderr_sha256": _safe_process_hash(completed.stderr),
                            "delivery_evidence": recovery,
                            "safe_to_replay": False,
                        },
                    )
                    outcomes.append(
                        {
                            "work_item_id": work_id,
                            "decision": "no_source_permanent_stop_no_replay",
                            "incident_entry_sha256": incident["entry_sha256"],
                        }
                    )
                    stopped = True
                    stop_reason = "no_verifiable_source"
                    continue
                source = _source_for_work_item(paths, work_id)
                if source is None or source.path.resolve() != artifact_path.resolve():
                    raise IntegrityError("subprocess source is not the reserved work-item source")
                event, complete, issues = _finalize_source(
                    cell=cell,
                    bundle=bundle,
                    paths=paths,
                    reservation=reservation,
                    runner_run_id=runner_run_id,
                    source=source,
                )
                outcomes.append(
                    {
                        "work_item_id": work_id,
                        "decision": (
                            "source_finalized_complete"
                            if complete
                            else "source_finalized_incomplete_permanent_stop"
                        ),
                        "source_artifact_sha256": source.artifact_sha256,
                        "ledger_entry_sha256": event["entry_sha256"],
                        "issues": issues,
                    }
                )
                if not complete:
                    stopped = True
                    stop_reason = "incomplete_cell"

    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "record_role": "single_sequential_v2_v3_execution_receipt",
        "preflight_sha256": preflight["artifact_sha256"],
        "runner_run_id": runner_run_id,
        "status": "complete" if not stopped else "failed_closed",
        "stop_reason": stop_reason,
        "subprocesses_started": subprocesses_started,
        "outcomes": outcomes,
        "manual_retries": 0,
        "second_invocation_of_reserved_work": False,
        "completed_at": _utc_now(),
        "official_fit_eligible": False,
    }
    receipt_path = _write_addressed(
        receipt_payload,
        directory=output_root,
        prefix="frontier-coverage-continuation-receipt",
    )
    receipt = _load_addressed(
        receipt_path,
        label="continuation receipt",
        expected_schema=RECEIPT_SCHEMA_VERSION,
    )
    ledger_heads: dict[str, str | None] = {}
    for kind in ("v2_continuation", "v3_replacement"):
        entries = load_dataset_ledger(_run_paths(output_root, kind).ledger)
        ledger_heads[kind] = entries[-1]["entry_sha256"] if entries else None
    closure_payload = {
        "schema_version": CLOSURE_SCHEMA_VERSION,
        "record_role": "permanent_v2_v3_work_and_identifier_closure",
        "preflight_sha256": preflight["artifact_sha256"],
        "receipt_sha256": receipt["artifact_sha256"],
        "status": "closed_complete" if not stopped else "closed_failed_incomplete",
        "stop_reason": stop_reason,
        "ledger_heads": ledger_heads,
        "closed_work_item_ids": sorted(
            cell.work_item.work_item_id for cell in bundle.cells
        ),
        "closed_run_ids": sorted(cell.run_id for cell in bundle.cells),
        "closed_attempt_ids_sha256": sha256_json(
            sorted(
                str(slot["attempt_id"])
                for cell in bundle.cells
                for slot in cell.attempt_slots
            )
        ),
        "safe_to_replay_any_reserved_or_planned_work": False,
        "future_execution_with_same_preflight_permitted": False,
        "original_v1_failures_superseded": False,
        "official_fit_eligible": False,
        "permitted_analysis": "comparison_graph_diagnostics_only",
    }
    closure_path = _write_addressed(
        closure_payload,
        directory=output_root,
        prefix="frontier-coverage-continuation-closure",
    )
    return receipt_path, closure_path


def _phase_coordinate(event: Mapping[str, Any]) -> tuple[str, str, int]:
    phase = str(event.get("phase") or "")
    if phase.startswith("cohere_direct_"):
        phase = phase.removeprefix("cohere_direct_")
    return (
        str(event.get("arm_id") or ""),
        phase,
        int(event.get("attempt_index", -1)),
    )


def _audit_source(
    *, cell: ContinuationCell, bundle: RuntimeBundle, source: DatasetSource
) -> tuple[list[str], set[str], set[str], int, int]:
    failures: list[str] = []
    artifact, digest = _verify_live_artifact(source.path)
    if digest != source.artifact_sha256:
        failures.append("source_digest_scan_mismatch")
    expected = {
        "run_id": cell.run_id,
        "dataset_work_item_id": cell.work_item.work_item_id,
        "dataset_task_id": cell.work_item.task.public_id,
        "prompt_sha256": cell.work_item.task.prompt_sha256,
        "category": cell.work_item.task.family,
        "requested_model_id": cell.work_item.candidate.model_id,
        "requested_provider": cell.work_item.candidate.provider_tag,
        "candidate_manifest_sha256": cell.route_manifest_sha256,
        "endpoint_execution_contract_sha256": cell.work_item.endpoint_execution_sha256,
        "execution_policy_sha256": cell.work_item.execution_policy_sha256,
        "requested_conditions": list(cell.conditions),
        "official": False,
        "rank_eligible": False,
        "research_result": False,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            failures.append(f"source_{field}_mismatch")
    epicure = artifact.get("epicure")
    if not isinstance(epicure, Mapping) or any(
        epicure.get(field) != bundle.epicure[field]
        for field in ("release_id", "bundle_sha256", "application_sha256")
    ) or artifact.get("epicure_tool_schema_sha256") != bundle.epicure["tool_schema_sha256"]:
        failures.append("source_epicure_identity_mismatch")
    events = [
        item for item in artifact.get("provider_attempt_events") or [] if isinstance(item, Mapping)
    ]
    planned = {
        (str(slot["arm_id"]), str(slot["phase"]), int(slot["attempt_index"])): str(
            slot["attempt_id"]
        )
        for slot in cell.attempt_slots
    }
    external_starts = [
        event
        for event in events
        if event.get("event_type") in {"request_started", "mcp_session_started", "mcp_call_started"}
    ]
    attempt_ids: set[str] = set()
    for event in external_starts:
        coordinate = _phase_coordinate(event)
        attempt_id = str(event.get("attempt_id") or "")
        if planned.get(coordinate) != attempt_id:
            failures.append("external_attempt_outside_prefrozen_slot_pool")
        if attempt_id in attempt_ids:
            failures.append("external_attempt_id_reused")
        attempt_ids.add(attempt_id)
    accepted = [event for event in events if event.get("event_type") == "response_received"]
    accepted_ids = {str(event.get("generation_id") or "") for event in accepted}
    if "" in accepted_ids:
        failures.append("accepted_response_without_generation_id")
    results = artifact.get("results")
    result_ids: set[str] = set()
    actual_cost_micros = 0
    successful_tools = 0
    if not isinstance(results, Mapping):
        failures.append("source_results_missing")
        results = {}
    for condition in cell.conditions:
        result = results.get(condition)
        if not isinstance(result, Mapping):
            failures.append(f"{condition}_result_missing")
            continue
        answer = str(result.get("answer_markdown") or "").strip()
        if len(answer) < 20:
            failures.append(f"{condition}_answer_not_substantive")
        generation_ids = [str(value) for value in result.get("generation_ids") or []]
        metadata = [
            item for item in result.get("generation_metadata") or [] if isinstance(item, Mapping)
        ]
        metadata_ids = {str(item.get("generation_id") or "") for item in metadata}
        if set(generation_ids) != metadata_ids or len(generation_ids) != len(set(generation_ids)):
            failures.append(f"{condition}_generation_metadata_bijection_failed")
        if cell.work_item.candidate.execution_backend == "openrouter":
            if result.get("cost_reconciled") is not True or any(
                item.get("reconciled") is not True for item in metadata
            ):
                failures.append(f"{condition}_openrouter_cost_not_reconciled")
        else:
            if (
                result.get("cost_reconciled") is not False
                or result.get("billing_reconciliation_status")
                != "provider_charge_unavailable"
                or any(item.get("reconciled") is not False for item in metadata)
            ):
                failures.append(f"{condition}_cohere_rate_card_accounting_invalid")
        calculated_cost = sum(int(item.get("cost_micros") or 0) for item in metadata)
        if calculated_cost != int(result.get("cost_micros") or 0):
            failures.append(f"{condition}_cost_sum_mismatch")
        actual_cost_micros += calculated_cost
        if result_ids.intersection(generation_ids):
            failures.append("generation_id_reused_across_conditions")
        result_ids.update(generation_ids)
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
                failures.append("epicure_on_has_no_successful_tool_call")
    incomplete = [
        item
        for item in artifact.get("incomplete_generation_metadata") or []
        if isinstance(item, Mapping)
    ]
    incomplete_ids = {str(item.get("generation_id") or "") for item in incomplete}
    if incomplete_ids:
        failures.append("incomplete_generation_metadata_present")
        result_ids.update(incomplete_ids)
        actual_cost_micros += sum(int(item.get("cost_micros") or 0) for item in incomplete)
    if accepted_ids != result_ids:
        failures.append("accepted_generation_id_bijection_failed")
    if int(artifact.get("budget", {}).get("actual_cost_micros") or 0) != actual_cost_micros:
        failures.append("source_budget_cost_mismatch")
    result_hashes = sorted(
        str(item.get("result_sha256") or "")
        for item in artifact.get("mcp_trace_events") or []
        if isinstance(item, Mapping)
    )
    completed_hashes = sorted(
        str(item.get("payload_sha256") or "")
        for item in events
        if item.get("event_type") == "mcp_call_completed"
    )
    if result_hashes != completed_hashes or any(not _is_sha256(value) for value in result_hashes):
        failures.append("mcp_trace_result_hash_bijection_failed")
    return sorted(set(failures)), attempt_ids, result_ids, actual_cost_micros, successful_tools


def build_postrun_audit(
    *,
    preflight_path: Path,
    receipt_path: Path,
    closure_path: Path,
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    preflight = _load_addressed(
        preflight_path,
        label="continuation preflight",
        expected_schema=PREFLIGHT_SCHEMA_VERSION,
    )
    receipt = _load_addressed(
        receipt_path,
        label="continuation receipt",
        expected_schema=RECEIPT_SCHEMA_VERSION,
    )
    closure = _load_addressed(
        closure_path,
        label="continuation closure",
        expected_schema=CLOSURE_SCHEMA_VERSION,
    )
    failures: list[str] = []
    if preflight.get("source_code") != _source_bundle(project_root):
        failures.append("generation_or_auditor_source_changed_after_preflight")
    if (
        receipt.get("preflight_sha256") != preflight["artifact_sha256"]
        or closure.get("preflight_sha256") != preflight["artifact_sha256"]
        or closure.get("receipt_sha256") != receipt["artifact_sha256"]
        or closure.get("safe_to_replay_any_reserved_or_planned_work") is not False
        or closure.get("official_fit_eligible") is not False
    ):
        failures.append("receipt_or_closure_binding_mismatch")
    exact = preflight.get("exact_inputs")
    if not isinstance(exact, Mapping):
        raise IntegrityError("preflight exact inputs are absent")

    def bound_path(name: str) -> Path:
        record = exact.get(name)
        if not isinstance(record, Mapping):
            raise IntegrityError(f"audit input is absent: {name}")
        path = project_root / str(record.get("path") or "")
        if _file_sha256(path) != record.get("physical_sha256"):
            raise IntegrityError(f"audit input changed: {name}")
        return path

    route_manifests = [project_root / str(value) for value in exact["route_manifests"]]
    bundle = build_runtime_bundle(
        project_root=project_root,
        v2_plan_path=bound_path("v2_plan"),
        v3_plan_path=bound_path("v3_plan"),
        v1_materialization_path=bound_path("v1_materialization"),
        task_validity_path=bound_path("task_validity"),
        route_manifest_paths=route_manifests,
        stopped_audit_path=bound_path("stopped_audit"),
        orphan_closure_path=bound_path("orphan_closure"),
        v1_ledger_path=bound_path("v1_ledger"),
        v1_source_directory=project_root / str(exact["v1_source_directory"]),
        v1_response_directory=project_root / str(exact["v1_response_directory"]),
    )
    if bundle.document["artifact_sha256"] != preflight.get("materialization", {}).get(
        "sha256"
    ):
        failures.append("runtime_materialization_no_longer_reconstructs")
    all_attempt_ids: set[str] = set()
    all_generation_ids: set[str] = set()
    usable_cells = 0
    usable_arms = 0
    provider_generations = 0
    successful_tools = 0
    total_cost_micros = 0
    cell_records: list[dict[str, Any]] = []
    for kind in ("v2_continuation", "v3_replacement"):
        cells = [cell for cell in bundle.cells if cell.plan_kind == kind]
        paths = _run_paths(output_root, kind)
        accounting, responses = _state(cells, paths, label=f"audit_{kind}")
        entries = load_dataset_ledger(paths.ledger)
        incidents_by_work = {
            str(entry.get("work_item_id") or ""): entry
            for entry in entries
            if entry.get("event_type") == "execution_incident"
        }
        expected_head = entries[-1]["entry_sha256"] if entries else None
        if closure.get("ledger_heads", {}).get(kind) != expected_head:
            failures.append(f"{kind}_ledger_head_mismatch")
        for cell in cells:
            work_id = cell.work_item.work_item_id
            reservation = accounting.reservations.get(work_id)
            finalization = accounting.finalizations.get(work_id)
            source = accounting.sources.get(work_id)
            local_failures: list[str] = []
            if reservation is None:
                cell_records.append(
                    {
                        "plan_kind": kind,
                        "work_item_id": work_id,
                        "status": "not_started_after_permanent_stop",
                        "failures": [],
                    }
                )
                continue
            namespace = preflight["fresh_namespaces"][
                "v2_reservation_namespace_sha256"
                if kind == "v2_continuation"
                else "v3_reservation_namespace_sha256"
            ]
            exact_fields = _reservation_fields(
                cell=cell, bundle=bundle, namespace_sha256=str(namespace)
            )
            if any(reservation.get(field) != value for field, value in exact_fields.items()):
                local_failures.append("reservation_differs_from_frozen_cell")
            if reservation.get("credential_preflight") not in {
                "not_applicable",
                "prefixed_cohere_present_before_reservation",
            }:
                local_failures.append("reservation_credential_preflight_invalid")
            if source is None:
                local_failures.append("reservation_has_no_source")
                incident = incidents_by_work.get(work_id)
                delivery = (
                    incident.get("delivery_evidence")
                    if isinstance(incident, Mapping)
                    else None
                )
                delivery_class = (
                    str(delivery.get("delivery_classification") or "")
                    if isinstance(delivery, Mapping)
                    else ""
                )
                if delivery_class not in {
                    "no_journal_delivery_state_unknown",
                    "uncertain_delivery_or_unreconciled_generation",
                    "provider_generation_observed_source_missing",
                    "pre_request_or_safe_provider_rejection_no_generation",
                    "journal_present_delivery_state_requires_manual_reconciliation",
                }:
                    local_failures.append("missing_machine_verifiable_delivery_classification")
            else:
                source_failures, attempts, generations, cost_micros, tool_count = _audit_source(
                    cell=cell, bundle=bundle, source=source
                )
                local_failures.extend(source_failures)
                if all_attempt_ids.intersection(attempts):
                    local_failures.append("attempt_id_overlaps_another_cell")
                if all_generation_ids.intersection(generations):
                    local_failures.append("generation_id_overlaps_another_cell")
                all_attempt_ids.update(attempts)
                all_generation_ids.update(generations)
                total_cost_micros += cost_micros
                successful_tools += tool_count
                provider_generations += len(generations)
            expected_responses = {
                (work_id, condition) for condition in cell.conditions
            }
            observed_responses = {key for key in responses if key[0] == work_id}
            if finalization is None:
                local_failures.append("source_not_finalized")
            elif (
                finalization.get("complete_required_conditions") is True
                and observed_responses != expected_responses
            ):
                local_failures.append("finalized_response_set_mismatch")
            complete = not local_failures and finalization is not None and (
                finalization.get("complete_required_conditions") is True
            )
            if complete:
                usable_cells += 1
                usable_arms += 2
            failures.extend(f"{work_id}:{value}" for value in local_failures)
            cell_records.append(
                {
                    "plan_kind": kind,
                    "work_item_id": work_id,
                    "status": "source_reconstructed_complete" if complete else "failed_closed",
                    "source_sha256": source.artifact_sha256 if source else None,
                    "delivery_classification": (
                        incidents_by_work.get(work_id, {})
                        .get("delivery_evidence", {})
                        .get("delivery_classification")
                    ),
                    "failures": sorted(set(local_failures)),
                }
            )
    closed_ids = sorted(cell.work_item.work_item_id for cell in bundle.cells)
    if closure.get("closed_work_item_ids") != closed_ids:
        failures.append("closure_work_item_set_mismatch")
    expected_complete = receipt.get("status") == "complete"
    if expected_complete and usable_cells != len(bundle.cells):
        failures.append("receipt_claims_complete_but_not_all_cells_reconstruct")
    if not expected_complete and closure.get("status") != "closed_failed_incomplete":
        failures.append("failed_receipt_lacks_failed_closed_status")
    unique_failures = sorted(set(failures))
    completed_all = not unique_failures and usable_cells == len(bundle.cells)
    payload = {
        "schema_version": POSTRUN_AUDIT_SCHEMA_VERSION,
        "record_role": "source_reconstructed_v2_v3_continuation_postrun_audit",
        "preflight_sha256": preflight["artifact_sha256"],
        "receipt_sha256": receipt["artifact_sha256"],
        "closure_sha256": closure["artifact_sha256"],
        "decision": (
            "passed_all_cells_source_reconstructed"
            if completed_all
            else "failed_closed_one_or_more_cells_incomplete"
        ),
        "failures": unique_failures,
        "counts": {
            "planned_cells": len(bundle.cells),
            "usable_cells": usable_cells,
            "planned_real_arms": EXPECTED_ARMS,
            "usable_real_arms": usable_arms,
            "provider_generations": provider_generations,
            "successful_epicure_tool_calls": successful_tools,
            "synthetic_arms": 0,
        },
        "accounting": {
            "actual_cost_micros": total_cost_micros,
            "actual_cost_usd": _decimal_text(
                Decimal(total_cost_micros) / Decimal(1_000_000)
            ),
            "openrouter_costs_require_generation_reconciliation": True,
            "cohere_costs_are_frozen_rate_card_estimates": True,
        },
        "identifier_audit": {
            "attempt_ids_sha256": sha256_json(sorted(all_attempt_ids)),
            "generation_ids_sha256": sha256_json(sorted(all_generation_ids)),
            "attempt_id_count": len(all_attempt_ids),
            "generation_id_count": len(all_generation_ids),
            "same_planned_work_replay_permitted": False,
        },
        "cells": cell_records,
        "v1_disposition": {
            "failure_records_preserved": True,
            "original_reliability_failures_superseded": False,
            "retired_work_items_replayed": 0,
        },
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "official_preference_or_uplift_fit_eligible": False,
            "permitted_analysis": "comparison_graph_diagnostics_only",
            "quality_judgments": 0,
        },
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _input_record(project_root: Path, path: Path) -> dict[str, str]:
    return {
        "path": _relative(project_root, path),
        "physical_sha256": _file_sha256(path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "execute", "audit"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v2-plan", type=Path)
    parser.add_argument("--v3-plan", type=Path)
    parser.add_argument("--v1-materialization", type=Path)
    parser.add_argument("--task-validity", type=Path)
    parser.add_argument("--route-manifest", type=Path, action="append", default=[])
    parser.add_argument("--stopped-audit", type=Path)
    parser.add_argument("--orphan-closure", type=Path)
    parser.add_argument("--v1-run-root", type=Path)
    parser.add_argument("--budget-audit", type=Path)
    parser.add_argument("--supplemental-run-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--global-ledger",
        type=Path,
        default=Path("artifacts/frontier-contract/ledger.jsonl"),
    )
    parser.add_argument(
        "--global-artifact-directory",
        type=Path,
        default=Path("artifacts/live-smoke"),
    )
    parser.add_argument(
        "--global-corrections-directory",
        type=Path,
        default=Path("artifacts/corrections"),
    )
    parser.add_argument(
        "--global-reconciliation-directory",
        type=Path,
        default=Path("artifacts/frontier-contract/reconciliations"),
    )
    parser.add_argument("--v4-route-plan", type=Path)
    parser.add_argument("--v4-route-audit", type=Path)
    parser.add_argument("--v4-route-closure", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--execution-closure", type=Path)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--process-timeout-seconds", type=int, default=3600)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "execute":
        if args.preflight is None:
            raise SystemExit("--preflight is required")
        receipt, closure = execute_preflight(
            preflight_path=args.preflight.resolve(),
            project_root=project_root,
            output_root=output_root,
            confirmation=args.confirm,
            process_timeout_seconds=args.process_timeout_seconds,
        )
        print(
            json.dumps(
                {
                    "status": "permanently_closed_after_execution",
                    "receipt": str(receipt),
                    "closure": str(closure),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "audit":
        if args.preflight is None or args.receipt is None or args.execution_closure is None:
            raise SystemExit("--preflight, --receipt, and --execution-closure are required")
        audit = build_postrun_audit(
            preflight_path=args.preflight.resolve(),
            receipt_path=args.receipt.resolve(),
            closure_path=args.execution_closure.resolve(),
            project_root=project_root,
            output_root=output_root,
        )
        path = _write_addressed(
            {key: value for key, value in audit.items() if key != "artifact_sha256"},
            directory=output_root,
            prefix="frontier-coverage-continuation-postrun-audit",
        )
        print(json.dumps({"status": audit["decision"], "audit": str(path)}, sort_keys=True))
        return
    required = {
        "v2_plan": args.v2_plan,
        "v3_plan": args.v3_plan,
        "v1_materialization": args.v1_materialization,
        "task_validity": args.task_validity,
        "stopped_audit": args.stopped_audit,
        "orphan_closure": args.orphan_closure,
        "v1_run_root": args.v1_run_root,
        "budget_audit": args.budget_audit,
        "v4_route_plan": args.v4_route_plan,
        "v4_route_audit": args.v4_route_audit,
        "v4_route_closure": args.v4_route_closure,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing or len(args.route_manifest) != 2:
        raise SystemExit(f"preflight inputs are incomplete: {missing}")
    resolved = {name: value.resolve() for name, value in required.items()}
    v1_root = resolved["v1_run_root"]
    route_manifests = [path.resolve() for path in args.route_manifest]
    bundle = build_runtime_bundle(
        project_root=project_root,
        v2_plan_path=resolved["v2_plan"],
        v3_plan_path=resolved["v3_plan"],
        v1_materialization_path=resolved["v1_materialization"],
        task_validity_path=resolved["task_validity"],
        route_manifest_paths=route_manifests,
        stopped_audit_path=resolved["stopped_audit"],
        orphan_closure_path=resolved["orphan_closure"],
        v1_ledger_path=v1_root / "ledger.jsonl",
        v1_source_directory=v1_root / "source",
        v1_response_directory=v1_root / "responses",
    )
    v4_route = _verify_v4_gate(
        project_root=project_root,
        plan_path=resolved["v4_route_plan"],
        audit_path=resolved["v4_route_audit"],
        closure_path=resolved["v4_route_closure"],
    )
    input_paths: dict[str, Any] = {
        "v2_plan": _input_record(project_root, resolved["v2_plan"]),
        "v3_plan": _input_record(project_root, resolved["v3_plan"]),
        "v1_materialization": _input_record(project_root, resolved["v1_materialization"]),
        "task_validity": _input_record(project_root, resolved["task_validity"]),
        "stopped_audit": _input_record(project_root, resolved["stopped_audit"]),
        "orphan_closure": _input_record(project_root, resolved["orphan_closure"]),
        "v1_ledger": _input_record(project_root, v1_root / "ledger.jsonl"),
        "v1_source_directory": _relative(project_root, v1_root / "source"),
        "v1_response_directory": _relative(project_root, v1_root / "responses"),
        "route_manifests": [_relative(project_root, path) for path in route_manifests],
        "global_ledger": _relative(project_root, args.global_ledger.resolve()),
    }
    supplemental = [
        SupplementalRun(
            source_directory=root.resolve() / "source",
            ledger_path=root.resolve() / "ledger.jsonl",
            corrections_directory=(
                root.resolve() / "corrections"
                if (root.resolve() / "corrections").exists()
                else None
            ),
        )
        for root in args.supplemental_run_root
    ]
    preflight = build_preflight(
        bundle=bundle,
        project_root=project_root,
        output_root=output_root,
        input_paths=input_paths,
        budget_audit_path=resolved["budget_audit"],
        supplemental_runs=supplemental,
        v1_run=SupplementalRun(v1_root / "source", v1_root / "ledger.jsonl"),
        v1_orphan_closure_path=resolved["orphan_closure"],
        global_ledger_path=args.global_ledger.resolve(),
        global_artifact_directory=args.global_artifact_directory.resolve(),
        global_corrections_directory=args.global_corrections_directory.resolve(),
        global_reconciliation_directory=args.global_reconciliation_directory.resolve(),
        v4_route=v4_route,
        environment=os.environ,
    )
    path = _write_addressed(
        {key: value for key, value in preflight.items() if key != "artifact_sha256"},
        directory=output_root,
        prefix="frontier-coverage-continuation-preflight",
    )
    print(
        json.dumps(
            {
                "status": preflight["status"],
                "preflight": str(path),
                "budget": preflight["budget"],
                "blockers": preflight["blockers"],
                "provider_calls": 0,
                "epicure_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
