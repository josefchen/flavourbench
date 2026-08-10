"""Append-only v4 recovery for the permanently closed coverage continuation.

The v2/v3 continuation is immutable.  This module first reconstructs its
terminal audit from raw sources, ledgers, responses, and frozen inputs.  It
then creates wholly new model-task cells in two ordered phases:

1. seven cells that were never reserved in the closed continuation; and
2. one GLM-specific replacement for the incomplete GLM substitution cell.

Every cell receives an independent terminal disposition.  A provider failure
therefore closes only that fresh work item and cannot stop later unrelated
cells.  The GLM-specific phase is additionally barred until a source-
reconstructing audit proves that all seven first-phase cells were dispositioned.

``freeze``, ``preflight``, and ``audit`` are network-free.  ``execute`` is the
only command that may start provider/MCP subprocesses and requires an exact
phase-specific confirmation string.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import frontier_coverage_continuation_executor as parent_executor
from .frontier_contract_runner import (
    AUTHORIZED_TOTAL_CAP_USD,
    DEFAULT_ADMISSION_FRACTION,
    AdmissionDenied,
    IntegrityError,
    _exclusive_runner_lock,
    _extract_artifact_path,
    _safe_process_hash,
    load_candidate_manifest,
    select_candidates,
)
from .frontier_contract_runner import (
    load_ledger as load_frontier_ledger,
)
from .frontier_coverage_continuation import (
    _attempt_slots,
    require_prefixed_credential_before_reservation,
)
from .frontier_coverage_continuation_executor import (
    ContinuationCell,
    RunPaths,
    RuntimeBundle,
    _audit_source,
    _finalize_source,
    _recovery_evidence,
    _source_for_work_item,
)
from .frontier_coverage_continuation_executor import (
    build_postrun_audit as build_parent_postrun_audit,
)
from .frontier_coverage_continuation_executor import (
    build_runtime_bundle as build_parent_runtime_bundle,
)
from .frontier_coverage_repair_executor import (
    SupplementalRun,
    _decimal,
    _decimal_text,
    _global_ledger_state,
    _run_accounting,
    _verify_budget_audit,
)
from .real_dataset_runner import (
    WorkItem,
    _dataset_ledger_lock,
    _subprocess_command,
    append_dataset_ledger_event,
    dataset_ledger_state,
    derive_conditions_forecast,
    load_dataset_ledger,
    load_development_task_inventory,
    scan_response_artifacts,
    task_registry_sha256,
)
from .real_task_bank import sha256_json
from .run_journal import JournalIntegrityError, load_run_journal

PLAN_SCHEMA_VERSION = "flavourbench-frontier-coverage-recovery-v4-plan-v1"
PREFLIGHT_SCHEMA_VERSION = "flavourbench-frontier-coverage-recovery-v4-preflight-v1"
RECEIPT_SCHEMA_VERSION = "flavourbench-frontier-coverage-recovery-v4-receipt-v1"
CLOSURE_SCHEMA_VERSION = "flavourbench-frontier-coverage-recovery-v4-closure-v1"
AUDIT_SCHEMA_VERSION = "flavourbench-frontier-coverage-recovery-v4-audit-v1"

RECOVERY_PHASE = "untouched_recovery"
GLM_PHASE = "glm_specific_replacement"
PHASES = (RECOVERY_PHASE, GLM_PHASE)
RECOVERY_CONFIRMATION = "RUN_EXACT_V4_UNTOUCHED_RECOVERY_14_REAL_ARMS"
GLM_CONFIRMATION = "RUN_EXACT_V4_GLM_REPLACEMENT_2_REAL_ARMS"
RECOVERY_NAMESPACE = uuid.UUID("86e516ee-3451-4b1e-9f5c-d64443b8c4f2")

PARENT_PREFLIGHT_SHA256 = "371d8f8ec33645d3bbe47db486939c71a7d745f49b076616318cb96414615b46"
PARENT_MATERIALIZATION_SHA256 = "0afc602aecf1240dfa3fc6a15daff15907505b42dc844b4f0e3285f84051682d"
PARENT_RECEIPT_SHA256 = "788ae734229a27aa7efeade9c4c60f95d106e0efda9b7408122cbdab64e1a0d9"
PARENT_CLOSURE_SHA256 = "b18eb9eb94bdc6d251cb5f33c5c94b64795c18fa50e56bacc233d38e6cc144c4"
PARENT_AUDIT_SHA256 = "4ac6e792d06f832e9e0215a47b28087f6b0ad91bd5de745b7e5f683c2eabc008"
PARENT_SOURCE_MIGRATION_SCHEMA = "flavourbench-historical-source-migration-v1"
PARENT_SOURCE_MIGRATION_SHA256 = (
    "5ffca007ec4085633babb8cb8c545071e7887bbd056bffb7101d76791d47ea73"
)
PARENT_SOURCE_MIGRATED_PATH = "src/flavourbench/service_cohere.py"
PARENT_SOURCE_MIGRATED_SHA256 = (
    "f1f335fa4edc56a11b51735c038b87d493040c30ff8d5d7ab6ec2d0084947628"
)
PARENT_SOURCE_MIGRATED_BYTES = 26_816
PARENT_SOURCE_DRIFT_FAILURE = "generation_or_auditor_source_changed_after_preflight"
HISTORICAL_EXPOSURE_VIEW_SCHEMA = "flavourbench-historical-exposure-view-v2"
HISTORICAL_EXPOSURE_VIEW_SHA256 = (
    "8ffb32261039af076101de5ad5426c1ecf75623a31addfeb66eb3ceb8270c562"
)
HISTORICAL_EXPOSURE_PLAN_SHA256 = (
    "730b426cfa5b7481446b4618166a2e6f75107c52ec26243283ef10ccbe01c0b8"
)
HISTORICAL_EXPOSURE_RECORD_COUNT = 248
HISTORICAL_EXPOSURE_RECORDS_SHA256 = (
    "75cd731c3101cf32007525172adad15c4cc86e2644a02695743d2951350d0e13"
)
HISTORICAL_EXPOSURE_SELF_OUTPUT_COUNT = 8
HISTORICAL_EXPOSURE_SELF_OUTPUTS_SHA256 = (
    "a238968bc0413509b22551d61074916b5659b29525c31147354716cfc82d5189"
)
HISTORICAL_EXPOSURE_OWNED_ROOTS = (
    "frontier-coverage-recovery-v4/untouched-recovery/source/",
    "frontier-coverage-recovery-v4/glm-specific-replacement/source/",
)

REASONING_V4_ROUTE_PLAN_SHA256 = "2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352"
REASONING_V4_RECEIPT_SHA256 = "172f4a08003656371de69c0907975f83761597338b159031b16052417d575852"
REASONING_V4_AUDIT_SHA256 = "c90617d7b6a8cab918bf0f50f7190f8ad8f49badb5ce036c7c9fa716d7d9a959"
REASONING_V4_CLOSURE_SHA256 = "807aa054e7f0aaaa770630adae7696bba8fc24251d7ed2b08082b46a0edfde87"
REASONING_V4_GEMINI_WORK_ITEM_ID = (
    "189d76023f42d7b14912b61daa8b98fde587b86b05695908b522b45ba9175002"
)
REASONING_V4_GEMINI_RUN_ID = "19125098-99b0-58af-b87b-a6260a9c5bd3"
REASONING_V4_GEMINI_RESERVATION_SHA256 = (
    "db19e86ac60a9fa9d0c34a7787b7b383e4aa2b3ec30eec4006628ffd7e8a4e26"
)
REASONING_V4_GEMINI_INCIDENT_SHA256 = (
    "86a99483395c09d57fcc1ada43bce5a8c4a2e5930f2a554ac099f47f02291e0c"
)
REASONING_V4_GEMINI_ERROR_SHA256 = (
    "71cd44184907309cc160fb501e395865355a417f4907a1a6ec1e3a6fa3ef0e83"
)
REASONING_V4_GEMINI_RESERVE_USD = Decimal("0.6765315")
REASONING_V4_LEDGER_SHA256 = "90d2e9a8092dbb4af286159afad2304257e3300feedf4d1460313c37ff1f1783"
REASONING_V4_JOURNAL_SHA256 = "2ec728d34d01bb758e5f91fd78f2e3c1e0bc68aa3ec3984ad9d97fa7b2dda152"

REASONING_V5_ENDPOINT_SNAPSHOT_SHA256 = (
    "ce46706dd7c2cb0605c3dd5abc34f36714f09a6074e155b18298393f14a38262"
)
REASONING_V5_ROUTE_PLAN_SHA256 = "0481ecd9c8260967275e18a72d4ed265352d35ca2254f554ba55053bc61bb71c"
REASONING_V5_SONNET_RECEIPT_SHA256 = (
    "6b54b77c744016dd17714b25f7f0e2795600fb204d02e37f11165451a35de7a6"
)
REASONING_V5_SONNET_AUDIT_SHA256 = (
    "4c5e4a6fb796f9791fbf5e1889d3a09fd52fa3a1e67f0e50a0dbb6daeba49feb"
)
REASONING_V5_SONNET_CLOSURE_SHA256 = (
    "99c194969edabe33ebfb942c1bf053515c871c953c65b8b8372400c4b245f068"
)
REASONING_V5_GEMINI_RECEIPT_SHA256 = (
    "157e3aaeb8faf02830c927ddbe035dcb7414900cf900c59bd23db57bf918b803"
)
REASONING_V5_GEMINI_AUDIT_SHA256 = (
    "63da19f18b9c2f3104d6ef775969cc1a0c8750ef5bcacf03cf9f6bfdd0223f23"
)
REASONING_V5_GEMINI_CLOSURE_SHA256 = (
    "44ba45a5c967744ffb9d9b107511a3104c4b0dfd6d708c8fdfddcf28c5ce0c04"
)
REASONING_V5_AGGREGATE_AUDIT_SHA256 = (
    "30271cb2108274271700be203d0eb3c7efde53875ca927c5425021ba27c32a35"
)
REASONING_V5_AGGREGATE_CLOSURE_SHA256 = (
    "e6ce615dbe15c29ae8066990371f7512b532eea2689ddb83fd5917b059d8859b"
)
REASONING_V5_SONNET_LEDGER_SHA256 = (
    "bcef4bc8fcc99ab6a7303e106c24903d37baaf225d8cb21a4fcf42343567c592"
)
REASONING_V5_GEMINI_LEDGER_SHA256 = (
    "03c0f15d6c73c6bdcca930e63e7e16528583fab88605720a46d46f7b439795eb"
)
REASONING_V5_SONNET_LEDGER_HEAD_SHA256 = (
    "36971758fda854a7e179b15fbeffecd1e90befccac9b7e77fd13ea37335a71fc"
)
REASONING_V5_GEMINI_LEDGER_HEAD_SHA256 = (
    "902a8af19270f8d0c5d9adaea13d80c40c9c1055db48d475866773d5cb9a055e"
)
REASONING_V5_SONNET_SOURCE_EXPOSURE_USD = Decimal("0.061742")
REASONING_V5_SONNET_AUDIT_COST_USD = Decimal("0.033475")
REASONING_V5_GEMINI_SOURCE_EXPOSURE_USD = Decimal("0.066503")
REASONING_V5_TOTAL_SOURCE_EXPOSURE_USD = Decimal("0.128245")
FAILED_PRE_RESERVATION_PREFLIGHT_SHA256 = (
    "5b219789f1d3862b0652ba573d449fb905ab851db2eb2bdda4c2a5e938b88592"
)
FAILED_PRE_RESERVATION_SUPERSESSION_SCHEMA = (
    "flavourbench-historical-pre-reservation-supersession-v1"
)
FAILED_PRE_RESERVATION_SUPERSESSION_SHA256 = (
    "873a00db424c8dc5d27dc12e251a7126f9b411534c667d7f0ec080738dffb8cf"
)
V4_EXECUTION_PREFLIGHT_SHA256 = (
    "c2cf6aa4d6397f6034114dfb9ead0b446895256a7a84705fbd3a55c70d742268"
)

PARENT_COMPLETE_WORK_ITEM = "650b3ced16656fbd66460d556128614385a4411be217f061769420b766d74ad3"
PARENT_INCOMPLETE_GLM_WORK_ITEM = "6033ee45cacbaf80257b18d820d3f90b095bda6d756fa3af65fd6d628da76ab6"

# Six non-GLM cells are deliberately first.  The previously untouched GLM
# composition cell is seventh.  Only after the whole phase is dispositioned
# may the separate GLM substitution replacement be admitted.
CELL_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        RECOVERY_PHASE,
        "242a96c4a9ac7912a95490d5df20f013d90ccb12e855423a830f1977107c5844",
        "fb-s0-evidence-001",
    ),
    (
        RECOVERY_PHASE,
        "1341863cf5c9f66a847ea15e0026068fb6644a7f65ff0efa18f64cafae9e8888",
        "fb-s0-evidence-006",
    ),
    (
        RECOVERY_PHASE,
        "7468dc89ca8f7a9cdc0db242d1e5411c9be2c900cfac6522f589ea6184a54a8c",
        "fb-s0-evidence-024",
    ),
    (
        RECOVERY_PHASE,
        "290165c68a641b31bfce99d481849356e37490680f618f7231a93b77486f58c7",
        "fb-s0-substitution-012",
    ),
    (
        RECOVERY_PHASE,
        "43a22f2cf2869bb016d42dc56101f5090a744585d845fe15c6ca879c225b7fd0",
        "fb-s0-cookability-002",
    ),
    (
        RECOVERY_PHASE,
        "90dbbba3412e8593310df9a6addd64e04b8c2cb8acce75e0836de301f279dd41",
        "fb-s0-cookability-004",
    ),
    (
        RECOVERY_PHASE,
        "f0ab8e1b38daef642fa0568b4dc755ae479246421136226eb2d7444e2434969b",
        "fb-s0-composition-002",
    ),
    (
        GLM_PHASE,
        PARENT_INCOMPLETE_GLM_WORK_ITEM,
        "fb-s0-substitution-027",
    ),
)

QUARANTINED_TASK_IDS = frozenset(
    {
        "fb-s0-composition-006",
        "fb-s0-composition-008",
        "fb-s0-composition-009",
        "fb-s0-cookability-003",
    }
)

EXPECTED_PARENT_STATUSES = {
    PARENT_COMPLETE_WORK_ITEM: "source_reconstructed_complete",
    PARENT_INCOMPLETE_GLM_WORK_ITEM: "failed_closed",
    **{source_id: "not_started_after_permanent_stop" for _, source_id, _ in CELL_SPECS[:7]},
}

SOURCE_FILES = (
    "src/flavourbench/frontier_coverage_recovery_v4.py",
    "src/flavourbench/frontier_coverage_continuation_executor.py",
    "src/flavourbench/frontier_coverage_continuation.py",
    "src/flavourbench/frontier_coverage_repair_executor.py",
    "src/flavourbench/continuation_openrouter_pair.py",
    "src/flavourbench/real_dataset_runner.py",
    "src/flavourbench/live_smoke.py",
    "src/flavourbench/direct_cohere_pair.py",
    "src/flavourbench/provider.py",
    "src/flavourbench/service_cohere.py",
    "src/flavourbench/run_journal.py",
    "src/flavourbench/reasoning_effort_route_gate_v4.py",
    "src/flavourbench/reasoning_effort_route_gate_v5.py",
)

OBSERVATIONAL_SOURCE_FAILURES = frozenset(
    {
        "epicure_off_result_missing",
        "epicure_on_result_missing",
        "epicure_off_answer_not_substantive",
        "epicure_on_answer_not_substantive",
        "incomplete_generation_metadata_present",
        "epicure_on_has_no_successful_tool_call",
    }
)


@dataclass(frozen=True)
class ParentState:
    bundle: RuntimeBundle
    preflight: Mapping[str, Any]
    receipt: Mapping[str, Any]
    closure: Mapping[str, Any]
    audit: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
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
    expected_schema: str | None = None,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    value = _load_json(path, label=label)
    digest = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if (
        not _is_sha256(digest)
        or sha256_json(payload) != digest
        or str(digest) not in path.name
        or (expected_schema is not None and value.get("schema_version") != expected_schema)
        or (expected_digest is not None and digest != expected_digest)
    ):
        raise IntegrityError(f"{label} schema or content address does not verify")
    return value


def _write_addressed(payload: Mapping[str, Any], *, directory: Path, prefix: str) -> Path:
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


def _relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError as error:
        raise IntegrityError("bound path is outside the FlavourBench project root") from error


def _bound_path(project_root: Path, record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise IntegrityError(f"{label} binding is absent")
    path = project_root / str(record.get("path") or "")
    if _file_sha256(path) != record.get("physical_sha256"):
        raise IntegrityError(f"{label} physical digest changed")
    return path


def _source_bundle(project_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"execution source is missing or non-regular: {relative}")
        files.append({"path": relative, "sha256": _file_sha256(path), "bytes": path.stat().st_size})
    return {"files": files, "bundle_sha256": sha256_json(files)}


def _artifact(parent_root: Path, prefix: str, digest: str) -> Path:
    path = parent_root / f"{prefix}-{digest}.json"
    if not path.is_file():
        raise IntegrityError(f"parent artifact is missing: {path.name}")
    return path


def _parent_source_migration_path(project_root: Path) -> Path:
    return (
        project_root
        / "artifacts/season1/current-quality-run/historical-source-migrations"
        / f"historical-source-migration-{PARENT_SOURCE_MIGRATION_SHA256}.json"
    )


def _verify_parent_historical_source_view(
    *,
    project_root: Path,
    preflight: Mapping[str, Any],
    migration_path: Path | None = None,
    current_source_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify one archived historical source without governing prospective code.

    Every non-migrated source remains byte-identical to the old preflight.  The
    sole exception is reconstructed from an exact regular-file archive whose
    bytes and digest equal the original preflight row.  The current adapter is
    still inspected for path-set integrity, but its bytes are deliberately not
    substituted into prospective execution.
    """

    path = migration_path or _parent_source_migration_path(project_root)
    migration = _load_addressed(
        path,
        label="historical source migration",
        expected_schema=PARENT_SOURCE_MIGRATION_SCHEMA,
        expected_digest=PARENT_SOURCE_MIGRATION_SHA256,
    )
    expected_bundle = preflight.get("source_code")
    if not isinstance(expected_bundle, Mapping):
        raise IntegrityError("parent preflight source bundle is absent")
    if (
        migration.get("historical_preflight_sha256") != PARENT_PREFLIGHT_SHA256
        or migration.get("historical_audit_sha256") != PARENT_AUDIT_SHA256
        or migration.get("historical_source_bundle_sha256")
        != expected_bundle.get("bundle_sha256")
        or migration.get("allowed_current_divergence_paths")
        != [PARENT_SOURCE_MIGRATED_PATH]
        or migration.get("prospective_execution_source_override") is not False
        or migration.get("provider_or_epicure_calls_made")
        != {"provider_completions": 0, "epicure": 0}
    ):
        raise IntegrityError("historical source migration binding differs")

    expected_files = expected_bundle.get("files")
    if not isinstance(expected_files, list):
        raise IntegrityError("parent preflight source inventory is absent")
    expected_by_path = {
        str(record.get("path") or ""): dict(record)
        for record in expected_files
        if isinstance(record, Mapping)
    }
    if len(expected_by_path) != len(expected_files):
        raise IntegrityError("parent preflight source inventory is malformed")
    migrated = expected_by_path.get(PARENT_SOURCE_MIGRATED_PATH)
    archived = migration.get("archived_source")
    if (
        migrated
        != {
            "path": PARENT_SOURCE_MIGRATED_PATH,
            "sha256": PARENT_SOURCE_MIGRATED_SHA256,
            "bytes": PARENT_SOURCE_MIGRATED_BYTES,
        }
        or not isinstance(archived, Mapping)
        or archived.get("original_path") != PARENT_SOURCE_MIGRATED_PATH
        or archived.get("sha256") != PARENT_SOURCE_MIGRATED_SHA256
        or archived.get("bytes") != PARENT_SOURCE_MIGRATED_BYTES
    ):
        raise IntegrityError("archived historical source identity differs")
    archive_path = project_root / str(archived.get("path") or "")
    try:
        archive_path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise IntegrityError("historical source archive escapes the project root") from error
    if (
        archive_path.is_symlink()
        or not archive_path.is_file()
        or archive_path.stat().st_size != PARENT_SOURCE_MIGRATED_BYTES
        or _file_sha256(archive_path) != PARENT_SOURCE_MIGRATED_SHA256
    ):
        raise IntegrityError("historical source archive is missing or tampered")

    observed_bundle = (
        dict(current_source_bundle)
        if current_source_bundle is not None
        else parent_executor._source_bundle(project_root)
    )
    observed_files = observed_bundle.get("files")
    if not isinstance(observed_files, list):
        raise IntegrityError("current parent source inventory is absent")
    observed_by_path = {
        str(record.get("path") or ""): dict(record)
        for record in observed_files
        if isinstance(record, Mapping)
    }
    if set(observed_by_path) != set(expected_by_path):
        raise IntegrityError("current and historical source path sets differ")
    for source_path, expected in expected_by_path.items():
        if source_path == PARENT_SOURCE_MIGRATED_PATH:
            continue
        if observed_by_path[source_path] != expected:
            raise IntegrityError(f"unmigrated historical source differs: {source_path}")
    return migration


def _verify_parent_audit_source_migration(
    *,
    project_root: Path,
    preflight: Mapping[str, Any],
    frozen_audit: Mapping[str, Any],
    rebuilt_audit: Mapping[str, Any],
) -> None:
    """Accept only the exact audit delta caused by the governed source move."""

    migration = _verify_parent_historical_source_view(
        project_root=project_root,
        preflight=preflight,
    )
    frozen_failures = frozen_audit.get("failures")
    rebuilt_failures = rebuilt_audit.get("failures")
    if not isinstance(frozen_failures, list) or not isinstance(rebuilt_failures, list):
        raise IntegrityError("historical audit failure inventory is malformed")
    if sorted(rebuilt_failures) != sorted([*frozen_failures, PARENT_SOURCE_DRIFT_FAILURE]):
        raise IntegrityError("historical audit differs beyond the governed source move")
    normalized = copy.deepcopy(dict(rebuilt_audit))
    normalized["failures"] = [
        value for value in rebuilt_failures if value != PARENT_SOURCE_DRIFT_FAILURE
    ]
    normalized.pop("artifact_sha256", None)
    normalized["artifact_sha256"] = sha256_json(normalized)
    if (
        normalized != dict(frozen_audit)
        or migration.get("drifted_rebuild_sha256") != rebuilt_audit.get("artifact_sha256")
        or migration.get("normalized_rebuild_sha256") != PARENT_AUDIT_SHA256
        or migration.get("allowed_audit_delta")
        != {"added_failure": PARENT_SOURCE_DRIFT_FAILURE}
    ):
        raise IntegrityError("historical audit migration does not recover the frozen audit")


def reconstruct_parent(*, project_root: Path, parent_root: Path) -> ParentState:
    preflight_path = _artifact(
        parent_root, "frontier-coverage-continuation-preflight", PARENT_PREFLIGHT_SHA256
    )
    receipt_path = _artifact(
        parent_root, "frontier-coverage-continuation-receipt", PARENT_RECEIPT_SHA256
    )
    closure_path = _artifact(
        parent_root, "frontier-coverage-continuation-closure", PARENT_CLOSURE_SHA256
    )
    audit_path = _artifact(
        parent_root, "frontier-coverage-continuation-postrun-audit", PARENT_AUDIT_SHA256
    )
    preflight = _load_addressed(
        preflight_path,
        label="parent preflight",
        expected_schema="flavourbench-frontier-coverage-continuation-preflight-v1",
        expected_digest=PARENT_PREFLIGHT_SHA256,
    )
    receipt = _load_addressed(
        receipt_path,
        label="parent receipt",
        expected_schema="flavourbench-frontier-coverage-continuation-receipt-v1",
        expected_digest=PARENT_RECEIPT_SHA256,
    )
    closure = _load_addressed(
        closure_path,
        label="parent closure",
        expected_schema="flavourbench-frontier-coverage-continuation-closure-v1",
        expected_digest=PARENT_CLOSURE_SHA256,
    )
    audit = _load_addressed(
        audit_path,
        label="parent audit",
        expected_schema="flavourbench-frontier-coverage-continuation-postrun-audit-v1",
        expected_digest=PARENT_AUDIT_SHA256,
    )
    rebuilt_audit = build_parent_postrun_audit(
        preflight_path=preflight_path,
        receipt_path=receipt_path,
        closure_path=closure_path,
        project_root=project_root,
        output_root=parent_root,
    )
    if rebuilt_audit != audit:
        _verify_parent_audit_source_migration(
            project_root=project_root,
            preflight=preflight,
            frozen_audit=audit,
            rebuilt_audit=rebuilt_audit,
        )
    exact = preflight.get("exact_inputs")
    if not isinstance(exact, Mapping):
        raise IntegrityError("parent preflight exact inputs are absent")
    route_manifests = [project_root / str(value) for value in exact["route_manifests"]]
    bundle = build_parent_runtime_bundle(
        project_root=project_root,
        v2_plan_path=_bound_path(project_root, exact.get("v2_plan"), label="parent v2 plan"),
        v3_plan_path=_bound_path(project_root, exact.get("v3_plan"), label="parent v3 plan"),
        v1_materialization_path=_bound_path(
            project_root, exact.get("v1_materialization"), label="parent v1 materialization"
        ),
        task_validity_path=_bound_path(
            project_root, exact.get("task_validity"), label="parent task validity"
        ),
        route_manifest_paths=route_manifests,
        stopped_audit_path=_bound_path(
            project_root, exact.get("stopped_audit"), label="parent stopped audit"
        ),
        orphan_closure_path=_bound_path(
            project_root, exact.get("orphan_closure"), label="parent orphan closure"
        ),
        v1_ledger_path=_bound_path(project_root, exact.get("v1_ledger"), label="parent v1 ledger"),
        v1_source_directory=project_root / str(exact["v1_source_directory"]),
        v1_response_directory=project_root / str(exact["v1_response_directory"]),
    )
    if bundle.document.get("artifact_sha256") != PARENT_MATERIALIZATION_SHA256:
        raise IntegrityError("parent materialization digest differs")
    statuses = {
        str(cell.get("work_item_id") or ""): str(cell.get("status") or "")
        for cell in audit.get("cells") or []
        if isinstance(cell, Mapping)
    }
    if any(statuses.get(work_id) != status for work_id, status in EXPECTED_PARENT_STATUSES.items()):
        raise IntegrityError("parent complete/incomplete/untouched disposition differs")
    if (
        receipt.get("status") != "failed_closed"
        or closure.get("status") != "closed_failed_incomplete"
        or closure.get("safe_to_replay_any_reserved_or_planned_work") is not False
        or audit.get("counts", {}).get("usable_cells") != 1
        or audit.get("counts", {}).get("planned_cells") != 9
    ):
        raise IntegrityError("parent terminal policy differs")
    return ParentState(bundle, preflight, receipt, closure, audit)


def _scan_model_task_exposure(root: Path, *, model_ids: set[str]) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("exposure root must be a regular directory")
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        model_id = document.get("requested_model_id")
        task_id = document.get("dataset_task_id")
        if model_id not in model_ids or not isinstance(task_id, str):
            continue
        records.append(
            {
                "model_id": str(model_id),
                "task_id": task_id,
                "path": str(path.relative_to(root)),
                "physical_sha256": _file_sha256(path),
                "artifact_sha256": str(document.get("artifact_sha256") or ""),
            }
        )
    return {
        "root": str(root),
        "record_count": len(records),
        "records_sha256": sha256_json(records),
        "records": records,
    }


def _historical_exposure_view_path(project_root: Path) -> Path:
    return (
        project_root
        / "artifacts/season1/current-quality-run/historical-source-migrations"
        / f"historical-exposure-view-{HISTORICAL_EXPOSURE_VIEW_SHA256}.json"
    )


def _assert_no_model_task_overlap(
    *, cells: Sequence[Mapping[str, Any]], exposure_records: Sequence[Mapping[str, Any]]
) -> None:
    exposed = {
        (str(record.get("model_id") or ""), str(record.get("task_id") or ""))
        for record in exposure_records
    }
    overlap = sorted(
        (str(cell.get("model_id") or ""), str(cell.get("task_id") or ""))
        for cell in cells
        if (
            str(cell.get("model_id") or ""),
            str(cell.get("task_id") or ""),
        )
        in exposed
    )
    if overlap:
        raise IntegrityError("historical plan contains a pre-freeze model-task overlap")


def _verify_historical_exposure_view(
    *,
    project_root: Path,
    exposure_root: Path,
    historical_plan: Mapping[str, Any],
    view_path: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct the exact exposure universe visible when the plan froze.

    The embedded snapshot is content-addressed by the frozen plan and every
    listed physical source is rehashed.  The only later records excluded are
    the eight exact outputs created by this plan under its two owned source
    roots.  Any other later record remains fatal.  The ordinary planner still
    performs a live recursive scan when ``historical_plan`` is absent.
    """

    view = _load_addressed(
        view_path or _historical_exposure_view_path(project_root),
        label="historical exposure view",
        expected_schema=HISTORICAL_EXPOSURE_VIEW_SCHEMA,
        expected_digest=HISTORICAL_EXPOSURE_VIEW_SHA256,
    )
    snapshot = historical_plan.get("model_task_exposure_snapshot")
    if not isinstance(snapshot, Mapping):
        raise IntegrityError("historical plan exposure snapshot is absent")
    records = snapshot.get("records")
    if not isinstance(records, list):
        raise IntegrityError("historical plan exposure records are absent")
    self_outputs = view.get("exact_later_self_outputs")
    policy = view.get("policy")
    if (
        historical_plan.get("artifact_sha256") != HISTORICAL_EXPOSURE_PLAN_SHA256
        or view.get("historical_plan_sha256") != HISTORICAL_EXPOSURE_PLAN_SHA256
        or view.get("supersedes_artifact_sha256")
        != "1a3d0802b1faaabef09f8e0a59f054d8d31b128d976dc718d1f0c201085455b3"
        or view.get("snapshot_field") != "model_task_exposure_snapshot"
        or view.get("record_count") != HISTORICAL_EXPOSURE_RECORD_COUNT
        or view.get("records_sha256") != HISTORICAL_EXPOSURE_RECORDS_SHA256
        or snapshot.get("record_count") != HISTORICAL_EXPOSURE_RECORD_COUNT
        or snapshot.get("records_sha256") != HISTORICAL_EXPOSURE_RECORDS_SHA256
        or len(records) != HISTORICAL_EXPOSURE_RECORD_COUNT
        or sha256_json(records) != HISTORICAL_EXPOSURE_RECORDS_SHA256
        or view.get("owned_execution_output_roots")
        != list(HISTORICAL_EXPOSURE_OWNED_ROOTS)
        or not isinstance(self_outputs, Mapping)
        or self_outputs.get("record_count") != HISTORICAL_EXPOSURE_SELF_OUTPUT_COUNT
        or self_outputs.get("records_sha256")
        != HISTORICAL_EXPOSURE_SELF_OUTPUTS_SHA256
        or not isinstance(self_outputs.get("records"), list)
        or len(self_outputs.get("records")) != HISTORICAL_EXPOSURE_SELF_OUTPUT_COUNT
        or sha256_json(self_outputs.get("records"))
        != HISTORICAL_EXPOSURE_SELF_OUTPUTS_SHA256
        or policy
        != {
            "historical_replay_only": True,
            "live_scan_remains_default_for_new_plans": True,
            "all_nonowned_current_records_must_equal_frozen_snapshot": True,
            "same_model_task_artifact_outside_owned_roots_is_fatal": True,
            "unknown_output_inside_owned_roots_is_fatal": True,
            "later_self_outputs_change_historical_view": False,
        }
        or view.get("provider_or_epicure_calls_made")
        != {"provider_completions": 0, "epicure": 0}
        or view.get("official") is not False
        or view.get("rank_eligible") is not False
    ):
        raise IntegrityError("historical exposure view binding differs")

    observed_paths: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise IntegrityError("historical exposure record is malformed")
        relative = str(raw.get("path") or "")
        if not relative or relative in observed_paths:
            raise IntegrityError("historical exposure path is absent or duplicated")
        observed_paths.add(relative)
        source_path = exposure_root / relative
        try:
            source_path.resolve().relative_to(exposure_root.resolve())
        except ValueError as error:
            raise IntegrityError("historical exposure source escapes its root") from error
        if (
            source_path.is_symlink()
            or not source_path.is_file()
            or _file_sha256(source_path) != raw.get("physical_sha256")
        ):
            raise IntegrityError(f"historical exposure source differs: {relative}")
        document = _load_json(source_path, label="historical exposure source")
        if (
            document.get("artifact_sha256") != raw.get("artifact_sha256")
            or document.get("requested_model_id") != raw.get("model_id")
            or document.get("dataset_task_id") != raw.get("task_id")
        ):
            raise IntegrityError(f"historical exposure semantics differ: {relative}")

    model_ids = {
        str(cell.get("model_id") or "")
        for cell in historical_plan.get("cells") or []
        if isinstance(cell, Mapping)
    }
    live_records = _scan_model_task_exposure(exposure_root, model_ids=model_ids)["records"]
    owned_records = [
        record
        for record in live_records
        if any(
            str(record.get("path") or "").startswith(prefix)
            for prefix in HISTORICAL_EXPOSURE_OWNED_ROOTS
        )
    ]
    nonowned_records = [record for record in live_records if record not in owned_records]
    if owned_records != self_outputs.get("records"):
        raise IntegrityError("historical replay owned self-output set differs")
    if nonowned_records != records:
        raise IntegrityError("historical replay has a later nonowned exposure")
    _assert_no_model_task_overlap(
        cells=historical_plan.get("cells") or [], exposure_records=records
    )
    return copy.deepcopy(dict(snapshot))


def _prior_identifiers(parent: ParentState) -> set[str]:
    identifiers: set[str] = set()
    for cell in parent.bundle.cells:
        identifiers.update(
            {
                cell.cell_id,
                cell.run_id,
                cell.work_item.work_item_id,
                *map(str, cell.arm_ids.values()),
                *(str(slot.get("attempt_id") or "") for slot in cell.attempt_slots),
            }
        )
    for outcome in parent.receipt.get("outcomes") or []:
        if isinstance(outcome, Mapping):
            identifiers.add(str(outcome.get("work_item_id") or ""))
    identifiers.discard("")
    return identifiers


def _task_and_manifest_inputs(
    *, project_root: Path, parent: ParentState
) -> tuple[Path, tuple[Path, ...]]:
    exact = parent.preflight["exact_inputs"]
    task_path = _bound_path(project_root, exact.get("task_validity"), label="task validity")
    manifests = tuple(project_root / str(value) for value in exact["route_manifests"])
    for path in manifests:
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("route manifest disappeared")
    return task_path, manifests


def _runtime_cells(
    *,
    project_root: Path,
    parent: ParentState,
    plan: Mapping[str, Any],
) -> tuple[ContinuationCell, ...]:
    task_path, manifests = _task_and_manifest_inputs(project_root=project_root, parent=parent)
    tasks, _ = load_development_task_inventory(task_path)
    task_index = {task.public_id: task for task in tasks}
    registry_sha = task_registry_sha256(tasks)
    candidates: dict[tuple[str, str, str], tuple[Any, Path, str]] = {}
    for path in manifests:
        manifest = load_candidate_manifest(path, expected_digest="")
        digest = str(manifest.get("content_address", {}).get("digest") or "")
        for candidate in select_candidates(manifest):
            key = (
                candidate.model_id,
                candidate.provider_tag,
                candidate.execution_backend,
            )
            candidates[key] = (candidate, path, digest)
    parent_cells = {cell.work_item.work_item_id: cell for cell in parent.bundle.cells}
    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 8:
        raise IntegrityError("v4 plan does not contain eight exact cells")
    runtime: list[ContinuationCell] = []
    identifiers: set[str] = set()
    for ordinal, raw in enumerate(raw_cells, start=1):
        if not isinstance(raw, Mapping):
            raise IntegrityError("v4 cell is not an object")
        source = parent_cells.get(str(raw.get("source_closed_work_item_id") or ""))
        task = task_index.get(str(raw.get("task_id") or ""))
        if source is None or task is None:
            raise IntegrityError("v4 source cell or task is absent")
        key = (
            str(raw.get("model_id") or ""),
            str(raw.get("provider_tag") or ""),
            str(raw.get("execution_backend") or ""),
        )
        selected = candidates.get(key)
        if selected is None:
            raise IntegrityError("v4 route is not in the frozen parent manifests")
        candidate, manifest_path, manifest_sha = selected
        if (
            source.work_item.candidate.model_id != candidate.model_id
            or source.route_manifest_sha256 != manifest_sha
            or raw.get("route_manifest_sha256") != manifest_sha
            or raw.get("prompt_sha256") != task.prompt_sha256
            or raw.get("task_family") != task.family
            or task.public_id in QUARANTINED_TASK_IDS
            or raw.get("phase") not in PHASES
        ):
            raise IntegrityError("v4 cell route, task, phase, or quarantine binding differs")
        arm_ids = raw.get("arm_ids")
        attempt_slots = raw.get("attempt_slots")
        if (
            not _is_sha256(raw.get("work_item_id"))
            or not _is_sha256(raw.get("cell_id"))
            or not isinstance(arm_ids, Mapping)
            or set(arm_ids) != {"epicure_off", "epicure_on"}
            or not all(_is_sha256(value) for value in arm_ids.values())
            or not isinstance(attempt_slots, list)
            or not attempt_slots
        ):
            raise IntegrityError("v4 identifiers are malformed")
        try:
            uuid.UUID(str(raw.get("run_id") or ""))
        except ValueError as error:
            raise IntegrityError("v4 run ID is malformed") from error
        new = {
            str(raw["work_item_id"]),
            str(raw["cell_id"]),
            str(raw["run_id"]),
            *map(str, arm_ids.values()),
            *(
                str(slot.get("attempt_id") or "")
                for slot in attempt_slots
                if isinstance(slot, Mapping)
            ),
        }
        if "" in new or identifiers.intersection(new):
            raise IntegrityError("v4 identifiers overlap within the plan")
        identifiers.update(new)
        work_item = WorkItem(
            ordinal=ordinal,
            work_item_id=str(raw["work_item_id"]),
            manifest_sha256=manifest_sha,
            task_registry_sha256=registry_sha,
            task=task,
            candidate=candidate,
            endpoint_execution_sha256=candidate.endpoint_execution_sha256,
            execution_policy_sha256=parent.bundle.execution_policy.sha256,
            execution_policy=parent.bundle.execution_policy,
        )
        forecast = derive_conditions_forecast(
            work_item,
            policy=parent.bundle.execution_policy,
            conditions=("epicure_off", "epicure_on"),
        )
        if _decimal_text(forecast.forecast_usd) != raw.get("reserved_worst_case_usd"):
            raise IntegrityError("v4 cell forecast no longer reconstructs")
        runtime.append(
            ContinuationCell(
                plan_kind=str(raw["phase"]),
                plan_sha256=str(plan["artifact_sha256"]),
                cell_id=str(raw["cell_id"]),
                run_id=str(raw["run_id"]),
                arm_ids={key: str(value) for key, value in arm_ids.items()},
                attempt_slots=tuple(dict(slot) for slot in attempt_slots),
                work_item=work_item,
                route_manifest_path=manifest_path,
                route_manifest_sha256=manifest_sha,
                forecast_usd=forecast.forecast_usd,
                source_work_item_id=str(raw["source_closed_work_item_id"]),
            )
        )
    if [cell.plan_kind for cell in runtime] != [RECOVERY_PHASE] * 7 + [GLM_PHASE]:
        raise IntegrityError("v4 phase order differs")
    if any(cell.work_item.candidate.model_id == "z-ai/glm-5.2" for cell in runtime[:6]):
        raise IntegrityError("GLM appears before unrelated recovery cells")
    return tuple(runtime)


def build_plan(
    *,
    project_root: Path,
    parent_root: Path,
    quarantine_path: Path,
    exposure_root: Path,
    historical_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parent = reconstruct_parent(project_root=project_root, parent_root=parent_root)
    quarantine = _load_addressed(
        quarantine_path,
        label="task quarantine",
        expected_schema="flavourbench-current-frontier-task-quarantine-v1",
        expected_digest="e095c45ed27b0639a8eefae13a028c653fdea493999e095c2a757818ebbb7a15",
    )
    if {str(item.get("task_id")) for item in quarantine.get("records") or []} != set(
        QUARANTINED_TASK_IDS
    ):
        raise IntegrityError("task quarantine set differs")
    task_path, _ = _task_and_manifest_inputs(project_root=project_root, parent=parent)
    tasks, task_source = load_development_task_inventory(task_path)
    task_index = {task.public_id: task for task in tasks}
    parent_cells = {cell.work_item.work_item_id: cell for cell in parent.bundle.cells}
    model_ids = {
        parent_cells[source_id].work_item.candidate.model_id for _, source_id, _ in CELL_SPECS
    }
    exposure = (
        _verify_historical_exposure_view(
            project_root=project_root,
            exposure_root=exposure_root,
            historical_plan=historical_plan,
        )
        if historical_plan is not None
        else _scan_model_task_exposure(exposure_root, model_ids=model_ids)
    )
    exposed_pairs = {
        (str(record["model_id"]), str(record["task_id"])) for record in exposure["records"]
    }
    old_identifiers = _prior_identifiers(parent)
    new_identifiers: set[str] = set()
    cells: list[dict[str, Any]] = []
    for ordinal, (phase, source_id, task_id) in enumerate(CELL_SPECS, start=1):
        source = parent_cells.get(source_id)
        task = task_index.get(task_id)
        if source is None or task is None:
            raise IntegrityError("v4 source or alternate task is absent")
        model_id = source.work_item.candidate.model_id
        if (
            task.family != source.work_item.task.family
            or task.public_id in QUARANTINED_TASK_IDS
            or (model_id, task.public_id) in exposed_pairs
            or task.public_id == source.work_item.task.public_id
        ):
            raise IntegrityError("alternate task is quarantined, exposed, drifted, or unchanged")
        basis = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "phase": phase,
            "phase_ordinal": ordinal if phase == RECOVERY_PHASE else 1,
            "source_closed_work_item_id": source_id,
            "source_closed_cell_id": source.cell_id,
            "model_id": model_id,
            "provider_tag": source.work_item.candidate.provider_tag,
            "execution_backend": source.work_item.candidate.execution_backend,
            "route_manifest_sha256": source.route_manifest_sha256,
            "task_id": task.public_id,
            "task_family": task.family,
            "prompt_sha256": task.prompt_sha256,
            "conditions": ["epicure_off", "epicure_on"],
            "development_only": True,
        }
        cell_id = sha256_json({**basis, "identifier_role": "cell"})
        work_id = sha256_json({**basis, "identifier_role": "work_item"})
        run_id = str(uuid.uuid5(RECOVERY_NAMESPACE, f"run:{cell_id}"))
        arm_ids = {
            condition: sha256_json(
                {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "work_item_id": work_id,
                    "condition": condition,
                }
            )
            for condition in ("epicure_off", "epicure_on")
        }
        attempts = _attempt_slots(
            run_id,
            cell_id,
            ("epicure_off", "epicure_on"),
            namespace=RECOVERY_NAMESPACE,
        )
        identifiers = {
            cell_id,
            work_id,
            run_id,
            *arm_ids.values(),
            *(str(slot["attempt_id"]) for slot in attempts),
        }
        if old_identifiers.intersection(identifiers) or new_identifiers.intersection(identifiers):
            raise IntegrityError("v4 identifier collides with a closed or new identifier")
        new_identifiers.update(identifiers)
        work_item = WorkItem(
            ordinal=ordinal,
            work_item_id=work_id,
            manifest_sha256=source.route_manifest_sha256,
            task_registry_sha256=task_registry_sha256(tasks),
            task=task,
            candidate=source.work_item.candidate,
            endpoint_execution_sha256=source.work_item.endpoint_execution_sha256,
            execution_policy_sha256=parent.bundle.execution_policy.sha256,
            execution_policy=parent.bundle.execution_policy,
        )
        forecast = derive_conditions_forecast(
            work_item,
            policy=parent.bundle.execution_policy,
            conditions=("epicure_off", "epicure_on"),
        )
        cells.append(
            {
                **basis,
                "cell_id": cell_id,
                "work_item_id": work_id,
                "run_id": run_id,
                "arm_ids": arm_ids,
                "attempt_slots": attempts,
                "attempt_slots_sha256": sha256_json(attempts),
                "reserved_worst_case_usd": _decimal_text(forecast.forecast_usd),
                "alternate_non_quarantined_task": True,
                "no_prior_model_task_exposure_at_freeze": True,
                "fresh_identifiers_disjoint_from_parent": True,
                "official_fit_eligible": False,
            }
        )
    recovery_budget = sum(
        (_decimal(cell["reserved_worst_case_usd"], field="cell forecast") for cell in cells[:7]),
        Decimal(0),
    )
    glm_budget = _decimal(cells[7]["reserved_worst_case_usd"], field="GLM forecast")
    parent_statuses = {
        str(cell.get("work_item_id")): str(cell.get("status"))
        for cell in parent.audit.get("cells") or []
        if isinstance(cell, Mapping)
    }
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "frozen_zero_provider_or_mcp_calls",
        "sources": {
            "parent_preflight_sha256": PARENT_PREFLIGHT_SHA256,
            "parent_materialization_sha256": PARENT_MATERIALIZATION_SHA256,
            "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
            "parent_closure_sha256": PARENT_CLOSURE_SHA256,
            "parent_reconstructed_audit_sha256": PARENT_AUDIT_SHA256,
            "task_validity_sha256": task_source["artifact_sha256"],
            "task_registry_sha256": task_registry_sha256(tasks),
            "quarantine_sha256": quarantine["artifact_sha256"],
            "exposure_snapshot_records_sha256": exposure["records_sha256"],
            "exposure_snapshot_record_count": exposure["record_count"],
        },
        "parent_disposition": {
            "complete_cell": {
                "work_item_id": PARENT_COMPLETE_WORK_ITEM,
                "status": parent_statuses[PARENT_COMPLETE_WORK_ITEM],
                "preserved": True,
            },
            "incomplete_glm_cell": {
                "work_item_id": PARENT_INCOMPLETE_GLM_WORK_ITEM,
                "status": parent_statuses[PARENT_INCOMPLETE_GLM_WORK_ITEM],
                "failures": [
                    failure
                    for failure in parent.audit.get("failures") or []
                    if str(failure).startswith(PARENT_INCOMPLETE_GLM_WORK_ITEM)
                ],
                "preserved": True,
                "replayed": False,
            },
            "seven_unstarted_parent_work_item_ids": [
                source_id for _, source_id, _ in CELL_SPECS[:7]
            ],
            "parent_identifiers_reopened": 0,
            "parent_reliability_observations_superseded": 0,
        },
        "task_quarantine": {
            "excluded_task_ids": sorted(QUARANTINED_TASK_IDS),
            "selected_quarantined_tasks": 0,
        },
        "model_task_exposure_snapshot": exposure,
        "execution_order": {
            "phase_1": {
                "name": RECOVERY_PHASE,
                "cells": 7,
                "first_six_are_non_glm": True,
                "seventh_is_previously_untouched_glm_composition": True,
                "cell_failures_are_isolated": True,
            },
            "barrier": "source_reconstructed_complete_disposition_audit_for_all_seven_cells",
            "phase_2": {
                "name": GLM_PHASE,
                "cells": 1,
                "purpose": "alternate-task replacement for incomplete parent GLM substitution cell",
                "separate_confirmation_and_ledger": True,
            },
        },
        "cells": cells,
        "counts": {
            "recovery_cells": 7,
            "glm_specific_cells": 1,
            "planned_real_arms": 16,
            "planned_synthetic_arms": 0,
            "provider_calls_by_freeze": 0,
            "epicure_calls_by_freeze": 0,
            "fresh_work_item_ids": 8,
        },
        "budget": {
            "currency": "USD",
            "recovery_phase_worst_case_usd": _decimal_text(recovery_budget),
            "glm_phase_worst_case_usd": _decimal_text(glm_budget),
            "total_worst_case_usd": _decimal_text(recovery_budget + glm_budget),
        },
        "execution_policy": parent.bundle.execution_policy.document(),
        "execution_policy_sha256": parent.bundle.execution_policy.sha256,
        "reasoning_effort_disclosure": {
            "intermediate": parent.bundle.execution_policy.intermediate_reasoning_effort,
            "final": parent.bundle.execution_policy.final_reasoning_effort,
        },
        "epicure": dict(parent.bundle.epicure),
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "official_preference_or_uplift_fit_eligible": False,
            "permitted_analysis": "coverage_and_reliability_diagnostics_only",
            "replacement_observations_are_not_missing_at_random": True,
            "human_quality_judgments": 0,
        },
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _phase_paths(output_root: Path, phase: str) -> RunPaths:
    name = "untouched-recovery" if phase == RECOVERY_PHASE else "glm-specific-replacement"
    root = output_root / name
    return RunPaths(root, root / "source", root / "responses", root / "ledger.jsonl")


def _ledger_head(path: Path) -> str | None:
    entries = load_dataset_ledger(path)
    return str(entries[-1]["entry_sha256"]) if entries else None


def _frontier_ledger_head(path: Path) -> str | None:
    entries = load_frontier_ledger(path)
    return str(entries[-1]["entry_sha256"]) if entries else None


def _credential_blockers(environment: Mapping[str, str]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    required = (
        "FLAVOURBENCH_OPENROUTER_API_KEY",
        "FLAVOURBENCH_COHERE_API_KEY",
        "FLAVOURBENCH_MCP_URL",
        "FLAVOURBENCH_MCP_TOKEN",
    )
    for variable in required:
        if not environment.get(variable):
            blockers.append({"gate": "credential_or_private_route_presence", "variable": variable})
    base = environment.get("FLAVOURBENCH_OPENROUTER_BASE_URL", "")
    if "gateway.ai.cloudflare.com" in base and not environment.get(
        "FLAVOURBENCH_CLOUDFLARE_AI_GATEWAY_TOKEN"
    ):
        blockers.append(
            {
                "gate": "cloudflare_gateway_token_presence",
                "variable": "FLAVOURBENCH_CLOUDFLARE_AI_GATEWAY_TOKEN",
            }
        )
    return blockers


def verify_reasoning_v4_terminal_orphan(
    *,
    project_root: Path,
    route_plan_path: Path,
    receipt_path: Path,
    audit_path: Path,
    closure_path: Path,
    ledger_path: Path,
    journal_path: Path,
    source_directory: Path,
) -> dict[str, Any]:
    """Verify the one exact reasoning-v4 pre-generation terminal orphan.

    The retained reservation remains budget exposure.  This verifier only
    changes its admission classification from active/unresolved to terminal
    no-replay after reconstructing the source audit and closure byte-for-byte.
    """

    from .reasoning_effort_route_gate_v4 import build_route_audit, verify_closure

    route_plan = _load_addressed(
        route_plan_path,
        label="reasoning-v4 route plan",
        expected_schema="flavourbench-reasoning-effort-route-gate-plan-v4",
        expected_digest=REASONING_V4_ROUTE_PLAN_SHA256,
    )
    receipt = _load_addressed(
        receipt_path,
        label="reasoning-v4 receipt",
        expected_schema="flavourbench-reasoning-effort-route-gate-execution-receipt-v1",
        expected_digest=REASONING_V4_RECEIPT_SHA256,
    )
    audit = _load_addressed(
        audit_path,
        label="reasoning-v4 audit",
        expected_schema="flavourbench-reasoning-effort-route-gate-audit-v1",
        expected_digest=REASONING_V4_AUDIT_SHA256,
    )
    closure = _load_addressed(
        closure_path,
        label="reasoning-v4 closure",
        expected_schema="flavourbench-reasoning-effort-route-gate-closure-v1",
        expected_digest=REASONING_V4_CLOSURE_SHA256,
    )
    rebuilt_audit_payload = build_route_audit(
        route_plan=route_plan,
        receipt=receipt,
        receipt_path=receipt_path,
        ledger_path=ledger_path,
        source_directory=source_directory,
        repo_root=project_root.parent,
    )
    rebuilt_audit = {
        **rebuilt_audit_payload,
        "artifact_sha256": sha256_json(rebuilt_audit_payload),
    }
    if rebuilt_audit != audit:
        raise IntegrityError("reasoning-v4 audit does not exactly source-reconstruct")
    if not verify_closure(closure, route_plan=route_plan, audit=audit):
        raise IntegrityError("reasoning-v4 closure integrity does not verify")

    ledger = load_dataset_ledger(ledger_path)
    if (
        _file_sha256(ledger_path) != REASONING_V4_LEDGER_SHA256
        or len(ledger) != 6
        or ledger[-1].get("entry_sha256") != REASONING_V4_GEMINI_INCIDENT_SHA256
    ):
        raise IntegrityError("reasoning-v4 terminal ledger differs")
    reservations = [
        item
        for item in ledger
        if item.get("event_type") == "reservation_created"
        and item.get("work_item_id") == REASONING_V4_GEMINI_WORK_ITEM_ID
    ]
    incidents = [
        item
        for item in ledger
        if item.get("event_type") == "execution_incident"
        and item.get("work_item_id") == REASONING_V4_GEMINI_WORK_ITEM_ID
    ]
    if len(reservations) != 1 or len(incidents) != 1:
        raise IntegrityError("reasoning-v4 Gemini lifecycle is not unique")
    reservation = reservations[0]
    incident = incidents[0]
    if (
        reservation.get("entry_sha256") != REASONING_V4_GEMINI_RESERVATION_SHA256
        or reservation.get("run_id") != REASONING_V4_GEMINI_RUN_ID
        or reservation.get("model_id") != "google/gemini-3.6-flash"
        or reservation.get("provider_endpoint") != "google-ai-studio/flex"
        or reservation.get("route_plan_sha256") != REASONING_V4_ROUTE_PLAN_SHA256
        or reservation.get("replay_permitted") is not False
        or _decimal(reservation.get("reserved_usd"), field="reasoning-v4 reserve")
        != REASONING_V4_GEMINI_RESERVE_USD
        or incident.get("entry_sha256") != REASONING_V4_GEMINI_INCIDENT_SHA256
        or incident.get("reservation_entry_sha256") != REASONING_V4_GEMINI_RESERVATION_SHA256
        or incident.get("incident") != "no_verified_source_or_uncertain_delivery_no_replay"
        or incident.get("error_type") != "RuntimeError"
        or incident.get("error_sha256") != REASONING_V4_GEMINI_ERROR_SHA256
        or incident.get("replay_permitted") is not False
    ):
        raise IntegrityError("reasoning-v4 Gemini reservation or incident differs")

    try:
        journal = load_run_journal(journal_path)
    except JournalIntegrityError as error:
        raise IntegrityError(
            "reasoning-v4 Gemini journal is not the exact pre-request stop"
        ) from error
    if (
        _file_sha256(journal_path) != REASONING_V4_JOURNAL_SHA256
        or journal_path.name
        != (f".flavourbench-live-smoke-journal-{REASONING_V4_GEMINI_RUN_ID}.inprogress.jsonl")
        or [item.get("event_type") for item in journal] != ["run_started", "openrouter_key_status"]
        or any(
            item.get("event_type")
            in {
                "provider_attempt",
                "request_started",
                "response_received",
                "mcp_session_started",
                "mcp_call_started",
                "mcp_call_completed",
            }
            for item in journal
        )
    ):
        raise IntegrityError("reasoning-v4 Gemini journal is not the exact pre-request stop")
    journal_start = journal[0].get("payload")
    if (
        journal[0].get("run_id") != REASONING_V4_GEMINI_RUN_ID
        or not isinstance(journal_start, Mapping)
        or journal_start.get("dataset_work_item_id") != REASONING_V4_GEMINI_WORK_ITEM_ID
        or journal_start.get("requested_model_id") != "google/gemini-3.6-flash"
        or journal_start.get("requested_provider") != "google-ai-studio/flex"
    ):
        raise IntegrityError("reasoning-v4 Gemini journal identity differs")
    if source_directory.is_symlink() or not source_directory.is_dir():
        raise IntegrityError("reasoning-v4 source directory is absent or non-regular")
    source_work_ids: set[str] = set()
    for path in sorted(source_directory.glob("*.json")):
        document = _load_json(path, label="reasoning-v4 source artifact")
        source_work_ids.add(str(document.get("dataset_work_item_id") or ""))
    if REASONING_V4_GEMINI_WORK_ITEM_ID in source_work_ids:
        raise IntegrityError("reasoning-v4 Gemini unexpectedly has a complete source")

    receipt_blockers = receipt.get("final_budget", {}).get("blockers") or []
    expected_blocker = {
        "gate": "active_reservation_without_source",
        "reservation_entry_sha256": REASONING_V4_GEMINI_RESERVATION_SHA256,
        "reserved_usd": _decimal_text(REASONING_V4_GEMINI_RESERVE_USD),
        "run": "reasoning_effort_route_gate_v4",
        "work_item_id": REASONING_V4_GEMINI_WORK_ITEM_ID,
    }
    incident_outcomes = [
        item
        for item in receipt.get("outcomes") or []
        if isinstance(item, Mapping)
        and item.get("work_item_id") == REASONING_V4_GEMINI_WORK_ITEM_ID
    ]
    closed = closure.get("closed_identifiers") or {}
    if (
        receipt_blockers != [expected_blocker]
        or receipt.get("status") != "failed_or_incomplete_closed"
        or receipt.get("failed_suffix_reopened") is not False
        or receipt.get("uncertain_delivery_replayed") is not False
        or receipt.get("final_budget", {}).get("route_gate_orphan_reservation_usd")
        != _decimal_text(REASONING_V4_GEMINI_RESERVE_USD)
        or incident_outcomes
        != [
            {
                "decision": "execution_incident_reservation_retained_no_replay",
                "incident_entry_sha256": REASONING_V4_GEMINI_INCIDENT_SHA256,
                "work_item_id": REASONING_V4_GEMINI_WORK_ITEM_ID,
            }
        ]
        or audit.get("execution_receipt", {}).get("artifact_sha256") != REASONING_V4_RECEIPT_SHA256
        or audit.get("decision") != "failed_one_or_more_predicates"
        or f"missing_source:{REASONING_V4_GEMINI_WORK_ITEM_ID}" not in (audit.get("failures") or [])
        or closure.get("route_gate_audit_sha256") != REASONING_V4_AUDIT_SHA256
        or closure.get("route_plan_sha256") != REASONING_V4_ROUTE_PLAN_SHA256
        or closure.get("execution_receipt", {}).get("artifact_sha256")
        != REASONING_V4_RECEIPT_SHA256
        or REASONING_V4_GEMINI_WORK_ITEM_ID not in (closed.get("work_item_ids") or [])
        or REASONING_V4_GEMINI_RUN_ID not in (closed.get("run_ids") or [])
        or closed.get("replay_permitted") is not False
        or closure.get("decision", {}).get("failed_or_unattempted_suffix_closed") is not True
    ):
        raise IntegrityError("reasoning-v4 receipt/audit/closure terminal binding differs")

    payload = {
        "schema_version": "flavourbench-reasoning-v4-terminal-orphan-resolution-v1",
        "classification": "verified_pre_request_terminal_no_replay_reservation",
        "work_item_id": REASONING_V4_GEMINI_WORK_ITEM_ID,
        "run_id": REASONING_V4_GEMINI_RUN_ID,
        "reservation_entry_sha256": REASONING_V4_GEMINI_RESERVATION_SHA256,
        "incident_entry_sha256": REASONING_V4_GEMINI_INCIDENT_SHA256,
        "error_sha256": REASONING_V4_GEMINI_ERROR_SHA256,
        "reserved_usd_retained_as_conservative_exposure": _decimal_text(
            REASONING_V4_GEMINI_RESERVE_USD
        ),
        "reservation_released": False,
        "terminal_no_replay": True,
        "request_boundary": {
            "provider_completion_request_events_for_orphan": 0,
            "provider_generation_ids_for_orphan": 0,
            "mcp_sessions_for_orphan": 0,
            "mcp_tool_calls_for_orphan": 0,
            "account_status_events": 1,
        },
        "evidence": {
            "route_plan": {
                "path": _relative(project_root, route_plan_path),
                "artifact_sha256": REASONING_V4_ROUTE_PLAN_SHA256,
                "physical_sha256": _file_sha256(route_plan_path),
            },
            "receipt": {
                "path": _relative(project_root, receipt_path),
                "artifact_sha256": REASONING_V4_RECEIPT_SHA256,
                "physical_sha256": _file_sha256(receipt_path),
            },
            "audit": {
                "path": _relative(project_root, audit_path),
                "artifact_sha256": REASONING_V4_AUDIT_SHA256,
                "physical_sha256": _file_sha256(audit_path),
            },
            "closure": {
                "path": _relative(project_root, closure_path),
                "artifact_sha256": REASONING_V4_CLOSURE_SHA256,
                "physical_sha256": _file_sha256(closure_path),
            },
            "ledger": {
                "path": _relative(project_root, ledger_path),
                "physical_sha256": REASONING_V4_LEDGER_SHA256,
                "head_entry_sha256": REASONING_V4_GEMINI_INCIDENT_SHA256,
            },
            "journal": {
                "path": _relative(project_root, journal_path),
                "physical_sha256": REASONING_V4_JOURNAL_SHA256,
                "head_entry_sha256": str(journal[-1]["entry_sha256"]),
            },
            "source_directory": _relative(project_root, source_directory),
        },
    }
    return {**payload, "verification_sha256": sha256_json(payload)}


def _terminal_orphan_blocker_matches(
    item: Mapping[str, Any], resolution: Mapping[str, Any]
) -> bool:
    return bool(
        resolution.get("terminal_no_replay") is True
        and resolution.get("classification")
        == "verified_pre_request_terminal_no_replay_reservation"
        and item.get("gate") == "active_reservation_without_source"
        and item.get("work_item_id") == REASONING_V4_GEMINI_WORK_ITEM_ID
        and item.get("reservation_entry_sha256") == REASONING_V4_GEMINI_RESERVATION_SHA256
        and _decimal(item.get("reserved_usd"), field="terminal orphan blocker")
        == REASONING_V4_GEMINI_RESERVE_USD
    )


def _verify_reasoning_v5_endpoint_metadata(
    *,
    project_root: Path,
    plan: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    endpoint_id: str,
    endpoint_root: Path,
    ledger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = {
        str(item.get("endpoint_id") or ""): item
        for item in snapshot.get("records") or []
        if isinstance(item, Mapping)
    }
    snapshot_record = records.get(endpoint_id)
    if not isinstance(snapshot_record, Mapping):
        raise IntegrityError(f"reasoning-v5 {endpoint_id} snapshot record is absent")
    items = {
        str(item.get("work_item_id") or ""): item
        for item in plan.get("work_items") or []
        if isinstance(item, Mapping) and item.get("endpoint_id") == endpoint_id
    }
    reservations = [item for item in ledger if item.get("event_type") == "reservation_created"]
    expected_reservation_count = 1 if endpoint_id == "sonnet" else 2
    if len(reservations) != expected_reservation_count:
        raise IntegrityError(f"reasoning-v5 {endpoint_id} reservation count differs")
    attestation_directory = endpoint_root / "endpoint-attestations"
    if attestation_directory.is_symlink() or not attestation_directory.is_dir():
        raise IntegrityError(f"reasoning-v5 {endpoint_id} attestation directory differs")
    attestation_paths = sorted(attestation_directory.glob("*.json"))
    if len(attestation_paths) != expected_reservation_count:
        raise IntegrityError(f"reasoning-v5 {endpoint_id} attestation inventory differs")
    by_digest = {
        path.name.rsplit("-", 1)[-1].removesuffix(".json"): path for path in attestation_paths
    }
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reservation in reservations:
        work_item_id = str(reservation.get("work_item_id") or "")
        item = items.get(work_item_id)
        if not isinstance(item, Mapping):
            raise IntegrityError(f"reasoning-v5 {endpoint_id} reservation is not planned")
        coordinate = item.get("route_coordinate")
        if not isinstance(coordinate, Mapping):
            raise IntegrityError(f"reasoning-v5 {endpoint_id} route coordinate is absent")
        attestation_sha256 = str(reservation.get("endpoint_attestation_sha256") or "")
        attestation_path = by_digest.get(attestation_sha256)
        if attestation_path is None:
            raise IntegrityError(f"reasoning-v5 {endpoint_id} attestation binding is absent")
        expected_name = (
            f"{endpoint_id}-pre-admission-{coordinate['variant_id']}-{attestation_sha256}.json"
        )
        if attestation_path.name != expected_name:
            raise IntegrityError(f"reasoning-v5 {endpoint_id} attestation filename differs")
        attestation = _load_addressed(
            attestation_path,
            label=f"reasoning-v5 {endpoint_id} endpoint attestation",
            expected_schema="flavourbench-reasoning-effort-endpoint-admission-attestation-v5",
            expected_digest=attestation_sha256,
        )
        expected_reservation = {
            "work_item_id": work_item_id,
            "route_cell_id": item.get("route_cell_id"),
            "run_id": item.get("run_id"),
            "arm_ids": item.get("arm_ids"),
            "model_id": coordinate.get("model_id"),
            "canonical_model_slug": coordinate.get("canonical_model_slug"),
            "provider_endpoint": coordinate.get("provider_endpoint"),
            "actual_provider_name": coordinate.get("actual_provider_name"),
            "variant_id": coordinate.get("variant_id"),
            "intermediate_reasoning_effort": coordinate.get("intermediate_reasoning_effort"),
            "final_reasoning_effort": coordinate.get("final_reasoning_effort"),
            "endpoint_snapshot_sha256": REASONING_V5_ENDPOINT_SNAPSHOT_SHA256,
            "raw_endpoint_execution_contract_sha256": coordinate.get(
                "snapshot_raw_execution_contract_sha256"
            ),
            "semantic_endpoint_execution_contract_sha256": coordinate.get(
                "semantic_execution_contract_sha256"
            ),
            "route_plan_sha256": REASONING_V5_ROUTE_PLAN_SHA256,
            "reserved_usd": item.get("worst_case_reserve_usd"),
            "replay_permitted": False,
            "quality_observations": 0,
            "rank_eligible": False,
        }
        if any(reservation.get(key) != value for key, value in expected_reservation.items()):
            raise IntegrityError(f"reasoning-v5 {endpoint_id} reservation metadata differs")
        if (
            attestation.get("record_role") != "pre_reservation_zero_generation_endpoint_attestation"
            or attestation.get("endpoint_id") != endpoint_id
            or attestation.get("model") != snapshot_record.get("model")
            or attestation.get("raw_execution_contract")
            != snapshot_record.get("raw_execution_contract")
            or attestation.get("raw_execution_contract_sha256")
            != snapshot_record.get("raw_execution_contract_sha256")
            or attestation.get("semantic_execution_contract")
            != snapshot_record.get("semantic_execution_contract")
            or attestation.get("semantic_execution_contract_sha256")
            != snapshot_record.get("semantic_execution_contract_sha256")
            or attestation.get("counts")
            != {
                "catalog_http_gets": 2,
                "provider_completion_requests": 0,
                "epicure_calls": 0,
            }
        ):
            raise IntegrityError(f"reasoning-v5 {endpoint_id} endpoint metadata differs")
        seen.add(attestation_sha256)
        evidence.append(
            {
                "path": _relative(project_root, attestation_path),
                "artifact_sha256": attestation_sha256,
                "physical_sha256": _file_sha256(attestation_path),
                "work_item_id": work_item_id,
                "variant_id": coordinate.get("variant_id"),
                "raw_execution_contract_sha256": attestation.get("raw_execution_contract_sha256"),
                "semantic_execution_contract_sha256": attestation.get(
                    "semantic_execution_contract_sha256"
                ),
            }
        )
    if seen != set(by_digest):
        raise IntegrityError(f"reasoning-v5 {endpoint_id} has an unbound attestation")
    return sorted(evidence, key=lambda item: str(item["work_item_id"]))


def verify_reasoning_v5_terminal_endpoints(
    *,
    project_root: Path,
    route_plan_path: Path,
    endpoint_snapshot_path: Path,
    sonnet_root: Path,
    sonnet_receipt_path: Path,
    sonnet_audit_path: Path,
    sonnet_closure_path: Path,
    gemini_root: Path,
    gemini_receipt_path: Path,
    gemini_audit_path: Path,
    gemini_closure_path: Path,
    aggregate_audit_path: Path,
    aggregate_closure_path: Path,
) -> dict[str, Any]:
    """Reconstruct and bind the exact terminal reasoning-v5 endpoint runs."""

    from .reasoning_effort_route_gate_v5 import (
        build_aggregate_audit,
        build_aggregate_closure,
        build_endpoint_audit,
        build_endpoint_closure,
        validate_route_plan,
    )

    route_plan = _load_addressed(
        route_plan_path,
        label="reasoning-v5 route plan",
        expected_schema="flavourbench-reasoning-effort-route-gate-plan-v5",
        expected_digest=REASONING_V5_ROUTE_PLAN_SHA256,
    )
    endpoint_snapshot = _load_addressed(
        endpoint_snapshot_path,
        label="reasoning-v5 endpoint snapshot",
        expected_schema="flavourbench-openrouter-endpoint-snapshot-v5",
        expected_digest=REASONING_V5_ENDPOINT_SNAPSHOT_SHA256,
    )
    try:
        validate_route_plan(route_plan, repo_root=project_root.parent)
    except RuntimeError as error:
        raise IntegrityError("reasoning-v5 route plan no longer validates") from error

    endpoint_configs = {
        "sonnet": {
            "root": sonnet_root,
            "receipt_path": sonnet_receipt_path,
            "audit_path": sonnet_audit_path,
            "closure_path": sonnet_closure_path,
            "receipt_sha256": REASONING_V5_SONNET_RECEIPT_SHA256,
            "audit_sha256": REASONING_V5_SONNET_AUDIT_SHA256,
            "closure_sha256": REASONING_V5_SONNET_CLOSURE_SHA256,
            "ledger_sha256": REASONING_V5_SONNET_LEDGER_SHA256,
            "ledger_head_sha256": REASONING_V5_SONNET_LEDGER_HEAD_SHA256,
            "source_exposure_usd": REASONING_V5_SONNET_SOURCE_EXPOSURE_USD,
            "audit_cost_usd": REASONING_V5_SONNET_AUDIT_COST_USD,
            "source_count": 1,
            "receipt_status": "failed_or_incomplete_closed",
            "audit_decision": "failed_one_or_more_predicates",
            "endpoint_qualified": False,
        },
        "gemini": {
            "root": gemini_root,
            "receipt_path": gemini_receipt_path,
            "audit_path": gemini_audit_path,
            "closure_path": gemini_closure_path,
            "receipt_sha256": REASONING_V5_GEMINI_RECEIPT_SHA256,
            "audit_sha256": REASONING_V5_GEMINI_AUDIT_SHA256,
            "closure_sha256": REASONING_V5_GEMINI_CLOSURE_SHA256,
            "ledger_sha256": REASONING_V5_GEMINI_LEDGER_SHA256,
            "ledger_head_sha256": REASONING_V5_GEMINI_LEDGER_HEAD_SHA256,
            "source_exposure_usd": REASONING_V5_GEMINI_SOURCE_EXPOSURE_USD,
            "audit_cost_usd": REASONING_V5_GEMINI_SOURCE_EXPOSURE_USD,
            "source_count": 2,
            "receipt_status": "two_pair_sources_available",
            "audit_decision": "passed_all_predicates",
            "endpoint_qualified": True,
        },
    }
    endpoint_results: dict[str, Any] = {}
    endpoint_documents: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for endpoint_id, config in endpoint_configs.items():
        root = Path(config["root"])
        if root.is_symlink() or not root.is_dir():
            raise IntegrityError(f"reasoning-v5 {endpoint_id} root differs")
        receipt = _load_addressed(
            Path(config["receipt_path"]),
            label=f"reasoning-v5 {endpoint_id} receipt",
            expected_schema="flavourbench-reasoning-effort-endpoint-receipt-v5",
            expected_digest=str(config["receipt_sha256"]),
        )
        audit = _load_addressed(
            Path(config["audit_path"]),
            label=f"reasoning-v5 {endpoint_id} audit",
            expected_schema="flavourbench-reasoning-effort-endpoint-audit-v5",
            expected_digest=str(config["audit_sha256"]),
        )
        closure = _load_addressed(
            Path(config["closure_path"]),
            label=f"reasoning-v5 {endpoint_id} closure",
            expected_schema="flavourbench-reasoning-effort-endpoint-closure-v5",
            expected_digest=str(config["closure_sha256"]),
        )
        rebuilt_audit_payload = build_endpoint_audit(
            plan=route_plan,
            endpoint_id=endpoint_id,
            receipt_path=Path(config["receipt_path"]),
            endpoint_root=root,
            repo_root=project_root.parent,
        )
        rebuilt_audit = {
            **rebuilt_audit_payload,
            "artifact_sha256": sha256_json(rebuilt_audit_payload),
        }
        if rebuilt_audit != audit:
            raise IntegrityError(f"reasoning-v5 {endpoint_id} audit does not reconstruct")
        rebuilt_closure_payload = build_endpoint_closure(
            plan=route_plan, endpoint_id=endpoint_id, audit=audit
        )
        rebuilt_closure = {
            **rebuilt_closure_payload,
            "artifact_sha256": sha256_json(rebuilt_closure_payload),
        }
        if rebuilt_closure != closure:
            raise IntegrityError(f"reasoning-v5 {endpoint_id} closure does not reconstruct")

        ledger_path = root / "ledger.jsonl"
        ledger = load_dataset_ledger(ledger_path)
        reservations, finalizations = dataset_ledger_state(ledger)
        planned_ids = {
            str(item["work_item_id"])
            for item in route_plan["work_items"]
            if item["endpoint_id"] == endpoint_id
        }
        if (
            _file_sha256(ledger_path) != config["ledger_sha256"]
            or _ledger_head(ledger_path) != config["ledger_head_sha256"]
            or set(reservations) != set(finalizations)
            or len(reservations) != int(config["source_count"])
            or not set(reservations) <= planned_ids
            or any(item.get("event_type") == "execution_incident" for item in ledger)
            or set(closure.get("closed_identifiers", {}).get("work_item_ids") or []) != planned_ids
            or closure.get("closed_identifiers", {}).get("replay_permitted") is not False
            or closure.get("decision", {}).get("endpoint_identifiers_permanently_closed")
            is not True
            or closure.get("decision", {}).get("endpoint_qualified")
            is not config["endpoint_qualified"]
            or receipt.get("status") != config["receipt_status"]
            or audit.get("decision") != config["audit_decision"]
        ):
            raise IntegrityError(f"reasoning-v5 {endpoint_id} terminal lifecycle differs")

        accounting = _run_accounting(
            SupplementalRun(root / "source", ledger_path),
            label=f"reasoning_v5_terminal_{endpoint_id}",
        )
        source_exposure = Decimal(config["source_exposure_usd"])
        audit_cost = Decimal(config["audit_cost_usd"])
        if (
            accounting.actual_cost_usd != source_exposure
            or accounting.exposure_usd != source_exposure
            or accounting.orphan_reservation_usd != 0
            or accounting.blockers
            or accounting.source_count != int(config["source_count"])
            or _decimal(audit.get("accounting", {}).get("actual_cost_usd"), field="v5 audit cost")
            != audit_cost
        ):
            raise IntegrityError(f"reasoning-v5 {endpoint_id} source accounting differs")
        metadata = _verify_reasoning_v5_endpoint_metadata(
            project_root=project_root,
            plan=route_plan,
            snapshot=endpoint_snapshot,
            endpoint_id=endpoint_id,
            endpoint_root=root,
            ledger=ledger,
        )
        endpoint_results[endpoint_id] = {
            "terminal_classification": (
                "passed_both_pairs_closed_no_replay"
                if endpoint_id == "gemini"
                else "failed_first_pair_unattempted_suffix_closed_no_replay"
            ),
            "root": _relative(project_root, root),
            "ledger": {
                "path": _relative(project_root, ledger_path),
                "physical_sha256": config["ledger_sha256"],
                "head_entry_sha256": config["ledger_head_sha256"],
            },
            "receipt": {
                "path": _relative(project_root, Path(config["receipt_path"])),
                "artifact_sha256": config["receipt_sha256"],
                "physical_sha256": _file_sha256(Path(config["receipt_path"])),
            },
            "audit": {
                "path": _relative(project_root, Path(config["audit_path"])),
                "artifact_sha256": config["audit_sha256"],
                "physical_sha256": _file_sha256(Path(config["audit_path"])),
            },
            "closure": {
                "path": _relative(project_root, Path(config["closure_path"])),
                "artifact_sha256": config["closure_sha256"],
                "physical_sha256": _file_sha256(Path(config["closure_path"])),
            },
            "planned_work_items": len(planned_ids),
            "reserved_and_finalized_work_items": len(reservations),
            "source_conservative_exposure_usd": _decimal_text(source_exposure),
            "endpoint_audit_cost_scope_usd": _decimal_text(audit_cost),
            "reconciled_cost_outside_endpoint_audit_scope_usd": _decimal_text(
                source_exposure - audit_cost
            ),
            "endpoint_attestations": metadata,
        }
        endpoint_documents[endpoint_id] = (audit, closure)

    aggregate_audit = _load_addressed(
        aggregate_audit_path,
        label="reasoning-v5 aggregate audit",
        expected_schema="flavourbench-reasoning-effort-route-gate-audit-v5",
        expected_digest=REASONING_V5_AGGREGATE_AUDIT_SHA256,
    )
    aggregate_closure = _load_addressed(
        aggregate_closure_path,
        label="reasoning-v5 aggregate closure",
        expected_schema="flavourbench-reasoning-effort-route-gate-closure-v5",
        expected_digest=REASONING_V5_AGGREGATE_CLOSURE_SHA256,
    )
    rebuilt_aggregate_payload = build_aggregate_audit(
        plan=route_plan,
        gemini_audit_path=gemini_audit_path,
        gemini_closure_path=gemini_closure_path,
        sonnet_audit_path=sonnet_audit_path,
        sonnet_closure_path=sonnet_closure_path,
        repo_root=project_root.parent,
    )
    rebuilt_aggregate = {
        **rebuilt_aggregate_payload,
        "artifact_sha256": sha256_json(rebuilt_aggregate_payload),
    }
    if rebuilt_aggregate != aggregate_audit:
        raise IntegrityError("reasoning-v5 aggregate audit does not reconstruct")
    rebuilt_aggregate_closure_payload = build_aggregate_closure(
        plan=route_plan, aggregate_audit=aggregate_audit
    )
    rebuilt_aggregate_closure = {
        **rebuilt_aggregate_closure_payload,
        "artifact_sha256": sha256_json(rebuilt_aggregate_closure_payload),
    }
    if rebuilt_aggregate_closure != aggregate_closure:
        raise IntegrityError("reasoning-v5 aggregate closure does not reconstruct")
    aggregate_narrow_cost = _decimal(
        aggregate_audit.get("accounting", {}).get("fresh_endpoint_actual_cost_usd"),
        field="reasoning-v5 aggregate cost",
    )
    if (
        aggregate_audit.get("decision") != "failed_one_or_more_predicates"
        or aggregate_audit.get("failures") != ["sonnet_did_not_pass"]
        or aggregate_closure.get("decision")
        != {
            "route_gate_qualified": False,
            "full_study_zero_call_preflight_permitted": False,
            "all_old_and_new_route_identifiers_closed": True,
            "replay_permitted": False,
        }
        or aggregate_narrow_cost != Decimal("0.099978")
    ):
        raise IntegrityError("reasoning-v5 aggregate terminal decision differs")

    payload = {
        "schema_version": "flavourbench-reasoning-v5-terminal-resolution-v1",
        "classification": "verified_endpoint_scoped_terminal_no_replay",
        "route_gate_qualified": False,
        "full_sensitivity_study_admitted": False,
        "coverage_recovery_blocked": False,
        "all_v5_identifiers_closed": True,
        "replay_permitted": False,
        "route_plan": {
            "path": _relative(project_root, route_plan_path),
            "artifact_sha256": REASONING_V5_ROUTE_PLAN_SHA256,
            "physical_sha256": _file_sha256(route_plan_path),
        },
        "endpoint_snapshot": {
            "path": _relative(project_root, endpoint_snapshot_path),
            "artifact_sha256": REASONING_V5_ENDPOINT_SNAPSHOT_SHA256,
            "physical_sha256": _file_sha256(endpoint_snapshot_path),
        },
        "endpoints": endpoint_results,
        "aggregate": {
            "audit": {
                "path": _relative(project_root, aggregate_audit_path),
                "artifact_sha256": REASONING_V5_AGGREGATE_AUDIT_SHA256,
                "physical_sha256": _file_sha256(aggregate_audit_path),
            },
            "closure": {
                "path": _relative(project_root, aggregate_closure_path),
                "artifact_sha256": REASONING_V5_AGGREGATE_CLOSURE_SHA256,
                "physical_sha256": _file_sha256(aggregate_closure_path),
            },
            "decision": "failed_sonnet_all_identifiers_closed",
        },
        "accounting": {
            "source_conservative_exposure_usd": _decimal_text(
                REASONING_V5_TOTAL_SOURCE_EXPOSURE_USD
            ),
            "aggregate_quality_audit_cost_scope_usd": _decimal_text(aggregate_narrow_cost),
            "reconciled_failed_arm_cost_outside_aggregate_scope_usd": _decimal_text(
                REASONING_V5_TOTAL_SOURCE_EXPOSURE_USD - aggregate_narrow_cost
            ),
            "orphan_reservation_usd": "0",
            "budget_basis": "complete_source_generation_accounting_not_narrow_quality_audit",
        },
        "endpoint_metadata": {
            "attestations": sum(
                len(item["endpoint_attestations"]) for item in endpoint_results.values()
            ),
            "catalog_http_gets": 6,
            "provider_completion_requests_by_attestations": 0,
            "epicure_calls_by_attestations": 0,
        },
    }
    return {**payload, "verification_sha256": sha256_json(payload)}


def _supplemental_accounting(
    *,
    roots: Sequence[Path],
    seen: set[str],
    terminal_orphan_resolution: Mapping[str, Any],
    terminal_orphan_root: Path,
) -> tuple[Decimal, Decimal, list[dict[str, Any]], list[dict[str, Any]], int]:
    exposure = Decimal(0)
    actual = Decimal(0)
    blockers: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    resolved_terminal_blockers = 0
    terminal_root_seen = 0
    for index, root in enumerate(roots, start=1):
        accounting = _run_accounting(
            SupplementalRun(root / "source", root / "ledger.jsonl"),
            label=f"v4_supplemental_{index}",
        )
        if seen.intersection(accounting.artifact_sha256s):
            raise IntegrityError("supplemental accounting duplicates a prior budget input")
        seen.update(accounting.artifact_sha256s)
        exposure += accounting.exposure_usd + accounting.orphan_reservation_usd
        actual += accounting.actual_cost_usd
        is_terminal_root = root.resolve() == terminal_orphan_root.resolve()
        terminal_root_seen += int(is_terminal_root)
        for item in accounting.blockers:
            if (
                item.get("work_item_id")
                == "63cf4b5c57e627ae17d150c6d0a37d30b7f59bee1c1f9a301a6c48c30b700a79"
                and _decimal(item.get("reserved_usd"), field="closed zero reserve") == 0
            ):
                continue
            if is_terminal_root and _terminal_orphan_blocker_matches(
                item, terminal_orphan_resolution
            ):
                resolved_terminal_blockers += 1
                continue
            blockers.append(dict(item))
        bindings.append(
            {
                "root": str(root),
                "ledger_head_sha256": _ledger_head(root / "ledger.jsonl"),
                "exposure_usd": _decimal_text(
                    accounting.exposure_usd + accounting.orphan_reservation_usd
                ),
                "actual_cost_usd": _decimal_text(accounting.actual_cost_usd),
            }
        )
    if terminal_root_seen != 1 or resolved_terminal_blockers != 1:
        raise IntegrityError(
            "reasoning-v4 terminal orphan root/blocker was not verified exactly once"
        )
    return exposure, actual, blockers, bindings, resolved_terminal_blockers


def _regular_file_inventory(root: Path, *, label: str) -> list[dict[str, Any]]:
    """Return a deterministic physical inventory and reject filesystem aliases."""

    if root.is_symlink() or not root.is_dir():
        raise IntegrityError(f"{label} must be a regular directory")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrityError(f"{label} contains a symlink: {path.relative_to(root)}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise IntegrityError(f"{label} contains a non-regular entry")
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "physical_sha256": _file_sha256(path),
            }
        )
    return records


def _failed_pre_reservation_supersession_path(project_root: Path) -> Path:
    return (
        project_root
        / "artifacts/season1/current-quality-run/historical-source-migrations"
        / (
            "historical-pre-reservation-supersession-"
            f"{FAILED_PRE_RESERVATION_SUPERSESSION_SHA256}.json"
        )
    )


def _load_supersession_artifact(
    *,
    project_root: Path,
    record: object,
    label: str,
    expected_schema: str,
) -> tuple[Path, dict[str, Any]]:
    path = _bound_path(project_root, record, label=label)
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise IntegrityError(f"{label} escapes the project root") from error
    if not isinstance(record, Mapping):
        raise IntegrityError(f"{label} binding is absent")
    document = _load_addressed(
        path,
        label=label,
        expected_schema=expected_schema,
        expected_digest=str(record.get("artifact_sha256") or ""),
    )
    return path, document


def _verify_failed_pre_reservation_supersession(
    *,
    project_root: Path,
    output_root: Path,
    plan: Mapping[str, Any],
    supersession_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the failed zero-call point and the exact later closed execution.

    The first attempt happened before any reservation append or subprocess.
    Its empty-root predicate is historical, not a requirement that later valid
    execution disappear.  This verifier accepts that temporal supersession only
    when every later file is the frozen, permanently closed v4 execution.
    """

    supersession = _load_addressed(
        supersession_path or _failed_pre_reservation_supersession_path(project_root),
        label="failed pre-reservation supersession",
        expected_schema=FAILED_PRE_RESERVATION_SUPERSESSION_SCHEMA,
        expected_digest=FAILED_PRE_RESERVATION_SUPERSESSION_SHA256,
    )
    historical = supersession.get("historical_failed_attempt")
    later = supersession.get("later_closed_execution")
    interpretation = supersession.get("interpretation")
    if not isinstance(historical, Mapping) or not isinstance(later, Mapping):
        raise IntegrityError("failed pre-reservation supersession is incomplete")
    if (
        supersession.get("historical_plan_sha256") != HISTORICAL_EXPOSURE_PLAN_SHA256
        or plan.get("artifact_sha256") != HISTORICAL_EXPOSURE_PLAN_SHA256
        or historical.get("phase") != RECOVERY_PHASE
        or historical.get("historical_phase_root_inventory") != ["ledger.jsonl.lock"]
        or historical.get("reservations_appended") != 0
        or historical.get("provider_completions") != 0
        or historical.get("epicure_calls") != 0
        or later.get("execution_preflight_sha256") != V4_EXECUTION_PREFLIGHT_SHA256
        or later.get("all_identifiers_permanently_closed") is not True
        or later.get("same_identifier_replay_permitted") is not False
        or later.get("unknown_or_unclosed_later_artifacts_permitted") is not False
        or interpretation
        != {
            "historical_zero_call_fact_preserved": True,
            "later_closed_files_do_not_retroactively_change_the_historical_snapshot": True,
            "later_execution_does_not_convert_the_failed_attempt_into_a_provider_call": True,
            "prospective_execution_authorized": False,
        }
        or supersession.get("provider_or_epicure_calls_made_by_supersession")
        != {"provider_completions": 0, "epicure": 0}
        or supersession.get("official") is not False
        or supersession.get("rank_eligible") is not False
    ):
        raise IntegrityError("failed pre-reservation supersession policy differs")

    _, failed_preflight = _load_supersession_artifact(
        project_root=project_root,
        record=historical.get("failed_zero_call_preflight"),
        label="failed zero-call preflight",
        expected_schema=PREFLIGHT_SCHEMA_VERSION,
    )
    _, incident_carrier = _load_supersession_artifact(
        project_root=project_root,
        record=historical.get("incident_carrier_preflight"),
        label="failed-attempt incident carrier",
        expected_schema=PREFLIGHT_SCHEMA_VERSION,
    )
    incident = incident_carrier.get("exact_inputs", {}).get(
        "failed_pre_reservation_attempt"
    )
    if not isinstance(incident, Mapping):
        raise IntegrityError("failed pre-reservation incident record is absent")
    failed_binding = historical.get("failed_zero_call_preflight")
    incident_failed_binding = incident.get("failed_preflight")
    if not isinstance(failed_binding, Mapping) or not isinstance(
        incident_failed_binding, Mapping
    ):
        raise IntegrityError("failed pre-reservation binding is absent")
    if (
        failed_preflight.get("artifact_sha256") != FAILED_PRE_RESERVATION_PREFLIGHT_SHA256
        or failed_preflight.get("status") != "admissible_zero_call_preflight"
        or failed_preflight.get("calls") != {"provider": 0, "epicure": 0}
        or failed_preflight.get("plan", {}).get("sha256")
        != HISTORICAL_EXPOSURE_PLAN_SHA256
        or incident_carrier.get("artifact_sha256") != V4_EXECUTION_PREFLIGHT_SHA256
        or incident_carrier.get("status") != "admissible_zero_call_preflight"
        or incident_carrier.get("calls") != {"provider": 0, "epicure": 0}
        or incident_carrier.get("plan", {}).get("sha256")
        != HISTORICAL_EXPOSURE_PLAN_SHA256
        or sha256_json(incident) != historical.get("incident_record_sha256")
        or any(
            incident_failed_binding.get(field) != failed_binding.get(field)
            for field in ("path", "artifact_sha256", "physical_sha256")
        )
        or incident_failed_binding.get("source_bundle_sha256")
        != failed_preflight.get("source_code", {}).get("bundle_sha256")
        or incident.get("classification")
        != "failed_before_reservation_append_and_subprocess_boundary"
        or incident.get("phase") != RECOVERY_PHASE
        or incident.get("filesystem")
        != {
            "inventory": ["ledger.jsonl.lock"],
            "ledger_exists": False,
            "source_directory_exists": False,
            "response_directory_exists": False,
            "run_journals": 0,
            "phase_receipts": 0,
            "phase_closures": 0,
            "phase_audits": 0,
        }
        or incident.get("calls") != {"provider": 0, "epicure": 0}
        or incident.get("identifier_disposition")
        != {
            "planned_cells": 7,
            "reservations_appended": 0,
            "work_item_ids_started": [],
            "run_ids_started": [],
            "attempt_ids_started": [],
            "fresh_identifiers_may_be_preserved": True,
        }
        or incident.get("observed_exception_type") != "IntegrityError"
        or incident.get("observed_exception")
        != "dataset ledger event overrides protected hash-chain fields"
        or incident.get("observed_exception_sha256")
        != _safe_process_hash("dataset ledger event overrides protected hash-chain fields")
    ):
        raise IntegrityError("failed pre-reservation zero-call evidence differs")

    output_inventory = _regular_file_inventory(output_root, label="closed v4 output root")
    expected_output_inventory = later.get("output_root_inventory")
    if not isinstance(expected_output_inventory, Mapping) or (
        len(output_inventory) != expected_output_inventory.get("record_count")
        or sha256_json(output_inventory) != expected_output_inventory.get("records_sha256")
    ):
        raise IntegrityError("closed v4 output inventory differs")

    phase_bindings = later.get("phases")
    if not isinstance(phase_bindings, Mapping) or set(phase_bindings) != set(PHASES):
        raise IntegrityError("closed v4 phase bindings differ")
    for phase, expected_cells in ((RECOVERY_PHASE, 7), (GLM_PHASE, 1)):
        binding = phase_bindings.get(phase)
        if not isinstance(binding, Mapping):
            raise IntegrityError(f"closed v4 {phase} binding is absent")
        phase_root = project_root / str(binding.get("phase_root") or "")
        if phase_root.resolve() != _phase_paths(output_root, phase).root.resolve():
            raise IntegrityError(f"closed v4 {phase} root binding differs")
        phase_inventory = _regular_file_inventory(
            phase_root, label=f"closed v4 {phase} root"
        )
        expected_inventory = binding.get("phase_root_inventory")
        if not isinstance(expected_inventory, Mapping) or (
            len(phase_inventory) != expected_inventory.get("record_count")
            or sha256_json(phase_inventory) != expected_inventory.get("records_sha256")
        ):
            raise IntegrityError(f"closed v4 {phase} inventory differs")

        artifacts: dict[str, tuple[Path, dict[str, Any]]] = {}
        for role, schema in (
            ("receipt", RECEIPT_SCHEMA_VERSION),
            ("closure", CLOSURE_SCHEMA_VERSION),
            ("audit", AUDIT_SCHEMA_VERSION),
        ):
            artifacts[role] = _load_supersession_artifact(
                project_root=project_root,
                record=binding.get(role),
                label=f"closed v4 {phase} {role}",
                expected_schema=schema,
            )
            unique = _find_phase_artifact(output_root, phase, role)
            if unique is None or unique.resolve() != artifacts[role][0].resolve():
                raise IntegrityError(f"closed v4 {phase} {role} is not unique")
        receipt = artifacts["receipt"][1]
        closure = artifacts["closure"][1]
        audit = artifacts["audit"][1]
        if (
            binding.get("planned_cells") != expected_cells
            or receipt.get("phase") != phase
            or closure.get("phase") != phase
            or audit.get("phase") != phase
            or receipt.get("preflight_sha256") != V4_EXECUTION_PREFLIGHT_SHA256
            or closure.get("preflight_sha256") != V4_EXECUTION_PREFLIGHT_SHA256
            or audit.get("preflight_sha256") != V4_EXECUTION_PREFLIGHT_SHA256
            or receipt.get("plan_sha256") != HISTORICAL_EXPOSURE_PLAN_SHA256
            or closure.get("plan_sha256") != HISTORICAL_EXPOSURE_PLAN_SHA256
            or audit.get("plan_sha256") != HISTORICAL_EXPOSURE_PLAN_SHA256
            or closure.get("receipt_sha256") != receipt.get("artifact_sha256")
            or audit.get("receipt_sha256") != receipt.get("artifact_sha256")
            or audit.get("closure_sha256") != closure.get("artifact_sha256")
            or receipt.get("status")
            != "all_phase_cells_received_terminal_disposition"
            or len(receipt.get("outcomes") or []) != expected_cells
            or closure.get("status") != "permanently_closed_all_phase_identifiers"
            or closure.get("all_cells_received_terminal_disposition") is not True
            or closure.get("safe_to_replay_any_parent_or_v4_work") is not False
            or closure.get("future_execution_with_same_phase_permitted") is not False
            or len(closure.get("closed_work_item_ids") or []) != expected_cells
            or closure.get("ledger_head_sha256") != binding.get("ledger_head_sha256")
            or _ledger_head(phase_root / "ledger.jsonl")
            != binding.get("ledger_head_sha256")
            or audit.get("decision") != "passed_complete_phase_disposition"
            or audit.get("integrity_failures") != []
            or audit.get("counts", {}).get("planned_cells") != expected_cells
            or audit.get("counts", {}).get("terminally_dispositioned_cells")
            != expected_cells
            or audit.get("counts", {}).get("synthetic_arms") != 0
            or len(audit.get("cells") or []) != expected_cells
            or audit.get("claim_boundary", {}).get("official") is not False
            or audit.get("claim_boundary", {}).get("rank_eligible") is not False
        ):
            raise IntegrityError(f"closed v4 {phase} terminal chain differs")
    return copy.deepcopy(dict(incident))


def build_preflight(
    *,
    project_root: Path,
    plan_path: Path,
    parent_root: Path,
    quarantine_path: Path,
    exposure_root: Path,
    output_root: Path,
    budget_audit_path: Path,
    supplemental_roots: Sequence[Path],
    global_ledger_path: Path,
    global_artifact_directory: Path,
    global_corrections_directory: Path | None,
    global_reconciliation_directory: Path | None,
    reasoning_v4_route_plan_path: Path,
    reasoning_v4_receipt_path: Path,
    reasoning_v4_audit_path: Path,
    reasoning_v4_closure_path: Path,
    reasoning_v4_ledger_path: Path,
    reasoning_v4_journal_path: Path,
    reasoning_v4_source_directory: Path,
    reasoning_v5_route_plan_path: Path,
    reasoning_v5_endpoint_snapshot_path: Path,
    reasoning_v5_sonnet_root: Path,
    reasoning_v5_sonnet_receipt_path: Path,
    reasoning_v5_sonnet_audit_path: Path,
    reasoning_v5_sonnet_closure_path: Path,
    reasoning_v5_gemini_root: Path,
    reasoning_v5_gemini_receipt_path: Path,
    reasoning_v5_gemini_audit_path: Path,
    reasoning_v5_gemini_closure_path: Path,
    reasoning_v5_aggregate_audit_path: Path,
    reasoning_v5_aggregate_closure_path: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    plan = _load_addressed(plan_path, label="v4 recovery plan", expected_schema=PLAN_SCHEMA_VERSION)
    parent = reconstruct_parent(project_root=project_root, parent_root=parent_root)
    rebuilt = build_plan(
        project_root=project_root,
        parent_root=parent_root,
        quarantine_path=quarantine_path,
        exposure_root=exposure_root,
        historical_plan=plan,
    )
    if rebuilt != plan:
        raise IntegrityError("v4 plan no longer reconstructs from its bound sources")
    cells = _runtime_cells(project_root=project_root, parent=parent, plan=plan)
    blockers: list[dict[str, Any]] = _credential_blockers(environment)
    budget_audit, seen_digests = _verify_budget_audit(
        budget_audit_path,
        project_root=project_root.parent,
        cap_usd=AUTHORIZED_TOTAL_CAP_USD,
        admission_fraction=DEFAULT_ADMISSION_FRACTION,
    )
    terminal_orphan_resolution = verify_reasoning_v4_terminal_orphan(
        project_root=project_root,
        route_plan_path=reasoning_v4_route_plan_path,
        receipt_path=reasoning_v4_receipt_path,
        audit_path=reasoning_v4_audit_path,
        closure_path=reasoning_v4_closure_path,
        ledger_path=reasoning_v4_ledger_path,
        journal_path=reasoning_v4_journal_path,
        source_directory=reasoning_v4_source_directory,
    )
    reasoning_v5_terminal_resolution = verify_reasoning_v5_terminal_endpoints(
        project_root=project_root,
        route_plan_path=reasoning_v5_route_plan_path,
        endpoint_snapshot_path=reasoning_v5_endpoint_snapshot_path,
        sonnet_root=reasoning_v5_sonnet_root,
        sonnet_receipt_path=reasoning_v5_sonnet_receipt_path,
        sonnet_audit_path=reasoning_v5_sonnet_audit_path,
        sonnet_closure_path=reasoning_v5_sonnet_closure_path,
        gemini_root=reasoning_v5_gemini_root,
        gemini_receipt_path=reasoning_v5_gemini_receipt_path,
        gemini_audit_path=reasoning_v5_gemini_audit_path,
        gemini_closure_path=reasoning_v5_gemini_closure_path,
        aggregate_audit_path=reasoning_v5_aggregate_audit_path,
        aggregate_closure_path=reasoning_v5_aggregate_closure_path,
    )
    failed_attempt_record = _verify_failed_pre_reservation_supersession(
        project_root=project_root,
        output_root=output_root,
        plan=plan,
    )
    exact = parent.preflight["exact_inputs"]
    v1_root = (project_root / str(exact["v1_source_directory"])).parent
    accounting_roots = [
        *supplemental_roots,
        reasoning_v5_sonnet_root,
        reasoning_v5_gemini_root,
        v1_root,
        parent_root / "v2",
    ]
    (
        supplemental_exposure,
        supplemental_actual,
        supplemental_blockers,
        bindings,
        resolved_terminal_blockers,
    ) = _supplemental_accounting(
        roots=accounting_roots,
        seen=set(seen_digests),
        terminal_orphan_resolution=terminal_orphan_resolution,
        terminal_orphan_root=reasoning_v4_ledger_path.parent,
    )
    blockers.extend(supplemental_blockers)
    global_active, global_blockers = _global_ledger_state(
        ledger_path=global_ledger_path,
        artifact_directory=global_artifact_directory,
        corrections_directory=global_corrections_directory,
        reconciliation_directory=global_reconciliation_directory,
    )
    blockers.extend(dict(item) for item in global_blockers)
    # The current roots are the exact, terminally closed execution verified
    # above.  They postdate this historical preflight and therefore are not
    # prospective-admission blockers in its reconstruction.
    baseline = _decimal(budget_audit.get("current_total_exposure_usd"), field="baseline exposure")
    current = baseline + global_active + supplemental_exposure
    outstanding = sum((cell.forecast_usd for cell in cells), Decimal(0))
    projected = current + outstanding
    ceiling = AUTHORIZED_TOTAL_CAP_USD * DEFAULT_ADMISSION_FRACTION
    if projected > ceiling or projected > AUTHORIZED_TOTAL_CAP_USD:
        blockers.append(
            {
                "gate": "shared_budget_projection",
                "projected_total_exposure_usd": _decimal_text(projected),
            }
        )
    exposure = _verify_historical_exposure_view(
        project_root=project_root,
        exposure_root=exposure_root,
        historical_plan=plan,
    )
    exposed_pairs = {
        (str(record["model_id"]), str(record["task_id"])) for record in exposure["records"]
    }
    for cell in cells:
        pair = (cell.work_item.candidate.model_id, cell.work_item.task.public_id)
        if pair in exposed_pairs:
            blockers.append(
                {
                    "gate": "planned_model_task_pair_now_exposed",
                    "model_id": pair[0],
                    "task_id": pair[1],
                }
            )
    payload = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": (
            "admissible_zero_call_preflight" if not blockers else "blocked_zero_call_preflight"
        ),
        "plan": {
            "path": _relative(project_root, plan_path),
            "sha256": plan["artifact_sha256"],
            "physical_sha256": _file_sha256(plan_path),
        },
        "parent_execution": {
            "root": _relative(project_root, parent_root),
            "preflight_sha256": PARENT_PREFLIGHT_SHA256,
            "receipt_sha256": PARENT_RECEIPT_SHA256,
            "closure_sha256": PARENT_CLOSURE_SHA256,
            "reconstructed_audit_sha256": PARENT_AUDIT_SHA256,
        },
        "exact_inputs": {
            "quarantine": {
                "path": _relative(project_root, quarantine_path),
                "physical_sha256": _file_sha256(quarantine_path),
            },
            "exposure_root": _relative(project_root, exposure_root),
            "budget_audit": {
                "path": _relative(project_root, budget_audit_path),
                "physical_sha256": _file_sha256(budget_audit_path),
            },
            "global_ledger": _relative(project_root, global_ledger_path),
            "global_ledger_head_sha256": _frontier_ledger_head(global_ledger_path),
            "supplemental_bindings": [
                {**binding, "root": _relative(project_root, Path(binding["root"]))}
                for binding in bindings
            ],
            "reasoning_v4_terminal_orphan": terminal_orphan_resolution,
            "reasoning_v5_terminal_endpoints": reasoning_v5_terminal_resolution,
            "failed_pre_reservation_attempt": failed_attempt_record,
        },
        "source_code": _source_bundle(project_root),
        "live_exposure_snapshot_sha256": exposure["records_sha256"],
        "credentials": {
            "presence_only_not_validity_test": True,
            "checked_before_any_reservation": True,
            "unprefixed_cohere_alias_accepted": False,
        },
        "budget": {
            "currency": "USD",
            "baseline_exposure_usd": _decimal_text(baseline),
            "global_active_reservation_usd": _decimal_text(global_active),
            "supplemental_actual_cost_usd": _decimal_text(supplemental_actual),
            "supplemental_exposure_usd": _decimal_text(supplemental_exposure),
            "terminal_no_replay_reservation_exposure_usd": _decimal_text(
                REASONING_V4_GEMINI_RESERVE_USD
            ),
            "terminal_no_replay_blockers_resolved": resolved_terminal_blockers,
            "terminal_reservation_released": False,
            "reasoning_v5_source_conservative_exposure_usd": _decimal_text(
                REASONING_V5_TOTAL_SOURCE_EXPOSURE_USD
            ),
            "reasoning_v5_orphan_reservation_usd": "0",
            "current_total_exposure_usd": _decimal_text(current),
            "recovery_phase_worst_case_usd": plan["budget"]["recovery_phase_worst_case_usd"],
            "glm_phase_worst_case_usd": plan["budget"]["glm_phase_worst_case_usd"],
            "outstanding_worst_case_usd": _decimal_text(outstanding),
            "projected_total_exposure_usd": _decimal_text(projected),
            "admission_ceiling_usd": _decimal_text(ceiling),
            "hard_cap_usd": _decimal_text(AUTHORIZED_TOTAL_CAP_USD),
            "admission_allowed": not blockers,
        },
        "execution": {
            "phase_order": list(PHASES),
            "recovery_confirmation": RECOVERY_CONFIRMATION,
            "glm_confirmation": GLM_CONFIRMATION,
            "provider_attempts_per_phase_max": parent.bundle.execution_policy.max_provider_attempts,
            "per_cell_failure_action": "close_fresh_cell_no_replay_and_continue",
            "glm_phase_barrier": "passing_recovery_phase_complete_disposition_audit",
        },
        "blockers": blockers,
        "calls": {"provider": 0, "epicure": 0},
        "synthetic_arms": 0,
        "claim_boundary": plan["claim_boundary"],
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _reservation_fields(
    *, cell: ContinuationCell, bundle: RuntimeBundle, namespace_sha256: str
) -> dict[str, Any]:
    return {
        "phase": cell.plan_kind,
        "plan_sha256": cell.plan_sha256,
        "reservation_namespace_sha256": namespace_sha256,
        "cell_id": cell.cell_id,
        "frozen_run_id": cell.run_id,
        "source_closed_work_item_id": cell.source_work_item_id,
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
        "permitted_analysis": "coverage_and_reliability_diagnostics_only",
    }


def _append_reservation(
    *,
    paths: RunPaths,
    runner_run_id: str,
    cell: ContinuationCell,
    bundle: RuntimeBundle,
    namespace_sha256: str,
    environment: Mapping[str, str],
) -> Mapping[str, Any]:
    backend = cell.work_item.candidate.execution_backend
    require_prefixed_credential_before_reservation(backend, environment)
    return append_dataset_ledger_event(
        paths.ledger,
        {
            "event_type": "reservation_created",
            "runner_run_id": runner_run_id,
            "work_item_id": cell.work_item.work_item_id,
            "reserved_usd": _decimal_text(cell.forecast_usd),
            "execution_backend": backend,
            "credential_preflight": (
                "prefixed_cohere_present_before_reservation"
                if backend == "cohere_direct"
                else "not_applicable"
            ),
            "safe_to_replay_parent_or_v4_work": False,
            **_reservation_fields(cell=cell, bundle=bundle, namespace_sha256=namespace_sha256),
        },
    )


def _find_phase_artifact(output_root: Path, phase: str, role: str) -> Path | None:
    paths = sorted(output_root.glob(f"frontier-coverage-recovery-v4-{phase}-{role}-*.json"))
    if len(paths) > 1:
        raise IntegrityError(f"multiple v4 {phase} {role} artifacts exist")
    return paths[0] if paths else None


def _verify_external_heads(project_root: Path, preflight: Mapping[str, Any]) -> None:
    exact = preflight["exact_inputs"]
    global_path = project_root / str(exact["global_ledger"])
    if _frontier_ledger_head(global_path) != exact.get("global_ledger_head_sha256"):
        raise AdmissionDenied("global budget ledger changed after v4 preflight")
    for binding in exact.get("supplemental_bindings") or []:
        root = project_root / str(binding["root"])
        if _ledger_head(root / "ledger.jsonl") != binding.get("ledger_head_sha256"):
            raise AdmissionDenied("supplemental budget ledger changed after v4 preflight")


def _verify_terminal_resolution_from_preflight(
    project_root: Path, preflight: Mapping[str, Any]
) -> None:
    exact = preflight.get("exact_inputs")
    resolution = exact.get("reasoning_v4_terminal_orphan") if isinstance(exact, Mapping) else None
    if not isinstance(resolution, Mapping):
        raise IntegrityError("reasoning-v4 terminal orphan resolution is absent")
    evidence = resolution.get("evidence")
    if not isinstance(evidence, Mapping):
        raise IntegrityError("reasoning-v4 terminal orphan evidence is absent")

    root = project_root.resolve()

    def evidence_path(label: str) -> Path:
        record = evidence.get(label)
        if not isinstance(record, Mapping):
            raise IntegrityError(f"reasoning-v4 {label} evidence binding is absent")
        raw = Path(str(record.get("path") or ""))
        if raw.is_absolute():
            raise IntegrityError(f"reasoning-v4 {label} evidence path is not relative")
        path = (root / raw).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise IntegrityError(
                f"reasoning-v4 {label} evidence path escapes the project root"
            ) from error
        return path

    source_raw = Path(str(evidence.get("source_directory") or ""))
    if source_raw.is_absolute():
        raise IntegrityError("reasoning-v4 source directory binding is not relative")
    source_directory = (root / source_raw).resolve()
    try:
        source_directory.relative_to(root)
    except ValueError as error:
        raise IntegrityError(
            "reasoning-v4 source directory binding escapes the project root"
        ) from error

    rebuilt = verify_reasoning_v4_terminal_orphan(
        project_root=project_root,
        route_plan_path=evidence_path("route_plan"),
        receipt_path=evidence_path("receipt"),
        audit_path=evidence_path("audit"),
        closure_path=evidence_path("closure"),
        ledger_path=evidence_path("ledger"),
        journal_path=evidence_path("journal"),
        source_directory=source_directory,
    )
    if rebuilt != dict(resolution):
        raise IntegrityError("reasoning-v4 terminal orphan resolution changed after preflight")


def _verify_reasoning_v5_resolution_from_preflight(
    project_root: Path, preflight: Mapping[str, Any]
) -> None:
    exact = preflight.get("exact_inputs")
    resolution = (
        exact.get("reasoning_v5_terminal_endpoints") if isinstance(exact, Mapping) else None
    )
    if not isinstance(resolution, Mapping):
        raise IntegrityError("reasoning-v5 terminal endpoint resolution is absent")
    root = project_root.resolve()

    def relative_path(value: object, *, label: str) -> Path:
        raw = Path(str(value or ""))
        if raw.is_absolute():
            raise IntegrityError(f"reasoning-v5 {label} path is not relative")
        path = (root / raw).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"reasoning-v5 {label} path escapes the project root") from error
        return path

    def record_path(record: object, *, label: str) -> Path:
        if not isinstance(record, Mapping):
            raise IntegrityError(f"reasoning-v5 {label} binding is absent")
        return relative_path(record.get("path"), label=label)

    endpoints = resolution.get("endpoints")
    aggregate = resolution.get("aggregate")
    if not isinstance(endpoints, Mapping) or not isinstance(aggregate, Mapping):
        raise IntegrityError("reasoning-v5 endpoint or aggregate evidence is absent")
    sonnet = endpoints.get("sonnet")
    gemini = endpoints.get("gemini")
    if not isinstance(sonnet, Mapping) or not isinstance(gemini, Mapping):
        raise IntegrityError("reasoning-v5 endpoint evidence is incomplete")
    rebuilt = verify_reasoning_v5_terminal_endpoints(
        project_root=project_root,
        route_plan_path=record_path(resolution.get("route_plan"), label="route plan"),
        endpoint_snapshot_path=record_path(
            resolution.get("endpoint_snapshot"), label="endpoint snapshot"
        ),
        sonnet_root=relative_path(sonnet.get("root"), label="Sonnet root"),
        sonnet_receipt_path=record_path(sonnet.get("receipt"), label="Sonnet receipt"),
        sonnet_audit_path=record_path(sonnet.get("audit"), label="Sonnet audit"),
        sonnet_closure_path=record_path(sonnet.get("closure"), label="Sonnet closure"),
        gemini_root=relative_path(gemini.get("root"), label="Gemini root"),
        gemini_receipt_path=record_path(gemini.get("receipt"), label="Gemini receipt"),
        gemini_audit_path=record_path(gemini.get("audit"), label="Gemini audit"),
        gemini_closure_path=record_path(gemini.get("closure"), label="Gemini closure"),
        aggregate_audit_path=record_path(aggregate.get("audit"), label="aggregate audit"),
        aggregate_closure_path=record_path(aggregate.get("closure"), label="aggregate closure"),
    )
    if rebuilt != dict(resolution):
        raise IntegrityError("reasoning-v5 terminal endpoint resolution changed after preflight")


def _load_execution_context(
    *,
    preflight_path: Path,
    project_root: Path,
    output_root: Path,
    allowed_output_exposure_phases: frozenset[str] = frozenset(),
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    RuntimeBundle,
    tuple[ContinuationCell, ...],
    ParentState,
]:
    preflight = _load_addressed(
        preflight_path, label="v4 preflight", expected_schema=PREFLIGHT_SCHEMA_VERSION
    )
    if preflight.get("status") != "admissible_zero_call_preflight":
        raise AdmissionDenied("v4 preflight is not admissible")
    if preflight.get("source_code") != _source_bundle(project_root):
        raise IntegrityError("v4 execution source changed after preflight")
    plan_record = preflight.get("plan")
    plan_path = project_root / str(plan_record.get("path") or "")
    if _file_sha256(plan_path) != plan_record.get("physical_sha256"):
        raise IntegrityError("v4 plan physical digest changed")
    plan = _load_addressed(
        plan_path,
        label="v4 recovery plan",
        expected_schema=PLAN_SCHEMA_VERSION,
        expected_digest=str(plan_record.get("sha256") or ""),
    )
    parent_root = project_root / str(preflight["parent_execution"]["root"])
    parent = reconstruct_parent(project_root=project_root, parent_root=parent_root)
    cells = _runtime_cells(project_root=project_root, parent=parent, plan=plan)
    _verify_external_heads(project_root, preflight)
    _verify_terminal_resolution_from_preflight(project_root, preflight)
    _verify_reasoning_v5_resolution_from_preflight(project_root, preflight)
    blockers = _credential_blockers(os.environ)
    if blockers:
        raise AdmissionDenied(f"v4 runtime credentials/routes are absent: {blockers}")
    exposure_root = project_root / str(preflight["exact_inputs"]["exposure_root"])
    exposure = _scan_model_task_exposure(
        exposure_root, model_ids={cell.work_item.candidate.model_id for cell in cells}
    )
    planned = {(cell.work_item.candidate.model_id, cell.work_item.task.public_id) for cell in cells}
    allowed_source_roots = tuple(
        _phase_paths(output_root, phase).source.resolve()
        for phase in allowed_output_exposure_phases
    )
    unexpected_exposures: list[tuple[str, str, str]] = []
    for record in exposure["records"]:
        pair = (str(record["model_id"]), str(record["task_id"]))
        if pair not in planned:
            continue
        path = (exposure_root / str(record["path"])).resolve()
        if any(path.is_relative_to(root) for root in allowed_source_roots):
            continue
        unexpected_exposures.append((*pair, str(record["path"])))
    if unexpected_exposures:
        raise AdmissionDenied(
            "a planned v4 model-task pair became exposed outside its admitted phase: "
            f"{unexpected_exposures}"
        )
    bundle = RuntimeBundle(
        document=plan,
        cells=cells,
        epicure=parent.bundle.epicure,
        execution_policy=parent.bundle.execution_policy,
    )
    return preflight, plan, bundle, cells, parent


def _barrier_audit(*, project_root: Path, output_root: Path, path: Path) -> Mapping[str, Any]:
    audit = _load_addressed(
        path, label="recovery phase audit", expected_schema=AUDIT_SCHEMA_VERSION
    )
    if (
        audit.get("phase") != RECOVERY_PHASE
        or audit.get("decision") != "passed_complete_phase_disposition"
        or audit.get("counts", {}).get("terminally_dispositioned_cells") != 7
        or audit.get("claim_boundary", {}).get("official") is not False
    ):
        raise AdmissionDenied("GLM phase requires a passing seven-cell disposition audit")
    expected = _find_phase_artifact(output_root, RECOVERY_PHASE, "audit")
    if expected is None or expected.resolve() != path.resolve():
        raise IntegrityError("GLM barrier audit is not the unique recovery audit")
    if _file_sha256(path) != _file_sha256(expected):
        raise IntegrityError("GLM barrier audit physical digest differs")
    del project_root
    return audit


def _execute_cell(
    *,
    cell: ContinuationCell,
    bundle: RuntimeBundle,
    paths: RunPaths,
    runner_run_id: str,
    namespace_sha256: str,
    process_timeout_seconds: int,
) -> tuple[dict[str, Any], bool]:
    work_id = cell.work_item.work_item_id
    with _dataset_ledger_lock(paths.ledger):
        entries = load_dataset_ledger(paths.ledger)
        reservations, finalizations = dataset_ledger_state(entries)
        if work_id in finalizations:
            return {"work_item_id": work_id, "decision": "skip_terminally_finalized"}, False
        reservation = reservations.get(work_id)
        source = _source_for_work_item(paths, work_id) if reservation else None
        if reservation is not None and source is None:
            evidence = _recovery_evidence(paths, work_id)
            incident = append_dataset_ledger_event(
                paths.ledger,
                {
                    "event_type": "execution_incident",
                    "runner_run_id": runner_run_id,
                    "work_item_id": work_id,
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "incident": "reserved_without_source_cell_closed_no_replay_continue",
                    "delivery_evidence": evidence,
                    "terminal_disposition": True,
                    "safe_to_replay": False,
                },
            )
            return {
                "work_item_id": work_id,
                "decision": "reserved_without_source_closed_no_replay_continue",
                "incident_entry_sha256": incident["entry_sha256"],
            }, False
        if reservation is not None and source is not None:
            try:
                event, complete, issues = _finalize_source(
                    cell=cell,
                    bundle=bundle,
                    paths=paths,
                    reservation=reservation,
                    runner_run_id=runner_run_id,
                    source=source,
                )
            except IntegrityError as error:
                incident = append_dataset_ledger_event(
                    paths.ledger,
                    {
                        "event_type": "execution_incident",
                        "runner_run_id": runner_run_id,
                        "work_item_id": work_id,
                        "reservation_entry_sha256": reservation["entry_sha256"],
                        "incident": "source_integrity_failure_cell_closed_no_replay_continue",
                        "error_sha256": _safe_process_hash(str(error)),
                        "terminal_disposition": True,
                        "safe_to_replay": False,
                    },
                )
                return {
                    "work_item_id": work_id,
                    "decision": "source_integrity_failure_closed_continue",
                    "incident_entry_sha256": incident["entry_sha256"],
                }, False
            return {
                "work_item_id": work_id,
                "decision": (
                    "recovered_source_complete"
                    if complete
                    else "recovered_source_incomplete_closed_continue"
                ),
                "source_artifact_sha256": source.artifact_sha256,
                "ledger_entry_sha256": event["entry_sha256"],
                "issues": issues,
            }, False

        reservation = _append_reservation(
            paths=paths,
            runner_run_id=runner_run_id,
            cell=cell,
            bundle=bundle,
            namespace_sha256=namespace_sha256,
            environment=os.environ,
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
                raise IntegrityError("OpenRouter command is not the qualified runner")
            command[2] = "flavourbench.continuation_openrouter_pair"
        command.extend(["--frozen-run-id", cell.run_id])
        command.extend(
            [
                "--frozen-attempt-slots-json",
                json.dumps(list(cell.attempt_slots), separators=(",", ":"), sort_keys=True),
            ]
        )
        environment = os.environ.copy()
        environment.update(bundle.execution_policy.settings_environment())
        environment["FLAVOURBENCH_OPENROUTER_MAX_PROMPT_PRICE_PER_MTOK"] = _decimal_text(
            forecast.price_envelope.prompt_usd_per_mtok
        )
        environment["FLAVOURBENCH_OPENROUTER_MAX_COMPLETION_PRICE_PER_MTOK"] = _decimal_text(
            forecast.price_envelope.completion_usd_per_mtok
        )
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
            evidence = _recovery_evidence(paths, work_id)
            incident = append_dataset_ledger_event(
                paths.ledger,
                {
                    "event_type": "execution_incident",
                    "runner_run_id": runner_run_id,
                    "work_item_id": work_id,
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "incident": "subprocess_timeout_cell_closed_no_replay_continue",
                    "timeout_seconds": process_timeout_seconds,
                    "output_sha256": _safe_process_hash(str(error.output or "")),
                    "delivery_evidence": evidence,
                    "terminal_disposition": True,
                    "safe_to_replay": False,
                },
            )
            return {
                "work_item_id": work_id,
                "decision": "timeout_closed_no_replay_continue",
                "incident_entry_sha256": incident["entry_sha256"],
            }, True
        artifact_path = _extract_artifact_path(completed.stdout, paths.source)
        if artifact_path is None or not artifact_path.exists():
            evidence = _recovery_evidence(paths, work_id)
            incident = append_dataset_ledger_event(
                paths.ledger,
                {
                    "event_type": "execution_incident",
                    "runner_run_id": runner_run_id,
                    "work_item_id": work_id,
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "incident": "no_verifiable_source_cell_closed_no_replay_continue",
                    "subprocess_returncode": completed.returncode,
                    "stdout_sha256": _safe_process_hash(completed.stdout),
                    "stderr_sha256": _safe_process_hash(completed.stderr),
                    "delivery_evidence": evidence,
                    "terminal_disposition": True,
                    "safe_to_replay": False,
                },
            )
            return {
                "work_item_id": work_id,
                "decision": "no_source_closed_no_replay_continue",
                "incident_entry_sha256": incident["entry_sha256"],
            }, True
        source = _source_for_work_item(paths, work_id)
        if source is None or source.path.resolve() != artifact_path.resolve():
            incident = append_dataset_ledger_event(
                paths.ledger,
                {
                    "event_type": "execution_incident",
                    "runner_run_id": runner_run_id,
                    "work_item_id": work_id,
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "incident": "source_work_item_binding_failure_closed_no_replay_continue",
                    "terminal_disposition": True,
                    "safe_to_replay": False,
                },
            )
            return {
                "work_item_id": work_id,
                "decision": "source_binding_failure_closed_continue",
                "incident_entry_sha256": incident["entry_sha256"],
            }, True
        try:
            event, complete, issues = _finalize_source(
                cell=cell,
                bundle=bundle,
                paths=paths,
                reservation=reservation,
                runner_run_id=runner_run_id,
                source=source,
            )
        except IntegrityError as error:
            incident = append_dataset_ledger_event(
                paths.ledger,
                {
                    "event_type": "execution_incident",
                    "runner_run_id": runner_run_id,
                    "work_item_id": work_id,
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "incident": "source_integrity_failure_cell_closed_no_replay_continue",
                    "source_artifact_sha256": source.artifact_sha256,
                    "error_sha256": _safe_process_hash(str(error)),
                    "terminal_disposition": True,
                    "safe_to_replay": False,
                },
            )
            return {
                "work_item_id": work_id,
                "decision": "source_integrity_failure_closed_continue",
                "incident_entry_sha256": incident["entry_sha256"],
            }, True
        return {
            "work_item_id": work_id,
            "decision": (
                "source_finalized_complete"
                if complete
                else "source_finalized_incomplete_closed_continue"
            ),
            "source_artifact_sha256": source.artifact_sha256,
            "ledger_entry_sha256": event["entry_sha256"],
            "issues": issues,
        }, True


def _collect_independent_dispositions(
    cells: Sequence[ContinuationCell],
    execute_one: Callable[[ContinuationCell], tuple[dict[str, Any], bool]],
) -> tuple[list[dict[str, Any]], int]:
    """Run every fresh cell even when an earlier cell reports failure.

    Provider exceptions are converted to terminal cell outcomes inside
    ``_execute_cell``.  This loop deliberately has no batch-level stop flag.
    """

    outcomes: list[dict[str, Any]] = []
    subprocesses_started = 0
    for cell in cells:
        outcome, started = execute_one(cell)
        outcomes.append(outcome)
        subprocesses_started += int(started)
    return outcomes, subprocesses_started


def execute_phase(
    *,
    preflight_path: Path,
    project_root: Path,
    output_root: Path,
    phase: str,
    confirmation: str,
    process_timeout_seconds: int,
    recovery_audit_path: Path | None = None,
) -> tuple[Path, Path]:
    expected_confirmation = RECOVERY_CONFIRMATION if phase == RECOVERY_PHASE else GLM_CONFIRMATION
    if phase not in PHASES or confirmation != expected_confirmation:
        raise AdmissionDenied(f"{phase} execution requires --confirm {expected_confirmation}")
    preflight, plan, bundle, all_cells, _ = _load_execution_context(
        preflight_path=preflight_path,
        project_root=project_root,
        output_root=output_root,
        allowed_output_exposure_phases=(
            frozenset() if phase == RECOVERY_PHASE else frozenset({RECOVERY_PHASE})
        ),
    )
    if _find_phase_artifact(output_root, phase, "closure") is not None:
        raise AdmissionDenied(f"v4 {phase} identifiers are permanently closed")
    if phase == GLM_PHASE:
        if recovery_audit_path is None:
            raise AdmissionDenied("GLM phase requires --recovery-audit")
        _barrier_audit(project_root=project_root, output_root=output_root, path=recovery_audit_path)
    cells = tuple(cell for cell in all_cells if cell.plan_kind == phase)
    expected_count = 7 if phase == RECOVERY_PHASE else 1
    if len(cells) != expected_count:
        raise IntegrityError("v4 phase cell count differs")
    runner_run_id = str(uuid.uuid4())
    paths = _phase_paths(output_root, phase)
    namespace_sha256 = sha256_json(
        {"plan_sha256": plan["artifact_sha256"], "phase": phase, "role": "reservations"}
    )
    global_ledger = project_root / str(preflight["exact_inputs"]["global_ledger"])
    with _exclusive_runner_lock(global_ledger):
        _verify_external_heads(project_root, preflight)
        outcomes, subprocesses_started = _collect_independent_dispositions(
            cells,
            lambda cell: _execute_cell(
                cell=cell,
                bundle=bundle,
                paths=paths,
                runner_run_id=runner_run_id,
                namespace_sha256=namespace_sha256,
                process_timeout_seconds=process_timeout_seconds,
            ),
        )
    if len(outcomes) != len(cells):
        raise IntegrityError("v4 phase did not disposition every cell")
    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "phase": phase,
        "preflight_sha256": preflight["artifact_sha256"],
        "plan_sha256": plan["artifact_sha256"],
        "runner_run_id": runner_run_id,
        "status": "all_phase_cells_received_terminal_disposition",
        "subprocesses_started": subprocesses_started,
        "outcomes": outcomes,
        "manual_retries": 0,
        "same_identifier_replays": 0,
        "completed_at": _utc_now(),
        "official_fit_eligible": False,
    }
    receipt_path = _write_addressed(
        receipt_payload,
        directory=output_root,
        prefix=f"frontier-coverage-recovery-v4-{phase}-receipt",
    )
    receipt = _load_addressed(
        receipt_path, label="v4 phase receipt", expected_schema=RECEIPT_SCHEMA_VERSION
    )
    closure_payload = {
        "schema_version": CLOSURE_SCHEMA_VERSION,
        "phase": phase,
        "preflight_sha256": preflight["artifact_sha256"],
        "plan_sha256": plan["artifact_sha256"],
        "receipt_sha256": receipt["artifact_sha256"],
        "status": "permanently_closed_all_phase_identifiers",
        "all_cells_received_terminal_disposition": True,
        "ledger_head_sha256": _ledger_head(paths.ledger),
        "closed_work_item_ids": sorted(cell.work_item.work_item_id for cell in cells),
        "closed_run_ids": sorted(cell.run_id for cell in cells),
        "closed_attempt_ids_sha256": sha256_json(
            sorted(str(slot["attempt_id"]) for cell in cells for slot in cell.attempt_slots)
        ),
        "safe_to_replay_any_parent_or_v4_work": False,
        "future_execution_with_same_phase_permitted": False,
        "parent_failures_superseded": False,
        "official_fit_eligible": False,
    }
    closure_path = _write_addressed(
        closure_payload,
        directory=output_root,
        prefix=f"frontier-coverage-recovery-v4-{phase}-closure",
    )
    return receipt_path, closure_path


def build_phase_audit(
    *,
    preflight_path: Path,
    receipt_path: Path,
    closure_path: Path,
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    preflight = _load_addressed(
        preflight_path, label="v4 preflight", expected_schema=PREFLIGHT_SCHEMA_VERSION
    )
    receipt = _load_addressed(
        receipt_path, label="v4 receipt", expected_schema=RECEIPT_SCHEMA_VERSION
    )
    closure = _load_addressed(
        closure_path, label="v4 closure", expected_schema=CLOSURE_SCHEMA_VERSION
    )
    phase = str(receipt.get("phase") or "")
    if phase not in PHASES or closure.get("phase") != phase:
        raise IntegrityError("v4 audit phase binding differs")
    preflight_loaded, plan, bundle, all_cells, _ = _load_execution_context(
        preflight_path=preflight_path,
        project_root=project_root,
        output_root=output_root,
        allowed_output_exposure_phases=(
            frozenset({RECOVERY_PHASE}) if phase == RECOVERY_PHASE else frozenset(PHASES)
        ),
    )
    cells = tuple(cell for cell in all_cells if cell.plan_kind == phase)
    failures: list[str] = []
    if (
        preflight_loaded["artifact_sha256"] != preflight["artifact_sha256"]
        or receipt.get("preflight_sha256") != preflight["artifact_sha256"]
        or closure.get("preflight_sha256") != preflight["artifact_sha256"]
        or receipt.get("plan_sha256") != plan["artifact_sha256"]
        or closure.get("plan_sha256") != plan["artifact_sha256"]
        or closure.get("receipt_sha256") != receipt["artifact_sha256"]
        or closure.get("all_cells_received_terminal_disposition") is not True
        or closure.get("safe_to_replay_any_parent_or_v4_work") is not False
    ):
        failures.append("preflight_receipt_closure_binding_mismatch")
    paths = _phase_paths(output_root, phase)
    entries = load_dataset_ledger(paths.ledger)
    if closure.get("ledger_head_sha256") != _ledger_head(paths.ledger):
        failures.append("phase_ledger_head_mismatch")
    reservations, finalizations = dataset_ledger_state(entries)
    responses = scan_response_artifacts(paths.responses)
    incidents = {
        str(entry.get("work_item_id") or ""): entry
        for entry in entries
        if entry.get("event_type") == "execution_incident"
    }
    namespace_sha256 = sha256_json(
        {"plan_sha256": plan["artifact_sha256"], "phase": phase, "role": "reservations"}
    )
    all_attempt_ids: set[str] = set()
    all_generation_ids: set[str] = set()
    total_cost_micros = 0
    successful_tools = 0
    usable_cells = 0
    terminal_cells = 0
    reliability_failures = 0
    records: list[dict[str, Any]] = []
    for cell in cells:
        work_id = cell.work_item.work_item_id
        reservation = reservations.get(work_id)
        finalization = finalizations.get(work_id)
        incident = incidents.get(work_id)
        source = _source_for_work_item(paths, work_id) if reservation else None
        local_integrity: list[str] = []
        observations: list[str] = []
        if reservation is None:
            local_integrity.append("missing_reservation_and_terminal_disposition")
        else:
            exact_fields = _reservation_fields(
                cell=cell, bundle=bundle, namespace_sha256=namespace_sha256
            )
            if any(reservation.get(field) != value for field, value in exact_fields.items()):
                local_integrity.append("reservation_differs_from_frozen_cell")
        if source is not None:
            source_findings, attempts, generations, cost_micros, tools = _audit_source(
                cell=cell, bundle=bundle, source=source
            )
            observations.extend(
                finding for finding in source_findings if finding in OBSERVATIONAL_SOURCE_FAILURES
            )
            local_integrity.extend(
                finding
                for finding in source_findings
                if finding not in OBSERVATIONAL_SOURCE_FAILURES
            )
            if all_attempt_ids.intersection(attempts):
                local_integrity.append("attempt_id_overlaps_another_v4_cell")
            if all_generation_ids.intersection(generations):
                local_integrity.append("generation_id_overlaps_another_v4_cell")
            all_attempt_ids.update(attempts)
            all_generation_ids.update(generations)
            total_cost_micros += cost_micros
            successful_tools += tools
            observed = {key for key in responses if key[0] == work_id}
            expected = {(work_id, condition) for condition in cell.conditions}
            if finalization is None and incident is None:
                local_integrity.append("source_has_no_terminal_ledger_disposition")
            elif finalization is not None:
                terminal_cells += 1
                complete = finalization.get("complete_required_conditions") is True
                if complete and observed != expected:
                    local_integrity.append("complete_finalization_response_set_mismatch")
                if complete and not source_findings:
                    usable_cells += 1
                else:
                    reliability_failures += 1
            else:
                terminal_cells += 1
                reliability_failures += 1
        elif reservation is not None and incident is not None:
            delivery = incident.get("delivery_evidence")
            if (
                incident.get("safe_to_replay") is not False
                or incident.get("terminal_disposition") is not True
            ):
                local_integrity.append("incident_is_not_terminal_no_replay")
            if "source_integrity_failure" not in str(incident.get("incident") or ""):
                classification = (
                    str(delivery.get("delivery_classification") or "")
                    if isinstance(delivery, Mapping)
                    else ""
                )
                if classification not in {
                    "no_journal_delivery_state_unknown",
                    "uncertain_delivery_or_unreconciled_generation",
                    "provider_generation_observed_source_missing",
                    "pre_request_or_safe_provider_rejection_no_generation",
                    "journal_present_delivery_state_requires_manual_reconciliation",
                }:
                    local_integrity.append("incident_delivery_classification_missing")
            terminal_cells += 1
            reliability_failures += 1
        elif reservation is not None:
            local_integrity.append("reservation_has_no_source_or_terminal_incident")
        failures.extend(f"{work_id}:{failure}" for failure in local_integrity)
        records.append(
            {
                "work_item_id": work_id,
                "model_id": cell.work_item.candidate.model_id,
                "task_id": cell.work_item.task.public_id,
                "terminal_disposition": not local_integrity
                and (finalization is not None or incident is not None),
                "usable_complete_pair": (
                    finalization is not None
                    and finalization.get("complete_required_conditions") is True
                    and not local_integrity
                    and not observations
                ),
                "observed_reliability_failures": sorted(set(observations)),
                "integrity_failures": sorted(set(local_integrity)),
                "source_sha256": source.artifact_sha256 if source else None,
            }
        )
    if sorted(closure.get("closed_work_item_ids") or []) != sorted(
        cell.work_item.work_item_id for cell in cells
    ):
        failures.append("closure_work_item_set_mismatch")
    unique_failures = sorted(set(failures))
    disposition_pass = not unique_failures and terminal_cells == len(cells)
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "phase": phase,
        "preflight_sha256": preflight["artifact_sha256"],
        "plan_sha256": plan["artifact_sha256"],
        "receipt_sha256": receipt["artifact_sha256"],
        "closure_sha256": closure["artifact_sha256"],
        "decision": (
            "passed_complete_phase_disposition"
            if disposition_pass
            else "failed_closed_incomplete_or_unverifiable_disposition"
        ),
        "integrity_failures": unique_failures,
        "counts": {
            "planned_cells": len(cells),
            "terminally_dispositioned_cells": terminal_cells,
            "usable_complete_cells": usable_cells,
            "usable_real_arms": usable_cells * 2,
            "reliability_failure_cells": reliability_failures,
            "provider_generations": len(all_generation_ids),
            "successful_epicure_tool_calls": successful_tools,
            "synthetic_arms": 0,
        },
        "accounting": {
            "actual_cost_micros": total_cost_micros,
            "actual_cost_usd": _decimal_text(Decimal(total_cost_micros) / Decimal(1_000_000)),
        },
        "identifier_audit": {
            "attempt_id_count": len(all_attempt_ids),
            "generation_id_count": len(all_generation_ids),
            "attempt_ids_sha256": sha256_json(sorted(all_attempt_ids)),
            "generation_ids_sha256": sha256_json(sorted(all_generation_ids)),
            "same_identifier_replay_permitted": False,
        },
        "cells": records,
        "parent_disposition": {
            "parent_audit_sha256": PARENT_AUDIT_SHA256,
            "parent_complete_cell_preserved": True,
            "parent_incomplete_glm_failure_preserved": True,
            "parent_reliability_failures_superseded": False,
        },
        "claim_boundary": plan["claim_boundary"],
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "preflight", "execute", "audit"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--exposure-root", type=Path)
    parser.add_argument("--plan", type=Path)
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
    parser.add_argument("--reasoning-v4-route-plan", type=Path)
    parser.add_argument("--reasoning-v4-receipt", type=Path)
    parser.add_argument("--reasoning-v4-audit", type=Path)
    parser.add_argument("--reasoning-v4-closure", type=Path)
    parser.add_argument("--reasoning-v4-ledger", type=Path)
    parser.add_argument("--reasoning-v4-journal", type=Path)
    parser.add_argument("--reasoning-v4-source-directory", type=Path)
    parser.add_argument("--reasoning-v5-route-plan", type=Path)
    parser.add_argument("--reasoning-v5-endpoint-snapshot", type=Path)
    parser.add_argument("--reasoning-v5-sonnet-root", type=Path)
    parser.add_argument("--reasoning-v5-sonnet-receipt", type=Path)
    parser.add_argument("--reasoning-v5-sonnet-audit", type=Path)
    parser.add_argument("--reasoning-v5-sonnet-closure", type=Path)
    parser.add_argument("--reasoning-v5-gemini-root", type=Path)
    parser.add_argument("--reasoning-v5-gemini-receipt", type=Path)
    parser.add_argument("--reasoning-v5-gemini-audit", type=Path)
    parser.add_argument("--reasoning-v5-gemini-closure", type=Path)
    parser.add_argument("--reasoning-v5-aggregate-audit", type=Path)
    parser.add_argument("--reasoning-v5-aggregate-closure", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--recovery-audit", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--execution-closure", type=Path)
    parser.add_argument("--process-timeout-seconds", type=int, default=3600)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "freeze":
        required = (args.parent_root, args.quarantine, args.exposure_root)
        if any(value is None for value in required):
            raise SystemExit("freeze requires --parent-root, --quarantine, and --exposure-root")
        plan = build_plan(
            project_root=project_root,
            parent_root=args.parent_root.resolve(),
            quarantine_path=args.quarantine.resolve(),
            exposure_root=args.exposure_root.resolve(),
        )
        path = _write_addressed(
            {key: value for key, value in plan.items() if key != "artifact_sha256"},
            directory=output_root,
            prefix="frontier-coverage-recovery-v4-plan",
        )
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "plan": str(path),
                    "budget": plan["budget"],
                    "provider_calls": 0,
                    "epicure_calls": 0,
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "preflight":
        required = (
            args.plan,
            args.parent_root,
            args.quarantine,
            args.exposure_root,
            args.budget_audit,
            args.reasoning_v4_route_plan,
            args.reasoning_v4_receipt,
            args.reasoning_v4_audit,
            args.reasoning_v4_closure,
            args.reasoning_v4_ledger,
            args.reasoning_v4_journal,
            args.reasoning_v4_source_directory,
            args.reasoning_v5_route_plan,
            args.reasoning_v5_endpoint_snapshot,
            args.reasoning_v5_sonnet_root,
            args.reasoning_v5_sonnet_receipt,
            args.reasoning_v5_sonnet_audit,
            args.reasoning_v5_sonnet_closure,
            args.reasoning_v5_gemini_root,
            args.reasoning_v5_gemini_receipt,
            args.reasoning_v5_gemini_audit,
            args.reasoning_v5_gemini_closure,
            args.reasoning_v5_aggregate_audit,
            args.reasoning_v5_aggregate_closure,
        )
        if any(value is None for value in required):
            raise SystemExit("preflight inputs are incomplete")
        preflight = build_preflight(
            project_root=project_root,
            plan_path=args.plan.resolve(),
            parent_root=args.parent_root.resolve(),
            quarantine_path=args.quarantine.resolve(),
            exposure_root=args.exposure_root.resolve(),
            output_root=output_root,
            budget_audit_path=args.budget_audit.resolve(),
            supplemental_roots=[path.resolve() for path in args.supplemental_run_root],
            global_ledger_path=args.global_ledger.resolve(),
            global_artifact_directory=args.global_artifact_directory.resolve(),
            global_corrections_directory=args.global_corrections_directory.resolve(),
            global_reconciliation_directory=args.global_reconciliation_directory.resolve(),
            reasoning_v4_route_plan_path=args.reasoning_v4_route_plan.resolve(),
            reasoning_v4_receipt_path=args.reasoning_v4_receipt.resolve(),
            reasoning_v4_audit_path=args.reasoning_v4_audit.resolve(),
            reasoning_v4_closure_path=args.reasoning_v4_closure.resolve(),
            reasoning_v4_ledger_path=args.reasoning_v4_ledger.resolve(),
            reasoning_v4_journal_path=args.reasoning_v4_journal.resolve(),
            reasoning_v4_source_directory=args.reasoning_v4_source_directory.resolve(),
            reasoning_v5_route_plan_path=args.reasoning_v5_route_plan.resolve(),
            reasoning_v5_endpoint_snapshot_path=args.reasoning_v5_endpoint_snapshot.resolve(),
            reasoning_v5_sonnet_root=args.reasoning_v5_sonnet_root.resolve(),
            reasoning_v5_sonnet_receipt_path=args.reasoning_v5_sonnet_receipt.resolve(),
            reasoning_v5_sonnet_audit_path=args.reasoning_v5_sonnet_audit.resolve(),
            reasoning_v5_sonnet_closure_path=args.reasoning_v5_sonnet_closure.resolve(),
            reasoning_v5_gemini_root=args.reasoning_v5_gemini_root.resolve(),
            reasoning_v5_gemini_receipt_path=args.reasoning_v5_gemini_receipt.resolve(),
            reasoning_v5_gemini_audit_path=args.reasoning_v5_gemini_audit.resolve(),
            reasoning_v5_gemini_closure_path=args.reasoning_v5_gemini_closure.resolve(),
            reasoning_v5_aggregate_audit_path=args.reasoning_v5_aggregate_audit.resolve(),
            reasoning_v5_aggregate_closure_path=args.reasoning_v5_aggregate_closure.resolve(),
            environment=os.environ,
        )
        path = _write_addressed(
            {key: value for key, value in preflight.items() if key != "artifact_sha256"},
            directory=output_root,
            prefix="frontier-coverage-recovery-v4-preflight",
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
        return
    if args.command == "execute":
        if args.preflight is None or args.phase is None:
            raise SystemExit("execute requires --preflight and --phase")
        receipt, closure = execute_phase(
            preflight_path=args.preflight.resolve(),
            project_root=project_root,
            output_root=output_root,
            phase=args.phase,
            confirmation=args.confirm,
            process_timeout_seconds=args.process_timeout_seconds,
            recovery_audit_path=(args.recovery_audit.resolve() if args.recovery_audit else None),
        )
        print(
            json.dumps(
                {
                    "status": "phase_permanently_closed",
                    "phase": args.phase,
                    "receipt": str(receipt),
                    "closure": str(closure),
                },
                sort_keys=True,
            )
        )
        return
    if args.preflight is None or args.receipt is None or args.execution_closure is None:
        raise SystemExit("audit requires --preflight, --receipt, and --execution-closure")
    audit = build_phase_audit(
        preflight_path=args.preflight.resolve(),
        receipt_path=args.receipt.resolve(),
        closure_path=args.execution_closure.resolve(),
        project_root=project_root,
        output_root=output_root,
    )
    path = _write_addressed(
        {key: value for key, value in audit.items() if key != "artifact_sha256"},
        directory=output_root,
        prefix=f"frontier-coverage-recovery-v4-{audit['phase']}-audit",
    )
    print(json.dumps({"status": audit["decision"], "audit": str(path)}, sort_keys=True))


if __name__ == "__main__":
    run()
