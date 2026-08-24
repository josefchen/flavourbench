"""Execute and independently verify the reasoning-effort v4 route gate.

This module is intentionally outside the generation-source bundle frozen by
``reasoning_effort_sensitivity_v4``.  It may orchestrate the six diagnostic
pairs, but it may not change the provider, live-smoke, journal, policy, MCP, or
dataset-runner code whose hashes are already part of the route plan.

Planning is the default and performs no provider or MCP calls.  Paid execution
requires the exact confirmation token in the frozen plan, holds both the shared
frontier-budget lock and a route-local ledger lock, admits one matched pair at
a time, and permanently refuses to replay a reservation whose delivery state
is uncertain.  The audit reopens every live artifact and hash-chained journal;
no summary or self-consistent hash-only receipt can qualify the route.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .config import get_settings
from .execution_policy import ExecutionPolicy
from .frontier_contract_runner import (
    AdmissionDenied,
    IntegrityError,
    _exclusive_runner_lock,
    _verify_live_artifact,
    scan_live_smoke_artifacts,
)
from .frontier_coverage_continuation import verify_orphan_closure
from .frontier_coverage_repair_executor import (
    SupplementalRun,
    _global_ledger_state,
    _run_accounting,
    _verify_budget_audit,
)
from .live_smoke import CONFIRMATION as LIVE_SMOKE_CONFIRMATION
from .live_smoke import live_smoke
from .real_dataset_runner import (
    _dataset_ledger_lock,
    append_dataset_ledger_event,
    dataset_ledger_state,
    load_dataset_ledger,
)
from .reasoning_effort_sensitivity_v4 import (
    BASELINE_SCHEMA,
    FREEZE_NONCE,
    HISTORY_SCHEMA,
    NAMESPACE,
    PREFLIGHT_SCHEMA,
    ROUTE_CONFIRMATION,
    ROUTE_PLAN_SCHEMA,
    RUNNER_ASSETS_SCHEMA,
    STUDY_PLAN_SCHEMA,
    _artifact_verifies,
    _attempt_slots,
    _decimal_text,
    _manifest,
    _sha256,
    verify_frozen,
)
from .response_envelope_route_v4 import _policy_from_manifest
from .run_journal import load_run_journal

EXECUTION_PLAN_SCHEMA = "flavourbench-reasoning-effort-route-gate-execution-plan-v1"
EXECUTION_RECEIPT_SCHEMA = "flavourbench-reasoning-effort-route-gate-execution-receipt-v1"
AUDIT_SCHEMA = "flavourbench-reasoning-effort-route-gate-audit-v1"
CLOSURE_SCHEMA = "flavourbench-reasoning-effort-route-gate-closure-v1"

EXPECTED_DIGESTS = {
    "history": "308ac12ebdf375d83337d55a98a0c5aef055f6cb9b26d74795bf09d14b80b386",
    "baseline": "1fce54a13e2f844ae7a5d6b2d6f97eee4a8f37d58520d026c25ebe31cb2970e6",
    "route_plan": "2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352",
    "study_plan": "733977cc3eac48316244adcf9beb726824505173b9fe52140cb664ad35d348c0",
    "runner_assets": "f4516e382422add2a0a68b17857e7b724090e6b49542158cc2927b6cb8be6ebf",
    "preflight": "a7396f64a4db08dc1eef8425b59eb61f21836bdc5a8c572f12748f6ee3e239f7",
}
EXPECTED_SCHEMAS = {
    "history": HISTORY_SCHEMA,
    "baseline": BASELINE_SCHEMA,
    "route_plan": ROUTE_PLAN_SCHEMA,
    "study_plan": STUDY_PLAN_SCHEMA,
    "runner_assets": RUNNER_ASSETS_SCHEMA,
    "preflight": PREFLIGHT_SCHEMA,
}
CONDITIONS = ("epicure_off", "epicure_on")
MINIMUM_FINAL_CHARACTERS = 100
MINIMUM_FINAL_WORDS = 20


class RouteGateError(RuntimeError):
    """A frozen input, execution boundary, or evidence predicate failed closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RouteGateError(f"expected a regular non-symlink file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise RouteGateError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise RouteGateError(f"{field} must be finite and non-negative")
    return parsed


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RouteGateError(f"input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RouteGateError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise RouteGateError(f"expected a JSON object: {path}")
    return value


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    root = repo_root.resolve()
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RouteGateError(f"path escapes the evaluation repository: {value}") from error
    return path


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError as error:
        raise RouteGateError(f"output is outside the evaluation repository: {path}") from error


def _write_artifact(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RouteGateError(f"content-addressed output conflict: {path}")
        return path
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _artifact_document_verifies(document: object, schema: str) -> bool:
    return _artifact_verifies(document, schema)


def _source_bundle_verifies(repo_root: Path, route_plan: Mapping[str, Any]) -> bool:
    source = route_plan.get("source_code")
    if not isinstance(source, Mapping):
        return False
    records = source.get("files")
    if not isinstance(records, list) or not records:
        return False
    observed: list[dict[str, Any]] = []
    try:
        for record in records:
            if not isinstance(record, Mapping):
                return False
            relative = str(record.get("path") or "")
            path = _repo_path(repo_root, relative)
            current = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            if current != dict(record):
                return False
            observed.append(current)
    except (OSError, RouteGateError):
        return False
    return source.get("bundle_sha256") == _sha256(observed)


def _manifest_reference(route_plan: Mapping[str, Any], repo_root: Path) -> Path:
    reference = (route_plan.get("source_artifacts") or {}).get("manifest_v29")
    if not isinstance(reference, Mapping):
        raise RouteGateError("route plan lacks its high-resource manifest binding")
    path = _repo_path(repo_root, str(reference.get("path") or ""))
    if (
        _file_sha256(path) != reference.get("file_sha256")
        or path.stat().st_size != reference.get("bytes")
    ):
        raise RouteGateError("high-resource manifest differs from the frozen binding")
    manifest = _manifest(path)
    if manifest.get("content_address", {}).get("digest") != reference.get(
        "semantic_sha256"
    ):
        raise RouteGateError("high-resource manifest semantic digest differs")
    return path


def _prior_closed_identifiers(
    route_plan: Mapping[str, Any], repo_root: Path
) -> dict[str, set[str]]:
    reference = (route_plan.get("source_artifacts") or {}).get("v4_low_closure")
    if not isinstance(reference, Mapping):
        raise RouteGateError("route plan lacks the prior v4 closure binding")
    path = _repo_path(repo_root, str(reference.get("path") or ""))
    if _file_sha256(path) != reference.get("file_sha256"):
        raise RouteGateError("prior v4 closure physical digest differs")
    closure = _regular_json(path)
    if closure.get("artifact_sha256") != reference.get("semantic_sha256"):
        raise RouteGateError("prior v4 closure semantic digest differs")
    identifiers = closure.get("closed_identifiers")
    if not isinstance(identifiers, Mapping):
        raise RouteGateError("prior v4 closure has no identifier inventory")
    return {
        key: {str(value) for value in identifiers.get(key) or [] if value}
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


def _validate_route_plan_shape(
    route_plan: Mapping[str, Any], *, repo_root: Path
) -> None:
    if not _artifact_document_verifies(route_plan, ROUTE_PLAN_SCHEMA):
        raise RouteGateError("route plan content address or schema does not verify")
    if route_plan.get("artifact_sha256") != EXPECTED_DIGESTS["route_plan"]:
        raise RouteGateError("route plan is not the authoritative frozen v4 artifact")
    if route_plan.get("execution", {}).get("confirmation") != ROUTE_CONFIRMATION:
        raise RouteGateError("route plan execution confirmation differs")
    if (
        route_plan.get("counts")
        != {
            "effort_variants": 2,
            "matched_pairs": 6,
            "models": 3,
            "quality_observations": 0,
            "response_arms": 12,
            "synthetic_arms": 0,
        }
        or route_plan.get("claim_boundary")
        != {
            "official": False,
            "quality_effect_estimable": False,
            "rank_eligible": False,
            "route_gate_only": True,
        }
    ):
        raise RouteGateError("route plan count or claim boundary differs")
    task = route_plan.get("task") or {}
    if (
        task.get("synthetic") is not False
        or task.get("quarantined") is not False
        or hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
        != task.get("prompt_sha256")
    ):
        raise RouteGateError("route task is synthetic, quarantined, or hash-mismatched")
    epicure = route_plan.get("epicure") or {}
    if (
        epicure.get("public_reconstruction_complete") is not False
        or epicure.get("rank_eligible") is not False
        or not epicure.get("release_id")
        or any(
            not _is_sha256(epicure.get(field))
            for field in ("bundle_sha256", "application_sha256", "tool_schema_sha256")
        )
    ):
        raise RouteGateError("Epicure route identity or claim boundary is malformed")
    variants = {
        str(item.get("variant_id") or ""): dict(item)
        for item in route_plan.get("variants") or []
        if isinstance(item, Mapping)
    }
    expected_variants = {
        "provider_default": {
            "variant_id": "provider_default",
            "intermediate_reasoning_effort": None,
            "final_reasoning_effort": None,
            "request_semantics": "reasoning_parameter_omitted",
        },
        "explicit_high": {
            "variant_id": "explicit_high",
            "intermediate_reasoning_effort": "high",
            "final_reasoning_effort": "high",
            "request_semantics": "reasoning_effort_explicit_high",
        },
    }
    if variants != expected_variants:
        raise RouteGateError("reasoning variants differ from absent/default and explicit high")
    models = {
        str(item.get("model_id") or ""): item
        for item in route_plan.get("models") or []
        if isinstance(item, Mapping)
    }
    if len(models) != 3:
        raise RouteGateError("route plan does not contain three exact models")
    work_items = route_plan.get("work_items")
    if not isinstance(work_items, list) or len(work_items) != 6:
        raise RouteGateError("route plan does not contain six work items")
    prior = _prior_closed_identifiers(route_plan, repo_root)
    seen: dict[str, set[str]] = {
        "route_cell_ids": set(),
        "work_item_ids": set(),
        "run_ids": set(),
        "arm_ids": set(),
        "attempt_ids": set(),
    }
    reserve = Decimal(0)
    for item in work_items:
        if not isinstance(item, Mapping):
            raise RouteGateError("route work item is malformed")
        coordinate = item.get("route_coordinate")
        if not isinstance(coordinate, Mapping):
            raise RouteGateError("route coordinate is absent")
        model_id = str(coordinate.get("model_id") or "")
        variant_id = str(coordinate.get("variant_id") or "")
        model = models.get(model_id)
        variant = variants.get(variant_id)
        if model is None or variant is None:
            raise RouteGateError("route coordinate names an unknown model or variant")
        expected_coordinate = {
            "schema_version": "flavourbench-reasoning-effort-route-coordinate-v4",
            "freeze_nonce": FREEZE_NONCE,
            "model_id": model_id,
            "canonical_model_slug": model.get("canonical_model_slug"),
            "provider_endpoint": model.get("provider_endpoint"),
            "actual_provider_name": model.get("actual_provider_name"),
            "endpoint_execution_contract_sha256": model.get(
                "endpoint_execution_contract_sha256"
            ),
            "provider_controls": model.get("provider_controls"),
            "task_id": task.get("task_id"),
            "prompt_sha256": task.get("prompt_sha256"),
            "variant_id": variant_id,
            "intermediate_reasoning_effort": variant.get(
                "intermediate_reasoning_effort"
            ),
            "final_reasoning_effort": variant.get("final_reasoning_effort"),
            "epicure_bundle_sha256": epicure.get("bundle_sha256"),
            "epicure_application_sha256": epicure.get("application_sha256"),
            "epicure_tool_schema_sha256": epicure.get("tool_schema_sha256"),
        }
        if dict(coordinate) != expected_coordinate:
            raise RouteGateError("route coordinate differs from frozen model/variant inputs")
        route_cell_id = _sha256(expected_coordinate)
        work_item_id = _sha256(
            {"route_cell_id": route_cell_id, "role": "effort-v4-gate"}
        )
        run_id = str(
            __import__("uuid").uuid5(NAMESPACE, f"{route_cell_id}:{work_item_id}")
        )
        arms = [f"{run_id}:epicure_off", f"{run_id}:epicure_on"]
        slots = _attempt_slots(run_id, route_cell_id, FREEZE_NONCE)
        exact = {
            "route_cell_id": route_cell_id,
            "work_item_id": work_item_id,
            "run_id": run_id,
            "arm_ids": arms,
            "attempt_slots": slots,
        }
        if any(item.get(field) != value for field, value in exact.items()):
            raise RouteGateError("route work identifiers do not rederive")
        collections = {
            "route_cell_ids": [route_cell_id],
            "work_item_ids": [work_item_id],
            "run_ids": [run_id],
            "arm_ids": arms,
            "attempt_ids": [str(slot["attempt_id"]) for slot in slots],
        }
        for key, values in collections.items():
            if set(values) & seen[key] or set(values) & prior.get(key, set()):
                raise RouteGateError(f"route {key} overlap a prior or sibling identifier")
            seen[key].update(values)
        if item.get("diagnostic_outputs_reused") is not False:
            raise RouteGateError("route diagnostic outputs are marked for reuse")
        reserve += _decimal(item.get("worst_case_reserve_usd"), field="pair reserve")
    if route_plan.get("execution_order") != [
        item["work_item_id"] for item in work_items
    ]:
        raise RouteGateError("execution order differs from the frozen work-item order")
    if reserve != _decimal(
        route_plan.get("budget", {}).get("route_gate_worst_case_usd"),
        field="route gate reserve",
    ):
        raise RouteGateError("work-item reserves do not sum to the route budget")
    acceptance = route_plan.get("acceptance") or {}
    required_true = {
        "all_generation_costs_reconciled",
        "all_six_pairs_required",
        "each_arm_substantive",
        "each_epicure_off_arm_has_zero_tool_calls",
        "each_epicure_on_arm_has_successful_real_tool_call",
        "explicit_high_requires_reasoning_effort_high",
        "provider_default_requires_reasoning_field_absent",
        "request_contract_reconstructed_from_journal",
        "stop_and_close_suffix_on_first_failure",
    }
    if any(acceptance.get(key) is not True for key in required_true) or any(
        acceptance.get(key) is not False
        for key in (
            "diagnostic_outputs_enter_quality_fit",
            "identity_substitution_allowed",
            "replay_permitted",
        )
    ):
        raise RouteGateError("route acceptance policy differs")
    if not _source_bundle_verifies(repo_root, route_plan):
        raise RouteGateError("current generation-source bundle differs from the freeze")
    _manifest_reference(route_plan, repo_root)


def load_authoritative_inputs(
    *,
    repo_root: Path,
    history_path: Path,
    baseline_path: Path,
    route_plan_path: Path,
    study_plan_path: Path,
    runner_assets_path: Path,
    preflight_path: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "history": history_path,
        "baseline": baseline_path,
        "route_plan": route_plan_path,
        "study_plan": study_plan_path,
        "runner_assets": runner_assets_path,
        "preflight": preflight_path,
    }
    documents = {name: _regular_json(path) for name, path in paths.items()}
    for name, document in documents.items():
        if (
            document.get("artifact_sha256") != EXPECTED_DIGESTS[name]
            or not _artifact_document_verifies(document, EXPECTED_SCHEMAS[name])
        ):
            raise RouteGateError(f"{name} is not the authoritative content-addressed artifact")
    if not verify_frozen(
        repo_root=repo_root,
        history_path=history_path,
        baseline_path=baseline_path,
        route_plan_path=route_plan_path,
        study_plan_path=study_plan_path,
        runner_assets_path=runner_assets_path,
        preflight_path=preflight_path,
    ):
        raise RouteGateError("the complete frozen reasoning-effort v4 package does not rederive")
    _validate_route_plan_shape(documents["route_plan"], repo_root=repo_root)
    return documents


def _variant_policy(
    route_plan: Mapping[str, Any], work_item: Mapping[str, Any], repo_root: Path
) -> ExecutionPolicy:
    manifest = _manifest(_manifest_reference(route_plan, repo_root))
    base = _policy_from_manifest(manifest)
    coordinate = work_item["route_coordinate"]
    policy = replace(
        base,
        intermediate_reasoning_effort=coordinate["intermediate_reasoning_effort"],
        final_reasoning_effort=coordinate["final_reasoning_effort"],
    )
    policy.validate()
    return policy


def _manifest_endpoint(
    route_plan: Mapping[str, Any], work_item: Mapping[str, Any], repo_root: Path
) -> Mapping[str, Any]:
    manifest = _manifest(_manifest_reference(route_plan, repo_root))
    model_id = work_item["route_coordinate"]["model_id"]
    matches = [
        record
        for record in manifest.get("models") or []
        if isinstance(record, Mapping)
        and (record.get("model") or {}).get("id") == model_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("endpoint"), Mapping):
        raise RouteGateError(f"manifest lacks one exact route for {model_id}")
    return matches[0]["endpoint"]


def _expected_provider_controls(
    route_plan: Mapping[str, Any], work_item: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    coordinate = work_item["route_coordinate"]
    controls = dict(coordinate["provider_controls"])
    endpoint = _manifest_endpoint(route_plan, work_item, repo_root)
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        raise RouteGateError("frozen endpoint has no price envelope")
    try:
        prompt = float(Decimal(str(pricing["prompt"])) * Decimal(1_000_000))
        completion = float(Decimal(str(pricing["completion"])) * Decimal(1_000_000))
    except (KeyError, InvalidOperation, ValueError) as error:
        raise RouteGateError("frozen endpoint price envelope is malformed") from error
    controls["max_price"] = {"prompt": prompt, "completion": completion}
    return controls


def _ledger_descriptor(ledger_path: Path) -> dict[str, Any]:
    entries = load_dataset_ledger(ledger_path)
    return {
        "path": str(ledger_path),
        "sha256": (
            _file_sha256(ledger_path)
            if ledger_path.exists()
            else hashlib.sha256(b"").hexdigest()
        ),
        "entry_count": len(entries),
        "head_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
    }


def _source_map(source_directory: Path) -> dict[str, tuple[Path, dict[str, Any], str]]:
    scan = scan_live_smoke_artifacts(source_directory)
    sources: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for exposure in scan.artifacts:
        artifact, digest = _verify_live_artifact(exposure.path)
        work_item_id = str(artifact.get("dataset_work_item_id") or "")
        if not _is_sha256(work_item_id) or work_item_id in sources:
            raise RouteGateError("route source has an absent, malformed, or duplicate work-item ID")
        sources[work_item_id] = (exposure.path, artifact, digest)
    return sources


def _budget_state(
    *,
    route_plan: Mapping[str, Any],
    budget_audit_path: Path,
    project_root: Path,
    supplemental_runs: Sequence[SupplementalRun],
    source_directory: Path,
    ledger_path: Path,
    global_ledger_path: Path,
    global_artifact_directory: Path,
    global_corrections_directory: Path | None,
    global_reconciliation_directory: Path | None,
    cap_usd: Decimal,
    admission_fraction: Decimal,
    retired_zero_reservation_closure_path: Path | None = None,
    retired_zero_reservation_audit_path: Path | None = None,
) -> dict[str, Any]:
    audit, seen = _verify_budget_audit(
        budget_audit_path,
        project_root=project_root,
        cap_usd=cap_usd,
        admission_fraction=admission_fraction,
    )
    global_active, global_blockers = _global_ledger_state(
        ledger_path=global_ledger_path,
        artifact_directory=global_artifact_directory,
        corrections_directory=global_corrections_directory,
        reconciliation_directory=global_reconciliation_directory,
    )
    seen_digests = set(seen)
    supplemental_exposure = Decimal(0)
    supplemental_actual = Decimal(0)
    supplemental_orphan = Decimal(0)
    blockers: list[Mapping[str, Any]] = list(global_blockers)
    resolved_zero_reservations: list[dict[str, Any]] = []
    retired_closure: Mapping[str, Any] | None = None
    if (
        retired_zero_reservation_closure_path is not None
        or retired_zero_reservation_audit_path is not None
    ):
        if (
            retired_zero_reservation_closure_path is None
            or retired_zero_reservation_audit_path is None
        ):
            raise RouteGateError(
                "the retired zero-reservation closure and audit must be supplied together"
            )
        matching_ledgers = [
            run.ledger_path
            for run in supplemental_runs
            if run.ledger_path.exists()
            and any(
                entry.get("closure_artifact_sha256")
                == _regular_json(retired_zero_reservation_closure_path).get(
                    "artifact_sha256"
                )
                for entry in load_dataset_ledger(run.ledger_path)
            )
        ]
        if len(matching_ledgers) != 1:
            raise RouteGateError(
                "retired zero-reservation closure is not linked from exactly one "
                "supplemental ledger"
            )
        retired_closure = verify_orphan_closure(
            closure_path=retired_zero_reservation_closure_path,
            audit_path=retired_zero_reservation_audit_path,
            ledger_path=matching_ledgers[0],
        )
    for index, run in enumerate(supplemental_runs):
        accounting = _run_accounting(run, label=f"reasoning_route_supplemental_{index + 1}")
        overlap = seen_digests & set(accounting.artifact_sha256s)
        if overlap:
            raise RouteGateError("a supplemental source is duplicated in the budget audit")
        seen_digests.update(accounting.artifact_sha256s)
        supplemental_exposure += accounting.exposure_usd
        supplemental_actual += accounting.actual_cost_usd
        supplemental_orphan += accounting.orphan_reservation_usd
        for blocker in accounting.blockers:
            if (
                retired_closure is not None
                and blocker.get("work_item_id") == retired_closure.get("work_item_id")
                and _decimal(blocker.get("reserved_usd"), field="retired reservation")
                == 0
            ):
                resolved_zero_reservations.append(
                    {
                        "work_item_id": blocker.get("work_item_id"),
                        "reservation_entry_sha256": blocker.get(
                            "reservation_entry_sha256"
                        ),
                        "closure_artifact_sha256": retired_closure.get(
                            "artifact_sha256"
                        ),
                        "reserved_usd": "0",
                        "exposure_retained_usd": "0",
                        "work_item_retired": True,
                        "safe_to_replay": False,
                    }
                )
            else:
                blockers.append(blocker)
    own = _run_accounting(
        SupplementalRun(source_directory=source_directory, ledger_path=ledger_path),
        label="reasoning_effort_route_gate_v4",
    )
    overlap = seen_digests & set(own.artifact_sha256s)
    if overlap:
        raise RouteGateError("a route-gate source is duplicated in prior budget evidence")
    blockers.extend(own.blockers)
    baseline = _decimal(audit.get("current_total_exposure_usd"), field="baseline exposure")
    current = (
        baseline
        + global_active
        + supplemental_exposure
        + supplemental_orphan
        + own.exposure_usd
        + own.orphan_reservation_usd
    )
    reservations, _ = dataset_ledger_state(load_dataset_ledger(ledger_path))
    outstanding = sum(
        (
            _decimal(item["worst_case_reserve_usd"], field="outstanding pair reserve")
            for item in route_plan["work_items"]
            if item["work_item_id"] not in reservations
        ),
        Decimal(0),
    )
    projected = current + outstanding
    ceiling = cap_usd * admission_fraction
    allowed = not blockers and projected <= ceiling and projected <= cap_usd
    return {
        "currency": "USD",
        "baseline_audit_sha256": audit["artifact_sha256"],
        "baseline_exposure_usd": _decimal_text(baseline),
        "global_active_reservation_usd": _decimal_text(global_active),
        "supplemental_actual_cost_usd": _decimal_text(supplemental_actual),
        "supplemental_conservative_exposure_usd": _decimal_text(supplemental_exposure),
        "supplemental_orphan_reservation_usd": _decimal_text(supplemental_orphan),
        "route_gate_actual_cost_usd": _decimal_text(own.actual_cost_usd),
        "route_gate_conservative_exposure_usd": _decimal_text(own.exposure_usd),
        "route_gate_orphan_reservation_usd": _decimal_text(own.orphan_reservation_usd),
        "current_total_exposure_usd": _decimal_text(current),
        "outstanding_route_gate_worst_case_usd": _decimal_text(outstanding),
        "projected_total_exposure_usd": _decimal_text(projected),
        "admission_ceiling_usd": _decimal_text(ceiling),
        "hard_cap_usd": _decimal_text(cap_usd),
        "budget_within_limits": projected <= ceiling and projected <= cap_usd,
        "admission_allowed": allowed,
        "blockers": [dict(item) for item in blockers],
        "retired_zero_reservation_resolutions": resolved_zero_reservations,
    }


def build_execution_plan(
    *,
    route_plan: Mapping[str, Any],
    budget: Mapping[str, Any],
    ledger_path: Path,
    source_directory: Path,
) -> dict[str, Any]:
    entries = load_dataset_ledger(ledger_path)
    reservations, finalizations = dataset_ledger_state(entries)
    sources = _source_map(source_directory)
    known = {str(item["work_item_id"]) for item in route_plan["work_items"]}
    if (set(reservations) | set(finalizations) | set(sources)) - known:
        raise RouteGateError("route ledger or source directory contains an unknown work item")
    decisions = []
    for item in route_plan["work_items"]:
        work_item_id = item["work_item_id"]
        finalization = finalizations.get(work_item_id)
        if finalization is not None:
            decision = (
                "skip_finalized_pass"
                if finalization.get("route_gate_pair_passed") is True
                else "stop_finalized_failure_suffix_closed"
            )
        elif work_item_id in sources and work_item_id in reservations:
            decision = "recover_source_and_audit_without_provider_call"
        elif work_item_id in reservations:
            decision = "stop_reserved_without_source_no_replay"
        elif budget.get("admission_allowed") is True:
            decision = "admit_one_pair_after_exact_reservation"
        else:
            decision = "block_before_provider_call"
        decisions.append(
            {
                "work_item_id": work_item_id,
                "model_id": item["route_coordinate"]["model_id"],
                "provider_endpoint": item["route_coordinate"]["provider_endpoint"],
                "variant_id": item["route_coordinate"]["variant_id"],
                "conditions": list(CONDITIONS),
                "worst_case_reserve_usd": item["worst_case_reserve_usd"],
                "decision": decision,
            }
        )
    return {
        "schema_version": EXECUTION_PLAN_SCHEMA,
        "record_role": "zero_call_or_sequential_reasoning_effort_route_gate_plan",
        "route_plan_sha256": route_plan["artifact_sha256"],
        "status": (
            "admissible_dry_run"
            if budget.get("admission_allowed") is True
            else "blocked_dry_run"
        ),
        "budget": dict(budget),
        "ledger": _ledger_descriptor(ledger_path),
        "source_directory": str(source_directory),
        "decisions": decisions,
        "counts": {
            "planned_pairs": 6,
            "planned_arms": 12,
            "existing_sources": len(sources),
            "finalized_pairs": len(finalizations),
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "execution": {
            "confirmation": ROUTE_CONFIRMATION,
            "single_pair_admission": True,
            "shared_frontier_budget_lock": True,
            "local_append_only_ledger_lock": True,
            "uncertain_delivery_replay": False,
            "provider_calls_made_by_plan": 0,
            "epicure_calls_made_by_plan": 0,
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "official": False,
            "rank_eligible": False,
            "quality_observations": 0,
            "enters_sensitivity_fit": False,
        },
    }


def _request_reasoning_predicate(
    request_contract: Mapping[str, Any], variant_id: str
) -> bool:
    if variant_id == "provider_default":
        return (
            request_contract.get("reasoning_field_present") is False
            and request_contract.get("reasoning") is None
        )
    if variant_id == "explicit_high":
        return (
            request_contract.get("reasoning_field_present") is True
            and request_contract.get("reasoning")
            == {"effort": "high", "exclude": True}
        )
    return False


def _journal_evidence(
    source: Mapping[str, Any], source_path: Path
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    failures: list[str] = []
    descriptor = source.get("run_journal")
    if not isinstance(descriptor, Mapping):
        return [], ["run_journal_descriptor_missing"], None
    journal_path = source_path.parent / str(descriptor.get("filename") or "")
    try:
        entries = load_run_journal(journal_path)
        physical = _file_sha256(journal_path)
    except Exception:
        return [], ["run_journal_failed_hash_chain_verification"], None
    if (
        descriptor.get("sha256") != physical
        or descriptor.get("entry_count") != len(entries)
        or descriptor.get("head_entry_sha256") != entries[-1].get("entry_sha256")
        or descriptor.get("run_id") != source.get("run_id")
        or descriptor.get("finalized") is not True
    ):
        failures.append("run_journal_descriptor_mismatch")
    attempts = [
        dict(entry.get("payload") or {})
        for entry in entries
        if entry.get("event_type") == "provider_attempt"
    ]
    traces = [
        dict(entry.get("payload") or {})
        for entry in entries
        if entry.get("event_type") == "mcp_trace"
    ]
    if attempts != list(source.get("provider_attempt_events") or []):
        failures.append("journal_provider_events_differ_from_source")
    if traces != list(source.get("mcp_trace_events") or []):
        failures.append("journal_mcp_events_differ_from_source")
    return entries, failures, physical


def _audit_pair_source(
    *,
    route_plan: Mapping[str, Any],
    work_item: Mapping[str, Any],
    source_path: Path,
    source: Mapping[str, Any],
    source_digest: str,
    repo_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    coordinate = work_item["route_coordinate"]
    policy = _variant_policy(route_plan, work_item, repo_root)
    expected_controls = _expected_provider_controls(route_plan, work_item, repo_root)
    _, journal_failures, journal_sha = _journal_evidence(source, source_path)
    failures.extend(journal_failures)
    if (
        source.get("status") != "complete"
        or source.get("errors") != {}
        or source.get("official") is not False
        or source.get("rank_eligible") is not False
        or source.get("research_result") is not False
    ):
        failures.append("source_status_or_claim_boundary_invalid")
    if (
        source.get("dataset_work_item_id") != work_item["work_item_id"]
        or source.get("dataset_task_id") != route_plan["task"]["task_id"]
        or source.get("run_id") != work_item["run_id"]
        or source.get("requested_conditions") != list(CONDITIONS)
        or source.get("prompt_sha256") != route_plan["task"]["prompt_sha256"]
    ):
        failures.append("work_task_run_or_condition_binding_mismatch")
    model_contract = source.get("model_contract") or {}
    endpoint_contract = source.get("endpoint_contract") or {}
    if (
        source.get("requested_model_id") != coordinate["model_id"]
        or source.get("requested_provider") != coordinate["provider_endpoint"]
        or model_contract.get("canonical_slug") != coordinate["canonical_model_slug"]
        or endpoint_contract.get("provider_name") != coordinate["actual_provider_name"]
        or endpoint_contract.get("tag") != coordinate["provider_endpoint"]
        or source.get("endpoint_execution_contract_sha256")
        != coordinate["endpoint_execution_contract_sha256"]
    ):
        failures.append("fixed_route_or_catalog_identity_mismatch")
    if (
        source.get("provider_routing_controls") != expected_controls
        or source.get("provider_routing_controls_sha256") != _sha256(expected_controls)
    ):
        failures.append("fixed_provider_controls_mismatch")
    if (
        source.get("execution_policy_sha256") != policy.sha256
        or source.get("execution_policy") != policy.document()
        or source.get("frozen_generation_contract", {}).get(
            "intermediate_reasoning_effort"
        )
        != coordinate["intermediate_reasoning_effort"]
        or source.get("frozen_generation_contract", {}).get("final_reasoning_effort")
        != coordinate["final_reasoning_effort"]
    ):
        failures.append("variant_execution_policy_mismatch")
    epicure = source.get("epicure") or {}
    if any(
        epicure.get(field) != route_plan["epicure"][field]
        for field in ("release_id", "bundle_sha256", "application_sha256")
    ) or source.get("epicure_tool_schema_sha256") != route_plan["epicure"][
        "tool_schema_sha256"
    ]:
        failures.append("epicure_runtime_identity_mismatch")
    events = [
        event
        for event in source.get("provider_attempt_events") or []
        if isinstance(event, Mapping)
    ]
    starts = [event for event in events if event.get("event_type") == "request_started"]
    responses = [event for event in events if event.get("event_type") == "response_received"]
    rejections = [event for event in events if event.get("event_type") == "request_rejected"]
    if any(
        event.get("event_type") in {"uncertain_delivery", "invalid_response"}
        for event in events
    ):
        failures.append("unsafe_or_uncertain_provider_event")
    planned_slots = {
        (
            str(slot["arm_id"]),
            str(slot["phase"]),
            int(slot["attempt_index"]),
        ): str(slot["attempt_id"])
        for slot in work_item["attempt_slots"]
    }
    external_starts = [
        event
        for event in events
        if event.get("event_type")
        in {"request_started", "mcp_session_started", "mcp_call_started"}
    ]
    attempt_ids = [str(event.get("attempt_id") or "") for event in external_starts]
    for event in external_starts:
        key = (
            str(event.get("arm_id") or ""),
            str(event.get("phase") or ""),
            int(event.get("attempt_index", -1)),
        )
        if planned_slots.get(key) != event.get("attempt_id"):
            failures.append("external_attempt_outside_prefrozen_slot_pool")
    if len(attempt_ids) != len(set(attempt_ids)) or any(not value for value in attempt_ids):
        failures.append("attempt_id_absent_or_reused")
    request_keys: list[str] = []
    phase_keys: dict[tuple[str, str], set[str]] = {}
    for event in starts:
        metadata = event.get("metadata")
        request = metadata.get("request_contract") if isinstance(metadata, Mapping) else None
        if (
            not isinstance(request, Mapping)
            or metadata.get("request_contract_sha256") != _sha256(request)
        ):
            failures.append("request_semantics_projection_missing_or_invalid")
            continue
        if request.get("model") != coordinate["model_id"]:
            failures.append("request_model_differs_from_frozen_route")
        if request.get("provider") != expected_controls:
            failures.append("request_provider_controls_differ_from_frozen_route")
        if not _request_reasoning_predicate(request, coordinate["variant_id"]):
            failures.append(f"{coordinate['variant_id']}_reasoning_request_semantics_failed")
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
        request_key = str(event.get("request_key_sha256") or "")
        request_keys.append(request_key)
        phase_keys.setdefault((arm_id, phase), set()).add(request_key)
        if not _is_sha256(request_key):
            failures.append("request_key_is_not_sha256")
    if any(len(values) != 1 for values in phase_keys.values()):
        failures.append("retry_chain_changed_idempotency_key")
    if len(set(request_keys)) != len(phase_keys):
        failures.append("request_key_reused_across_arm_or_phase")
    start_ids = {str(event.get("attempt_id") or "") for event in starts}
    terminal_ids = {
        str(event.get("attempt_id") or "") for event in [*responses, *rejections]
    }
    if start_ids != terminal_ids:
        failures.append("provider_request_terminal_event_bijection_failed")
    if any(
        not isinstance(event.get("metadata"), Mapping)
        or event["metadata"].get("response_envelope", {}).get(
            "accepted_chat_completion"
        )
        is not True
        or event["metadata"].get("response_envelope", {}).get("classification")
        != "chat_completions"
        or str(event["metadata"].get("openrouter_cache_status") or "").upper()
        == "HIT"
        or str(event["metadata"].get("cloudflare_cache_status") or "").upper()
        == "HIT"
        for event in responses
    ):
        failures.append("accepted_response_envelope_or_cache_attestation_invalid")
    accepted_generation_ids = [
        str(event.get("generation_id") or "") for event in responses
    ]
    results = source.get("results") or {}
    if set(results) != set(CONDITIONS):
        failures.append("exact_two_arm_results_missing")
    observed_generation_ids: list[str] = []
    metadata_records: list[Mapping[str, Any]] = []
    substantive: dict[str, Any] = {}
    successful_tools = 0
    for condition in CONDITIONS:
        result = results.get(condition)
        if not isinstance(result, Mapping):
            continue
        if (
            result.get("actual_model_id") != coordinate["canonical_model_slug"]
            or result.get("actual_provider") != coordinate["actual_provider_name"]
            or result.get("finish_reason") not in {"stop", "end_turn"}
            or result.get("final_response_mode") != "plain_text"
            or result.get("cost_reconciled") is not True
        ):
            failures.append(f"{condition}_identity_finish_or_accounting_invalid")
        answer = str(result.get("answer_markdown") or "").strip()
        words = len(answer.split())
        passed_substantive = (
            len(answer) >= MINIMUM_FINAL_CHARACTERS and words >= MINIMUM_FINAL_WORDS
        )
        substantive[condition] = {
            "characters": len(answer),
            "words": words,
            "passed": passed_substantive,
        }
        if not passed_substantive:
            failures.append(f"{condition}_final_answer_not_substantive")
        if any(
            isinstance(item, Mapping) and item.get("truncated") is True
            for item in result.get("intermediate_outputs") or []
        ):
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
        observed_generation_ids.extend(
            str(value) for value in result.get("generation_ids") or []
        )
        metadata_records.extend(
            value
            for value in result.get("generation_metadata") or []
            if isinstance(value, Mapping)
        )
    if (
        not observed_generation_ids
        or len(observed_generation_ids) != len(set(observed_generation_ids))
        or set(observed_generation_ids) != set(accepted_generation_ids)
    ):
        failures.append("accepted_generation_id_bijection_failed")
    metadata_ids = [str(item.get("generation_id") or "") for item in metadata_records]
    if (
        len(metadata_ids) != len(set(metadata_ids))
        or set(metadata_ids) != set(observed_generation_ids)
        or any(
            item.get("reconciled") is not True
            or item.get("model") != coordinate["canonical_model_slug"]
            or item.get("provider") != coordinate["actual_provider_name"]
            for item in metadata_records
        )
    ):
        failures.append("generation_metadata_cost_identity_bijection_failed")
    try:
        actual_cost_micros = sum(
            int(item.get("cost_micros") or 0) for item in metadata_records
        )
    except (TypeError, ValueError):
        actual_cost_micros = 0
        failures.append("generation_cost_is_not_integer_micros")
    source_budget = source.get("budget") or {}
    reserve = _decimal(work_item["worst_case_reserve_usd"], field="pair reserve")
    if (
        source_budget.get("all_generation_costs_reconciled") is not True
        or int(source_budget.get("actual_cost_micros") or 0) != actual_cost_micros
        or Decimal(actual_cost_micros) / Decimal(1_000_000) > reserve
        or source.get("incomplete_generation_metadata") != []
    ):
        failures.append("cost_reconciliation_or_pair_reserve_failed")
    mcp_events = [
        event for event in events if str(event.get("event_type") or "").startswith("mcp_")
    ]
    mcp_starts = {
        str(event.get("attempt_id") or "")
        for event in mcp_events
        if event.get("event_type") == "mcp_call_started"
    }
    mcp_completions = {
        str(event.get("attempt_id") or "")
        for event in mcp_events
        if event.get("event_type") == "mcp_call_completed"
    }
    if not mcp_starts or mcp_starts != mcp_completions:
        failures.append("mcp_call_start_completion_bijection_failed")
    if not any(event.get("event_type") == "mcp_session_started" for event in mcp_events):
        failures.append("mcp_session_start_missing")
    if not any(event.get("event_type") == "mcp_session_attested" for event in mcp_events):
        failures.append("mcp_session_attestation_missing")
    completed_hashes = sorted(
        str(event.get("payload_sha256") or "")
        for event in mcp_events
        if event.get("event_type") == "mcp_call_completed"
    )
    traced_hashes = sorted(
        str(event.get("result_sha256") or "")
        for event in source.get("mcp_trace_events") or []
        if isinstance(event, Mapping)
    )
    if (
        not traced_hashes
        or completed_hashes != traced_hashes
        or any(not _is_sha256(value) for value in traced_hashes)
    ):
        failures.append("mcp_result_hash_bijection_failed")
    unique_failures = sorted(set(failures))
    return {
        "work_item_id": work_item["work_item_id"],
        "route_cell_id": work_item["route_cell_id"],
        "run_id": work_item["run_id"],
        "model_id": coordinate["model_id"],
        "canonical_model_slug": coordinate["canonical_model_slug"],
        "provider_endpoint": coordinate["provider_endpoint"],
        "actual_provider_name": coordinate["actual_provider_name"],
        "variant_id": coordinate["variant_id"],
        "request_semantics": (
            "reasoning_field_absent"
            if coordinate["variant_id"] == "provider_default"
            else "reasoning_effort_explicit_high"
        ),
        "source": {
            "path": _relative(repo_root, source_path),
            "artifact_sha256": source_digest,
            "journal_sha256": journal_sha,
        },
        "decision": "passed_all_predicates" if not unique_failures else "failed",
        "failures": unique_failures,
        "counts": {
            "provider_requests": len(starts),
            "accepted_chat_completions": len(responses),
            "successful_epicure_tool_calls": successful_tools,
            "epicure_off_tool_calls": len(
                (results.get("epicure_off") or {}).get("tool_trace") or []
            ),
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "substantive_integrity": substantive,
        "identifiers": {
            "attempt_ids": sorted(attempt_ids),
            "generation_ids": sorted(observed_generation_ids),
            "request_key_sha256s": sorted(set(request_keys)),
        },
        "accounting": {
            "actual_cost_micros": actual_cost_micros,
            "actual_cost_usd": _decimal_text(
                Decimal(actual_cost_micros) / Decimal(1_000_000)
            ),
            "reserved_worst_case_usd": work_item["worst_case_reserve_usd"],
            "reconciled": "cost_reconciliation_or_pair_reserve_failed"
            not in unique_failures,
        },
    }


def _receipt_source_references(
    source_directory: Path, *, repo_root: Path
) -> list[dict[str, Any]]:
    return [
        {
            "work_item_id": work_item_id,
            "path": _relative(repo_root, path),
            "artifact_sha256": digest,
        }
        for work_item_id, (path, _, digest) in sorted(_source_map(source_directory).items())
    ]


def build_route_audit(
    *,
    route_plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_path: Path,
    ledger_path: Path,
    source_directory: Path,
    repo_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    if not _artifact_document_verifies(receipt, EXECUTION_RECEIPT_SCHEMA):
        failures.append("execution_receipt_does_not_verify")
    entries = load_dataset_ledger(ledger_path)
    reservations, finalizations = dataset_ledger_state(entries)
    sources = _source_map(source_directory)
    known = {str(item["work_item_id"]) for item in route_plan["work_items"]}
    if (set(reservations) | set(finalizations) | set(sources)) - known:
        failures.append("unknown_work_item_in_ledger_or_source_directory")
    descriptor = _ledger_descriptor(ledger_path)
    receipt_descriptor = receipt.get("ledger") or {}
    receipt_should_be_complete = len(sources) == 6 and len(finalizations) == 6
    if (
        receipt.get("route_plan_sha256") != route_plan.get("artifact_sha256")
        or receipt.get("status")
        != (
            "six_pair_sources_available"
            if receipt_should_be_complete
            else "failed_or_incomplete_closed"
        )
        or receipt.get("total_source_pairs") != len(sources)
        or receipt.get("total_finalized_pairs") != len(finalizations)
        or receipt.get("quality_observations") != 0
        or receipt.get("rank_eligible") is not False
        or receipt.get("retry_outside_prefrozen_provider_phases") is not False
        or receipt.get("uncertain_delivery_replayed") is not False
        or receipt.get("failed_suffix_reopened") is not False
        or receipt.get("source_artifacts")
        != _receipt_source_references(source_directory, repo_root=repo_root)
        or receipt_descriptor.get("path") != _relative(repo_root, ledger_path)
        or receipt_descriptor.get("sha256") != descriptor["sha256"]
        or receipt_descriptor.get("entry_count") != descriptor["entry_count"]
        or receipt_descriptor.get("head_entry_sha256")
        != descriptor["head_entry_sha256"]
    ):
        failures.append("execution_receipt_source_or_ledger_binding_mismatch")
    pair_audits: list[dict[str, Any]] = []
    all_attempts: set[str] = set()
    all_generations: set[str] = set()
    all_request_keys: set[str] = set()
    total_cost_micros = 0
    suffix_closed = False
    for item in route_plan["work_items"]:
        work_item_id = item["work_item_id"]
        source_record = sources.get(work_item_id)
        reservation = reservations.get(work_item_id)
        finalization = finalizations.get(work_item_id)
        if suffix_closed:
            if source_record is not None or reservation is not None or finalization is not None:
                failures.append("work_observed_after_failed_suffix_boundary")
            continue
        if source_record is None:
            failures.append(f"missing_source:{work_item_id}")
            suffix_closed = True
            continue
        if reservation is None or finalization is None:
            failures.append(f"source_without_complete_ledger_lifecycle:{work_item_id}")
        path, source, digest = source_record
        pair = _audit_pair_source(
            route_plan=route_plan,
            work_item=item,
            source_path=path,
            source=source,
            source_digest=digest,
            repo_root=repo_root,
        )
        pair_audits.append(pair)
        if (
            finalization is None
            or finalization.get("source_artifact_sha256") != digest
            or finalization.get("route_gate_pair_passed")
            is not (pair["decision"] == "passed_all_predicates")
        ):
            failures.append(f"ledger_finalization_differs_from_source_audit:{work_item_id}")
        identifiers = pair["identifiers"]
        for label, target in (
            ("attempt_ids", all_attempts),
            ("generation_ids", all_generations),
            ("request_key_sha256s", all_request_keys),
        ):
            values = set(identifiers[label])
            if target & values:
                failures.append(f"cross_pair_{label}_overlap")
            target.update(values)
        total_cost_micros += int(pair["accounting"]["actual_cost_micros"])
        if pair["decision"] != "passed_all_predicates":
            suffix_closed = True
    if len(pair_audits) != 6:
        failures.append("all_six_pair_sources_are_required")
    if any(pair["decision"] != "passed_all_predicates" for pair in pair_audits):
        failures.append("one_or_more_pair_predicates_failed")
    planned_attempt_ids = {
        str(slot["attempt_id"])
        for item in route_plan["work_items"]
        for slot in item["attempt_slots"]
    }
    if not all_attempts <= planned_attempt_ids:
        failures.append("observed_attempt_outside_complete_frozen_pool")
    prior = _prior_closed_identifiers(route_plan, repo_root)
    if all_attempts & prior["attempt_ids"]:
        failures.append("attempt_id_replays_prior_route")
    if all_generations & prior["generation_ids"]:
        failures.append("generation_id_replays_prior_route")
    if all_request_keys & prior["request_key_sha256s"]:
        failures.append("request_key_replays_prior_route")
    unique_failures = sorted(set(failures))
    passed = not unique_failures
    return {
        "schema_version": AUDIT_SCHEMA,
        "record_role": "source_reconstructed_reasoning_effort_default_high_route_gate",
        "route_plan_sha256": route_plan["artifact_sha256"],
        "execution_receipt": {
            "path": _relative(repo_root, receipt_path),
            "artifact_sha256": receipt.get("artifact_sha256"),
        },
        "ledger": {
            "path": _relative(repo_root, ledger_path),
            "sha256": descriptor["sha256"],
            "entry_count": descriptor["entry_count"],
            "head_entry_sha256": descriptor["head_entry_sha256"],
        },
        "decision": "passed_all_predicates" if passed else "failed_one_or_more_predicates",
        "failures": unique_failures,
        "pair_audits": pair_audits,
        "counts": {
            "attempted_pairs": len(pair_audits),
            "usable_pairs": 6 if passed else 0,
            "intended_arms": 12,
            "usable_arms": 12 if passed else 0,
            "provider_requests": sum(
                pair["counts"]["provider_requests"] for pair in pair_audits
            ),
            "accepted_chat_completions": sum(
                pair["counts"]["accepted_chat_completions"] for pair in pair_audits
            ),
            "successful_epicure_tool_calls": sum(
                pair["counts"]["successful_epicure_tool_calls"]
                for pair in pair_audits
            ),
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "identifier_audit": {
            "planned_attempt_ids": sorted(planned_attempt_ids),
            "observed_attempt_ids": sorted(all_attempts),
            "observed_generation_ids": sorted(all_generations),
            "observed_request_key_sha256s": sorted(all_request_keys),
            "attempt_id_prior_overlap": sorted(all_attempts & prior["attempt_ids"]),
            "generation_id_prior_overlap": sorted(all_generations & prior["generation_ids"]),
            "request_key_prior_overlap": sorted(
                all_request_keys & prior["request_key_sha256s"]
            ),
        },
        "accounting": {
            "actual_cost_micros": total_cost_micros,
            "actual_cost_usd": _decimal_text(
                Decimal(total_cost_micros) / Decimal(1_000_000)
            ),
            "route_gate_worst_case_usd": route_plan["budget"][
                "route_gate_worst_case_usd"
            ],
            "all_generation_costs_reconciled": passed
            or all(
                pair["accounting"]["reconciled"] is True for pair in pair_audits
            ),
        },
        "study_admission": {
            "authorized": passed,
            "scope": "materialize_a_fresh_zero_call_full_study_preflight_only",
            "full_48_pair_study_executed": False,
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "quality_observations": 0,
            "official": False,
            "rank_eligible": False,
            "enters_sensitivity_fit": False,
        },
    }


def build_closure(
    *,
    route_plan: Mapping[str, Any],
    audit: Mapping[str, Any],
    receipt_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if not _artifact_document_verifies(audit, AUDIT_SCHEMA):
        raise RouteGateError("route-gate audit does not verify")
    identifiers = audit.get("identifier_audit") or {}
    return {
        "schema_version": CLOSURE_SCHEMA,
        "record_role": "permanent_reasoning_effort_v4_route_identifier_closure",
        "route_plan_sha256": route_plan["artifact_sha256"],
        "route_gate_audit_sha256": audit["artifact_sha256"],
        "execution_receipt": {
            "path": _relative(repo_root, receipt_path),
            "artifact_sha256": audit["execution_receipt"]["artifact_sha256"],
        },
        "closed_identifiers": {
            "route_cell_ids": sorted(
                str(item["route_cell_id"]) for item in route_plan["work_items"]
            ),
            "work_item_ids": sorted(
                str(item["work_item_id"]) for item in route_plan["work_items"]
            ),
            "run_ids": sorted(str(item["run_id"]) for item in route_plan["work_items"]),
            "arm_ids": sorted(
                str(value)
                for item in route_plan["work_items"]
                for value in item["arm_ids"]
            ),
            "attempt_ids": sorted(identifiers.get("planned_attempt_ids") or []),
            "used_attempt_ids": sorted(identifiers.get("observed_attempt_ids") or []),
            "unused_attempt_ids": sorted(
                set(identifiers.get("planned_attempt_ids") or [])
                - set(identifiers.get("observed_attempt_ids") or [])
            ),
            "generation_ids": sorted(identifiers.get("observed_generation_ids") or []),
            "request_key_sha256s": sorted(
                identifiers.get("observed_request_key_sha256s") or []
            ),
            "replay_permitted": False,
        },
        "decision": {
            "route_gate_qualified": audit.get("decision") == "passed_all_predicates",
            "full_study_zero_call_preflight_permitted": audit.get(
                "study_admission", {}
            ).get("authorized")
            is True,
            "full_48_pair_study_execution_performed": False,
            "failed_or_unattempted_suffix_closed": True,
        },
        "cost": audit.get("accounting"),
        "claim_boundary": audit.get("claim_boundary"),
    }


def verify_closure(
    closure: object, *, route_plan: Mapping[str, Any], audit: Mapping[str, Any]
) -> bool:
    if not _artifact_document_verifies(closure, CLOSURE_SCHEMA):
        return False
    assert isinstance(closure, Mapping)
    identifiers = audit.get("identifier_audit") or {}
    expected_identifiers = {
        "route_cell_ids": sorted(
            str(item["route_cell_id"]) for item in route_plan["work_items"]
        ),
        "work_item_ids": sorted(
            str(item["work_item_id"]) for item in route_plan["work_items"]
        ),
        "run_ids": sorted(str(item["run_id"]) for item in route_plan["work_items"]),
        "arm_ids": sorted(
            str(value)
            for item in route_plan["work_items"]
            for value in item["arm_ids"]
        ),
        "attempt_ids": sorted(identifiers.get("planned_attempt_ids") or []),
        "used_attempt_ids": sorted(identifiers.get("observed_attempt_ids") or []),
        "unused_attempt_ids": sorted(
            set(identifiers.get("planned_attempt_ids") or [])
            - set(identifiers.get("observed_attempt_ids") or [])
        ),
        "generation_ids": sorted(identifiers.get("observed_generation_ids") or []),
        "request_key_sha256s": sorted(
            identifiers.get("observed_request_key_sha256s") or []
        ),
        "replay_permitted": False,
    }
    expected_decision = {
        "route_gate_qualified": audit.get("decision") == "passed_all_predicates",
        "full_study_zero_call_preflight_permitted": audit.get(
            "study_admission", {}
        ).get("authorized")
        is True,
        "full_48_pair_study_execution_performed": False,
        "failed_or_unattempted_suffix_closed": True,
    }
    return bool(
        closure.get("route_plan_sha256") == route_plan.get("artifact_sha256")
        and closure.get("route_gate_audit_sha256") == audit.get("artifact_sha256")
        and closure.get("closed_identifiers") == expected_identifiers
        and closure.get("decision") == expected_decision
        and closure.get("cost") == audit.get("accounting")
        and closure.get("claim_boundary") == audit.get("claim_boundary")
    )


def _require_live_environment_before_reservation() -> None:
    """Reject missing authority or credentials before the ledger is mutated."""

    settings = get_settings()
    missing: list[str] = []
    if settings.execution_mode != "live":
        missing.append("FLAVOURBENCH_EXECUTION_MODE=live")
    if not settings.live_authorized:
        missing.append("FLAVOURBENCH_LIVE_AUTHORIZED=true")
    if not settings.openrouter_api_key:
        missing.append("FLAVOURBENCH_OPENROUTER_API_KEY")
    if not settings.mcp_token:
        missing.append("FLAVOURBENCH_MCP_TOKEN")
    if (
        "gateway.ai.cloudflare.com" in settings.openrouter_base_url
        and not settings.cloudflare_ai_gateway_token
    ):
        missing.append("FLAVOURBENCH_CLOUDFLARE_AI_GATEWAY_TOKEN")
    if missing:
        raise AdmissionDenied(
            "reasoning-effort route gate failed its pre-reservation environment gate: "
            + ", ".join(missing)
        )


@contextmanager
def _policy_environment(
    *,
    policy: ExecutionPolicy,
    endpoint: Mapping[str, Any],
) -> Iterable[None]:
    pricing = endpoint.get("pricing") or {}
    updates = {
        **policy.settings_environment(),
        "FLAVOURBENCH_OPENROUTER_MAX_PROMPT_PRICE_PER_MTOK": _decimal_text(
            _decimal(pricing.get("prompt"), field="endpoint prompt price")
            * Decimal(1_000_000)
        ),
        "FLAVOURBENCH_OPENROUTER_MAX_COMPLETION_PRICE_PER_MTOK": _decimal_text(
            _decimal(pricing.get("completion"), field="endpoint completion price")
            * Decimal(1_000_000)
        ),
        "FLAVOURBENCH_OPENROUTER_ZDR": "false",
    }
    before = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        get_settings.cache_clear()
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


def _live_args(
    *,
    route_plan: Mapping[str, Any],
    work_item: Mapping[str, Any],
    repo_root: Path,
    source_directory: Path,
) -> argparse.Namespace:
    coordinate = work_item["route_coordinate"]
    policy = _variant_policy(route_plan, work_item, repo_root)
    manifest_reference = route_plan["source_artifacts"]["manifest_v29"]
    return argparse.Namespace(
        confirm=LIVE_SMOKE_CONFIRMATION,
        cap_usd=_decimal(work_item["worst_case_reserve_usd"], field="pair cap"),
        model_id=coordinate["model_id"],
        provider_slug=coordinate["provider_endpoint"],
        prompt=route_plan["task"]["prompt"],
        category=route_plan["task"]["family"],
        skip_tool_contract=True,
        contract_only=False,
        condition=None,
        plain_text_final=True,
        tool_catalog_bytes_bound=policy.tool_catalog_bytes_bound,
        require_epicure_call=True,
        evidence_protocol=policy.evidence_protocol,
        intermediate_reasoning_effort=coordinate["intermediate_reasoning_effort"],
        final_reasoning_effort=coordinate["final_reasoning_effort"],
        output_dir=str(source_directory),
        candidate_manifest_sha256=manifest_reference["semantic_sha256"],
        sequential_arms=True,
        dataset_work_item_id=work_item["work_item_id"],
        dataset_task_id=route_plan["task"]["task_id"],
        expected_canonical_model_slug=coordinate["canonical_model_slug"],
        expected_endpoint_execution_sha256=coordinate[
            "endpoint_execution_contract_sha256"
        ],
        expected_execution_policy_sha256=policy.sha256,
        expected_epicure_release_id=route_plan["epicure"]["release_id"],
        expected_epicure_bundle_sha256=route_plan["epicure"]["bundle_sha256"],
        expected_epicure_application_sha256=route_plan["epicure"][
            "application_sha256"
        ],
        expected_epicure_tool_schema_sha256=route_plan["epicure"][
            "tool_schema_sha256"
        ],
        frozen_run_id=work_item["run_id"],
        frozen_attempt_slots=work_item["attempt_slots"],
    )


def _reservation_event(
    *, route_plan: Mapping[str, Any], work_item: Mapping[str, Any], budget: Mapping[str, Any]
) -> dict[str, Any]:
    coordinate = work_item["route_coordinate"]
    return {
        "event_type": "reservation_created",
        "runner_run_id": "reasoning-effort-v4-route-gate",
        "work_item_id": work_item["work_item_id"],
        "route_plan_sha256": route_plan["artifact_sha256"],
        "route_cell_id": work_item["route_cell_id"],
        "run_id": work_item["run_id"],
        "arm_ids": list(work_item["arm_ids"]),
        "model_id": coordinate["model_id"],
        "canonical_model_slug": coordinate["canonical_model_slug"],
        "provider_endpoint": coordinate["provider_endpoint"],
        "actual_provider_name": coordinate["actual_provider_name"],
        "endpoint_execution_contract_sha256": coordinate[
            "endpoint_execution_contract_sha256"
        ],
        "variant_id": coordinate["variant_id"],
        "intermediate_reasoning_effort": coordinate["intermediate_reasoning_effort"],
        "final_reasoning_effort": coordinate["final_reasoning_effort"],
        "conditions": list(CONDITIONS),
        "reserved_usd": work_item["worst_case_reserve_usd"],
        "total_exposure_before_usd": budget["current_total_exposure_usd"],
        "projected_all_remaining_usd": budget["projected_total_exposure_usd"],
        "replay_permitted": False,
        "quality_observations": 0,
        "rank_eligible": False,
    }


async def _execute_locked(
    *,
    route_plan: Mapping[str, Any],
    repo_root: Path,
    budget_audit_path: Path,
    supplemental_runs: Sequence[SupplementalRun],
    source_directory: Path,
    ledger_path: Path,
    global_ledger_path: Path,
    global_artifact_directory: Path,
    global_corrections_directory: Path | None,
    global_reconciliation_directory: Path | None,
    cap_usd: Decimal,
    admission_fraction: Decimal,
    retired_zero_reservation_closure_path: Path | None,
    retired_zero_reservation_audit_path: Path | None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    new_invocations = 0
    source_directory.mkdir(parents=True, exist_ok=True)
    for work_item in route_plan["work_items"]:
        entries = load_dataset_ledger(ledger_path)
        reservations, finalizations = dataset_ledger_state(entries)
        sources = _source_map(source_directory)
        work_item_id = work_item["work_item_id"]
        finalization = finalizations.get(work_item_id)
        reservation = reservations.get(work_item_id)
        source_record = sources.get(work_item_id)
        if finalization is not None:
            passed = finalization.get("route_gate_pair_passed") is True
            outcomes.append(
                {
                    "work_item_id": work_item_id,
                    "decision": "skip_finalized_pass" if passed else "stop_finalized_failure",
                }
            )
            if not passed:
                break
            continue
        if reservation is not None:
            if source_record is None:
                outcomes.append(
                    {
                        "work_item_id": work_item_id,
                        "decision": "stop_reserved_without_source_no_replay",
                        "reservation_entry_sha256": reservation["entry_sha256"],
                    }
                )
                break
            source_path, source, digest = source_record
            pair_audit = _audit_pair_source(
                route_plan=route_plan,
                work_item=work_item,
                source_path=source_path,
                source=source,
                source_digest=digest,
                repo_root=repo_root,
            )
            finalized = append_dataset_ledger_event(
                ledger_path,
                {
                    "event_type": "source_artifact_recorded",
                    "runner_run_id": "reasoning-effort-v4-route-gate",
                    "work_item_id": work_item_id,
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "source_artifact_sha256": digest,
                    "source_path": _relative(repo_root, source_path),
                    "route_gate_pair_passed": pair_audit["decision"]
                    == "passed_all_predicates",
                    "pair_audit_sha256": _sha256(pair_audit),
                    "actual_cost_usd": pair_audit["accounting"]["actual_cost_usd"],
                    "quality_observations": 0,
                    "rank_eligible": False,
                },
            )
            passed = pair_audit["decision"] == "passed_all_predicates"
            outcomes.append(
                {
                    "work_item_id": work_item_id,
                    "decision": (
                        "recovered_source_pass_without_provider_call"
                        if passed
                        else "recovered_source_failure_suffix_closed"
                    ),
                    "source_artifact_sha256": digest,
                    "ledger_entry_sha256": finalized["entry_sha256"],
                    "failures": pair_audit["failures"],
                }
            )
            if not passed:
                break
            continue
        if source_record is not None:
            raise RouteGateError("route source exists without a prior reservation")
        budget = _budget_state(
            route_plan=route_plan,
            budget_audit_path=budget_audit_path,
            project_root=repo_root,
            supplemental_runs=supplemental_runs,
            source_directory=source_directory,
            ledger_path=ledger_path,
            global_ledger_path=global_ledger_path,
            global_artifact_directory=global_artifact_directory,
            global_corrections_directory=global_corrections_directory,
            global_reconciliation_directory=global_reconciliation_directory,
            cap_usd=cap_usd,
            admission_fraction=admission_fraction,
            retired_zero_reservation_closure_path=(
                retired_zero_reservation_closure_path
            ),
            retired_zero_reservation_audit_path=retired_zero_reservation_audit_path,
        )
        if budget["admission_allowed"] is not True:
            outcomes.append(
                {
                    "work_item_id": work_item_id,
                    "decision": "stop_shared_budget_not_admissible",
                    "blockers": budget["blockers"],
                }
            )
            break
        reservation = append_dataset_ledger_event(
            ledger_path,
            _reservation_event(route_plan=route_plan, work_item=work_item, budget=budget),
        )
        policy = _variant_policy(route_plan, work_item, repo_root)
        endpoint = _manifest_endpoint(route_plan, work_item, repo_root)
        summary: Mapping[str, Any] | None = None
        try:
            with _policy_environment(policy=policy, endpoint=endpoint):
                settings = get_settings()
                if settings.execution_mode != "live" or not settings.live_authorized:
                    raise AdmissionDenied(
                        "route execution requires live mode and explicit live authorization"
                    )
                new_invocations += 1
                summary = await live_smoke(
                    _live_args(
                        route_plan=route_plan,
                        work_item=work_item,
                        repo_root=repo_root,
                        source_directory=source_directory,
                    )
                )
        except Exception as error:
            incident = append_dataset_ledger_event(
                ledger_path,
                {
                    "event_type": "execution_incident",
                    "runner_run_id": "reasoning-effort-v4-route-gate",
                    "work_item_id": work_item_id,
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "incident": "no_verified_source_or_uncertain_delivery_no_replay",
                    "error_type": type(error).__name__,
                    "error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                    "replay_permitted": False,
                },
            )
            outcomes.append(
                {
                    "work_item_id": work_item_id,
                    "decision": "execution_incident_reservation_retained_no_replay",
                    "incident_entry_sha256": incident["entry_sha256"],
                }
            )
            break
        artifact_path = Path(str((summary or {}).get("artifact") or ""))
        if (
            not artifact_path.is_file()
            or artifact_path.is_symlink()
            or artifact_path.resolve().parent != source_directory.resolve()
        ):
            incident = append_dataset_ledger_event(
                ledger_path,
                {
                    "event_type": "execution_incident",
                    "runner_run_id": "reasoning-effort-v4-route-gate",
                    "work_item_id": work_item_id,
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "incident": "no_verifiable_artifact_reservation_retained_no_replay",
                    "summary_sha256": _sha256(summary),
                    "replay_permitted": False,
                },
            )
            outcomes.append(
                {
                    "work_item_id": work_item_id,
                    "decision": "no_artifact_reservation_retained_no_replay",
                    "incident_entry_sha256": incident["entry_sha256"],
                }
            )
            break
        source, digest = _verify_live_artifact(artifact_path)
        if source.get("dataset_work_item_id") != work_item_id:
            raise RouteGateError("returned source is not the reserved work item")
        pair_audit = _audit_pair_source(
            route_plan=route_plan,
            work_item=work_item,
            source_path=artifact_path,
            source=source,
            source_digest=digest,
            repo_root=repo_root,
        )
        passed = pair_audit["decision"] == "passed_all_predicates"
        finalization = append_dataset_ledger_event(
            ledger_path,
            {
                "event_type": "source_artifact_recorded",
                "runner_run_id": "reasoning-effort-v4-route-gate",
                "work_item_id": work_item_id,
                "reservation_entry_sha256": reservation["entry_sha256"],
                "source_artifact_sha256": digest,
                "source_path": _relative(repo_root, artifact_path),
                "route_gate_pair_passed": passed,
                "pair_audit_sha256": _sha256(pair_audit),
                "actual_cost_usd": pair_audit["accounting"]["actual_cost_usd"],
                "quality_observations": 0,
                "rank_eligible": False,
            },
        )
        outcomes.append(
            {
                "work_item_id": work_item_id,
                "decision": (
                    "source_finalized_pair_pass"
                    if passed
                    else "source_finalized_pair_failure_suffix_closed"
                ),
                "source_artifact_sha256": digest,
                "ledger_entry_sha256": finalization["entry_sha256"],
                "failures": pair_audit["failures"],
            }
        )
        if not passed:
            break
    final_budget = _budget_state(
        route_plan=route_plan,
        budget_audit_path=budget_audit_path,
        project_root=repo_root,
        supplemental_runs=supplemental_runs,
        source_directory=source_directory,
        ledger_path=ledger_path,
        global_ledger_path=global_ledger_path,
        global_artifact_directory=global_artifact_directory,
        global_corrections_directory=global_corrections_directory,
        global_reconciliation_directory=global_reconciliation_directory,
        cap_usd=cap_usd,
        admission_fraction=admission_fraction,
        retired_zero_reservation_closure_path=retired_zero_reservation_closure_path,
        retired_zero_reservation_audit_path=retired_zero_reservation_audit_path,
    )
    return outcomes, new_invocations, final_budget


def _build_receipt(
    *,
    route_plan: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    new_invocations: int,
    final_budget: Mapping[str, Any],
    ledger_path: Path,
    source_directory: Path,
    repo_root: Path,
) -> dict[str, Any]:
    entries = load_dataset_ledger(ledger_path)
    _, finalizations = dataset_ledger_state(entries)
    sources = _source_map(source_directory)
    passed = len(finalizations) == 6 and all(
        item.get("route_gate_pair_passed") is True for item in finalizations.values()
    )
    descriptor = _ledger_descriptor(ledger_path)
    return {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "record_role": "six_pair_sequential_reasoning_effort_route_gate_receipt",
        "route_plan_sha256": route_plan["artifact_sha256"],
        "status": "six_pair_sources_available" if passed else "failed_or_incomplete_closed",
        "new_pair_invocations_this_command": new_invocations,
        "total_source_pairs": len(sources),
        "total_finalized_pairs": len(finalizations),
        "source_artifacts": _receipt_source_references(
            source_directory, repo_root=repo_root
        ),
        "ledger": {
            "path": _relative(repo_root, ledger_path),
            "sha256": descriptor["sha256"],
            "entry_count": descriptor["entry_count"],
            "head_entry_sha256": descriptor["head_entry_sha256"],
        },
        "outcomes": [dict(item) for item in outcomes],
        "final_budget": dict(final_budget),
        "retry_outside_prefrozen_provider_phases": False,
        "uncertain_delivery_replayed": False,
        "failed_suffix_reopened": False,
        "quality_observations": 0,
        "rank_eligible": False,
    }


def verify_acceptance_paths(
    *,
    route_plan: Mapping[str, Any],
    receipt_path: Path,
    audit_path: Path,
    closure_path: Path,
    ledger_path: Path,
    source_directory: Path,
    repo_root: Path,
) -> bool:
    try:
        _validate_route_plan_shape(route_plan, repo_root=repo_root)
        receipt = _regular_json(receipt_path)
        audit = _regular_json(audit_path)
        closure = _regular_json(closure_path)
        expected_audit = build_route_audit(
            route_plan=route_plan,
            receipt=receipt,
            receipt_path=receipt_path,
            ledger_path=ledger_path,
            source_directory=source_directory,
            repo_root=repo_root,
        )
        expected_audit = {**expected_audit, "artifact_sha256": _sha256(expected_audit)}
        if dict(audit) != expected_audit or audit.get("decision") != "passed_all_predicates":
            return False
        expected_closure = build_closure(
            route_plan=route_plan,
            audit=audit,
            receipt_path=receipt_path,
            repo_root=repo_root,
        )
        expected_closure = {**expected_closure, "artifact_sha256": _sha256(expected_closure)}
        return dict(closure) == expected_closure
    except (OSError, ValueError, TypeError, KeyError, IntegrityError, RouteGateError):
        return False


def _add_frozen_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--study-plan", type=Path, required=True)
    parser.add_argument("--runner-assets", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget-audit", type=Path, required=True)
    parser.add_argument("--supplemental-run-root", type=Path, action="append", default=[])
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--global-budget-lock-path", type=Path, required=True)
    parser.add_argument("--global-artifact-directory", type=Path, required=True)
    parser.add_argument("--global-corrections-directory", type=Path)
    parser.add_argument("--global-reconciliation-directory", type=Path)
    parser.add_argument("--retired-zero-reservation-closure", type=Path)
    parser.add_argument("--retired-zero-reservation-audit", type=Path)
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("100"))
    parser.add_argument("--admission-fraction", type=Decimal, default=Decimal("0.85"))
    parser.add_argument("--output-directory", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    _add_frozen_arguments(plan)
    _add_run_arguments(plan)
    execute = sub.add_parser("execute")
    _add_frozen_arguments(execute)
    _add_run_arguments(execute)
    execute.add_argument("--confirm", required=True)
    audit = sub.add_parser("audit")
    _add_frozen_arguments(audit)
    audit.add_argument("--receipt", type=Path, required=True)
    audit.add_argument("--source-directory", type=Path, required=True)
    audit.add_argument("--ledger", type=Path, required=True)
    audit.add_argument("--output-directory", type=Path, required=True)
    close = sub.add_parser("close")
    _add_frozen_arguments(close)
    close.add_argument("--receipt", type=Path, required=True)
    close.add_argument("--audit", type=Path, required=True)
    close.add_argument("--output-directory", type=Path, required=True)
    verify = sub.add_parser("verify")
    _add_frozen_arguments(verify)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--audit", type=Path, required=True)
    verify.add_argument("--closure", type=Path, required=True)
    verify.add_argument("--source-directory", type=Path, required=True)
    verify.add_argument("--ledger", type=Path, required=True)
    return parser


def _supplemental(arguments: argparse.Namespace) -> list[SupplementalRun]:
    return [
        SupplementalRun(
            source_directory=root / "source",
            corrections_directory=(
                root / "corrections" if (root / "corrections").exists() else None
            ),
            ledger_path=root / "ledger.jsonl",
        )
        for root in arguments.supplemental_run_root
    ]


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    repo_root = arguments.repo_root.resolve()
    inputs = load_authoritative_inputs(
        repo_root=repo_root,
        history_path=arguments.history,
        baseline_path=arguments.baseline,
        route_plan_path=arguments.route_plan,
        study_plan_path=arguments.study_plan,
        runner_assets_path=arguments.runner_assets,
        preflight_path=arguments.preflight,
    )
    route_plan = inputs["route_plan"]
    if arguments.command in {"plan", "execute"}:
        if arguments.command == "execute" and arguments.confirm != ROUTE_CONFIRMATION:
            raise AdmissionDenied(f"execution requires --confirm {ROUTE_CONFIRMATION}")
        if arguments.command == "execute":
            _require_live_environment_before_reservation()
        with _exclusive_runner_lock(arguments.global_budget_lock_path):
            with _dataset_ledger_lock(arguments.ledger):
                budget = _budget_state(
                    route_plan=route_plan,
                    budget_audit_path=arguments.budget_audit,
                    project_root=repo_root,
                    supplemental_runs=_supplemental(arguments),
                    source_directory=arguments.source_directory,
                    ledger_path=arguments.ledger,
                    global_ledger_path=arguments.global_budget_lock_path,
                    global_artifact_directory=arguments.global_artifact_directory,
                    global_corrections_directory=arguments.global_corrections_directory,
                    global_reconciliation_directory=arguments.global_reconciliation_directory,
                    cap_usd=arguments.cap_usd,
                    admission_fraction=arguments.admission_fraction,
                    retired_zero_reservation_closure_path=(
                        arguments.retired_zero_reservation_closure
                    ),
                    retired_zero_reservation_audit_path=(
                        arguments.retired_zero_reservation_audit
                    ),
                )
                if arguments.command == "plan":
                    payload = build_execution_plan(
                        route_plan=route_plan,
                        budget=budget,
                        ledger_path=arguments.ledger,
                        source_directory=arguments.source_directory,
                    )
                    path = _write_artifact(
                        arguments.output_directory,
                        "reasoning-effort-v4-route-gate-execution-plan",
                        payload,
                    )
                else:
                    if budget["admission_allowed"] is not True:
                        raise AdmissionDenied("reasoning-effort route gate is not budget-admitted")
                    outcomes, new_invocations, final_budget = asyncio.run(
                        _execute_locked(
                            route_plan=route_plan,
                            repo_root=repo_root,
                            budget_audit_path=arguments.budget_audit,
                            supplemental_runs=_supplemental(arguments),
                            source_directory=arguments.source_directory,
                            ledger_path=arguments.ledger,
                            global_ledger_path=arguments.global_budget_lock_path,
                            global_artifact_directory=arguments.global_artifact_directory,
                            global_corrections_directory=(
                                arguments.global_corrections_directory
                            ),
                            global_reconciliation_directory=(
                                arguments.global_reconciliation_directory
                            ),
                            cap_usd=arguments.cap_usd,
                            admission_fraction=arguments.admission_fraction,
                            retired_zero_reservation_closure_path=(
                                arguments.retired_zero_reservation_closure
                            ),
                            retired_zero_reservation_audit_path=(
                                arguments.retired_zero_reservation_audit
                            ),
                        )
                    )
                    payload = _build_receipt(
                        route_plan=route_plan,
                        outcomes=outcomes,
                        new_invocations=new_invocations,
                        final_budget=final_budget,
                        ledger_path=arguments.ledger,
                        source_directory=arguments.source_directory,
                        repo_root=repo_root,
                    )
                    path = _write_artifact(
                        arguments.output_directory,
                        "reasoning-effort-v4-route-gate-execution-receipt",
                        payload,
                    )
    elif arguments.command == "audit":
        receipt = _regular_json(arguments.receipt)
        payload = build_route_audit(
            route_plan=route_plan,
            receipt=receipt,
            receipt_path=arguments.receipt,
            ledger_path=arguments.ledger,
            source_directory=arguments.source_directory,
            repo_root=repo_root,
        )
        path = _write_artifact(
            arguments.output_directory, "reasoning-effort-v4-route-gate-audit", payload
        )
    elif arguments.command == "close":
        receipt = _regular_json(arguments.receipt)
        audit = _regular_json(arguments.audit)
        if (
            audit.get("execution_receipt", {}).get("artifact_sha256")
            != receipt.get("artifact_sha256")
        ):
            raise RouteGateError("audit and receipt do not share one execution")
        payload = build_closure(
            route_plan=route_plan,
            audit=audit,
            receipt_path=arguments.receipt,
            repo_root=repo_root,
        )
        path = _write_artifact(
            arguments.output_directory, "reasoning-effort-v4-route-gate-closure", payload
        )
    else:
        passed = verify_acceptance_paths(
            route_plan=route_plan,
            receipt_path=arguments.receipt,
            audit_path=arguments.audit,
            closure_path=arguments.closure,
            ledger_path=arguments.ledger,
            source_directory=arguments.source_directory,
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
                "status": document.get("status"),
                "decision": document.get("decision"),
                "provider_calls_made_by_builder": 0,
                "epicure_calls_made_by_builder": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
