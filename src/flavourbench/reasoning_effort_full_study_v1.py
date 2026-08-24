"""Freeze and reconstruct the task-wave reasoning-effort sensitivity study.

The study uses 24 family-balanced, non-quarantined development tasks.  Every
task wave contains seven fresh matched Epicure pairs: explicit low and high on
Sonnet, Gemini, and DeepSeek, plus Gemini's omitted provider default.  Freeze,
verify, preflight, and audit operations are network-free; paid execution lives
in :mod:`flavourbench.reasoning_effort_full_study_executor_v1`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import reasoning_effort_route_gate_v5 as v5
from . import reasoning_effort_route_gate_v6 as v6
from .frontier_manifest import verify_manifest_content_address
from .reasoning_effort_source_closure_v1 import (
    SourceClosureError,
    build_source_closure,
    verify_source_closure,
)

PLAN_SCHEMA = "flavourbench-reasoning-effort-task-wave-plan-v2"
PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-task-wave-preflight-v2"
BOUND_PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-bound-admission-preflight-v2"
TASK_SELECTION_SCHEMA = "flavourbench-reasoning-effort-task-wave-selection-v2"
MANIFEST_SCHEMA = "flavourbench-reasoning-effort-task-wave-manifest-v2"
ENDPOINT_AUDIT_SCHEMA = "flavourbench-reasoning-effort-task-wave-endpoint-audit-v2"
ENDPOINT_CLOSURE_SCHEMA = "flavourbench-reasoning-effort-task-wave-endpoint-closure-v2"

FREEZE_NONCE = "reasoning-effort-task-waves-v3-2026-08-04"
SELECTION_SEED = "reasoning-effort-full-v2-24-task-wave-selection"
NAMESPACE = uuid.UUID("5b8446e0-cc27-4cf0-8a26-430783495eda")
CONFIRMATION = "RUN_REASONING_EFFORT_V2_ONE_COMPLETE_FAMILY_BLOCK"

ENDPOINTS = {
    "sonnet": {
        "model_id": "anthropic/claude-sonnet-5",
        "canonical_model_slug": "anthropic/claude-sonnet-5-20260630",
        "provider_endpoint": "anthropic",
        "actual_provider_name": "Anthropic",
        "provider_default_effort": "high",
    },
    "gemini": {
        "model_id": "google/gemini-3.6-flash",
        "canonical_model_slug": "google/gemini-3.6-flash-20260721",
        "provider_endpoint": "google-ai-studio/flex",
        "actual_provider_name": "Google AI Studio",
        "provider_default_effort": "medium",
    },
    "deepseek": {
        "model_id": "deepseek/deepseek-v4-flash-0731",
        "canonical_model_slug": "deepseek/deepseek-v4-flash-20260731",
        "provider_endpoint": "deepinfra/fp4",
        "actual_provider_name": "DeepInfra",
        "provider_default_effort": "high",
    },
}
MODEL_TO_ENDPOINT = {value["model_id"]: key for key, value in ENDPOINTS.items()}
VARIANTS = {
    "explicit_low": {
        "variant_id": "explicit_low",
        "intermediate_reasoning_effort": "low",
        "final_reasoning_effort": "low",
        "request_semantics": "reasoning_effort_explicit_low",
    },
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
VARIANTS_BY_ENDPOINT = {
    "sonnet": ("explicit_low", "explicit_high"),
    "gemini": ("explicit_low", "provider_default", "explicit_high"),
    "deepseek": ("explicit_low", "explicit_high"),
}
TASK_FAMILIES = ("substitution", "composition", "cookability", "evidence")
CURRENT_EXPOSURE_USD = Decimal("48.01944682666666666666666666")
ADMISSION_CEILING_USD = Decimal("85")
HARD_CAP_USD = Decimal("100")


class FullStudyError(RuntimeError):
    """A frozen identity, protocol, budget, or source predicate failed."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FullStudyError(f"expected a regular non-symlink file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FullStudyError(f"expected a regular non-symlink JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FullStudyError(f"expected a JSON object: {path}")
    return value


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _file_ref(repo_root: Path, path: Path) -> dict[str, Any]:
    document = _regular_json(path)
    semantic = document.get("artifact_sha256") or (document.get("content_address") or {}).get(
        "digest"
    )
    return {
        "path": _relative(repo_root, path),
        "bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
        "semantic_sha256": semantic if isinstance(semantic, str) else _file_sha256(path),
    }


def _write_artifact(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(body)
    document = {**body, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FullStudyError(f"content-addressed artifact conflict: {path}")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as out:
        temporary = Path(out.name)
        out.write(rendered)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _write_manifest(directory: Path, payload: Mapping[str, Any]) -> Path:
    body = {key: value for key, value in payload.items() if key != "content_address"}
    digest = _sha256(body)
    manifest = {
        **body,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }
    if not verify_manifest_content_address(manifest):
        raise FullStudyError("derived manifest content address failed")
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"flavourbench-reasoning-effort-task-wave-v2-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FullStudyError(f"content-addressed manifest conflict: {path}")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as out:
        temporary = Path(out.name)
        out.write(rendered)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _one(paths: Sequence[Path], label: str) -> Path:
    if len(paths) != 1:
        raise FullStudyError(f"expected one {label}; found {len(paths)}")
    return paths[0]


def _paths(repo_root: Path) -> dict[str, Path]:
    root = repo_root / "flavourbench"
    current = root / "artifacts/season1/current-quality-run"
    v6root = current / "reasoning-effort-sensitivity-v6"
    return {
        "root": root,
        "current": current,
        "base_manifest": _one(
            list((current / "manifest-v29-high-resource").glob("*.json")), "v29 manifest"
        ),
        "task_validity": root / "artifacts/season1/task-validity/development-v2/"
        "development-task-validity-v2-86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json",
        "quarantine": _one(
            list((current / "task-quarantine-v1").glob("current-frontier-task-quarantine-*.json")),
            "task quarantine",
        ),
        "v5_snapshot": _one(
            list(
                (current / "reasoning-effort-sensitivity-v5/endpoint-snapshot").glob(
                    "reasoning-effort-v5-endpoint-snapshot-*.json"
                )
            ),
            "v5 endpoint snapshot",
        ),
        "v6_receipt": _one(
            list(
                (v6root / "admission/receipts").glob("reasoning-effort-v6-rebased-receipt-*.json")
            ),
            "v6 receipt",
        ),
        "v6_audit": _one(
            list((v6root / "aggregate").glob("reasoning-effort-v6-aggregate-audit-*.json")),
            "v6 aggregate audit",
        ),
        "v6_closure": _one(
            list((v6root / "aggregate").glob("reasoning-effort-v6-aggregate-closure-*.json")),
            "v6 aggregate closure",
        ),
        "v6_bridge": _one(
            list((v6root / "bridge").glob("reasoning-effort-v6-fanout-bridge-audit-*.json")),
            "v6 fanout bridge",
        ),
    }


def _artifact_ok(document: Mapping[str, Any], schema: str) -> bool:
    digest = document.get("artifact_sha256")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return document.get("schema_version") == schema and digest == _sha256(body)


def reconstruct_current_exposure(*, repo_root: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute terminal coverage and v6 costs from source inventories."""

    from .frontier_coverage_repair_executor import SupplementalRun, _run_accounting

    current = repo_root / "flavourbench/artifacts/season1/current-quality-run"
    historical = Decimal(str(receipt["budget"]["historical_v5_total_exposure_usd"]))
    coverage_actual = Decimal(0)
    coverage_exposure = Decimal(0)
    coverage_sources = 0
    blockers: list[dict[str, Any]] = []
    for label in ("untouched-recovery", "glm-specific-replacement"):
        root = current / "frontier-coverage-recovery-v4" / label
        accounting = _run_accounting(
            SupplementalRun(source_directory=root / "source", ledger_path=root / "ledger.jsonl"),
            label=f"reasoning_wave_baseline_{label}",
        )
        coverage_actual += accounting.actual_cost_usd
        coverage_exposure += accounting.exposure_usd + accounting.orphan_reservation_usd
        coverage_sources += accounting.source_count
        blockers.extend(dict(value) for value in accounting.blockers)
    v6root = current / "reasoning-effort-sensitivity-v6/sonnet"
    v6_accounting = _run_accounting(
        SupplementalRun(source_directory=v6root / "source", ledger_path=v6root / "ledger.jsonl"),
        label="reasoning_wave_baseline_v6",
    )
    blockers.extend(dict(value) for value in v6_accounting.blockers)
    total = (
        historical
        + coverage_exposure
        + v6_accounting.exposure_usd
        + v6_accounting.orphan_reservation_usd
    )
    recorded = Decimal(str(receipt["budget"]["complete_rebased_total_exposure_usd"]))
    if (
        coverage_sources != 8
        or coverage_actual != Decimal("0.448965")
        or coverage_exposure != Decimal("0.448965")
        or v6_accounting.source_count != 2
        or v6_accounting.actual_cost_usd != Decimal("0.244312")
        or v6_accounting.exposure_usd != Decimal("0.244312")
        or v6_accounting.orphan_reservation_usd != 0
        or blockers
        or total != recorded
        or total != CURRENT_EXPOSURE_USD
    ):
        raise FullStudyError("source-backed current exposure does not rederive")
    return {
        "historical_v5_exposure_usd": _decimal_text(historical),
        "coverage_source_count": coverage_sources,
        "coverage_actual_and_exposure_usd": _decimal_text(coverage_exposure),
        "v6_source_count": v6_accounting.source_count,
        "v6_actual_and_exposure_usd": _decimal_text(v6_accounting.exposure_usd),
        "current_total_exposure_usd": _decimal_text(total),
        "blockers": [],
    }


def _model_record(manifest: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    records = [
        dict(record)
        for record in manifest.get("models") or []
        if (record.get("model") or {}).get("id") == model_id
    ]
    if len(records) != 1:
        raise FullStudyError(f"manifest lacks one route for {model_id}")
    return records[0]


def _derive_policy(base: Mapping[str, Any], variant: str):
    base_policy = v6.v6_policy(base)
    value = VARIANTS[variant]
    policy = replace(
        base_policy,
        intermediate_reasoning_effort=value["intermediate_reasoning_effort"],
        final_reasoning_effort=value["final_reasoning_effort"],
        pair_arm_scheduling="concurrent",
    )
    policy.validate()
    return policy


def _common_protocol_projection(policy: Any) -> dict[str, Any]:
    document = copy.deepcopy(policy.document())
    document.pop("content_address", None)
    document["reasoning"] = "VARIANT_ONLY"
    return document


def _derive_manifest(
    *,
    base: Mapping[str, Any],
    variant: str,
    task_selection_sha: str,
    v6_closure_sha: str,
) -> dict[str, Any]:
    if not verify_manifest_content_address(dict(base)):
        raise FullStudyError("base manifest does not verify")
    policy = _derive_policy(base, variant)
    payload = copy.deepcopy(dict(base))
    payload.pop("content_address", None)
    payload["schema_version"] = MANIFEST_SCHEMA
    payload["manifest_role"] = "fixed_route_task_wave_reasoning_sensitivity"
    payload["status"] = "frozen_not_executed"
    if variant == "provider_default":
        payload["models"] = [copy.deepcopy(_model_record(base, ENDPOINTS["gemini"]["model_id"]))]
    else:
        payload["models"] = [
            copy.deepcopy(_model_record(base, fixed["model_id"])) for fixed in ENDPOINTS.values()
        ]
    design = payload["run_design"]
    design["assignments_per_model"] = 24
    design["selected_task_count"] = 24
    design["expected_pairs"] = 24 * len(payload["models"])
    design["expected_arms"] = 2 * design["expected_pairs"]
    design["execution_policy"] = policy.document()
    design["execution_policy_sha256"] = policy.sha256
    design["task_source"] = {
        "artifact_sha256": task_selection_sha,
        "source_class": "licensed_real_human_authored_public_questions",
        "selected_tasks": 24,
        "synthetic_tasks": 0,
        "confirmatory_eligible": False,
        "rank_eligible": False,
    }
    protocol = design["generation_protocol"]
    protocol["intermediate_reasoning_effort"] = VARIANTS[variant]["intermediate_reasoning_effort"]
    protocol["final_reasoning_effort"] = VARIANTS[variant]["final_reasoning_effort"]
    protocol["provider_default_parameter_omitted"] = variant == "provider_default"
    protocol["tool_fanout_acceptance"] = {
        "catalog_tool_count": 13,
        "max_tool_calls_per_round": 13,
        "max_tool_calls_total": 13,
        "at_most_one_complete_catalog_sweep": True,
        "client_side_only": True,
    }
    payload["source"] = {
        **dict(payload.get("source") or {}),
        "base_manifest_sha256": base["content_address"]["digest"],
        "qualified_v6_closure_sha256": v6_closure_sha,
        "task_selection_sha256": task_selection_sha,
        "freeze_nonce": FREEZE_NONCE,
        "variant_id": variant,
    }
    payload["governance"] = {
        **dict(payload.get("governance") or {}),
        "manifest_class": "unranked_task_wave_reasoning_sensitivity_v2",
        "official": False,
        "rank_eligible": False,
        "ranking_use": "prohibited",
        "same_identifier_replay_permitted": False,
        "provider_substitution_permitted": False,
    }
    payload["budget"] = {
        "currency": "USD",
        "hard_cap_usd": "100",
        "admission_ceiling_usd": "85",
        "generation_spend_authorized_by_manifest": False,
        "admission_unit": "one_complete_four_task_family_balanced_block",
    }
    return payload


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
        for call_index in range(13):
            coordinates.append((on, f"mcp_tool_{round_index}_{call_index}", 0))
    return [
        {
            "arm_id": arm_id,
            "phase": phase,
            "attempt_index": attempt,
            "attempt_id": str(
                uuid.uuid5(
                    NAMESPACE,
                    f"{FREEZE_NONCE}:{route_cell_id}:{arm_id}:{phase}:{attempt}",
                )
            ),
        }
        for arm_id, phase, attempt in coordinates
    ]


def _historical_identifiers(current: Path) -> dict[str, set[str]]:
    result = {key: set() for key in ("work_item_ids", "run_ids", "arm_ids", "attempt_ids")}
    for path in sorted(current.glob("**/source/*.json")):
        try:
            source = _regular_json(path)
        except (FullStudyError, json.JSONDecodeError):
            continue
        work = str(source.get("dataset_work_item_id") or "")
        run = str(source.get("run_id") or "")
        if work:
            result["work_item_ids"].add(work)
        if run:
            result["run_ids"].add(run)
            result["arm_ids"].update(
                f"{run}:{condition}" for condition in ("epicure_off", "epicure_on")
            )
        for event in source.get("provider_attempt_events") or []:
            if isinstance(event, Mapping) and event.get("attempt_id"):
                result["attempt_ids"].add(str(event["attempt_id"]))
    for path in sorted(current.glob("**/*closure*.json")):
        try:
            closed = _regular_json(path).get("closed_identifiers") or {}
        except (FullStudyError, json.JSONDecodeError):
            continue
        for key in result:
            result[key].update(str(value) for value in closed.get(key) or [] if value)
    return result


def _endpoint_semantics(
    *,
    base: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    repo_root: Path,
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    snapshot_records = {
        str(record["endpoint_id"]): record for record in snapshot.get("records") or []
    }
    result: dict[str, dict[str, Any]] = {}
    for endpoint_id, fixed in ENDPOINTS.items():
        record = _model_record(base, fixed["model_id"])
        endpoint = record["endpoint"]
        if (
            endpoint.get("tag") != fixed["provider_endpoint"]
            or endpoint.get("provider_name") != fixed["actual_provider_name"]
            or (record.get("model") or {}).get("canonical_slug") != fixed["canonical_model_slug"]
        ):
            raise FullStudyError(f"frozen endpoint identity differs for {endpoint_id}")
        raw = v5.raw_endpoint_contract(endpoint)
        semantic = v5.semantic_endpoint_contract(raw)
        if endpoint_id in {"sonnet", "gemini"}:
            observed = snapshot_records[endpoint_id]
            semantic = dict(observed["semantic_execution_contract"])
            semantic_sha = str(observed["semantic_execution_contract_sha256"])
            source = _file_ref(repo_root, paths["v5_snapshot"])
        else:
            semantic_sha = _sha256(semantic)
            source = _file_ref(repo_root, paths["base_manifest"])
        result[endpoint_id] = {
            **fixed,
            "provider_controls": (record.get("request_policy") or {}).get("provider"),
            "semantic_execution_contract": semantic,
            "semantic_execution_contract_sha256": semantic_sha,
            "freeze_source": source,
            "fresh_catalog_attestation_before_each_atomic_family_block": True,
        }
    return result


def _selection(
    *, task_validity_path: Path, quarantine: Mapping[str, Any]
) -> tuple[list[Any], str, dict[str, Any]]:
    from .real_dataset_runner import load_development_task_inventory, select_balanced_tasks

    inventory, source = load_development_task_inventory(task_validity_path)
    quarantined = {str(record.get("task_id")) for record in quarantine.get("records") or []}
    eligible = [task for task in inventory if task.public_id not in quarantined]
    selected, registry = select_balanced_tasks(
        tasks_per_family=6, seed=SELECTION_SEED, tasks=eligible
    )
    counts = Counter(task.family for task in selected)
    if counts != Counter({family: 6 for family in TASK_FAMILIES}):
        raise FullStudyError("selected task set is not exactly six per family")
    return selected, registry, source


def _task_wave_order(tasks: Sequence[Any]) -> list[Any]:
    by_family: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        by_family[task.family].append(task)
    for family in TASK_FAMILIES:
        by_family[family].sort(
            key=lambda task: _sha256(
                {"seed": FREEZE_NONCE, "family": family, "task_id": task.public_id}
            )
        )
    family_order = sorted(
        TASK_FAMILIES, key=lambda family: _sha256({"seed": FREEZE_NONCE, "family": family})
    )
    waves: list[Any] = []
    for cycle in range(6):
        rotated = family_order[cycle % 4 :] + family_order[: cycle % 4]
        waves.extend(by_family[family][cycle] for family in rotated)
    if len(waves) != 24:
        raise FullStudyError("wave schedule is not 24 tasks")
    return waves


def canonical_task_wave_identity(
    *, tasks: Sequence[Mapping[str, Any]], waves: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the one selection/order contract shared with human evaluation."""

    selected_tasks = sorted(
        (
            {
                "task_id": str(task["task_id"]),
                "family": str(task["family"]),
                "prompt_sha256": str(task["prompt_sha256"]),
            }
            for task in tasks
        ),
        key=lambda task: (task["task_id"], task["family"], task["prompt_sha256"]),
    )
    ordered_waves = [
        {
            "wave_id": str(wave["wave_id"]),
            "task_id": str(wave["task_id"]),
            "task_family": str(wave["task_family"]),
            "prompt_sha256": str(wave["prompt_sha256"]),
        }
        for wave in waves
    ]
    selected_set_coordinates = sorted(
        (task["task_id"], task["family"], task["prompt_sha256"]) for task in selected_tasks
    )
    return {
        "schema_version": "flavourbench-reasoning-task-wave-identity-v2",
        "selected_tasks": selected_tasks,
        "selected_task_set_sha256": _sha256(selected_set_coordinates),
        "ordered_waves": ordered_waves,
        "wave_order_sha256": _sha256(ordered_waves),
    }


def verify_human_protocol_binding(
    *, plan: Mapping[str, Any], human_protocol: Mapping[str, Any]
) -> None:
    """Require exact task, prompt, selection, plan, and wave-order identity."""

    digest = human_protocol.get("artifact_sha256")
    body = {key: value for key, value in human_protocol.items() if key != "artifact_sha256"}
    if not isinstance(digest, str) or digest != _sha256(body):
        raise FullStudyError("human protocol artifact content address failed")
    identity = plan["task_wave_identity"]
    expected = {
        "study_plan_sha256": plan["artifact_sha256"],
        "task_selection_artifact_sha256": plan["source_artifacts"]["task_selection"][
            "semantic_sha256"
        ],
        "selected_task_set_sha256": identity["selected_task_set_sha256"],
        "wave_order_sha256": identity["wave_order_sha256"],
        "selected_tasks": identity["selected_tasks"],
        "ordered_waves": identity["ordered_waves"],
    }
    if human_protocol.get("reasoning_task_wave_binding") != expected:
        raise FullStudyError(
            "human protocol task IDs, prompt hashes, selected set, or wave order differ"
        )
    if (human_protocol.get("source_bindings") or {}).get("executor_study_plan", {}).get(
        "semantic_sha256"
    ) != plan["artifact_sha256"]:
        raise FullStudyError("human protocol does not bind the exact executor plan")
    plan_items = {item["work_item_id"]: item for item in plan["work_items"]}
    expected_arms = {
        (
            arm_id,
            item_id,
            item["route_coordinate"]["task_id"],
            item["route_coordinate"]["endpoint_id"],
            arm_id.rsplit(":", 1)[-1],
            item["route_coordinate"]["variant_id"],
        )
        for item_id, item in plan_items.items()
        for arm_id in item["arm_ids"]
    }
    human_arms = human_protocol.get("arm_coordinates") or []
    observed_arms = {
        (
            arm.get("executor_arm_id"),
            arm.get("executor_work_item_id"),
            arm.get("task_id"),
            arm.get("endpoint_id"),
            arm.get("condition"),
            arm.get("variant"),
        )
        for arm in human_arms
        if isinstance(arm, Mapping)
    }
    if (
        len(human_arms) != 336
        or len({arm.get("arm_coordinate_id") for arm in human_arms}) != 336
        or observed_arms != expected_arms
    ):
        raise FullStudyError("human protocol 336-arm graph differs from the executor plan")
    expected_presentations = {
        presentation["presentation_id"]: presentation
        for presentation in plan["human_evaluation"]["presentations"]
    }
    cells = human_protocol.get("comparison_cells") or []
    if (
        len(cells) != 240
        or len({cell.get("cell_id") for cell in cells}) != 240
        or {cell.get("executor_presentation_id") for cell in cells} != set(expected_presentations)
    ):
        raise FullStudyError("human protocol 240-cell graph differs from the executor plan")
    arm_by_coordinate = {
        arm["arm_coordinate_id"]: arm for arm in human_arms if isinstance(arm, Mapping)
    }
    for cell in cells:
        source = expected_presentations[str(cell["executor_presentation_id"])]
        lower = arm_by_coordinate.get(cell.get("lower_arm_coordinate_id")) or {}
        upper = arm_by_coordinate.get(cell.get("upper_arm_coordinate_id")) or {}
        if (
            cell.get("task_id") != source["task_id"]
            or cell.get("endpoint_id") != source["endpoint_id"]
            or cell.get("condition") != source["condition"]
            or cell.get("lower_variant") != source["first_variant"]
            or cell.get("upper_variant") != source["second_variant"]
            or lower.get("executor_work_item_id") != source["first_work_item_id"]
            or upper.get("executor_work_item_id") != source["second_work_item_id"]
        ):
            raise FullStudyError("human comparison cell does not rederive from its plan cell")


def _human_presentations(
    *, waves: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    item_map = {
        (
            item["route_coordinate"]["endpoint_id"],
            item["route_coordinate"]["task_id"],
            item["route_coordinate"]["variant_id"],
        ): item
        for item in items
    }
    records: list[dict[str, Any]] = []
    contrasts = {
        "sonnet": (("explicit_low", "explicit_high", "primary_low_high"),),
        "gemini": (
            ("explicit_low", "explicit_high", "primary_low_high"),
            ("explicit_low", "provider_default", "secondary_low_default"),
            ("provider_default", "explicit_high", "secondary_default_high"),
        ),
        "deepseek": (("explicit_low", "explicit_high", "primary_low_high"),),
    }
    for wave in waves:
        task_id = wave["task_id"]
        for endpoint, endpoint_contrasts in contrasts.items():
            for first, second, contrast in endpoint_contrasts:
                for condition in ("epicure_off", "epicure_on"):
                    coordinate = {
                        "schema_version": "flavourbench-reasoning-presentation-v2",
                        "wave_id": wave["wave_id"],
                        "task_id": task_id,
                        "task_family": wave["task_family"],
                        "endpoint_id": endpoint,
                        "condition": condition,
                        "contrast": contrast,
                        "first_variant": first,
                        "second_variant": second,
                    }
                    records.append(
                        {
                            **coordinate,
                            "presentation_id": _sha256(coordinate),
                            "first_work_item_id": item_map[(endpoint, task_id, first)][
                                "work_item_id"
                            ],
                            "second_work_item_id": item_map[(endpoint, task_id, second)][
                                "work_item_id"
                            ],
                        }
                    )
    # Exactly half of every endpoint/family/condition/contrast stratum has the
    # first variant on the left, fixed before any answer exists.
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["endpoint_id"],
                record["task_family"],
                record["condition"],
                record["contrast"],
            )
        ].append(record)
    final: list[dict[str, Any]] = []
    for values in grouped.values():
        values.sort(
            key=lambda value: _sha256({"side": FREEZE_NONCE, "id": value["presentation_id"]})
        )
        if len(values) != 6:
            raise FullStudyError("human-presentation stratum is not six tasks")
        for index, value in enumerate(values):
            first_left = index < 3
            final.append(
                {
                    **value,
                    "left_work_item_id": value[
                        "first_work_item_id" if first_left else "second_work_item_id"
                    ],
                    "right_work_item_id": value[
                        "second_work_item_id" if first_left else "first_work_item_id"
                    ],
                    "first_variant_on_left": first_left,
                }
            )
    return sorted(final, key=lambda value: value["presentation_id"])


def build_plan(*, repo_root: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    from .real_dataset_runner import (
        WorkItem,
        derive_pair_forecast,
        select_candidates,
    )

    paths = _paths(repo_root)
    base = _regular_json(paths["base_manifest"])
    quarantine = _regular_json(paths["quarantine"])
    snapshot = _regular_json(paths["v5_snapshot"])
    receipt = _regular_json(paths["v6_receipt"])
    v6_audit = _regular_json(paths["v6_audit"])
    v6_closure = _regular_json(paths["v6_closure"])
    bridge = _regular_json(paths["v6_bridge"])
    if (
        not verify_manifest_content_address(base)
        or receipt.get("artifact_sha256")
        != "54389db9554837178a1dd42a3cd6e639a68195bba3e479d5fdf5e713353f3152"
        or v6_audit.get("artifact_sha256")
        != "84a3e4f2102a5d1be17d819985aad663210ce6d7e73d56042aaec65c8007d2d1"
        or v6_closure.get("artifact_sha256")
        != "7b7b6b0de6e2951bc4bea8100716d32296c8117c46aafd910cab50f4256414f5"
        or bridge.get("artifact_sha256")
        != "9d389ac19fff5a57d801c3ee076f38276793413c260b47cf135601ab441a81f4"
        or v6_closure.get("decision", {}).get("route_gate_qualified") is not True
        or v6_closure.get("decision", {}).get("replay_permitted") is not False
        or v6_audit.get("decision") != "passed_all_predicates"
    ):
        raise FullStudyError("qualified v6 predecessor evidence differs")
    baseline = reconstruct_current_exposure(repo_root=repo_root, receipt=receipt)
    selected, registry, source = _selection(
        task_validity_path=paths["task_validity"], quarantine=quarantine
    )
    selected_ids = {task.public_id for task in selected}
    quarantined_ids = {str(record.get("task_id")) for record in quarantine.get("records") or []}
    if selected_ids & quarantined_ids:
        raise FullStudyError("selected tasks intersect the current quarantine")
    task_selection = {
        "schema_version": TASK_SELECTION_SCHEMA,
        "record_role": "quality_blind_family_balanced_task_wave_selection",
        "selection_seed": SELECTION_SEED,
        "source_task_validity": _file_ref(repo_root, paths["task_validity"]),
        "source_quarantine": _file_ref(repo_root, paths["quarantine"]),
        "source_registry_sha256": registry,
        "selection_method": "balanced_sha256_order_after_current_quarantine",
        "quality_observations_used": 0,
        "tasks": [
            {
                "task_id": task.public_id,
                "family": task.family,
                "prompt": task.prompt,
                "prompt_sha256": task.prompt_sha256,
                "synthetic": False,
                "quarantined": False,
            }
            for task in selected
        ],
        "counts": {
            "tasks": 24,
            "tasks_per_family": 6,
            "synthetic_tasks": 0,
            "quarantined_tasks": 0,
        },
        "claim_boundary": {"official": False, "rank_eligible": False},
    }
    task_selection_path = _write_artifact(
        output_dir / "tasks", "reasoning-effort-task-wave-selection-v2", task_selection
    )
    task_selection_document = _regular_json(task_selection_path)

    policies = {variant: _derive_policy(base, variant) for variant in VARIANTS}
    projections = {_sha256(_common_protocol_projection(policy)) for policy in policies.values()}
    if len(projections) != 1:
        raise FullStudyError("reasoning variants differ outside the reasoning field")
    required_common = {
        "max_tool_calls_per_round": 13,
        "max_tool_calls_total": 13,
        "max_tool_rounds": 3,
        "max_output_tokens": 8192,
        "max_intermediate_tokens": 8192,
        "max_provider_attempts": 2,
        "max_tool_result_bytes": 16384,
        "max_cumulative_tool_result_bytes": 65536,
        "evidence_protocol": "matched_evidence_v2",
        "final_response_mode": "plain_text",
        "pair_arm_scheduling": "concurrent",
    }
    for policy in policies.values():
        if any(getattr(policy, key) != value for key, value in required_common.items()):
            raise FullStudyError("a variant differs from the exact common protocol")

    manifests: dict[str, dict[str, Any]] = {}
    manifest_paths: dict[str, Path] = {}
    for variant in VARIANTS:
        payload = _derive_manifest(
            base=base,
            variant=variant,
            task_selection_sha=task_selection_document["artifact_sha256"],
            v6_closure_sha=v6_closure["artifact_sha256"],
        )
        path = _write_manifest(output_dir / "manifests", payload)
        manifest_paths[variant] = path
        manifests[variant] = _regular_json(path)

    endpoint_contracts = _endpoint_semantics(
        base=base, snapshot=snapshot, repo_root=repo_root, paths=paths
    )
    candidates = {candidate.model_id: candidate for candidate in select_candidates(base, ())}
    epicure = {
        "release_id": "exploratory-unmatched-1790-runtime",
        "bundle_sha256": "98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1",
        "application_sha256": "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313",
        "tool_schema_sha256": "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd",
        "public_reconstruction_complete": False,
        "official_or_rank_eligible": False,
    }

    work_items: list[dict[str, Any]] = []
    item_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for task in selected:
        for endpoint_id, fixed in ENDPOINTS.items():
            for variant in VARIANTS_BY_ENDPOINT[endpoint_id]:
                policy = policies[variant]
                manifest = manifests[variant]
                coordinate = {
                    "schema_version": "flavourbench-reasoning-effort-task-wave-coordinate-v2",
                    "freeze_nonce": FREEZE_NONCE,
                    "qualified_v6_closure_sha256": v6_closure["artifact_sha256"],
                    "task_selection_sha256": task_selection_document["artifact_sha256"],
                    "task_id": task.public_id,
                    "task_family": task.family,
                    "prompt_sha256": task.prompt_sha256,
                    "endpoint_id": endpoint_id,
                    "model_id": fixed["model_id"],
                    "canonical_model_slug": fixed["canonical_model_slug"],
                    "provider_endpoint": fixed["provider_endpoint"],
                    "actual_provider_name": fixed["actual_provider_name"],
                    "semantic_execution_contract_sha256": endpoint_contracts[endpoint_id][
                        "semantic_execution_contract_sha256"
                    ],
                    "provider_controls": endpoint_contracts[endpoint_id]["provider_controls"],
                    "variant_id": variant,
                    "intermediate_reasoning_effort": VARIANTS[variant][
                        "intermediate_reasoning_effort"
                    ],
                    "final_reasoning_effort": VARIANTS[variant]["final_reasoning_effort"],
                    "execution_policy_sha256": policy.sha256,
                    "manifest_sha256": manifest["content_address"]["digest"],
                    "epicure_bundle_sha256": epicure["bundle_sha256"],
                    "epicure_application_sha256": epicure["application_sha256"],
                    "epicure_tool_schema_sha256": epicure["tool_schema_sha256"],
                }
                route_cell_id = _sha256(coordinate)
                work_item_id = _sha256(
                    {"route_cell_id": route_cell_id, "role": "reasoning-effort-task-wave-v2"}
                )
                run_id = str(uuid.uuid5(NAMESPACE, f"{route_cell_id}:{work_item_id}"))
                candidate = candidates[fixed["model_id"]]
                forecast_item = WorkItem(
                    ordinal=1,
                    work_item_id=work_item_id,
                    manifest_sha256=manifest["content_address"]["digest"],
                    task_registry_sha256=registry,
                    task=task,
                    candidate=candidate,
                    endpoint_execution_sha256="0" * 64,
                    execution_policy_sha256=policy.sha256,
                    execution_policy=policy,
                )
                forecast = derive_pair_forecast(forecast_item, policy=policy)
                item = {
                    "route_coordinate": coordinate,
                    "route_cell_id": route_cell_id,
                    "work_item_id": work_item_id,
                    "run_id": run_id,
                    "arm_ids": [f"{run_id}:epicure_off", f"{run_id}:epicure_on"],
                    "attempt_slots": _attempt_slots(run_id, route_cell_id),
                    "task": {
                        "task_id": task.public_id,
                        "family": task.family,
                        "prompt": task.prompt,
                        "prompt_sha256": task.prompt_sha256,
                        "synthetic": False,
                        "quarantined": False,
                    },
                    "manifest": _file_ref(repo_root, manifest_paths[variant]),
                    "forecast": forecast.public_payload(),
                    "worst_case_reserve_usd": _decimal_text(forecast.forecast_usd),
                    "same_identifier_replay_permitted": False,
                }
                work_items.append(item)
                item_index[(endpoint_id, task.public_id, variant)] = item
    if len(work_items) != 168:
        raise FullStudyError("task-wave study is not 168 matched pairs")

    ordered_tasks = _task_wave_order(selected)
    waves: list[dict[str, Any]] = []
    for ordinal, task in enumerate(ordered_tasks, start=1):
        items = [
            item_index[(endpoint_id, task.public_id, variant)]
            for endpoint_id in ENDPOINTS
            for variant in VARIANTS_BY_ENDPOINT[endpoint_id]
        ]
        items.sort(
            key=lambda item: _sha256(
                {
                    "wave": FREEZE_NONCE,
                    "task_id": task.public_id,
                    "work_item_id": item["work_item_id"],
                }
            )
        )
        wave_coordinate = {
            "schema_version": "flavourbench-reasoning-effort-task-wave-v2",
            "freeze_nonce": FREEZE_NONCE,
            "wave_ordinal": ordinal,
            "task_id": task.public_id,
            "task_family": task.family,
            "prompt_sha256": task.prompt_sha256,
            "work_item_ids": [item["work_item_id"] for item in items],
        }
        reserve = sum(Decimal(item["worst_case_reserve_usd"]) for item in items)
        waves.append(
            {
                **wave_coordinate,
                "wave_id": _sha256(wave_coordinate),
                "worst_case_reserve_usd": _decimal_text(reserve),
                "response_arms": 14,
                "matched_pairs": 7,
                "individually_admissible": False,
                "admitted_only_as_member_of_family_block": True,
            }
        )
    family_prefix = Counter()
    for index, wave in enumerate(waves, start=1):
        family_prefix[wave["task_family"]] += 1
        if index % 4 == 0 and len(set(family_prefix.values())) != 1:
            raise FullStudyError("each four-wave prefix is not family balanced")
    admission_blocks: list[dict[str, Any]] = []
    for block_index in range(6):
        block_waves = waves[block_index * 4 : (block_index + 1) * 4]
        if Counter(wave["task_family"] for wave in block_waves) != Counter(
            {family: 1 for family in TASK_FAMILIES}
        ):
            raise FullStudyError("admission block is not one task per family")
        coordinate = {
            "schema_version": "flavourbench-reasoning-effort-family-block-v2",
            "freeze_nonce": FREEZE_NONCE,
            "block_ordinal": block_index + 1,
            "wave_ids": [wave["wave_id"] for wave in block_waves],
            "task_ids": [wave["task_id"] for wave in block_waves],
            "work_item_ids": [item_id for wave in block_waves for item_id in wave["work_item_ids"]],
        }
        reserve = sum(Decimal(wave["worst_case_reserve_usd"]) for wave in block_waves)
        admission_blocks.append(
            {
                **coordinate,
                "admission_block_id": _sha256(coordinate),
                "task_families": [wave["task_family"] for wave in block_waves],
                "worst_case_reserve_usd": _decimal_text(reserve),
                "tasks": 4,
                "matched_pairs": 28,
                "response_arms": 56,
                "atomic_admission": True,
                "partial_block_start_permitted": False,
            }
        )

    historical = _historical_identifiers(paths["current"])
    fresh = {
        "work_item_ids": [item["work_item_id"] for item in work_items],
        "run_ids": [item["run_id"] for item in work_items],
        "arm_ids": [arm for item in work_items for arm in item["arm_ids"]],
        "attempt_ids": [
            slot["attempt_id"] for item in work_items for slot in item["attempt_slots"]
        ],
    }
    for key, values in fresh.items():
        if len(values) != len(set(values)) or set(values) & historical[key]:
            raise FullStudyError(f"fresh {key} overlap historical or sibling evidence")

    presentations = _human_presentations(waves=waves, items=work_items)
    primary = [value for value in presentations if value["contrast"] == "primary_low_high"]
    secondary = [value for value in presentations if value["contrast"] != "primary_low_high"]
    if len(primary) != 144 or len(secondary) != 96:
        raise FullStudyError("human presentation counts are not 144 primary plus 96 secondary")
    repeat_ids = [
        value["presentation_id"]
        for value in sorted(
            presentations,
            key=lambda value: _sha256({"repeat": FREEZE_NONCE, "id": value["presentation_id"]}),
        )[:24]
    ]
    task_wave_identity = canonical_task_wave_identity(
        tasks=task_selection_document["tasks"], waves=waves
    )

    source_closure = build_source_closure(repo_root=repo_root)
    total_future = sum(Decimal(block["worst_case_reserve_usd"]) for block in admission_blocks)
    first_block = Decimal(admission_blocks[0]["worst_case_reserve_usd"])

    plan = {
        "schema_version": PLAN_SCHEMA,
        "record_role": "route_qualified_randomized_task_wave_reasoning_sensitivity",
        "study_id": "frontier-reasoning-effort-task-waves-v3",
        "freeze_nonce": FREEZE_NONCE,
        "status": "frozen_not_executed",
        "supersedes": {
            "artifact_sha256": ("03731cb5e509bc40ec733bc5c55ee91ad035b04e1c4adaf64684437751fb1f0c"),
            "reason": (
                "the predecessor was retired after a benchmark-pipeline parser hardcoded "
                "sequential scheduling for manifests frozen as concurrent and stopped the "
                "first block before any provider completion or Epicure call"
            ),
            "pipeline_incident_sha256": (
                "5457d837103a165cfab969ea9f50a9b640e80c4b6a73c4775225b49539942402"
            ),
            "zero_call_recovery_receipt_sha256": (
                "0a882f9003aca0fbdbb4137a24ada1dd6b827c0d80b870e5c46c466058b1da0f"
            ),
        },
        "source_artifacts": {
            "base_manifest": _file_ref(repo_root, paths["base_manifest"]),
            "task_validity": _file_ref(repo_root, paths["task_validity"]),
            "task_quarantine": _file_ref(repo_root, paths["quarantine"]),
            "task_selection": _file_ref(repo_root, task_selection_path),
            "v5_endpoint_snapshot": _file_ref(repo_root, paths["v5_snapshot"]),
            "v6_rebased_receipt": _file_ref(repo_root, paths["v6_receipt"]),
            "v6_aggregate_audit": _file_ref(repo_root, paths["v6_audit"]),
            "v6_aggregate_closure": _file_ref(repo_root, paths["v6_closure"]),
            "v6_fanout_bridge": _file_ref(repo_root, paths["v6_bridge"]),
            "derived_manifests": {
                variant: _file_ref(repo_root, path) for variant, path in manifest_paths.items()
            },
        },
        "source_code": source_closure,
        "route_admission": {
            "v6_route_gate_qualified": True,
            "v6_aggregate_closure_sha256": v6_closure["artifact_sha256"],
            "route_gate_outputs_reused_in_quality_fit": False,
            "same_identifier_replay_permitted": False,
            "fanout_policy": "13_per_round_and_13_total_for_every_new_cell",
        },
        "epicure": epicure,
        "models": endpoint_contracts,
        "variants": list(VARIANTS.values()),
        "common_protocol": {
            **required_common,
            "decoding_temperature": policies["explicit_low"].decoding_temperature,
            "decoding_top_p": policies["explicit_low"].decoding_top_p,
            "decoding_seed": policies["explicit_low"].decoding_seed,
            "tool_argument_repair_turns": policies["explicit_low"].tool_argument_repair_turns,
            "required_tool_contract_max_intermediate_tokens": policies[
                "explicit_low"
            ].required_tool_contract_max_intermediate_tokens,
            "common_projection_sha256": next(iter(projections)),
            "only_variant_specific_field": "reasoning.intermediate_effort_and_final_effort",
            "attempt_slots_per_pair": 56,
        },
        "tasks": task_selection_document["tasks"],
        "task_wave_identity": task_wave_identity,
        "work_items": sorted(work_items, key=lambda item: item["work_item_id"]),
        "task_waves": waves,
        "wave_execution_order": [wave["wave_id"] for wave in waves],
        "admission_blocks": admission_blocks,
        "block_execution_order": [block["admission_block_id"] for block in admission_blocks],
        "execution_roots": {
            "coordinator": (
                "flavourbench/artifacts/season1/current-quality-run/"
                "reasoning-effort-task-waves-v3/runs/coordinator"
            ),
            "endpoints": {
                endpoint: (
                    "flavourbench/artifacts/season1/current-quality-run/"
                    f"reasoning-effort-task-waves-v3/runs/{endpoint}"
                )
                for endpoint in ENDPOINTS
            },
        },
        "execution": {
            "module": "flavourbench.reasoning_effort_full_study_executor_v1",
            "confirmation": CONFIRMATION,
            "max_new_family_blocks_per_command": 1,
            "global_lock_held_for_complete_four_wave_family_block": True,
            "fresh_catalog_attestation_for_all_three_endpoints_before_block_reservation": True,
            "partial_family_block_admission_permitted": False,
            "provider_substitution_permitted": False,
            "same_identifier_replay_permitted": False,
            "cross_bound_human_protocol_required": True,
        },
        "design": {
            "tasks": 24,
            "tasks_per_family": 6,
            "endpoints": 3,
            "primary_low_high_matched_pairs": 144,
            "secondary_gemini_default_matched_pairs": 24,
            "total_new_matched_epicure_pairs": 168,
            "total_new_real_response_arms": 336,
            "synthetic_arms": 0,
            "pairs_per_task_wave": 7,
            "arms_per_task_wave": 14,
            "family_balanced_admission_blocks": 6,
            "task_waves_per_admission_block": 4,
            "pairs_per_admission_block": 28,
            "arms_per_admission_block": 56,
            "wave_order": "pre_randomized_family_balanced_four-wave_cycles",
            "within_wave_order": "pre_randomized_seven_endpoint_variant_cells",
        },
        "budget": {
            "currency": "USD",
            "source_reconstructed_current_exposure": baseline,
            "current_total_exposure_usd": _decimal_text(CURRENT_EXPOSURE_USD),
            "all_24_waves_worst_case_usd": _decimal_text(total_future),
            "all_24_waves_projected_usd": _decimal_text(CURRENT_EXPOSURE_USD + total_future),
            "first_block_worst_case_usd": _decimal_text(first_block),
            "first_block_admissible_now": CURRENT_EXPOSURE_USD + first_block
            <= ADMISSION_CEILING_USD,
            "admission_ceiling_usd": "85",
            "hard_cap_usd": "100",
            "all_blocks_pre_reserved": False,
            "reservation_unit": "one_complete_four_task_family_balanced_block",
            "next_block_reaudited_under_global_lock": True,
            "stop_admitting_at_85_percent": True,
            "drain_at_95_percent": True,
            "hard_stop_at_100_percent": True,
            "uncompleted_suffix_due_budget_preserves_exact_family_balance": True,
        },
        "failure_policy": {
            "source_and_ledgers_endpoint_isolated": True,
            "coordinator_family_block_ledger_append_only": True,
            "whole_four_wave_family_block_terminalized_before_next_admission": True,
            "failed_pair_replay_or_replacement_permitted": False,
            "reconciled_failed_source_retained_as_reliability_failure": True,
            "pre_generation_failure_terminalized_without_replay": True,
            "uncertain_delivery_without_source_retains_full_block_reserve_and_stops": True,
        },
        "human_evaluation": {
            "frozen_before_generation": True,
            "presentations": presentations,
            "counts": {
                "primary_low_high_presentations": 144,
                "secondary_gemini_adjacent_dose_presentations": 96,
                "total_unique_presentations": 240,
                "minimum_independent_raters_per_presentation": 2,
                "position_swapped_repeat_presentations_per_rater": 24,
            },
            "position_swapped_repeat_presentation_ids": repeat_ids,
            "blinding": {
                "prompt_visible": True,
                "model_endpoint_variant_and_epicure_condition_hidden_until_ballot_lock": True,
                "left_right_assignment_frozen": True,
                "reviewer_access_to_identity_plan_during_adjudication": False,
            },
            "rater_admission": {
                "qualified_culinary_reviewer": True,
                "independent_ballots": True,
                "minimum_raters": 2,
                "identity_or_condition_lookup_prohibited": True,
                "conflicts_and_affiliations_disclosed": True,
            },
            "workflow": {
                "task_validity_sealed_before_answers_revealed": True,
                "task_validity_choices": ["valid", "invalid", "uncertain"],
                "preference_choices": ["left", "right", "tie", "both_bad"],
                "rubric_dimensions": [
                    "task_completion",
                    "constraint_compliance",
                    "coherence",
                    "sensory_promise",
                    "cookability",
                    "clarity",
                    "originality",
                    "evidence_use",
                    "calibration",
                ],
                "comparative_rationale_required": True,
                "confidence_required": True,
                "failure_tags_optional": True,
            },
        },
        "analysis_contract": {
            "co_primary_estimands": [
                "Sonnet explicit-high minus explicit-low within task and Epicure condition",
                "Gemini explicit-high minus explicit-low within task and Epicure condition",
                "DeepSeek explicit-high minus explicit-low within task and Epicure condition",
            ],
            "pooled_default_to_high_estimand": False,
            "provider_default_semantics": {
                "sonnet": "omitted default is high; route diagnostic only",
                "gemini": "omitted default is medium; secondary dose-response level",
                "deepseek": "omitted default is high; route diagnostic only",
            },
            "secondary_estimands": [
                "Gemini provider-default/medium minus explicit-low",
                "Gemini explicit-high minus provider-default/medium",
            ],
            "quality_source": "new blinded human judgments only",
            "independent_task_clusters_planned": 24,
            "unit_of_analysis": (
                "task cluster; arms, raters, conditions, and contrasts are repeated measures"
            ),
            "primary_inference": (
                "endpoint-specific task-cluster randomization/sign analysis with "
                "simultaneous intervals"
            ),
            "multiplicity": "Holm correction across the three co-primary endpoint contrasts",
            "rater_dependence": (
                "rater and task crossed dependence retained; no ballot row treated as independent"
            ),
            "missingness": {
                "complete_case": "only waves with both compared sources and valid ballots",
                "worst_best_bounds": (
                    "assign every missing task cluster against/for the observed direction"
                ),
                "budget_truncation": (
                    "report scheduled and completed waves; never model- or cost-selected "
                    "partial waves"
                ),
                "operational_failures": "retained in reliability and never regenerated",
                "invalid_tasks": "excluded by the frozen task-validity rule and reported by family",
            },
            "minimum_reporting": [
                "scheduled waves",
                "completed waves",
                "tasks per family",
                "usable pairs",
                "failure rate",
                "cluster interval",
                "worst/best missingness bounds",
                "repeat agreement",
                "cost and latency",
            ],
            "ranking_use": "prohibited; fixed-endpoint development sensitivity only",
        },
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "enters_primary_leaderboard": False,
            "quality_result_available_before_blinded_judgment": False,
            "generalizes_to_unmeasured_models_or_all_culinary_tasks": False,
            "epicure_public_reconstruction_complete": False,
        },
        "calls_made_by_freeze": {"provider_completions": 0, "epicure": 0, "catalog_gets": 0},
    }
    return plan, manifest_paths


def validate_plan(plan: Mapping[str, Any], *, repo_root: Path) -> None:
    if not _artifact_ok(plan, PLAN_SCHEMA):
        raise FullStudyError("task-wave plan content address or schema failed")
    source_artifacts = plan.get("source_artifacts") or {}
    for reference in source_artifacts.values():
        records = (
            reference.values()
            if isinstance(reference, Mapping) and "path" not in reference
            else [reference]
        )
        for record in records:
            if not isinstance(record, Mapping) or "path" not in record:
                continue
            path = repo_root / str(record["path"])
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != record.get("bytes")
                or _file_sha256(path) != record.get("file_sha256")
            ):
                raise FullStudyError(f"frozen source artifact differs: {path}")
    try:
        verify_source_closure(expected=plan.get("source_code") or {}, repo_root=repo_root)
    except SourceClosureError as error:
        raise FullStudyError(f"source closure does not rederive: {error}") from error
    items = plan.get("work_items") or []
    waves = plan.get("task_waves") or []
    if (
        len(items) != 168
        or len(waves) != 24
        or sum(len(item.get("attempt_slots") or []) for item in items) != 168 * 56
        or sum(wave.get("matched_pairs", 0) for wave in waves) != 168
    ):
        raise FullStudyError("task-wave counts or attempt slots differ")
    item_ids = {item["work_item_id"] for item in items}
    if len(item_ids) != 168 or any(
        len(wave.get("work_item_ids") or []) != 7 or not set(wave["work_item_ids"]) <= item_ids
        for wave in waves
    ):
        raise FullStudyError("wave membership is malformed")
    blocks = plan.get("admission_blocks") or []
    flattened_wave_ids = [wave_id for block in blocks for wave_id in block.get("wave_ids") or []]
    if (
        len(blocks) != 6
        or plan.get("block_execution_order")
        != [block.get("admission_block_id") for block in blocks]
        or flattened_wave_ids != plan.get("wave_execution_order")
        or any(
            len(block.get("wave_ids") or []) != 4
            or len(block.get("work_item_ids") or []) != 28
            or Counter(block.get("task_families") or [])
            != Counter({family: 1 for family in TASK_FAMILIES})
            or block.get("matched_pairs") != 28
            or block.get("response_arms") != 56
            for block in blocks
        )
    ):
        raise FullStudyError("family-balanced admission blocks are malformed")
    presentations = plan.get("human_evaluation", {}).get("presentations") or []
    if len(presentations) != 240:
        raise FullStudyError("human presentation plan is not 240 comparisons")
    if plan.get("task_wave_identity") != canonical_task_wave_identity(
        tasks=plan.get("tasks") or [], waves=waves
    ):
        raise FullStudyError("canonical task-wave identity does not rederive")


def build_preflight(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    from .frontier_coverage_repair_executor import _global_ledger_state

    validate_plan(plan, repo_root=repo_root)
    receipt = _regular_json(repo_root / plan["source_artifacts"]["v6_rebased_receipt"]["path"])
    baseline = reconstruct_current_exposure(repo_root=repo_root, receipt=receipt)
    artifacts = repo_root / "flavourbench/artifacts"
    active, blockers = _global_ledger_state(
        ledger_path=artifacts / "frontier-contract/ledger.jsonl",
        artifact_directory=artifacts / "live-smoke",
        corrections_directory=artifacts / "corrections",
        reconciliation_directory=artifacts / "frontier-contract/reconciliations",
    )
    run_files: list[str] = []
    roots = [plan["execution_roots"]["coordinator"], *plan["execution_roots"]["endpoints"].values()]
    for relative in roots:
        root = repo_root / relative
        if root.exists():
            run_files.extend(
                _relative(repo_root, path)
                for path in root.rglob("*")
                if path.is_file() and not path.name.endswith(".lock")
            )
    first_reserve = Decimal(plan["admission_blocks"][0]["worst_case_reserve_usd"])
    budget_ready = (
        active == 0
        and not blockers
        and not run_files
        and CURRENT_EXPOSURE_USD + first_reserve <= ADMISSION_CEILING_USD
    )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "record_role": "zero_call_pre_protocol_preflight_for_family_block_sensitivity",
        "study_plan_sha256": plan["artifact_sha256"],
        "decision": "awaiting_cross_bound_human_protocol",
        "supersedes": {
            "artifact_sha256": ("cca0e9578c7b6760df8389bdacca9967b673d595b5c23147bd88348cefc8cbcf"),
            "reason": "predecessor was bound to the retired pre-generation parser-defect plan",
        },
        "checks": {
            "source_reconstructed_baseline": baseline,
            "global_active_reservation_usd": _decimal_text(active),
            "global_blockers": [dict(value) for value in blockers],
            "fresh_execution_roots": run_files == [],
            "first_family_block_reserve_usd": _decimal_text(first_reserve),
            "first_family_block_projected_usd": _decimal_text(CURRENT_EXPOSURE_USD + first_reserve),
            "first_family_block_below_85_percent_admission": CURRENT_EXPOSURE_USD + first_reserve
            <= ADMISSION_CEILING_USD,
            "budget_and_empty_roots_ready": budget_ready,
            "planned_tasks": 24,
            "planned_pairs": 168,
            "planned_real_arms": 336,
            "synthetic_arms": 0,
            "quarantined_tasks": 0,
            "human_protocol_frozen_and_cross_verified": False,
        },
        "execution": {
            "module": plan["execution"]["module"],
            "confirmation": plan["execution"]["confirmation"],
            "max_new_family_blocks_per_command": 1,
            "partial_family_block_start_permitted": False,
            "provider_or_epicure_calls_made_by_preflight": False,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def build_bound_preflight(
    *, plan: Mapping[str, Any], human_protocol: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Bind the independently frozen human graph before live admission."""

    verify_human_protocol_binding(plan=plan, human_protocol=human_protocol)
    preliminary = build_preflight(plan=plan, repo_root=repo_root)
    ready = preliminary["checks"]["budget_and_empty_roots_ready"] is True
    return {
        "schema_version": BOUND_PREFLIGHT_SCHEMA,
        "record_role": "cross_bound_zero_call_family_block_admission_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "decision": "first_family_block_admissible" if ready else "blocked_before_external_calls",
        "supersedes": {
            "artifact_sha256": ("9c5cb664b5708fccfa49e20f8c362736786e705b06444ed7c59f5013181e8d8e"),
            "reason": "predecessor was bound to the retired pre-generation parser-defect plan",
        },
        "checks": {
            **dict(preliminary["checks"]),
            "human_protocol_frozen_and_cross_verified": True,
            "human_arm_coordinates_verified": 336,
            "human_comparison_cells_verified": 240,
            "task_selection_and_prompt_hashes_identical": True,
            "wave_order_identical": True,
            "family_balanced_admission_blocks": 6,
        },
        "execution": preliminary["execution"],
        "calls_made_by_bound_preflight": {
            "provider_completions": 0,
            "epicure": 0,
            "catalog_gets": 0,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def verify_bound_preflight(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
) -> None:
    verify_human_protocol_binding(plan=plan, human_protocol=human_protocol)
    if (
        not _artifact_ok(bound_preflight, BOUND_PREFLIGHT_SCHEMA)
        or bound_preflight.get("study_plan_sha256") != plan["artifact_sha256"]
        or bound_preflight.get("human_protocol_sha256") != human_protocol["artifact_sha256"]
        or bound_preflight.get("decision") != "first_family_block_admissible"
        or bound_preflight.get("checks", {}).get("human_protocol_frozen_and_cross_verified")
        is not True
    ):
        raise FullStudyError("post-protocol admission preflight is absent or invalid")


def _item_plan_view(plan: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(plan),
        "task": dict(item["task"]),
        "source_artifacts": {
            **dict(plan.get("source_artifacts") or {}),
            "manifest_v29": dict(item["manifest"]),
        },
    }


def pair_audit(
    *, plan: Mapping[str, Any], item: Mapping[str, Any], source_path: Path, repo_root: Path
) -> dict[str, Any]:
    source, digest = v5._verify_live_artifact(source_path)
    raw = v5.raw_endpoint_contract(source.get("endpoint_contract") or {})
    adapted = copy.deepcopy(dict(item))
    adapted["route_coordinate"]["endpoint_execution_contract_sha256"] = _sha256(raw)
    pair = v5._adapted_pair_audit(
        plan=_item_plan_view(plan, item),
        item=adapted,
        source_path=source_path,
        source=source,
        digest=digest,
        repo_root=repo_root,
    )
    expected_semantic = item["route_coordinate"]["semantic_execution_contract_sha256"]
    observed_semantic = _sha256(v5.semantic_endpoint_contract(raw))
    failures = [
        value
        for value in pair.get("failures") or []
        if value != "source_endpoint_semantic_contract_differs_from_v5_freeze"
    ]
    if item["route_coordinate"]["variant_id"] == "explicit_low":
        failures = [
            value
            for value in failures
            if value != "explicit_low_reasoning_request_semantics_failed"
        ]
        starts = [
            event
            for event in source.get("provider_attempt_events") or []
            if isinstance(event, Mapping) and event.get("event_type") == "request_started"
        ]
        low_requests = [(event.get("metadata") or {}).get("request_contract") for event in starts]
        if not low_requests or any(
            not isinstance(request, Mapping)
            or request.get("reasoning_field_present") is not True
            or request.get("reasoning") != {"effort": "low", "exclude": True}
            for request in low_requests
        ):
            failures.append("explicit_low_reasoning_request_semantics_failed")
    if observed_semantic != expected_semantic:
        failures.append("source_endpoint_semantic_contract_differs_from_task_wave_freeze")
    on = (source.get("results") or {}).get("epicure_on") or {}
    fanout = [
        int(record["tool_call_count"])
        for record in on.get("intermediate_outputs") or []
        if isinstance(record, Mapping) and record.get("tool_call_count") is not None
    ]
    if max(fanout or [0]) > 13:
        failures.append("round_fanout_exceeds_13")
    if len(on.get("tool_trace") or []) > 13:
        failures.append("total_tool_calls_exceed_13")
    pair["failures"] = sorted(set(failures))
    pair["decision"] = "passed_all_predicates" if not pair["failures"] else "failed"
    pair["task_wave"] = {
        "observed_semantic_execution_contract_sha256": observed_semantic,
        "frozen_semantic_execution_contract_sha256": expected_semantic,
        "maximum_round_fanout": max(fanout or [0]),
        "executed_tool_calls_total": len(on.get("tool_trace") or []),
    }
    return pair


def freeze(*, repo_root: Path, output_dir: Path) -> dict[str, Path]:
    plan, _ = build_plan(repo_root=repo_root, output_dir=output_dir)
    plan_path = _write_artifact(output_dir / "plan", "reasoning-effort-task-wave-plan-v2", plan)
    plan_document = _regular_json(plan_path)
    validate_plan(plan_document, repo_root=repo_root)
    preflight = build_preflight(plan=plan_document, repo_root=repo_root)
    preflight_path = _write_artifact(
        output_dir / "preflight", "reasoning-effort-task-wave-preflight-v2", preflight
    )
    return {"plan": plan_path, "preflight": preflight_path}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "flavourbench/artifacts/season1/current-quality-run/reasoning-effort-task-waves-v3"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    verify = sub.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    binding = sub.add_parser("verify-human-protocol")
    binding.add_argument("--plan", type=Path, required=True)
    binding.add_argument("--human-protocol", type=Path, required=True)
    bound = sub.add_parser("bind-human-protocol")
    bound.add_argument("--plan", type=Path, required=True)
    bound.add_argument("--human-protocol", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = repo_root / output
    if args.command == "freeze":
        result = freeze(repo_root=repo_root, output_dir=output)
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
        return
    plan = _regular_json(args.plan)
    validate_plan(plan, repo_root=repo_root)
    if args.command in {"verify-human-protocol", "bind-human-protocol"}:
        human_protocol = _regular_json(args.human_protocol)
        verify_human_protocol_binding(plan=plan, human_protocol=human_protocol)
        if args.command == "bind-human-protocol":
            bound_preflight = build_bound_preflight(
                plan=plan, human_protocol=human_protocol, repo_root=repo_root
            )
            path = _write_artifact(
                output / "bound-preflight",
                "reasoning-effort-bound-admission-preflight-v2",
                bound_preflight,
            )
            print(json.dumps({"bound_preflight": str(path)}, indent=2))
            return
        print(
            json.dumps(
                {
                    "verified": True,
                    "plan_sha256": plan["artifact_sha256"],
                    "human_protocol_sha256": human_protocol["artifact_sha256"],
                },
                indent=2,
            )
        )
        return
    print(json.dumps({"verified": True, "plan_sha256": plan["artifact_sha256"]}, indent=2))


if __name__ == "__main__":
    run()
