"""Reconstruct terminal coverage evidence and freeze the residual bridge.

All commands in this module are offline.  They read content-addressed records,
write new content-addressed governance artifacts, and never invoke a provider
or Epicure MCP endpoint.  The residual plan is intentionally development-only:
it repairs model-pair-by-family support geometry but cannot create preference
judgments or an official ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from . import frontier_coverage_recovery_v4 as recovery_v4
from .frontier_contract_runner import IntegrityError, load_candidate_manifest, select_candidates
from .frontier_coverage_continuation import _attempt_slots
from .frontier_coverage_postrun import (
    FAMILIES,
    Arm,
    _arena_side_layout,
    _arm_from_document,
    _components,
    _uplift_identity_commitment,
    _verify_uplift_pool,
)
from .frontier_coverage_recovery_v4 import (
    QUARANTINED_TASK_IDS,
    _file_sha256,
    _load_addressed,
    _relative,
    _task_and_manifest_inputs,
    build_phase_audit,
    reconstruct_parent,
)
from .frontier_coverage_repair_executor import _decimal_text
from .frontier_multirun_assets import _verify_artifact
from .real_dataset_runner import (
    WorkItem,
    derive_conditions_forecast,
    load_development_task_inventory,
    task_registry_sha256,
)
from .real_task_bank import sha256_json

RESIDUAL_PLAN_SCHEMA_VERSION = "flavourbench-frontier-coverage-residual-v5-plan-v1"
RESIDUAL_NAMESPACE = uuid.UUID("b11cd988-e35f-4d57-8221-8917f2bf9463")
PRIMARY_PLAN_SCHEMA_VERSION = "flavourbench-frontier-coverage-primary-on-v5-plan-v2"
PRIMARY_NAMESPACE = uuid.UUID("aa2d30cb-0939-43ca-833e-89eb7cf054dd")
COHERE_ROUTE_GATE_SCHEMA_VERSION = "flavourbench-cohere-continuation-route-gate-v1"
PRIMARY_PREFLIGHT_SCHEMA_VERSION = "flavourbench-frontier-coverage-primary-preflight-v1"

BASE_ARENA_SHA256 = "407e7fc6413e6d009c942eb51d9603d7cb958f0f282ffe90e1dc8ff28c3b6ac3"
BASE_ARENA_RELATIVE = (
    "artifacts/season1/current-quality-run/"
    "frontier-model-arena-review-pool-quarantine-v1/"
    f"frontier-model-arena-review-pool-{BASE_ARENA_SHA256}.json"
)

# Each selected task is a fresh shared anchor for one family.  Composition
# needs only the four endpoints still implicated in its three residual holes;
# the other families need the whole panel except an already successful v4
# anchor on evidence-001 and substitution-012 respectively.
RESIDUAL_TASK_MODELS: tuple[tuple[str, tuple[str, ...] | str], ...] = (
    (
        "fb-s0-composition-010",
        (
            "z-ai/glm-5.2",
            "cohere/command-a-plus-05-2026",
            "cohere/command-a-reasoning-08-2025",
            "x-ai/grok-4.5",
        ),
    ),
    ("fb-s0-cookability-012", "all"),
    ("fb-s0-evidence-001", "all_except_minimax"),
    ("fb-s0-substitution-012", "all_except_deepseek_pro"),
)

EXPECTED_PRIOR_EXPOSURES = {
    "fb-s0-composition-010": frozenset(),
    "fb-s0-cookability-012": frozenset(),
    "fb-s0-evidence-001": frozenset({"minimax/minimax-m3"}),
    "fb-s0-substitution-012": frozenset({"deepseek/deepseek-v4-pro"}),
}

EVIDENCE_DECISION_TOKEN_OVERRIDE = 16_384
COHERE_CONTINUATION_CONTRACT = {
    "schema_version": "flavourbench-cohere-reasoning-continuation-contract-v1",
    "scope": "all_fresh_cohere_residual_cells",
    "provider_message_invariant": (
        "preserve opaque Cohere assistant content blocks across staged turns while "
        "excluding thinking blocks from benchmark-visible text"
    ),
    "tool_turn_invariant": (
        "preserve assistant content together with tool_calls and bind each tool result "
        "to its exact tool_call_id"
    ),
    "validation": "offline_projection_contract_tests_required_before_admission",
}

PRIMARY_PLAN_SHA256 = "f79850aaa6a9b256340c2932ae376e6887e387b7bded6ce2ffd06d7caa3dc308"
PRIMARY_PLAN_RELATIVE = (
    "artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/"
    f"frontier-coverage-primary-on-v5-plan-{PRIMARY_PLAN_SHA256}.json"
)
V4_PREFLIGHT_SHA256 = "c2cf6aa4d6397f6034114dfb9ead0b446895256a7a84705fbd3a55c70d742268"
V4_PHASE1_AUDIT_SHA256 = "6166f1451c9b3fad14aed6838a7aa55a6cf577c7c310ee45d758a5ebcccafe15"
V4_PHASE2_AUDIT_SHA256 = "0698fead0375847f3b72f4198584cce1332afc993ccc98f77c39193861292ab7"
V6_ROUTE_PLAN_SHA256 = "905f41ba1cd50915d6aa8fc11f5f582930e045e9dc0586dae98549ad21fa6a2c"
V6_EXECUTION_PLAN_SHA256 = "4091db95f115d79aa454821aa0700941284dd1a8795e787c5db7d2a405121d54"
V6_AUDIT_SHA256 = "7135eb6819172e253bf678d7ec1a02ca10dede3452f4a0aad4adaebaccab4d3e"
V6_CLOSURE_SHA256 = "748fedad8bfd403c17232058ecace1c7aede79f0754d00dd7ae2e51fabcf6bcf"
V6_RECEIPT_SHA256 = "0b2edd08d073471c4f9eec11abede052160c1f76d9fce16be4cb968f072a2351"
V6_AGGREGATE_AUDIT_SHA256 = "84a3e4f2102a5d1be17d819985aad663210ce6d7e73d56042aaec65c8007d2d1"
V6_AGGREGATE_CLOSURE_SHA256 = "7b7b6b0de6e2951bc4bea8100716d32296c8117c46aafd910cab50f4256414f5"
V6_SOURCE_SHA256S = (
    "356965186144dac3eb74c7a1ccbe7d4431440887f88691eff2999baba37a5d21",
    "263cfb0634e7eea5b3e9134d77ad6b35702256f7e3f3e3154cbecf838775c4b7",
)

AGGREGATE_RECEIPT_SCHEMA_VERSION = "flavourbench-frontier-coverage-v4-aggregate-receipt-v1"
AGGREGATE_AUDIT_SCHEMA_VERSION = "flavourbench-frontier-coverage-v4-aggregate-audit-v1"
CORRECTED_UPLIFT_SCHEMA_VERSION = "flavourbench-frontier-corrected-uplift-input-v2"
CORRECTED_ARENA_SCHEMA_VERSION = "flavourbench-frontier-corrected-arena-input-v2"
CORRECTED_COVERAGE_SCHEMA_VERSION = "flavourbench-frontier-corrected-coverage-metrics-v2"

CORRECTED_ARENA_SHA256 = "234f5b5e3364f0e0f2fddc0f23d47d1d670df509c5707e35cb713183264c5c5e"
CORRECTED_COVERAGE_SHA256 = "68863208cf4ffc4772ee55378e3ce82a66988b8933e3ef1ced34edf290afa695"
AGGREGATE_AUDIT_SHA256 = "43f808b791847cd35aca637c8edb32df67055f4bf8aab588bbd1f4bb7bd9e822"
AGGREGATE_RECEIPT_SHA256 = "9542d711192bed5f10e0dc4bfca80e48539bf864064a8262e57176c7dde1ea45"
POSTRUN_RELATIVE = "artifacts/season1/current-quality-run/frontier-coverage-v4-postrun"

V4_PREFLIGHT_SHA256 = "c2cf6aa4d6397f6034114dfb9ead0b446895256a7a84705fbd3a55c70d742268"
V4_PHASE1_AUDIT_SHA256 = "6166f1451c9b3fad14aed6838a7aa55a6cf577c7c310ee45d758a5ebcccafe15"
V4_PHASE1_RECEIPT_SHA256 = "f6d0babc3b6275a5067c8446c4da2783a4fa6a0afea3c725c944eca2838cfcda"
V4_PHASE1_CLOSURE_SHA256 = "d0dd8c0075f8e2f8dc7a8988a3d2d1ccec6400b3a8b2198c5644a741bc66d287"
V4_PHASE2_AUDIT_SHA256 = "0698fead0375847f3b72f4198584cce1332afc993ccc98f77c39193861292ab7"
V4_PHASE2_RECEIPT_SHA256 = "be0fce2a5fea950e77a8619124858acdb7fdb3e4552518a9cbe29f9f31c9b910"
V4_PHASE2_CLOSURE_SHA256 = "3630d08f5d1734a0a0c67f585380b82f3a1fa96beba2502be970d767ba969cab"

STRICT_POOL_SHA256 = "0da4c58326a936daef3d9e6ac606cfb5abaff2e9d93784754c56a302c662f38c"
HIGH_POOL_SHA256 = "cd47055d12e6360a1ad0bfaa73fe4b2cef5bd1f5666150968bdfeeaf9eca024c"
STRICT_POOL_RELATIVE = (
    "artifacts/season1/current-quality-run/frontier-strict-review-pool-quarantine-v1/"
    f"frontier-multirun-review-pool-{STRICT_POOL_SHA256}.json"
)
HIGH_POOL_RELATIVE = (
    "artifacts/season1/current-quality-run/"
    "frontier-high-resource-review-pool-quarantine-v1/"
    f"frontier-multirun-review-pool-{HIGH_POOL_SHA256}.json"
)

SUCCESS_WORK_ITEMS = {
    # The original repair contributes four complete uplift coordinates.  The
    # DeepSeek coordinate reuses its already committed historical Epicure-on
    # arm; every other listed work item has both new conditions.
    "78261a7ef93c2ab5a82e59c75753339735f354bbe689380d5bf98ff8be1f764c": {
        "origin": "original_repair_new_pair",
        "conditions": ("epicure_off", "epicure_on"),
    },
    "4634a9408376d7437a6e16665baa12ef6d58553e3e46eeca8ca4c6b75f728c46": {
        "origin": "original_repair_with_historical_on_reuse",
        "conditions": ("epicure_off",),
        "historical_on_sha256": "c36c7b6c32b608a9eb09ac4cab631d9736d59ce505d75dd6dd897bd07ab8a096",
    },
    "cf936d3911566f3ec1c627150c2345f49562f443c4d705eda426174597d14776": {
        "origin": "original_repair_new_pair",
        "conditions": ("epicure_off", "epicure_on"),
    },
    "8d4d4cb2b65cbd787c9b7bf40a5589fedb73183a92a48992cf5dc643dffa61fb": {
        "origin": "original_repair_new_pair",
        "conditions": ("epicure_off", "epicure_on"),
    },
    "650b3ced16656fbd66460d556128614385a4411be217f061769420b766d74ad3": {
        "origin": "closed_continuation_new_pair",
        "conditions": ("epicure_off", "epicure_on"),
    },
    "bb4f0c2f8d55a212e3ecd5dbeca6d004c90ea5b162d42d7328cf63f29b8ee9a4": {
        "origin": "v4_phase1_new_pair",
        "conditions": ("epicure_off", "epicure_on"),
    },
    "dd39f32db48e931e9df31d81a7fb5785881b3c8589ab7025c86616f13a80b3b0": {
        "origin": "v4_phase1_new_pair",
        "conditions": ("epicure_off", "epicure_on"),
    },
    "4aa65fb3b3df86120612ab666c8e2f47ba49552de798984df18df3c75343e2a4": {
        "origin": "v4_phase2_new_pair",
        "conditions": ("epicure_off", "epicure_on"),
    },
}

RESIDUAL_FAILURES = {
    "f664fa3227461d1964115b23ecfa3f3666d8675f03df95856fd7f492bac06ddf": {
        "model_id": "mistralai/mistral-medium-3-5",
        "task_id": "fb-s0-evidence-006",
        "failed_condition": "epicure_on",
        "error": "ProviderError: provider evidence-decision turn did not finish normally",
        "diagnosis": "provider_declared_non_normal_evidence_decision_finish",
    },
    "98cf8dfd76a666081fc1dbff06322424e48d3a4391adb96abcdef037ca0f88b0": {
        "model_id": "x-ai/grok-4.5",
        "task_id": "fb-s0-evidence-024",
        "failed_condition": "epicure_on",
        "error": "ProviderError: provider evidence-decision turn did not finish normally",
        "diagnosis": "provider_declared_non_normal_evidence_decision_finish",
    },
    "6194e07b1d784e88e903395092ef6c35eb74b25efabd707f87fc04697ff87944": {
        "model_id": "mistralai/mistral-medium-3-5",
        "task_id": "fb-s0-cookability-002",
        "failed_condition": "epicure_on",
        "error": "ProviderError: provider evidence-decision turn did not finish normally",
        "diagnosis": "provider_declared_non_normal_evidence_decision_finish",
    },
    "6e05317b08fe9d7b9507ac99cb6b652bc36aff85679c219f262239a9532ed80c": {
        "model_id": "cohere/command-a-reasoning-08-2025",
        "task_id": "fb-s0-cookability-004",
        "failed_condition": "epicure_off",
        "error": (
            "ProviderError: Cohere request rejected with HTTP 422: No valid response "
            "generated. Try updating messages"
        ),
        "diagnosis": "cohere_staged_message_continuation_rejected_http_422",
    },
    "bfda28289735a75d4b3f0255c9e21bb2d0e2a10aab76d7b26a6ce8073972f50d": {
        "model_id": "z-ai/glm-5.2",
        "task_id": "fb-s0-composition-002",
        "failed_condition": "epicure_on",
        "error": "ProviderError: provider tool-call fan-out (8) exceeded the per-round cap (6)",
        "diagnosis": "tool_fanout_8_exceeded_frozen_per_round_cap_6",
    },
}


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


def _load_arena(path: Path) -> dict[str, Any]:
    arena = _load_addressed(
        path,
        label="quarantine-corrected arena",
        expected_schema="flavourbench-frontier-model-arena-review-pool-v1",
        expected_digest=BASE_ARENA_SHA256,
    )
    observed = arena.get("observed")
    if (
        not isinstance(observed, Mapping)
        or observed.get("candidate_comparisons") != 876
        or observed.get("source_response_arms") != 185
        or observed.get("missing_model_pair_family_cells") != 94
        or arena.get("claim_boundary", {}).get("official") is not False
        or arena.get("claim_boundary", {}).get("rank_eligible") is not False
        or observed.get("synthetic_arms") != 0
    ):
        raise IntegrityError("quarantine-corrected arena boundary differs")
    return arena


def _iter_documents(root: Path):  # type: ignore[no-untyped-def]
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("exposure root must be a regular directory")
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            yield path, value


def _actual_exposure_snapshot(
    root: Path, *, model_ids: frozenset[str], task_ids: frozenset[str]
) -> dict[str, Any]:
    """Index actual source/response records, excluding plans and analyses."""

    admitted_schemas = {
        "flavourbench-live-smoke-v1",
        "flavourbench-real-exploratory-response-v1",
    }
    records: list[dict[str, str]] = []
    coordinates: set[tuple[str, str]] = set()
    for path, document in _iter_documents(root):
        if document.get("schema_version") not in admitted_schemas:
            continue
        model_id = document.get("requested_model_id")
        task_id = document.get("dataset_task_id")
        if task_id is None and isinstance(document.get("model"), Mapping):
            model_id = document["model"].get("requested_model_id")
        if task_id is None and isinstance(document.get("task"), Mapping):
            task_id = document["task"].get("public_id")
        if model_id not in model_ids or task_id not in task_ids:
            continue
        coordinates.add((str(model_id), str(task_id)))
        records.append(
            {
                "model_id": str(model_id),
                "task_id": str(task_id),
                "schema_version": str(document.get("schema_version")),
                "path": str(path.relative_to(root)),
                "physical_sha256": _file_sha256(path),
                "artifact_sha256": str(document.get("artifact_sha256") or ""),
            }
        )
    return {
        "root": str(root),
        "record_count": len(records),
        "records_sha256": sha256_json(records),
        "coordinate_count": len(coordinates),
        "coordinates_sha256": sha256_json(sorted(coordinates)),
        "coordinates": [
            {"model_id": model_id, "task_id": task_id} for model_id, task_id in sorted(coordinates)
        ],
    }


def _collect_prior_identifiers(root: Path) -> set[str]:
    singular = {
        "cell_id",
        "work_item_id",
        "dataset_work_item_id",
        "run_id",
        "arm_id",
        "attempt_id",
    }
    plural = {
        "closed_work_item_ids",
        "closed_run_ids",
        "work_item_ids",
        "run_ids",
        "arm_ids",
        "attempt_ids",
    }
    found: set[str] = set()

    def visit(value: object, key: str = "") -> None:
        if key in singular and isinstance(value, str) and value:
            found.add(value)
        elif key in plural and isinstance(value, Mapping):
            found.update(str(item) for item in value.values() if isinstance(item, str))
        elif key in plural and isinstance(value, list):
            found.update(str(item) for item in value if isinstance(item, str))
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)

    for _path, document in _iter_documents(root):
        # Re-freezing must reconstruct the same plan after the content-addressed
        # plan itself has been written beneath the exposure root.  Its fresh
        # identifiers are outputs of this function, not prior execution IDs.
        if document.get("schema_version") in {
            RESIDUAL_PLAN_SCHEMA_VERSION,
            PRIMARY_PLAN_SCHEMA_VERSION,
        }:
            continue
        visit(document)
    for path in sorted(root.rglob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            visit(value)
    found.discard("")
    return found


def _candidate_index(manifest_paths: Sequence[Path]):
    result: dict[str, tuple[Any, Path, str]] = {}
    for path in manifest_paths:
        manifest = load_candidate_manifest(path, expected_digest="")
        digest = str(manifest.get("content_address", {}).get("digest") or "")
        for candidate in select_candidates(manifest):
            if candidate.model_id in result:
                raise IntegrityError(
                    f"model appears in multiple residual route manifests: {candidate.model_id}"
                )
            result[candidate.model_id] = (candidate, path, digest)
    return result


def _forecast_with_evidence_override(work_item: WorkItem, *, policy, enabled: bool):  # type: ignore[no-untyped-def]
    forecast = derive_conditions_forecast(
        work_item, policy=policy, conditions=("epicure_off", "epicure_on")
    )
    if not enabled:
        return forecast.forecast_usd, {
            "base_forecast_usd": _decimal_text(forecast.forecast_usd),
            "evidence_decision_extra_completion_tokens": 0,
            "extra_forecast_usd": "0",
        }
    extra_tokens = EVIDENCE_DECISION_TOKEN_OVERRIDE - policy.max_intermediate_tokens
    if extra_tokens <= 0:
        raise IntegrityError("evidence-decision override is not larger than the base limit")
    extra = Decimal(extra_tokens) * (
        forecast.price_envelope.completion_usd_per_token
        + forecast.price_envelope.reasoning_usd_per_token
    )
    return forecast.forecast_usd + extra, {
        "base_forecast_usd": _decimal_text(forecast.forecast_usd),
        "evidence_decision_extra_completion_tokens": extra_tokens,
        "extra_forecast_usd": _decimal_text(extra),
    }


def build_residual_plan(
    *,
    project_root: Path,
    parent_root: Path,
    exposure_root: Path,
    arena_path: Path,
) -> dict[str, Any]:
    parent = reconstruct_parent(project_root=project_root, parent_root=parent_root)
    arena = _load_arena(arena_path)
    model_order = tuple(str(value) for value in arena.get("model_order") or [])
    if len(model_order) != 16 or len(set(model_order)) != 16:
        raise IntegrityError("residual plan requires the exact 16-model corrected panel")
    task_path, manifest_paths = _task_and_manifest_inputs(project_root=project_root, parent=parent)
    tasks, task_source = load_development_task_inventory(task_path)
    task_index = {task.public_id: task for task in tasks}
    registry_sha = task_registry_sha256(tasks)
    candidates = _candidate_index(manifest_paths)
    if set(candidates) != set(model_order):
        raise IntegrityError("route manifests do not exactly cover the 16-model panel")
    selected_task_ids = frozenset(task_id for task_id, _ in RESIDUAL_TASK_MODELS)
    if selected_task_ids & QUARANTINED_TASK_IDS:
        raise IntegrityError("residual schedule selected a quarantined task")
    if any(task_id not in task_index for task_id in selected_task_ids):
        raise IntegrityError("residual schedule task is absent from frozen task validity")
    exposure = _actual_exposure_snapshot(
        exposure_root,
        model_ids=frozenset(model_order),
        task_ids=selected_task_ids,
    )
    observed: dict[str, set[str]] = {task_id: set() for task_id in selected_task_ids}
    for row in exposure["coordinates"]:
        observed[str(row["task_id"])].add(str(row["model_id"]))
    if any(
        frozenset(observed[task_id]) != expected
        for task_id, expected in EXPECTED_PRIOR_EXPOSURES.items()
    ):
        raise IntegrityError("residual task exposure snapshot differs from the frozen expectation")

    prior_identifiers = _collect_prior_identifiers(exposure_root)
    new_identifiers: set[str] = set()
    base_policy = parent.bundle.execution_policy
    glm_policy = replace(
        base_policy,
        max_tool_calls_per_round=13,
        max_tool_calls_total=13,
    )
    cohere_contract_sha = sha256_json(COHERE_CONTINUATION_CONTRACT)
    cells: list[dict[str, Any]] = []
    ordinal = 0
    for task_id, selector in RESIDUAL_TASK_MODELS:
        task = task_index[task_id]
        if selector == "all":
            selected_models = model_order
        elif selector == "all_except_minimax":
            selected_models = tuple(model for model in model_order if model != "minimax/minimax-m3")
        elif selector == "all_except_deepseek_pro":
            selected_models = tuple(
                model for model in model_order if model != "deepseek/deepseek-v4-pro"
            )
        else:
            selected_models = tuple(selector)
        for model_id in selected_models:
            ordinal += 1
            candidate, manifest_path, manifest_sha = candidates[model_id]
            if (model_id, task_id) in {
                (str(row["model_id"]), str(row["task_id"])) for row in exposure["coordinates"]
            }:
                raise IntegrityError("residual schedule would replay an exposed model-task cell")
            glm_override = model_id == "z-ai/glm-5.2" and task.family == "composition"
            evidence_override = task.family == "evidence" and model_id in {
                "mistralai/mistral-medium-3-5",
                "x-ai/grok-4.5",
            }
            policy = glm_policy if glm_override else base_policy
            basis = {
                "schema_version": RESIDUAL_PLAN_SCHEMA_VERSION,
                "ordinal": ordinal,
                "model_id": model_id,
                "provider_tag": candidate.provider_tag,
                "execution_backend": candidate.execution_backend,
                "route_manifest_sha256": manifest_sha,
                "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
                "task_id": task.public_id,
                "task_family": task.family,
                "prompt_sha256": task.prompt_sha256,
                "conditions": ["epicure_off", "epicure_on"],
                "development_only": True,
                "fresh_no_replay": True,
            }
            cell_id = sha256_json({**basis, "identifier_role": "cell"})
            work_id = sha256_json({**basis, "identifier_role": "work_item"})
            run_id = str(uuid.uuid5(RESIDUAL_NAMESPACE, f"run:{cell_id}"))
            arm_ids = {
                condition: sha256_json(
                    {
                        "schema_version": RESIDUAL_PLAN_SCHEMA_VERSION,
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
                namespace=RESIDUAL_NAMESPACE,
            )
            identifiers = {
                cell_id,
                work_id,
                run_id,
                *arm_ids.values(),
                *(str(slot["attempt_id"]) for slot in attempts),
            }
            if prior_identifiers & identifiers or new_identifiers & identifiers:
                raise IntegrityError("residual identifier overlaps prior or new work")
            new_identifiers.update(identifiers)
            work_item = WorkItem(
                ordinal=ordinal,
                work_item_id=work_id,
                manifest_sha256=manifest_sha,
                task_registry_sha256=registry_sha,
                task=task,
                candidate=candidate,
                endpoint_execution_sha256=candidate.endpoint_execution_sha256,
                execution_policy_sha256=policy.sha256,
                execution_policy=policy,
            )
            forecast, forecast_detail = _forecast_with_evidence_override(
                work_item, policy=policy, enabled=evidence_override
            )
            policy_override = {
                "glm_tool_fanout_13_by_13": glm_override,
                "evidence_decision_max_tokens": (
                    EVIDENCE_DECISION_TOKEN_OVERRIDE if evidence_override else None
                ),
                "cohere_continuation_contract_sha256": (
                    cohere_contract_sha if candidate.execution_backend == "cohere_direct" else None
                ),
                "non_retroactive": bool(
                    glm_override
                    or evidence_override
                    or candidate.execution_backend == "cohere_direct"
                ),
            }
            cells.append(
                {
                    **basis,
                    "cell_id": cell_id,
                    "work_item_id": work_id,
                    "run_id": run_id,
                    "arm_ids": arm_ids,
                    "attempt_slots": attempts,
                    "attempt_slots_sha256": sha256_json(attempts),
                    "route_manifest_path": _relative(project_root, manifest_path),
                    "execution_policy": policy.document(),
                    "execution_policy_sha256": policy.sha256,
                    "policy_override": policy_override,
                    "forecast_detail": forecast_detail,
                    "reserved_worst_case_usd": _decimal_text(forecast),
                    "no_prior_model_task_exposure_at_freeze": True,
                    "fresh_identifiers_disjoint_from_all_prior_records": True,
                    "official_fit_eligible": False,
                }
            )

    if len(cells) != 50:
        raise IntegrityError("residual schedule is not the exact 50-cell minimum bridge")
    family_counts = Counter(str(cell["task_family"]) for cell in cells)
    batch_rows: list[dict[str, Any]] = []
    for batch_ordinal, model_id in enumerate(model_order, start=1):
        batch_cells = [cell for cell in cells if cell["model_id"] == model_id]
        if not batch_cells:
            raise IntegrityError("every residual endpoint must receive an isolated batch")
        batch_rows.append(
            {
                "batch_ordinal": batch_ordinal,
                "batch_id": sha256_json(
                    {
                        "schema_version": RESIDUAL_PLAN_SCHEMA_VERSION,
                        "model_id": model_id,
                        "work_item_ids": [cell["work_item_id"] for cell in batch_cells],
                    }
                ),
                "model_id": model_id,
                "provider_tag": batch_cells[0]["provider_tag"],
                "execution_backend": batch_cells[0]["execution_backend"],
                "cell_count": len(batch_cells),
                "work_item_ids": [cell["work_item_id"] for cell in batch_cells],
                "worst_case_usd": _decimal_text(
                    sum(
                        (Decimal(str(cell["reserved_worst_case_usd"])) for cell in batch_cells),
                        Decimal(0),
                    )
                ),
                "failure_isolation": "close_only_this_fresh_cell_then_continue_endpoint_batch",
                "separate_ledger_and_terminal_closure": True,
            }
        )
    total_worst_case = sum(
        (Decimal(str(cell["reserved_worst_case_usd"])) for cell in cells), Decimal(0)
    )
    payload = {
        "schema_version": RESIDUAL_PLAN_SCHEMA_VERSION,
        "status": "frozen_zero_provider_or_mcp_calls",
        "purpose": "minimum_no_replay_shared_anchor_bridge_for_remaining_support_holes",
        "sources": {
            "parent_reconstructed_audit_sha256": parent.audit["artifact_sha256"],
            "base_arena_sha256": arena["artifact_sha256"],
            "task_validity_sha256": task_source["artifact_sha256"],
            "task_registry_sha256": registry_sha,
            "route_manifest_sha256s": sorted(
                str(load_candidate_manifest(path, expected_digest="")["content_address"]["digest"])
                for path in manifest_paths
            ),
            "exposure_snapshot": exposure,
            "prior_identifier_count": len(prior_identifiers),
            "prior_identifiers_sha256": sha256_json(sorted(prior_identifiers)),
        },
        "support_basis": {
            "corrected_current_comparisons": 915,
            "corrected_current_unique_arms": 192,
            "model_pair_family_cells": 480,
            "missing_cells_before": 73,
            "missing_cells_before_by_family": {
                "composition": 3,
                "cookability": 20,
                "evidence": 27,
                "substitution": 23,
            },
            "projected_missing_cells_after_all_50_cells_usable": 0,
            "projection_is_not_an_observation": True,
        },
        "selected_tasks": [
            {
                "task_id": task_id,
                "task_family": task_index[task_id].family,
                "prompt_sha256": task_index[task_id].prompt_sha256,
                "quarantined": False,
                "prior_exposed_models": sorted(EXPECTED_PRIOR_EXPOSURES[task_id]),
            }
            for task_id, _ in RESIDUAL_TASK_MODELS
        ],
        "cells": cells,
        "endpoint_isolated_batches": batch_rows,
        "counts": {
            "fresh_cells": 50,
            "fresh_real_arms": 100,
            "synthetic_arms": 0,
            "endpoint_isolated_batches": 16,
            "cells_by_family": {
                family: family_counts[family]
                for family in ("composition", "cookability", "evidence", "substitution")
            },
            "provider_calls_by_freeze": 0,
            "epicure_calls_by_freeze": 0,
        },
        "policy_overrides": {
            "base_execution_policy_sha256": base_policy.sha256,
            "glm_composition_execution_policy": glm_policy.document(),
            "glm_override_scope": "z-ai/glm-5.2_on_fb-s0-composition-010_only",
            "evidence_decision_override_scope": [
                "mistralai/mistral-medium-3-5_on_fb-s0-evidence-001",
                "x-ai/grok-4.5_on_fb-s0-evidence-001",
            ],
            "evidence_decision_max_tokens": EVIDENCE_DECISION_TOKEN_OVERRIDE,
            "cohere_continuation_contract": COHERE_CONTINUATION_CONTRACT,
            "cohere_continuation_contract_sha256": cohere_contract_sha,
            "all_overrides_non_retroactive": True,
        },
        "budget": {
            "currency": "USD",
            "residual_worst_case_usd": _decimal_text(total_worst_case),
            "transactional_reservation_unit": "endpoint_isolated_batch",
            "admission_requires_fresh_source_reconstructed_preflight": True,
        },
        "execution_order": {
            "one_endpoint_batch_at_a_time": True,
            "one_fresh_cell_failure_does_not_stop_unrelated_cells": True,
            "same_identifier_replay_permitted": False,
            "provider_substitution_permitted": False,
            "manual_replay_permitted": False,
        },
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "official_preference_or_uplift_fit_eligible": False,
            "permitted_analysis": "coverage_geometry_and_reliability_diagnostics_only",
            "human_quality_judgments": 0,
            "synthetic_arms": 0,
        },
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _find_addressed_by_digest(root: Path, digest: str, *, label: str) -> Path:
    matches = [
        path for path in root.rglob(f"*{digest}*.json") if path.is_file() and not path.is_symlink()
    ]
    if len(matches) != 1:
        raise IntegrityError(f"{label} did not resolve exactly once: {digest}")
    return matches[0]


def _source_by_work_item(root: Path, work_item_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(document, dict)
            and document.get("schema_version") == "flavourbench-live-smoke-v1"
            and document.get("dataset_work_item_id") == work_item_id
        ):
            verified = _verify_artifact(path)
            matches.append((path, verified))
    if len(matches) != 1:
        raise IntegrityError(f"work item source did not resolve exactly once: {work_item_id}")
    return matches[0]


def _responses_by_work_item(
    root: Path, work_item_id: str
) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.rglob(f"response-{work_item_id}-*.json")):
        document = _verify_artifact(path)
        condition = str(document.get("condition") or "")
        if condition in result:
            raise IntegrityError(f"duplicate normalized response for {work_item_id}:{condition}")
        result[condition] = (path, document)
    return result


def _verify_success_arms(
    *, current_root: Path, expected_epicure: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Arm]]:
    """Reconstruct the exact eight usable uplift coordinates from raw sources."""

    source_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    all_arms: dict[str, Arm] = {}
    generation_ids: set[str] = set()
    for work_item_id, specification in SUCCESS_WORK_ITEMS.items():
        source_path, source = _source_by_work_item(current_root, work_item_id)
        expected_conditions = tuple(specification["conditions"])
        if (
            source.get("status") != "complete"
            or tuple(sorted((source.get("results") or {}).keys()))
            != tuple(sorted(expected_conditions))
            or source.get("errors") != {}
            or source.get("budget", {}).get("all_generation_costs_reconciled") is not True
        ):
            raise IntegrityError(f"usable source contract differs: {work_item_id}")
        responses = _responses_by_work_item(current_root, work_item_id)
        if set(responses) != set(expected_conditions):
            raise IntegrityError(f"usable normalized response set differs: {work_item_id}")
        by_condition: dict[str, Arm] = {}
        response_bindings: dict[str, dict[str, str]] = {}
        for condition in expected_conditions:
            path, document = responses[condition]
            arm = _arm_from_document(document, expected_epicure=expected_epicure)
            if (
                arm.work_item_id != work_item_id
                or arm.condition != condition
                or arm.source_artifact_sha256 != source["artifact_sha256"]
            ):
                raise IntegrityError("normalized success arm changed source or condition")
            if generation_ids.intersection(arm.generation_ids):
                raise IntegrityError("usable coverage arms duplicate a provider generation")
            generation_ids.update(arm.generation_ids)
            by_condition[condition] = arm
            all_arms[arm.response_artifact_sha256] = arm
            response_bindings[condition] = {
                "path": str(path.relative_to(current_root)),
                "artifact_sha256": arm.response_artifact_sha256,
                "physical_sha256": _file_sha256(path),
            }
        historical_on_sha = specification.get("historical_on_sha256")
        if historical_on_sha:
            historical_path = _find_addressed_by_digest(
                current_root,
                str(historical_on_sha),
                label="historical DeepSeek Epicure-on response",
            )
            historical_document = _verify_artifact(historical_path)
            historical_on = _arm_from_document(
                historical_document, expected_epicure=expected_epicure
            )
            off = by_condition["epicure_off"]
            if (
                historical_on.condition != "epicure_on"
                or historical_on.model_id != off.model_id
                or historical_on.task_id != off.task_id
                or historical_on.family != off.family
                or historical_on.prompt_sha256 != off.prompt_sha256
                or historical_on.provider_tag != off.provider_tag
                or historical_on.canonical_model_slug != off.canonical_model_slug
            ):
                raise IntegrityError("historical DeepSeek Epicure-on reuse changed coordinate")
            if generation_ids.intersection(historical_on.generation_ids):
                raise IntegrityError("historical DeepSeek reuse duplicates a new generation")
            generation_ids.update(historical_on.generation_ids)
            by_condition["epicure_on"] = historical_on
            all_arms[historical_on.response_artifact_sha256] = historical_on
            response_bindings["epicure_on"] = {
                "path": str(historical_path.relative_to(current_root)),
                "artifact_sha256": historical_on.response_artifact_sha256,
                "physical_sha256": _file_sha256(historical_path),
            }
        if set(by_condition) != {"epicure_off", "epicure_on"}:
            raise IntegrityError("usable coordinate is not a complete uplift pair")
        off, on = by_condition["epicure_off"], by_condition["epicure_on"]
        if any(
            getattr(off, field) != getattr(on, field)
            for field in (
                "task_id",
                "family",
                "prompt_sha256",
                "model_id",
                "canonical_model_slug",
                "provider_tag",
            )
        ):
            raise IntegrityError("usable uplift coordinate changed between conditions")
        cost_micros = int(source.get("budget", {}).get("actual_cost_micros") or 0)
        if cost_micros != sum(
            int((source["results"][condition] or {}).get("cost_micros") or 0)
            for condition in expected_conditions
        ):
            raise IntegrityError("usable source cost does not sum across requested conditions")
        source_records.append(
            {
                "work_item_id": work_item_id,
                "model_id": off.model_id,
                "task_id": off.task_id,
                "task_family": off.family,
                "status": "source_reconstructed_usable_complete_pair",
                "origin": specification["origin"],
                "source": {
                    "path": str(source_path.relative_to(current_root)),
                    "artifact_sha256": source["artifact_sha256"],
                    "physical_sha256": _file_sha256(source_path),
                },
                "response_bindings": response_bindings,
                "source_actual_cost_micros": cost_micros,
                "historical_condition_reused": bool(historical_on_sha),
            }
        )
        candidates.append(
            {
                "origin": specification["origin"],
                "task_id": off.task_id,
                "family": off.family,
                "prompt_sha256": off.prompt_sha256,
                "model_id": off.model_id,
                "canonical_model_slug": off.canonical_model_slug,
                "provider_tag": off.provider_tag,
                "condition_work_item_ids": {
                    condition: arm.work_item_id for condition, arm in by_condition.items()
                },
                "arms": by_condition,
            }
        )
    if len(candidates) != 8 or len(all_arms) != 16:
        raise IntegrityError("source reconstruction did not produce eight disjoint uplift pairs")
    return source_records, candidates, all_arms


def _verify_residual_failures(current_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for work_item_id, expected in RESIDUAL_FAILURES.items():
        path, source = _source_by_work_item(current_root, work_item_id)
        errors = source.get("errors")
        results = source.get("results")
        incomplete = source.get("incomplete_generation_metadata")
        if (
            source.get("status") != "failed_or_unreconciled"
            or source.get("requested_model_id") != expected["model_id"]
            or source.get("dataset_task_id") != expected["task_id"]
            or not isinstance(errors, Mapping)
            or errors.get(expected["failed_condition"]) != expected["error"]
            or not isinstance(results, Mapping)
            or set(results)
            != (
                {"epicure_on"} if expected["failed_condition"] == "epicure_off" else {"epicure_off"}
            )
            or not isinstance(incomplete, list)
        ):
            raise IntegrityError(f"residual failure evidence changed: {work_item_id}")
        responses = _responses_by_work_item(current_root, work_item_id)
        if set(responses) != set(results):
            raise IntegrityError("partial failure response set differs from raw source")
        records.append(
            {
                "work_item_id": work_item_id,
                "model_id": expected["model_id"],
                "task_id": expected["task_id"],
                "failed_condition": expected["failed_condition"],
                "error": expected["error"],
                "diagnosis": expected["diagnosis"],
                "source": {
                    "path": str(path.relative_to(current_root)),
                    "artifact_sha256": source["artifact_sha256"],
                    "physical_sha256": _file_sha256(path),
                },
                "actual_cost_micros": int(source.get("budget", {}).get("actual_cost_micros") or 0),
                "incomplete_generation_count": len(incomplete),
                "retained_for_reliability_only": True,
                "preference_or_uplift_pool_admitted": False,
                "safe_to_replay": False,
            }
        )
    if len(records) != 5:
        raise IntegrityError("residual failure reconstruction count differs")
    return records


def _load_phase_artifact(root: Path, prefix: str, digest: str) -> Path:
    return root / f"{prefix}-{digest}.json"


def _reconstruct_v4_phases(
    *, project_root: Path, v4_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight = _load_phase_artifact(
        v4_root, "frontier-coverage-recovery-v4-preflight", V4_PREFLIGHT_SHA256
    )
    phase1_receipt = _load_phase_artifact(
        v4_root,
        "frontier-coverage-recovery-v4-untouched_recovery-receipt",
        V4_PHASE1_RECEIPT_SHA256,
    )
    phase1_closure = _load_phase_artifact(
        v4_root,
        "frontier-coverage-recovery-v4-untouched_recovery-closure",
        V4_PHASE1_CLOSURE_SHA256,
    )
    phase2_receipt = _load_phase_artifact(
        v4_root,
        "frontier-coverage-recovery-v4-glm_specific_replacement-receipt",
        V4_PHASE2_RECEIPT_SHA256,
    )
    phase2_closure = _load_phase_artifact(
        v4_root,
        "frontier-coverage-recovery-v4-glm_specific_replacement-closure",
        V4_PHASE2_CLOSURE_SHA256,
    )
    # The historical audit function checks credential *presence* even though
    # its audit branch never makes a network request.  Supply non-secret
    # sentinels so this offline reconstruction cannot depend on live secrets.
    with patch.dict(
        os.environ,
        {
            "FLAVOURBENCH_OPENROUTER_API_KEY": "offline-audit-sentinel",
            "FLAVOURBENCH_COHERE_API_KEY": "offline-audit-sentinel",
            "FLAVOURBENCH_MCP_URL": "http://offline-audit.invalid/mcp",
            "FLAVOURBENCH_MCP_TOKEN": "offline-audit-sentinel",
        },
        clear=False,
    ):
        original_scan = recovery_v4._scan_model_task_exposure

        def phase1_time_scan(root: Path, *, model_ids: set[str]) -> dict[str, Any]:
            snapshot = original_scan(root, model_ids=model_ids)
            records = [
                row
                for row in snapshot["records"]
                if "frontier-coverage-recovery-v4/glm-specific-replacement/" not in str(row["path"])
            ]
            return {
                **snapshot,
                "record_count": len(records),
                "records_sha256": sha256_json(records),
                "records": records,
            }

        # Recreate the phase-1 audit's historical visibility boundary: the
        # separately barred GLM phase did not yet exist when phase 1 closed.
        with patch.object(recovery_v4, "_scan_model_task_exposure", phase1_time_scan):
            phase1 = build_phase_audit(
                preflight_path=preflight,
                receipt_path=phase1_receipt,
                closure_path=phase1_closure,
                project_root=project_root,
                output_root=v4_root,
            )
        phase2 = build_phase_audit(
            preflight_path=preflight,
            receipt_path=phase2_receipt,
            closure_path=phase2_closure,
            project_root=project_root,
            output_root=v4_root,
        )
    if (
        phase1.get("artifact_sha256") != V4_PHASE1_AUDIT_SHA256
        or phase2.get("artifact_sha256") != V4_PHASE2_AUDIT_SHA256
        or phase1.get("decision") != "passed_complete_phase_disposition"
        or phase2.get("decision") != "passed_complete_phase_disposition"
        or phase1.get("counts", {}).get("usable_complete_cells") != 2
        or phase2.get("counts", {}).get("usable_complete_cells") != 1
        or phase1.get("accounting", {}).get("actual_cost_micros") != 424_886
        or phase2.get("accounting", {}).get("actual_cost_micros") != 24_079
    ):
        raise IntegrityError("v4 terminal phases no longer reconstruct exactly")
    return phase1, phase2


def _build_corrected_uplift(
    *,
    strict: Mapping[str, Any],
    high: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    aggregate_source_sha256: str,
) -> dict[str, Any]:
    base_items: list[dict[str, Any]] = []
    for stratum, pool in (("strict", strict), ("high-resource", high)):
        for item in pool.get("items") or []:
            copied = dict(item)
            copied["execution_stratum"] = stratum
            copied["origin"] = "retained_quarantine_corrected_base"
            base_items.append(copied)
    source_commitment = sha256_json(
        {
            "strict_pool_sha256": strict["artifact_sha256"],
            "high_pool_sha256": high["artifact_sha256"],
            "aggregate_source_sha256": aggregate_source_sha256,
            "new_response_sets": sorted(
                sorted(
                    candidate["arms"][condition].response_artifact_sha256
                    for condition in ("epicure_off", "epicure_on")
                )
                for candidate in candidates
            ),
        }
    )
    new_items: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates, key=lambda row: (str(row["model_id"]), str(row["task_id"]))
    ):
        arms = candidate["arms"]
        pair_key = sha256_json(
            {
                "model_id": candidate["model_id"],
                "task_id": candidate["task_id"],
                "responses": {
                    condition: arms[condition].response_artifact_sha256
                    for condition in ("epicure_off", "epicure_on")
                },
            }
        )
        on_left = (
            int(
                hashlib.sha256(f"{source_commitment}:{pair_key}:side".encode()).hexdigest()[:2],
                16,
            )
            % 2
            == 0
        )
        sides = (
            (("left", "epicure_on"), ("right", "epicure_off"))
            if on_left
            else (("left", "epicure_off"), ("right", "epicure_on"))
        )
        item: dict[str, Any] = {
            "review_item_id": hashlib.sha256(
                f"{source_commitment}:{pair_key}:review-item".encode()
            ).hexdigest(),
            "pair_key": pair_key,
            "origin": candidate["origin"],
            "execution_stratum": "high-resource-coverage-recovery",
            "condition_work_item_ids": dict(candidate["condition_work_item_ids"]),
            "task_id": candidate["task_id"],
            "task_family": candidate["family"],
            "prompt_sha256": candidate["prompt_sha256"],
            "requested_model_id": candidate["model_id"],
            "canonical_model_slug": candidate["canonical_model_slug"],
            "provider_tag": candidate["provider_tag"],
        }
        for side, condition in sides:
            item[side] = arms[condition].uplift_manifest()
        new_items.append(item)
    items = base_items + new_items
    response_digests = [
        str(item[side]["response_artifact_sha256"]) for item in items for side in ("left", "right")
    ]
    if len(response_digests) != len(set(response_digests)):
        raise IntegrityError("corrected uplift pool reuses a response across pairs")
    family_counts = Counter(str(item["task_family"]) for item in items)
    model_order = tuple(str(value) for value in high.get("model_order") or [])
    model_counts = Counter(str(item["requested_model_id"]) for item in items)
    new_arms = {
        arm.response_artifact_sha256: arm
        for candidate in candidates
        for arm in candidate["arms"].values()
    }
    observed = {
        "retained_strict_pairs": int(strict["observed"]["candidate_pairs"]),
        "retained_high_resource_pairs": int(high["observed"]["candidate_pairs"]),
        "coverage_recovery_pairs_added": len(new_items),
        "candidate_pairs": len(items),
        "source_arms": 2 * len(items),
        "unique_task_ids": len({str(item["task_id"]) for item in items}),
        "distinct_tasks": len({str(item["task_id"]) for item in items}),
        "candidate_pairs_by_family": {family: family_counts[family] for family in FAMILIES},
        "candidate_pairs_by_model": {model_id: model_counts[model_id] for model_id in model_order},
        "real_provider_calls": int(strict["observed"]["real_provider_calls"])
        + int(high["observed"]["real_provider_calls"])
        + sum(len(arm.generation_ids) for arm in new_arms.values()),
        "real_epicure_calls": int(strict["observed"]["real_epicure_calls"])
        + int(high["observed"]["real_epicure_calls"])
        + sum(arm.tool_calls for arm in new_arms.values()),
        "successful_real_epicure_calls": int(strict["observed"]["successful_real_epicure_calls"])
        + int(high["observed"]["successful_real_epicure_calls"])
        + sum(arm.successful_tool_calls for arm in new_arms.values()),
        "reviewed_source_cost_micros": int(strict["observed"]["reviewed_source_cost_micros"])
        + int(high["observed"]["reviewed_source_cost_micros"])
        + sum(arm.cost_micros for arm in new_arms.values()),
        "left_epicure_on": sum(item["left"]["condition"] == "epicure_on" for item in items),
        "right_epicure_on": sum(item["right"]["condition"] == "epicure_on" for item in items),
        "synthetic_arms": 0,
    }
    if len(items) != 187 or observed["source_arms"] != 374:
        raise IntegrityError("corrected uplift pool is not the exact 187-pair pool")
    payload = {
        "schema_version": CORRECTED_UPLIFT_SCHEMA_VERSION,
        "artifact_role": "source_reconstructed_corrected_development_uplift_pool",
        "status": "verified_real_development_input",
        "track": "epicure_uplift",
        "source": {
            "strict_pool_sha256": strict["artifact_sha256"],
            "high_resource_pool_sha256": high["artifact_sha256"],
            "aggregate_source_sha256": aggregate_source_sha256,
            "source_commitment_sha256": source_commitment,
            "historical_raw_artifacts_mutated": False,
        },
        "selection_policy": {
            "paired_same_model_same_task": True,
            "complete_pairs_only": True,
            "failed_or_partial_pairs_retained_for_reliability_only": True,
            "raw_answers_edited": False,
            "deterministic_side_assignment": True,
        },
        "observed": observed,
        "epicure": high["epicure"],
        "identity_commitment_sha256": _uplift_identity_commitment(items),
        "model_order": list(model_order),
        "model_contracts": high["model_contracts"],
        "items": items,
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "permitted_use": "blinded real-output development review",
            "prohibited_use": "quality or uplift ranking before real judgments",
            "synthetic_arms": 0,
        },
    }
    document = {**payload, "artifact_sha256": sha256_json(payload)}
    _verify_uplift_pool(document, label="source-reconstructed corrected uplift")
    return document


def _build_corrected_arena(
    *,
    base: Mapping[str, Any],
    success_arms: Mapping[str, Arm],
    aggregate_source_sha256: str,
) -> dict[str, Any]:
    model_order = tuple(str(value) for value in base.get("model_order") or [])
    coordinates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in base.get("items") or []:
        stratum = str(item["execution_stratum"])
        for side in ("left", "right"):
            arm = dict(item[side])
            key = (stratum, str(item["task_id"]), str(arm["requested_model_id"]))
            row = {
                "stratum": stratum,
                "task_id": str(item["task_id"]),
                "family": str(item["task_family"]),
                "prompt_sha256": str(item["prompt_sha256"]),
                "model_id": str(arm["requested_model_id"]),
                "execution_policy_sha256": str(item["execution_policy_sha256"]),
                "arm": arm,
                "new_arm": False,
                "generation_count": None,
                "tool_calls": None,
                "successful_tool_calls": None,
                "cost_micros": None,
            }
            prior = coordinates.setdefault(key, row)
            if prior != row:
                raise IntegrityError("base arena coordinate resolves to multiple arms")
    arena_success_work_items = {
        "78261a7ef93c2ab5a82e59c75753339735f354bbe689380d5bf98ff8be1f764c",
        "cf936d3911566f3ec1c627150c2345f49562f443c4d705eda426174597d14776",
        "8d4d4cb2b65cbd787c9b7bf40a5589fedb73183a92a48992cf5dc643dffa61fb",
        "650b3ced16656fbd66460d556128614385a4411be217f061769420b766d74ad3",
        "bb4f0c2f8d55a212e3ecd5dbeca6d004c90ea5b162d42d7328cf63f29b8ee9a4",
        "dd39f32db48e931e9df31d81a7fb5785881b3c8589ab7025c86616f13a80b3b0",
        "4aa65fb3b3df86120612ab666c8e2f47ba49552de798984df18df3c75343e2a4",
    }
    added = 0
    for arm in success_arms.values():
        if arm.condition != "epicure_on" or arm.work_item_id not in arena_success_work_items:
            continue
        key = ("high-resource", arm.task_id, arm.model_id)
        if key in coordinates:
            raise IntegrityError("new coverage arena arm duplicates a retained coordinate")
        response_policy = None
        # The normalized response binds the policy through its response model;
        # all seven admitted on-arms use the frozen high-resource base policy.
        response_policy = "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d"
        coordinates[key] = {
            "stratum": "high-resource",
            "task_id": arm.task_id,
            "family": arm.family,
            "prompt_sha256": arm.prompt_sha256,
            "model_id": arm.model_id,
            "execution_policy_sha256": response_policy,
            "arm": arm.arena_manifest(),
            "new_arm": True,
            "generation_count": len(arm.generation_ids),
            "tool_calls": arm.tool_calls,
            "successful_tool_calls": arm.successful_tool_calls,
            "cost_micros": arm.cost_micros,
        }
        added += 1
    if added != 7 or len(coordinates) != 192:
        raise IntegrityError("corrected arena does not contain 185 retained plus seven new arms")
    source_commitment = sha256_json(
        {
            "base_arena_sha256": base["artifact_sha256"],
            "aggregate_source_sha256": aggregate_source_sha256,
            "coordinates": sorted(
                (
                    {
                        "stratum": key[0],
                        "task_id": key[1],
                        "model_id": key[2],
                        "response_artifact_sha256": row["arm"]["response_artifact_sha256"],
                    }
                    for key, row in coordinates.items()
                ),
                key=lambda item: (item["stratum"], item["task_id"], item["model_id"]),
            ),
        }
    )
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in coordinates.values():
        by_task[(str(row["stratum"]), str(row["task_id"]))].append(row)
    candidates: list[dict[str, Any]] = []
    for (stratum, task_id), rows in sorted(by_task.items()):
        ordered = sorted(rows, key=lambda row: model_order.index(str(row["model_id"])))
        for left, right in itertools.combinations(ordered, 2):
            if left["execution_policy_sha256"] != right["execution_policy_sha256"]:
                raise IntegrityError("arena candidate crosses execution policies")
            model_a, model_b = sorted((str(left["model_id"]), str(right["model_id"])))
            candidate_id = hashlib.sha256(
                f"{source_commitment}:{stratum}:{task_id}:{model_a}:{model_b}:arena-candidate".encode()
            ).hexdigest()
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "stratum": stratum,
                    "task_id": task_id,
                    "family": left["family"],
                    "prompt_sha256": left["prompt_sha256"],
                    "model_a": model_a,
                    "model_b": model_b,
                    "execution_policy_sha256": left["execution_policy_sha256"],
                }
            )
    layout = _arena_side_layout(candidates, source_commitment)
    items: list[dict[str, Any]] = []
    presentations: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_pair_family: Counter[tuple[str, str, str]] = Counter()
    edges: list[tuple[str, str]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (
            str(row["stratum"]),
            str(row["task_id"]),
            str(row["model_a"]),
            str(row["model_b"]),
        ),
    ):
        left_model, right_model = layout[str(candidate["candidate_id"])]
        left = coordinates[(candidate["stratum"], candidate["task_id"], left_model)]
        right = coordinates[(candidate["stratum"], candidate["task_id"], right_model)]
        items.append(
            {
                "review_item_id": hashlib.sha256(
                    f"{source_commitment}:{candidate['candidate_id']}:review-item".encode()
                ).hexdigest(),
                "task_id": candidate["task_id"],
                "task_family": candidate["family"],
                "prompt_sha256": candidate["prompt_sha256"],
                "execution_stratum": candidate["stratum"],
                "execution_policy_sha256": candidate["execution_policy_sha256"],
                "left": left["arm"],
                "right": right["arm"],
            }
        )
        for row in (left, right):
            presentations[str(row["arm"]["response_artifact_sha256"])] += 1
        pair = tuple(sorted((left_model, right_model)))
        family = str(candidate["family"])
        by_family[family] += 1
        by_pair_family[(pair[0], pair[1], family)] += 1
        edges.append(pair)
    pair_family = [
        {
            "model_a": model_a,
            "model_b": model_b,
            "task_family": family,
            "candidate_comparisons": by_pair_family[(*tuple(sorted((model_a, model_b))), family)],
        }
        for model_a, model_b in itertools.combinations(model_order, 2)
        for family in FAMILIES
    ]
    missing = [row for row in pair_family if row["candidate_comparisons"] == 0]
    missing_by_family = Counter(str(row["task_family"]) for row in missing)
    unpaired = [
        {
            "execution_stratum": row["stratum"],
            "task_id": row["task_id"],
            "task_family": row["family"],
            "model_id": row["model_id"],
            "response_artifact_sha256": row["arm"]["response_artifact_sha256"],
            "reason": "no_same_task_same_policy_peer_yet",
        }
        for row in coordinates.values()
        if presentations[row["arm"]["response_artifact_sha256"]] == 0
    ]
    reuse = list(presentations.values())
    if (
        len(items) != 915
        or len(missing) != 73
        or dict(sorted(missing_by_family.items()))
        != {"composition": 3, "cookability": 20, "evidence": 27, "substitution": 23}
        or len(unpaired) != 4
        or _components(edges, model_order) != [sorted(model_order)]
    ):
        raise IntegrityError(
            "corrected arena support geometry differs from source reconstruction: "
            f"comparisons={len(items)}, missing={len(missing)}, "
            f"missing_by_family={dict(sorted(missing_by_family.items()))}, "
            f"unpaired={len(unpaired)}, components={_components(edges, model_order)}"
        )
    observed = {
        "base_candidate_comparisons": int(base["observed"]["candidate_comparisons"]),
        "coverage_recovery_candidate_comparisons_added": len(items)
        - int(base["observed"]["candidate_comparisons"]),
        "candidate_comparisons": len(items),
        "source_response_arms": len(coordinates),
        "compared_response_arms": len(presentations),
        "unpaired_response_arms": len(unpaired),
        "candidate_comparisons_by_family": {family: by_family[family] for family in FAMILIES},
        "candidate_comparisons_by_model_pair_family": pair_family,
        "model_pair_family_cells": len(pair_family),
        "missing_model_pair_family_cells": len(missing),
        "missing_model_pair_family_cells_by_family": {
            family: missing_by_family[family] for family in FAMILIES
        },
        "missing_model_pair_family_cell_records": missing,
        "task_stratum_clusters": len(by_task),
        "unique_task_ids": len({row["task_id"] for row in coordinates.values()}),
        "synthetic_arms": 0,
        "evidence_units": {
            "raw_comparison_rows": len(items),
            "unique_response_arms": len(coordinates),
            "compared_response_arms": len(presentations),
            "unpaired_response_arms": len(unpaired),
            "response_arm_presentations": 2 * len(items),
            "minimum_comparisons_per_compared_response_arm": min(reuse),
            "maximum_comparisons_per_compared_response_arm": max(reuse),
            "independence_unit_for_uncertainty": "task_and_response_cluster",
            "comparison_rows_treated_as_independent": False,
            "cluster_by_task_required": True,
            "cluster_by_response_required": True,
            "scalar_effective_sample_size_claimed": False,
        },
    }
    payload = {
        "schema_version": CORRECTED_ARENA_SCHEMA_VERSION,
        "artifact_role": "source_reconstructed_corrected_development_model_arena_pool",
        "status": "verified_real_development_input",
        "track": "model_arena",
        "source": {
            "base_arena_sha256": base["artifact_sha256"],
            "aggregate_source_sha256": aggregate_source_sha256,
            "source_commitment_sha256": source_commitment,
            "historical_raw_artifacts_mutated": False,
        },
        "selection_policy": {
            "same_task": True,
            "same_execution_stratum": True,
            "same_execution_policy": True,
            "epicure_on_only": True,
            "all_available_unordered_model_pairs": True,
            "raw_answers_edited": False,
            "failed_responses_retained_for_reliability_only": True,
        },
        "observed": observed,
        "unpaired_epicure_on_registry": sorted(
            unpaired, key=lambda row: (row["task_id"], row["model_id"])
        ),
        "epicure": base["epicure"],
        "model_order": list(model_order),
        "model_contracts": base["model_contracts"],
        "items": items,
        "items_commitment_sha256": sha256_json(items),
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "family_specific_ranking_supported": False,
            "permitted_use": "blinded real-output development review and support diagnostics",
            "prohibited_use": "model-quality ranking before real judgments",
            "synthetic_arms": 0,
        },
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def materialize_aggregate(
    *,
    project_root: Path,
    current_root: Path,
    parent_root: Path,
    v4_root: Path,
    arena_path: Path,
    strict_path: Path,
    high_path: Path,
) -> dict[str, dict[str, Any]]:
    parent = reconstruct_parent(project_root=project_root, parent_root=parent_root)
    phase1, phase2 = _reconstruct_v4_phases(project_root=project_root, v4_root=v4_root)
    arena_base = _load_arena(arena_path)
    strict = _load_addressed(
        strict_path,
        label="quarantine-corrected strict uplift",
        expected_schema="flavourbench-frontier-multirun-review-pool-v1",
        expected_digest=STRICT_POOL_SHA256,
    )
    high = _load_addressed(
        high_path,
        label="quarantine-corrected high-resource uplift",
        expected_schema="flavourbench-frontier-multirun-review-pool-v1",
        expected_digest=HIGH_POOL_SHA256,
    )
    _verify_uplift_pool(strict, label="quarantine-corrected strict uplift")
    _verify_uplift_pool(high, label="quarantine-corrected high-resource uplift")
    if strict.get("epicure") != high.get("epicure") or high.get("epicure") != arena_base.get(
        "epicure"
    ):
        raise IntegrityError("base review pools do not share one Epicure identity")
    success_records, candidates, success_arms = _verify_success_arms(
        current_root=current_root, expected_epicure=high["epicure"]
    )
    failures = _verify_residual_failures(current_root)
    success_cost = sum(int(row["source_actual_cost_micros"]) for row in success_records)
    failure_cost = sum(int(row["actual_cost_micros"]) for row in failures)
    aggregate_basis = {
        "parent_audit_sha256": parent.audit["artifact_sha256"],
        "v4_phase1_audit_sha256": phase1["artifact_sha256"],
        "v4_phase2_audit_sha256": phase2["artifact_sha256"],
        "base_strict_pool_sha256": strict["artifact_sha256"],
        "base_high_resource_pool_sha256": high["artifact_sha256"],
        "base_arena_sha256": arena_base["artifact_sha256"],
        "success_source_sha256s": sorted(
            str(row["source"]["artifact_sha256"]) for row in success_records
        ),
        "failure_source_sha256s": sorted(str(row["source"]["artifact_sha256"]) for row in failures),
    }
    aggregate_source_sha256 = sha256_json(aggregate_basis)
    receipt_payload = {
        "schema_version": AGGREGATE_RECEIPT_SCHEMA_VERSION,
        "record_role": "source_reconstructed_terminal_coverage_receipt",
        "status": "terminal_8_usable_5_residual_failures",
        "source_commitment": aggregate_basis,
        "source_commitment_sha256": aggregate_source_sha256,
        "phase_reconstruction": {
            "v4_phase1": {
                "audit_sha256": phase1["artifact_sha256"],
                "usable_cells": phase1["counts"]["usable_complete_cells"],
                "failure_cells": phase1["counts"]["reliability_failure_cells"],
                "actual_cost_micros": phase1["accounting"]["actual_cost_micros"],
            },
            "v4_phase2": {
                "audit_sha256": phase2["artifact_sha256"],
                "usable_cells": phase2["counts"]["usable_complete_cells"],
                "failure_cells": phase2["counts"]["reliability_failure_cells"],
                "actual_cost_micros": phase2["accounting"]["actual_cost_micros"],
            },
        },
        "usable_cells": success_records,
        "residual_failure_cells": failures,
        "counts": {
            "current_endpoint_family_cells": 13,
            "source_reconstructed_usable_cells": len(success_records),
            "source_reconstructed_usable_real_arms": 2 * len(success_records),
            "residual_failure_cells": len(failures),
            "partial_failure_arms_retained_for_reliability": len(failures),
            "synthetic_arms": 0,
            "provider_calls_by_materialization": 0,
            "epicure_calls_by_materialization": 0,
        },
        "accounting": {
            "usable_source_actual_cost_micros": success_cost,
            "failure_source_actual_cost_micros": failure_cost,
            "current_13_cell_source_actual_cost_micros": success_cost + failure_cost,
            "current_13_cell_source_actual_cost_usd": _decimal_text(
                Decimal(success_cost + failure_cost) / Decimal(1_000_000)
            ),
            "v4_phase1_actual_cost_usd": "0.424886",
            "v4_phase2_actual_cost_usd": "0.024079",
        },
        "missingness": {
            "replacement_observations_missing_at_random": False,
            "failure_mechanisms": dict(Counter(row["diagnosis"] for row in failures)),
            "failed_cells_enter_preference_or_uplift_fit": False,
            "failed_cells_enter_reliability_metrics": True,
        },
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "quality_judgments": 0,
            "synthetic_arms": 0,
        },
    }
    receipt = {**receipt_payload, "artifact_sha256": sha256_json(receipt_payload)}
    uplift = _build_corrected_uplift(
        strict=strict,
        high=high,
        candidates=candidates,
        aggregate_source_sha256=aggregate_source_sha256,
    )
    arena = _build_corrected_arena(
        base=arena_base,
        success_arms=success_arms,
        aggregate_source_sha256=aggregate_source_sha256,
    )
    coverage_payload = {
        "schema_version": CORRECTED_COVERAGE_SCHEMA_VERSION,
        "artifact_role": "source_reconstructed_corrected_development_coverage_metrics",
        "source": {
            "aggregate_receipt_sha256": receipt["artifact_sha256"],
            "corrected_uplift_sha256": uplift["artifact_sha256"],
            "corrected_arena_sha256": arena["artifact_sha256"],
            "base_arena_sha256": arena_base["artifact_sha256"],
        },
        "uplift": {
            "retained_pairs": int(strict["observed"]["candidate_pairs"])
            + int(high["observed"]["candidate_pairs"]),
            "pairs_added": 8,
            "pairs_after": int(uplift["observed"]["candidate_pairs"]),
        },
        "model_arena": {
            "comparisons_before": int(arena_base["observed"]["candidate_comparisons"]),
            "comparisons_added": int(
                arena["observed"]["coverage_recovery_candidate_comparisons_added"]
            ),
            "comparisons_after": int(arena["observed"]["candidate_comparisons"]),
            "unique_response_arms_before": int(arena_base["observed"]["source_response_arms"]),
            "unique_response_arms_after": int(arena["observed"]["source_response_arms"]),
            "unpaired_response_arms_after": int(arena["observed"]["unpaired_response_arms"]),
            "model_pair_family_cells": int(arena["observed"]["model_pair_family_cells"]),
            "missing_cells_before": int(arena_base["observed"]["missing_model_pair_family_cells"]),
            "missing_cells_after": int(arena["observed"]["missing_model_pair_family_cells"]),
            "missing_cells_after_by_family": arena["observed"][
                "missing_model_pair_family_cells_by_family"
            ],
        },
        "inference": {
            "raw_comparisons_are_independent": False,
            "cluster_by_task": True,
            "cluster_by_response": True,
            "reason": "915 rows reuse 192 answers; four answers have no same-task peer",
        },
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "family_specific_ranking_supported": False,
            "synthetic_arms": 0,
        },
    }
    coverage = {**coverage_payload, "artifact_sha256": sha256_json(coverage_payload)}
    audit_payload = {
        "schema_version": AGGREGATE_AUDIT_SCHEMA_VERSION,
        "record_role": "independent_reconstruction_audit",
        "decision": "passed_exact_source_reconstruction",
        "receipt_sha256": receipt["artifact_sha256"],
        "corrected_uplift_sha256": uplift["artifact_sha256"],
        "corrected_arena_sha256": arena["artifact_sha256"],
        "corrected_coverage_sha256": coverage["artifact_sha256"],
        "checks": {
            "parent_terminal_audit_reconstructed": True,
            "v4_phase1_terminal_audit_reconstructed": True,
            "v4_phase2_terminal_audit_reconstructed": True,
            "eight_usable_coordinates_resolved_from_raw_sources": True,
            "five_failure_diagnoses_resolved_from_raw_sources": True,
            "all_normalized_response_content_addresses_verified": True,
            "synthetic_arms_zero": True,
            "failed_cells_excluded_from_preference_and_uplift_pools": True,
            "arena_support_recomputed_from_items": True,
            "uplift_counts_recomputed_from_items": True,
        },
        "counts": {
            "usable_cells": 8,
            "residual_failure_cells": 5,
            "corrected_uplift_pairs": 187,
            "corrected_uplift_arms": 374,
            "corrected_arena_comparisons": 915,
            "corrected_arena_unique_arms": 192,
            "corrected_arena_unpaired_arms": 4,
            "missing_model_pair_family_cells": 73,
            "synthetic_arms": 0,
        },
        "claim_boundary": receipt["claim_boundary"],
    }
    audit = {**audit_payload, "artifact_sha256": sha256_json(audit_payload)}
    return {
        "receipt": receipt,
        "audit": audit,
        "uplift": uplift,
        "arena": arena,
        "coverage": coverage,
    }


def _policy_attempt_slots(
    *,
    run_id: str,
    cell_id: str,
    max_tool_rounds: int,
    max_tool_calls_per_round: int,
    max_provider_attempts: int,
) -> list[dict[str, Any]]:
    """Allocate the exact potential on-arm coordinates for one policy.

    The per-round slots cover every legal distribution of the separately
    bounded total call count.  They do not imply that every slot may be used in
    one run.
    """

    arm_id = f"{run_id}:epicure_on"
    coordinates: list[tuple[str, int]] = []
    phases = [
        "planning",
        *(f"tool_round_{index}" for index in range(max_tool_rounds)),
        "final",
    ]
    for phase in phases:
        coordinates.extend((phase, attempt) for attempt in range(max_provider_attempts))
    coordinates.append(("mcp_session", 0))
    for round_index in range(max_tool_rounds):
        for call_index in range(max_tool_calls_per_round):
            coordinates.append((f"mcp_tool_{round_index}_{call_index}", 0))
    return [
        {
            "arm_id": arm_id,
            "phase": phase,
            "attempt_index": attempt_index,
            "attempt_id": str(
                uuid.uuid5(
                    PRIMARY_NAMESPACE,
                    f"{run_id}:{cell_id}:{arm_id}:{phase}:{attempt_index}",
                )
            ),
        }
        for phase, attempt_index in coordinates
    ]


def _arena_coordinate_projection(
    *, corrected_arena: Mapping[str, Any], planned_cells: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    model_order = tuple(str(value) for value in corrected_arena.get("model_order") or [])
    coordinates: set[tuple[str, str, str, str, str]] = set()
    for item in corrected_arena.get("items") or []:
        for side in ("left", "right"):
            arm = item[side]
            coordinates.add(
                (
                    str(item["execution_stratum"]),
                    str(item["task_id"]),
                    str(arm["requested_model_id"]),
                    str(item["execution_policy_sha256"]),
                    str(item["task_family"]),
                )
            )
    # Four source-reconstructed on-arms currently have no same-task peer and
    # therefore do not occur in an arena item.  All were generated with the
    # base high-resource policy; bind them explicitly from the registry.
    for row in corrected_arena.get("unpaired_epicure_on_registry") or []:
        coordinates.add(
            (
                str(row["execution_stratum"]),
                str(row["task_id"]),
                str(row["model_id"]),
                "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d",
                str(row["task_family"]),
            )
        )

    def summarize(rows: set[tuple[str, str, str, str, str]]) -> dict[str, Any]:
        grouped: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        for stratum, task_id, model_id, policy_sha, family in rows:
            grouped[(stratum, task_id, policy_sha, family)].add(model_id)
        counts: Counter[tuple[str, str, str]] = Counter()
        comparisons = 0
        for (_stratum, _task_id, _policy, family), models in grouped.items():
            for left, right in itertools.combinations(sorted(models, key=model_order.index), 2):
                pair = tuple(sorted((left, right)))
                counts[(pair[0], pair[1], family)] += 1
                comparisons += 1
        records = [
            {
                "model_a": model_a,
                "model_b": model_b,
                "task_family": family,
                "candidate_comparisons": counts[(*tuple(sorted((model_a, model_b))), family)],
            }
            for model_a, model_b in itertools.combinations(model_order, 2)
            for family in FAMILIES
        ]
        missing = [row for row in records if row["candidate_comparisons"] == 0]
        by_family = Counter(str(row["task_family"]) for row in missing)
        return {
            "comparisons": comparisons,
            "coordinates": len(rows),
            "model_pair_family_cells": len(records),
            "missing_cells": len(missing),
            "missing_cells_by_family": {family: by_family[family] for family in FAMILIES},
            "cell_records_sha256": sha256_json(records),
            "missing_cell_records": missing,
        }

    before = summarize(coordinates)
    if (
        before["comparisons"] != corrected_arena["observed"]["candidate_comparisons"]
        or before["coordinates"] != corrected_arena["observed"]["source_response_arms"]
        or before["missing_cells"] != corrected_arena["observed"]["missing_model_pair_family_cells"]
    ):
        raise IntegrityError("corrected arena coordinates do not reproduce support metrics")
    projected = set(coordinates)
    for cell in planned_cells:
        coordinate = (
            "high-resource",
            str(cell["task_id"]),
            str(cell["model_id"]),
            str(cell["execution_policy_sha256"]),
            str(cell["task_family"]),
        )
        if coordinate in projected:
            raise IntegrityError("primary on-arm plan reuses an existing arena coordinate")
        projected.add(coordinate)
    after = summarize(projected)
    if after["missing_cells"] != 0:
        raise IntegrityError("primary on-arm schedule does not project zero support holes")
    return {"before": before, "projected_after_all_usable": after}


def build_primary_on_plan(
    *,
    project_root: Path,
    parent_root: Path,
    exposure_root: Path,
    corrected_arena_path: Path,
    corrected_coverage_path: Path,
    aggregate_audit_path: Path,
) -> dict[str, Any]:
    parent = reconstruct_parent(project_root=project_root, parent_root=parent_root)
    corrected_arena = _load_addressed(
        corrected_arena_path,
        label="source-reconstructed corrected arena",
        expected_schema=CORRECTED_ARENA_SCHEMA_VERSION,
        expected_digest=CORRECTED_ARENA_SHA256,
    )
    corrected_coverage = _load_addressed(
        corrected_coverage_path,
        label="source-reconstructed corrected coverage",
        expected_schema=CORRECTED_COVERAGE_SCHEMA_VERSION,
        expected_digest=CORRECTED_COVERAGE_SHA256,
    )
    aggregate_audit = _load_addressed(
        aggregate_audit_path,
        label="source-reconstructed aggregate audit",
        expected_schema=AGGREGATE_AUDIT_SCHEMA_VERSION,
        expected_digest=AGGREGATE_AUDIT_SHA256,
    )
    if (
        aggregate_audit.get("decision") != "passed_exact_source_reconstruction"
        or aggregate_audit.get("corrected_arena_sha256") != corrected_arena["artifact_sha256"]
        or aggregate_audit.get("corrected_coverage_sha256") != corrected_coverage["artifact_sha256"]
        or corrected_coverage.get("model_arena", {}).get("missing_cells_after") != 73
    ):
        raise IntegrityError("primary plan support inputs do not form one passing chain")
    model_order = tuple(str(value) for value in corrected_arena.get("model_order") or [])
    task_path, manifest_paths = _task_and_manifest_inputs(project_root=project_root, parent=parent)
    tasks, task_source = load_development_task_inventory(task_path)
    task_index = {task.public_id: task for task in tasks}
    registry_sha = task_registry_sha256(tasks)
    candidates = _candidate_index(manifest_paths)
    if set(candidates) != set(model_order):
        raise IntegrityError("primary route manifests do not exactly cover the 16-model panel")
    selected_task_ids = frozenset(task_id for task_id, _ in RESIDUAL_TASK_MODELS)
    exposure = _actual_exposure_snapshot(
        exposure_root,
        model_ids=frozenset(model_order),
        task_ids=selected_task_ids,
    )
    observed_exposure: dict[str, set[str]] = {task_id: set() for task_id in selected_task_ids}
    for row in exposure["coordinates"]:
        observed_exposure[str(row["task_id"])].add(str(row["model_id"]))
    if any(
        frozenset(observed_exposure[task_id]) != expected
        for task_id, expected in EXPECTED_PRIOR_EXPOSURES.items()
    ):
        raise IntegrityError("primary task exposure snapshot changed")
    prior_identifiers = _collect_prior_identifiers(exposure_root)
    policy = parent.bundle.execution_policy
    if (
        policy.sha256 != "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d"
        or policy.max_tool_rounds != 3
        or policy.max_tool_calls_per_round != 6
        or policy.max_tool_calls_total != 12
        or policy.max_intermediate_tokens != 8192
        or policy.max_output_tokens != 8192
    ):
        raise IntegrityError("primary plan requires the exact comparable high-resource policy")
    new_identifiers: set[str] = set()
    cells: list[dict[str, Any]] = []
    ordinal = 0
    exposed_coordinates = {
        (str(row["model_id"]), str(row["task_id"])) for row in exposure["coordinates"]
    }
    for task_id, selector in RESIDUAL_TASK_MODELS:
        task = task_index[task_id]
        if selector == "all":
            selected_models = model_order
        elif selector == "all_except_minimax":
            selected_models = tuple(model for model in model_order if model != "minimax/minimax-m3")
        elif selector == "all_except_deepseek_pro":
            selected_models = tuple(
                model for model in model_order if model != "deepseek/deepseek-v4-pro"
            )
        else:
            selected_models = tuple(selector)
        for model_id in selected_models:
            if (model_id, task_id) in exposed_coordinates:
                raise IntegrityError("primary plan would replay an exposed model-task coordinate")
            ordinal += 1
            candidate, manifest_path, manifest_sha = candidates[model_id]
            basis = {
                "schema_version": PRIMARY_PLAN_SCHEMA_VERSION,
                "ordinal": ordinal,
                "model_id": model_id,
                "provider_tag": candidate.provider_tag,
                "execution_backend": candidate.execution_backend,
                "route_manifest_sha256": manifest_sha,
                "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
                "task_id": task.public_id,
                "task_family": task.family,
                "prompt_sha256": task.prompt_sha256,
                "conditions": ["epicure_on"],
                "execution_policy_sha256": policy.sha256,
                "development_only": True,
                "fresh_no_replay": True,
            }
            cell_id = sha256_json({**basis, "identifier_role": "cell"})
            work_id = sha256_json({**basis, "identifier_role": "work_item"})
            run_id = str(uuid.uuid5(PRIMARY_NAMESPACE, f"run:{cell_id}"))
            arm_id = sha256_json(
                {
                    "schema_version": PRIMARY_PLAN_SCHEMA_VERSION,
                    "work_item_id": work_id,
                    "condition": "epicure_on",
                }
            )
            attempts = _policy_attempt_slots(
                run_id=run_id,
                cell_id=cell_id,
                max_tool_rounds=policy.max_tool_rounds,
                max_tool_calls_per_round=policy.max_tool_calls_per_round,
                max_provider_attempts=policy.max_provider_attempts,
            )
            identifiers = {
                cell_id,
                work_id,
                run_id,
                arm_id,
                *(str(slot["attempt_id"]) for slot in attempts),
            }
            if prior_identifiers & identifiers or new_identifiers & identifiers:
                raise IntegrityError("primary identifier overlaps rejected, closed, or new work")
            new_identifiers.update(identifiers)
            work_item = WorkItem(
                ordinal=ordinal,
                work_item_id=work_id,
                manifest_sha256=manifest_sha,
                task_registry_sha256=registry_sha,
                task=task,
                candidate=candidate,
                endpoint_execution_sha256=candidate.endpoint_execution_sha256,
                execution_policy_sha256=policy.sha256,
                execution_policy=policy,
            )
            forecast = derive_conditions_forecast(
                work_item, policy=policy, conditions=("epicure_on",)
            )
            cells.append(
                {
                    **basis,
                    "cell_id": cell_id,
                    "work_item_id": work_id,
                    "run_id": run_id,
                    "arm_ids": {"epicure_on": arm_id},
                    "attempt_slots": attempts,
                    "attempt_slots_sha256": sha256_json(attempts),
                    "attempt_slot_contract": {
                        "conditions": ["epicure_on"],
                        "provider_phases": [
                            "planning",
                            "tool_round_0",
                            "tool_round_1",
                            "tool_round_2",
                            "final",
                        ],
                        "provider_attempts_per_phase": 2,
                        "mcp_sessions": 1,
                        "potential_mcp_slots_per_round": 6,
                        "tool_rounds": 3,
                        "total_tool_call_cap": 12,
                        "attempt_slot_count": len(attempts),
                    },
                    "route_manifest_path": _relative(project_root, manifest_path),
                    "reserved_worst_case_usd": _decimal_text(forecast.forecast_usd),
                    "forecast": {
                        "conditions": ["epicure_on"],
                        "request_bound": forecast.request_bound,
                        "total_completion_tokens_bound": forecast.total_completion_tokens_bound,
                        "prompt_tokens_per_request_bound": _decimal_text(
                            forecast.prompt_tokens_per_request_bound
                        ),
                    },
                    "no_prior_model_task_exposure_at_freeze": True,
                    "fresh_identifiers_disjoint_from_all_prior_records": True,
                    "official_fit_eligible": False,
                }
            )
    if len(cells) != 50:
        raise IntegrityError("primary plan is not exactly 50 fresh Epicure-on arms")
    support = _arena_coordinate_projection(corrected_arena=corrected_arena, planned_cells=cells)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_model[str(cell["model_id"])].append(cell)
    batch_rows: list[dict[str, Any]] = []
    for model_id in model_order:
        model_cells = sorted(by_model[model_id], key=lambda row: str(row["task_family"]))
        batch_rows.append(
            {
                "model_id": model_id,
                "provider_tag": model_cells[0]["provider_tag"],
                "execution_backend": model_cells[0]["execution_backend"],
                "cell_count": len(model_cells),
                "task_families": [str(row["task_family"]) for row in model_cells],
                "work_item_ids": [str(row["work_item_id"]) for row in model_cells],
                "worst_case_usd": _decimal_text(
                    sum(
                        (Decimal(str(row["reserved_worst_case_usd"])) for row in model_cells),
                        Decimal(0),
                    )
                ),
                "one_active_batch_at_a_time": True,
                "fresh_rebase_required_before_reservation": True,
                "separate_ledger_and_terminal_closure": True,
            }
        )
    # Deterministically zig-zag across the cost distribution.  Every endpoint
    # batch already mixes all families available to that model, so this avoids
    # a cost- or family-ordered acquisition wave while remaining reproducible.
    seed = corrected_coverage["artifact_sha256"]
    cost_sorted = sorted(
        batch_rows,
        key=lambda row: (
            Decimal(str(row["worst_case_usd"])),
            hashlib.sha256(f"{seed}:{row['model_id']}:batch-order".encode()).hexdigest(),
        ),
    )
    balanced_order: list[dict[str, Any]] = []
    left, right = 0, len(cost_sorted) - 1
    take_high = int(seed[:2], 16) % 2 == 1
    while left <= right:
        if take_high:
            balanced_order.append(cost_sorted[right])
            right -= 1
        else:
            balanced_order.append(cost_sorted[left])
            left += 1
        take_high = not take_high
    for index, batch in enumerate(balanced_order, start=1):
        batch["execution_ordinal"] = index
        batch["batch_id"] = sha256_json(
            {
                "schema_version": PRIMARY_PLAN_SCHEMA_VERSION,
                "execution_ordinal": index,
                "model_id": batch["model_id"],
                "work_item_ids": batch["work_item_ids"],
            }
        )
    anchor_gates = []
    for task_id, _selector in RESIDUAL_TASK_MODELS:
        task_cells = [cell for cell in cells if cell["task_id"] == task_id]
        retained = sorted(EXPECTED_PRIOR_EXPOSURES[task_id])
        anchor_gates.append(
            {
                "task_id": task_id,
                "task_family": task_index[task_id].family,
                "new_work_item_ids": [cell["work_item_id"] for cell in task_cells],
                "new_arm_count": len(task_cells),
                "retained_compatible_anchor_models": retained,
                "complete_anchor_model_count": len(task_cells) + len(retained),
                "support_claim_gate": (
                    "all new work items source-reconstructed usable under the exact primary "
                    "policy before recomputing this family's holes"
                ),
                "partial_anchor_support_claim_permitted": False,
            }
        )
    total = sum((Decimal(str(cell["reserved_worst_case_usd"])) for cell in cells), Decimal(0))
    payload = {
        "schema_version": PRIMARY_PLAN_SCHEMA_VERSION,
        "status": "frozen_blocked_pending_route_gate_and_rebased_admission",
        "purpose": "primary_comparable_epicure_on_connectivity_repair",
        "sources": {
            "aggregate_audit": {
                "path": _relative(project_root, aggregate_audit_path),
                "artifact_sha256": aggregate_audit["artifact_sha256"],
                "physical_sha256": _file_sha256(aggregate_audit_path),
            },
            "corrected_arena": {
                "path": _relative(project_root, corrected_arena_path),
                "artifact_sha256": corrected_arena["artifact_sha256"],
                "physical_sha256": _file_sha256(corrected_arena_path),
            },
            "corrected_coverage": {
                "path": _relative(project_root, corrected_coverage_path),
                "artifact_sha256": corrected_coverage["artifact_sha256"],
                "physical_sha256": _file_sha256(corrected_coverage_path),
            },
            "task_validity_sha256": task_source["artifact_sha256"],
            "task_registry_sha256": registry_sha,
            "route_manifest_sha256s": sorted(
                str(load_candidate_manifest(path, expected_digest="")["content_address"]["digest"])
                for path in manifest_paths
            ),
            "exposure_snapshot": exposure,
            "prior_identifier_count": len(prior_identifiers),
            "prior_identifiers_sha256": sha256_json(sorted(prior_identifiers)),
            "rejected_predecessor_plan_sha256": (
                "e3f96fb63a17e36d43fd33635b43b473ccc477a9e3457c809ceca2b439a8c59d"
            ),
        },
        "primary_protocol": {
            "one_uniform_execution_policy": True,
            "execution_policy": policy.document(),
            "execution_policy_sha256": policy.sha256,
            "conditions": ["epicure_on"],
            "max_tool_rounds": 3,
            "max_tool_calls_per_round": 6,
            "max_tool_calls_total": 12,
            "max_intermediate_tokens": 8192,
            "max_final_tokens": 8192,
            "diagnostic_13_by_13_glm_cells_in_primary": 0,
            "diagnostic_16k_evidence_cells_in_primary": 0,
            "protocol_incommensurable_arms_pooled": False,
        },
        "support_reconstruction": support,
        "cells": cells,
        "endpoint_isolated_batches_in_balanced_order": balanced_order,
        "batch_order_policy": {
            "method": "deterministic_cost_zigzag_with_family_mixed_endpoint_batches_v1",
            "seed_sha256": seed,
            "quality_outcomes_used": 0,
            "one_active_endpoint_batch_at_a_time": True,
        },
        "anchor_completeness_gates": anchor_gates,
        "counts": {
            "primary_fresh_cells": 50,
            "primary_fresh_real_arms": 50,
            "primary_epicure_on_arms": 50,
            "primary_epicure_off_arms": 0,
            "endpoint_isolated_batches": 16,
            "anchor_tasks": 4,
            "synthetic_arms": 0,
            "provider_calls_by_freeze": 0,
            "epicure_calls_by_freeze": 0,
        },
        "budget": {
            "currency": "USD",
            "primary_worst_case_usd": _decimal_text(total),
            "aggregate_reservation_on_freeze_usd": "0",
            "transactional_reservation_unit": "one_endpoint_isolated_batch",
            "one_active_batch_at_a_time": True,
            "each_batch_requires_current_exposure_rebase": True,
            "admission_ceiling_fraction": "0.85",
            "hard_cap_usd": "100",
            "admission_granted_by_plan": False,
        },
        "route_gates": {
            "cohere_transport_contract_sha256": sha256_json(COHERE_CONTINUATION_CONTRACT),
            "cohere_projection_contract_test_required": True,
            "cohere_route_admission_artifact": None,
            "all_routes_blocked_until_separate_preflight": True,
        },
        "diagnostic_recovery": {
            "glm_13_by_13": "excluded_from_primary_and_not_authorized",
            "mistral_grok_16k_evidence": "excluded_from_primary_and_not_authorized",
            "may_close_primary_support_holes": False,
            "execution_before_governance_reaudit_permitted": False,
        },
        "missingness": {
            "all_50_primary_arms_usable_assumed_by_projection": True,
            "projection_is_not_observed_support": True,
            "partial_or_failed_arms_retained_for_reliability_only": True,
            "anchor_failure_missing_at_random": False,
        },
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "quality_judgments": 0,
            "family_specific_ranking_supported": False,
            "permitted_analysis": "connectivity_and_reliability_diagnostics_only",
            "synthetic_arms": 0,
        },
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _cohere_projection_contract() -> dict[str, Any]:
    """Exercise the transport projection without contacting Cohere."""

    from .provider import _assistant_continuation_message
    from .service_cohere import _openai_response, _request_payload

    opaque_content = [
        {"type": "thinking", "thinking": "opaque fixture", "signature": "sig-fixture"},
        {"type": "text", "text": "Visible note."},
    ]
    normalized = _openai_response(
        {
            "id": "offline-projection",
            "finish_reason": "TOOL_CALL",
            "message": {
                "role": "assistant",
                "content": opaque_content,
                "tool_plan": "Use Epicure once.",
                "tool_calls": [
                    {
                        "id": "fixture-call-id",
                        "type": "function",
                        "function": {
                            "name": "find_pairings",
                            "arguments": '{"ingredients":["pear","miso"]}',
                        },
                    }
                ],
            },
            "usage": {"tokens": {"input_tokens": 1, "output_tokens": 1}},
        },
        response_model="command-a-reasoning-08-2025",
    )
    assistant = normalized["choices"][0]["message"]
    continuation = _assistant_continuation_message(
        assistant, empty_content_fallback="No visible note."
    )
    replay = _request_payload(
        {
            "model": "command-a-reasoning-08-2025",
            "messages": [
                {"role": "user", "content": "Find a pairing."},
                {**continuation, "tool_calls": assistant["tool_calls"]},
                {
                    "role": "tool",
                    "tool_call_id": "fixture-call-id",
                    "name": "find_pairings",
                    "content": "fixture result",
                },
            ],
        }
    )
    replayed_assistant = replay["messages"][1]
    replayed_tool = replay["messages"][2]
    checks = {
        "opaque_content_blocks_preserved": replayed_assistant.get("content") == opaque_content,
        "visible_text_excludes_thinking": assistant.get("content") == "Visible note."
        and "opaque fixture" not in str(assistant.get("content")),
        "tool_plan_preserved": replayed_assistant.get("tool_plan") == "Use Epicure once.",
        "assistant_tool_call_id_preserved": (
            (replayed_assistant.get("tool_calls") or [{}])[0].get("id") == "fixture-call-id"
        ),
        "tool_result_id_bound": replayed_tool.get("tool_call_id") == "fixture-call-id",
    }
    if not all(checks.values()):
        raise IntegrityError("offline Cohere continuation projection failed")
    projection = {
        "checks": checks,
        "visible_content_sha256": sha256_json(assistant["content"]),
        "opaque_content_sha256": sha256_json(opaque_content),
        "replayed_assistant_sha256": sha256_json(replayed_assistant),
        "replayed_tool_sha256": sha256_json(replayed_tool),
    }
    return {**projection, "projection_sha256": sha256_json(projection)}


def _run_cohere_pytest(project_root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_service_cohere.py"]
    result = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    match = re.search(r"(?P<count>\d+) passed", output)
    if result.returncode != 0 or match is None or int(match.group("count")) < 5:
        raise IntegrityError("offline Cohere contract pytest gate failed")
    stable_summary = re.sub(r" in [0-9.]+s", " in <elapsed>s", output)
    return {
        "command": "PYTHONPATH=src .venv/bin/pytest -q tests/test_service_cohere.py",
        "exit_code": result.returncode,
        "passed_tests": int(match.group("count")),
        "stable_summary_sha256": hashlib.sha256(stable_summary.encode()).hexdigest(),
        "provider_calls": 0,
        "epicure_calls": 0,
    }


def _file_binding(project_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(project_root, path),
        "physical_sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _load_truncated_addressed(path: Path, *, label: str, expected_digest: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IntegrityError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} is not an object")
    digest = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    source_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        digest != expected_digest
        or source_digest != expected_digest
        or expected_digest[:12] not in path.name
    ):
        raise IntegrityError(f"{label} content address does not verify")
    return value


def build_cohere_route_gate(
    *,
    project_root: Path,
    primary_plan_path: Path,
    pytest_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _load_addressed(
        primary_plan_path,
        label="primary Epicure-on coverage plan",
        expected_schema=PRIMARY_PLAN_SCHEMA_VERSION,
        expected_digest=PRIMARY_PLAN_SHA256,
    )
    contract_sha = sha256_json(COHERE_CONTINUATION_CONTRACT)
    if (
        contract_sha != "fd8e888da053b40de27fd006315df3535ac7934aafcdeba40363d3e114e81b0a"
        or plan.get("route_gates", {}).get("cohere_transport_contract_sha256") != contract_sha
        or int(pytest_evidence.get("exit_code", 1)) != 0
        or int(pytest_evidence.get("passed_tests", 0)) < 5
    ):
        raise IntegrityError("Cohere route-gate inputs do not verify")
    projection = _cohere_projection_contract()
    source_files = [
        project_root / "src/flavourbench/service_cohere.py",
        project_root / "src/flavourbench/provider.py",
        project_root / "tests/test_service_cohere.py",
    ]
    cohere_cells = [cell for cell in plan["cells"] if str(cell["model_id"]).startswith("cohere/")]
    if len(cohere_cells) != 8:
        raise IntegrityError("primary plan Cohere cell count changed")
    payload = {
        "schema_version": COHERE_ROUTE_GATE_SCHEMA_VERSION,
        "status": "passed_offline_transport_contract_only",
        "record_role": "content_addressed_cohere_staged_continuation_route_gate",
        "primary_plan": {
            **_file_binding(project_root, primary_plan_path),
            "artifact_sha256": plan["artifact_sha256"],
            "cohere_cells": len(cohere_cells),
            "cohere_work_item_ids": sorted(cell["work_item_id"] for cell in cohere_cells),
        },
        "contract": {
            "document": COHERE_CONTINUATION_CONTRACT,
            "sha256": contract_sha,
            "projection": projection,
        },
        "implementation": {
            "files": [_file_binding(project_root, path) for path in source_files],
            "bundle_sha256": sha256_json(
                [_file_binding(project_root, path) for path in source_files]
            ),
        },
        "verification": dict(pytest_evidence),
        "decision": {
            "cohere_transport_contract_qualified": True,
            "provider_connectivity_tested": False,
            "provider_semantics_claimed": False,
            "paid_execution_admission_granted": False,
            "independent_governance_go_still_required": True,
        },
        "calls": {"provider": 0, "epicure": 0},
        "claim_boundary": plan["claim_boundary"],
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _live_source_inventory(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != "flavourbench-live-smoke-v1"
        ):
            continue
        digest = value.get("artifact_sha256")
        payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
        source_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if not isinstance(digest, str) or source_digest != digest:
            raise IntegrityError("live source inventory contains an invalid content address")
        records.append(
            {
                "path": str(path.relative_to(root)),
                "artifact_sha256": digest,
                "completed_at": str(value.get("completed_at") or ""),
                "actual_cost_micros": int(value.get("budget", {}).get("actual_cost_micros") or 0),
            }
        )
    if not records:
        raise IntegrityError("live source inventory is empty")
    return {
        "artifact_count": len(records),
        "records_sha256": sha256_json(records),
        "latest_completed_at": max(row["completed_at"] for row in records),
    }


def build_primary_preflight(
    *,
    project_root: Path,
    current_root: Path,
    primary_plan_path: Path,
    cohere_gate_path: Path,
) -> dict[str, Any]:
    primary = _load_addressed(
        primary_plan_path,
        label="primary Epicure-on coverage plan",
        expected_schema=PRIMARY_PLAN_SCHEMA_VERSION,
        expected_digest=PRIMARY_PLAN_SHA256,
    )
    gate = _load_addressed(
        cohere_gate_path,
        label="Cohere continuation route gate",
        expected_schema=COHERE_ROUTE_GATE_SCHEMA_VERSION,
    )
    if (
        gate.get("primary_plan", {}).get("artifact_sha256") != primary["artifact_sha256"]
        or gate.get("decision", {}).get("cohere_transport_contract_qualified") is not True
    ):
        raise IntegrityError("Cohere route gate is not bound to the primary plan")
    for source in gate.get("implementation", {}).get("files", []):
        path = project_root / str(source.get("path") or "")
        if _file_binding(project_root, path) != dict(source):
            raise IntegrityError("Cohere route-gated implementation changed")

    v4_root = current_root / "frontier-coverage-recovery-v4"
    v6_root = current_root / "reasoning-effort-sensitivity-v6"
    paths = {
        "v4_preflight": v4_root
        / f"frontier-coverage-recovery-v4-preflight-{V4_PREFLIGHT_SHA256}.json",
        "v4_phase1_audit": v4_root
        / f"frontier-coverage-recovery-v4-untouched_recovery-audit-{V4_PHASE1_AUDIT_SHA256}.json",
        "v4_phase2_audit": v4_root
        / (
            "frontier-coverage-recovery-v4-glm_specific_replacement-audit-"
            f"{V4_PHASE2_AUDIT_SHA256}.json"
        ),
        "v6_route_plan": v6_root
        / "route-gate"
        / f"reasoning-effort-v6-sonnet-route-gate-plan-{V6_ROUTE_PLAN_SHA256}.json",
        "v6_execution_plan": v6_root
        / "route-gate"
        / f"reasoning-effort-v6-sonnet-execution-plan-{V6_EXECUTION_PLAN_SHA256}.json",
        "v6_audit": v6_root
        / "sonnet/audits"
        / f"reasoning-effort-v6-sonnet-audit-{V6_AUDIT_SHA256}.json",
        "v6_closure": v6_root
        / "sonnet/closures"
        / f"reasoning-effort-v6-sonnet-closure-{V6_CLOSURE_SHA256}.json",
        "v6_receipt": v6_root
        / "sonnet/receipts"
        / f"reasoning-effort-v6-sonnet-receipt-{V6_RECEIPT_SHA256}.json",
        "v6_aggregate_audit": v6_root
        / "aggregate"
        / f"reasoning-effort-v6-aggregate-audit-{V6_AGGREGATE_AUDIT_SHA256}.json",
        "v6_aggregate_closure": v6_root
        / "aggregate"
        / f"reasoning-effort-v6-aggregate-closure-{V6_AGGREGATE_CLOSURE_SHA256}.json",
    }
    expected = {
        "v4_preflight": V4_PREFLIGHT_SHA256,
        "v4_phase1_audit": V4_PHASE1_AUDIT_SHA256,
        "v4_phase2_audit": V4_PHASE2_AUDIT_SHA256,
        "v6_route_plan": V6_ROUTE_PLAN_SHA256,
        "v6_execution_plan": V6_EXECUTION_PLAN_SHA256,
        "v6_audit": V6_AUDIT_SHA256,
        "v6_closure": V6_CLOSURE_SHA256,
        "v6_receipt": V6_RECEIPT_SHA256,
        "v6_aggregate_audit": V6_AGGREGATE_AUDIT_SHA256,
        "v6_aggregate_closure": V6_AGGREGATE_CLOSURE_SHA256,
    }
    docs = {
        label: _load_addressed(path, label=label, expected_digest=expected[label])
        for label, path in paths.items()
    }
    source_paths = [
        next((v6_root / "sonnet/source").glob(f"*-{digest[:12]}.json"))
        for digest in V6_SOURCE_SHA256S
    ]
    sources = [
        _load_truncated_addressed(path, label="v6 source", expected_digest=digest)
        for path, digest in zip(source_paths, V6_SOURCE_SHA256S, strict=True)
    ]
    v4_preflight = docs["v4_preflight"]
    phase1 = docs["v4_phase1_audit"]
    phase2 = docs["v4_phase2_audit"]
    v6_audit = docs["v6_audit"]
    v6_receipt = docs["v6_receipt"]
    if (
        v4_preflight.get("status") != "admissible_zero_call_preflight"
        or phase1.get("preflight_sha256") != V4_PREFLIGHT_SHA256
        or phase2.get("preflight_sha256") != V4_PREFLIGHT_SHA256
        or int(phase1.get("accounting", {}).get("actual_cost_micros") or -1) != 424886
        or int(phase2.get("accounting", {}).get("actual_cost_micros") or -1) != 24079
        or docs["v6_execution_plan"].get("route_plan_sha256") != V6_ROUTE_PLAN_SHA256
        or v6_receipt.get("route_plan_sha256") != V6_ROUTE_PLAN_SHA256
        or docs["v6_closure"].get("audit_sha256") != V6_AUDIT_SHA256
        or docs["v6_aggregate_audit"]
        .get("inputs", {})
        .get("sonnet_v6_audit", {})
        .get("artifact_sha256")
        != V6_AUDIT_SHA256
        or docs["v6_aggregate_closure"].get("aggregate_audit_sha256") != V6_AGGREGATE_AUDIT_SHA256
        or v6_audit.get("decision") != "passed_all_predicates"
        or int(v6_audit.get("accounting", {}).get("actual_cost_micros") or -1) != 244312
        or sum(int(source["budget"]["actual_cost_micros"]) for source in sources) != 244312
        or {row["artifact_sha256"] for row in v6_receipt.get("source_artifacts", [])}
        != set(V6_SOURCE_SHA256S)
    ):
        raise IntegrityError("post-v4 or Sonnet-v6 accounting chain does not verify")

    baseline = Decimal(str(v4_preflight["budget"]["current_total_exposure_usd"]))
    phase1_cost = Decimal(424886) / Decimal(1_000_000)
    phase2_cost = Decimal(24079) / Decimal(1_000_000)
    v6_cost = Decimal(244312) / Decimal(1_000_000)
    current = baseline + phase1_cost + phase2_cost + v6_cost
    primary_worst = Decimal(str(primary["budget"]["primary_worst_case_usd"]))
    projected = current + primary_worst
    ceiling = Decimal("85")
    hard_cap = Decimal("100")
    if current != Decimal("48.01944682666666666666666666") or projected > ceiling:
        raise IntegrityError("primary plan no longer fits the rebased budget envelope")
    inventory = _live_source_inventory(current_root)
    if inventory["latest_completed_at"] != max(source["completed_at"] for source in sources):
        raise IntegrityError("a live source newer than the bound Sonnet-v6 sources exists")
    max_batch = max(
        Decimal(str(batch["worst_case_usd"]))
        for batch in primary["endpoint_isolated_batches_in_balanced_order"]
    )
    payload = {
        "schema_version": PRIMARY_PREFLIGHT_SCHEMA_VERSION,
        "status": "budget_fits_but_blocked_pending_independent_governance_go",
        "primary_plan": {
            **_file_binding(project_root, primary_plan_path),
            "artifact_sha256": primary["artifact_sha256"],
        },
        "cohere_route_gate": {
            **_file_binding(project_root, cohere_gate_path),
            "artifact_sha256": gate["artifact_sha256"],
        },
        "accounting_chain": {
            label: {
                **_file_binding(project_root, paths[label]),
                "artifact_sha256": docs[label]["artifact_sha256"],
            }
            for label in paths
        },
        "v6_sources": [
            {
                **_file_binding(project_root, path),
                "artifact_sha256": source["artifact_sha256"],
                "actual_cost_micros": source["budget"]["actual_cost_micros"],
            }
            for path, source in zip(source_paths, sources, strict=True)
        ],
        "live_source_snapshot": inventory,
        "budget": {
            "currency": "USD",
            "historical_v4_preflight_exposure_usd": _decimal_text(baseline),
            "v4_phase1_actual_cost_usd": _decimal_text(phase1_cost),
            "v4_phase2_actual_cost_usd": _decimal_text(phase2_cost),
            "sonnet_v6_actual_cost_usd": _decimal_text(v6_cost),
            "rebased_current_exposure_usd": _decimal_text(current),
            "primary_worst_case_usd": _decimal_text(primary_worst),
            "projected_total_exposure_usd": _decimal_text(projected),
            "headroom_to_admission_ceiling_usd": _decimal_text(ceiling - projected),
            "admission_ceiling_usd": _decimal_text(ceiling),
            "hard_cap_usd": _decimal_text(hard_cap),
            "largest_endpoint_batch_worst_case_usd": _decimal_text(max_batch),
            "mathematical_budget_fit": True,
            "admission_granted": False,
        },
        "execution_gate": {
            "one_active_endpoint_batch_at_a_time": True,
            "fresh_source_and_ledger_rebase_before_every_reservation": True,
            "complete_anchor_batch_required_before_support_claim": True,
            "independent_governance_go_required": True,
            "provider_or_mcp_call_permitted_by_this_artifact": False,
        },
        "support": {
            "observed_missing_cells": 73,
            "projected_missing_cells_if_all_50_arms_usable": 0,
            "projection_is_observed": False,
        },
        "calls": {"provider": 0, "epicure": 0},
        "claim_boundary": primary["claim_boundary"],
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-residual")
    freeze.add_argument("--project-root", type=Path, default=Path("."))
    freeze.add_argument(
        "--parent-root",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1"
        ),
    )
    freeze.add_argument(
        "--exposure-root",
        type=Path,
        default=Path("artifacts/season1/current-quality-run"),
    )
    freeze.add_argument("--arena", type=Path, default=Path(BASE_ARENA_RELATIVE))
    freeze.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/season1/current-quality-run/frontier-coverage-residual-v5"),
    )
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--project-root", type=Path, default=Path("."))
    materialize.add_argument(
        "--current-root",
        type=Path,
        default=Path("artifacts/season1/current-quality-run"),
    )
    materialize.add_argument(
        "--parent-root",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1"
        ),
    )
    materialize.add_argument(
        "--v4-root",
        type=Path,
        default=Path("artifacts/season1/current-quality-run/frontier-coverage-recovery-v4"),
    )
    materialize.add_argument("--arena", type=Path, default=Path(BASE_ARENA_RELATIVE))
    materialize.add_argument("--strict", type=Path, default=Path(STRICT_POOL_RELATIVE))
    materialize.add_argument("--high", type=Path, default=Path(HIGH_POOL_RELATIVE))
    materialize.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/season1/current-quality-run/frontier-coverage-v4-postrun"),
    )
    primary = subparsers.add_parser("freeze-primary-on")
    primary.add_argument("--project-root", type=Path, default=Path("."))
    primary.add_argument(
        "--parent-root",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1"
        ),
    )
    primary.add_argument(
        "--exposure-root",
        type=Path,
        default=Path("artifacts/season1/current-quality-run"),
    )
    primary.add_argument(
        "--corrected-arena",
        type=Path,
        default=Path(
            f"{POSTRUN_RELATIVE}/frontier-corrected-development-arena-{CORRECTED_ARENA_SHA256}.json"
        ),
    )
    primary.add_argument(
        "--corrected-coverage",
        type=Path,
        default=Path(
            f"{POSTRUN_RELATIVE}/frontier-corrected-development-coverage-"
            f"{CORRECTED_COVERAGE_SHA256}.json"
        ),
    )
    primary.add_argument(
        "--aggregate-audit",
        type=Path,
        default=Path(
            f"{POSTRUN_RELATIVE}/frontier-coverage-v4-aggregate-audit-{AGGREGATE_AUDIT_SHA256}.json"
        ),
    )
    primary.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5"),
    )
    gate = subparsers.add_parser("freeze-cohere-gate")
    gate.add_argument("--project-root", type=Path, default=Path("."))
    gate.add_argument("--primary-plan", type=Path, default=Path(PRIMARY_PLAN_RELATIVE))
    gate.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/route-gate"
        ),
    )
    preflight = subparsers.add_parser("freeze-primary-preflight")
    preflight.add_argument("--project-root", type=Path, default=Path("."))
    preflight.add_argument(
        "--current-root",
        type=Path,
        default=Path("artifacts/season1/current-quality-run"),
    )
    preflight.add_argument("--primary-plan", type=Path, default=Path(PRIMARY_PLAN_RELATIVE))
    preflight.add_argument("--cohere-gate", type=Path, required=True)
    preflight.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/preflight"
        ),
    )
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command in {"freeze-cohere-gate", "freeze-primary-preflight"}:
        project_root = args.project_root.resolve()

        def bound(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()

        if args.command == "freeze-cohere-gate":
            evidence = _run_cohere_pytest(project_root)
            gate = build_cohere_route_gate(
                project_root=project_root,
                primary_plan_path=bound(args.primary_plan),
                pytest_evidence=evidence,
            )
            path = _write_addressed(
                {key: value for key, value in gate.items() if key != "artifact_sha256"},
                directory=bound(args.output_dir),
                prefix="frontier-coverage-primary-cohere-route-gate",
            )
            print(
                json.dumps(
                    {
                        "status": gate["status"],
                        "path": str(path),
                        "artifactSha256": gate["artifact_sha256"],
                        "passedTests": gate["verification"]["passed_tests"],
                        "providerCalls": 0,
                        "epicureCalls": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        preflight = build_primary_preflight(
            project_root=project_root,
            current_root=bound(args.current_root),
            primary_plan_path=bound(args.primary_plan),
            cohere_gate_path=bound(args.cohere_gate),
        )
        path = _write_addressed(
            {key: value for key, value in preflight.items() if key != "artifact_sha256"},
            directory=bound(args.output_dir),
            prefix="frontier-coverage-primary-preflight",
        )
        print(
            json.dumps(
                {
                    "status": preflight["status"],
                    "path": str(path),
                    "artifactSha256": preflight["artifact_sha256"],
                    "budget": preflight["budget"],
                    "providerCalls": 0,
                    "epicureCalls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "freeze-primary-on":
        project_root = args.project_root.resolve()

        def bound(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()

        plan = build_primary_on_plan(
            project_root=project_root,
            parent_root=bound(args.parent_root),
            exposure_root=bound(args.exposure_root),
            corrected_arena_path=bound(args.corrected_arena),
            corrected_coverage_path=bound(args.corrected_coverage),
            aggregate_audit_path=bound(args.aggregate_audit),
        )
        path = _write_addressed(
            {key: value for key, value in plan.items() if key != "artifact_sha256"},
            directory=bound(args.output_dir),
            prefix="frontier-coverage-primary-on-v5-plan",
        )
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "path": str(path),
                    "artifactSha256": plan["artifact_sha256"],
                    "freshRealArms": plan["counts"]["primary_fresh_real_arms"],
                    "worstCaseUsd": plan["budget"]["primary_worst_case_usd"],
                    "supportBefore": plan["support_reconstruction"]["before"],
                    "projectedSupportAfter": plan["support_reconstruction"][
                        "projected_after_all_usable"
                    ],
                    "providerCalls": 0,
                    "epicureCalls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "materialize":
        project_root = args.project_root.resolve()

        def bound(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (project_root / path).resolve()

        documents = materialize_aggregate(
            project_root=project_root,
            current_root=bound(args.current_root),
            parent_root=bound(args.parent_root),
            v4_root=bound(args.v4_root),
            arena_path=bound(args.arena),
            strict_path=bound(args.strict),
            high_path=bound(args.high),
        )
        output_dir = bound(args.output_dir)
        prefixes = {
            "receipt": "frontier-coverage-v4-aggregate-receipt",
            "audit": "frontier-coverage-v4-aggregate-audit",
            "uplift": "frontier-corrected-development-uplift",
            "arena": "frontier-corrected-development-arena",
            "coverage": "frontier-corrected-development-coverage",
        }
        paths = {
            label: _write_addressed(
                {key: value for key, value in document.items() if key != "artifact_sha256"},
                directory=output_dir,
                prefix=prefixes[label],
            )
            for label, document in documents.items()
        }
        print(
            json.dumps(
                {
                    "status": documents["audit"]["decision"],
                    "artifacts": {
                        label: {
                            "path": str(paths[label]),
                            "artifactSha256": document["artifact_sha256"],
                        }
                        for label, document in documents.items()
                    },
                    "counts": documents["audit"]["counts"],
                    "providerCalls": 0,
                    "epicureCalls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "freeze-residual":
        project_root = args.project_root.resolve()
        plan = build_residual_plan(
            project_root=project_root,
            parent_root=(project_root / args.parent_root).resolve()
            if not args.parent_root.is_absolute()
            else args.parent_root.resolve(),
            exposure_root=(project_root / args.exposure_root).resolve()
            if not args.exposure_root.is_absolute()
            else args.exposure_root.resolve(),
            arena_path=(project_root / args.arena).resolve()
            if not args.arena.is_absolute()
            else args.arena.resolve(),
        )
        output_dir = (
            (project_root / args.output_dir).resolve()
            if not args.output_dir.is_absolute()
            else args.output_dir.resolve()
        )
        path = _write_addressed(
            {key: value for key, value in plan.items() if key != "artifact_sha256"},
            directory=output_dir,
            prefix="frontier-coverage-residual-v5-plan",
        )
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "artifactSha256": plan["artifact_sha256"],
                    "path": str(path),
                    "freshCells": plan["counts"]["fresh_cells"],
                    "freshRealArms": plan["counts"]["fresh_real_arms"],
                    "projectedMissingCells": plan["support_basis"][
                        "projected_missing_cells_after_all_50_cells_usable"
                    ],
                    "providerCalls": 0,
                    "epicureCalls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    run()
