"""Materialize corrected paper inputs after the frozen coverage repair completes.

This command is a read-only integration step over immutable records.  It verifies
the corrected strict, high-resource, and model-arena pools, resolves every base
arm against its historical normalized response, verifies the exact 13-cell /
25-arm coverage state through the append-only repair ledger, and writes new
content-addressed analysis inputs.  Historical v27--v44 files are never edited.

The output remains development evidence: adding one shared anchor task closes
comparison-graph holes but does not create adequate family-specific evidence or
quality judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .current_pilot_review_import import _response_answer, _validate_tool_trace
from .frontier_contract_runner import IntegrityError
from .frontier_coverage_repair_executor import (
    EXPECTED_ARM_COUNT,
    EXPECTED_CELL_COUNT,
    HIGH_RESOURCE_STRATUM,
    CoverageMaterialization,
    CoverageState,
    _coverage_state,
    _load_content_addressed_artifact,
    _write_artifact,
    build_materialization,
)
from .frontier_multirun_assets import _verify_artifact
from .real_task_bank import sha256_json

UPLIFT_SCHEMA_VERSION = "flavourbench-frontier-corrected-uplift-input-v1"
ARENA_SCHEMA_VERSION = "flavourbench-frontier-corrected-arena-input-v1"
COVERAGE_SCHEMA_VERSION = "flavourbench-frontier-corrected-coverage-metrics-v1"
BUNDLE_SCHEMA_VERSION = "flavourbench-frontier-corrected-paper-input-bundle-v1"
BASE_UPLIFT_SCHEMA_VERSION = "flavourbench-frontier-multirun-review-pool-v1"
BASE_ARENA_SCHEMA_VERSION = "flavourbench-frontier-model-arena-review-pool-v1"
FAMILIES = ("composition", "cookability", "evidence", "substitution")


@dataclass(frozen=True)
class Arm:
    """One verified normalized real response used by a corrected input."""

    condition: str
    task_id: str
    family: str
    prompt_sha256: str
    model_id: str
    canonical_model_slug: str
    provider_tag: str
    actual_model_id: str
    actual_provider: str
    response_artifact_sha256: str
    source_artifact_sha256: str
    answer_sha256: str
    generation_ids: tuple[str, ...]
    tool_calls: int
    successful_tool_calls: int
    cost_micros: int
    work_item_id: str

    def uplift_manifest(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "response_artifact_sha256": self.response_artifact_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "answer_sha256": self.answer_sha256,
            "actual_model_id": self.actual_model_id,
            "actual_provider": self.actual_provider,
            "generation_ids_sha256": sha256_json(list(self.generation_ids)),
        }

    def arena_manifest(self) -> dict[str, Any]:
        return {
            **self.uplift_manifest(),
            "requested_model_id": self.model_id,
            "canonical_model_slug": self.canonical_model_slug,
        }


@dataclass(frozen=True)
class PostrunDocuments:
    strict: Mapping[str, Any]
    high: Mapping[str, Any]
    arena: Mapping[str, Any]
    coverage: Mapping[str, Any]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _address(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{label} must be an object")
    return value


def _require_items(document: Mapping[str, Any], *, label: str) -> list[Mapping[str, Any]]:
    items = document.get("items")
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise IntegrityError(f"{label} items are malformed")
    return list(items)


def _uplift_identity_commitment(items: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, str]] = []
    for item in items:
        for side in ("left", "right"):
            arm = _require_mapping(item.get(side), label=f"uplift {side} arm")
            rows.append(
                {
                    "review_item_id": str(item.get("review_item_id") or ""),
                    "side": side,
                    "condition": str(arm.get("condition") or ""),
                    "requested_model_id": str(item.get("requested_model_id") or ""),
                    "actual_model_id": str(arm.get("actual_model_id") or ""),
                    "actual_provider": str(arm.get("actual_provider") or ""),
                }
            )
    return sha256_json(rows)


def _verify_uplift_pool(document: Mapping[str, Any], *, label: str) -> None:
    items = _require_items(document, label=label)
    observed = _require_mapping(document.get("observed"), label=f"{label} observed")
    claim = _require_mapping(document.get("claim_boundary"), label=f"{label} claim boundary")
    model_order = tuple(str(value) for value in document.get("model_order") or [])
    coordinates: set[tuple[str, str]] = set()
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    left_on = 0
    response_digests: set[str] = set()
    for item in items:
        task_id = str(item.get("task_id") or "")
        model_id = str(item.get("requested_model_id") or "")
        family = str(item.get("task_family") or "")
        if (
            not task_id
            or not model_id
            or family not in FAMILIES
            or model_id not in model_order
            or (task_id, model_id) in coordinates
        ):
            raise IntegrityError(f"{label} has a duplicate or invalid model-task pair")
        conditions: set[str] = set()
        for side in ("left", "right"):
            arm = _require_mapping(item.get(side), label=f"{label} {side} arm")
            condition = str(arm.get("condition") or "")
            digest = str(arm.get("response_artifact_sha256") or "")
            if condition not in {"epicure_off", "epicure_on"} or len(digest) != 64:
                raise IntegrityError(f"{label} contains an invalid arm")
            if digest in response_digests:
                raise IntegrityError(f"{label} repeats a response across uplift pairs")
            response_digests.add(digest)
            conditions.add(condition)
        if conditions != {"epicure_off", "epicure_on"}:
            raise IntegrityError(f"{label} item is not a complete uplift pair")
        coordinates.add((task_id, model_id))
        family_counts[family] += 1
        model_counts[model_id] += 1
        left_on += int(item["left"]["condition"] == "epicure_on")
    exact = {
        "candidate_pairs": len(items),
        "source_arms": 2 * len(items),
        "candidate_pairs_by_family": dict(family_counts),
        "candidate_pairs_by_model": {
            model_id: model_counts[model_id] for model_id in model_order
        },
        "left_epicure_on": left_on,
        "right_epicure_on": len(items) - left_on,
        "synthetic_arms": 0,
    }
    if any(observed.get(field) != value for field, value in exact.items()):
        raise IntegrityError(f"{label} observed counts do not reproduce from its items")
    if (
        document.get("identity_commitment_sha256") != _uplift_identity_commitment(items)
        or claim.get("official") is not False
        or claim.get("rank_eligible") is not False
        or claim.get("research_result") is not False
        or int(claim.get("quality_judgments") or 0) != 0
    ):
        raise IntegrityError(f"{label} commitment or claim boundary does not verify")


def _verify_arena_pool(document: Mapping[str, Any]) -> None:
    items = _require_items(document, label="base arena")
    observed = _require_mapping(document.get("observed"), label="base arena observed")
    claim = _require_mapping(document.get("claim_boundary"), label="base arena boundary")
    if (
        document.get("track") != "model_arena"
        or document.get("items_commitment_sha256") != sha256_json(items)
        or observed.get("candidate_comparisons") != len(items)
        or observed.get("synthetic_arms") != 0
        or claim.get("official") is not False
        or claim.get("rank_eligible") is not False
        or claim.get("research_result") is not False
        or int(claim.get("quality_judgments") or 0) != 0
    ):
        raise IntegrityError("base arena commitment or claim boundary does not verify")


def load_base_inputs(
    *,
    strict_path: Path,
    high_path: Path,
    arena_path: Path,
    materialization: CoverageMaterialization,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    strict, _ = _load_content_addressed_artifact(
        strict_path, label="corrected strict uplift pool", schema_version=BASE_UPLIFT_SCHEMA_VERSION
    )
    high, _ = _load_content_addressed_artifact(
        high_path,
        label="corrected high-resource uplift pool",
        schema_version=BASE_UPLIFT_SCHEMA_VERSION,
    )
    arena, arena_sha256 = _load_content_addressed_artifact(
        arena_path, label="corrected model arena", schema_version=BASE_ARENA_SCHEMA_VERSION
    )
    _verify_uplift_pool(strict, label="corrected strict uplift pool")
    _verify_uplift_pool(high, label="corrected high-resource uplift pool")
    _verify_arena_pool(arena)
    if arena_sha256 != materialization.document["source"]["corrected_arena_sha256"]:
        raise IntegrityError("base arena differs from the frozen coverage materialization")
    if strict.get("epicure") != materialization.epicure or high.get("epicure") != (
        materialization.epicure
    ) or arena.get("epicure") != materialization.epicure:
        raise IntegrityError("base inputs and coverage repair use different Epicure provenance")
    high_policy = _require_mapping(high.get("selection_policy"), label="high policy")
    if high_policy.get("execution_policy_sha256") != materialization.policy.sha256:
        raise IntegrityError("high-resource pool and coverage execution policy differ")
    strata = _require_mapping(arena.get("source"), label="arena source").get("strata")
    strata = _require_mapping(strata, label="arena strata")
    strict_stratum = _require_mapping(strata.get("strict"), label="strict arena stratum")
    high_stratum = _require_mapping(
        strata.get(HIGH_RESOURCE_STRATUM), label="high-resource arena stratum"
    )
    if (
        strict_stratum.get("aggregate_sha256")
        != _require_mapping(strict.get("source"), label="strict source").get(
            "summary_content_address"
        )
        or high_stratum.get("aggregate_sha256")
        != _require_mapping(high.get("source"), label="high source").get(
            "summary_content_address"
        )
        or high_stratum.get("execution_policy_sha256") != materialization.policy.sha256
    ):
        raise IntegrityError("uplift pools are not the strata used by the corrected arena")
    if tuple(high.get("model_order") or []) != tuple(arena.get("model_order") or []):
        raise IntegrityError("high-resource and arena frozen model orders differ")
    return strict, high, arena


def load_historical_responses(directories: Sequence[Path]) -> dict[str, Mapping[str, Any]]:
    """Verify and index immutable historical normalized responses by content address."""

    if not directories:
        raise IntegrityError("historical response directories are required")
    indexed: dict[str, Mapping[str, Any]] = {}
    for directory in sorted({path.resolve() for path in directories}, key=str):
        if directory.is_symlink() or not directory.is_dir():
            raise IntegrityError(f"historical response directory is invalid: {directory}")
        for path in sorted(directory.glob("*.json")):
            try:
                document = _verify_artifact(path)
            except RuntimeError as error:
                raise IntegrityError(str(error)) from error
            digest = str(document.get("artifact_sha256") or "")
            if digest in indexed:
                raise IntegrityError(f"historical response artifact is duplicated: {digest}")
            indexed[digest] = document
    return indexed


def _arm_from_document(
    document: Mapping[str, Any],
    *,
    expected_epicure: Mapping[str, str],
) -> Arm:
    model = _require_mapping(document.get("model"), label="response model")
    task = _require_mapping(document.get("task"), label="response task")
    response = _require_mapping(document.get("response"), label="response payload")
    provenance = _require_mapping(document.get("provenance"), label="response provenance")
    source = _require_mapping(document.get("source"), label="response source")
    epicure = _require_mapping(provenance.get("epicure"), label="response Epicure provenance")
    condition = str(document.get("condition") or "")
    generation_ids_raw = response.get("generation_ids")
    if not isinstance(generation_ids_raw, list) or not generation_ids_raw:
        raise IntegrityError("response has no provider generation identifiers")
    generation_ids = tuple(str(value) for value in generation_ids_raw)
    if any(not value for value in generation_ids) or len(set(generation_ids)) != len(
        generation_ids
    ):
        raise IntegrityError("response generation identifiers are empty or duplicated")
    calls, errors = _validate_tool_trace(document)
    answer = _response_answer(document)
    cost_micros = response.get("cost_micros")
    if isinstance(cost_micros, bool) or not isinstance(cost_micros, int) or cost_micros < 0:
        raise IntegrityError("response cost_micros is invalid")
    exact_epicure = {
        "release_id": str(epicure.get("release_id") or ""),
        "bundle_sha256": str(epicure.get("bundle_sha256") or ""),
        "application_sha256": str(epicure.get("application_sha256") or ""),
        "tool_schema_sha256": str(provenance.get("epicure_tool_schema_sha256") or ""),
    }
    if (
        document.get("schema_version") != "flavourbench-real-exploratory-response-v1"
        or document.get("official") is not False
        or document.get("rank_eligible") is not False
        or document.get("research_result") is not False
        or document.get("research_release_eligible") is not False
        or condition not in {"epicure_off", "epicure_on"}
        or bool(provenance.get("epicure_access")) != (condition == "epicure_on")
        or (condition == "epicure_off" and calls != 0)
        or (condition == "epicure_on" and calls - errors <= 0)
        or exact_epicure != dict(expected_epicure)
        or model.get("actual_model_id") != response.get("actual_model_id")
    ):
        raise IntegrityError("response crossed the corrected real-arm contract")
    return Arm(
        condition=condition,
        task_id=str(task.get("public_id") or ""),
        family=str(task.get("family") or ""),
        prompt_sha256=str(task.get("prompt_sha256") or ""),
        model_id=str(model.get("requested_model_id") or ""),
        canonical_model_slug=str(model.get("canonical_model_slug") or ""),
        provider_tag=str(model.get("provider_tag") or ""),
        actual_model_id=str(model.get("actual_model_id") or ""),
        actual_provider=str(model.get("actual_provider") or ""),
        response_artifact_sha256=str(document.get("artifact_sha256") or ""),
        source_artifact_sha256=str(source.get("artifact_sha256") or ""),
        answer_sha256=_sha256_text(answer),
        generation_ids=generation_ids,
        tool_calls=calls,
        successful_tool_calls=calls - errors,
        cost_micros=cost_micros,
        work_item_id=str(document.get("work_item_id") or ""),
    )


def _assert_arm_manifest(
    arm: Arm,
    manifest: Mapping[str, Any],
    *,
    arena: bool,
) -> None:
    expected = arm.arena_manifest() if arena else arm.uplift_manifest()
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise IntegrityError("response content does not reproduce its committed arm manifest")


def resolve_base_arms(
    *,
    strict: Mapping[str, Any],
    high: Mapping[str, Any],
    arena: Mapping[str, Any],
    historical: Mapping[str, Mapping[str, Any]],
    epicure: Mapping[str, str],
) -> tuple[
    dict[str, Arm],
    dict[str, Arm],
    dict[tuple[str, str, str], Arm],
]:
    required: set[str] = set()
    for document in (strict, high, arena):
        for item in _require_items(document, label="base input"):
            for side in ("left", "right"):
                required.add(str(item[side]["response_artifact_sha256"]))
    missing = sorted(required - set(historical))
    if missing:
        raise IntegrityError(
            f"historical response index is incomplete ({len(missing)} missing artifacts)"
        )
    parsed = {
        digest: _arm_from_document(historical[digest], expected_epicure=epicure)
        for digest in sorted(required)
    }
    strict_arms: dict[str, Arm] = {}
    high_arms: dict[str, Arm] = {}
    for pool, target, label in (
        (strict, strict_arms, "strict"),
        (high, high_arms, "high-resource"),
    ):
        for item in _require_items(pool, label=f"{label} pool"):
            for side in ("left", "right"):
                meta = _require_mapping(item[side], label=f"{label} arm")
                arm = parsed[str(meta["response_artifact_sha256"])]
                _assert_arm_manifest(arm, meta, arena=False)
                if (
                    arm.task_id != item.get("task_id")
                    or arm.family != item.get("task_family")
                    or arm.prompt_sha256 != item.get("prompt_sha256")
                    or arm.model_id != item.get("requested_model_id")
                    or arm.canonical_model_slug != item.get("canonical_model_slug")
                    or arm.provider_tag != item.get("provider_tag")
                ):
                    raise IntegrityError(f"{label} arm identity differs from its pool item")
                target[arm.response_artifact_sha256] = arm
    arena_arms: dict[tuple[str, str, str], Arm] = {}
    for item in _require_items(arena, label="base arena"):
        stratum = str(item.get("execution_stratum") or "")
        for side in ("left", "right"):
            meta = _require_mapping(item[side], label="base arena arm")
            arm = parsed[str(meta["response_artifact_sha256"])]
            _assert_arm_manifest(arm, meta, arena=True)
            coordinate = (stratum, arm.task_id, arm.model_id)
            prior = arena_arms.setdefault(coordinate, arm)
            if prior != arm:
                raise IntegrityError("arena reuses a coordinate with a different response")
            if (
                arm.task_id != item.get("task_id")
                or arm.family != item.get("task_family")
                or arm.prompt_sha256 != item.get("prompt_sha256")
            ):
                raise IntegrityError("arena arm identity differs from its item")
    generation_ids: set[str] = set()
    for arm in {arm.response_artifact_sha256: arm for arm in parsed.values()}.values():
        if generation_ids.intersection(arm.generation_ids):
            raise IntegrityError("historical response generation identifier is duplicated")
        generation_ids.update(arm.generation_ids)
    return strict_arms, high_arms, arena_arms


def resolve_coverage_arms(
    materialization: CoverageMaterialization,
    state: CoverageState,
) -> dict[tuple[str, str], Arm]:
    if (
        state.accounting.source_count != EXPECTED_CELL_COUNT
        or len(state.accounting.sources) != EXPECTED_CELL_COUNT
        or len(state.accounting.finalizations) != EXPECTED_CELL_COUNT
        or len(state.responses) != EXPECTED_ARM_COUNT
        or state.accounting.blockers
    ):
        raise IntegrityError("coverage repair is incomplete; exactly 13 finalized sources required")
    cells = {cell.work_item.work_item_id: cell for cell in materialization.cells}
    arms: dict[tuple[str, str], Arm] = {}
    generation_ids: set[str] = set()
    for key, response in sorted(state.responses.items()):
        work_item_id, condition = key
        cell = cells.get(work_item_id)
        source = state.accounting.sources.get(work_item_id)
        if cell is None or source is None or condition not in cell.conditions:
            raise IntegrityError("coverage response is outside the frozen cell schedule")
        try:
            document = _verify_artifact(response.path)
        except RuntimeError as error:
            raise IntegrityError(str(error)) from error
        if document.get("artifact_sha256") != response.artifact_sha256:
            raise IntegrityError("coverage response digest differs from the scanned state")
        arm = _arm_from_document(document, expected_epicure=materialization.epicure)
        if (
            arm.work_item_id != work_item_id
            or arm.condition != condition
            or arm.task_id != cell.work_item.task.public_id
            or arm.family != cell.work_item.task.family
            or arm.prompt_sha256 != cell.work_item.task.prompt_sha256
            or arm.model_id != cell.work_item.candidate.model_id
            or arm.canonical_model_slug != cell.work_item.candidate.canonical_model_slug
            or arm.provider_tag != cell.work_item.candidate.provider_tag
            or arm.source_artifact_sha256 != source.artifact_sha256
        ):
            raise IntegrityError("coverage response identity differs from its frozen cell")
        if generation_ids.intersection(arm.generation_ids):
            raise IntegrityError("coverage generation identifier is duplicated")
        generation_ids.update(arm.generation_ids)
        arms[key] = arm
    expected_keys = {
        (cell.work_item.work_item_id, condition)
        for cell in materialization.cells
        for condition in cell.conditions
    }
    if set(arms) != expected_keys:
        raise IntegrityError("coverage response set is not the exact frozen 25-arm set")
    return arms


def _base_uplift_candidates(
    pool: Mapping[str, Any],
    arms_by_digest: Mapping[str, Arm],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in _require_items(pool, label="uplift pool"):
        by_condition: dict[str, Arm] = {}
        for side in ("left", "right"):
            arm = arms_by_digest[str(item[side]["response_artifact_sha256"])]
            by_condition[arm.condition] = arm
        candidates.append(
            {
                "origin": "retained_corrected_base",
                "task_id": str(item["task_id"]),
                "family": str(item["task_family"]),
                "prompt_sha256": str(item["prompt_sha256"]),
                "model_id": str(item["requested_model_id"]),
                "canonical_model_slug": str(item["canonical_model_slug"]),
                "provider_tag": str(item["provider_tag"]),
                "condition_work_item_ids": {
                    condition: arm.work_item_id for condition, arm in by_condition.items()
                },
                "arms": by_condition,
            }
        )
    return candidates


def _coverage_uplift_candidates(
    materialization: CoverageMaterialization,
    coverage_arms: Mapping[tuple[str, str], Arm],
    arena_arms: Mapping[tuple[str, str, str], Arm],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for cell in materialization.cells:
        work_item_id = cell.work_item.work_item_id
        by_condition = {
            condition: coverage_arms[(work_item_id, condition)]
            for condition in cell.conditions
        }
        if "epicure_on" not in by_condition:
            coordinate = (
                HIGH_RESOURCE_STRATUM,
                cell.work_item.task.public_id,
                cell.work_item.candidate.model_id,
            )
            historical_on = arena_arms.get(coordinate)
            if historical_on is None or historical_on.condition != "epicure_on":
                raise IntegrityError("partial repair has no committed historical Epicure-on arm")
            by_condition["epicure_on"] = historical_on
        if set(by_condition) != {"epicure_off", "epicure_on"}:
            raise IntegrityError("coverage cell did not resolve to one complete uplift pair")
        off, on = by_condition["epicure_off"], by_condition["epicure_on"]
        if (
            off.model_id != on.model_id
            or off.task_id != on.task_id
            or off.prompt_sha256 != on.prompt_sha256
            or off.family != on.family
            or off.canonical_model_slug != on.canonical_model_slug
            or off.provider_tag != on.provider_tag
        ):
            raise IntegrityError("partial-condition completion changed model, task, or route")
        candidates.append(
            {
                "origin": (
                    "coverage_repair_with_historical_on_reuse"
                    if len(cell.conditions) == 1
                    else "coverage_repair_new_pair"
                ),
                "schedule_cell_id": cell.schedule_cell_id,
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
    return candidates


def _uplift_on_left(
    candidates: Sequence[Mapping[str, Any]], source_commitment: str, model_order: Sequence[str]
) -> set[str]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for candidate in candidates:
        model_id = str(candidate["model_id"])
        pair_key = sha256_json(
            {
                "model_id": model_id,
                "task_id": candidate["task_id"],
                "responses": {
                    condition: candidate["arms"][condition].response_artifact_sha256
                    for condition in ("epicure_off", "epicure_on")
                },
            }
        )
        score = _sha256_text(
            f"{source_commitment}:{model_id}:{candidate['family']}:{pair_key}:left-right"
        )
        grouped[model_id].append((score, pair_key))
    on_left: set[str] = set()
    for model_id in model_order:
        ordered = sorted(grouped.get(model_id, []))
        on_left.update(pair_key for _, pair_key in ordered[: len(ordered) // 2])
        if len(ordered) % 2 and int(
            _sha256_text(f"{source_commitment}:{model_id}:odd")[:2], 16
        ) % 2 == 0:
            on_left.add(ordered[len(ordered) // 2][1])
    return on_left


def _build_uplift_input(
    *,
    base: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    execution_stratum: str,
    materialization: CoverageMaterialization,
    coverage_pairs_added: int,
) -> dict[str, Any]:
    model_order = tuple(str(value) for value in base.get("model_order") or [])
    source_commitment = sha256_json(
        {
            "base_pool_sha256": base["artifact_sha256"],
            "coverage_materialization_sha256": materialization.document["artifact_sha256"],
            "candidate_response_sets": sorted(
                sorted(
                    candidate["arms"][condition].response_artifact_sha256
                    for condition in ("epicure_off", "epicure_on")
                )
                for candidate in candidates
            ),
        }
    )
    on_left = _uplift_on_left(candidates, source_commitment, model_order)
    items: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    unique_arms: dict[str, Arm] = {}
    for candidate in sorted(
        candidates,
        key=lambda value: (
            model_order.index(str(value["model_id"])),
            str(value["family"]),
            str(value["task_id"]),
        ),
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
        sides = (
            (("left", "epicure_on"), ("right", "epicure_off"))
            if pair_key in on_left
            else (("left", "epicure_off"), ("right", "epicure_on"))
        )
        review_item_id = _sha256_text(f"{source_commitment}:{pair_key}:review-item")
        item: dict[str, Any] = {
            "review_item_id": review_item_id,
            "pair_key": pair_key,
            "origin": candidate["origin"],
            "condition_work_item_ids": dict(candidate["condition_work_item_ids"]),
            "task_id": candidate["task_id"],
            "task_family": candidate["family"],
            "prompt_sha256": candidate["prompt_sha256"],
            "requested_model_id": candidate["model_id"],
            "canonical_model_slug": candidate["canonical_model_slug"],
            "provider_tag": candidate["provider_tag"],
        }
        if candidate.get("schedule_cell_id"):
            item["schedule_cell_id"] = candidate["schedule_cell_id"]
        for side, condition in sides:
            arm = arms[condition]
            item[side] = arm.uplift_manifest()
            unique_arms[arm.response_artifact_sha256] = arm
        items.append(item)
        family_counts[str(candidate["family"])] += 1
        model_counts[str(candidate["model_id"])] += 1
    if len(unique_arms) != 2 * len(items):
        raise IntegrityError("corrected uplift input reuses an arm across different pairs")
    generation_ids: set[str] = set()
    for arm in unique_arms.values():
        if generation_ids.intersection(arm.generation_ids):
            raise IntegrityError("corrected uplift input duplicates a provider generation")
        generation_ids.update(arm.generation_ids)
    observed = {
        "candidate_pairs": len(items),
        "source_arms": len(unique_arms),
        "unique_task_ids": len({str(item["task_id"]) for item in items}),
        "distinct_tasks": len({str(item["task_id"]) for item in items}),
        "candidate_pairs_by_family": {
            family: family_counts[family] for family in FAMILIES
        },
        "candidate_pairs_by_model": {
            model_id: model_counts[model_id] for model_id in model_order
        },
        "real_provider_calls": sum(len(arm.generation_ids) for arm in unique_arms.values()),
        "real_epicure_calls": sum(arm.tool_calls for arm in unique_arms.values()),
        "successful_real_epicure_calls": sum(
            arm.successful_tool_calls for arm in unique_arms.values()
        ),
        "synthetic_arms": 0,
        "reviewed_source_cost_micros": sum(arm.cost_micros for arm in unique_arms.values()),
        "left_epicure_on": sum(item["left"]["condition"] == "epicure_on" for item in items),
        "right_epicure_on": sum(
            item["right"]["condition"] == "epicure_on" for item in items
        ),
        "coverage_repair_pairs_added": coverage_pairs_added,
        "coverage_repair_new_real_arms": (
            EXPECTED_ARM_COUNT if execution_stratum == HIGH_RESOURCE_STRATUM else 0
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": UPLIFT_SCHEMA_VERSION,
        "artifact_role": "corrected_paper_regeneration_uplift_input",
        "track": "epicure_uplift",
        "execution_stratum": execution_stratum,
        "status": "verified_real_development_input",
        "source": {
            "base_corrected_pool_sha256": base["artifact_sha256"],
            "coverage_materialization_sha256": materialization.document["artifact_sha256"],
            "coverage_schedule_sha256": materialization.schedule_sha256,
            "source_commitment_sha256": source_commitment,
            "historical_raw_artifacts_mutated": False,
        },
        "selection_policy": {
            **dict(_require_mapping(base.get("selection_policy"), label="base selection")),
            "partial_condition_completion_across_content_addressed_work_items": True,
            "all_arm_identities_reverified_from_normalized_real_responses": True,
            "deterministic_model_balanced_side_assignment": True,
        },
        "observed": observed,
        "epicure": dict(materialization.epicure),
        "identity_commitment_sha256": _uplift_identity_commitment(items),
        "model_order": list(model_order),
        "model_contracts": base.get("model_contracts"),
        "items": items,
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "permitted_use": "blinded real-output development review and paper regeneration",
            "prohibited_use": "quality or uplift estimate before real judgments are admitted",
            "synthetic_arms": 0,
        },
    }
    document = _address(payload)
    _verify_corrected_uplift(document)
    return document


def _verify_corrected_uplift(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != UPLIFT_SCHEMA_VERSION:
        raise IntegrityError("corrected uplift schema is invalid")
    _verify_uplift_pool(document, label="corrected uplift input")


def _arena_side_layout(
    candidates: Sequence[Mapping[str, Any]], source_commitment: str
) -> dict[str, tuple[str, str]]:
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for candidate in candidates:
        pair = tuple(sorted((str(candidate["model_a"]), str(candidate["model_b"]))))
        candidate_id = str(candidate["candidate_id"])
        grouped[(pair[0], pair[1])].append(
            (_sha256_text(f"{source_commitment}:{candidate_id}:side-order"), candidate_id)
        )
    layout: dict[str, tuple[str, str]] = {}
    for pair, rows in grouped.items():
        base = int(_sha256_text(f"{source_commitment}:{pair}:orientation")[:2], 16) % 2
        for ordinal, (_, candidate_id) in enumerate(sorted(rows)):
            layout[candidate_id] = pair if (ordinal + base) % 2 == 0 else pair[::-1]
    return layout


def _components(edges: Sequence[tuple[str, str]], model_order: Sequence[str]) -> list[list[str]]:
    adjacency = {model_id: set() for model_id in model_order}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(model_order)
    result: list[list[str]] = []
    while unseen:
        queue = deque([min(unseen)])
        component: set[str] = set()
        while queue:
            model_id = queue.popleft()
            if model_id in component:
                continue
            component.add(model_id)
            queue.extend(adjacency[model_id] - component)
        unseen -= component
        result.append(sorted(component))
    return sorted(result, key=lambda value: (-len(value), value))


def _build_arena_input(
    *,
    base: Mapping[str, Any],
    base_arms: Mapping[tuple[str, str, str], Arm],
    materialization: CoverageMaterialization,
    coverage_arms: Mapping[tuple[str, str], Arm],
) -> dict[str, Any]:
    model_order = tuple(str(value) for value in base.get("model_order") or [])
    arms = dict(base_arms)
    for cell in materialization.cells:
        if "epicure_on" not in cell.conditions:
            continue
        arm = coverage_arms[(cell.work_item.work_item_id, "epicure_on")]
        coordinate = (HIGH_RESOURCE_STRATUM, arm.task_id, arm.model_id)
        if coordinate in arms:
            raise IntegrityError("coverage Epicure-on arm duplicates an arena coordinate")
        arms[coordinate] = arm
    by_task: dict[tuple[str, str], list[Arm]] = defaultdict(list)
    for (stratum, task_id, _), arm in arms.items():
        by_task[(stratum, task_id)].append(arm)
    source_commitment = sha256_json(
        {
            "base_arena_sha256": base["artifact_sha256"],
            "coverage_materialization_sha256": materialization.document["artifact_sha256"],
            "arms": sorted(arm.response_artifact_sha256 for arm in arms.values()),
        }
    )
    candidates: list[dict[str, Any]] = []
    for (stratum, task_id), task_arms in sorted(by_task.items()):
        ordered = sorted(task_arms, key=lambda arm: model_order.index(arm.model_id))
        for left, right in itertools.combinations(ordered, 2):
            model_a, model_b = sorted((left.model_id, right.model_id))
            candidate_id = _sha256_text(
                f"{source_commitment}:{stratum}:{task_id}:{model_a}:{model_b}:arena-candidate"
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "stratum": stratum,
                    "task_id": task_id,
                    "family": left.family,
                    "prompt_sha256": left.prompt_sha256,
                    "model_a": model_a,
                    "model_b": model_b,
                }
            )
    layout = _arena_side_layout(candidates, source_commitment)
    items: list[dict[str, Any]] = []
    response_presentations: Counter[str] = Counter()
    comparison_by_model: Counter[str] = Counter()
    comparison_by_pair: Counter[tuple[str, str]] = Counter()
    comparison_by_family: Counter[str] = Counter()
    comparison_by_stratum: Counter[str] = Counter()
    comparison_by_pair_family: Counter[tuple[str, str, str]] = Counter()
    left_by_model: Counter[str] = Counter()
    edges: list[tuple[str, str]] = []
    for candidate in sorted(
        candidates,
        key=lambda value: (
            str(value["stratum"]),
            str(value["task_id"]),
            str(value["model_a"]),
            str(value["model_b"]),
        ),
    ):
        candidate_id = str(candidate["candidate_id"])
        left_model, right_model = layout[candidate_id]
        stratum = str(candidate["stratum"])
        task_id = str(candidate["task_id"])
        left = arms[(stratum, task_id, left_model)]
        right = arms[(stratum, task_id, right_model)]
        review_item_id = _sha256_text(f"{source_commitment}:{candidate_id}:review-item")
        item = {
            "review_item_id": review_item_id,
            "task_id": task_id,
            "task_family": candidate["family"],
            "prompt_sha256": candidate["prompt_sha256"],
            "execution_stratum": stratum,
            "execution_policy_sha256": (
                materialization.policy.sha256
                if stratum == HIGH_RESOURCE_STRATUM
                else base["source"]["strata"][stratum]["execution_policy_sha256"]
            ),
            "left": left.arena_manifest(),
            "right": right.arena_manifest(),
        }
        items.append(item)
        for side, arm in (("left", left), ("right", right)):
            response_presentations[arm.response_artifact_sha256] += 1
            comparison_by_model[arm.model_id] += 1
            left_by_model[arm.model_id] += int(side == "left")
        pair = tuple(sorted((left.model_id, right.model_id)))
        family = str(candidate["family"])
        comparison_by_pair[pair] += 1
        comparison_by_family[family] += 1
        comparison_by_stratum[stratum] += 1
        comparison_by_pair_family[(pair[0], pair[1], family)] += 1
        edges.append(pair)
    pair_family_cells = [
        {
            "model_a": model_a,
            "model_b": model_b,
            "task_family": family,
            "candidate_comparisons": comparison_by_pair_family[
                (*tuple(sorted((model_a, model_b))), family)
            ],
        }
        for model_a, model_b in itertools.combinations(model_order, 2)
        for family in FAMILIES
    ]
    missing = [row for row in pair_family_cells if row["candidate_comparisons"] == 0]
    components = _components(edges, model_order)
    unique_arms = {arm.response_artifact_sha256: arm for arm in arms.values()}
    generation_ids: set[str] = set()
    for arm in unique_arms.values():
        if generation_ids.intersection(arm.generation_ids):
            raise IntegrityError("corrected arena duplicates a provider generation")
        generation_ids.update(arm.generation_ids)
    reuse = list(response_presentations.values())
    model_family_tasks = {
        model_id: {
            family: len(
                {
                    task_id
                    for (stratum, task_id, observed_model), arm in arms.items()
                    if observed_model == model_id and arm.family == family
                }
            )
            for family in FAMILIES
        }
        for model_id in model_order
    }
    base_observed = _require_mapping(base.get("observed"), label="base arena observed")
    observed = {
        "base_candidate_comparisons": int(base_observed["candidate_comparisons"]),
        "coverage_repair_candidate_comparisons_added": len(items)
        - int(base_observed["candidate_comparisons"]),
        "candidate_comparisons": len(items),
        "unique_task_ids": len({arm.task_id for arm in unique_arms.values()}),
        "task_stratum_clusters": len(by_task),
        "distinct_tasks": len({arm.task_id for arm in unique_arms.values()}),
        "models": len(model_order),
        "source_response_arms": len(unique_arms),
        "base_source_response_arms": int(base_observed["source_response_arms"]),
        "coverage_repair_epicure_on_arms_added": len(unique_arms)
        - int(base_observed["source_response_arms"]),
        "projected_response_presentations": 2 * len(items),
        "candidate_comparisons_by_family": {
            family: comparison_by_family[family] for family in FAMILIES
        },
        "candidate_comparisons_by_stratum": dict(sorted(comparison_by_stratum.items())),
        "comparison_exposures_by_model": {
            model_id: comparison_by_model[model_id] for model_id in model_order
        },
        "candidate_comparisons_by_model_pair": [
            {
                "model_a": model_a,
                "model_b": model_b,
                "candidate_comparisons": comparison_by_pair[
                    tuple(sorted((model_a, model_b)))
                ],
            }
            for model_a, model_b in itertools.combinations(model_order, 2)
        ],
        "candidate_comparisons_by_model_pair_family": pair_family_cells,
        "model_pair_family_cells": len(pair_family_cells),
        "missing_model_pair_family_cells": len(missing),
        "missing_model_pair_family_cell_records": missing,
        "model_family_unique_task_counts": model_family_tasks,
        "left_presentations_by_model": {
            model_id: left_by_model[model_id] for model_id in model_order
        },
        "unique_provider_generation_ids": len(generation_ids),
        "unique_epicure_calls": sum(arm.tool_calls for arm in unique_arms.values()),
        "unique_successful_epicure_calls": sum(
            arm.successful_tool_calls for arm in unique_arms.values()
        ),
        "projected_provider_calls": sum(
            len(unique_arms[digest].generation_ids) * presentations
            for digest, presentations in response_presentations.items()
        ),
        "projected_epicure_calls": sum(
            unique_arms[digest].tool_calls * presentations
            for digest, presentations in response_presentations.items()
        ),
        "projected_successful_epicure_calls": sum(
            unique_arms[digest].successful_tool_calls * presentations
            for digest, presentations in response_presentations.items()
        ),
        "unique_source_cost_micros": sum(arm.cost_micros for arm in unique_arms.values()),
        "comparison_graph_component_sizes": [len(component) for component in components],
        "evidence_units": {
            "raw_comparison_rows": len(items),
            "unique_task_ids": len({arm.task_id for arm in unique_arms.values()}),
            "task_stratum_clusters": len(by_task),
            "unique_response_arms": len(unique_arms),
            "response_arm_presentations": 2 * len(items),
            "minimum_comparisons_per_reused_response_arm": min(reuse),
            "maximum_comparisons_per_reused_response_arm": max(reuse),
            "median_comparisons_per_reused_response_arm": median(reuse),
            "independence_unit_for_uncertainty": "task_cluster",
            "response_reuse_policy": (
                "shared response arms remain locked inside their task cluster during "
                "resampling; comparison rows are not treated as independent"
            ),
            "scalar_effective_sample_size_claimed": False,
        },
        "synthetic_arms": 0,
    }
    payload: dict[str, Any] = {
        "schema_version": ARENA_SCHEMA_VERSION,
        "artifact_role": "corrected_paper_regeneration_model_arena_input",
        "status": "verified_real_development_input",
        "track": "model_arena",
        "source": {
            "base_corrected_arena_sha256": base["artifact_sha256"],
            "coverage_materialization_sha256": materialization.document["artifact_sha256"],
            "coverage_schedule_sha256": materialization.schedule_sha256,
            "source_commitment_sha256": source_commitment,
            "strata": base["source"]["strata"],
            "historical_raw_artifacts_mutated": False,
        },
        "selection_policy": {
            **dict(_require_mapping(base.get("selection_policy"), label="arena policy")),
            "coverage_repair_all_new_on_arms_admitted": True,
            "all_arm_identities_reverified_from_normalized_real_responses": True,
            "one_shared_anchor_per_repaired_endpoint_family_cell": True,
        },
        "observed": observed,
        "epicure": dict(materialization.epicure),
        "model_order": list(model_order),
        "model_contracts": base["model_contracts"],
        "items": items,
        "items_commitment_sha256": sha256_json(items),
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "permitted_use": "blinded real-output development review and paper regeneration",
            "prohibited_use": "model-quality ranking before real judgments are admitted",
            "family_specific_ranking_supported": False,
            "synthetic_arms": 0,
        },
    }
    document = _address(payload)
    if len(missing) != 0 or components != [sorted(model_order)]:
        raise IntegrityError("corrected arena did not close support holes in one connected graph")
    return document


def _build_coverage_metrics(
    *,
    strict: Mapping[str, Any],
    high: Mapping[str, Any],
    arena: Mapping[str, Any],
    base_high: Mapping[str, Any],
    base_arena: Mapping[str, Any],
    materialization: CoverageMaterialization,
    coverage_arms: Mapping[tuple[str, str], Arm],
) -> dict[str, Any]:
    before_cells = _require_mapping(base_arena["observed"], label="base arena observed")[
        "candidate_comparisons_by_model_pair_family"
    ]
    after_cells = arena["observed"]["candidate_comparisons_by_model_pair_family"]
    before_missing = Counter(
        str(row["task_family"])
        for row in before_cells
        if int(row["candidate_comparisons"]) == 0
    )
    after_missing = Counter(
        str(row["task_family"])
        for row in after_cells
        if int(row["candidate_comparisons"]) == 0
    )
    anchors = Counter(cell.work_item.task.family for cell in materialization.cells)
    source_records: list[dict[str, Any]] = []
    for cell in materialization.cells:
        work_item_id = cell.work_item.work_item_id
        cell_arms = {
            condition: coverage_arms[(work_item_id, condition)]
            for condition in cell.conditions
        }
        source_digests = {arm.source_artifact_sha256 for arm in cell_arms.values()}
        if len(source_digests) != 1:
            raise IntegrityError("one coverage cell resolved to multiple source records")
        source_records.append(
            {
                "ordinal": cell.work_item.ordinal,
                "schedule_cell_id": cell.schedule_cell_id,
                "work_item_id": work_item_id,
                "task_id": cell.work_item.task.public_id,
                "task_family": cell.work_item.task.family,
                "prompt_sha256": cell.work_item.task.prompt_sha256,
                "model_id": cell.work_item.candidate.model_id,
                "canonical_model_slug": cell.work_item.candidate.canonical_model_slug,
                "provider_tag": cell.work_item.candidate.provider_tag,
                "execution_backend": cell.work_item.candidate.execution_backend,
                "endpoint_execution_sha256": cell.work_item.endpoint_execution_sha256,
                "route_manifest_sha256": cell.route_manifest_sha256,
                "execution_policy_sha256": cell.work_item.execution_policy_sha256,
                "source_artifact_sha256": next(iter(source_digests)),
                "required_new_conditions": list(cell.conditions),
                "existing_real_conditions_reused": list(cell.existing_conditions),
                "arm_ids": list(cell.arm_ids),
                "response_artifact_sha256s_by_condition": {
                    condition: arm.response_artifact_sha256
                    for condition, arm in sorted(cell_arms.items())
                },
                "generation_ids_sha256s_by_condition": {
                    condition: sha256_json(list(arm.generation_ids))
                    for condition, arm in sorted(cell_arms.items())
                },
            }
        )
    payload: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "artifact_role": "corrected_paper_regeneration_coverage_metrics",
        "status": "verified_real_development_input",
        "source": {
            "coverage_materialization_sha256": materialization.document["artifact_sha256"],
            "coverage_schedule_sha256": materialization.schedule_sha256,
            "strict_input_sha256": strict["artifact_sha256"],
            "high_resource_input_sha256": high["artifact_sha256"],
            "arena_input_sha256": arena["artifact_sha256"],
            "historical_high_pool_sha256": base_high["artifact_sha256"],
            "historical_arena_sha256": base_arena["artifact_sha256"],
        },
        "coverage_repair": {
            "frozen_endpoint_task_cells": EXPECTED_CELL_COUNT,
            "new_real_arms": len(coverage_arms),
            "full_new_pairs": 12,
            "partial_condition_completions": 1,
            "historical_real_arms_reused": 1,
            "synthetic_arms": 0,
            "anchor_cells_by_family": {
                family: anchors[family] for family in FAMILIES
            },
            "source_records": source_records,
            "source_records_commitment_sha256": sha256_json(source_records),
            "epicure": dict(materialization.epicure),
        },
        "uplift": {
            "strict_pairs_before": int(strict["observed"]["candidate_pairs"]),
            "strict_pairs_after": int(strict["observed"]["candidate_pairs"]),
            "high_resource_pairs_before": int(base_high["observed"]["candidate_pairs"]),
            "high_resource_pairs_after": int(high["observed"]["candidate_pairs"]),
            "combined_pairs_after": int(strict["observed"]["candidate_pairs"])
            + int(high["observed"]["candidate_pairs"]),
        },
        "model_arena": {
            "comparisons_before": int(base_arena["observed"]["candidate_comparisons"]),
            "comparisons_added": int(
                arena["observed"]["coverage_repair_candidate_comparisons_added"]
            ),
            "comparisons_after": int(arena["observed"]["candidate_comparisons"]),
            "unique_response_arms_before": int(
                base_arena["observed"]["source_response_arms"]
            ),
            "unique_response_arms_after": int(arena["observed"]["source_response_arms"]),
            "model_pair_family_cells": int(arena["observed"]["model_pair_family_cells"]),
            "missing_cells_before": sum(before_missing.values()),
            "missing_cells_before_by_family": {
                family: before_missing[family] for family in FAMILIES
            },
            "missing_cells_after": sum(after_missing.values()),
            "missing_cells_after_by_family": {
                family: after_missing[family] for family in FAMILIES
            },
        },
        "paper_macros": {
            "FrontierCorrectedUpliftPairs": int(strict["observed"]["candidate_pairs"])
            + int(high["observed"]["candidate_pairs"]),
            "FrontierCorrectedArenaAnswers": int(arena["observed"]["source_response_arms"]),
            "FrontierCorrectedArenaComparisons": int(
                arena["observed"]["candidate_comparisons"]
            ),
            "FrontierCorrectedMissingModelPairFamilyCells": sum(after_missing.values()),
        },
        "interpretation": {
            "comparison_graph_holes_closed": sum(after_missing.values()) == 0,
            "family_specific_ranking_supported": False,
            "reason": (
                "the repair adds one shared real anchor task to each weakest endpoint-family "
                "cell; this repairs support connectivity but remains insufficient for "
                "family-specific model ranking"
            ),
            "quality_judgments_added": 0,
            "quality_ranking_supported": False,
        },
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "synthetic_arms": 0,
        },
    }
    return _address(payload)


def build_corrected_documents(
    *,
    materialization: CoverageMaterialization,
    coverage_state: CoverageState,
    strict_base: Mapping[str, Any],
    high_base: Mapping[str, Any],
    arena_base: Mapping[str, Any],
    historical_responses: Mapping[str, Mapping[str, Any]],
) -> PostrunDocuments:
    strict_arms, high_arms, arena_arms = resolve_base_arms(
        strict=strict_base,
        high=high_base,
        arena=arena_base,
        historical=historical_responses,
        epicure=materialization.epicure,
    )
    coverage_arms = resolve_coverage_arms(materialization, coverage_state)
    historical_generation_ids = {
        generation_id
        for arm in {**strict_arms, **high_arms}.values()
        for generation_id in arm.generation_ids
    } | {
        generation_id for arm in arena_arms.values() for generation_id in arm.generation_ids
    }
    coverage_generation_ids = {
        generation_id for arm in coverage_arms.values() for generation_id in arm.generation_ids
    }
    if historical_generation_ids & coverage_generation_ids:
        raise IntegrityError("coverage provider generation ID overlaps historical records")
    historical_response_digests = set(strict_arms) | set(high_arms) | {
        arm.response_artifact_sha256 for arm in arena_arms.values()
    }
    if historical_response_digests & {
        arm.response_artifact_sha256 for arm in coverage_arms.values()
    }:
        raise IntegrityError("coverage response artifact overlaps historical records")

    strict_candidates = _base_uplift_candidates(strict_base, strict_arms)
    high_candidates = _base_uplift_candidates(high_base, high_arms)
    high_candidates.extend(
        _coverage_uplift_candidates(materialization, coverage_arms, arena_arms)
    )
    strict = _build_uplift_input(
        base=strict_base,
        candidates=strict_candidates,
        execution_stratum="strict",
        materialization=materialization,
        coverage_pairs_added=0,
    )
    high = _build_uplift_input(
        base=high_base,
        candidates=high_candidates,
        execution_stratum=HIGH_RESOURCE_STRATUM,
        materialization=materialization,
        coverage_pairs_added=EXPECTED_CELL_COUNT,
    )
    arena = _build_arena_input(
        base=arena_base,
        base_arms=arena_arms,
        materialization=materialization,
        coverage_arms=coverage_arms,
    )
    coverage = _build_coverage_metrics(
        strict=strict,
        high=high,
        arena=arena,
        base_high=high_base,
        base_arena=arena_base,
        materialization=materialization,
        coverage_arms=coverage_arms,
    )
    return PostrunDocuments(strict=strict, high=high, arena=arena, coverage=coverage)


def materialize_postrun(
    *,
    schedule_path: Path,
    arena_base_path: Path,
    strict_base_path: Path,
    high_base_path: Path,
    task_validity_path: Path,
    route_manifest_paths: Sequence[Path],
    historical_response_directories: Sequence[Path],
    coverage_source_directory: Path,
    coverage_corrections_directory: Path | None,
    coverage_response_directory: Path,
    coverage_ledger_path: Path,
    output_directory: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Path]]:
    """Verify completed immutable inputs and write a content-addressed paper bundle."""

    materialization = build_materialization(
        schedule_path=schedule_path,
        arena_path=arena_base_path,
        task_validity_path=task_validity_path,
        route_manifest_paths=route_manifest_paths,
    )
    strict_base, high_base, arena_base = load_base_inputs(
        strict_path=strict_base_path,
        high_path=high_base_path,
        arena_path=arena_base_path,
        materialization=materialization,
    )
    historical = load_historical_responses(historical_response_directories)
    state = _coverage_state(
        materialization,
        source_directory=coverage_source_directory,
        corrections_directory=coverage_corrections_directory,
        response_directory=coverage_response_directory,
        ledger_path=coverage_ledger_path,
    )
    documents = build_corrected_documents(
        materialization=materialization,
        coverage_state=state,
        strict_base=strict_base,
        high_base=high_base,
        arena_base=arena_base,
        historical_responses=historical,
    )
    paths = {
        "strict": _write_artifact(
            documents.strict, directory=output_directory, prefix="frontier-corrected-strict-input"
        ),
        "high": _write_artifact(
            documents.high,
            directory=output_directory,
            prefix="frontier-corrected-high-resource-input",
        ),
        "arena": _write_artifact(
            documents.arena, directory=output_directory, prefix="frontier-corrected-arena-input"
        ),
        "coverage": _write_artifact(
            documents.coverage,
            directory=output_directory,
            prefix="frontier-corrected-coverage-metrics",
        ),
    }
    artifact_records = {
        label: {
            "filename": path.name,
            "artifact_sha256": documents_by_label[label]["artifact_sha256"],
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for label, path in paths.items()
        for documents_by_label in (
            {
                "strict": documents.strict,
                "high": documents.high,
                "arena": documents.arena,
                "coverage": documents.coverage,
            },
        )
    }
    bundle_payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_role": "paper_regeneration_input_bundle",
        "status": "verified_real_development_inputs",
        "source": {
            "coverage_materialization_sha256": materialization.document["artifact_sha256"],
            "coverage_schedule_sha256": materialization.schedule_sha256,
            "base_strict_pool_sha256": strict_base["artifact_sha256"],
            "base_high_resource_pool_sha256": high_base["artifact_sha256"],
            "base_arena_sha256": arena_base["artifact_sha256"],
        },
        "artifacts": artifact_records,
        "paper_regeneration": {
            "strict_uplift_input": artifact_records["strict"]["artifact_sha256"],
            "high_resource_uplift_input": artifact_records["high"]["artifact_sha256"],
            "model_arena_input": artifact_records["arena"]["artifact_sha256"],
            "coverage_metrics_input": artifact_records["coverage"]["artifact_sha256"],
            "historical_v27_v44_raw_artifacts_mutated": False,
            "deterministic": True,
        },
        "counts": {
            "coverage_source_records": EXPECTED_CELL_COUNT,
            "coverage_new_real_arms": EXPECTED_ARM_COUNT,
            "synthetic_arms": 0,
            "quality_judgments": 0,
        },
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "family_specific_ranking_supported": False,
        },
    }
    bundle = _address(bundle_payload)
    paths["bundle"] = _write_artifact(
        bundle, directory=output_directory, prefix="frontier-corrected-paper-input-bundle"
    )
    return bundle, paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--arena-base", type=Path, required=True)
    parser.add_argument("--strict-base", type=Path, required=True)
    parser.add_argument("--high-base", type=Path, required=True)
    parser.add_argument("--task-validity", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--historical-response-directory", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--coverage-source-directory",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/source"
        ),
    )
    parser.add_argument(
        "--coverage-corrections-directory",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/corrections"
        ),
    )
    parser.add_argument(
        "--coverage-response-directory",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/responses"
        ),
    )
    parser.add_argument(
        "--coverage-ledger",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/ledger.jsonl"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-corrected-paper-inputs-v1"
        ),
    )
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        bundle, paths = materialize_postrun(
            schedule_path=arguments.schedule,
            arena_base_path=arguments.arena_base,
            strict_base_path=arguments.strict_base,
            high_base_path=arguments.high_base,
            task_validity_path=arguments.task_validity,
            route_manifest_paths=arguments.route_manifest,
            historical_response_directories=arguments.historical_response_directory,
            coverage_source_directory=arguments.coverage_source_directory,
            coverage_corrections_directory=arguments.coverage_corrections_directory,
            coverage_response_directory=arguments.coverage_response_directory,
            coverage_ledger_path=arguments.coverage_ledger,
            output_directory=arguments.output_directory,
        )
    except IntegrityError as error:
        print(
            json.dumps(
                {
                    "status": "blocked_fail_closed",
                    "reason": str(error),
                    "provider_calls_made": 0,
                    "epicure_calls_made": 0,
                    "artifacts_written": 0,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    print(
        json.dumps(
            {
                "status": bundle["status"],
                "bundle_sha256": bundle["artifact_sha256"],
                "paths": {label: str(path.resolve()) for label, path in paths.items()},
                "provider_calls_made": 0,
                "epicure_calls_made": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
