"""Freeze a real-call schedule that closes exact-frontier family coverage holes.

The schedule is an append-only development contract.  It never fabricates an
answer and never performs a provider call.  For each culinary family it chooses
the non-quarantined task with the greatest existing matched-pair coverage, then
lists the exact endpoint-task conditions still needed to give every endpoint a
common matched anchor in that family.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .current_frontier_task_quarantine import (
    quarantine_binding,
    quarantine_task_ids,
)
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-frontier-coverage-repair-schedule-v1"
FAMILIES = ("composition", "cookability", "evidence", "substitution")


class FrontierCoverageRepairError(ValueError):
    """Coverage inputs or the resulting collection contract are invalid."""


def _verify_content_address(document: Mapping[str, Any], label: str) -> None:
    digest = document.get("artifact_sha256")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if not isinstance(digest, str) or digest != sha256_json(payload):
        raise FrontierCoverageRepairError(f"{label} content address is invalid")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FrontierCoverageRepairError(f"{path} is not a JSON object")
    return value


def _verify_task_validity(document: Mapping[str, Any]) -> None:
    _verify_content_address(document, "task validity")
    if (
        document.get("schema_version") != "flavourbench-development-task-validity-v2"
        or document.get("claim_boundary", {}).get("synthetic_tasks") != 0
        or document.get("claim_boundary", {}).get("supports_official_leaderboard")
        is not False
    ):
        raise FrontierCoverageRepairError(
            "coverage repair requires the real, non-confirmatory development validity dossier"
        )


def build_coverage_repair_schedule(
    *,
    arena_pool: Mapping[str, Any],
    uplift_pools: Sequence[Mapping[str, Any]],
    task_validity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic no-call contract for the missing real arm cells."""

    _verify_content_address(arena_pool, "model-arena pool")
    _verify_task_validity(task_validity)
    if arena_pool.get("track") != "model_arena":
        raise FrontierCoverageRepairError("coverage input is not a model-arena pool")
    if not uplift_pools:
        raise FrontierCoverageRepairError("at least one uplift pool is required")
    for index, pool in enumerate(uplift_pools):
        _verify_content_address(pool, f"uplift pool {index}")
        if pool.get("claim_boundary", {}).get("synthetic_arms") not in {None, 0}:
            raise FrontierCoverageRepairError("synthetic uplift arms are prohibited")
        if pool.get("observed", {}).get("synthetic_arms") != 0:
            raise FrontierCoverageRepairError("synthetic uplift arms are prohibited")

    expected_quarantine = quarantine_binding()
    observed_quarantine = arena_pool.get("selection_policy", {}).get("task_quarantine")
    if observed_quarantine != expected_quarantine:
        raise FrontierCoverageRepairError("arena pool is not bound to the current quarantine")
    for pool in uplift_pools:
        if pool.get("selection_policy", {}).get("task_quarantine") != expected_quarantine:
            raise FrontierCoverageRepairError(
                "uplift pool is not bound to the current quarantine"
            )

    roster = tuple(str(value) for value in arena_pool.get("model_order", []))
    if len(roster) < 2 or len(set(roster)) != len(roster):
        raise FrontierCoverageRepairError("arena roster is missing or duplicated")
    model_contracts = arena_pool.get("model_contracts")
    if not isinstance(model_contracts, dict) or set(model_contracts) != set(roster):
        raise FrontierCoverageRepairError("arena model contracts do not bind the roster")

    held = quarantine_task_ids()
    on_arms: set[tuple[str, str]] = set()
    task_metadata: dict[str, dict[str, str]] = {}
    current_pair_family_support: set[tuple[str, str, str]] = set()
    for raw_item in arena_pool.get("items", []):
        if not isinstance(raw_item, dict):
            raise FrontierCoverageRepairError("arena item is malformed")
        task_id = str(raw_item["task_id"])
        if task_id in held:
            raise FrontierCoverageRepairError("quarantined task entered corrected arena pool")
        family = str(raw_item["task_family"])
        prompt_sha256 = str(raw_item["prompt_sha256"])
        prior = task_metadata.setdefault(
            task_id,
            {
                "task_id": task_id,
                "family": family,
                "prompt_sha256": prompt_sha256,
            },
        )
        if prior["family"] != family or prior["prompt_sha256"] != prompt_sha256:
            raise FrontierCoverageRepairError("task metadata changes within the arena pool")
        side_models: list[str] = []
        for side in ("left", "right"):
            side_record = raw_item.get(side)
            if not isinstance(side_record, dict):
                raise FrontierCoverageRepairError("arena side record is malformed")
            model_id = str(side_record["requested_model_id"])
            on_arms.add((task_id, model_id))
            side_models.append(model_id)
        first, second = sorted(side_models)
        current_pair_family_support.add((first, second, family))

    paired_arms: set[tuple[str, str]] = set()
    for pool in uplift_pools:
        pool_roster = tuple(str(value) for value in pool.get("model_order", []))
        if not set(pool_roster) <= set(roster) or pool_roster != tuple(
            model_id for model_id in roster if model_id in set(pool_roster)
        ):
            # Stratum-specific pools can omit the Cohere extension but cannot add
            # or reorder an endpoint outside the arena roster.
            raise FrontierCoverageRepairError("uplift roster crosses the arena contract")
        for raw_item in pool.get("items", []):
            if not isinstance(raw_item, dict):
                raise FrontierCoverageRepairError("uplift item is malformed")
            task_id = str(raw_item["task_id"])
            model_id = str(raw_item["requested_model_id"])
            if task_id in held:
                raise FrontierCoverageRepairError("quarantined task entered corrected uplift pool")
            if model_id not in roster:
                raise FrontierCoverageRepairError("uplift endpoint is outside the arena roster")
            paired_arms.add((task_id, model_id))

    validity_tasks = {
        str(row["task_id"]): row
        for row in task_validity.get("tasks", [])
        if isinstance(row, dict)
    }
    for task_id, metadata in task_metadata.items():
        validity = validity_tasks.get(task_id)
        if (
            validity is None
            or validity.get("prompt_sha256") != metadata["prompt_sha256"]
            or validity.get("rank_eligible") is not False
        ):
            raise FrontierCoverageRepairError(
                "arena task is not bound to the supplied development validity dossier"
            )

    anchors: list[dict[str, Any]] = []
    for family in FAMILIES:
        candidates = [
            metadata for metadata in task_metadata.values() if metadata["family"] == family
        ]
        if not candidates:
            raise FrontierCoverageRepairError(f"no non-quarantined {family} task remains")
        chosen = min(
            candidates,
            key=lambda row: (
                -sum((row["task_id"], model_id) in paired_arms for model_id in roster),
                -sum((row["task_id"], model_id) in on_arms for model_id in roster),
                row["task_id"],
            ),
        )
        anchors.append(
            {
                **chosen,
                "selection_rule": (
                    "maximum existing matched-pair endpoint coverage, then maximum "
                    "Epicure-on endpoint coverage, then lexical task ID"
                ),
                "existing_matched_endpoint_count": sum(
                    (chosen["task_id"], model_id) in paired_arms for model_id in roster
                ),
                "existing_epicure_on_endpoint_count": sum(
                    (chosen["task_id"], model_id) in on_arms for model_id in roster
                ),
            }
        )

    missing_cells: list[dict[str, Any]] = []
    for anchor in anchors:
        task_id = str(anchor["task_id"])
        family = str(anchor["family"])
        for model_id in roster:
            has_pair = (task_id, model_id) in paired_arms
            has_on = (task_id, model_id) in on_arms
            existing_conditions = ["epicure_off", "epicure_on"] if has_pair else (
                ["epicure_on"] if has_on else []
            )
            required_conditions = [] if has_pair else (
                ["epicure_off"] if has_on else ["epicure_off", "epicure_on"]
            )
            if required_conditions:
                missing_cells.append(
                    {
                        "schedule_cell_id": sha256_json(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "family": family,
                                "task_id": task_id,
                                "model_id": model_id,
                                "required_conditions": required_conditions,
                            }
                        ),
                        "family": family,
                        "task_id": task_id,
                        "prompt_sha256": anchor["prompt_sha256"],
                        "model_id": model_id,
                        "model_contract": model_contracts[model_id],
                        "existing_real_conditions": existing_conditions,
                        "required_new_conditions": required_conditions,
                        "required_new_real_arms": len(required_conditions),
                    }
                )

    total_pair_family_cells = len(tuple(itertools.combinations(roster, 2))) * len(FAMILIES)
    current_missing = total_pair_family_cells - len(current_pair_family_support)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "append_only_real_call_coverage_repair_contract",
        "status": "frozen_development_schedule_no_calls_executed",
        "source": {
            "arena_pool_sha256": arena_pool["artifact_sha256"],
            "uplift_pool_sha256s": sorted(
                str(pool["artifact_sha256"]) for pool in uplift_pools
            ),
            "task_validity_sha256": task_validity["artifact_sha256"],
            "task_quarantine_sha256": expected_quarantine["artifact_sha256"],
        },
        "roster": list(roster),
        "anchors": anchors,
        "missing_endpoint_task_cells": missing_cells,
        "counts": {
            "models": len(roster),
            "families": len(FAMILIES),
            "anchor_endpoint_task_cells": len(roster) * len(FAMILIES),
            "already_complete_matched_anchor_cells": len(roster) * len(FAMILIES)
            - len(missing_cells),
            "missing_endpoint_task_cells": len(missing_cells),
            "required_new_real_arms": sum(
                int(row["required_new_real_arms"]) for row in missing_cells
            ),
            "current_model_pair_family_cells": total_pair_family_cells,
            "current_missing_model_pair_family_cells": current_missing,
            "projected_missing_model_pair_family_cells_after_schedule": 0,
            "synthetic_tasks": 0,
            "synthetic_arms": 0,
        },
        "collection_policy": {
            "conditions": ["epicure_off", "epicure_on"],
            "reuse_existing_content_addressed_real_arms": True,
            "provider_calls_per_missing_condition": 1,
            "provider_fallbacks": False,
            "provider_substitution": False,
            "task_family_balance_gate": (
                "every endpoint must have a complete matched pair on the common family anchor"
            ),
            "model_pair_family_balance_gate": (
                "every unordered model pair must share at least one Epicure-on task per family"
            ),
            "paid_calls_executed_by_this_artifact": 0,
        },
        "quarantine": expected_quarantine,
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "quality_judgments": 0,
            "quality_ranking": False,
            "epicure_uplift_estimate": False,
            "zero_synthetic_tasks": True,
            "zero_synthetic_arms": True,
            "schedule_completion_is_not_sufficient_for_publication": True,
        },
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def write_coverage_repair_schedule(value: Mapping[str, Any], output_dir: Path) -> Path:
    _verify_content_address(value, "coverage repair schedule")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"frontier-coverage-repair-{value['artifact_sha256']}.json"
    rendered = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise FrontierCoverageRepairError("content-addressed schedule output conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-pool", type=Path, required=True)
    parser.add_argument("--uplift-pool", type=Path, action="append", required=True)
    parser.add_argument("--task-validity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    schedule = build_coverage_repair_schedule(
        arena_pool=_load(arguments.arena_pool),
        uplift_pools=[_load(path) for path in arguments.uplift_pool],
        task_validity=_load(arguments.task_validity),
    )
    path = write_coverage_repair_schedule(schedule, arguments.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": schedule["status"],
                "artifact_sha256": schedule["artifact_sha256"],
                "counts": schedule["counts"],
                "anchors": schedule["anchors"],
                "output": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
