from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import flavourbench.epicure_selection_route_manifest_v79 as manifest_v79
from flavourbench.epicure_selection_powered_plan_v80 import (
    build_plan as build_panel_1,
)
from flavourbench.epicure_selection_powered_plan_v80 import verify_plan as verify_panel_1
from flavourbench.epicure_selection_powered_plan_v81 import (
    build_plan as build_panel_2,
)
from flavourbench.epicure_selection_powered_plan_v81 import verify_plan as verify_panel_2
from flavourbench.epicure_selection_powered_runner import validate_inputs
from flavourbench.epicure_selection_route_manifest_v45 import FABLE_MODEL_ID
from flavourbench.epicure_selection_route_manifest_v79 import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / (
    "benchmark/powered-v73/manifest/"
    "flavourbench-frontier-refresh-27-"
    "993e6ad499f7a26f52baf25cc22c52e8a8938703365dc1814f0c17b156d73368.json"
)
PANEL_1_PREDECESSOR = ROOT / (
    "benchmark/powered-v74/plan/"
    "epicure-selection-analysis-plan-"
    "55fef84440b6dda3db7ad44ba3947a3b2d81f127c2ad81fce7e6fbd38c8df6c0.json"
)
PANEL_2_PREDECESSOR = ROOT / (
    "benchmark/powered-v75/plan/"
    "epicure-selection-analysis-plan-"
    "48d9d8f12d6da1910621d15ea7f26750119745ecde3b94fb8dfb3ec5c382cf75.json"
)
MANIFEST = ROOT / (
    "benchmark/powered-v79/manifest/"
    "flavourbench-frontier-refresh-27-"
    "5387bcaba56551d72f54694606c7e17fafaa00343c6f9d44cabf13b221481971.json"
)
PANEL_1 = ROOT / (
    "benchmark/powered-v80/plan/"
    "epicure-selection-analysis-plan-"
    "d8b699ea0e01e42edf93e2fe433d469ce55a6365c27d2038b65bc09ff01e5c6a.json"
)
PANEL_2 = ROOT / (
    "benchmark/powered-v81/plan/"
    "epicure-selection-analysis-plan-"
    "b126bacf50c05c1a548056c153280ceba5f9e8b182bea95df216abb430c02e0b.json"
)
TASKSET_1 = ROOT / (
    "benchmark/powered-v44/taskset/"
    "epicure-selection-taskset-"
    "a33bf28db372090015118371417b0e8ed1254f416d03d2c2c5816a6a752beb41.json"
)
REPEAT_1 = ROOT / (
    "benchmark/powered-v44/plan/"
    "epicure-selection-repeat-panel-"
    "96f766df855b93ad1495ec386c70ad88e42c4f896be24b0538cf1084da3c124a.json"
)
TASKSET_2 = ROOT / (
    "benchmark/powered-v45/taskset/"
    "epicure-selection-taskset-"
    "925ba9d1d4be9c2b7a1e9956ecd6c18d34ffcad22eee28522f16892922c91e3f.json"
)
REPEAT_2 = ROOT / (
    "benchmark/powered-v45/plan/"
    "epicure-selection-repeat-panel-"
    "36d8c12ff883ead78e53406844ad386eb8999168a61d6931fe17135a2c73acfe.json"
)
PREDECESSOR_RELEASE = ROOT / (
    "paper/generated/powered/"
    "flavourbench-powered-release-"
    "7aeddf27998b0a8ed0b961cab035e4793305ea120f73ce8f3baa47e4db612cf7.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fable_manifest_changes_only_the_exact_first_party_route(monkeypatch) -> None:
    source = _load(SOURCE_MANIFEST)
    prior_row = next(row for row in source["models"] if row["model"]["id"] == FABLE_MODEL_ID)
    endpoint = copy.deepcopy(prior_row["endpoint"])
    endpoint.update(
        {
            "tag": "anthropic",
            "provider_name": "Anthropic",
            "status": 0,
            "supported_parameters": ["max_tokens", "reasoning", "response_format"],
        }
    )

    async def fetch(model_id: str):
        assert model_id == FABLE_MODEL_ID
        return [endpoint]

    monkeypatch.setattr(manifest_v79, "_fetch_endpoints", fetch)
    successor = asyncio.run(manifest_v79.build(source_path=SOURCE_MANIFEST))
    assert verify_manifest(successor)
    source_rows = {row["model"]["id"]: row for row in source["models"]}
    final_rows = {row["model"]["id"]: row for row in successor["models"]}
    assert final_rows[FABLE_MODEL_ID]["endpoint"]["tag"] == "anthropic"
    assert all(
        final_rows[model_id] == row
        for model_id, row in source_rows.items()
        if model_id != FABLE_MODEL_ID
    )


def test_both_fable_plans_preserve_every_non_fable_roster_row() -> None:
    manifest = _load(MANIFEST)
    for predecessor_path, builder, verifier, panel in (
        (PANEL_1_PREDECESSOR, build_panel_1, verify_panel_1, 1),
        (PANEL_2_PREDECESSOR, build_panel_2, verify_panel_2, 2),
    ):
        predecessor = _load(predecessor_path)
        successor = builder(
            predecessor=predecessor,
            predecessor_physical_sha256=manifest_v79._sha256_file(predecessor_path),
            manifest=manifest,
            manifest_physical_sha256=manifest_v79._sha256_file(MANIFEST),
        )
        assert verifier(successor)
        prior_rows = {row["model_id"]: row for row in predecessor["roster"]["models"]}
        final_rows = {row["model_id"]: row for row in successor["roster"]["models"]}
        assert final_rows[FABLE_MODEL_ID]["provider_tag"] == "anthropic"
        assert all(
            final_rows[model_id] == row
            for model_id, row in prior_rows.items()
            if model_id != FABLE_MODEL_ID
        )
        assert successor["budget"]["successor_scope"].endswith(
            f"panel-{panel} Fable first-party block"
        )


def test_powered_runner_accepts_both_materialized_fable_plans() -> None:
    manifest = _load(MANIFEST)
    for taskset, repeat, plan in (
        (TASKSET_1, REPEAT_1, PANEL_1),
        (TASKSET_2, REPEAT_2, PANEL_2),
    ):
        loaded = validate_inputs(
            manifest_path=MANIFEST,
            manifest_sha256=manifest["content_address"]["digest"],
            taskset_path=taskset,
            repeat_panel_path=repeat,
            plan_path=plan,
            predecessor_release_path=PREDECESSOR_RELEASE,
        )
        assert loaded[3]["artifact_sha256"] == _load(plan)["artifact_sha256"]
