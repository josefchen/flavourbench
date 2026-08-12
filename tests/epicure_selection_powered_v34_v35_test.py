from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from flavourbench.epicure_selection_powered_plan_v34 import (
    PREDECESSOR_MODEL_IDS,
    SUCCESSOR_MODEL_IDS,
)
from flavourbench.epicure_selection_powered_plan_v34 import (
    verify_plan as verify_v34_plan,
)
from flavourbench.epicure_selection_powered_plan_v35 import (
    COHERE_CONCURRENCY,
    transport_commitment,
)
from flavourbench.epicure_selection_powered_plan_v35 import (
    verify_plan as verify_v35_plan,
)
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.epicure_selection_route_manifest_v32 import ROUTE_MAX_OUTPUT_TOKENS

ROOT = Path(__file__).resolve().parents[1]


def _only_json(directory: Path) -> Path:
    paths = list(directory.glob("*.json"))
    assert len(paths) == 1
    return paths[0]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _physical(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_clean_cohere_successor_plans_are_exact_and_non_pooled() -> None:
    v33_path = _only_json(ROOT / "benchmark/powered-v33/plan")
    v34_path = _only_json(ROOT / "benchmark/powered-v34/plan")
    v35_path = _only_json(ROOT / "benchmark/powered-v35/plan")
    v33 = _load(v33_path)
    v34 = _load(v34_path)
    v35 = _load(v35_path)
    assert verify_v34_plan(v34)
    assert verify_v35_plan(v35)
    assert v34["inputs"]["plan_v33_predecessor"] == {
        "semantic_sha256": v33["artifact_sha256"],
        "physical_sha256": _physical(v33_path),
    }
    assert v35["inputs"]["plan_v34_predecessor"] == {
        "semantic_sha256": v34["artifact_sha256"],
        "physical_sha256": _physical(v34_path),
    }
    rows = {row["model_id"]: row for row in v35["roster"]["models"]}
    assert not set(PREDECESSOR_MODEL_IDS) & set(rows)
    assert set(SUCCESSOR_MODEL_IDS) <= set(rows)
    for model_id in SUCCESSOR_MODEL_IDS:
        assert rows[model_id]["final_max_output_tokens"] == ROUTE_MAX_OUTPUT_TOKENS
        assert (
            v35["execution"]["collection_concurrency"]["per_model_by_model_id"][model_id]
            == COHERE_CONCURRENCY
        )
    lineage = v35["execution"]["cohere_route_successor"]
    assert lineage["reuse_direct_responses"] is False
    assert lineage["cross_provider_score_pooling"] is False
    assert lineage["transport_checks_reused_as_primary"] is False


def test_v35_binds_all_sixteen_predetermined_transport_responses() -> None:
    raw_run = ROOT / "benchmark/powered-v34/run"
    if not (raw_run / "attempts/provider-attempts.jsonl").is_file():
        pytest.skip("raw pre-release transport calibration is outside the compact public checkout")
    v34_path = _only_json(ROOT / "benchmark/powered-v34/plan")
    v35_path = _only_json(ROOT / "benchmark/powered-v35/plan")
    v34 = _load(v34_path)
    v35 = _load(v35_path)
    observed = transport_commitment(
        raw_run,
        expected_plan_sha256=str(v34["artifact_sha256"]),
    )
    recorded = v35["inputs"]["calibration_v34"]
    assert all(recorded[key] == value for key, value in observed.items())
    assert observed["response_count"] == 16
    assert observed["used_as_primary_data"] is False
    assert observed["scores_or_selections_inspected_before_successor_freeze"] is False


def test_cohere_generation_uses_the_frozen_route_specific_ceiling() -> None:
    manifest_path = _only_json(ROOT / "benchmark/powered-v34/manifest")
    plan_path = _only_json(ROOT / "benchmark/powered-v35/plan")
    taskset_path = _only_json(ROOT / "benchmark/powered-v16/taskset")
    repeat_path = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
    predecessor_path = ROOT / "paper/generated/epicure-native/epicure-native-release.json"
    manifest, taskset, repeat, plan, _, candidates = validate_inputs(
        manifest_path=manifest_path,
        manifest_sha256=str(_load(manifest_path)["content_address"]["digest"]),
        taskset_path=taskset_path,
        repeat_panel_path=repeat_path,
        plan_path=plan_path,
        predecessor_release_path=predecessor_path,
    )
    command_a = next(
        candidate for candidate in candidates if candidate.model_id == "cohere/command-a"
    )
    cell = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=[command_a],
        phase="primary",
    )[0]
    spec, _ = build_generation_spec(
        cell=cell,
        plan=plan,
        manifest_sha256=manifest["content_address"]["digest"],
        taskset=taskset,
        reserve_micros=100_000,
        execution_policy=selection_execution_policy_v31(),
    )
    assert spec.decoding_parameters is not None
    assert spec.decoding_parameters["max_tokens"] == ROUTE_MAX_OUTPUT_TOKENS
    assert spec.final_reasoning_effort is None
