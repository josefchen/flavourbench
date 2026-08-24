from __future__ import annotations

import copy
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

import flavourbench.reasoning_effort_full_study_executor_v1 as executor
import flavourbench.reasoning_effort_full_study_v1 as study
from flavourbench.response_envelope_route_v4 import _policy_from_manifest

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
STUDY_ROOT = ROOT / "artifacts/season1/current-quality-run/reasoning-effort-task-waves-v3"
PLAN = (
    STUDY_ROOT
    / "plan"
    / (
        "reasoning-effort-task-wave-plan-v2-"
        "99b8f70ae81aa3a7b7e79a45bb4253cb58d26306f90ab5b9c4f09a6938f1a301.json"
    )
)
PREFLIGHT = (
    STUDY_ROOT
    / "preflight"
    / (
        "reasoning-effort-task-wave-preflight-v2-"
        "8d0e79d9ccbe0b62aa2486421757d3d31739aac4accea594cb58bdd18bd94f04.json"
    )
)
HUMAN_PROTOCOL = (
    ROOT
    / "artifacts/season1/current-quality-run/reasoning-effort-human-protocol-v2"
    / (
        "reasoning-effort-human-protocol-"
        "cd2a234f617158304a5eb4efed1c6e34198cd857f2de124b10dee09fdec370a8.json"
    )
)
BOUND_PREFLIGHT = (
    STUDY_ROOT
    / "bound-preflight"
    / (
        "reasoning-effort-bound-admission-preflight-v2-"
        "58d509d8c9c4276ad9c497789652a9ba55a50320846b7dda5eb72853bfe25910.json"
    )
)
RETIRED_PLAN = (
    ROOT
    / "artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2/plan"
    / (
        "reasoning-effort-task-wave-plan-v2-"
        "03731cb5e509bc40ec733bc5c55ee91ad035b04e1c4adaf64684437751fb1f0c.json"
    )
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_plan_freezes_real_balanced_tasks_pairs_and_common_protocol() -> None:
    plan = _json(PLAN)
    study.validate_plan(plan, repo_root=REPO_ROOT)
    assert plan["artifact_sha256"] == (
        "99b8f70ae81aa3a7b7e79a45bb4253cb58d26306f90ab5b9c4f09a6938f1a301"
    )
    assert plan["supersedes"]["artifact_sha256"] == (
        "03731cb5e509bc40ec733bc5c55ee91ad035b04e1c4adaf64684437751fb1f0c"
    )
    source = plan["source_code"]
    assert source["schema_version"] == "flavourbench-reasoning-effort-source-closure-v1"
    assert len(source["modules"]) == 61
    assert len(source["import_edges"]) == 233
    assert len(source["execution_environment"]["distributions"]) == 30
    assert Counter(task["family"] for task in plan["tasks"]) == Counter(
        {family: 6 for family in study.TASK_FAMILIES}
    )
    assert not any(task["synthetic"] or task["quarantined"] for task in plan["tasks"])
    assert len(plan["work_items"]) == 168
    assert sum(len(item["arm_ids"]) for item in plan["work_items"]) == 336
    assert all(len(item["attempt_slots"]) == 56 for item in plan["work_items"])
    common = plan["common_protocol"]
    assert common["max_tool_calls_per_round"] == 13
    assert common["max_tool_calls_total"] == 13
    assert common["max_tool_rounds"] == 3
    assert common["max_output_tokens"] == 8192
    assert common["max_intermediate_tokens"] == 8192
    assert common["max_provider_attempts"] == 2
    assert common["evidence_protocol"] == "matched_evidence_v2"
    assert common["final_response_mode"] == "plain_text"
    assert common["pair_arm_scheduling"] == "concurrent"
    assert (
        plan["execution"][
            "fresh_catalog_attestation_for_all_three_endpoints_before_block_reservation"
        ]
        is True
    )
    for model in plan["models"].values():
        assert model["fresh_catalog_attestation_before_each_atomic_family_block"] is True
        assert "fresh_catalog_attestation_before_every_wave" not in model


def test_each_atomic_admission_block_preserves_exact_family_balance() -> None:
    plan = _json(PLAN)
    waves = {wave["wave_id"]: wave for wave in plan["task_waves"]}
    assert len(plan["admission_blocks"]) == 6
    assert plan["block_execution_order"] == [
        block["admission_block_id"] for block in plan["admission_blocks"]
    ]
    for block in plan["admission_blocks"]:
        assert Counter(block["task_families"]) == Counter(
            {family: 1 for family in study.TASK_FAMILIES}
        )
        assert [waves[wave_id]["task_family"] for wave_id in block["wave_ids"]] == block[
            "task_families"
        ]
        assert len(block["work_item_ids"]) == 28
        assert block["matched_pairs"] == 28
        assert block["response_arms"] == 56
        assert block["partial_block_start_permitted"] is False


def test_every_manifest_round_trips_the_frozen_policy_and_new_ids_are_disjoint() -> None:
    plan = _json(PLAN)
    for item in plan["work_items"]:
        manifest = executor._manifest(plan, item, REPO_ROOT)
        policy = executor._prospective_policy_from_manifest(manifest)
        coordinate = item["route_coordinate"]
        assert policy.pair_arm_scheduling == plan["common_protocol"]["pair_arm_scheduling"]
        assert policy.sha256 == coordinate["execution_policy_sha256"]
        assert policy.intermediate_reasoning_effort == coordinate["intermediate_reasoning_effort"]
        assert policy.final_reasoning_effort == coordinate["final_reasoning_effort"]


def test_manifest_policy_parser_preserves_legacy_and_requires_prospective_opt_in() -> None:
    plan = _json(PLAN)
    manifest = executor._manifest(plan, plan["work_items"][0], REPO_ROOT)
    assert _policy_from_manifest(manifest).pair_arm_scheduling == "sequential"
    assert (
        executor._prospective_policy_from_manifest(manifest).pair_arm_scheduling
        == "concurrent"
    )

    retired = _json(RETIRED_PLAN)

    def identities(document: dict) -> dict[str, set[str]]:
        work_items = document["work_items"]
        return {
            "work": {item["work_item_id"] for item in work_items},
            "run": {item["run_id"] for item in work_items},
            "arm": {arm_id for item in work_items for arm_id in item["arm_ids"]},
            "attempt": {
                slot["attempt_id"] for item in work_items for slot in item["attempt_slots"]
            },
            "presentation": {
                row["presentation_id"] for row in document["human_evaluation"]["presentations"]
            },
        }

    current_ids = identities(plan)
    retired_ids = identities(retired)
    assert all(current_ids[key].isdisjoint(retired_ids[key]) for key in current_ids)


def test_preliminary_preflight_cannot_claim_human_protocol_is_frozen() -> None:
    plan = _json(PLAN)
    preflight = _json(PREFLIGHT)
    assert study._artifact_ok(preflight, study.PREFLIGHT_SCHEMA)
    assert preflight["study_plan_sha256"] == plan["artifact_sha256"]
    assert preflight["decision"] == "awaiting_cross_bound_human_protocol"
    assert preflight["checks"]["human_protocol_frozen_and_cross_verified"] is False
    assert preflight["checks"]["budget_and_empty_roots_ready"] is True
    assert Decimal(preflight["checks"]["first_family_block_reserve_usd"]) == Decimal(
        plan["admission_blocks"][0]["worst_case_reserve_usd"]
    )
    assert Decimal(preflight["checks"]["first_family_block_projected_usd"]) < Decimal("85")
    assert preflight["execution"]["provider_or_epicure_calls_made_by_preflight"] is False


def test_external_human_graph_and_bound_preflight_cross_verify() -> None:
    plan = _json(PLAN)
    human = _json(HUMAN_PROTOCOL)
    bound = _json(BOUND_PREFLIGHT)
    study.verify_human_protocol_binding(plan=plan, human_protocol=human)
    study.verify_bound_preflight(plan=plan, human_protocol=human, bound_preflight=bound)
    assert human["supersedes"]["artifact_sha256"] == (
        "42fb1b5ea606034d4eb62eb813c957b87ffee44392e1c8f11322bf61fe7002ea"
    )
    assert bound["supersedes"]["artifact_sha256"] == (
        "9c5cb664b5708fccfa49e20f8c362736786e705b06444ed7c59f5013181e8d8e"
    )
    assert bound["decision"] == "first_family_block_admissible"
    assert bound["checks"]["human_arm_coordinates_verified"] == 336
    assert bound["checks"]["human_comparison_cells_verified"] == 240
    assert bound["calls_made_by_bound_preflight"] == {
        "provider_completions": 0,
        "epicure": 0,
        "catalog_gets": 0,
    }


def test_cross_verifier_rejects_a_self_consistent_but_wrong_human_arm() -> None:
    plan = _json(PLAN)
    human = copy.deepcopy(_json(HUMAN_PROTOCOL))
    human["arm_coordinates"][0]["condition"] = "epicure_wrong"
    body = {key: value for key, value in human.items() if key != "artifact_sha256"}
    human["artifact_sha256"] = study._sha256(body)
    with pytest.raises(study.FullStudyError, match="336-arm graph differs"):
        study.verify_human_protocol_binding(plan=plan, human_protocol=human)


def test_active_block_counts_the_whole_reserve_without_pair_reservations(
    tmp_path: Path,
) -> None:
    plan = _json(PLAN)
    block = plan["admission_blocks"][0]
    ledger = tmp_path / "coordinator/ledger.jsonl"
    reservation = executor._append_ledger(
        ledger,
        role="coordinator",
        event={
            "event_type": "family_block_reservation_created",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "work_item_ids": block["work_item_ids"],
            "reserved_usd": block["worst_case_reserve_usd"],
            "pair_reservations_created": 0,
        },
    )
    assert reservation["pair_reservations_created"] == 0
    state = executor._coordinator_state(plan, executor._load_ledger(ledger, role="coordinator"))
    assert state["active_block_id"] == block["admission_block_id"]
    endpoint_roots = {endpoint: tmp_path / endpoint for endpoint in study.ENDPOINTS}
    accounting = executor._accounting(
        plan=plan,
        repo_root=REPO_ROOT,
        coordinator_ledger=ledger,
        endpoint_roots=endpoint_roots,
    )
    assert Decimal(accounting["active_block_full_reserve_usd"]) == Decimal(
        block["worst_case_reserve_usd"]
    )
    assert accounting["active_block_sources_counted_inside_reserve_only"] is True
    assert accounting["new_block_admission_allowed"] is False


def test_zero_cost_terminal_block_releases_only_the_block_reservation(
    tmp_path: Path,
) -> None:
    plan = _json(PLAN)
    block = plan["admission_blocks"][0]
    ledger = tmp_path / "coordinator/ledger.jsonl"
    reservation = executor._append_ledger(
        ledger,
        role="coordinator",
        event={
            "event_type": "family_block_reservation_created",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "work_item_ids": block["work_item_ids"],
            "reserved_usd": block["worst_case_reserve_usd"],
        },
    )
    for item_id in block["work_item_ids"]:
        executor._append_ledger(
            ledger,
            role="coordinator",
            event={
                "event_type": "family_block_item_terminalized",
                "study_plan_sha256": plan["artifact_sha256"],
                "admission_block_id": block["admission_block_id"],
                "task_wave_id": executor._item_wave_id(plan, item_id),
                "work_item_id": item_id,
                "block_reservation_entry_sha256": reservation["entry_sha256"],
                "disposition": "pre_generation_failure_zero_cost",
                "actual_cost_usd": "0",
            },
        )
    terminal = executor._terminalize_block(plan=plan, block=block, coordinator_ledger=ledger)
    assert terminal["event_type"] == "family_block_terminalized"
    assert terminal["terminal_pairs"] == 28
    assert terminal["pre_generation_failure_pairs"] == 28
    assert terminal["actual_cost_usd"] == "0"
    endpoint_roots = {endpoint: tmp_path / endpoint for endpoint in study.ENDPOINTS}
    accounting = executor._accounting(
        plan=plan,
        repo_root=REPO_ROOT,
        coordinator_ledger=ledger,
        endpoint_roots=endpoint_roots,
    )
    assert accounting["completed_family_blocks"] == 1
    assert accounting["completed_task_waves"] == 4
    assert accounting["active_block_id"] is None
    assert accounting["next_block_id"] == plan["admission_blocks"][1]["admission_block_id"]
