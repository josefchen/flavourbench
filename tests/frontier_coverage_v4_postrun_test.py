from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from flavourbench.frontier_coverage_v4_postrun import (
    AGGREGATE_AUDIT_SHA256,
    COHERE_ROUTE_GATE_SCHEMA_VERSION,
    CORRECTED_ARENA_SHA256,
    CORRECTED_COVERAGE_SHA256,
    PRIMARY_PLAN_SCHEMA_VERSION,
    PRIMARY_PLAN_SHA256,
    PRIMARY_PREFLIGHT_SCHEMA_VERSION,
    _cohere_projection_contract,
)
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
POSTRUN = CURRENT / "frontier-coverage-v4-postrun"
PRIMARY = CURRENT / "frontier-coverage-primary-on-v5"
PLAN = PRIMARY / f"frontier-coverage-primary-on-v5-plan-{PRIMARY_PLAN_SHA256}.json"
GATE = (
    PRIMARY
    / "route-gate"
    / (
        "frontier-coverage-primary-cohere-route-gate-"
        "a89c319e32ba169645173809b1019a51b549dfdc22cab75f06c4d5718cb8f918.json"
    )
)
PREFLIGHT = (
    PRIMARY
    / "preflight"
    / (
        "frontier-coverage-primary-preflight-"
        "4b0be120e32f5f8e448742a1411ed48cccf64f0af29c359a28cf0f6a1eaa1797.json"
    )
)


def _addressed(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("artifact_sha256")
    assert sha256_json(value) == digest
    return {**value, "artifact_sha256": digest}


def test_source_reconstructed_aggregate_has_exact_corrected_grain() -> None:
    audit = _addressed(
        POSTRUN / f"frontier-coverage-v4-aggregate-audit-{AGGREGATE_AUDIT_SHA256}.json"
    )
    arena = _addressed(
        POSTRUN / f"frontier-corrected-development-arena-{CORRECTED_ARENA_SHA256}.json"
    )
    coverage = _addressed(
        POSTRUN / f"frontier-corrected-development-coverage-{CORRECTED_COVERAGE_SHA256}.json"
    )
    assert audit["counts"] == {
        "corrected_arena_comparisons": 915,
        "corrected_arena_unique_arms": 192,
        "corrected_arena_unpaired_arms": 4,
        "corrected_uplift_arms": 374,
        "corrected_uplift_pairs": 187,
        "missing_model_pair_family_cells": 73,
        "residual_failure_cells": 5,
        "synthetic_arms": 0,
        "usable_cells": 8,
    }
    assert arena["observed"]["candidate_comparisons"] == 915
    assert coverage["inference"]["cluster_by_task"] is True
    assert coverage["inference"]["cluster_by_response"] is True


def test_primary_plan_is_uniform_fresh_on_only_and_projects_zero_holes() -> None:
    plan = _addressed(PLAN)
    assert plan["schema_version"] == PRIMARY_PLAN_SCHEMA_VERSION
    assert plan["counts"]["primary_fresh_real_arms"] == 50
    assert plan["counts"]["primary_epicure_off_arms"] == 0
    assert plan["support_reconstruction"]["before"]["missing_cells"] == 73
    assert plan["support_reconstruction"]["projected_after_all_usable"]["missing_cells"] == 0
    assert plan["support_reconstruction"]["projected_after_all_usable"]["comparisons"] == 1281
    assert plan["primary_protocol"]["execution_policy_sha256"] == (
        "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d"
    )
    identifiers: set[str] = set()
    for cell in plan["cells"]:
        assert cell["conditions"] == ["epicure_on"]
        assert cell["attempt_slot_contract"]["attempt_slot_count"] == 29
        assert len(cell["attempt_slots"]) == 29
        assert not any(slot["phase"] == "tool_round_3" for slot in cell["attempt_slots"])
        assert not any(slot["phase"].startswith("mcp_tool_3_") for slot in cell["attempt_slots"])
        local = {
            cell["cell_id"],
            cell["work_item_id"],
            cell["run_id"],
            cell["arm_ids"]["epicure_on"],
            *(slot["attempt_id"] for slot in cell["attempt_slots"]),
        }
        assert not identifiers.intersection(local)
        identifiers.update(local)


def test_cohere_gate_is_offline_and_preserves_opaque_continuation() -> None:
    gate = _addressed(GATE)
    projection = _cohere_projection_contract()
    assert gate["schema_version"] == COHERE_ROUTE_GATE_SCHEMA_VERSION
    assert gate["verification"]["passed_tests"] == 5
    assert gate["calls"] == {"epicure": 0, "provider": 0}
    assert gate["contract"]["projection"] == projection
    assert all(projection["checks"].values())
    assert gate["decision"]["paid_execution_admission_granted"] is False


def test_preflight_rebases_all_terminal_costs_but_does_not_admit() -> None:
    preflight = _addressed(PREFLIGHT)
    budget = preflight["budget"]
    assert preflight["schema_version"] == PRIMARY_PREFLIGHT_SCHEMA_VERSION
    assert preflight["status"] == ("budget_fits_but_blocked_pending_independent_governance_go")
    assert Decimal(budget["rebased_current_exposure_usd"]) == Decimal(
        "48.01944682666666666666666666"
    )
    assert Decimal(budget["projected_total_exposure_usd"]) == Decimal(
        "82.27921310999999999999999999"
    )
    assert Decimal(budget["projected_total_exposure_usd"]) < Decimal("85")
    assert budget["admission_granted"] is False
    assert preflight["execution_gate"]["independent_governance_go_required"] is True
    assert preflight["calls"] == {"epicure": 0, "provider": 0}
