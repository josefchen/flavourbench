from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from flavourbench.reasoning_effort_sensitivity_v4 import MODELS, verify_frozen

REPO = Path(__file__).resolve().parents[2]
ROOT = (
    REPO
    / "flavourbench/artifacts/season1/current-quality-run"
    / "reasoning-effort-sensitivity-v4"
)


def _one(pattern: str) -> Path:
    matches = list(ROOT.glob(pattern))
    assert len(matches) == 1
    return matches[0]


def _load(pattern: str) -> dict:
    return json.loads(_one(pattern).read_text(encoding="utf-8"))


def test_frozen_no_call_package_reconstructs() -> None:
    paths = {
        "history_path": _one("reasoning-effort-v4-history-audit-*.json"),
        "baseline_path": _one("reasoning-effort-v4-low-baseline-audit-*.json"),
        "route_plan_path": _one("reasoning-effort-v4-route-gate-plan-*.json"),
        "study_plan_path": _one("reasoning-effort-v4-study-plan-*.json"),
        "runner_assets_path": _one("reasoning-effort-v4-runner-assets-*.json"),
        "preflight_path": _one("reasoning-effort-v4-preflight-*.json"),
    }
    assert verify_frozen(repo_root=REPO, **paths)

    preflight = json.loads(paths["preflight_path"].read_text(encoding="utf-8"))
    assert preflight["decision"] == (
        "blocked_before_full_provider_calls_pending_six_pair_route_gate"
    )
    assert preflight["provider_calls_made_by_preflight"] is False
    assert preflight["epicure_calls_made_by_preflight"] is False


def test_design_is_real_balanced_bounded_and_nonranking() -> None:
    baseline = _load("reasoning-effort-v4-low-baseline-audit-*.json")
    route = _load("reasoning-effort-v4-route-gate-plan-*.json")
    study = _load("reasoning-effort-v4-study-plan-*.json")
    assets = _load("reasoning-effort-v4-runner-assets-*.json")

    assert baseline["counts"]["source_reconstructed_complete_low_pairs"] == 23
    assert baseline["counts"]["immutable_missing_low_pairs"] == 1
    assert baseline["counts"]["synthetic_arms"] == 0
    assert route["counts"] == {
        "effort_variants": 2,
        "matched_pairs": 6,
        "models": 3,
        "quality_observations": 0,
        "response_arms": 12,
        "synthetic_arms": 0,
    }
    assert {item["model_id"] for item in study["models"]} == set(MODELS)
    assert Counter(task["family"] for task in study["tasks"]) == Counter(
        {"substitution": 2, "composition": 2, "cookability": 2, "evidence": 2}
    )
    assert study["factorial_design"]["new_default_high_pairs"] == 48
    assert study["factorial_design"]["synthetic_arms"] == 0
    assert Decimal(study["budget"]["total_new_worst_case_usd"]) == Decimal(
        "26.98645665333333333333333334"
    )
    assert Decimal(study["budget"]["projected_total_exposure_usd"]) < Decimal("85")
    assert study["claim_boundary"]["enters_primary_leaderboard"] is False

    assert len(assets["execution_schedule"]) == 48
    blocks: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for row in assets["execution_schedule"]:
        key = (row["panel_id"], row["model_id"], row["task_id"])
        blocks[key] = tuple(row["effort_order_in_block"])
    assert len(blocks) == 24
    assert Counter(order[0] for order in blocks.values()) == Counter(
        {"provider_default": 12, "explicit_high": 12}
    )
    for variant in assets["variants"]:
        command = variant["single_pair_dry_run_command"]
        assert command[command.index("--max-new-pairs") + 1] == "1"
        assert "--execute" not in command
    assert assets["execution_command"]["implemented"] is False
