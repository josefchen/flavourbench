from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from flavourbench.qwen_exploratory_release_projection import (
    EXPECTED,
    build_qwen_projection,
)

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
QWEN = CURRENT / "qwencloud-smoke-v1"
PREDECESSOR = QWEN / "runs/20260808T205957Z-a9e863df14ef.json"
PREDECESSOR_JOURNAL = QWEN / (
    "runs/flavourbench-live-smoke-journal-"
    "2db2e258607889d833cd3680ec573e6c49a6cd9bc882207cc76053b78b3b03f3.jsonl"
)
SUCCESSOR = QWEN / "runs-successor-v2/20260808T211702Z-124a38f1c990.json"
SUCCESSOR_JOURNAL = QWEN / (
    "runs-successor-v2/flavourbench-live-smoke-journal-"
    "e7c04218f1e910cee466e80d9ea9be917215db0a66a2e7022252d4acc6f9835d.jsonl"
)
RECOVERY = QWEN / (
    "governance/qwencloud-zero-call-recovery-"
    "7bb5f1392a2422437edc138b14940cd92736caa6bc6328acbf4b2dd73e8d479a.json"
)
LEDGER = QWEN / "governance/qwencloud-exploratory-ledger-v1.jsonl"
ROUTE = CURRENT / (
    "qwencloud-route-20260808-successor-v2/qwencloud-route-manifest-"
    "1e646c713945f9e492be99a49daae139cf8c6b799cbedeb5fc197d285771f0d4.json"
)
PREFLIGHT = QWEN / (
    "preflight-successor-v2/preflight-"
    "22580b7bd1c38039f9bbdfe2061974bf0944819ea4063a431381caf39d48b1d5.json"
)
GO_TEMPLATE = QWEN / (
    "governance/qwencloud-one-pair-pi-go-template-"
    "324fa0408946eb483a7103c305f4a394ebd6d24f6aee79643cd832500c57dea1.json"
)
AUTHORIZATION = QWEN / (
    "governance/qwencloud-one-pair-human-pi-go-"
    "93ac04e043027e333e283a966accb995ffb6b9255eb810b821495d2ae4f343cf.json"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            child for item in value.values() for child in _all_keys(item)
        }
    if isinstance(value, list):
        return {child for item in value for child in _all_keys(item)}
    return set()


def test_qwen_public_projection_is_redacted_append_only_and_unranked(
    tmp_path: Path,
) -> None:
    sources = (
        PREDECESSOR,
        PREDECESSOR_JOURNAL,
        SUCCESSOR,
        SUCCESSOR_JOURNAL,
        RECOVERY,
        LEDGER,
        ROUTE,
        PREFLIGHT,
        GO_TEMPLATE,
        AUTHORIZATION,
    )
    before = {path: _digest(path) for path in sources}
    output = build_qwen_projection(
        predecessor_source_path=PREDECESSOR,
        predecessor_journal_path=PREDECESSOR_JOURNAL,
        successor_source_path=SUCCESSOR,
        successor_journal_path=SUCCESSOR_JOURNAL,
        recovery_path=RECOVERY,
        ledger_path=LEDGER,
        route_path=ROUTE,
        preflight_path=PREFLIGHT,
        go_template_path=GO_TEMPLATE,
        authorization_path=AUTHORIZATION,
        output_dir=tmp_path,
    )
    assert before == {path: _digest(path) for path in sources}
    assert output.name == (
        "qwencloud-exploratory-operational-projection-"
        "b2f7790b3eb18d1df083397ce02b5296c549e5ed3ddb3d3f32ea776db3ddca04.json"
    )

    projection = json.loads(output.read_text(encoding="utf-8"))
    assert projection["artifact_sha256"] == (
        "b2f7790b3eb18d1df083397ce02b5296c549e5ed3ddb3d3f32ea776db3ddca04"
    )
    assert projection["model_identity"]["display_name"] == "Qwen 3.8 Max"
    assert projection["model_identity"]["frozen_release"] is False
    assert projection["predecessor_reliability_run"]["provider_rejections"] == 1
    assert projection["successor_operational_run"] == {
        "completed_off_on_pairs": 1,
        "delivered_response_arms": 2,
        "epicure_tool_names": ["list_targets", "flavour_correlations"],
        "finish_reasons": ["stop", "stop"],
        "provider_rejections": 0,
        "provider_requests": 6,
        "provider_responses": 6,
        "provider_retries": 0,
        "real_epicure_calls": 2,
        "retained_budget_ceiling_usd": "2",
        "returned_stage_usage": {
            "completion_tokens": 14574,
            "normalized_result_reasoning_token_fields": 0,
            "prompt_tokens": 12413,
            "provider_generation_responses": 6,
            "reasoning_tokens": 9332,
        },
        "status": "complete_unpriced_budget_ceiling",
        "successful_real_epicure_calls": 2,
        "synthetic_arms": 0,
    }
    assert projection["combined_reliability_accounting"] == {
        "cumulative_retained_budget_ceiling_usd": "4",
        "eligible_for_numeric_cost_plot": False,
        "provider_charge_available": False,
        "provider_cost_reconciled": False,
        "provider_rejections": 1,
        "provider_requests": 11,
        "provider_responses": 10,
        "recorded_zero_cost_means": "unknown_not_free",
    }
    boundary = projection["claim_boundary"]
    assert boundary["quality_judgments"] == 0
    assert boundary["leaderboard_comparisons_authorized"] == 0
    assert boundary["included_in_current_uplift_pool"] is False
    assert boundary["included_in_current_model_arena_pool"] is False
    assert boundary["included_in_any_quality_fit"] is False
    assert boundary["changes_186_pair_uplift_count"] is False
    assert boundary["changes_915_comparison_arena_count"] is False

    assert len(projection["source_commitments"]) == 10
    assert sum(
        item["distributed_in_arxiv_source"]
        for item in projection["source_commitments"]
    ) == 1
    assert {
        item["semantic_sha256"]
        for item in projection["source_commitments"]
        if "semantic_sha256" in item
    } >= {
        EXPECTED["predecessor_source_semantic"],
        EXPECTED["successor_source_semantic"],
        EXPECTED["recovery_semantic"],
    }

    forbidden_keys = {
        "answer_markdown",
        "arguments",
        "human_pi",
        "intermediate_outputs",
        "output_json",
        "prompt",
        "result",
        "tool_trace",
    }
    assert not (_all_keys(projection) & forbidden_keys)
    serialized = json.dumps(projection, sort_keys=True)
    assert "/home/" not in serialized
    assert "remy-simpc4" not in serialized
    assert "sk-" not in serialized
