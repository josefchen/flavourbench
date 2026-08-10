from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from flavourbench.direct_kimi_pair import _rate_card_result_is_accounted
from flavourbench.execution_policy import MATCHED_TOOL_ACCESS_PROTOCOL_V1, ExecutionPolicy
from flavourbench.frontier_contract_runner import select_candidates
from flavourbench.frontier_manifest import verify_manifest_content_address
from flavourbench.routed_development_manifest import build_manifest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / (
    "artifacts/season1/current-quality-run/manifest-v20-runner-contract-fix/"
    "flavourbench-openrouter-unranked-"
    "8de80c774eed0a8bf2c4bee7e19775eb949b338ef1eb91d8ec3d085b9685e945.json"
)
TASKS = ROOT / (
    "artifacts/season1/task-validity/development-v2/development-task-validity-v2-"
    "5ffd81a44267291413bc8a638d15391ec2b51decdda270550f81ca17ec587846.json"
)
KIMI_CATALOG = ROOT / (
    "artifacts/frontier-refresh/2026-07-28/kimi-code-direct/k3-v2/catalog/"
    "kimi-catalog-ca3fcfdba2612735b6365afdefa6af90f373ea4a30c6d939a23dd01e75b3b8ba.json"
)
KIMI_SMOKE = ROOT / (
    "artifacts/frontier-refresh/2026-07-28/kimi-code-direct/k3-v2/compatibility/"
    "kimi-2f5052c9fd15-63496d48c2f81dd26efd02d7ba56a84b4946d3666c7ab1092de8fa3e47208313.json"
)
BEDROCK_CATALOG = ROOT / (
    "artifacts/bedrock/catalog-2026-08-02-eu/"
    "bedrock-catalog-09b42d5a659acca11a6697013f39ad80f2eaeead789614b817773febd874b23a.json"
)
BEDROCK_FABLE = ROOT / (
    "artifacts/frontier-refresh/2026-08-02/bedrock-fable5-global-recheck-v1/"
    "frontier-refresh-contract-summary-"
    "dd581ced5b0610d9d5cf8538104cea442d27431ce2fbb5e039bc41c5bb7820bb.json"
)
BEDROCK_CLAUDE_EU = ROOT / (
    "artifacts/frontier-refresh/2026-08-02/bedrock-claude5-eu-recheck-v1/"
    "frontier-refresh-contract-summary-"
    "392ad8f1abfe512118629d133726277a5ad6d0103be6fdc3893c9c45ef3f2851.json"
)
BEDROCK_US_WEST_CATALOG = ROOT / (
    "artifacts/frontier-refresh/2026-08-03/bedrock-us-west-2-global/"
    "bedrock-catalog-6bf8a1930f1d63c0e99e36bb243a9363fdd1c83e93b14f7efbddf998aa5e2763.json"
)
BEDROCK_US_WEST_RECEIPT = ROOT / (
    "artifacts/frontier-refresh/2026-08-03/bedrock-claude5-global-us-west-2-smokes/"
    "frontier-refresh-contract-summary-"
    "ebfbcc5983f6442aa91a6dd571cb231e3590ba4f71a3a05913c968a7dbae3c26.json"
)


def _policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        max_output_tokens=8192,
        max_intermediate_tokens=2048,
        max_tool_rounds=2,
        max_tool_calls_per_round=16,
        max_tool_calls_total=32,
        max_cumulative_tool_result_bytes=98_304,
        max_provider_attempts=2,
        decoding_temperature=1.0,
        decoding_top_p=0.95,
        decoding_seed=20260715,
        final_response_mode="plain_text",
        matched_planning=False,
        evidence_protocol=MATCHED_TOOL_ACCESS_PROTOCOL_V1,
        intermediate_reasoning_effort="low",
        final_reasoning_effort="low",
        tool_catalog_bytes_bound=24_000,
    )


def test_routed_manifest_freezes_direct_kimi_and_evidenced_fallbacks() -> None:
    manifest = build_manifest(
        base_manifest_path=BASE,
        expected_base_manifest_sha256=(
            "8de80c774eed0a8bf2c4bee7e19775eb949b338ef1eb91d8ec3d085b9685e945"
        ),
        task_validity_path=TASKS,
        kimi_catalog_path=KIMI_CATALOG,
        kimi_compatibility_path=KIMI_SMOKE,
        bedrock_catalog_path=BEDROCK_CATALOG,
        bedrock_fable_receipt_path=BEDROCK_FABLE,
        bedrock_claude_eu_receipt_path=BEDROCK_CLAUDE_EU,
        cap_usd=Decimal("100"),
        execution_policy=_policy(),
    )

    assert verify_manifest_content_address(manifest)
    candidates = select_candidates(manifest)
    assert len(candidates) == 14
    direct = [
        candidate for candidate in candidates if candidate.execution_backend == "kimi_direct"
    ]
    fallback = [
        candidate for candidate in candidates if candidate.execution_backend == "openrouter"
    ]
    assert [candidate.model_id for candidate in direct] == ["moonshotai/kimi-k3"]
    assert direct[0].canonical_model_slug == "k3"
    assert direct[0].provider_tag == "kimi-code-direct"
    assert direct[0].cost_accounting_policy == "provider_usage_times_frozen_rate_card"
    assert len(fallback) == 13
    assert all(candidate.route_selection["fallback_used"] is True for candidate in fallback)
    assert all(
        candidate.route_selection["generation_time_automatic_fallback"] is False
        for candidate in candidates
    )
    claude_reasons = {
        candidate.model_id: candidate.route_selection["selection_reason"]
        for candidate in fallback
        if candidate.model_id.startswith("anthropic/")
    }
    assert set(claude_reasons.values()) == {
        "bedrock_account_access_denied_before_generation"
    }
    deepseek = next(
        candidate
        for candidate in fallback
        if candidate.model_id == "deepseek/deepseek-v4-pro"
    )
    assert deepseek.route_selection["selection_reason"] == "bedrock_exact_model_absent"
    assert Decimal(manifest["budget"]["bounded_forecast_usd"]) <= Decimal("85")


def test_direct_kimi_rate_card_accounting_is_strictly_typed() -> None:
    result = {
        "cost_reconciled": False,
        "cost_accounting_basis": "frozen_rate_card_times_kimi_returned_usage",
        "billing_reconciliation_status": "provider_charge_unavailable",
        "generation_ids": ["gen-1"],
        "cost_micros": 19,
        "generation_metadata": [
            {
                "generation_id": "gen-1",
                "cost_micros": 19,
                "reconciled": False,
                "accounting_basis": "frozen_rate_card_times_kimi_returned_usage",
                "billing_reconciliation_status": "provider_charge_unavailable",
            }
        ],
    }
    assert _rate_card_result_is_accounted(result)
    result["generation_metadata"][0]["reconciled"] = True
    assert not _rate_card_result_is_accounted(result)


def test_routed_manifest_binds_us_west_catalog_to_unified_denial_receipt() -> None:
    manifest = build_manifest(
        base_manifest_path=BASE,
        expected_base_manifest_sha256=(
            "8de80c774eed0a8bf2c4bee7e19775eb949b338ef1eb91d8ec3d085b9685e945"
        ),
        task_validity_path=TASKS,
        kimi_catalog_path=KIMI_CATALOG,
        kimi_compatibility_path=KIMI_SMOKE,
        bedrock_catalog_path=BEDROCK_US_WEST_CATALOG,
        bedrock_fable_receipt_path=None,
        bedrock_claude_eu_receipt_path=None,
        bedrock_access_receipt_path=BEDROCK_US_WEST_RECEIPT,
        cap_usd=Decimal("100"),
        execution_policy=_policy(),
    )

    assert verify_manifest_content_address(manifest)
    assert manifest["source"]["bedrock_catalog_region"] == "us-west-2"
    assert manifest["routing_policy"]["bedrock_catalog_region"] == "us-west-2"
    assert manifest["source"]["bedrock_access_denial_receipts"] == [
        {
            "artifact_sha256": (
                "ebfbcc5983f6442aa91a6dd571cb231e3590ba4f71a3a05913c968a7dbae3c26"
            ),
            "requested_endpoint_ids": [
                "global.anthropic.claude-fable-5",
                "global.anthropic.claude-opus-5",
                "global.anthropic.claude-sonnet-5",
            ],
            "failed_pre_generation": 3,
            "real_provider_calls": 0,
            "real_epicure_calls": 0,
        }
    ]


def test_routed_manifest_freezes_targeted_floor_replenishment_before_generation() -> None:
    targets = [
        "x-ai/grok-4.5",
        "minimax/minimax-m3",
        "mistralai/mistral-medium-3-5",
    ]
    manifest = build_manifest(
        base_manifest_path=BASE,
        expected_base_manifest_sha256=(
            "8de80c774eed0a8bf2c4bee7e19775eb949b338ef1eb91d8ec3d085b9685e945"
        ),
        task_validity_path=TASKS,
        kimi_catalog_path=KIMI_CATALOG,
        kimi_compatibility_path=KIMI_SMOKE,
        bedrock_catalog_path=BEDROCK_US_WEST_CATALOG,
        bedrock_fable_receipt_path=None,
        bedrock_claude_eu_receipt_path=None,
        bedrock_access_receipt_path=BEDROCK_US_WEST_RECEIPT,
        cap_usd=Decimal("100"),
        execution_policy=_policy(),
        target_model_ids=targets,
    )

    assert verify_manifest_content_address(manifest)
    assert [entry["model"]["id"] for entry in manifest["models"]] == targets
    assert manifest["selection"]["model_count"] == 3
    assert manifest["selection"]["route_counts"] == {
        "kimi_direct": 0,
        "bedrock": 0,
        "openrouter_fallback": 3,
    }
    assert manifest["selection"]["targeting"] == {
        "method": "operator-frozen operational completion-floor replenishment",
        "selected_model_ids": targets,
        "quality_outcomes_used": False,
    }
    assert manifest["run_design"]["expected_pairs"] == 12
    assert manifest["run_design"]["expected_arms"] == 24
