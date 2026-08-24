from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from flavourbench.current_development_manifest import build_manifest
from flavourbench.execution_policy import ExecutionPolicy
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / (
    "artifacts/frontier-refresh/2026-08-01/current-route-registry/aggregate/"
    "current-route-registry-"
    "b300d460ec3d93dbfdaea64e0809abf858fa9efb570d0bddeac28566b6cdf010.json"
)
TASK_VALIDITY = ROOT / (
    "artifacts/season1/task-validity/development-v2/"
    "development-task-validity-v2-"
    "5ffd81a44267291413bc8a638d15391ec2b51decdda270550f81ca17ec587846.json"
)


def test_current_manifest_freezes_all_passed_openrouter_routes_before_quality() -> None:
    manifest = build_manifest(
        registry_path=REGISTRY,
        route_catalog_root=ROOT / "artifacts",
        task_validity_path=TASK_VALIDITY,
        repository_root=ROOT,
        tasks_per_family=1,
        assignments_per_model=4,
        cap_usd=Decimal("100"),
        execution_policy=ExecutionPolicy(
            max_output_tokens=8192,
            max_intermediate_tokens=2048,
            max_tool_rounds=2,
            max_tool_calls_per_round=16,
            max_tool_calls_total=16,
            max_cumulative_tool_result_bytes=65_536,
            final_response_mode="plain_text",
            matched_planning=True,
            evidence_protocol="matched_evidence_v2",
            intermediate_reasoning_effort="low",
            final_reasoning_effort="low",
        ),
    )

    assert verify_manifest_content_address(manifest)
    assert len(manifest["models"]) == 14
    assert manifest["run_design"]["expected_pairs"] == 56
    assert manifest["run_design"]["expected_arms"] == 112
    assert Decimal(manifest["budget"]["bounded_forecast_usd"]) <= Decimal("85")
    assert manifest["generation_calls_made"] == 0
    assert manifest["official_results_authorised"] is False
    assert manifest["selection"]["quality_observations_used"] == 0
    assert manifest["run_design"]["generation_protocol"]["final_response_mode"] == ("plain_text")
    assert manifest["run_design"]["execution_policy"]["schema_version"] == (
        "flavourbench-real-execution-policy-v7"
    )
    protocol = manifest["run_design"]["generation_protocol"]
    assert protocol["evidence_protocol"] == "matched_evidence_v2"
    assert protocol["required_tool_contract_protocol"] == "direct_tool_first_v1"
    assert protocol["required_tool_contract_max_intermediate_tokens"] == 2048
    assert protocol["required_tool_contract"]["rank_eligible"] is False
    assert protocol["required_tool_contract_sha256"] == (
        protocol["required_tool_contract"]["content_address"]["digest"]
    )
    assert protocol["intermediate_reasoning_effort"] == "low"
    assert protocol["final_reasoning_effort"] == "low"
    model_ids = {entry["model"]["id"] for entry in manifest["models"]}
    assert {
        "openai/gpt-5.6-sol-pro",
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.6-flash",
        "x-ai/grok-4.5",
        "moonshotai/kimi-k3",
        "z-ai/glm-5.2",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash-0731",
        "minimax/minimax-m3",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "mistralai/mistral-medium-3-5",
    } == model_ids
    excluded = {
        item["model_id"]: item["reason"] for item in manifest["selection"]["excluded_lanes"]
    }
    assert excluded["command-a-plus-05-2026"] == (
        "direct_provider_lane_requires_separate_cost_governor"
    )
    assert excluded["qwen/qwen3.7-max"] == "exact_route_contract_failed"
