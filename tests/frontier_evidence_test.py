from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.frontier_evidence import FrontierEvidenceError, build_bundle
from flavourbench.real_task_bank import sha256_json


def write_artifact(
    path: Path,
    *,
    model: str,
    provider: str,
    index: int,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": (
            "flavourbench-frontier-refresh-openrouter-contract-smoke-v1"
            if provider == "openrouter"
            else f"flavourbench-{provider}-epicure-contract-smoke-v1"
        ),
        "status": "smoke_passed",
        "provider": None if provider == "openrouter" else provider,
        "requested_model_id": model,
        "prompt_sha256": "a" * 64,
        "response_schema_sha256": "b" * 64,
        "epicure_tool_catalog_sha256": "c" * 64,
        "generation_ids": [f"generation-{index}-a", f"generation-{index}-b"],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "wall_clock_latency_ms": 100,
        "real_provider_calls": 2,
        "real_epicure_calls": 1,
        "complete_epicure_trace": [{"is_error": False}],
        "official": False,
        "rank_eligible": False,
    }
    if provider == "openrouter":
        canonical = f"{model}-dated"
        payload.update(
            {
                "canonical_slug": canonical,
                "returned_provider_name": "Direct Provider",
                "generation_accounting": [
                    {"model": canonical, "provider_name": "Direct Provider"},
                    {"model": canonical, "provider_name": "Direct Provider"},
                ],
                "cost_usd": "0.01",
            }
        )
    elif provider == "kimi_code_direct":
        payload.update(
            {
                "returned_model_ids": [model, model],
                "cost_status": "managed_service_returned_no_per_generation_cost",
            }
        )
    else:
        payload.update(
            {
                "catalog_entry": {"name": model},
                "cost_status": "no_per_generation_cost_returned_by_provider",
            }
        )
    document = {**payload, "artifact_sha256": sha256_json(payload)}
    path.write_text(json.dumps(document))
    return path


def five_artifacts(tmp_path: Path) -> list[Path]:
    specifications = [
        ("anthropic/claude-sonnet-5", "openrouter"),
        ("anthropic/claude-opus-5", "openrouter"),
        ("command-a-plus-05-2026", "cohere_direct"),
        ("command-a-reasoning-08-2025", "cohere_direct"),
        ("k3", "kimi_code_direct"),
    ]
    return [
        write_artifact(
            tmp_path / f"artifact-{index}.json",
            model=model,
            provider=provider,
            index=index,
        )
        for index, (model, provider) in enumerate(specifications)
    ]


def test_bundle_requires_real_contract_and_separates_cost_coverage(tmp_path: Path) -> None:
    payload = build_bundle(five_artifacts(tmp_path))

    assert payload["status"] == "all_selected_contract_smokes_passed"
    assert payload["counts"] == {
        "models": 5,
        "provider_lanes": 3,
        "provider_generations": 10,
        "epicure_calls": 5,
        "response_returned_model_verified": 3,
        "known_cost_models": 2,
    }
    assert payload["known_reconciled_cost_usd"] == "0.02"
    assert payload["interpretation"]["leaderboard_observation"] is False


def test_bundle_rejects_mutated_content_address(tmp_path: Path) -> None:
    paths = five_artifacts(tmp_path)
    document = json.loads(paths[0].read_text())
    document["status"] = "failed"
    paths[0].write_text(json.dumps(document))

    with pytest.raises(FrontierEvidenceError, match="content address"):
        build_bundle(paths)


def test_bundle_rejects_model_identity_substitution(tmp_path: Path) -> None:
    paths = five_artifacts(tmp_path)
    document = json.loads(paths[-1].read_text())
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    payload["returned_model_ids"] = ["substitute", "substitute"]
    paths[-1].write_text(json.dumps({**payload, "artifact_sha256": sha256_json(payload)}))

    with pytest.raises(FrontierEvidenceError, match="returned-model mismatch"):
        build_bundle(paths)
