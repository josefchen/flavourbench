from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from flavourbench.bedrock_b2_manifest import (
    B2ForecastPolicy,
    BedrockB2BudgetExceeded,
    BedrockB2ManifestError,
    build_b2_manifest,
    select_tasks,
    verify_b2_manifest_content_address,
    write_b2_manifest,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write(path: Path, value: object) -> Path:
    path.write_bytes(_canonical(value) + b"\n")
    return path


def _endpoint(path: Path, model_id: str = "vendor/real-model-v1", price: str = "1") -> Path:
    endpoint = {
        "schema_version": "flavourbench-bedrock-endpoint-manifest-v2",
        "official": False,
        "rank_eligible": False,
        "contracts": [
            {
                "canonical_model_id": model_id,
                "bedrock_target_id": "global.vendor.real-model-v1:0",
                "expected_foundation_model_ids": ["vendor.real-model-v1:0"],
                "supports_converse": True,
                "supports_tool_use": True,
                "supports_structured_output": True,
                "price": {
                    "input_per_million_usd": price,
                    "output_per_million_usd": price,
                    "price_sha256": "a" * 64,
                },
            }
        ],
    }
    wrapper = {
        "schema_version": "flavourbench-bedrock-smoke-manifest-v1",
        "official": False,
        "rank_eligible": False,
        "endpoint_manifest": endpoint,
    }
    wrapper["manifest_sha256"] = hashlib.sha256(_canonical(wrapper)).hexdigest()
    return _write(path, wrapper)


@pytest.fixture
def local_contracts(tmp_path: Path) -> tuple[Path, Path, Path]:
    endpoint = _endpoint(tmp_path / "endpoint.json")
    epicure = _write(tmp_path / "epicure.json", {"release_id": "unmatched-real-runtime"})
    tools = _write(tmp_path / "tools.json", [{"name": "find_pairings"}])
    return endpoint, epicure, tools


def _build(local_contracts: tuple[Path, Path, Path], **overrides: object) -> dict:
    endpoint, epicure, tools = local_contracts
    kwargs = {
        "endpoint_contract_paths": [endpoint],
        "epicure_contract_path": epicure,
        "tool_contract_path": tools,
        "tasks": select_tasks(
            [
                "sub-001",
                "sub-002",
                "comp-001",
                "comp-002",
                "cook-001",
                "cook-002",
                "evid-001",
                "evid-002",
            ]
        ),
        "frozen_at": "2026-07-16T00:00:00Z",
    }
    kwargs.update(overrides)
    return build_b2_manifest(**kwargs)  # type: ignore[arg-type]


def _task_ids() -> list[str]:
    return [
        "sub-001",
        "sub-002",
        "comp-001",
        "comp-002",
        "cook-001",
        "cook-002",
        "evid-001",
        "evid-002",
    ]


def test_freezes_exact_common_task_cartesian_workload(local_contracts) -> None:
    manifest = _build(local_contracts)

    assert manifest["official"] is False
    assert manifest["rank_eligible"] is False
    assert manifest["provider_calls_made"] == 0
    assert manifest["counts"] == {"models": 1, "tasks": 8, "arms": 16, "arms_per_model": 16}
    assert {task["family"] for task in manifest["tasks"]} == {
        "substitution",
        "composition",
        "cookability",
        "evidence",
    }
    for task_id in {task["task_id"] for task in manifest["tasks"]}:
        task_arms = [arm for arm in manifest["arms"] if arm["task_id"] == task_id]
        assert {arm["condition"] for arm in task_arms} == {"epicure_off", "epicure_on"}
    endpoint_ref = manifest["models"][0]["endpoint_contract_reference"]
    assert len(endpoint_ref["file_sha256"]) == 64
    assert len(endpoint_ref["endpoint_manifest_sha256"]) == 64
    assert len(endpoint_ref["endpoint_contract_sha256"]) == 64
    assert verify_b2_manifest_content_address(manifest)


def test_content_address_is_immutable(local_contracts, tmp_path: Path) -> None:
    manifest = _build(local_contracts)
    path = write_b2_manifest(manifest, tmp_path / "out")
    assert path.name.endswith(f"{manifest['content_address']['digest']}.json")
    assert write_b2_manifest(manifest, tmp_path / "out") == path

    changed = copy.deepcopy(manifest)
    changed["budget"]["cap_usd"] = "99"
    assert not verify_b2_manifest_content_address(changed)
    with pytest.raises(BedrockB2ManifestError, match="invalid content address"):
        write_b2_manifest(changed, tmp_path / "out")


@pytest.mark.parametrize(
    "task_ids",
    [
        ["sub-001"] * 2 + ["comp-001", "comp-002", "cook-001", "cook-002", "evid-001", "evid-002"],
        [
            "sub-001",
            "sub-002",
            "comp-001",
            "comp-002",
            "cook-001",
            "cook-002",
            "cook-003",
            "evid-001",
        ],
    ],
)
def test_rejects_duplicate_or_incomplete_task_coverage(task_ids) -> None:
    with pytest.raises(BedrockB2ManifestError):
        select_tasks(task_ids)


def test_rejects_mock_and_duplicate_models(local_contracts, tmp_path: Path) -> None:
    _, epicure, tools = local_contracts
    mock = _endpoint(tmp_path / "mock.json", "vendor/mock-model")
    with pytest.raises(BedrockB2ManifestError, match="mock/fixture"):
        build_b2_manifest(
            endpoint_contract_paths=[mock],
            epicure_contract_path=epicure,
            tool_contract_path=tools,
            tasks=select_tasks(_task_ids()),
        )

    first = _endpoint(tmp_path / "first.json")
    second = _endpoint(tmp_path / "second.json")
    with pytest.raises(BedrockB2ManifestError, match="duplicate canonical model"):
        build_b2_manifest(
            endpoint_contract_paths=[first, second],
            epicure_contract_path=epicure,
            tool_contract_path=tools,
            tasks=select_tasks(_task_ids()),
        )


def test_rejects_missing_contract_and_whole_block_over_cap(local_contracts, tmp_path: Path) -> None:
    endpoint, epicure, tools = local_contracts
    with pytest.raises(BedrockB2ManifestError, match="missing Epicure lineage contract"):
        _build((endpoint, tmp_path / "missing.json", tools))

    expensive = _endpoint(tmp_path / "expensive.json", price="1000")
    with pytest.raises(BedrockB2BudgetExceeded, match="whole-block forecast"):
        _build((expensive, epicure, tools), cap_usd="100")
    with pytest.raises(BedrockB2ManifestError, match="cannot exceed USD 100"):
        _build(local_contracts, cap_usd="101")


def test_forecast_is_whole_block_not_per_arm(local_contracts) -> None:
    policy = B2ForecastPolicy(
        max_off_generations=1,
        max_on_generations=2,
        max_input_tokens_per_generation=100,
        max_output_tokens_per_generation=100,
    )
    manifest = _build(local_contracts, forecast_policy=policy)
    assert Decimal(manifest["budget"]["whole_block_worst_case_usd"]) == Decimal("0.0048")
