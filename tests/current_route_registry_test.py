from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flavourbench.current_frontier_paper_assets import (
    CurrentFrontierPaperAssetError,
    _load_registry,
    _render_table,
)
from flavourbench.current_route_registry import REGISTRY_ORDER
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT
    / "artifacts/frontier-refresh/2026-08-01/current-route-registry/aggregate"
    / "current-route-registry-b300d460ec3d93dbfdaea64e0809abf858fa9efb570d0bddeac28566b6cdf010.json"
)


def test_current_route_registry_is_unranked_and_contains_requested_frontier_models() -> None:
    registry = _load_registry(REGISTRY)
    counts = registry["counts"]
    assert counts == {
        "models": 16,
        "contract_passed": 15,
        "contract_failed": 1,
        "real_provider_generations_in_passed_receipts": 30,
        "real_epicure_calls_in_passed_receipts": 15,
        "quality_observations": 0,
        "rankable_comparisons": 0,
    }
    by_name = {row["display_name"]: row for row in registry["models"]}
    for name in (
        "Anthropic Claude Sonnet 5",
        "Anthropic Claude Opus 5",
        "MoonshotAI Kimi K3",
        "Z.AI GLM 5.2",
        "DeepSeek V4 Pro",
        "DeepSeek V4 Flash 0731",
    ):
        assert by_name[name]["contract_status"] == "passed_unranked"
        assert by_name[name]["quality_observations"] == 0
    assert by_name["Qwen 3.7 Max"]["contract_status"] == "failed_pre_generation"


def test_current_route_paper_assets_refuse_quality_data(tmp_path: Path) -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    mutated["models"][0]["quality_observations"] = 1
    payload = {key: value for key, value in mutated.items() if key != "artifact_sha256"}
    mutated["artifact_sha256"] = sha256_json(payload)
    path = tmp_path / "mutated-registry.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(CurrentFrontierPaperAssetError, match="claim boundary"):
        _load_registry(path)


def test_current_route_order_is_unique_and_table_labels_quality_sample_size_zero() -> None:
    model_ids = [model_id for model_id, _source, _display_name in REGISTRY_ORDER]
    display_names = [display_name for _model_id, _source, display_name in REGISTRY_ORDER]
    assert len(model_ids) == len(set(model_ids)) == 16
    assert len(display_names) == len(set(display_names)) == 16

    table = _render_table(_load_registry(REGISTRY))
    assert "Claude Sonnet 5" in table
    assert "Kimi K3" in table
    assert "Anthropic Claude Sonnet 5" not in table
    assert "Qwen 3.7 Max" in table and "failed" in table
    assert table.count(" & 0 \\") == 16
