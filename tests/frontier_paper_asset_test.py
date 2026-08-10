from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.frontier_evidence import SCHEMA_VERSION
from flavourbench.frontier_paper_asset import (
    FrontierPaperAssetError,
    load_bundle,
    render_table,
)
from flavourbench.real_task_bank import sha256_json


def bundle_document() -> dict[str, object]:
    models = []
    specifications = [
        (
            "anthropic/claude-opus-5",
            "openrouter",
            "generation_accounting_model_and_provider",
            "0.04",
        ),
        (
            "anthropic/claude-sonnet-5",
            "openrouter",
            "generation_accounting_model_and_provider",
            "0.02",
        ),
        (
            "command-a-plus-05-2026",
            "cohere_direct",
            "authenticated_catalog_and_exact_request_only",
            None,
        ),
        (
            "command-a-reasoning-08-2025",
            "cohere_direct",
            "authenticated_catalog_and_exact_request_only",
            None,
        ),
        ("k3", "kimi_code_direct", "response_returned_exact_catalog_model", None),
    ]
    for model, provider, identity, cost in specifications:
        models.append(
            {
                "requested_model_id": model,
                "provider_lane": provider,
                "verification": identity,
                "real_provider_calls": 2,
                "real_epicure_calls": 1,
                "cost_usd": cost,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "all_selected_contract_smokes_passed",
        "models": models,
        "official": False,
        "rank_eligible": False,
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def test_table_has_one_row_per_model_and_marks_unknown_cost(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle_document()))

    rendered = render_table(load_bundle(path))

    assert rendered.count(" \\\\") == 6
    assert "Claude Opus 5 & OpenRouter, Anthropic" in rendered
    assert "Command A Reasoning & Cohere direct" in rendered
    assert rendered.count("not returned") == 3


def test_bundle_content_address_is_required(tmp_path: Path) -> None:
    document = bundle_document()
    document["status"] = "failed"
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(document))

    with pytest.raises(FrontierPaperAssetError, match="content address"):
        load_bundle(path)
