from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.season0_openrouter_compatibility import (
    OpenRouterCompatibilityError,
    OpenRouterTarget,
    _finish_contract,
    _structured_response_format,
    load_targets,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_load_targets_binds_exact_model_provider_and_capabilities(tmp_path: Path) -> None:
    roster = {
        "slots": [
            {
                "canonical_name": "Model",
                "provider": "openrouter",
                "endpoint_id": "lab/model",
                "canonical_slug": "lab/model-20260701",
                "provider_slug": "lab/flex",
            }
        ]
    }
    manifest = {
        "content_address": {"sha256": "a" * 64},
        "models": [
            {
                "model": {
                    "id": "lab/model",
                    "canonical_slug": "lab/model-20260701",
                    "name": "Model",
                },
                "endpoint_document_sha256": "b" * 64,
                "endpoint": {
                    "tag": "lab/flex",
                    "provider_name": "Lab",
                    "pricing": {"prompt": "0.1"},
                    "supported_parameters": [
                        "max_tokens",
                        "response_format",
                        "structured_outputs",
                        "tools",
                        "tool_choice",
                    ],
                },
            }
        ],
    }
    targets = load_targets(
        _write(tmp_path / "roster.json", roster),
        _write(tmp_path / "manifest.json", manifest),
    )
    assert len(targets) == 1
    assert targets[0].canonical_slug == "lab/model-20260701"
    assert targets[0].provider_slug == "lab/flex"


def test_load_targets_refuses_provider_drift(tmp_path: Path) -> None:
    roster = {
        "slots": [
            {
                "provider": "openrouter",
                "endpoint_id": "lab/model",
                "canonical_slug": "lab/model-20260701",
                "provider_slug": "wrong",
            }
        ]
    }
    manifest = {
        "content_address": {"sha256": "a" * 64},
        "models": [
            {
                "model": {"id": "lab/model", "canonical_slug": "lab/model-20260701"},
                "endpoint": {"tag": "right", "supported_parameters": []},
            }
        ],
    }
    with pytest.raises(OpenRouterCompatibilityError, match="provider tag drift"):
        load_targets(
            _write(tmp_path / "roster.json", roster),
            _write(tmp_path / "manifest.json", manifest),
        )


def test_finish_contract_rejects_truncation() -> None:
    assert _finish_contract("stop") == ("smoke_passed", None, None)
    status, error_type, error = _finish_contract("length")
    assert status == "failed_incomplete_finish"
    assert error_type == "IncompleteFinish"
    assert error == "final generation ended with length"


def test_strict_structured_output_contract_is_provider_enforced() -> None:
    target = OpenRouterTarget(
        display_name="Model",
        model_id="lab/model",
        canonical_slug="lab/model-20260731",
        provider_slug="lab/fp8",
        provider_name="Lab",
        supported_parameters=(
            "max_tokens",
            "response_format",
            "structured_outputs",
            "tools",
            "tool_choice",
        ),
        endpoint_document_sha256="a" * 64,
        pricing={},
        source_manifest_sha256="b" * 64,
    )

    response_format = _structured_response_format(target)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_strict_structured_output_contract_refuses_capability_drift() -> None:
    target = OpenRouterTarget(
        display_name="Model",
        model_id="lab/model",
        canonical_slug="lab/model-20260731",
        provider_slug="lab/fp8",
        provider_name="Lab",
        supported_parameters=("max_tokens", "tools", "tool_choice"),
        endpoint_document_sha256="a" * 64,
        pricing={},
        source_manifest_sha256="b" * 64,
    )

    with pytest.raises(OpenRouterCompatibilityError, match="strict structured output"):
        _structured_response_format(target)
