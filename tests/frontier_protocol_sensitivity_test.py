from copy import deepcopy
from pathlib import Path

import pytest

from flavourbench.frontier_multirun_assets import RunInput, verify_runs
from flavourbench.frontier_protocol_sensitivity import (
    FrontierProtocolSensitivityError,
    compare_strata,
    write_assets,
)
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "artifacts/season1/current-quality-run"

STRICT_RUNS = [
    RunInput(
        summary=PILOT_ROOT
        / "pilot-v27-eight-pairs/summaries"
        / (
            "real-exploratory-summary-"
            "d0876f6e7b70d9803468b766b4df91f983fcf684c463766bbe9be1b35cda7018.json"
        ),
        sources=PILOT_ROOT / "pilot-v27-eight-pairs/source",
        responses=PILOT_ROOT / "pilot-v27-eight-pairs/responses",
    ),
    RunInput(
        summary=PILOT_ROOT
        / "pilot-v28-replenishment/summaries"
        / (
            "real-exploratory-summary-"
            "a3e6674274a270d51424e86d1726b3a52abade109ec27590f5adc4bde8fa5a05.json"
        ),
        sources=PILOT_ROOT / "pilot-v28-replenishment/source",
        responses=PILOT_ROOT / "pilot-v28-replenishment/responses",
    ),
]

HIGH_RESOURCE_RUNS = [
    RunInput(
        summary=PILOT_ROOT
        / "pilot-v29-high-resource/summaries"
        / (
            "real-exploratory-summary-"
            "9710525c84feed31ed7ddfa6ae172cff915b36b8fdab8f7dedbeba2bdb0c8084.json"
        ),
        sources=PILOT_ROOT / "pilot-v29-high-resource/source",
        responses=PILOT_ROOT / "pilot-v29-high-resource/responses",
    ),
    RunInput(
        summary=PILOT_ROOT
        / "pilot-v30-floor-replenishment/summaries"
        / (
            "real-exploratory-summary-"
            "6fe0e3ff11572069900bb1a06b24bc7377ea6440c92d24e4567f5138db4553b6.json"
        ),
        sources=PILOT_ROOT / "pilot-v30-floor-replenishment/source",
        responses=PILOT_ROOT / "pilot-v30-floor-replenishment/responses",
    ),
    RunInput(
        summary=PILOT_ROOT
        / "pilot-v32-floor-replenishment/summaries"
        / (
            "real-exploratory-summary-"
            "26b0392db5c4e1ae3a4e8f7ce53b4981f0b9bf0ab9e9ea27d71fd5b9a17593fe.json"
        ),
        sources=PILOT_ROOT / "pilot-v32-floor-replenishment/source",
        responses=PILOT_ROOT / "pilot-v32-floor-replenishment/responses",
    ),
    RunInput(
        summary=PILOT_ROOT
        / "pilot-v33-mistral-floor/summaries"
        / (
            "real-exploratory-summary-"
            "93e134e2bacd766afb5bb18cd558d6d352991b5acb05eb162464a9ff47b3260e.json"
        ),
        sources=PILOT_ROOT / "pilot-v33-mistral-floor/source",
        responses=PILOT_ROOT / "pilot-v33-mistral-floor/responses",
    ),
]

COHERE_RUNS = [
    RunInput(
        summary=PILOT_ROOT
        / version
        / "summaries"
        / f"real-exploratory-summary-{digest}.json",
        sources=PILOT_ROOT / version / "source",
        responses=PILOT_ROOT / version / "responses",
    )
    for version, digest in (
        (
            "pilot-v42-cohere-direct",
            "b32df510da8125b91248bdc29f8f7c7cc6b9ab2abccabe762e89cfa00b9965b0",
        ),
        (
            "pilot-v43-cohere-direct",
            "814af7f7aaa5be5b76674292ef3a5f09a05a1303f969ab78ddbf47536cf68e2d",
        ),
        (
            "pilot-v44-cohere-direct",
            "b39a1de038f40f8d16a73597788ed9263d91dd4a54e556fcd29578b4284d8b30",
        ),
    )
]


def test_protocol_sensitivity_keeps_real_disjoint_strata_separate(tmp_path: Path) -> None:
    strict = verify_runs(STRICT_RUNS).aggregate
    high = verify_runs(HIGH_RESOURCE_RUNS).aggregate
    comparison = compare_strata(strict, high)

    assert comparison.aggregate["task_sets_disjoint"] is True
    assert comparison.aggregate["quality_ranking"] is False
    assert comparison.aggregate["synthetic_tasks"] == 0
    assert comparison.aggregate["strict"]["complete_pairs"] == 101
    assert comparison.aggregate["strict"]["scheduled_pairs"] == 168
    assert comparison.aggregate["high_resource"]["complete_pairs"] == 90
    assert comparison.aggregate["high_resource"]["scheduled_pairs"] == 128
    assert comparison.aggregate["combined_inventory"]["complete_pairs"] == 191
    assert comparison.aggregate["combined_inventory"]["completed_response_arms"] == 480
    assert comparison.aggregate["combined_inventory"]["provider_generation_ids"] == 1739
    assert comparison.aggregate["combined_inventory"]["epicure_calls"] == 454
    assert (
        comparison.aggregate["combined_inventory"][
            "models_with_at_least_eight_complete_pairs"
        ]
        == 14
    )
    assert comparison.aggregate["combined_inventory"]["minimum_complete_pairs_per_model"] == 8
    assert comparison.aggregate["combined_inventory"]["maximum_complete_pairs_per_model"] == 19
    assert all(row["minimum_eight_complete_pairs"] for row in comparison.model_rows)
    assert comparison.aggregate["combined_inventory"]["synthetic_tasks"] == 0
    assert len(comparison.model_rows) == 14
    assert {row["display_name"] for row in comparison.model_rows} >= {
        "Claude Opus 5",
        "Claude Sonnet 5",
        "DeepSeek V4 Pro",
        "DeepSeek V4 Flash",
        "Kimi K3",
    }
    outputs = write_assets(comparison, tmp_path)
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    assert outputs["figure"].with_suffix(".svg").is_file()


def test_protocol_sensitivity_rejects_task_overlap() -> None:
    strict = verify_runs(STRICT_RUNS).aggregate
    high = deepcopy(verify_runs(HIGH_RESOURCE_RUNS).aggregate)
    high["tasks"][0] = deepcopy(strict["tasks"][0])
    high["task_set_sha256"] = sha256_json(high["tasks"])
    high["artifact_sha256"] = sha256_json(
        {key: value for key, value in high.items() if key != "artifact_sha256"}
    )

    with pytest.raises(FrontierProtocolSensitivityError, match="disjoint tasks"):
        compare_strata(strict, high)


def test_cohere_provider_charge_is_unpriced_instead_of_zero(tmp_path: Path) -> None:
    strict = verify_runs(STRICT_RUNS).aggregate
    high = verify_runs([*HIGH_RESOURCE_RUNS, *COHERE_RUNS]).aggregate

    assert high["cost"]["provider_charge_complete"] is False
    assert high["cost"]["unpriced_model_ids"] == [
        "cohere/command-a-plus-05-2026",
        "cohere/command-a-reasoning-08-2025",
    ]
    cohere_rows = [
        row for row in high["model_rows"] if row["execution_backend"] == "cohere_direct"
    ]
    assert len(cohere_rows) == 2
    assert all(row["conservative_cost_exposure_usd"] is None for row in cohere_rows)
    assert all(row["cost_display_status"] == "provider_charge_unavailable" for row in cohere_rows)

    outputs = write_assets(compare_strata(strict, high), tmp_path)
    table = outputs["table"].read_text(encoding="utf-8")
    assert table.count("not returned") == 2
    assert "53.708 + 2 unpriced" in table
