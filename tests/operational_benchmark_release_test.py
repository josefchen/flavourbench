from collections import Counter
from pathlib import Path

from flavourbench.frontier_multirun_assets import RunInput
from flavourbench.operational_benchmark_release import (
    build_operational_benchmark_release,
    render_release,
)
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/season1/current-quality-run"
TASK_BANK = (
    ROOT
    / "data/season0/frozen"
    / "season0-real-task-bank-1ce969bdee4124fa44bab46a04feda2a0ebeddf4d37c49c0264b48b3833a4313.json"
)
RUNS = (
    ("pilot-v29-high-resource", "9710525c84feed31ed7ddfa6ae172cff915b36b8fdab8f7dedbeba2bdb0c8084"),
    (
        "pilot-v30-floor-replenishment",
        "6fe0e3ff11572069900bb1a06b24bc7377ea6440c92d24e4567f5138db4553b6",
    ),
    (
        "pilot-v32-floor-replenishment",
        "26b0392db5c4e1ae3a4e8f7ce53b4981f0b9bf0ab9e9ea27d71fd5b9a17593fe",
    ),
    ("pilot-v33-mistral-floor", "93e134e2bacd766afb5bb18cd558d6d352991b5acb05eb162464a9ff47b3260e"),
    ("pilot-v42-cohere-direct", "b32df510da8125b91248bdc29f8f7c7cc6b9ab2abccabe762e89cfa00b9965b0"),
    ("pilot-v43-cohere-direct", "814af7f7aaa5be5b76674292ef3a5f09a05a1303f969ab78ddbf47536cf68e2d"),
    ("pilot-v44-cohere-direct", "b39a1de038f40f8d16a73597788ed9263d91dd4a54e556fcd29578b4284d8b30"),
)


def _inputs() -> list[RunInput]:
    return [
        RunInput(
            RUN_ROOT / root / "summaries" / f"real-exploratory-summary-{digest}.json",
            RUN_ROOT / root / "source",
            RUN_ROOT / root / "responses",
        )
        for root, digest in RUNS
    ]


def test_builds_public_reproducibility_dataset_from_verified_raw_runs() -> None:
    release = build_operational_benchmark_release(_inputs(), TASK_BANK)

    assert release["artifact_sha256"] == sha256_json(
        {key: value for key, value in release.items() if key != "artifact_sha256"}
    )
    assert release["status"] == "public_reproducibility_dataset"
    assert len(release["tasks"]) == 16
    assert len(release["models"]) == 16
    assert len(release["pair_records"]) == 152
    assert sum(row["verified_pair_complete"] for row in release["pair_records"]) == 110
    assert (
        sum(
            row[condition]["epicure_calls"]
            for row in release["pair_records"]
            for condition in ("epicure_off", "epicure_on")
        )
        == 273
    )
    assert (
        sum(
            row[condition]["epicure_successful_calls"]
            for row in release["pair_records"]
            for condition in ("epicure_off", "epicure_on")
        )
        == 207
    )
    assert all(task["prompt"] and task["source_url"] for task in release["tasks"])
    assert {task["source_license"] for task in release["tasks"]} <= {
        "CC BY-SA 3.0",
        "CC BY-SA 4.0",
    }
    assert release["claim_boundary"]["raw_provider_text_distributed"] is False
    assert release["claim_boundary"]["culinary_quality_ranking_supported"] is False

    counts = Counter(row["model_id"] for row in release["pair_records"])
    assert set(counts) == {row["model_id"] for row in release["models"]}
    assert min(counts.values()) >= 8
    assert render_release(release).endswith(b"\n")


def test_public_dataset_contains_no_response_or_tool_payload_text() -> None:
    release = build_operational_benchmark_release(_inputs(), TASK_BANK)
    payload = render_release(release)

    forbidden = (
        b'"answer_markdown"',
        b'"output_json"',
        b'"tool_trace"',
        b'"arguments"',
        b'"result"',
        b'"human_reference"',
    )
    assert all(marker not in payload for marker in forbidden)
