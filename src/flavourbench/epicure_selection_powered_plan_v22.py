"""Freeze the v22 powered plan with provider-safe GPT-5.6 concurrency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan import run_commitment
from .epicure_selection_powered_plan_v21 import verify_plan as verify_v21_plan

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v19"
PLAN_VERSION = "flavourbench-selection-20x640-v19"
SUCCESSOR_RUN_CAP_USD = "74"
GPT_MODEL_IDS = frozenset(
    {
        "openai/gpt-5.6-sol-pro",
        "openai/gpt-5.6-terra-pro",
        "openai/gpt-5.6-luna-pro",
    }
)


class SelectionPoweredPlanV22Error(RuntimeError):
    """The v22 provider-safe successor failed verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV22Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV22Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    calibration_v21: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v21_plan(predecessor):
        raise SelectionPoweredPlanV22Error("v22 requires the exact v21 predecessor")
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v21_calibration_before_primary_responses"
    document["inputs"]["plan_v21_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v21"] = {
        **dict(calibration_v21),
        "interface_finding": (
            "GPT-5.6 Luna produced two upstream 429 failures in its first eight observed cells "
            "while Sol, Terra, and Luna could jointly issue twelve concurrent requests"
        ),
        "successor_change": (
            "single-flight each GPT-5.6 route while preserving all task, scoring, reasoning, "
            "eligibility, route, and inference contracts"
        ),
        "captured_responses_remain_calibration_only": True,
    }
    overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
    for model_id in GPT_MODEL_IDS:
        overrides[model_id] = 1
    document["execution"]["collection_concurrency"]["reason"] = (
        "parallelize independent routes while single-flighting direct Kimi, direct Cohere, "
        "the exact Nemotron route, and each GPT-5.6 route sharing the upstream OpenAI provider"
    )
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v21["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV22Error("constructed v22 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
        calibration = document["inputs"]["calibration_v21"]
        predecessor = document["inputs"]["plan_v21_predecessor"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and all(overrides.get(model_id) == 1 for model_id in GPT_MODEL_IDS)
        and overrides.get("nvidia/nemotron-3-ultra-550b-a55b") == 1
        and calibration.get("response_count") == 181
        and calibration.get("used_as_primary_data") is False
        and calibration.get("captured_responses_remain_calibration_only") is True
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and document["design"]["primary_provider_calls"] == 12_800
        and document["inference"]["bootstrap_resamples"] == 50_000
        and document["inference"]["permutation_resamples"] == 100_000
        and document["budget"]["hard_cap"] == SUCCESSOR_RUN_CAP_USD
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV22Error("content-addressed plan conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-plan", type=Path, required=True)
    parser.add_argument("--calibration-v21-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    calibration = run_commitment(args.calibration_v21_directory, expected_responses=181)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        calibration_v21=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
