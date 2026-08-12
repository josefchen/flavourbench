"""Freeze the transport-repaired v18 powered FlavourBench analysis plan."""

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
from .epicure_selection_powered_plan import verify_plan as verify_v17_plan

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v15"
PLAN_VERSION = "flavourbench-selection-20x640-v15"
NEMOTRON_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b"
QWEN_MODEL_ID = "qwen/qwen3.8-max"
SUCCESSOR_RUN_CAP_USD = "80"


class SelectionPoweredPlanV18Error(RuntimeError):
    """The v18 transport-repaired plan failed verification."""


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
        raise SelectionPoweredPlanV18Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV18Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    calibration_v17: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the sole outcome-blind transport repair to the frozen v17 plan."""
    if not verify_v17_plan(predecessor):
        raise SelectionPoweredPlanV18Error("v18 requires the exact valid v17 predecessor")
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_transport_calibration_before_primary_responses"
    document["inputs"]["plan_v17_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v17"] = {
        **dict(calibration_v17),
        "interface_findings": [
            "BaseTen Nemotron produced repeated HTTP 429 responses under four concurrent calls",
            "two Nemotron responses exhausted the 8192-token final-response ceiling",
        ],
        "successor_changes": [
            "freeze one in-flight request for the exact Nemotron route",
            "freeze minimal hidden reasoning for Nemotron as already used for Qwen",
        ],
        "interrupted_after_decisive_transport_finding": True,
        "captured_responses_remain_calibration_only": True,
    }
    for row in document["roster"]["models"]:
        if row["model_id"] == NEMOTRON_MODEL_ID:
            row["final_reasoning_effort"] = "minimal"
    concurrency = document["execution"]["collection_concurrency"]
    concurrency["per_model_by_model_id"] = {NEMOTRON_MODEL_ID: 1}
    concurrency["reason"] = (
        "parallelize independent OpenRouter cells while single-flighting direct Kimi, direct "
        "Cohere, and the rate-limited exact Nemotron route"
    )
    document["execution"]["reasoning_control"] = (
        "minimal hidden reasoning for Qwen and Nemotron; provider-fixed for other OpenRouter "
        "and Kimi routes; Cohere selection adapters use their frozen bounded modes"
    )
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v17["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV18Error("constructed v18 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        concurrency = document["execution"]["collection_concurrency"]
        calibration = document["inputs"]["calibration_v17"]
        predecessor = document["inputs"]["plan_v17_predecessor"]
        reasoning = {row["model_id"]: row["final_reasoning_effort"] for row in roster}
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and concurrency.get("per_model_by_model_id") == {NEMOTRON_MODEL_ID: 1}
        and reasoning.get(NEMOTRON_MODEL_ID) == "minimal"
        and reasoning.get(QWEN_MODEL_ID) == "minimal"
        and calibration.get("response_count") == 835
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
            raise SelectionPoweredPlanV18Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v17-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    calibration = run_commitment(args.calibration_v17_directory, expected_responses=835)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        calibration_v17=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
