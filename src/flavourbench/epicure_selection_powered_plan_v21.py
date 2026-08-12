"""Freeze the v21 powered plan with reasoning disabled on two verbose routes."""

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
from .epicure_selection_powered_plan_v20 import verify_plan as verify_v20_plan

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v18"
PLAN_VERSION = "flavourbench-selection-20x640-v18"
SUCCESSOR_RUN_CAP_USD = "75"
NO_REASONING_MODEL_IDS = frozenset(
    {
        "deepseek/deepseek-v4-flash-0731",
        "z-ai/glm-5.2",
    }
)


class SelectionPoweredPlanV21Error(RuntimeError):
    """The v21 no-reasoning successor failed verification."""


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
        raise SelectionPoweredPlanV21Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV21Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    calibration_v20: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v20_plan(predecessor):
        raise SelectionPoweredPlanV21Error("v21 requires the exact v20 predecessor")
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v20_calibration_before_primary_responses"
    document["inputs"]["plan_v20_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v20"] = {
        **dict(calibration_v20),
        "interface_finding": (
            "minimal reasoning still produced two abnormal long DeepSeek Flash responses in "
            "31 observed cells and one abnormal long GLM response in 38 observed cells"
        ),
        "successor_change": (
            "disable reasoning for the exact DeepSeek Flash and GLM routes; preserve minimal "
            "reasoning on the other twelve controllable routes"
        ),
        "captured_responses_remain_calibration_only": True,
    }
    for row in document["roster"]["models"]:
        if row["model_id"] in NO_REASONING_MODEL_IDS:
            row["final_reasoning_effort"] = "none"
    document["execution"]["reasoning_control"] = (
        "reasoning disabled on DeepSeek Flash and GLM; minimal hidden reasoning on the other "
        "twelve endpoints that advertise reasoning_effort; provider-fixed on six endpoints"
    )
    document["execution"]["reasoning_disabled_model_ids"] = sorted(NO_REASONING_MODEL_IDS)
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v20["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV21Error("constructed v21 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        controlled = set(document["execution"]["reasoning_control_model_ids"])
        disabled = set(document["execution"]["reasoning_disabled_model_ids"])
        calibration = document["inputs"]["calibration_v20"]
        predecessor = document["inputs"]["plan_v20_predecessor"]
        reasoning = {row["model_id"]: row["final_reasoning_effort"] for row in roster}
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(controlled) == 14
        and disabled == NO_REASONING_MODEL_IDS
        and disabled <= controlled
        and all(reasoning.get(model_id) == "none" for model_id in disabled)
        and all(reasoning.get(model_id) == "minimal" for model_id in controlled - disabled)
        and all(
            reasoning.get(row["model_id"]) == "provider_fixed"
            for row in roster
            if row["model_id"] not in controlled
        )
        and calibration.get("response_count") == 579
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
            raise SelectionPoweredPlanV21Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v20-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    calibration = run_commitment(args.calibration_v20_directory, expected_responses=579)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        calibration_v20=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
