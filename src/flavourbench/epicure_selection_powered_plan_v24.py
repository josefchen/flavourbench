"""Freeze the v24 powered plan on the Morph Kimi route."""

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
from .epicure_selection_powered_plan_v23 import (
    KIMI_MODEL_ID,
    _roster_row,
)
from .epicure_selection_powered_plan_v23 import (
    verify_plan as verify_v23_plan,
)
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v21"
PLAN_VERSION = "flavourbench-selection-20x640-v21"
SUCCESSOR_RUN_CAP_USD = "73"
KIMI_ROUTE = "morph"


class SelectionPoweredPlanV24Error(RuntimeError):
    """The v24 Morph-route successor failed verification."""


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
        raise SelectionPoweredPlanV24Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV24Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    calibration_v23: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v23_plan(predecessor):
        raise SelectionPoweredPlanV24Error("v24 requires the exact v23 predecessor")
    candidates = select_candidates(manifest)
    if len(candidates) != 20:
        raise SelectionPoweredPlanV24Error("v24 manifest does not contain 20 candidates")
    effort = {
        row["model_id"]: row["final_reasoning_effort"] for row in predecessor["roster"]["models"]
    }
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v23_calibration_before_primary_responses"
    document["inputs"]["plan_v23_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v23"] = {
        **dict(calibration_v23),
        "interface_finding": (
            "the exact DeepInfra Kimi route returned upstream HTTP 429 for both bounded checks"
        ),
        "successor_change": (
            "freeze Kimi to the exact Morph OpenRouter route; preserve low reasoning, "
            "single-flight execution, no fallback, and all analysis contracts"
        ),
        "captured_responses_remain_calibration_only": True,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["roster"]["models"] = [
        _roster_row(candidate, effort[candidate.model_id]) for candidate in candidates
    ]
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v23["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV24Error("constructed v24 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        rows = {row["model_id"]: row for row in roster}
        calibration = document["inputs"]["calibration_v23"]
        predecessor = document["inputs"]["plan_v23_predecessor"]
        route = document["inputs"]["route_manifest"]
        overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
    except (KeyError, TypeError):
        return False
    kimi = rows.get(KIMI_MODEL_ID) or {}
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(rows) == 20
        and kimi.get("execution_backend") == "openrouter"
        and kimi.get("provider_tag") == KIMI_ROUTE
        and kimi.get("final_reasoning_effort") == "low"
        and overrides.get(KIMI_MODEL_ID) == 1
        and calibration.get("response_count") == 2
        and calibration.get("used_as_primary_data") is False
        and calibration.get("captured_responses_remain_calibration_only") is True
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and isinstance(route.get("semantic_sha256"), str)
        and len(route["semantic_sha256"]) == 64
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
            raise SelectionPoweredPlanV24Error("content-addressed plan conflict")
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--calibration-v23-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    calibration = run_commitment(args.calibration_v23_directory, expected_responses=2)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        calibration_v23=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
