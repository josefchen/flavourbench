"""Freeze the v19 powered plan with the repaired exact Nemotron route."""

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
from .epicure_selection_powered_plan_v18 import NEMOTRON_MODEL_ID, QWEN_MODEL_ID
from .epicure_selection_powered_plan_v18 import verify_plan as verify_v18_plan
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v16"
PLAN_VERSION = "flavourbench-selection-20x640-v16"
SUCCESSOR_RUN_CAP_USD = "80"


class SelectionPoweredPlanV19Error(RuntimeError):
    """The exact route-bound v19 plan failed verification."""


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
        raise SelectionPoweredPlanV19Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV19Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    calibration_v18: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v18_plan(predecessor):
        raise SelectionPoweredPlanV19Error("v19 requires the exact v18 predecessor")
    candidates = select_candidates(manifest)
    if len(candidates) != 20:
        raise SelectionPoweredPlanV19Error("route successor does not contain 20 candidates")
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_route_calibration_before_primary_responses"
    document["inputs"]["plan_v18_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["inputs"]["calibration_v18"] = {
        **dict(calibration_v18),
        "interface_finding": (
            "the exact BaseTen route returned HTTP 429 for all eight sequential single-flight "
            "requests before billable generation"
        ),
        "successor_change": (
            "pin the restored exact Together route from a fresh endpoint catalog read"
        ),
        "captured_responses_remain_calibration_only": True,
    }
    roster = []
    for candidate in candidates:
        roster.append(
            {
                "slot_id": candidate.slot_id,
                "model_id": candidate.model_id,
                "model_name": candidate.model_name,
                "canonical_model_slug": candidate.canonical_model_slug,
                "execution_backend": candidate.execution_backend,
                "provider_tag": candidate.provider_tag,
                "provider_name": candidate.provider_name,
                "endpoint_sha256": candidate.endpoint_sha256,
                "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
                "backend_contract_sha256": candidate.backend_contract_sha256,
                "final_reasoning_effort": (
                    "minimal"
                    if candidate.model_id in {QWEN_MODEL_ID, NEMOTRON_MODEL_ID}
                    else "provider_fixed"
                ),
            }
        )
    document["roster"] = {"model_count": 20, "fallbacks": "disabled", "models": roster}
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v18["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV19Error("constructed v19 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        calibration = document["inputs"]["calibration_v18"]
        route = document["inputs"]["route_manifest"]
        predecessor = document["inputs"]["plan_v18_predecessor"]
        reasoning = {row["model_id"]: row["final_reasoning_effort"] for row in roster}
        providers = {row["model_id"]: row["provider_tag"] for row in roster}
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len({row["model_id"] for row in roster}) == 20
        and providers.get(NEMOTRON_MODEL_ID) == "together"
        and reasoning.get(NEMOTRON_MODEL_ID) == "minimal"
        and reasoning.get(QWEN_MODEL_ID) == "minimal"
        and document["execution"]["collection_concurrency"]["per_model_by_model_id"]
        == {NEMOTRON_MODEL_ID: 1}
        and calibration.get("response_count") == 8
        and calibration.get("used_as_primary_data") is False
        and calibration.get("captured_responses_remain_calibration_only") is True
        and isinstance(route.get("semantic_sha256"), str)
        and len(route["semantic_sha256"]) == 64
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
            raise SelectionPoweredPlanV19Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v18-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    calibration = run_commitment(args.calibration_v18_directory, expected_responses=8)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        calibration_v18=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
