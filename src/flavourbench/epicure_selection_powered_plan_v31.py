"""Freeze v31 with a uniform 16,384-token response ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan import run_commitment, selection_execution_policy
from .epicure_selection_powered_plan_v30 import verify_plan as verify_v30_plan
from .execution_policy import ExecutionPolicy, verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v28"
PLAN_VERSION = "flavourbench-selection-20x640-v28"
SUCCESSOR_RUN_CAP_USD = "72"
MAX_OUTPUT_TOKENS = 16_384


class SelectionPoweredPlanV31Error(RuntimeError):
    """The v31 uniform-ceiling successor failed verification."""


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
        raise SelectionPoweredPlanV31Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV31Error("plan input is not a JSON object")
    return value


def selection_execution_policy_v31() -> ExecutionPolicy:
    """Change only the symmetric final-response ceiling."""

    return replace(selection_execution_policy(), max_output_tokens=MAX_OUTPUT_TOKENS)


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    calibration_v30: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v30_plan(predecessor):
        raise SelectionPoweredPlanV31Error("v31 requires the exact v30 predecessor")
    candidates = select_candidates(manifest)
    current_roster = [candidate.model_id for candidate in candidates]
    predecessor_roster = [row["model_id"] for row in predecessor["roster"]["models"]]
    if current_roster != predecessor_roster or len(current_roster) != 20:
        raise SelectionPoweredPlanV31Error("v31 must retain the exact v30 roster and order")
    predecessor_route = predecessor["inputs"]["route_manifest"]
    if (
        manifest["content_address"]["digest"] != predecessor_route["semantic_sha256"]
        or manifest_physical_sha256 != predecessor_route["physical_sha256"]
    ):
        raise SelectionPoweredPlanV31Error("v31 must reuse the exact v30 route manifest")

    policy = selection_execution_policy_v31()
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v30_calibration_before_primary_responses"
    document["inputs"]["plan_v30_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v30"] = {
        **dict(calibration_v30),
        "interface_finding": (
            "the exact CoreWeave Nemotron 3.5 Lightning route produced four normal responses; "
            "one further response reached the uniform 8192-token ceiling with 8165 provider-"
            "accounted completion tokens before the eight-cell check was stopped"
        ),
        "successor_change": (
            "raise max_output_tokens uniformly from 8192 to 16384 for all twenty models; retain "
            "the exact roster, routes, prompts, task order, parser, Epicure scores, and inference"
        ),
        "captured_responses_remain_calibration_only": True,
    }
    document["execution"]["execution_policy"] = policy.document()
    document["execution"]["execution_policy_sha256"] = policy.sha256
    document["execution"]["uniform_output_ceiling_successor"] = {
        "predecessor_max_output_tokens": 8_192,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "applies_to_all_models": True,
        "prompt_or_scoring_change": False,
        "route_or_roster_change": False,
        "required_before_full_collection": "eight_completed_normal_lightning_responses",
    }
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v30["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV31Error("constructed v31 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        execution = document["execution"]
        policy_document = execution["execution_policy"]
        successor = execution["uniform_output_ceiling_successor"]
        calibration = document["inputs"]["calibration_v30"]
        predecessor = document["inputs"]["plan_v30_predecessor"]
        route = document["inputs"]["route_manifest"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len({row["model_id"] for row in roster}) == 20
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and execution.get("execution_policy_sha256") == policy.sha256
        and policy_document["limits"]["max_output_tokens"] == MAX_OUTPUT_TOKENS
        and successor
        == {
            "predecessor_max_output_tokens": 8_192,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "applies_to_all_models": True,
            "prompt_or_scoring_change": False,
            "route_or_roster_change": False,
            "required_before_full_collection": "eight_completed_normal_lightning_responses",
        }
        and calibration.get("response_count") == 5
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
            raise SelectionPoweredPlanV31Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v30-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    calibration = run_commitment(args.calibration_v30_directory, expected_responses=5)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        calibration_v30=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
