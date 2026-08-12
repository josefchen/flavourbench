"""Freeze the v20 powered plan with bounded reasoning on supporting routes."""

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
from .epicure_selection_powered_plan_v19 import verify_plan as verify_v19_plan
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v17"
PLAN_VERSION = "flavourbench-selection-20x640-v17"
SUCCESSOR_RUN_CAP_USD = "76"


class SelectionPoweredPlanV20Error(RuntimeError):
    """The bounded-reasoning v20 plan failed verification."""


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
        raise SelectionPoweredPlanV20Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV20Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    calibration_v19: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v19_plan(predecessor):
        raise SelectionPoweredPlanV20Error("v20 requires the exact v19 predecessor")
    candidates = select_candidates(manifest)
    if len(candidates) != 20:
        raise SelectionPoweredPlanV20Error("v20 manifest does not contain 20 candidates")
    supported = {
        candidate.model_id
        for candidate in candidates
        if "reasoning_effort" in set(candidate.endpoint.get("supported_parameters") or [])
    }
    if len(supported) != 14:
        raise SelectionPoweredPlanV20Error("reasoning-control support set changed")
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_reasoning_calibration_before_primary_responses"
    document["inputs"]["plan_v19_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v19"] = {
        **dict(calibration_v19),
        "interface_finding": (
            "four reasoning routes produced eight length terminations before 934 sealed cells, "
            "including four of 45 DeepSeek Flash cells"
        ),
        "successor_change": (
            "freeze minimal hidden reasoning for every exact endpoint that advertises the "
            "reasoning_effort control; leave unsupported endpoints provider-fixed"
        ),
        "captured_responses_remain_calibration_only": True,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    for row in document["roster"]["models"]:
        row["final_reasoning_effort"] = (
            "minimal" if row["model_id"] in supported else "provider_fixed"
        )
    document["execution"]["reasoning_control"] = (
        "minimal hidden reasoning on all 14 endpoints that advertise reasoning_effort; "
        "provider-fixed on the six endpoints without that control"
    )
    document["execution"]["reasoning_control_model_ids"] = sorted(supported)
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v19["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV20Error("constructed v20 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        controlled = document["execution"]["reasoning_control_model_ids"]
        calibration = document["inputs"]["calibration_v19"]
        predecessor = document["inputs"]["plan_v19_predecessor"]
        reasoning = {row["model_id"]: row["final_reasoning_effort"] for row in roster}
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(controlled) == 14
        and len(set(controlled)) == 14
        and all(reasoning.get(model_id) == "minimal" for model_id in controlled)
        and all(
            reasoning.get(row["model_id"]) == "provider_fixed"
            for row in roster
            if row["model_id"] not in controlled
        )
        and calibration.get("response_count") == 934
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
            raise SelectionPoweredPlanV20Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v19-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    calibration = run_commitment(args.calibration_v19_directory, expected_responses=934)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        calibration_v19=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
