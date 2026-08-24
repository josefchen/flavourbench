"""Freeze the v23 powered plan on the exact OpenRouter Kimi route."""

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
from .epicure_selection_powered_plan_v22 import verify_plan as verify_v22_plan
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v20"
PLAN_VERSION = "flavourbench-selection-20x640-v20"
SUCCESSOR_RUN_CAP_USD = "73"
KIMI_MODEL_ID = "moonshotai/kimi-k3"
KIMI_ROUTE = "deepinfra/bf16"


class SelectionPoweredPlanV23Error(RuntimeError):
    """The v23 Kimi-route successor failed verification."""


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
        raise SelectionPoweredPlanV23Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV23Error("plan input is not a JSON object")
    return value


def _roster_row(candidate: Any, effort: str) -> dict[str, Any]:
    return {
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
        "final_reasoning_effort": effort,
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    calibration_v22: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v22_plan(predecessor):
        raise SelectionPoweredPlanV23Error("v23 requires the exact v22 predecessor")
    candidates = select_candidates(manifest)
    if len(candidates) != 20:
        raise SelectionPoweredPlanV23Error("v23 manifest does not contain 20 candidates")
    prior_effort = {
        row["model_id"]: row["final_reasoning_effort"] for row in predecessor["roster"]["models"]
    }
    prior_effort[KIMI_MODEL_ID] = "low"
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v22_calibration_before_primary_responses"
    document["inputs"]["plan_v22_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v22"] = {
        **dict(calibration_v22),
        "interface_finding": (
            "direct Kimi returned billing-cycle quota HTTP 403 responses despite single-flight"
        ),
        "successor_change": (
            "freeze Kimi to the exact DeepInfra BF16 OpenRouter route with no fallback and low "
            "reasoning; preserve every other model route and analysis contract"
        ),
        "captured_responses_remain_calibration_only": True,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["roster"]["models"] = [
        _roster_row(candidate, prior_effort[candidate.model_id]) for candidate in candidates
    ]
    controlled = set(document["execution"]["reasoning_control_model_ids"])
    controlled.add(KIMI_MODEL_ID)
    document["execution"]["reasoning_control_model_ids"] = sorted(controlled)
    document["execution"]["reasoning_control"] = (
        "reasoning disabled on DeepSeek Flash and GLM; low reasoning on Kimi K3; minimal hidden "
        "reasoning on the other twelve controllable endpoints; provider-fixed on five endpoints"
    )
    overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
    overrides[KIMI_MODEL_ID] = 1
    document["execution"]["collection_concurrency"]["reason"] = (
        "parallelize independent routes while single-flighting exact Kimi, exact Nemotron, each "
        "GPT-5.6 route, and direct Cohere through its shared backend limiter"
    )
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v22["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV23Error("constructed v23 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        rows = {row["model_id"]: row for row in roster}
        controlled = set(document["execution"]["reasoning_control_model_ids"])
        disabled = set(document["execution"]["reasoning_disabled_model_ids"])
        overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
        calibration = document["inputs"]["calibration_v22"]
        predecessor = document["inputs"]["plan_v22_predecessor"]
        route = document["inputs"]["route_manifest"]
    except (KeyError, TypeError):
        return False
    kimi = rows.get(KIMI_MODEL_ID) or {}
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(rows) == 20
        and len(controlled) == 15
        and disabled == {"deepseek/deepseek-v4-flash-0731", "z-ai/glm-5.2"}
        and kimi.get("execution_backend") == "openrouter"
        and kimi.get("provider_tag") == KIMI_ROUTE
        and kimi.get("final_reasoning_effort") == "low"
        and overrides.get(KIMI_MODEL_ID) == 1
        and calibration.get("response_count") == 186
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
            raise SelectionPoweredPlanV23Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v22-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    calibration = run_commitment(args.calibration_v22_directory, expected_responses=186)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        calibration_v22=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
