"""Freeze v28 with Fable restored and rate-limited routes single-flighted."""

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
from .epicure_selection_powered_plan_v23 import _roster_row
from .epicure_selection_powered_plan_v27 import verify_plan as verify_v27_plan
from .epicure_selection_route_manifest_v28 import (
    FABLE_CANONICAL_ID,
    FABLE_MODEL_ID,
    NEMOTRON_MODEL_ID,
    REPLACEMENT_PROVIDER,
    REPLACEMENT_TAG,
)
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v25"
PLAN_VERSION = "flavourbench-selection-20x640-v25"
SUCCESSOR_RUN_CAP_USD = "72"
LLAMA_MODEL_ID = "meta-llama/llama-4-maverick"
FABLE_CHECK_TASK_IDS = frozenset(
    {
        "fb-executable-constraint-031",
        "fb-executable-cultural_composition-072",
        "fb-executable-cultural_composition-085",
        "fb-executable-pairing-029",
        "fb-executable-pairing-063",
        "fb-executable-pairing-088",
        "fb-executable-pairing-114",
        "fb-executable-substitution-079",
    }
)


class SelectionPoweredPlanV28Error(RuntimeError):
    """The v28 roster/concurrency successor failed verification."""


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
        raise SelectionPoweredPlanV28Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV28Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    calibration_v27: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v27_plan(predecessor):
        raise SelectionPoweredPlanV28Error("v28 requires the exact v27 predecessor")
    candidates = select_candidates(manifest)
    if len(candidates) != 20:
        raise SelectionPoweredPlanV28Error("v28 manifest does not contain 20 candidates")
    effort = {
        row["model_id"]: row["final_reasoning_effort"] for row in predecessor["roster"]["models"]
    }
    effort.pop(NEMOTRON_MODEL_ID)
    effort[FABLE_MODEL_ID] = "minimal"
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v27_calibration_before_primary_responses"
    document["inputs"]["plan_v27_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v27"] = {
        **dict(calibration_v27),
        "interface_findings": [
            "Llama Maverick exhausted two HTTP 429 retries in its first 52 recorded cells",
            (
                "Nemotron recorded one ambiguous HTTP 503 and one 7971-token length "
                "termination in 11 cells"
            ),
        ],
        "successor_changes": [
            "single-flight the unchanged exact Llama route",
            (
                "replace the under-reliable Nemotron slot with Claude Fable 5 on exact "
                "Amazon Bedrock after the user-confirmed usage-limit repair"
            ),
        ],
        "captured_responses_remain_calibration_only": True,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["roster"]["models"] = [
        _roster_row(candidate, effort[candidate.model_id]) for candidate in candidates
    ]
    controlled = set(document["execution"]["reasoning_control_model_ids"])
    controlled.remove(NEMOTRON_MODEL_ID)
    controlled.add(FABLE_MODEL_ID)
    document["execution"]["reasoning_control_model_ids"] = sorted(controlled)
    document["execution"]["reasoning_control"] = (
        "reasoning disabled on DeepSeek Flash and GLM; low reasoning on Kimi K3; minimal hidden "
        "reasoning on the other twelve controllable endpoints; provider-fixed on five endpoints"
    )
    overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
    overrides.pop(NEMOTRON_MODEL_ID)
    overrides[FABLE_MODEL_ID] = 1
    overrides[LLAMA_MODEL_ID] = 1
    document["execution"]["collection_concurrency"]["reason"] = (
        "parallelize independent routes while single-flighting Fable, Llama, DeepSeek V4 Pro, "
        "exact Kimi, and each GPT-5.6 route; direct Cohere uses its shared backend limiter"
    )
    document["execution"]["fable_requalification"] = {
        "task_ids": sorted(FABLE_CHECK_TASK_IDS),
        "selection_basis": "the same eight transport/content-filter probes used before removal",
        "required_before_full_collection": "eight_completed_normal_responses",
        "score_adaptive": False,
    }
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v27["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV28Error("constructed v28 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        rows = {row["model_id"]: row for row in roster}
        controlled = set(document["execution"]["reasoning_control_model_ids"])
        overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
        check = document["execution"]["fable_requalification"]
        calibration = document["inputs"]["calibration_v27"]
        predecessor = document["inputs"]["plan_v27_predecessor"]
        route = document["inputs"]["route_manifest"]
    except (KeyError, TypeError):
        return False
    fable = rows.get(FABLE_MODEL_ID) or {}
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(rows) == 20
        and NEMOTRON_MODEL_ID not in rows
        and len(controlled) == 15
        and FABLE_MODEL_ID in controlled
        and NEMOTRON_MODEL_ID not in controlled
        and fable.get("canonical_model_slug") == FABLE_CANONICAL_ID
        and fable.get("execution_backend") == "openrouter"
        and fable.get("provider_tag") == REPLACEMENT_TAG
        and fable.get("provider_name") == REPLACEMENT_PROVIDER
        and fable.get("final_reasoning_effort") == "minimal"
        and overrides.get(FABLE_MODEL_ID) == 1
        and overrides.get(LLAMA_MODEL_ID) == 1
        and NEMOTRON_MODEL_ID not in overrides
        and frozenset(check.get("task_ids") or []) == FABLE_CHECK_TASK_IDS
        and check.get("score_adaptive") is False
        and calibration.get("response_count") == 639
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
            raise SelectionPoweredPlanV28Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v27-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    calibration = run_commitment(args.calibration_v27_directory, expected_responses=639)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        calibration_v27=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
