"""Freeze complete Cohere successor blocks after exact transport checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_native_powered_runner import _semantic_valid
from .epicure_selection_powered_plan_v31 import (
    MAX_OUTPUT_TOKENS,
    selection_execution_policy_v31,
)
from .epicure_selection_powered_plan_v34 import (
    PRIMARY_CELLS,
    REPEAT_CELLS,
    SUCCESSOR_MODEL_IDS,
    SUCCESSOR_RUN_CAP_USD,
    TRANSPORT_CELLS_PER_MODEL,
    TRANSPORT_TASK_IDS,
)
from .epicure_selection_powered_plan_v34 import verify_plan as verify_v34_plan
from .epicure_selection_route_manifest_v32 import (
    OPENROUTER_PROVIDER_NAME,
    OPENROUTER_PROVIDER_TAG,
    ROUTE_MAX_OUTPUT_TOKENS,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v32"
PLAN_VERSION = "flavourbench-selection-20x640-v32"
COHERE_CONCURRENCY = 4
TRANSPORT_RESPONSE_COUNT = TRANSPORT_CELLS_PER_MODEL * len(SUCCESSOR_MODEL_IDS)


class SelectionPoweredPlanV35Error(RuntimeError):
    """The v35 complete Cohere successor plan failed verification."""


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
        raise SelectionPoweredPlanV35Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV35Error("plan input is not a JSON object")
    return value


def transport_commitment(
    run_directory: Path,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    paths = sorted((run_directory / "responses/primary").glob("*/response-*.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise SelectionPoweredPlanV35Error("transport response is not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not _semantic_valid(value):
            raise SelectionPoweredPlanV35Error("transport response failed integrity")
        rows.append(value)
    if len(rows) != TRANSPORT_RESPONSE_COUNT:
        raise SelectionPoweredPlanV35Error("Cohere transport check is incomplete")
    by_model = Counter(str(row.get("model_id")) for row in rows)
    if by_model != Counter(
        {model_id: TRANSPORT_CELLS_PER_MODEL for model_id in SUCCESSOR_MODEL_IDS}
    ):
        raise SelectionPoweredPlanV35Error("Cohere transport model coverage changed")
    expected_tasks = set(TRANSPORT_TASK_IDS)
    canonical_by_model: dict[str, str] = {}
    for model_id in SUCCESSOR_MODEL_IDS:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        if {str(row["task_id"]) for row in model_rows} != expected_tasks:
            raise SelectionPoweredPlanV35Error("Cohere transport task coverage changed")
        for row in model_rows:
            generation = row.get("generation") or {}
            decoding = generation.get("decoding") or {}
            if (
                row.get("status") != "completed"
                or row.get("plan_sha256") != expected_plan_sha256
                or row.get("panel") != "primary"
                or row.get("execution_backend") != "openrouter"
                or row.get("provider_route") != OPENROUTER_PROVIDER_TAG
                or generation.get("actual_provider") != OPENROUTER_PROVIDER_NAME
                or generation.get("finish_reason") != "stop"
                or decoding.get("max_tokens") != ROUTE_MAX_OUTPUT_TOKENS
            ):
                raise SelectionPoweredPlanV35Error("Cohere transport response is not normal")
        identities = {
            str((row.get("generation") or {}).get("actual_model_id")) for row in model_rows
        }
        if len(identities) != 1 or "" in identities:
            raise SelectionPoweredPlanV35Error("Cohere transport identity is ambiguous")
        canonical_by_model[model_id] = identities.pop()

    journal = run_directory / "attempts/provider-attempts.jsonl"
    if journal.is_symlink() or not journal.is_file():
        raise SelectionPoweredPlanV35Error("Cohere transport journal is unavailable")
    return {
        "response_count": len(rows),
        "responses_per_model": dict(sorted(by_model.items())),
        "task_ids": list(TRANSPORT_TASK_IDS),
        "actual_model_ids": dict(sorted(canonical_by_model.items())),
        "actual_provider": OPENROUTER_PROVIDER_NAME,
        "response_artifact_set_sha256": _sha256(
            sorted(str(row["artifact_sha256"]) for row in rows)
        ),
        "attempt_journal_physical_sha256": _sha256_file(journal),
        "spend_micros": sum(
            int((row.get("generation") or {}).get("cost_micros") or 0) for row in rows
        ),
        "used_as_primary_data": False,
        "scores_or_selections_inspected_before_successor_freeze": False,
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v34_plan(predecessor):
        raise SelectionPoweredPlanV35Error("v35 requires the exact v34 predecessor")
    candidates = select_candidates(manifest)
    if [candidate.model_id for candidate in candidates] != [
        row["model_id"] for row in predecessor["roster"]["models"]
    ]:
        raise SelectionPoweredPlanV35Error("v35 route roster changed")
    predecessor_route = predecessor["inputs"]["route_manifest"]
    if (
        manifest["content_address"]["digest"] != predecessor_route["semantic_sha256"]
        or manifest_physical_sha256 != predecessor_route["physical_sha256"]
    ):
        raise SelectionPoweredPlanV35Error("v35 must retain the exact v34 route manifest")
    if calibration.get("response_count") != TRANSPORT_RESPONSE_COUNT:
        raise SelectionPoweredPlanV35Error("v35 transport commitment is incomplete")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = (
        "preregistered_after_sixteen_normal_cohere_checks_before_full_successor_blocks"
    )
    document["inputs"]["plan_v34_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v34"] = {
        **dict(calibration),
        "interface_finding": (
            "both exact Cohere routes completed all eight predetermined tasks with their frozen "
            "model identity, Cohere provider identity, stop finish, and 1800-token ceiling"
        ),
        "captured_responses_remain_calibration_only": True,
    }
    recovery = document["execution"]["cohere_route_successor"]
    recovery.update(
        {
            "transport_check_source_plan_sha256": predecessor["artifact_sha256"],
            "transport_checks_reused_as_primary": False,
            "successor_execution_order": [
                "complete_both_full_primary_blocks_under_v35",
                "verify_complete_exact_provider_blocks_without_inspecting_scores",
                "complete_both_repeat_blocks_under_v35",
            ],
        }
    )
    overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
    for model_id in SUCCESSOR_MODEL_IDS:
        overrides[model_id] = COHERE_CONCURRENCY
    document["execution"]["cohere_concurrency_successor"] = {
        "model_ids": list(SUCCESSOR_MODEL_IDS),
        "predecessor_concurrency": 1,
        "successor_concurrency": COHERE_CONCURRENCY,
        "basis": "sixteen_normal_exact_route_responses",
        "changes_scoring_or_inference": False,
        "changes_prompt_or_decoding": False,
        "changes_provider_or_model_identity": False,
    }
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["budget"]["program_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV35Error("constructed v35 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        rows = {row["model_id"]: row for row in roster}
        execution = document["execution"]
        recovery = execution["cohere_route_successor"]
        concurrency = execution["cohere_concurrency_successor"]
        calibration = document["inputs"]["calibration_v34"]
        predecessor = document["inputs"]["plan_v34_predecessor"]
        policy_document = execution["execution_policy"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(rows) == 20
        and all(
            rows[model_id].get("provider_tag") == OPENROUTER_PROVIDER_TAG
            and rows[model_id].get("provider_name") == OPENROUTER_PROVIDER_NAME
            and rows[model_id].get("final_max_output_tokens") == ROUTE_MAX_OUTPUT_TOKENS
            for model_id in SUCCESSOR_MODEL_IDS
        )
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and execution.get("execution_policy_sha256") == policy.sha256
        and policy_document["limits"]["max_output_tokens"] == MAX_OUTPUT_TOKENS
        and concurrency
        == {
            "model_ids": list(SUCCESSOR_MODEL_IDS),
            "predecessor_concurrency": 1,
            "successor_concurrency": COHERE_CONCURRENCY,
            "basis": "sixteen_normal_exact_route_responses",
            "changes_scoring_or_inference": False,
            "changes_prompt_or_decoding": False,
            "changes_provider_or_model_identity": False,
        }
        and all(
            execution["collection_concurrency"]["per_model_by_model_id"].get(model_id)
            == COHERE_CONCURRENCY
            for model_id in SUCCESSOR_MODEL_IDS
        )
        and recovery.get("transport_check_source_plan_sha256") == predecessor["semantic_sha256"]
        and recovery.get("transport_checks_reused_as_primary") is False
        and recovery.get("successor_primary_cells_per_model") == PRIMARY_CELLS
        and recovery.get("successor_repeat_cells_per_model") == REPEAT_CELLS
        and recovery.get("reuse_direct_responses") is False
        and recovery.get("cross_provider_score_pooling") is False
        and calibration.get("response_count") == TRANSPORT_RESPONSE_COUNT
        and calibration.get("responses_per_model")
        == {model_id: TRANSPORT_CELLS_PER_MODEL for model_id in sorted(SUCCESSOR_MODEL_IDS)}
        and calibration.get("task_ids") == list(TRANSPORT_TASK_IDS)
        and calibration.get("used_as_primary_data") is False
        and calibration.get("captured_responses_remain_calibration_only") is True
        and calibration.get("scores_or_selections_inspected_before_successor_freeze") is False
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and document["design"]["primary_provider_calls"] == 12_800
        and document["inference"]["bootstrap_resamples"] == 50_000
        and document["inference"]["permutation_resamples"] == 100_000
        and document["budget"]["hard_cap"] == SUCCESSOR_RUN_CAP_USD
        and document["budget"]["program_cap"] == SUCCESSOR_RUN_CAP_USD
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV35Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v34-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    calibration = transport_commitment(
        args.calibration_v34_directory,
        expected_plan_sha256=predecessor["artifact_sha256"],
    )
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        calibration=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
