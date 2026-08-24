"""Freeze final Qwen/DeepSeek transport settings for the 26-model run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from .epicure_selection_powered_plan_v37 import (
    NEW_MODEL_IDS,
)
from .epicure_selection_powered_plan_v37 import (
    verify_plan as verify_plan_v37,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26_v37 import FINAL_PANEL_ORDER

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v38"
PLAN_VERSION = "flavourbench-selection-26x640-v38"
QWEN_ID = "qwen/qwen3.8-2.4t-a95b"
DEEPSEEK_ID = "deepseek/deepseek-v4-pro-0813"
FINAL_MAX_OUTPUT_TOKENS = {QWEN_ID: 16_384, DEEPSEEK_ID: 4_096}


class SelectionPoweredPlanV38Error(RuntimeError):
    """The final transport-setting successor failed verification."""


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
        raise SelectionPoweredPlanV38Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV38Error("input is not a JSON object")
    return value


def _transport_commitment(directory: Path, *, plan_sha256: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((directory / "responses/primary").glob("*/response-*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(row)
        recorded = str(payload.pop("artifact_sha256", ""))
        if recorded != _sha256(payload) or row.get("plan_sha256") != plan_sha256:
            raise SelectionPoweredPlanV38Error("v37 transport response failed integrity")
        rows.append(row)
    if len(rows) != 32 or len({str(row["model_id"]) for row in rows}) != 8:
        raise SelectionPoweredPlanV38Error("v37 transport panel is incomplete")
    return {
        "plan_sha256": plan_sha256,
        "response_count": len(rows),
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "response_artifact_set_sha256": _sha256(
            sorted(str(row["artifact_sha256"]) for row in rows)
        ),
        "attempt_journal_physical_sha256": _sha256_file(
            directory / "attempts/provider-attempts.jsonl"
        ),
        "scores_or_selections_used": False,
        "responses_used_as_final_score_data": False,
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    transport: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_plan_v37(predecessor):
        raise SelectionPoweredPlanV38Error("v38 requires the exact v37 predecessor")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != FINAL_PANEL_ORDER:
        raise SelectionPoweredPlanV38Error("v38 manifest roster/order changed")
    predecessor_route = predecessor["inputs"]["route_manifest"]
    if (
        predecessor_route["semantic_sha256"] != manifest["content_address"]["digest"]
        or predecessor_route["physical_sha256"] != manifest_physical_sha256
    ):
        raise SelectionPoweredPlanV38Error("v38 must retain the exact v37 route manifest")
    if transport.get("completed_count") != 24 or transport.get("failed_count") != 8:
        raise SelectionPoweredPlanV38Error("v37 transport boundary changed")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v37_transport_before_clean_complete_blocks"
    document["inputs"]["plan_v37_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v37"] = {
        **dict(transport),
        "interface_findings": [
            "Qwen rejected reasoning effort none with HTTP 400",
            "one DeepSeek response exhausted the 2048-token ceiling",
            "Fable refusals persisted on the exact Anthropic route and remain scored failures",
        ],
        "successor_change": (
            "restore minimal Qwen reasoning with a 16384-token ceiling; retain minimal DeepSeek "
            "reasoning with a 4096-token ceiling; change no prompt, task, score, or inference"
        ),
    }
    rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
    rows[QWEN_ID]["final_reasoning_effort"] = "minimal"
    rows[QWEN_ID]["final_max_output_tokens"] = FINAL_MAX_OUTPUT_TOKENS[QWEN_ID]
    rows[DEEPSEEK_ID]["final_reasoning_effort"] = "minimal"
    rows[DEEPSEEK_ID]["final_max_output_tokens"] = FINAL_MAX_OUTPUT_TOKENS[DEEPSEEK_ID]
    document["execution"]["frontier_refresh_successor"].update(
        {
            "transport_plan_sha256": predecessor["artifact_sha256"],
            "transport_responses_used_as_score_data": False,
            "new_model_ids": list(NEW_MODEL_IDS),
        }
    )
    document["execution"]["reasoning_control"] = (
        "reasoning disabled for Seed 2.1 Turbo; minimal on the other controllable successor "
        "routes; route-specific final ceilings bound Qwen and DeepSeek"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV38Error("constructed v38 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = list(document["roster"]["models"])
        rows = {str(row["model_id"]): row for row in roster}
        policy_document = document["execution"]["execution_policy"]
        calibration = document["inputs"]["calibration_v37"]
        predecessor = document["inputs"]["plan_v37_predecessor"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == 26
        and tuple(row["model_id"] for row in roster) == FINAL_PANEL_ORDER
        and rows[QWEN_ID].get("final_reasoning_effort") == "minimal"
        and rows[QWEN_ID].get("final_max_output_tokens") == FINAL_MAX_OUTPUT_TOKENS[QWEN_ID]
        and rows[DEEPSEEK_ID].get("final_reasoning_effort") == "minimal"
        and rows[DEEPSEEK_ID].get("final_max_output_tokens") == FINAL_MAX_OUTPUT_TOKENS[DEEPSEEK_ID]
        and calibration.get("response_count") == 32
        and calibration.get("completed_count") == 24
        and calibration.get("failed_count") == 8
        and calibration.get("scores_or_selections_used") is False
        and calibration.get("responses_used_as_final_score_data") is False
        and document["execution"]["frontier_refresh_successor"].get(
            "transport_responses_used_as_score_data"
        )
        is False
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["design"].get("primary_provider_calls") == 26 * 640
        and document["design"].get("repeat_provider_calls") == 26 * 64
        and document["budget"].get("hard_cap") == "200"
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV38Error("content-addressed plan conflict")
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
    parser.add_argument("--transport-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    transport = _transport_commitment(
        args.transport_directory, plan_sha256=predecessor["artifact_sha256"]
    )
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        transport=transport,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
