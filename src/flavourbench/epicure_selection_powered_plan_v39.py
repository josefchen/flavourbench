"""Freeze a clean DeepSeek transport repair for the 26-model release."""

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

from .epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from .epicure_selection_powered_plan_v37 import NEW_MODEL_IDS
from .epicure_selection_powered_plan_v38 import verify_plan as verify_plan_v38
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26_v37 import FINAL_PANEL_ORDER

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v39"
PLAN_VERSION = "flavourbench-selection-26x640-v39"
DEEPSEEK_ID = "deepseek/deepseek-v4-pro-0813"
FINAL_MAX_OUTPUT_TOKENS = 16_384


class SelectionPoweredPlanV39Error(RuntimeError):
    """The clean DeepSeek transport successor failed verification."""


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
        raise SelectionPoweredPlanV39Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV39Error("input is not a JSON object")
    return value


def _verified_attempt_event(document: Mapping[str, Any], *, plan_sha256: str) -> Mapping[str, Any]:
    payload = dict(document)
    recorded = str(payload.pop("event_sha256", ""))
    event = payload.get("event")
    if (
        recorded != _sha256(payload)
        or payload.get("plan_sha256") != plan_sha256
        or not isinstance(event, Mapping)
    ):
        raise SelectionPoweredPlanV39Error("v38 attempt journal failed integrity")
    return event


def _transport_commitment(directory: Path, *, plan_sha256: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((directory / "responses/primary").glob("*/response-*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(row)
        recorded = str(payload.pop("artifact_sha256", ""))
        if recorded != _sha256(payload) or row.get("plan_sha256") != plan_sha256:
            raise SelectionPoweredPlanV39Error("v38 primary response failed integrity")
        rows.append(row)
    counts = Counter(str(row["model_id"]) for row in rows)
    if len(rows) != 8 * 640 or set(counts) != set(NEW_MODEL_IDS) or set(counts.values()) != {640}:
        raise SelectionPoweredPlanV39Error("v38 successor primary panel is incomplete")
    statuses = Counter(str(row["status"]) for row in rows if row["model_id"] == DEEPSEEK_ID)
    finish_reasons = Counter(
        str((row.get("generation") or {}).get("finish_reason") or "missing")
        for row in rows
        if row["model_id"] == DEEPSEEK_ID
    )
    deepseek_arms = {str(row["arm_id"]) for row in rows if str(row["model_id"]) == DEEPSEEK_ID}
    provider_finish_reasons: Counter[str] = Counter()
    journal_path = directory / "attempts/provider-attempts.jsonl"
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        document = json.loads(line)
        event = _verified_attempt_event(document, plan_sha256=plan_sha256)
        if event.get("arm_id") in deepseek_arms and event.get("event_type") == "response_received":
            provider_finish_reasons[
                str((event.get("metadata") or {}).get("finish_reason") or "missing")
            ] += 1
    return {
        "plan_sha256": plan_sha256,
        "response_count": len(rows),
        "per_model_response_count": dict(sorted(counts.items())),
        "deepseek_status_counts": dict(sorted(statuses.items())),
        "deepseek_finish_reason_counts": dict(sorted(finish_reasons.items())),
        "deepseek_provider_finish_reason_counts": dict(sorted(provider_finish_reasons.items())),
        "response_artifact_set_sha256": _sha256(
            sorted(str(row["artifact_sha256"]) for row in rows)
        ),
        "attempt_journal_physical_sha256": _sha256_file(journal_path),
        "previous_run_spend_micros": sum(
            int((row.get("generation") or {}).get("cost_micros") or 0) for row in rows
        ),
        "scores_or_selections_used": False,
        "responses_used_as_final_deepseek_score_data": False,
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    transport: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_plan_v38(predecessor):
        raise SelectionPoweredPlanV39Error("v39 requires the exact v38 predecessor")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != FINAL_PANEL_ORDER:
        raise SelectionPoweredPlanV39Error("v39 manifest roster/order changed")
    predecessor_route = predecessor["inputs"]["route_manifest"]
    if (
        predecessor_route["semantic_sha256"] != manifest["content_address"]["digest"]
        or predecessor_route["physical_sha256"] != manifest_physical_sha256
    ):
        raise SelectionPoweredPlanV39Error("v39 must retain the exact v37 route manifest")
    deepseek_statuses = transport.get("deepseek_status_counts") or {}
    if (
        transport.get("response_count") != 8 * 640
        or sum(int(value) for value in deepseek_statuses.values()) != 640
        or int(deepseek_statuses.get("failed", 0)) == 0
        or int((transport.get("deepseek_provider_finish_reason_counts") or {}).get("length", 0))
        == 0
    ):
        raise SelectionPoweredPlanV39Error("v38 DeepSeek transport boundary changed")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v38_transport_before_clean_deepseek_block"
    document["inputs"]["plan_v38_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v38_primary_transport"] = {
        **dict(transport),
        "interface_finding": (
            "the exact DeepSeek route frequently exhausted the 4096-token final ceiling"
        ),
        "successor_change": (
            "raise only DeepSeek's final ceiling to 16384 tokens; preserve its exact route, "
            "reasoning setting, prompt, tasks, score maps, and inference"
        ),
    }
    rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
    rows[DEEPSEEK_ID]["final_max_output_tokens"] = FINAL_MAX_OUTPUT_TOKENS
    document["execution"]["collection_concurrency"]["per_model_by_model_id"][DEEPSEEK_ID] = 8
    document["execution"]["frontier_refresh_successor"].update(
        {
            "new_model_ids": list(NEW_MODEL_IDS),
            "retained_v38_new_model_ids": [
                model_id for model_id in NEW_MODEL_IDS if model_id != DEEPSEEK_ID
            ],
            "rerun_model_ids": [DEEPSEEK_ID],
            "v38_deepseek_responses_used_as_score_data": False,
        }
    )
    document["execution"]["reasoning_control"] = (
        "v38 settings retained; only the DeepSeek final output ceiling increases to 16384"
    )
    prior_spend_micros = int(transport["previous_run_spend_micros"])
    remaining_micros = 200_000_000 - prior_spend_micros
    if remaining_micros <= 0:
        raise SelectionPoweredPlanV39Error("v38 spent the complete aggregate program cap")
    document["budget"]["prior_v38_spend_micros"] = prior_spend_micros
    document["budget"]["hard_cap"] = f"{remaining_micros / 1_000_000:.6f}"
    document["budget"]["aggregate_program_cap"] = "200"
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV39Error("constructed v39 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = list(document["roster"]["models"])
        rows = {str(row["model_id"]): row for row in roster}
        policy_document = document["execution"]["execution_policy"]
        calibration = document["inputs"]["calibration_v38_primary_transport"]
        predecessor = document["inputs"]["plan_v38_predecessor"]
        successor = document["execution"]["frontier_refresh_successor"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == 26
        and tuple(row["model_id"] for row in roster) == FINAL_PANEL_ORDER
        and rows[DEEPSEEK_ID].get("final_reasoning_effort") == "minimal"
        and rows[DEEPSEEK_ID].get("final_max_output_tokens") == FINAL_MAX_OUTPUT_TOKENS
        and document["execution"]["collection_concurrency"]
        .get("per_model_by_model_id", {})
        .get(DEEPSEEK_ID)
        == 8
        and calibration.get("response_count") == 8 * 640
        and sum(calibration.get("per_model_response_count", {}).values()) == 8 * 640
        and calibration.get("deepseek_provider_finish_reason_counts", {}).get("length", 0) > 0
        and isinstance(calibration.get("previous_run_spend_micros"), int)
        and calibration.get("scores_or_selections_used") is False
        and calibration.get("responses_used_as_final_deepseek_score_data") is False
        and successor.get("rerun_model_ids") == [DEEPSEEK_ID]
        and set(successor.get("retained_v38_new_model_ids", []))
        == set(NEW_MODEL_IDS) - {DEEPSEEK_ID}
        and successor.get("v38_deepseek_responses_used_as_score_data") is False
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["design"].get("primary_provider_calls") == 26 * 640
        and document["design"].get("repeat_provider_calls") == 26 * 64
        and document["budget"].get("prior_v38_spend_micros")
        == calibration.get("previous_run_spend_micros")
        and document["budget"].get("hard_cap")
        == f"{(200_000_000 - int(calibration['previous_run_spend_micros'])) / 1_000_000:.6f}"
        and document["budget"].get("aggregate_program_cap") == "200"
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV39Error("content-addressed plan conflict")
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
