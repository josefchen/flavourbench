"""Freeze a complete, non-selective Fable transport repair for the 26-model release."""

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
from .epicure_selection_powered_plan_v39 import DEEPSEEK_ID
from .epicure_selection_powered_plan_v39 import verify_plan as verify_plan_v39
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26_v37 import FINAL_PANEL_ORDER

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v40"
PLAN_VERSION = "flavourbench-selection-26x640-v40"
FABLE_ID = "anthropic/claude-fable-5"
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
PROGRAM_CAP_MICROS = 200_000_000


class SelectionPoweredPlanV40Error(RuntimeError):
    """The clean Fable transport successor failed verification."""


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
        raise SelectionPoweredPlanV40Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV40Error("input is not a JSON object")
    return value


def _load_verified_responses(
    directory: Path,
    *,
    panel: str,
    slot_id: str,
    model_id: str,
    plan_sha256: str,
    expected: int,
) -> list[dict[str, Any]]:
    paths = sorted((directory / "responses" / panel / slot_id).glob("response-*.json"))
    if len(paths) != expected:
        raise SelectionPoweredPlanV40Error(
            f"{model_id} {panel} response panel has {len(paths)} rows; expected {expected}"
        )
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(row)
        recorded = str(payload.pop("artifact_sha256", ""))
        if (
            recorded != _sha256(payload)
            or row.get("plan_sha256") != plan_sha256
            or row.get("model_id") != model_id
            or row.get("panel") != panel
        ):
            raise SelectionPoweredPlanV40Error(f"response failed integrity: {path}")
        cell_id = str(row.get("cell_id") or "")
        if path.name != f"response-{cell_id}-{recorded}.json":
            raise SelectionPoweredPlanV40Error("response filename is not content addressed")
        identity = (str(row.get("model_id")), str(row.get("task_id")))
        if identity in identities:
            raise SelectionPoweredPlanV40Error("response task identity is duplicated")
        identities.add(identity)
        rows.append(row)
    return rows


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("status")) for row in rows).items()))


def _spend(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(int((row.get("generation") or {}).get("cost_micros") or 0) for row in rows)


def _v38_fable_transport_commitment(
    directory: Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    row = next(
        (value for value in plan["roster"]["models"] if value["model_id"] == FABLE_ID),
        None,
    )
    if row is None:
        raise SelectionPoweredPlanV40Error("v38 Fable roster row is absent")
    primary = _load_verified_responses(
        directory,
        panel="primary",
        slot_id=str(row["slot_id"]),
        model_id=FABLE_ID,
        plan_sha256=str(plan["artifact_sha256"]),
        expected=PRIMARY_TASKS,
    )
    repeat = _load_verified_responses(
        directory,
        panel="repeat",
        slot_id=str(row["slot_id"]),
        model_id=FABLE_ID,
        plan_sha256=str(plan["artifact_sha256"]),
        expected=REPEAT_TASKS,
    )
    primary_statuses = _status_counts(primary)
    repeat_statuses = _status_counts(repeat)
    error_types = Counter(
        str((value.get("error") or {}).get("type") or "missing")
        for value in primary + repeat
        if value.get("status") != "completed"
    )
    error_message_hashes = Counter(
        hashlib.sha256(
            str((value.get("error") or {}).get("message") or "").encode("utf-8")
        ).hexdigest()
        for value in primary + repeat
        if value.get("status") != "completed"
    )
    journal = directory / "attempts/provider-attempts.jsonl"
    if journal.is_symlink() or not journal.is_file():
        raise SelectionPoweredPlanV40Error("v38 attempt journal is absent")
    return {
        "plan_sha256": plan["artifact_sha256"],
        "primary_response_count": len(primary),
        "repeat_response_count": len(repeat),
        "primary_status_counts": primary_statuses,
        "repeat_status_counts": repeat_statuses,
        "primary_completion_rate": primary_statuses.get("completed", 0) / PRIMARY_TASKS,
        "repeat_completion_rate": repeat_statuses.get("completed", 0) / REPEAT_TASKS,
        "failed_error_type_counts": dict(sorted(error_types.items())),
        "failed_error_message_sha256_counts": dict(sorted(error_message_hashes.items())),
        "response_artifact_set_sha256": _sha256(
            sorted(str(value["artifact_sha256"]) for value in primary + repeat)
        ),
        "attempt_journal_physical_sha256": _sha256_file(journal),
        "spend_micros": _spend(primary + repeat),
        "aggregate_fable_score_was_inspected_before_repair": True,
        "task_level_scores_or_selections_used_to_change_execution_contract": False,
        "execution_contract_changed": False,
        "complete_old_fable_block_used_as_final_score_data": False,
    }


def _v39_deepseek_commitment(
    directory: Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    row = next(
        (value for value in plan["roster"]["models"] if value["model_id"] == DEEPSEEK_ID),
        None,
    )
    if row is None:
        raise SelectionPoweredPlanV40Error("v39 DeepSeek roster row is absent")
    primary = _load_verified_responses(
        directory,
        panel="primary",
        slot_id=str(row["slot_id"]),
        model_id=DEEPSEEK_ID,
        plan_sha256=str(plan["artifact_sha256"]),
        expected=PRIMARY_TASKS,
    )
    repeat = _load_verified_responses(
        directory,
        panel="repeat",
        slot_id=str(row["slot_id"]),
        model_id=DEEPSEEK_ID,
        plan_sha256=str(plan["artifact_sha256"]),
        expected=REPEAT_TASKS,
    )
    return {
        "plan_sha256": plan["artifact_sha256"],
        "primary_response_count": len(primary),
        "repeat_response_count": len(repeat),
        "primary_status_counts": _status_counts(primary),
        "repeat_status_counts": _status_counts(repeat),
        "response_artifact_set_sha256": _sha256(
            sorted(str(value["artifact_sha256"]) for value in primary + repeat)
        ),
        "spend_micros": _spend(primary + repeat),
        "responses_used_as_final_deepseek_score_data": True,
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    fable_transport: Mapping[str, Any],
    deepseek_source: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_plan_v39(predecessor):
        raise SelectionPoweredPlanV40Error("v40 requires the exact v39 predecessor")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != FINAL_PANEL_ORDER:
        raise SelectionPoweredPlanV40Error("v40 manifest roster/order changed")
    predecessor_route = predecessor["inputs"]["route_manifest"]
    if (
        predecessor_route["semantic_sha256"] != manifest["content_address"]["digest"]
        or predecessor_route["physical_sha256"] != manifest_physical_sha256
    ):
        raise SelectionPoweredPlanV40Error("v40 must retain the exact v37 route manifest")
    primary_statuses = fable_transport.get("primary_status_counts") or {}
    repeat_statuses = fable_transport.get("repeat_status_counts") or {}
    if (
        fable_transport.get("primary_response_count") != PRIMARY_TASKS
        or fable_transport.get("repeat_response_count") != REPEAT_TASKS
        or sum(int(value) for value in primary_statuses.values()) != PRIMARY_TASKS
        or sum(int(value) for value in repeat_statuses.values()) != REPEAT_TASKS
        or int(primary_statuses.get("completed", 0))
        >= int(predecessor["eligibility"]["minimum_completed_tasks"])
        or int(primary_statuses.get("failed", 0)) == 0
        or fable_transport.get("execution_contract_changed") is not False
        or fable_transport.get("task_level_scores_or_selections_used_to_change_execution_contract")
        is not False
    ):
        raise SelectionPoweredPlanV40Error("v38 Fable transport boundary changed")
    if (
        deepseek_source.get("primary_response_count") != PRIMARY_TASKS
        or deepseek_source.get("repeat_response_count") != REPEAT_TASKS
        or deepseek_source.get("responses_used_as_final_deepseek_score_data") is not True
    ):
        raise SelectionPoweredPlanV40Error("v39 DeepSeek response source is incomplete")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "transport_repair_frozen_before_complete_clean_fable_recollection"
    document["inputs"]["plan_v39_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v38_fable_transport"] = {
        **dict(fable_transport),
        "eligibility_finding": (
            "the complete v38 Fable block fell below the pre-existing 608-of-640 availability floor"
        ),
        "successor_change": (
            "none: recollect all 640 primary and 64 repeat cells in a fresh namespace after "
            "the provider usage limit was restored; do not retain or selectively patch any v38 cell"
        ),
    }
    document["inputs"]["retained_v39_deepseek_response_source"] = dict(deepseek_source)
    successor = document["execution"]["frontier_refresh_successor"]
    successor.update(
        {
            "new_model_ids": list(NEW_MODEL_IDS),
            "new_provider_calls": PRIMARY_TASKS + REPEAT_TASKS,
            "retained_v38_new_model_ids": [
                model_id for model_id in NEW_MODEL_IDS if model_id not in {DEEPSEEK_ID, FABLE_ID}
            ],
            "retained_v39_new_model_ids": [DEEPSEEK_ID],
            "rerun_model_ids": [FABLE_ID],
            "v38_deepseek_responses_used_as_score_data": False,
            "v38_fable_responses_used_as_score_data": False,
            "v39_deepseek_responses_used_as_score_data": True,
            "full_fable_block_replacement": True,
            "selective_failed_cell_retry": False,
            "score_or_result_adaptive_execution_change": False,
        }
    )
    document["execution"]["reasoning_control"] = (
        "all v39 settings retained; Fable is recollected under its exact v38 route, prompt, "
        "decoding, output ceiling, reasoning setting, and concurrency"
    )
    document["budget"]["successor_scope"] = "one complete 640-primary plus 64-repeat Fable block"
    prior_v38 = int(predecessor["budget"]["prior_v38_spend_micros"])
    prior_v39 = int(deepseek_source["spend_micros"])
    prior_program = prior_v38 + prior_v39
    remaining = PROGRAM_CAP_MICROS - prior_program
    if remaining <= 0:
        raise SelectionPoweredPlanV40Error("frontier refresh exhausted its aggregate cap")
    document["budget"]["prior_v38_spend_micros"] = prior_v38
    document["budget"]["prior_v39_deepseek_spend_micros"] = prior_v39
    document["budget"]["prior_frontier_program_spend_micros"] = prior_program
    document["budget"]["superseded_fable_spend_micros"] = int(fable_transport["spend_micros"])
    document["budget"]["hard_cap"] = f"{remaining / 1_000_000:.6f}"
    document["budget"]["aggregate_program_cap"] = "200"
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV40Error("constructed v40 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = list(document["roster"]["models"])
        rows = {str(row["model_id"]): row for row in roster}
        policy_document = document["execution"]["execution_policy"]
        calibration = document["inputs"]["calibration_v38_fable_transport"]
        predecessor = document["inputs"]["plan_v39_predecessor"]
        deepseek = document["inputs"]["retained_v39_deepseek_response_source"]
        successor = document["execution"]["frontier_refresh_successor"]
        prior_v38 = int(document["budget"]["prior_v38_spend_micros"])
        prior_v39 = int(document["budget"]["prior_v39_deepseek_spend_micros"])
    except (KeyError, TypeError, ValueError):
        return False
    policy = selection_execution_policy_v31()
    fable_v39 = next(
        (row for row in roster if row.get("model_id") == FABLE_ID),
        None,
    )
    retained = set(successor.get("retained_v38_new_model_ids", []))
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == 26
        and tuple(row["model_id"] for row in roster) == FINAL_PANEL_ORDER
        and fable_v39 == rows[FABLE_ID]
        and calibration.get("primary_response_count") == PRIMARY_TASKS
        and calibration.get("repeat_response_count") == REPEAT_TASKS
        and sum(calibration.get("primary_status_counts", {}).values()) == PRIMARY_TASKS
        and sum(calibration.get("repeat_status_counts", {}).values()) == REPEAT_TASKS
        and calibration.get("primary_status_counts", {}).get("completed", PRIMARY_TASKS)
        < int(document["eligibility"]["minimum_completed_tasks"])
        and calibration.get("primary_status_counts", {}).get("failed", 0) > 0
        and calibration.get("aggregate_fable_score_was_inspected_before_repair") is True
        and calibration.get("task_level_scores_or_selections_used_to_change_execution_contract")
        is False
        and calibration.get("execution_contract_changed") is False
        and calibration.get("complete_old_fable_block_used_as_final_score_data") is False
        and deepseek.get("primary_response_count") == PRIMARY_TASKS
        and deepseek.get("repeat_response_count") == REPEAT_TASKS
        and deepseek.get("responses_used_as_final_deepseek_score_data") is True
        and successor.get("rerun_model_ids") == [FABLE_ID]
        and successor.get("new_provider_calls") == PRIMARY_TASKS + REPEAT_TASKS
        and successor.get("retained_v39_new_model_ids") == [DEEPSEEK_ID]
        and retained == set(NEW_MODEL_IDS) - {DEEPSEEK_ID, FABLE_ID}
        and successor.get("v38_fable_responses_used_as_score_data") is False
        and successor.get("v39_deepseek_responses_used_as_score_data") is True
        and successor.get("full_fable_block_replacement") is True
        and successor.get("selective_failed_cell_retry") is False
        and successor.get("score_or_result_adaptive_execution_change") is False
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["design"].get("primary_provider_calls") == 26 * PRIMARY_TASKS
        and document["design"].get("repeat_provider_calls") == 26 * REPEAT_TASKS
        and document["budget"].get("prior_frontier_program_spend_micros") == prior_v38 + prior_v39
        and document["budget"].get("hard_cap")
        == f"{(PROGRAM_CAP_MICROS - prior_v38 - prior_v39) / 1_000_000:.6f}"
        and document["budget"].get("aggregate_program_cap") == "200"
        and document["budget"].get("successor_scope")
        == "one complete 640-primary plus 64-repeat Fable block"
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV40Error("content-addressed plan conflict")
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
    parser.add_argument("--v38-run-directory", type=Path, required=True)
    parser.add_argument("--v39-run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    v38_semantic = predecessor["inputs"]["plan_v38_predecessor"]["semantic_sha256"]
    v38_plan = _load(
        args.v38_run_directory.parent
        / "plan"
        / f"epicure-selection-analysis-plan-{v38_semantic}.json"
    )
    fable_transport = _v38_fable_transport_commitment(args.v38_run_directory, plan=v38_plan)
    deepseek_source = _v39_deepseek_commitment(args.v39_run_directory, plan=predecessor)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        fable_transport=fable_transport,
        deepseek_source=deepseek_source,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
