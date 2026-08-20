"""Freeze a complete Azure-routed Fable block after v40 transport failure."""

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
from .epicure_selection_powered_plan_v40 import FABLE_ID
from .epicure_selection_powered_plan_v40 import verify_plan as verify_plan_v40
from .epicure_selection_route_manifest_v41 import SELECTED_PROVIDER, SELECTED_TAG
from .epicure_selection_route_manifest_v41 import verify_manifest as verify_manifest_v41
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26_v37 import FINAL_PANEL_ORDER

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v41"
PLAN_VERSION = "flavourbench-selection-26x640-v41"
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
PROGRAM_CAP_MICROS = 200_000_000


class SelectionPoweredPlanV41Error(RuntimeError):
    """The clean Azure Fable successor failed verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV41Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV41Error("input is not a JSON object")
    return value


def _v40_transport_commitment(
    directory: Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    slot = next(
        (str(row["slot_id"]) for row in plan["roster"]["models"] if row["model_id"] == FABLE_ID),
        None,
    )
    if slot is None:
        raise SelectionPoweredPlanV41Error("v40 Fable slot is absent")
    paths = sorted((directory / "responses" / "primary" / slot).glob("response-*.json"))
    if not paths or len(paths) >= PRIMARY_TASKS:
        raise SelectionPoweredPlanV41Error("v40 must be a non-empty interrupted Fable block")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(row)
        recorded = str(payload.pop("artifact_sha256", ""))
        if (
            recorded != _sha256(payload)
            or row.get("plan_sha256") != plan["artifact_sha256"]
            or row.get("model_id") != FABLE_ID
            or row.get("panel") != "primary"
            or path.name != f"response-{row.get('cell_id')}-{recorded}.json"
        ):
            raise SelectionPoweredPlanV41Error("v40 response failed integrity")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in identities:
            raise SelectionPoweredPlanV41Error("v40 response identity is invalid")
        identities.add(task_id)
        rows.append(row)
    statuses = Counter(str(row.get("status")) for row in rows)
    finish_reasons: Counter[str] = Counter()
    persisted_arm_ids = {str(row["arm_id"]) for row in rows}
    received_arm_ids: set[str] = set()
    journal = directory / "attempts/provider-attempts.jsonl"
    if journal.is_symlink() or not journal.is_file():
        raise SelectionPoweredPlanV41Error("v40 attempt journal is absent")
    for line in journal.read_text(encoding="utf-8").splitlines():
        document = json.loads(line)
        payload = dict(document)
        recorded = str(payload.pop("event_sha256", ""))
        event = payload.get("event")
        if (
            recorded != _sha256(payload)
            or payload.get("plan_sha256") != plan["artifact_sha256"]
            or not isinstance(event, Mapping)
        ):
            raise SelectionPoweredPlanV41Error("v40 attempt journal failed integrity")
        if event.get("event_type") == "response_received":
            arm_id = str(event.get("arm_id") or "")
            if not arm_id or arm_id in received_arm_ids:
                raise SelectionPoweredPlanV41Error("v40 response-received identity is invalid")
            received_arm_ids.add(arm_id)
            finish_reasons[
                str((event.get("metadata") or {}).get("finish_reason") or "missing")
            ] += 1
    reservations = {int(row["budget"]["reserved_micros"]) for row in rows}
    if len(reservations) != 1 or not persisted_arm_ids <= received_arm_ids:
        raise SelectionPoweredPlanV41Error("v40 response reservation lineage is invalid")
    reservation = reservations.pop()
    artifactless = received_arm_ids - persisted_arm_ids
    settled_spend = sum(int((row.get("generation") or {}).get("cost_micros") or 0) for row in rows)
    return {
        "plan_sha256": plan["artifact_sha256"],
        "response_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "provider_finish_reason_counts": dict(sorted(finish_reasons.items())),
        "response_received_count": len(received_arm_ids),
        "artifactless_response_received_count": len(artifactless),
        "artifactless_response_received_set_sha256": _sha256(sorted(artifactless)),
        "response_artifact_set_sha256": _sha256(
            sorted(str(row["artifact_sha256"]) for row in rows)
        ),
        "attempt_journal_physical_sha256": _sha256_file(journal),
        "settled_spend_micros": settled_spend,
        "reservation_micros_per_cell": reservation,
        "unsettled_reservation_micros": len(artifactless) * reservation,
        "bounded_exposure_micros": settled_spend + len(artifactless) * reservation,
        "interrupted_before_complete_block": True,
        "task_scores_or_selections_used_for_route_choice": False,
        "responses_used_as_final_fable_score_data": False,
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    transport: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_plan_v40(predecessor):
        raise SelectionPoweredPlanV41Error("v41 requires the exact v40 predecessor")
    if not verify_manifest_v41(manifest):
        raise SelectionPoweredPlanV41Error("v41 requires the exact Azure route manifest")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != FINAL_PANEL_ORDER:
        raise SelectionPoweredPlanV41Error("v41 manifest roster/order changed")
    statuses = transport.get("status_counts") or {}
    finish_reasons = transport.get("provider_finish_reason_counts") or {}
    if (
        not 0 < int(transport.get("response_count", 0)) < PRIMARY_TASKS
        or sum(int(value) for value in statuses.values()) != transport["response_count"]
        or int(statuses.get("failed", 0)) == 0
        or int(finish_reasons.get("content_filter", 0)) == 0
        or transport.get("response_received_count")
        != transport["response_count"] + transport.get("artifactless_response_received_count", -1)
        or int(transport.get("artifactless_response_received_count", 0)) <= 0
        or transport.get("bounded_exposure_micros")
        != transport.get("settled_spend_micros", 0)
        + transport.get("unsettled_reservation_micros", 0)
        or transport.get("interrupted_before_complete_block") is not True
        or transport.get("task_scores_or_selections_used_for_route_choice") is not False
        or transport.get("responses_used_as_final_fable_score_data") is not False
    ):
        raise SelectionPoweredPlanV41Error("v40 transport boundary changed")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "azure_route_frozen_before_complete_clean_fable_recollection"
    document["inputs"]["plan_v40_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["inputs"]["calibration_v40_fable_transport"] = {
        **dict(transport),
        "transport_finding": "exact Anthropic route returned repeated content_filter finishes",
        "successor_change": (
            "switch only Fable's exact provider route from Anthropic to Azure; preserve the "
            "dated model identity, prompts, tasks, decoding, output ceiling, and concurrency"
        ),
    }
    candidate_by_id = {candidate.model_id: candidate for candidate in candidates}
    fable = candidate_by_id[FABLE_ID]
    rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
    rows[FABLE_ID].update(
        {
            "canonical_model_slug": fable.canonical_model_slug,
            "execution_backend": fable.execution_backend,
            "provider_tag": fable.provider_tag,
            "provider_name": fable.endpoint["provider_name"],
            "endpoint_sha256": fable.endpoint_sha256,
            "endpoint_execution_sha256": fable.endpoint_execution_sha256,
            "backend_contract_sha256": fable.backend_contract_sha256,
        }
    )
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
            "fable_selected_provider_tag": SELECTED_TAG,
            "fable_selected_provider_name": SELECTED_PROVIDER,
            "v38_fable_responses_used_as_score_data": False,
            "v40_fable_responses_used_as_score_data": False,
            "full_fable_block_replacement": True,
            "selective_failed_cell_retry": False,
            "score_or_result_adaptive_execution_change": False,
        }
    )
    document["execution"]["reasoning_control"] = (
        "all v40 settings retained; only Fable's exact provider route changes to Azure"
    )
    prior_program = int(predecessor["budget"]["prior_frontier_program_spend_micros"])
    v40_exposure = int(transport["bounded_exposure_micros"])
    remaining = PROGRAM_CAP_MICROS - prior_program - v40_exposure
    if remaining <= 0:
        raise SelectionPoweredPlanV41Error("frontier refresh exhausted its aggregate cap")
    document["budget"]["prior_v40_fable_settled_spend_micros"] = int(
        transport["settled_spend_micros"]
    )
    document["budget"]["prior_v40_fable_unsettled_reservation_micros"] = int(
        transport["unsettled_reservation_micros"]
    )
    document["budget"]["prior_v40_fable_bounded_exposure_micros"] = v40_exposure
    document["budget"]["prior_frontier_program_spend_micros"] = prior_program + v40_exposure
    document["budget"]["hard_cap"] = f"{remaining / 1_000_000:.6f}"
    document["budget"]["successor_scope"] = (
        "one complete 640-primary plus 64-repeat Azure-routed Fable block"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV41Error("constructed v41 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = list(document["roster"]["models"])
        rows = {str(row["model_id"]): row for row in roster}
        policy_document = document["execution"]["execution_policy"]
        transport = document["inputs"]["calibration_v40_fable_transport"]
        predecessor = document["inputs"]["plan_v40_predecessor"]
        successor = document["execution"]["frontier_refresh_successor"]
        prior_program = int(document["budget"]["prior_frontier_program_spend_micros"])
        prior_v40 = int(document["budget"]["prior_v40_fable_bounded_exposure_micros"])
    except (KeyError, TypeError, ValueError):
        return False
    policy = selection_execution_policy_v31()
    fable = rows.get(FABLE_ID, {})
    prior_before_v40 = prior_program - prior_v40
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == 26
        and tuple(row["model_id"] for row in roster) == FINAL_PANEL_ORDER
        and fable.get("canonical_model_slug") == "anthropic/claude-5-fable-20260609"
        and fable.get("execution_backend") == "openrouter"
        and fable.get("provider_tag") == SELECTED_TAG
        and fable.get("provider_name") == SELECTED_PROVIDER
        and fable.get("final_max_output_tokens") == 2_048
        and fable.get("final_reasoning_effort") == "minimal"
        and 0 < transport.get("response_count", 0) < PRIMARY_TASKS
        and sum(transport.get("status_counts", {}).values()) == transport.get("response_count")
        and transport.get("status_counts", {}).get("failed", 0) > 0
        and transport.get("provider_finish_reason_counts", {}).get("content_filter", 0) > 0
        and transport.get("response_received_count")
        == transport.get("response_count") + transport.get("artifactless_response_received_count")
        and transport.get("artifactless_response_received_count", 0) > 0
        and transport.get("bounded_exposure_micros")
        == transport.get("settled_spend_micros") + transport.get("unsettled_reservation_micros")
        and transport.get("unsettled_reservation_micros")
        == transport.get("artifactless_response_received_count")
        * transport.get("reservation_micros_per_cell")
        and transport.get("interrupted_before_complete_block") is True
        and transport.get("task_scores_or_selections_used_for_route_choice") is False
        and transport.get("responses_used_as_final_fable_score_data") is False
        and successor.get("rerun_model_ids") == [FABLE_ID]
        and successor.get("retained_v39_new_model_ids") == [DEEPSEEK_ID]
        and set(successor.get("retained_v38_new_model_ids", []))
        == set(NEW_MODEL_IDS) - {DEEPSEEK_ID, FABLE_ID}
        and successor.get("fable_selected_provider_tag") == SELECTED_TAG
        and successor.get("v38_fable_responses_used_as_score_data") is False
        and successor.get("v40_fable_responses_used_as_score_data") is False
        and successor.get("full_fable_block_replacement") is True
        and successor.get("selective_failed_cell_retry") is False
        and successor.get("score_or_result_adaptive_execution_change") is False
        and successor.get("new_provider_calls") == PRIMARY_TASKS + REPEAT_TASKS
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["budget"].get("hard_cap")
        == f"{(PROGRAM_CAP_MICROS - prior_before_v40 - prior_v40) / 1_000_000:.6f}"
        and document["budget"].get("aggregate_program_cap") == "200"
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV41Error("content-addressed plan conflict")
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
    parser.add_argument("--v40-run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    transport = _v40_transport_commitment(args.v40_run_directory, plan=predecessor)
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
