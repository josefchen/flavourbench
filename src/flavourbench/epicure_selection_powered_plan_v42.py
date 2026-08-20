"""Freeze the full Google-global Fable block after bounded route qualification."""

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
from .epicure_selection_powered_plan_v39 import DEEPSEEK_ID
from .epicure_selection_powered_plan_v40 import FABLE_ID
from .epicure_selection_powered_plan_v41 import verify_plan as verify_plan_v41
from .epicure_selection_route_manifest_v42 import SELECTED_PROVIDER, SELECTED_TAG
from .epicure_selection_route_manifest_v42 import verify_manifest as verify_manifest_v42
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26_v37 import FINAL_PANEL_ORDER

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v42"
PLAN_VERSION = "flavourbench-selection-26x640-v42"
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
PROGRAM_CAP_MICROS = 200_000_000
PROBE_RESERVATION_MICROS_PER_CALL = 860_160
ROUTE_PROBE = (
    ("amazon-bedrock", 1, 3),
    ("google-vertex/global", 3, 1),
    ("google-vertex/europe", 3, 1),
    ("amazon-bedrock/claude-on-aws", 1, 3),
)


class SelectionPoweredPlanV42Error(RuntimeError):
    """The complete Google-global Fable successor failed verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV42Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV42Error("input is not a JSON object")
    return value


def _v41_transport_commitment(
    directory: Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    slot = next(
        (str(row["slot_id"]) for row in plan["roster"]["models"] if row["model_id"] == FABLE_ID),
        None,
    )
    if slot is None:
        raise SelectionPoweredPlanV42Error("v41 Fable slot is absent")
    paths = sorted((directory / "responses" / "primary" / slot).glob("response-*.json"))
    if len(paths) != 4:
        raise SelectionPoweredPlanV42Error("v41 must contain exactly four pilot responses")
    rows: list[dict[str, Any]] = []
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
            raise SelectionPoweredPlanV42Error("v41 response failed integrity")
        rows.append(row)
    statuses = Counter(str(row["status"]) for row in rows)
    journal = directory / "attempts/provider-attempts.jsonl"
    finish_reasons: Counter[str] = Counter()
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
            raise SelectionPoweredPlanV42Error("v41 attempt journal failed integrity")
        if event.get("event_type") == "response_received":
            finish_reasons[
                str((event.get("metadata") or {}).get("finish_reason") or "missing")
            ] += 1
    return {
        "plan_sha256": plan["artifact_sha256"],
        "response_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "provider_finish_reason_counts": dict(sorted(finish_reasons.items())),
        "response_artifact_set_sha256": _sha256(
            sorted(str(row["artifact_sha256"]) for row in rows)
        ),
        "attempt_journal_physical_sha256": _sha256_file(journal),
        "settled_spend_micros": sum(
            int((row.get("generation") or {}).get("cost_micros") or 0) for row in rows
        ),
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
    if not verify_plan_v41(predecessor) or not verify_manifest_v42(manifest):
        raise SelectionPoweredPlanV42Error("v42 requires exact v41 plan and v42 manifest")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != FINAL_PANEL_ORDER:
        raise SelectionPoweredPlanV42Error("v42 manifest roster/order changed")
    if (
        transport.get("response_count") != 4
        or transport.get("status_counts") != {"completed": 1, "failed": 3}
        or transport.get("provider_finish_reason_counts") != {"content_filter": 3, "stop": 1}
        or transport.get("task_scores_or_selections_used_for_route_choice") is not False
        or transport.get("responses_used_as_final_fable_score_data") is not False
    ):
        raise SelectionPoweredPlanV42Error("v41 Azure pilot boundary changed")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "google_global_route_frozen_before_complete_fable_block"
    document["inputs"]["plan_v41_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["inputs"]["calibration_v41_fable_transport"] = dict(transport)
    document["inputs"]["bounded_fable_route_probe"] = {
        "route_order": [row[0] for row in ROUTE_PROBE],
        "cells_per_route": 4,
        "results": [
            {
                "provider_tag": tag,
                "normal_completions": completed,
                "content_filter_finishes": failed,
            }
            for tag, completed, failed in ROUTE_PROBE
        ],
        "selection_rule": (
            "maximize normal completions, then choose the first route in the frozen order"
        ),
        "selected_provider_tag": SELECTED_TAG,
        "answers_or_scores_used": False,
        "response_payloads_retained_as_score_data": False,
        "provider_calls": 16,
        "unsettled_reservation_micros": 16 * PROBE_RESERVATION_MICROS_PER_CALL,
    }
    candidate = next(value for value in candidates if value.model_id == FABLE_ID)
    roster_row = next(
        value for value in document["roster"]["models"] if value["model_id"] == FABLE_ID
    )
    roster_row.update(
        {
            "canonical_model_slug": candidate.canonical_model_slug,
            "execution_backend": candidate.execution_backend,
            "provider_tag": candidate.provider_tag,
            "provider_name": candidate.endpoint["provider_name"],
            "endpoint_sha256": candidate.endpoint_sha256,
            "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
            "backend_contract_sha256": candidate.backend_contract_sha256,
        }
    )
    successor = document["execution"]["frontier_refresh_successor"]
    successor.update(
        {
            "new_provider_calls": PRIMARY_TASKS + REPEAT_TASKS,
            "fable_selected_provider_tag": SELECTED_TAG,
            "fable_selected_provider_name": SELECTED_PROVIDER,
            "v41_fable_responses_used_as_score_data": False,
            "route_probe_responses_used_as_score_data": False,
            "full_fable_block_replacement": True,
            "selective_failed_cell_retry": False,
            "score_or_result_adaptive_execution_change": False,
        }
    )
    document["execution"]["reasoning_control"] = (
        "all v41 settings retained; only Fable's exact provider route changes to Google global"
    )
    prior = int(predecessor["budget"]["prior_frontier_program_spend_micros"])
    v41_spend = int(transport["settled_spend_micros"])
    probe_exposure = 16 * PROBE_RESERVATION_MICROS_PER_CALL
    remaining = PROGRAM_CAP_MICROS - prior - v41_spend - probe_exposure
    if remaining <= 0:
        raise SelectionPoweredPlanV42Error("frontier refresh exhausted its aggregate cap")
    document["budget"]["prior_v41_fable_spend_micros"] = v41_spend
    document["budget"]["route_probe_bounded_exposure_micros"] = probe_exposure
    document["budget"]["prior_frontier_program_spend_micros"] = prior + v41_spend + probe_exposure
    document["budget"]["hard_cap"] = f"{remaining / 1_000_000:.6f}"
    document["budget"]["successor_scope"] = (
        "one complete 640-primary plus 64-repeat Google-global Fable block"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV42Error("constructed v42 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = list(document["roster"]["models"])
        fable = next(row for row in roster if row["model_id"] == FABLE_ID)
        transport = document["inputs"]["calibration_v41_fable_transport"]
        probe = document["inputs"]["bounded_fable_route_probe"]
        successor = document["execution"]["frontier_refresh_successor"]
        policy_document = document["execution"]["execution_policy"]
        prior = int(document["budget"]["prior_frontier_program_spend_micros"])
        v41_spend = int(document["budget"]["prior_v41_fable_spend_micros"])
        probe_exposure = int(document["budget"]["route_probe_bounded_exposure_micros"])
    except (KeyError, StopIteration, TypeError, ValueError):
        return False
    policy = selection_execution_policy_v31()
    before = prior - v41_spend - probe_exposure
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == 26
        and tuple(row["model_id"] for row in roster) == FINAL_PANEL_ORDER
        and fable.get("canonical_model_slug") == "anthropic/claude-5-fable-20260609"
        and fable.get("provider_tag") == SELECTED_TAG
        and fable.get("provider_name") == SELECTED_PROVIDER
        and fable.get("final_max_output_tokens") == 2_048
        and fable.get("final_reasoning_effort") == "minimal"
        and transport.get("status_counts") == {"completed": 1, "failed": 3}
        and transport.get("provider_finish_reason_counts") == {"content_filter": 3, "stop": 1}
        and transport.get("task_scores_or_selections_used_for_route_choice") is False
        and transport.get("responses_used_as_final_fable_score_data") is False
        and probe.get("route_order") == [row[0] for row in ROUTE_PROBE]
        and probe.get("selected_provider_tag") == SELECTED_TAG
        and probe.get("answers_or_scores_used") is False
        and probe.get("response_payloads_retained_as_score_data") is False
        and probe.get("provider_calls") == 16
        and probe.get("unsettled_reservation_micros") == 16 * PROBE_RESERVATION_MICROS_PER_CALL
        and successor.get("rerun_model_ids") == [FABLE_ID]
        and successor.get("retained_v39_new_model_ids") == [DEEPSEEK_ID]
        and successor.get("fable_selected_provider_tag") == SELECTED_TAG
        and successor.get("v38_fable_responses_used_as_score_data") is False
        and successor.get("v40_fable_responses_used_as_score_data") is False
        and successor.get("v41_fable_responses_used_as_score_data") is False
        and successor.get("route_probe_responses_used_as_score_data") is False
        and successor.get("full_fable_block_replacement") is True
        and successor.get("selective_failed_cell_retry") is False
        and successor.get("new_provider_calls") == PRIMARY_TASKS + REPEAT_TASKS
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["budget"].get("hard_cap")
        == f"{(PROGRAM_CAP_MICROS - before - v41_spend - probe_exposure) / 1_000_000:.6f}"
        and document["budget"].get("aggregate_program_cap") == "200"
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV42Error("content-addressed plan conflict")
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
    parser.add_argument("--v41-run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    transport = _v41_transport_commitment(args.v41_run_directory, plan=predecessor)
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
