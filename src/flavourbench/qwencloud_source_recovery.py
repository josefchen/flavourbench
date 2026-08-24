"""Reconcile a complete Qwen source rejected only by local digest canonicalization."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .qwencloud_smoke_admission import (
    CONDITIONS,
    EPICURE_MCP_URL,
    EPICURE_PROVENANCE_URL,
    MODEL_ID,
    PROVIDER_SLUG,
    QwenCloudSmokeAdmissionError,
    _live_source_sha256,
    _regular_json,
    _sha256,
    _write_content_addressed,
    load_ledger,
    terminalize_source,
    validate_ledger_state,
    verify_go_template,
    verify_human_pi_authorization,
    verify_preflight_artifact,
)
from .run_journal import load_run_journal

SCHEMA_VERSION = "flavourbench-qwencloud-zero-call-source-recovery-v1"
CONFIRMATION = "TERMINALIZE_COMPLETE_QWEN_SOURCE_WITH_ZERO_PROVIDER_CALLS_V1"


def _source_generation_ids(source: dict[str, Any]) -> set[str]:
    results = source.get("results")
    if not isinstance(results, dict) or set(results) != set(CONDITIONS):
        raise QwenCloudSmokeAdmissionError("recovery source lacks the exact two arms")
    generation_ids: set[str] = set()
    for condition in CONDITIONS:
        result = results.get(condition)
        if (
            not isinstance(result, dict)
            or result.get("actual_model_id") != MODEL_ID
            or result.get("actual_provider") != PROVIDER_SLUG
            or result.get("finish_reason") not in {"stop", "end_turn"}
            or result.get("cost_reconciled") is not False
            or result.get("billing_reconciliation_status")
            != "provider_rate_and_charge_unavailable"
        ):
            raise QwenCloudSmokeAdmissionError(
                f"recovery source arm is incomplete or identity-mismatched: {condition}"
            )
        ids = result.get("generation_ids")
        if not isinstance(ids, list) or len(ids) != 3 or any(not item for item in ids):
            raise QwenCloudSmokeAdmissionError(
                f"recovery source arm lacks three staged generation IDs: {condition}"
            )
        generation_ids.update(map(str, ids))
    if len(generation_ids) != 6:
        raise QwenCloudSmokeAdmissionError("recovery source generation IDs are duplicated")
    return generation_ids


def build_recovery(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    template = verify_go_template(
        args.go_template,
        expected_sha256=args.expected_go_template_sha256,
    )
    preflight = verify_preflight_artifact(
        args.preflight,
        expected_sha256=args.expected_preflight_sha256,
        template=template,
    )
    authorization = verify_human_pi_authorization(
        args.human_pi_authorization,
        expected_sha256=args.expected_human_pi_authorization_sha256,
        template=template,
        preflight=preflight,
    )
    source = _regular_json(args.source, label="complete QwenCloud source")
    source_sha = str(source.get("artifact_sha256") or "")
    source_body = {key: value for key, value in source.items() if key != "artifact_sha256"}
    utf8_digest = _sha256(source_body)
    if (
        source_sha != args.expected_source_sha256
        or _live_source_sha256(source_body) != source_sha
        or utf8_digest == source_sha
        or source.get("status") != "complete_unpriced_budget_ceiling"
        or source.get("run_id") != template["execution"]["frozen_run_id"]
        or source.get("candidate_manifest_sha256")
        != template["model_identity"]["route_manifest_sha256"]
        or source.get("dataset_work_item_id")
        != template["reservation"]["work_item_id"]
        or source.get("errors") != {}
        or source.get("official") is not False
        or source.get("rank_eligible") is not False
        or source.get("requested_conditions") != list(CONDITIONS)
    ):
        raise QwenCloudSmokeAdmissionError(
            "source is not a complete producer-addressed Qwen successor pair"
        )
    generation_ids = _source_generation_ids(source)
    budget = source.get("budget")
    if (
        not isinstance(budget, dict)
        or budget.get("cap_usd") != template["reservation"]["full_ceiling_usd"]
        or budget.get("provider_cost_known") is not False
        or budget.get("full_unpriced_budget_ceiling_retained") is not True
        or budget.get("retained_exposure_usd")
        != template["reservation"]["full_ceiling_usd"]
        or budget.get("all_generation_usage_accounted") is not True
    ):
        raise QwenCloudSmokeAdmissionError("source budget or usage accounting is incomplete")
    traces = source.get("mcp_trace_events")
    on_arm = f"{source['run_id']}:epicure_on"
    successful_traces = [
        trace
        for trace in traces or []
        if isinstance(trace, dict)
        and trace.get("arm_id") == on_arm
        and trace.get("is_error") is False
    ]
    if len(successful_traces) < 1:
        raise QwenCloudSmokeAdmissionError("source lacks a successful real Epicure trace")
    if source.get("epicure_transport") != {
        "mcp_url": EPICURE_MCP_URL,
        "provenance_url": EPICURE_PROVENANCE_URL,
    }:
        raise QwenCloudSmokeAdmissionError("source used a different Epicure transport")

    journal_sha = hashlib.sha256(args.journal.read_bytes()).hexdigest()
    entries = load_run_journal(args.journal)
    response_events = [
        entry
        for entry in entries
        if entry.get("event_type") == "provider_attempt"
        and isinstance(entry.get("payload"), dict)
        and entry["payload"].get("event_type") == "response_received"
    ]
    journal_generation_ids = {
        str(entry["payload"].get("generation_id") or "") for entry in response_events
    }
    finalized = entries[-1]
    final_payload = finalized.get("payload")
    if (
        journal_sha != args.expected_journal_sha256
        or args.journal.name
        != f"flavourbench-live-smoke-journal-{journal_sha}.jsonl"
        or entries[0].get("run_id") != source["run_id"]
        or journal_generation_ids != generation_ids
        or len(response_events) != 6
        or finalized.get("event_type") != "run_finalized"
        or not isinstance(final_payload, dict)
        or final_payload.get("status")
        != "generation_complete_unpriced_budget_ceiling"
        or set(final_payload.get("generation_ids") or []) != generation_ids
        or final_payload.get("error_keys") != []
        or final_payload.get("all_generation_usage_rate_card_accounted") is not True
    ):
        raise QwenCloudSmokeAdmissionError("journal does not reconstruct the complete source")

    ledger_entries = load_ledger(args.ledger)
    ledger_state = validate_ledger_state(ledger_entries)
    reservation_sha = str(template["reservation"]["entry_sha256"])
    start = ledger_state.starts.get(reservation_sha)
    incident = ledger_state.incidents.get(reservation_sha)
    if (
        start is None
        or incident is None
        or incident.get("entry_sha256") != args.expected_incident_entry_sha256
        or start.get("go_template_sha256") != template["artifact_sha256"]
        or start.get("human_pi_authorization_sha256")
        != authorization["artifact_sha256"]
        or start.get("preflight_artifact_sha256") != preflight["artifact_sha256"]
    ):
        raise QwenCloudSmokeAdmissionError("ledger incident does not bind the complete source run")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "recovery_decision": (
            "terminalize_complete_source_after_local_digest_validator_fix"
        ),
        "provider_calls_made": False,
        "epicure_calls_made": False,
        "source_mutated": False,
        "source_artifact_sha256": source_sha,
        "source_filename": args.source.name,
        "journal_sha256": journal_sha,
        "journal_filename": args.journal.name,
        "journal_head_entry_sha256": entries[-1]["entry_sha256"],
        "journal_entry_count": len(entries),
        "incident_entry_sha256": incident["entry_sha256"],
        "execution_start_entry_sha256": start["entry_sha256"],
        "reservation_entry_sha256": reservation_sha,
        "go_template_sha256": template["artifact_sha256"],
        "human_pi_authorization_sha256": authorization["artifact_sha256"],
        "preflight_artifact_sha256": preflight["artifact_sha256"],
        "run_id": source["run_id"],
        "model_identity": {
            "requested": MODEL_ID,
            "returned_models": [MODEL_ID],
            "provider": PROVIDER_SLUG,
            "identity_kind": "mutable_alias",
            "official": False,
            "rank_eligible": False,
        },
        "observed_counts": {
            "response_arms": 2,
            "provider_generation_responses": len(response_events),
            "successful_real_epicure_calls": len(successful_traces),
            "synthetic_arms": 0,
            "quality_comparisons_authorized": 0,
        },
        "generation_ids": sorted(generation_ids),
        "epicure_tool_names": [str(trace.get("name") or "") for trace in successful_traces],
        "budget": {
            "provider_cost_known": False,
            "retained_exposure_usd": template["reservation"]["full_ceiling_usd"],
            "cumulative_season_retained_exposure_usd": template["reservation"][
                "season_retained_exposure_usd"
            ],
        },
        "root_cause": {
            "producer_canonicalization": "json_sort_keys_separators_ascii_escaped",
            "rejected_validator_canonicalization": (
                "json_sort_keys_separators_utf8_unescaped"
            ),
            "producer_digest_matches_source": True,
            "prior_validator_digest_sha256": utf8_digest,
            "scientific_or_provider_predicate_failed": False,
            "local_digest_validator_only": True,
        },
        "claim_boundary": {
            "exploratory_development_only": True,
            "official": False,
            "rank_eligible": False,
            "provider_cost_reconciled": False,
            "leaderboard_comparisons_authorized": 0,
        },
    }
    path = _write_content_addressed(args.output_dir, "qwencloud-zero-call-recovery", payload)
    return path, template, authorization


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--expected-journal-sha256", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--expected-incident-entry-sha256", required=True)
    parser.add_argument("--go-template", type=Path, required=True)
    parser.add_argument("--expected-go-template-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--human-pi-authorization", type=Path, required=True)
    parser.add_argument("--expected-human-pi-authorization-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"pass --confirm {CONFIRMATION}")
    try:
        recovery_path, template, authorization = build_recovery(args)
        recovery = _regular_json(recovery_path, label="QwenCloud zero-call recovery")
        terminal = terminalize_source(
            ledger_path=args.ledger,
            template=template,
            authorization=authorization,
            artifact_path=args.source,
            recovery_artifact_path=recovery_path,
            expected_recovery_artifact_sha256=recovery["artifact_sha256"],
        )
    except (OSError, QwenCloudSmokeAdmissionError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "status": "complete_source_terminalized_after_zero_call_recovery",
                "provider_calls_made": False,
                "epicure_calls_made": False,
                "source_mutated": False,
                "recovery_artifact": str(recovery_path.resolve()),
                "recovery_artifact_sha256": recovery["artifact_sha256"],
                "source_artifact_sha256": args.expected_source_sha256,
                "terminal_entry_sha256": terminal["entry_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
