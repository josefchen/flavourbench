"""Freeze and verify the prospective public-source task-validation campaign.

The campaign uses exact, licensed human questions.  It makes no model, Epicure,
provider, or source-network calls.  Two admitted culinary validators assess each
prompt independently.  A third person adjudicates only a disagreement.  Rights
and contamination checks are audited once over the sealed bank and its reserve,
instead of assigning two extra reviewers to every task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .prospective_task_acquisition import (
    ASSIGNMENT_SCHEMA,
    BUNDLE_SCHEMA,
    RECEIPT_SCHEMA,
    canonical_json_bytes,
    canonical_sha256,
    verify_artifact,
)

CAMPAIGN_SCHEMA = "flavourbench-public-source-task-validation-campaign-v1"
QUALITY_REPORT_SCHEMA = "flavourbench-public-source-task-campaign-quality-report-v1"
READINESS_SCHEMA = "flavourbench-public-source-task-campaign-readiness-v1"
LEDGER_EVENT_SCHEMA = "flavourbench-task-validation-ledger-event-v1"
ZERO_SHA256 = "0" * 64
FAMILIES = ("substitution", "composition", "cookability", "evidence")
VALID_DECISIONS = frozenset({"valid", "revise", "exclude"})
REQUIRED_CHECKS = (
    "construct_fit",
    "context_complete",
    "coherent_question",
    "general_track_scope",
    "answer_leakage_absent",
    "discrimination_value",
)


class TaskValidationCampaignError(ValueError):
    """A campaign artifact or append-only event failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskValidationCampaignError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise TaskValidationCampaignError(f"artifact must be a JSON object: {path.name}")
    return value


def _verify_embedded_digest(document: Mapping[str, Any], schema: str) -> None:
    try:
        verify_artifact(document, schema_version=schema)
    except ValueError as exc:
        raise TaskValidationCampaignError(str(exc)) from exc


def _content_address(document: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    document["artifact_sha256"] = canonical_sha256(payload)
    return document


def _write_content_addressed(document: Mapping[str, Any], output_dir: Path, prefix: str) -> Path:
    schema = str(document.get("schema_version", ""))
    _verify_embedded_digest(document, schema)
    digest = str(document["artifact_sha256"])
    destination = output_dir / f"{prefix}-{digest}.json"
    rendered = canonical_json_bytes(document) + b"\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise TaskValidationCampaignError("content-addressed path contains different bytes")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def load_inputs(
    *, bundle_path: Path, assignment_path: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = _load_json(bundle_path)
    assignment = _load_json(assignment_path)
    receipt = _load_json(receipt_path)
    _verify_embedded_digest(bundle, BUNDLE_SCHEMA)
    _verify_embedded_digest(assignment, ASSIGNMENT_SCHEMA)
    _verify_embedded_digest(receipt, RECEIPT_SCHEMA)
    if assignment.get("source_candidate_bundle_sha256") != bundle.get("artifact_sha256"):
        raise TaskValidationCampaignError("assignment points to another candidate bundle")
    if receipt.get("candidate_bundle_sha256") != bundle.get("artifact_sha256"):
        raise TaskValidationCampaignError("acquisition receipt points to another candidate bundle")
    if receipt.get("assignment_artifact_sha256") != assignment.get("artifact_sha256"):
        raise TaskValidationCampaignError("acquisition receipt points to another assignment")
    if any(int(receipt.get(key, -1)) != 0 for key in ("model_calls", "epicure_calls")):
        raise TaskValidationCampaignError("task acquisition contains a model or Epicure call")
    if int(receipt.get("answer_endpoint_requests", -1)) != 0:
        raise TaskValidationCampaignError("task acquisition requested source answers")
    return bundle, assignment, receipt


def _validate_schedule(assignment: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = assignment.get("assignment_rows")
    if not isinstance(rows, list) or len(rows) != 180:
        raise TaskValidationCampaignError("campaign requires the frozen 180-candidate slate")
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    family_counts: Counter[str] = Counter()
    validated: list[dict[str, Any]] = []
    family_ordinals: Counter[str] = Counter()
    for expected_ordinal, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise TaskValidationCampaignError("assignment row is not an object")
        candidate_id = str(raw.get("candidate_id", ""))
        prompt = str(raw.get("prompt", ""))
        prompt_sha256 = str(raw.get("prompt_sha256", ""))
        family = str(raw.get("allocation_family_hidden_from_blind_reviewer", ""))
        if int(raw.get("assignment_ordinal", -1)) != expected_ordinal:
            raise TaskValidationCampaignError("candidate schedule is not contiguous")
        if candidate_id in ids or prompt_sha256 in prompt_hashes:
            raise TaskValidationCampaignError("candidate slate contains a duplicate")
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha256:
            raise TaskValidationCampaignError("prompt hash differs from prompt text")
        if family not in FAMILIES:
            raise TaskValidationCampaignError("candidate has an unknown scheduling family")
        source = raw.get("source_metadata_visible_after_blind_decision")
        if not isinstance(source, dict):
            raise TaskValidationCampaignError("candidate lacks source provenance")
        effective_license = source.get("content_license") or source.get(
            "revision_content_license"
        )
        if effective_license != "CC BY-SA 4.0":
            raise TaskValidationCampaignError("candidate lacks an accepted effective licence")
        attribution = source.get("attribution")
        if not isinstance(attribution, dict) or not attribution.get("profile_url"):
            raise TaskValidationCampaignError("candidate lacks source attribution")
        if source.get("source_answer_payload_requested") is not False:
            raise TaskValidationCampaignError("candidate source answer was requested")
        ids.add(candidate_id)
        prompt_hashes.add(prompt_sha256)
        family_counts[family] += 1
        family_ordinals[family] += 1
        validated.append(
            {
                "schedule_ordinal": expected_ordinal,
                "family_activation_ordinal": family_ordinals[family],
                "candidate_id": candidate_id,
                "prompt_sha256": prompt_sha256,
                "scheduling_family": family,
                "source_question_id": int(source["question_id"]),
                "source_revision_guid": str(source["revision_guid"]),
                "source_record_sha256": canonical_sha256(source),
                "rank_eligible": False,
            }
        )
    if family_counts != Counter({family: 45 for family in FAMILIES}):
        raise TaskValidationCampaignError("candidate slate is not balanced at 45 per family")
    return validated


def build_quality_report(
    *,
    bundle: Mapping[str, Any],
    assignment: Mapping[str, Any],
    receipt: Mapping[str, Any],
    physical_hashes: Mapping[str, str],
) -> dict[str, Any]:
    rows = assignment["assignment_rows"]
    authors: Counter[tuple[object, object]] = Counter()
    lengths: list[int] = []
    created: list[str] = []
    heuristic_patterns = {
        "direct_url_present": re.compile(r"https?://", re.IGNORECASE),
        "visual_or_video_reference": re.compile(
            r"\b(?:video|photo|picture|pictured|image|screenshot|link below|kindly see)\b",
            re.IGNORECASE,
        ),
        "specialist_safety_term": re.compile(
            r"\b(?:safe|safety|dangerous|danger|fire|explosion|poison|toxic|"
            r"botul|salmonella|spoil|mould|mold|undercook|canning)\b",
            re.IGNORECASE,
        ),
        "health_or_diet_term": re.compile(
            r"\b(?:doctor|medical|health|diet|intoleran|allerg|bladder|celiac|"
            r"coeliac|illness|disease|pregnan|diabet)\b",
            re.IGNORECASE,
        ),
        "possible_self_resolution_marker": re.compile(
            r"\b(?:update|edit|solved|figured it out|turns out|what worked)\b",
            re.IGNORECASE,
        ),
    }
    heuristic_hits: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_question_api_license: list[dict[str, object]] = []
    for row in rows:
        source = row["source_metadata_visible_after_blind_decision"]
        author = source["attribution"]
        authors[(author.get("source_user_id"), author.get("profile_url"))] += 1
        prompt = str(row["prompt"])
        lengths.append(len(prompt))
        created.append(str(source["created_utc"]))
        for check_id, pattern in heuristic_patterns.items():
            if pattern.search(prompt):
                heuristic_hits[check_id].append(
                    {
                        "candidate_id": str(row["candidate_id"]),
                        "prompt_sha256": str(row["prompt_sha256"]),
                    }
                )
        if source.get("question_content_license") is None:
            missing_question_api_license.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "source_question_id": int(source["question_id"]),
                    "effective_license": str(source.get("content_license")),
                    "terminal_revision_license": str(source.get("revision_content_license")),
                }
            )
    sorted_lengths = sorted(lengths)
    report = {
        "schema_version": QUALITY_REPORT_SCHEMA,
        "source_artifacts": {
            "candidate_bundle_sha256": bundle["artifact_sha256"],
            "review_assignment_sha256": assignment["artifact_sha256"],
            "acquisition_receipt_sha256": receipt["artifact_sha256"],
            "physical_file_sha256": dict(sorted(physical_hashes.items())),
        },
        "dataset_and_grain": {
            "candidate_bundle_records": int(bundle["counts"]["candidate_records"]),
            "provisional_pass_records": int(bundle["counts"]["provisional_screen_pass"]),
            "scheduled_candidate_records": len(rows),
            "scheduled_per_family": {
                family: sum(
                    row["allocation_family_hidden_from_blind_reviewer"] == family
                    for row in rows
                )
                for family in FAMILIES
            },
            "target_validated_tasks": 120,
            "target_per_family": 30,
            "reserve_candidates": 60,
        },
        "checks": {
            "unique_candidate_ids": len({row["candidate_id"] for row in rows}),
            "unique_prompt_hashes": len({row["prompt_sha256"] for row in rows}),
            "unique_source_questions": len(
                {
                    row["source_metadata_visible_after_blind_decision"]["question_id"]
                    for row in rows
                }
            ),
            "unique_attributed_authors": len(authors),
            "maximum_candidates_per_author": max(authors.values()),
            "prompt_characters": {
                "minimum": sorted_lengths[0],
                "median": (
                    sorted_lengths[len(sorted_lengths) // 2 - 1]
                    + sorted_lengths[len(sorted_lengths) // 2]
                )
                / 2,
                "maximum": sorted_lengths[-1],
            },
            "source_created_utc": {"minimum": min(created), "maximum": max(created)},
            "source_answer_payloads_requested": 0,
            "synthetic_tasks": 0,
            "model_calls": int(receipt["model_calls"]),
            "epicure_calls": int(receipt["epicure_calls"]),
        },
        "manual_review_triggers": {
            check_id: {"count": len(hits), "records": hits}
            for check_id, hits in sorted(heuristic_hits.items())
        },
        "license_field_anomalies": {
            "question_api_license_null_count": len(missing_question_api_license),
            "records": missing_question_api_license,
            "treatment": (
                "Retain only for review because the terminal revision and effective source "
                "licence are CC BY-SA 4.0; include it in the mandatory human rights sample."
            ),
        },
        "assessment": {
            "candidate_provenance": "ready_for_blind_human_validation",
            "task_validity": "not_yet_established",
            "official_task_bank": False,
            "rank_eligible": False,
            "contamination_claim": "public_source_contamination_risk_cannot_be_eliminated",
            "automated_trigger_interpretation": "review_priority_not_human_ground_truth",
        },
    }
    return _content_address(report)


def build_campaign_packet(
    *,
    bundle: Mapping[str, Any],
    assignment: Mapping[str, Any],
    receipt: Mapping[str, Any],
    physical_hashes: Mapping[str, str],
) -> dict[str, Any]:
    schedule = _validate_schedule(assignment)
    packet = {
        "schema_version": CAMPAIGN_SCHEMA,
        "status": "frozen_instrument_no_human_ballots",
        "source_artifacts": {
            "candidate_bundle_sha256": bundle["artifact_sha256"],
            "review_assignment_sha256": assignment["artifact_sha256"],
            "acquisition_receipt_sha256": receipt["artifact_sha256"],
            "physical_file_sha256": dict(sorted(physical_hashes.items())),
        },
        "target": {
            "validated_tasks": 120,
            "validated_tasks_per_family": 30,
            "candidate_slate": 180,
            "candidate_slate_per_scheduling_family": 45,
            "reserve_candidates": 60,
            "activation": (
                "fixed schedule order; stop only when human-confirmed quotas reach 30 in "
                "each family or the 180-candidate slate is exhausted"
            ),
            "effect_based_or_model_result_based_stopping": False,
        },
        "validation_protocol": {
            "source_role": (
                "exact attributed public human question under CC BY-SA 4.0; the source author "
                "is not represented as an enrolled reviewer"
            ),
            "blind_validators_per_candidate": 2,
            "validator_blinding": [
                "other-validator-ballot",
                "source-answer-text",
                "source-metadata-until-ballot-sealed",
                "model-output",
                "model-and-Epicure-condition",
                "provisional-scheduling-family",
            ],
            "ballot_checks": list(REQUIRED_CHECKS),
            "ballot_outputs": [
                "decision",
                "human-confirmed-family",
                "construct-cell",
                "difficulty-tier",
                "independent-solution-outline",
                "success-criteria",
                "permitted-variations",
                "disqualifying-errors",
                "objective-surface-checks-when-valid",
            ],
            "consensus_rule": (
                "both ballots valid; all checks pass; family, construct-cell, and difficulty "
                "agree; both validators countersign the deterministic union criterion pack"
            ),
            "adjudication_trigger": (
                "decision or label disagreement, any failed check, or refusal to countersign "
                "the merged criterion pack"
            ),
            "adjudicator_per_disagreement": 1,
            "adjudicator_independence": (
                "different season person commitment from both validators; qualified for the "
                "resolved family; no model output visible"
            ),
            "unanimous_candidate_adjudication": False,
            "recording": "append-only hash-chained events with idempotent event identifiers",
        },
        "batch_audits": {
            "rights_and_attribution": {
                "automated_coverage": "100_percent_of_sealed_bank_and_reserve",
                "human_sample": "24 seed-committed records, six per family, plus every anomaly",
                "auditor": "one privately verified public-pseudonymous person",
            },
            "contamination": {
                "automated_coverage": (
                    "100_percent_exact_fuzzy_ngram_semantic_and_captured_web_replay"
                ),
                "human_sample": "24 seed-committed records, six per family, plus every hit",
                "auditor": "one privately verified public-pseudonymous person",
                "claim_boundary": "contamination-limited; never contamination-free",
            },
            "coi": (
                "the two batch auditors are different people and neither validated or "
                "adjudicated a task in the audited campaign"
            ),
        },
        "identity_and_privacy": {
            "private_identity_verification": True,
            "season_scoped_person_commitment": True,
            "raw_identity_in_ballot_ledger": False,
            "public_identity": "stable campaign pseudonym only",
            "one_person_cannot_fill_both_validator_slots_on_one_candidate": True,
            "source_author_attribution_retained_for_license": True,
            "source_author_is_not_a_reviewer_identity": True,
            "no_placeholder_or_machine_generated_reviewers": True,
        },
        "minimum_human_workload": {
            "best_case_blind_ballots": 240,
            "full_slate_blind_ballots": 360,
            "best_case_pack_countersignatures": 240,
            "full_slate_pack_countersignatures": 360,
            "adjudications": {"minimum": 0, "maximum": 180},
            "batch_audit_signoffs": 2,
            "best_case_human_actions": 482,
            "maximum_actions_without_revision_rounds": 902,
            "minimum_distinct_people_with_full_cross_task_reuse": 5,
            "recommended_distinct_people": 8,
            "planning_hours_not_observed": {
                "assumptions": (
                    "8 to 12 minutes per blind ballot, 1 to 3 minutes per pack "
                    "countersignature, 8 to 15 minutes per adjudication, and 4 to 8 hours "
                    "for each batch audit"
                ),
                "best_case_range": [44, 76],
                "full_slate_without_adjudication_range": [62, 106],
                "full_slate_with_180_adjudications_range": [86, 151],
            },
        },
        "literature_basis": [
            {
                "work": "GPQA",
                "url": "https://arxiv.org/abs/2311.12022",
                "adopted_principle": (
                    "domain-matched independent validation and sealed revision history"
                ),
            },
            {
                "work": "Dynabench",
                "url": "https://arxiv.org/abs/2104.14337",
                "adopted_principle": "human-authored examples checked by another person",
            },
            {
                "work": "LiveBench",
                "url": "https://arxiv.org/abs/2406.19314",
                "adopted_principle": "dated source provenance and explicit contamination limits",
            },
            {
                "work": "Datasheets for Datasets",
                "url": "https://arxiv.org/abs/1803.09010",
                "adopted_principle": (
                    "document collection, composition, intended use, and limitations"
                ),
            },
        ],
        "candidate_schedule": schedule,
        "observations": {
            "human_ballots": 0,
            "adjudications": 0,
            "batch_audits": 0,
            "model_calls": 0,
            "epicure_calls": 0,
            "synthetic_tasks": 0,
        },
        "claim_boundary": {
            "instrument_is_validation_evidence": False,
            "human_validated_tasks": 0,
            "official_task_bank": False,
            "rank_eligible": False,
            "contamination_free": False,
            "database_import_authorized": False,
            "service_adapter_required_before_live_ballots": True,
        },
    }
    return _content_address(packet)


def verify_campaign_packet(packet: Mapping[str, Any]) -> None:
    _verify_embedded_digest(packet, CAMPAIGN_SCHEMA)
    target = packet.get("target")
    protocol = packet.get("validation_protocol")
    workload = packet.get("minimum_human_workload")
    boundary = packet.get("claim_boundary")
    schedule = packet.get("candidate_schedule")
    if not all(isinstance(value, Mapping) for value in (target, protocol, workload, boundary)):
        raise TaskValidationCampaignError("campaign packet is incomplete")
    if not isinstance(schedule, list) or len(schedule) != 180:
        raise TaskValidationCampaignError("campaign schedule is incomplete")
    assert isinstance(target, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(workload, Mapping)
    assert isinstance(boundary, Mapping)
    if int(target.get("validated_tasks", 0)) != 120:
        raise TaskValidationCampaignError("campaign target drifted")
    if int(protocol.get("blind_validators_per_candidate", 0)) != 2:
        raise TaskValidationCampaignError("campaign does not require two validators")
    if protocol.get("unanimous_candidate_adjudication") is not False:
        raise TaskValidationCampaignError("unanimous candidates must not require adjudication")
    if int(workload.get("minimum_distinct_people_with_full_cross_task_reuse", 0)) != 5:
        raise TaskValidationCampaignError("campaign role separation drifted")
    if boundary.get("contamination_free") is not False:
        raise TaskValidationCampaignError("public-source campaign cannot claim contamination-free")
    if boundary.get("rank_eligible") is not False:
        raise TaskValidationCampaignError("empty campaign cannot be rank eligible")


def build_readiness_decision(
    *, campaign: Mapping[str, Any], quality_report: Mapping[str, Any]
) -> dict[str, Any]:
    verify_campaign_packet(campaign)
    _verify_embedded_digest(quality_report, QUALITY_REPORT_SCHEMA)
    decision = {
        "schema_version": READINESS_SCHEMA,
        "bound_artifacts": {
            "campaign_sha256": campaign["artifact_sha256"],
            "quality_report_sha256": quality_report["artifact_sha256"],
        },
        "assessment_type": "technical_readiness_not_human_release_authorization",
        "gates": [
            {
                "gate": "source_acquisition_and_provenance",
                "decision": "go",
                "basis": (
                    "exact attributed public questions, terminal revisions, zero source-answer "
                    "requests, and zero model or Epicure calls"
                ),
            },
            {
                "gate": "frozen_validation_instrument",
                "decision": "go",
                "basis": "balanced fixed slate, target, reserve, workload, and claim limits",
            },
            {
                "gate": "reviewer_enrollment",
                "decision": "conditional_go",
                "basis": (
                    "real privately verified people only, after the service adapter passes "
                    "identity, qualification, conflict, and concurrency tests"
                ),
            },
            {
                "gate": "live_ballot_collection",
                "decision": "no_go",
                "basis": "the deployed API still implements the superseded per-task role law",
            },
            {
                "gate": "official_task_bank",
                "decision": "no_go",
                "basis": "zero human ballots and zero batch audits have been recorded",
            },
            {
                "gate": "contamination_free_claim",
                "decision": "permanent_no_go",
                "basis": "the questions are drawn from a public source",
            },
            {
                "gate": "model_generation_and_ranking",
                "decision": "no_go",
                "basis": "no admitted task bank exists and the campaign authorizes no model call",
            },
        ],
        "overall": "conditional_go_for_no_generation_campaign_after_adapter_review",
        "remaining_blockers": [
            "runtime adapter for blind ballots and pack confirmations",
            "disagreement-only adjudication queue",
            "campaign-level rights and contamination audit queues",
            "120 admitted tasks with 30 per human-confirmed family",
            "synchronized study design, bank importer, paper, and public status page",
        ],
        "claim_boundary": {
            "human_ballots": 0,
            "official": False,
            "rank_eligible": False,
            "model_or_epicure_calls_authorized": False,
            "human_release_authority_exercised": False,
        },
    }
    return _content_address(decision)


def make_ledger_event(
    *,
    campaign_sha256: str,
    sequence: int,
    event_id: str,
    event_type: Literal[
        "blind_ballot",
        "criterion_pack_confirmation",
        "adjudication",
        "rights_batch_audit",
        "contamination_batch_audit",
    ],
    reviewer_pseudonym: str,
    person_commitment_sha256: str,
    reviewer_admission_receipt_sha256: str,
    payload: Mapping[str, Any],
    previous_event_sha256: str,
) -> dict[str, Any]:
    event = {
        "schema_version": LEDGER_EVENT_SCHEMA,
        "campaign_sha256": campaign_sha256,
        "sequence": sequence,
        "event_id": event_id,
        "event_type": event_type,
        "reviewer_pseudonym": reviewer_pseudonym,
        "person_commitment_sha256": person_commitment_sha256,
        "reviewer_admission_receipt_sha256": reviewer_admission_receipt_sha256,
        "payload": dict(payload),
        "previous_event_sha256": previous_event_sha256,
    }
    event["event_sha256"] = canonical_sha256(event)
    return event


def verify_event_chain(
    events: Sequence[Mapping[str, Any]], *, campaign_sha256: str
) -> list[dict[str, Any]]:
    prior = ZERO_SHA256
    event_ids: set[str] = set()
    verified: list[dict[str, Any]] = []
    for expected_sequence, raw in enumerate(events, start=1):
        event = dict(raw)
        embedded = event.pop("event_sha256", None)
        if event.get("schema_version") != LEDGER_EVENT_SCHEMA:
            raise TaskValidationCampaignError("ledger event schema drifted")
        if event.get("campaign_sha256") != campaign_sha256:
            raise TaskValidationCampaignError("ledger event points to another campaign")
        if int(event.get("sequence", 0)) != expected_sequence:
            raise TaskValidationCampaignError("ledger sequence is not contiguous")
        if event.get("previous_event_sha256") != prior:
            raise TaskValidationCampaignError("ledger predecessor hash differs")
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in event_ids:
            raise TaskValidationCampaignError("ledger event identifier is missing or reused")
        person = str(event.get("person_commitment_sha256", ""))
        admission = str(event.get("reviewer_admission_receipt_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", person) or not re.fullmatch(
            r"[0-9a-f]{64}", admission
        ):
            raise TaskValidationCampaignError("ledger event lacks identity admission binding")
        calculated = canonical_sha256(event)
        if embedded != calculated:
            raise TaskValidationCampaignError("ledger event digest differs")
        prior = calculated
        event_ids.add(event_id)
        verified.append({**event, "event_sha256": calculated})
    return verified


def _ballot_is_valid(ballot: Mapping[str, Any]) -> bool:
    checks = ballot.get("checks")
    return bool(
        ballot.get("decision") == "valid"
        and isinstance(checks, Mapping)
        and all(checks.get(check) is True for check in REQUIRED_CHECKS)
        and ballot.get("source_metadata_seen") is False
        and ballot.get("other_ballot_seen") is False
        and ballot.get("model_outputs_seen") is False
    )


def _merged_pack(ballots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "success_criteria",
        "permitted_variations",
        "disqualifying_errors",
        "objective_checks",
    )
    merged: dict[str, Any] = {}
    for field in fields:
        values: set[str] = set()
        for ballot in ballots:
            raw_values = ballot.get(field, [])
            if not isinstance(raw_values, list):
                raise TaskValidationCampaignError(f"ballot {field} must be a list")
            for value in raw_values:
                normalized = " ".join(str(value).split())
                if normalized:
                    values.add(normalized)
        merged[field] = sorted(values)
    return merged


def merged_criterion_pack(
    *, candidate_id: str, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the deterministic two-ballot criterion pack for one candidate.

    The runtime uses this public helper after both blind ballots are sealed.  It
    deliberately returns no reviewer identity or scheduling-family metadata.
    """

    ballots = [
        event["payload"]
        for event in events
        if event.get("event_type") == "blind_ballot"
        and event.get("payload", {}).get("candidate_id") == candidate_id
    ]
    if len(ballots) != 2:
        raise TaskValidationCampaignError(
            "a merged criterion pack requires exactly two blind ballots"
        )
    return _merged_pack(ballots)


def derive_candidate_state(
    *, candidate_id: str, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    candidate_events = [
        event for event in events if event.get("payload", {}).get("candidate_id") == candidate_id
    ]
    ballots = [event for event in candidate_events if event["event_type"] == "blind_ballot"]
    ballot_people = {str(event["person_commitment_sha256"]) for event in ballots}
    if len(ballots) > 2 or len(ballot_people) != len(ballots):
        raise TaskValidationCampaignError("candidate ballot slots violate person uniqueness")
    if len(ballots) < 2:
        return {"status": "awaiting_blind_ballots", "ballots": len(ballots)}
    payloads = [event["payload"] for event in ballots]
    labels = {
        (
            payload.get("family"),
            payload.get("construct_cell_id"),
            payload.get("difficulty_tier"),
        )
        for payload in payloads
    }
    consensus_possible = all(_ballot_is_valid(payload) for payload in payloads) and len(labels) == 1
    merged_pack = _merged_pack(payloads)
    merged_pack_sha256 = canonical_sha256(merged_pack)
    confirmations = [
        event
        for event in candidate_events
        if event["event_type"] == "criterion_pack_confirmation"
    ]
    confirmation_by_person = {
        str(event["person_commitment_sha256"]): event for event in confirmations
    }
    if len(confirmation_by_person) != len(confirmations) or any(
        person not in ballot_people for person in confirmation_by_person
    ):
        raise TaskValidationCampaignError("criterion pack confirmation has an invalid reviewer")
    confirmations_pass = bool(
        len(confirmations) == 2
        and all(
            event["payload"].get("accepted") is True
            and event["payload"].get("criterion_pack_sha256") == merged_pack_sha256
            for event in confirmations
        )
    )
    adjudications = [
        event for event in candidate_events if event["event_type"] == "adjudication"
    ]
    if len(adjudications) > 1:
        raise TaskValidationCampaignError("candidate has duplicate adjudications")
    confirmation_refused = any(
        event["payload"].get("accepted") is not True
        or event["payload"].get("criterion_pack_sha256") != merged_pack_sha256
        for event in confirmations
    )
    disagreement = not consensus_possible or confirmation_refused
    if not disagreement and confirmations_pass:
        family, construct_cell, difficulty = next(iter(labels))
        return {
            "status": "validated_consensus",
            "family": family,
            "construct_cell_id": construct_cell,
            "difficulty_tier": difficulty,
            "criterion_pack_sha256": merged_pack_sha256,
            "adjudication_required": False,
        }
    if not disagreement:
        return {
            "status": "awaiting_criterion_pack_confirmations",
            "criterion_pack_sha256": merged_pack_sha256,
            "adjudication_required": False,
        }
    if not adjudications:
        return {"status": "awaiting_adjudication", "adjudication_required": True}
    adjudication = adjudications[0]
    if str(adjudication["person_commitment_sha256"]) in ballot_people:
        raise TaskValidationCampaignError("a candidate validator adjudicated the same candidate")
    payload = adjudication["payload"]
    if payload.get("model_outputs_seen") is not False:
        raise TaskValidationCampaignError("adjudicator was not model-output blind")
    decision = str(payload.get("decision", ""))
    if decision not in {"approve", "revise", "reject"}:
        raise TaskValidationCampaignError("adjudication decision is invalid")
    if decision != "approve":
        return {"status": f"adjudicated_{decision}", "adjudication_required": True}
    return {
        "status": "validated_adjudicated",
        "family": payload.get("family"),
        "construct_cell_id": payload.get("construct_cell_id"),
        "difficulty_tier": payload.get("difficulty_tier"),
        "criterion_pack_sha256": payload.get("criterion_pack_sha256"),
        "adjudication_required": True,
    }


def public_event_view(event: Mapping[str, Any]) -> dict[str, Any]:
    """Remove private linkage while retaining a stable public reviewer pseudonym."""

    return {
        key: value
        for key, value in event.items()
        if key
        not in {
            "person_commitment_sha256",
            "reviewer_admission_receipt_sha256",
        }
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/season1/task-validation-campaign-v6"),
    )
    args = parser.parse_args(argv)
    bundle, assignment, receipt = load_inputs(
        bundle_path=args.bundle,
        assignment_path=args.assignment,
        receipt_path=args.receipt,
    )
    physical_hashes = {
        "candidate_bundle": _sha256_file(args.bundle),
        "review_assignment": _sha256_file(args.assignment),
        "acquisition_receipt": _sha256_file(args.receipt),
    }
    packet = build_campaign_packet(
        bundle=bundle,
        assignment=assignment,
        receipt=receipt,
        physical_hashes=physical_hashes,
    )
    report = build_quality_report(
        bundle=bundle,
        assignment=assignment,
        receipt=receipt,
        physical_hashes=physical_hashes,
    )
    verify_campaign_packet(packet)
    readiness = build_readiness_decision(campaign=packet, quality_report=report)
    packet_path = _write_content_addressed(packet, args.output_dir, "campaign")
    report_path = _write_content_addressed(report, args.output_dir, "quality-report")
    readiness_path = _write_content_addressed(readiness, args.output_dir, "readiness")
    print(
        json.dumps(
            {
                "campaign": str(packet_path),
                "campaign_sha256": packet["artifact_sha256"],
                "quality_report": str(report_path),
                "quality_report_sha256": report["artifact_sha256"],
                "readiness": str(readiness_path),
                "readiness_sha256": readiness["artifact_sha256"],
                "human_ballots": 0,
                "official": False,
                "rank_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
