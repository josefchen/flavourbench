from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .expert_calibration import (
    SCHEMA_VERSION as REPLACEMENT_CANDIDATE_SCHEMA_VERSION,
)
from .expert_calibration import (
    TASK_SCOPE_QUARANTINE,
    TASK_SCOPE_REVIEW_SHA256,
    ExpertCalibrationError,
    _assert_governance_review_contracts,
)
from .expert_review import RUBRIC_DIMENSIONS, TASK_FAMILIES

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "flavourbench/artifacts/season1/human-review"
DEFAULT_CANDIDATE_ROOT = ROOT / "flavourbench/artifacts/expert-calibration"
DEFAULT_REPLACEMENT_CANDIDATE_DIR = ROOT / "flavourbench/artifacts/expert-calibration/candidate-v11"
CONSENT_DIR = ROOT / "protocol/consent"
ACCEPTED_FINAL_FINISH_REASONS = frozenset({"completed", "end_turn", "stop", "stop_sequence"})
CANDIDATE_DIRECTORY_PATTERN = re.compile(r"^candidate-v([1-9][0-9]*)$")
CANDIDATE_FILENAME_PATTERN = re.compile(r"^candidate-pack-([0-9a-f]{64})\.json$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

CONTROL_PLANE_SQL = r"""
WITH review_session AS (
  SELECT event.entity_id AS review_session_id,
         event.payload_json::jsonb AS payload,
         event.created_at
  FROM run_events AS event
  WHERE event.entity_type = 'expert_review_session'
    AND event.event_type = 'expert_review_session_opened'
    AND event.payload_json::jsonb ->> 'admission_pathway' = 'anonymous_external_rater'
    AND EXISTS (
      SELECT 1
      FROM run_events AS restriction
      WHERE restriction.entity_type = 'expert_review_session'
        AND restriction.entity_id = event.entity_id
        AND restriction.event_type = 'expert_review_batch_restricted'
        AND restriction.payload_json::jsonb ->> 'evidence_status'
            = 'restricted_operational_qa'
    )
  ORDER BY event.created_at DESC
  LIMIT 1
),
anonymous_reviewer AS (
  SELECT reviewer.id,
         reviewer.cohort
  FROM expert_reviewers AS reviewer
  CROSS JOIN review_session AS review_session
  WHERE reviewer.id = review_session.payload ->> 'reviewer_id'
    AND reviewer.cohort = 'expert_independent'
  LIMIT 1
),
submitted AS (
  SELECT event.created_at, event.payload_json::jsonb AS payload
  FROM run_events AS event
  CROSS JOIN review_session AS review_session
  WHERE event.entity_type = 'expert_review_assignment'
    AND event.event_type = 'expert_review_assignment_submitted'
    AND event.payload_json::jsonb ->> 'review_session_id' = review_session.review_session_id
),
review_rows AS (
  SELECT jsonb_build_object(
    'created_at', submitted.created_at,
    'battle_id', submitted.payload ->> 'battle_id',
    'task_public_id', task.public_id,
    'mode', submitted.payload ->> 'mode',
    'track', submitted.payload ->> 'track',
    'category', submitted.payload ->> 'category',
    'choice', submitted.payload ->> 'normalized_choice',
    'duration_ms', (submitted.payload ->> 'duration_ms')::bigint,
    'answer_review_duration_ms',
      (submitted.payload ->> 'answer_review_duration_ms')::bigint,
    'speed_flag', COALESCE((submitted.payload ->> 'speed_flag')::boolean, false),
    'task_validity',
      submitted.payload #>> '{normalized_rubric,review_metadata,task_validity}',
    'reason_tags', COALESCE(submitted.payload -> 'normalized_reason_tags', '[]'::jsonb),
    'left_failure_tags', COALESCE(
      submitted.payload #> '{normalized_rubric,review_metadata,left_failure_tags}',
      '[]'::jsonb
    ),
    'right_failure_tags', COALESCE(
      submitted.payload #> '{normalized_rubric,review_metadata,right_failure_tags}',
      '[]'::jsonb
    ),
    'rubric', submitted.payload -> 'normalized_rubric',
    'primary_choice', primary_vote.choice,
    'primary_rubric', primary_vote.rubric_json::jsonb,
    'left_condition', left_arm.condition,
    'right_condition', right_arm.condition,
    'left_status', left_arm.status,
    'right_status', right_arm.status,
    'left_finish_reason', left_arm.finish_reason,
    'right_finish_reason', right_arm.finish_reason,
    'battle_manifest_sha256', battle.manifest_sha256,
    'left_answer_sha256', left_arm.answer_markdown_sha256,
    'right_answer_sha256', right_arm.answer_markdown_sha256,
    'model_ids', COALESCE(arms.model_ids, '[]'::jsonb)
  ) AS record
  FROM submitted
  JOIN battles AS battle
    ON battle.id = submitted.payload ->> 'battle_id'
  JOIN tasks AS task ON task.id = battle.task_id
  LEFT JOIN votes AS primary_vote
    ON primary_vote.battle_id = submitted.payload ->> 'battle_id'
   AND primary_vote.cohort = 'expert_independent'
  LEFT JOIN response_arms AS left_arm
    ON left_arm.battle_id = submitted.payload ->> 'battle_id'
   AND left_arm.side = 'left'
  LEFT JOIN response_arms AS right_arm
    ON right_arm.battle_id = submitted.payload ->> 'battle_id'
   AND right_arm.side = 'right'
  LEFT JOIN LATERAL (
    SELECT jsonb_agg(response_arms.model_id ORDER BY response_arms.side) AS model_ids
    FROM response_arms
    WHERE response_arms.battle_id = submitted.payload ->> 'battle_id'
  ) AS arms ON TRUE
),
self_report AS (
  SELECT run_events.payload_json::jsonb AS payload
  FROM run_events
  CROSS JOIN review_session
  WHERE entity_type = 'expert_review_session'
    AND entity_id = review_session.review_session_id
    AND event_type = 'expert_review_batch_self_reported_quality_signal'
  ORDER BY run_events.created_at DESC
  LIMIT 1
),
restriction AS (
  SELECT run_events.payload_json::jsonb AS payload
  FROM run_events
  CROSS JOIN review_session
  WHERE entity_type = 'expert_review_session'
    AND entity_id = review_session.review_session_id
    AND event_type = 'expert_review_batch_restricted'
  ORDER BY run_events.created_at DESC
  LIMIT 1
),
pool AS (
  SELECT event.payload_json::jsonb AS payload
  FROM run_events AS event
  CROSS JOIN review_session AS review_session
  WHERE event.entity_type = 'author_evaluator_pool'
    AND event.event_type = 'author_evaluator_pool_imported'
    AND event.entity_id = review_session.payload ->> 'anonymous_external_pool_sha256'
    AND event.payload_json::jsonb ->> 'candidate_pack_sha256' = event.entity_id
  ORDER BY event.created_at DESC
  LIMIT 1
),
pool_season AS (
  SELECT season.id AS season_id
  FROM seasons AS season
  CROSS JOIN review_session AS review_session
  WHERE season.manifest_sha256 = review_session.payload ->> 'anonymous_external_pool_sha256'
  ORDER BY season.created_at
  LIMIT 1
),
source_classes AS (
  SELECT COALESCE(
    jsonb_object_agg(source_class, source_count),
    '{}'::jsonb
  ) AS payload
  FROM (
    SELECT provenance_json::jsonb ->> 'sourceClass' AS source_class,
           count(*) AS source_count
    FROM tasks
    CROSS JOIN pool_season
    WHERE tasks.season_id = pool_season.season_id
    GROUP BY provenance_json::jsonb ->> 'sourceClass'
  ) AS grouped
)
SELECT jsonb_build_object(
  'schema_revision', (SELECT version_num FROM alembic_version),
  'reviewer', COALESCE((
    SELECT jsonb_build_object(
      'cohort', cohort,
      'qualification_verified', false,
      'identity_collection_prohibited',
        COALESCE((review_session.payload ->> 'identity_collection_prohibited')::boolean, false),
      'qualification_self_attested',
        COALESCE(
          review_session.payload ->> 'qualification_basis'
            = 'reviewer_self_attestation_unverified',
          false
        ),
      'independence_self_attested',
        COALESCE(
          review_session.payload ->> 'independence_basis'
            = 'reviewer_self_attestation',
          false
        ),
      'consent_reference_present',
        COALESCE(
          length((SELECT payload ->> 'consent_document_sha256' FROM restriction)) = 64,
          false
        ),
      'consent_document_sha256',
        (SELECT payload ->> 'consent_document_sha256' FROM restriction),
      'independent_expert_validation_claim',
        false
    )
    FROM anonymous_reviewer
    CROSS JOIN review_session
  ), 'null'::jsonb),
  'session', COALESCE((
    SELECT jsonb_build_object(
      'opened_at', created_at,
      'target_presentations', (payload ->> 'target_judgments')::integer,
      'protocol_version', payload ->> 'protocol_version',
      'protocol_sha256', payload ->> 'protocol_sha256',
      'admission_pathway', payload ->> 'admission_pathway',
      'source_pool_sha256', payload ->> 'anonymous_external_pool_sha256'
    )
    FROM review_session
  ), 'null'::jsonb),
  'source_pool', COALESCE((SELECT payload FROM pool), '{}'::jsonb),
  'reviewer_self_report', COALESCE((SELECT payload FROM self_report), '{}'::jsonb),
  'restriction', COALESCE((SELECT payload FROM restriction), '{}'::jsonb),
  'retained_provider_generation_ids', (
    SELECT COALESCE(sum(json_array_length(provider_generation_ids_json)), 0)
    FROM response_arms
    JOIN battles ON battles.id = response_arms.battle_id
    CROSS JOIN pool_season
    WHERE battles.season_id = pool_season.season_id
  ),
  'task_state', jsonb_build_object(
    'total', (SELECT count(*) FROM tasks CROSS JOIN pool_season
              WHERE tasks.season_id = pool_season.season_id),
    'calibration', (SELECT count(*) FROM tasks CROSS JOIN pool_season
                    WHERE tasks.season_id = pool_season.season_id
                      AND split = 'calibration'),
    'synthetic', (SELECT count(*) FROM tasks CROSS JOIN pool_season
                  WHERE tasks.season_id = pool_season.season_id
                    AND COALESCE(
                      (provenance_json::jsonb ->> 'synthetic')::boolean, false
                    )),
    'source_classes', (SELECT payload FROM source_classes)
  ),
  'records', COALESCE((SELECT jsonb_agg(record ORDER BY record ->> 'created_at')
                       FROM review_rows), '[]'::jsonb)
);
"""


class AnonymousReviewReportError(RuntimeError):
    """The identity-minimized review report could not be built safely."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _run(*command: str) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def control_plane_snapshot() -> dict[str, Any]:
    payload = _run(
        "sudo",
        "-n",
        "docker",
        "exec",
        "epicure-flavourbench-db-1",
        "psql",
        "-U",
        "flavourbench_bootstrap",
        "-d",
        "flavourbench",
        "-Atc",
        CONTROL_PLANE_SQL,
    )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise AnonymousReviewReportError("control-plane query did not return an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _historical_pool_sha256(snapshot: Mapping[str, Any]) -> str:
    session = snapshot.get("session")
    source_pool = snapshot.get("source_pool")
    if not isinstance(session, Mapping) or not isinstance(source_pool, Mapping):
        raise AnonymousReviewReportError("historical source-pool provenance is unavailable")
    session_sha256 = session.get("source_pool_sha256")
    event_sha256 = source_pool.get("candidate_pack_sha256")
    if (
        not isinstance(session_sha256, str)
        or SHA256_PATTERN.fullmatch(session_sha256) is None
        or not isinstance(event_sha256, str)
        or SHA256_PATTERN.fullmatch(event_sha256) is None
    ):
        raise AnonymousReviewReportError("historical source-pool digest is unavailable")
    if event_sha256 != session_sha256:
        raise AnonymousReviewReportError(
            "source-pool import does not match the historical review session"
        )
    return session_sha256


def _verified_candidate_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    directory_match = CANDIDATE_DIRECTORY_PATTERN.fullmatch(path.parent.name)
    filename_match = CANDIDATE_FILENAME_PATTERN.fullmatch(path.name)
    if directory_match is None or filename_match is None:
        raise AnonymousReviewReportError("candidate evidence path is not versioned and addressed")
    if filename_match.group(1) != expected_sha256:
        raise AnonymousReviewReportError("candidate filename digest does not match its binding")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnonymousReviewReportError(f"candidate evidence cannot be read: {path}") from exc
    if not isinstance(value, dict):
        raise AnonymousReviewReportError("candidate evidence is not a JSON object")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    calculated = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    expected_schema = f"flavourbench-expert-calibration-candidate-v{directory_match.group(1)}"
    if (
        value.get("artifact_sha256") != expected_sha256
        or calculated != expected_sha256
        or value.get("schema_version") != expected_schema
    ):
        raise AnonymousReviewReportError("candidate evidence digest or schema is invalid")
    return value


def _resolve_candidate_artifact(candidate_root: Path, candidate_sha256: str) -> Path:
    if CANDIDATE_DIRECTORY_PATTERN.fullmatch(candidate_root.name):
        matches = [candidate_root / f"candidate-pack-{candidate_sha256}.json"]
        matches = [path for path in matches if path.is_file()]
    else:
        matches = sorted(
            path
            for path in candidate_root.glob(f"candidate-v*/candidate-pack-{candidate_sha256}.json")
            if path.is_file()
        )
    if len(matches) != 1:
        raise AnonymousReviewReportError(
            "historical candidate digest must resolve to exactly one versioned artifact"
        )
    _verified_candidate_artifact(matches[0], candidate_sha256)
    return matches[0]


def attach_candidate_evidence(
    snapshot: dict[str, Any], candidate_root: Path = DEFAULT_CANDIDATE_ROOT
) -> dict[str, Any]:
    candidate_sha256 = _historical_pool_sha256(snapshot)
    path = _resolve_candidate_artifact(candidate_root, candidate_sha256)
    value = _verified_candidate_artifact(path, candidate_sha256)
    observed = value.get("observed")
    if not isinstance(observed, Mapping):
        raise AnonymousReviewReportError("candidate evidence has no observed-count block")
    snapshot["candidate_evidence"] = dict(observed)
    snapshot["candidate_source"] = {
        "path": _display_path(path),
        "file_sha256": _sha256_file(path),
        "artifact_sha256": candidate_sha256,
        "schema_version": value.get("schema_version"),
    }
    return snapshot


def attach_replacement_candidate_evidence(
    snapshot: dict[str, Any],
    candidate_dir: Path = DEFAULT_REPLACEMENT_CANDIDATE_DIR,
) -> dict[str, Any]:
    paths = sorted(candidate_dir.glob("candidate-pack-*.json"))
    if len(paths) != 1:
        raise AnonymousReviewReportError("exactly one replacement candidate pack must be present")
    path = paths[0]
    filename_match = CANDIDATE_FILENAME_PATTERN.fullmatch(path.name)
    if filename_match is None:
        raise AnonymousReviewReportError("replacement candidate filename is not content-addressed")
    calculated = filename_match.group(1)
    value = _verified_candidate_artifact(path, calculated)
    if value.get("schema_version") != REPLACEMENT_CANDIDATE_SCHEMA_VERSION:
        raise AnonymousReviewReportError("replacement candidate schema is not current")
    items = value.get("items")
    if not isinstance(items, list):
        raise AnonymousReviewReportError("replacement candidate has no item coordinates")
    try:
        _assert_governance_review_contracts(items)
    except ExpertCalibrationError as exc:
        raise AnonymousReviewReportError(
            "replacement candidate is not bound to the governed scope review"
        ) from exc
    snapshot["replacement_candidate"] = {
        "path": _display_path(path),
        "artifact_sha256": calculated,
        "schema_version": value.get("schema_version"),
        "observed": value.get("observed"),
        "selection_policy": value.get("selection_policy"),
    }
    snapshot["scope_governance"] = {
        "artifact_sha256": TASK_SCOPE_REVIEW_SHA256,
        "schema_version": "flavourbench-specialist-scope-review-v1",
        "quarantined_task_count": len(TASK_SCOPE_QUARANTINE),
    }
    return snapshot


def attach_consent_evidence(snapshot: dict[str, Any]) -> dict[str, Any]:
    reviewer = snapshot.get("reviewer")
    if not isinstance(reviewer, Mapping):
        raise AnonymousReviewReportError("reviewer provenance is unavailable")
    digest = reviewer.get("consent_document_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise AnonymousReviewReportError("reviewer consent digest is unavailable")
    matches = [
        path
        for path in sorted(CONSENT_DIR.glob("*"))
        if path.is_file() and _sha256_file(path) == digest
    ]
    if len(matches) != 1:
        raise AnonymousReviewReportError("consent digest does not resolve to one local document")
    path = matches[0]
    opening = path.read_text(encoding="utf-8")[:800].lower()
    active = "status: active" in opening and "status: not active" not in opening
    snapshot["consent_evidence"] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "status": "active" if active else "inactive_or_unapproved",
        "collection_permitted": active,
    }
    return snapshot


def _wilson_interval(successes: float, sample_size: int) -> list[float] | None:
    if sample_size < 1:
        return None
    probability = successes / sample_size
    z = 1.959963984540054
    denominator = 1 + z**2 / sample_size
    center = (probability + z**2 / (2 * sample_size)) / denominator
    half_width = (
        z
        * math.sqrt(probability * (1 - probability) / sample_size + z**2 / (4 * sample_size**2))
        / denominator
    )
    return [round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4)]


def _dimension_reliability(records: Sequence[Mapping[str, Any]]) -> tuple[float | None, int]:
    differences: list[float] = []
    for row in records:
        primary = row.get("primary_rubric")
        repeat = row.get("rubric")
        if not isinstance(primary, Mapping) or not isinstance(repeat, Mapping):
            continue
        for side in ("left", "right"):
            first_side = primary.get(side)
            second_side = repeat.get(side)
            if not isinstance(first_side, Mapping) or not isinstance(second_side, Mapping):
                continue
            for dimension in RUBRIC_DIMENSIONS:
                first = first_side.get(dimension)
                second = second_side.get(dimension)
                if isinstance(first, int) and isinstance(second, int):
                    differences.append(abs(first - second))
    if not differences:
        return None, 0
    return round(sum(differences) / len(differences), 4), len(differences)


def _uplift_outcome(row: Mapping[str, Any]) -> str:
    choice = str(row.get("choice", ""))
    if choice in {"tie", "both_bad"}:
        return choice
    if choice not in {"left", "right"}:
        raise AnonymousReviewReportError(f"invalid stored preference choice: {choice!r}")
    condition = row.get(f"{choice}_condition")
    if condition == "epicure_on":
        return "epicure_win"
    if condition == "epicure_off":
        return "epicure_loss"
    raise AnonymousReviewReportError("uplift presentation lacks an Epicure on/off side pair")


def _counter_payload(counter: Counter[str], labels: Sequence[str]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in labels}


def build_report(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    reviewer = snapshot.get("reviewer")
    session = snapshot.get("session")
    records = snapshot.get("records")
    if not isinstance(reviewer, Mapping) or not isinstance(session, Mapping):
        raise AnonymousReviewReportError("no active anonymous external review session exists")
    if not isinstance(records, list):
        raise AnonymousReviewReportError("review records are unavailable")

    historical_pool_sha256 = _historical_pool_sha256(snapshot)
    candidate_source = snapshot.get("candidate_source")
    if (
        not isinstance(candidate_source, Mapping)
        or candidate_source.get("artifact_sha256") != historical_pool_sha256
    ):
        raise AnonymousReviewReportError(
            "candidate artifact does not match the historical review session"
        )
    mismatched_battles = [
        row.get("battle_id")
        for row in records
        if not isinstance(row, Mapping)
        or row.get("battle_manifest_sha256") != historical_pool_sha256
    ]
    if mismatched_battles:
        raise AnonymousReviewReportError(
            "review records do not belong to the historical source pool"
        )
    scope_governance = snapshot.get("scope_governance")
    if not isinstance(scope_governance, Mapping) or scope_governance != {
        "artifact_sha256": TASK_SCOPE_REVIEW_SHA256,
        "schema_version": "flavourbench-specialist-scope-review-v1",
        "quarantined_task_count": len(TASK_SCOPE_QUARANTINE),
    }:
        raise AnonymousReviewReportError("governed specialist-scope evidence is unavailable")

    primary = [row for row in records if row.get("mode") == "primary"]
    repeats = [row for row in records if row.get("mode") == "reliability_repeat"]
    unsupported_modes = [
        row.get("mode")
        for row in records
        if row.get("mode") not in {"primary", "reliability_repeat"}
    ]
    if unsupported_modes:
        raise AnonymousReviewReportError(f"unsupported review mode: {unsupported_modes[0]!r}")

    target = int(session.get("target_presentations", 0))
    if target < 1:
        raise AnonymousReviewReportError("the review target is not positive")

    def finish_clean(row: Mapping[str, Any]) -> bool:
        return all(
            row.get(f"{side}_status") == "complete"
            and str(row.get(f"{side}_finish_reason") or "").lower() in ACCEPTED_FINAL_FINISH_REASONS
            for side in ("left", "right")
        )

    durations = [int(row["duration_ms"]) for row in records if row.get("duration_ms") is not None]
    answer_durations = [
        int(row["answer_review_duration_ms"])
        for row in records
        if row.get("answer_review_duration_ms") is not None
    ]
    primary_validity = Counter(str(row.get("task_validity") or "missing") for row in primary)
    choice_counts = Counter(str(row.get("choice")) for row in primary)
    by_family = Counter(str(row.get("category")) for row in primary)
    by_track = Counter(str(row.get("track")) for row in primary)
    clean_primary = [row for row in primary if finish_clean(row)]
    non_normal_primary = [row for row in primary if not finish_clean(row)]
    non_normal_arms = sum(
        str(row.get(f"{side}_finish_reason") or "").lower() not in ACCEPTED_FINAL_FINISH_REASONS
        or row.get(f"{side}_status") != "complete"
        for row in primary
        for side in ("left", "right")
    )

    comparable_repeats = [
        row for row in repeats if row.get("primary_choice") in {"left", "right", "tie", "both_bad"}
    ]
    repeat_matches = sum(
        row.get("primary_choice") == row.get("choice") for row in comparable_repeats
    )
    dimension_difference, dimension_comparisons = _dimension_reliability(comparable_repeats)
    self_report = snapshot.get("reviewer_self_report")
    if not isinstance(self_report, Mapping):
        self_report = {}
    recognized_repeats = int(self_report.get("recognized_reliability_repeats", 0))
    repeat_interpretable = recognized_repeats == 0

    def uplift_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts: Counter[str] = Counter(_uplift_outcome(row) for row in rows)
        effective_n = counts["epicure_win"] + counts["tie"] + counts["epicure_loss"]
        successes = counts["epicure_win"] + 0.5 * counts["tie"]
        return {
            "n": len(rows),
            "epicure_wins": counts["epicure_win"],
            "ties": counts["tie"],
            "epicure_losses": counts["epicure_loss"],
            "both_bad": counts["both_bad"],
            "effective_n_excluding_both_bad": effective_n,
            "tie_adjusted_epicure_preference_share": (
                round(successes / effective_n, 4) if effective_n else None
            ),
        }

    uplift_rows = [row for row in primary if row.get("track") == "epicure_uplift"]
    clean_uplift_rows = [row for row in clean_primary if row.get("track") == "epicure_uplift"]

    failure_tags: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_scores: dict[str, list[int]] = defaultdict(list)
    clean_evidence_scores: dict[str, list[int]] = defaultdict(list)
    invented_outcomes: Counter[str] = Counter()
    for row in primary:
        rubric = row.get("rubric")
        for side in ("left", "right"):
            condition = str(row.get(f"{side}_condition") or "missing")
            tags = row.get(f"{side}_failure_tags")
            if isinstance(tags, list):
                for tag in tags:
                    failure_tags[str(tag)][condition] += 1
                    if tag == "invented_evidence":
                        choice = str(row.get("choice"))
                        invented_outcomes[
                            choice
                            if choice in {"tie", "both_bad"}
                            else "win"
                            if choice == side
                            else "loss"
                        ] += 1
            if isinstance(rubric, Mapping):
                side_rubric = rubric.get(side)
                if isinstance(side_rubric, Mapping):
                    score = side_rubric.get("evidence_use")
                    if isinstance(score, int):
                        evidence_scores[condition].append(score)
                        if finish_clean(row):
                            clean_evidence_scores[condition].append(score)

    task_public_ids = {
        str(row.get("task_public_id")) for row in primary if row.get("task_public_id")
    }
    scope_quarantine_reviewed = sorted(task_public_ids.intersection(TASK_SCOPE_QUARANTINE))
    model_ids = {
        str(model_id)
        for row in primary
        for model_id in row.get("model_ids", [])
        if isinstance(model_id, str)
    }
    observed_at = max(
        (str(row.get("created_at")) for row in records if row.get("created_at")),
        default=str(session.get("opened_at")),
    )
    source_pool = snapshot.get("source_pool")
    if not isinstance(source_pool, Mapping):
        source_pool = {}
    task_state = snapshot.get("task_state")
    if not isinstance(task_state, Mapping):
        task_state = {}
    candidate_evidence = snapshot.get("candidate_evidence")
    if not isinstance(candidate_evidence, Mapping):
        candidate_evidence = {}
    consent = snapshot.get("consent_evidence")
    if not isinstance(consent, Mapping):
        consent = {
            "status": "unresolved",
            "collection_permitted": False,
        }
    restriction = snapshot.get("restriction")
    if not isinstance(restriction, Mapping):
        restriction = {}

    replacement = snapshot.get("replacement_candidate")
    if not isinstance(replacement, Mapping):
        replacement = {}
    replacement_observed = replacement.get("observed")
    if not isinstance(replacement_observed, Mapping):
        replacement_observed = {}

    report = {
        "schema_version": "flavourbench-human-review-operational-qa-v3",
        "observed_at": observed_at,
        "scope": "restricted_operational_quality_assurance",
        "provenance": {
            "database_schema_revision": snapshot.get("schema_revision"),
            "protocol_version": session.get("protocol_version"),
            "protocol_sha256": session.get("protocol_sha256"),
            "admission_pathway": session.get("admission_pathway"),
            "application_record_pseudonymous": bool(reviewer.get("identity_collection_prohibited")),
            "qualification_basis": "self_attested_unverified",
            "independence_basis": "self_attested_unverified",
            "qualification_verified": bool(reviewer.get("qualification_verified")),
            "independent_expert_validation_claim": False,
            "consent": dict(consent),
        },
        "source_pool": {
            "historical_review_session_pool_sha256": historical_pool_sha256,
            "real_output_pairs": int(source_pool.get("battle_count", 0)),
            "paid_source_arms": int(source_pool.get("source_arm_count", 0)),
            "provider_calls": int(candidate_evidence.get("real_provider_calls", 0)),
            "retained_provider_generation_ids": int(
                snapshot.get("retained_provider_generation_ids", 0)
            ),
            "epicure_calls": int(candidate_evidence.get("real_epicure_calls", 0)),
            "successful_epicure_calls": int(
                candidate_evidence.get("successful_real_epicure_calls", 0)
            ),
            "synthetic_arms": int(source_pool.get("synthetic_arm_count", 0)),
            "rank_eligible_battles": int(source_pool.get("rank_eligible_battle_count", 0)),
            "data_stratum": source_pool.get("data_stratum"),
            "run_class": source_pool.get("run_class"),
            "candidate_artifact": snapshot.get("candidate_source"),
        },
        "quality_funnel": {
            "candidate_pairs": int(source_pool.get("battle_count", 0)),
            "unique_primary_pairs_reviewed": len(primary),
            "finish_clean_primary_pairs": len(clean_primary),
            "finish_clean_non_both_bad_preferences": sum(
                row.get("choice") != "both_bad" for row in clean_primary
            ),
            "unseen_candidate_pairs": max(
                0, int(source_pool.get("battle_count", 0)) - len(primary)
            ),
            "non_normal_response_arms": non_normal_arms,
            "affected_primary_pairs": len(non_normal_primary),
        },
        "review_progress": {
            "completed_presentations": len(records),
            "target_presentations": target,
            "completion_rate": round(len(records) / target, 4),
            "unique_primary_judgments": len(primary),
            "repeat_presentations": len(repeats),
            "by_family": _counter_payload(by_family, TASK_FAMILIES),
            "by_track": _counter_payload(by_track, ("model_arena", "epicure_uplift")),
            "models_observed": len(model_ids),
            "speed_flags": sum(bool(row.get("speed_flag")) for row in records),
            "primary_task_validity": dict(sorted(primary_validity.items())),
            "median_total_review_seconds": (
                round(statistics.median(durations) / 1000, 1) if durations else None
            ),
            "mean_total_review_seconds": (
                round(statistics.fmean(durations) / 1000, 1) if durations else None
            ),
            "median_answer_review_seconds": (
                round(statistics.median(answer_durations) / 1000, 1) if answer_durations else None
            ),
        },
        "restricted_diagnostic_preferences": {
            "publication_permitted": False,
            "all_primary_choices": _counter_payload(
                choice_counts, ("left", "right", "tie", "both_bad")
            ),
            "all_primary_condition_normalized": uplift_summary(uplift_rows),
            "finish_clean_sensitivity": uplift_summary(clean_uplift_rows),
            "reason": (
                "Collected under inactive consent; selected development pool; one rater; "
                "post hoc exclusions."
            ),
        },
        "repeat_check": {
            "completed_repeats": len(repeats),
            "comparable_repeats": len(comparable_repeats),
            "recognized_repeats": recognized_repeats,
            "exact_preference_matches": repeat_matches,
            "observed_exact_agreement": (
                round(repeat_matches / len(comparable_repeats), 4) if comparable_repeats else None
            ),
            "observed_mean_absolute_dimension_difference": dimension_difference,
            "dimension_comparisons": dimension_comparisons,
            "reliability_interpretable": repeat_interpretable,
            "interpretation": (
                "uninterpretable_because_repeats_were_recognized_and_scores_mirrored"
                if not repeat_interpretable
                else "limited_unrecognized_repeat_check"
            ),
        },
        "completion_audit": {
            "accepted_finish_reasons": sorted(ACCEPTED_FINAL_FINISH_REASONS),
            "non_normal_response_arms": non_normal_arms,
            "affected_primary_pairs": len(non_normal_primary),
            "replacement_candidate": {
                "artifact_sha256": replacement.get("artifact_sha256"),
                "schema_version": replacement.get("schema_version"),
                "pairs": int(replacement_observed.get("candidate_pairs", 0)),
                "source_arms": int(replacement_observed.get("source_arms", 0)),
                "provider_calls": int(replacement_observed.get("real_provider_calls", 0)),
                "epicure_calls": int(replacement_observed.get("real_epicure_calls", 0)),
                "synthetic_arms": int(replacement_observed.get("synthetic_arms", 0)),
            },
        },
        "scope_audit": {
            "governance_review": dict(scope_governance),
            "governed_quarantine_tasks": len(TASK_SCOPE_QUARANTINE),
            "general_track_quarantine_tasks_reviewed": len(scope_quarantine_reviewed),
            "task_public_ids": scope_quarantine_reviewed,
            "response_specific_safety_reports": int(
                self_report.get("reported_potential_safety_hazards", 0)
            ),
            "safety_reports_verified": 0,
            "safety_status": "pending_qualified_food_safety_adjudication",
        },
        "failure_tag_audit": {
            "counts_by_tag_and_condition": {
                tag: {
                    "epicure_on": counts["epicure_on"],
                    "epicure_off": counts["epicure_off"],
                }
                for tag, counts in sorted(failure_tags.items())
            },
            "legacy_combined_tags": ["invented_evidence", "unsafe_or_impractical"],
            "prospective_protocol": "flavourbench-blinded-pair-review-v6",
        },
        "evidence_use_signal": {
            "status": "post_hoc_hypothesis_generating_only",
            "legacy_invented_evidence_tags": {
                "epicure_on": failure_tags["invented_evidence"]["epicure_on"],
                "epicure_off": failure_tags["invented_evidence"]["epicure_off"],
                "tagged_arm_outcomes": _counter_payload(
                    invented_outcomes, ("win", "tie", "loss", "both_bad")
                ),
            },
            "mean_evidence_use_score": {
                condition: (round(statistics.fmean(scores), 4) if scores else None)
                for condition, scores in sorted(evidence_scores.items())
            },
            "finish_clean_mean_evidence_use_score": {
                condition: (round(statistics.fmean(scores), 4) if scores else None)
                for condition, scores in sorted(clean_evidence_scores.items())
            },
            "prospective_taxonomy": [
                "evidence_trace_mismatch",
                "entity_resolution_mismatch",
                "similarity_as_functional_proof",
                "similarity_as_mechanism",
                "axis_as_measured_quantity",
                "score_as_normative_truth",
                "selective_evidence",
                "irrelevant_evidence",
                "false_precision",
            ],
        },
        "task_pool": {
            "tasks": int(task_state.get("total", 0)),
            "calibration_tasks": int(task_state.get("calibration", 0)),
            "synthetic_tasks": int(task_state.get("synthetic", 0)),
            "source_classes": dict(task_state.get("source_classes", {})),
            "season1_confirmatory_tasks": 0,
        },
        "claim_boundary": {
            "database_restriction_recorded": bool(restriction),
            "evidence_status": "restricted_operational_qa",
            "research_use": False,
            "paper_use": False,
            "rank_eligible": False,
            "leaderboard_use": False,
            "official_season1_evidence": False,
            "independent_expert_validation": False,
            "quality_leaderboard_supported": False,
            "limitations": [
                "the bound consent document is marked inactive",
                "one pseudonymous application record with unverified self-attestation",
                "the response pool is selected retrospective development data",
                "four non-normal arms affect three reviewed pairs",
                "all observed repeats were recognized and deliberately mirrored",
                "specialist-scope and safety reports lack qualified adjudication",
            ],
        },
    }
    return report


def render_markdown(report: Mapping[str, Any], digest: str) -> str:
    progress = report["review_progress"]
    funnel = report["quality_funnel"]
    clean = report["restricted_diagnostic_preferences"]["finish_clean_sensitivity"]
    repeat = report["repeat_check"]
    evidence = report["evidence_use_signal"]["legacy_invented_evidence_tags"]
    replacement = report["completion_audit"]["replacement_candidate"]
    source_pool = report["source_pool"]
    scope = report["scope_audit"]
    return "\n".join(
        (
            "# Human-review operational QA audit",
            "",
            f"Artifact SHA-256: `{digest}`",
            "",
            "This report documents a contained quality incident. The records are restricted to "
            "operational QA. They do not support a paper result, a leaderboard, or an Epicure "
            "effect estimate.",
            "",
            "## Evidence boundary",
            "",
            f"- Presentations: `{progress['completed_presentations']}` / "
            f"`{progress['target_presentations']}`",
            f"- Unique primary comparisons: `{progress['unique_primary_judgments']}`",
            f"- Repeat presentations: `{progress['repeat_presentations']}`",
            f"- Historical session pool: `{source_pool['historical_review_session_pool_sha256']}`",
            "- Consent status: `inactive or unapproved`",
            "- Permitted use: `restricted operational QA only`",
            "",
            "## Completion audit",
            "",
            f"The candidate pool contained `{funnel['candidate_pairs']}` pairs. "
            f"`{funnel['unique_primary_pairs_reviewed']}` received a primary review, "
            f"`{funnel['finish_clean_primary_pairs']}` were finish-clean, and "
            f"`{funnel['finish_clean_non_both_bad_preferences']}` retained a usable "
            "preference after excluding both-bad.",
            "",
            f"The replacement pack contains `{replacement['pairs']}` pairs, "
            f"`{replacement['source_arms']}` real arms, and `{replacement['synthetic_arms']}` "
            "synthetic arms. It excludes non-normal completions and the task-scope quarantine.",
            "",
            "## Restricted sensitivity check",
            "",
            f"After removing the three affected pairs, the stored condition-normalized counts are "
            f"`{clean['epicure_wins']}` wins, `{clean['ties']}` ties, "
            f"`{clean['epicure_losses']}` losses, and `{clean['both_bad']}` both-bad. "
            "These counts are retained only to verify exclusion logic.",
            "",
            "## Repeat and tagging diagnostics",
            "",
            f"All `{repeat['recognized_repeats']}` observed repeats were recognized, so the "
            "agreement statistic is not interpretable as reliability. The legacy "
            f"invented-evidence tag occurred on `{evidence['epicure_on']}` Epicure-on arms and "
            f"`{evidence['epicure_off']}` Epicure-off arms. This is a post hoc signal for a "
            "prospective evidence-misuse taxonomy, not a causal result.",
            "",
            "Two responses were reported as potentially unsafe. Neither report is counted as a "
            "verified safety error pending qualified food-safety adjudication.",
            "",
            "## Governed scope audit",
            "",
            f"The authoritative scope review quarantines `{scope['governed_quarantine_tasks']}` "
            "task coordinates. Of the historical primary comparisons, "
            f"`{scope['general_track_quarantine_tasks_reviewed']}` used a quarantined task. "
            "This retrospective classification does not relabel the v11 replacement pool.",
            "",
        )
    )


def write_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(canonical_bytes(report)).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    json_path = output_dir / f"restricted-operational-qa-{digest}.json"
    markdown_path = output_dir / f"restricted-operational-qa-{digest}.md"
    json_path.write_text(
        json.dumps({**report, "artifact_sha256": digest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report, digest), encoding="utf-8")
    json_path.chmod(0o600)
    markdown_path.chmod(0o600)
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    snapshot = control_plane_snapshot()
    attach_candidate_evidence(snapshot)
    attach_replacement_candidate_evidence(snapshot)
    attach_consent_evidence(snapshot)
    report = build_report(snapshot)
    json_path, markdown_path = write_report(report, args.output_dir.resolve())
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    run()
