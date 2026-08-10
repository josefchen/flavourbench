from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from .construct_blueprint import BLUEPRINT, BLUEPRINT_SHA256
from .current_catalog_audit import verify_catalog_audit_content_address
from .development_task_validation import (
    DevelopmentTaskValidationError,
    verify_validation_packet,
)
from .expert_calibration import TASK_SCOPE_QUARANTINE, TASK_SCOPE_REVIEW_SHA256
from .frontier_manifest import verify_manifest_content_address
from .models import RESEARCH_ARCHIVE_SIGNATURE_CONTEXT
from .research_release import ResearchReleaseError, verify_archive
from .season1_method_validation import verify_artifact as verify_method_validation
from .task_contributor_protocol import PROTOCOL_SCOPE as TASK_CONTRIBUTOR_PROTOCOL_SCOPE
from .task_contributor_protocol import PROTOCOL_SHA256 as TASK_CONTRIBUTOR_PROTOCOL_SHA256
from .task_contributor_protocol import PROTOCOL_VERSION as TASK_CONTRIBUTOR_PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = ROOT / (
    "flavourbench/artifacts/frontier-refresh/2026-08-01/current-route-registry/aggregate/"
    "current-route-registry-"
    "b300d460ec3d93dbfdaea64e0809abf858fa9efb570d0bddeac28566b6cdf010.json"
)
DEFAULT_PARITY_AUDIT = ROOT / (
    "flavourbench/artifacts/season1/readiness/structured-parity/"
    "structured-parity-cost-audit-"
    "065780668d256e8815f8fdc1eb13dc6f9ffb3d514dd9fec67c650d6032478f80.json"
)
DEFAULT_HUMAN_DIR = ROOT / "flavourbench/artifacts/season1/human-review"
DEFAULT_REAL_HUMAN_PILOT = ROOT / (
    "flavourbench/artifacts/season1/human-review/real-human-pilot-v1/"
    "real-human-pilot-quality-"
    "933b35a81b96a364b805bed5f39c29fd8f79fb6a29868d5d6fd7c152d2557430.json"
)
DEFAULT_REAL_PILOT_VALIDATORS = ROOT / (
    "flavourbench/artifacts/season1/validators/real-pilot-v1/"
    "real-arm-validator-audit-"
    "5ff552da8c8ad7a7bcee277bd28f2dd18b4d41f1c3c9425e0707bec41184a40d.json"
)
DEFAULT_SURFACE_CLEAN_TASKS = ROOT / (
    "flavourbench/artifacts/season1/task-validity/development-v2/"
    "development-task-validity-v2-"
    "5ffd81a44267291413bc8a638d15391ec2b51decdda270550f81ca17ec587846.json"
)
DEFAULT_DEVELOPMENT_VALIDATION_PACKET = ROOT / (
    "flavourbench/artifacts/season1/task-validity/human-validation-current-v2/"
    "development-task-human-validation-v2-"
    "c45023aee6cf8ff91437c08c16ae20498b2d025e9a0155aedd44898de1d7fbb1.json"
)
DEVELOPMENT_VALIDATION_PACKET_SHA256 = (
    "c45023aee6cf8ff91437c08c16ae20498b2d025e9a0155aedd44898de1d7fbb1"
)
DEFAULT_CURRENT_MODEL_MANIFEST = ROOT / (
    "flavourbench/artifacts/season1/current-quality-run/manifest-v13-evidence-boundary/"
    "flavourbench-openrouter-unranked-"
    "12f411f86c67af5555036851713290bcaf04e1d725bada5af937839753e7db54.json"
)
DEFAULT_CURRENT_CATALOG_AUDIT = ROOT / (
    "flavourbench/artifacts/season1/current-quality-run/catalog-audit-v2/"
    "current-model-catalog-audit-"
    "9a4507f9c83da65e3e2fe1fd03e147c36ef216dd490a7067bcd79440f1d28947.json"
)
DEFAULT_STUDY_DESIGN = ROOT / "flavourbench/contracts/season1/season1-study-design-v5.json"
DEFAULT_ROBUSTNESS_EVIDENCE_CONTRACT = ROOT / (
    "flavourbench/contracts/season1/season1-validity-robustness-evidence-v1.md"
)
ROBUSTNESS_EVIDENCE_CONTRACT_SHA256 = (
    "37aa50d8dba05410aaa7862d4e2c55674fa992b5cb415ea3ae2b6ece95357056"
)
DEFAULT_METHOD_VALIDATION = ROOT / (
    "flavourbench/contracts/season1/method-validation/"
    "season1-statistical-method-validation-"
    "0b4345e523fdaa97d1b406cd1f2165540d0f9ad338bb49f3ac656da73e3c1933.json"
)
DEFAULT_EPICURE_RELEASE = (
    ROOT / "flavourbench/contracts/epicure/epicure-mcp-1790-r1.release-candidate.json"
)
DEFAULT_PUBLIC_CONSENT = ROOT / "protocol/consent/PUBLIC-RESEARCH-CONSENT-v1-DRAFT.md"
DEFAULT_OUTPUT = ROOT / "flavourbench/artifacts/season1/readiness"
HUMAN_QA_SCHEMA_VERSION = "flavourbench-human-review-operational-qa-v3"
EXPECTED_REVIEWED_QUARANTINE_TASKS = 7

SERVICES = (
    "epicure-flavourbench-db-1",
    "epicure-flavourbench-api-1",
    "epicure-flavourbench-worker-1",
    "epicure-mcp-1",
)

_CONTROL_PLANE_SQL_TEMPLATE = r"""
WITH target_season AS (
  SELECT id, slug, epicure_release_id
  FROM seasons
  WHERE slug = 'season-1'
)
SELECT jsonb_build_object(
  'schema_revision', (SELECT version_num FROM alembic_version),
  'scope', jsonb_build_object(
    'season_id', (SELECT id FROM target_season),
    'season_slug', (SELECT slug FROM target_season),
    'all_release_counts_season_scoped', true
  ),
  'counts', jsonb_build_object(
    'catalog_models', (SELECT count(*) FROM catalog_models),
    'season_models', (SELECT count(*) FROM season_models
      WHERE season_id = (SELECT id FROM target_season)),
    'tasks', (SELECT count(*) FROM tasks
      WHERE season_id = (SELECT id FROM target_season)),
    'task_evidence_artifacts', (SELECT count(*) FROM task_evidence_artifacts evidence
      JOIN tasks task ON task.id = evidence.task_id
      WHERE task.season_id = (SELECT id FROM target_season)),
    'battles', (SELECT count(*) FROM battles
      WHERE season_id = (SELECT id FROM target_season)),
    'response_arms', (SELECT count(*) FROM response_arms arm
      JOIN battles battle ON battle.id = arm.battle_id
      WHERE battle.season_id = (SELECT id FROM target_season)),
    'votes', (SELECT count(*) FROM votes vote
      JOIN battles battle ON battle.id = vote.battle_id
      WHERE battle.season_id = (SELECT id FROM target_season)),
    'jobs', (SELECT count(*) FROM jobs job
      JOIN battles battle ON battle.id = job.battle_id
      WHERE battle.season_id = (SELECT id FROM target_season)),
    'generation_attempts', (SELECT count(*) FROM generation_attempts attempt
      JOIN response_arms arm ON arm.id = attempt.arm_id
      JOIN battles battle ON battle.id = arm.battle_id
      WHERE battle.season_id = (SELECT id FROM target_season)),
    'tool_calls', (SELECT count(*) FROM tool_calls call
      JOIN response_arms arm ON arm.id = call.arm_id
      JOIN battles battle ON battle.id = arm.battle_id
      WHERE battle.season_id = (SELECT id FROM target_season)),
    'official_battles', (SELECT count(*) FROM battles
      WHERE season_id = (SELECT id FROM target_season)
        AND run_class = 'official'),
    'rank_eligible_battles', (SELECT count(*) FROM battles
      WHERE season_id = (SELECT id FROM target_season)
        AND run_class = 'official' AND rank_eligible IS TRUE),
    'valid_public_preferences', (SELECT count(*) FROM votes vote
      JOIN battles battle ON battle.id = vote.battle_id
      WHERE battle.season_id = (SELECT id FROM target_season)
        AND battle.run_class = 'official' AND battle.rank_eligible IS TRUE
        AND vote.cohort = 'public' AND vote.choice IN ('left', 'right', 'tie')),
    'valid_expert_judgments', (SELECT count(*) FROM votes vote
      JOIN battles battle ON battle.id = vote.battle_id
      WHERE battle.season_id = (SELECT id FROM target_season)
        AND battle.run_class = 'official' AND battle.rank_eligible IS TRUE
        AND vote.cohort = 'expert_independent'
        AND vote.choice IN ('left', 'right', 'tie')),
    'official_epicure_releases',
      (SELECT count(*) FROM epicure_releases release
       WHERE release.official_eligible IS TRUE
         AND release.release_id = (SELECT epicure_release_id FROM target_season)),
    'research_release_archives', (SELECT count(*) FROM research_release_archives archive
      WHERE archive.season_id = (SELECT id FROM target_season)
        AND archive.archive_class = 'internal_official'),
    'active_verified_independent_reviewers',
      (SELECT count(DISTINCT reviewer.id) FROM expert_reviewers reviewer
       JOIN run_events review
         ON review.payload_json::jsonb ->> 'reviewer_id' = reviewer.id
       JOIN tasks task ON task.id = review.entity_id
       WHERE task.season_id = (SELECT id FROM target_season)
         AND review.entity_type = 'task'
         AND review.event_type = 'confirmatory_task_review_recorded'
         AND review.payload_json::jsonb ->> 'decision' = 'approve'
         AND COALESCE(
           (review.payload_json::jsonb ->> 'independent_of_author')::boolean, false
         )
         AND reviewer.active IS TRUE AND reviewer.qualification_verified IS TRUE
         AND reviewer.cohort = 'expert_independent'),
    'active_anonymous_external_raters',
      (SELECT count(DISTINCT reviewer.id) FROM expert_reviewers reviewer
       JOIN run_events review
         ON review.payload_json::jsonb ->> 'reviewer_id' = reviewer.id
       JOIN tasks task ON task.id = review.entity_id
       WHERE task.season_id = (SELECT id FROM target_season)
         AND review.entity_type = 'task'
         AND review.event_type = 'confirmatory_task_review_recorded'
         AND review.payload_json::jsonb ->> 'decision' = 'approve'
         AND reviewer.active IS TRUE AND reviewer.cohort = 'expert_independent'
         AND reviewer.profile_json::jsonb ? 'anonymous_external_pool_sha256')
  ),
  'season_1', COALESCE((
    SELECT jsonb_build_object(
      'id', id,
      'slug', slug,
      'status', status,
      'official', official,
      'frozen_at', frozen_at,
      'manifest_sha256', manifest_sha256,
      'prompt_registry_sha256', prompt_registry_sha256,
      'tool_registry_sha256', tool_registry_sha256,
      'epicure_release_id', epicure_release_id,
      'analysis_plan_sha256', analysis_plan_sha256,
      'protocol_bundle_sha256', protocol_bundle_sha256,
      'budget_cap_micros', budget_cap_micros,
      'budget_used_micros', budget_used_micros,
      'budget_reserved_micros', budget_reserved_micros
    )
    FROM seasons WHERE slug = 'season-1'
  ), 'null'::jsonb),
  'task_bank', jsonb_build_object(
    'season1_eligible', (SELECT count(*) FROM tasks
      WHERE season_id = (SELECT id FROM target_season) AND COALESCE(
      (provenance_json::jsonb ->> 'season1_eligible')::boolean, false
    )),
    'scored', (SELECT count(*) FROM tasks
      WHERE season_id = (SELECT id FROM target_season)
        AND split = 'scored' AND COALESCE(
      (provenance_json::jsonb ->> 'season1_eligible')::boolean, false
    )),
    'development', (SELECT count(*) FROM tasks
      WHERE season_id = (SELECT id FROM target_season)
        AND split = 'development' AND COALESCE(
      (provenance_json::jsonb ->> 'season1_eligible')::boolean, false
    )),
    'private_reserve', (SELECT count(*) FROM tasks
      WHERE season_id = (SELECT id FROM target_season)
        AND split = 'private_reserve' AND COALESCE(
      (provenance_json::jsonb ->> 'season1_eligible')::boolean, false
    )),
    'calibration_only', (SELECT count(*) FROM tasks
      WHERE season_id = (SELECT id FROM target_season) AND split = 'calibration'),
    'human_task_candidates', (SELECT count(*) FROM run_events candidate
      WHERE candidate.entity_type = 'task_candidate'
        AND candidate.event_type = 'task_candidate_submitted'
        AND EXISTS (SELECT 1 FROM tasks task
          WHERE task.season_id = (SELECT id FROM target_season)
            AND task.provenance_json::jsonb ->> 'source_candidate_id' = candidate.entity_id)),
    'task_candidates_submitted_total', (SELECT count(*) FROM run_events candidate
      WHERE candidate.entity_type = 'task_candidate'
        AND candidate.event_type = 'task_candidate_submitted'
        AND candidate.payload_json::jsonb ->> 'construct_blueprint_sha256' =
          '__CONSTRUCT_BLUEPRINT_SHA256__'
        AND candidate.payload_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'),
    'active_task_candidates', (SELECT count(*) FROM run_events candidate
      WHERE candidate.entity_type = 'task_candidate'
        AND candidate.event_type = 'task_candidate_submitted'
        AND candidate.payload_json::jsonb ->> 'construct_blueprint_sha256' =
          '__CONSTRUCT_BLUEPRINT_SHA256__'
        AND candidate.payload_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'
        AND NOT EXISTS (SELECT 1 FROM run_events withdrawal
          WHERE withdrawal.entity_type = 'task_candidate'
            AND withdrawal.entity_id = candidate.entity_id
            AND withdrawal.event_type = 'task_candidate_withdrawal_recorded')),
    'withdrawn_task_candidates', (SELECT count(DISTINCT withdrawal.entity_id)
      FROM run_events withdrawal
      JOIN run_events candidate
        ON candidate.entity_type = 'task_candidate'
       AND candidate.entity_id = withdrawal.entity_id
       AND candidate.event_type = 'task_candidate_submitted'
      WHERE withdrawal.entity_type = 'task_candidate'
        AND withdrawal.event_type = 'task_candidate_withdrawal_recorded'
        AND candidate.payload_json::jsonb ->> 'construct_blueprint_sha256' =
          '__CONSTRUCT_BLUEPRINT_SHA256__'
        AND candidate.payload_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'),
    'duplicate_candidate_withdrawals', (SELECT count(*) FROM (
      SELECT withdrawal.entity_id
      FROM run_events withdrawal
      WHERE withdrawal.entity_type = 'task_candidate'
        AND withdrawal.event_type = 'task_candidate_withdrawal_recorded'
      GROUP BY withdrawal.entity_id
      HAVING count(*) > 1
    ) duplicate_withdrawals),
    'withdrawn_imported_task_candidates', (SELECT count(DISTINCT withdrawal.entity_id)
      FROM run_events withdrawal
      WHERE withdrawal.entity_type = 'task_candidate'
        AND withdrawal.event_type = 'task_candidate_withdrawal_recorded'
        AND EXISTS (SELECT 1 FROM tasks task
          WHERE task.provenance_json::jsonb ->> 'source_candidate_id' =
            withdrawal.entity_id)),
    'legacy_task_candidate_reviews', (SELECT count(*) FROM run_events review
      WHERE review.entity_type = 'task_candidate'
        AND review.event_type = 'task_candidate_review_recorded'
        AND EXISTS (SELECT 1 FROM tasks task
          WHERE task.season_id = (SELECT id FROM target_season)
            AND task.provenance_json::jsonb ->> 'source_candidate_id' = review.entity_id)),
    'blind_prompt_only_validity_records', (SELECT count(*) FROM run_events review
      WHERE review.entity_type = 'task_candidate'
        AND review.event_type = 'task_candidate_blind_validity_recorded'
        AND review.payload_json::jsonb ->> 'decision' = 'valid'
        AND COALESCE(
          (review.payload_json::jsonb ->> 'author_pack_visible')::boolean, true
        ) IS FALSE
        AND COALESCE(
          (review.payload_json::jsonb ->> 'model_outputs_visible')::boolean, true
        ) IS FALSE
        AND EXISTS (SELECT 1 FROM tasks task
          WHERE task.season_id = (SELECT id FROM target_season)
            AND task.provenance_json::jsonb ->> 'source_candidate_id' = review.entity_id)),
    'independent_reconciliation_records', (SELECT count(*) FROM run_events review
      WHERE review.entity_type = 'task_candidate'
        AND review.event_type = 'task_candidate_reconciliation_recorded'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND COALESCE(
          (review.payload_json::jsonb ->> 'model_outputs_visible')::boolean, true
        ) IS FALSE
        AND EXISTS (SELECT 1 FROM tasks task
          WHERE task.season_id = (SELECT id FROM target_season)
            AND task.provenance_json::jsonb ->> 'source_candidate_id' = review.entity_id)),
    'independent_adjudication_records', (SELECT count(*) FROM run_events review
      WHERE review.entity_type = 'task_candidate'
        AND review.event_type = 'task_candidate_adjudication_recorded'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND COALESCE(
          (review.payload_json::jsonb ->> 'model_outputs_visible')::boolean, true
        ) IS FALSE
        AND COALESCE(
          (review.payload_json::jsonb ->> 'source_reviewer')::boolean, true
        ) IS FALSE
        AND EXISTS (SELECT 1 FROM tasks task
          WHERE task.season_id = (SELECT id FROM target_season)
            AND task.provenance_json::jsonb ->> 'source_candidate_id' = review.entity_id)),
    'validator_contract_human_reviews', (SELECT count(*) FROM run_events review
      JOIN tasks task ON task.id = review.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND review.entity_type = 'task'
        AND review.event_type = 'confirmatory_task_evidence_review_recorded'
        AND review.payload_json::jsonb ->> 'evidence_type' = 'validator_contract'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND COALESCE(
          (review.payload_json::jsonb ->> 'independent_of_task_roles')::boolean,
          false
        )),
    'contamination_audit_human_reviews', (SELECT count(*) FROM run_events review
      JOIN tasks task ON task.id = review.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND review.entity_type = 'task'
        AND review.event_type = 'confirmatory_task_evidence_review_recorded'
        AND review.payload_json::jsonb ->> 'evidence_type' = 'contamination_audit'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND COALESCE(
          (review.payload_json::jsonb ->> 'independent_of_task_roles')::boolean,
          false
        )),
    'candidates_approved_for_bank_assembly', (SELECT count(*) FROM run_events submitted
      WHERE submitted.entity_type = 'task_candidate'
        AND submitted.event_type = 'task_candidate_submitted'
        AND EXISTS (SELECT 1 FROM tasks task
          WHERE task.season_id = (SELECT id FROM target_season)
            AND task.provenance_json::jsonb ->> 'source_candidate_id' = submitted.entity_id)
        AND EXISTS (
          SELECT 1 FROM run_events adjudication
          WHERE adjudication.entity_id = submitted.entity_id
            AND adjudication.event_type = 'task_candidate_adjudication_recorded'
            AND adjudication.payload_json::jsonb ->> 'decision' = 'approve'
            AND adjudication.payload_json::jsonb ->> 'criterion_pack_sha256'
              ~ '^[0-9a-f]{64}$'
        )),
    'task_contributor_accounts_total', (SELECT count(*) FROM expert_reviewers contributor
      WHERE contributor.active IS TRUE
        AND contributor.profile_json::jsonb ->> 'admission_pathway' = 'task_contributor'),
    'current_protocol_task_contributor_accounts', (SELECT count(*)
      FROM expert_reviewers contributor
      WHERE contributor.active IS TRUE
        AND contributor.profile_json::jsonb ->> 'admission_pathway' = 'task_contributor'
        AND contributor.profile_json::jsonb ->> 'task_contributor_status' = 'active'
        AND contributor.profile_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'
        AND contributor.profile_json::jsonb ->> 'task_contributor_protocol_sha256' =
          '__TASK_CONTRIBUTOR_PROTOCOL_SHA256__'
        AND contributor.profile_json::jsonb ->> 'task_contributor_protocol_scope' =
          '__TASK_CONTRIBUTOR_PROTOCOL_SCOPE__'
        AND COALESCE(
          (contributor.profile_json::jsonb ->> 'task_contributor_protocol_accepted')::boolean,
          false
        )
        AND EXISTS (SELECT 1 FROM run_events acceptance
          WHERE acceptance.id = contributor.profile_json::jsonb
              ->> 'task_contributor_protocol_acceptance_event_id'
            AND acceptance.entity_type = 'task_contributor'
            AND acceptance.entity_id = contributor.id
            AND acceptance.event_type = 'task_contributor_protocol_accepted'
            AND acceptance.payload_json::jsonb ->> 'protocol_version' =
              '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'
            AND acceptance.payload_json::jsonb ->> 'protocol_sha256' =
              '__TASK_CONTRIBUTOR_PROTOCOL_SHA256__')),
    'active_anonymous_task_contributors', (SELECT count(DISTINCT contributor.id)
      FROM expert_reviewers contributor
      JOIN tasks task
        ON task.provenance_json::jsonb ->> 'human_author_id' = contributor.id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND contributor.active IS TRUE
        AND contributor.profile_json::jsonb ->> 'admission_pathway' = 'task_contributor'
        AND contributor.profile_json::jsonb ->> 'task_contributor_status' = 'active'
        AND contributor.profile_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'
        AND contributor.profile_json::jsonb ->> 'task_contributor_protocol_sha256' =
          '__TASK_CONTRIBUTOR_PROTOCOL_SHA256__'
        AND task.provenance_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'
        AND task.provenance_json::jsonb ->> 'task_contributor_protocol_sha256' =
          '__TASK_CONTRIBUTOR_PROTOCOL_SHA256__'),
    'verified_unique_task_contributors', (SELECT count(DISTINCT
      contributor.profile_json::jsonb ->> 'person_uniqueness_commitment_sha256')
      FROM expert_reviewers contributor
      JOIN tasks task
        ON task.provenance_json::jsonb ->> 'human_author_id' = contributor.id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND contributor.active IS TRUE
        AND contributor.profile_json::jsonb ->> 'person_uniqueness_verified' = 'true'
        AND contributor.profile_json::jsonb ->> 'person_uniqueness_method' =
          'admin-witnessed-season-hmac-v1'
        AND contributor.profile_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'
        AND contributor.profile_json::jsonb ->> 'task_contributor_protocol_sha256' =
          '__TASK_CONTRIBUTOR_PROTOCOL_SHA256__'
        AND task.provenance_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'
        AND task.provenance_json::jsonb ->> 'task_contributor_protocol_sha256' =
          '__TASK_CONTRIBUTOR_PROTOCOL_SHA256__'
        AND contributor.profile_json::jsonb
          ->> 'person_uniqueness_commitment_sha256' ~ '^[0-9a-f]{64}$'),
    'author_person_binding_tasks', (SELECT count(*) FROM tasks task
      JOIN expert_reviewers contributor
        ON task.provenance_json::jsonb ->> 'human_author_id' = contributor.id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )
        AND task.provenance_json::jsonb
          ->> 'human_author_person_commitment_sha256' =
          contributor.profile_json::jsonb
            ->> 'person_uniqueness_commitment_sha256'
        AND task.provenance_json::jsonb ->> 'task_contributor_protocol_version' =
          '__TASK_CONTRIBUTOR_PROTOCOL_VERSION__'
        AND task.provenance_json::jsonb ->> 'task_contributor_protocol_sha256' =
          '__TASK_CONTRIBUTOR_PROTOCOL_SHA256__'
        AND task.provenance_json::jsonb
          ->> 'task_contributor_protocol_acceptance_event_id' =
          contributor.profile_json::jsonb
            ->> 'task_contributor_protocol_acceptance_event_id'),
    'synthetic', (SELECT count(*) FROM tasks
      WHERE season_id = (SELECT id FROM target_season) AND COALESCE(
      (provenance_json::jsonb ->> 'synthetic')::boolean, false
    )),
    'independent_task_approvals', (SELECT count(*) FROM run_events review
      JOIN tasks task ON task.id = review.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND review.entity_type = 'task'
        AND review.event_type = 'confirmatory_task_review_recorded'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND COALESCE(
          (review.payload_json::jsonb ->> 'independent_of_author')::boolean, false
        )),
    'task_review_evidence_store_available', COALESCE((SELECT
      count(*) > 0
        AND bool_and(
        task.provenance_json::jsonb ->> 'evidence_registry_status' = 'verified'
        AND task.provenance_json::jsonb ->> 'task_evidence_root_sha256'
          ~ '^[0-9a-f]{64}$'
        AND (SELECT count(*) FROM task_evidence_artifacts evidence
          WHERE evidence.task_id = task.id AND evidence.revision_ordinal = 1) = 2
        AND EXISTS (SELECT 1 FROM task_evidence_artifacts validator
          WHERE validator.task_id = task.id
            AND validator.evidence_type = 'validator_contract'
            AND validator.artifact_sha256 =
              task.provenance_json::jsonb ->> 'validator_contract_sha256')
        AND EXISTS (SELECT 1 FROM task_evidence_artifacts audit
          WHERE audit.task_id = task.id
            AND audit.evidence_type = 'contamination_audit'
            AND audit.artifact_sha256 =
              task.provenance_json::jsonb ->> 'contamination_audit_sha256')
        AND (SELECT count(*) FROM run_events evidence_review
          WHERE evidence_review.entity_type = 'task'
            AND evidence_review.entity_id = task.id
            AND evidence_review.event_type =
              'confirmatory_task_evidence_review_recorded') = 2
        AND EXISTS (SELECT 1 FROM run_events evidence_review
          WHERE evidence_review.entity_type = 'task'
            AND evidence_review.entity_id = task.id
            AND evidence_review.event_type =
              'confirmatory_task_evidence_review_recorded'
            AND evidence_review.payload_json::jsonb ->> 'evidence_type' =
              'validator_contract'
            AND evidence_review.payload_json::jsonb ->> 'artifact_sha256' =
              task.provenance_json::jsonb ->> 'validator_contract_sha256'
            AND evidence_review.payload_json::jsonb ->> 'review_event_sha256' =
              task.provenance_json::jsonb -> 'validator_contract_review'
                ->> 'review_event_sha256'
            AND COALESCE((evidence_review.payload_json::jsonb
              ->> 'independent_of_task_roles')::boolean, false))
        AND EXISTS (SELECT 1 FROM run_events evidence_review
          WHERE evidence_review.entity_type = 'task'
            AND evidence_review.entity_id = task.id
            AND evidence_review.event_type =
              'confirmatory_task_evidence_review_recorded'
            AND evidence_review.payload_json::jsonb ->> 'evidence_type' =
              'contamination_audit'
            AND evidence_review.payload_json::jsonb ->> 'artifact_sha256' =
              task.provenance_json::jsonb ->> 'contamination_audit_sha256'
            AND evidence_review.payload_json::jsonb ->> 'review_event_sha256' =
              task.provenance_json::jsonb -> 'contamination_audit_review'
                ->> 'review_event_sha256'
            AND COALESCE((evidence_review.payload_json::jsonb
              ->> 'independent_of_task_roles')::boolean, false))
        AND EXISTS (SELECT 1 FROM run_events calibration
          WHERE calibration.entity_type = 'season'
            AND calibration.entity_id = task.season_id
            AND calibration.event_type = 'confirmatory_task_bank_imported'
            AND calibration.payload_json::jsonb ->> 'validator_calibration_status'
              = 'verified'
            AND calibration.payload_json::jsonb
              ->> 'validator_calibration_artifact_sha256' =
              task.provenance_json::jsonb
                ->> 'validator_calibration_artifact_sha256'
            AND calibration.payload_json::jsonb
              ->> 'contamination_calibration_status' = 'verified'
            AND calibration.payload_json::jsonb
              ->> 'contamination_calibration_artifact_sha256' =
              task.provenance_json::jsonb
                ->> 'contamination_calibration_artifact_sha256')
        AND NOT EXISTS (SELECT 1 FROM task_evidence_artifacts correction
          JOIN task_evidence_artifacts prior
            ON prior.id = correction.supersedes_artifact_id
          WHERE prior.task_id = task.id)
      )
      FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )), false),
    'construct_blueprint_verified', COALESCE((SELECT
      count(*) = 240
      AND count(DISTINCT task.provenance_json::jsonb ->> 'construct_blueprint_sha256') = 1
      AND bool_and(
        task.provenance_json::jsonb ->> 'construct_blueprint_sha256'
          ~ '^[0-9a-f]{64}$'
        AND task.provenance_json::jsonb ->> 'construct_cell_id' IS NOT NULL
        AND task.provenance_json::jsonb ->> 'difficulty_tier' IS NOT NULL
      )
      FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )), false),
    'construct_blueprint_sha256', (SELECT min(
      task.provenance_json::jsonb ->> 'construct_blueprint_sha256')
      FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )),
    'criterion_pack_tasks', (SELECT count(*) FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )
        AND task.provenance_json::jsonb ->> 'criterion_pack_sha256'
          ~ '^[0-9a-f]{64}$'
        AND task.provenance_json::jsonb -> 'criterion_pack' IS NOT NULL),
    'contamination_replay_tasks', (SELECT count(*) FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )
        AND task.provenance_json::jsonb ->> 'contamination_scan_bundle_sha256'
          ~ '^[0-9a-f]{64}$'
        AND task.provenance_json::jsonb ->> 'contamination_audit_status' = 'pass'),
    'contamination_replay_bundle_count', (SELECT count(DISTINCT
      task.provenance_json::jsonb ->> 'contamination_scan_bundle_sha256') FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )
        AND task.provenance_json::jsonb ->> 'contamination_scan_bundle_sha256'
          ~ '^[0-9a-f]{64}$'),
    'surface_diagnostic_tasks', (SELECT count(*) FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'objective_validator_possible')::boolean,
          false
        )),
    'sealed_task_lifecycles', (SELECT count(*) FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )
        AND task.provenance_json::jsonb ->> 'authored_at' IS NOT NULL
        AND task.provenance_json::jsonb ->> 'sealed_at' IS NOT NULL
        AND task.provenance_json::jsonb ->> 'task_lifecycle_seal_sha256'
          ~ '^[0-9a-f]{64}$'
        AND (SELECT count(*) FROM run_events authored
          WHERE authored.entity_type = 'task' AND authored.entity_id = task.id
            AND authored.event_type = 'confirmatory_task_authorship_recorded') = 1
        AND (SELECT count(*) FROM run_events sealed
          WHERE sealed.entity_type = 'task' AND sealed.entity_id = task.id
            AND sealed.event_type = 'confirmatory_task_sealed'
            AND sealed.payload_json::jsonb ->> 'lifecycle_seal_sha256' =
              task.provenance_json::jsonb ->> 'task_lifecycle_seal_sha256') = 1),
    'first_used_task_lifecycles', (SELECT count(*) FROM tasks task
      WHERE task.season_id = (SELECT id FROM target_season)
        AND COALESCE(
          (task.provenance_json::jsonb ->> 'season1_eligible')::boolean, false
        )
        AND EXISTS (SELECT 1 FROM run_events first_use
          WHERE first_use.entity_type = 'task' AND first_use.entity_id = task.id
            AND first_use.event_type = 'confirmatory_task_first_used')),
    'preseal_task_uses', (SELECT count(*) FROM run_events first_use
      JOIN tasks task ON task.id = first_use.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND first_use.entity_type = 'task'
        AND first_use.event_type = 'confirmatory_task_first_used'
        AND first_use.created_at < task.created_at),
    'retired_confirmatory_tasks', (SELECT count(*) FROM run_events retirement
      JOIN tasks task ON task.id = retirement.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND retirement.entity_type = 'task'
        AND retirement.event_type = 'confirmatory_task_retired'),
    'required_surface_diagnostic_tasks', 96,
    'validator_calibration_verified', COALESCE((SELECT bool_and(
      event.payload_json::jsonb ->> 'validator_calibration_status' = 'verified'
      AND event.payload_json::jsonb ->> 'validator_calibration_artifact_sha256'
        ~ '^[0-9a-f]{64}$'
      AND event.payload_json::jsonb ->> 'validator_calibration_receipt_sha256'
        ~ '^[0-9a-f]{64}$'
      AND (event.payload_json::jsonb ->> 'validator_calibration_case_count')::int >= 120
    ) FROM run_events event
      WHERE event.entity_type = 'season'
        AND event.entity_id = (SELECT id FROM target_season)
        AND event.event_type = 'confirmatory_task_bank_imported'), false),
    'validator_calibration_artifact_sha256', (SELECT min(
      event.payload_json::jsonb ->> 'validator_calibration_artifact_sha256')
      FROM run_events event
      WHERE event.entity_type = 'season'
        AND event.entity_id = (SELECT id FROM target_season)
        AND event.event_type = 'confirmatory_task_bank_imported'),
    'contamination_calibration_verified', COALESCE((SELECT bool_and(
      event.payload_json::jsonb ->> 'contamination_calibration_status' = 'verified'
      AND event.payload_json::jsonb ->> 'contamination_calibration_artifact_sha256'
        ~ '^[0-9a-f]{64}$'
      AND event.payload_json::jsonb ->> 'contamination_calibration_receipt_sha256'
        ~ '^[0-9a-f]{64}$'
      AND (event.payload_json::jsonb ->> 'contamination_calibration_case_count')::int >= 150
      AND (event.payload_json::jsonb
        ->> 'contamination_calibration_precision_milli')::int >= 950
      AND (event.payload_json::jsonb
        ->> 'contamination_calibration_recall_milli')::int >= 900
      AND (event.payload_json::jsonb
        ->> 'contamination_calibration_paraphrase_recall_milli')::int >= 850
    ) FROM run_events event
      WHERE event.entity_type = 'season'
        AND event.entity_id = (SELECT id FROM target_season)
        AND event.event_type = 'confirmatory_task_bank_imported'), false),
    'contamination_calibration_artifact_sha256', (SELECT min(
      event.payload_json::jsonb ->> 'contamination_calibration_artifact_sha256')
      FROM run_events event
      WHERE event.entity_type = 'season'
        AND event.entity_id = (SELECT id FROM target_season)
        AND event.event_type = 'confirmatory_task_bank_imported'),
    'contamination_calibration_case_count', (SELECT min(
      (event.payload_json::jsonb ->> 'contamination_calibration_case_count')::int)
      FROM run_events event
      WHERE event.entity_type = 'season'
        AND event.entity_id = (SELECT id FROM target_season)
        AND event.event_type = 'confirmatory_task_bank_imported'),
    'contamination_calibration_precision_milli', (SELECT min(
      (event.payload_json::jsonb ->> 'contamination_calibration_precision_milli')::int)
      FROM run_events event
      WHERE event.entity_type = 'season'
        AND event.entity_id = (SELECT id FROM target_season)
        AND event.event_type = 'confirmatory_task_bank_imported'),
    'contamination_calibration_recall_milli', (SELECT min(
      (event.payload_json::jsonb ->> 'contamination_calibration_recall_milli')::int)
      FROM run_events event
      WHERE event.entity_type = 'season'
        AND event.entity_id = (SELECT id FROM target_season)
        AND event.event_type = 'confirmatory_task_bank_imported'),
    'contamination_calibration_paraphrase_recall_milli', (SELECT min(
      (event.payload_json::jsonb
        ->> 'contamination_calibration_paraphrase_recall_milli')::int)
      FROM run_events event
      WHERE event.entity_type = 'season'
        AND event.entity_id = (SELECT id FROM target_season)
        AND event.event_type = 'confirmatory_task_bank_imported')
  ),
  'independent_reviewers_by_family', jsonb_build_object(
    'substitution', (SELECT count(DISTINCT reviewer.id) FROM expert_reviewers reviewer
      JOIN run_events review ON review.payload_json::jsonb ->> 'reviewer_id' = reviewer.id
      JOIN tasks task ON task.id = review.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND task.family = 'substitution' AND review.entity_type = 'task'
        AND review.event_type = 'confirmatory_task_review_recorded'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND reviewer.active IS TRUE AND reviewer.qualification_verified IS TRUE
        AND reviewer.cohort = 'expert_independent'
        AND reviewer.qualification_json::jsonb ? 'substitution'),
    'composition', (SELECT count(DISTINCT reviewer.id) FROM expert_reviewers reviewer
      JOIN run_events review ON review.payload_json::jsonb ->> 'reviewer_id' = reviewer.id
      JOIN tasks task ON task.id = review.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND task.family = 'composition' AND review.entity_type = 'task'
        AND review.event_type = 'confirmatory_task_review_recorded'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND reviewer.active IS TRUE AND reviewer.qualification_verified IS TRUE
        AND reviewer.cohort = 'expert_independent'
        AND reviewer.qualification_json::jsonb ? 'composition'),
    'cookability', (SELECT count(DISTINCT reviewer.id) FROM expert_reviewers reviewer
      JOIN run_events review ON review.payload_json::jsonb ->> 'reviewer_id' = reviewer.id
      JOIN tasks task ON task.id = review.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND task.family = 'cookability' AND review.entity_type = 'task'
        AND review.event_type = 'confirmatory_task_review_recorded'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND reviewer.active IS TRUE AND reviewer.qualification_verified IS TRUE
        AND reviewer.cohort = 'expert_independent'
        AND reviewer.qualification_json::jsonb ? 'cookability'),
    'evidence', (SELECT count(DISTINCT reviewer.id) FROM expert_reviewers reviewer
      JOIN run_events review ON review.payload_json::jsonb ->> 'reviewer_id' = reviewer.id
      JOIN tasks task ON task.id = review.entity_id
      WHERE task.season_id = (SELECT id FROM target_season)
        AND task.family = 'evidence' AND review.entity_type = 'task'
        AND review.event_type = 'confirmatory_task_review_recorded'
        AND review.payload_json::jsonb ->> 'decision' = 'approve'
        AND reviewer.active IS TRUE AND reviewer.qualification_verified IS TRUE
        AND reviewer.cohort = 'expert_independent'
        AND reviewer.qualification_json::jsonb ? 'evidence')
  ),
  'provider_budgets', COALESCE((
    SELECT jsonb_object_agg(execution_backend, jsonb_build_object(
      'currency', currency,
      'cap_micros', budget_cap_micros,
      'used_micros', budget_used_micros,
      'reserved_micros', budget_reserved_micros,
      'valid_until', valid_until,
      'account_authorization_bound',
        account_authorization_envelope_sha256 NOT IN ('unresolved', 'unfrozen')
    ))
    FROM season_provider_budgets
    WHERE season_id = (SELECT id FROM seasons WHERE slug = 'season-1')
  ), '{}'::jsonb),
  'research_consent', jsonb_build_object(
    'consented_battles', (SELECT count(*) FROM battles
      WHERE season_id = (SELECT id FROM target_season) AND research_consent IS TRUE),
    'nonconsented_battles', (SELECT count(*) FROM battles
      WHERE season_id = (SELECT id FROM target_season) AND research_consent IS FALSE),
    'release_approved_battles',
      (SELECT count(*) FROM battles
       WHERE season_id = (SELECT id FROM target_season)
         AND release_review_status = 'approved')
  ),
  'development_task_validation', jsonb_build_object(
    'packet_sha256', '__DEVELOPMENT_VALIDATION_PACKET_SHA256__',
    'blind_validity_records', (SELECT count(*) FROM run_events event
      WHERE event.entity_type = 'development_task_validation'
        AND event.event_type = 'development_task_blind_validity_recorded'
        AND event.payload_json::jsonb ->> 'packet_sha256' =
          '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'
        AND COALESCE(
          (event.payload_json::jsonb ->> 'independent_review')::boolean,
          false
        )),
    'criterion_pack_records', (SELECT count(*) FROM run_events event
      WHERE event.entity_type = 'development_task_validation'
        AND event.event_type = 'development_task_criteria_recorded'
        AND event.payload_json::jsonb ->> 'packet_sha256' =
          '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'
        AND COALESCE(
          (event.payload_json::jsonb ->> 'independent_review')::boolean,
          false
        )),
    'adjudication_records', (SELECT count(*) FROM run_events event
      WHERE event.entity_type = 'development_task_validation'
        AND event.event_type = 'development_task_adjudication_recorded'
        AND event.payload_json::jsonb ->> 'packet_sha256' =
          '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'),
    'distinct_independent_reviewers', (SELECT count(DISTINCT
      event.payload_json::jsonb ->> 'reviewer_id') FROM run_events event
      WHERE event.entity_type = 'development_task_validation'
        AND event.payload_json::jsonb ->> 'packet_sha256' =
          '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'
        AND COALESCE(
          (event.payload_json::jsonb ->> 'independent_review')::boolean,
          false
        )),
    'tasks_with_three_complete_reviews', (SELECT count(*) FROM (
      SELECT blind.entity_id
      FROM run_events blind
      WHERE blind.entity_type = 'development_task_validation'
        AND blind.event_type = 'development_task_blind_validity_recorded'
        AND blind.payload_json::jsonb ->> 'packet_sha256' =
          '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'
        AND COALESCE(
          (blind.payload_json::jsonb ->> 'independent_review')::boolean,
          false
        )
        AND (
          blind.payload_json::jsonb ->> 'decision' <> 'valid'
          OR EXISTS (SELECT 1 FROM run_events criteria
            WHERE criteria.entity_type = 'development_task_validation'
              AND criteria.entity_id = blind.entity_id
              AND criteria.event_type = 'development_task_criteria_recorded'
              AND criteria.payload_json::jsonb ->> 'packet_sha256' =
                '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'
              AND criteria.payload_json::jsonb ->> 'reviewer_id' =
                blind.payload_json::jsonb ->> 'reviewer_id'
              AND COALESCE(
                (criteria.payload_json::jsonb ->> 'independent_review')::boolean,
                false
              ))
        )
      GROUP BY blind.entity_id
      HAVING count(DISTINCT blind.payload_json::jsonb ->> 'reviewer_id') = 3
    ) completed),
    'unanimously_valid_tasks', (SELECT count(*) FROM (
      SELECT blind.entity_id
      FROM run_events blind
      WHERE blind.entity_type = 'development_task_validation'
        AND blind.event_type = 'development_task_blind_validity_recorded'
        AND blind.payload_json::jsonb ->> 'packet_sha256' =
          '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'
        AND blind.payload_json::jsonb ->> 'decision' = 'valid'
        AND COALESCE(
          (blind.payload_json::jsonb ->> 'independent_review')::boolean,
          false
        )
        AND EXISTS (SELECT 1 FROM run_events criteria
          WHERE criteria.entity_type = 'development_task_validation'
            AND criteria.entity_id = blind.entity_id
            AND criteria.event_type = 'development_task_criteria_recorded'
            AND criteria.payload_json::jsonb ->> 'packet_sha256' =
              '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'
            AND criteria.payload_json::jsonb ->> 'reviewer_id' =
              blind.payload_json::jsonb ->> 'reviewer_id'
            AND COALESCE(
              (criteria.payload_json::jsonb ->> 'independent_review')::boolean,
              false
            ))
      GROUP BY blind.entity_id
      HAVING count(DISTINCT blind.payload_json::jsonb ->> 'reviewer_id') = 3
    ) unanimous),
    'adjudicated_valid_tasks', (SELECT count(DISTINCT event.entity_id)
      FROM run_events event
      WHERE event.entity_type = 'development_task_validation'
        AND event.event_type = 'development_task_adjudication_recorded'
        AND event.payload_json::jsonb ->> 'packet_sha256' =
          '__DEVELOPMENT_VALIDATION_PACKET_SHA256__'
        AND event.payload_json::jsonb ->> 'decision' = 'valid')
  ),
  'fixture_rows', jsonb_build_object(
    'catalog_models', (SELECT count(*) FROM catalog_models
      WHERE model_id LIKE 'flavourbench/mock/%' OR model_id LIKE 'flavourbench/mock-%'),
    'season_models', (SELECT count(*) FROM season_models
      WHERE season_id = (SELECT id FROM target_season)
        AND (execution_backend = 'mock' OR provider_slug = 'mock')),
    'battles', (SELECT count(*) FROM battles
      WHERE season_id = (SELECT id FROM target_season) AND run_class = 'mock')
  ),
  'leaderboard_snapshots', COALESCE((
    SELECT jsonb_agg(jsonb_build_object(
      'id', snapshot.id,
      'track', snapshot.track,
      'cohort', snapshot.cohort,
      'category', snapshot.category,
      'data_stratum', snapshot.data_stratum,
      'controlled_run_id', snapshot.controlled_run_id,
      'publication_status', snapshot.publication_status,
      'input_sha256', snapshot.input_sha256,
      'input_evidence_sha256', snapshot.input_evidence_sha256,
      'input_evidence_json', snapshot.input_evidence_json,
      'payload_sha256', snapshot.payload_sha256,
      'payload_json', snapshot.payload_json,
      'evidence_cutoff_at', snapshot.evidence_cutoff_at,
      'created_at', snapshot.created_at
    ) ORDER BY snapshot.created_at, snapshot.id)
    FROM leaderboard_snapshots snapshot
    WHERE snapshot.season_id = (SELECT id FROM target_season)
      AND snapshot.publication_status IN ('draft', 'published')
  ), '[]'::jsonb),
  'research_release_archives', COALESCE((
    SELECT jsonb_agg(jsonb_build_object(
      'id', archive.id,
      'season_id', archive.season_id,
      'archive_class', archive.archive_class,
      'schema_version', archive.schema_version,
      'snapshot_ids_json', archive.snapshot_ids_json,
      'snapshot_set_sha256', archive.snapshot_set_sha256,
      'manifest_json', archive.manifest_json,
      'manifest_sha256', archive.manifest_sha256,
      'archive_sha256', archive.archive_sha256,
      'storage_object_key', archive.storage_object_key,
      'size_bytes', archive.size_bytes,
      'member_count', archive.member_count,
      'source_date_epoch', archive.source_date_epoch,
      'requirements_lock_sha256', archive.requirements_lock_sha256,
      'build_image_digest', archive.build_image_digest,
      'signature_algorithm', archive.signature_algorithm,
      'signing_key_id', archive.signing_key_id,
      'public_key_pem', archive.public_key_pem,
      'public_key_sha256', archive.public_key_sha256,
      'signature_base64', archive.signature_base64,
      'created_at', archive.created_at
    ) ORDER BY archive.created_at, archive.id)
    FROM research_release_archives archive
    WHERE archive.season_id = (SELECT id FROM target_season)
  ), '[]'::jsonb)
);
"""

CONTROL_PLANE_SQL = (
    _CONTROL_PLANE_SQL_TEMPLATE.replace(
        "__TASK_CONTRIBUTOR_PROTOCOL_VERSION__",
        TASK_CONTRIBUTOR_PROTOCOL_VERSION,
    )
    .replace(
        "__TASK_CONTRIBUTOR_PROTOCOL_SHA256__",
        TASK_CONTRIBUTOR_PROTOCOL_SHA256,
    )
    .replace(
        "__TASK_CONTRIBUTOR_PROTOCOL_SCOPE__",
        TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
    )
    .replace(
        "__CONSTRUCT_BLUEPRINT_SHA256__",
        BLUEPRINT_SHA256,
    )
    .replace(
        "__DEVELOPMENT_VALIDATION_PACKET_SHA256__",
        DEVELOPMENT_VALIDATION_PACKET_SHA256,
    )
)


class Season1ReadinessError(RuntimeError):
    """The Season 1 readiness snapshot could not be built safely."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Season1ReadinessError(f"{path} does not contain a JSON object")
    return value


def verify_embedded_digest(value: dict[str, Any], field: str = "artifact_sha256") -> bool:
    embedded = value.get(field)
    if not isinstance(embedded, str) or len(embedded) != 64:
        return False
    payload = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest() == embedded


def _evidence_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _evidence_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        return None
    return value


def _evidence_sha256(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return value
    return None


def _valid_task_family_map(value: object, *, task_count: int) -> dict[str, str] | None:
    families = {"composition", "cookability", "evidence", "substitution"}
    if not isinstance(value, dict) or len(value) != task_count:
        return None
    normalized: dict[str, str] = {}
    for task_id, family in value.items():
        normalized_task_id = _evidence_id(task_id)
        if normalized_task_id is None or family not in families:
            return None
        normalized[normalized_task_id] = str(family)
    expected_per_family = task_count // len(families)
    if any(list(normalized.values()).count(family) != expected_per_family for family in families):
        return None
    return normalized


def _valid_robustness_common(
    value: dict[str, Any],
    *,
    schema_version: str,
    study_design_sha256: str,
) -> bool:
    return bool(
        verify_embedded_digest(value)
        and value.get("schema_version") == schema_version
        and value.get("status") == "complete"
        and value.get("study_design_artifact_sha256") == study_design_sha256
        and _evidence_int(value.get("synthetic_observations")) == 0
    )


def valid_post_collection_item_audit(
    value: dict[str, Any], *, study_design_sha256: str
) -> bool:
    counts = value.get("counts")
    records = value.get("task_records")
    if not isinstance(counts, dict) or not isinstance(records, list):
        return False
    task_ids: set[str] = set()
    random_tasks = 0
    anomaly_tasks = 0
    confirmed_defect_records = 0
    for record in records:
        if not isinstance(record, dict):
            return False
        task_id = _evidence_id(record.get("task_id"))
        task_sha256 = _evidence_sha256(record.get("task_content_sha256"))
        reasons = record.get("selection_reasons")
        auditors = record.get("auditor_commitments_sha256")
        material_defect = record.get("material_defect")
        if not (
            task_id is not None
            and task_id not in task_ids
            and task_sha256 is not None
            and isinstance(reasons, list)
            and reasons
            and set(reasons).issubset({"random", "anomaly"})
            and len(reasons) == len(set(reasons))
            and isinstance(auditors, list)
            and len(auditors) >= 2
            and len(auditors) == len(set(auditors))
            and all(_evidence_sha256(auditor) is not None for auditor in auditors)
            and isinstance(material_defect, bool)
        ):
            return False
        task_ids.add(task_id)
        random_tasks += int("random" in reasons)
        anomaly_tasks += int("anomaly" in reasons)
        if material_defect:
            confirmed_defect_records += 1
            if not (
                record.get("resolution_status") == "retired_and_snapshots_recomputed"
                and _evidence_sha256(record.get("challenge_artifact_sha256")) is not None
                and _evidence_sha256(record.get("retirement_event_sha256")) is not None
                and _evidence_sha256(record.get("snapshot_recomputation_artifact_sha256"))
                is not None
            ):
                return False
        elif record.get("resolution_status") != "no_material_defect":
            return False
    anomaly_flagged = _evidence_int(counts.get("anomaly_flagged_tasks"))
    confirmed_defects = _evidence_int(counts.get("confirmed_material_defects"))
    return bool(
        _valid_robustness_common(
            value,
            schema_version="flavourbench-season1-post-collection-item-audit-v1",
            study_design_sha256=study_design_sha256,
        )
        and _evidence_int(counts.get("population_tasks"), 0) == 240
        and _evidence_int(counts.get("random_tasks_audited"), 0) >= 60
        and _evidence_int(counts.get("random_tasks_audited")) == random_tasks
        and anomaly_flagged >= 0
        and _evidence_int(counts.get("anomaly_flagged_tasks_audited")) == anomaly_flagged
        and anomaly_flagged == anomaly_tasks
        and _evidence_int(counts.get("unique_tasks_audited")) == len(task_ids)
        and _evidence_int(counts.get("minimum_independent_auditors_per_task"), 0) >= 2
        and _evidence_int(counts.get("unresolved_material_defects")) == 0
        and confirmed_defects >= 0
        and confirmed_defects == confirmed_defect_records
        and _evidence_int(counts.get("retired_material_defects")) == confirmed_defects
        and value.get("sampling_seed_committed_before_model_results") is True
        and value.get("original_task_roles_excluded") is True
        and value.get("affected_snapshots_recomputed") is True
    )


def valid_generation_reliability_panel(
    value: dict[str, Any], *, study_design_sha256: str
) -> bool:
    counts = value.get("counts")
    records = value.get("cell_records")
    task_families = _valid_task_family_map(value.get("task_families"), task_count=20)
    if not isinstance(counts, dict) or not isinstance(records, list) or task_families is None:
        return False
    cells: set[tuple[str, str, str]] = set()
    arm_ids: set[str] = set()
    endpoint_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            return False
        task_id = _evidence_id(record.get("task_id"))
        endpoint_id = _evidence_id(record.get("endpoint_id"))
        condition = record.get("condition")
        observed_arm_ids = record.get("arm_ids")
        retry_attempt_ids = record.get("provider_retry_attempt_ids")
        if not (
            task_id in task_families
            and endpoint_id is not None
            and condition in {"epicure_off", "epicure_on"}
            and isinstance(observed_arm_ids, list)
            and len(observed_arm_ids) == 3
            and len(observed_arm_ids) == len(set(observed_arm_ids))
            and all(_evidence_id(arm_id) is not None for arm_id in observed_arm_ids)
            and isinstance(retry_attempt_ids, list)
            and len(retry_attempt_ids) == len(set(retry_attempt_ids))
            and all(_evidence_id(attempt_id) is not None for attempt_id in retry_attempt_ids)
        ):
            return False
        cell = (str(task_id), endpoint_id, str(condition))
        if cell in cells or any(str(arm_id) in arm_ids for arm_id in observed_arm_ids):
            return False
        cells.add(cell)
        endpoint_ids.add(endpoint_id)
        arm_ids.update(str(arm_id) for arm_id in observed_arm_ids)
    return bool(
        _valid_robustness_common(
            value,
            schema_version="flavourbench-season1-generation-reliability-panel-v1",
            study_design_sha256=study_design_sha256,
        )
        and _evidence_int(counts.get("tasks"), 0) == 20
        and _evidence_int(counts.get("endpoints"), 0) == 16
        and _evidence_int(counts.get("conditions"), 0) == 2
        and _evidence_int(counts.get("independent_generations_per_cell"), 0) == 3
        and _evidence_int(counts.get("response_arms"), 0) == 1920
        and _evidence_int(counts.get("incremental_response_arms"), 0) == 1280
        and len(records) == 20 * 16 * 2
        and len(cells) == 20 * 16 * 2
        and len(endpoint_ids) == 16
        and len(arm_ids) == 1920
        and value.get("retries_counted_as_repetitions") is False
        and value.get("ranking_use") == "reported-separately-not-pooled"
        and _evidence_sha256(value.get("metrics_artifact_sha256")) is not None
    )


def valid_prompt_sensitivity_audit(
    value: dict[str, Any], *, study_design_sha256: str
) -> bool:
    counts = value.get("counts")
    records = value.get("arm_records")
    task_families = _valid_task_family_map(value.get("task_families"), task_count=20)
    if not isinstance(counts, dict) or not isinstance(records, list) or task_families is None:
        return False
    cells: set[tuple[str, str, str]] = set()
    arm_ids: set[str] = set()
    endpoint_ids: set[str] = set()
    prompt_variants: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            return False
        task_id = _evidence_id(record.get("task_id"))
        endpoint_id = _evidence_id(record.get("endpoint_id"))
        prompt_sha256 = _evidence_sha256(record.get("prompt_variant_sha256"))
        arm_id = _evidence_id(record.get("arm_id"))
        if not (
            task_id in task_families
            and endpoint_id is not None
            and prompt_sha256 is not None
            and arm_id is not None
            and record.get("rank_eligible") is False
        ):
            return False
        cell = (str(task_id), endpoint_id, prompt_sha256)
        if cell in cells or arm_id in arm_ids:
            return False
        cells.add(cell)
        arm_ids.add(arm_id)
        endpoint_ids.add(endpoint_id)
        prompt_variants.add(prompt_sha256)
    return bool(
        _valid_robustness_common(
            value,
            schema_version="flavourbench-season1-prompt-sensitivity-audit-v1",
            study_design_sha256=study_design_sha256,
        )
        and value.get("split") == "development"
        and _evidence_int(counts.get("tasks"), 0) == 20
        and _evidence_int(counts.get("endpoints"), 0) == 8
        and _evidence_int(counts.get("prompt_variants"), 0) == 3
        and _evidence_int(counts.get("response_arms"), 0) == 480
        and len(records) == 20 * 8 * 3
        and len(cells) == 20 * 8 * 3
        and len(arm_ids) == 480
        and len(endpoint_ids) == 8
        and len(prompt_variants) == 3
        and value.get("selection_after_results") is False
        and value.get("rank_eligible") is False
        and _evidence_sha256(value.get("metrics_artifact_sha256")) is not None
    )


def valid_practical_cookability_execution(
    value: dict[str, Any], *, study_design_sha256: str
) -> bool:
    counts = value.get("counts")
    records = value.get("execution_records")
    if not isinstance(counts, dict) or not isinstance(records, list):
        return False
    execution_ids: set[str] = set()
    task_ids: set[str] = set()
    output_cooks: dict[str, set[str]] = {}
    output_tasks: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        execution_id = _evidence_id(record.get("execution_id"))
        task_id = _evidence_id(record.get("task_id"))
        output_id = _evidence_id(record.get("output_id"))
        cook_commitment = _evidence_sha256(record.get("cook_commitment_sha256"))
        elapsed_seconds = _evidence_int(record.get("elapsed_seconds"), 0)
        deviations = record.get("instruction_deviations")
        acceptability = _evidence_int(record.get("blinded_acceptability"), 0)
        if not (
            execution_id is not None
            and execution_id not in execution_ids
            and task_id is not None
            and output_id is not None
            and cook_commitment is not None
            and record.get("model_and_condition_blinded") is True
            and isinstance(record.get("completed"), bool)
            and elapsed_seconds > 0
            and isinstance(deviations, list)
            and all(isinstance(item, str) for item in deviations)
            and record.get("yield_recorded") is True
            and 1 <= acceptability <= 5
        ):
            return False
        prior_task = output_tasks.setdefault(output_id, task_id)
        if prior_task != task_id:
            return False
        execution_ids.add(execution_id)
        task_ids.add(task_id)
        output_cooks.setdefault(output_id, set()).add(cook_commitment)
    return bool(
        _valid_robustness_common(
            value,
            schema_version="flavourbench-season1-practical-cookability-execution-v1",
            study_design_sha256=study_design_sha256,
        )
        and _evidence_int(counts.get("tasks"), 0) == 24
        and _evidence_int(counts.get("outputs"), 0) == 24
        and _evidence_int(counts.get("independent_cooks_per_output"), 0) == 2
        and _evidence_int(counts.get("kitchen_executions"), 0) == 48
        and len(records) == 48
        and len(execution_ids) == 48
        and len(task_ids) == 24
        and len(output_cooks) == 24
        and all(len(cooks) == 2 for cooks in output_cooks.values())
        and value.get("model_and_condition_blinded") is True
        and value.get("ranking_use") == "construct-validity-only-not-pooled"
        and _evidence_sha256(value.get("output_selection_artifact_sha256")) is not None
        and _evidence_sha256(value.get("rubric_association_artifact_sha256")) is not None
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _archive_canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _enrich_archive_verification(database: dict[str, Any]) -> None:
    rows = database.get("research_release_archives", [])
    if not isinstance(rows, list):
        database["research_release_archives"] = []
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata_signature_valid = False
        try:
            public_key = load_pem_public_key(str(row["public_key_pem"]).encode("ascii"))
            signature = base64.b64decode(str(row["signature_base64"]), validate=True)
            if not isinstance(public_key, Ed25519PublicKey):
                raise ValueError("not an Ed25519 public key")
            public_key.verify(
                signature,
                RESEARCH_ARCHIVE_SIGNATURE_CONTEXT + bytes.fromhex(str(row["archive_sha256"])),
            )
            metadata_signature_valid = True
        except (InvalidSignature, KeyError, TypeError, ValueError):
            metadata_signature_valid = False
        file_verification: dict[str, Any] = {}
        path = Path(str(row.get("storage_object_key", "")))
        if metadata_signature_valid and path.is_file():
            try:
                file_verification = verify_archive(
                    archive_path=path,
                    signature_base64=str(row["signature_base64"]),
                    public_key_pem=str(row["public_key_pem"]),
                    expected_archive_sha256=str(row["archive_sha256"]),
                )
            except (OSError, ResearchReleaseError):
                file_verification = {}
        row["verification"] = {
            "metadata_signature_valid": metadata_signature_valid,
            "archive_file_verified": bool(file_verification.get("signature_valid")),
            "inventory_valid": bool(file_verification.get("inventory_valid")),
            "reproducible_metadata_valid": bool(
                file_verification.get("reproducible_metadata_valid")
            ),
            "verified_manifest_sha256": file_verification.get("manifest_sha256"),
        }


def valid_statistical_snapshot(value: dict[str, Any]) -> bool:
    payload = value.get("payload_json")
    evidence = value.get("input_evidence_json")
    if not isinstance(payload, dict) or not isinstance(evidence, dict):
        return False
    observations = evidence.get("analysis_observations")
    payload_ids = payload.get("preference_observation_ids")
    evidence_ids = (
        observations.get("preference_observation_ids") if isinstance(observations, dict) else None
    )
    rows = payload.get("rows")
    acceptance = payload.get("statistical_acceptance")
    expected_observation_sha256 = (
        _canonical_sha256({"vote_ids": payload_ids}) if isinstance(payload_ids, list) else None
    )
    return bool(
        value.get("publication_status") in {"draft", "published"}
        and value.get("data_stratum") == "controlled"
        and value.get("category") == "all"
        and isinstance(value.get("controlled_run_id"), str)
        and value.get("track") in {"model_arena", "epicure_uplift"}
        and value.get("cohort") in {"public", "expert_independent"}
        and value.get("input_sha256") == _canonical_sha256(payload)
        and value.get("payload_sha256") == _canonical_sha256(payload)
        and value.get("input_evidence_sha256") == _canonical_sha256(evidence)
        and payload.get("track") == value.get("track")
        and payload.get("cohort") == value.get("cohort")
        and payload.get("category") == value.get("category")
        and payload.get("data_stratum") == value.get("data_stratum")
        and payload.get("controlled_run_id") == value.get("controlled_run_id")
        and isinstance(payload_ids, list)
        and payload_ids == sorted(set(payload_ids))
        and payload_ids == evidence_ids
        and payload.get("preference_observation_sha256") == expected_observation_sha256
        and isinstance(observations, dict)
        and observations.get("preference_observation_sha256") == expected_observation_sha256
        and isinstance(acceptance, dict)
        and acceptance.get("status") == "pass"
        and payload.get("ranking_status") == "estimated"
        and payload.get("bootstrap_replicates") == 5_000
        and isinstance(rows, list)
        and rows
        and all(isinstance(row, dict) and row.get("provisional") is False for row in rows)
    )


def accepted_snapshot_cells(database: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshots = database.get("leaderboard_snapshots", [])
    if not isinstance(snapshots, list):
        return {}
    accepted: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or not valid_statistical_snapshot(snapshot):
            continue
        key = f"{snapshot['track']}:{snapshot['cohort']}"
        accepted[key] = snapshot
    return accepted


def valid_research_archive(
    value: dict[str, Any],
    *,
    season_id: str,
    snapshot_ids: list[str],
    robustness_evidence_sha256: dict[str, str] | None = None,
) -> bool:
    manifest = value.get("manifest_json")
    verification = value.get("verification")
    observed_snapshot_ids = value.get("snapshot_ids_json")
    expected_snapshot_ids = sorted(set(snapshot_ids))
    return bool(
        value.get("season_id") == season_id
        and value.get("archive_class") == "internal_official"
        and value.get("schema_version") == "flavourbench-research-release-v1"
        and observed_snapshot_ids == expected_snapshot_ids
        and value.get("snapshot_set_sha256")
        == _archive_canonical_sha256({"snapshot_ids": expected_snapshot_ids})
        and isinstance(manifest, dict)
        and value.get("manifest_sha256") == _archive_canonical_sha256(manifest)
        and manifest.get("snapshot_set_sha256") == value.get("snapshot_set_sha256")
        and manifest.get("requirements_lock_sha256") == value.get("requirements_lock_sha256")
        and manifest.get("build_image_digest") == value.get("build_image_digest")
        and (
            robustness_evidence_sha256 is None
            or manifest.get("robustness_evidence_sha256")
            == dict(sorted(robustness_evidence_sha256.items()))
        )
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("archive_sha256", "")))
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("requirements_lock_sha256", "")))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("build_image_digest", "")))
        and value.get("signature_algorithm") == "Ed25519"
        and int(value.get("size_bytes", 0)) > 0
        and int(value.get("member_count", 0)) > 0
        and int(value.get("source_date_epoch", -1)) == 0
        and isinstance(verification, dict)
        and verification.get("metadata_signature_valid") is True
        and verification.get("archive_file_verified") is True
        and verification.get("inventory_valid") is True
        and verification.get("reproducible_metadata_valid") is True
        and verification.get("verified_manifest_sha256") == value.get("manifest_sha256")
    )


def valid_restricted_human_report(value: dict[str, Any]) -> bool:
    boundary = value.get("claim_boundary")
    scope = value.get("scope_audit")
    source_pool = value.get("source_pool")
    completion = value.get("completion_audit")
    if not all(isinstance(item, dict) for item in (boundary, scope, source_pool, completion)):
        return False

    task_ids = scope.get("task_public_ids")
    governance = scope.get("governance_review")
    candidate = source_pool.get("candidate_artifact")
    replacement = completion.get("replacement_candidate")
    if not all(isinstance(item, dict) for item in (governance, candidate, replacement)):
        return False

    historical_pool_sha256 = source_pool.get("historical_review_session_pool_sha256")
    source_artifact_sha256 = candidate.get("artifact_sha256")
    replacement_artifact_sha256 = replacement.get("artifact_sha256")
    return bool(
        verify_embedded_digest(value)
        and value.get("schema_version") == HUMAN_QA_SCHEMA_VERSION
        and boundary.get("evidence_status") == "restricted_operational_qa"
        and all(
            boundary.get(field) is False
            for field in ("paper_use", "research_use", "rank_eligible", "leaderboard_use")
        )
        and isinstance(task_ids, list)
        and all(isinstance(task_id, str) for task_id in task_ids)
        and len(task_ids) == EXPECTED_REVIEWED_QUARANTINE_TASKS
        and len(set(task_ids)) == EXPECTED_REVIEWED_QUARANTINE_TASKS
        and set(task_ids).issubset(TASK_SCOPE_QUARANTINE)
        and scope.get("general_track_quarantine_tasks_reviewed")
        == EXPECTED_REVIEWED_QUARANTINE_TASKS
        and scope.get("governed_quarantine_tasks") == len(TASK_SCOPE_QUARANTINE)
        and governance.get("schema_version") == "flavourbench-specialist-scope-review-v1"
        and governance.get("artifact_sha256") == TASK_SCOPE_REVIEW_SHA256
        and governance.get("quarantined_task_count") == len(TASK_SCOPE_QUARANTINE)
        and isinstance(historical_pool_sha256, str)
        and historical_pool_sha256 == source_artifact_sha256
        and replacement_artifact_sha256 != historical_pool_sha256
    )


def valid_real_human_pilot(value: dict[str, Any]) -> bool:
    inventory = value.get("real_data_inventory")
    boundary = value.get("claim_boundary")
    validity = value.get("task_validity_diagnostic")
    return bool(
        verify_embedded_digest(value)
        and value.get("schema_version") == "flavourbench-real-human-pilot-quality-v1"
        and value.get("status") == "restricted_real_human_pilot_diagnostic"
        and isinstance(inventory, dict)
        and inventory.get("synthetic_arms") == 0
        and int(inventory.get("primary_human_judgments", 0)) > 0
        and isinstance(validity, dict)
        and int(validity.get("later_governance_scope_disagreements", 0)) > 0
        and isinstance(boundary, dict)
        and boundary.get("real_model_outputs") is True
        and boundary.get("real_epicure_calls") is True
        and boundary.get("real_human_judgments") is True
        and boundary.get("synthetic_observations") == 0
        and all(
            boundary.get(field) is False
            for field in (
                "paper_use",
                "research_release_use",
                "official_leaderboard_use",
                "model_ranking_use",
            )
        )
    )


def valid_real_pilot_validator_audit(value: dict[str, Any]) -> bool:
    observed = value.get("observed")
    boundary = value.get("claim_boundary")
    return bool(
        verify_embedded_digest(value)
        and value.get("schema_version") == "flavourbench-real-arm-validator-audit-v1"
        and value.get("status") == "restricted_real_arm_validator_audit"
        and isinstance(observed, dict)
        and observed.get("real_response_arms") == 192
        and observed.get("synthetic_or_mock_arms") == 0
        and observed.get("validator_receipts_verified") == 1344
        and isinstance(boundary, dict)
        and boundary.get("synthetic_observations") == 0
        and boundary.get("human_quality_judgments_added") == 0
        and boundary.get("official_leaderboard_use") is False
        and boundary.get("model_ranking_use") is False
    )


def valid_surface_clean_development_tasks(value: dict[str, Any]) -> bool:
    counts = value.get("counts")
    boundary = value.get("claim_boundary")
    return bool(
        verify_embedded_digest(value)
        and value.get("schema_version") == "flavourbench-development-task-validity-v2"
        and value.get("status")
        == "surface_clean_source_verified_development_candidate_not_confirmatory"
        and isinstance(counts, dict)
        and counts.get("synthetic_tasks") == 0
        and counts.get("selected_development_tasks") == 40
        and int(counts.get("surface_dependency_quarantined", 0)) > 0
        and isinstance(boundary, dict)
        and boundary.get("real_human_authored_tasks") is True
        and boundary.get("official") is False
        and boundary.get("rank_eligible") is False
    )


def valid_development_validation_packet(value: dict[str, Any]) -> bool:
    try:
        verify_validation_packet(value)
    except DevelopmentTaskValidationError:
        return False
    counts = value.get("counts")
    policy = value.get("review_policy")
    boundary = value.get("claim_boundary")
    return bool(
        value.get("artifact_sha256") == DEVELOPMENT_VALIDATION_PACKET_SHA256
        and isinstance(counts, dict)
        and counts.get("tasks") == 40
        and counts.get("per_family")
        == {
            "substitution": 10,
            "composition": 10,
            "cookability": 10,
            "evidence": 10,
        }
        and counts.get("synthetic_tasks") == 0
        and counts.get("sealed_human_reviews") == 0
        and isinstance(policy, dict)
        and policy.get("required_independent_reviewers_per_task") == 3
        and policy.get("unanimous_valid_completes_source_review") is True
        and policy.get("adjudication_trigger") == "any_nonunanimous_decision"
        and isinstance(boundary, dict)
        and boundary.get("packet_itself_is_human_validation_evidence") is False
        and boundary.get("rank_eligible") is False
    )


def valid_current_model_manifest(value: dict[str, Any]) -> bool:
    design = value.get("run_design")
    protocol = design.get("generation_protocol") if isinstance(design, dict) else None
    policy = design.get("execution_policy") if isinstance(design, dict) else None
    task_source = design.get("task_source") if isinstance(design, dict) else None
    return bool(
        verify_manifest_content_address(value)
        and value.get("manifest_role") == "current_frontier_real_development_quality_run"
        and value.get("generation_calls_made") == 0
        and value.get("official_results_authorised") is False
        and isinstance(value.get("models"), list)
        and len(value["models"]) == 14
        and isinstance(protocol, dict)
        and protocol.get("evidence_protocol") == "matched_evidence_v2"
        and isinstance(policy, dict)
        and policy.get("limits", {}).get("max_output_tokens") == 8192
        and isinstance(task_source, dict)
        and task_source.get("artifact_sha256")
        == "5ffd81a44267291413bc8a638d15391ec2b51decdda270550f81ca17ec587846"
        and task_source.get("synthetic_tasks") == 0
    )


def valid_current_catalog_audit(value: dict[str, Any]) -> bool:
    counts = value.get("counts")
    boundary = value.get("claim_boundary")
    return bool(
        verify_catalog_audit_content_address(value)
        and isinstance(counts, dict)
        and counts.get("manifest_models") == 14
        and counts.get("models_discovered") == 14
        and counts.get("exact_provider_endpoints_matched") == 14
        and counts.get("freshness_contract_passed") == 14
        and counts.get("freshness_contract_failed") == 0
        and counts.get("provider_generations") == 0
        and counts.get("quality_observations") == 0
        and counts.get("spend_usd") == "0"
        and isinstance(boundary, dict)
        and boundary.get("live_catalog_network_requests") == 15
        and boundary.get("provider_generation_requests") == 0
        and boundary.get("epicure_calls") == 0
        and boundary.get("quality_observations") == 0
        and boundary.get("rank_eligible") is False
        and boundary.get("catalog_presence_is_not_execution_compatibility") is True
        and boundary.get("catalog_presence_is_not_model_quality") is True
    )


def latest_human_report(directory: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for path in directory.glob("operational-qa/restricted-operational-qa-*.json"):
        value = load_json(path)
        if not valid_restricted_human_report(value):
            continue
        candidates.append((str(value.get("observed_at", "")), path))
    if not candidates:
        raise Season1ReadinessError("no valid restricted human-review QA report is available")
    return max(candidates)[1]


def run_command(*command: str) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def control_plane_state() -> dict[str, Any]:
    payload = run_command(
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
        raise Season1ReadinessError("control-plane query did not return an object")
    _enrich_archive_verification(value)
    return value


def service_state(name: str) -> dict[str, Any]:
    payload = run_command(
        "sudo",
        "-n",
        "docker",
        "inspect",
        "--format",
        (
            '{"status":{{json .State.Status}},'
            '"health":{{if .State.Health}}{{json .State.Health.Status}}{{else}}null{{end}},'
            '"image_id":{{json .Image}},'
            '"image_name":{{json .Config.Image}}}'
        ),
        name,
    )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise Season1ReadinessError(f"{name} inspection did not return an object")
    return value


def consent_state(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    status_match = re.search(r"^Status:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    status = status_match.group(1).rstrip(".") if status_match else "missing_status"
    return {
        "status": status,
        "active": status.strip().lower() == "active",
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256_file(path),
    }


def source_record(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256_file(path),
        "embedded_artifact_sha256": value.get("artifact_sha256"),
        "schema_version": value.get("schema_version"),
    }


def build_report(
    *,
    panel: dict[str, Any],
    parity: dict[str, Any],
    human: dict[str, Any],
    study_design: dict[str, Any],
    method_validation: dict[str, Any],
    epicure_release: dict[str, Any],
    database: dict[str, Any],
    services: dict[str, dict[str, Any]],
    consent: dict[str, Any],
    sources: list[dict[str, Any]],
    robustness_evidence: dict[str, dict[str, Any]] | None = None,
    real_human_pilot: dict[str, Any] | None = None,
    real_pilot_validators: dict[str, Any] | None = None,
    surface_clean_tasks: dict[str, Any] | None = None,
    development_validation_packet: dict[str, Any] | None = None,
    current_model_manifest: dict[str, Any] | None = None,
    current_catalog_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    database_scope = database.get("scope")
    if not (
        isinstance(database_scope, dict)
        and database_scope.get("season_slug") == "season-1"
        and isinstance(database_scope.get("season_id"), str)
        and database_scope.get("season_id")
        and database_scope.get("all_release_counts_season_scoped") is True
    ):
        raise Season1ReadinessError("control-plane evidence is not explicitly scoped to Season 1")
    counts = database["counts"]
    season = database["season_1"]
    if not isinstance(season, dict) or season.get("id") != database_scope["season_id"]:
        raise Season1ReadinessError(
            "control-plane Season 1 identity does not match the scoped count set"
        )
    task_bank = database["task_bank"]
    provider_budgets = database["provider_budgets"]
    reviewers = database["independent_reviewers_by_family"]
    fixture_rows = database["fixture_rows"]

    services_healthy = all(
        state.get("status") == "running" and state.get("health") == "healthy"
        for state in services.values()
    )
    fixture_free = all(int(value) == 0 for value in fixture_rows.values())
    primary_collection = study_design.get("primary_controlled_collection", {})
    arena_collection = (
        primary_collection.get("model_arena", {}) if isinstance(primary_collection, dict) else {}
    )
    uplift_collection = (
        primary_collection.get("epicure_uplift", {}) if isinstance(primary_collection, dict) else {}
    )
    design_task_bank = study_design.get("task_bank", {})
    task_admission = (
        design_task_bank.get("admission", {}) if isinstance(design_task_bank, dict) else {}
    )
    required_task_count = int(design_task_bank.get("total", 0) or 0)
    required_blind_validity_records = required_task_count * int(
        task_admission.get("blind_prompt_only_solutions_per_task", 0) or 0
    )
    required_reconciliations = required_task_count * int(
        task_admission.get("independent_reconciliations_per_task", 0) or 0
    )
    required_adjudications = required_task_count * int(
        task_admission.get("independent_adjudications_per_task", 0) or 0
    )
    required_completed_validity_records = int(
        task_admission.get("minimum_human_validity_records", 0) or 0
    )
    required_validator_contract_reviews = required_task_count * int(
        task_admission.get("independent_validator_contract_reviews_per_task", 0) or 0
    )
    required_contamination_audit_reviews = required_task_count * int(
        task_admission.get("independent_contamination_audit_reviews_per_task", 0) or 0
    )
    required_surface_diagnostics = int(
        design_task_bank.get("minimum_surface_diagnostic_coverage", 0) or 0
    )
    contamination_policy = (
        design_task_bank.get("contamination", {}) if isinstance(design_task_bank, dict) else {}
    )
    contamination_calibration_policy = (
        contamination_policy.get("labeled_detection_calibration", {})
        if isinstance(contamination_policy, dict)
        else {}
    )
    required_contamination_calibration_cases = int(
        contamination_calibration_policy.get("minimum_cases", 0) or 0
    )
    required_contamination_precision_milli = round(
        1000 * float(contamination_calibration_policy.get("minimum_overall_precision", 0) or 0)
    )
    required_contamination_recall_milli = round(
        1000 * float(contamination_calibration_policy.get("minimum_overall_recall", 0) or 0)
    )
    required_paraphrase_recall_milli = round(
        1000 * float(contamination_calibration_policy.get("minimum_paraphrase_recall", 0) or 0)
    )
    robustness_policy = study_design.get("validity_and_robustness", {})
    post_collection_audit_policy = (
        robustness_policy.get("post_collection_item_audit", {})
        if isinstance(robustness_policy, dict)
        else {}
    )
    reliability_panel_policy = (
        robustness_policy.get("generation_reliability_panel", {})
        if isinstance(robustness_policy, dict)
        else {}
    )
    prompt_sensitivity_policy = (
        robustness_policy.get("prompt_sensitivity_audit", {})
        if isinstance(robustness_policy, dict)
        else {}
    )
    cookability_execution_policy = (
        robustness_policy.get("practical_cookability_execution", {})
        if isinstance(robustness_policy, dict)
        else {}
    )
    design_frozen = bool(
        verify_embedded_digest(study_design)
        and study_design.get("schema_version") == "flavourbench-season1-study-design-v5"
        and study_design.get("status")
        == "prospective-design-superseding-v4-before-scored-collection"
        and study_design.get("task_bank", {}).get("total") == 240
        and arena_collection.get("total_battles") == 3200
        and arena_collection.get("endpoint_appearances") == 6400
        and uplift_collection.get("total_pairs") == 3200
        and primary_collection.get("total_model_response_arms") == 12800
        and study_design.get("analysis", {}).get("no_composite_score") is True
        and study_design.get("claim_boundary", {}).get("synthetic_observations")
        == "prohibited-from-all-scored-and-supplemental-empirical-evidence"
        and contamination_calibration_policy.get("required_before_scored_collection") is True
        and required_contamination_calibration_cases >= 150
        and required_contamination_precision_milli >= 950
        and required_contamination_recall_milli >= 900
        and required_paraphrase_recall_milli >= 850
        and int(post_collection_audit_policy.get("minimum_random_tasks", 0) or 0) >= 60
        and post_collection_audit_policy.get("all_anomaly_flagged_tasks") is True
        and int(
            post_collection_audit_policy.get("minimum_independent_auditors_per_task", 0) or 0
        )
        >= 2
        and post_collection_audit_policy.get(
            "release_requires_zero_unresolved_material_defects"
        )
        is True
        and int(reliability_panel_policy.get("task_count", 0) or 0) == 20
        and int(reliability_panel_policy.get("endpoint_count", 0) or 0) == 16
        and int(
            reliability_panel_policy.get("independent_generations_per_cell", 0) or 0
        )
        == 3
        and int(reliability_panel_policy.get("total_panel_arms", 0) or 0) == 1920
        and reliability_panel_policy.get("retries_are_not_repetitions") is True
        and int(prompt_sensitivity_policy.get("total_response_arms", 0) or 0) == 480
        and prompt_sensitivity_policy.get("ranking_use")
        == "development-only-non-ranking-audit"
        and int(cookability_execution_policy.get("task_count", 0) or 0) == 24
        and int(cookability_execution_policy.get("total_kitchen_executions", 0) or 0) == 48
        and int(
            robustness_policy.get(
                "total_planned_real_model_response_arms_including_robustness", 0
            )
            or 0
        )
        == 14560
        and sha256_file(DEFAULT_ROBUSTNESS_EVIDENCE_CONTRACT)
        == ROBUSTNESS_EVIDENCE_CONTRACT_SHA256
    )
    robustness_evidence = robustness_evidence or {}
    design_sha256 = str(study_design.get("artifact_sha256", ""))
    post_collection_item_audit_ready = valid_post_collection_item_audit(
        robustness_evidence.get("post_collection_item_audit", {}),
        study_design_sha256=design_sha256,
    )
    generation_reliability_panel_ready = valid_generation_reliability_panel(
        robustness_evidence.get("generation_reliability_panel", {}),
        study_design_sha256=design_sha256,
    )
    prompt_sensitivity_audit_ready = valid_prompt_sensitivity_audit(
        robustness_evidence.get("prompt_sensitivity_audit", {}),
        study_design_sha256=design_sha256,
    )
    practical_cookability_execution_ready = valid_practical_cookability_execution(
        robustness_evidence.get("practical_cookability_execution", {}),
        study_design_sha256=design_sha256,
    )
    robustness_results_ready = all(
        (
            post_collection_item_audit_ready,
            generation_reliability_panel_ready,
            prompt_sensitivity_audit_ready,
            practical_cookability_execution_ready,
        )
    )
    statistical_methods_ready = verify_method_validation(
        method_validation,
        reproduce=True,
    )
    panel_counts = panel.get("counts", {})
    passed_route_count = int(panel_counts.get("contract_passed", 0) or 0)
    failed_route_count = int(panel_counts.get("contract_failed", 0) or 0)
    exact_panel_contract = bool(
        verify_embedded_digest(panel)
        and panel.get("schema_version") == "flavourbench-current-route-registry-v1"
        and panel_counts.get("models") == 16
        and passed_route_count + failed_route_count == 16
        and panel_counts.get("real_provider_generations_in_passed_receipts")
        == 2 * passed_route_count
        and panel_counts.get("real_epicure_calls_in_passed_receipts") == passed_route_count
        and panel_counts.get("quality_observations") == 0
        and panel_counts.get("rankable_comparisons") == 0
        and panel.get("rank_eligible") is False
    )
    all_route_contracts_ready = bool(
        exact_panel_contract and passed_route_count == 16 and failed_route_count == 0
    )
    structured_parity = bool(
        verify_embedded_digest(parity)
        and parity.get("selected_contract_evidence", {}).get("routes") == 8
        and parity.get("selected_contract_evidence", {}).get("all_normal_stop") is True
        and parity.get("selected_contract_evidence", {}).get("all_strict_structured_output") is True
        and parity.get("claim_boundary", {}).get("rank_eligible") is False
    )
    human_digest_valid = verify_embedded_digest(human)
    human_boundary = human.get("claim_boundary", {})
    human_restricted = valid_restricted_human_report(human)
    if not human_restricted:
        raise Season1ReadinessError(
            "human-review input is not a valid fail-closed operational-QA artifact"
        )
    if real_human_pilot is not None and not valid_real_human_pilot(real_human_pilot):
        raise Season1ReadinessError("restricted real-human pilot evidence failed closed")
    if real_pilot_validators is not None and not valid_real_pilot_validator_audit(
        real_pilot_validators
    ):
        raise Season1ReadinessError("real-pilot validator evidence failed closed")
    if surface_clean_tasks is not None and not valid_surface_clean_development_tasks(
        surface_clean_tasks
    ):
        raise Season1ReadinessError("surface-clean development task evidence failed closed")
    development_validation = database.get("development_task_validation", {})
    if development_validation_packet is not None:
        if not valid_development_validation_packet(development_validation_packet):
            raise Season1ReadinessError("development task-validation packet failed closed")
        if not isinstance(development_validation, dict) or (
            development_validation.get("packet_sha256")
            != development_validation_packet.get("artifact_sha256")
        ):
            raise Season1ReadinessError(
                "live development task-validation records do not bind the current packet"
            )
    if current_model_manifest is not None and not valid_current_model_manifest(
        current_model_manifest
    ):
        raise Season1ReadinessError("current-model prospective manifest failed closed")
    if current_catalog_audit is not None:
        if not valid_current_catalog_audit(current_catalog_audit):
            raise Season1ReadinessError("current-model catalog audit failed closed")
        if current_model_manifest is None or current_catalog_audit.get(
            "source_manifest_sha256"
        ) != current_model_manifest.get("content_address", {}).get("digest"):
            raise Season1ReadinessError(
                "current-model catalog audit does not bind the prospective manifest"
            )
    task_bank_ready = bool(
        int(task_bank["season1_eligible"]) == 240
        and int(task_bank["scored"]) == 160
        and int(task_bank["development"]) == 40
        and int(task_bank["private_reserve"]) == 40
        and int(task_bank["synthetic"]) == 0
    )
    task_review_ready = bool(
        int(task_bank["independent_task_approvals"]) >= required_reconciliations
        and int(task_bank.get("blind_prompt_only_validity_records", 0))
        >= required_blind_validity_records
        and int(task_bank.get("independent_reconciliation_records", 0)) >= required_reconciliations
        and int(task_bank.get("independent_adjudication_records", 0)) >= required_adjudications
        and int(task_bank.get("independent_reconciliation_records", 0))
        + int(task_bank.get("independent_adjudication_records", 0))
        >= required_completed_validity_records
    )
    task_evidence_review_ready = bool(
        task_bank.get("task_review_evidence_store_available") is True
        and int(task_bank.get("validator_contract_human_reviews", 0))
        == required_validator_contract_reviews
        and int(task_bank.get("contamination_audit_human_reviews", 0))
        == required_contamination_audit_reviews
    )
    minimum_distinct_authors = int(BLUEPRINT["authorship"]["minimum_distinct_authors"])
    task_contributor_accounts_total = int(
        task_bank.get("task_contributor_accounts_total", 0)
    )
    current_protocol_task_contributors = int(
        task_bank.get("current_protocol_task_contributor_accounts", 0)
    )
    task_contributor_protocol_ready = bool(
        current_protocol_task_contributors >= minimum_distinct_authors
    )
    task_candidate_withdrawal_ready = bool(
        int(task_bank.get("duplicate_candidate_withdrawals", 0)) == 0
        and int(task_bank.get("withdrawn_imported_task_candidates", 0)) == 0
    )
    author_person_uniqueness_ready = bool(
        int(task_bank.get("verified_unique_task_contributors", 0)) >= minimum_distinct_authors
        and int(task_bank.get("verified_unique_task_contributors", 0))
        == int(task_bank.get("active_anonymous_task_contributors", 0))
        and int(task_bank.get("author_person_binding_tasks", 0)) == required_task_count
    )
    construct_blueprint_ready = bool(
        task_bank.get("construct_blueprint_verified") is True
        and isinstance(task_bank.get("construct_blueprint_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(task_bank["construct_blueprint_sha256"]))
    )
    criterion_pack_ready = bool(
        int(task_bank.get("criterion_pack_tasks", 0)) == required_task_count
    )
    contamination_replay_ready = bool(
        int(task_bank.get("contamination_replay_tasks", 0)) == required_task_count
        and int(task_bank.get("contamination_replay_bundle_count", 0)) == 1
    )
    surface_diagnostic_ready = bool(
        int(task_bank.get("surface_diagnostic_tasks", 0)) >= required_surface_diagnostics
    )
    task_lifecycle_ready = bool(
        int(task_bank.get("sealed_task_lifecycles", 0)) == required_task_count
        and int(task_bank.get("preseal_task_uses", 0)) == 0
        and int(task_bank.get("retired_confirmatory_tasks", 0)) == 0
    )
    validator_calibration_ready = bool(
        task_bank.get("validator_calibration_verified") is True
        and isinstance(task_bank.get("validator_calibration_artifact_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(task_bank["validator_calibration_artifact_sha256"]))
    )
    contamination_calibration_ready = bool(
        task_bank.get("contamination_calibration_verified") is True
        and isinstance(task_bank.get("contamination_calibration_artifact_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(task_bank["contamination_calibration_artifact_sha256"]),
        )
        and int(task_bank.get("contamination_calibration_case_count", 0) or 0)
        >= required_contamination_calibration_cases
        and int(task_bank.get("contamination_calibration_precision_milli", 0) or 0)
        >= required_contamination_precision_milli
        and int(task_bank.get("contamination_calibration_recall_milli", 0) or 0)
        >= required_contamination_recall_milli
        and int(task_bank.get("contamination_calibration_paraphrase_recall_milli", 0) or 0)
        >= required_paraphrase_recall_milli
    )
    independent_reviewer_ready = all(int(value) >= 2 for value in reviewers.values())
    lineage_ready = bool(
        int(counts["official_epicure_releases"]) >= 1
        and epicure_release.get("rank_eligible") is True
        and epicure_release.get("rights", {}).get("status") == "cleared"
    )
    provider_budget_ready = bool(
        {"bedrock", "openrouter"}.issubset(provider_budgets)
        and all(
            int(provider_budgets[backend]["cap_micros"]) > 0
            and provider_budgets[backend]["account_authorization_bound"] is True
            for backend in ("bedrock", "openrouter")
        )
    )
    season_run_frozen = bool(
        isinstance(season, dict)
        and season.get("status") == "active"
        and season.get("frozen_at") is not None
        and season.get("official") is True
        and season.get("manifest_sha256") not in {None, "unfrozen"}
        and season.get("prompt_registry_sha256") not in {None, "unfrozen"}
        and season.get("tool_registry_sha256") not in {None, "unfrozen"}
        and season.get("analysis_plan_sha256") not in {None, "unfrozen"}
        and season.get("protocol_bundle_sha256") not in {None, "unfrozen"}
    )

    closed_generation_ready = all(
        (
            services_healthy,
            fixture_free,
            design_frozen,
            statistical_methods_ready,
            all_route_contracts_ready,
            structured_parity,
            task_bank_ready,
            task_review_ready,
            task_evidence_review_ready,
            task_candidate_withdrawal_ready,
            author_person_uniqueness_ready,
            construct_blueprint_ready,
            criterion_pack_ready,
            contamination_replay_ready,
            surface_diagnostic_ready,
            task_lifecycle_ready,
            validator_calibration_ready,
            contamination_calibration_ready,
            lineage_ready,
            provider_budget_ready,
            season_run_frozen,
        )
    )
    controlled_collection_ready = bool(closed_generation_ready and independent_reviewer_ready)
    public_collection_ready = bool(controlled_collection_ready and consent["active"])
    snapshots = accepted_snapshot_cells(database)
    required_snapshot_cells = {
        "model_arena:public",
        "model_arena:expert_independent",
        "epicure_uplift:public",
        "epicure_uplift:expert_independent",
    }
    expert_snapshot_coverage = True
    for key in (
        "model_arena:expert_independent",
        "epicure_uplift:expert_independent",
    ):
        snapshot = snapshots.get(key)
        payload = snapshot.get("payload_json", {}) if snapshot else {}
        coverage = payload.get("rater_coverage", {}) if isinstance(payload, dict) else {}
        expert_snapshot_coverage &= bool(
            isinstance(coverage, dict)
            and int(coverage.get("unique_comparisons", 0)) >= 800
            and int(coverage.get("minimum_distinct_raters_per_comparison", 0)) >= 2
            and int(coverage.get("comparisons_with_two_or_more_distinct_raters", 0)) >= 800
        )
    result_sample_ready = bool(
        required_snapshot_cells.issubset(snapshots) and expert_snapshot_coverage
    )
    accepted_snapshot_ids = sorted(
        str(snapshots[key]["id"]) for key in required_snapshot_cells if key in snapshots
    )
    archive_rows = database.get("research_release_archives", [])
    accepted_archives = (
        [
            row
            for row in archive_rows
            if isinstance(row, dict)
            and valid_research_archive(
                row,
                season_id=str(database_scope["season_id"]),
                snapshot_ids=accepted_snapshot_ids,
                robustness_evidence_sha256={
                    name: str(robustness_evidence[name]["artifact_sha256"])
                    for name in (
                        "post_collection_item_audit",
                        "generation_reliability_panel",
                        "prompt_sensitivity_audit",
                        "practical_cookability_execution",
                    )
                }
                if robustness_results_ready
                else {},
            )
        ]
        if isinstance(archive_rows, list) and len(accepted_snapshot_ids) == 4
        else []
    )
    research_archive_ready = len(accepted_archives) == 1
    leaderboard_release_ready = bool(
        controlled_collection_ready
        and result_sample_ready
        and robustness_results_ready
        and research_archive_ready
    )

    gates = {
        "production_services": {
            "status": "pass" if services_healthy and fixture_free else "blocked",
            "services_healthy": services_healthy,
            "fixture_free": fixture_free,
        },
        "prospective_study_design": {
            "status": "pass" if design_frozen else "blocked",
            "artifact_sha256": study_design.get("artifact_sha256"),
            "target_population": study_design.get("target_population"),
        },
        "post_collection_item_audit": {
            "status": "pass" if post_collection_item_audit_ready else "not_started",
            "required": post_collection_audit_policy,
            "artifact_sha256": robustness_evidence.get("post_collection_item_audit", {}).get(
                "artifact_sha256"
            ),
            "evidence_contract": {
                "path": str(DEFAULT_ROBUSTNESS_EVIDENCE_CONTRACT.relative_to(ROOT)),
                "file_sha256": ROBUSTNESS_EVIDENCE_CONTRACT_SHA256,
            },
        },
        "generation_reliability_panel": {
            "status": "pass" if generation_reliability_panel_ready else "not_started",
            "required": reliability_panel_policy,
            "artifact_sha256": robustness_evidence.get("generation_reliability_panel", {}).get(
                "artifact_sha256"
            ),
        },
        "prompt_sensitivity_audit": {
            "status": "pass" if prompt_sensitivity_audit_ready else "not_started",
            "required": prompt_sensitivity_policy,
            "artifact_sha256": robustness_evidence.get("prompt_sensitivity_audit", {}).get(
                "artifact_sha256"
            ),
        },
        "practical_cookability_execution": {
            "status": "pass" if practical_cookability_execution_ready else "not_started",
            "required": cookability_execution_policy,
            "artifact_sha256": robustness_evidence.get(
                "practical_cookability_execution", {}
            ).get("artifact_sha256"),
        },
        "statistical_method_validation": {
            "status": "pass" if statistical_methods_ready else "blocked",
            "artifact_sha256": method_validation.get("artifact_sha256"),
            "artifact_class": method_validation.get("artifact_class"),
            "acceptance": method_validation.get("acceptance"),
            "claim_boundary": method_validation.get("claim_boundary"),
        },
        "exact_model_contracts": {
            "status": (
                "pass_contract_only"
                if all_route_contracts_ready and structured_parity
                else "partial_contract_only"
                if exact_panel_contract and structured_parity
                else "blocked"
            ),
            "models": panel.get("counts", {}).get("models"),
            "contract_passed": panel.get("counts", {}).get("contract_passed"),
            "contract_failed": panel.get("counts", {}).get("contract_failed"),
            "real_provider_generations": panel.get("counts", {}).get(
                "real_provider_generations_in_passed_receipts"
            ),
            "real_epicure_calls": panel.get("counts", {}).get(
                "real_epicure_calls_in_passed_receipts"
            ),
            "quality_observations": 0,
            "rankable_comparisons": 0,
        },
        "prospective_roster_catalog_freshness": {
            "status": "pass_catalog_only" if current_catalog_audit is not None else "blocked",
            "artifact_sha256": (
                current_catalog_audit.get("artifact_sha256")
                if current_catalog_audit is not None
                else None
            ),
            "observed_at": (
                current_catalog_audit.get("observed_at")
                if current_catalog_audit is not None
                else None
            ),
            "counts": (
                current_catalog_audit.get("counts", {})
                if current_catalog_audit is not None
                else {}
            ),
            "quality_observations": 0,
            "rank_eligible": False,
        },
        "confirmatory_task_bank": {
            "status": "pass" if task_bank_ready else "blocked",
            "observed": {
                "total": int(task_bank["season1_eligible"]),
                "scored": int(task_bank["scored"]),
                "development": int(task_bank["development"]),
                "private_reserve": int(task_bank["private_reserve"]),
                "synthetic": int(task_bank["synthetic"]),
                "human_task_candidates": int(task_bank.get("human_task_candidates", 0)),
                "submitted_candidates_total": int(
                    task_bank.get("task_candidates_submitted_total", 0)
                ),
                "active_candidates": int(task_bank.get("active_task_candidates", 0)),
                "withdrawn_candidates": int(
                    task_bank.get("withdrawn_task_candidates", 0)
                ),
                "legacy_candidate_review_events": int(
                    task_bank.get("legacy_task_candidate_reviews", 0)
                ),
                "blind_prompt_only_validity_records": int(
                    task_bank.get("blind_prompt_only_validity_records", 0)
                ),
                "independent_reconciliation_records": int(
                    task_bank.get("independent_reconciliation_records", 0)
                ),
                "independent_adjudication_records": int(
                    task_bank.get("independent_adjudication_records", 0)
                ),
                "candidates_approved_for_bank_assembly": int(
                    task_bank.get("candidates_approved_for_bank_assembly", 0)
                ),
                "active_anonymous_task_contributors": int(
                    task_bank.get("active_anonymous_task_contributors", 0)
                ),
                "task_contributor_accounts_total": task_contributor_accounts_total,
                "current_protocol_task_contributor_accounts": (
                    current_protocol_task_contributors
                ),
            },
            "required": study_design["task_bank"],
        },
        "task_candidate_withdrawal_integrity": {
            "status": "pass" if task_candidate_withdrawal_ready else "blocked",
            "observed": {
                "withdrawn_candidates": int(
                    task_bank.get("withdrawn_task_candidates", 0)
                ),
                "duplicate_withdrawal_events": int(
                    task_bank.get("duplicate_candidate_withdrawals", 0)
                ),
                "withdrawn_candidates_imported": int(
                    task_bank.get("withdrawn_imported_task_candidates", 0)
                ),
            },
            "required": {
                "append_only_event_schema": "flavourbench-task-candidate-withdrawal-v1",
                "maximum_events_per_candidate": 1,
                "withdrawn_candidates_imported": 0,
                "transactional_candidate_lock": True,
            },
            "claim_boundary": (
                "Withdrawal applies before task-bank import; later corrections use the "
                "public challenge, retirement, and snapshot-withdrawal process."
            ),
        },
        "three_stage_task_validity": {
            "status": "pass" if task_review_ready else "blocked",
            "observed": {
                "blind_prompt_only_records": int(
                    task_bank.get("blind_prompt_only_validity_records", 0)
                ),
                "reconciled_source_validations": int(
                    task_bank.get("independent_reconciliation_records", 0)
                ),
                "independent_adjudications": int(
                    task_bank.get("independent_adjudication_records", 0)
                ),
                "completed_human_validity_records": int(
                    task_bank.get("independent_reconciliation_records", 0)
                )
                + int(task_bank.get("independent_adjudication_records", 0)),
            },
            "required": {
                "blind_prompt_only_records": required_blind_validity_records,
                "reconciled_source_validations": required_reconciliations,
                "independent_adjudications": required_adjudications,
                "completed_human_validity_records": required_completed_validity_records,
            },
            "evidence_store_available": bool(task_bank["task_review_evidence_store_available"]),
            "legacy_one_step_reviews_counted": False,
        },
        "independent_task_evidence_reviews": {
            "status": "pass" if task_evidence_review_ready else "blocked",
            "observed": {
                "validator_contract_reviews": int(
                    task_bank.get("validator_contract_human_reviews", 0)
                ),
                "contamination_audit_reviews": int(
                    task_bank.get("contamination_audit_human_reviews", 0)
                ),
            },
            "required": {
                "validator_contract_reviews": required_validator_contract_reviews,
                "contamination_audit_reviews": required_contamination_audit_reviews,
            },
            "evidence_store_available": bool(task_bank["task_review_evidence_store_available"]),
            "role_policy": (
                "two distinct qualified reviewers per task, each independent of the author, "
                "source reviewers, and adjudicator"
            ),
        },
        "privacy_preserving_person_uniqueness": {
            "status": "pass" if author_person_uniqueness_ready else "blocked",
            "observed": {
                "contributor_accounts": int(task_bank.get("active_anonymous_task_contributors", 0)),
                "verified_unique_people": int(
                    task_bank.get("verified_unique_task_contributors", 0)
                ),
                "tasks_bound_to_person_commitment": int(
                    task_bank.get("author_person_binding_tasks", 0)
                ),
            },
            "required": {
                "minimum_unique_people": minimum_distinct_authors,
                "tasks_bound_to_person_commitment": required_task_count,
                "one_account_per_person": True,
            },
            "privacy": (
                "administrator-witnessed, season-specific HMAC commitments; raw identity "
                "handles are not retained"
            ),
        },
        "task_contributor_protocol_capacity": {
            "status": "pass" if task_contributor_protocol_ready else "blocked",
            "observed": {
                "current_protocol_contributors": current_protocol_task_contributors,
                "legacy_or_pending_contributors": max(
                    0,
                    task_contributor_accounts_total - current_protocol_task_contributors,
                ),
                "protocol_version": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
                "protocol_sha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
                "protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
            },
            "required": {
                "minimum_current_protocol_contributors": minimum_distinct_authors,
                "exact_protocol_binding": True,
                "append_only_acceptance_event": True,
            },
            "claim_boundary": (
                "Task-contributor acceptance governs authorship and redistribution only; it "
                "does not authorize output rating or clear the human-subjects determination."
            ),
        },
        "construct_blueprint": {
            "status": "pass" if construct_blueprint_ready else "blocked",
            "artifact_sha256": task_bank.get("construct_blueprint_sha256"),
        },
        "adjudicated_criterion_packs": {
            "status": "pass" if criterion_pack_ready else "blocked",
            "observed_tasks": int(task_bank.get("criterion_pack_tasks", 0)),
            "required_tasks": required_task_count,
            "delivery_policy": "criterion pack is bound to the task and shown to output raters",
        },
        "contamination_replay": {
            "status": "pass" if contamination_replay_ready else "blocked",
            "observed_tasks": int(task_bank.get("contamination_replay_tasks", 0)),
            "required_tasks": required_task_count,
            "observed_content_addressed_bundles": int(
                task_bank.get("contamination_replay_bundle_count", 0)
            ),
            "required_content_addressed_bundles": 1,
            "methods": design_task_bank.get("contamination", {}).get("methods", []),
        },
        "surface_diagnostics": {
            "status": "pass" if surface_diagnostic_ready else "blocked",
            "observed_tasks": int(task_bank.get("surface_diagnostic_tasks", 0)),
            "required_tasks": required_surface_diagnostics,
            "claim_boundary": "surface-constraint diagnostics are not culinary correctness",
        },
        "task_lifecycle_integrity": {
            "status": "pass" if task_lifecycle_ready else "blocked",
            "observed": {
                "sealed_tasks": int(task_bank.get("sealed_task_lifecycles", 0)),
                "first_used_tasks": int(task_bank.get("first_used_task_lifecycles", 0)),
                "preseal_uses": int(task_bank.get("preseal_task_uses", 0)),
                "retired_tasks": int(task_bank.get("retired_confirmatory_tasks", 0)),
            },
            "required": {
                "sealed_tasks": required_task_count,
                "preseal_uses": 0,
                "retired_tasks_at_admission": 0,
            },
            "event_order": [
                "authored_at",
                "sealed_at",
                "first_used_at",
                "released_at",
                "retired_at",
            ],
        },
        "validator_calibration": {
            "status": "pass" if validator_calibration_ready else "blocked",
            "artifact_sha256": task_bank.get("validator_calibration_artifact_sha256"),
            "minimum_precision": task_admission.get("validator_calibration_minimum_precision"),
            "minimum_recall": task_admission.get("validator_calibration_minimum_recall"),
        },
        "contamination_detection_calibration": {
            "status": "pass" if contamination_calibration_ready else "blocked",
            "artifact_sha256": task_bank.get("contamination_calibration_artifact_sha256"),
            "observed": {
                "cases": int(task_bank.get("contamination_calibration_case_count", 0) or 0),
                "precision": int(task_bank.get("contamination_calibration_precision_milli", 0) or 0)
                / 1000,
                "recall": int(task_bank.get("contamination_calibration_recall_milli", 0) or 0)
                / 1000,
                "paraphrase_recall": int(
                    task_bank.get("contamination_calibration_paraphrase_recall_milli", 0) or 0
                )
                / 1000,
            },
            "required": {
                "cases": required_contamination_calibration_cases,
                "precision": required_contamination_precision_milli / 1000,
                "recall": required_contamination_recall_milli / 1000,
                "paraphrase_recall": required_paraphrase_recall_milli / 1000,
            },
            "label_policy": (
                "two independent human label commitments per case; model outputs excluded"
            ),
        },
        "epicure_release": {
            "status": "pass" if lineage_ready else "blocked",
            "candidate_release_id": epicure_release.get("release_id"),
            "candidate_status": epicure_release.get("status"),
            "bundle_sha256": epicure_release.get("bundle", {}).get("sha256"),
            "rights_status": epicure_release.get("rights", {}).get("status"),
            "rank_eligible": epicure_release.get("rank_eligible"),
        },
        "provider_budget_epochs": {
            "status": "pass" if provider_budget_ready else "blocked",
            "observed": provider_budgets,
            "authorized_external_caps": {"bedrock_usd": 5000, "openrouter_usd": 100},
            "note": "The owner-authorized caps exist in governance records but are not yet bound "
            "to the draft Season 1 database row.",
        },
        "frozen_run_bundle": {
            "status": "pass" if season_run_frozen else "blocked",
            "observed": season,
        },
        "public_research_consent": {
            "status": "pass" if consent["active"] else "blocked",
            "document_status": consent["status"],
            "consented_battles": int(database["research_consent"]["consented_battles"]),
        },
        "independent_output_review": {
            "status": "pass" if independent_reviewer_ready else "blocked",
            "verified_reviewers_by_family": reviewers,
            "required_per_family": 2,
            "required_unique_comparisons_per_track": 800,
            "required_distinct_raters_per_comparison": 2,
        },
        "minimum_result_sample": {
            "status": "pass" if result_sample_ready else "not_started",
            "accepted_snapshot_cells": sorted(snapshots),
            "required_snapshot_cells": sorted(required_snapshot_cells),
            "expert_snapshot_coverage": expert_snapshot_coverage,
            "source": "content-addressed canonical analysis snapshots",
        },
        "signed_research_archive": {
            "status": "pass" if research_archive_ready else "blocked",
            "accepted_archive_ids": [str(row["id"]) for row in accepted_archives],
            "accepted_snapshot_ids": accepted_snapshot_ids,
            "required_archive_class": "internal_official",
            "requirements": (
                "Ed25519 signature, exact four-snapshot membership, deterministic tar metadata, "
                "member inventory, dependency lock, and content-addressed build image"
            ),
        },
    }

    return {
        "schema_version": "flavourbench-season1-readiness-audit-v13",
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": "prospective_season1_release_readiness",
        "control_plane": {
            "services": services,
            "database": database,
            "services_healthy": services_healthy,
            "fixture_free": fixture_free,
        },
        "candidate_panel": {
            "artifact_sha256": panel.get("artifact_sha256"),
            "status": panel.get("status"),
            "counts": panel.get("counts"),
            "strict_parity_cost": parity.get("selected_contract_evidence", {}).get(
                "reconciled_cost_usd"
            ),
            "qualification_exposure_interval_usd": parity.get(
                "total_qualification_exposure_interval_usd"
            ),
            "quality_observations": 0,
            "rank_eligible": False,
        },
        "restricted_human_review_qa": {
            "schema_version": human.get("schema_version"),
            "artifact_sha256": human.get("artifact_sha256"),
            "digest_valid": human_digest_valid,
            "evidence_status": human_boundary.get("evidence_status"),
            "historical_review_session_pool_sha256": human.get("source_pool", {}).get(
                "historical_review_session_pool_sha256"
            ),
            "scope_governance_artifact_sha256": human.get("scope_audit", {})
            .get("governance_review", {})
            .get("artifact_sha256"),
            "governed_quarantine_tasks": human.get("scope_audit", {}).get(
                "governed_quarantine_tasks"
            ),
            "reviewed_quarantine_tasks": human.get("scope_audit", {}).get(
                "general_track_quarantine_tasks_reviewed"
            ),
            "completed_presentations": human.get("review_progress", {}).get(
                "completed_presentations"
            ),
            "unique_primary_judgments": human.get("review_progress", {}).get(
                "unique_primary_judgments"
            ),
            "non_normal_response_arms": human.get("completion_audit", {}).get(
                "non_normal_response_arms"
            ),
            "replacement_candidate_sha256": human.get("completion_audit", {})
            .get("replacement_candidate", {})
            .get("artifact_sha256"),
            "human_evidence_eligible": False,
            "preference_aggregates_republished": False,
            "repeatability_aggregates_republished": False,
            "claim_boundary": human_boundary,
        },
        "restricted_real_human_pilot": {
            "status": "verified_restricted" if real_human_pilot is not None else "unavailable",
            "artifact_sha256": (
                real_human_pilot.get("artifact_sha256")
                if real_human_pilot is not None
                else None
            ),
            "official_prospective_quality_observations": 0,
            "primary_human_judgments": (
                real_human_pilot.get("real_data_inventory", {}).get(
                    "primary_human_judgments"
                )
                if real_human_pilot is not None
                else 0
            ),
            "finish_clean_primary_judgments": (
                real_human_pilot.get("real_data_inventory", {}).get(
                    "finish_clean_primary_judgments"
                )
                if real_human_pilot is not None
                else 0
            ),
            "task_scope_disagreements": (
                real_human_pilot.get("task_validity_diagnostic", {}).get(
                    "later_governance_scope_disagreements"
                )
                if real_human_pilot is not None
                else 0
            ),
            "synthetic_observations": 0,
            "paper_use": False,
            "leaderboard_use": False,
        },
        "real_pilot_deterministic_validation": {
            "status": (
                "verified_restricted" if real_pilot_validators is not None else "unavailable"
            ),
            "artifact_sha256": (
                real_pilot_validators.get("artifact_sha256")
                if real_pilot_validators is not None
                else None
            ),
            "validator_version": (
                real_pilot_validators.get("validator_version")
                if real_pilot_validators is not None
                else None
            ),
            "observed": (
                real_pilot_validators.get("observed", {})
                if real_pilot_validators is not None
                else {}
            ),
            "human_quality_judgments_added": 0,
            "paper_use": False,
            "leaderboard_use": False,
        },
        "surface_clean_real_human_development_pool": {
            "status": "verified_development" if surface_clean_tasks is not None else "unavailable",
            "artifact_sha256": (
                surface_clean_tasks.get("artifact_sha256")
                if surface_clean_tasks is not None
                else None
            ),
            "counts": surface_clean_tasks.get("counts", {}) if surface_clean_tasks else {},
            "confirmatory_eligible": False,
            "rank_eligible": False,
        },
        "development_task_validation_campaign": {
            "status": (
                "instrument_ready_awaiting_real_reviews"
                if development_validation_packet is not None
                and int(development_validation.get("blind_validity_records", 0)) == 0
                else "real_reviews_in_progress"
                if development_validation_packet is not None
                else "unavailable"
            ),
            "packet_sha256": (
                development_validation_packet.get("artifact_sha256")
                if development_validation_packet is not None
                else None
            ),
            "task_count": (
                development_validation_packet.get("counts", {}).get("tasks", 0)
                if development_validation_packet is not None
                else 0
            ),
            "required_independent_reviews_per_task": (
                development_validation_packet.get("review_policy", {}).get(
                    "required_independent_reviewers_per_task", 0
                )
                if development_validation_packet is not None
                else 0
            ),
            "assignment_policy": (
                development_validation_packet.get("review_policy", {}).get(
                    "assignment_policy"
                )
                if development_validation_packet is not None
                else None
            ),
            "statistics_policy": (
                development_validation_packet.get("statistics_policy", {})
                if development_validation_packet is not None
                else {}
            ),
            "required_independent_reviews_total": (
                int(development_validation_packet.get("counts", {}).get("tasks", 0))
                * int(
                    development_validation_packet.get("review_policy", {}).get(
                        "required_independent_reviewers_per_task", 0
                    )
                )
                if development_validation_packet is not None
                else 0
            ),
            "observed": development_validation if development_validation_packet else {},
            "independently_validated_tasks": (
                int(development_validation.get("unanimously_valid_tasks", 0))
                + int(development_validation.get("adjudicated_valid_tasks", 0))
                if development_validation_packet is not None
                else 0
            ),
            "unanimous_valid_skips_adjudication": True,
            "nonunanimous_records_require_fourth_person": True,
            "model_outputs_visible_during_validation": False,
            "confirmatory_eligible": False,
            "rank_eligible": False,
        },
        "prospective_current_model_manifest": {
            "status": "frozen_no_calls" if current_model_manifest is not None else "unavailable",
            "manifest_sha256": (
                current_model_manifest.get("content_address", {}).get("digest")
                if current_model_manifest is not None
                else None
            ),
            "model_count": (
                len(current_model_manifest.get("models", []))
                if current_model_manifest is not None
                else 0
            ),
            "model_ids": (
                sorted(
                    str(entry.get("model", {}).get("id"))
                    for entry in current_model_manifest.get("models", [])
                    if isinstance(entry, dict) and isinstance(entry.get("model"), dict)
                )
                if current_model_manifest is not None
                else []
            ),
            "evidence_protocol": (
                current_model_manifest.get("run_design", {})
                .get("generation_protocol", {})
                .get("evidence_protocol")
                if current_model_manifest is not None
                else None
            ),
            "max_output_tokens": (
                current_model_manifest.get("run_design", {})
                .get("execution_policy", {})
                .get("limits", {})
                .get("max_output_tokens")
                if current_model_manifest is not None
                else None
            ),
            "planned_real_arms": (
                current_model_manifest.get("run_design", {}).get("expected_arms")
                if current_model_manifest is not None
                else 0
            ),
            "provider_calls_made": False,
            "rank_eligible": False,
        },
        "current_model_catalog_audit": {
            "status": (
                "freshness_verified_no_generation"
                if current_catalog_audit is not None
                else "unavailable"
            ),
            "artifact_sha256": (
                current_catalog_audit.get("artifact_sha256")
                if current_catalog_audit is not None
                else None
            ),
            "observed_at": (
                current_catalog_audit.get("observed_at")
                if current_catalog_audit is not None
                else None
            ),
            "counts": (
                current_catalog_audit.get("counts", {})
                if current_catalog_audit is not None
                else {}
            ),
            "claim_boundary": (
                current_catalog_audit.get("claim_boundary", {})
                if current_catalog_audit is not None
                else {}
            ),
        },
        "release_gates": gates,
        "decision": {
            "closed_generation_admission_ready": closed_generation_ready,
            "controlled_collection_ready": controlled_collection_ready,
            "public_collection_ready": public_collection_ready,
            "quality_leaderboard_release_ready": leaderboard_release_ready,
            "current_result_status": "no_prospective_season1_quality_observations",
            "evidence_inventory": {
                "official_prospective_quality_observations": 0,
                "restricted_real_human_pilot_judgments": (
                    real_human_pilot.get("real_data_inventory", {}).get(
                        "primary_human_judgments", 0
                    )
                    if real_human_pilot is not None
                    else 0
                ),
                "restricted_real_model_arms_with_validator_receipts": (
                    real_pilot_validators.get("observed", {}).get("real_response_arms", 0)
                    if real_pilot_validators is not None
                    else 0
                ),
                "development_task_validation_packet_tasks": (
                    development_validation_packet.get("counts", {}).get("tasks", 0)
                    if development_validation_packet is not None
                    else 0
                ),
                "development_independent_blind_validity_records": int(
                    development_validation.get("blind_validity_records", 0)
                ),
                "development_independently_validated_tasks": (
                    int(development_validation.get("unanimously_valid_tasks", 0))
                    + int(development_validation.get("adjudicated_valid_tasks", 0))
                ),
                "synthetic_observations": 0,
            },
            "paper_boundary": (
                "The existing manuscript is a retrospective measurement audit. A Season 1 "
                "benchmark-results paper begins only after the frozen prospective collection."
            ),
            "next_actions": [
                "complete three independent blind validations and criterion packs for each "
                "of the 40 public calibration tasks; adjudicate only non-unanimous records",
                "retain the frozen construct blueprint and independently calibrate surface "
                "diagnostics",
                "retain a reproduced passing statistical method-validation artifact",
                "admit at least 20 distinct task authors under the exact current contributor "
                "protocol and append one acceptance event per author",
                "collect and seal 240 human-authored tasks with 480 blind prompt-only "
                "solutions, 480 reconciliations, 240 adjudications, and 480 independent "
                "evidence inspections",
                "replay the frozen five-method contamination audit over one "
                "content-addressed bundle",
                "pass the independently labeled contamination detector calibration, including "
                "the paraphrase-recall floor",
                "publish and register the exact Epicure MCP release",
                "bind Bedrock and OpenRouter budget epochs to the frozen Season 1 run",
                "activate reviewer consent documents and complete the human-subjects "
                "determination before public collection",
                "freeze the final model manifest, prompt registry, tools, and analysis bundle",
                "collect the controlled schedule and two-rater independent expert sample",
                "seal the post-collection item audit, three-run reliability panel, development "
                "prompt-sensitivity audit, and blinded kitchen-execution study",
                "seal four statistically accepted canonical analysis snapshots",
                "build and independently verify the signed snapshot-bound research archive",
            ],
        },
        "sources": sources,
    }


def render_markdown(report: dict[str, Any], digest: str) -> str:
    panel = report["candidate_panel"]
    human = report["restricted_human_review_qa"]
    pilot = report["restricted_real_human_pilot"]
    validators = report["real_pilot_deterministic_validation"]
    task_pool = report["surface_clean_real_human_development_pool"]
    validation_campaign = report["development_task_validation_campaign"]
    current_manifest = report["prospective_current_model_manifest"]
    current_catalog = report["current_model_catalog_audit"]
    rows = [
        f"| {name.replace('_', ' ')} | {gate['status']} |"
        for name, gate in report["release_gates"].items()
    ]
    decision = report["decision"]
    return "\n".join(
        (
            "# FlavourBench Season 1 readiness",
            "",
            f"Artifact SHA-256: `{digest}`",
            "",
            "## What is ready",
            "",
            f"The `{panel['counts']['models']}`-endpoint candidate panel has real provider and "
            "Epicure contract evidence under the production structured-output contract. This is "
            "compatibility evidence, not a quality ranking.",
            "",
            f"A human-review QA batch contains `{human['completed_presentations']}` "
            "presentations. It is restricted because the bound consent document was inactive; "
            "no preference or repeatability aggregate is republished here, and the batch remains "
            "ineligible for research, paper, rating, or leaderboard use.",
            "",
            f"Its governed scope audit quarantines `{human['governed_quarantine_tasks']}` tasks; "
            f"`{human['reviewed_quarantine_tasks']}` of those tasks occurred in this review batch.",
            "",
            "The restricted diagnostic contains "
            f"`{pilot['primary_human_judgments']}` real primary judgments, of which "
            f"`{pilot['finish_clean_primary_judgments']}` are finish-clean. These are explicitly "
            "excluded from the paper and leaderboard.",
            "",
            "Deterministic validator v1.3 receipts cover "
            f"`{validators.get('observed', {}).get('real_response_arms', 0)}` real arms. They "
            "identify non-normal completions and review triggers; they do not assign culinary "
            "quality.",
            "",
            "The v2 development pool retains "
            f"`{task_pool.get('counts', {}).get('surface_clean_general_track_candidates', 0)}` "
            "surface-clean, real-human general-track candidates after specialist and missing-"
            "context quarantine.",
            "",
            "The current calibration instrument contains "
            f"`{validation_campaign['task_count']}` tasks and requires "
            f"`{validation_campaign['required_independent_reviews_per_task']}` independent "
            "answer-blind reviews per task. The live database contains "
            f"`{validation_campaign.get('observed', {}).get('blind_validity_records', 0)}` "
            "sealed blind records and "
            f"`{validation_campaign['independently_validated_tasks']}` independently validated "
            "tasks at this cutoff. Missing labels are not imputed; agreement statistics remain "
            "undefined until a task has three complete reviews.",
            "",
            f"A no-call manifest freezes `{current_manifest['model_count']}` current endpoints "
            f"under `{current_manifest['evidence_protocol']}` with an "
            f"`{current_manifest['max_output_tokens']}`-token final-response ceiling. It is a "
            "prospective engineering contract, not result evidence.",
            "",
            "The live catalog audit matched "
            f"`{current_catalog.get('counts', {}).get('freshness_contract_passed', 0)}` of "
            f"`{current_catalog.get('counts', {}).get('manifest_models', 0)}` frozen routes at "
            f"`{current_catalog.get('observed_at')}`. It made no provider generations or Epicure "
            "calls and therefore contributes no quality observations.",
            "",
            "## Release gates",
            "",
            "| Gate | Status |",
            "|---|---|",
            *rows,
            "",
            "## Decision",
            "",
            f"- Closed generation admission ready: "
            f"`{str(decision['closed_generation_admission_ready']).lower()}`",
            f"- Public collection ready: `{str(decision['public_collection_ready']).lower()}`",
            f"- Quality leaderboard release ready: "
            f"`{str(decision['quality_leaderboard_release_ready']).lower()}`",
            "",
            decision["paper_boundary"],
            "",
            "## Remaining work",
            "",
            *(f"{index}. {action}" for index, action in enumerate(decision["next_actions"], 1)),
            "",
        )
    )


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(canonical_bytes(report)).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"season1-readiness-audit-{digest}.json"
    markdown_path = output_dir / f"season1-readiness-audit-{digest}.md"
    json_path.write_text(
        json.dumps({**report, "artifact_sha256": digest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report, digest), encoding="utf-8")
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--parity-audit", type=Path, default=DEFAULT_PARITY_AUDIT)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--human-review-dir", type=Path, default=DEFAULT_HUMAN_DIR)
    parser.add_argument("--real-human-pilot", type=Path, default=DEFAULT_REAL_HUMAN_PILOT)
    parser.add_argument(
        "--real-pilot-validators", type=Path, default=DEFAULT_REAL_PILOT_VALIDATORS
    )
    parser.add_argument(
        "--surface-clean-tasks", type=Path, default=DEFAULT_SURFACE_CLEAN_TASKS
    )
    parser.add_argument(
        "--development-validation-packet",
        type=Path,
        default=DEFAULT_DEVELOPMENT_VALIDATION_PACKET,
    )
    parser.add_argument(
        "--current-model-manifest", type=Path, default=DEFAULT_CURRENT_MODEL_MANIFEST
    )
    parser.add_argument(
        "--current-catalog-audit", type=Path, default=DEFAULT_CURRENT_CATALOG_AUDIT
    )
    parser.add_argument("--study-design", type=Path, default=DEFAULT_STUDY_DESIGN)
    parser.add_argument("--method-validation", type=Path, default=DEFAULT_METHOD_VALIDATION)
    parser.add_argument("--epicure-release", type=Path, default=DEFAULT_EPICURE_RELEASE)
    parser.add_argument("--public-consent", type=Path, default=DEFAULT_PUBLIC_CONSENT)
    parser.add_argument("--post-collection-item-audit", type=Path)
    parser.add_argument("--generation-reliability-panel", type=Path)
    parser.add_argument("--prompt-sensitivity-audit", type=Path)
    parser.add_argument("--practical-cookability-execution", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    panel_path = args.panel.resolve()
    parity_path = args.parity_audit.resolve()
    human_path = (
        args.human_review.resolve()
        if args.human_review
        else latest_human_report(args.human_review_dir.resolve())
    )
    design_path = args.study_design.resolve()
    method_validation_path = args.method_validation.resolve()
    epicure_path = args.epicure_release.resolve()
    consent_path = args.public_consent.resolve()
    panel = load_json(panel_path)
    parity = load_json(parity_path)
    human = load_json(human_path)
    real_human_pilot_path = args.real_human_pilot.resolve()
    real_pilot_validators_path = args.real_pilot_validators.resolve()
    surface_clean_tasks_path = args.surface_clean_tasks.resolve()
    development_validation_packet_path = args.development_validation_packet.resolve()
    current_model_manifest_path = args.current_model_manifest.resolve()
    current_catalog_audit_path = args.current_catalog_audit.resolve()
    real_human_pilot = load_json(real_human_pilot_path)
    real_pilot_validators = load_json(real_pilot_validators_path)
    surface_clean_tasks = load_json(surface_clean_tasks_path)
    development_validation_packet = load_json(development_validation_packet_path)
    current_model_manifest = load_json(current_model_manifest_path)
    current_catalog_audit = load_json(current_catalog_audit_path)
    design = load_json(design_path)
    method_validation = load_json(method_validation_path)
    epicure = load_json(epicure_path)
    robustness_paths = {
        "post_collection_item_audit": args.post_collection_item_audit,
        "generation_reliability_panel": args.generation_reliability_panel,
        "prompt_sensitivity_audit": args.prompt_sensitivity_audit,
        "practical_cookability_execution": args.practical_cookability_execution,
    }
    robustness_evidence = {
        name: load_json(path.resolve())
        for name, path in robustness_paths.items()
        if path is not None
    }
    services = {name: service_state(name) for name in SERVICES}
    consent = consent_state(consent_path)
    sources = [
        source_record(panel_path, panel),
        source_record(parity_path, parity),
        source_record(human_path, human),
        source_record(real_human_pilot_path, real_human_pilot),
        source_record(real_pilot_validators_path, real_pilot_validators),
        source_record(surface_clean_tasks_path, surface_clean_tasks),
        source_record(development_validation_packet_path, development_validation_packet),
        source_record(current_model_manifest_path, current_model_manifest),
        source_record(current_catalog_audit_path, current_catalog_audit),
        source_record(design_path, design),
        source_record(method_validation_path, method_validation),
        source_record(epicure_path, epicure),
        {
            "path": str(DEFAULT_ROBUSTNESS_EVIDENCE_CONTRACT.relative_to(ROOT)),
            "file_sha256": sha256_file(DEFAULT_ROBUSTNESS_EVIDENCE_CONTRACT),
            "schema_version": "flavourbench-season1-validity-robustness-evidence-v1",
        },
        consent,
        *(
            source_record(path.resolve(), robustness_evidence[name])
            for name, path in robustness_paths.items()
            if path is not None
        ),
    ]
    report = build_report(
        panel=panel,
        parity=parity,
        human=human,
        study_design=design,
        method_validation=method_validation,
        epicure_release=epicure,
        database=control_plane_state(),
        services=services,
        consent=consent,
        sources=sources,
        robustness_evidence=robustness_evidence,
        real_human_pilot=real_human_pilot,
        real_pilot_validators=real_pilot_validators,
        surface_clean_tasks=surface_clean_tasks,
        development_validation_packet=development_validation_packet,
        current_model_manifest=current_model_manifest,
        current_catalog_audit=current_catalog_audit,
    )
    json_path, markdown_path = write_report(report, args.output_dir.resolve())
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    run()
