"""Harden reviewer and task-validation trigger function resolution.

Revision ID: 0033_reviewer_task_guard_hardening
Revises: 0032_task_validation_audit_singleton
"""

from __future__ import annotations

from alembic import op

revision = "0033_reviewer_task_guard_hardening"
down_revision = "0032_task_validation_audit_singleton"
branch_labels = None
depends_on = None


FUNCTION_DEFINITIONS = (
    """
    CREATE OR REPLACE FUNCTION public.flavourbench_reviewer_evidence_append_only_v1()
    RETURNS trigger LANGUAGE plpgsql
    SET search_path = pg_catalog, public
    AS $$
    BEGIN
      RAISE EXCEPTION 'reviewer identity and admission evidence is append-only';
    END;
    $$
    """,
    """
    CREATE OR REPLACE FUNCTION public.flavourbench_reviewer_credential_lifecycle_v1()
    RETURNS trigger LANGUAGE plpgsql
    SET search_path = pg_catalog, public
    AS $$
    BEGIN
      IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'reviewer access credentials cannot be deleted';
      END IF;
      IF (NEW.id, NEW.season_id, NEW.reviewer_id, NEW.identity_binding_id,
          NEW.credential_prefix, NEW.secret_hmac_sha256, NEW.hmac_key_id,
          NEW.credential_kind, NEW.scopes_json::text, NEW.maximum_uses,
          NEW.not_before, NEW.expires_at, NEW.created_at)
         IS DISTINCT FROM
         (OLD.id, OLD.season_id, OLD.reviewer_id, OLD.identity_binding_id,
          OLD.credential_prefix, OLD.secret_hmac_sha256, OLD.hmac_key_id,
          OLD.credential_kind, OLD.scopes_json::text, OLD.maximum_uses,
          OLD.not_before, OLD.expires_at, OLD.created_at) THEN
        RAISE EXCEPTION 'reviewer credential contract is immutable';
      END IF;
      IF OLD.status IN ('consumed', 'revoked') THEN
        RAISE EXCEPTION 'terminal reviewer credential is immutable';
      END IF;
      IF NEW.use_count < OLD.use_count OR NEW.use_count > OLD.use_count + 1 THEN
        RAISE EXCEPTION 'reviewer credential use count is not monotone';
      END IF;
      IF NEW.status = 'active' AND
         (NEW.use_count >= NEW.maximum_uses OR NEW.consumed_at IS NOT NULL OR
          NEW.revoked_at IS NOT NULL) THEN
        RAISE EXCEPTION 'active reviewer credential has terminal state';
      ELSIF NEW.status = 'consumed' AND
            (NEW.use_count <> NEW.maximum_uses OR NEW.consumed_at IS NULL OR
             NEW.revoked_at IS NOT NULL) THEN
        RAISE EXCEPTION 'consumed reviewer credential is incomplete';
      ELSIF NEW.status = 'revoked' AND
            (NEW.revoked_at IS NULL OR NEW.consumed_at IS NOT NULL) THEN
        RAISE EXCEPTION 'revoked reviewer credential is incomplete';
      END IF;
      RETURN NEW;
    END;
    $$
    """,
    """
    CREATE OR REPLACE FUNCTION public.flavourbench_reviewer_family_admission_guard_v1()
    RETURNS trigger LANGUAGE plpgsql
    SET search_path = pg_catalog, public
    AS $$
    DECLARE aligned_count integer;
    DECLARE calibration_required boolean;
    DECLARE minimum_accuracy integer;
    BEGIN
      IF NEW.admission_policy_json->>'schema_version' <>
         'flavourbench-reviewer-admission-policy-v1' THEN
        RAISE EXCEPTION 'reviewer admission policy schema is invalid';
      END IF;
      calibration_required :=
        (NEW.admission_policy_json->>'requires_calibration')::boolean;
      minimum_accuracy :=
        (NEW.admission_policy_json->>'minimum_accuracy_milli')::integer;
      IF minimum_accuracy < 0 OR minimum_accuracy > 1000 THEN
        RAISE EXCEPTION 'reviewer admission accuracy threshold is invalid';
      END IF;
      SELECT count(*) INTO aligned_count
      FROM public.reviewer_identity_bindings AS binding
      JOIN public.reviewer_qualification_evidence AS qualification
        ON qualification.id = NEW.qualification_evidence_id
      JOIN public.expert_reviewers AS reviewer ON reviewer.id = NEW.reviewer_id
      WHERE binding.id = NEW.identity_binding_id
        AND binding.season_id = NEW.season_id
        AND binding.reviewer_id = NEW.reviewer_id
        AND binding.assurance_level = 'server_verified'
        AND binding.roles_json::jsonb ? NEW.review_role
        AND qualification.season_id = NEW.season_id
        AND qualification.reviewer_id = NEW.reviewer_id
        AND qualification.identity_binding_id = binding.id
        AND qualification.family = NEW.family
        AND qualification.verification_status = 'verified'
        AND NEW.valid_from >= qualification.verified_at
        AND (qualification.valid_until IS NULL OR
             NEW.valid_until <= qualification.valid_until)
        AND ((qualification.affiliation_class = 'independent_external' AND
              NEW.cohort = 'expert_independent' AND
              qualification.independence_verified IS TRUE AND
              qualification.conflict_cleared IS TRUE) OR
             (qualification.affiliation_class = 'product_affiliated' AND
              NEW.cohort = 'expert_product_affiliated') OR
             (qualification.affiliation_class = 'provider_affiliated' AND
              NEW.cohort = 'expert_provider_affiliated'))
        AND reviewer.active IS TRUE;
      IF aligned_count <> 1 THEN
        RAISE EXCEPTION 'reviewer admission identity or qualification is misaligned';
      END IF;
      IF calibration_required AND NEW.calibration_ballot_id IS NULL THEN
        RAISE EXCEPTION 'reviewer admission requires calibration';
      END IF;
      IF NEW.calibration_ballot_id IS NOT NULL THEN
        SELECT count(*) INTO aligned_count
        FROM public.reviewer_calibration_ballots AS ballot
        JOIN public.reviewer_calibration_sets AS calibration_set
          ON calibration_set.id = ballot.calibration_set_id
        WHERE ballot.id = NEW.calibration_ballot_id
          AND ballot.season_id = NEW.season_id
          AND ballot.reviewer_id = NEW.reviewer_id
          AND ballot.identity_binding_id = NEW.identity_binding_id
          AND ballot.passed IS TRUE
          AND ballot.accuracy_milli >= minimum_accuracy
          AND ballot.completed_at <= NEW.valid_from
          AND calibration_set.season_id = NEW.season_id
          AND calibration_set.family = NEW.family
          AND calibration_set.synthetic_arms = 0;
        IF aligned_count <> 1 THEN
          RAISE EXCEPTION 'reviewer admission calibration is misaligned';
        END IF;
      END IF;
      RETURN NEW;
    END;
    $$
    """,
    """
    CREATE OR REPLACE FUNCTION public.flavourbench_verified_expert_vote_guard_v1()
    RETURNS trigger LANGUAGE plpgsql
    SET search_path = pg_catalog, public
    AS $$
    DECLARE aligned_count integer;
    BEGIN
      IF NEW.provenance_status = 'expert_verified_v1' THEN
        IF NEW.cohort = 'public' THEN
          RAISE EXCEPTION 'verified reviewer provenance is restricted to expert cohorts';
        END IF;
        SELECT count(*) INTO aligned_count
        FROM public.battles AS battle
        JOIN public.reviewer_identity_bindings AS binding
          ON binding.id = NEW.reviewer_identity_binding_id
        JOIN public.reviewer_family_admissions AS admission
          ON admission.id = NEW.reviewer_family_admission_id
        JOIN public.expert_reviewers AS reviewer
          ON reviewer.id = NEW.reviewer_id
        WHERE battle.id = NEW.battle_id
          AND binding.season_id = battle.season_id
          AND binding.reviewer_id = NEW.reviewer_id
          AND binding.assurance_level = 'server_verified'
          AND admission.season_id = battle.season_id
          AND admission.reviewer_id = NEW.reviewer_id
          AND admission.identity_binding_id = binding.id
          AND admission.family = battle.category
          AND admission.review_role = 'output_rater'
          AND admission.cohort = NEW.cohort
          AND COALESCE(NEW.created_at, now()) BETWEEN admission.valid_from
                                                  AND admission.valid_until
          AND reviewer.active IS TRUE;
        IF aligned_count <> 1 THEN
          RAISE EXCEPTION 'verified expert vote is not backed by one active family admission';
        END IF;
      END IF;
      RETURN NEW;
    END;
    $$
    """,
    """
    CREATE OR REPLACE FUNCTION public.flavourbench_task_validation_append_only_v1()
    RETURNS trigger LANGUAGE plpgsql
    SET search_path = pg_catalog, public
    AS $$
    BEGIN
      RAISE EXCEPTION 'task-validation campaign evidence is append-only';
    END;
    $$
    """,
    """
    CREATE OR REPLACE FUNCTION public.flavourbench_task_validation_event_guard_v1()
    RETURNS trigger LANGUAGE plpgsql
    SET search_path = pg_catalog, public
    AS $$
    DECLARE aligned_count integer;
    DECLARE expected_role text;
    DECLARE expected_kind text;
    BEGIN
      SELECT count(*) INTO aligned_count
      FROM public.reviewer_identity_bindings AS binding
      JOIN public.expert_reviewers AS reviewer ON reviewer.id = NEW.reviewer_id
      WHERE binding.id = NEW.identity_binding_id
        AND binding.season_id = NEW.season_id
        AND binding.reviewer_id = NEW.reviewer_id
        AND binding.person_commitment_sha256 = NEW.person_commitment_sha256
        AND binding.assurance_level = 'server_verified'
        AND reviewer.active IS TRUE;
      IF aligned_count <> 1 THEN
        RAISE EXCEPTION 'task-validation event identity is inadmissible';
      END IF;

      IF NEW.event_type IN (
          'blind_ballot', 'criterion_pack_confirmation', 'adjudication'
      ) THEN
        expected_role := CASE WHEN NEW.event_type = 'adjudication'
                              THEN 'task_adjudicator' ELSE 'task_validator' END;
        SELECT count(*) INTO aligned_count
        FROM public.reviewer_family_admissions AS admission
        WHERE admission.id = NEW.family_admission_id
          AND admission.season_id = NEW.season_id
          AND admission.reviewer_id = NEW.reviewer_id
          AND admission.identity_binding_id = NEW.identity_binding_id
          AND admission.review_role = expected_role
          AND admission.cohort = 'expert_independent'
          AND admission.evidence_bundle_sha256 =
              NEW.reviewer_admission_receipt_sha256
          AND COALESCE(NEW.created_at, now()) BETWEEN admission.valid_from
                                                  AND admission.valid_until;
      ELSE
        expected_kind := CASE WHEN NEW.event_type = 'rights_batch_audit'
                              THEN 'rights' ELSE 'contamination' END;
        SELECT count(*) INTO aligned_count
        FROM public.task_validation_audit_authorizations AS audit_auth
        WHERE audit_auth.id = NEW.audit_authorization_id
          AND audit_auth.season_id = NEW.season_id
          AND audit_auth.reviewer_id = NEW.reviewer_id
          AND audit_auth.identity_binding_id = NEW.identity_binding_id
          AND audit_auth.audit_kind = expected_kind
          AND audit_auth.authorization_sha256 =
              NEW.reviewer_admission_receipt_sha256;
      END IF;
      IF aligned_count <> 1 THEN
        RAISE EXCEPTION 'task-validation event authority is inadmissible';
      END IF;
      RETURN NEW;
    END;
    $$
    """,
)


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"unsupported database dialect for 0033: {dialect}")
    if dialect == "postgresql":
        for definition in FUNCTION_DEFINITIONS:
            op.execute(definition)


def downgrade() -> None:
    raise RuntimeError(
        "downgrade across reviewer and task-validation guard hardening is prohibited"
    )
