"""Bind campaign-v6 rights audits to the exact verified replay.

Revision ID: 0034_task_validation_replay_binding
Revises: 0033_reviewer_task_guard_hardening
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import sqlalchemy as sa

from alembic import op

revision = "0034_task_validation_replay_binding"
down_revision = "0033_reviewer_task_guard_hardening"
branch_labels = None
depends_on = None

CAMPAIGN_SHA256 = "76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709"
REPLAY_SHA256 = "89f6dede2826e27bcd69eb764e32bd7a203b371f0098831c78c1077383383157"
PLAN_SHA256 = "d550f516c33a7ff77f81d9b3d542e3752eb886c9d3d6591bc6c96a9fec5894c2"
SAMPLE_SEED_SHA256 = "aae16a208727c7f64b4a89607929c783891afc580f7def0b6fdb4d6849ea49f7"
SAMPLE_IDS = (
    "25edd53f-82c2-5ebb-b058-807487e6db8b",
    "976ab5c3-0a48-55c5-8373-ba377f63caf5",
    "17a39314-e09e-5f38-9cb2-768f3b7c643c",
    "fb42473c-ea53-521a-a0f5-9a72efb6c250",
    "6735309d-3f6e-5e18-9b62-4a6d817c9e6a",
    "e89c6c7f-b3ea-5e5c-92d5-aff512690230",
    "5982509e-5a77-55d0-b283-d95a427f0c63",
    "423b3c67-2c8b-52e9-9b45-e8d8ad0dca05",
    "a1c64071-58ee-5a78-a1dc-e32e853ea572",
    "210d72af-ea8f-54ba-b231-5ced2e448195",
    "04b7528e-f430-5461-9bf8-4ea7e682549f",
    "5968388e-b7e9-5d10-b4df-e59394ec5d04",
    "8abdb882-8b8b-5792-aa99-5aaa0c6f7dcb",
    "dc041268-8129-5ffb-a280-af204c0c2c31",
    "a2c0e720-12b2-5005-be8c-8385ae81d3d5",
    "e13f4620-45a7-5a00-be5c-ef429f0674b5",
    "848c2474-0a5c-5468-a21e-59d168196e15",
    "9c622e63-5e68-598d-96e5-c8d76f6eb3ed",
    "310a6f75-6487-53c9-b26d-7e8a4d470573",
    "be6124d6-2d96-57f9-aa48-2d6af4eee159",
    "4f132dc2-f3ac-5bfa-8d2c-02cc99925ba7",
    "1c1ddc8f-48b7-5a13-af06-493865b5e0d0",
    "a4e901d3-ea6c-532c-9fd5-0323648f1b1b",
    "39e1c00d-26be-54ff-81a8-39216ffc42b7",
)
ANOMALY_IDS = ("210d72af-ea8f-54ba-b231-5ced2e448195",)
REQUIRED_IDS = (
    "04b7528e-f430-5461-9bf8-4ea7e682549f",
    "17a39314-e09e-5f38-9cb2-768f3b7c643c",
    "1c1ddc8f-48b7-5a13-af06-493865b5e0d0",
    "210d72af-ea8f-54ba-b231-5ced2e448195",
    "25edd53f-82c2-5ebb-b058-807487e6db8b",
    "310a6f75-6487-53c9-b26d-7e8a4d470573",
    "39e1c00d-26be-54ff-81a8-39216ffc42b7",
    "423b3c67-2c8b-52e9-9b45-e8d8ad0dca05",
    "4f132dc2-f3ac-5bfa-8d2c-02cc99925ba7",
    "5968388e-b7e9-5d10-b4df-e59394ec5d04",
    "5982509e-5a77-55d0-b283-d95a427f0c63",
    "6735309d-3f6e-5e18-9b62-4a6d817c9e6a",
    "848c2474-0a5c-5468-a21e-59d168196e15",
    "8abdb882-8b8b-5792-aa99-5aaa0c6f7dcb",
    "976ab5c3-0a48-55c5-8373-ba377f63caf5",
    "9c622e63-5e68-598d-96e5-c8d76f6eb3ed",
    "a1c64071-58ee-5a78-a1dc-e32e853ea572",
    "a2c0e720-12b2-5005-be8c-8385ae81d3d5",
    "a4e901d3-ea6c-532c-9fd5-0323648f1b1b",
    "be6124d6-2d96-57f9-aa48-2d6af4eee159",
    "dc041268-8129-5ffb-a280-af204c0c2c31",
    "e13f4620-45a7-5a00-be5c-ef429f0674b5",
    "e89c6c7f-b3ea-5e5c-92d5-aff512690230",
    "fb42473c-ea53-521a-a0f5-9a72efb6c250",
)
FORMAL_CONTAMINATION_METHODS = ("exact", "fuzzy", "ngram", "semantic", "web")


def _plan() -> dict[str, object]:
    return {
        "schema_version": "flavourbench-task-validation-batch-audit-plan-v2",
        "campaign_sha256": CAMPAIGN_SHA256,
        "audit_kind": "rights",
        "sample_seed_commitment_sha256": SAMPLE_SEED_SHA256,
        "sample_candidate_ids": list(SAMPLE_IDS),
        "anomaly_or_hit_candidate_ids": list(ANOMALY_IDS),
        "required_candidate_ids": list(REQUIRED_IDS),
        "automated_evidence_sha256": REPLAY_SHA256,
        "automated_evidence_verified": True,
        "automated_evidence_scope": "rights_snapshot_integrity_and_local_prompt_risk_only",
        "rights_snapshot_integrity_verified": True,
        "local_prompt_risk_replay_verified": True,
        "contamination_campaign_coverage_verified": False,
        "formal_contamination_methods_required": list(FORMAL_CONTAMINATION_METHODS),
        "model_outputs_available": False,
        "rank_eligible": False,
    }


PLAN = _plan()
PLAN_JSON = json.dumps(PLAN, separators=(",", ":"))
if (
    hashlib.sha256(json.dumps(PLAN, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    != PLAN_SHA256
):  # pragma: no cover - migration source invariant
    raise RuntimeError("0034 frozen audit-plan digest differs")


def _json_object(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, Mapping) else None
    return None


def _event_binding_valid(payload: object) -> bool:
    value = _json_object(payload)
    return bool(
        value is not None
        and value.get("audit_kind") == "rights"
        and value.get("audit_plan_sha256") == PLAN_SHA256
        and value.get("automated_evidence_sha256") == REPLAY_SHA256
        and value.get("automated_evidence_verified") is True
        and value.get("rights_snapshot_integrity_verified") is True
        and value.get("local_prompt_risk_replay_verified") is True
        and value.get("contamination_campaign_coverage_verified") is False
        and value.get("reviewed_candidate_ids") == list(REQUIRED_IDS)
    )


def _assert_existing_rows() -> None:
    bind = op.get_bind()
    authorizations = list(
        bind.execute(
            sa.text(
                "SELECT id, campaign_sha256, audit_kind, automated_evidence_sha256, "
                "audit_plan_json, audit_plan_sha256 "
                "FROM task_validation_audit_authorizations"
            )
        ).mappings()
    )
    valid_ids: set[str] = set()
    for row in authorizations:
        if not (
            row["campaign_sha256"] == CAMPAIGN_SHA256
            and row["audit_kind"] == "rights"
            and row["automated_evidence_sha256"] == REPLAY_SHA256
            and _json_object(row["audit_plan_json"]) == PLAN
            and row["audit_plan_sha256"] == PLAN_SHA256
        ):
            raise RuntimeError("0034 refuses an unbound historical audit authorization")
        valid_ids.add(str(row["id"]))
    events = list(
        bind.execute(
            sa.text(
                "SELECT event_type, audit_authorization_id, payload_json "
                "FROM task_validation_campaign_events "
                "WHERE event_type IN ('rights_batch_audit', 'contamination_batch_audit')"
            )
        ).mappings()
    )
    if any(
        row["event_type"] != "rights_batch_audit"
        or str(row["audit_authorization_id"]) not in valid_ids
        or not _event_binding_valid(row["payload_json"])
        for row in events
    ):
        raise RuntimeError("0034 refuses an unbound historical audit event")


def _create_postgresql_guards() -> None:
    plan_literal = PLAN_JSON.replace("'", "''")
    required_literal = json.dumps(list(REQUIRED_IDS), separators=(",", ":")).replace("'", "''")
    op.get_bind().exec_driver_sql(
        f"""
        CREATE OR REPLACE FUNCTION public.flavourbench_task_validation_audit_replay_guard_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF NEW.campaign_sha256 <> '{CAMPAIGN_SHA256}'
             OR NEW.audit_kind <> 'rights'
             OR NEW.automated_evidence_sha256 <> '{REPLAY_SHA256}'
             OR NEW.audit_plan_sha256 <> '{PLAN_SHA256}'
             OR NEW.audit_plan_json::jsonb <> '{plan_literal}'::jsonb THEN
            RAISE EXCEPTION 'task-validation audit replay binding is inadmissible';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.get_bind().exec_driver_sql(
        f"""
        CREATE OR REPLACE FUNCTION public.flavourbench_task_validation_event_replay_guard_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        DECLARE aligned_count integer;
        BEGIN
          IF NEW.event_type IN ('rights_batch_audit', 'contamination_batch_audit') THEN
            IF NEW.event_type <> 'rights_batch_audit'
               OR NEW.payload_json->>'audit_kind' IS DISTINCT FROM 'rights'
               OR NEW.payload_json->>'audit_plan_sha256' IS DISTINCT FROM '{PLAN_SHA256}'
               OR NEW.payload_json->>'automated_evidence_sha256'
                  IS DISTINCT FROM '{REPLAY_SHA256}'
               OR (NEW.payload_json::jsonb)->'automated_evidence_verified'
                  IS DISTINCT FROM 'true'::jsonb
               OR (NEW.payload_json::jsonb)->'rights_snapshot_integrity_verified'
                  IS DISTINCT FROM 'true'::jsonb
               OR (NEW.payload_json::jsonb)->'local_prompt_risk_replay_verified'
                  IS DISTINCT FROM 'true'::jsonb
               OR (NEW.payload_json::jsonb)->'contamination_campaign_coverage_verified'
                  IS DISTINCT FROM 'false'::jsonb
               OR (NEW.payload_json::jsonb)->'reviewed_candidate_ids'
                  IS DISTINCT FROM '{required_literal}'::jsonb THEN
              RAISE EXCEPTION 'task-validation audit event replay binding is inadmissible';
            END IF;
            SELECT pg_catalog.count(*) INTO aligned_count
            FROM public.task_validation_audit_authorizations AS audit_auth
            WHERE audit_auth.id = NEW.audit_authorization_id
              AND audit_auth.campaign_sha256 = '{CAMPAIGN_SHA256}'
              AND audit_auth.audit_kind = 'rights'
              AND audit_auth.automated_evidence_sha256 = '{REPLAY_SHA256}'
              AND audit_auth.audit_plan_sha256 = '{PLAN_SHA256}'
              AND audit_auth.audit_plan_json::jsonb = '{plan_literal}'::jsonb;
            IF aligned_count <> 1 THEN
              RAISE EXCEPTION 'task-validation audit event replay authority is inadmissible';
            END IF;
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_task_validation_audit_authorizations_replay_v1
        BEFORE INSERT ON public.task_validation_audit_authorizations
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_task_validation_audit_replay_guard_v1()
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_task_validation_campaign_events_replay_v1
        BEFORE INSERT ON public.task_validation_campaign_events
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_task_validation_event_replay_guard_v1()
        """
    )


def _create_sqlite_guards() -> None:
    sample = json.dumps(list(SAMPLE_IDS), separators=(",", ":"))
    anomalies = json.dumps(list(ANOMALY_IDS), separators=(",", ":"))
    required = json.dumps(list(REQUIRED_IDS), separators=(",", ":"))
    methods = json.dumps(list(FORMAL_CONTAMINATION_METHODS), separators=(",", ":"))
    op.execute(
        f"""
        CREATE TRIGGER trg_task_validation_audit_authorizations_replay_v1
        BEFORE INSERT ON task_validation_audit_authorizations FOR EACH ROW
        WHEN NEW.campaign_sha256 <> '{CAMPAIGN_SHA256}'
          OR NEW.audit_kind <> 'rights'
          OR NEW.automated_evidence_sha256 <> '{REPLAY_SHA256}'
          OR NEW.audit_plan_sha256 <> '{PLAN_SHA256}'
          OR (SELECT count(*) FROM json_each(NEW.audit_plan_json)) <> 16
          OR json_extract(NEW.audit_plan_json, '$.schema_version') <>
             'flavourbench-task-validation-batch-audit-plan-v2'
          OR json_extract(NEW.audit_plan_json, '$.campaign_sha256') <> '{CAMPAIGN_SHA256}'
          OR json_extract(NEW.audit_plan_json, '$.audit_kind') <> 'rights'
          OR json_extract(NEW.audit_plan_json, '$.sample_seed_commitment_sha256') <>
             '{SAMPLE_SEED_SHA256}'
          OR json_extract(NEW.audit_plan_json, '$.sample_candidate_ids') <> json('{sample}')
          OR json_extract(NEW.audit_plan_json, '$.anomaly_or_hit_candidate_ids') <>
             json('{anomalies}')
          OR json_extract(NEW.audit_plan_json, '$.required_candidate_ids') <>
             json('{required}')
          OR json_extract(NEW.audit_plan_json, '$.automated_evidence_sha256') <>
             '{REPLAY_SHA256}'
          OR json_extract(NEW.audit_plan_json, '$.automated_evidence_verified') IS NOT 1
          OR json_extract(NEW.audit_plan_json, '$.automated_evidence_scope') <>
             'rights_snapshot_integrity_and_local_prompt_risk_only'
          OR json_extract(NEW.audit_plan_json, '$.rights_snapshot_integrity_verified') IS NOT 1
          OR json_extract(NEW.audit_plan_json, '$.local_prompt_risk_replay_verified') IS NOT 1
          OR json_extract(NEW.audit_plan_json, '$.contamination_campaign_coverage_verified')
             IS NOT 0
          OR json_extract(NEW.audit_plan_json, '$.formal_contamination_methods_required') <>
             json('{methods}')
          OR json_extract(NEW.audit_plan_json, '$.model_outputs_available') IS NOT 0
          OR json_extract(NEW.audit_plan_json, '$.rank_eligible') IS NOT 0
        BEGIN
          SELECT RAISE(ABORT, 'task-validation audit replay binding is inadmissible');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER trg_task_validation_campaign_events_replay_v1
        BEFORE INSERT ON task_validation_campaign_events FOR EACH ROW
        WHEN NEW.event_type IN ('rights_batch_audit', 'contamination_batch_audit')
         AND (
           NEW.event_type <> 'rights_batch_audit'
           OR json_extract(NEW.payload_json, '$.audit_kind') IS NOT 'rights'
           OR json_extract(NEW.payload_json, '$.audit_plan_sha256') IS NOT '{PLAN_SHA256}'
           OR json_extract(NEW.payload_json, '$.automated_evidence_sha256') IS NOT
              '{REPLAY_SHA256}'
           OR json_extract(NEW.payload_json, '$.automated_evidence_verified') IS NOT 1
           OR json_extract(NEW.payload_json, '$.rights_snapshot_integrity_verified') IS NOT 1
           OR json_extract(NEW.payload_json, '$.local_prompt_risk_replay_verified') IS NOT 1
           OR json_extract(NEW.payload_json, '$.contamination_campaign_coverage_verified')
              IS NOT 0
           OR json_extract(NEW.payload_json, '$.reviewed_candidate_ids')
              IS NOT json('{required}')
           OR NOT EXISTS (
             SELECT 1 FROM task_validation_audit_authorizations AS audit_auth
             WHERE audit_auth.id = NEW.audit_authorization_id
               AND audit_auth.campaign_sha256 = '{CAMPAIGN_SHA256}'
               AND audit_auth.audit_kind = 'rights'
               AND audit_auth.automated_evidence_sha256 = '{REPLAY_SHA256}'
               AND audit_auth.audit_plan_sha256 = '{PLAN_SHA256}'
           )
         )
        BEGIN
          SELECT RAISE(ABORT, 'task-validation audit event replay binding is inadmissible');
        END;
        """
    )


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"unsupported database dialect for 0034: {dialect}")
    _assert_existing_rows()
    if dialect == "postgresql":
        _create_postgresql_guards()
    else:
        _create_sqlite_guards()


def downgrade() -> None:
    raise RuntimeError("downgrade across task-validation automated replay bindings is prohibited")
