from __future__ import annotations

import hashlib
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import Base

settings = get_settings()
EXPECTED_SCHEMA_REVISION = "0035_participant_lifecycle_privacy"
_BUDGET_AUTHORITY_FUNCTIONS = {
    "flavourbench_reserve_battle_budget": {
        "signature": "text",
        "flavourbench_api": True,
        "flavourbench_worker": False,
    },
    "flavourbench_settle_battle_budget": {
        "signature": "text",
        "flavourbench_api": True,
        "flavourbench_worker": True,
    },
    "flavourbench_apply_bedrock_billing_adjustment": {
        "signature": "text",
        "flavourbench_api": False,
        "flavourbench_worker": False,
    },
    "flavourbench_register_bedrock_billing_adjustment": {
        "signature": "text,jsonb",
        "flavourbench_api": True,
        "flavourbench_worker": False,
    },
}
_NORMAL_FINISH_GUARD_FUNCTION_BODIES = {
    "flavourbench_response_arm_normal_finish_guard": """
        BEGIN
            IF NEW.status = 'complete'
               AND pg_catalog.lower(pg_catalog.btrim(
                    COALESCE(NEW.finish_reason::text, ''::text)
               )) NOT IN ('completed', 'end_turn', 'stop', 'stop_sequence') THEN
                RAISE EXCEPTION
                    'complete response arm requires a normal provider finish reason';
            END IF;
            RETURN NEW;
        END;
    """,
    "flavourbench_vote_normal_finish_guard": """
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM public.battles AS battle
                JOIN public.response_arms AS left_arm
                  ON left_arm.id = battle.left_arm_id
                 AND left_arm.battle_id = battle.id
                 AND left_arm.side = 'left'
                 AND left_arm.status = 'complete'
                 AND pg_catalog.lower(pg_catalog.btrim(
                        COALESCE(left_arm.finish_reason::text, ''::text)
                     )) IN ('completed', 'end_turn', 'stop', 'stop_sequence')
                JOIN public.response_arms AS right_arm
                  ON right_arm.id = battle.right_arm_id
                 AND right_arm.battle_id = battle.id
                 AND right_arm.side = 'right'
                 AND right_arm.status = 'complete'
                 AND pg_catalog.lower(pg_catalog.btrim(
                        COALESCE(right_arm.finish_reason::text, ''::text)
                     )) IN ('completed', 'end_turn', 'stop', 'stop_sequence')
                WHERE battle.id = NEW.battle_id
                  AND battle.status = 'complete'
            ) THEN
                RAISE EXCEPTION
                    'vote requires two normally finished response arms';
            END IF;
            RETURN NEW;
        END;
    """,
}
_NORMAL_FINISH_GUARD_TRIGGERS = {
    "trg_response_arm_normal_finish_guard": {
        "table": "response_arms",
        "function": "flavourbench_response_arm_normal_finish_guard",
        "type": 23,
        "definition_fragment": "before insert or update of status, finish_reason",
    },
    "trg_vote_normal_finish_guard": {
        "table": "votes",
        "function": "flavourbench_vote_normal_finish_guard",
        "type": 7,
        "definition_fragment": "before insert on public.votes",
    },
}
_REVIEWER_TASK_VALIDATION_GUARD_BODY_SHA256 = {
    "flavourbench_reviewer_evidence_append_only_v1": (
        "ed1f3cc26aa23c6555092577b02ff9eed50a1cf80d2f8add585d54dfef76a468"
    ),
    "flavourbench_reviewer_credential_lifecycle_v1": (
        "a5cedcee7a6755f79eec9726a3bfbbcd52ee27dcc117aff975186d758a44699b"
    ),
    "flavourbench_reviewer_family_admission_guard_v1": (
        "c11507f54d47fe966d64e727803f10db0f1d5063a9c75646c98f0ad7afeb551a"
    ),
    "flavourbench_verified_expert_vote_guard_v1": (
        "00f6783b5b442fb95b0214a8a094af1ea6ec672a8aef2600469e2c64ba4397e6"
    ),
    "flavourbench_task_validation_append_only_v1": (
        "ff815d16c1344ce8679304f9ed5dee0e80300840db9db9db71c2693e3abc443d"
    ),
    "flavourbench_task_validation_event_guard_v1": (
        "8f09b823567d75a407963ed60245a541755a9792d5d0e3d86d7e214623732ba3"
    ),
    "flavourbench_task_validation_audit_replay_guard_v1": (
        "758ea11ce67eeef35115e22b86b2bb6f259fd7961390f810cf954fcd5d46c87e"
    ),
    "flavourbench_task_validation_event_replay_guard_v1": (
        "9f6f84452444326c24d79ba48bacddb923187e083b84cddb89417773e012e834"
    ),
    "flavourbench_task_validation_candidate_capacity_v1": (
        "ce9931e59c90046960a8c66f16103075b841b1595f428951fab6eab018539f8b"
    ),
}
_REVIEWER_TASK_VALIDATION_GUARD_TRIGGERS = {
    "trg_reviewer_identity_bindings_append_only_v1": (
        "reviewer_identity_bindings",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_qualification_evidence_append_only_v1": (
        "reviewer_qualification_evidence",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_calibration_sets_append_only_v1": (
        "reviewer_calibration_sets",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_calibration_ballots_append_only_v1": (
        "reviewer_calibration_ballots",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_family_admissions_append_only_v1": (
        "reviewer_family_admissions",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_access_credentials_lifecycle_v1": (
        "reviewer_access_credentials",
        "flavourbench_reviewer_credential_lifecycle_v1",
        27,
    ),
    "trg_reviewer_family_admissions_guard_v1": (
        "reviewer_family_admissions",
        "flavourbench_reviewer_family_admission_guard_v1",
        7,
    ),
    "trg_votes_verified_expert_provenance_v1": (
        "votes",
        "flavourbench_verified_expert_vote_guard_v1",
        23,
    ),
    "trg_task_validation_audit_authorizations_append_only_v1": (
        "task_validation_audit_authorizations",
        "flavourbench_task_validation_append_only_v1",
        27,
    ),
    "trg_task_validation_campaign_events_append_only_v1": (
        "task_validation_campaign_events",
        "flavourbench_task_validation_append_only_v1",
        27,
    ),
    "trg_task_validation_campaign_events_authority_v1": (
        "task_validation_campaign_events",
        "flavourbench_task_validation_event_guard_v1",
        7,
    ),
    "trg_task_validation_audit_authorizations_replay_v1": (
        "task_validation_audit_authorizations",
        "flavourbench_task_validation_audit_replay_guard_v1",
        7,
    ),
    "trg_task_validation_campaign_events_replay_v1": (
        "task_validation_campaign_events",
        "flavourbench_task_validation_event_replay_guard_v1",
        7,
    ),
    "trg_task_validation_campaign_events_candidate_capacity_v1": (
        "task_validation_campaign_events",
        "flavourbench_task_validation_candidate_capacity_v1",
        7,
    ),
}
_PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256 = {
    "flavourbench_participant_append_only_v1": (
        "ab523c8f49d33e8cf3dc0c0057808754cb82739407199c9494ce90452da770f2"
    ),
    "flavourbench_consent_acceptance_guard_v1": (
        "3f855d976badebf3921e14573337a38bbc8dcd0a381ebb5004c20a43205066da"
    ),
    "flavourbench_participation_insert_guard_v1": (
        "b584d2f4c1601c1208fb0860ede66200b7736115c22ca9ee3081e38612fa9bce"
    ),
    "flavourbench_withdrawal_receipt_guard_v1": (
        "50462ba5f3d7e19ff60b424fc53bbf5effb049b8a0282c88dfabfd2364ac49c8"
    ),
    "flavourbench_retention_schedule_guard_v1": (
        "5f146896a852f42117d56ee61ded4df7fa11b0ba7bb260665fc3b7d9e317034c"
    ),
    "flavourbench_deletion_receipt_guard_v1": (
        "b3325e94124fa9cc8b9eecb26114518523838959fa7e22a32c6a6ebfc7db9c0f"
    ),
    "flavourbench_enrollment_offer_lifecycle_v1": (
        "b88946323266d1ddf363ad1967476d5bd1ec17de0aabf3de59a7f4d02aa9acda"
    ),
    "flavourbench_participation_lifecycle_v1": (
        "0f1dd7037e212627be8c4c40f2aa0e02277582b4797552b865543e616cb9da6c"
    ),
    "flavourbench_participant_forward_authority_v1": (
        "fec9709f754aba87d3fe71bda067590951cb2e32a9e7b17577de61619849650f"
    ),
    "flavourbench_reviewer_privacy_lifecycle_v1": (
        "eacdfd5aa7364b2bbd8b9849387e55ce10ef36ec928d341e5df8fa016984d8eb"
    ),
}
_PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS = {
    **{
        f"trg_{table}_append_only_v1": (table, "flavourbench_participant_append_only_v1", 27, None)
        for table in (
            "reviewer_consent_acceptances",
            "reviewer_withdrawal_receipts",
            "reviewer_retention_schedules",
            "reviewer_deletion_receipts",
        )
    },
    "trg_reviewer_consent_acceptances_guard_v1": (
        "reviewer_consent_acceptances",
        "flavourbench_consent_acceptance_guard_v1",
        7,
        None,
    ),
    "trg_reviewer_participation_lifecycles_insert_guard_v1": (
        "reviewer_participation_lifecycles",
        "flavourbench_participation_insert_guard_v1",
        7,
        None,
    ),
    "trg_reviewer_withdrawal_receipts_guard_v1": (
        "reviewer_withdrawal_receipts",
        "flavourbench_withdrawal_receipt_guard_v1",
        7,
        None,
    ),
    "trg_reviewer_retention_schedules_guard_v1": (
        "reviewer_retention_schedules",
        "flavourbench_retention_schedule_guard_v1",
        7,
        None,
    ),
    "trg_reviewer_deletion_receipts_guard_v1": (
        "reviewer_deletion_receipts",
        "flavourbench_deletion_receipt_guard_v1",
        7,
        None,
    ),
    "trg_reviewer_enrollment_offers_lifecycle_v1": (
        "reviewer_enrollment_offers",
        "flavourbench_enrollment_offer_lifecycle_v1",
        27,
        None,
    ),
    "trg_reviewer_participation_lifecycles_lifecycle_v1": (
        "reviewer_participation_lifecycles",
        "flavourbench_participation_lifecycle_v1",
        27,
        None,
    ),
    "trg_reviewer_identity_bindings_participant_consent_v1": (
        "reviewer_identity_bindings",
        "flavourbench_participant_forward_authority_v1",
        7,
        "identity",
    ),
    "trg_reviewer_qualification_evidence_participant_consent_v1": (
        "reviewer_qualification_evidence",
        "flavourbench_participant_forward_authority_v1",
        7,
        "qualification",
    ),
    "trg_reviewer_access_credentials_participation_v1": (
        "reviewer_access_credentials",
        "flavourbench_participant_forward_authority_v1",
        23,
        "credential",
    ),
    "trg_controlled_run_reviewers_participation_v1": (
        "controlled_run_reviewers",
        "flavourbench_participant_forward_authority_v1",
        23,
        "assignment",
    ),
    "trg_votes_participation_v1": (
        "votes",
        "flavourbench_participant_forward_authority_v1",
        23,
        "vote",
    ),
    "trg_task_validation_campaign_events_participation_v1": (
        "task_validation_campaign_events",
        "flavourbench_participant_forward_authority_v1",
        7,
        "task_event",
    ),
    "trg_task_validation_audit_authorizations_participation_v1": (
        "task_validation_audit_authorizations",
        "flavourbench_participant_forward_authority_v1",
        7,
        "task_audit_authorization",
    ),
    "trg_expert_reviewers_privacy_lifecycle_v1": (
        "expert_reviewers",
        "flavourbench_reviewer_privacy_lifecycle_v1",
        19,
        None,
    ),
}


def _normalize_sql(value: object) -> str:
    return " ".join(str(value).split())


def _assert_postgresql_reviewer_task_validation_guards(connection: object) -> None:
    function_rows = connection.execute(  # type: ignore[attr-defined]
        text(
            """
            SELECT p.proname,
                   pg_catalog.pg_get_userbyid(p.proowner) AS owner,
                   p.prosecdef,
                   l.lanname AS language,
                   pg_catalog.pg_get_function_identity_arguments(p.oid) AS arguments,
                   pg_catalog.pg_get_function_result(p.oid) AS result,
                   COALESCE(pg_catalog.array_to_string(p.proconfig, ','), '') AS config,
                   p.prosrc
            FROM pg_catalog.pg_proc AS p
            JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
            JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang
            WHERE n.nspname = 'public' AND p.proname = ANY(:names)
            """
        ),
        {"names": list(_REVIEWER_TASK_VALIDATION_GUARD_BODY_SHA256)},
    ).mappings()
    functions = {str(row["proname"]): row for row in function_rows}
    if set(functions) != set(_REVIEWER_TASK_VALIDATION_GUARD_BODY_SHA256):
        raise RuntimeError("database reviewer/task-validation guard functions are incomplete")
    for name, expected_body_sha256 in _REVIEWER_TASK_VALIDATION_GUARD_BODY_SHA256.items():
        row = functions[name]
        config = str(row["config"]).replace('"', "")
        if (
            row["owner"] != "flavourbench_owner"
            or row["prosecdef"] is not False
            or row["language"] != "plpgsql"
            or row["arguments"] != ""
            or row["result"] != "trigger"
            or config != "search_path=pg_catalog, public"
        ):
            raise RuntimeError(
                f"database reviewer/task-validation guard metadata is unsafe: {name}"
            )
        observed_body_sha256 = hashlib.sha256(_normalize_sql(row["prosrc"]).encode()).hexdigest()
        if observed_body_sha256 != expected_body_sha256:
            raise RuntimeError(
                f"database reviewer/task-validation guard body is stale or unsafe: {name}"
            )

    trigger_rows = connection.execute(  # type: ignore[attr-defined]
        text(
            """
            SELECT t.tgname,
                   t.tgenabled,
                   t.tgtype,
                   table_class.relname AS table_name,
                   function_proc.proname AS function_name
            FROM pg_catalog.pg_trigger AS t
            JOIN pg_catalog.pg_class AS table_class ON table_class.oid = t.tgrelid
            JOIN pg_catalog.pg_namespace AS table_namespace
              ON table_namespace.oid = table_class.relnamespace
            JOIN pg_catalog.pg_proc AS function_proc ON function_proc.oid = t.tgfoid
            JOIN pg_catalog.pg_namespace AS function_namespace
              ON function_namespace.oid = function_proc.pronamespace
            WHERE table_namespace.nspname = 'public'
              AND function_namespace.nspname = 'public'
              AND NOT t.tgisinternal
              AND t.tgname = ANY(:names)
            """
        ),
        {"names": list(_REVIEWER_TASK_VALIDATION_GUARD_TRIGGERS)},
    ).mappings()
    triggers = {str(row["tgname"]): row for row in trigger_rows}
    if set(triggers) != set(_REVIEWER_TASK_VALIDATION_GUARD_TRIGGERS):
        raise RuntimeError("database reviewer/task-validation guard triggers are incomplete")
    for name, (
        table_name,
        function_name,
        trigger_type,
    ) in _REVIEWER_TASK_VALIDATION_GUARD_TRIGGERS.items():
        row = triggers[name]
        if (
            row["tgenabled"] != "O"
            or row["table_name"] != table_name
            or row["function_name"] != function_name
            or row["tgtype"] != trigger_type
        ):
            raise RuntimeError(f"database reviewer/task-validation guard trigger is unsafe: {name}")


def _assert_postgresql_participant_lifecycle_guards(connection: object) -> None:
    function_rows = connection.execute(  # type: ignore[attr-defined]
        text(
            """
            SELECT p.proname,
                   pg_catalog.pg_get_userbyid(p.proowner) AS owner,
                   p.prosecdef,
                   l.lanname AS language,
                   pg_catalog.pg_get_function_identity_arguments(p.oid) AS arguments,
                   pg_catalog.pg_get_function_result(p.oid) AS result,
                   COALESCE(pg_catalog.array_to_string(p.proconfig, ','), '') AS config,
                   p.prosrc
            FROM pg_catalog.pg_proc AS p
            JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
            JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang
            WHERE n.nspname = 'public' AND p.proname = ANY(:names)
            """
        ),
        {"names": list(_PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256)},
    ).mappings()
    functions = {str(row["proname"]): row for row in function_rows}
    if set(functions) != set(_PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256):
        raise RuntimeError("database participant-lifecycle guard functions are incomplete")
    for name, expected_sha256 in _PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256.items():
        row = functions[name]
        config = str(row["config"]).replace('"', "")
        if (
            row["owner"] != "flavourbench_owner"
            or row["prosecdef"] is not False
            or row["language"] != "plpgsql"
            or row["arguments"] != ""
            or row["result"] != "trigger"
            or config != "search_path=pg_catalog, public"
        ):
            raise RuntimeError(f"database participant-lifecycle guard metadata is unsafe: {name}")
        observed = hashlib.sha256(_normalize_sql(row["prosrc"]).encode()).hexdigest()
        if observed != expected_sha256:
            raise RuntimeError(f"database participant-lifecycle guard body is stale: {name}")

    trigger_rows = connection.execute(  # type: ignore[attr-defined]
        text(
            """
            SELECT t.tgname,
                   t.tgenabled,
                   t.tgtype,
                   table_class.relname AS table_name,
                   function_proc.proname AS function_name,
                   pg_catalog.pg_get_triggerdef(t.oid, false) AS definition
            FROM pg_catalog.pg_trigger AS t
            JOIN pg_catalog.pg_class AS table_class ON table_class.oid = t.tgrelid
            JOIN pg_catalog.pg_namespace AS table_namespace
              ON table_namespace.oid = table_class.relnamespace
            JOIN pg_catalog.pg_proc AS function_proc ON function_proc.oid = t.tgfoid
            JOIN pg_catalog.pg_namespace AS function_namespace
              ON function_namespace.oid = function_proc.pronamespace
            WHERE table_namespace.nspname = 'public'
              AND function_namespace.nspname = 'public'
              AND NOT t.tgisinternal
              AND t.tgname = ANY(:names)
            """
        ),
        {"names": list(_PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS)},
    ).mappings()
    triggers = {str(row["tgname"]): row for row in trigger_rows}
    if set(triggers) != set(_PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS):
        raise RuntimeError("database participant-lifecycle guard triggers are incomplete")
    for name, expected in _PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS.items():
        table_name, function_name, trigger_type, argument = expected
        row = triggers[name]
        definition = _normalize_sql(row["definition"]).lower()
        if (
            row["tgenabled"] != "O"
            or row["table_name"] != table_name
            or row["function_name"] != function_name
            or row["tgtype"] != trigger_type
            or (argument is not None and f"'{argument}'" not in definition)
        ):
            raise RuntimeError(f"database participant-lifecycle guard trigger is unsafe: {name}")


def _assert_postgresql_normal_finish_guards(connection: object) -> None:
    function_rows = connection.execute(  # type: ignore[attr-defined]
        text(
            """
            SELECT p.proname,
                   pg_catalog.pg_get_userbyid(p.proowner) AS owner,
                   p.prosecdef,
                   l.lanname AS language,
                   pg_catalog.pg_get_function_identity_arguments(p.oid) AS arguments,
                   pg_catalog.pg_get_function_result(p.oid) AS result,
                   COALESCE(pg_catalog.array_to_string(p.proconfig, ','), '') AS config,
                   p.prosrc
            FROM pg_catalog.pg_proc AS p
            JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
            JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang
            WHERE n.nspname = 'public' AND p.proname = ANY(:names)
            """
        ),
        {"names": list(_NORMAL_FINISH_GUARD_FUNCTION_BODIES)},
    ).mappings()
    functions = {str(row["proname"]): row for row in function_rows}
    if set(functions) != set(_NORMAL_FINISH_GUARD_FUNCTION_BODIES):
        raise RuntimeError("database normal-finish guard functions are incomplete")
    for name, expected_body in _NORMAL_FINISH_GUARD_FUNCTION_BODIES.items():
        row = functions[name]
        config = str(row["config"]).replace('"', "")
        if (
            row["owner"] != "flavourbench_owner"
            or row["prosecdef"] is not False
            or row["language"] != "plpgsql"
            or row["arguments"] != ""
            or row["result"] != "trigger"
            or config != "search_path=pg_catalog, public"
        ):
            raise RuntimeError(f"database normal-finish guard metadata is unsafe: {name}")
        if _normalize_sql(row["prosrc"]) != _normalize_sql(expected_body):
            raise RuntimeError(f"database normal-finish guard body is stale or unsafe: {name}")

    trigger_rows = connection.execute(  # type: ignore[attr-defined]
        text(
            """
            SELECT t.tgname,
                   t.tgenabled,
                   t.tgtype,
                   table_class.relname AS table_name,
                   function_proc.proname AS function_name,
                   pg_catalog.pg_get_triggerdef(t.oid, false) AS definition
            FROM pg_catalog.pg_trigger AS t
            JOIN pg_catalog.pg_class AS table_class ON table_class.oid = t.tgrelid
            JOIN pg_catalog.pg_namespace AS table_namespace
              ON table_namespace.oid = table_class.relnamespace
            JOIN pg_catalog.pg_proc AS function_proc ON function_proc.oid = t.tgfoid
            JOIN pg_catalog.pg_namespace AS function_namespace
              ON function_namespace.oid = function_proc.pronamespace
            WHERE table_namespace.nspname = 'public'
              AND function_namespace.nspname = 'public'
              AND NOT t.tgisinternal
              AND t.tgname = ANY(:names)
            """
        ),
        {"names": list(_NORMAL_FINISH_GUARD_TRIGGERS)},
    ).mappings()
    triggers = {str(row["tgname"]): row for row in trigger_rows}
    if set(triggers) != set(_NORMAL_FINISH_GUARD_TRIGGERS):
        raise RuntimeError("database normal-finish guard triggers are incomplete or disabled")
    for name, expected in _NORMAL_FINISH_GUARD_TRIGGERS.items():
        row = triggers[name]
        definition = _normalize_sql(row["definition"]).lower()
        if (
            row["tgenabled"] != "O"
            or row["table_name"] != expected["table"]
            or row["function_name"] != expected["function"]
            or row["tgtype"] != expected["type"]
            or str(expected["definition_fragment"]) not in definition
        ):
            raise RuntimeError(f"database normal-finish guard trigger is unsafe: {name}")


def _assert_postgresql_budget_authority(connection: object) -> None:
    function_rows = connection.execute(  # type: ignore[attr-defined]
        text(
            """
            SELECT p.proname,
                   pg_catalog.pg_get_userbyid(p.proowner) AS owner,
                   p.prosecdef,
                   COALESCE(pg_catalog.array_to_string(p.proconfig, ','), '') AS config
            FROM pg_catalog.pg_proc AS p
            JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
            WHERE n.nspname = 'public'
              AND p.proname = ANY(:names)
              AND (
                  (
                      p.proname IN (
                          'flavourbench_reserve_battle_budget',
                          'flavourbench_settle_battle_budget'
                      )
                      AND pg_catalog.pg_get_function_identity_arguments(p.oid) =
                          'p_battle_id text'
                  )
                  OR (
                      p.proname = 'flavourbench_apply_bedrock_billing_adjustment'
                      AND pg_catalog.pg_get_function_identity_arguments(p.oid) =
                          'p_crosscheck_id text'
                  )
                  OR (
                      p.proname = 'flavourbench_register_bedrock_billing_adjustment'
                      AND pg_catalog.pg_get_function_identity_arguments(p.oid) =
                          'p_season_id text, p_request jsonb'
                  )
              )
            """
        ),
        {"names": list(_BUDGET_AUTHORITY_FUNCTIONS)},
    ).mappings()
    functions = {str(row["proname"]): row for row in function_rows}
    if set(functions) != set(_BUDGET_AUTHORITY_FUNCTIONS):
        raise RuntimeError("database budget authority functions are incomplete")
    for name, row in functions.items():
        config = str(row["config"]).replace('"', "")
        if (
            row["owner"] != "flavourbench_owner"
            or row["prosecdef"] is not True
            or "search_path=pg_catalog, public" not in config
        ):
            raise RuntimeError(f"database budget authority metadata is unsafe: {name}")
        signature = str(_BUDGET_AUTHORITY_FUNCTIONS[name]["signature"])
        for role in ("flavourbench_api", "flavourbench_worker"):
            expected = bool(_BUDGET_AUTHORITY_FUNCTIONS[name][role])
            observed = bool(
                connection.execute(  # type: ignore[attr-defined]
                    text("SELECT pg_catalog.has_function_privilege(:role, :signature, 'EXECUTE')"),
                    {"role": role, "signature": f"public.{name}({signature})"},
                ).scalar_one()
            )
            if observed is not expected:
                raise RuntimeError(f"database budget authority grant is unsafe: {name}/{role}")
        if bool(
            connection.execute(  # type: ignore[attr-defined]
                text("SELECT pg_catalog.has_function_privilege('public', :signature, 'EXECUTE')"),
                {"signature": f"public.{name}({signature})"},
            ).scalar_one()
        ):
            raise RuntimeError(f"database budget authority is executable by PUBLIC: {name}")

    required_triggers = {
        "trg_budget_counter_authority_seasons",
        "trg_budget_counter_authority_season_provider_budgets",
        "trg_budget_counter_authority_provider_account_budgets",
        "trg_budget_counter_authority_controlled_runs",
        "trg_battle_reservation_authority",
        "trg_cost_event_authority",
        "trg_bedrock_membership_seal",
        "trg_generation_attempt_arm_authority",
    }
    enabled_triggers = set(
        connection.execute(  # type: ignore[attr-defined]
            text(
                """
                SELECT t.tgname
                FROM pg_catalog.pg_trigger AS t
                JOIN pg_catalog.pg_class AS c ON c.oid = t.tgrelid
                JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND NOT t.tgisinternal
                  AND t.tgenabled = 'O' AND t.tgname = ANY(:names)
                """
            ),
            {"names": list(required_triggers)},
        ).scalars()
    )
    if enabled_triggers != required_triggers:
        raise RuntimeError("database budget authority triggers are incomplete or disabled")

    protected_columns = {
        "flavourbench_api": {
            ("seasons", "budget_used_micros"),
            ("seasons", "budget_reserved_micros"),
            ("season_provider_budgets", "budget_used_micros"),
            ("season_provider_budgets", "budget_reserved_micros"),
            ("provider_account_budgets", "budget_used_micros"),
            ("provider_account_budgets", "budget_reserved_micros"),
            ("controlled_runs", "budget_used_micros"),
            ("controlled_runs", "budget_reserved_micros"),
            ("battles", "reserved_cost_micros"),
            ("battles", "provider_reservations_json"),
        },
        "flavourbench_worker": {
            ("seasons", "status"),
            ("seasons", "budget_used_micros"),
            ("seasons", "budget_reserved_micros"),
            ("season_provider_budgets", "budget_used_micros"),
            ("season_provider_budgets", "budget_reserved_micros"),
            ("provider_account_budgets", "budget_used_micros"),
            ("provider_account_budgets", "budget_reserved_micros"),
            ("controlled_runs", "budget_used_micros"),
            ("controlled_runs", "budget_reserved_micros"),
            ("battles", "reserved_cost_micros"),
            ("battles", "provider_reservations_json"),
        },
    }
    for role, columns in protected_columns.items():
        for table_name, column_name in columns:
            if bool(
                connection.execute(  # type: ignore[attr-defined]
                    text(
                        "SELECT pg_catalog.has_column_privilege("
                        ":role, :table_name, :column_name, 'UPDATE')"
                    ),
                    {
                        "role": role,
                        "table_name": f"public.{table_name}",
                        "column_name": column_name,
                    },
                ).scalar_one()
            ):
                raise RuntimeError(
                    f"runtime role can update protected ledger column: "
                    f"{role}/{table_name}.{column_name}"
                )
        for table_name in (
            "bedrock_billing_crosschecks",
            "bedrock_billing_crosscheck_arms",
        ):
            if bool(
                connection.execute(  # type: ignore[attr-defined]
                    text("SELECT pg_catalog.has_table_privilege(:role, :table_name, 'INSERT')"),
                    {"role": role, "table_name": f"public.{table_name}"},
                ).scalar_one()
            ):
                raise RuntimeError(
                    f"runtime role can directly insert governed billing evidence: "
                    f"{role}/{table_name}"
                )

    required_constraints = {
        "ck_response_arms_nonnegative_cost",
        "ck_cost_events_nonnegative_charge",
        "ck_cost_events_nonpositive_release",
        "ck_bedrock_billing_crosschecks_not_self_superseding",
    }
    constraints = set(
        connection.execute(  # type: ignore[attr-defined]
            text(
                """
                SELECT con.conname
                FROM pg_catalog.pg_constraint AS con
                JOIN pg_catalog.pg_namespace AS n ON n.oid = con.connamespace
                WHERE n.nspname = 'public' AND con.convalidated
                  AND con.conname = ANY(:names)
                """
            ),
            {"names": list(required_constraints)},
        ).scalars()
    )
    if constraints != required_constraints:
        raise RuntimeError("database budget authority constraints are incomplete")


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args=connect_args,
)
if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_database() -> None:
    if settings.auto_create_schema:
        Base.metadata.create_all(bind=engine)


def assert_production_fixture_free(
    session: Session,
    *,
    environment: str | None = None,
) -> None:
    """Reject a production database that still contains development fixtures.

    The first public prototype used ``flavourbench/mock-*`` catalog entries and
    mock execution rows.  Those records are useful in isolated test databases,
    but carrying them into a commercial deployment would make the public model
    catalog ambiguous and could admit a job against a non-provider endpoint.
    """

    if (environment or settings.environment) != "production":
        return
    connection = session.connection()
    inspector = inspect(connection)
    fixture_queries = {
        "catalog_models": (
            "SELECT 1 FROM catalog_models WHERE model_id LIKE 'flavourbench/mock-%' LIMIT 1"
        ),
        "season_models": (
            "SELECT 1 FROM season_models "
            "WHERE execution_backend = 'mock' OR provider_slug = 'mock' LIMIT 1"
        ),
        "battles": "SELECT 1 FROM battles WHERE run_class = 'mock' LIMIT 1",
    }
    for table_name, query in fixture_queries.items():
        if inspector.has_table(table_name) and connection.execute(text(query)).first() is not None:
            raise RuntimeError(
                "production database contains legacy fixture records; "
                "quarantine the prototype volume and provision a fresh database"
            )


def database_readiness(
    session: Session,
    *,
    expected_role: str | None = None,
) -> dict[str, str]:
    """Verify the live connection, governed schema, and runtime identity."""

    connection = session.connection()
    connection.execute(text("SELECT 1"))
    dialect = connection.dialect.name
    role = (
        str(connection.execute(text("SELECT current_user")).scalar_one())
        if dialect == "postgresql"
        else "sqlite"
    )
    schema_revision = (
        str(connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one())
        if inspect(connection).has_table("alembic_version")
        else "unmanaged"
    )
    if settings.environment == "production":
        if schema_revision != EXPECTED_SCHEMA_REVISION:
            raise RuntimeError("database schema revision does not match the service release")
        if expected_role is not None and role != expected_role:
            raise RuntimeError("database connection uses the wrong runtime role")
        _assert_postgresql_budget_authority(connection)
        _assert_postgresql_normal_finish_guards(connection)
        _assert_postgresql_reviewer_task_validation_guards(connection)
        _assert_postgresql_participant_lifecycle_guards(connection)
        assert_production_fixture_free(session)
    return {
        "database": "ready",
        "databaseDialect": dialect,
        "databaseRole": role,
        "schemaRevision": schema_revision,
    }


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
