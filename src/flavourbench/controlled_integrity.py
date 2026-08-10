"""Cross-table integrity checks for frozen commercial controlled runs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from .arena import CONTROLLED_SCHEDULER_VERSION, controlled_side_is_reversed
from .models import Battle, ControlledRunAssignment, ResponseArm


class ControlledRunIntegrityError(RuntimeError):
    """A persisted assignment, battle, or arm no longer matches the signed schedule."""


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _terminal_arm_evidence_is_complete(arm: ResponseArm, *, content_redacted: bool) -> bool:
    if arm.completed_at is None or arm.output_json_sha256 is None:
        return False
    if arm.status == "complete":
        return bool(
            (arm.answer_markdown is not None or content_redacted)
            and arm.answer_markdown_sha256
            and arm.actual_provider_slug
            and arm.actual_model_id
            and arm.generation_id
            and arm.provider_generation_ids_json
            and arm.cost_reconciled
            and arm.cost_accounting_basis != "unrecorded"
            and arm.billing_reconciliation_status != "unrecorded"
        )
    if arm.status == "failed":
        return bool(
            arm.cost_reconciled
            and arm.cost_accounting_basis != "unrecorded"
            and arm.billing_reconciliation_status != "unrecorded"
        )
    return False


def verify_controlled_assignment_battle(
    session: Session,
    assignment: ControlledRunAssignment,
    battle: Battle,
    *,
    require_terminal: bool = False,
) -> tuple[ResponseArm, ResponseArm]:
    if assignment.status != "queued" or assignment.battle_id != battle.id:
        raise ControlledRunIntegrityError("controlled assignment is not bound to this battle")
    if (
        battle.controlled_run_id != assignment.controlled_run_id
        or battle.data_stratum != "controlled"
        or battle.task_id != assignment.task_id
        or battle.task_revision != assignment.task_revision
        or battle.prompt_sha256 != assignment.task_prompt_sha256
        or battle.category != assignment.task_family
        or battle.track != assignment.track
        or battle.assignment_seed != assignment.assignment_seed
        or battle.scheduler_version != CONTROLLED_SCHEDULER_VERSION
        or battle.track_assignment_probability != "1/1"
        or battle.model_assignment_probability != "1/1"
        or battle.side_assignment_probability != "1/2"
    ):
        raise ControlledRunIntegrityError(
            "controlled battle scientific fields differ from the frozen assignment"
        )
    if require_terminal and battle.status not in {"complete", "failed"}:
        raise ControlledRunIntegrityError("controlled battle is not terminal")
    if (
        battle.left_arm_id is None
        or battle.right_arm_id is None
        or battle.left_arm_id == battle.right_arm_id
    ):
        raise ControlledRunIntegrityError("controlled battle arm links are incomplete")
    left = session.get(ResponseArm, battle.left_arm_id)
    right = session.get(ResponseArm, battle.right_arm_id)
    if (
        left is None
        or right is None
        or left.battle_id != battle.id
        or right.battle_id != battle.id
        or left.side != "left"
        or right.side != "right"
    ):
        raise ControlledRunIntegrityError(
            "controlled battle arm ownership or side differs from its links"
        )
    if require_terminal:
        allowed_statuses = (
            {"complete"}
            if battle.status == "complete"
            else {
                "complete",
                "failed",
            }
        )
        if (
            battle.completed_at is None
            or left.status not in allowed_statuses
            or right.status not in allowed_statuses
            or not _terminal_arm_evidence_is_complete(left, content_redacted=battle.prompt_redacted)
            or not _terminal_arm_evidence_is_complete(
                right, content_redacted=battle.prompt_redacted
            )
            or _as_utc(left.completed_at) > _as_utc(battle.completed_at)
            or _as_utc(right.completed_at) > _as_utc(battle.completed_at)
        ):
            raise ControlledRunIntegrityError(
                "controlled battle terminal arm evidence is incomplete"
            )

    model_ids = assignment.model_ids_json
    if not isinstance(model_ids, list) or any(not isinstance(item, str) for item in model_ids):
        raise ControlledRunIntegrityError("controlled assignment model list is invalid")
    if assignment.track == "model_arena":
        if len(model_ids) != 2 or len(set(model_ids)) != 2:
            raise ControlledRunIntegrityError("model-arena assignment requires two models")
        expected = [(model_ids[0], "epicure_on"), (model_ids[1], "epicure_on")]
    elif assignment.track == "epicure_uplift":
        if len(model_ids) != 1:
            raise ControlledRunIntegrityError("uplift assignment requires one model")
        expected = [(model_ids[0], "epicure_on"), (model_ids[0], "epicure_off")]
    else:
        raise ControlledRunIntegrityError("controlled assignment track is invalid")
    if controlled_side_is_reversed(assignment.assignment_seed):
        expected.reverse()
    realized = [(left.model_id, left.condition), (right.model_id, right.condition)]
    if realized != expected:
        raise ControlledRunIntegrityError(
            "controlled battle arm identities differ from the committed side assignment"
        )
    return left, right


def verify_controlled_run_bijection(
    session: Session,
    assignments: Sequence[ControlledRunAssignment],
    battles: Sequence[Battle],
    *,
    require_terminal: bool = False,
) -> None:
    if not assignments:
        raise ControlledRunIntegrityError("controlled run has no frozen assignments")
    assignment_battle_ids = [assignment.battle_id for assignment in assignments]
    battle_ids = [battle.id for battle in battles]
    if (
        any(battle_id is None for battle_id in assignment_battle_ids)
        or len(set(assignment_battle_ids)) != len(assignment_battle_ids)
        or len(set(battle_ids)) != len(battle_ids)
        or set(assignment_battle_ids) != set(battle_ids)
    ):
        raise ControlledRunIntegrityError(
            "controlled assignments and battles do not form an exact bijection"
        )
    by_id = {battle.id: battle for battle in battles}
    for assignment in assignments:
        verify_controlled_assignment_battle(
            session,
            assignment,
            by_id[str(assignment.battle_id)],
            require_terminal=require_terminal,
        )
