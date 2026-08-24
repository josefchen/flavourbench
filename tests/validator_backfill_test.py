from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from flavourbench.models import Base, Battle, CatalogModel, ResponseArm, ToolCall, ValidatorResult
from flavourbench.validator_backfill import backfill_and_build_report
from flavourbench.validators import VALIDATOR_VERSION


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _arm(
    *, battle_id: str, side: str, answer: str, output: dict[str, object], finish: str
) -> dict[str, object]:
    return {
        "id": f"arm-{side}",
        "battle_id": battle_id,
        "side": side,
        "condition": "epicure_on" if side == "left" else "epicure_off",
        "model_id": "real/model-v1",
        "execution_backend": "openrouter",
        "provider_slug": "fixed-provider",
        "actual_provider_slug": "Fixed Provider",
        "actual_model_id": "real/model-v1",
        "generation_id": f"generation-{side}",
        "status": "complete",
        "answer_markdown": answer,
        "answer_markdown_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "output_json": output,
        "output_json_sha256": _json_sha256(output),
        "prompt_sha256": "a" * 64,
        "schema_sha256": "b" * 64,
        "tool_schema_sha256": "c" * 64,
        "epicure_release_id": "epicure-real-v1",
        "epicure_bundle_sha256": "d" * 64,
        "finish_reason": finish,
        "cost_reconciled": True,
        "completed_at": datetime.now(UTC),
    }


def test_backfill_is_append_only_idempotent_and_exposes_real_failures() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    prompt = "Use https://example.test and the attached photo to interpret the evidence."
    answer_left = (
        "The similarity data confirms that this ingredient is a functional binder for the "
        "recipe. Test a small batch before changing the entire preparation."
    )
    answer_right = (
        "Run a small controlled batch and compare texture at equal hydration. Record the result "
        "before changing the full recipe, because technique may explain the difference."
    )
    output_left = {
        "answer_markdown": answer_left,
        "ingredient_mentions": [],
        "constraints_addressed": [],
        "uncertainties": [],
    }
    output_right = {
        "answer_markdown": answer_right,
        "ingredient_mentions": [],
        "constraints_addressed": [],
        "uncertainties": [],
    }
    with Session(engine) as session:
        session.add(
            CatalogModel(
                model_id="real/model-v1",
                canonical_slug="real/model-v1",
                name="Real Model v1",
                status="smoke_passed",
            )
        )
        session.flush()
        completed_at = datetime.now(UTC)
        session.execute(
            Battle.__table__.insert().values(
                id="pilot-battle",
                season_id="pilot-season",
                run_class="pilot",
                rank_eligible=False,
                data_stratum="development",
                track="epicure_uplift",
                category="evidence",
                prompt=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                client_nonce_sha256="e" * 64,
                research_consent=False,
                retention_basis="development_research",
                requester_pseudonym="f" * 64,
                status="complete",
                left_arm_id="arm-left",
                right_arm_id="arm-right",
                completed_at=completed_at,
                retention_until=datetime.now(UTC) + timedelta(days=30),
            )
        )
        left = _arm(
            battle_id="pilot-battle",
            side="left",
            answer=answer_left,
            output=output_left,
            finish="stop",
        )
        right = _arm(
            battle_id="pilot-battle",
            side="right",
            answer=answer_right,
            output=output_right,
            finish="length",
        )
        left["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        right["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        session.execute(ResponseArm.__table__.insert(), [left, right])
        session.add(
            ToolCall(
                arm_id="arm-left",
                round_index=0,
                call_index=0,
                tool_name="find_pairings",
                arguments_json={"ingredient": "sample"},
                result_text="real tool result",
                result_sha256=hashlib.sha256(b"real tool result").hexdigest(),
                is_error=False,
            )
        )
        session.flush()

        first = backfill_and_build_report(session)
        assert first["observed"]["real_response_arms"] == 2
        assert first["observed"]["synthetic_or_mock_arms"] == 0
        assert first["observed"]["validator_receipts_verified"] == 14
        assert first["observed"]["non_normal_final_completion_arms"] == 1
        assert first["observed"]["evidence_claim_boundary_warnings"] == 1
        assert first["observed"]["task_surface_integrity_warnings"] == 2

        second = backfill_and_build_report(session)
        assert second == first
        receipts = list(
            session.scalars(
                select(ValidatorResult).where(
                    ValidatorResult.validator_version == VALIDATOR_VERSION
                )
            )
        )
        assert len(receipts) == 14
        evidence = next(
            receipt
            for receipt in receipts
            if receipt.arm_id == "arm-left"
            and receipt.validator_name == "evidence_claim_boundary"
        )
        assert evidence.status == "warn"
        assert "sentence_sha256" in evidence.detail_json["findings"][0]
        assert "sentence" not in evidence.detail_json["findings"][0]
