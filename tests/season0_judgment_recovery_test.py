from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from flavourbench import season0_judgment_recovery as recovery
from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_judge_manifest import MAX_OUTPUT_TOKENS
from flavourbench.season0_judge_protocol import (
    DIMENSIONS,
    JUDGE_SYSTEM_PROMPT_SHA256,
    JUDGMENT_SCHEMA_SHA256,
    PROTOCOL_VERSION,
)
from flavourbench.season0_judging import _atomic_write, build_work_items
from flavourbench.season0_judgment_recovery import (
    CONFIRMATION,
    _attempt_exposure,
    _is_documented_throttle_rejection,
    _plan_payload,
    freeze_recovery_plan,
    recover_throttles,
)


def _sealed(value: dict[str, object]) -> dict[str, object]:
    return {**value, "artifact_sha256": sha256_json(value)}


def _record(judgment_id: str, error_type: str) -> dict[str, object]:
    return _sealed(
        {
            "judgment_id": judgment_id,
            "comparison_id": f"comparison-{judgment_id}",
            "judge": {"judge_id": "judge-1"},
            "orientation": "original",
            "status": "failed",
            "error_type": error_type,
            "reservation_usd": "0.25",
            "result": {
                "estimated_cost_usd": None,
                "usage": None,
                "raw_output_sha256": None,
                "request_id_sha256": None,
            },
        }
    )


def test_recovery_plan_selects_only_documented_throttle_rejections() -> None:
    throttle = _record("throttle", "ThrottlingException")
    timeout = _record("timeout", "ReadTimeoutError")
    summary_payload = {
        "status": "collection_complete",
        "synthetic_judgments": 0,
        "task_bank_artifact_sha256": "a" * 64,
        "comparison_manifest_artifact_sha256": "b" * 64,
        "judge_manifest_artifact_sha256": "c" * 64,
        "judge_cost_envelope_artifact_sha256": "d" * 64,
        "counts": {"planned_judgments": 2, "terminal_judgments": 2},
        "judgment_artifact_sha256s": sorted(
            [throttle["artifact_sha256"], timeout["artifact_sha256"]]
        ),
    }
    plan = _plan_payload(
        original_summary=summary_payload,
        latest={"throttle": throttle, "timeout": timeout},
        original_summary_sha256="e" * 64,
    )

    assert plan["counts"]["eligible_documented_throttle_rejections"] == 1
    assert [row["judgment_id"] for row in plan["recovery_items"]] == ["throttle"]
    assert "ReadTimeoutError" in plan["excluded_error_types"]


def test_documented_throttles_are_zero_delivery_not_conservative_reservations() -> None:
    throttle = _record("throttle", "ThrottlingException")
    timeout = _record("timeout", "ReadTimeoutError")
    success = {
        "status": "success",
        "reservation_usd": "1.00",
        "result": {"estimated_cost_usd": "0.10"},
    }

    attributed, reservations, rejected = _attempt_exposure([throttle, timeout, success])

    assert _is_documented_throttle_rejection(throttle) is True
    assert attributed == Decimal("0.10")
    assert reservations == Decimal("0.25")
    assert rejected == 1


def _artifact(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "artifact_sha256": sha256_json(payload)}


@pytest.mark.asyncio
async def test_real_recovery_is_append_only_idempotent_and_one_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_bank = _artifact(
        {
            "synthetic_tasks": 0,
            "tasks": [
                {
                    "task_id": "task-1",
                    "family": "composition",
                    "prompt": "How should these ingredients be balanced?",
                    "human_reference": {"text": "Balance salt, acid, and aroma."},
                }
            ],
        }
    )
    left_answer = "Use acid cautiously and taste after each addition."
    right_answer = "Build the aromatic base first, then balance with salt."
    arms_dir = tmp_path / "arms"
    _atomic_write(
        arms_dir,
        "arm-left",
        {
            "arm_id": "left",
            "completed_at": "2026-07-16T10:00:00Z",
            "result": {"answer_markdown": left_answer},
        },
    )
    _atomic_write(
        arms_dir,
        "arm-right",
        {
            "arm_id": "right",
            "completed_at": "2026-07-16T10:00:00Z",
            "result": {"answer_markdown": right_answer},
        },
    )
    comparison_manifest = _artifact(
        {
            "synthetic_comparisons": 0,
            "task_bank_artifact_sha256": task_bank["artifact_sha256"],
            "comparisons": [
                {
                    "comparison_id": "comparison-1",
                    "judgable": True,
                    "track": "model_arena",
                    "task_id": "task-1",
                    "task_family": "composition",
                    "left_arm_id": "left",
                    "right_arm_id": "right",
                    "left": {
                        "season_model_id": "model-left",
                        "condition": "epicure_on",
                        "answer_sha256": recovery.sha256_text(left_answer),
                    },
                    "right": {
                        "season_model_id": "model-right",
                        "condition": "epicure_on",
                        "answer_sha256": recovery.sha256_text(right_answer),
                    },
                }
            ],
        }
    )
    judge = {
        "judge_id": "judge-1",
        "display_name": "Judge One",
        "canonical_model_id": "provider/judge-one",
        "requested_endpoint_id": "provider.judge-one-v1",
        "input_usd_per_million": "1.0",
        "output_usd_per_million": "2.0",
    }
    judge_manifest = _artifact(
        {
            "task_bank_artifact_sha256": task_bank["artifact_sha256"],
            "hard_cap_usd": "5000",
            "judges": [judge],
            "protocol": {
                "version": PROTOCOL_VERSION,
                "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
                "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
        }
    )
    cost_envelope = _artifact({"hard_cap_usd": "5000"})
    output_dir = tmp_path / "judging"
    judgment_dir = output_dir / "judgments"
    items = build_work_items(comparison_manifest, judge_manifest)
    assert len(items) == 2
    records = []
    for index, item in enumerate(items):
        payload: dict[str, object] = {
            "schema_version": recovery.JUDGMENT_SCHEMA_VERSION,
            "judgment_id": item.judgment_id,
            "comparison_id": "comparison-1",
            "track": "model_arena",
            "task_id": "task-1",
            "task_family": "composition",
            "judge": judge,
            "orientation": item.orientation,
            "status": "failed" if index == 0 else "success",
            "error_type": "ThrottlingException" if index == 0 else None,
            "reservation_usd": "0.250000000",
            "completed_at": f"2026-07-16T10:0{index + 1}:00Z",
            "result": {
                "estimated_cost_usd": None if index == 0 else "0.010000000",
                "usage": None,
                "raw_output_sha256": None,
                "request_id_sha256": None,
            },
            "synthetic": False,
        }
        path = _atomic_write(judgment_dir, f"judgment-{item.judgment_id}", payload)
        records.append(json.loads(path.read_bytes()))
    summary = _artifact(
        {
            "status": "collection_complete",
            "synthetic_judgments": 0,
            "task_bank_artifact_sha256": task_bank["artifact_sha256"],
            "comparison_manifest_artifact_sha256": comparison_manifest[
                "artifact_sha256"
            ],
            "judge_manifest_artifact_sha256": judge_manifest["artifact_sha256"],
            "judge_cost_envelope_artifact_sha256": cost_envelope["artifact_sha256"],
            "counts": {"planned_judgments": 2, "terminal_judgments": 2},
            "judgment_artifact_sha256s": sorted(
                str(record["artifact_sha256"]) for record in records
            ),
        }
    )
    plan = freeze_recovery_plan(
        original_summary=summary,
        judgments_dir=judgment_dir,
        output_dir=tmp_path / "plans",
    )
    assert plan["counts"]["eligible_documented_throttle_rejections"] == 1
    persisted_plan = json.loads(Path(str(plan["plan_path"])).read_bytes())

    judgment = {
        "choice": "tie",
        "left": {
            "scores": {dimension: 3 for dimension in DIMENSIONS},
            "fatal_failure": False,
            "summary": "Useful answer.",
        },
        "right": {
            "scores": {dimension: 3 for dimension in DIMENSIONS},
            "fatal_failure": False,
            "summary": "Useful answer.",
        },
        "confidence": "medium",
        "reason_tags": ["none"],
        "rationale": "The answers are similarly useful and practical.",
    }

    class FakeRuntime:
        calls = 0

        def converse(self, **_: object) -> dict[str, object]:
            self.calls += 1
            return {
                "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
                "output": {"message": {"content": [{"text": json.dumps(judgment)}]}},
                "stopReason": "end_turn",
                "ResponseMetadata": {"RequestId": "request-1"},
            }

    runtime = FakeRuntime()
    settings = SimpleNamespace(
        enabled=True,
        live_authorized=True,
        stage="season",
        hard_cap_usd=Decimal("5000"),
    )
    monkeypatch.setattr(
        recovery.BedrockLaneSettings,
        "from_environ",
        classmethod(lambda cls: settings),
    )
    monkeypatch.setattr(
        recovery,
        "create_boto3_clients",
        lambda settings, client_config: SimpleNamespace(runtime=runtime),
    )

    first = await recover_throttles(
        task_bank=task_bank,
        arms_dir=arms_dir,
        comparison_manifest=comparison_manifest,
        judge_manifest=judge_manifest,
        cost_envelope=cost_envelope,
        plan=persisted_plan,
        output_dir=output_dir,
        cap_usd=Decimal("5000"),
        confirmation=CONFIRMATION,
    )
    assert runtime.calls == 1
    assert first["counts"]["terminal_judgments"] == 2
    assert first["counts"]["provider_attempt_records"] == 3
    assert first["counts"]["recovery_attempts"] == 1
    assert first["counts"]["recovered_to_success"] == 1
    assert first["counts"]["documented_zero_delivery_throttle_rejections"] == 1
    assert len(list(judgment_dir.glob("judgment-*.json"))) == 3

    second = await recover_throttles(
        task_bank=task_bank,
        arms_dir=arms_dir,
        comparison_manifest=comparison_manifest,
        judge_manifest=judge_manifest,
        cost_envelope=cost_envelope,
        plan=persisted_plan,
        output_dir=output_dir,
        cap_usd=Decimal("5000"),
        confirmation=CONFIRMATION,
    )
    assert runtime.calls == 1
    assert second["artifact_sha256"] == first["artifact_sha256"]
    assert second["counts"]["provider_attempt_records"] == 3
