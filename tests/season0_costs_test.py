import json
from decimal import Decimal
from pathlib import Path

from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_costs import _rate_cost, reconcile_costs


def test_rate_cost_uses_returned_token_classes() -> None:
    cost = _rate_cost(
        {
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "cache_read_input_tokens": 10_000,
            "cache_write_input_tokens": 1_000,
        },
        {
            "input": "3",
            "output": "15",
            "cache_read_input": "0.3",
            "cache_write_input": "3.75",
        },
    )
    assert cost == Decimal("4.50675")


def test_unattributed_call_uses_reservation_for_conservative_exposure(
    tmp_path: Path,
) -> None:
    arms = tmp_path / "arms"
    arms.mkdir()
    # Research archives may shorten filenames while retaining payload identity.
    (arms / "000000.json").write_text(
        json.dumps(
            {
                "arm_id": "one",
                "status": "failed",
                "delivery_state": "uncertain",
                "reservation_usd": "0.25",
                "model": {
                    "season_model_id": "model-1",
                    "provider": "bedrock",
                    "display_name": "Model 1",
                    "requested_endpoint_id": "endpoint-1",
                },
                "result": None,
            }
        ),
        encoding="utf-8",
    )
    result = reconcile_costs(
        arms_dir=arms,
        rate_card={
            "status": "frozen",
            "rates_per_million_tokens": {
                "endpoint-1": {"input": "1", "output": "2"}
            },
        },
        output_dir=tmp_path / "costs",
    )
    assert result["cost_usd"]["combined_attributed"] == "0.000000000"
    assert result["cost_usd"]["combined_conservative_exposure"] == "0.250000000"


def test_unattributed_rows_are_content_ordered(tmp_path: Path) -> None:
    arms = tmp_path / "arms"
    arms.mkdir()
    for filename, arm_id in (("z.json", "arm-z"), ("a.json", "arm-a")):
        (arms / filename).write_text(
            json.dumps(
                {
                    "arm_id": arm_id,
                    "status": "failed",
                    "delivery_state": "uncertain",
                    "reservation_usd": "0.25",
                    "model": {
                        "season_model_id": "model-1",
                        "provider": "bedrock",
                        "display_name": "Model 1",
                        "requested_endpoint_id": "endpoint-1",
                    },
                    "result": None,
                }
            ),
            encoding="utf-8",
        )

    result = reconcile_costs(
        arms_dir=arms,
        rate_card={
            "status": "frozen",
            "rates_per_million_tokens": {
                "endpoint-1": {"input": "1", "output": "2"}
            },
        },
        output_dir=tmp_path / "costs",
    )

    assert [row["arm_id"] for row in result["unattributed"]] == ["arm-a", "arm-z"]


def test_openrouter_rejections_and_delayed_generation_are_fully_attributed(
    tmp_path: Path,
) -> None:
    arms = tmp_path / "arms"
    corrections = tmp_path / "corrections"
    arms.mkdir()
    corrections.mkdir()
    base_model = {
        "season_model_id": "model-or",
        "provider": "openrouter",
        "display_name": "Model OR",
        "requested_endpoint_id": "requested/model",
        "canonical_model_id": "returned/model",
    }
    rejected_payload = {
        "arm_id": "rejected",
        "status": "failed",
        "delivery_state": "safe_pre_inference",
        "error_type": "SafePreInferenceError",
        "reservation_usd": "0.50",
        "model": base_model,
        "result": None,
    }
    rejected = {**rejected_payload, "artifact_sha256": sha256_json(rejected_payload)}
    (arms / "000000.json").write_text(json.dumps(rejected), encoding="utf-8")
    delayed_payload = {
        "arm_id": "delayed",
        "status": "failed",
        "delivery_state": "uncertain",
        "error_type": "UncertainDeliveryError",
        "reservation_usd": "0.50",
        "model": base_model,
        "result": None,
    }
    delayed = {**delayed_payload, "artifact_sha256": sha256_json(delayed_payload)}
    (arms / "000001.json").write_text(json.dumps(delayed), encoding="utf-8")
    correction_payload = {
        "schema_version": "flavourbench-season0-cost-correction-v1",
        "arm_id": "delayed",
        "source_arm_artifact_sha256": delayed["artifact_sha256"],
        "provider": "openrouter",
        "generation_id": "gen-delayed",
        "accounting": {
            "generation_id": "gen-delayed",
            "total_cost_usd": "0.125",
            "provider_name": "Provider",
            "model": "returned/model",
            "tokens_prompt": 100,
            "tokens_completion": 50,
            "reconciled": True,
        },
    }
    correction = {
        **correction_payload,
        "artifact_sha256": sha256_json(correction_payload),
    }
    (corrections / "000000.json").write_text(
        json.dumps(correction), encoding="utf-8"
    )
    result = reconcile_costs(
        arms_dir=arms,
        rate_card={"status": "frozen", "rates_per_million_tokens": {}},
        output_dir=tmp_path / "costs",
        corrections_dir=corrections,
    )
    assert result["counts"] == {
        "arms": 2,
        "attributed_arms": 2,
        "unattributed_arms": 0,
        "zero_charge_explicit_rejections": 1,
        "recovered_generation_corrections": 1,
    }
    assert result["cost_usd"]["openrouter_generation_metadata_actual"] == "0.125000000"
    assert result["complete_openrouter_request_level_attribution"] is True
    assert result["complete_exposure_accounting"] is True
