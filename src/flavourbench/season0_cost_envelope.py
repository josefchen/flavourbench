"""Freeze scored-run reservations from the final paid Season 0 calibration."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json
from .season0_costs import _latest_arms, _load, _rate_cost

SCHEMA_VERSION = "flavourbench-season0-cost-envelope-v1"


class CostEnvelopeError(RuntimeError):
    """The final calibration cannot safely authorize a scored collection."""


def _artifact_sha(document: Mapping[str, Any], *, label: str) -> str:
    expected = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if expected != actual:
        raise CostEnvelopeError(f"{label} artifact hash mismatch")
    return actual


def freeze_cost_envelope(
    *,
    arms_dir: Path,
    model_manifest: Mapping[str, Any],
    rate_card: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    manifest_sha = _artifact_sha(model_manifest, label="model manifest")
    rates = rate_card.get("rates_per_million_tokens")
    if not isinstance(rates, Mapping):
        raise CostEnvelopeError("rate card has no model rates")
    rows = _latest_arms(arms_dir)
    expected_models = {str(model["season_model_id"]): model for model in model_manifest["models"]}
    if len(rows) != 8 * len(expected_models):
        raise CostEnvelopeError("final calibration must contain exactly eight arms per model")
    envelopes: dict[str, dict[str, Any]] = {}
    provider_forecasts = {"bedrock": Decimal(0), "openrouter": Decimal(0)}
    infrastructure_failures: list[dict[str, str]] = []
    for model_id, model in expected_models.items():
        model_rows = [
            row for row in rows if row.get("model", {}).get("season_model_id") == model_id
        ]
        if len(model_rows) != 8:
            raise CostEnvelopeError(f"{model_id} does not have eight calibration arms")
        costs: list[Decimal] = []
        for row in model_rows:
            contracts = row.get("contracts")
            if (
                not isinstance(contracts, Mapping)
                or contracts.get("model_manifest_artifact_sha256") != manifest_sha
            ):
                raise CostEnvelopeError("calibration arm is bound to another manifest")
            if row.get("delivery_state") in {"uncertain", "safe_pre_inference"}:
                infrastructure_failures.append(
                    {
                        "arm_id": str(row.get("arm_id") or ""),
                        "delivery_state": str(row.get("delivery_state") or ""),
                    }
                )
            result = row.get("result")
            if not isinstance(result, Mapping) or not isinstance(result.get("usage"), Mapping):
                raise CostEnvelopeError("every sent calibration arm must retain usage")
            if model["provider"] == "openrouter":
                if result.get("cost_status") != "openrouter_generation_metadata_reconciled":
                    raise CostEnvelopeError("OpenRouter calibration cost is not reconciled")
                costs.append(Decimal(str(result["actual_cost_usd"])))
            else:
                endpoint_rates = rates.get(model["requested_endpoint_id"])
                if not isinstance(endpoint_rates, Mapping):
                    raise CostEnvelopeError("Bedrock calibration model has no frozen rate")
                costs.append(_rate_cost(result["usage"], endpoint_rates))
        maximum = max(costs)
        mean = sum(costs, Decimal(0)) / Decimal(len(costs))
        reservation = maximum * Decimal(2)
        forecast = mean * Decimal(240)
        provider_forecasts[str(model["provider"])] += forecast
        envelopes[model_id] = {
            "display_name": model["display_name"],
            "provider": model["provider"],
            "calibration_arms": len(costs),
            "mean_arm_cost_usd": format(mean, ".9f"),
            "median_arm_cost_usd": format(Decimal(str(statistics.median(costs))), ".9f"),
            "maximum_arm_cost_usd": format(maximum, ".9f"),
            "scored_arm_reservation_usd": format(reservation, ".9f"),
            "forecast_240_scored_arms_usd": format(forecast, ".9f"),
        }
    if infrastructure_failures:
        raise CostEnvelopeError(
            "final calibration contains infrastructure or uncertain-delivery failures"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "season": "Season 0",
        "status": "frozen_for_scored_admission",
        "synthetic_arms": 0,
        "model_manifest_artifact_sha256": manifest_sha,
        "model_set_sha256": model_manifest["model_set_sha256"],
        "execution_contract_sha256": sha256_json(model_manifest["execution_contract"]),
        "rate_card_sha256": sha256_json(rate_card),
        "calibration_arm_artifact_sha256s": sorted(str(row["artifact_sha256"]) for row in rows),
        "reservation_rule": "two_times_maximum_final_calibration_arm_cost",
        "models": envelopes,
        "forecast_usd": {
            "bedrock": format(provider_forecasts["bedrock"], ".9f"),
            "openrouter": format(provider_forecasts["openrouter"], ".9f"),
            "combined": format(sum(provider_forecasts.values()), ".9f"),
        },
        "hard_caps_usd": {"bedrock": "5000", "openrouter": "100"},
        "admission_stop_fraction": "0.85",
        "forecast_within_admission_caps": (
            provider_forecasts["bedrock"] < Decimal("4250")
            and provider_forecasts["openrouter"] < Decimal("85")
        ),
    }
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"season0-cost-envelope-{digest}.json"
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(destination)
    return {**payload, "summary_path": str(destination)}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--rate-card", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze_cost_envelope(
        arms_dir=args.arms_dir,
        model_manifest=_load(args.model_manifest),
        rate_card=_load(args.rate_card),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
