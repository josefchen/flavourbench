"""Reconcile real Season 0 arm usage against frozen provider cost evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json
from .season0_arm_corrections import validate_arm_interpretation_correction

SCHEMA_VERSION = "flavourbench-season0-cost-audit-v1"
MILLION = Decimal("1000000")


class CostAuditError(RuntimeError):
    """Cost evidence is missing, internally inconsistent, or not attributable."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise CostAuditError(f"expected a JSON object: {path}")
    return value


def _latest_arms(directory: Path) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in directory.glob("*.json"):
        value = _load(path)
        arm_id = value.get("arm_id")
        if not isinstance(arm_id, str):
            continue
        prior = by_id.get(arm_id)
        if prior is None or str(value.get("completed_at") or "") > str(
            prior.get("completed_at") or ""
        ):
            by_id[arm_id] = value
    return [by_id[arm_id] for arm_id in sorted(by_id)]


def _cost_corrections(directory: Path | None) -> dict[str, dict[str, Any]]:
    if directory is None:
        return {}
    by_arm: dict[str, dict[str, Any]] = {}
    generation_ids: set[str] = set()
    for path in directory.glob("*.json"):
        value = _load(path)
        claimed = value.get("artifact_sha256")
        actual = sha256_json(
            {key: item for key, item in value.items() if key != "artifact_sha256"}
        )
        if claimed != actual:
            raise CostAuditError(f"cost correction artifact hash mismatch: {path}")
        if value.get("schema_version") != "flavourbench-season0-cost-correction-v1":
            raise CostAuditError(f"unsupported cost correction schema: {path}")
        arm_id = value.get("arm_id")
        generation_id = value.get("generation_id")
        if not isinstance(arm_id, str) or not isinstance(generation_id, str):
            raise CostAuditError(f"cost correction lacks an arm or generation ID: {path}")
        if arm_id in by_arm or generation_id in generation_ids:
            raise CostAuditError("cost corrections repeat an arm or generation ID")
        by_arm[arm_id] = value
        generation_ids.add(generation_id)
    return by_arm


def _rate_cost(usage: Mapping[str, Any], rates: Mapping[str, Any]) -> Decimal:
    input_tokens = Decimal(int(usage.get("input_tokens") or 0))
    output_tokens = Decimal(int(usage.get("output_tokens") or 0))
    cache_read = Decimal(int(usage.get("cache_read_input_tokens") or 0))
    cache_write = Decimal(int(usage.get("cache_write_input_tokens") or 0))
    if cache_read and rates.get("cache_read_input") is None:
        raise CostAuditError("cache-read usage has no frozen rate")
    if cache_write and rates.get("cache_write_input") is None:
        raise CostAuditError("cache-write usage has no frozen rate")
    return (
        input_tokens * Decimal(str(rates["input"]))
        + output_tokens * Decimal(str(rates["output"]))
        + cache_read * Decimal(str(rates.get("cache_read_input") or 0))
        + cache_write * Decimal(str(rates.get("cache_write_input") or 0))
    ) / MILLION


def reconcile_costs(
    *,
    arms_dir: Path,
    rate_card: Mapping[str, Any],
    output_dir: Path,
    corrections_dir: Path | None = None,
    arm_interpretation_correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    interpretation = validate_arm_interpretation_correction(
        correction=arm_interpretation_correction,
        arms_dir=arms_dir,
    )
    rates = rate_card.get("rates_per_million_tokens")
    if not isinstance(rates, Mapping):
        raise CostAuditError("rate card contains no model rates")
    rows = _latest_arms(arms_dir)
    corrections = _cost_corrections(corrections_dir)
    correction_artifact_sha256s: list[str] = []
    recovered_generation_corrections = 0
    zero_charge_explicit_rejections = 0
    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "provider": "",
            "display_name": "",
            "arms": 0,
            "attributed_arms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": Decimal(0),
        }
    )
    unattributed: list[dict[str, str]] = []

    def record_unattributed(
        row: Mapping[str, Any], provider: str, reason: str
    ) -> None:
        reservation = Decimal(str(row.get("reservation_usd") or "0"))
        unattributed.append(
            {
                "arm_id": str(row.get("arm_id") or ""),
                "provider": provider,
                "reason": reason,
                "conservative_reservation_usd": format(reservation, ".9f"),
            }
        )
    for row in rows:
        model = row.get("model")
        if not isinstance(model, Mapping):
            continue
        model_id = str(model.get("season_model_id") or "")
        provider = str(model.get("provider") or "")
        bucket = by_model[model_id]
        bucket["provider"] = provider
        bucket["display_name"] = str(model.get("display_name") or "")
        bucket["arms"] += 1
        result = row.get("result")
        if not isinstance(result, Mapping):
            correction = corrections.pop(str(row.get("arm_id") or ""), None)
            if correction is not None:
                accounting = correction.get("accounting")
                if (
                    provider != "openrouter"
                    or row.get("status") != "failed"
                    or correction.get("provider") != "openrouter"
                    or correction.get("source_arm_artifact_sha256")
                    != row.get("artifact_sha256")
                    or not isinstance(accounting, Mapping)
                    or accounting.get("reconciled") is not True
                    or accounting.get("generation_id")
                    != correction.get("generation_id")
                    or accounting.get("model") != model.get("canonical_model_id")
                    or not accounting.get("provider_name")
                ):
                    raise CostAuditError("OpenRouter cost correction binding mismatch")
                cost = Decimal(str(accounting.get("total_cost_usd") or ""))
                if not cost.is_finite() or cost < 0:
                    raise CostAuditError("OpenRouter cost correction is not a valid amount")
                bucket["attributed_arms"] += 1
                bucket["input_tokens"] += int(accounting.get("tokens_prompt") or 0)
                bucket["output_tokens"] += int(
                    accounting.get("tokens_completion") or 0
                )
                bucket["cost_usd"] += cost
                correction_artifact_sha256s.append(str(correction["artifact_sha256"]))
                recovered_generation_corrections += 1
                continue
            if (
                provider == "openrouter"
                and row.get("status") == "failed"
                and row.get("delivery_state") == "safe_pre_inference"
                and row.get("error_type") == "SafePreInferenceError"
            ):
                bucket["attributed_arms"] += 1
                zero_charge_explicit_rejections += 1
                continue
            if row.get("status") != "not_admitted":
                record_unattributed(row, provider, "missing_result")
            continue
        usage = result.get("usage")
        if not isinstance(usage, Mapping):
            record_unattributed(row, provider, "missing_usage")
            continue
        if provider == "openrouter":
            actual = result.get("actual_cost_usd")
            if actual is None or result.get("cost_status") != (
                "openrouter_generation_metadata_reconciled"
            ):
                record_unattributed(
                    row, provider, "openrouter_generation_cost_unreconciled"
                )
                continue
            cost = Decimal(str(actual))
        elif provider == "bedrock":
            endpoint_id = str(model.get("requested_endpoint_id") or "")
            model_rates = rates.get(endpoint_id)
            if not isinstance(model_rates, Mapping):
                record_unattributed(row, provider, f"missing_rate:{endpoint_id}")
                continue
            cost = _rate_cost(usage, model_rates)
        else:
            record_unattributed(row, provider, "unknown_provider")
            continue
        bucket["attributed_arms"] += 1
        bucket["input_tokens"] += int(usage.get("input_tokens") or 0)
        bucket["output_tokens"] += int(usage.get("output_tokens") or 0)
        bucket["cost_usd"] += cost

    if corrections:
        raise CostAuditError("cost corrections reference no current unattributed arm")

    unattributed.sort(
        key=lambda row: (
            row["arm_id"],
            row["provider"],
            row["reason"],
            row["conservative_reservation_usd"],
        )
    )

    models = {
        model_id: {
            **{key: value for key, value in values.items() if key != "cost_usd"},
            "cost_usd": format(values["cost_usd"], ".9f"),
        }
        for model_id, values in sorted(by_model.items())
    }
    bedrock_total = sum(
        (values["cost_usd"] for values in by_model.values() if values["provider"] == "bedrock"),
        Decimal(0),
    )
    openrouter_total = sum(
        (
            values["cost_usd"]
            for values in by_model.values()
            if values["provider"] == "openrouter"
        ),
        Decimal(0),
    )
    unattributed_reservations = {
        provider: sum(
            (
                Decimal(row["conservative_reservation_usd"])
                for row in unattributed
                if row["provider"] == provider
            ),
            Decimal(0),
        )
        for provider in ("bedrock", "openrouter")
    }
    unattributed_reservation_total = sum(
        unattributed_reservations.values(), Decimal(0)
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "synthetic_arms": 0,
        "arm_interpretation_correction_artifact_sha256": (
            interpretation.artifact_sha256 if interpretation is not None else None
        ),
        "arm_interpretation_correction_count": (
            len(interpretation.arm_ids) if interpretation is not None else 0
        ),
        "rate_card_sha256": sha256_json(rate_card),
        "rate_card_status": rate_card.get("status"),
        "counts": {
            "arms": len(rows),
            "attributed_arms": sum(value["attributed_arms"] for value in by_model.values()),
            "unattributed_arms": len(unattributed),
            "zero_charge_explicit_rejections": zero_charge_explicit_rejections,
            "recovered_generation_corrections": recovered_generation_corrections,
        },
        "cost_usd": {
            "bedrock_published_rate_estimate": format(bedrock_total, ".9f"),
            "openrouter_generation_metadata_actual": format(openrouter_total, ".9f"),
            "combined_attributed": format(bedrock_total + openrouter_total, ".9f"),
            "unattributed_conservative_reservations": format(
                unattributed_reservation_total, ".9f"
            ),
            "combined_conservative_exposure": format(
                bedrock_total + openrouter_total + unattributed_reservation_total,
                ".9f",
            ),
            "unattributed_reservations_by_provider": {
                provider: format(value, ".9f")
                for provider, value in unattributed_reservations.items()
            },
        },
        "invoice_reconciliation": {
            "bedrock": "pending_aws_cur_2_aggregate_crosscheck",
            "openrouter": "complete_per_generation_or_explicit_pre_inference_rejection",
        },
        "models": models,
        "unattributed": unattributed,
        "cost_correction_artifact_sha256s": sorted(correction_artifact_sha256s),
        "complete_openrouter_request_level_attribution": not any(
            row["provider"] == "openrouter" for row in unattributed
        ),
        "complete_exposure_accounting": all(
            Decimal(row["conservative_reservation_usd"]) > 0 for row in unattributed
        ),
        "complete_request_level_attribution": not unattributed,
    }
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"cost-audit-{digest}.json"
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
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
    parser.add_argument("--rate-card", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corrections-dir", type=Path)
    parser.add_argument("--arm-interpretation-correction", type=Path)
    args = parser.parse_args(argv)
    result = reconcile_costs(
        arms_dir=args.arms_dir,
        rate_card=_load(args.rate_card),
        output_dir=args.output_dir,
        corrections_dir=args.corrections_dir,
        arm_interpretation_correction=(
            _load(args.arm_interpretation_correction)
            if args.arm_interpretation_correction is not None
            else None
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
