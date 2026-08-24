"""Compare verified exact-frontier pilot strata without pooling them.

The two strata intentionally use disjoint human-authored tasks.  This module
therefore reports operational completion descriptively and refuses to label the
comparison as either a causal resource effect or a model-quality ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .current_pilot_assets import (
    COHERE_MODEL_ORDER,
    EXTENDED_MODEL_ORDER,
    FAMILY_ORDER,
    MODEL_ORDER,
)
from .frontier_multirun_assets import (
    ASSET_SCHEMA_VERSION as MULTIRUN_SCHEMA_VERSION,
)
from .frontier_multirun_assets import (
    BLUE,
    INK,
    LIGHT,
    ORANGE,
    _configure_matplotlib,
)
from .real_task_bank import sha256_json

SENSITIVITY_SCHEMA_VERSION = "flavourbench-frontier-protocol-sensitivity-v1"


class FrontierProtocolSensitivityError(RuntimeError):
    """A protocol-comparison integrity or claim-boundary check failed."""


@dataclass(frozen=True)
class VerifiedProtocolSensitivity:
    """Content-addressed comparison and its publication tables."""

    aggregate: dict[str, Any]
    model_rows: tuple[dict[str, Any], ...]
    family_rows: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrontierProtocolSensitivityError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise FrontierProtocolSensitivityError(f"expected a JSON object: {path}")
    return value


def _verify_multirun_aggregate(document: Mapping[str, Any], *, label: str) -> None:
    digest = str(document.get("artifact_sha256") or "")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if digest != sha256_json(payload):
        raise FrontierProtocolSensitivityError(f"{label} aggregate hash does not verify")
    totals = document.get("totals")
    tasks = document.get("tasks")
    if (
        document.get("schema_version") != MULTIRUN_SCHEMA_VERSION
        or document.get("official") is not False
        or document.get("rank_eligible") is not False
        or document.get("quality_ranking") is not False
        or document.get("synthetic_tasks") != 0
        or not isinstance(totals, Mapping)
        or totals.get("synthetic_tasks") != 0
        or totals.get("quality_judgments") != 0
        or not isinstance(tasks, list)
        or len(tasks) != totals.get("distinct_tasks")
        or document.get("task_set_sha256") != sha256_json(tasks)
    ):
        raise FrontierProtocolSensitivityError(
            f"{label} aggregate crossed its real-development claim boundary"
        )
    task_ids: set[str] = set()
    for row in tasks:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("task_id"), str)
            or not isinstance(row.get("prompt_sha256"), str)
            or len(str(row["prompt_sha256"])) != 64
            or str(row["task_id"]) in task_ids
        ):
            raise FrontierProtocolSensitivityError(f"{label} task provenance is invalid")
        task_ids.add(str(row["task_id"]))


def _index_rows(
    document: Mapping[str, Any], key: str, identity: str, *, label: str
) -> dict[str, Mapping[str, Any]]:
    rows = document.get(key)
    if not isinstance(rows, list):
        raise FrontierProtocolSensitivityError(f"{label} {key} is missing")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FrontierProtocolSensitivityError(f"{label} {key} row is invalid")
        value = str(row.get(identity) or "")
        if not value or value in indexed:
            raise FrontierProtocolSensitivityError(f"{label} {key} identity is invalid")
        indexed[value] = row
    return indexed


def compare_strata(
    strict: Mapping[str, Any], high_resource: Mapping[str, Any]
) -> VerifiedProtocolSensitivity:
    """Verify and compare two disjoint real-call protocol strata."""

    _verify_multirun_aggregate(strict, label="strict")
    _verify_multirun_aggregate(high_resource, label="high-resource")
    if strict.get("execution_policy_sha256") == high_resource.get("execution_policy_sha256"):
        raise FrontierProtocolSensitivityError("the execution-policy hashes must differ")

    strict_tasks = {str(row["task_id"]) for row in strict["tasks"]}
    resource_tasks = {str(row["task_id"]) for row in high_resource["tasks"]}
    overlap = strict_tasks & resource_tasks
    if overlap:
        raise FrontierProtocolSensitivityError(
            "protocol sensitivity requires disjoint tasks; overlap: " + ", ".join(sorted(overlap))
        )

    strict_models = _index_rows(strict, "model_rows", "model_id", label="strict")
    resource_models = _index_rows(high_resource, "model_rows", "model_id", label="high-resource")
    strict_model_ids = set(strict_models)
    resource_model_ids = set(resource_models)
    if strict_model_ids != set(MODEL_ORDER):
        raise FrontierProtocolSensitivityError("the frozen strict 14-model panel changed")
    if resource_model_ids == set(MODEL_ORDER):
        active_model_order = MODEL_ORDER
    elif resource_model_ids == set(EXTENDED_MODEL_ORDER):
        active_model_order = EXTENDED_MODEL_ORDER
    else:
        raise FrontierProtocolSensitivityError(
            "the high-resource panel must be the frozen core panel or its Cohere extension"
        )

    model_rows: list[dict[str, Any]] = []
    for manifest_ordinal, model_id in enumerate(active_model_order, start=1):
        right = resource_models[model_id]
        left = strict_models.get(model_id)
        if left is not None:
            identity_fields = (
                "display_name",
                "canonical_model_slug",
                "provider_tag",
                "execution_backend",
            )
            if any(left.get(field) != right.get(field) for field in identity_fields):
                raise FrontierProtocolSensitivityError(
                    f"model identity or exact route changed: {model_id}"
                )
        elif model_id not in COHERE_MODEL_ORDER:
            raise FrontierProtocolSensitivityError(
                f"an undeclared endpoint is absent from the strict stratum: {model_id}"
            )
        strict_complete = int(left["complete_pairs"]) if left is not None else 0
        strict_scheduled = int(left["scheduled_pairs"]) if left is not None else 0
        strict_calls = int(left["epicure_calls"]) if left is not None else 0
        strict_successful_calls = (
            int(left["epicure_successful_calls"]) if left is not None else 0
        )
        strict_cost = (
            float(left["conservative_cost_exposure_usd"])
            if left is not None and left["conservative_cost_exposure_usd"] is not None
            else None
        )
        high_resource_cost = (
            float(right["conservative_cost_exposure_usd"])
            if right["conservative_cost_exposure_usd"] is not None
            else None
        )
        total_cost = (
            strict_cost + high_resource_cost
            if strict_cost is not None and high_resource_cost is not None
            else None
        )
        model_rows.append(
            {
                "manifest_ordinal": manifest_ordinal,
                "model_id": model_id,
                "display_name": str(right["display_name"]),
                "canonical_model_slug": str(right["canonical_model_slug"]),
                "provider_tag": str(right["provider_tag"]),
                "strict_observed": left is not None,
                "strict_complete_pairs": strict_complete if left is not None else None,
                "strict_scheduled_pairs": strict_scheduled if left is not None else None,
                "strict_completion_rate": (
                    float(left["pair_completion_rate"]) if left is not None else None
                ),
                "strict_wilson_lower_95": (
                    float(left["pair_completion_wilson_lower_95"])
                    if left is not None
                    else None
                ),
                "strict_wilson_upper_95": (
                    float(left["pair_completion_wilson_upper_95"])
                    if left is not None
                    else None
                ),
                "high_resource_complete_pairs": int(right["complete_pairs"]),
                "high_resource_scheduled_pairs": int(right["scheduled_pairs"]),
                "high_resource_completion_rate": float(right["pair_completion_rate"]),
                "high_resource_wilson_lower_95": float(right["pair_completion_wilson_lower_95"]),
                "high_resource_wilson_upper_95": float(right["pair_completion_wilson_upper_95"]),
                "total_complete_pairs": strict_complete + int(right["complete_pairs"]),
                "total_scheduled_pairs": strict_scheduled + int(right["scheduled_pairs"]),
                "minimum_eight_complete_pairs": strict_complete
                + int(right["complete_pairs"])
                >= 8,
                "strict_epicure_calls": strict_calls,
                "strict_epicure_successful_calls": strict_successful_calls,
                "high_resource_epicure_calls": int(right["epicure_calls"]),
                "high_resource_epicure_successful_calls": int(right["epicure_successful_calls"]),
                "total_epicure_calls": strict_calls + int(right["epicure_calls"]),
                "total_epicure_successful_calls": strict_successful_calls
                + int(right["epicure_successful_calls"]),
                "strict_conservative_cost_exposure_usd": strict_cost,
                "high_resource_conservative_cost_exposure_usd": high_resource_cost,
                "total_conservative_cost_exposure_usd": total_cost,
                "cost_display_status": (
                    "known_usd_exposure"
                    if total_cost is not None
                    else "provider_charge_unavailable"
                ),
                "descriptive_difference_percentage_points": (
                    100
                    * (
                        float(right["pair_completion_rate"])
                        - float(left["pair_completion_rate"])
                    )
                    if left is not None
                    else None
                ),
                "quality_judgments": 0,
                "rank_eligible": False,
            }
        )

    strict_families = _index_rows(strict, "family_rows", "family", label="strict")
    resource_families = _index_rows(high_resource, "family_rows", "family", label="high-resource")
    if set(strict_families) != set(FAMILY_ORDER) or set(resource_families) != set(FAMILY_ORDER):
        raise FrontierProtocolSensitivityError("the task-family panel changed")
    family_rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        left = strict_families[family]
        right = resource_families[family]
        family_rows.append(
            {
                "family": family,
                "display_name": str(left["display_name"]),
                "strict_complete_pairs": int(left["complete_pairs"]),
                "strict_scheduled_pairs": int(left["scheduled_pairs"]),
                "strict_completion_rate": float(left["pair_completion_rate"]),
                "strict_wilson_lower_95": float(left["pair_completion_wilson_lower_95"]),
                "strict_wilson_upper_95": float(left["pair_completion_wilson_upper_95"]),
                "high_resource_complete_pairs": int(right["complete_pairs"]),
                "high_resource_scheduled_pairs": int(right["scheduled_pairs"]),
                "high_resource_completion_rate": float(right["pair_completion_rate"]),
                "high_resource_wilson_lower_95": float(right["pair_completion_wilson_lower_95"]),
                "high_resource_wilson_upper_95": float(right["pair_completion_wilson_upper_95"]),
                "total_complete_pairs": int(left["complete_pairs"])
                + int(right["complete_pairs"]),
                "total_scheduled_pairs": int(left["scheduled_pairs"])
                + int(right["scheduled_pairs"]),
                "descriptive_difference_percentage_points": 100
                * (float(right["pair_completion_rate"]) - float(left["pair_completion_rate"])),
            }
        )

    strict_totals = strict["totals"]
    resource_totals = high_resource["totals"]
    strict_cost = strict["cost"]
    resource_cost = high_resource["cost"]
    combined = {
        "models": len(active_model_order),
        "distinct_tasks": int(strict_totals["distinct_tasks"])
        + int(resource_totals["distinct_tasks"]),
        "scheduled_pairs": int(strict_totals["scheduled_pairs"])
        + int(resource_totals["scheduled_pairs"]),
        "complete_pairs": int(strict_totals["complete_pairs"])
        + int(resource_totals["complete_pairs"]),
        "completed_response_arms": int(strict_totals["completed_response_arms"])
        + int(resource_totals["completed_response_arms"]),
        "provider_generation_ids": int(strict_totals["provider_generation_ids"])
        + int(resource_totals["provider_generation_ids"]),
        "epicure_calls": int(strict_totals["epicure_calls"])
        + int(resource_totals["epicure_calls"]),
        "epicure_successful_calls": int(strict_totals["epicure_successful_calls"])
        + int(resource_totals["epicure_successful_calls"]),
        "known_conservative_exposure_subtotal_usd": float(
            strict_cost["known_conservative_exposure_subtotal_usd"]
        )
        + float(resource_cost["known_conservative_exposure_subtotal_usd"]),
        "provider_charge_complete": bool(strict_cost["provider_charge_complete"])
        and bool(resource_cost["provider_charge_complete"]),
        "unpriced_model_ids": sorted(
            {
                *strict_cost["unpriced_model_ids"],
                *resource_cost["unpriced_model_ids"],
            }
        ),
        "synthetic_tasks": 0,
        "quality_judgments": 0,
    }
    combined["models_with_at_least_eight_complete_pairs"] = sum(
        bool(row["minimum_eight_complete_pairs"]) for row in model_rows
    )
    combined["minimum_complete_pairs_per_model"] = min(
        int(row["total_complete_pairs"]) for row in model_rows
    )
    combined["maximum_complete_pairs_per_model"] = max(
        int(row["total_complete_pairs"]) for row in model_rows
    )
    aggregate: dict[str, Any] = {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "status": "verified_real_protocol_sensitivity",
        "official": False,
        "rank_eligible": False,
        "quality_ranking": False,
        "synthetic_tasks": 0,
        "quality_judgments": 0,
        "strict": {
            "source_artifact_sha256": strict["artifact_sha256"],
            "execution_policy_sha256": strict["execution_policy_sha256"],
            "task_set_sha256": strict["task_set_sha256"],
            "distinct_tasks": int(strict_totals["distinct_tasks"]),
            "scheduled_pairs": int(strict_totals["scheduled_pairs"]),
            "complete_pairs": int(strict_totals["complete_pairs"]),
            "completed_response_arms": int(strict_totals["completed_response_arms"]),
            "provider_generation_ids": int(strict_totals["provider_generation_ids"]),
            "epicure_calls": int(strict_totals["epicure_calls"]),
            "epicure_successful_calls": int(strict_totals["epicure_successful_calls"]),
            "known_conservative_exposure_subtotal_usd": float(
                strict_cost["known_conservative_exposure_subtotal_usd"]
            ),
            "provider_charge_complete": bool(strict_cost["provider_charge_complete"]),
            "unpriced_model_ids": list(strict_cost["unpriced_model_ids"]),
        },
        "high_resource": {
            "source_artifact_sha256": high_resource["artifact_sha256"],
            "execution_policy_sha256": high_resource["execution_policy_sha256"],
            "task_set_sha256": high_resource["task_set_sha256"],
            "distinct_tasks": int(resource_totals["distinct_tasks"]),
            "scheduled_pairs": int(resource_totals["scheduled_pairs"]),
            "complete_pairs": int(resource_totals["complete_pairs"]),
            "completed_response_arms": int(resource_totals["completed_response_arms"]),
            "provider_generation_ids": int(resource_totals["provider_generation_ids"]),
            "epicure_calls": int(resource_totals["epicure_calls"]),
            "epicure_successful_calls": int(resource_totals["epicure_successful_calls"]),
            "known_conservative_exposure_subtotal_usd": float(
                resource_cost["known_conservative_exposure_subtotal_usd"]
            ),
            "provider_charge_complete": bool(resource_cost["provider_charge_complete"]),
            "unpriced_model_ids": list(resource_cost["unpriced_model_ids"]),
        },
        "combined_inventory": combined,
        "task_sets_disjoint": True,
        "model_rows": model_rows,
        "family_rows": family_rows,
        "interpretation": "descriptive_operational_completion_only",
        "limitations": [
            "The task sets are disjoint, so between-stratum differences are not "
            "causal resource effects.",
            "Pair completion is an execution-reliability measure, not a model-quality score.",
            "Wilson intervals are descriptive binomial intervals and do not model task clustering.",
            "No preference or rubric judgments are present in either stratum.",
            "The combined USD cost is a known-exposure subtotal because direct Cohere "
            "provider charges were not returned.",
            (
                "The two Cohere endpoints were prespecified as a high-resource extension and "
                "were not run in the earlier strict stratum."
                if active_model_order == EXTENDED_MODEL_ORDER
                else "The same endpoint panel was used in both protocol strata."
            ),
        ],
    }
    aggregate["artifact_sha256"] = sha256_json(aggregate)
    return VerifiedProtocolSensitivity(
        aggregate=aggregate,
        model_rows=tuple(model_rows),
        family_rows=tuple(family_rows),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _draw_interval_panel(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
) -> None:
    ordered = list(rows)[::-1]
    y = np.arange(len(ordered), dtype=float)
    offset = 0.14
    strict_rate = np.asarray(
        [
            100 * float(row["strict_completion_rate"])
            if row["strict_completion_rate"] is not None
            else np.nan
            for row in ordered
        ]
    )
    strict_lower = np.asarray(
        [
            100 * float(row["strict_wilson_lower_95"])
            if row["strict_wilson_lower_95"] is not None
            else np.nan
            for row in ordered
        ]
    )
    strict_upper = np.asarray(
        [
            100 * float(row["strict_wilson_upper_95"])
            if row["strict_wilson_upper_95"] is not None
            else np.nan
            for row in ordered
        ]
    )
    resource_rate = np.asarray(
        [100 * float(row["high_resource_completion_rate"]) for row in ordered]
    )
    resource_lower = np.asarray(
        [100 * float(row["high_resource_wilson_lower_95"]) for row in ordered]
    )
    resource_upper = np.asarray(
        [100 * float(row["high_resource_wilson_upper_95"]) for row in ordered]
    )
    ax.errorbar(
        strict_rate,
        y - offset,
        xerr=np.vstack(
            (
                np.maximum(0.0, strict_rate - strict_lower),
                np.maximum(0.0, strict_upper - strict_rate),
            )
        ),
        fmt="o",
        color=BLUE,
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=4.5,
        elinewidth=0.8,
        capsize=2.2,
        label="Strict protocol",
        zorder=3,
    )
    ax.errorbar(
        resource_rate,
        y + offset,
        xerr=np.vstack(
            (
                np.maximum(0.0, resource_rate - resource_lower),
                np.maximum(0.0, resource_upper - resource_rate),
            )
        ),
        fmt="D",
        color=ORANGE,
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=0.9,
        markersize=4.1,
        elinewidth=0.8,
        capsize=2.2,
        label="High-resource protocol",
        zorder=3,
    )
    ax.set_yticks(y, [str(row["display_name"]) for row in ordered])
    ax.set_xlim(0, 145)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Complete matched pairs (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axvline(100, color=INK, linewidth=0.55)
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(105, len(ordered) - 0.15, "strict", color=BLUE, fontsize=6.6, ha="center")
    ax.text(
        119,
        len(ordered) - 0.15,
        "high",
        color=ORANGE,
        fontsize=6.6,
        ha="center",
    )
    ax.text(136, len(ordered) - 0.15, "total", color=INK, fontsize=6.6, ha="center")
    for index, row in enumerate(ordered):
        strict_count = (
            f"{row['strict_complete_pairs']}/{row['strict_scheduled_pairs']}"
            if row["strict_scheduled_pairs"] is not None
            else "--"
        )
        ax.text(
            105,
            index,
            strict_count,
            color=BLUE,
            fontsize=6.5,
            ha="center",
            va="center",
        )
        ax.text(
            119,
            index,
            f"{row['high_resource_complete_pairs']}/{row['high_resource_scheduled_pairs']}",
            color=ORANGE,
            fontsize=6.5,
            ha="center",
            va="center",
        )
        ax.text(
            136,
            index,
            f"{row['total_complete_pairs']}/{row['total_scheduled_pairs']}",
            color=INK,
            fontsize=6.5,
            ha="center",
            va="center",
        )


def _render_figure(comparison: VerifiedProtocolSensitivity, output: Path) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    model_count = int(comparison.aggregate["combined_inventory"]["models"])
    has_high_resource_extension = any(
        not bool(row.get("strict_observed", True)) for row in comparison.model_rows
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.15, 7.65 if model_count > 14 else 7.35),
        gridspec_kw={"height_ratios": (3.3, 1.18), "hspace": 0.42},
    )
    _draw_interval_panel(
        axes[0], comparison.model_rows, title="a  Model-level operational completion"
    )
    _draw_interval_panel(
        axes[1], comparison.family_rows, title="b  Task-family operational completion"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.977),
    )
    figure.suptitle(
        "Protocol sensitivity in the real exact-frontier pilot",
        x=0.075,
        y=0.986,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.013,
        (
            "Points show real matched-pair completion; whiskers are 95% Wilson intervals. "
            "The protocol strata use disjoint human-authored tasks.\n"
            "Differences are descriptive and are neither causal resource effects nor "
            f"quality scores. All {model_count} exact endpoints have at least eight complete "
            "pairs in total."
            + (
                " Cohere endpoints were not run in the strict stratum."
                if has_high_resource_extension
                else ""
            )
        ),
        fontsize=6.7,
        color="#59636E",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=300)
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def write_assets(comparison: VerifiedProtocolSensitivity, output_dir: Path) -> dict[str, Path]:
    """Write the verified comparison, source table, macros, and vector figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    digest = str(comparison.aggregate["artifact_sha256"])
    aggregate = output_dir / f"frontier-protocol-sensitivity-{digest}.json"
    aggregate.write_text(
        json.dumps(comparison.aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = output_dir / "frontier-protocol-sensitivity-provenance.json"
    provenance.write_text(
        json.dumps(comparison.aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_csv = output_dir / "frontier-protocol-sensitivity-models.csv"
    family_csv = output_dir / "frontier-protocol-sensitivity-families.csv"
    _write_csv(model_csv, comparison.model_rows)
    _write_csv(family_csv, comparison.family_rows)
    strict = comparison.aggregate["strict"]
    high = comparison.aggregate["high_resource"]
    inventory = comparison.aggregate["combined_inventory"]
    strict_rate = 100 * strict["complete_pairs"] / strict["scheduled_pairs"]
    high_rate = 100 * high["complete_pairs"] / high["scheduled_pairs"]
    macros = output_dir / "frontier-protocol-sensitivity-macros.tex"
    macros.write_text(
        "\n".join(
            [
                rf"\newcommand{{\FrontierExactModelCount}}{{{inventory['models']}}}",
                rf"\newcommand{{\FrontierProtocolTaskCount}}{{{inventory['distinct_tasks']}}}",
                rf"\newcommand{{\FrontierCombinedScheduledPairs}}{{{inventory['scheduled_pairs']}}}",
                rf"\newcommand{{\FrontierCombinedCompletePairs}}{{{inventory['complete_pairs']}}}",
                rf"\newcommand{{\FrontierCompletedResponseArms}}{{{inventory['completed_response_arms']}}}",
                rf"\newcommand{{\FrontierProviderGenerations}}{{{inventory['provider_generation_ids']}}}",
                rf"\newcommand{{\FrontierEpicureCalls}}{{{inventory['epicure_calls']}}}",
                rf"\newcommand{{\FrontierEpicureSuccessfulCalls}}{{{inventory['epicure_successful_calls']}}}",
                rf"\newcommand{{\FrontierQualityJudgments}}{{{inventory['quality_judgments']}}}",
                rf"\newcommand{{\FrontierCombinedExposure}}{{\${inventory['known_conservative_exposure_subtotal_usd']:.3f}}}",
                rf"\newcommand{{\FrontierCombinedUnpricedModelCount}}{{{len(inventory['unpriced_model_ids'])}}}",
                rf"\newcommand{{\FrontierModelsAtLeastEightPairs}}{{{inventory['models_with_at_least_eight_complete_pairs']}}}",
                rf"\newcommand{{\FrontierMinimumCompletePairs}}{{{inventory['minimum_complete_pairs_per_model']}}}",
                rf"\newcommand{{\FrontierMaximumCompletePairs}}{{{inventory['maximum_complete_pairs_per_model']}}}",
                rf"\newcommand{{\FrontierStrictTaskCount}}{{{strict['distinct_tasks']}}}",
                rf"\newcommand{{\FrontierStrictScheduledPairs}}{{{strict['scheduled_pairs']}}}",
                rf"\newcommand{{\FrontierStrictCompletePairs}}{{{strict['complete_pairs']}}}",
                rf"\newcommand{{\FrontierStrictPairRate}}{{{strict_rate:.1f}\%}}",
                rf"\newcommand{{\FrontierStrictResponseArms}}{{{strict['completed_response_arms']}}}",
                rf"\newcommand{{\FrontierStrictProviderGenerations}}{{{strict['provider_generation_ids']}}}",
                rf"\newcommand{{\FrontierStrictEpicureCalls}}{{{strict['epicure_calls']}}}",
                rf"\newcommand{{\FrontierStrictEpicureSuccessfulCalls}}{{{strict['epicure_successful_calls']}}}",
                rf"\newcommand{{\FrontierStrictExposure}}{{\${strict['known_conservative_exposure_subtotal_usd']:.3f}}}",
                rf"\newcommand{{\FrontierHighResourceTaskCount}}{{{high['distinct_tasks']}}}",
                rf"\newcommand{{\FrontierHighResourceScheduledPairs}}{{{high['scheduled_pairs']}}}",
                rf"\newcommand{{\FrontierHighResourceCompletePairs}}{{{high['complete_pairs']}}}",
                rf"\newcommand{{\FrontierHighResourcePairRate}}{{{high_rate:.1f}\%}}",
                rf"\newcommand{{\FrontierHighResourceResponseArms}}{{{high['completed_response_arms']}}}",
                rf"\newcommand{{\FrontierHighResourceProviderGenerations}}{{{high['provider_generation_ids']}}}",
                rf"\newcommand{{\FrontierHighResourceEpicureCalls}}{{{high['epicure_calls']}}}",
                rf"\newcommand{{\FrontierHighResourceEpicureSuccessfulCalls}}{{{high['epicure_successful_calls']}}}",
                rf"\newcommand{{\FrontierHighResourceExposure}}{{\${high['known_conservative_exposure_subtotal_usd']:.3f}}}",
                rf"\newcommand{{\FrontierHighResourceUnpricedModelCount}}{{{len(high['unpriced_model_ids'])}}}",
                rf"\newcommand{{\FrontierStrictAssetHash}}{{{strict['source_artifact_sha256']}}}",
                rf"\newcommand{{\FrontierHighResourceAssetHash}}{{{high['source_artifact_sha256']}}}",
                rf"\newcommand{{\FrontierProtocolSensitivityHash}}{{{digest}}}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    caption = output_dir / "frontier-protocol-sensitivity-caption.tex"
    caption.write_text(
        (
            "Operational completion under two frozen execution protocols. Points report "
            "the fraction of scheduled matched Epicure-off/on pairs for which both "
            "normalized responses were recovered and the treatment arm contained at "
            "least one successful Epicure call; whiskers are 95\\% Wilson intervals. "
            f"The strict stratum completed {strict['complete_pairs']} of "
            f"{strict['scheduled_pairs']} pairs and the high-resource stratum completed "
            f"{high['complete_pairs']} of {high['scheduled_pairs']}. The strata use "
            "disjoint human-authored tasks, so differences are descriptive and should "
            "not be read as causal resource effects or model-quality scores.\n"
        ),
        encoding="utf-8",
    )
    table = output_dir / "frontier-protocol-sensitivity-table.tex"
    table_lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Endpoint & Strict & High & Combined & Epicure ok/all & Exposure (\$) \\",
        r"\midrule",
    ]
    for row in comparison.model_rows:
        strict_count = (
            f"{row['strict_complete_pairs']}/{row['strict_scheduled_pairs']}"
            if row["strict_scheduled_pairs"] is not None
            else r"\textemdash"
        )
        exposure = (
            f"{row['total_conservative_cost_exposure_usd']:.3f}"
            if row["total_conservative_cost_exposure_usd"] is not None
            else r"not returned"
        )
        table_lines.append(
            f"{row['display_name']} & "
            f"{strict_count} & "
            f"{row['high_resource_complete_pairs']}/"
            f"{row['high_resource_scheduled_pairs']} & "
            f"{row['total_complete_pairs']}/{row['total_scheduled_pairs']} & "
            f"{row['total_epicure_successful_calls']}/{row['total_epicure_calls']} & "
            f"{exposure} \\\\"
        )
    table_lines.extend(
        [
            r"\midrule",
            "Total & "
            f"{strict['complete_pairs']}/{strict['scheduled_pairs']} & "
            f"{high['complete_pairs']}/{high['scheduled_pairs']} & "
            f"{inventory['complete_pairs']}/{inventory['scheduled_pairs']} & "
            f"{inventory['epicure_successful_calls']}/{inventory['epicure_calls']} & "
            f"{inventory['known_conservative_exposure_subtotal_usd']:.3f} "
            f"+ {len(inventory['unpriced_model_ids'])} unpriced \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    table.write_text("\n".join(table_lines), encoding="utf-8")
    figure = output_dir / "frontier-protocol-sensitivity.pdf"
    _render_figure(comparison, figure)
    return {
        "aggregate": aggregate,
        "provenance": provenance,
        "model_csv": model_csv,
        "family_csv": family_csv,
        "macros": macros,
        "caption": caption,
        "table": table,
        "figure": figure,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--high-resource", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    comparison = compare_strata(_read_json(arguments.strict), _read_json(arguments.high_resource))
    paths = write_assets(comparison, arguments.output_dir)
    print(
        json.dumps(
            {
                "status": "verified",
                "artifact_sha256": comparison.aggregate["artifact_sha256"],
                "task_sets_disjoint": comparison.aggregate["task_sets_disjoint"],
                "outputs": {key: str(value.resolve()) for key, value in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
