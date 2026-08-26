#!/usr/bin/env python3
"""Build paper and dataset assets for the preregistered reward-transfer study."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from flavourbench.reward_transfer import verify_content_addressed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = REPOSITORY / "hf/dataset/data-analysis/reward-transfer-primary-analysis.json"
DEFAULT_PUBLIC = REPOSITORY / "hf/dataset/data-analysis/reward-transfer-public-analysis.json"
DEFAULT_GENERATED = REPOSITORY / "paper/generated/complete-core"
DEFAULT_FIGURES = REPOSITORY / "paper/figures/complete-core"
DEFAULT_DATASET_ASSETS = REPOSITORY / "hf/dataset/assets"
CONDITIONS = ("pretrained_base", "sft_format_control", "sft_epicure_optimum")


class RewardTransferAssetError(RuntimeError):
    """A released analysis or rendered paper asset is incomplete."""


def _load(path: Path, *, split: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RewardTransferAssetError(f"analysis is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    verify_content_addressed(value, label=f"{split} reward-transfer analysis")
    expected_status = (
        "primary_analysis_complete_before_public_replication"
        if split == "primary"
        else "public_replication_analysis_complete"
    )
    if (
        value.get("schema_version") != "flavourbench-reward-transfer-analysis-v1"
        or value.get("status") != expected_status
        or value.get("split") != split
    ):
        raise RewardTransferAssetError(f"{split} analysis contract differs")
    return value


def _condition(analysis: Mapping[str, Any], condition: str) -> Mapping[str, Any]:
    rows = analysis.get("condition_summaries")
    if not isinstance(rows, list):
        raise RewardTransferAssetError("condition summaries are absent")
    matches = [
        row for row in rows if isinstance(row, Mapping) and row.get("condition") == condition
    ]
    if len(matches) != 1:
        raise RewardTransferAssetError(f"condition summary differs: {condition}")
    return matches[0]


def _contrast(analysis: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    key = "confirmatory_contrast" if split == "primary" else "replication_contrast"
    value = analysis.get(key)
    if not isinstance(value, Mapping):
        raise RewardTransferAssetError(f"{split} contrast is absent")
    return value


def _secondary(analysis: Mapping[str, Any], suffix: str) -> Mapping[str, Any]:
    rows = analysis.get("secondary_contrasts")
    if not isinstance(rows, list):
        raise RewardTransferAssetError("secondary contrasts are absent")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("label", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RewardTransferAssetError(f"secondary contrast differs: {suffix}")
    return matches[0]


def _family(summary: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    rows = summary.get("per_family")
    if not isinstance(rows, list):
        raise RewardTransferAssetError("family summaries are absent")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("family") == family]
    if len(matches) != 1:
        raise RewardTransferAssetError(f"family summary differs: {family}")
    return matches[0]


def _tex_p_value(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}"
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / (10**exponent)
    if coefficient >= 9.995:
        coefficient /= 10
        exponent += 1
    return rf"{coefficient:.2f}\!\times\!10^{{{exponent}}}"


def _tex_p(value: float) -> str:
    rendered = _tex_p_value(value)
    return f"${rendered}$" if value < 0.001 else rendered


def _plain_p(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}"
    return f"{value:.1e}".replace("e-0", "e-")


def _macro(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _render_macros(primary: Mapping[str, Any], public: Mapping[str, Any]) -> bytes:
    primary_contrast = _contrast(primary, "primary")
    public_contrast = _contrast(public, "public")
    primary_treatment = _condition(primary, "sft_epicure_optimum")
    primary_control = _condition(primary, "sft_format_control")
    public_treatment = _condition(public, "sft_epicure_optimum")
    public_control = _condition(public, "sft_format_control")
    primary_base = _condition(primary, "pretrained_base")
    public_base = _condition(public, "pretrained_base")
    primary_control_base = _secondary(primary, "sft_format_control_minus_pretrained_base")
    public_control_base = _secondary(public, "sft_format_control_minus_pretrained_base")
    primary_interval = primary_contrast["confidence_interval_95"]
    public_interval = public_contrast["confidence_interval_95"]
    primary_control_base_interval = primary_control_base["confidence_interval_95"]
    public_control_base_interval = public_control_base["confidence_interval_95"]
    lines = [
        "% Auto-generated by build_reward_transfer_assets.py.",
        _macro("FBTransferPrimaryTasks", int(primary["tasks"])),
        _macro("FBTransferPublicTasks", int(public["tasks"])),
        _macro("FBTransferPrimaryChance", f"{float(primary['chance_baseline_score']):.2f}"),
        _macro("FBTransferPublicChance", f"{float(public['chance_baseline_score']):.2f}"),
        _macro("FBTransferPrimaryBase", f"{float(primary_base['score_unconditional_mean']):.2f}"),
        _macro(
            "FBTransferPrimaryControl",
            f"{float(primary_control['score_unconditional_mean']):.2f}",
        ),
        _macro(
            "FBTransferPrimaryTreatment",
            f"{float(primary_treatment['score_unconditional_mean']):.2f}",
        ),
        _macro("FBTransferPrimaryBaseParse", f"{100 * float(primary_base['parse_rate_mean']):.2f}"),
        _macro(
            "FBTransferPrimaryControlParse",
            f"{100 * float(primary_control['parse_rate_mean']):.2f}",
        ),
        _macro(
            "FBTransferPrimaryTreatmentParse",
            f"{100 * float(primary_treatment['parse_rate_mean']):.2f}",
        ),
        _macro("FBTransferPrimaryGain", f"{float(primary_contrast['estimate_points']):.2f}"),
        _macro("FBTransferPrimaryCILow", f"{float(primary_interval[0]):.2f}"),
        _macro("FBTransferPrimaryCIHigh", f"{float(primary_interval[1]):.2f}"),
        _macro(
            "FBTransferPrimaryP",
            _tex_p_value(float(primary_contrast["two_sided_sign_flip_p"])),
        ),
        _macro(
            "FBTransferPrimaryControlBaseGain",
            f"{float(primary_control_base['estimate_points']):.2f}",
        ),
        _macro(
            "FBTransferPrimaryControlBaseCILow",
            f"{float(primary_control_base_interval[0]):.2f}",
        ),
        _macro(
            "FBTransferPrimaryControlBaseCIHigh",
            f"{float(primary_control_base_interval[1]):.2f}",
        ),
        _macro(
            "FBTransferPrimaryControlBaseP",
            _tex_p_value(float(primary_control_base["two_sided_sign_flip_p"])),
        ),
        _macro("FBTransferPublicBase", f"{float(public_base['score_unconditional_mean']):.2f}"),
        _macro(
            "FBTransferPublicControl",
            f"{float(public_control['score_unconditional_mean']):.2f}",
        ),
        _macro(
            "FBTransferPublicTreatment",
            f"{float(public_treatment['score_unconditional_mean']):.2f}",
        ),
        _macro("FBTransferPublicBaseParse", f"{100 * float(public_base['parse_rate_mean']):.2f}"),
        _macro(
            "FBTransferPublicControlParse",
            f"{100 * float(public_control['parse_rate_mean']):.2f}",
        ),
        _macro(
            "FBTransferPublicTreatmentParse",
            f"{100 * float(public_treatment['parse_rate_mean']):.2f}",
        ),
        _macro("FBTransferPublicGain", f"{float(public_contrast['estimate_points']):.2f}"),
        _macro("FBTransferPublicCILow", f"{float(public_interval[0]):.2f}"),
        _macro("FBTransferPublicCIHigh", f"{float(public_interval[1]):.2f}"),
        _macro(
            "FBTransferPublicP",
            _tex_p_value(float(public_contrast["two_sided_sign_flip_p"])),
        ),
        _macro(
            "FBTransferPublicControlBaseGain",
            f"{float(public_control_base['estimate_points']):.2f}",
        ),
        _macro(
            "FBTransferPublicControlBaseCILow",
            f"{float(public_control_base_interval[0]):.2f}",
        ),
        _macro(
            "FBTransferPublicControlBaseCIHigh",
            f"{float(public_control_base_interval[1]):.2f}",
        ),
        _macro(
            "FBTransferPublicControlBaseP",
            _tex_p_value(float(public_control_base["two_sided_sign_flip_p"])),
        ),
        "",
    ]
    return "\n".join(lines).encode()


def _render_table(primary: Mapping[str, Any], public: Mapping[str, Any]) -> bytes:
    lines = [
        "% Auto-generated by build_reward_transfer_assets.py.",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Evaluation & Base & Format control & Epicure SFT & $\Delta$ [95\% CI] & $p$ \\",
        r"\midrule",
    ]
    for label, split, analysis in (
        ("Anchor-disjoint transfer", "primary", primary),
        ("Public-map replication", "public", public),
    ):
        base = _condition(analysis, "pretrained_base")
        control = _condition(analysis, "sft_format_control")
        treatment = _condition(analysis, "sft_epicure_optimum")
        contrast = _contrast(analysis, split)
        interval = contrast["confidence_interval_95"]
        lines.append(
            f"{label} ($n={int(analysis['tasks'])}$) & "
            f"{float(base['score_unconditional_mean']):.2f} & "
            f"{float(control['score_unconditional_mean']):.2f} & "
            f"{float(treatment['score_unconditional_mean']):.2f} & "
            f"+{float(contrast['estimate_points']):.2f} "
            f"[{float(interval[0]):.2f}, {float(interval[1]):.2f}] & "
            f"{_tex_p(float(contrast['two_sided_sign_flip_p']))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines).encode()


def _render_family_table(primary: Mapping[str, Any], public: Mapping[str, Any]) -> bytes:
    lines = [
        "% Auto-generated by build_reward_transfer_assets.py.",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Evaluation & Family & Base & Format control & Epicure SFT & $\Delta$ \\",
        r"\midrule",
    ]
    families = (
        ("substitution", "Substitution"),
        ("pairing", "Pairing"),
        ("constraint", "Constraint"),
    )
    for split_index, (split_label, analysis) in enumerate(
        (("Transfer", primary), ("Public", public))
    ):
        base = _condition(analysis, "pretrained_base")
        control = _condition(analysis, "sft_format_control")
        treatment = _condition(analysis, "sft_epicure_optimum")
        for family_key, family_label in families:
            base_score = float(_family(base, family_key)["score_unconditional_mean"])
            control_score = float(_family(control, family_key)["score_unconditional_mean"])
            treatment_score = float(_family(treatment, family_key)["score_unconditional_mean"])
            lines.append(
                f"{split_label} & {family_label} & {base_score:.2f} & {control_score:.2f} & "
                f"{treatment_score:.2f} & {treatment_score - control_score:+.2f} \\\\"
            )
        if split_index == 0:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines).encode()


def _runs_by_seed(summary: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = summary.get("runs")
    if not isinstance(rows, list):
        raise RewardTransferAssetError("condition runs are absent")
    result = {
        int(row["training_seed"]): row
        for row in rows
        if isinstance(row, Mapping) and row.get("training_seed") is not None
    }
    if len(result) != 3:
        raise RewardTransferAssetError("trained condition does not contain three unique seeds")
    return result


def _render_seed_table(primary: Mapping[str, Any], public: Mapping[str, Any]) -> bytes:
    primary_control = _runs_by_seed(_condition(primary, "sft_format_control"))
    primary_treatment = _runs_by_seed(_condition(primary, "sft_epicure_optimum"))
    public_control = _runs_by_seed(_condition(public, "sft_format_control"))
    public_treatment = _runs_by_seed(_condition(public, "sft_epicure_optimum"))
    seeds = sorted(primary_control)
    if not (set(seeds) == set(primary_treatment) == set(public_control) == set(public_treatment)):
        raise RewardTransferAssetError("training seeds differ across conditions or evaluations")
    lines = [
        "% Auto-generated by build_reward_transfer_assets.py.",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        (
            r"& \multicolumn{3}{c}{Anchor-disjoint transfer} "
            r"& \multicolumn{3}{c}{Public-map replication} \\"
        ),
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Seed & Control & Epicure & $\Delta$ & Control & Epicure & $\Delta$ \\",
        r"\midrule",
    ]
    for seed in seeds:
        primary_control_score = float(primary_control[seed]["score_unconditional"])
        primary_treatment_score = float(primary_treatment[seed]["score_unconditional"])
        public_control_score = float(public_control[seed]["score_unconditional"])
        public_treatment_score = float(public_treatment[seed]["score_unconditional"])
        lines.append(
            f"{seed} & {primary_control_score:.2f} & {primary_treatment_score:.2f} & "
            f"{primary_treatment_score - primary_control_score:+.2f} & "
            f"{public_control_score:.2f} & {public_treatment_score:.2f} & "
            f"{public_treatment_score - public_control_score:+.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines).encode()


def _render_figure(primary: Mapping[str, Any], public: Mapping[str, Any]) -> tuple[bytes, bytes]:
    ink = "#262B33"
    neutral = "#69736E"
    control_color = "#B77A08"
    treatment_color = "#168C7A"
    grid = "#DDE2DF"
    analyses = (primary, public)
    split_names = (
        f"Anchor-disjoint\ntransfer ({int(primary['tasks'])})",
        f"Public maps\n({int(public['tasks'])})",
    )

    figure = plt.figure(figsize=(7.15, 2.75), facecolor="white")
    grid_spec = figure.add_gridspec(1, 2, width_ratios=(1.45, 1.0), wspace=0.34)
    score_axis = figure.add_subplot(grid_spec[0, 0])
    effect_axis = figure.add_subplot(grid_spec[0, 1])
    score_axis.set_facecolor("white")
    effect_axis.set_facecolor("white")

    offsets = {"pretrained_base": -0.24, "sft_format_control": 0.0, "sft_epicure_optimum": 0.24}
    colors = {
        "pretrained_base": neutral,
        "sft_format_control": control_color,
        "sft_epicure_optimum": treatment_color,
    }
    markers = {"pretrained_base": "D", "sft_format_control": "o", "sft_epicure_optimum": "o"}
    for group, analysis in enumerate(analyses):
        control_runs = _condition(analysis, "sft_format_control")["runs"]
        treatment_runs = _condition(analysis, "sft_epicure_optimum")["runs"]
        if len(control_runs) != 3 or len(treatment_runs) != 3:
            raise RewardTransferAssetError("trained condition does not contain three seeds")
        for control_run, treatment_run in zip(control_runs, treatment_runs, strict=True):
            if control_run["training_seed"] != treatment_run["training_seed"]:
                raise RewardTransferAssetError("matched training seeds differ")
            score_axis.plot(
                [group + offsets["sft_format_control"], group + offsets["sft_epicure_optimum"]],
                [control_run["score_unconditional"], treatment_run["score_unconditional"]],
                color="#BEC5C1",
                linewidth=0.8,
                zorder=1,
            )
        for condition in CONDITIONS:
            summary = _condition(analysis, condition)
            x = group + offsets[condition]
            run_scores = [float(row["score_unconditional"]) for row in summary["runs"]]
            if condition != "pretrained_base":
                score_axis.scatter(
                    [x] * len(run_scores),
                    run_scores,
                    s=15,
                    facecolors="white" if condition == "sft_format_control" else colors[condition],
                    edgecolors=colors[condition],
                    linewidths=0.9,
                    zorder=3,
                )
            score_axis.scatter(
                [x],
                [summary["score_unconditional_mean"]],
                s=48,
                marker=markers[condition],
                facecolors="white" if condition == "sft_format_control" else colors[condition],
                edgecolors=colors[condition],
                linewidths=1.4,
                zorder=4,
                label={
                    "pretrained_base": "Pretrained base",
                    "sft_format_control": "Format control",
                    "sft_epicure_optimum": "Epicure SFT",
                }[condition]
                if group == 0
                else None,
            )
    score_axis.set_xlim(-0.52, 1.52)
    maximum_score = max(
        float(_condition(analysis, condition)["score_unconditional_mean"])
        for analysis in analyses
        for condition in CONDITIONS
    )
    score_axis.set_ylim(0, min(100, max(55, 10 * np.ceil((maximum_score + 5) / 10))))
    score_axis.set_xticks((0, 1), split_names)
    score_axis.set_ylabel("Score (0–100)")
    score_axis.set_title("Condition scores", loc="left", fontsize=9, fontweight="bold", color=ink)
    score_axis.grid(axis="y", color=grid, linewidth=0.7)
    score_axis.spines[["top", "right", "left"]].set_visible(False)
    score_axis.tick_params(axis="x", length=0)
    score_axis.tick_params(axis="y", colors=neutral, labelsize=7)
    score_axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=7,
        ncol=3,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0,
    )

    positions = np.asarray((1.0, 0.0))
    contrasts = (_contrast(primary, "primary"), _contrast(public, "public"))
    estimates = np.asarray([float(row["estimate_points"]) for row in contrasts])
    intervals = np.asarray([row["confidence_interval_95"] for row in contrasts], dtype=float)
    effect_axis.axvline(0, color=ink, linewidth=0.9, zorder=0)
    effect_axis.axvline(3, color=control_color, linewidth=0.9, linestyle=(0, (2, 2)), zorder=0)
    effect_axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates)),
        fmt="o",
        markersize=5.5,
        color=treatment_color,
        ecolor=treatment_color,
        elinewidth=1.6,
        capsize=3,
        zorder=3,
    )
    for estimate, position, contrast in zip(estimates, positions, contrasts, strict=True):
        effect_axis.text(
            estimate,
            position + 0.2,
            f"+{estimate:.2f}  [{contrast['confidence_interval_95'][0]:.2f}, "
            f"{contrast['confidence_interval_95'][1]:.2f}]\n"
            f"p={_plain_p(float(contrast['two_sided_sign_flip_p']))}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=ink,
        )
    effect_axis.text(3.15, -0.47, "+3 practical threshold", fontsize=6.5, color=control_color)
    effect_axis.set_yticks(positions, ("Transfer", "Public maps"))
    x_low = min(-2.0, float(intervals[:, 0].min()) - 2)
    x_high = float(intervals[:, 1].max()) + 3
    effect_axis.set_xlim(x_low, x_high)
    effect_axis.set_ylim(-0.62, 1.62)
    effect_axis.set_xlabel("Epicure SFT minus control (points)")
    effect_axis.set_title("Treatment effect and 95% CI", loc="left", fontsize=9, fontweight="bold")
    effect_axis.grid(axis="x", color=grid, linewidth=0.7)
    effect_axis.spines[["top", "right", "left"]].set_visible(False)
    effect_axis.tick_params(axis="y", length=0)
    effect_axis.tick_params(axis="x", colors=neutral, labelsize=7)

    for axis in (score_axis, effect_axis):
        axis.spines["bottom"].set_color("#AAB2AE")
        axis.spines["bottom"].set_linewidth(0.7)
        axis.title.set_color(ink)
        axis.xaxis.label.set_color(ink)
        axis.yaxis.label.set_color(ink)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.22)

    pdf_handle = io.BytesIO()
    figure.savefig(
        pdf_handle,
        format="pdf",
        metadata={
            "Title": "Epicure reward-transfer experiment",
            "Author": "FlavourBench",
            "Creator": "FlavourBench",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    png_handle = io.BytesIO()
    figure.savefig(
        png_handle,
        format="png",
        dpi=240,
        metadata={"Software": "FlavourBench"},
    )
    plt.close(figure)
    return pdf_handle.getvalue(), png_handle.getvalue()


def _write(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise RewardTransferAssetError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--dataset-assets", type=Path, default=DEFAULT_DATASET_ASSETS)
    args = parser.parse_args()
    primary = _load(args.primary, split="primary")
    public = _load(args.public, split="public")
    if public.get("primary_analysis_artifact_sha256") != primary["artifact_sha256"]:
        raise RewardTransferAssetError("public analysis does not bind the primary analysis")
    macros = _render_macros(primary, public)
    table = _render_table(primary, public)
    family_table = _render_family_table(primary, public)
    seed_table = _render_seed_table(primary, public)
    pdf, png = _render_figure(primary, public)
    outputs = {
        args.generated / "complete-core-reward-transfer-macros.tex": macros,
        args.generated / "complete-core-reward-transfer-table.tex": table,
        args.generated / "complete-core-reward-transfer-family-table.tex": family_table,
        args.generated / "complete-core-reward-transfer-seed-table.tex": seed_table,
        args.figures / "complete-core-reward-transfer.pdf": pdf,
        args.dataset_assets / "complete-core-reward-transfer.png": png,
    }
    for path, payload in outputs.items():
        _write(path, payload)
    manifest = {
        "schema_version": "flavourbench-reward-transfer-paper-assets-v1",
        "primary_analysis_artifact_sha256": primary["artifact_sha256"],
        "public_analysis_artifact_sha256": public["artifact_sha256"],
        "chart_contract": {
            "question": "Does Epicure SFT outperform format-only SFT on disjoint reward maps?",
            "form": "condition dot plot plus paired-effect forest plot",
            "uncertainty": "prespecified crossed seed-anchor bootstrap",
            "non_color_encoding": "diamond base, open control, filled treatment",
            "score_axis": "zero based",
        },
        "files": [
            {
                "path": path.relative_to(REPOSITORY).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(outputs.items(), key=lambda item: str(item[0]))
        ],
    }
    manifest["artifact_sha256"] = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    _write(
        args.generated / "complete-core-reward-transfer-asset-manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(manifest["artifact_sha256"])


if __name__ == "__main__":
    main()
