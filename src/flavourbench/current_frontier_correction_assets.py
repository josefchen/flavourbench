"""Render the post-collection task-quarantine accounting used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class CorrectionAssetError(RuntimeError):
    """A correction input failed its content or cross-artifact checks."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_verified(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CorrectionAssetError(f"input must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CorrectionAssetError(f"input is not an object: {path}")
    digest = value.get("artifact_sha256")
    unhashed = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if not isinstance(digest, str) or hashlib.sha256(_canonical(unhashed)).hexdigest() != digest:
        raise CorrectionAssetError(f"content address does not verify: {path}")
    return value


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or value < 0:
        raise CorrectionAssetError(f"missing non-negative integer: {key}")
    return value


def _integer_or_zero(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    if not isinstance(value, int) or value < 0:
        raise CorrectionAssetError(f"invalid non-negative integer: {key}")
    return value


def build_summary(
    *,
    arena: Mapping[str, Any],
    strict_uplift: Mapping[str, Any],
    high_uplift: Mapping[str, Any],
    quarantine: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check and summarize gross, excluded, and retained evidence units."""

    quarantine_sha256 = str(quarantine.get("artifact_sha256") or "")
    task_records = quarantine.get("tasks")
    if not isinstance(task_records, list):
        task_records = quarantine.get("quarantined_tasks")
    if not isinstance(task_records, list):
        task_records = quarantine.get("records")
    if not isinstance(task_records, list):
        raise CorrectionAssetError("quarantine has no task records")
    task_ids = sorted(
        str(record.get("task_id") or "")
        for record in task_records
        if isinstance(record, Mapping) and record.get("task_id")
    )
    if len(task_ids) != 4 or len(set(task_ids)) != 4:
        raise CorrectionAssetError("expected four unique quarantined tasks")

    pools = (arena, strict_uplift, high_uplift)
    for pool in pools:
        binding = (pool.get("selection_policy") or {}).get("task_quarantine")
        if not isinstance(binding, Mapping) or binding.get("artifact_sha256") != quarantine_sha256:
            raise CorrectionAssetError("review pool does not bind the quarantine artifact")
        if sorted(str(value) for value in binding.get("task_ids") or []) != task_ids:
            raise CorrectionAssetError("review pool quarantine task set differs")

    arena_observed = arena.get("observed")
    strict_observed = strict_uplift.get("observed")
    high_observed = high_uplift.get("observed")
    counts = coverage.get("counts")
    if not all(
        isinstance(value, Mapping)
        for value in (arena_observed, strict_observed, high_observed, counts)
    ):
        raise CorrectionAssetError("correction input is missing observed counts")
    assert isinstance(arena_observed, Mapping)
    assert isinstance(strict_observed, Mapping)
    assert isinstance(high_observed, Mapping)
    assert isinstance(counts, Mapping)

    gross_arena = _integer(arena_observed, "source_candidate_comparisons_before_task_quarantine")
    retained_arena = _integer(arena_observed, "candidate_comparisons")
    excluded_arena = _integer(arena_observed, "task_quarantined_candidate_comparisons")
    gross_answers = _integer(arena_observed, "source_response_arms_before_task_quarantine")
    retained_answers = _integer(arena_observed, "source_response_arms")
    excluded_answers = _integer(arena_observed, "task_quarantined_source_response_arms")
    gross_tasks = _integer(arena_observed, "source_unique_task_ids_before_task_quarantine")
    retained_tasks = _integer(arena_observed, "unique_task_ids")

    gross_uplift = sum(
        _integer(observed, "source_candidate_pairs_before_task_quarantine")
        for observed in (strict_observed, high_observed)
    )
    retained_uplift = sum(
        _integer(observed, "candidate_pairs")
        for observed in (strict_observed, high_observed)
    )
    excluded_uplift = sum(
        _integer(observed, "task_quarantined_candidate_pairs")
        for observed in (strict_observed, high_observed)
    )
    strict_by_model = strict_observed.get("candidate_pairs_by_model")
    high_by_model = high_observed.get("candidate_pairs_by_model")
    if not isinstance(strict_by_model, Mapping) or not isinstance(high_by_model, Mapping):
        raise CorrectionAssetError("uplift pools have no per-model counts")
    model_ids = sorted({str(value) for value in (*strict_by_model, *high_by_model)})
    uplift_by_model = {
        model_id: _integer_or_zero(strict_by_model, model_id)
        + _integer_or_zero(high_by_model, model_id)
        for model_id in model_ids
    }

    for gross, retained, excluded, label in (
        (gross_arena, retained_arena, excluded_arena, "arena comparisons"),
        (gross_answers, retained_answers, excluded_answers, "arena answers"),
        (gross_uplift, retained_uplift, excluded_uplift, "uplift pairs"),
    ):
        if retained + excluded != gross:
            raise CorrectionAssetError(f"gross accounting does not close for {label}")
    if retained_tasks + len(task_ids) != gross_tasks:
        raise CorrectionAssetError("task accounting does not close")

    evidence_units = arena_observed.get("evidence_units")
    if not isinstance(evidence_units, Mapping):
        raise CorrectionAssetError("arena pool has no evidence-unit record")
    current_missing = _integer(counts, "current_missing_model_pair_family_cells")
    total_cells = _integer(counts, "current_model_pair_family_cells")
    if current_missing != _integer(arena_observed, "missing_model_pair_family_cells"):
        raise CorrectionAssetError("coverage schedule and arena pool disagree")

    return {
        "arena_comparisons": {
            "gross": gross_arena,
            "excluded": excluded_arena,
            "retained": retained_arena,
        },
        "arena_answers": {
            "gross": gross_answers,
            "excluded": excluded_answers,
            "retained": retained_answers,
        },
        "tasks": {
            "gross": gross_tasks,
            "excluded": len(task_ids),
            "retained": retained_tasks,
            "quarantined_task_ids": task_ids,
        },
        "uplift_pairs": {
            "gross": gross_uplift,
            "excluded": excluded_uplift,
            "retained": retained_uplift,
            "retained_by_model": uplift_by_model,
            "minimum_per_model": min(uplift_by_model.values()),
            "maximum_per_model": max(uplift_by_model.values()),
            "models_below_eight": sum(value < 8 for value in uplift_by_model.values()),
        },
        "dependence": {
            "task_stratum_clusters": _integer(evidence_units, "task_stratum_clusters"),
            "minimum_reuse": _integer(
                evidence_units, "minimum_comparisons_per_reused_response_arm"
            ),
            "median_reuse": _integer(
                evidence_units, "median_comparisons_per_reused_response_arm"
            ),
            "maximum_reuse": _integer(
                evidence_units, "maximum_comparisons_per_reused_response_arm"
            ),
        },
        "coverage": {
            "missing_cells": current_missing,
            "total_cells": total_cells,
            "missing_endpoint_task_cells": _integer(counts, "missing_endpoint_task_cells"),
            "scheduled_real_arms": _integer(counts, "required_new_real_arms"),
            "projected_missing_cells": _integer(
                counts, "projected_missing_model_pair_family_cells_after_schedule"
            ),
            "calls_completed": False,
        },
        "bindings": {
            "arena_pool_sha256": arena["artifact_sha256"],
            "strict_uplift_pool_sha256": strict_uplift["artifact_sha256"],
            "high_uplift_pool_sha256": high_uplift["artifact_sha256"],
            "quarantine_sha256": quarantine_sha256,
            "coverage_schedule_sha256": coverage["artifact_sha256"],
        },
    }


def _macro(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def write_assets(summary: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    macros = output_dir / "current-frontier-correction-macros.tex"
    arena = summary["arena_comparisons"]
    answers = summary["arena_answers"]
    tasks = summary["tasks"]
    uplift = summary["uplift_pairs"]
    dependence = summary["dependence"]
    coverage = summary["coverage"]
    macros.write_text(
        "\n".join(
            [
                _macro("FrontierGrossArenaComparisons", arena["gross"]),
                _macro("FrontierExcludedArenaComparisons", arena["excluded"]),
                _macro("FrontierRetainedArenaComparisons", arena["retained"]),
                _macro("FrontierGrossArenaAnswers", answers["gross"]),
                _macro("FrontierExcludedArenaAnswers", answers["excluded"]),
                _macro("FrontierRetainedArenaAnswers", answers["retained"]),
                _macro("FrontierGrossArenaTaskCount", tasks["gross"]),
                _macro("FrontierQuarantinedArenaTaskCount", tasks["excluded"]),
                _macro("FrontierRetainedArenaTaskCount", tasks["retained"]),
                _macro("FrontierGrossUpliftPairs", uplift["gross"]),
                _macro("FrontierExcludedUpliftPairs", uplift["excluded"]),
                _macro("FrontierRetainedUpliftPairs", uplift["retained"]),
                _macro("FrontierRetainedUpliftMinimum", uplift["minimum_per_model"]),
                _macro("FrontierRetainedUpliftMaximum", uplift["maximum_per_model"]),
                _macro("FrontierRetainedUpliftModelsBelowEight", uplift["models_below_eight"]),
                _macro("FrontierTaskStratumClusters", dependence["task_stratum_clusters"]),
                _macro("FrontierResponseReuseMinimum", dependence["minimum_reuse"]),
                _macro("FrontierResponseReuseMedian", dependence["median_reuse"]),
                _macro("FrontierResponseReuseMaximum", dependence["maximum_reuse"]),
                _macro("FrontierMissingPairFamilyCells", coverage["missing_cells"]),
                _macro("FrontierPairFamilyCells", coverage["total_cells"]),
                _macro("FrontierCoverageRepairCells", coverage["missing_endpoint_task_cells"]),
                _macro("FrontierCoverageRepairArms", coverage["scheduled_real_arms"]),
                _macro(
                    "FrontierProjectedMissingPairFamilyCells",
                    coverage["projected_missing_cells"],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    table = output_dir / "current-frontier-correction-table.tex"
    table.write_text(
        "\n".join(
            [
                r"\begin{tabular}{@{}lrrr@{}}",
                r"\toprule",
                r"Evidence unit & Gross & Held & Retained \\",
                r"\midrule",
                (
                    f"Tasks in arena & {tasks['gross']} & {tasks['excluded']} & "
                    f"{tasks['retained']} \\\\"
                ),
                (
                    f"Epicure-on answers & {answers['gross']} & {answers['excluded']} & "
                    f"{answers['retained']} \\\\"
                ),
                (
                    f"Arena comparisons & {arena['gross']:,} & {arena['excluded']} & "
                    f"{arena['retained']} \\\\"
                ),
                (
                    f"Matched uplift pairs & {uplift['gross']} & {uplift['excluded']} & "
                    f"{uplift['retained']} \\\\"
                ),
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = output_dir / "current-frontier-correction-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"macros": macros, "table": table, "summary": summary_path}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-pool", type=Path, required=True)
    parser.add_argument("--strict-uplift-pool", type=Path, required=True)
    parser.add_argument("--high-uplift-pool", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = build_summary(
        arena=_read_verified(arguments.arena_pool),
        strict_uplift=_read_verified(arguments.strict_uplift_pool),
        high_uplift=_read_verified(arguments.high_uplift_pool),
        quarantine=_read_verified(arguments.quarantine),
        coverage=_read_verified(arguments.coverage),
    )
    paths = write_assets(summary, arguments.output_dir)
    print(json.dumps({key: str(path.resolve()) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    run()
