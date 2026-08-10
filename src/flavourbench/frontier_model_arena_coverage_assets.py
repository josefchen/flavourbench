"""Render source-bound coverage assets from a corrected model-arena manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from .current_pilot_review_import import ReviewPool
from .frontier_model_arena_review_pool import (
    _render_coverage_figure,
    _write_coverage_csv,
)
from .real_task_bank import sha256_json

PROVENANCE_SCHEMA = "flavourbench-model-arena-coverage-assets-v1"


class CoverageAssetError(RuntimeError):
    """A corrected arena or generated asset failed its integrity contract."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_addressed(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CoverageAssetError("arena input must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageAssetError("arena input is not valid JSON") from error
    if not isinstance(value, dict):
        raise CoverageAssetError("arena input must be an object")
    digest = value.get("artifact_sha256")
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if not isinstance(digest, str) or sha256_json(body) != digest:
        raise CoverageAssetError("arena content address does not verify")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    observed = manifest.get("observed")
    items = manifest.get("items")
    model_order = manifest.get("model_order")
    boundary = manifest.get("claim_boundary")
    if (
        manifest.get("track") != "model_arena"
        or not isinstance(observed, Mapping)
        or not isinstance(items, list)
        or not isinstance(model_order, list)
        or len(model_order) < 2
        or len(set(map(str, model_order))) != len(model_order)
        or not isinstance(boundary, Mapping)
        or boundary.get("official") is not False
        or boundary.get("rank_eligible") is not False
        or observed.get("synthetic_arms") != 0
        or observed.get("candidate_comparisons") != len(items)
    ):
        raise CoverageAssetError("arena claim boundary or evidence counts are invalid")
    response_counts: Counter[str] = Counter()
    roster = set(map(str, model_order))
    for item in items:
        if not isinstance(item, Mapping):
            raise CoverageAssetError("arena item must be an object")
        sides: list[Mapping[str, Any]] = []
        for side in ("left", "right"):
            arm = item.get(side)
            if not isinstance(arm, Mapping):
                raise CoverageAssetError("arena item is missing a side")
            sides.append(arm)
        models = [str(arm.get("requested_model_id") or "") for arm in sides]
        responses = [str(arm.get("response_artifact_sha256") or "") for arm in sides]
        if (
            models[0] == models[1]
            or any(model not in roster for model in models)
            or any(len(value) != 64 for value in responses)
        ):
            raise CoverageAssetError("arena item has an invalid model or response identity")
        response_counts.update(responses)
    evidence = observed.get("evidence_units")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("comparison_rows_treated_as_independent") is not False
        or evidence.get("scalar_effective_sample_size_claimed") is not False
    ):
        raise CoverageAssetError("arena does not preserve the dependence boundary")


def _validate_uplift(uplift: Mapping[str, Any]) -> None:
    observed = uplift.get("observed")
    boundary = uplift.get("claim_boundary")
    if (
        uplift.get("track") != "epicure_uplift"
        or not isinstance(observed, Mapping)
        or not isinstance(boundary, Mapping)
        or boundary.get("official") is not False
        or boundary.get("rank_eligible") is not False
        or observed.get("synthetic_arms") != 0
        or not isinstance(observed.get("candidate_pairs"), int)
        or not isinstance(observed.get("coverage_recovery_pairs_added"), int)
    ):
        raise CoverageAssetError("uplift claim boundary or evidence counts are invalid")


def _validate_coverage(
    coverage: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    uplift: Mapping[str, Any] | None,
) -> None:
    arena_counts = coverage.get("model_arena")
    uplift_counts = coverage.get("uplift")
    boundary = coverage.get("claim_boundary")
    if (
        not isinstance(arena_counts, Mapping)
        or not isinstance(uplift_counts, Mapping)
        or not isinstance(boundary, Mapping)
        or boundary.get("official") is not False
        or boundary.get("rank_eligible") is not False
        or boundary.get("synthetic_arms") != 0
        or arena_counts.get("comparisons_after") != manifest["observed"]["candidate_comparisons"]
        or arena_counts.get("missing_cells_after")
        != manifest["observed"]["missing_model_pair_family_cells"]
        or arena_counts.get("unique_response_arms_after")
        != manifest["observed"]["source_response_arms"]
        or (
            uplift is not None
            and uplift_counts.get("pairs_after") != uplift["observed"]["candidate_pairs"]
        )
    ):
        raise CoverageAssetError("coverage audit does not match corrected evidence")


def _macro(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _write_current_macros(
    *,
    manifest: Mapping[str, Any],
    uplift: Mapping[str, Any],
    coverage: Mapping[str, Any] | None,
    response_counts: Counter[str],
    path: Path,
) -> None:
    observed = manifest["observed"]
    uplift_observed = uplift["observed"]
    reuse = sorted(response_counts.values())
    missing_by_family = observed["missing_model_pair_family_cells_by_family"]
    arena_counts = coverage.get("model_arena") if coverage is not None else None
    answers_added = (
        arena_counts["unique_response_arms_after"] - arena_counts["unique_response_arms_before"]
        if isinstance(arena_counts, Mapping)
        else 0
    )
    values = [
        _macro("FrontierCurrentArenaComparisons", observed["candidate_comparisons"]),
        _macro(
            "FrontierCurrentArenaComparisonsAdded",
            observed["coverage_recovery_candidate_comparisons_added"],
        ),
        _macro("FrontierCurrentArenaSourceAnswers", observed["source_response_arms"]),
        _macro("FrontierCurrentArenaAnswersAdded", answers_added),
        _macro("FrontierCurrentArenaComparedAnswers", observed["compared_response_arms"]),
        _macro("FrontierCurrentArenaUnpairedAnswers", observed["unpaired_response_arms"]),
        _macro("FrontierCurrentArenaTaskCount", observed["unique_task_ids"]),
        _macro("FrontierCurrentArenaTaskStratumClusters", observed["task_stratum_clusters"]),
        _macro("FrontierCurrentResponseReuseMinimum", min(reuse)),
        _macro("FrontierCurrentResponseReuseMedian", int(median(reuse))),
        _macro("FrontierCurrentResponseReuseMaximum", max(reuse)),
        _macro(
            "FrontierCurrentMissingPairFamilyCells", observed["missing_model_pair_family_cells"]
        ),
        _macro("FrontierCurrentPairFamilyCells", observed["model_pair_family_cells"]),
        _macro("FrontierCurrentMissingCompositionCells", missing_by_family["composition"]),
        _macro("FrontierCurrentMissingCookabilityCells", missing_by_family["cookability"]),
        _macro("FrontierCurrentMissingEvidenceCells", missing_by_family["evidence"]),
        _macro("FrontierCurrentMissingSubstitutionCells", missing_by_family["substitution"]),
        _macro("FrontierCurrentUpliftPairs", uplift_observed["candidate_pairs"]),
        _macro("FrontierCurrentUpliftPairsAdded", uplift_observed["coverage_recovery_pairs_added"]),
        _macro("FrontierCurrentUpliftArms", uplift_observed["source_arms"]),
        _macro("FrontierCurrentUpliftTaskCount", uplift_observed["unique_task_ids"]),
        "",
    ]
    path.write_text("\n".join(values), encoding="utf-8")


def render_assets(
    arena_path: Path,
    output_dir: Path,
    uplift_path: Path | None = None,
    coverage_path: Path | None = None,
) -> dict[str, Path]:
    manifest = _read_addressed(arena_path)
    _validate_manifest(manifest)
    uplift = _read_addressed(uplift_path) if uplift_path is not None else None
    if uplift is not None:
        _validate_uplift(uplift)
    coverage = _read_addressed(coverage_path) if coverage_path is not None else None
    if coverage is not None:
        _validate_coverage(coverage, manifest=manifest, uplift=uplift)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise CoverageAssetError("output directory must be a regular directory")
    pool = ReviewPool(manifest=manifest, pairs=tuple())
    csv_path = output_dir / "frontier-model-arena-coverage.csv"
    pdf_path = output_dir / "frontier-model-arena-coverage.pdf"
    _write_coverage_csv(pool, csv_path)
    _render_coverage_figure(pool, pdf_path)
    svg_path = pdf_path.with_suffix(".svg")
    response_counts = Counter(
        str(item[side]["response_artifact_sha256"])
        for item in manifest["items"]
        for side in ("left", "right")
    )
    assets: dict[str, dict[str, str]] = {
        "csv": {"filename": csv_path.name, "sha256": _file_sha256(csv_path)},
        "pdf": {"filename": pdf_path.name, "sha256": _file_sha256(pdf_path)},
        "svg": {"filename": svg_path.name, "sha256": _file_sha256(svg_path)},
    }
    result = {"csv": csv_path, "pdf": pdf_path, "svg": svg_path}
    macros_path: Path | None = None
    if uplift is not None:
        macros_path = output_dir / "frontier-model-arena-current-macros.tex"
        _write_current_macros(
            manifest=manifest,
            uplift=uplift,
            coverage=coverage,
            response_counts=response_counts,
            path=macros_path,
        )
        assets["macros"] = {
            "filename": macros_path.name,
            "sha256": _file_sha256(macros_path),
        }
        result["macros"] = macros_path
    provenance_body = {
        "schema_version": PROVENANCE_SCHEMA,
        "source_arena_sha256": manifest["artifact_sha256"],
        "source_arena_physical_sha256": _file_sha256(arena_path),
        "candidate_comparison_rows": len(manifest["items"]),
        "compared_response_arms": len(response_counts),
        "response_reuse_frequency": {
            str(reuse): count for reuse, count in sorted(Counter(response_counts.values()).items())
        },
        "model_pair_family_cells": manifest["observed"].get("model_pair_family_cells"),
        "missing_model_pair_family_cells": manifest["observed"].get(
            "missing_model_pair_family_cells"
        ),
        "source_uplift_sha256": uplift.get("artifact_sha256") if uplift else None,
        "source_coverage_sha256": coverage.get("artifact_sha256") if coverage else None,
        "candidate_uplift_pairs": (
            uplift["observed"]["candidate_pairs"] if uplift is not None else None
        ),
        "assets": assets,
        "claim_boundary": {
            "official": False,
            "quality_judgments": 0,
            "comparison_rows_are_independent": False,
            "permitted_use": "development coverage and dependence diagnostics",
        },
    }
    provenance = {
        **provenance_body,
        "artifact_sha256": sha256_json(provenance_body),
    }
    provenance_path = output_dir / "frontier-model-arena-coverage-provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["provenance"] = provenance_path
    return result


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--uplift", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    paths = render_assets(
        arguments.arena,
        arguments.output_dir,
        arguments.uplift,
        arguments.coverage,
    )
    print(
        json.dumps(
            {key: str(path.resolve()) for key, path in paths.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
