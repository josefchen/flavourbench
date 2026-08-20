"""Verify the compact, provider-free FlavourBench powered release.

This replay intentionally needs only the checked-in release JSON and its two CSV
tables. It verifies their content addresses and the complete statistical result
contract without making provider or Epicure calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
from typing import Any

COVERAGE_REPAIR_MODEL_IDS = [
    "anthropic/claude-opus-5",
    "google/gemini-3.1-pro-preview",
    "deepseek/deepseek-v4-pro-0813",
    "minimax/minimax-m3",
    "nvidia/nemotron-3.5-lightning",
    "tencent/hy3",
    "meta/muse-glimmer-30b",
    "anthropic/claude-fable-5",
    "thinkingmachines/inkling",
]


class PoweredReleaseError(RuntimeError):
    """Raised when the public release fails closed."""


def _reject_constant(value: str) -> None:
    raise PoweredReleaseError(f"non-finite JSON constant: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise PoweredReleaseError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoweredReleaseError(f"cannot read release: {path}") from exc
    if not isinstance(document, dict):
        raise PoweredReleaseError("release root must be an object")
    return document


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoweredReleaseError(message)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PoweredReleaseError(f"cannot read table: {path}") from exc
    return list(csv.DictReader(io.StringIO(text)))


def verify_release(path: Path) -> dict[str, Any]:
    """Verify one compact release and return a short reproducibility summary."""

    _require(path.is_file() and not path.is_symlink(), "release must be a regular file")
    release = _load_json(path)
    stated = str(release.get("artifact_sha256", ""))
    semantic_payload = dict(release)
    semantic_payload.pop("artifact_sha256", None)
    _require(len(stated) == 64, "release semantic hash is malformed")
    _require(_canonical_sha256(semantic_payload) == stated, "release semantic hash failed")
    _require(stated in path.name, "release filename is not content addressed")
    _require(release.get("status") == "final_complete", "release is not final_complete")
    schema = str(release.get("schema_version") or "")
    _require(
        schema
        in {
            "flavourbench-selection-powered-release-v1",
            "flavourbench-selection-powered-release-v2-anchor-free",
            "flavourbench-selection-powered-joint-release-v1",
        },
        "unexpected release schema",
    )
    success_only = schema in {
        "flavourbench-selection-powered-release-v2-anchor-free",
        "flavourbench-selection-powered-joint-release-v1",
    }
    joint = schema == "flavourbench-selection-powered-joint-release-v1"

    analysis = release.get("analysis")
    _require(isinstance(analysis, dict), "analysis is missing")
    _require(analysis.get("status") == "final_complete", "analysis is not final_complete")
    models = analysis.get("models")
    pairs = analysis.get("pairwise_comparisons")
    repeats = analysis.get("repeatability")
    _require(isinstance(models, list) and len(models) >= 2, "expected at least two models")
    model_count = len(models)
    pair_count = model_count * (model_count - 1) // 2
    _require(isinstance(pairs, list) and len(pairs) == pair_count, "pairwise row count failed")
    _require(isinstance(repeats, list) and len(repeats) == model_count, "repeat row count failed")

    model_ids = [str(row["model_id"]) for row in models]
    _require(len(set(model_ids)) == model_count, "model IDs are not unique")
    availability_key = "coverage" if success_only else "availability"
    scheduled_tasks = 1280 if joint else 640
    scheduled_repeats = 128 if joint else 64
    _require(
        all(int(row[availability_key]["scheduled"]) == scheduled_tasks for row in models),
        f"every model must have {scheduled_tasks} scheduled primary tasks",
    )
    if success_only:
        _require(analysis.get("dnf_classification") is False, "v2 unexpectedly enables DNF")
        _require(
            analysis.get("failure_handling")
            == "excluded_from_quality_score_and_retained_in_coverage",
            "v2 failure handling changed",
        )
        _require(
            all(
                row.get("score_status") == "scored"
                and 0 < int(row["coverage"]["valid_scored"]) <= scheduled_tasks
                for row in models
            ),
            "v2 model score or coverage contract failed",
        )
    _require(
        all(
            int(row.get("scheduled", row.get("tasks", -1))) == scheduled_repeats for row in repeats
        ),
        f"every model must have {scheduled_repeats} scheduled repeat tasks",
    )
    _require({str(row["model_id"]) for row in repeats} == set(model_ids), "repeat roster drift")

    expected_pairs = {frozenset(pair) for pair in combinations(model_ids, 2)}
    observed_pairs = {
        frozenset((str(row["left_model_id"]), str(row["right_model_id"]))) for row in pairs
    }
    _require(observed_pairs == expected_pairs, "pairwise comparison grid is incomplete")

    inputs = release.get("inputs")
    _require(isinstance(inputs, dict), "release inputs are missing")
    if joint:
        primary_count = int(inputs["panel_1_primary"]["count"]) + int(
            inputs["panel_2_primary"]["count"]
        )
        repeat_count = int(inputs["panel_1_repeat"]["count"]) + int(
            inputs["panel_2_repeat"]["count"]
        )
        design = analysis.get("design") or {}
        inference = analysis.get("inference") or {}
        _require(design.get("unique_anchor_clusters") == 1178, "joint anchor count failed")
        _require(design.get("shared_anchor_clusters") == 102, "shared anchor count failed")
        _require(
            inference.get("independence_unit") == "anchor_ingredient"
            and inference.get("independent_cluster_count") == 1178
            and inference.get("shared_anchor_tasks_move_together") is True,
            "joint cluster-inference contract failed",
        )
        _require(
            isinstance(analysis.get("panel_replication"), dict),
            "joint replication diagnostic is missing",
        )
        lineage = inputs.get("response_lineage") or {}
        replacement_plan = lineage.get("panel_2_replacement_plan_sha256")
        if replacement_plan is not None:
            _require(
                isinstance(replacement_plan, str)
                and len(replacement_plan) == 64
                and isinstance(lineage.get("panel_2_base_plan_sha256"), str)
                and len(lineage["panel_2_base_plan_sha256"]) == 64
                and replacement_plan != lineage["panel_2_base_plan_sha256"],
                "panel-2 replacement plan lineage is malformed",
            )
            _require(
                lineage.get("panel_2_replacement_model_ids")
                == [
                    "openai/gpt-5.6-luna-pro",
                    "deepseek/deepseek-v4-flash-0731",
                ]
                and lineage.get("panel_2_superseded_route_responses_used") is False,
                "panel-2 replacement source contract failed",
            )
        panel_1_coverage_plan = lineage.get("panel_1_coverage_repair_plan_sha256")
        panel_2_coverage_plan = lineage.get("panel_2_coverage_repair_plan_sha256")
        if panel_1_coverage_plan is not None or panel_2_coverage_plan is not None:
            _require(
                isinstance(panel_1_coverage_plan, str)
                and len(panel_1_coverage_plan) == 64
                and isinstance(panel_2_coverage_plan, str)
                and len(panel_2_coverage_plan) == 64
                and panel_1_coverage_plan != lineage["panel_1_base_plan_sha256"]
                and panel_1_coverage_plan != lineage["panel_1_qwen_replacement_plan_sha256"]
                and panel_2_coverage_plan != lineage["panel_2_base_plan_sha256"]
                and panel_2_coverage_plan != replacement_plan,
                "complete coverage-repair plan lineage is malformed",
            )
            _require(
                lineage.get("panel_1_coverage_repair_model_ids") == COVERAGE_REPAIR_MODEL_IDS
                and lineage.get("panel_2_coverage_repair_model_ids") == COVERAGE_REPAIR_MODEL_IDS
                and set(COVERAGE_REPAIR_MODEL_IDS) <= set(model_ids)
                and lineage.get("panel_1_superseded_coverage_route_responses_used") is False
                and lineage.get("panel_2_superseded_coverage_route_responses_used") is False
                and lineage.get("panel_1_fable_replacement_plan_sha256") is None
                and isinstance(lineage.get("panel_1_superseded_fable_replacement_plan_sha256"), str)
                and len(lineage["panel_1_superseded_fable_replacement_plan_sha256"]) == 64,
                "complete coverage-repair response source contract failed",
            )
        panel_1_deepseek_plan = lineage.get("panel_1_deepseek_repair_plan_sha256")
        panel_2_deepseek_plan = lineage.get("panel_2_deepseek_repair_plan_sha256")
        if panel_1_deepseek_plan is not None or panel_2_deepseek_plan is not None:
            _require(
                isinstance(panel_1_deepseek_plan, str)
                and len(panel_1_deepseek_plan) == 64
                and isinstance(panel_2_deepseek_plan, str)
                and len(panel_2_deepseek_plan) == 64
                and panel_1_deepseek_plan != panel_1_coverage_plan
                and panel_2_deepseek_plan != panel_2_coverage_plan,
                "DeepSeek repair plan lineage is malformed",
            )
            _require(
                lineage.get("panel_1_deepseek_repair_model_ids")
                == ["deepseek/deepseek-v4-pro-0813"]
                and lineage.get("panel_2_deepseek_repair_model_ids")
                == ["deepseek/deepseek-v4-pro-0813"]
                and lineage.get("panel_1_deepseek_repair_provider_tag")
                == lineage.get("panel_2_deepseek_repair_provider_tag")
                and lineage.get("panel_1_deepseek_repair_provider_tag")
                in {"baseten/fp4", "gmicloud/fp8"}
                and lineage.get("panel_1_superseded_deepseek_route_responses_used") is False
                and lineage.get("panel_2_superseded_deepseek_route_responses_used") is False,
                "DeepSeek repair response source contract failed",
            )
            _require(
                lineage.get("deepseek_quality_scores_inspected_before_source_freeze") is False,
                "DeepSeek route selection inspected quality before source freeze",
            )
    else:
        primary_count = int(inputs["primary_responses"]["count"])
        repeat_count = int(inputs["repeat_responses"]["count"])
    _require(
        primary_count == model_count * scheduled_tasks,
        "primary response count differs from the model/task grid",
    )
    _require(
        repeat_count == model_count * scheduled_repeats,
        "repeat response count differs from the model/repeat grid",
    )

    tables = release.get("tables")
    _require(isinstance(tables, dict), "release table commitments are missing")
    table_rows: dict[str, list[dict[str, str]]] = {}
    for label, expected_rows in (("leaderboard", model_count), ("pairwise", pair_count)):
        table = tables.get(label)
        _require(isinstance(table, dict), f"{label} table commitment is missing")
        table_path = path.parent / str(table.get("filename", ""))
        _require(table_path.is_file() and not table_path.is_symlink(), f"{label} table is missing")
        _require(_file_sha256(table_path) == table.get("sha256"), f"{label} table hash failed")
        rows = _csv_rows(table_path)
        _require(len(rows) == expected_rows, f"{label} table row count failed")
        table_rows[label] = rows

    _require(
        {row["model_id"] for row in table_rows["leaderboard"]} == set(model_ids),
        "leaderboard roster differs from release",
    )
    table_pairs = {
        frozenset((row["left_model_id"], row["right_model_id"])) for row in table_rows["pairwise"]
    }
    _require(table_pairs == expected_pairs, "pairwise CSV grid differs from release")

    ranked = sorted(
        (row for row in models if row.get("point_estimate_rank") is not None),
        key=lambda row: (int(row["point_estimate_rank"]), str(row["model_id"])),
    )
    _require(ranked, "release has no ranked model")
    if success_only:
        _require(len(ranked) == model_count, "v2 must rank every scored model")
    _require(
        [int(row["point_estimate_rank"]) for row in ranked] == list(range(1, len(ranked) + 1)),
        "point ranks are not contiguous",
    )
    return {
        "status": "verified",
        "release": path.name,
        "artifact_sha256": stated,
        "models": len(models),
        "tasks_per_model": scheduled_tasks,
        "independent_anchor_clusters": 1178 if joint else scheduled_tasks,
        "primary_responses": model_count * scheduled_tasks,
        "repeat_responses": model_count * scheduled_repeats,
        "pairwise_comparisons": len(pairs),
        "leader_model_id": ranked[0]["model_id"],
        "leader_score": ranked[0]["flavourbench_score"],
        "definitive_top_model_id": analysis.get("definitive_top_model_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.release), sort_keys=True))


if __name__ == "__main__":
    main()
