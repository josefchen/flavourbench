"""Build the public Epicure-grounded operational leaderboard.

This leaderboard intentionally answers a machine-verifiable operational question:
did an endpoint complete both arms of a frozen real-task pair, with the Epicure-on
arm completing at least one successful real Epicure tool call?  It does not infer
culinary quality or human preference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-epicure-operational-leaderboard-v1"
SOURCE_SCHEMA_VERSION = "flavourbench-frontier-multirun-assets-v1"
QWEN_SCHEMA_VERSION = "flavourbench-qwencloud-exploratory-operational-projection-v1"

EXPECTED_SOURCE_SEMANTIC_SHA256 = "c0bd526a2776a25adfbd2c43b98b8f15c143a8cb93b957ba961d0e9efe626688"
EXPECTED_SOURCE_PHYSICAL_SHA256 = "377a6afffab5c3b6072be8157fb08fb0a1d94e59900a69d68b39c5ac268c2252"
EXPECTED_EXECUTION_POLICY_SHA256 = (
    "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d"
)
EXPECTED_TASK_SET_SHA256 = "dc6a5ecea427ac2fec8198be9a852d1ca29d4033cd26bee9f240bd7c19fb2a92"
EXPECTED_QWEN_SEMANTIC_SHA256 = "b2f7790b3eb18d1df083397ce02b5296c549e5ed3ddb3d3f32ea776db3ddca04"
EXPECTED_QWEN_PHYSICAL_SHA256 = "9343a2959d3acf3079fb91b2bd7ff608af421532b826f7b98917c88b76a7f85c"

REQUIRED_PANEL = {
    "moonshotai/kimi-k3",
    "cohere/command-a-plus-05-2026",
    "cohere/command-a-reasoning-08-2025",
}


class EpicureOperationalLeaderboardError(RuntimeError):
    """The frozen evidence does not satisfy the publication contract."""


def _reject_constant(value: str) -> None:
    raise EpicureOperationalLeaderboardError(f"non-finite JSON value is forbidden: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EpicureOperationalLeaderboardError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EpicureOperationalLeaderboardError(f"invalid JSON: {path}") from error
    if not isinstance(document, dict):
        raise EpicureOperationalLeaderboardError(f"expected a JSON object: {path}")
    return document


def _physical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_content_address(
    path: Path,
    document: Mapping[str, Any],
    *,
    expected_physical: str,
    expected_semantic: str,
) -> None:
    if _physical_sha256(path) != expected_physical:
        raise EpicureOperationalLeaderboardError(f"physical source drift: {path}")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("artifact_sha256") != expected_semantic
        or sha256_json(payload) != expected_semantic
    ):
        raise EpicureOperationalLeaderboardError(f"semantic source drift: {path}")


def _validate_source(path: Path) -> dict[str, Any]:
    source = _read_json(path)
    _verify_content_address(
        path,
        source,
        expected_physical=EXPECTED_SOURCE_PHYSICAL_SHA256,
        expected_semantic=EXPECTED_SOURCE_SEMANTIC_SHA256,
    )
    if (
        source.get("schema_version") != SOURCE_SCHEMA_VERSION
        or source.get("status") != "verified_real_development_pilot"
        or source.get("official") is not False
        or source.get("rank_eligible") is not False
        or source.get("quality_ranking") is not False
        or source.get("synthetic_tasks") != 0
        or source.get("execution_policy_sha256") != EXPECTED_EXECUTION_POLICY_SHA256
        or source.get("task_set_sha256") != EXPECTED_TASK_SET_SHA256
    ):
        raise EpicureOperationalLeaderboardError("source evidence boundary or protocol drifted")
    totals = source.get("totals")
    rows = source.get("model_rows")
    if not isinstance(totals, Mapping) or not isinstance(rows, list):
        raise EpicureOperationalLeaderboardError("source totals or model rows are missing")
    expected_totals = {
        "models": 16,
        "runs": 7,
        "distinct_tasks": 16,
        "task_families": 4,
        "scheduled_pairs": 152,
        "complete_pairs": 110,
        "completed_response_arms": 262,
        "epicure_calls": 273,
        "epicure_successful_calls": 207,
        "quality_judgments": 0,
        "synthetic_tasks": 0,
    }
    for key, value in expected_totals.items():
        if totals.get(key) != value:
            raise EpicureOperationalLeaderboardError(f"source total drifted: {key}")
    model_ids = [str(row.get("model_id") or "") for row in rows if isinstance(row, Mapping)]
    if len(rows) != 16 or len(set(model_ids)) != 16 or not REQUIRED_PANEL <= set(model_ids):
        raise EpicureOperationalLeaderboardError("the common 16-model panel is incomplete")
    return source


def _validate_qwen(path: Path) -> dict[str, Any]:
    qwen = _read_json(path)
    _verify_content_address(
        path,
        qwen,
        expected_physical=EXPECTED_QWEN_PHYSICAL_SHA256,
        expected_semantic=EXPECTED_QWEN_SEMANTIC_SHA256,
    )
    identity = qwen.get("model_identity")
    run = qwen.get("successor_operational_run")
    boundary = qwen.get("claim_boundary")
    if (
        qwen.get("schema_version") != QWEN_SCHEMA_VERSION
        or qwen.get("status") != "verified_exploratory_unranked_post_freeze_addendum"
        or not isinstance(identity, Mapping)
        or identity.get("requested_model_id") != "qwen3.8-max"
        or identity.get("identity_kind") != "mutable_alias"
        or identity.get("frozen_release") is not False
        or not isinstance(run, Mapping)
        or run.get("completed_off_on_pairs") != 1
        or run.get("delivered_response_arms") != 2
        or run.get("real_epicure_calls") != 2
        or run.get("successful_real_epicure_calls") != 2
        or not isinstance(boundary, Mapping)
        or boundary.get("official") is not False
        or boundary.get("rank_eligible") is not False
        or boundary.get("quality_judgments") != 0
        or boundary.get("leaderboard_comparisons_authorized") != 0
    ):
        raise EpicureOperationalLeaderboardError("Qwen extension boundary drifted")
    return qwen


def _score_key(row: Mapping[str, Any]) -> tuple[float, float]:
    return (
        float(row["pair_completion_wilson_lower_95"]),
        float(row["pair_completion_rate"]),
    )


def _published_row(
    source_row: Mapping[str, Any],
    *,
    rank: int,
    display_order: int,
) -> dict[str, Any]:
    scheduled = int(source_row["scheduled_pairs"])
    complete = int(source_row["complete_pairs"])
    successes = int(source_row["epicure_successful_calls"])
    calls = int(source_row["epicure_calls"])
    if (
        scheduled < 8
        or not 0 <= complete <= scheduled
        or not 0 <= successes <= calls
        or source_row.get("quality_judgments") != 0
        or source_row.get("rank_eligible") is not False
    ):
        raise EpicureOperationalLeaderboardError("invalid source model row")
    return {
        "operational_rank": rank,
        "display_order": display_order,
        "model_id": str(source_row["model_id"]),
        "display_name": str(source_row["display_name"]),
        "canonical_model_slug": str(source_row["canonical_model_slug"]),
        "execution_backend": str(source_row["execution_backend"]),
        "provider_tag": str(source_row["provider_tag"]),
        "scheduled_pairs": scheduled,
        "verified_complete_pairs": complete,
        "verified_pair_completion_rate": float(source_row["pair_completion_rate"]),
        "verified_pair_completion_wilson_lower_95": float(
            source_row["pair_completion_wilson_lower_95"]
        ),
        "verified_pair_completion_wilson_upper_95": float(
            source_row["pair_completion_wilson_upper_95"]
        ),
        "epicure_calls": calls,
        "epicure_successful_calls": successes,
        "epicure_tool_call_success_rate": float(source_row["tool_call_success_rate"]),
        "completed_response_arms": int(source_row["completed_arms_for_latency"]),
        "latency_median_seconds": float(source_row["latency_median_s"]),
        "evidence_tier": "qualified" if complete >= 8 else "provisional",
        "automated_operational_rank_eligible": True,
        "culinary_quality_rank_eligible": False,
        "quality_judgments": 0,
        "cost_display_status": str(source_row["cost_display_status"]),
        "known_conservative_cost_exposure_usd": source_row["conservative_cost_exposure_usd"],
    }


def build_operational_leaderboard(source_path: Path, qwen_path: Path) -> dict[str, Any]:
    """Verify frozen inputs and return one content-addressed public artifact."""

    source = _validate_source(source_path)
    qwen = _validate_qwen(qwen_path)
    source_rows = source["model_rows"]
    if not isinstance(source_rows, list):
        raise EpicureOperationalLeaderboardError("source model rows are missing")
    ordered = sorted(
        source_rows,
        key=lambda row: (
            -_score_key(row)[0],
            -_score_key(row)[1],
            str(row["model_id"]),
        ),
    )
    dense_rank = 0
    previous_score: tuple[float, float] | None = None
    rows: list[dict[str, Any]] = []
    for display_order, row in enumerate(ordered, start=1):
        score = _score_key(row)
        if score != previous_score:
            dense_rank += 1
            previous_score = score
        rows.append(_published_row(row, rank=dense_rank, display_order=display_order))

    qwen_identity = qwen["model_identity"]
    qwen_run = qwen["successor_operational_run"]
    assert isinstance(qwen_identity, Mapping)
    assert isinstance(qwen_run, Mapping)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "public_release_candidate",
        "observed_through": "2026-08-08",
        "leaderboard_scope": "epicure_grounded_automated_operational",
        "official_within_scope": True,
        "source": {
            "semantic_sha256": EXPECTED_SOURCE_SEMANTIC_SHA256,
            "physical_sha256": EXPECTED_SOURCE_PHYSICAL_SHA256,
            "execution_policy_sha256": EXPECTED_EXECUTION_POLICY_SHA256,
            "task_set_sha256": EXPECTED_TASK_SET_SHA256,
            "protocol_stratum": "high_resource",
            "task_source": source["task_source"],
        },
        "estimand": {
            "name": "verified_matched_pair_completion",
            "unit": "scheduled_real_task_epicure_off_on_pair",
            "success_condition": (
                "Both frozen-protocol response arms complete; the Epicure-off arm makes no "
                "tool call; the Epicure-on arm completes at least one successful real "
                "Epicure call; all source and response content addresses verify."
            ),
            "primary_order": (
                "Descending two-sided 95% Wilson lower confidence bound for verified pair "
                "completion, then point estimate; exact model ID orders evidence ties only."
            ),
            "tie_policy": "dense_rank_on_identical_lower_bound_and_point_estimate",
            "interpretation": (
                "Operational reliability under the frozen Epicure-assisted protocol; higher "
                "is better."
            ),
            "not_an_estimand": [
                "culinary answer quality",
                "human preference",
                "Epicure uplift",
                "general model intelligence",
            ],
        },
        "eligibility": {
            "minimum_scheduled_pairs": 8,
            "qualified_evidence_minimum_complete_pairs": 8,
            "qualified_and_provisional_rows_are_ranked": True,
            "protocol_pooling": False,
            "synthetic_tasks_allowed": False,
        },
        "totals": {
            "ranked_models": len(rows),
            "qualified_models": sum(row["evidence_tier"] == "qualified" for row in rows),
            "provisional_models": sum(row["evidence_tier"] == "provisional" for row in rows),
            **dict(source["totals"]),
        },
        "rows": rows,
        "unranked_extensions": [
            {
                "model_id": str(qwen_identity["requested_model_id"]),
                "display_name": str(qwen_identity["display_name"]),
                "identity_kind": str(qwen_identity["identity_kind"]),
                "completed_pairs": int(qwen_run["completed_off_on_pairs"]),
                "delivered_response_arms": int(qwen_run["delivered_response_arms"]),
                "epicure_calls": int(qwen_run["real_epicure_calls"]),
                "epicure_successful_calls": int(qwen_run["successful_real_epicure_calls"]),
                "status": "insufficient_comparable_evidence",
                "rank": None,
                "reason": (
                    "One post-freeze exploratory pair used a different protocol/runtime and a "
                    "mutable provider alias; it is displayed but not pooled or ranked."
                ),
                "source_semantic_sha256": EXPECTED_QWEN_SEMANTIC_SHA256,
                "source_physical_sha256": EXPECTED_QWEN_PHYSICAL_SHA256,
            }
        ],
        "claim_boundary": {
            "automated_operational_leaderboard_official": True,
            "culinary_quality_leaderboard_official": False,
            "human_preference_leaderboard_official": False,
            "epicure_uplift_leaderboard_official": False,
            "quality_judgments": 0,
            "epicure_is_ground_truth_for": [
                "successful tool execution",
                "frozen protocol compliance",
                "content-addressed pair completion",
            ],
            "epicure_is_not_claimed_as_ground_truth_for": [
                "subjective taste",
                "culinary usefulness",
                "human preference",
            ],
        },
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact


def render_artifact(artifact: Mapping[str, Any]) -> bytes:
    """Return stable pretty JSON while preserving the semantic content address."""

    return (json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _latex(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def render_latex_table(artifact: Mapping[str, Any]) -> str:
    """Render the ranked panel with a plain 0--100 FlavourBench score."""

    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 16:
        raise EpicureOperationalLeaderboardError("operational rows are missing")
    lines = [
        r"\begin{tabular}{@{}rlrrrr@{}}",
        r"\toprule",
        r"Rank & Model & Score & Completed & Completion [95\% CI] & Successful calls \\",
        r"\midrule",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            raise EpicureOperationalLeaderboardError("invalid operational row")
        rate = 100 * float(row["verified_pair_completion_rate"])
        lower = 100 * float(row["verified_pair_completion_wilson_lower_95"])
        upper = 100 * float(row["verified_pair_completion_wilson_upper_95"])
        lines.append(
            f"{int(row['operational_rank'])} & {_latex(str(row['display_name']))} & "
            f"{lower:.1f} & "
            f"{int(row['verified_complete_pairs'])}/{int(row['scheduled_pairs'])} & "
            f"{rate:.1f} [{lower:.1f}, {upper:.1f}] & "
            f"{int(row['epicure_successful_calls'])}/{int(row['epicure_calls'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--check-artifact", type=Path)
    parser.add_argument("--check-table", type=Path)
    parser.add_argument("--latex-table", action="store_true")
    args = parser.parse_args(argv)
    artifact = build_operational_leaderboard(args.source, args.qwen)
    output = render_artifact(artifact)
    if args.check_artifact is not None:
        if args.check_artifact.read_bytes() != output:
            raise EpicureOperationalLeaderboardError("materialized artifact is stale")
    if args.check_table is not None:
        if args.check_table.read_text(encoding="utf-8") != render_latex_table(artifact):
            raise EpicureOperationalLeaderboardError("materialized LaTeX table is stale")
    if args.check_artifact is not None or args.check_table is not None:
        print(artifact["artifact_sha256"])
        return 0
    # Stdout-only by design; publication uses a separately reviewed file edit.
    import sys

    if args.latex_table:
        sys.stdout.write(render_latex_table(artifact))
    else:
        sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
