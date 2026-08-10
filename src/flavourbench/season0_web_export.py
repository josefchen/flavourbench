"""Build a raw-text-free web visualization payload from the Season 0 analysis."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-season0-web-results-v1"


class WebExportError(RuntimeError):
    """The analysis cannot support a safe aggregate web export."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise WebExportError("analysis is not a JSON object")
    return value


def _verify(document: Mapping[str, Any], label: str) -> str:
    claimed = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise WebExportError(f"{label} artifact hash mismatch")
    return actual


def _model_family(canonical_model_id: str) -> str:
    owner = re.split(r"[/.]", canonical_model_id, maxsplit=1)[0].lower()
    return {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "qwen": "Alibaba Qwen",
        "mistral": "Mistral",
        "amazon": "Amazon",
        "minimax": "MiniMax",
        "google": "Google",
    }.get(owner, owner.title())


def _model_catalog(model_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for model in model_manifest.get("models", []):
        if not isinstance(model, Mapping):
            raise WebExportError("model manifest contains a malformed entry")
        endpoint = model.get("endpoint")
        endpoint = endpoint if isinstance(endpoint, Mapping) else {}
        canonical = str(model.get("canonical_model_id") or "")
        output.append(
            {
                "id": model["season_model_id"],
                "canonicalSlug": model.get("canonical_slug") or canonical,
                "name": model["display_name"],
                "family": _model_family(canonical),
                "openWeight": model.get("slot_role") == "open_weight",
                "status": "season_eligible",
                "supportsTools": True,
                "supportsStructuredOutput": True,
                "contextLength": endpoint.get("context_length"),
                "slotRole": model["slot_role"],
                "provider": model["provider"],
                "providerName": model.get("provider_name"),
                "requestedEndpointId": model["requested_endpoint_id"],
                "compatibilityArtifactSha256": model["compatibility_artifact_sha256"],
            }
        )
    if not output:
        raise WebExportError("model manifest has no frozen systems")
    return output


def _rows(
    arena_rows: Sequence[Mapping[str, Any]],
    uplift_rows: Sequence[Mapping[str, Any]],
    operational: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    uplift_by_id = {str(row["season_model_id"]): row for row in uplift_rows}
    output = []
    for arena in arena_rows:
        model_id = str(arena["season_model_id"])
        uplift = uplift_by_id[model_id]
        metrics = operational[model_id]
        judgments = int(arena.get("judgments") or 0)
        output.append(
            {
                "season_model_id": model_id,
                "competitor_id": arena["display_name"],
                "provider": metrics["provider"],
                "rating": arena.get("rating"),
                "rating_lower": arena.get("rating_lower"),
                "rating_upper": arena.get("rating_upper"),
                "battles": int(arena.get("comparisons") or 0),
                "model_provisional": int(arena.get("comparisons") or 0) < 100,
                "epicure_win_share": uplift.get("epicure_win_share"),
                "interval_lower": uplift.get("interval_lower"),
                "interval_upper": uplift.get("interval_upper"),
                "epicure_wins": int(uplift.get("epicure_wins") or 0),
                "unaided_wins": int(uplift.get("unaided_wins") or 0),
                "ties": int(uplift.get("ties") or 0),
                "uplift_pairs": int(uplift.get("comparisons") or 0),
                "uplift_provisional": int(uplift.get("comparisons") or 0) < 50,
                "both_bad": int(arena.get("both_bad") or 0),
                "both_bad_rate": int(arena.get("both_bad") or 0) / judgments if judgments else None,
                "average_cost_micros": round(
                    float(metrics.get("mean_arm_cost_usd") or 0) * 1_000_000
                ),
                "average_latency_ms": metrics.get("latency_median_ms"),
                "latency_p95_ms": metrics.get("latency_p95_ms"),
                "invalid_response_rate": metrics.get("invalid_response_rate"),
                "end_to_end_failure_rate": metrics.get("end_to_end_failure_rate"),
                "provider_route_failure_rate": metrics.get("provider_route_failure_rate"),
                "identity_leak_rate": metrics.get("identity_leak_rate"),
                "tool_success_rate": metrics.get("tool_success_rate"),
                "epicure_on_tool_use_rate": metrics.get("epicure_on_tool_use_rate"),
                "response_arms": int(metrics.get("arms") or 0),
                "answer_words_median": metrics.get("answer_words_median"),
            }
        )
    return output


def build_export(
    analysis: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    judgment_summary: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    analysis_sha = _verify(analysis, "analysis")
    model_manifest_sha = _verify(model_manifest, "model manifest")
    judgment_summary_sha = _verify(judgment_summary, "judgment summary")
    if analysis.get("model_manifest_artifact_sha256") != model_manifest_sha:
        raise WebExportError("analysis is bound to another model manifest")
    if analysis.get("synthetic_arms") != 0 or analysis.get("synthetic_judgments") != 0:
        raise WebExportError("web release refuses non-provider observations")
    judgment_counts = judgment_summary.get("counts")
    if (
        judgment_summary.get("status") != "collection_complete"
        or judgment_summary.get("synthetic_judgments") != 0
        or judgment_summary.get("comparison_manifest_artifact_sha256")
        != analysis.get("comparison_manifest_artifact_sha256")
        or judgment_summary.get("judge_manifest_artifact_sha256")
        != analysis.get("judge_manifest_artifact_sha256")
        or not isinstance(judgment_counts, Mapping)
        or int(judgment_counts.get("terminal_judgments") or 0)
        != int(analysis["counts"]["judgment_records"])
    ):
        raise WebExportError("judgment summary is incomplete or not bound to the analysis")
    operational = analysis.get("operational_metrics")
    if not isinstance(operational, Mapping):
        raise WebExportError("analysis has no operational metrics")
    global_rows = _rows(analysis["model_leaderboard"], analysis["uplift_leaderboard"], operational)
    family_rows = {
        family: _rows(
            analysis["model_leaderboard_by_family"][family],
            analysis["uplift_leaderboard_by_family"][family],
            operational,
        )
        for family in ("substitution", "composition", "cookability", "evidence")
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "season": "Season 0",
        "cohort": "Automated · 4 judges · swap-controlled · non-self consensus",
        "status": "frozen_automated_cohort",
        "analysis_artifact_sha256": analysis_sha,
        "task_bank_artifact_sha256": analysis["task_bank_artifact_sha256"],
        "model_manifest_artifact_sha256": analysis["model_manifest_artifact_sha256"],
        "comparison_manifest_artifact_sha256": analysis["comparison_manifest_artifact_sha256"],
        "judge_manifest_artifact_sha256": analysis["judge_manifest_artifact_sha256"],
        "judgment_summary_artifact_sha256": judgment_summary_sha,
        "target_cost_audit_artifact_sha256": analysis["target_cost_audit_artifact_sha256"],
        "counts": analysis["counts"],
        "judging": {
            "terminal_judgment_identities": int(judgment_counts["terminal_judgments"]),
            "provider_attempt_records": int(
                judgment_counts.get("provider_attempt_records")
                or judgment_counts["terminal_judgments"]
            ),
            "successful_judgments": int(judgment_counts["success"]),
            "failed_judgments": int(judgment_counts["failed"]),
            "first_pass_documented_throttle_rejections": int(
                judgment_counts.get("first_pass_documented_throttle_rejections") or 0
            ),
            "recovery_attempts": int(judgment_counts.get("recovery_attempts") or 0),
            "recovered_to_success": int(judgment_counts.get("recovered_to_success") or 0),
            "recovery_failures": int(judgment_counts.get("recovery_failures") or 0),
            "estimated_conservative_cost_usd": float(judgment_summary["estimated_cost_usd"]),
        },
        "models": _model_catalog(model_manifest),
        "rows": global_rows,
        "rows_by_family": family_rows,
        "panel_uplift": analysis["panel_uplift"],
        "panel_uplift_dimensions": analysis.get("panel_uplift_dimensions", []),
        "judge_diagnostics": analysis["judge_diagnostics"],
        "judge_family_balanced_sensitivity": analysis["judge_family_balanced_sensitivity"],
        "arena_graph_diagnostics": analysis["arena_graph_diagnostics"],
        "arena_task_cluster_bootstrap": analysis["arena_task_cluster_bootstrap"],
        "reference_overlap_audit": analysis["reference_overlap_audit"],
        "verbosity_diagnostics": analysis["verbosity_diagnostics"],
        "methods": analysis["methods"],
        "privacy": {
            "contains_prompts": False,
            "contains_answers": False,
            "contains_human_references": False,
            "contains_judge_rationales": False,
            "contains_personal_data": False,
        },
    }
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"season0-web-results-{digest}.json"
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
    return {**document, "output_path": str(destination)}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--judgment-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_export(
        _load(args.analysis),
        _load(args.model_manifest),
        _load(args.judgment_summary),
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": result["artifact_sha256"],
                "rows": len(result["rows"]),
                "output_path": result["output_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
