"""Freeze a complete-case 26-model leaderboard before final quality analysis."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v54 import _sha256, _sha256_file
from .epicure_selection_powered_plan_v77 import PLAN_SCHEMA_VERSION as PREDECESSOR_SCHEMA
from .epicure_selection_powered_plan_v77 import PLAN_VERSION as PREDECESSOR_VERSION
from .epicure_selection_powered_plan_v77 import verify_plan as verify_plan_v77
from .epicure_selection_route_manifest_v45 import FABLE_MODEL_ID

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v78"
PLAN_VERSION = "flavourbench-selection-26x1280-complete-case-ranking-v78"
PRIMARY_TASKS = 1_280
REPEAT_TASKS = 128
RANKED_MODEL_COUNT = 26
PAIRWISE_HYPOTHESES = RANKED_MODEL_COUNT * (RANKED_MODEL_COUNT - 1) // 2


class SelectionPoweredPlanV78Error(RuntimeError):
    """The complete-case public-leaderboard plan failed verification."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV78Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV78Error("joint-plan input is not a JSON object")
    return value


def build_plan(*, predecessor: Mapping[str, Any], predecessor_path: Path) -> dict[str, Any]:
    if not verify_plan_v77(predecessor):
        raise SelectionPoweredPlanV78Error("v78 requires the exact v77 predecessor")
    model_ids = [str(row["model_id"]) for row in predecessor["roster"]["models"]]
    if len(model_ids) != 27 or model_ids.count(FABLE_MODEL_ID) != 1:
        raise SelectionPoweredPlanV78Error("v78 requires one Fable row in the 27-model roster")
    ranked_model_ids = [model_id for model_id in model_ids if model_id != FABLE_MODEL_ID]

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "rank_eligibility_frozen_after_coverage_before_quality_analysis"
    document["inputs"]["joint_plan_v77_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": _sha256_file(predecessor_path),
    }
    document["roster"]["ranked_model_count"] = RANKED_MODEL_COUNT
    document["roster"]["pairwise_hypotheses"] = PAIRWISE_HYPOTHESES
    document["eligibility"] = {
        "schema_version": "flavourbench-complete-case-rank-eligibility-v1",
        "ranked_model_ids": ranked_model_ids,
        "coverage_diagnostic_model_ids": [FABLE_MODEL_ID],
        "minimum_valid_primary_tasks": PRIMARY_TASKS,
        "minimum_valid_repeat_tasks": REPEAT_TASKS,
        "requires_every_scheduled_primary_and_repeat_cell": True,
        "failed_content_filtered_or_unparseable_cells_ranked_as_zero": False,
        "incomplete_models_emitted_as_rank_rows": False,
        "dnf_rows_emitted": False,
        "coverage_diagnostic_reports_quality_score": False,
        "eligibility_uses_status_finish_and_parseability_only": True,
        "quality_scores_or_model_selections_used": False,
        "fable_exclusion_reason": (
            "systematic model refusals across independently frozen Anthropic, AWS-backed, "
            "and Google-backed route blocks prevent complete-case ranking"
        ),
        "excluded_response_artifacts_preserved": True,
    }
    document["source_rules"].update(
        {
            "ranked_response_set_complete_case_only": True,
            "coverage_diagnostic_model_ids": [FABLE_MODEL_ID],
            "coverage_decision_uses_quality_scores_or_selections": False,
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV78Error("constructed v78 joint plan failed verification")
    return document


def _as_v77(document: Mapping[str, Any]) -> dict[str, Any]:
    prior = copy.deepcopy(document)
    prior.pop("artifact_sha256", None)
    prior["schema_version"] = PREDECESSOR_SCHEMA
    prior["plan_version"] = PREDECESSOR_VERSION
    prior["status"] = "joint_source_lineage_frozen_after_coverage_before_quality_analysis"
    prior["inputs"].pop("joint_plan_v77_predecessor")
    prior["roster"].pop("ranked_model_count")
    prior["roster"]["pairwise_hypotheses"] = 351
    prior.pop("eligibility")
    for key in (
        "ranked_response_set_complete_case_only",
        "coverage_diagnostic_model_ids",
        "coverage_decision_uses_quality_scores_or_selections",
    ):
        prior["source_rules"].pop(key)
    prior["artifact_sha256"] = _sha256(prior)
    return prior


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        eligibility = document["eligibility"]
        roster = document["roster"]
        ranked = eligibility["ranked_model_ids"]
        diagnostic = eligibility["coverage_diagnostic_model_ids"]
        predecessor = document["inputs"]["joint_plan_v77_predecessor"]
    except (KeyError, TypeError):
        return False
    roster_ids = [str(row["model_id"]) for row in roster.get("models") or []]
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and document.get("status")
        == "rank_eligibility_frozen_after_coverage_before_quality_analysis"
        and recorded == _sha256(payload)
        and verify_plan_v77(_as_v77(document))
        and roster.get("model_count") == 27
        and roster.get("ranked_model_count") == RANKED_MODEL_COUNT
        and roster.get("pairwise_hypotheses") == PAIRWISE_HYPOTHESES
        and len(ranked) == RANKED_MODEL_COUNT
        and len(set(ranked)) == RANKED_MODEL_COUNT
        and ranked == [model_id for model_id in roster_ids if model_id != FABLE_MODEL_ID]
        and diagnostic == [FABLE_MODEL_ID]
        and eligibility.get("minimum_valid_primary_tasks") == PRIMARY_TASKS
        and eligibility.get("minimum_valid_repeat_tasks") == REPEAT_TASKS
        and eligibility.get("requires_every_scheduled_primary_and_repeat_cell") is True
        and eligibility.get("failed_content_filtered_or_unparseable_cells_ranked_as_zero") is False
        and eligibility.get("incomplete_models_emitted_as_rank_rows") is False
        and eligibility.get("dnf_rows_emitted") is False
        and eligibility.get("coverage_diagnostic_reports_quality_score") is False
        and eligibility.get("eligibility_uses_status_finish_and_parseability_only") is True
        and eligibility.get("quality_scores_or_model_selections_used") is False
        and eligibility.get("excluded_response_artifacts_preserved") is True
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and isinstance(predecessor.get("physical_sha256"), str)
        and len(predecessor["physical_sha256"]) == 64
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = (
        directory / f"epicure-selection-joint-analysis-plan-{document['artifact_sha256']}.json"
    )
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV78Error("content-addressed joint-plan conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        _write(
            build_plan(predecessor=_load(args.predecessor), predecessor_path=args.predecessor),
            args.output_directory,
        )
    )


if __name__ == "__main__":
    run()
