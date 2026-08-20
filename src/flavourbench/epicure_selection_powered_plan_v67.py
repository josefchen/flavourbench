"""Freeze joint 27-model inference before observing the GLM-5.3 blocks."""

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
from .epicure_selection_powered_plan_v64 import PLAN_SCHEMA_VERSION as PREDECESSOR_SCHEMA
from .epicure_selection_powered_plan_v64 import PLAN_VERSION as PREDECESSOR_VERSION
from .epicure_selection_powered_plan_v64 import verify_plan as verify_plan_v64
from .epicure_selection_powered_plan_v65 import verify_plan as verify_plan_v65
from .epicure_selection_powered_plan_v66 import verify_plan as verify_plan_v66

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v67"
PLAN_VERSION = "flavourbench-selection-27x1280-two-panel-glm53-limited-run-v67"


class SelectionPoweredPlanV67Error(RuntimeError):
    """The 27-model joint plan failed verification."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV67Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV67Error("joint-plan input is not a JSON object")
    return value


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _sha256_file(path),
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    panel_1_plan: Mapping[str, Any],
    panel_1_plan_path: Path,
    panel_2_plan: Mapping[str, Any],
    panel_2_plan_path: Path,
) -> dict[str, Any]:
    if (
        not verify_plan_v64(predecessor)
        or not verify_plan_v65(panel_1_plan)
        or not verify_plan_v66(panel_2_plan)
    ):
        raise SelectionPoweredPlanV67Error("v67 requires exact v64, v65, and v66 plans")
    before = predecessor["roster"]["models"]
    panel_1 = panel_1_plan["roster"]["models"]
    panel_2 = panel_2_plan["roster"]["models"]
    if panel_1[-1] != panel_2[-1] or panel_1[-1]["model_id"] != "z-ai/glm-5.3" or len(before) != 26:
        raise SelectionPoweredPlanV67Error("additive GLM-5.3 roster lineage differs")

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "joint_analysis_frozen_before_glm53_quality_inspection"
    document["inputs"]["joint_plan_v64_predecessor"] = _pin(predecessor, predecessor_path)
    prior_panel_1 = copy.deepcopy(document["inputs"]["panel_1_plan"])
    prior_panel_2 = copy.deepcopy(document["inputs"]["panel_2_plan"])
    document["inputs"]["panel_1_plan"] = _pin(panel_1_plan, panel_1_plan_path)
    document["inputs"]["panel_2_plan"] = _pin(panel_2_plan, panel_2_plan_path)
    document["roster"]["model_count"] = 27
    document["roster"]["pairwise_hypotheses"] = 351
    document["roster"]["models"] = copy.deepcopy(before) + [copy.deepcopy(panel_1[-1])]
    document["design"]["primary_model_task_cells"] = 34_560
    document["design"]["repeat_model_task_cells"] = 3_456
    document["power"]["familywise_comparisons"] = 351
    document["source_rules"].update(
        {
            "panel_1_uses_complete_glm53_limited_run_block": True,
            "panel_2_uses_complete_glm53_limited_run_block": True,
            "glm53_limited_run_model_ids": ["z-ai/glm-5.3"],
            "glm53_requested_and_returned_model_id": "glm-5.3",
            "glm53_finite_cli_only": True,
            "glm53_standing_service": False,
            "glm53_automatic_fallback": False,
            "glm53_included_without_quality_scores_or_selections": True,
            "superseded_panel_1_plan_before_glm53": prior_panel_1,
            "superseded_panel_2_plan_before_glm53": prior_panel_2,
            "superseded_joint_counts_before_glm53": {
                "model_count": 26,
                "pairwise_hypotheses": 325,
                "primary_model_task_cells": 33_280,
                "repeat_model_task_cells": 3_328,
                "familywise_comparisons": 325,
            },
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV67Error("constructed v67 joint plan failed verification")
    return document


def _as_v64(document: Mapping[str, Any]) -> dict[str, Any]:
    prior = copy.deepcopy(document)
    prior.pop("artifact_sha256", None)
    prior["schema_version"] = PREDECESSOR_SCHEMA
    prior["plan_version"] = PREDECESSOR_VERSION
    prior["status"] = "joint_analysis_frozen_before_gmicloud_quality_inspection"
    prior["inputs"].pop("joint_plan_v64_predecessor")
    source = prior["source_rules"]
    prior["inputs"]["panel_1_plan"] = source.pop("superseded_panel_1_plan_before_glm53")
    prior["inputs"]["panel_2_plan"] = source.pop("superseded_panel_2_plan_before_glm53")
    counts = source.pop("superseded_joint_counts_before_glm53")
    prior["roster"]["model_count"] = counts["model_count"]
    prior["roster"]["pairwise_hypotheses"] = counts["pairwise_hypotheses"]
    prior["roster"]["models"] = prior["roster"]["models"][:-1]
    prior["design"]["primary_model_task_cells"] = counts["primary_model_task_cells"]
    prior["design"]["repeat_model_task_cells"] = counts["repeat_model_task_cells"]
    prior["power"]["familywise_comparisons"] = counts["familywise_comparisons"]
    for key in (
        "panel_1_uses_complete_glm53_limited_run_block",
        "panel_2_uses_complete_glm53_limited_run_block",
        "glm53_limited_run_model_ids",
        "glm53_requested_and_returned_model_id",
        "glm53_finite_cli_only",
        "glm53_standing_service",
        "glm53_automatic_fallback",
        "glm53_included_without_quality_scores_or_selections",
    ):
        source.pop(key)
    prior["artifact_sha256"] = _sha256(prior)
    return prior


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        source = document["source_rules"]
        predecessor = document["inputs"]["joint_plan_v64_predecessor"]
        panel_1 = document["inputs"]["panel_1_plan"]
        panel_2 = document["inputs"]["panel_2_plan"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and document.get("status") == "joint_analysis_frozen_before_glm53_quality_inspection"
        and recorded == _sha256(payload)
        and verify_plan_v64(_as_v64(document))
        and document["roster"].get("model_count") == 27
        and document["roster"].get("pairwise_hypotheses") == 351
        and len(document["roster"].get("models") or []) == 27
        and document["roster"]["models"][-1].get("model_id") == "z-ai/glm-5.3"
        and document["design"].get("primary_model_task_cells") == 34_560
        and document["design"].get("repeat_model_task_cells") == 3_456
        and document["power"].get("familywise_comparisons") == 351
        and source.get("panel_1_uses_complete_glm53_limited_run_block") is True
        and source.get("panel_2_uses_complete_glm53_limited_run_block") is True
        and source.get("glm53_limited_run_model_ids") == ["z-ai/glm-5.3"]
        and source.get("glm53_finite_cli_only") is True
        and source.get("glm53_standing_service") is False
        and source.get("glm53_automatic_fallback") is False
        and source.get("glm53_included_without_quality_scores_or_selections") is True
        and all(
            isinstance(pin.get("semantic_sha256"), str)
            and len(pin["semantic_sha256"]) == 64
            and isinstance(pin.get("physical_sha256"), str)
            and len(pin["physical_sha256"]) == 64
            for pin in (predecessor, panel_1, panel_2)
        )
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = (
        directory / f"epicure-selection-joint-analysis-plan-{document['artifact_sha256']}.json"
    )
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV67Error("content-addressed joint-plan conflict")
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
    parser.add_argument("--panel-1-plan", type=Path, required=True)
    parser.add_argument("--panel-2-plan", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        _write(
            build_plan(
                predecessor=_load(args.predecessor),
                predecessor_path=args.predecessor,
                panel_1_plan=_load(args.panel_1_plan),
                panel_1_plan_path=args.panel_1_plan,
                panel_2_plan=_load(args.panel_2_plan),
                panel_2_plan_path=args.panel_2_plan,
            ),
            args.output_directory,
        )
    )


if __name__ == "__main__":
    run()
