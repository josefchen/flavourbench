"""Freeze final two-panel inference with complete coverage-repair sources."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v53 import (
    PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V53,
)
from .epicure_selection_powered_plan_v53 import PLAN_VERSION as PLAN_VERSION_V53
from .epicure_selection_powered_plan_v53 import _sha256, _sha256_file
from .epicure_selection_powered_plan_v53 import verify_plan as verify_plan_v53
from .epicure_selection_powered_plan_v54 import verify_plan as verify_plan_v54
from .epicure_selection_powered_plan_v55 import verify_plan as verify_plan_v55
from .epicure_selection_route_manifest_v54 import REPLACEMENT_MODEL_IDS

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v56"
PLAN_VERSION = "flavourbench-selection-26x1280-two-panel-complete-coverage-repair-v56"


class SelectionPoweredPlanV56Error(RuntimeError):
    """The final two-panel complete-coverage source plan failed verification."""


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _sha256_file(path),
    }


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV56Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV56Error("joint-plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    panel_1_replacement_plan: Mapping[str, Any],
    panel_1_replacement_plan_path: Path,
    panel_2_replacement_plan: Mapping[str, Any],
    panel_2_replacement_plan_path: Path,
) -> dict[str, Any]:
    if (
        not verify_plan_v53(predecessor)
        or not verify_plan_v55(panel_1_replacement_plan)
        or not verify_plan_v54(panel_2_replacement_plan)
    ):
        raise SelectionPoweredPlanV56Error("v56 requires exact v53, v55, and v54 inputs")
    before_ids = [str(row["model_id"]) for row in predecessor["roster"]["models"]]
    panel_1_ids = [str(row["model_id"]) for row in panel_1_replacement_plan["roster"]["models"]]
    panel_2_ids = [str(row["model_id"]) for row in panel_2_replacement_plan["roster"]["models"]]
    if before_ids != panel_1_ids or before_ids != panel_2_ids:
        raise SelectionPoweredPlanV56Error("coverage repair changed model identities")

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "joint_analysis_frozen_before_coverage_repair_quality_inspection"
    document["inputs"]["joint_plan_v53_predecessor"] = _pin(predecessor, predecessor_path)
    prior_panel_1 = copy.deepcopy(document["inputs"]["panel_1_plan"])
    prior_panel_2 = copy.deepcopy(document["inputs"]["panel_2_plan"])
    document["inputs"]["panel_1_plan"] = _pin(
        panel_1_replacement_plan, panel_1_replacement_plan_path
    )
    document["inputs"]["panel_2_plan"] = _pin(
        panel_2_replacement_plan, panel_2_replacement_plan_path
    )
    document["source_rules"].update(
        {
            "panel_1_uses_complete_coverage_repair_blocks": True,
            "panel_2_uses_complete_coverage_repair_blocks": True,
            "complete_coverage_repair_model_ids": REPLACEMENT_MODEL_IDS,
            "coverage_routes_selected_without_quality_scores_or_selections": True,
            "superseded_panel_1_plan_before_coverage_repair": prior_panel_1,
            "superseded_panel_2_plan_before_coverage_repair": prior_panel_2,
            "superseded_coverage_route_responses_used": False,
            "cross_route_response_pooling": False,
            "selective_failed_cell_retry": False,
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV56Error("constructed v56 joint plan failed verification")
    return document


def _as_v53(document: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = copy.deepcopy(document)
    predecessor.pop("artifact_sha256", None)
    predecessor["schema_version"] = PLAN_SCHEMA_VERSION_V53
    predecessor["plan_version"] = PLAN_VERSION_V53
    predecessor["status"] = "joint_analysis_frozen_before_replacement_quality_score_inspection"
    predecessor["inputs"].pop("joint_plan_v53_predecessor", None)
    source = predecessor["source_rules"]
    predecessor["inputs"]["panel_1_plan"] = source.pop(
        "superseded_panel_1_plan_before_coverage_repair"
    )
    predecessor["inputs"]["panel_2_plan"] = source.pop(
        "superseded_panel_2_plan_before_coverage_repair"
    )
    for key in (
        "panel_1_uses_complete_coverage_repair_blocks",
        "panel_2_uses_complete_coverage_repair_blocks",
        "complete_coverage_repair_model_ids",
        "coverage_routes_selected_without_quality_scores_or_selections",
        "superseded_coverage_route_responses_used",
    ):
        source.pop(key, None)
    predecessor["artifact_sha256"] = _sha256(predecessor)
    return predecessor


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        source = document["source_rules"]
        predecessor = document["inputs"]["joint_plan_v53_predecessor"]
        panel_1 = document["inputs"]["panel_1_plan"]
        panel_2 = document["inputs"]["panel_2_plan"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status")
        == "joint_analysis_frozen_before_coverage_repair_quality_inspection"
        and verify_plan_v53(_as_v53(document))
        and source.get("panel_1_uses_complete_coverage_repair_blocks") is True
        and source.get("panel_2_uses_complete_coverage_repair_blocks") is True
        and source.get("complete_coverage_repair_model_ids") == REPLACEMENT_MODEL_IDS
        and source.get("coverage_routes_selected_without_quality_scores_or_selections") is True
        and source.get("superseded_coverage_route_responses_used") is False
        and source.get("cross_route_response_pooling") is False
        and source.get("selective_failed_cell_retry") is False
        and isinstance(
            (source.get("superseded_panel_1_plan_before_coverage_repair") or {}).get(
                "semantic_sha256"
            ),
            str,
        )
        and isinstance(
            (source.get("superseded_panel_2_plan_before_coverage_repair") or {}).get(
                "semantic_sha256"
            ),
            str,
        )
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
            raise SelectionPoweredPlanV56Error("content-addressed joint-plan conflict")
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
    parser.add_argument("--panel-1-replacement-plan", type=Path, required=True)
    parser.add_argument("--panel-2-replacement-plan", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor)
    panel_1 = _load(args.panel_1_replacement_plan)
    panel_2 = _load(args.panel_2_replacement_plan)
    document = build_plan(
        predecessor=predecessor,
        predecessor_path=args.predecessor,
        panel_1_replacement_plan=panel_1,
        panel_1_replacement_plan_path=args.panel_1_replacement_plan,
        panel_2_replacement_plan=panel_2,
        panel_2_replacement_plan_path=args.panel_2_replacement_plan,
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
