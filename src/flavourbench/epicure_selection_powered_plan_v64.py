"""Freeze joint inference before inspecting the GMICloud DeepSeek blocks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v53 import _sha256, _sha256_file
from .epicure_selection_powered_plan_v60 import (
    PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V60,
)
from .epicure_selection_powered_plan_v60 import PLAN_VERSION as PLAN_VERSION_V60
from .epicure_selection_powered_plan_v60 import verify_plan as verify_plan_v60
from .epicure_selection_powered_plan_v62 import verify_plan as verify_plan_v62
from .epicure_selection_powered_plan_v63 import verify_plan as verify_plan_v63
from .epicure_selection_route_manifest_v61 import DEEPSEEK_PRO_MODEL_ID, ROUTE_TAG

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v64"
PLAN_VERSION = "flavourbench-selection-26x1280-two-panel-deepseek-gmicloud-block-v64"


class SelectionPoweredPlanV64Error(RuntimeError):
    """The final GMICloud joint source plan failed verification."""


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _sha256_file(path),
    }


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV64Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV64Error("joint-plan input is not a JSON object")
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
        not verify_plan_v60(predecessor)
        or not verify_plan_v62(panel_1_replacement_plan)
        or not verify_plan_v63(panel_2_replacement_plan)
    ):
        raise SelectionPoweredPlanV64Error("v64 requires exact v60, v62, and v63 inputs")
    before_ids = [str(row["model_id"]) for row in predecessor["roster"]["models"]]
    panel_1_ids = [str(row["model_id"]) for row in panel_1_replacement_plan["roster"]["models"]]
    panel_2_ids = [str(row["model_id"]) for row in panel_2_replacement_plan["roster"]["models"]]
    if before_ids != panel_1_ids or before_ids != panel_2_ids:
        raise SelectionPoweredPlanV64Error("GMICloud repair changed model identities")

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "joint_analysis_frozen_before_gmicloud_quality_inspection"
    document["inputs"]["joint_plan_v60_predecessor"] = _pin(predecessor, predecessor_path)
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
            "panel_1_uses_deepseek_gmicloud_complete_block": True,
            "panel_2_uses_deepseek_gmicloud_complete_block": True,
            "deepseek_gmicloud_complete_block_model_ids": [DEEPSEEK_PRO_MODEL_ID],
            "deepseek_gmicloud_exact_route": ROUTE_TAG,
            "gmicloud_selected_without_quality_scores_or_selections": True,
            "superseded_panel_1_plan_before_gmicloud": prior_panel_1,
            "superseded_panel_2_plan_before_gmicloud": prior_panel_2,
            "superseded_baseten_responses_used": False,
            "gmicloud_cross_route_response_pooling": False,
            "gmicloud_selective_failed_cell_retry": False,
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV64Error("constructed v64 joint plan failed verification")
    return document


def _as_v60(document: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = copy.deepcopy(document)
    predecessor.pop("artifact_sha256", None)
    predecessor["schema_version"] = PLAN_SCHEMA_VERSION_V60
    predecessor["plan_version"] = PLAN_VERSION_V60
    predecessor["status"] = "joint_analysis_frozen_before_deepseek_repair_quality_inspection"
    predecessor["inputs"].pop("joint_plan_v60_predecessor", None)
    source = predecessor["source_rules"]
    predecessor["inputs"]["panel_1_plan"] = source.pop("superseded_panel_1_plan_before_gmicloud")
    predecessor["inputs"]["panel_2_plan"] = source.pop("superseded_panel_2_plan_before_gmicloud")
    for key in (
        "panel_1_uses_deepseek_gmicloud_complete_block",
        "panel_2_uses_deepseek_gmicloud_complete_block",
        "deepseek_gmicloud_complete_block_model_ids",
        "deepseek_gmicloud_exact_route",
        "gmicloud_selected_without_quality_scores_or_selections",
        "superseded_baseten_responses_used",
        "gmicloud_cross_route_response_pooling",
        "gmicloud_selective_failed_cell_retry",
    ):
        source.pop(key, None)
    predecessor["artifact_sha256"] = _sha256(predecessor)
    return predecessor


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        source = document["source_rules"]
        predecessor = document["inputs"]["joint_plan_v60_predecessor"]
        panel_1 = document["inputs"]["panel_1_plan"]
        panel_2 = document["inputs"]["panel_2_plan"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status") == "joint_analysis_frozen_before_gmicloud_quality_inspection"
        and verify_plan_v60(_as_v60(document))
        and source.get("panel_1_uses_deepseek_gmicloud_complete_block") is True
        and source.get("panel_2_uses_deepseek_gmicloud_complete_block") is True
        and source.get("deepseek_gmicloud_complete_block_model_ids") == [DEEPSEEK_PRO_MODEL_ID]
        and source.get("deepseek_gmicloud_exact_route") == ROUTE_TAG
        and source.get("gmicloud_selected_without_quality_scores_or_selections") is True
        and source.get("superseded_baseten_responses_used") is False
        and source.get("gmicloud_cross_route_response_pooling") is False
        and source.get("gmicloud_selective_failed_cell_retry") is False
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
            raise SelectionPoweredPlanV64Error("content-addressed joint-plan conflict")
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
