"""Freeze joint inference with complete panel-2 transport-route replacements."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v51 import (
    PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V51,
)
from .epicure_selection_powered_plan_v51 import PLAN_VERSION as PLAN_VERSION_V51
from .epicure_selection_powered_plan_v51 import verify_plan as verify_plan_v51
from .epicure_selection_powered_plan_v52 import verify_plan as verify_plan_v52
from .epicure_selection_route_manifest_v52 import REPLACEMENT_MODEL_IDS

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v53"
PLAN_VERSION = "flavourbench-selection-26x1280-two-panel-route-repair-v53"


class SelectionPoweredPlanV53Error(RuntimeError):
    """The final response-blind joint-source plan failed verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _sha256_file(path),
    }


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV53Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV53Error("joint-plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    panel_2_replacement_plan: Mapping[str, Any],
    panel_2_replacement_plan_path: Path,
) -> dict[str, Any]:
    if not verify_plan_v51(predecessor) or not verify_plan_v52(panel_2_replacement_plan):
        raise SelectionPoweredPlanV53Error("v53 requires exact v51 and v52 predecessors")
    before_ids = [str(row["model_id"]) for row in predecessor["roster"]["models"]]
    after_ids = [str(row["model_id"]) for row in panel_2_replacement_plan["roster"]["models"]]
    if before_ids != after_ids:
        raise SelectionPoweredPlanV53Error("route repair changed the joint model identities")

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "joint_analysis_frozen_before_replacement_quality_score_inspection"
    document["inputs"]["joint_plan_v51_predecessor"] = _pin(predecessor, predecessor_path)
    superseded_panel_2_plan = copy.deepcopy(document["inputs"]["panel_2_plan"])
    document["inputs"]["panel_2_plan"] = _pin(
        panel_2_replacement_plan, panel_2_replacement_plan_path
    )
    document["source_rules"].update(
        {
            "panel_2_uses_complete_route_replacement_blocks": True,
            "panel_2_replacement_model_ids": REPLACEMENT_MODEL_IDS,
            "panel_2_route_replacements_selected_without_quality_scores_or_selections": True,
            "superseded_panel_2_responses_used": False,
            "superseded_panel_2_plan": superseded_panel_2_plan,
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV53Error("constructed v53 joint plan failed verification")
    return document


def _as_v51(document: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = copy.deepcopy(document)
    predecessor.pop("artifact_sha256", None)
    predecessor["schema_version"] = PLAN_SCHEMA_VERSION_V51
    predecessor["plan_version"] = PLAN_VERSION_V51
    predecessor["status"] = "joint_analysis_frozen_before_fable_replacement_or_panel_2_execution"
    predecessor["inputs"].pop("joint_plan_v51_predecessor", None)
    predecessor["inputs"]["panel_2_plan"] = predecessor["source_rules"].pop(
        "superseded_panel_2_plan"
    )
    for key in (
        "panel_2_uses_complete_route_replacement_blocks",
        "panel_2_replacement_model_ids",
        "panel_2_route_replacements_selected_without_quality_scores_or_selections",
        "superseded_panel_2_responses_used",
    ):
        predecessor["source_rules"].pop(key, None)
    predecessor["artifact_sha256"] = _sha256(predecessor)
    return predecessor


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        source = document["source_rules"]
        predecessor = document["inputs"]["joint_plan_v51_predecessor"]
        panel_2 = document["inputs"]["panel_2_plan"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status")
        == "joint_analysis_frozen_before_replacement_quality_score_inspection"
        and verify_plan_v51(_as_v51(document))
        and source.get("panel_2_uses_complete_route_replacement_blocks") is True
        and source.get("panel_2_replacement_model_ids") == REPLACEMENT_MODEL_IDS
        and source.get("panel_2_route_replacements_selected_without_quality_scores_or_selections")
        is True
        and source.get("superseded_panel_2_responses_used") is False
        and source.get("cross_route_response_pooling") is False
        and source.get("selective_failed_cell_retry") is False
        and isinstance((source.get("superseded_panel_2_plan") or {}).get("semantic_sha256"), str)
        and isinstance((source.get("superseded_panel_2_plan") or {}).get("physical_sha256"), str)
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and isinstance(predecessor.get("physical_sha256"), str)
        and len(predecessor["physical_sha256"]) == 64
        and isinstance(panel_2.get("semantic_sha256"), str)
        and len(panel_2["semantic_sha256"]) == 64
        and isinstance(panel_2.get("physical_sha256"), str)
        and len(panel_2["physical_sha256"]) == 64
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = (
        directory / f"epicure-selection-joint-analysis-plan-{document['artifact_sha256']}.json"
    )
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV53Error("content-addressed joint-plan conflict")
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
    parser.add_argument("--panel-2-replacement-plan", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor)
    panel_2 = _load(args.panel_2_replacement_plan)
    document = build_plan(
        predecessor=predecessor,
        predecessor_path=args.predecessor,
        panel_2_replacement_plan=panel_2,
        panel_2_replacement_plan_path=args.panel_2_replacement_plan,
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
