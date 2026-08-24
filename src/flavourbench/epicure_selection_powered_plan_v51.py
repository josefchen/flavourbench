"""Freeze the joint analysis after complete Qwen and Fable route replacement."""

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

from .epicure_selection_powered_plan_v48 import (
    PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V48,
)
from .epicure_selection_powered_plan_v48 import PLAN_VERSION as PLAN_VERSION_V48
from .epicure_selection_powered_plan_v48 import build_plan as build_plan_v48
from .epicure_selection_powered_plan_v48 import verify_plan as verify_plan_v48
from .epicure_selection_powered_plan_v49 import verify_plan as verify_plan_v49
from .epicure_selection_powered_plan_v50 import verify_plan as verify_plan_v50

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v51"
PLAN_VERSION = "flavourbench-selection-26x1280-two-panel-fable-bedrock-v51"


class SelectionPoweredPlanV51Error(RuntimeError):
    """The post-route-recovery joint plan failed verification."""


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


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV51Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV51Error("joint-plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    panel_1_plan: Mapping[str, Any],
    panel_1_plan_path: Path,
    panel_1_taskset: Mapping[str, Any],
    panel_1_taskset_path: Path,
    panel_1_repeat: Mapping[str, Any],
    panel_1_repeat_path: Path,
    panel_2_plan: Mapping[str, Any],
    panel_2_plan_path: Path,
    panel_2_taskset: Mapping[str, Any],
    panel_2_taskset_path: Path,
    panel_2_repeat: Mapping[str, Any],
    panel_2_repeat_path: Path,
) -> dict[str, Any]:
    if not verify_plan_v48(predecessor):
        raise SelectionPoweredPlanV51Error("v51 requires the frozen v48 inference predecessor")
    if not verify_plan_v50(panel_1_plan) or not verify_plan_v49(panel_2_plan):
        raise SelectionPoweredPlanV51Error("v51 requires exact Fable-recovered panel plans")
    document = build_plan_v48(
        panel_1_plan=panel_1_plan,
        panel_1_plan_path=panel_1_plan_path,
        panel_1_taskset=panel_1_taskset,
        panel_1_taskset_path=panel_1_taskset_path,
        panel_1_repeat=panel_1_repeat,
        panel_1_repeat_path=panel_1_repeat_path,
        panel_2_plan=panel_2_plan,
        panel_2_plan_path=panel_2_plan_path,
        panel_2_taskset=panel_2_taskset,
        panel_2_taskset_path=panel_2_taskset_path,
        panel_2_repeat=panel_2_repeat,
        panel_2_repeat_path=panel_2_repeat_path,
    )
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "joint_analysis_frozen_before_fable_replacement_or_panel_2_execution"
    document["inputs"]["joint_plan_v48_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": _sha256_file(predecessor_path),
    }
    document["source_rules"].update(
        {
            "panel_1_uses_complete_fable_replacement_block": True,
            "panel_2_fable_route_frozen_before_collection": True,
            "fable_bedrock_route_selected_without_quality_scores_or_selections": True,
            "superseded_fable_responses_used": False,
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV51Error("constructed v51 joint plan failed verification")
    return document


def _as_v48(document: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = copy.deepcopy(document)
    predecessor.pop("artifact_sha256", None)
    predecessor["schema_version"] = PLAN_SCHEMA_VERSION_V48
    predecessor["plan_version"] = PLAN_VERSION_V48
    predecessor["status"] = "joint_analysis_frozen_before_any_quality_score_inspection"
    predecessor["inputs"].pop("joint_plan_v48_predecessor", None)
    for key in (
        "panel_1_uses_complete_fable_replacement_block",
        "panel_2_fable_route_frozen_before_collection",
        "fable_bedrock_route_selected_without_quality_scores_or_selections",
        "superseded_fable_responses_used",
    ):
        predecessor["source_rules"].pop(key, None)
    predecessor["artifact_sha256"] = _sha256(predecessor)
    return predecessor


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        source = document["source_rules"]
        predecessor = document["inputs"]["joint_plan_v48_predecessor"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status")
        == "joint_analysis_frozen_before_fable_replacement_or_panel_2_execution"
        and verify_plan_v48(_as_v48(document))
        and source.get("panel_1_uses_complete_fable_replacement_block") is True
        and source.get("panel_2_fable_route_frozen_before_collection") is True
        and source.get("fable_bedrock_route_selected_without_quality_scores_or_selections") is True
        and source.get("superseded_fable_responses_used") is False
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
            raise SelectionPoweredPlanV51Error("content-addressed joint-plan conflict")
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
    parser.add_argument("--panel-1-taskset", type=Path, required=True)
    parser.add_argument("--panel-1-repeat-panel", type=Path, required=True)
    parser.add_argument("--panel-2-plan", type=Path, required=True)
    parser.add_argument("--panel-2-taskset", type=Path, required=True)
    parser.add_argument("--panel-2-repeat-panel", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "predecessor": args.predecessor,
        "panel_1_plan": args.panel_1_plan,
        "panel_1_taskset": args.panel_1_taskset,
        "panel_1_repeat": args.panel_1_repeat_panel,
        "panel_2_plan": args.panel_2_plan,
        "panel_2_taskset": args.panel_2_taskset,
        "panel_2_repeat": args.panel_2_repeat_panel,
    }
    documents = {label: _load(path) for label, path in paths.items()}
    document = build_plan(
        predecessor=documents["predecessor"],
        predecessor_path=paths["predecessor"],
        panel_1_plan=documents["panel_1_plan"],
        panel_1_plan_path=paths["panel_1_plan"],
        panel_1_taskset=documents["panel_1_taskset"],
        panel_1_taskset_path=paths["panel_1_taskset"],
        panel_1_repeat=documents["panel_1_repeat"],
        panel_1_repeat_path=paths["panel_1_repeat"],
        panel_2_plan=documents["panel_2_plan"],
        panel_2_plan_path=paths["panel_2_plan"],
        panel_2_taskset=documents["panel_2_taskset"],
        panel_2_taskset_path=paths["panel_2_taskset"],
        panel_2_repeat=documents["panel_2_repeat"],
        panel_2_repeat_path=paths["panel_2_repeat"],
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
