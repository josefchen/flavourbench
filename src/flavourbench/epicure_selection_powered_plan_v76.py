"""Freeze joint 27-model inference before inspecting refreshed DeepSeek quality."""

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
from .epicure_selection_powered_plan_v67 import PLAN_SCHEMA_VERSION as PREDECESSOR_SCHEMA
from .epicure_selection_powered_plan_v67 import PLAN_VERSION as PREDECESSOR_VERSION
from .epicure_selection_powered_plan_v67 import verify_plan as verify_plan_v67
from .epicure_selection_powered_plan_v74 import verify_plan as verify_plan_v74
from .epicure_selection_powered_plan_v75 import verify_plan as verify_plan_v75
from .epicure_selection_route_manifest_v54 import DEEPSEEK_PRO_MODEL_ID

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v76"
PLAN_VERSION = "flavourbench-selection-27x1280-deepseek-contract-refresh-v76"


class SelectionPoweredPlanV76Error(RuntimeError):
    """The refreshed 27-model joint plan failed verification."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV76Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV76Error("joint-plan input is not a JSON object")
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
        not verify_plan_v67(predecessor)
        or not verify_plan_v74(panel_1_plan)
        or not verify_plan_v75(panel_2_plan)
    ):
        raise SelectionPoweredPlanV76Error("v76 requires exact v67, v74, and v75 plans")
    before = predecessor["roster"]["models"]
    panel_1_rows = panel_1_plan["roster"]["models"]
    panel_2_rows = panel_2_plan["roster"]["models"]
    if panel_1_rows[12] != panel_2_rows[12]:
        raise SelectionPoweredPlanV76Error("refreshed DeepSeek panel rows differ")
    after = copy.deepcopy(before)
    after[12] = copy.deepcopy(panel_1_rows[12])
    changed = [
        index
        for index, (left, right) in enumerate(zip(before, after, strict=True))
        if left != right
    ]
    if len(before) != 27 or changed != [12] or after[12]["model_id"] != DEEPSEEK_PRO_MODEL_ID:
        raise SelectionPoweredPlanV76Error("v76 changed more than the DeepSeek route row")

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "joint_analysis_frozen_before_refreshed_deepseek_quality_inspection"
    document["inputs"]["joint_plan_v67_predecessor"] = _pin(predecessor, predecessor_path)
    prior_panel_1 = copy.deepcopy(document["inputs"]["panel_1_plan"])
    prior_panel_2 = copy.deepcopy(document["inputs"]["panel_2_plan"])
    document["inputs"]["panel_1_plan"] = _pin(panel_1_plan, panel_1_plan_path)
    document["inputs"]["panel_2_plan"] = _pin(panel_2_plan, panel_2_plan_path)
    document["roster"]["models"] = copy.deepcopy(after)
    document["source_rules"].update(
        {
            "panel_1_uses_complete_refreshed_deepseek_block": True,
            "panel_2_uses_complete_refreshed_deepseek_block": True,
            "refreshed_deepseek_model_ids": [DEEPSEEK_PRO_MODEL_ID],
            "refreshed_deepseek_provider_tag": "gmicloud/fp8",
            "refreshed_deepseek_selective_retry": False,
            "refreshed_deepseek_prior_responses_used": False,
            "refreshed_deepseek_cross_contract_pooling": False,
            "refreshed_deepseek_included_without_quality_scores_or_selections": True,
            "superseded_panel_1_plan_before_deepseek_contract_refresh": prior_panel_1,
            "superseded_panel_2_plan_before_deepseek_contract_refresh": prior_panel_2,
            "superseded_deepseek_roster_row": copy.deepcopy(before[12]),
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV76Error("constructed v76 joint plan failed verification")
    return document


def _as_v67(document: Mapping[str, Any]) -> dict[str, Any]:
    prior = copy.deepcopy(document)
    prior.pop("artifact_sha256", None)
    prior["schema_version"] = PREDECESSOR_SCHEMA
    prior["plan_version"] = PREDECESSOR_VERSION
    prior["status"] = "joint_analysis_frozen_before_glm53_quality_inspection"
    prior["inputs"].pop("joint_plan_v67_predecessor")
    source = prior["source_rules"]
    prior["inputs"]["panel_1_plan"] = source.pop(
        "superseded_panel_1_plan_before_deepseek_contract_refresh"
    )
    prior["inputs"]["panel_2_plan"] = source.pop(
        "superseded_panel_2_plan_before_deepseek_contract_refresh"
    )
    prior["roster"]["models"][12] = source.pop("superseded_deepseek_roster_row")
    for key in (
        "panel_1_uses_complete_refreshed_deepseek_block",
        "panel_2_uses_complete_refreshed_deepseek_block",
        "refreshed_deepseek_model_ids",
        "refreshed_deepseek_provider_tag",
        "refreshed_deepseek_selective_retry",
        "refreshed_deepseek_prior_responses_used",
        "refreshed_deepseek_cross_contract_pooling",
        "refreshed_deepseek_included_without_quality_scores_or_selections",
    ):
        source.pop(key)
    prior["artifact_sha256"] = _sha256(prior)
    return prior


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        source = document["source_rules"]
        predecessor = document["inputs"]["joint_plan_v67_predecessor"]
        panel_1 = document["inputs"]["panel_1_plan"]
        panel_2 = document["inputs"]["panel_2_plan"]
        deepseek = document["roster"]["models"][12]
    except (IndexError, KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and document.get("status")
        == "joint_analysis_frozen_before_refreshed_deepseek_quality_inspection"
        and recorded == _sha256(payload)
        and verify_plan_v67(_as_v67(document))
        and document["roster"].get("model_count") == 27
        and document["roster"].get("pairwise_hypotheses") == 351
        and deepseek.get("model_id") == DEEPSEEK_PRO_MODEL_ID
        and deepseek.get("provider_tag") == "gmicloud/fp8"
        and source.get("panel_1_uses_complete_refreshed_deepseek_block") is True
        and source.get("panel_2_uses_complete_refreshed_deepseek_block") is True
        and source.get("refreshed_deepseek_model_ids") == [DEEPSEEK_PRO_MODEL_ID]
        and source.get("refreshed_deepseek_provider_tag") == "gmicloud/fp8"
        and source.get("refreshed_deepseek_selective_retry") is False
        and source.get("refreshed_deepseek_prior_responses_used") is False
        and source.get("refreshed_deepseek_cross_contract_pooling") is False
        and source.get("refreshed_deepseek_included_without_quality_scores_or_selections") is True
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
            raise SelectionPoweredPlanV76Error("content-addressed joint-plan conflict")
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
