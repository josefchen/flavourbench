"""Freeze DeepSeek price-only response lineage before final quality analysis."""

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
from .epicure_selection_powered_plan_v62 import verify_plan as verify_plan_v62
from .epicure_selection_powered_plan_v63 import verify_plan as verify_plan_v63
from .epicure_selection_powered_plan_v76 import PLAN_SCHEMA_VERSION as PREDECESSOR_SCHEMA
from .epicure_selection_powered_plan_v76 import PLAN_VERSION as PREDECESSOR_VERSION
from .epicure_selection_powered_plan_v76 import verify_plan as verify_plan_v76
from .epicure_selection_route_manifest_v54 import DEEPSEEK_PRO_MODEL_ID

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v77"
PLAN_VERSION = "flavourbench-selection-27x1280-deepseek-price-lineage-v77"


class SelectionPoweredPlanV77Error(RuntimeError):
    """The DeepSeek price-only response-lineage plan failed verification."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV77Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV77Error("joint-plan input is not a JSON object")
    return value


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _sha256_file(path),
    }


def _row(plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in plan["roster"]["models"] if row.get("model_id") == DEEPSEEK_PRO_MODEL_ID]
    if len(rows) != 1:
        raise SelectionPoweredPlanV77Error("DeepSeek roster row is not unique")
    return dict(rows[0])


def _price_only_difference(
    prior_plan: Mapping[str, Any], current_plan: Mapping[str, Any]
) -> tuple[str, str]:
    prior = _row(prior_plan)
    current = _row(current_plan)
    differing = {key for key in set(prior) | set(current) if prior.get(key) != current.get(key)}
    if differing != {"endpoint_sha256"}:
        raise SelectionPoweredPlanV77Error(
            "DeepSeek predecessor differs beyond endpoint price metadata"
        )
    if prior.get("endpoint_execution_sha256") != current.get("endpoint_execution_sha256"):
        raise SelectionPoweredPlanV77Error("DeepSeek inference contract changed")
    return str(prior["endpoint_sha256"]), str(current["endpoint_sha256"])


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    panel_1_prior_plan: Mapping[str, Any],
    panel_1_prior_plan_path: Path,
    panel_2_prior_plan: Mapping[str, Any],
    panel_2_prior_plan_path: Path,
) -> dict[str, Any]:
    if not (
        verify_plan_v76(predecessor)
        and verify_plan_v62(panel_1_prior_plan)
        and verify_plan_v63(panel_2_prior_plan)
    ):
        raise SelectionPoweredPlanV77Error("v77 requires exact v76, v62, and v63 plans")
    panel_1_prior, panel_1_current = _price_only_difference(
        panel_1_prior_plan, {"roster": predecessor["roster"]}
    )
    panel_2_prior, panel_2_current = _price_only_difference(
        panel_2_prior_plan, {"roster": predecessor["roster"]}
    )
    if (panel_1_prior, panel_1_current) != (panel_2_prior, panel_2_current):
        raise SelectionPoweredPlanV77Error("DeepSeek price-only changes differ by panel")

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "joint_source_lineage_frozen_after_coverage_before_quality_analysis"
    document["inputs"].update(
        {
            "joint_plan_v76_predecessor": _pin(predecessor, predecessor_path),
            "panel_1_prior_deepseek_plan_v62": _pin(panel_1_prior_plan, panel_1_prior_plan_path),
            "panel_2_prior_deepseek_plan_v63": _pin(panel_2_prior_plan, panel_2_prior_plan_path),
        }
    )
    document["source_rules"].update(
        {
            "deepseek_response_source_order": {
                "panel_1": ["v62_prior_block", "v74_refresh_block", "v74_completion_blocks"],
                "panel_2": ["v63_prior_block", "v75_refresh_block", "v75_completion_blocks"],
            },
            "deepseek_response_selection_rule": (
                "first_completed_parseable_response_in_frozen_source_directory_order"
            ),
            "deepseek_prior_responses_used": True,
            "deepseek_cross_contract_pooling": True,
            "deepseek_cross_contract_allowed_roster_differences": ["endpoint_sha256"],
            "deepseek_cross_contract_difference_class": "price_metadata_only",
            "deepseek_prior_endpoint_sha256": panel_1_prior,
            "deepseek_current_endpoint_sha256": panel_1_current,
            "deepseek_endpoint_execution_sha256_unchanged": True,
            "deepseek_provider_and_provider_tag_unchanged": True,
            "deepseek_prompt_task_parser_and_inference_bytes_unchanged": True,
            "coverage_and_parseability_inspected_before_source_freeze": True,
            "quality_scores_and_model_selections_inspected_for_source_decision": False,
            "failed_and_malformed_response_artifacts_preserved": True,
            "failed_and_malformed_response_artifacts_used_as_score_data": False,
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV77Error("constructed v77 joint plan failed verification")
    return document


def _as_v76(document: Mapping[str, Any]) -> dict[str, Any]:
    prior = copy.deepcopy(document)
    prior.pop("artifact_sha256", None)
    prior["schema_version"] = PREDECESSOR_SCHEMA
    prior["plan_version"] = PREDECESSOR_VERSION
    prior["status"] = "joint_analysis_frozen_before_refreshed_deepseek_quality_inspection"
    for key in (
        "joint_plan_v76_predecessor",
        "panel_1_prior_deepseek_plan_v62",
        "panel_2_prior_deepseek_plan_v63",
    ):
        prior["inputs"].pop(key)
    for key in (
        "deepseek_response_source_order",
        "deepseek_response_selection_rule",
        "deepseek_prior_responses_used",
        "deepseek_cross_contract_pooling",
        "deepseek_cross_contract_allowed_roster_differences",
        "deepseek_cross_contract_difference_class",
        "deepseek_prior_endpoint_sha256",
        "deepseek_current_endpoint_sha256",
        "deepseek_endpoint_execution_sha256_unchanged",
        "deepseek_provider_and_provider_tag_unchanged",
        "deepseek_prompt_task_parser_and_inference_bytes_unchanged",
        "coverage_and_parseability_inspected_before_source_freeze",
        "quality_scores_and_model_selections_inspected_for_source_decision",
        "failed_and_malformed_response_artifacts_preserved",
        "failed_and_malformed_response_artifacts_used_as_score_data",
    ):
        prior["source_rules"].pop(key)
    prior["artifact_sha256"] = _sha256(prior)
    return prior


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        inputs = document["inputs"]
        rules = document["source_rules"]
        pins = (
            inputs["joint_plan_v76_predecessor"],
            inputs["panel_1_prior_deepseek_plan_v62"],
            inputs["panel_2_prior_deepseek_plan_v63"],
        )
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and document.get("status")
        == "joint_source_lineage_frozen_after_coverage_before_quality_analysis"
        and recorded == _sha256(payload)
        and verify_plan_v76(_as_v76(document))
        and rules.get("deepseek_response_selection_rule")
        == "first_completed_parseable_response_in_frozen_source_directory_order"
        and rules.get("deepseek_prior_responses_used") is True
        and rules.get("deepseek_cross_contract_pooling") is True
        and rules.get("deepseek_cross_contract_allowed_roster_differences") == ["endpoint_sha256"]
        and rules.get("deepseek_cross_contract_difference_class") == "price_metadata_only"
        and rules.get("deepseek_endpoint_execution_sha256_unchanged") is True
        and rules.get("deepseek_provider_and_provider_tag_unchanged") is True
        and rules.get("deepseek_prompt_task_parser_and_inference_bytes_unchanged") is True
        and rules.get("coverage_and_parseability_inspected_before_source_freeze") is True
        and rules.get("quality_scores_and_model_selections_inspected_for_source_decision") is False
        and rules.get("failed_and_malformed_response_artifacts_preserved") is True
        and rules.get("failed_and_malformed_response_artifacts_used_as_score_data") is False
        and all(
            isinstance(pin.get("semantic_sha256"), str)
            and len(pin["semantic_sha256"]) == 64
            and isinstance(pin.get("physical_sha256"), str)
            and len(pin["physical_sha256"]) == 64
            for pin in pins
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
            raise SelectionPoweredPlanV77Error("content-addressed joint-plan conflict")
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
    parser.add_argument("--panel-1-prior-plan", type=Path, required=True)
    parser.add_argument("--panel-2-prior-plan", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        _write(
            build_plan(
                predecessor=_load(args.predecessor),
                predecessor_path=args.predecessor,
                panel_1_prior_plan=_load(args.panel_1_prior_plan),
                panel_1_prior_plan_path=args.panel_1_prior_plan,
                panel_2_prior_plan=_load(args.panel_2_prior_plan),
                panel_2_prior_plan_path=args.panel_2_prior_plan,
            ),
            args.output_directory,
        )
    )


if __name__ == "__main__":
    run()
