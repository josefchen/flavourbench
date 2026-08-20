"""Freeze panel 1 of the additive GLM-5.3 limited benchmark run."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v23 import _roster_row
from .epicure_selection_powered_plan_v54 import _sha256, _sha256_file
from .epicure_selection_powered_plan_v62 import PLAN_SCHEMA_VERSION as PREDECESSOR_SCHEMA_V62
from .epicure_selection_powered_plan_v62 import PLAN_VERSION as PREDECESSOR_VERSION_V62
from .epicure_selection_powered_plan_v62 import verify_plan as verify_plan_v62
from .epicure_selection_route_manifest_v65 import MODEL_ID
from .epicure_selection_route_manifest_v65 import verify_manifest as verify_manifest_v65
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v65"
PLAN_VERSION = "flavourbench-selection-27x640-panel-1-glm53-limited-run-v65"


class SelectionPoweredPlanV65Error(RuntimeError):
    """The panel-1 GLM-5.3 addition plan failed verification."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV65Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV65Error("plan input is not a JSON object")
    return value


def _pin(document: Mapping[str, Any], path: Path, *, manifest: bool = False) -> dict[str, str]:
    return {
        "semantic_sha256": str(
            document["content_address"]["digest"] if manifest else document["artifact_sha256"]
        ),
        "physical_sha256": _sha256_file(path),
    }


def _build_panel_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    verify_predecessor: Callable[[Mapping[str, Any]], bool],
    predecessor_schema: str,
    predecessor_version: str,
    schema_version: str,
    plan_version: str,
    status: str,
    predecessor_key: str,
    panel: int,
) -> dict[str, Any]:
    if not verify_predecessor(predecessor) or not verify_manifest_v65(manifest):
        raise SelectionPoweredPlanV65Error("GLM-5.3 plan inputs failed verification")
    candidates = select_candidates(manifest)
    if len(candidates) != 27 or candidates[-1].model_id != MODEL_ID:
        raise SelectionPoweredPlanV65Error("GLM-5.3 manifest roster differs")

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = schema_version
    document["plan_version"] = plan_version
    document["status"] = status
    document["inputs"][predecessor_key] = _pin(predecessor, predecessor_path)
    prior_manifest = copy.deepcopy(document["inputs"]["route_manifest"])
    document["inputs"]["route_manifest"] = _pin(manifest, manifest_path, manifest=True)

    glm = candidates[-1]
    glm_row = _roster_row(glm, "provider_fixed")
    glm_row["final_max_output_tokens"] = 16_384
    prior_collection = copy.deepcopy(document["execution"]["collection_concurrency"])
    prior_reasoning = str(document["execution"]["reasoning_control"])
    prior_scope = str(document["budget"]["successor_scope"])
    prior_pilot_cells = int(document["execution"]["pilot"]["cells"])
    document["roster"]["model_count"] = 27
    document["roster"]["models"].append(glm_row)
    document["execution"]["pilot"]["cells"] = 108
    document["execution"]["collection_concurrency"]["per_model_by_backend"]["zai_coding_direct"] = 1
    document["execution"]["collection_concurrency"]["per_model_by_model_id"][MODEL_ID] = 1
    document["execution"]["collection_concurrency"]["reason"] = (
        f"single-flight the separately authorized finite GLM-5.3 panel-{panel} block"
    )
    document["execution"]["reasoning_control"] = (
        prior_reasoning
        + "; GLM-5.3 uses the provider-fixed Coding Plan mode under its one-run permission"
    )
    document["budget"]["successor_scope"] = (
        f"one complete 640+64 panel-{panel} GLM-5.3 Coding Plan block; subscription quota, "
        "no per-call price claim"
    )
    document["execution"]["glm53_limited_run_addition"] = {
        "schema_version": "flavourbench-score-blind-additive-model-block-v1",
        "panel": panel,
        "new_model_ids": [MODEL_ID],
        "primary_cells_per_model": 640,
        "repeat_cells_per_model": 64,
        "complete_block_required": True,
        "provider_backend": "zai_coding_direct",
        "requested_model_id": "glm-5.3",
        "exact_returned_model_required": "glm-5.3",
        "finite_cli_only": True,
        "standing_service": False,
        "automatic_fallback": False,
        "quality_scores_or_selections_used_for_inclusion": False,
        "previous_26_model_rows_preserved": document["roster"]["models"][:-1]
        == predecessor["roster"]["models"],
        "superseded_route_manifest": prior_manifest,
        "superseded_collection_concurrency": prior_collection,
        "superseded_reasoning_control": prior_reasoning,
        "superseded_budget_successor_scope": prior_scope,
        "superseded_pilot_cells": prior_pilot_cells,
    }
    document["artifact_sha256"] = _sha256(document)
    if not _verify_panel_plan(
        document,
        verify_predecessor=verify_predecessor,
        predecessor_schema=predecessor_schema,
        predecessor_version=predecessor_version,
        schema_version=schema_version,
        plan_version=plan_version,
        status=status,
        predecessor_key=predecessor_key,
        panel=panel,
    ):
        raise SelectionPoweredPlanV65Error("constructed GLM-5.3 plan failed verification")
    return document


def _as_predecessor(
    document: Mapping[str, Any],
    *,
    predecessor_schema: str,
    predecessor_version: str,
    predecessor_key: str,
) -> dict[str, Any]:
    prior = copy.deepcopy(document)
    prior.pop("artifact_sha256", None)
    addition = prior["execution"].pop("glm53_limited_run_addition")
    prior["schema_version"] = predecessor_schema
    prior["plan_version"] = predecessor_version
    prior["status"] = (
        "panel_1_deepseek_gmicloud_block_frozen_before_execution"
        if predecessor_schema == PREDECESSOR_SCHEMA_V62
        else "panel_2_deepseek_gmicloud_block_frozen_before_execution"
    )
    prior["inputs"].pop(predecessor_key)
    prior["inputs"]["route_manifest"] = addition["superseded_route_manifest"]
    prior["roster"]["model_count"] = 26
    prior["roster"]["models"] = prior["roster"]["models"][:-1]
    prior["execution"]["pilot"]["cells"] = addition["superseded_pilot_cells"]
    prior["execution"]["collection_concurrency"] = addition["superseded_collection_concurrency"]
    prior["execution"]["reasoning_control"] = addition["superseded_reasoning_control"]
    prior["budget"]["successor_scope"] = addition["superseded_budget_successor_scope"]
    prior["artifact_sha256"] = _sha256(prior)
    return prior


def _verify_panel_plan(
    document: Mapping[str, Any],
    *,
    verify_predecessor: Callable[[Mapping[str, Any]], bool],
    predecessor_schema: str,
    predecessor_version: str,
    schema_version: str,
    plan_version: str,
    status: str,
    predecessor_key: str,
    panel: int,
) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        addition = document["execution"]["glm53_limited_run_addition"]
        rows = document["roster"]["models"]
        glm = rows[-1]
        predecessor_pin = document["inputs"][predecessor_key]
        manifest_pin = document["inputs"]["route_manifest"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == schema_version
        and document.get("plan_version") == plan_version
        and document.get("status") == status
        and recorded == _sha256(payload)
        and verify_predecessor(
            _as_predecessor(
                document,
                predecessor_schema=predecessor_schema,
                predecessor_version=predecessor_version,
                predecessor_key=predecessor_key,
            )
        )
        and document["roster"].get("model_count") == 27
        and len(rows) == 27
        and glm.get("model_id") == MODEL_ID
        and glm.get("execution_backend") == "zai_coding_direct"
        and glm.get("final_reasoning_effort") == "provider_fixed"
        and glm.get("final_max_output_tokens") == 16_384
        and document["execution"]["pilot"].get("cells") == 108
        and addition.get("panel") == panel
        and addition.get("new_model_ids") == [MODEL_ID]
        and addition.get("complete_block_required") is True
        and addition.get("finite_cli_only") is True
        and addition.get("standing_service") is False
        and addition.get("automatic_fallback") is False
        and addition.get("quality_scores_or_selections_used_for_inclusion") is False
        and addition.get("previous_26_model_rows_preserved") is True
        and all(
            isinstance(pin.get("semantic_sha256"), str)
            and len(pin["semantic_sha256"]) == 64
            and isinstance(pin.get("physical_sha256"), str)
            and len(pin["physical_sha256"]) == 64
            for pin in (predecessor_pin, manifest_pin)
        )
    )


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    return _build_panel_plan(
        predecessor=predecessor,
        predecessor_path=predecessor_path,
        manifest=manifest,
        manifest_path=manifest_path,
        verify_predecessor=verify_plan_v62,
        predecessor_schema=PREDECESSOR_SCHEMA_V62,
        predecessor_version=PREDECESSOR_VERSION_V62,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_1_glm53_limited_run_frozen_before_execution",
        predecessor_key="plan_v62_predecessor",
        panel=1,
    )


def verify_plan(document: Mapping[str, Any]) -> bool:
    return _verify_panel_plan(
        document,
        verify_predecessor=verify_plan_v62,
        predecessor_schema=PREDECESSOR_SCHEMA_V62,
        predecessor_version=PREDECESSOR_VERSION_V62,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_1_glm53_limited_run_frozen_before_execution",
        predecessor_key="plan_v62_predecessor",
        panel=1,
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV65Error("content-addressed plan conflict")
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        _write(
            build_plan(
                predecessor=_load(args.predecessor),
                predecessor_path=args.predecessor,
                manifest=_load(args.manifest),
                manifest_path=args.manifest,
            ),
            args.output_directory,
        )
    )


if __name__ == "__main__":
    run()
