"""Freeze panel 1 for a complete Anthropic-routed Claude Fable 5 rerun."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v23 import _roster_row
from .epicure_selection_powered_plan_v54 import _sha256, _sha256_file
from .epicure_selection_powered_plan_v70 import _load
from .epicure_selection_powered_plan_v74 import verify_plan as verify_plan_v74
from .epicure_selection_route_manifest_v45 import FABLE_MODEL_ID
from .epicure_selection_route_manifest_v79 import PROVIDER_NAME, ROUTE_TAG
from .epicure_selection_route_manifest_v79 import verify_manifest as verify_manifest_v79
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v80"
PLAN_VERSION = "flavourbench-selection-27x640-panel-1-fable-anthropic-v80"
PRIMARY_TASKS = 640
REPEAT_TASKS = 64


class SelectionPoweredPlanV80Error(RuntimeError):
    """The complete panel-1 first-party Fable plan failed verification."""


def _build_fable_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    verify_predecessor: Callable[[Mapping[str, Any]], bool],
    schema_version: str,
    plan_version: str,
    status: str,
    predecessor_key: str,
    execution_key: str,
    panel: int,
) -> dict[str, Any]:
    if not verify_predecessor(predecessor) or not verify_manifest_v79(manifest):
        raise SelectionPoweredPlanV80Error("Fable predecessor or manifest failed verification")
    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    candidates = {candidate.model_id: candidate for candidate in select_candidates(manifest)}
    if list(prior_rows) != list(candidates):
        raise SelectionPoweredPlanV80Error("Fable route roster differs from predecessor")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = schema_version
    document["plan_version"] = plan_version
    document["status"] = status
    document["inputs"][predecessor_key] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["superseded_route_manifest_v73"] = document["inputs"]["route_manifest"]
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }

    final_rows = json.loads(json.dumps(predecessor["roster"]["models"]))
    index = next(i for i, row in enumerate(final_rows) if row["model_id"] == FABLE_MODEL_ID)
    prior = prior_rows[FABLE_MODEL_ID]
    replacement = _roster_row(candidates[FABLE_MODEL_ID], str(prior["final_reasoning_effort"]))
    if "final_max_output_tokens" in prior:
        replacement["final_max_output_tokens"] = int(prior["final_max_output_tokens"])
    final_rows[index] = replacement
    if any(
        row != predecessor["roster"]["models"][row_index]
        for row_index, row in enumerate(final_rows)
        if row["model_id"] != FABLE_MODEL_ID
    ):
        raise SelectionPoweredPlanV80Error("Fable plan changed a non-Fable roster row")
    document["roster"]["models"] = final_rows
    document["execution"][execution_key] = {
        "schema_version": "flavourbench-score-blind-complete-model-route-refresh-v1",
        "source_plan_sha256": predecessor["artifact_sha256"],
        "replacement_model_ids": [FABLE_MODEL_ID],
        "replacement_provider_tags": {FABLE_MODEL_ID: ROUTE_TAG},
        "replacement_provider_names": {FABLE_MODEL_ID: PROVIDER_NAME},
        "replacement_primary_cells_per_model": PRIMARY_TASKS,
        "replacement_repeat_cells_per_model": REPEAT_TASKS,
        "fixed_transport_pilot_required": True,
        "replacement_blocks_must_be_complete": True,
        "superseded_responses_used": False,
        "cross_route_response_pooling": False,
        "selective_failed_cell_retry": False,
        "selection_uses_scores_or_selections": False,
        "selection_uses_transport_status_only": True,
        "automatic_fallback": False,
        "quality_score_definition": "successful_and_parseable_only",
    }
    document["execution"]["reasoning_control"] = (
        f"replace all 640 primary and 64 repeat panel-{panel} Fable cells through the exact "
        "Anthropic first-party route; preserve tasks, prompts, decoding, scoring, and inference"
    )
    document["execution"]["collection_concurrency"] = {
        "global": 2,
        "per_model_default": 2,
        "per_model_by_backend": {"openrouter": 2},
        "per_model_by_model_id": {FABLE_MODEL_ID: 2},
        "reason": "one exact-route complete Fable block with bounded concurrency",
    }
    document["budget"].update(
        {
            "aggregate_program_cap": "800",
            "program_cap": "800",
            "hard_cap": "800",
            "successor_scope": f"one complete 640+64 panel-{panel} Fable first-party block",
        }
    )
    document["artifact_sha256"] = _sha256(document)
    return document


def _verify_fable_plan(
    document: Mapping[str, Any],
    *,
    schema_version: str,
    plan_version: str,
    status: str,
    predecessor_key: str,
    execution_key: str,
) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        replacement = document["execution"][execution_key]
        rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
        predecessor = document["inputs"][predecessor_key]
        manifest = document["inputs"]["route_manifest"]
        superseded = document["inputs"]["superseded_route_manifest_v73"]
    except (KeyError, TypeError):
        return False
    row = rows.get(FABLE_MODEL_ID) or {}
    return bool(
        document.get("schema_version") == schema_version
        and document.get("plan_version") == plan_version
        and document.get("status") == status
        and recorded == _sha256(payload)
        and len(rows) == 27
        and row.get("provider_tag") == ROUTE_TAG
        and row.get("provider_name") == PROVIDER_NAME
        and replacement.get("replacement_model_ids") == [FABLE_MODEL_ID]
        and replacement.get("replacement_provider_tags") == {FABLE_MODEL_ID: ROUTE_TAG}
        and replacement.get("replacement_primary_cells_per_model") == PRIMARY_TASKS
        and replacement.get("replacement_repeat_cells_per_model") == REPEAT_TASKS
        and replacement.get("fixed_transport_pilot_required") is True
        and replacement.get("replacement_blocks_must_be_complete") is True
        and replacement.get("superseded_responses_used") is False
        and replacement.get("cross_route_response_pooling") is False
        and replacement.get("selective_failed_cell_retry") is False
        and replacement.get("selection_uses_scores_or_selections") is False
        and replacement.get("selection_uses_transport_status_only") is True
        and replacement.get("automatic_fallback") is False
        and verify_policy_document(document["execution"]["execution_policy"])
        and all(
            isinstance(pin.get("semantic_sha256"), str)
            and len(pin["semantic_sha256"]) == 64
            and isinstance(pin.get("physical_sha256"), str)
            and len(pin["physical_sha256"]) == 64
            for pin in (predecessor, manifest, superseded)
        )
    )


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    document = _build_fable_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=predecessor_physical_sha256,
        manifest=manifest,
        manifest_physical_sha256=manifest_physical_sha256,
        verify_predecessor=verify_plan_v74,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_1_fable_anthropic_complete_block_frozen_before_execution",
        predecessor_key="plan_v74_predecessor",
        execution_key="fable_anthropic_complete_block_v80",
        panel=1,
    )
    if not verify_plan(document):
        raise SelectionPoweredPlanV80Error("constructed v80 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    return _verify_fable_plan(
        document,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_1_fable_anthropic_complete_block_frozen_before_execution",
        predecessor_key="plan_v74_predecessor",
        execution_key="fable_anthropic_complete_block_v80",
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV80Error("content-addressed plan conflict")
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
    predecessor = _load(args.predecessor)
    manifest = _load(args.manifest)
    print(
        _write(
            build_plan(
                predecessor=predecessor,
                predecessor_physical_sha256=_sha256_file(args.predecessor),
                manifest=manifest,
                manifest_physical_sha256=_sha256_file(args.manifest),
            ),
            args.output_directory,
        )
    )


if __name__ == "__main__":
    run()
