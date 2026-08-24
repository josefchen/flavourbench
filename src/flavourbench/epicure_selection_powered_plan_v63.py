"""Freeze the panel-2 DeepSeek all-cell GMICloud replacement."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v54 import _sha256_file
from .epicure_selection_powered_plan_v54 import verify_plan as verify_plan_v54
from .epicure_selection_powered_plan_v62 import (
    _build_panel_plan,
    _load,
    _verify_panel_plan,
    _write,
)

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v63"
PLAN_VERSION = "flavourbench-selection-26x640-panel-2-deepseek-gmicloud-block-v63"


class SelectionPoweredPlanV63Error(RuntimeError):
    """The panel-2 GMICloud replacement plan failed verification."""


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    document = _build_panel_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=predecessor_physical_sha256,
        manifest=manifest,
        manifest_physical_sha256=manifest_physical_sha256,
        verify_predecessor=verify_plan_v54,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_2_deepseek_gmicloud_block_frozen_before_execution",
        predecessor_key="plan_v54_predecessor",
        execution_key="deepseek_complete_block_replacement_v63",
        panel=2,
    )
    if not verify_plan(document):
        raise SelectionPoweredPlanV63Error("constructed v63 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    return _verify_panel_plan(
        document,
        verify_predecessor=verify_plan_v54,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_2_deepseek_gmicloud_block_frozen_before_execution",
        predecessor_key="plan_v54_predecessor",
        execution_key="deepseek_complete_block_replacement_v63",
    )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor)
    manifest = _load(args.manifest)
    document = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
