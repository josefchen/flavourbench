"""Freeze panel 2 of the additive GLM-5.3 limited benchmark run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v63 import PLAN_SCHEMA_VERSION as PREDECESSOR_SCHEMA_V63
from .epicure_selection_powered_plan_v63 import PLAN_VERSION as PREDECESSOR_VERSION_V63
from .epicure_selection_powered_plan_v63 import verify_plan as verify_plan_v63
from .epicure_selection_powered_plan_v65 import (
    SelectionPoweredPlanV65Error,
    _build_panel_plan,
    _load,
    _verify_panel_plan,
    _write,
)

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v66"
PLAN_VERSION = "flavourbench-selection-27x640-panel-2-glm53-limited-run-v66"


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
        verify_predecessor=verify_plan_v63,
        predecessor_schema=PREDECESSOR_SCHEMA_V63,
        predecessor_version=PREDECESSOR_VERSION_V63,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_2_glm53_limited_run_frozen_before_execution",
        predecessor_key="plan_v63_predecessor",
        panel=2,
    )


def verify_plan(document: Mapping[str, Any]) -> bool:
    return _verify_panel_plan(
        document,
        verify_predecessor=verify_plan_v63,
        predecessor_schema=PREDECESSOR_SCHEMA_V63,
        predecessor_version=PREDECESSOR_VERSION_V63,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_2_glm53_limited_run_frozen_before_execution",
        predecessor_key="plan_v63_predecessor",
        panel=2,
    )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        document = build_plan(
            predecessor=_load(args.predecessor),
            predecessor_path=args.predecessor,
            manifest=_load(args.manifest),
            manifest_path=args.manifest,
        )
    except SelectionPoweredPlanV65Error:
        raise
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
