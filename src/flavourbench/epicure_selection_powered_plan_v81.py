"""Freeze panel 2 for a complete Anthropic-routed Claude Fable 5 rerun."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v54 import _sha256_file
from .epicure_selection_powered_plan_v75 import verify_plan as verify_plan_v75
from .epicure_selection_powered_plan_v80 import (
    SelectionPoweredPlanV80Error,
    _build_fable_plan,
    _load,
    _verify_fable_plan,
    _write,
)

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v81"
PLAN_VERSION = "flavourbench-selection-27x640-panel-2-fable-anthropic-v81"


class SelectionPoweredPlanV81Error(SelectionPoweredPlanV80Error):
    """The complete panel-2 first-party Fable plan failed verification."""


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
        verify_predecessor=verify_plan_v75,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_2_fable_anthropic_complete_block_frozen_before_execution",
        predecessor_key="plan_v75_predecessor",
        execution_key="fable_anthropic_complete_block_v81",
        panel=2,
    )
    if not verify_plan(document):
        raise SelectionPoweredPlanV81Error("constructed v81 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    return _verify_fable_plan(
        document,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_2_fable_anthropic_complete_block_frozen_before_execution",
        predecessor_key="plan_v75_predecessor",
        execution_key="fable_anthropic_complete_block_v81",
    )


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
