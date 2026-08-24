"""Freeze the panel-2 DeepSeek full-block price-contract refresh."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v54 import _sha256_file
from .epicure_selection_powered_plan_v66 import verify_plan as verify_plan_v66
from .epicure_selection_powered_plan_v70 import (
    _build_panel_plan,
    _load,
    _verify_panel_plan,
    _write,
)
from .epicure_selection_route_manifest_v69 import PROVIDER_NAME, ROUTE_TAG
from .epicure_selection_route_manifest_v69 import verify_manifest as verify_manifest_v69

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v71"
PLAN_VERSION = "flavourbench-selection-27x640-panel-2-deepseek-price-refresh-v71"


class SelectionPoweredPlanV71Error(RuntimeError):
    """The panel-2 DeepSeek price-refresh plan failed verification."""


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
        verify_predecessor=verify_plan_v66,
        verify_manifest=verify_manifest_v69,
        route_tag=ROUTE_TAG,
        provider_name=PROVIDER_NAME,
        superseded_provider_tags=["gmicloud/fp8"],
        collection_concurrency=1,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_2_deepseek_price_refresh_frozen_before_execution",
        predecessor_key="plan_v66_predecessor",
        execution_key="deepseek_price_contract_refresh_v71",
        panel=2,
    )
    if not verify_plan(document):
        raise SelectionPoweredPlanV71Error("constructed v71 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    return _verify_panel_plan(
        document,
        schema_version=PLAN_SCHEMA_VERSION,
        plan_version=PLAN_VERSION,
        status="panel_2_deepseek_price_refresh_frozen_before_execution",
        predecessor_key="plan_v66_predecessor",
        execution_key="deepseek_price_contract_refresh_v71",
        route_tag=ROUTE_TAG,
        provider_name=PROVIDER_NAME,
        superseded_provider_tags=["gmicloud/fp8"],
        collection_concurrency=1,
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
