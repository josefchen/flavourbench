"""Freeze the no-generation Qwen 3.8 tool-auto successor route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .qwencloud_catalog import (
    CATALOG_SCHEMA_VERSION,
    QwenCloudCatalogError,
    build_unranked_qwen38_alias_route_manifest,
    verify_content_address,
    write_qwencloud_route_manifest,
)

CONFIRMATION = "FREEZE_QWEN38_TOOL_AUTO_SUCCESSOR_NO_GENERATION_V1"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument("--predecessor-failure-sha256", required=True)
    parser.add_argument("--cap-usd", default="2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"pass --confirm {CONFIRMATION}")
    try:
        catalog = json.loads(args.catalog.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit("could not read QwenCloud catalog") from error
    if (
        not isinstance(catalog, dict)
        or catalog.get("artifact_sha256") != args.expected_catalog_sha256
        or not verify_content_address(catalog, CATALOG_SCHEMA_VERSION)
    ):
        raise SystemExit("QwenCloud catalog content address does not verify")
    try:
        manifest = build_unranked_qwen38_alias_route_manifest(
            catalog_artifact=catalog,
            cap_usd=args.cap_usd,
            allow_mutable_alias_exploratory=True,
            tool_auto_successor_failure_sha256=args.predecessor_failure_sha256,
        )
        path = write_qwencloud_route_manifest(manifest, args.output_dir)
    except (OSError, QwenCloudCatalogError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "status": "qwen38_tool_auto_successor_frozen_no_external_calls",
                "provider_calls_made": False,
                "epicure_calls_made": False,
                "artifact": str(path.resolve()),
                "artifact_sha256": manifest["content_address"]["digest"],
                "predecessor_failure_artifact_sha256": (
                    args.predecessor_failure_sha256
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
