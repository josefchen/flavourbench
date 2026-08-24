"""Freeze a no-generation QwenCloud execution route from an authenticated catalog."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from .qwencloud_catalog import (
    CATALOG_SCHEMA_VERSION,
    QwenCloudCatalogError,
    build_unranked_qwen37_route_manifest,
    build_unranked_qwen38_alias_route_manifest,
    verify_content_address,
    write_qwencloud_route_manifest,
)


def freeze_route(args: argparse.Namespace) -> Path:
    catalog_path = args.catalog.resolve()
    if catalog_path.is_symlink() or not catalog_path.is_file():
        raise QwenCloudCatalogError("QwenCloud catalog must be a regular file")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or not verify_content_address(
        catalog, CATALOG_SCHEMA_VERSION
    ):
        raise QwenCloudCatalogError("QwenCloud catalog content address does not verify")
    if args.expected_catalog_sha256 != catalog.get("artifact_sha256"):
        raise QwenCloudCatalogError("QwenCloud catalog differs from the expected digest")
    if args.model_id == "qwen3.8-max":
        manifest = build_unranked_qwen38_alias_route_manifest(
            catalog_artifact=catalog,
            cap_usd=args.cap_usd,
            allow_mutable_alias_exploratory=args.allow_mutable_alias_exploratory,
        )
    else:
        if args.allow_mutable_alias_exploratory:
            raise QwenCloudCatalogError(
                "mutable-alias opt-in cannot be applied to the dated Qwen route"
            )
        manifest = build_unranked_qwen37_route_manifest(
            catalog_artifact=catalog,
            cap_usd=args.cap_usd,
        )
    return write_qwencloud_route_manifest(manifest, args.output_dir)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument(
        "--model-id",
        choices=["qwen3.7-max-2026-06-08", "qwen3.8-max"],
        required=True,
    )
    parser.add_argument("--cap-usd", type=Decimal, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-mutable-alias-exploratory", action="store_true")
    args = parser.parse_args()
    try:
        path = freeze_route(args)
    except (json.JSONDecodeError, OSError, QwenCloudCatalogError) as error:
        raise SystemExit(str(error)) from error
    print(path)


if __name__ == "__main__":
    run()
