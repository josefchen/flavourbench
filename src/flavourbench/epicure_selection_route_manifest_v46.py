"""Freeze replication-2 routes with only Qwen A95B replaced by Alibaba."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_route_manifest_v43 import verify_manifest as verify_manifest_v43
from .epicure_selection_route_manifest_v45 import (
    QWEN_MODEL_ID,
    ROUTE_SPECS,
)
from .epicure_selection_route_manifest_v45 import (
    verify_manifest as verify_manifest_v45,
)
from .frontier_manifest import verify_manifest_content_address

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-replication-2-route-v46"


class SelectionRouteManifestV46Error(RuntimeError):
    """The replication-2 route manifest failed verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionRouteManifestV46Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionRouteManifestV46Error("manifest input is not a JSON object")
    return value


def build(
    *,
    source: Mapping[str, Any],
    source_physical_sha256: str,
    recovery: Mapping[str, Any],
    recovery_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_manifest_v43(source) or not verify_manifest_v45(recovery):
        raise SelectionRouteManifestV46Error("route predecessor failed verification")
    source_rows = {str(row["model"]["id"]): row for row in source["models"]}
    recovery_rows = {str(row["model"]["id"]): row for row in recovery["models"]}
    if set(source_rows) != set(recovery_rows) or QWEN_MODEL_ID not in source_rows:
        raise SelectionRouteManifestV46Error("route predecessors have different rosters")
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    rows = {str(row["model"]["id"]): row for row in document["models"]}
    rows[QWEN_MODEL_ID].clear()
    rows[QWEN_MODEL_ID].update(copy.deepcopy(recovery_rows[QWEN_MODEL_ID]))
    spec = ROUTE_SPECS[QWEN_MODEL_ID]
    document["status"] = "unranked_candidate"
    document["replication_route_v46"] = {
        "schema_version": REFRESH_SCHEMA_VERSION,
        "replication_index": 2,
        "source_manifest": {
            "semantic_sha256": source["content_address"]["digest"],
            "physical_sha256": source_physical_sha256,
        },
        "qwen_recovery_manifest": {
            "semantic_sha256": recovery["content_address"]["digest"],
            "physical_sha256": recovery_physical_sha256,
        },
        "changed_model_ids": [QWEN_MODEL_ID],
        "selected_exact_tag": spec["tag"],
        "selected_provider": spec["provider"],
        "reasoning_effort": spec["reasoning_effort"],
        "automatic_fallback": False,
        "all_other_model_entries_byte_preserved": True,
        "selection_uses_status_and_finish_metadata_only": True,
        "quality_scores_or_selections_used": False,
        "first_panel_responses_reused": False,
    }
    digest = _sha256(document)
    document["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest(document):
        raise SelectionRouteManifestV46Error("constructed route manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        refresh = document["replication_route_v46"]
        rows = {str(row["model"]["id"]): row for row in document["models"]}
        qwen = rows[QWEN_MODEL_ID]
        spec = ROUTE_SPECS[QWEN_MODEL_ID]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("status") == "unranked_candidate"
        and verify_manifest_content_address(document)
        and len(rows) == 26
        and qwen["endpoint"].get("tag") == spec["tag"]
        and qwen["endpoint"].get("provider_name") == spec["provider"]
        and qwen["request_policy"]["provider"].get("only") == [spec["tag"]]
        and qwen["request_policy"]["provider"].get("allow_fallbacks") is False
        and refresh.get("replication_index") == 2
        and refresh.get("changed_model_ids") == [QWEN_MODEL_ID]
        and refresh.get("selected_exact_tag") == spec["tag"]
        and refresh.get("automatic_fallback") is False
        and refresh.get("all_other_model_entries_byte_preserved") is True
        and refresh.get("selection_uses_status_and_finish_metadata_only") is True
        and refresh.get("quality_scores_or_selections_used") is False
        and refresh.get("first_panel_responses_reused") is False
        and all(
            isinstance((refresh.get(label) or {}).get("semantic_sha256"), str)
            and isinstance((refresh.get(label) or {}).get("physical_sha256"), str)
            for label in ("source_manifest", "qwen_recovery_manifest")
        )
        and document.get("generation_calls_made") == 0
        and document.get("official_results_authorised") is False
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-frontier-refresh-26-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV46Error("content-addressed manifest conflict")
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
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    source = _load(args.source_manifest)
    recovery = _load(args.recovery_manifest)
    document = build(
        source=source,
        source_physical_sha256=_sha256_file(args.source_manifest),
        recovery=recovery,
        recovery_physical_sha256=_sha256_file(args.recovery_manifest),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
