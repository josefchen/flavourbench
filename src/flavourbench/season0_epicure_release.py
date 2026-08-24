"""Freeze the exact live Epicure intervention used by FlavourBench Season 0."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .config import get_settings
from .mcp_client import McpSession, tool_catalog_sha256
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-season0-epicure-intervention-v1"
INTERVENTION_ID = "epicure-mcp-opaque-1790-v1"


class EpicureFreezeError(RuntimeError):
    """The live Epicure intervention did not match its frozen evidence."""


def _atomic_write(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise EpicureFreezeError("content-addressed Epicure intervention conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def build_intervention(
    provenance: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
    frozen_tool_catalog: Sequence[Mapping[str, Any]],
    *,
    observed_at: str,
    provenance_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = ("release_id", "bundle_sha256", "application_sha256", "ingredient_count")
    if any(not provenance.get(field) for field in required):
        raise EpicureFreezeError("Epicure provenance omitted a required identity field")
    live_tools = [dict(tool) for tool in tools]
    frozen_tools = [dict(tool) for tool in frozen_tool_catalog]
    live_sha = tool_catalog_sha256(live_tools)
    frozen_sha = tool_catalog_sha256(frozen_tools)
    if live_sha != frozen_sha:
        raise EpicureFreezeError("live Epicure tools differ from the frozen tool catalog")
    if int(provenance["ingredient_count"]) != 1_790:
        raise EpicureFreezeError("Season 0 requires the attested 1,790-ingredient bundle")
    return {
        "schema_version": SCHEMA_VERSION,
        "intervention_id": INTERVENTION_ID,
        "status": "frozen_opaque_domain_tool_intervention",
        "observed_at": observed_at,
        "provenance_source": dict(provenance_source or {"mode": "live_endpoint"}),
        "runtime": {
            "release_id": str(provenance["release_id"]),
            "bundle_sha256": str(provenance["bundle_sha256"]),
            "application_sha256": str(provenance["application_sha256"]),
            "tool_catalog_sha256": live_sha,
            "ingredient_count": int(provenance["ingredient_count"]),
            "embedding_dimensions": int(provenance.get("embedding_dimensions") or 0),
            "tool_count": len(live_tools),
        },
        "lineage_statement": {
            "public_release_match": False,
            "classification": "opaque_proprietary_runtime_snapshot",
            "claim_boundary": (
                "The intervention is identified by application, bundle, and tool hashes. "
                "It is not represented as a public Cooc/Core/Chem release."
            ),
            "similarity_as_ground_truth": False,
            "similarity_use": "explanatory_evidence_only",
        },
        "tools": live_tools,
    }


async def freeze(
    tool_catalog_path: Path, prior_attestation_path: Path | None = None
) -> dict[str, Any]:
    frozen = json.loads(tool_catalog_path.read_bytes())
    if not isinstance(frozen, list) or not all(isinstance(item, Mapping) for item in frozen):
        raise EpicureFreezeError("frozen Epicure tool catalog is invalid")
    settings = get_settings()
    provenance_url = settings.mcp_url.removesuffix("/mcp").rstrip("/") + "/provenance"
    headers = {"Authorization": f"Bearer {settings.mcp_token}"} if settings.mcp_token else {}
    provenance_source: dict[str, Any] = {
        "mode": "live_endpoint",
        "url": provenance_url,
    }
    async with httpx.AsyncClient(headers=headers, timeout=settings.mcp_timeout_seconds) as client:
        response = await client.get(provenance_url)
    if response.status_code == 404 and prior_attestation_path is not None:
        prior = json.loads(prior_attestation_path.read_bytes())
        provenance = prior.get("epicure") if isinstance(prior, Mapping) else None
        provenance_source = {
            "mode": "prior_live_attestation_plus_current_tool_revalidation",
            "prior_artifact_sha256": sha256_json(
                {key: value for key, value in prior.items() if key != "artifact_sha256"}
            ),
            "prior_epicure_sha256": sha256_json(provenance),
            "current_provenance_endpoint_status": 404,
        }
    else:
        response.raise_for_status()
        provenance = response.json()
    if not isinstance(provenance, Mapping):
        raise EpicureFreezeError("Epicure provenance endpoint returned a non-object")
    async with McpSession() as mcp:
        tools = await mcp.list_tools()
    return build_intervention(
        provenance,
        tools,
        frozen,
        observed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        provenance_source=provenance_source,
    )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-catalog", type=Path, required=True)
    parser.add_argument("--prior-attestation", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/season0/epicure"))
    args = parser.parse_args(argv)
    intervention = asyncio.run(freeze(args.tool_catalog, args.prior_attestation))
    path = _atomic_write(args.output_dir, "epicure-intervention", intervention)
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "intervention_id": intervention["intervention_id"],
                "runtime": intervention["runtime"],
                "lineage_statement": intervention["lineage_statement"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
