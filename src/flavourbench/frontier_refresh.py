"""Freeze and smoke-test a dated frontier-model refresh without altering Season 0."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .execution_policy import assert_legacy_paid_cli_allowed
from .real_task_bank import sha256_json, sha256_text
from .season0_compatibility import (
    _smoke_target as smoke_bedrock_target,
)
from .season0_compatibility import (
    load_find_pairings as load_bedrock_tool,
)
from .season0_compatibility import (
    load_targets as load_bedrock_targets,
)
from .season0_openrouter_compatibility import (
    _headers as openrouter_headers,
)
from .season0_openrouter_compatibility import (
    _smoke as smoke_openrouter_target,
)
from .season0_openrouter_compatibility import (
    load_find_pairings as load_openrouter_tool,
)
from .season0_openrouter_compatibility import (
    load_targets as load_openrouter_targets,
)
from .season0_openrouter_routes import snapshot as snapshot_openrouter_routes

ROUTE_SCHEMA = "flavourbench-frontier-refresh-route-catalog-v1"
BEDROCK_ARM_SCHEMA = "flavourbench-frontier-refresh-bedrock-contract-smoke-v1"
OPENROUTER_ARM_SCHEMA = "flavourbench-frontier-refresh-openrouter-contract-smoke-v1"
SUMMARY_SCHEMA = "flavourbench-frontier-refresh-contract-summary-v1"
CONFIRMATION = "RUN_REAL_FRONTIER_REFRESH_CONTRACT_SMOKES_V1"


class FrontierRefreshError(RuntimeError):
    """The frontier refresh could not be frozen or smoke-tested safely."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise FrontierRefreshError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise FrontierRefreshError(f"expected a JSON object: {path}")
    return value


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
            raise FrontierRefreshError("content-addressed artifact conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _redacted_error(error: Exception) -> str:
    value = re.sub(r"(?<!\d)\d{12}(?!\d)", "<account-redacted>", str(error))
    value = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "<credential-redacted>", value)
    return value[:600]


def _validate_roster(roster: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    slots = roster.get("slots")
    if roster.get("schema_version") != "flavourbench-frontier-refresh-roster-v1":
        raise FrontierRefreshError("unexpected frontier-refresh roster schema")
    if not isinstance(slots, list) or not slots:
        raise FrontierRefreshError("frontier-refresh roster has no slots")
    ids: set[str] = set()
    endpoints: set[str] = set()
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise FrontierRefreshError("frontier-refresh roster contains a non-object slot")
        snapshot_id = str(slot.get("snapshot_model_id") or "")
        endpoint_id = str(slot.get("endpoint_id") or "")
        provider = str(slot.get("provider") or "")
        if not snapshot_id or snapshot_id in ids:
            raise FrontierRefreshError("frontier-refresh model IDs must be present and unique")
        if not endpoint_id or endpoint_id in endpoints:
            raise FrontierRefreshError("frontier-refresh endpoints must be present and unique")
        if provider not in {"bedrock", "openrouter"}:
            raise FrontierRefreshError(f"unsupported provider in refresh roster: {provider}")
        ids.add(snapshot_id)
        endpoints.add(endpoint_id)
    return slots


async def freeze_routes(roster_path: Path, output_dir: Path) -> Path:
    roster = _load(roster_path)
    _validate_roster(roster)
    catalog = await snapshot_openrouter_routes(roster_path)
    payload = {
        **{key: value for key, value in catalog.items() if key != "schema_version"},
        "schema_version": ROUTE_SCHEMA,
        "snapshot_label": str(roster.get("snapshot_label") or ""),
        "status": "frontier_refresh_exact_routes_pending_live_smoke",
        "official": False,
        "rank_eligible": False,
    }
    return _atomic_write(output_dir, "frontier-refresh-route-catalog", payload)


async def _run_bedrock_smokes(
    *,
    roster_path: Path,
    catalog_path: Path,
    tool_catalog_path: Path,
    output_dir: Path,
    concurrency: int,
) -> tuple[list[dict[str, Any]], int]:
    targets = load_bedrock_targets(roster_path, catalog_path)
    tool, tool_sha = load_bedrock_tool(tool_catalog_path)
    settings = BedrockLaneSettings.from_environ()
    if not settings.enabled or not settings.live_authorized:
        raise FrontierRefreshError("Bedrock live authorization is not enabled")
    runtime = create_boto3_clients(settings).runtime
    semaphore = asyncio.Semaphore(concurrency)

    async def one(target: Any) -> tuple[Any, dict[str, Any]]:
        async with semaphore:
            try:
                payload = await smoke_bedrock_target(
                    runtime,
                    target,
                    tool,
                    tool_sha,
                    artifact_schema_version=BEDROCK_ARM_SCHEMA,
                    request_phase="frontier_refresh_20260728_contract_smoke",
                )
            except Exception as error:  # noqa: BLE001 - immutable redacted failure evidence
                payload = {
                    "schema_version": BEDROCK_ARM_SCHEMA,
                    "status": "failed",
                    "display_name": target.display_name,
                    "requested_target_id": target.target_id,
                    "catalog_sha256": target.catalog_sha256,
                    "error_type": type(error).__name__,
                    "error": _redacted_error(error),
                    "official": False,
                    "rank_eligible": False,
                }
            return target, payload

    outcomes = await asyncio.gather(*(one(target) for target in targets))
    artifacts: list[dict[str, Any]] = []
    provider_calls = 0
    for target, payload in outcomes:
        path = _atomic_write(
            output_dir,
            f"bedrock-{sha256_text(target.target_id)[:12]}",
            payload,
        )
        provider_calls += int(payload.get("provider_calls") or 0)
        artifacts.append(
            {
                "provider": "bedrock",
                "requested_endpoint_id": target.target_id,
                "display_name": target.display_name,
                "status": payload["status"],
                "artifact_path": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "provider_calls": int(payload.get("provider_calls") or 0),
                "epicure_calls": int(payload.get("real_epicure_calls") or 0),
                "cost_status": "usage_recorded_rate_card_pending",
                "error_type": payload.get("error_type"),
            }
        )
    return artifacts, provider_calls


async def _run_openrouter_smokes(
    *,
    roster_path: Path,
    route_catalog_path: Path,
    tool_catalog_path: Path,
    output_dir: Path,
    base_url: str,
) -> tuple[list[dict[str, Any]], int, Decimal]:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "FLAVOURBENCH_OPENROUTER_API_KEY"
    )
    if not api_key:
        raise FrontierRefreshError("OpenRouter API key is not configured")
    gateway_token = os.environ.get("CLOUDFLARE_AI_GATEWAY_TOKEN") or ""
    targets = load_openrouter_targets(roster_path, route_catalog_path)
    tool, tool_sha = load_openrouter_tool(tool_catalog_path)
    title = "Epicure FlavourBench frontier refresh 2026-07-28"
    headers = openrouter_headers(
        api_key,
        gateway_token,
        base_url,
        request_title=title,
    )
    accounting_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "HTTP-Referer": "https://epicure.kaikaku.ai/flavourbench",
        "X-Title": title,
    }
    artifacts: list[dict[str, Any]] = []
    total_cost = Decimal(0)
    provider_calls = 0
    async with (
        httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=240,
        ) as generation_client,
        httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1/",
            headers=accounting_headers,
            timeout=120,
        ) as accounting_client,
    ):
        for target in targets:
            try:
                payload = await smoke_openrouter_target(
                    target,
                    tool,
                    tool_sha,
                    generation_client,
                    accounting_client,
                    artifact_schema_version=OPENROUTER_ARM_SCHEMA,
                    require_structured_output=True,
                )
            except Exception as error:  # noqa: BLE001 - immutable redacted failure evidence
                payload = {
                    "schema_version": OPENROUTER_ARM_SCHEMA,
                    "status": "failed",
                    "display_name": target.display_name,
                    "requested_model_id": target.model_id,
                    "canonical_slug": target.canonical_slug,
                    "requested_provider_slug": target.provider_slug,
                    "source_manifest_sha256": target.source_manifest_sha256,
                    "endpoint_document_sha256": target.endpoint_document_sha256,
                    "error_type": type(error).__name__,
                    "error": _redacted_error(error),
                    "official": False,
                    "rank_eligible": False,
                }
            path = _atomic_write(
                output_dir,
                f"openrouter-{sha256_text(target.model_id)[:12]}",
                payload,
            )
            cost = Decimal(str(payload.get("cost_usd") or "0"))
            total_cost += cost
            provider_calls += int(payload.get("real_provider_calls") or 0)
            artifacts.append(
                {
                    "provider": "openrouter",
                    "requested_endpoint_id": target.model_id,
                    "requested_provider_slug": target.provider_slug,
                    "display_name": target.display_name,
                    "status": payload["status"],
                    "artifact_path": str(path),
                    "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                    "provider_calls": int(payload.get("real_provider_calls") or 0),
                    "epicure_calls": int(payload.get("real_epicure_calls") or 0),
                    "cost_usd": format(cost, "f"),
                    "cost_status": (
                        "openrouter_generation_metadata_reconciled"
                        if payload.get("generation_costs_reconciled") is True
                        else "no_reconciled_cost"
                    ),
                    "error_type": payload.get("error_type"),
                }
            )
    return artifacts, provider_calls, total_cost


async def run_smokes(
    *,
    roster_path: Path,
    bedrock_catalog_path: Path,
    route_catalog_path: Path,
    tool_catalog_path: Path,
    output_dir: Path,
    base_url: str,
    cap_usd: Decimal,
    concurrency: int,
    providers: frozenset[str],
) -> Path:
    roster = _load(roster_path)
    slots = _validate_roster(roster)
    if not providers or not providers.issubset({"bedrock", "openrouter"}):
        raise FrontierRefreshError("providers must contain bedrock and/or openrouter")
    selected_slots = [slot for slot in slots if str(slot.get("provider")) in providers]
    if not selected_slots:
        raise FrontierRefreshError("the selected providers have no roster endpoints")
    if not 0 < cap_usd <= Decimal("20"):
        raise FrontierRefreshError("contract-smoke cap must be in (0, 20]")
    if not 1 <= concurrency <= 4:
        raise FrontierRefreshError("contract-smoke concurrency must be between one and four")
    if Decimal(len(selected_slots)) * Decimal("2") > cap_usd:
        raise FrontierRefreshError("$2-per-endpoint conservative reserve exceeds the smoke cap")
    bedrock_artifacts: list[dict[str, Any]] = []
    bedrock_calls = 0
    if "bedrock" in providers:
        bedrock_artifacts, bedrock_calls = await _run_bedrock_smokes(
            roster_path=roster_path,
            catalog_path=bedrock_catalog_path,
            tool_catalog_path=tool_catalog_path,
            output_dir=output_dir,
            concurrency=concurrency,
        )
    openrouter_artifacts: list[dict[str, Any]] = []
    openrouter_calls = 0
    openrouter_cost = Decimal(0)
    if "openrouter" in providers:
        openrouter_artifacts, openrouter_calls, openrouter_cost = await _run_openrouter_smokes(
            roster_path=roster_path,
            route_catalog_path=route_catalog_path,
            tool_catalog_path=tool_catalog_path,
            output_dir=output_dir,
            base_url=base_url,
        )
    artifacts = bedrock_artifacts + openrouter_artifacts
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "snapshot_label": str(roster.get("snapshot_label") or ""),
        "providers": sorted(providers),
        "status": (
            "all_contract_smokes_passed"
            if all(item["status"] == "smoke_passed" for item in artifacts)
            else "one_or_more_contract_smokes_failed"
        ),
        "roster_sha256": sha256_json(roster),
        "bedrock_catalog_sha256": str(_load(bedrock_catalog_path).get("catalog_sha256") or ""),
        "openrouter_route_catalog_artifact_sha256": str(
            _load(route_catalog_path).get("artifact_sha256") or ""
        ),
        "counts": {
            "endpoints": len(artifacts),
            "smoke_passed": sum(item["status"] == "smoke_passed" for item in artifacts),
            "failed": sum(item["status"] != "smoke_passed" for item in artifacts),
            "real_provider_calls": bedrock_calls + openrouter_calls,
            "real_epicure_calls": sum(int(item["epicure_calls"]) for item in artifacts),
        },
        "cost": {
            "openrouter_reconciled_usd": format(openrouter_cost, "f"),
            "bedrock": "usage_recorded_rate_card_pending",
            "smoke_cap_usd": format(cap_usd, "f"),
        },
        "artifacts": artifacts,
        "official": False,
        "rank_eligible": False,
    }
    return _atomic_write(output_dir, "frontier-refresh-contract-summary", payload)


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    routes = subparsers.add_parser("freeze-routes")
    routes.add_argument("--roster", type=Path, required=True)
    routes.add_argument("--output-dir", type=Path, required=True)
    smokes = subparsers.add_parser("run-smokes")
    smokes.add_argument("--roster", type=Path, required=True)
    smokes.add_argument("--bedrock-catalog", type=Path, required=True)
    smokes.add_argument("--route-catalog", type=Path, required=True)
    smokes.add_argument("--tool-catalog", type=Path, required=True)
    smokes.add_argument("--output-dir", type=Path, required=True)
    smokes.add_argument("--base-url", required=True)
    smokes.add_argument("--cap-usd", type=Decimal, default=Decimal("10"))
    smokes.add_argument("--concurrency", type=int, default=2)
    smokes.add_argument(
        "--provider",
        action="append",
        choices=("bedrock", "openrouter"),
        dest="providers",
        help="Run only this provider. Repeat to select both; defaults to both.",
    )
    smokes.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-routes":
        path = asyncio.run(freeze_routes(args.roster, args.output_dir))
        print(
            json.dumps(
                {
                    "operation": "freeze_routes",
                    "output": str(path),
                    "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                    "inference_calls": 0,
                },
                indent=2,
            )
        )
        return
    assert_legacy_paid_cli_allowed("flavourbench-frontier-refresh")
    if args.confirmation != CONFIRMATION:
        raise FrontierRefreshError(f"live smokes require --confirmation {CONFIRMATION}")
    path = asyncio.run(
        run_smokes(
            roster_path=args.roster,
            bedrock_catalog_path=args.bedrock_catalog,
            route_catalog_path=args.route_catalog,
            tool_catalog_path=args.tool_catalog,
            output_dir=args.output_dir,
            base_url=args.base_url,
            cap_usd=args.cap_usd,
            concurrency=args.concurrency,
            providers=frozenset(args.providers or ("bedrock", "openrouter")),
        )
    )
    print(json.dumps(_load(path), indent=2))
    print(json.dumps({"summary_path": str(path)}, indent=2))


if __name__ == "__main__":
    run()
