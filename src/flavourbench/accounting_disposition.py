"""Reconcile an absent OpenRouter generation against an immutable run receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

from .config import get_settings
from .matched_protocol_preflight import _verified_live_smoke

SCHEMA_VERSION = "flavourbench-openrouter-accounting-disposition-v1"
CONFIRMATION = "VERIFY_ACCOUNTING_ONLY"


class AccountingDispositionError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _decimal(value: object, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise AccountingDispositionError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise AccountingDispositionError(f"{field} is invalid")
    return parsed


def _generation_ids(artifact: dict[str, Any]) -> tuple[list[str], list[str]]:
    reconciled: list[str] = []
    for result in (artifact.get("results") or {}).values():
        if not isinstance(result, dict):
            continue
        reconciled.extend(
            str(item.get("generation_id") or "")
            for item in result.get("generation_metadata") or []
            if isinstance(item, dict) and item.get("reconciled") is True
        )
    unresolved = [
        str(item.get("generation_id") or "")
        for item in artifact.get("incomplete_generation_metadata") or []
        if isinstance(item, dict) and item.get("reconciled") is not True
    ]
    if not reconciled or any(not value for value in [*reconciled, *unresolved]):
        raise AccountingDispositionError("source receipt has incomplete generation identifiers")
    if len(set([*reconciled, *unresolved])) != len([*reconciled, *unresolved]):
        raise AccountingDispositionError("source receipt repeats a generation identifier")
    return reconciled, unresolved


def build_disposition(
    *,
    source_artifact_path: Path,
    unresolved_generation_id: str,
    client: httpx.Client,
) -> dict[str, Any]:
    artifact = _verified_live_smoke(source_artifact_path)
    reconciled_ids, unresolved_ids = _generation_ids(artifact)
    if unresolved_ids != [unresolved_generation_id]:
        raise AccountingDispositionError(
            "requested generation is not the receipt's sole unresolved identifier"
        )
    budget = artifact.get("budget") or {}
    before = _decimal(
        (budget.get("openrouter_key_before") or {}).get("usage_daily_usd"),
        "usage before",
    )
    after = _decimal(
        (budget.get("openrouter_key_after") or {}).get("usage_daily_usd"),
        "usage after",
    )
    account_delta = after - before
    if account_delta < 0:
        raise AccountingDispositionError("account usage moved backwards")

    rows: list[dict[str, Any]] = []
    accounted_total = Decimal(0)
    for generation_id in [*reconciled_ids, unresolved_generation_id]:
        response = client.get("generation", params={"id": generation_id})
        if response.status_code == 404:
            rows.append(
                {
                    "generation_id": generation_id,
                    "http_status": 404,
                    "provider_record_present": False,
                }
            )
            continue
        response.raise_for_status()
        data = response.json().get("data") or {}
        cost = _decimal(data.get("total_cost"), "generation total cost")
        accounted_total += cost
        rows.append(
            {
                "generation_id": generation_id,
                "http_status": response.status_code,
                "provider_record_present": True,
                "total_cost_usd": format(cost, "f"),
                "model": str(data.get("model") or "unknown"),
                "provider": str(data.get("provider_name") or "unknown"),
                "finish_reason": str(data.get("finish_reason") or "unknown"),
            }
        )
    target = next(row for row in rows if row["generation_id"] == unresolved_generation_id)
    if target["provider_record_present"] is not False:
        raise AccountingDispositionError(
            "generation now has provider metadata; use ordinary generation reconciliation"
        )
    if accounted_total != account_delta:
        raise AccountingDispositionError(
            "known provider generation costs do not exactly explain the account usage delta"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "account_scope_reconciled_generation_identity_unresolved",
        "official": False,
        "rank_eligible": False,
        "source_artifact_filename": source_artifact_path.name,
        "source_artifact_sha256": artifact["artifact_sha256"],
        "source_run_id": artifact.get("run_id"),
        "unresolved_generation_id": unresolved_generation_id,
        "provider_query_base_url": "https://openrouter.ai/api/v1",
        "account_usage": {
            "before_usd": format(before, "f"),
            "after_usd": format(after, "f"),
            "delta_usd": format(account_delta, "f"),
        },
        "known_generation_cost_total_usd": format(accounted_total, "f"),
        "unresolved_generation_incremental_cost_usd": "0",
        "generation_queries": rows,
        "disposition": {
            "budget_accounting": "closed_at_provider_account_scope",
            "generation_identity": "unresolved_provider_record_absent",
            "source_preflight_eligible": False,
            "source_preflight_may_be_retried": False,
            "new_protocol_requires_new_work_item_ids": True,
        },
        "provider_generation_calls_made": 0,
    }
    return {**payload, "artifact_sha256": _sha256(payload)}


def _write(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"openrouter-accounting-disposition-{payload['artifact_sha256']}.json"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise AccountingDispositionError("content-addressed disposition conflicts")
        return path
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    arguments = parser.parse_args(argv)
    if arguments.confirm != CONFIRMATION:
        raise AccountingDispositionError(f"requires --confirm {CONFIRMATION}")
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise AccountingDispositionError("OpenRouter credential is unavailable")
    with httpx.Client(
        base_url=settings.openrouter_accounting_base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        timeout=settings.openrouter_timeout_seconds,
    ) as client:
        payload = build_disposition(
            source_artifact_path=arguments.source_artifact,
            unresolved_generation_id=arguments.generation_id,
            client=client,
        )
    path = _write(arguments.output_dir, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "artifact": str(path),
                "artifact_sha256": payload["artifact_sha256"],
                "provider_generation_calls_made": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
