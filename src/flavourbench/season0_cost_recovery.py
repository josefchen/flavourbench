"""Recover delayed OpenRouter generation accounting without replaying an arm."""

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

from .execution_policy import assert_legacy_paid_cli_allowed
from .real_task_bank import sha256_json
from .season0_collection import _or_accounting

SCHEMA_VERSION = "flavourbench-season0-cost-correction-v1"


class CostRecoveryError(RuntimeError):
    """A delayed generation cannot be bound safely to a failed scored arm."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise CostRecoveryError(f"expected a JSON object: {path}")
    return value


def _verify(document: Mapping[str, Any], label: str) -> str:
    claimed = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise CostRecoveryError(f"{label} artifact hash mismatch")
    return actual


def _bound_model(arm: Mapping[str, Any], model_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    arm_model = arm.get("model")
    contracts = arm.get("contracts")
    if not isinstance(arm_model, Mapping) or not isinstance(contracts, Mapping):
        raise CostRecoveryError("arm has no model or contract binding")
    manifest_sha = _verify(model_manifest, "model manifest")
    if contracts.get("model_manifest_artifact_sha256") != manifest_sha:
        raise CostRecoveryError("arm is bound to another model manifest")
    matches = [
        model
        for model in model_manifest.get("models", [])
        if isinstance(model, Mapping)
        and model.get("season_model_id") == arm_model.get("season_model_id")
    ]
    if len(matches) != 1 or matches[0].get("provider") != "openrouter":
        raise CostRecoveryError("arm does not resolve to one OpenRouter manifest entry")
    return matches[0]


async def recover_generation(
    *,
    arm: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    generation_id: str,
    output_dir: Path,
    api_key: str,
) -> dict[str, Any]:
    arm_sha = _verify(arm, "source arm")
    model = _bound_model(arm, model_manifest)
    if (
        arm.get("status") != "failed"
        or arm.get("delivery_state") != "uncertain"
        or arm.get("error_type") != "UncertainDeliveryError"
        or generation_id not in str(arm.get("error") or "")
    ):
        raise CostRecoveryError("source arm is not the matching uncertain generation")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "HTTP-Referer": "https://epicure.kaikaku.ai/flavourbench",
        "X-Title": "Epicure FlavourBench Season 0 cost recovery",
    }
    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1/", headers=headers, timeout=120
    ) as client:
        accounting = await _or_accounting(client, generation_id, attempts=3)
    if (
        accounting.get("model") != model.get("canonical_model_id")
        or accounting.get("provider_name") != model.get("provider_name")
        or accounting.get("reconciled") is not True
    ):
        raise CostRecoveryError("recovered generation identity does not match the frozen route")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "correction_type": "delayed_openrouter_generation_accounting",
        "arm_id": arm["arm_id"],
        "source_arm_artifact_sha256": arm_sha,
        "model_manifest_artifact_sha256": model_manifest["artifact_sha256"],
        "provider": "openrouter",
        "generation_id": generation_id,
        "accounting": accounting,
        "retrieved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "response_status_unchanged": True,
        "rank_eligible": False,
        "synthetic": False,
    }
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"cost-correction-{arm['arm_id'][:16]}-{digest}.json"
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(destination)
    return {**document, "correction_path": str(destination)}


def run(argv: Sequence[str] | None = None) -> None:
    assert_legacy_paid_cli_allowed("flavourbench-season0-cost-recovery")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "FLAVOURBENCH_OPENROUTER_API_KEY"
    )
    if not api_key:
        raise CostRecoveryError("OpenRouter API key is not configured")
    result = asyncio.run(
        recover_generation(
            arm=_load(args.arm),
            model_manifest=_load(args.model_manifest),
            generation_id=args.generation_id,
            output_dir=args.output_dir,
            api_key=api_key,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
