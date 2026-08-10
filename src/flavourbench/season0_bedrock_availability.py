"""Freeze a credential-safe Bedrock model-availability diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-season0-bedrock-availability-v1"
CONFIRMATION = "RUN_REAL_SEASON0_BEDROCK_AVAILABILITY_V1"
STATUS_FIELDS = (
    "authorizationStatus",
    "entitlementAvailability",
    "regionAvailability",
)


class BedrockAvailabilityError(RuntimeError):
    """The control-plane diagnostic was malformed or unauthorized."""


def _atomic_write(directory: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"bedrock-availability-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise BedrockAvailabilityError("availability artifact content-address conflict")
        return destination
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def collect_availability(
    *,
    control: Any,
    model_ids: Sequence[str],
    region: str,
    observed_at: str,
) -> dict[str, Any]:
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise BedrockAvailabilityError("model IDs must be non-empty and unique")
    models = []
    for model_id in model_ids:
        response = control.get_foundation_model_availability(modelId=model_id)
        agreement = response.get("agreementAvailability")
        agreement_status = agreement.get("status") if isinstance(agreement, Mapping) else None
        statuses = {field: response.get(field) for field in STATUS_FIELDS}
        if not isinstance(agreement_status, str) or not all(
            isinstance(value, str) for value in statuses.values()
        ):
            raise BedrockAvailabilityError("Bedrock returned an incomplete availability contract")
        models.append(
            {
                "model_id": model_id,
                "agreement_status": agreement_status,
                "authorization_status": statuses["authorizationStatus"],
                "entitlement_availability": statuses["entitlementAvailability"],
                "region_availability": statuses["regionAvailability"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at": observed_at,
        "aws_region": region,
        "models": models,
        "inference_calls": 0,
        "official": False,
        "rank_eligible": False,
        "privacy": {
            "contains_account_id": False,
            "contains_credentials": False,
            "contains_response_metadata": False,
        },
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    if args.confirmation != CONFIRMATION:
        raise BedrockAvailabilityError(f"live query requires --confirmation {CONFIRMATION}")
    settings = BedrockLaneSettings.from_environ()
    if not settings.enabled or not settings.live_authorized:
        raise BedrockAvailabilityError("Bedrock live authorization is required")
    clients = create_boto3_clients(settings)
    payload = collect_availability(
        control=clients.control,
        model_ids=args.model_id,
        region=clients.region,
        observed_at=datetime.now(UTC).isoformat(),
    )
    path = _atomic_write(args.output_dir, payload)
    print(
        json.dumps(
            {
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "models": payload["models"],
                "inference_calls": 0,
                "path": str(path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
