from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

GPU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GPU_ROOT))

from common.manifest import DeploymentManifest, ManifestError, load_manifest
from lambda_cloud.api import LambdaCloudClient, LambdaCloudError

LIVE_ACK = "I_UNDERSTAND_THIS_CREATES_GPU_SPEND"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="External Lambda Cloud termination watchdog; run outside the GPU VM"
    )
    parser.add_argument("--lease", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--authorization-ticket", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=15)
    return parser


def _parse_time(value: object, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ManifestError(f"lease {field} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestError(f"lease {field} must include a timezone")
    return parsed.astimezone(UTC)


def _load_lease(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read watchdog lease {path}") from exc
    if not isinstance(value, dict):
        raise ManifestError("lease root must be an object")
    if value.get("provider") != "lambda_cloud":
        raise ManifestError("lease provider must be lambda_cloud")
    return value


def _write_lease(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    temporary.replace(path)


def _write_readiness(path: Path, manifest: DeploymentManifest) -> None:
    ready_path = Path(f"{path}.watchdog-ready")
    document = {
        "pid": os.getpid(),
        "checked_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest.spec_sha256,
    }
    temporary = ready_path.with_suffix(ready_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    temporary.replace(ready_path)


def _load_manifest_for_lease(lease: dict[str, Any]) -> DeploymentManifest:
    manifest = load_manifest(str(lease["manifest_path"]), expected_backend="lambda_cloud")
    if manifest.spec_sha256 != lease.get("manifest_sha256"):
        raise ManifestError("lease manifest hash does not match the frozen manifest")
    return manifest


def _live_gate(
    manifest: DeploymentManifest,
    *,
    apply: bool,
    authorization_ticket: str,
) -> None:
    if not apply:
        return
    if os.environ.get("FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED") != LIVE_ACK:
        raise ManifestError(
            "set FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED to the documented acknowledgement"
        )
    manifest.require_live_authorization(authorization_ticket)


def _cost_cents(lease: dict[str, Any], now: datetime) -> int:
    launched_at = _parse_time(lease["launched_at"], "launched_at")
    elapsed_seconds = max(0, math.ceil((now - launched_at).total_seconds()))
    billable_minutes = max(1, math.ceil(elapsed_seconds / 60))
    return math.ceil(int(lease["price_cents_per_hour"]) * billable_minutes / 60)


def _termination_reasons(lease: dict[str, Any], now: datetime) -> list[str]:
    reasons: list[str] = []
    if now >= _parse_time(lease["terminate_at"], "terminate_at"):
        reasons.append("absolute runtime deadline reached")
    if _cost_cents(lease, now) >= int(lease["maximum_cost_cents"]):
        reasons.append("hard monetary cap reached")
    return reasons


def _tags(instance: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("key")): str(item.get("value"))
        for item in instance.get("tags", [])
        if isinstance(item, dict)
    }


def _provider_safety_reasons(instance: dict[str, Any], lease: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    status_value = str(instance.get("status", "unknown"))
    if status_value in {"unhealthy", "preempted"}:
        reasons.append(f"provider reported {status_value}")
    tags = _tags(instance)
    if tags.get("flavourbench-manifest") != lease["manifest_sha256"]:
        reasons.append("instance manifest tag does not match lease")
    return reasons


def _tagged_instances(
    client: LambdaCloudClient,
    manifest: DeploymentManifest,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for instance in client.instances():
        if str(instance.get("status")) in {"terminated", "terminating"}:
            continue
        if _tags(instance).get("flavourbench-manifest") == manifest.spec_sha256:
            candidates.append(instance)
    return candidates


def _tick(
    lease_path: Path,
    lease: dict[str, Any],
    client: LambdaCloudClient | None,
    *,
    apply: bool,
) -> bool:
    now = datetime.now(UTC)
    reasons = _termination_reasons(lease, now)
    instance: dict[str, Any] | None = None
    if client is not None:
        instance = client.instance(str(lease["instance_id"]))
        status_value = str(instance.get("status", "unknown"))
        if status_value in {"terminated", "terminating"}:
            lease["state"] = status_value
            lease["watchdog_checked_at"] = now.isoformat()
            _write_lease(lease_path, lease)
            print(f"instance is {status_value}; watchdog complete")
            return True
        reasons.extend(_provider_safety_reasons(instance, lease))

    cost = _cost_cents(lease, now)
    print(
        f"{now.isoformat()} instance={lease['instance_id']} attributed_cost_cents={cost} "
        f"reasons={reasons or ['none']}"
    )
    if not reasons:
        return False
    if not apply:
        print("DRY RUN: watchdog would invoke Lambda Cloud terminate now")
        return True
    if client is None:
        raise LambdaCloudError("live watchdog has no Lambda Cloud client")

    result = client.terminate([str(lease["instance_id"])])
    lease["state"] = "termination_requested"
    lease["termination_requested_at"] = now.isoformat()
    lease["termination_reasons"] = sorted(set(reasons))
    lease["termination_response"] = result
    _write_lease(lease_path, lease)
    print("termination requested through Lambda Cloud API")
    return True


def main() -> int:
    args = _parser().parse_args()
    if not 5 <= args.interval_seconds <= 15:
        raise SystemExit("interval must be between 5 and 15 seconds")
    lease_path = Path(args.lease).resolve()
    try:
        if lease_path.exists():
            lease = _load_lease(lease_path)
            manifest = _load_manifest_for_lease(lease)
        else:
            if not args.loop or not args.manifest:
                raise ManifestError(
                    "a missing lease requires --loop and --manifest so the watchdog can arm first"
                )
            manifest = load_manifest(args.manifest, expected_backend="lambda_cloud")
            lease = {}
        _live_gate(
            manifest,
            apply=args.apply,
            authorization_ticket=args.authorization_ticket,
        )
        client = None
        if args.apply:
            client = LambdaCloudClient(os.environ.get("LAMBDA_API_KEY", ""))
        orphan_first_seen: datetime | None = None
        while True:
            _write_readiness(lease_path, manifest)
            if not lease_path.exists():
                if client is not None:
                    orphans = _tagged_instances(client, manifest)
                    if orphans and orphan_first_seen is None:
                        orphan_first_seen = datetime.now(UTC)
                        print(
                            "tagged instance appeared before its lease; allowing 30 seconds for "
                            "atomic lease persistence"
                        )
                    elif (
                        orphans
                        and orphan_first_seen is not None
                        and datetime.now(UTC) - orphan_first_seen >= timedelta(seconds=30)
                    ):
                        ids = [str(instance["id"]) for instance in orphans]
                        client.terminate(ids)
                        print(f"terminated orphaned tagged instances with no lease: {ids}")
                        return 0
                    elif not orphans:
                        orphan_first_seen = None
                print(f"watchdog armed; waiting for lease {lease_path}")
                time.sleep(args.interval_seconds)
                continue
            lease = _load_lease(lease_path)
            if _load_manifest_for_lease(lease).spec_sha256 != manifest.spec_sha256:
                raise ManifestError("new lease does not match the armed watchdog manifest")
            if _tick(lease_path, lease, client, apply=args.apply):
                return 0
            if not args.loop:
                print("single dry check complete; no provider mutation was made")
                return 0
            time.sleep(args.interval_seconds)
    except (ManifestError, LambdaCloudError) as exc:
        raise SystemExit(f"watchdog stopped: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
