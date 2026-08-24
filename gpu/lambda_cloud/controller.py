from __future__ import annotations

import argparse
import base64
import json
import math
import os
import secrets
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

GPU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GPU_ROOT))

from common.manifest import DeploymentManifest, ManifestError, load_manifest
from lambda_cloud.api import LambdaCloudClient, LambdaCloudError

LIVE_ACK = "I_UNDERSTAND_THIS_CREATES_GPU_SPEND"
TEMPLATE_PATH = GPU_ROOT / "lambda_cloud" / "cloud-init.yaml.tmpl"
GATEWAY_PATH = GPU_ROOT / "common" / "gateway.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and control one immutable Lambda Cloud inference epoch"
    )
    parser.add_argument(
        "command",
        choices=("plan", "render-cloud-init", "launch", "status", "terminate"),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-runtime-minutes", type=int, default=30)
    parser.add_argument("--lease")
    parser.add_argument("--instance-id")
    parser.add_argument("--authorization-ticket", default="")
    parser.add_argument("--apply", action="store_true")
    return parser


def _encode(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return base64.b64encode(raw).decode()


def _maximum_cost_cents(manifest: DeploymentManifest, runtime_minutes: int) -> int:
    price = int(manifest.document["cost"]["price_cents_per_hour"])
    return math.ceil(price * runtime_minutes / 60)


def _validate_runtime(manifest: DeploymentManifest, runtime_minutes: int) -> int:
    if runtime_minutes <= 0:
        raise ManifestError("max runtime must be positive")
    cost = manifest.document["cost"]
    if runtime_minutes > int(cost["maximum_runtime_minutes"]):
        raise ManifestError("requested runtime exceeds the frozen runtime cap")
    maximum_cost = _maximum_cost_cents(manifest, runtime_minutes)
    if maximum_cost > int(cost["maximum_epoch_cost_cents"]):
        raise ManifestError("projected epoch cost exceeds the frozen monetary cap")
    return maximum_cost


def _live_gate(manifest: DeploymentManifest, args: argparse.Namespace) -> None:
    if not args.apply:
        raise ManifestError("mutating commands require --apply")
    if os.environ.get("FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED") != LIVE_ACK:
        raise ManifestError(
            "set FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED to the documented acknowledgement"
        )
    manifest.require_live_authorization(args.authorization_ticket)


def _start_script(image: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
set -a
source /etc/flavourbench/runtime.env
set +a
VLLM_ARGS_JSON="$(tr -d '\\n' </opt/flavourbench/vllm-args.json)"
export VLLM_ARGS_JSON
exec /usr/bin/docker run --rm --gpus all --network host \\
  --name flavourbench-vllm \\
  --env-file /etc/flavourbench/runtime.env \\
  -e FLAVOURBENCH_VLLM_ARGS_JSON="$VLLM_ARGS_JSON" \\
  -v /opt/flavourbench:/opt/flavourbench:ro \\
  -v /var/lib/flavourbench/huggingface:/root/.cache/huggingface \\
  --entrypoint python \\
  {image} /opt/flavourbench/gateway.py
"""


def _systemd_unit() -> str:
    return """[Unit]
Description=FlavourBench immutable vLLM endpoint
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/sbin/flavourbench-gpu-start
Restart=on-failure
RestartSec=10
TimeoutStopSec=90
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
"""


def render_cloud_init(
    manifest: DeploymentManifest,
    *,
    gateway_key: str,
    internal_key: str,
) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    vllm_args = json.dumps(
        manifest.vllm_args(host="127.0.0.1", port=8001), separators=(",", ":")
    )
    runtime_env = "\n".join(
        [
            "FLAVOURBENCH_DEPLOYMENT_MANIFEST=/opt/flavourbench/deployment-manifest.json",
            "FLAVOURBENCH_GATEWAY_HOST=127.0.0.1",
            "FLAVOURBENCH_GATEWAY_PORT=8000",
            "FLAVOURBENCH_INTERNAL_VLLM_PORT=8001",
            f"FLAVOURBENCH_GATEWAY_API_KEY={gateway_key}",
            f"FLAVOURBENCH_INTERNAL_VLLM_API_KEY={internal_key}",
            "HF_HUB_DISABLE_TELEMETRY=1",
        ]
    ) + "\n"
    values = {
        "__MANIFEST_B64__": _encode(
            json.dumps(manifest.document, sort_keys=True, separators=(",", ":"))
        ),
        "__GATEWAY_B64__": _encode(GATEWAY_PATH.read_bytes()),
        "__VLLM_ARGS_B64__": _encode(vllm_args),
        "__RUNTIME_ENV_B64__": _encode(runtime_env),
        "__START_SCRIPT_B64__": _encode(
            _start_script(manifest.document["runtime"]["container_image"])
        ),
        "__SYSTEMD_UNIT_B64__": _encode(_systemd_unit()),
    }
    rendered = template
    for marker, value in values.items():
        rendered = rendered.replace(marker, value)
    if "__" in rendered:
        raise ManifestError("cloud-init template still contains unresolved markers")
    return rendered


def _write_lease(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    temporary.replace(path)


def _require_armed_watchdog(lease_path: Path, manifest: DeploymentManifest) -> None:
    ready_path = Path(f"{lease_path}.watchdog-ready")
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(
            f"external watchdog is not armed; expected fresh readiness file {ready_path}"
        ) from exc
    if not isinstance(ready, dict) or ready.get("manifest_sha256") != manifest.spec_sha256:
        raise ManifestError("watchdog readiness does not match the frozen manifest")
    try:
        checked_at = datetime.fromisoformat(str(ready.get("checked_at")))
    except ValueError as exc:
        raise ManifestError("watchdog readiness timestamp is invalid") from exc
    if checked_at.tzinfo is None:
        raise ManifestError("watchdog readiness timestamp has no timezone")
    if datetime.now(UTC) - checked_at.astimezone(UTC) > timedelta(seconds=30):
        raise ManifestError("external watchdog readiness heartbeat is stale")
    pid = int(ready.get("pid", 0))
    if pid <= 0:
        raise ManifestError("watchdog readiness has no process ID")
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise ManifestError("watchdog process is no longer running") from exc


def _load_lease(path: str | None) -> tuple[Path, dict[str, Any]]:
    if not path:
        raise ManifestError("--lease is required for this command")
    lease_path = Path(path).resolve()
    try:
        value = json.loads(lease_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read lease {lease_path}") from exc
    if not isinstance(value, dict):
        raise ManifestError("lease root must be an object")
    return lease_path, value


def _verify_capacity(client: LambdaCloudClient, manifest: DeploymentManifest) -> None:
    backend = manifest.document["backend"]
    hardware = manifest.document["hardware"]
    expected_price = int(manifest.document["cost"]["price_cents_per_hour"])
    entry = client.instance_types().get(hardware["lambda_instance_type"])
    if not isinstance(entry, dict):
        raise LambdaCloudError("frozen instance type is not currently offered")
    instance_type = entry.get("instance_type")
    if not isinstance(instance_type, dict):
        raise LambdaCloudError("instance type metadata is missing")
    if instance_type.get("name") != hardware["lambda_instance_type"]:
        raise LambdaCloudError("Lambda returned a different instance type identity")
    if instance_type.get("gpu_description") != hardware["expected_gpu_description"]:
        raise LambdaCloudError("live GPU description differs from the frozen manifest")
    specs = instance_type.get("specs")
    if not isinstance(specs, dict) or int(specs.get("gpus", -1)) != int(
        hardware["requested_gpu_count"]
    ):
        raise LambdaCloudError("live GPU count differs from the frozen manifest")
    if int(instance_type.get("price_cents_per_hour", -1)) != expected_price:
        raise LambdaCloudError("live instance price differs from the frozen price snapshot")
    regions = entry.get("regions_with_capacity_available", [])
    names = {region.get("name") for region in regions if isinstance(region, dict)}
    if backend["region_name"] not in names:
        raise LambdaCloudError("frozen region currently has no reported capacity")


def _launch(
    client: LambdaCloudClient,
    manifest: DeploymentManifest,
    args: argparse.Namespace,
    maximum_cost_cents: int,
) -> None:
    if not args.lease:
        raise ManifestError("launch requires --lease so the external watchdog can take ownership")
    lease_path = Path(args.lease).resolve()
    _require_armed_watchdog(lease_path, manifest)
    _verify_capacity(client, manifest)
    backend = manifest.document["backend"]
    gateway_key = secrets.token_urlsafe(32)
    internal_key = secrets.token_urlsafe(32)
    cloud_init = render_cloud_init(
        manifest,
        gateway_key=gateway_key,
        internal_key=internal_key,
    )
    payload = {
        "region_name": backend["region_name"],
        "instance_type_name": manifest.document["hardware"]["lambda_instance_type"],
        "ssh_key_names": [backend["ssh_key_name"]],
        "file_system_names": [],
        "name": f"flavourbench-{manifest.document['deployment_profile_id']}"[:64],
        "image": {"id": backend["image_id"]},
        "user_data": cloud_init,
        "tags": [
            {"key": "flavourbench-season", "value": manifest.document["season_slug"]},
            {"key": "flavourbench-profile", "value": manifest.document["deployment_profile_id"]},
            {"key": "flavourbench-manifest", "value": manifest.spec_sha256},
        ],
        "firewall_rulesets": [{"id": backend["firewall_ruleset_id"]}],
    }
    instance_ids = client.launch(payload)
    if len(instance_ids) != 1:
        if instance_ids:
            client.terminate(instance_ids)
        raise LambdaCloudError("expected exactly one instance; returned instances were terminated")
    launched_at = datetime.now(UTC)
    lease = {
        "schema_version": "1.0",
        "provider": "lambda_cloud",
        "instance_id": instance_ids[0],
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.spec_sha256,
        "launched_at": launched_at.isoformat(),
        "terminate_at": (launched_at + timedelta(minutes=args.max_runtime_minutes)).isoformat(),
        "price_cents_per_hour": manifest.document["cost"]["price_cents_per_hour"],
        "maximum_cost_cents": maximum_cost_cents,
        "gateway_bearer_token": gateway_key,
        "state": "launched",
    }
    try:
        _write_lease(lease_path, lease)
    except OSError as exc:
        client.terminate(instance_ids)
        raise LambdaCloudError(
            "instance launched but lease persistence failed; emergency termination requested"
        ) from exc
    print(f"launched instance {instance_ids[0]}")
    print(f"external watchdog lease: {lease_path}")


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = load_manifest(args.manifest, expected_backend="lambda_cloud")
        maximum_cost = _validate_runtime(manifest, args.max_runtime_minutes)
        print(f"manifest: {manifest.spec_sha256}")
        print(f"maximum runtime: {args.max_runtime_minutes} minutes")
        print(f"reserved upper-bound compute: {maximum_cost} cents USD")
        print(f"authorized for mutations: {manifest.mutations_authorized}")

        if args.command == "plan":
            print("dry run only; no Lambda Cloud API call was made")
            return 0
        if args.command == "render-cloud-init":
            print(
                render_cloud_init(
                    manifest,
                    gateway_key="DRY_RUN_NOT_A_SECRET",
                    internal_key="DRY_RUN_NOT_A_SECRET",
                )
            )
            return 0

        if args.command == "status":
            client = LambdaCloudClient(os.environ.get("LAMBDA_API_KEY", ""))
            _, lease = _load_lease(args.lease)
            instance = client.instance(str(lease["instance_id"]))
            print(json.dumps(instance, indent=2, sort_keys=True))
            return 0

        _live_gate(manifest, args)
        client = LambdaCloudClient(os.environ.get("LAMBDA_API_KEY", ""))
        if args.command == "launch":
            _launch(client, manifest, args, maximum_cost)
            return 0

        lease_path = None
        lease: dict[str, Any] = {}
        instance_id = args.instance_id
        if args.lease:
            lease_path, lease = _load_lease(args.lease)
            instance_id = str(lease["instance_id"])
        if not instance_id:
            raise ManifestError("terminate requires --instance-id or --lease")
        result = client.terminate([instance_id])
        print(json.dumps(result, indent=2, sort_keys=True))
        if lease_path:
            lease["state"] = "termination_requested"
            lease["termination_requested_at"] = datetime.now(UTC).isoformat()
            _write_lease(lease_path, lease)
    except (ManifestError, LambdaCloudError) as exc:
        raise SystemExit(f"Lambda Cloud controller refused action: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
