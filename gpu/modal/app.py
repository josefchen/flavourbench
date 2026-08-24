from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

GPU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GPU_ROOT))

from common.manifest import load_manifest

DEFAULT_MANIFEST = GPU_ROOT / "manifests" / "example-modal-qwen2.5-0.5b.json"
MANIFEST_PATH = Path(os.environ.get("FLAVOURBENCH_MANIFEST_PATH", DEFAULT_MANIFEST))
MANIFEST = load_manifest(MANIFEST_PATH, expected_backend="modal")
DOCUMENT = MANIFEST.document
BACKEND = DOCUMENT["backend"]
HARDWARE = DOCUMENT["hardware"]
SERVING = DOCUMENT["serving"]

image = (
    modal.Image.from_registry(DOCUMENT["runtime"]["container_image"])
    .entrypoint([])
    .add_local_file(
        GPU_ROOT / "common" / "gateway.py",
        "/opt/flavourbench/gateway.py",
        copy=True,
    )
    .add_local_file(
        MANIFEST_PATH,
        "/opt/flavourbench/deployment-manifest.json",
        copy=True,
    )
)

app = modal.App(
    name=BACKEND["app_name"],
    tags={
        "project": "flavourbench",
        "season": DOCUMENT["season_slug"],
        "deployment-profile": DOCUMENT["deployment_profile_id"],
        "manifest": MANIFEST.spec_sha256[:16],
    },
)


@app.server(
    image=image,
    gpu=HARDWARE["modal_gpu"],
    secrets=[modal.Secret.from_name(BACKEND["secret_name"])],
    env={
        "FLAVOURBENCH_DEPLOYMENT_MANIFEST": "/opt/flavourbench/deployment-manifest.json",
        "FLAVOURBENCH_GATEWAY_PORT": "8000",
        "FLAVOURBENCH_INTERNAL_VLLM_PORT": "8001",
        "FLAVOURBENCH_VLLM_ARGS_JSON": json.dumps(
            MANIFEST.vllm_args(host="127.0.0.1", port=8001), separators=(",", ":")
        ),
    },
    port=8000,
    min_containers=BACKEND["min_containers"],
    max_containers=BACKEND["max_containers"],
    target_concurrency=BACKEND["target_concurrency"],
    buffer_containers=BACKEND["buffer_containers"],
    scaledown_window=BACKEND["scaledown_window_seconds"],
    startup_timeout=BACKEND["startup_timeout_seconds"],
    exit_grace_period=BACKEND["exit_grace_period_seconds"],
    routing_region=BACKEND["routing_region"],
    compute_region=BACKEND["compute_region"],
)
class FlavourBenchVllmServer:
    @modal.enter()
    def start_gateway(self) -> None:
        self.process = subprocess.Popen(  # noqa: S603
            [sys.executable, "/opt/flavourbench/gateway.py"],
            start_new_session=True,
        )

    @modal.exit()
    def stop_gateway(self) -> None:
        if getattr(self, "process", None) is not None:
            self.process.terminate()
