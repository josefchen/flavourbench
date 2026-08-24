from __future__ import annotations

import base64
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from common.manifest import ManifestError, compute_spec_sha256, load_manifest
from lambda_cloud.controller import _maximum_cost_cents, _validate_runtime, render_cloud_init
from lambda_cloud.watchdog import _cost_cents, _termination_reasons

ROOT = Path(__file__).resolve().parents[1]
MODAL_MANIFEST = ROOT / "manifests" / "example-modal-qwen2.5-0.5b.json"
LAMBDA_MANIFEST = ROOT / "manifests" / "example-lambda-qwen2.5-0.5b.json"


class ManifestTests(unittest.TestCase):
    def test_examples_are_self_hashed_and_inert(self) -> None:
        for path in (MODAL_MANIFEST, LAMBDA_MANIFEST):
            manifest = load_manifest(path)
            self.assertEqual(manifest.spec_sha256, compute_spec_sha256(manifest.document))
            self.assertFalse(manifest.mutations_authorized)
            with self.assertRaises(ManifestError):
                manifest.require_live_authorization("GATE-A-NOT-AUTHORIZED")

    def test_vllm_command_is_fully_pinned(self) -> None:
        manifest = load_manifest(MODAL_MANIFEST, expected_backend="modal")
        command = manifest.vllm_args(host="127.0.0.1", port=8001)
        self.assertEqual(command[:3], ["vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"])
        self.assertIn("--revision", command)
        self.assertIn("--tokenizer-revision", command)
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertIn("--disable-log-requests", command)


class LambdaControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest(LAMBDA_MANIFEST, expected_backend="lambda_cloud")

    def test_cost_reservation_rounds_up_and_enforces_cap(self) -> None:
        self.assertEqual(_maximum_cost_cents(self.manifest, 30), 65)
        self.assertEqual(_validate_runtime(self.manifest, 30), 65)
        with self.assertRaises(ManifestError):
            _validate_runtime(self.manifest, 31)

    def test_cloud_init_is_resolved_and_binds_gateway_to_loopback(self) -> None:
        rendered = render_cloud_init(
            self.manifest,
            gateway_key="dry-gateway-key",
            internal_key="dry-internal-key",
        )
        self.assertNotIn("__MANIFEST_B64__", rendered)
        self.assertNotIn("__GATEWAY_B64__", rendered)
        self.assertIn("#cloud-config", rendered)
        lines = rendered.splitlines()
        path_index = lines.index("  - path: /etc/flavourbench/runtime.env")
        content_line = next(
            line for line in lines[path_index:] if line.startswith("    content: ")
        )
        runtime_env = base64.b64decode(content_line.removeprefix("    content: ")).decode()
        self.assertIn("FLAVOURBENCH_GATEWAY_HOST=127.0.0.1", runtime_env)

    def test_watchdog_cost_and_deadline(self) -> None:
        now = datetime.now(UTC)
        lease = {
            "launched_at": (now - timedelta(minutes=29)).isoformat(),
            "terminate_at": (now + timedelta(minutes=1)).isoformat(),
            "price_cents_per_hour": 129,
            "maximum_cost_cents": 65,
        }
        self.assertEqual(_cost_cents(lease, now), 63)
        self.assertEqual(_termination_reasons(lease, now), [])

        at_deadline = now + timedelta(minutes=1)
        reasons = _termination_reasons(lease, at_deadline)
        self.assertIn("absolute runtime deadline reached", reasons)
        self.assertIn("hard monetary cap reached", reasons)

    def test_manifest_json_remains_canonicalizable(self) -> None:
        document = json.loads(LAMBDA_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(document["spec_sha256"], compute_spec_sha256(document))


if __name__ == "__main__":
    unittest.main()
