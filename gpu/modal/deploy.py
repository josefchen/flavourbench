from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

GPU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GPU_ROOT))

from common.manifest import ManifestError, load_manifest

LIVE_ACK = "I_UNDERSTAND_THIS_CREATES_GPU_SPEND"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or deploy an immutable Modal server")
    parser.add_argument("command", choices=("plan", "deploy", "stop"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authorization-ticket", default="")
    parser.add_argument("--apply", action="store_true")
    return parser


def _live_gate(manifest, args: argparse.Namespace) -> None:
    if not args.apply:
        raise ManifestError("mutating commands require --apply")
    if os.environ.get("FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED") != LIVE_ACK:
        raise ManifestError(
            "set FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED to the documented acknowledgement"
        )
    manifest.require_live_authorization(args.authorization_ticket)


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = load_manifest(args.manifest, expected_backend="modal")
        backend = manifest.document["backend"]
        environment = backend["environment"]
        app_name = backend["app_name"]
        command = [
            sys.executable,
            "-m",
            "modal",
            "deploy",
            str(GPU_ROOT / "modal" / "app.py"),
            "--env",
            environment,
            "--strategy",
            "recreate",
        ]
        if args.command == "stop":
            command = [
                sys.executable,
                "-m",
                "modal",
                "app",
                "stop",
                app_name,
                "--env",
                environment,
                "--yes",
            ]

        print(f"manifest: {manifest.spec_sha256}")
        print(f"authorized for mutations: {manifest.mutations_authorized}")
        print("would run:", " ".join(command))
        if args.command == "plan":
            print("dry run only; no Modal API call was made")
            return 0

        _live_gate(manifest, args)
        environment_values = dict(os.environ)
        environment_values["FLAVOURBENCH_MANIFEST_PATH"] = str(manifest.path)
        subprocess.run(  # noqa: S603
            command,
            cwd=GPU_ROOT,
            env=environment_values,
            check=True,
        )
    except ManifestError as exc:
        raise SystemExit(f"refusing Modal mutation: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
