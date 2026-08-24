from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
PINNED_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.-]+)?$")
PLACEHOLDER_MARKERS = ("REPLACE_", "CHANGEME", "PLACEHOLDER")


class ManifestError(ValueError):
    """Raised when a deployment manifest cannot be treated as immutable."""


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def compute_spec_sha256(document: dict[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("spec_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ManifestError(f"{context}.{key} is required")
    return value


def _contains_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return any(marker in value.upper() for marker in PLACEHOLDER_MARKERS)
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


@dataclass(frozen=True)
class DeploymentManifest:
    path: Path
    document: dict[str, Any]
    spec_sha256: str

    @property
    def backend(self) -> str:
        return str(self.document["backend"]["kind"])

    @property
    def model_id(self) -> str:
        return str(self.document["model"]["repo_id"])

    @property
    def served_model_name(self) -> str:
        return str(self.document["model"]["served_model_name"])

    @property
    def mutations_authorized(self) -> bool:
        return bool(self.document["controls"]["mutations_authorized"])

    @property
    def authorization_ticket(self) -> str:
        return str(self.document["controls"]["authorization_ticket"])

    def require_live_authorization(self, supplied_ticket: str) -> None:
        if not self.mutations_authorized:
            raise ManifestError("manifest controls.mutations_authorized is false")
        if not self.document["controls"].get("official_runs_allowed", False):
            raise ManifestError("manifest controls.official_runs_allowed is false")
        if _contains_placeholder(self.document):
            raise ManifestError("authorized manifests cannot contain placeholder values")
        if not supplied_ticket or supplied_ticket != self.authorization_ticket:
            raise ManifestError("authorization ticket does not match the frozen manifest")

    def vllm_args(self, *, host: str, port: int) -> list[str]:
        model = self.document["model"]
        runtime = self.document["runtime"]
        serving = self.document["serving"]
        arguments = [
            "vllm",
            "serve",
            model["repo_id"],
            "--revision",
            model["revision"],
            "--tokenizer-revision",
            model["tokenizer_revision"],
            "--code-revision",
            model["code_revision"],
            "--served-model-name",
            model["served_model_name"],
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            serving["dtype"],
            "--max-model-len",
            str(serving["max_model_len"]),
            "--tensor-parallel-size",
            str(serving["tensor_parallel_size"]),
            "--pipeline-parallel-size",
            str(serving["pipeline_parallel_size"]),
            "--max-num-seqs",
            str(serving["max_num_seqs"]),
            "--gpu-memory-utilization",
            str(serving["gpu_memory_utilization"]),
            "--seed",
            str(serving["engine_seed"]),
            "--generation-config",
            "vllm",
            "--structured-outputs-config.backend",
            serving["structured_outputs_backend"],
            "--enable-request-id-headers",
            "--disable-log-requests",
        ]
        if serving.get("enable_auto_tool_choice", False):
            arguments.extend(
                ["--enable-auto-tool-choice", "--tool-call-parser", serving["tool_call_parser"]]
            )
        reasoning_parser = serving.get("reasoning_parser")
        if reasoning_parser:
            arguments.extend(["--reasoning-parser", reasoning_parser])
        if runtime["vllm_version"] not in runtime["container_image"]:
            raise ManifestError("container image reference must visibly pin the vLLM release")
        return arguments


def load_manifest(path: str | Path, *, expected_backend: str | None = None) -> DeploymentManifest:
    source = Path(path).resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read manifest {source}") from exc
    if not isinstance(document, dict):
        raise ManifestError("manifest root must be a JSON object")

    if document.get("schema_version") != "1.0":
        raise ManifestError("schema_version must be exactly 1.0")
    supplied_hash = str(_required(document, "spec_sha256", "manifest"))
    computed_hash = compute_spec_sha256(document)
    if supplied_hash != computed_hash:
        raise ManifestError(
            f"spec_sha256 mismatch: manifest has {supplied_hash}, computed {computed_hash}"
        )

    backend = _required(document, "backend", "manifest")
    model = _required(document, "model", "manifest")
    runtime = _required(document, "runtime", "manifest")
    serving = _required(document, "serving", "manifest")
    controls = _required(document, "controls", "manifest")
    decoding = _required(document, "decoding", "manifest")
    for name, value in {
        "backend": backend,
        "model": model,
        "runtime": runtime,
        "serving": serving,
        "controls": controls,
        "decoding": decoding,
    }.items():
        if not isinstance(value, dict):
            raise ManifestError(f"{name} must be an object")

    backend_kind = str(_required(backend, "kind", "backend"))
    if backend_kind not in {"modal", "lambda_cloud"}:
        raise ManifestError("backend.kind must be modal or lambda_cloud")
    if expected_backend and backend_kind != expected_backend:
        raise ManifestError(f"expected backend {expected_backend}, found {backend_kind}")

    for field in ("revision", "tokenizer_revision", "code_revision"):
        if not HEX_40.fullmatch(str(_required(model, field, "model"))):
            raise ManifestError(f"model.{field} must be a 40-character Git commit")
    if not HEX_64.fullmatch(str(_required(model, "weights_sha256", "model"))):
        raise ManifestError("model.weights_sha256 must be a SHA-256 digest")

    image = str(_required(runtime, "container_image", "runtime"))
    if "@sha256:" not in image or not HEX_64.fullmatch(image.rsplit("@sha256:", 1)[1]):
        raise ManifestError("runtime.container_image must use an OCI sha256 digest")
    for field in ("vllm_version", "modal_sdk_version"):
        if not PINNED_VERSION.fullmatch(str(_required(runtime, field, "runtime"))):
            raise ManifestError(f"runtime.{field} must be an exact semantic version")

    if serving.get("parallel_tool_calls") is not False:
        raise ManifestError("serving.parallel_tool_calls must be false")
    max_rounds = int(_required(serving, "max_tool_rounds", "serving"))
    if not 1 <= max_rounds <= 8:
        raise ManifestError("serving.max_tool_rounds must be between 1 and 8")
    if int(_required(decoding, "max_tokens", "decoding")) > 8192:
        raise ManifestError("decoding.max_tokens cannot exceed 8192")

    if controls.get("mutations_authorized", False) and _contains_placeholder(document):
        raise ManifestError("live-authorized manifests cannot contain placeholders")
    if controls.get("mutations_authorized", False) and not controls.get(
        "official_runs_allowed", False
    ):
        raise ManifestError("live-authorized manifests must explicitly decide official run status")

    return DeploymentManifest(source, document, computed_hash)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and hash FlavourBench GPU manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "hash"):
        child = subparsers.add_parser(command)
        child.add_argument("manifest")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "hash":
        try:
            document = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"manifest unreadable: {exc}") from exc
        if not isinstance(document, dict):
            raise SystemExit("manifest root must be an object")
        print(compute_spec_sha256(document))
        return 0
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        raise SystemExit(f"manifest invalid: {exc}") from exc
    print(f"valid {manifest.backend} manifest {manifest.spec_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
