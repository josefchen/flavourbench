"""Deterministic distributed execution for the Season 1 arena Monte Carlo.

This module schedules synthetic *method validation* only. Its records are never
model responses, votes, or benchmark quality observations. Production shards
use the frozen estimator equation and remain unable to publish a leaderboard.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .season1_arena_acceptance import (
    ARENA_INFERENCE_POLICY_SHA256,
    canonical_sha256,
    load_arena_inference_policy,
)
from .season1_arena_monte_carlo import (
    SCENARIOS,
    aggregate_production_results,
    bind_distributed_receipts,
    build_production_layout,
    engine_source_bundle,
    run_dataset,
    verify_production_result,
)

EXECUTION_CONTRACT_SCHEMA = "flavourbench-season1-arena-distributed-execution-v1"
EXECUTION_MANIFEST_SCHEMA = "flavourbench-season1-arena-distributed-manifest-v1"
SHARD_SPEC_SCHEMA = "flavourbench-season1-arena-production-shard-spec-v1"
SHARD_RESULT_SCHEMA = "flavourbench-season1-arena-production-shard-result-v1"
SHARD_TELEMETRY_SCHEMA = "flavourbench-season1-arena-shard-telemetry-v1"
DISTRIBUTED_AGGREGATE_SCHEMA = "flavourbench-season1-arena-distributed-aggregate-v1"
COST_PROJECTION_SCHEMA = "flavourbench-season1-arena-modal-cost-projection-v1"
MODAL_ADMISSION_SCHEMA = "flavourbench-season1-arena-modal-admission-v1"
PRODUCTION_SHARD_AUTHORIZATION = "I_AUTHORIZE_ONE_SEASON1_METHOD_VALIDATION_SHARD"


class DistributedMonteCarloError(RuntimeError):
    """Distributed evidence is incomplete, altered, or outside its frozen scope."""


def _project_root() -> Path:
    relative = Path(
        "contracts/season1/method-validation/"
        "season1-arena-distributed-execution-v2.json"
    )
    candidates = (
        Path.cwd().resolve(),
        Path.cwd().resolve() / "flavourbench",
        Path(__file__).resolve().parents[2],
        Path("/app"),
    )
    return next(
        (candidate for candidate in candidates if (candidate / relative).is_file()),
        candidates[2],
    )


ROOT = _project_root()
DEFAULT_EXECUTION_CONTRACT = ROOT / (
    "contracts/season1/method-validation/"
    "season1-arena-distributed-execution-v2.json"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_address(body: Mapping[str, Any], *, field: str = "artifact_sha256") -> dict[str, Any]:
    return {**dict(body), field: canonical_sha256(body)}


def _verify_content_address(
    document: Mapping[str, Any],
    *,
    field: str = "artifact_sha256",
    label: str,
) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if document.get(field) != canonical_sha256(body):
        raise DistributedMonteCarloError(f"{label} content address failed")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistributedMonteCarloError(f"{label} is unreadable") from exc
    if not isinstance(document, dict):
        raise DistributedMonteCarloError(f"{label} must be a JSON object")
    return document


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.exists() and path.read_bytes() != encoded:
            raise DistributedMonteCarloError(f"refusing to replace differing artifact: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_execution_contract(
    path: Path = DEFAULT_EXECUTION_CONTRACT,
) -> dict[str, Any]:
    document = _read_json(path, label="distributed execution contract")
    _verify_content_address(document, label="distributed execution contract")
    if (
        document.get("schema_version") != EXECUTION_CONTRACT_SCHEMA
        or document.get("status") != "frozen_before_distributed_execution"
        or document.get("policy_artifact_sha256") != ARENA_INFERENCE_POLICY_SHA256
        or document.get("layout_artifact_sha256")
        != build_production_layout()["artifact_sha256"]
        or document.get("engine_source_bundle_sha256")
        != engine_source_bundle()["artifact_sha256"]
    ):
        raise DistributedMonteCarloError("distributed execution contract identity is invalid")
    required = (
        "implementation_files",
        "production_image",
        "campaign",
        "modal",
        "pricing_snapshot",
    )
    if any(not isinstance(document.get(key), Mapping) for key in required):
        raise DistributedMonteCarloError("distributed execution contract is incomplete")
    return document


def _runtime_file(path: str) -> Path:
    candidate = ROOT / path
    if not candidate.is_file():
        raise DistributedMonteCarloError(f"runtime file is missing: {path}")
    return candidate


def verify_runtime_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the image and imported estimator before any resampling starts."""

    expected_files = contract.get("implementation_files")
    image = contract.get("production_image")
    if not isinstance(expected_files, Mapping) or not isinstance(image, Mapping):
        raise DistributedMonteCarloError("runtime identity contract is malformed")
    observed_files = {
        str(relative): _sha256_file(_runtime_file(str(relative)))
        for relative in expected_files
    }
    if observed_files != dict(expected_files):
        raise DistributedMonteCarloError("worker implementation hash mismatch")

    imported_sources = {
        "src/flavourbench/season1_arena_acceptance.py": (
            "flavourbench.season1_arena_acceptance"
        ),
        "src/flavourbench/season1_statistics.py": "flavourbench.season1_statistics",
        "src/flavourbench/season1_arena_monte_carlo.py": (
            "flavourbench.season1_arena_monte_carlo"
        ),
        "src/flavourbench/season1_arena_distributed.py": (
            "flavourbench.season1_arena_distributed"
        ),
    }
    imported_hashes: dict[str, str] = {}
    for relative, module_name in imported_sources.items():
        module = importlib.import_module(module_name)
        module_path = Path(str(module.__file__)).resolve()
        imported_hashes[relative] = _sha256_file(module_path)
        if imported_hashes[relative] != expected_files.get(relative):
            raise DistributedMonteCarloError(
                f"imported source differs from frozen image source: {module_name}"
            )

    expected_python = str(image.get("python_version"))
    observed_python = platform.python_version()
    if observed_python != expected_python:
        raise DistributedMonteCarloError(
            f"worker Python mismatch: expected {expected_python}, observed {observed_python}"
        )
    expected_machine = str(image.get("machine"))
    observed_machine = platform.machine()
    if observed_machine != expected_machine:
        raise DistributedMonteCarloError(
            f"worker machine mismatch: expected {expected_machine}, observed {observed_machine}"
        )

    expected_environment = image.get("thread_environment")
    if not isinstance(expected_environment, Mapping):
        raise DistributedMonteCarloError("production thread environment is absent")
    observed_environment = {
        str(name): os.getenv(str(name)) for name in expected_environment
    }
    if observed_environment != dict(expected_environment):
        raise DistributedMonteCarloError("worker thread environment mismatch")

    dependencies = image.get("dependency_versions")
    if not isinstance(dependencies, Mapping):
        raise DistributedMonteCarloError("production dependency identity is absent")
    observed_dependencies: dict[str, str] = {}
    for package, expected in dependencies.items():
        try:
            observed = importlib.metadata.version(str(package))
        except importlib.metadata.PackageNotFoundError as exc:
            raise DistributedMonteCarloError(
                f"worker dependency is absent: {package}"
            ) from exc
        observed_dependencies[str(package)] = observed
        if observed != expected:
            raise DistributedMonteCarloError(
                f"worker dependency mismatch for {package}: {observed} != {expected}"
            )

    policy = load_arena_inference_policy()
    if policy["artifact_sha256"] != contract.get("policy_artifact_sha256"):
        raise DistributedMonteCarloError("worker policy artifact mismatch")
    layout = build_production_layout()
    if layout["artifact_sha256"] != contract.get("layout_artifact_sha256"):
        raise DistributedMonteCarloError("worker layout artifact mismatch")
    return {
        "verified": True,
        "python_version": observed_python,
        "machine": observed_machine,
        "thread_environment": observed_environment,
        "implementation_files": observed_files,
        "imported_source_files": imported_hashes,
        "dependency_versions": observed_dependencies,
        "policy_artifact_sha256": policy["artifact_sha256"],
        "layout_artifact_sha256": layout["artifact_sha256"],
    }


def build_shard_spec(
    *,
    scenario: str,
    start: int,
    count: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    campaign = contract["campaign"]
    minimum_datasets = int(campaign["datasets_per_scenario"])
    if scenario not in SCENARIOS:
        raise DistributedMonteCarloError(f"unknown scenario: {scenario}")
    if start < 0 or count < 1 or start + count > minimum_datasets:
        raise DistributedMonteCarloError("shard range is outside the frozen campaign")
    body = {
        "schema_version": SHARD_SPEC_SCHEMA,
        "execution_contract_sha256": contract["artifact_sha256"],
        "policy_sha256": contract["policy_artifact_sha256"],
        "layout_sha256": contract["layout_artifact_sha256"],
        "engine_source_sha256": contract["implementation_files"][
            "src/flavourbench/season1_arena_monte_carlo.py"
        ],
        "engine_source_bundle_sha256": contract["engine_source_bundle_sha256"],
        "scenario": scenario,
        "start": start,
        "count": count,
        "bootstrap_replicates": int(campaign["bootstrap_replicates_per_dataset"]),
        "production_mode": True,
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
        },
    }
    return _content_address(body, field="shard_sha256")


def verify_shard_spec(
    spec: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> None:
    _verify_content_address(spec, field="shard_sha256", label="shard specification")
    expected = build_shard_spec(
        scenario=str(spec.get("scenario")),
        start=int(spec.get("start", -1)),
        count=int(spec.get("count", -1)),
        contract=contract,
    )
    if dict(spec) != expected:
        raise DistributedMonteCarloError("shard specification differs from frozen contract")


def build_execution_manifest(
    *,
    shard_size: int = 1,
    contract_path: Path = DEFAULT_EXECUTION_CONTRACT,
) -> dict[str, Any]:
    contract = load_execution_contract(contract_path)
    campaign = contract["campaign"]
    datasets = int(campaign["datasets_per_scenario"])
    maximum = int(campaign["maximum_datasets_per_shard"])
    if shard_size < 1 or shard_size > maximum:
        raise DistributedMonteCarloError(
            f"shard size must be between 1 and {maximum} datasets"
        )
    shards = []
    for scenario in SCENARIOS:
        for start in range(0, datasets, shard_size):
            shards.append(
                build_shard_spec(
                    scenario=scenario,
                    start=start,
                    count=min(shard_size, datasets - start),
                    contract=contract,
                )
            )
    body = {
        "schema_version": EXECUTION_MANIFEST_SCHEMA,
        "execution_contract_sha256": contract["artifact_sha256"],
        "policy_sha256": contract["policy_artifact_sha256"],
        "layout_sha256": contract["layout_artifact_sha256"],
        "shard_size": shard_size,
        "shard_count": len(shards),
        "dataset_record_count": sum(int(shard["count"]) for shard in shards),
        "shards": shards,
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
            "full_campaign_not_executed_by_manifest_creation": True,
        },
    }
    return _content_address(body)


def verify_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> None:
    _verify_content_address(manifest, label="distributed execution manifest")
    if (
        manifest.get("schema_version") != EXECUTION_MANIFEST_SCHEMA
        or manifest.get("execution_contract_sha256") != contract["artifact_sha256"]
        or manifest.get("policy_sha256") != contract["policy_artifact_sha256"]
        or manifest.get("layout_sha256") != contract["layout_artifact_sha256"]
    ):
        raise DistributedMonteCarloError("distributed execution manifest identity is invalid")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise DistributedMonteCarloError("distributed execution manifest lacks shards")
    expected_indices = set(range(int(contract["campaign"]["datasets_per_scenario"])))
    observed: dict[str, set[int]] = {scenario: set() for scenario in SCENARIOS}
    shard_ids: set[str] = set()
    for value in shards:
        if not isinstance(value, Mapping):
            raise DistributedMonteCarloError("manifest contains a non-object shard")
        verify_shard_spec(value, contract=contract)
        shard_id = str(value["shard_sha256"])
        if shard_id in shard_ids:
            raise DistributedMonteCarloError("manifest contains a duplicate shard")
        shard_ids.add(shard_id)
        scenario = str(value["scenario"])
        indices = set(range(int(value["start"]), int(value["start"]) + int(value["count"])))
        if observed[scenario] & indices:
            raise DistributedMonteCarloError("manifest shards overlap")
        observed[scenario] |= indices
    if any(indices != expected_indices for indices in observed.values()):
        raise DistributedMonteCarloError("manifest does not exactly cover the campaign")
    if (
        manifest.get("shard_count") != len(shards)
        or manifest.get("dataset_record_count") != len(SCENARIOS) * len(expected_indices)
    ):
        raise DistributedMonteCarloError("manifest campaign counts are inconsistent")


def _processor_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    match = re.search(r"^model name\s*:\s*(.+)$", path.read_text(errors="replace"), re.MULTILINE)
    return match.group(1).strip() if match else None


def _record_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        {"record_sha256s": [str(record["record_sha256"]) for record in records]}
    )


def execute_shard(
    spec: Mapping[str, Any],
    *,
    provider: str,
    contract_path: Path = DEFAULT_EXECUTION_CONTRACT,
) -> dict[str, dict[str, Any]]:
    """Run one content-addressed shard after verifying the worker image."""

    contract = load_execution_contract(contract_path)
    verify_shard_spec(spec, contract=contract)
    runtime = verify_runtime_identity(contract)
    started_at = datetime.now(UTC)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    records = [
        run_dataset(
            scenario=str(spec["scenario"]),
            dataset_index=dataset_index,
            bootstrap_replicates=int(spec["bootstrap_replicates"]),
            production_mode=True,
        )
        for dataset_index in range(
            int(spec["start"]), int(spec["start"]) + int(spec["count"])
        )
    ]
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    expected_indices = list(
        range(int(spec["start"]), int(spec["start"]) + int(spec["count"]))
    )
    for expected_index, record in zip(expected_indices, records, strict=True):
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("record_sha256") != canonical_sha256(payload)
            or record.get("scenario") != spec["scenario"]
            or record.get("dataset_index") != expected_index
            or record.get("bootstrap_replicates") != spec["bootstrap_replicates"]
            or record.get("policy_sha256") != spec["policy_sha256"]
            or record.get("layout_sha256") != spec["layout_sha256"]
            or record.get("engine_source_bundle_sha256")
            != spec["engine_source_bundle_sha256"]
            or record.get("production_mode") is not True
            or record.get("status") != "completed"
            or record.get("claim_boundary", {}).get("model_quality_evidence") is not False
        ):
            raise DistributedMonteCarloError("worker produced an ineligible dataset record")

    result_body = {
        "schema_version": SHARD_RESULT_SCHEMA,
        "shard_spec_sha256": spec["shard_sha256"],
        "execution_contract_sha256": contract["artifact_sha256"],
        "policy_sha256": contract["policy_artifact_sha256"],
        "layout_sha256": contract["layout_artifact_sha256"],
        "engine_source_sha256": spec["engine_source_sha256"],
        "engine_source_bundle_sha256": spec["engine_source_bundle_sha256"],
        "record_count": len(records),
        "record_set_sha256": _record_set_sha256(records),
        "records": records,
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
            "aggregate_acceptance_claimed": False,
        },
    }
    result = _content_address(result_body)
    pricing = contract["pricing_snapshot"]
    modal = contract["modal"]
    request_gib = float(modal["memory_request_mib"]) / 1024.0
    published_rate_estimate = wall_seconds * (
        float(modal["cpu_cores"]) * float(pricing["cpu_usd_per_core_second"])
        + request_gib * float(pricing["memory_usd_per_gib_second"])
    )
    telemetry_body = {
        "schema_version": SHARD_TELEMETRY_SCHEMA,
        "shard_spec_sha256": spec["shard_sha256"],
        "shard_result_sha256": result["artifact_sha256"],
        "provider": provider,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "dataset_count": len(records),
        "wall_seconds": round(wall_seconds, 6),
        "process_cpu_seconds": round(cpu_seconds, 6),
        "maximum_resident_set_kibibytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "processor_model": _processor_model(),
        "runtime_identity": runtime,
        "published_rate_compute_estimate_usd": round(published_rate_estimate, 9),
        "published_rate_snapshot_url": pricing["source_url"],
        "published_rate_snapshot_date": pricing["observed_date"],
        "billing_limit": (
            "Published-rate function compute estimate only; image build, storage, credits, "
            "tax, and invoice reconciliation are not observed here."
        ),
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
        },
    }
    return {"result": result, "telemetry": _content_address(telemetry_body)}


def verify_shard_result(
    spec: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> None:
    verify_shard_spec(spec, contract=contract)
    _verify_content_address(result, label="shard result")
    records = result.get("records")
    if not isinstance(records, list):
        raise DistributedMonteCarloError("shard result lacks records")
    indices = list(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))
    if (
        result.get("schema_version") != SHARD_RESULT_SCHEMA
        or result.get("shard_spec_sha256") != spec["shard_sha256"]
        or result.get("execution_contract_sha256") != contract["artifact_sha256"]
        or result.get("policy_sha256") != contract["policy_artifact_sha256"]
        or result.get("layout_sha256") != contract["layout_artifact_sha256"]
        or result.get("engine_source_sha256") != spec["engine_source_sha256"]
        or result.get("engine_source_bundle_sha256")
        != spec["engine_source_bundle_sha256"]
        or result.get("record_count") != int(spec["count"])
        or len(records) != int(spec["count"])
        or result.get("record_set_sha256") != _record_set_sha256(records)
        or result.get("claim_boundary", {}).get("model_quality_evidence") is not False
    ):
        raise DistributedMonteCarloError("shard result identity is invalid")
    for expected_index, record in zip(indices, records, strict=True):
        if not isinstance(record, Mapping):
            raise DistributedMonteCarloError("shard contains a non-object dataset record")
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("record_sha256") != canonical_sha256(payload)
            or record.get("scenario") != spec["scenario"]
            or record.get("dataset_index") != expected_index
            or record.get("bootstrap_replicates") != spec["bootstrap_replicates"]
            or record.get("production_mode") is not True
            or record.get("engine_source_bundle_sha256")
            != spec["engine_source_bundle_sha256"]
            or record.get("status") != "completed"
        ):
            raise DistributedMonteCarloError("shard dataset record is invalid")


def verify_shard_telemetry(telemetry: Mapping[str, Any]) -> None:
    _verify_content_address(telemetry, label="shard telemetry")
    if (
        telemetry.get("schema_version") != SHARD_TELEMETRY_SCHEMA
        or not isinstance(telemetry.get("wall_seconds"), (int, float))
        or float(telemetry["wall_seconds"]) <= 0
        or not isinstance(telemetry.get("dataset_count"), int)
        or int(telemetry["dataset_count"]) < 1
        or telemetry.get("claim_boundary", {}).get("model_quality_evidence") is not False
    ):
        raise DistributedMonteCarloError("shard telemetry is invalid")


def verify_modal_measurement_bundle(
    manifest: Mapping[str, Any],
    measurement: Mapping[str, Any],
    measurement_result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> None:
    verify_execution_manifest(manifest, contract=contract)
    verify_shard_telemetry(measurement)
    if measurement.get("provider") != "modal" or measurement.get("dataset_count") != 1:
        raise DistributedMonteCarloError(
            "admission requires one production-image dataset measured on Modal"
        )
    matches = [
        spec
        for spec in manifest["shards"]
        if spec["shard_sha256"] == measurement.get("shard_spec_sha256")
    ]
    if len(matches) != 1 or int(matches[0]["count"]) != 1:
        raise DistributedMonteCarloError("measurement shard is outside the campaign")
    verify_shard_result(matches[0], measurement_result, contract=contract)
    if measurement.get("shard_result_sha256") != measurement_result.get(
        "artifact_sha256"
    ):
        raise DistributedMonteCarloError("measurement does not bind its shard result")
    runtime = measurement.get("runtime_identity")
    image = contract["production_image"]
    if not isinstance(runtime, Mapping) or (
        runtime.get("verified") is not True
        or runtime.get("python_version") != image["python_version"]
        or runtime.get("machine") != image["machine"]
        or runtime.get("thread_environment") != image["thread_environment"]
        or runtime.get("dependency_versions") != image["dependency_versions"]
        or runtime.get("implementation_files") != contract["implementation_files"]
        or runtime.get("policy_artifact_sha256") != contract["policy_artifact_sha256"]
        or runtime.get("layout_artifact_sha256") != contract["layout_artifact_sha256"]
    ):
        raise DistributedMonteCarloError(
            "measurement runtime identity is not frozen-image evidence"
        )


def write_shard_bundle(
    output_directory: Path,
    spec: Mapping[str, Any],
    bundle: Mapping[str, Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> dict[str, str]:
    result = bundle.get("result")
    telemetry = bundle.get("telemetry")
    if not isinstance(result, Mapping) or not isinstance(telemetry, Mapping):
        raise DistributedMonteCarloError("worker returned an incomplete shard bundle")
    verify_shard_result(spec, result, contract=contract)
    verify_shard_telemetry(telemetry)
    if (
        telemetry.get("shard_spec_sha256") != spec["shard_sha256"]
        or telemetry.get("shard_result_sha256") != result["artifact_sha256"]
    ):
        raise DistributedMonteCarloError("shard telemetry does not bind its result")
    directory = output_directory / "shards" / str(spec["shard_sha256"])
    existing_results = list(directory.glob("result-*.json")) if directory.exists() else []
    expected_result = directory / f"result-{result['artifact_sha256']}.json"
    if any(path != expected_result for path in existing_results):
        raise DistributedMonteCarloError("deterministic shard re-execution diverged")
    result_path = expected_result
    telemetry_path = directory / f"telemetry-{telemetry['artifact_sha256']}.json"
    _atomic_json(result_path, result)
    _atomic_json(telemetry_path, telemetry)
    return {"result": str(result_path), "telemetry": str(telemetry_path)}


def _completed_results(
    manifest: Mapping[str, Any],
    output_directory: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_specs = {
        str(spec["shard_sha256"]): spec for spec in manifest["shards"]
    }
    completed: dict[str, dict[str, Any]] = {}
    shard_root = output_directory / "shards"
    for shard_id, spec in expected_specs.items():
        directory = shard_root / shard_id
        paths = sorted(directory.glob("result-*.json")) if directory.exists() else []
        if len(paths) > 1:
            raise DistributedMonteCarloError(f"multiple results exist for shard {shard_id}")
        if not paths:
            continue
        result = _read_json(paths[0], label=f"shard result {shard_id}")
        if paths[0].stem != f"result-{result.get('artifact_sha256')}":
            raise DistributedMonteCarloError("shard result filename is not content addressed")
        verify_shard_result(spec, result, contract=contract)
        completed[shard_id] = result
    if shard_root.exists():
        for path in shard_root.glob("*/result-*.json"):
            shard_id = path.parent.name
            if shard_id not in expected_specs:
                raise DistributedMonteCarloError("output directory contains a foreign shard")
    return completed


def execution_status(
    manifest: Mapping[str, Any],
    output_directory: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    verify_execution_manifest(manifest, contract=contract)
    completed = _completed_results(manifest, output_directory, contract=contract)
    missing = [
        str(spec["shard_sha256"])
        for spec in manifest["shards"]
        if str(spec["shard_sha256"]) not in completed
    ]
    completed_records = sum(int(result["record_count"]) for result in completed.values())
    body = {
        "schema_version": "flavourbench-season1-arena-distributed-status-v1",
        "execution_manifest_sha256": manifest["artifact_sha256"],
        "status": "complete" if not missing else "required_not_yet_executed",
        "completed_shards": len(completed),
        "required_shards": len(manifest["shards"]),
        "completed_dataset_records": completed_records,
        "required_dataset_records": int(manifest["dataset_record_count"]),
        "missing_shard_set_sha256": canonical_sha256({"shard_sha256s": missing}),
        "first_missing_shard_sha256s": missing[:20],
        "claim_boundary": {
            "production_gate_complete": not missing,
            "pass_claimed": False,
            "model_quality_evidence": False,
        },
    }
    return _content_address(body)


def aggregate_distributed_results(
    manifest: Mapping[str, Any],
    output_directory: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    status = execution_status(manifest, output_directory, contract=contract)
    completed = _completed_results(manifest, output_directory, contract=contract)
    if status["status"] != "complete":
        body = {
            "schema_version": DISTRIBUTED_AGGREGATE_SCHEMA,
            "status": "required_not_yet_executed",
            "execution_manifest_sha256": manifest["artifact_sha256"],
            "completed_shards": status["completed_shards"],
            "required_shards": status["required_shards"],
            "completed_dataset_records": status["completed_dataset_records"],
            "required_dataset_records": status["required_dataset_records"],
            "missing_shard_set_sha256": status["missing_shard_set_sha256"],
            "engine_result": None,
            "acceptance": None,
            "claim_boundary": {
                "production_gate_complete": False,
                "pass_claimed": False,
                "model_quality_evidence": False,
            },
        }
        return _content_address(body)

    records: list[dict[str, Any]] = []
    result_hashes: list[str] = []
    record_keys: set[tuple[str, int]] = set()
    for spec in manifest["shards"]:
        result = completed[str(spec["shard_sha256"])]
        result_hashes.append(str(result["artifact_sha256"]))
        for record in result["records"]:
            key = (str(record["scenario"]), int(record["dataset_index"]))
            if key in record_keys:
                raise DistributedMonteCarloError("distributed results duplicate a dataset")
            record_keys.add(key)
            records.append(record)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as handle:
        checkpoint = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        engine_result = aggregate_production_results([checkpoint])
    finally:
        checkpoint.unlink(missing_ok=True)
    shard_result_set_sha256 = canonical_sha256(
        {"shard_result_sha256s": result_hashes}
    )
    if engine_result.get("status") in {"pass", "fail"}:
        engine_result = bind_distributed_receipts(
            engine_result,
            execution_contract_sha256=str(contract["artifact_sha256"]),
            execution_manifest_sha256=str(manifest["artifact_sha256"]),
            shard_result_set_sha256=shard_result_set_sha256,
            shard_count=len(completed),
        )
    if engine_result.get("status") == "pass" and not verify_production_result(engine_result):
        raise DistributedMonteCarloError("engine aggregate claimed an unverifiable pass")
    distributed_status = str(engine_result.get("status"))
    body = {
        "schema_version": DISTRIBUTED_AGGREGATE_SCHEMA,
        "status": distributed_status,
        "execution_manifest_sha256": manifest["artifact_sha256"],
        "shard_result_set_sha256": shard_result_set_sha256,
        "completed_shards": len(completed),
        "required_shards": len(manifest["shards"]),
        "completed_dataset_records": len(records),
        "required_dataset_records": int(manifest["dataset_record_count"]),
        "engine_result": engine_result,
        "acceptance": engine_result.get("acceptance"),
        "claim_boundary": {
            "production_gate_complete": True,
            "pass_claimed": distributed_status == "pass",
            "model_quality_evidence": False,
        },
    }
    return _content_address(body)


def cost_projection(
    manifest: Mapping[str, Any],
    measurement: Mapping[str, Any],
    *,
    authorized_cap_usd: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    verify_execution_manifest(manifest, contract=contract)
    verify_shard_telemetry(measurement)
    if measurement.get("provider") != "modal":
        raise DistributedMonteCarloError("full-run projection requires Modal telemetry")
    dataset_count = int(measurement["dataset_count"])
    observed_per_dataset = float(measurement["wall_seconds"]) / dataset_count
    campaign = contract["campaign"]
    modal = contract["modal"]
    pricing = contract["pricing_snapshot"]
    total_records = int(manifest["dataset_record_count"])
    preflight_seconds = (
        float(campaign["preflight_cpu_hour_upper"]) * 3600.0 / total_records
    )
    projected_seconds_per_dataset = max(
        preflight_seconds,
        observed_per_dataset * float(campaign["measurement_runtime_multiplier"]),
    )
    rate = (
        float(modal["cpu_cores"]) * float(pricing["cpu_usd_per_core_second"])
        + float(modal["memory_request_mib"])
        / 1024.0
        * float(pricing["memory_usd_per_gib_second"])
    )
    raw_projection = projected_seconds_per_dataset * total_records * rate
    projected_upper = raw_projection * float(campaign["billing_safety_factor"])
    function_timeout_bound = (
        int(modal["function_timeout_seconds"]) * total_records * rate
    )
    body = {
        "schema_version": COST_PROJECTION_SCHEMA,
        "execution_manifest_sha256": manifest["artifact_sha256"],
        "measurement_telemetry_sha256": measurement["artifact_sha256"],
        "observed_seconds_per_dataset": round(observed_per_dataset, 6),
        "preflight_seconds_per_dataset_upper": round(preflight_seconds, 6),
        "projected_seconds_per_dataset": round(projected_seconds_per_dataset, 6),
        "dataset_records": total_records,
        "published_rate_usd_per_function_second": round(rate, 12),
        "projected_compute_usd_before_safety_factor": round(raw_projection, 6),
        "billing_safety_factor": float(campaign["billing_safety_factor"]),
        "projected_compute_upper_usd": round(projected_upper, 6),
        "function_timeout_compute_bound_usd": round(function_timeout_bound, 6),
        "authorized_cap_usd": round(float(authorized_cap_usd), 6),
        "admission_fraction": float(campaign["budget_admission_fraction"]),
        "admissible": (
            authorized_cap_usd > 0
            and projected_upper
            <= float(campaign["budget_admission_fraction"]) * authorized_cap_usd
        ),
        "billing_limit": (
            "Published-rate function CPU and requested-memory projection; the workspace hard "
            "budget remains the external absolute stop."
        ),
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
        },
    }
    return _content_address(body)


def _governance_modal_budget(study_path: Path) -> tuple[bool, float]:
    import yaml

    document = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise DistributedMonteCarloError("governance study record is malformed")
    budget = document.get("budget")
    compute = document.get("compute")
    if not isinstance(budget, dict) or not isinstance(compute, dict):
        raise DistributedMonteCarloError("governance study lacks Modal budget controls")
    return bool(compute.get("modal_enabled")), float(budget.get("modal_cap_usd", 0))


def build_modal_admission(
    manifest: Mapping[str, Any],
    measurement: Mapping[str, Any],
    measurement_result: Mapping[str, Any],
    *,
    governance_study: Path,
    maximum_authorized_usd: float,
    workspace_hard_budget_usd: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the artifact required *before* invoking the Modal CLI."""

    enabled, governance_cap = _governance_modal_budget(governance_study)
    if not enabled or governance_cap <= 0:
        raise DistributedMonteCarloError("governance keeps Modal disabled or at a zero cap")
    if maximum_authorized_usd <= 0 or workspace_hard_budget_usd <= 0:
        raise DistributedMonteCarloError("operator and workspace caps must be positive")
    verify_modal_measurement_bundle(
        manifest,
        measurement,
        measurement_result,
        contract=contract,
    )
    run_cap = min(
        governance_cap,
        float(maximum_authorized_usd),
        float(workspace_hard_budget_usd),
    )
    projected = cost_projection(
        manifest,
        measurement,
        authorized_cap_usd=run_cap,
        contract=contract,
    )
    if projected["admissible"] is not True:
        raise DistributedMonteCarloError(
            "measured full-run projection exceeds the 85% admission threshold"
        )
    body = {
        "schema_version": MODAL_ADMISSION_SCHEMA,
        "status": "admitted_for_bounded_full_campaign",
        "execution_contract_sha256": contract["artifact_sha256"],
        "execution_manifest_sha256": manifest["artifact_sha256"],
        "measurement_telemetry_sha256": measurement["artifact_sha256"],
        "governance_study_file_sha256": _sha256_file(governance_study),
        "governance_modal_cap_usd": governance_cap,
        "operator_maximum_authorized_usd": float(maximum_authorized_usd),
        "workspace_hard_budget_usd": float(workspace_hard_budget_usd),
        "effective_run_cap_usd": run_cap,
        "cost_projection": projected,
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
            "admission_artifact_creation_creates_cloud_spend": False,
        },
    }
    return _content_address(body)


def verify_modal_admission(
    admission: Mapping[str, Any],
    manifest: Mapping[str, Any],
    measurement: Mapping[str, Any],
    measurement_result: Mapping[str, Any],
    *,
    governance_study: Path,
    contract: Mapping[str, Any],
) -> None:
    _verify_content_address(admission, label="Modal full-run admission")
    expected = build_modal_admission(
        manifest,
        measurement,
        measurement_result,
        governance_study=governance_study,
        maximum_authorized_usd=float(
            admission.get("operator_maximum_authorized_usd", 0)
        ),
        workspace_hard_budget_usd=float(
            admission.get("workspace_hard_budget_usd", 0)
        ),
        contract=contract,
    )
    if dict(admission) != expected:
        raise DistributedMonteCarloError("Modal admission is stale or mismatched")


def _load_manifest(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(path, label="distributed execution manifest")
    verify_execution_manifest(manifest, contract=contract)
    return manifest


def _select_spec(manifest: Mapping[str, Any], shard_sha256: str) -> dict[str, Any]:
    matches = [
        spec for spec in manifest["shards"] if spec["shard_sha256"] == shard_sha256
    ]
    if len(matches) != 1:
        raise DistributedMonteCarloError("requested shard is absent or duplicated")
    return dict(matches[0])


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-manifest")
    freeze.add_argument("--shard-size", type=int, default=1)
    freeze.add_argument("--output", type=Path, required=True)
    status_command = commands.add_parser("status")
    status_command.add_argument("--manifest", type=Path, required=True)
    status_command.add_argument("--output-directory", type=Path, required=True)
    shard = commands.add_parser("run-shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--shard-sha256", required=True)
    shard.add_argument("--output-directory", type=Path, required=True)
    shard.add_argument("--provider", default="local-production-image")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--output-directory", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    projection = commands.add_parser("project-cost")
    projection.add_argument("--manifest", type=Path, required=True)
    projection.add_argument("--measurement", type=Path, required=True)
    projection.add_argument("--authorized-cap-usd", type=float, required=True)
    projection.add_argument("--output", type=Path, required=True)
    admission = commands.add_parser("admit-modal-full")
    admission.add_argument("--manifest", type=Path, required=True)
    admission.add_argument("--measurement", type=Path, required=True)
    admission.add_argument("--governance-study", type=Path, required=True)
    admission.add_argument("--maximum-authorized-usd", type=float, required=True)
    admission.add_argument("--workspace-hard-budget-usd", type=float, required=True)
    admission.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    contract = load_execution_contract()
    if arguments.command == "freeze-manifest":
        _atomic_json(
            arguments.output,
            build_execution_manifest(shard_size=arguments.shard_size),
        )
        return
    manifest = _load_manifest(arguments.manifest, contract)
    if arguments.command == "status":
        print(
            json.dumps(
                execution_status(manifest, arguments.output_directory, contract=contract),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "run-shard":
        if os.getenv("FLAVOURBENCH_SEASON1_PRODUCTION_SHARD_AUTHORIZED") != (
            PRODUCTION_SHARD_AUTHORIZATION
        ):
            raise DistributedMonteCarloError(
                "production method-validation shard requires explicit authorization"
            )
        spec = _select_spec(manifest, arguments.shard_sha256)
        bundle = execute_shard(spec, provider=arguments.provider)
        print(
            json.dumps(
                write_shard_bundle(
                    arguments.output_directory,
                    spec,
                    bundle,
                    contract=contract,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "aggregate":
        _atomic_json(
            arguments.output,
            aggregate_distributed_results(
                manifest,
                arguments.output_directory,
                contract=contract,
            ),
        )
        return
    measurement = _read_json(arguments.measurement, label="Modal measurement telemetry")
    if arguments.command == "admit-modal-full":
        result_path = arguments.measurement.parent / (
            f"result-{measurement.get('shard_result_sha256')}.json"
        )
        measurement_result = _read_json(
            result_path,
            label="measured Modal shard result",
        )
        _atomic_json(
            arguments.output,
            build_modal_admission(
                manifest,
                measurement,
                measurement_result,
                governance_study=arguments.governance_study,
                maximum_authorized_usd=arguments.maximum_authorized_usd,
                workspace_hard_budget_usd=arguments.workspace_hard_budget_usd,
                contract=contract,
            ),
        )
        return
    projected = cost_projection(
        manifest,
        measurement,
        authorized_cap_usd=arguments.authorized_cap_usd,
        contract=contract,
    )
    _atomic_json(arguments.output, projected)


if __name__ == "__main__":
    run()
