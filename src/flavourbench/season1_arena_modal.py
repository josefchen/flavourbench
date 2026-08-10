"""Modal CPU entrypoints for the frozen Season 1 arena method validation.

Use ``modal run ...::measure`` for one production-image measurement. The
``full`` entrypoint remains blocked unless the repository governance cap,
operator cap, workspace-budget attestation, and measured projection all pass.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import modal

from flavourbench.season1_arena_distributed import (
    DistributedMonteCarloError,
    _completed_results,
    _read_json,
    _select_spec,
    aggregate_distributed_results,
    cost_projection,
    load_execution_contract,
    verify_execution_manifest,
    verify_modal_admission,
    verify_shard_telemetry,
    write_shard_bundle,
)

APP_NAME = "flavourbench-season1-arena-method-validation"
MODAL_SDK_VERSION = "1.5.3"
FULL_RUN_AUTHORIZATION = "I_AUTHORIZE_THE_BOUNDED_SEASON1_MONTE_CARLO"
WORKSPACE_BUDGET_ATTESTATION = "I_CONFIRMED_THE_MODAL_WORKSPACE_HARD_BUDGET"

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / (
    "contracts/season1/method-validation/"
    "season1-arena-distributed-execution-v2.json"
)
CONTRACT = load_execution_contract(CONTRACT_PATH)
MODAL_CONFIG = CONTRACT["modal"]
if importlib.metadata.version("modal") != MODAL_SDK_VERSION:
    raise DistributedMonteCarloError("Modal controller SDK differs from the frozen version")

image = modal.Image.from_dockerfile(
    ROOT / "Dockerfile",
    context_dir=ROOT,
    ignore=[
        ".venv/**",
        ".pytest_cache/**",
        ".ruff_cache/**",
        "artifacts/**",
        "tests/**",
        "gpu/**",
        "**/__pycache__/**",
    ],
).env(
    {
        "PYTHONPATH": "/app/src",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "JAX_ENABLE_X64": "true",
    }
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=float(MODAL_CONFIG["cpu_cores"]),
    memory=(
        int(MODAL_CONFIG["memory_request_mib"]),
        int(MODAL_CONFIG["memory_limit_mib"]),
    ),
    timeout=int(MODAL_CONFIG["function_timeout_seconds"]),
    max_containers=1,
)
def measured_production_shard(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from flavourbench.season1_arena_distributed import execute_shard

    return execute_shard(spec, provider="modal", contract_path=CONTRACT_PATH)


@app.function(
    image=image,
    cpu=float(MODAL_CONFIG["cpu_cores"]),
    memory=(
        int(MODAL_CONFIG["memory_request_mib"]),
        int(MODAL_CONFIG["memory_limit_mib"]),
    ),
    timeout=int(MODAL_CONFIG["function_timeout_seconds"]),
    max_containers=int(MODAL_CONFIG["maximum_containers"]),
)
def production_shard(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from flavourbench.season1_arena_distributed import execute_shard

    return execute_shard(spec, provider="modal", contract_path=CONTRACT_PATH)


def _load_manifest(path: str) -> dict[str, Any]:
    manifest = _read_json(Path(path), label="distributed execution manifest")
    verify_execution_manifest(manifest, contract=CONTRACT)
    return manifest


def _function_second_rate() -> float:
    pricing = CONTRACT["pricing_snapshot"]
    return (
        float(MODAL_CONFIG["cpu_cores"])
        * float(pricing["cpu_usd_per_core_second"])
        + float(MODAL_CONFIG["memory_request_mib"])
        / 1024.0
        * float(pricing["memory_usd_per_gib_second"])
    )


def _maximum_single_shard_compute_usd(spec: dict[str, Any]) -> float:
    return (
        int(MODAL_CONFIG["function_timeout_seconds"])
        * int(spec["count"])
        * _function_second_rate()
    )


def _load_modal_measurement(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    measurement_path = Path(path)
    measurement = _read_json(measurement_path, label="Modal measurement telemetry")
    verify_shard_telemetry(measurement)
    if measurement.get("provider") != "modal":
        raise DistributedMonteCarloError("measurement was not produced on Modal")
    result = _read_json(
        measurement_path.parent
        / f"result-{measurement.get('shard_result_sha256')}.json",
        label="measured Modal shard result",
    )
    return measurement, result


def _existing_modal_estimate(output_directory: Path) -> float:
    total = 0.0
    for path in output_directory.glob("shards/*/telemetry-*.json"):
        telemetry = _read_json(path, label="existing shard telemetry")
        verify_shard_telemetry(telemetry)
        if telemetry.get("provider") == "modal":
            total += float(telemetry.get("published_rate_compute_estimate_usd", 0))
    return total


@app.local_entrypoint()
def measure(
    manifest: str,
    output_directory: str,
    maximum_authorized_usd: float,
    shard_sha256: str = "",
) -> None:
    """Run exactly one one-dataset shard in the frozen production image."""

    execution_manifest = _load_manifest(manifest)
    spec = (
        _select_spec(execution_manifest, shard_sha256)
        if shard_sha256
        else dict(execution_manifest["shards"][0])
    )
    if int(spec["count"]) != 1:
        raise DistributedMonteCarloError(
            "measurement requires a manifest with one dataset per shard"
        )
    compute_bound = _maximum_single_shard_compute_usd(spec)
    if maximum_authorized_usd <= 0 or compute_bound > maximum_authorized_usd:
        raise DistributedMonteCarloError(
            f"one-shard function compute bound ${compute_bound:.6f} exceeds authorization"
        )
    bundle = measured_production_shard.remote(spec)
    paths = write_shard_bundle(
        Path(output_directory), spec, bundle, contract=CONTRACT
    )
    print(
        json.dumps(
            {
                "status": "measured_one_production_image_shard",
                "shard_sha256": spec["shard_sha256"],
                "maximum_authorized_usd": maximum_authorized_usd,
                "function_compute_bound_usd": round(compute_bound, 9),
                **paths,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def full(
    manifest: str,
    output_directory: str,
    measurement: str,
    governance_study: str,
    admission: str,
    maximum_shards: int = 0,
) -> None:
    """Run resumable waves only after every monetary admission gate passes."""

    if os.getenv("FLAVOURBENCH_MODAL_FULL_RUN_AUTHORIZED") != FULL_RUN_AUTHORIZATION:
        raise DistributedMonteCarloError("full Modal campaign lacks operator authorization")
    if os.getenv("FLAVOURBENCH_MODAL_WORKSPACE_BUDGET_CONFIRMED") != (
        WORKSPACE_BUDGET_ATTESTATION
    ):
        raise DistributedMonteCarloError("Modal workspace hard budget is not attested")
    execution_manifest = _load_manifest(manifest)
    measured, measured_result = _load_modal_measurement(measurement)
    admitted = _read_json(Path(admission), label="Modal full-run admission")
    verify_modal_admission(
        admitted,
        execution_manifest,
        measured,
        measured_result,
        governance_study=Path(governance_study),
        contract=CONTRACT,
    )
    run_cap = float(admitted["effective_run_cap_usd"])
    projected = cost_projection(
        execution_manifest,
        measured,
        authorized_cap_usd=run_cap,
        contract=CONTRACT,
    )
    if projected["admissible"] is not True:
        raise DistributedMonteCarloError(
            "measured full-run projection exceeds the 85% admission threshold"
        )

    output = Path(output_directory)
    completed = _completed_results(execution_manifest, output, contract=CONTRACT)
    pending = [
        dict(spec)
        for spec in execution_manifest["shards"]
        if str(spec["shard_sha256"]) not in completed
    ]
    if maximum_shards > 0:
        pending = pending[:maximum_shards]
    wave_size = int(MODAL_CONFIG["admission_wave_size"])
    estimated_spend = _existing_modal_estimate(output)
    admission_limit = float(CONTRACT["campaign"]["budget_admission_fraction"]) * run_cap
    for offset in range(0, len(pending), wave_size):
        wave = pending[offset : offset + wave_size]
        wave_bound = sum(_maximum_single_shard_compute_usd(spec) for spec in wave)
        if estimated_spend + wave_bound > run_cap:
            raise DistributedMonteCarloError("next wave could exceed the bounded run cap")
        for spec, bundle in zip(
            wave,
            production_shard.map(wave, order_outputs=True),
            strict=True,
        ):
            write_shard_bundle(output, spec, bundle, contract=CONTRACT)
            estimated_spend += float(
                bundle["telemetry"]["published_rate_compute_estimate_usd"]
            )
        if estimated_spend >= admission_limit:
            print(
                json.dumps(
                    {
                        "status": "admission_drained_at_85_percent",
                        "estimated_compute_usd": round(estimated_spend, 6),
                        "run_cap_usd": run_cap,
                    },
                    sort_keys=True,
                )
            )
            return

    aggregate = aggregate_distributed_results(
        execution_manifest, output, contract=CONTRACT
    )
    aggregate_path = output / f"aggregate-{aggregate['artifact_sha256']}.json"
    from flavourbench.season1_arena_distributed import _atomic_json

    _atomic_json(aggregate_path, aggregate)
    print(
        json.dumps(
            {
                "status": aggregate["status"],
                "aggregate": str(aggregate_path),
                "estimated_compute_usd": round(estimated_spend, 6),
                "run_cap_usd": run_cap,
            },
            indent=2,
            sort_keys=True,
        )
    )
