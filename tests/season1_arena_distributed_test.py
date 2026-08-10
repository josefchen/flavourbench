from __future__ import annotations

import json
from pathlib import Path

import pytest

import flavourbench.season1_arena_distributed as distributed
from flavourbench.season1_arena_acceptance import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "contracts/season1/method-validation/"
    "season1-arena-distributed-execution-v2.json"
)
RETIRED_CONTRACT_PATH = ROOT / (
    "contracts/season1/method-validation/"
    "season1-arena-distributed-execution-v1.json"
)


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    return distributed.load_execution_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def manifest() -> dict[str, object]:
    return distributed.build_execution_manifest(shard_size=1, contract_path=CONTRACT_PATH)


def _fake_record(spec: dict[str, object], dataset_index: int) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "flavourbench-season1-arena-production-monte-carlo-v1",
        "policy_sha256": spec["policy_sha256"],
        "layout_sha256": spec["layout_sha256"],
        "engine_source_bundle_sha256": spec["engine_source_bundle_sha256"],
        "scenario": spec["scenario"],
        "dataset_index": dataset_index,
        "dataset_seed": dataset_index,
        "bootstrap_replicates": spec["bootstrap_replicates"],
        "engine": "production_equation_exact_task_by_rater_cluster_bootstrap",
        "production_mode": True,
        "status": "completed",
        "analysis": {
            "bootstrap_replicates_executed": spec["bootstrap_replicates"],
        },
        "claim_boundary": {
            "counts_toward_production_gate": True,
            "aggregate_acceptance_claimed": False,
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
        },
    }
    return {**body, "record_sha256": canonical_sha256(body)}


def _fake_telemetry(spec: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": distributed.SHARD_TELEMETRY_SCHEMA,
        "shard_spec_sha256": spec["shard_sha256"],
        "shard_result_sha256": result["artifact_sha256"],
        "provider": "modal",
        "started_at": "2026-08-03T00:00:00+00:00",
        "completed_at": "2026-08-03T00:01:00+00:00",
        "dataset_count": 1,
        "wall_seconds": 60.0,
        "process_cpu_seconds": 59.0,
        "maximum_resident_set_kibibytes": 350_000,
        "processor_model": "test",
        "runtime_identity": {"verified": True},
        "published_rate_compute_estimate_usd": 0.001,
        "published_rate_snapshot_url": "https://modal.com/pricing",
        "published_rate_snapshot_date": "2026-08-03",
        "billing_limit": "test fixture",
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
        },
    }
    return {**body, "artifact_sha256": canonical_sha256(body)}


def _fake_bundle(
    spec: dict[str, object], contract: dict[str, object]
) -> dict[str, dict[str, object]]:
    records = [_fake_record(spec, int(spec["start"]))]
    body: dict[str, object] = {
        "schema_version": distributed.SHARD_RESULT_SCHEMA,
        "shard_spec_sha256": spec["shard_sha256"],
        "execution_contract_sha256": contract["artifact_sha256"],
        "policy_sha256": contract["policy_artifact_sha256"],
        "layout_sha256": contract["layout_artifact_sha256"],
        "engine_source_sha256": spec["engine_source_sha256"],
        "engine_source_bundle_sha256": spec["engine_source_bundle_sha256"],
        "record_count": 1,
        "record_set_sha256": canonical_sha256(
            {"record_sha256s": [records[0]["record_sha256"]]}
        ),
        "records": records,
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
            "aggregate_acceptance_claimed": False,
        },
    }
    result = {**body, "artifact_sha256": canonical_sha256(body)}
    return {"result": result, "telemetry": _fake_telemetry(spec, result)}


def test_execution_contract_binds_every_worker_source(contract: dict[str, object]) -> None:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    digest = document.pop("artifact_sha256")
    assert digest == canonical_sha256(document)
    for relative, expected in contract["implementation_files"].items():
        assert distributed._sha256_file(ROOT / relative) == expected
    assert contract["contract_revision"] == 2
    assert "pyproject.toml" not in contract["implementation_files"]
    projection = contract["runtime_projection"]
    assert projection["path"] in contract["implementation_files"]
    assert projection["sha256"] == contract["implementation_files"][projection["path"]]


def test_retired_contract_is_immutable_and_explicitly_superseded(
    contract: dict[str, object],
) -> None:
    retired = json.loads(RETIRED_CONTRACT_PATH.read_text(encoding="utf-8"))
    digest = retired.pop("artifact_sha256")
    assert digest == canonical_sha256(retired)
    assert digest == "2fe57592e1850cfe32def554bc8a29c03184a95978b59e173d8d30e8aa6571c7"
    assert contract["supersedes_execution_contract_sha256"] == digest
    assert retired["implementation_files"]["pyproject.toml"] != distributed._sha256_file(
        ROOT / "pyproject.toml"
    )


def test_manifest_is_deterministic_and_exactly_covers_campaign(
    contract: dict[str, object], manifest: dict[str, object]
) -> None:
    assert manifest == distributed.build_execution_manifest(
        shard_size=1, contract_path=CONTRACT_PATH
    )
    distributed.verify_execution_manifest(manifest, contract=contract)
    assert manifest["shard_count"] == 16_000
    assert manifest["dataset_record_count"] == 16_000
    assert len({shard["shard_sha256"] for shard in manifest["shards"]}) == 16_000


def test_tampered_shard_spec_fails_closed(
    contract: dict[str, object], manifest: dict[str, object]
) -> None:
    spec = dict(manifest["shards"][0])
    spec["start"] = 1
    with pytest.raises(distributed.DistributedMonteCarloError, match="content address"):
        distributed.verify_shard_spec(spec, contract=contract)


def test_worker_checks_runtime_before_dataset_execution(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict[str, object],
    manifest: dict[str, object],
) -> None:
    spec = dict(manifest["shards"][0])
    calls: list[str] = []

    def verified(_: dict[str, object]) -> dict[str, object]:
        calls.append("runtime")
        return {"verified": True}

    def dataset(**kwargs: object) -> dict[str, object]:
        assert calls == ["runtime"]
        calls.append("dataset")
        return _fake_record(spec, int(kwargs["dataset_index"]))

    monkeypatch.setattr(distributed, "verify_runtime_identity", verified)
    monkeypatch.setattr(distributed, "run_dataset", dataset)
    bundle = distributed.execute_shard(
        spec,
        provider="modal",
        contract_path=CONTRACT_PATH,
    )
    distributed.verify_shard_result(spec, bundle["result"], contract=contract)
    assert calls == ["runtime", "dataset"]
    assert bundle["result"]["claim_boundary"]["model_quality_evidence"] is False


def test_shard_write_is_resumable_and_divergence_fails(
    tmp_path: Path,
    contract: dict[str, object],
    manifest: dict[str, object],
) -> None:
    spec = dict(manifest["shards"][0])
    bundle = _fake_bundle(spec, contract)
    first = distributed.write_shard_bundle(
        tmp_path, spec, bundle, contract=contract
    )
    second = distributed.write_shard_bundle(
        tmp_path, spec, bundle, contract=contract
    )
    assert first == second

    divergent = _fake_bundle(spec, contract)
    divergent["result"]["records"][0]["dataset_seed"] = 99
    record = divergent["result"]["records"][0]
    record["record_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    result_body = {
        key: value
        for key, value in divergent["result"].items()
        if key != "artifact_sha256"
    }
    result_body["record_set_sha256"] = canonical_sha256(
        {"record_sha256s": [record["record_sha256"]]}
    )
    divergent["result"] = {
        **result_body,
        "artifact_sha256": canonical_sha256(result_body),
    }
    divergent["telemetry"] = _fake_telemetry(spec, divergent["result"])
    with pytest.raises(distributed.DistributedMonteCarloError, match="diverged"):
        distributed.write_shard_bundle(
            tmp_path, spec, divergent, contract=contract
        )


def test_incomplete_distributed_aggregate_cannot_claim_acceptance(
    tmp_path: Path,
    contract: dict[str, object],
    manifest: dict[str, object],
) -> None:
    spec = dict(manifest["shards"][0])
    distributed.write_shard_bundle(
        tmp_path, spec, _fake_bundle(spec, contract), contract=contract
    )
    result = distributed.aggregate_distributed_results(
        manifest, tmp_path, contract=contract
    )
    assert result["status"] == "required_not_yet_executed"
    assert result["completed_dataset_records"] == 1
    assert result["required_dataset_records"] == 16_000
    assert result["acceptance"] is None
    assert result["claim_boundary"]["pass_claimed"] is False


def test_modal_projection_is_content_addressed_and_zero_cap_is_blocked(
    contract: dict[str, object], manifest: dict[str, object]
) -> None:
    spec = dict(manifest["shards"][0])
    bundle = _fake_bundle(spec, contract)
    zero = distributed.cost_projection(
        manifest,
        bundle["telemetry"],
        authorized_cap_usd=0,
        contract=contract,
    )
    assert zero["admissible"] is False
    assert zero["projected_compute_upper_usd"] > 0
    admitted = distributed.cost_projection(
        manifest,
        bundle["telemetry"],
        authorized_cap_usd=100,
        contract=contract,
    )
    assert admitted["admissible"] is True
    assert admitted["artifact_sha256"] == canonical_sha256(
        {key: value for key, value in admitted.items() if key != "artifact_sha256"}
    )


def test_modal_admission_fails_before_cli_under_current_governance(
    contract: dict[str, object], manifest: dict[str, object]
) -> None:
    spec = dict(manifest["shards"][0])
    bundle = _fake_bundle(spec, contract)
    with pytest.raises(
        distributed.DistributedMonteCarloError,
        match="disabled or at a zero cap",
    ):
        distributed.build_modal_admission(
            manifest,
            bundle["telemetry"],
            bundle["result"],
            governance_study=ROOT.parent / "protocol/study.yaml",
            maximum_authorized_usd=100,
            workspace_hard_budget_usd=100,
            contract=contract,
        )


def test_modal_admission_requires_frozen_runtime_measurement(
    tmp_path: Path,
    contract: dict[str, object],
    manifest: dict[str, object],
) -> None:
    study = tmp_path / "study.yaml"
    study.write_text(
        "budget:\n  modal_cap_usd: 100\ncompute:\n  modal_enabled: true\n",
        encoding="utf-8",
    )
    spec = dict(manifest["shards"][0])
    bundle = _fake_bundle(spec, contract)
    with pytest.raises(
        distributed.DistributedMonteCarloError,
        match="runtime identity",
    ):
        distributed.build_modal_admission(
            manifest,
            bundle["telemetry"],
            bundle["result"],
            governance_study=study,
            maximum_authorized_usd=100,
            workspace_hard_budget_usd=100,
            contract=contract,
        )

    telemetry_body = {
        key: value
        for key, value in bundle["telemetry"].items()
        if key != "artifact_sha256"
    }
    telemetry_body["runtime_identity"] = {
        "verified": True,
        "python_version": contract["production_image"]["python_version"],
        "machine": contract["production_image"]["machine"],
        "thread_environment": contract["production_image"]["thread_environment"],
        "dependency_versions": contract["production_image"]["dependency_versions"],
        "implementation_files": contract["implementation_files"],
        "policy_artifact_sha256": contract["policy_artifact_sha256"],
        "layout_artifact_sha256": contract["layout_artifact_sha256"],
    }
    telemetry = {
        **telemetry_body,
        "artifact_sha256": canonical_sha256(telemetry_body),
    }
    admission = distributed.build_modal_admission(
        manifest,
        telemetry,
        bundle["result"],
        governance_study=study,
        maximum_authorized_usd=100,
        workspace_hard_budget_usd=100,
        contract=contract,
    )
    distributed.verify_modal_admission(
        admission,
        manifest,
        telemetry,
        bundle["result"],
        governance_study=study,
        contract=contract,
    )
    assert admission["status"] == "admitted_for_bounded_full_campaign"
