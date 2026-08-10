from __future__ import annotations

from pathlib import Path

import pytest

import flavourbench.reasoning_effort_source_closure_v1 as closure

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


@pytest.fixture(scope="module")
def frozen_closure() -> dict:
    return closure.build_source_closure(repo_root=REPO_ROOT)


def test_closure_is_transitive_deterministic_and_environment_bound(
    frozen_closure: dict,
) -> None:
    repeated = closure.build_source_closure(repo_root=REPO_ROOT)
    assert repeated == frozen_closure
    modules = {row["module"] for row in frozen_closure["modules"]}
    assert set(closure.REQUIRED_MODULES) <= modules
    assert {
        "flavourbench.reasoning_effort_route_gate_v4",
        "flavourbench.response_envelope_route_v4",
        "flavourbench.frontier_coverage_repair_executor",
        "flavourbench.frontier_contract_runner",
        "flavourbench.config",
        "flavourbench.frontier_manifest",
        "flavourbench.provider",
        "flavourbench.mcp_client",
        "flavourbench.run_journal",
        "flavourbench.execution_policy",
    } <= modules
    assert len(modules) == len(frozen_closure["modules"])
    assert frozen_closure["module_inventory_sha256"] == closure._sha256(frozen_closure["modules"])
    assert frozen_closure["import_graph_sha256"] == closure._sha256(frozen_closure["import_edges"])
    assert {row["path"] for row in frozen_closure["runtime_files"]} == set(closure.RUNTIME_FILES)

    environment = frozen_closure["execution_environment"]
    assert environment["secrets_or_environment_values_recorded"] is False
    assert environment["python"]["implementation"] == "cpython"
    assert environment["python"]["soabi"]
    distributions = {row["name"]: row for row in environment["distributions"]}
    for owner in environment["third_party_import_owners"]:
        assert set(owner["distributions"]) <= set(distributions)
    assert {"httpx", "pydantic", "sqlalchemy", "boto3", "cryptography"} <= set(distributions)
    assert all(row["file_count"] > 0 for row in distributions.values())
    assert all(len(row["file_inventory_sha256"]) == 64 for row in distributions.values())


def test_monkeypatched_transitive_source_payload_drift_is_rejected(
    frozen_closure: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = closure._file_sha256

    def drift(path: Path) -> str:
        if path.name == "config.py":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(closure, "_file_sha256", drift)
    with pytest.raises(closure.SourceClosureError, match="current source/import graph"):
        closure.verify_source_closure(expected=frozen_closure, repo_root=REPO_ROOT)


def test_monkeypatched_import_graph_drift_is_rejected(
    frozen_closure: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = closure._imports

    def drift(*, repo_root: Path, module: str, path: Path):
        local, external = original(repo_root=repo_root, module=module, path=path)
        if module == "flavourbench.reasoning_effort_full_study_executor_v1":
            local = tuple(
                value for value in local if value != "flavourbench.reasoning_effort_route_gate_v4"
            )
        return local, external

    monkeypatch.setattr(closure, "_imports", drift)
    with pytest.raises(closure.SourceClosureError, match="current source/import graph"):
        closure.verify_source_closure(expected=frozen_closure, repo_root=REPO_ROOT)


def test_monkeypatched_installed_distribution_payload_drift_is_rejected(
    frozen_closure: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = closure._distribution_payload

    def drift(distribution):
        payload = original(distribution)
        if closure.canonicalize_name(distribution.metadata["Name"]) == "httpx":
            payload = {**payload, "file_inventory_sha256": "f" * 64}
        return payload

    monkeypatch.setattr(closure, "_distribution_payload", drift)
    with pytest.raises(closure.SourceClosureError, match="installed execution environment"):
        closure.verify_source_closure(expected=frozen_closure, repo_root=REPO_ROOT)


def test_monkeypatched_platform_drift_is_rejected(
    frozen_closure: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(closure.platform, "machine", lambda: "drifted-machine")
    with pytest.raises(closure.SourceClosureError, match="installed execution environment"):
        closure.verify_source_closure(expected=frozen_closure, repo_root=REPO_ROOT)
