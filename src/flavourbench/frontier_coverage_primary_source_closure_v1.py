"""Source and environment closure for the prospective coverage executor.

This module is deliberately independent of the historical v4 reconstruction
contract.  It follows every local import reachable from the successor freezer
and executor and binds the installed third-party distribution payloads.  It
never reads credentials and has no network-capable operation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .reasoning_effort_source_closure_v1 import (
    SourceClosureError,
    _distribution_closure,
    _imports,
    _interpreter_attestation,
    _module_ancestors,
    _module_path,
    _runtime_inventory,
    _sha256,
)

SCHEMA_VERSION = "flavourbench-frontier-coverage-primary-source-closure-v1"
ENVIRONMENT_SCHEMA = "flavourbench-frontier-coverage-primary-environment-v1"
ENTRYPOINT_MODULES = (
    "flavourbench.frontier_coverage_primary_executor_v1",
    "flavourbench.frontier_coverage_primary_source_closure_v1",
    "flavourbench.frontier_coverage_primary_successor_v1",
)
REQUIRED_MODULES = (
    "flavourbench.config",
    "flavourbench.direct_cohere_pair",
    "flavourbench.direct_kimi_pair",
    "flavourbench.execution_policy",
    "flavourbench.frontier_contract_runner",
    "flavourbench.frontier_coverage_primary_executor_v1",
    "flavourbench.frontier_coverage_primary_source_closure_v1",
    "flavourbench.frontier_coverage_primary_successor_v1",
    "flavourbench.live_smoke",
    "flavourbench.mcp_client",
    "flavourbench.provider",
    "flavourbench.run_journal",
)


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise SourceClosureError(f"source path escapes the repository: {path}") from error


def _source_inventory(
    repo_root: Path, entrypoints: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    pending = sorted(set(entrypoints))
    visited: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    external: set[str] = set()
    while pending:
        module = pending.pop(0)
        if module in visited:
            continue
        path = _module_path(repo_root, module)
        if path is None:
            raise SourceClosureError(f"source-closure module is absent: {module}")
        local, third_party = _imports(repo_root=repo_root, module=module, path=path)
        records[module] = {
            "module": module,
            "path": _relative(repo_root, path),
            "bytes": path.stat().st_size,
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "local_imports": list(local),
            "third_party_imports": list(third_party),
        }
        external.update(third_party)
        visited.add(module)
        expanded = set(local)
        for imported in local:
            expanded.update(_module_ancestors(repo_root, imported))
        pending = sorted(set(pending) | (expanded - visited))
    missing = sorted(set(REQUIRED_MODULES) - visited)
    if missing:
        raise SourceClosureError(
            "coverage governance modules are outside the source closure: " + ", ".join(missing)
        )
    modules = [records[module] for module in sorted(records)]
    edges = [
        {"importer": row["module"], "imported": imported}
        for row in modules
        for imported in row["local_imports"]
    ]
    return modules, edges, sorted(external)


def _environment(third_party_imports: Sequence[str]) -> dict[str, Any]:
    distributions, dependency_edges, import_owners = _distribution_closure(third_party_imports)
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "python": _interpreter_attestation(),
        "third_party_import_owners": import_owners,
        "distributions": distributions,
        "distribution_dependency_edges": dependency_edges,
        "secrets_or_environment_values_recorded": False,
    }
    return {**payload, "environment_sha256": _sha256(payload)}


def build_source_closure(*, repo_root: Path) -> dict[str, Any]:
    modules, edges, external = _source_inventory(repo_root, ENTRYPOINT_MODULES)
    runtime_files = _runtime_inventory(repo_root)
    environment = _environment(external)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entrypoint_modules": list(sorted(ENTRYPOINT_MODULES)),
        "required_modules": list(sorted(REQUIRED_MODULES)),
        "modules": modules,
        "import_edges": edges,
        "third_party_imports": external,
        "runtime_files": runtime_files,
        "execution_environment": environment,
        "module_inventory_sha256": _sha256(modules),
        "import_graph_sha256": _sha256(edges),
        "runtime_files_sha256": _sha256(runtime_files),
    }
    return {**payload, "closure_sha256": _sha256(payload)}


def verify_source_closure(*, expected: Mapping[str, Any], repo_root: Path) -> None:
    if expected.get("schema_version") != SCHEMA_VERSION:
        raise SourceClosureError("unexpected coverage source-closure schema")
    body = {key: value for key, value in expected.items() if key != "closure_sha256"}
    if expected.get("closure_sha256") != _sha256(body):
        raise SourceClosureError("coverage source-closure digest does not verify")
    environment = expected.get("execution_environment")
    if not isinstance(environment, Mapping):
        raise SourceClosureError("coverage execution-environment attestation is absent")
    environment_body = {
        key: value for key, value in environment.items() if key != "environment_sha256"
    }
    if environment.get("environment_sha256") != _sha256(environment_body):
        raise SourceClosureError("coverage environment digest does not verify")
    if dict(expected) != build_source_closure(repo_root=repo_root):
        raise SourceClosureError(
            "current coverage source graph or execution environment differs from freeze"
        )
