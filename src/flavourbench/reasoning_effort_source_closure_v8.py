"""Build the reproducible source closure for the V8 reasoning study."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import reasoning_effort_source_closure_v7 as v7

SCHEMA_VERSION = "flavourbench-reasoning-effort-source-closure-v8"
ENVIRONMENT_SCHEMA = "flavourbench-reasoning-effort-execution-environment-v8"

ENTRYPOINT_MODULES = (
    "flavourbench.reasoning_effort_full_study_executor_v8",
    "flavourbench.reasoning_effort_full_study_v8",
    "flavourbench.reasoning_effort_human_protocol",
)
REQUIRED_MODULES = (
    *(
        value
        for value in v7.REQUIRED_MODULES
        if not value.endswith("_executor_v7")
        and not value.endswith("_full_study_v7")
        and not value.endswith("_source_closure_v7")
    ),
    "flavourbench.reasoning_effort_full_study_executor_v8",
    "flavourbench.reasoning_effort_full_study_v8",
    "flavourbench.reasoning_effort_source_closure_v8",
)

SourceClosureError = v7.SourceClosureError
_sha256 = v7._sha256


def build_execution_environment_attestation(
    third_party_imports: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    distributions, dependency_edges, import_owners = v7.v6.v5.v2.v1._distribution_closure(
        third_party_imports
    )
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "python": v7.v6.v5.v2.v1._interpreter_attestation(),
        "third_party_import_owners": import_owners,
        "distributions": distributions,
        "distribution_dependency_edges": dependency_edges,
        "secrets_or_environment_values_recorded": False,
    }
    return {**payload, "environment_sha256": _sha256(payload)}


def build_source_closure(*, repo_root: Path) -> dict[str, Any]:
    modules, edges, external_imports = v7.v6.v5.v2.v1._source_inventory(
        repo_root, ENTRYPOINT_MODULES
    )
    names = {str(record["module"]) for record in modules}
    missing = sorted(set(REQUIRED_MODULES) - names)
    if missing:
        raise SourceClosureError(
            "V8 modules are outside the transitive source closure: " + ", ".join(missing)
        )
    runtime_files = v7.v6.v5.v2.v1._runtime_inventory(repo_root)
    environment = build_execution_environment_attestation(external_imports)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "package": v7.v6.v5.v2.v1.PACKAGE,
        "entrypoint_modules": list(sorted(ENTRYPOINT_MODULES)),
        "required_modules": list(sorted(REQUIRED_MODULES)),
        "modules": modules,
        "import_edges": edges,
        "third_party_imports": external_imports,
        "runtime_files": runtime_files,
        "execution_environment": environment,
        "module_inventory_sha256": _sha256(modules),
        "import_graph_sha256": _sha256(edges),
        "runtime_files_sha256": _sha256(runtime_files),
    }
    return {**payload, "closure_sha256": _sha256(payload)}


def verify_source_closure(*, expected: Mapping[str, Any], repo_root: Path) -> None:
    if expected.get("schema_version") != SCHEMA_VERSION:
        raise SourceClosureError("unexpected V8 source-closure schema")
    body = {key: value for key, value in expected.items() if key != "closure_sha256"}
    if expected.get("closure_sha256") != _sha256(body):
        raise SourceClosureError("V8 source-closure digest does not verify")
    environment = expected.get("execution_environment")
    if not isinstance(environment, Mapping):
        raise SourceClosureError("V8 execution-environment attestation is absent")
    environment_body = {
        key: value for key, value in environment.items() if key != "environment_sha256"
    }
    if environment.get("environment_sha256") != _sha256(environment_body):
        raise SourceClosureError("V8 execution-environment digest does not verify")
    if dict(expected) != build_source_closure(repo_root=repo_root):
        raise SourceClosureError("current V8 source/import graph or environment differs")
