"""Build the import-safe reasoning-effort successor source closure.

This successor deliberately leaves the v1 closure immutable.  It starts from
the new study and executor entry points, follows their complete in-package
import graph, and binds the same interpreter, installed distributions, and
runtime files as the original closure implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import reasoning_effort_source_closure_v1 as v1

SCHEMA_VERSION = "flavourbench-reasoning-effort-source-closure-v2"
ENVIRONMENT_SCHEMA = "flavourbench-reasoning-effort-execution-environment-v2"

ENTRYPOINT_MODULES = (
    "flavourbench.reasoning_effort_full_study_executor_v2",
    "flavourbench.reasoning_effort_full_study_v2",
    "flavourbench.reasoning_effort_human_protocol",
)

REQUIRED_MODULES = (
    *v1.REQUIRED_MODULES,
    "flavourbench.reasoning_effort_full_study_executor_v2",
    "flavourbench.reasoning_effort_full_study_v2",
    "flavourbench.reasoning_effort_source_closure_v2",
)

RUNTIME_FILES = v1.RUNTIME_FILES
SourceClosureError = v1.SourceClosureError
_sha256 = v1._sha256


def build_execution_environment_attestation(
    third_party_imports: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    distributions, dependency_edges, import_owners = v1._distribution_closure(
        third_party_imports
    )
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "python": v1._interpreter_attestation(),
        "third_party_import_owners": import_owners,
        "distributions": distributions,
        "distribution_dependency_edges": dependency_edges,
        "secrets_or_environment_values_recorded": False,
    }
    return {**payload, "environment_sha256": _sha256(payload)}


def build_source_closure(*, repo_root: Path) -> dict[str, Any]:
    modules, edges, external_imports = v1._source_inventory(repo_root, ENTRYPOINT_MODULES)
    module_names = {str(record["module"]) for record in modules}
    missing = sorted(set(REQUIRED_MODULES) - module_names)
    if missing:
        raise SourceClosureError(
            "successor governance modules are outside the transitive closure: "
            + ", ".join(missing)
        )
    runtime_files = v1._runtime_inventory(repo_root)
    execution_environment = build_execution_environment_attestation(external_imports)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "package": v1.PACKAGE,
        "entrypoint_modules": list(sorted(ENTRYPOINT_MODULES)),
        "required_modules": list(sorted(REQUIRED_MODULES)),
        "modules": modules,
        "import_edges": edges,
        "third_party_imports": external_imports,
        "runtime_files": runtime_files,
        "execution_environment": execution_environment,
        "module_inventory_sha256": _sha256(modules),
        "import_graph_sha256": _sha256(edges),
        "runtime_files_sha256": _sha256(runtime_files),
    }
    return {**payload, "closure_sha256": _sha256(payload)}


def verify_source_closure(*, expected: Mapping[str, Any], repo_root: Path) -> None:
    """Rebuild and compare the complete successor execution closure."""

    if expected.get("schema_version") != SCHEMA_VERSION:
        raise SourceClosureError("unexpected successor reasoning source-closure schema")
    body = {key: value for key, value in expected.items() if key != "closure_sha256"}
    if expected.get("closure_sha256") != _sha256(body):
        raise SourceClosureError("successor source-closure digest does not verify")
    environment = expected.get("execution_environment")
    if not isinstance(environment, Mapping):
        raise SourceClosureError("successor execution-environment attestation is absent")
    environment_body = {
        key: value for key, value in environment.items() if key != "environment_sha256"
    }
    if environment.get("environment_sha256") != _sha256(environment_body):
        raise SourceClosureError("successor execution-environment digest does not verify")
    if dict(expected) != build_source_closure(repo_root=repo_root):
        raise SourceClosureError(
            "current successor source/import graph or execution environment differs"
        )
