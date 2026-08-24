"""Build and verify the reasoning-study execution source closure.

The closure is intentionally derived from Python syntax rather than a hand-
maintained file list.  Starting from the study, executor, and human-protocol
entry points, it follows every import that resolves inside the ``flavourbench``
package.  It also binds the concrete interpreter and installed distribution
payloads needed by third-party imports.  No environment values, credentials,
network state, provider calls, or MCP calls enter the attestation.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

SCHEMA_VERSION = "flavourbench-reasoning-effort-source-closure-v1"
ENVIRONMENT_SCHEMA = "flavourbench-reasoning-effort-execution-environment-v1"
PACKAGE = "flavourbench"

ENTRYPOINT_MODULES = (
    "flavourbench.reasoning_effort_full_study_executor_v1",
    "flavourbench.reasoning_effort_full_study_v1",
    "flavourbench.reasoning_effort_human_protocol",
)

# These are governance-critical anchors, not a substitute for traversal.  A
# closure build fails if refactoring silently removes any anchor from the graph.
REQUIRED_MODULES = (
    "flavourbench.config",
    "flavourbench.execution_policy",
    "flavourbench.frontier_contract_runner",
    "flavourbench.frontier_coverage_repair_executor",
    "flavourbench.frontier_manifest",
    "flavourbench.live_smoke",
    "flavourbench.mcp_client",
    "flavourbench.provider",
    "flavourbench.reasoning_effort_full_study_executor_v1",
    "flavourbench.reasoning_effort_full_study_v1",
    "flavourbench.reasoning_effort_human_protocol",
    "flavourbench.reasoning_effort_route_gate_v4",
    "flavourbench.response_envelope_route_v4",
    "flavourbench.run_journal",
)

RUNTIME_FILES = (
    "flavourbench/Dockerfile",
    "flavourbench/pyproject.toml",
    "flavourbench/requirements.lock",
)


class SourceClosureError(RuntimeError):
    """The local source graph or execution environment cannot be attested."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SourceClosureError(f"source-closure path is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_root(repo_root: Path) -> Path:
    root = repo_root.resolve() / "flavourbench/src/flavourbench"
    if root.is_symlink() or not root.is_dir():
        raise SourceClosureError(f"flavourbench package root is unavailable: {root}")
    return root


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise SourceClosureError(f"closure path escapes the repository: {path}") from error


def _module_path(repo_root: Path, module: str) -> Path | None:
    if module != PACKAGE and not module.startswith(f"{PACKAGE}."):
        return None
    root = _package_root(repo_root)
    suffix = module.split(".")[1:]
    base = root.joinpath(*suffix)
    candidates = (
        [base.with_suffix(".py"), base / "__init__.py"] if suffix else [root / "__init__.py"]
    )
    matches = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(matches) > 1:
        raise SourceClosureError(f"ambiguous local module resolution: {module}")
    return matches[0] if matches else None


def _module_ancestors(repo_root: Path, module: str) -> set[str]:
    parts = module.split(".")
    result: set[str] = set()
    for length in range(1, len(parts) + 1):
        candidate = ".".join(parts[:length])
        if _module_path(repo_root, candidate) is not None:
            result.add(candidate)
    return result


def _resolved_from_base(module: str, path: Path, node: ast.ImportFrom) -> str:
    if not node.level:
        return str(node.module or "")
    current_package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    package_parts = current_package.split(".")
    remove = node.level - 1
    if remove >= len(package_parts):
        raise SourceClosureError(
            f"relative import escapes package in {module} at line {node.lineno}"
        )
    base = package_parts[: len(package_parts) - remove]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _imports(
    *, repo_root: Path, module: str, path: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise SourceClosureError(f"cannot parse local module {module}: {error}") from error

    local: set[str] = set()
    external: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top == PACKAGE:
                    if _module_path(repo_root, alias.name) is None:
                        raise SourceClosureError(
                            f"unresolved local import {alias.name!r} in {module}"
                        )
                    local.update(_module_ancestors(repo_root, alias.name))
                elif top not in sys.stdlib_module_names:
                    external.add(top)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _resolved_from_base(module, path, node)
        top = base.split(".", 1)[0] if base else ""
        if top == PACKAGE:
            base_path = _module_path(repo_root, base)
            if base_path is not None:
                local.update(_module_ancestors(repo_root, base))
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{base}.{alias.name}" if base else alias.name
                if _module_path(repo_root, candidate) is not None:
                    local.update(_module_ancestors(repo_root, candidate))
            if base_path is None and not any(
                _module_path(repo_root, f"{base}.{alias.name}") is not None
                for alias in node.names
                if alias.name != "*"
            ):
                raise SourceClosureError(f"unresolved local from-import {base!r} in {module}")
        elif top and top not in sys.stdlib_module_names:
            external.add(top)
    local.discard(module)
    return tuple(sorted(local)), tuple(sorted(external))


def _source_inventory(
    repo_root: Path, entrypoints: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    pending = sorted(set(entrypoints))
    visited: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    external_imports: set[str] = set()
    while pending:
        module = pending.pop(0)
        if module in visited:
            continue
        path = _module_path(repo_root, module)
        if path is None:
            raise SourceClosureError(f"source-closure entry module is absent: {module}")
        local, external = _imports(repo_root=repo_root, module=module, path=path)
        record = {
            "module": module,
            "path": _relative(repo_root, path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
            "local_imports": list(local),
            "third_party_imports": list(external),
        }
        records[module] = record
        external_imports.update(external)
        visited.add(module)
        pending = sorted(set(pending) | (set(local) - visited))

    missing = sorted(set(REQUIRED_MODULES) - visited)
    if missing:
        raise SourceClosureError(
            "governance-required modules are outside the transitive closure: " + ", ".join(missing)
        )
    modules = [records[module] for module in sorted(records)]
    edges = [
        {"importer": record["module"], "imported": imported}
        for record in modules
        for imported in record["local_imports"]
    ]
    return modules, edges, sorted(external_imports)


def _runtime_inventory(repo_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in sorted(RUNTIME_FILES):
        path = repo_root.resolve() / relative
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return records


def _distribution_payload(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    files = distribution.files
    if files is None:
        name = distribution.metadata["Name"]
        raise SourceClosureError(f"installed distribution has no auditable file inventory: {name}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for package_path in sorted(files, key=lambda value: str(value)):
        relative = str(package_path).replace(os.sep, "/")
        if relative in seen or relative.endswith(".pyc") or "/__pycache__/" in f"/{relative}":
            continue
        seen.add(relative)
        path = Path(distribution.locate_file(package_path))
        if path.is_symlink() or not path.is_file():
            raise SourceClosureError(
                f"installed distribution contains a non-regular payload: {relative}"
            )
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not records:
        name = distribution.metadata["Name"]
        raise SourceClosureError(f"installed distribution has an empty auditable payload: {name}")
    record_files = [row for row in records if row["path"].endswith(".dist-info/RECORD")]
    return {
        "file_count": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
        "file_inventory_sha256": _sha256(records),
        "record_file_sha256": (record_files[0]["sha256"] if len(record_files) == 1 else None),
    }


def _distribution_closure(
    third_party_imports: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    package_owners = importlib.metadata.packages_distributions()
    import_owners: list[dict[str, Any]] = []
    pending: set[str] = set()
    for import_name in sorted(set(third_party_imports)):
        owners = sorted({canonicalize_name(value) for value in package_owners.get(import_name, [])})
        if not owners:
            raise SourceClosureError(
                f"third-party import has no installed distribution owner: {import_name}"
            )
        import_owners.append({"import_name": import_name, "distributions": owners})
        pending.update(owners)

    environment = default_environment()
    environment["extra"] = ""
    visited: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str]] = set()
    while pending:
        name = sorted(pending)[0]
        pending.remove(name)
        if name in visited:
            continue
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise SourceClosureError(
                f"required installed distribution is absent: {name}"
            ) from error
        canonical_name = canonicalize_name(distribution.metadata["Name"])
        if canonical_name != name:
            raise SourceClosureError(
                f"distribution canonical name differs: requested {name}, got {canonical_name}"
            )
        dependencies: set[str] = set()
        for text in distribution.requires or []:
            try:
                requirement = Requirement(text)
            except InvalidRequirement as error:
                raise SourceClosureError(
                    f"invalid installed requirement for {name}: {text}"
                ) from error
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            dependency = canonicalize_name(requirement.name)
            try:
                importlib.metadata.distribution(dependency)
            except importlib.metadata.PackageNotFoundError as error:
                raise SourceClosureError(
                    f"active dependency {dependency} required by {name} is absent"
                ) from error
            dependencies.add(dependency)
            edges.add((name, dependency))
        payload = _distribution_payload(distribution)
        records[name] = {
            "name": name,
            "version": distribution.version,
            "dependencies": sorted(dependencies),
            **payload,
        }
        visited.add(name)
        pending.update(dependencies - visited)

    return (
        [records[name] for name in sorted(records)],
        [{"distribution": left, "dependency": right} for left, right in sorted(edges)],
        import_owners,
    )


def _interpreter_attestation() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    if executable.is_symlink() or not executable.is_file():
        raise SourceClosureError("Python executable is not a regular resolved file")
    return {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:5]),
        "hexversion": sys.hexversion,
        "cache_tag": sys.implementation.cache_tag,
        "abi_flags": getattr(sys, "abiflags", ""),
        "soabi": sysconfig.get_config_var("SOABI"),
        "platform": {
            "sys_platform": sys.platform,
            "system": platform.system(),
            "machine": platform.machine(),
            "architecture": list(platform.architecture()),
            "libc": list(platform.libc_ver()),
            "python_compiler": platform.python_compiler(),
        },
        "executable_bytes": executable.stat().st_size,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }


def build_execution_environment_attestation(
    third_party_imports: Sequence[str],
) -> dict[str, Any]:
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
    modules, edges, external_imports = _source_inventory(repo_root, ENTRYPOINT_MODULES)
    runtime_files = _runtime_inventory(repo_root)
    execution_environment = build_execution_environment_attestation(external_imports)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "package": PACKAGE,
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
    """Rebuild the complete closure and require byte-for-byte semantic identity."""

    if expected.get("schema_version") != SCHEMA_VERSION:
        raise SourceClosureError("unexpected reasoning source-closure schema")
    body = {key: value for key, value in expected.items() if key != "closure_sha256"}
    if expected.get("closure_sha256") != _sha256(body):
        raise SourceClosureError("frozen reasoning source-closure digest does not verify")
    environment = expected.get("execution_environment")
    if not isinstance(environment, Mapping):
        raise SourceClosureError("frozen execution-environment attestation is absent")
    environment_body = {
        key: value for key, value in environment.items() if key != "environment_sha256"
    }
    if environment.get("environment_sha256") != _sha256(environment_body):
        raise SourceClosureError("frozen execution-environment digest does not verify")
    current = build_source_closure(repo_root=repo_root)
    if dict(expected) != current:
        raise SourceClosureError(
            "current source/import graph or installed execution environment differs from freeze"
        )
