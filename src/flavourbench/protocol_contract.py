from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from .arena import CONTROLLED_SCHEDULER_VERSION, SCHEDULER_VERSION
from .config import Settings, get_settings
from .database import EXPECTED_SCHEMA_REVISION
from .execution_policy import (
    DIRECT_TOOL_CONTRACT_PROTOCOL,
    GOVERNED_EPICURE_PROTOCOLS,
    MATCHED_EVIDENCE_PROTOCOL_V2,
    MATCHED_EVIDENCE_PROTOCOLS,
    MATCHED_TOOL_ACCESS_PROTOCOL_V1,
    PORTABLE_TEXT_TOOL_PROTOCOL_V1,
)
from .provider import response_schema_sha256, system_prompt_sha256
from .validators import VALIDATOR_VERSION


def _project_root() -> Path:
    candidates = (Path.cwd().resolve(), Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / "requirements.lock").is_file() and (candidate / "alembic").is_dir():
            return candidate
    return candidates[-1]


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_sha256(filename: str) -> str:
    return hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()


def _project_sha256(relative_path: str) -> str:
    return hashlib.sha256((_project_root() / relative_path).read_bytes()).hexdigest()


def _local_module_imports(filename: str) -> set[str]:
    package_root = Path(__file__).parent
    tree = ast.parse((package_root / filename).read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                candidates.append(node.module.split(".", 1)[0])
            elif node.level:
                candidates.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif node.module and node.module.startswith("flavourbench."):
                candidates.append(node.module.split(".", 1)[1].split(".", 1)[0])
        elif isinstance(node, ast.Import):
            candidates.extend(
                alias.name.split(".", 1)[1].split(".", 1)[0]
                for alias in node.names
                if alias.name.startswith("flavourbench.")
            )
        for candidate in candidates:
            target = f"{candidate}.py"
            if (package_root / target).is_file():
                imports.add(target)
    return imports


def _implementation_source_names() -> tuple[str, ...]:
    pending = [
        "account_authority.py",
        "arena.py",
        "bedrock_auth.py",
        "bedrock_contract.py",
        "bedrock_manifest.py",
        "bedrock_provider.py",
        "budget_policy.py",
        "config.py",
        "controlled_integrity.py",
        "current_development_manifest.py",
        "database.py",
        "endpoint_contract.py",
        "engine.py",
        "execution_policy.py",
        "live_smoke.py",
        "main.py",
        "matched_protocol_preflight.py",
        "mcp_client.py",
        "models.py",
        "provider.py",
        "protocol_contract.py",
        "ranking.py",
        "real_dataset_runner.py",
        "schemas.py",
        "service_bedrock.py",
        "service_ranking.py",
        "validators.py",
        "worker.py",
    ]
    closure: set[str] = set()
    while pending:
        filename = pending.pop()
        if filename in closure:
            continue
        closure.add(filename)
        pending.extend(sorted(_local_module_imports(filename) - closure))
    return tuple(sorted(closure))


def _alembic_revision_hashes() -> dict[str, str]:
    revisions = sorted((_project_root() / "alembic" / "versions").glob("[0-9]*.py"))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in revisions}


def _alembic_head_evidence() -> tuple[str, str]:
    filename = f"{EXPECTED_SCHEMA_REVISION}.py"
    revision_hashes = _alembic_revision_hashes()
    try:
        return filename, revision_hashes[filename]
    except KeyError as exc:
        raise RuntimeError(f"expected Alembic head source is missing: {filename}") from exc


def build_protocol_bundle(
    *,
    tool_registry_sha256: str,
    epicure_release_id: str,
    epicure_bundle_sha256: str,
    epicure_application_sha256: str,
    analysis_plan_sha256: str,
    model_smoke_registry_sha256: str = "unfrozen",
    final_response_mode: str = "structured_json",
    evidence_protocol: str = "legacy_v6",
    required_tool_contract_sha256: str = "not_applicable",
    settings: Settings | None = None,
) -> tuple[dict[str, Any], str]:
    runtime = settings or get_settings()
    alembic_head_filename, alembic_head_sha256 = _alembic_head_evidence()
    alembic_revisions_sha256 = _alembic_revision_hashes()
    bundle: dict[str, Any] = {
        "schema_version": (
            "flavourbench-protocol-bundle-v10"
            if evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
            else "flavourbench-protocol-bundle-v9"
            if evidence_protocol == MATCHED_TOOL_ACCESS_PROTOCOL_V1
            else "flavourbench-protocol-bundle-v8"
            if evidence_protocol == MATCHED_EVIDENCE_PROTOCOL_V2
            else "flavourbench-protocol-bundle-v7"
            if evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS
            else "flavourbench-protocol-bundle-v2"
            if final_response_mode == "plain_text"
            else "flavourbench-protocol-bundle-v1"
        ),
        "system_prompt_sha256": {
            "epicure_off": system_prompt_sha256(
                "epicure_off", final_response_mode, evidence_protocol
            ),
            "epicure_on": system_prompt_sha256(
                "epicure_on", final_response_mode, evidence_protocol
            ),
        },
        "final_schema_sha256": response_schema_sha256(final_response_mode),
        "tool_registry_sha256": tool_registry_sha256,
        "epicure": {
            "release_id": epicure_release_id,
            "bundle_sha256": epicure_bundle_sha256,
            "application_sha256": epicure_application_sha256,
        },
        "analysis_plan_sha256": analysis_plan_sha256,
        "model_smoke_registry_sha256": model_smoke_registry_sha256,
        "execution_policy": {
            "max_tool_rounds": runtime.max_tool_rounds,
            "max_tool_calls_per_round": runtime.max_tool_calls_per_round,
            "max_tool_calls_total": runtime.max_tool_calls_total,
            "max_tool_result_bytes": runtime.max_tool_result_bytes,
            "max_cumulative_tool_result_bytes": runtime.max_cumulative_tool_result_bytes,
            "max_provider_attempts": runtime.max_provider_attempts,
            "invalid_tool_argument_repairs": 1,
            "provider_fallbacks": False,
            "provider_parameters_required": True,
            "evidence_protocol": evidence_protocol,
            "required_tool_contract_protocol": (
                DIRECT_TOOL_CONTRACT_PROTOCOL
                if evidence_protocol in GOVERNED_EPICURE_PROTOCOLS
                else "not_applicable"
            ),
            "required_tool_contract_sha256": required_tool_contract_sha256,
            "provider_data_collection": "deny",
        },
        "scheduler_versions": {
            "public": SCHEDULER_VERSION,
            "controlled": CONTROLLED_SCHEDULER_VERSION,
        },
        "validator_version": VALIDATOR_VERSION,
        "implementation_sha256": {
            name: _source_sha256(name) for name in _implementation_source_names()
        },
        "release_inputs": {
            "alembic_head": EXPECTED_SCHEMA_REVISION,
            "alembic_head_sha256": alembic_head_sha256,
            "alembic_revisions_sha256": alembic_revisions_sha256,
            "alembic_chain_sha256": canonical_sha256(alembic_revisions_sha256),
            "pyproject_sha256": _project_sha256("pyproject.toml"),
            "dependency_lock_sha256": _project_sha256("requirements.lock"),
            "dockerfile_sha256": _project_sha256("Dockerfile"),
            "container_image_digest": runtime.build_image_digest,
        },
        "analysis_dependency": "arena-rank==0.1.1",
    }
    if final_response_mode == "plain_text":
        bundle["final_response_mode"] = final_response_mode
    return bundle, canonical_sha256(bundle)
