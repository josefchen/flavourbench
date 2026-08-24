"""Freeze exact current OpenRouter routes for a real development-quality run."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .execution_policy import (
    GOVERNED_EPICURE_PROTOCOLS,
    MATCHED_EVIDENCE_PROTOCOLS,
    MATCHED_TOOL_ACCESS_PROTOCOL_V1,
    ExecutionPolicy,
)
from .frontier_contract_runner import ContractCandidate
from .frontier_manifest import verify_manifest_content_address
from .real_dataset_runner import (
    WorkItem,
    derive_pair_forecast,
    load_development_task_inventory,
    select_balanced_tasks,
    task_registry_sha256,
)
from .real_task_bank import sha256_json
from .tool_contract import required_tool_contract

SCHEMA_VERSION = "flavourbench-openrouter-candidate-manifest-v1"
MANIFEST_ROLE = "current_frontier_real_development_quality_run"
GENERATION_PROTOCOL_VERSION = "flavourbench-live-development-protocol-v9"
SELECTION_SEED = "flavourbench-current-frontier-quality-v2-matched-evidence"
DEFAULT_CAP_USD = Decimal("100")


class CurrentDevelopmentManifestError(RuntimeError):
    """Current route evidence could not produce an exact run manifest."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CurrentDevelopmentManifestError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CurrentDevelopmentManifestError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise CurrentDevelopmentManifestError(f"expected an object: {path}")
    return value


def _verified(path: Path, expected: str | None = None) -> dict[str, Any]:
    value = _load(path)
    recorded = str(value.get("artifact_sha256") or "")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    actual = sha256_json(payload)
    if recorded != actual or (expected is not None and recorded != expected):
        raise CurrentDevelopmentManifestError(f"content address does not verify: {path}")
    return value


def _catalogs(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    if root.is_symlink() or not root.is_dir():
        raise CurrentDevelopmentManifestError("route catalog root must be a directory")
    indexed: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(root.rglob("frontier-refresh-route-catalog-*.json")):
        value = _verified(path)
        digest = str(value["artifact_sha256"])
        if digest in indexed:
            previous = indexed[digest][1]
            if previous.read_bytes() != path.read_bytes():
                raise CurrentDevelopmentManifestError("conflicting route catalog digests")
            continue
        indexed[digest] = (value, path)
    if not indexed:
        raise CurrentDevelopmentManifestError("no verified route catalogs were found")
    return indexed


def _source_path(repository_root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else repository_root / path


def _candidate_from_route(
    *,
    index: int,
    record: Mapping[str, Any],
    route: Mapping[str, Any],
) -> tuple[dict[str, Any], ContractCandidate]:
    model = route.get("model")
    endpoint = route.get("endpoint")
    if not isinstance(model, Mapping) or not isinstance(endpoint, Mapping):
        raise CurrentDevelopmentManifestError("route lacks a model or endpoint contract")
    model_id = str(record.get("requested_model_id") or "")
    canonical = str(record.get("canonical_model_slug") or "")
    provider_tag = str(record.get("provider_endpoint") or "")
    if (
        route.get("model_id") != model_id
        or route.get("canonical_slug") != canonical
        or route.get("provider_slug") != provider_tag
        or model.get("id") != model_id
        or model.get("canonical_slug") != canonical
        or endpoint.get("model_id") != model_id
        or endpoint.get("tag") != provider_tag
    ):
        raise CurrentDevelopmentManifestError(f"route identity mismatch for {model_id}")
    supported = set(str(item) for item in endpoint.get("supported_parameters") or [])
    required = {
        "max_tokens",
        "reasoning",
        "tools",
        "tool_choice",
        "response_format",
        "structured_outputs",
    }
    if not required.issubset(supported):
        raise CurrentDevelopmentManifestError(
            f"{model_id}@{provider_tag} lacks required parameters"
        )
    slot_id = f"current-frontier-{index:02d}"
    entry = {
        "slot": {
            "slot_id": slot_id,
            "cohort": "current_frontier_development",
            "model_id": model_id,
            "open_weight_candidate": None,
            "rationale": (
                "Current exact route passed structured-final and real Epicure contract checks; "
                "inclusion is not a quality claim."
            ),
        },
        "model": dict(model),
        "endpoint": dict(endpoint),
        "endpoint_document_sha256": route.get("endpoint_document_sha256"),
        "endpoint_selection": {
            "method": "exact previously contract-passed provider route",
            "selected_exact_tag": provider_tag,
            "quality_observations_used": 0,
        },
        "request_policy": {
            "policy_scope": "request_enforced",
            "provider": {
                "only": [provider_tag],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            },
            "official_eligibility": "development_only",
        },
        "contract_evidence": {
            "source_artifact_sha256": record.get("source_artifact_sha256"),
            "real_provider_calls": record.get("provider_calls"),
            "real_epicure_calls": record.get("epicure_calls"),
            "actual_provider": record.get("actual_provider"),
            "identity_basis": record.get("identity_basis"),
            "contract_status": record.get("contract_status"),
        },
    }
    candidate = ContractCandidate(
        slot_id=slot_id,
        model_id=model_id,
        canonical_model_slug=canonical,
        model_name=str(record.get("display_name") or model_id),
        provider_tag=provider_tag,
        provider_name=str(endpoint.get("provider_name") or ""),
        endpoint_sha256=sha256_json(endpoint),
        endpoint_execution_sha256="derived_at_runner_load",
        endpoint=dict(endpoint),
    )
    return entry, candidate


def build_manifest(
    *,
    registry_path: Path,
    route_catalog_root: Path,
    task_validity_path: Path,
    repository_root: Path,
    tasks_per_family: int,
    assignments_per_model: int,
    cap_usd: Decimal,
    execution_policy: ExecutionPolicy,
    selection_seed: str = SELECTION_SEED,
) -> dict[str, Any]:
    if cap_usd <= 0 or cap_usd > DEFAULT_CAP_USD:
        raise CurrentDevelopmentManifestError("cap must be in (0, 100]")
    execution_policy.validate()
    registry = _verified(registry_path)
    if (
        registry.get("schema_version") != "flavourbench-current-route-registry-v1"
        or registry.get("official") is not False
        or registry.get("rank_eligible") is not False
    ):
        raise CurrentDevelopmentManifestError("unexpected current-route registry")
    records = registry.get("models")
    if not isinstance(records, list):
        raise CurrentDevelopmentManifestError("current-route registry has no models")
    catalog_index = _catalogs(route_catalog_root)

    entries: list[dict[str, Any]] = []
    candidates: list[ContractCandidate] = []
    exclusions: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise CurrentDevelopmentManifestError("registry contains a non-object model")
        model_id = str(record.get("requested_model_id") or "")
        endpoint = str(record.get("provider_endpoint") or "")
        if record.get("contract_status") != "passed_unranked":
            exclusions.append(
                {
                    "model_id": model_id,
                    "reason": "exact_route_contract_failed",
                    "contract_status": record.get("contract_status"),
                }
            )
            continue
        if endpoint == "cohere-direct":
            exclusions.append(
                {
                    "model_id": model_id,
                    "reason": "direct_provider_lane_requires_separate_cost_governor",
                    "contract_status": record.get("contract_status"),
                }
            )
            continue
        if (
            int(record.get("provider_calls") or 0) != 2
            or int(record.get("epicure_calls") or 0) != 1
        ):
            raise CurrentDevelopmentManifestError(f"incomplete contract counts for {model_id}")
        source = _verified(
            _source_path(repository_root, record.get("source_artifact_path")),
            str(record.get("source_artifact_sha256") or ""),
        )
        source_manifest_sha = str(source.get("source_manifest_sha256") or "")
        catalog_pair = catalog_index.get(source_manifest_sha)
        if catalog_pair is None:
            raise CurrentDevelopmentManifestError(
                f"no exact route catalog for {model_id}: {source_manifest_sha}"
            )
        catalog, _catalog_path = catalog_pair
        routes = catalog.get("routes")
        if not isinstance(routes, list):
            raise CurrentDevelopmentManifestError("route catalog has no route list")
        matches = [
            route
            for route in routes
            if isinstance(route, Mapping)
            and route.get("model_id") == model_id
            and route.get("provider_slug") == endpoint
        ]
        if len(matches) != 1:
            raise CurrentDevelopmentManifestError(
                f"expected exactly one exact route for {model_id}@{endpoint}"
            )
        route = matches[0]
        if (
            source.get("canonical_slug") != record.get("canonical_model_slug")
            or source.get("requested_provider_slug") != endpoint
            or source.get("endpoint_document_sha256") != route.get("endpoint_document_sha256")
        ):
            raise CurrentDevelopmentManifestError(
                f"contract receipt and route catalog disagree for {model_id}"
            )
        entry, candidate = _candidate_from_route(
            index=len(entries) + 1,
            record=record,
            route=route,
        )
        entries.append(entry)
        candidates.append(candidate)

    if len(entries) != 14:
        raise CurrentDevelopmentManifestError(
            f"expected 14 contract-passed OpenRouter routes, found {len(entries)}"
        )
    task_inventory, task_source = load_development_task_inventory(task_validity_path)
    selected_tasks, registry_sha = select_balanced_tasks(
        tasks_per_family=tasks_per_family,
        seed=selection_seed,
        tasks=task_inventory,
    )
    if not (
        len(selected_tasks) >= assignments_per_model >= 4
        and assignments_per_model <= tasks_per_family * 4
    ):
        raise CurrentDevelopmentManifestError("invalid assignments-per-model target")

    pair_forecast = Decimal(0)
    per_model_forecast: dict[str, Decimal] = {}
    for candidate in candidates:
        model_total = Decimal(0)
        for task in selected_tasks[:assignments_per_model]:
            placeholder = WorkItem(
                ordinal=0,
                work_item_id="0" * 64,
                manifest_sha256="0" * 64,
                task_registry_sha256=task_registry_sha256(task_inventory),
                task=task,
                candidate=candidate,
                endpoint_execution_sha256="0" * 64,
                execution_policy_sha256=execution_policy.sha256,
                execution_policy=execution_policy,
            )
            model_total += derive_pair_forecast(
                placeholder,
                policy=execution_policy,
            ).forecast_usd
        per_model_forecast[candidate.model_id] = model_total
        pair_forecast += model_total
    admission_ceiling = cap_usd * Decimal("0.85")
    if pair_forecast > admission_ceiling:
        raise CurrentDevelopmentManifestError(
            f"complete block forecast ${pair_forecast} exceeds 85% admission ceiling "
            f"${admission_ceiling}"
        )
    for entry in entries:
        model_id = str(entry["model"]["id"])
        entry["forecast"] = {
            "model_block_worst_case_usd": format(per_model_forecast[model_id], "f"),
            "pairs": assignments_per_model,
            "conditions_per_pair": 2,
        }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "unranked_candidate",
        "manifest_role": MANIFEST_ROLE,
        "official_results_authorised": False,
        "generation_calls_made": 0,
        "generation_spend_usd": "0",
        "observed_at": f"{registry['snapshot_date']}T00:00:00Z",
        "source": {
            "current_route_registry_sha256": registry["artifact_sha256"],
            "task_validity_artifact_sha256": task_source["artifact_sha256"],
            "task_candidate_coordinate_sha256": task_source["candidate_coordinate_sha256"],
            "task_registry_sha256": registry_sha,
        },
        "selection": {
            "method": "all current contract-passed OpenRouter routes before quality observation",
            "performance_claim": "none; inclusion is coverage, not a ranking",
            "model_count": len(entries),
            "excluded_lanes": exclusions,
            "quality_observations_used": 0,
        },
        "run_design": {
            "tasks_per_family_in_pool": tasks_per_family,
            "selection_seed": selection_seed,
            "selected_task_count": len(selected_tasks),
            "assignments_per_model": assignments_per_model,
            "expected_pairs": len(entries) * assignments_per_model,
            "expected_arms": len(entries) * assignments_per_model * 2,
            "conditions": ["epicure_off", "epicure_on"],
            "task_source": task_source,
            "execution_policy": execution_policy.document(),
            "execution_policy_sha256": execution_policy.sha256,
            "generation_protocol": {
                "schema_version": GENERATION_PROTOCOL_VERSION,
                "builder": "flavourbench.live_smoke.build_live_protocol_bundle",
                "full_epicure_catalog_required": True,
                "final_response_mode": execution_policy.final_response_mode,
                "matched_planning": execution_policy.matched_planning,
                "max_intermediate_tokens": execution_policy.max_intermediate_tokens,
                "required_tool_contract_max_intermediate_tokens": (
                    execution_policy.required_tool_contract_max_intermediate_tokens
                ),
                "evidence_protocol": execution_policy.evidence_protocol,
                "required_tool_contract_protocol": (
                    execution_policy.required_tool_contract_protocol
                ),
                "required_tool_contract": required_tool_contract(execution_policy),
                "required_tool_contract_sha256": required_tool_contract(execution_policy)[
                    "content_address"
                ]["digest"],
                "intermediate_reasoning_effort": (execution_policy.intermediate_reasoning_effort),
                "final_reasoning_effort": execution_policy.final_reasoning_effort,
                "tool_catalog_bytes_bound": execution_policy.tool_catalog_bytes_bound,
                "bindings": [
                    "candidate_manifest_sha256",
                    "dataset_work_item_id",
                    "dataset_task_id",
                    "prompt_sha256",
                    "task_family",
                    "canonical_model_slug",
                    "exact_provider_endpoint",
                    "endpoint_contract_sha256",
                    "execution_policy_sha256",
                    "response_schema_sha256",
                    "final_response_mode",
                    "matched_planning",
                    "max_intermediate_tokens",
                    "required_tool_contract_max_intermediate_tokens",
                    "evidence_protocol",
                    "required_tool_contract_protocol",
                    "required_tool_contract_sha256",
                    "intermediate_reasoning_effort",
                    "final_reasoning_effort",
                    "epicure_release_id",
                    "epicure_bundle_sha256",
                    "epicure_application_sha256",
                    "epicure_tool_schema_sha256",
                ],
            },
        },
        "budget": {
            "currency": "USD",
            "cap_usd": format(cap_usd, "f"),
            "admission_fraction": "0.85",
            "admission_ceiling_usd": format(admission_ceiling, "f"),
            "bounded_forecast_usd": format(pair_forecast, "f"),
            "headroom_to_admission_ceiling_usd": format(
                admission_ceiling - pair_forecast,
                "f",
            ),
            "within_cap": True,
            "forecast_policy": "real_dataset_runner_pair_reservation_v1",
        },
        "models": entries,
        "governance": {
            "manifest_class": "real_development_quality_run_candidate",
            "freeze_status": "exact_routes_and_workload_frozen_before_generation",
            "data_policy": "fixed endpoint, no fallbacks, require parameters, deny collection",
            "official": False,
            "rank_eligible": False,
            "required_before_quality_claims": [
                "complete real responses and cost reconciliation",
                "blinded judgments independent of model identity",
                "predeclared task-specific criteria",
                "uncertainty and sample-size reporting",
            ],
        },
    }
    digest = sha256_json(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest_content_address(payload):
        raise CurrentDevelopmentManifestError("internal manifest content address failed")
    return payload


def _write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    if not verify_manifest_content_address(payload):
        raise CurrentDevelopmentManifestError("refusing an invalid manifest")
    digest = str(payload["content_address"]["digest"])
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"flavourbench-openrouter-unranked-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise CurrentDevelopmentManifestError("content-addressed output conflict")
        return path
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_dir,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--route-catalog-root", type=Path, required=True)
    parser.add_argument("--task-validity", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks-per-family", type=int, default=1)
    parser.add_argument("--assignments-per-model", type=int, default=4)
    parser.add_argument("--selection-seed", default=SELECTION_SEED)
    parser.add_argument("--cap-usd", type=Decimal, default=DEFAULT_CAP_USD)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-intermediate-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-rounds", type=int, default=2)
    parser.add_argument("--max-cumulative-tool-result-bytes", type=int, default=65536)
    parser.add_argument("--max-tool-calls-per-round", type=int, default=16)
    parser.add_argument("--max-tool-calls-total", type=int, default=16)
    parser.add_argument("--max-provider-attempts", type=int, choices=[1, 2], default=2)
    parser.add_argument("--tool-catalog-bytes-bound", type=int, default=24000)
    parser.add_argument(
        "--final-response-mode",
        choices=["plain_text", "structured_json"],
        default="plain_text",
    )
    parser.add_argument(
        "--evidence-protocol",
        choices=sorted(GOVERNED_EPICURE_PROTOCOLS),
        default=MATCHED_TOOL_ACCESS_PROTOCOL_V1,
    )
    parser.add_argument(
        "--intermediate-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default="low",
    )
    parser.add_argument(
        "--final-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default="low",
    )
    arguments = parser.parse_args(argv)
    policy = ExecutionPolicy(
        max_output_tokens=arguments.max_output_tokens,
        max_intermediate_tokens=arguments.max_intermediate_tokens,
        max_tool_rounds=arguments.max_tool_rounds,
        max_cumulative_tool_result_bytes=(arguments.max_cumulative_tool_result_bytes),
        max_tool_calls_per_round=arguments.max_tool_calls_per_round,
        max_tool_calls_total=arguments.max_tool_calls_total,
        max_provider_attempts=arguments.max_provider_attempts,
        final_response_mode=arguments.final_response_mode,
        matched_planning=(
            arguments.final_response_mode == "plain_text"
            and arguments.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS
        ),
        evidence_protocol=arguments.evidence_protocol,
        intermediate_reasoning_effort=arguments.intermediate_reasoning_effort,
        final_reasoning_effort=arguments.final_reasoning_effort,
        tool_catalog_bytes_bound=arguments.tool_catalog_bytes_bound,
    )
    payload = build_manifest(
        registry_path=arguments.registry,
        route_catalog_root=arguments.route_catalog_root,
        task_validity_path=arguments.task_validity,
        repository_root=arguments.repository_root,
        tasks_per_family=arguments.tasks_per_family,
        assignments_per_model=arguments.assignments_per_model,
        cap_usd=arguments.cap_usd,
        execution_policy=policy,
        selection_seed=arguments.selection_seed,
    )
    path = _write(arguments.output_dir, payload)
    print(
        json.dumps(
            {
                "output": str(path),
                "manifest_sha256": payload["content_address"]["digest"],
                "models": payload["selection"]["model_count"],
                "run_design": payload["run_design"],
                "budget": payload["budget"],
                "provider_calls_made": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
