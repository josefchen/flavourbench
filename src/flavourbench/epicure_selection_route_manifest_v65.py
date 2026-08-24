"""Add the separately authorized GLM-5.3 finite-run route to the 26-model panel."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_route_manifest_v57 import _load, _sha256, _sha256_file
from .epicure_selection_route_manifest_v61 import verify_manifest as verify_manifest_v61
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-glm53-limited-run-addition-v65"
MODEL_ID = "z-ai/glm-5.3"
REQUESTED_MODEL_ID = "glm-5.3"
ROUTE_TAG = "zai-coding-plan-direct"
PERMISSION_SEMANTIC_SHA256 = "25baf3111dca323e4cd0f9edaa92569816f1a8b679350fcc3ec31425b83faf14"
PERMISSION_QUOTE_SHA256 = "efd593290cb43649f501243a9a5365d185367ac958371df859a89b76fa4358eb"
GLM53_DOC_SHA256 = "a26f0dc7e1ebb7406fd0b62e15240f70388055142c42d3e433daf8bf3085850a"


class SelectionRouteManifestV65Error(RuntimeError):
    """The additive GLM-5.3 route could not be frozen."""


def _semantic_valid(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == _sha256(payload))


def _permission_contract(permission: Mapping[str, Any]) -> dict[str, Any]:
    try:
        granted = permission["permission"]
        governed = permission["governed_execution"]
        docs = permission["public_documentation"]
    except (KeyError, TypeError) as error:
        raise SelectionRouteManifestV65Error("GLM-5.3 permission artifact is incomplete") from error
    if (
        not _semantic_valid(permission)
        or permission.get("artifact_sha256") != PERMISSION_SEMANTIC_SHA256
        or granted.get("provider") != "Z.ai"
        or granted.get("grantee") != "Josef Chen"
        or granted.get("model_id") != REQUESTED_MODEL_ID
        or granted.get("scope") != "one_finite_flavourbench_benchmark_run"
        or granted.get("permanent_running_function_authorized") is not False
        or granted.get("user_attested_written_permission") is not True
        or granted.get("permission_quote_sha256") != PERMISSION_QUOTE_SHA256
        or governed.get("total_cells") != 1_408
        or governed.get("standing_service") is not False
        or governed.get("calls_made") != 0
        or docs.get("model_page_observed_sha256") != GLM53_DOC_SHA256
        or docs.get("coding_plan_availability_stated") is not True
        or docs.get("general_api_stated_as_coming_soon") is not True
    ):
        raise SelectionRouteManifestV65Error("GLM-5.3 permission scope differs")
    return {
        "provider": "Z.ai",
        "grantee": "Josef Chen",
        "scope": "one_finite_flavourbench_benchmark_run",
        "permanent_running_function_authorized": False,
        "user_attested_written_permission": True,
        "permission_quote_sha256": PERMISSION_QUOTE_SHA256,
    }


def build(
    *,
    source_path: Path,
    permission_path: Path,
) -> dict[str, Any]:
    source = _load(source_path)
    permission = _load(permission_path)
    if not verify_manifest_v61(source):
        raise SelectionRouteManifestV65Error("v65 requires the exact v61 predecessor")
    permission_contract = _permission_contract(permission)
    if any(row.get("model", {}).get("id") == MODEL_ID for row in source["models"]):
        raise SelectionRouteManifestV65Error("GLM-5.3 is already present")

    endpoint = {
        "model_id": MODEL_ID,
        "provider_name": ROUTE_TAG,
        "tag": ROUTE_TAG,
        "name": "Z.ai Coding Plan | GLM-5.3 | finite benchmark permission",
        "status": 0,
        "quantization": "provider_managed",
        "context_length": 1_000_000,
        "max_completion_tokens": 128_000,
        "supported_parameters": ["max_tokens", "temperature"],
        "pricing": {
            "prompt": "0",
            "completion": "0",
            "request": "0",
            "internal_reasoning": "0",
            "accounting_note": "subscription quota; not a claim of free inference",
        },
    }
    backend_contract: dict[str, Any] = {
        "schema_version": "flavourbench-zai-coding-anthropic-contract-v1",
        "base_url": "https://api.z.ai/api/anthropic",
        "requested_model_id": REQUESTED_MODEL_ID,
        "expected_actual_provider_slug": ROUTE_TAG,
        "catalog_sha256": "60b34406d17fe6804b7c719e96886a8ad30082ada8b2187f85dcbc739cc7c25b",
        "catalog_entry_sha256": GLM53_DOC_SHA256,
        "identity_kind": "official_named_release",
        "model_release": "GLM-5.3",
        "data_policy": "deny",
        "allow_fallbacks": False,
        "rank_eligible_after_complete_block": True,
        "cost_reconciliation": "subscription_quota_no_per_generation_charge",
        "limited_run_permission": permission_contract,
    }
    row = {
        "slot": {
            "slot_id": "frontier-refresh-27",
            "cohort": "zai-glm53",
            "model_id": MODEL_ID,
            "open_weight_candidate": True,
            "rationale": "Latest named Z.ai release under written one-run benchmark permission.",
        },
        "model": {
            "id": MODEL_ID,
            "canonical_slug": REQUESTED_MODEL_ID,
            "name": "Z.ai: GLM-5.3",
            "description": "Z.ai GLM-5.3 official named release.",
            "created": 1786752000,
            "context_length": 1_000_000,
            "architecture": {
                "modality": "text->text",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "tokenizer": "Other",
                "instruct_type": None,
            },
            "supported_parameters": ["max_tokens", "temperature"],
            "pricing": {"prompt": "0", "completion": "0"},
            "top_provider": {
                "context_length": 1_000_000,
                "max_completion_tokens": 128_000,
                "is_moderated": False,
            },
        },
        "endpoint": endpoint,
        "endpoint_document_sha256": _sha256(endpoint),
        "endpoint_execution_sha256": endpoint_execution_contract_sha256(endpoint),
        "endpoint_selection": {
            "method": "provider-written finite-run permission plus official named-release docs",
            "selected_exact_tag": ROUTE_TAG,
            "eligible_endpoint_count": 1,
            "automatic_fallback": False,
            "quality_scores_or_selections_used": False,
        },
        "request_policy": {
            "official_eligibility": "eligible_after_two_complete_scored_blocks",
            "policy_scope": "request_enforced",
            "provider": {
                "only": [ROUTE_TAG],
                "allow_fallbacks": False,
                "data_collection": "deny",
                "require_parameters": True,
            },
        },
        "execution_route": {
            "policy": "exact_zai_coding_plan_limited_run_v1",
            "preferred_backend": "zai_coding_direct",
            "selected_backend": "zai_coding_direct",
            "fallback_used": False,
            "generation_time_automatic_fallback": False,
            "selection_frozen_before_generation": True,
            "selection_reason": "written provider permission for one finite benchmark run",
            "evidence": {
                "compatibility_artifact_sha256": PERMISSION_SEMANTIC_SHA256,
                "catalog_sha256": backend_contract["catalog_sha256"],
            },
        },
        "backend_contract": backend_contract,
        "backend_contract_sha256": _sha256(backend_contract),
        "cost_accounting_policy": "provider_subscription_quota_no_marginal_price",
        "contract_evidence": {
            "status": "permission_and_identity_frozen_before_any_generation",
            "permission_artifact_semantic_sha256": PERMISSION_SEMANTIC_SHA256,
            "permission_artifact_physical_sha256": _sha256_file(permission_path),
            "generation_calls": 0,
            "quality_observations": 0,
            "response_identity_required": REQUESTED_MODEL_ID,
            "standing_service_authorized": False,
        },
        "forecast": {
            "panels": 2,
            "primary_tasks_per_panel": 640,
            "repeat_tasks_per_panel": 64,
            "new_provider_calls": 1_408,
            "prompt_token_bound": 4_096,
            "route_max_output_tokens": 16_384,
            "billing_basis": "subscription_quota",
            "marginal_price_claimed": False,
        },
        "open_weight_evidence": {
            "status": "provider_documented_open_source_model_family",
            "quality_claim": False,
        },
    }

    document = copy.deepcopy(source)
    document.pop("content_address", None)
    document["models"].append(row)
    document.update(
        {
            "manifest_role": "frontier_refresh_27_glm53_limited_run_v65",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "glm53_limited_run_addition_v65": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest": {
                    "semantic_sha256": source["content_address"]["digest"],
                    "physical_sha256": _sha256_file(source_path),
                },
                "permission_artifact": {
                    "semantic_sha256": permission["artifact_sha256"],
                    "physical_sha256": _sha256_file(permission_path),
                },
                "added_model_ids": [MODEL_ID],
                "all_26_predecessor_entries_byte_preserved": document["models"][:-1]
                == source["models"],
                "finite_cli_only": True,
                "standing_service": False,
                "automatic_fallback": False,
                "quality_scores_or_selections_used": False,
                "complete_two_panel_blocks_required": True,
            },
        }
    )
    digest = _sha256(document)
    document["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest(document):
        raise SelectionRouteManifestV65Error("constructed v65 manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        addition = document["glm53_limited_run_addition_v65"]
        rows = document["models"]
        row = next(value for value in rows if value["model"]["id"] == MODEL_ID)
        contract = row["backend_contract"]
        permission = contract["limited_run_permission"]
    except (KeyError, StopIteration, TypeError):
        return False
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("status") == "unranked_candidate"
        and verify_manifest_content_address(document)
        and len(rows) == 27
        and len({value["model"]["id"] for value in rows}) == 27
        and row["slot"].get("slot_id") == "frontier-refresh-27"
        and row["endpoint"].get("tag") == ROUTE_TAG
        and row["endpoint"].get("provider_name") == ROUTE_TAG
        and row["model"].get("canonical_slug") == REQUESTED_MODEL_ID
        and row["execution_route"].get("selected_backend") == "zai_coding_direct"
        and row.get("backend_contract_sha256") == _sha256(contract)
        and contract.get("schema_version") == "flavourbench-zai-coding-anthropic-contract-v1"
        and contract.get("requested_model_id") == REQUESTED_MODEL_ID
        and contract.get("identity_kind") == "official_named_release"
        and contract.get("rank_eligible_after_complete_block") is True
        and permission.get("scope") == "one_finite_flavourbench_benchmark_run"
        and permission.get("permanent_running_function_authorized") is False
        and row.get("cost_accounting_policy") == "provider_subscription_quota_no_marginal_price"
        and addition.get("added_model_ids") == [MODEL_ID]
        and addition.get("all_26_predecessor_entries_byte_preserved") is True
        and addition.get("finite_cli_only") is True
        and addition.get("standing_service") is False
        and addition.get("automatic_fallback") is False
        and addition.get("quality_scores_or_selections_used") is False
        and addition.get("complete_two_panel_blocks_required") is True
        and document.get("generation_calls_made") == 0
        and document.get("official_results_authorised") is False
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-frontier-refresh-27-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV65Error("content-addressed manifest conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--permission", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        _write(
            build(source_path=args.source, permission_path=args.permission),
            args.output_directory,
        )
    )


if __name__ == "__main__":
    run()
