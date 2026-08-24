"""Build a blocked, offline 16-model Season-1 design alternative.

This module does not amend the active 14-model candidate.  It verifies exact local,
content-addressed evidence; constructs a fresh 16-model schedule; and freezes an
outcome-blind abstract human-sampling frame.  It has no provider, MCP, database,
reviewer, deployment, or publication integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frontier_manifest import verify_manifest_content_address
from .real_task_bank import sha256_json
from .task_campaign_study_design_successor import (
    verify_successor_design as verify_current_14_design,
)

SCHEMA_VERSION = "flavourbench-season1-16-model-study-design-alternative-v1-candidate"
STATUS = "blocked_offline_16_model_alternative_not_authorized"
SAMPLING_RECIPE_VERSION = "flavourbench-16-model-outcome-blind-sampling-v1"

FLAVOURBENCH_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FLAVOURBENCH_ROOT.parent

DEFAULT_CURRENT_14_ROSTER = (
    REPOSITORY_ROOT / "paper/flavourbench/provenance/routed-v30-manifest.json"
)
DEFAULT_QWEN_PROJECTION = REPOSITORY_ROOT / (
    "paper/flavourbench/provenance/qwencloud-exploratory-operational-projection.json"
)
DEFAULT_COHERE_MANIFEST = FLAVOURBENCH_ROOT / (
    "artifacts/season1/current-quality-run/"
    "manifest-v44-floor-replenishment-cohere-direct/"
    "flavourbench-cohere-unranked-"
    "a93791bb929bfc45d483ff031016760c6f042e6a7539fa9ef6f23f94b47ebabf.json"
)
DEFAULT_COHERE_CATALOG = FLAVOURBENCH_ROOT / (
    "artifacts/frontier-refresh/2026-08-03/cohere-direct/"
    "catalyst-key-v3-command-a-plus/catalog/"
    "cohere-catalog-"
    "b6d58d902b369130ca7c97b5a23f18d5a2c3ccc630b11204ac1d8218d04ff862.json"
)
DEFAULT_COHERE_SMOKE = FLAVOURBENCH_ROOT / (
    "artifacts/frontier-refresh/2026-08-03/cohere-direct/"
    "catalyst-key-v3-command-a-plus/compatibility/"
    "cohere-9d0a2c944810-"
    "9208ae6aa7897c23e5e55efe35e80da52e6415f72a2a26ab168f3fb957f838e5.json"
)
DEFAULT_CURRENT_14_DESIGN = FLAVOURBENCH_ROOT / (
    "artifacts/season1/study-design-v6-candidate/"
    "study-design-v6-candidate-"
    "e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e.json"
)
DEFAULT_CURRENT_14_SAMPLING = FLAVOURBENCH_ROOT / (
    "artifacts/season1/human-judgment-sampling-v1-candidate/"
    "human-judgment-sampling-v1-candidate-"
    "5a0b1bbeb20564c9e8fde78b958bbed723ee0cc3395c809267c3775adeed95f8.json"
)
DEFAULT_INFERENCE_ACCEPTANCE = FLAVOURBENCH_ROOT / (
    "contracts/season1/season1-arena-inference-acceptance-v1.json"
)
DEFAULT_OUTPUT_DIR = FLAVOURBENCH_ROOT / (
    "artifacts/season1/study-design-16-model-alternative-v1-candidate"
)

FAMILIES = ("substitution", "composition", "cookability", "evidence")
CONDITIONS = ("epicure_off", "epicure_on")
RATER_SLOTS = (1, 2)

CURRENT_14_MODELS: tuple[tuple[str, str], ...] = (
    ("moonshotai/kimi-k3", "k3"),
    ("openai/gpt-5.6-sol-pro", "openai/gpt-5.6-sol-pro-20260709"),
    ("anthropic/claude-fable-5", "anthropic/claude-5-fable-20260609"),
    ("anthropic/claude-opus-5", "anthropic/claude-opus-5-20260723"),
    ("anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5-20260630"),
    ("google/gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview-20260219"),
    ("google/gemini-3.6-flash", "google/gemini-3.6-flash-20260721"),
    ("x-ai/grok-4.5", "x-ai/grok-4.5-20260708"),
    ("z-ai/glm-5.2", "z-ai/glm-5.2-20260616"),
    ("deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro-20260423"),
    ("deepseek/deepseek-v4-flash-0731", "deepseek/deepseek-v4-flash-20260731"),
    ("minimax/minimax-m3", "minimax/minimax-m3-20260531"),
    (
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-ultra-550b-a55b-20260604",
    ),
    ("mistralai/mistral-medium-3-5", "mistralai/mistral-medium-3.5-20260430"),
)
QWEN_MODEL = ("qwen3.8-max", "qwen3.8-max")
COHERE_MODEL = ("cohere/command-a-plus-05-2026", "command-a-plus-05-2026")
EXPECTED_16_MODELS = (*CURRENT_14_MODELS, QWEN_MODEL, COHERE_MODEL)

# Pair coordinates receiving seven rather than six human arena selections.  The
# coordinates refer to the canonical K16 one-factorization.  The set has 80 edges,
# degree ten at every model, and the required 30/30/20 edge allocation across the
# three five-matching generation blocks (27/27/26 task uses).
ARENA_HUMAN_HIGH_TARGET_COORDINATES = frozenset(
    {
        *((matching, pair) for matching in (0, 1, 2, 5, 6, 7, 10, 11) for pair in range(8)),
        *((3, pair) for pair in range(6)),
        *((8, pair) for pair in range(6)),
        *((13, pair) for pair in range(1, 5)),
    }
)


class AlternativeDesignError(RuntimeError):
    """A bound source, design invariant, or authority boundary failed closed."""


@dataclass(frozen=True)
class SourceSpec:
    role: str
    reference_path: str
    schema_version: str
    semantic_sha256: str
    physical_sha256: str
    address_kind: str


SOURCE_SPECS = {
    "current_14_roster": SourceSpec(
        role="authoritative_current_14_model_routed_candidate",
        reference_path="paper/flavourbench/provenance/routed-v30-manifest.json",
        schema_version="flavourbench-routed-candidate-manifest-v1",
        semantic_sha256="e87a164d59bdd88eaf630c153755b5ec3c513e8b3770a17afac67037eb135910",
        physical_sha256="ac745d0ef19fbc9e17d38d4bd254e7688eee06a3280dbf9a3bbe032361b9d803",
        address_kind="content_address",
    ),
    "qwen_projection": SourceSpec(
        role="qwen_3_8_max_time_bounded_exploratory_projection",
        reference_path=(
            "paper/flavourbench/provenance/"
            "qwencloud-exploratory-operational-projection.json"
        ),
        schema_version="flavourbench-qwencloud-exploratory-operational-projection-v1",
        semantic_sha256="b2f7790b3eb18d1df083397ce02b5296c549e5ed3ddb3d3f32ea776db3ddca04",
        physical_sha256="9343a2959d3acf3079fb91b2bd7ff608af421532b826f7b98917c88b76a7f85c",
        address_kind="artifact_sha256",
    ),
    "cohere_manifest": SourceSpec(
        role="latest_audited_direct_cohere_route_manifest_v44",
        reference_path=(
            "flavourbench/artifacts/season1/current-quality-run/"
            "manifest-v44-floor-replenishment-cohere-direct/"
            "flavourbench-cohere-unranked-"
            "a93791bb929bfc45d483ff031016760c6f042e6a7539fa9ef6f23f94b47ebabf.json"
        ),
        schema_version="flavourbench-routed-candidate-manifest-v1",
        semantic_sha256="a93791bb929bfc45d483ff031016760c6f042e6a7539fa9ef6f23f94b47ebabf",
        physical_sha256="0e030fa5f3f8441e0195697286d3e5ba46c10c3ff977e7763a88dc82667104bb",
        address_kind="content_address",
    ),
    "cohere_catalog": SourceSpec(
        role="authenticated_direct_cohere_catalog_bound_by_v44",
        reference_path=(
            "flavourbench/artifacts/frontier-refresh/2026-08-03/cohere-direct/"
            "catalyst-key-v3-command-a-plus/catalog/"
            "cohere-catalog-"
            "b6d58d902b369130ca7c97b5a23f18d5a2c3ccc630b11204ac1d8218d04ff862.json"
        ),
        schema_version="flavourbench-cohere-catalog-v1",
        semantic_sha256="b6d58d902b369130ca7c97b5a23f18d5a2c3ccc630b11204ac1d8218d04ff862",
        physical_sha256="da6a1178fc1168a2e4e4b4fb7dc57320862a652d327e0b4e4cb869dc0e5f2e0a",
        address_kind="artifact_sha256",
    ),
    "cohere_smoke": SourceSpec(
        role="direct_cohere_command_a_plus_contract_smoke_bound_by_v44",
        reference_path=(
            "flavourbench/artifacts/frontier-refresh/2026-08-03/cohere-direct/"
            "catalyst-key-v3-command-a-plus/compatibility/"
            "cohere-9d0a2c944810-"
            "9208ae6aa7897c23e5e55efe35e80da52e6415f72a2a26ab168f3fb957f838e5.json"
        ),
        schema_version="flavourbench-cohere-epicure-contract-smoke-v1",
        semantic_sha256="9208ae6aa7897c23e5e55efe35e80da52e6415f72a2a26ab168f3fb957f838e5",
        physical_sha256="7e151a54d9aef74a17d2dfbba5dce057de659e2d6b524d6efda8327af20bc1d8",
        address_kind="artifact_sha256",
    ),
    "current_14_design": SourceSpec(
        role="current_14_model_13300_arm_design_for_comparison_only",
        reference_path=(
            "flavourbench/artifacts/season1/study-design-v6-candidate/"
            "study-design-v6-candidate-"
            "e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e.json"
        ),
        schema_version="flavourbench-season1-study-design-v6-candidate",
        semantic_sha256="e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e",
        physical_sha256="6affdc8f80e59476254834d8edc588c471a5bd7e86145e66448e4fb7b90118af",
        address_kind="artifact_sha256",
    ),
    "current_14_sampling": SourceSpec(
        role="current_14_model_human_sampling_frame_non_applicable_to_16",
        reference_path=(
            "flavourbench/artifacts/season1/human-judgment-sampling-v1-candidate/"
            "human-judgment-sampling-v1-candidate-"
            "5a0b1bbeb20564c9e8fde78b958bbed723ee0cc3395c809267c3775adeed95f8.json"
        ),
        schema_version="flavourbench-season1-human-judgment-sampling-v1-candidate",
        semantic_sha256="5a0b1bbeb20564c9e8fde78b958bbed723ee0cc3395c809267c3775adeed95f8",
        physical_sha256="6c7371daa9506cdf5dcee38c19ee48dd2938f36d259c1ee04b7c849c49977039",
        address_kind="artifact_sha256",
    ),
    "inference_acceptance": SourceSpec(
        role="frozen_official_arena_inference_acceptance_v1_non_applicable_audit",
        reference_path=(
            "flavourbench/contracts/season1/season1-arena-inference-acceptance-v1.json"
        ),
        schema_version="flavourbench-season1-arena-inference-acceptance-v1",
        semantic_sha256="bdc0fa93c6365cdcd45694d1d5500d82ccbd622f3be897be9217e252855ffff5",
        physical_sha256="02adfc4a32e2690c1f8f5ddce6edba3f1974159956027b88003a484f5a0655bc",
        address_kind="artifact_sha256",
    ),
}


@dataclass
class _FlowEdge:
    target: int
    capacity: int
    reverse: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AlternativeDesignError(message)


def _physical_sha256(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"source is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_source(path: Path, spec: SourceSpec) -> dict[str, Any]:
    _require(_physical_sha256(path) == spec.physical_sha256, f"{spec.role} physical digest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlternativeDesignError(f"invalid JSON for {spec.role}") from error
    _require(isinstance(value, dict), f"{spec.role} must be an object")
    _require(value.get("schema_version") == spec.schema_version, f"{spec.role} schema")
    if spec.address_kind == "content_address":
        address = value.get("content_address")
        _require(
            isinstance(address, Mapping)
            and address.get("digest") == spec.semantic_sha256
            and verify_manifest_content_address(value),
            f"{spec.role} semantic digest",
        )
    else:
        body = {key: item for key, item in value.items() if key != "artifact_sha256"}
        _require(
            value.get("artifact_sha256") == spec.semantic_sha256
            and sha256_json(body) == spec.semantic_sha256,
            f"{spec.role} semantic digest",
        )
    return value


def _source_commitment(spec: SourceSpec) -> dict[str, str]:
    return {
        "role": spec.role,
        "reference_path": spec.reference_path,
        "schema_version": spec.schema_version,
        "semantic_sha256": spec.semantic_sha256,
        "physical_sha256": spec.physical_sha256,
        "address_kind": spec.address_kind,
    }


def _verify_current_roster(
    roster: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    selection = roster.get("selection")
    governance = roster.get("governance")
    models = roster.get("models")
    _require(roster.get("status") == "unranked_candidate", "current roster status")
    _require(roster.get("official_results_authorised") is False, "current roster authority")
    _require(roster.get("generation_calls_made") == 0, "current roster call count")
    _require(
        isinstance(selection, Mapping)
        and selection.get("model_count") == 14
        and selection.get("quality_observations_used") == 0
        and selection.get("performance_claim") == "none; inclusion is coverage, not a ranking",
        "current roster selection boundary",
    )
    _require(
        isinstance(governance, Mapping)
        and governance.get("official") is False
        and governance.get("rank_eligible") is False,
        "current roster governance boundary",
    )
    _require(isinstance(models, list), "current roster models")
    observed: list[tuple[str, str]] = []
    panel_rows: list[dict[str, Any]] = []
    kimi_entry: Mapping[str, Any] | None = None
    for entry in models:
        _require(isinstance(entry, Mapping), "current roster model entry")
        model = entry.get("model")
        _require(isinstance(model, Mapping), "current roster model identity")
        model_id = str(model.get("id") or "")
        canonical = str(model.get("canonical_slug") or "")
        observed.append((model_id, canonical))
        identity_kind = (
            "provider_managed_direct_identifier_immutability_unproven"
            if model_id == "moonshotai/kimi-k3"
            else "source_bound_current_14_route_identity"
        )
        panel_rows.append(
            {
                "model_id": model_id,
                "canonical_model_slug": canonical,
                "display_name": str(model.get("name") or model_id),
                "identity_kind": identity_kind,
                "evidence_source_role": SOURCE_SPECS["current_14_roster"].role,
            }
        )
        if model_id == "moonshotai/kimi-k3":
            kimi_entry = entry
    _require(tuple(observed) == CURRENT_14_MODELS, "exact current 14 roster changed")
    _require(kimi_entry is not None, "Kimi K3 is absent from the current 14 roster")

    excluded = selection.get("excluded_lanes")
    _require(isinstance(excluded, list), "current roster exclusions")
    cohere_exclusion = [
        row
        for row in excluded
        if isinstance(row, Mapping) and row.get("model_id") == "command-a-plus-05-2026"
    ]
    _require(
        len(cohere_exclusion) == 1
        and cohere_exclusion[0].get("contract_status") == "passed_unranked"
        and cohere_exclusion[0].get("reason")
        == "direct_provider_lane_requires_separate_cost_governor",
        "current roster Cohere cost-governor exclusion changed",
    )
    return panel_rows, kimi_entry


def _verify_qwen_projection(projection: Mapping[str, Any]) -> dict[str, Any]:
    identity = projection.get("model_identity")
    boundary = projection.get("claim_boundary")
    accounting = projection.get("combined_reliability_accounting")
    run = projection.get("successor_operational_run")
    _require(
        projection.get("status") == "verified_exploratory_unranked_post_freeze_addendum",
        "Qwen projection status",
    )
    _require(
        isinstance(identity, Mapping)
        and identity.get("requested_model_id") == "qwen3.8-max"
        and identity.get("returned_model_ids") == ["qwen3.8-max"]
        and identity.get("provider") == "qwencloud-direct"
        and identity.get("identity_kind") == "mutable_alias"
        and identity.get("frozen_release") is False
        and identity.get("automatic_provider_fallback") is False
        and identity.get("catalog_observed_at") == "2026-08-08T16:29:04Z",
        "Qwen identity boundary",
    )
    _require(
        projection.get("recorded_at") == "2026-08-08T21:24:15.897913Z",
        "Qwen projection recording time",
    )
    _require(
        isinstance(accounting, Mapping)
        and accounting.get("provider_charge_available") is False
        and accounting.get("provider_cost_reconciled") is False
        and accounting.get("recorded_zero_cost_means") == "unknown_not_free",
        "Qwen cost boundary",
    )
    _require(
        isinstance(run, Mapping)
        and run.get("status") == "complete_unpriced_budget_ceiling"
        and run.get("delivered_response_arms") == 2,
        "Qwen operational evidence boundary",
    )
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("post_freeze_operational_addendum_only") is True
        and boundary.get("included_in_any_quality_fit") is False
        and boundary.get("quality_judgments") == 0
        and boundary.get("leaderboard_comparisons_authorized") == 0
        and boundary.get("official") is False
        and boundary.get("rank_eligible") is False
        and boundary.get("season_eligible") is False,
        "Qwen claim boundary",
    )
    return {
        "model_id": QWEN_MODEL[0],
        "canonical_model_slug": QWEN_MODEL[1],
        "display_name": str(identity["display_name"]),
        "identity_kind": "mutable_alias",
        "evidence_source_role": SOURCE_SPECS["qwen_projection"].role,
    }


def _verify_cohere_sources(
    manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
    smoke: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(
        catalog.get("provider") == "cohere_direct"
        and catalog.get("official") is False
        and catalog.get("rank_eligible") is False,
        "Cohere catalog boundary",
    )
    catalog_models = catalog.get("models")
    _require(isinstance(catalog_models, list), "Cohere catalog models")
    matches = [
        row
        for row in catalog_models
        if isinstance(row, Mapping) and row.get("name") == COHERE_MODEL[1]
    ]
    _require(len(matches) == 1, "Cohere catalog Command A+ identity")
    catalog_entry = matches[0]
    catalog_entry_sha256 = sha256_json(catalog_entry)
    _require(
        catalog_entry_sha256
        == "4b7d6826e53e09077d11ccdb0ae97f7f76a2a061b1bd498cbca4770b3d90b6d8",
        "Cohere catalog entry digest",
    )

    _require(
        smoke.get("status") == "smoke_passed"
        and smoke.get("requested_model_id") == COHERE_MODEL[1]
        and smoke.get("catalog_sha256") == SOURCE_SPECS["cohere_catalog"].semantic_sha256
        and smoke.get("catalog_entry_sha256") == catalog_entry_sha256
        and smoke.get("real_provider_calls") == 2
        and smoke.get("real_epicure_calls") == 1
        and smoke.get("cost_status") == "no_per_generation_cost_returned_by_provider"
        and smoke.get("official") is False
        and smoke.get("rank_eligible") is False,
        "Cohere smoke boundary",
    )
    _require(
        manifest.get("status") == "unranked_candidate"
        and manifest.get("manifest_role") == "current_frontier_routed_development_quality_run"
        and manifest.get("official_results_authorised") is False
        and manifest.get("generation_calls_made") == 0,
        "Cohere manifest boundary",
    )
    selection = manifest.get("selection")
    governance = manifest.get("governance")
    source = manifest.get("source")
    _require(
        isinstance(selection, Mapping)
        and selection.get("model_count") == 2
        and selection.get("quality_observations_used") == 0
        and selection.get("performance_claim") == "none; inclusion is coverage, not a ranking",
        "Cohere manifest selection",
    )
    _require(
        isinstance(governance, Mapping)
        and governance.get("official") is False
        and governance.get("rank_eligible") is False,
        "Cohere manifest governance",
    )
    _require(
        isinstance(source, Mapping)
        and source.get("cohere_catalog_sha256")
        == SOURCE_SPECS["cohere_catalog"].semantic_sha256
        and source.get("cohere_contract_smoke_sha256s", {}).get(COHERE_MODEL[1])
        == SOURCE_SPECS["cohere_smoke"].semantic_sha256,
        "Cohere manifest transitive evidence",
    )
    models = manifest.get("models")
    _require(isinstance(models, list), "Cohere manifest models")
    matches = [
        row
        for row in models
        if isinstance(row, Mapping)
        and isinstance(row.get("model"), Mapping)
        and row["model"].get("id") == COHERE_MODEL[0]
    ]
    _require(len(matches) == 1, "Cohere manifest Command A+ membership")
    entry = matches[0]
    model = entry.get("model")
    route = entry.get("execution_route")
    backend = entry.get("backend_contract")
    evidence = entry.get("contract_evidence")
    endpoint_selection = entry.get("endpoint_selection")
    _require(
        isinstance(model, Mapping)
        and model.get("canonical_slug") == COHERE_MODEL[1]
        and model.get("catalog_entry") == catalog_entry,
        "Cohere manifest model identity",
    )
    _require(
        isinstance(route, Mapping)
        and route.get("selected_backend") == "cohere_direct"
        and route.get("fallback_used") is False
        and route.get("generation_time_automatic_fallback") is False
        and route.get("evidence", {}).get("compatibility_artifact_sha256")
        == SOURCE_SPECS["cohere_smoke"].semantic_sha256,
        "Cohere route contract",
    )
    _require(
        isinstance(backend, Mapping)
        and backend.get("requested_model_id") == COHERE_MODEL[1]
        and backend.get("catalog_entry_sha256") == catalog_entry_sha256
        and backend.get("allow_fallbacks") is False
        and backend.get("season_eligible") is False,
        "Cohere backend contract",
    )
    _require(
        isinstance(evidence, Mapping)
        and evidence.get("contract_status") == "passed_unranked"
        and evidence.get("cost_status")
        == "public_free_rate_card_provider_charge_unavailable"
        and evidence.get("source_artifact_sha256")
        == SOURCE_SPECS["cohere_smoke"].semantic_sha256,
        "Cohere contract evidence",
    )
    _require(
        isinstance(endpoint_selection, Mapping)
        and endpoint_selection.get("quality_observations_used") == 0,
        "Cohere quality-selection boundary",
    )
    return entry


def _verify_current_design(design: Mapping[str, Any]) -> None:
    verify_current_14_design(design)
    _require(
        design.get("candidate_model_panel", {}).get("model_count") == 14
        and design.get("arithmetic", {}).get("total_planned_unique_real_response_arms")
        == 13_300,
        "current 14-model design comparison boundary",
    )


def _verify_current_sampling(sampling: Mapping[str, Any]) -> None:
    source_commitments = sampling.get("source_commitments")
    certificate = sampling.get("balance_certificate")
    _require(
        sampling.get("status") == "blocked_outcome_blind_sampling_frame_not_authorized",
        "current 14-model sampling status",
    )
    _require(
        isinstance(source_commitments, list)
        and source_commitments
        and source_commitments[0].get("semantic_sha256")
        == SOURCE_SPECS["current_14_design"].semantic_sha256,
        "current 14-model sampling source binding",
    )
    _require(
        isinstance(certificate, Mapping)
        and certificate.get("arena", {}).get("full_graph", {}).get("nodes") == 14
        and certificate.get("arena", {}).get("unique_comparisons") == 800
        and certificate.get("uplift", {}).get("unique_comparisons") == 800
        and certificate.get("human_presentations", {}).get("primary_judgment_slots") == 3_200
        and certificate.get("human_presentations", {}).get(
            "concealed_repeat_presentations"
        )
        == 400,
        "current 14-model sampling arithmetic",
    )


def _verify_inference_acceptance(acceptance: Mapping[str, Any]) -> None:
    global_fit = acceptance.get("global_fit")
    family_fit = acceptance.get("family_specific_fit")
    pairwise = acceptance.get("pairwise_reporting")
    simulation = acceptance.get("simulation_gate")
    _require(
        acceptance.get("status") == "frozen_precollection"
        and acceptance.get("scope") == "official_global_and_family_specific_model_arena_fits"
        and acceptance.get("study_design_artifact_sha256")
        == "7a63cfd6117338a3af16a422d5ee3458298fdc0ff2fd0abfe45fe851a7e54506"
        and acceptance.get("current_development_pool_is_not_grandfathered") is True,
        "frozen inference acceptance identity and non-grandfathering boundary",
    )
    _require(
        isinstance(global_fit, Mapping)
        and global_fit.get("required_admitted_scored_tasks") == 160
        and global_fit.get("required_admitted_scored_tasks_per_family") == 40
        and global_fit.get("minimum_unique_task_clusters_per_model_family") == 20
        and global_fit.get("minimum_unique_comparisons_per_model") == 100
        and global_fit.get("minimum_distinct_independent_raters_per_comparison") == 2,
        "frozen inference global-fit thresholds",
    )
    _require(
        isinstance(family_fit, Mapping)
        and family_fit.get("required_admitted_scored_tasks") == 40
        and family_fit.get("minimum_unique_task_clusters_per_model") == 20
        and family_fit.get("minimum_unique_comparisons_per_model") == 100
        and family_fit.get("minimum_distinct_independent_raters_per_comparison") == 2,
        "frozen inference family-fit thresholds",
    )
    _require(
        isinstance(pairwise, Mapping)
        and pairwise.get("minimum_shared_task_clusters_for_interval") == 10
        and pairwise.get("below_floor_action") == "suppress_pair_specific_interval",
        "frozen inference pairwise threshold",
    )
    _require(
        isinstance(simulation, Mapping)
        and simulation.get("status") == "required_not_yet_executed"
        and simulation.get("models") == 16
        and simulation.get("production_layout_required") is True
        and simulation.get("minimum_datasets_per_scenario") == 2_000
        and simulation.get("bootstrap_replicates") == 5_000
        and simulation.get("validated_surrogate_permitted") is False,
        "frozen inference simulation gate",
    )


def round_robin_matchings(model_ids: Sequence[str]) -> list[list[tuple[str, str]]]:
    """Return the canonical circle-method one-factorization of an even K_n."""

    players = list(model_ids)
    _require(len(players) >= 2 and len(players) % 2 == 0, "model count must be even")
    _require(len(set(players)) == len(players) and "" not in players, "model IDs must be unique")
    matchings: list[list[tuple[str, str]]] = []
    for _ in range(len(players) - 1):
        matchings.append(
            [
                tuple(sorted((players[index], players[-1 - index])))
                for index in range(len(players) // 2)
            ]
        )
        players = [players[0], players[-1], *players[1:-1]]
    pairs = [pair for matching in matchings for pair in matching]
    expected = len(model_ids) * (len(model_ids) - 1) // 2
    _require(len(pairs) == expected and len(set(pairs)) == expected, "K_n factorization")
    return matchings


def build_arena_schedule(model_ids: Sequence[str]) -> dict[str, Any]:
    """Build the fresh five-regular K16 generation schedule."""

    models = list(model_ids)
    _require(len(models) == 16 and len(set(models)) == 16, "arena requires exact K16")
    matchings = round_robin_matchings(models)
    matching_use: Counter[int] = Counter()
    pair_use: Counter[tuple[str, str]] = Counter()
    model_use: Counter[str] = Counter()
    slots: list[dict[str, Any]] = []
    ordinal = 0
    for family in FAMILIES:
        for family_ordinal in range(1, 21):
            ordinal += 1
            matching_indices = [
                ((ordinal - 1) * 5 + offset) % len(matchings) for offset in range(5)
            ]
            pairs = [pair for index in matching_indices for pair in matchings[index]]
            _require(len(pairs) == 40 and len(set(pairs)) == 40, "arena task pair count")
            degree = Counter(model for pair in pairs for model in pair)
            _require(degree == Counter({model: 5 for model in models}), "arena task degree")
            for matching_index in matching_indices:
                matching_use[matching_index] += 1
            pair_use.update(pairs)
            model_use.update(degree)
            slots.append(
                {
                    "design_slot_ordinal": ordinal,
                    "family": family,
                    "family_slot_ordinal": family_ordinal,
                    "matching_indices_zero_based": matching_indices,
                }
            )
    _require(matching_use == Counter({index: 27 if index < 10 else 26 for index in range(15)}),
             "arena matching repetition balance")
    _require(Counter(pair_use.values()) == Counter({27: 80, 26: 40}), "arena pair balance")
    _require(model_use == Counter({model: 400 for model in models}), "arena appearances")
    return {
        "task_binding": (
            "80 abstract scored-slot ordinals only; task IDs require a separate pre-output "
            "80/20/20 split freeze"
        ),
        "task_slots": 80,
        "tasks_per_family": 20,
        "graph": "five_regular_subgraph_of_K16_per_task",
        "comparisons_per_task": 40,
        "endpoint_degree_per_task": 5,
        "total_battles": 3_200,
        "total_response_arms": 6_400,
        "endpoint_appearances_per_model": 400,
        "distinct_endpoint_pairs": 120,
        "pair_repetition_distribution": {"26": 40, "27": 80},
        "factorization": {
            "matching_count": 15,
            "pairs_per_matching": 8,
            "matchings": [
                {
                    "matching_index_zero_based": index,
                    "model_pairs": [list(pair) for pair in matching],
                }
                for index, matching in enumerate(matchings)
            ],
        },
        "abstract_task_schedule": slots,
    }


def build_uplift_schedule(model_ids: Sequence[str]) -> dict[str, Any]:
    """Build 40 paired repetitions per scored task with exact model/family totals."""

    models = list(model_ids)
    _require(len(models) == 16 and len(set(models)) == 16, "uplift requires 16 models")
    totals: Counter[str] = Counter()
    family_totals = {family: Counter() for family in FAMILIES}
    slots: list[dict[str, Any]] = []
    ordinal = 0
    for family in FAMILIES:
        for family_ordinal in range(1, 21):
            ordinal += 1
            third_group = models[:8] if family_ordinal % 2 else models[8:]
            repetitions = {model: 2 + int(model in third_group) for model in models}
            _require(sum(repetitions.values()) == 40, "uplift task pair count")
            totals.update(repetitions)
            family_totals[family].update(repetitions)
            slots.append(
                {
                    "design_slot_ordinal": ordinal,
                    "family": family,
                    "family_slot_ordinal": family_ordinal,
                    "models_with_third_repetition": third_group,
                }
            )
    _require(totals == Counter({model: 200 for model in models}), "uplift model totals")
    _require(
        all(
            counts == Counter({model: 50 for model in models})
            for counts in family_totals.values()
        ),
        "uplift model-family totals",
    )
    return {
        "task_binding": (
            "80 abstract scored-slot ordinals only; task IDs require a separate pre-output "
            "80/20/20 split freeze"
        ),
        "task_slots": 80,
        "tasks_per_family": 20,
        "conditions": list(CONDITIONS),
        "paired_repetitions_per_task": 40,
        "paired_repetitions_per_model_task": [2, 3],
        "paired_repetitions_per_model_family": 50,
        "paired_repetitions_per_model": 200,
        "total_pairs": 3_200,
        "total_response_arms": 6_400,
        "abstract_task_schedule": slots,
    }


def _add_flow_edge(
    graph: list[list[_FlowEdge]], source: int, target: int, capacity: int
) -> _FlowEdge:
    forward = _FlowEdge(target=target, capacity=capacity, reverse=len(graph[target]))
    backward = _FlowEdge(target=source, capacity=0, reverse=len(graph[source]))
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _push_flow(
    graph: list[list[_FlowEdge]],
    levels: Sequence[int],
    next_edge: list[int],
    node: int,
    sink: int,
    available: int,
) -> int:
    if node == sink:
        return available
    while next_edge[node] < len(graph[node]):
        edge = graph[node][next_edge[node]]
        if edge.capacity > 0 and levels[edge.target] == levels[node] + 1:
            sent = _push_flow(
                graph,
                levels,
                next_edge,
                edge.target,
                sink,
                min(available, edge.capacity),
            )
            if sent:
                edge.capacity -= sent
                graph[edge.target][edge.reverse].capacity += sent
                return sent
        next_edge[node] += 1
    return 0


def _maximum_flow(graph: list[list[_FlowEdge]], source: int, sink: int) -> int:
    total = 0
    while True:
        levels = [-1] * len(graph)
        levels[source] = 0
        pending = deque([source])
        while pending:
            node = pending.popleft()
            for edge in graph[node]:
                if edge.capacity > 0 and levels[edge.target] < 0:
                    levels[edge.target] = levels[node] + 1
                    pending.append(edge.target)
        if levels[sink] < 0:
            return total
        next_edge = [0] * len(graph)

        while sent := _push_flow(graph, levels, next_edge, source, sink, 10**9):
            total += sent


def _sampling_recipe() -> dict[str, Any]:
    body = {
        "recipe_version": SAMPLING_RECIPE_VERSION,
        "outcome_blind": True,
        "source_semantic_sha256s": {
            key: SOURCE_SPECS[key].semantic_sha256
            for key in ("current_14_roster", "qwen_projection", "cohere_manifest")
        },
        "arena": {
            "source_candidates_per_task": 40,
            "selected_per_task": 10,
            "selection_algorithm": (
                "deterministic Dinic max-flow in canonical task, matching, pair order"
            ),
            "pair_target_rule": (
                "six selections per K16 pair plus one for each listed high-target coordinate"
            ),
            "high_target_matching_pair_coordinates_zero_based": [
                list(coordinate) for coordinate in sorted(ARENA_HUMAN_HIGH_TARGET_COORDINATES)
            ],
        },
        "uplift": {
            "selected_distinct_models_per_task": 10,
            "model_index_formula": (
                "indices (design_slot_ordinal - 1 + delta) mod 16 for delta=0..9"
            ),
            "source_repetition_index_formula": (
                "1 + uint64_be(sha256(recipe_version|uplift|task|model)[0:8]) "
                "mod generated repetitions available"
            ),
        },
        "rater_slots": {
            "abstract_slots_per_comparison": list(RATER_SLOTS),
            "distinct_people_required": True,
            "reviewer_identities_assigned": 0,
        },
        "concealed_repeats": {
            "rate_of_primary_judgment_slots": "0.125",
            "repeats_per_task": 5,
            "category_order": [
                {"track": track, "rater_slot": slot}
                for track, slot in (
                    ("model_arena", 1),
                    ("model_arena", 2),
                    ("epicure_uplift", 1),
                    ("epicure_uplift", 2),
                )
            ],
            "rule": (
                "one SHA-256-ranked source slot per category plus a second from category "
                "(design_slot_ordinal - 1) mod 4"
            ),
            "same_rater_as_source_required": True,
        },
    }
    return {**body, "recipe_sha256": sha256_json(body)}


def _comparison_id(identity: Mapping[str, Any]) -> str:
    return f"comparison-{sha256_json(identity)}"


def _judgment_slot_id(comparison_id: str, rater_slot: int) -> str:
    identity = {"comparison_id": comparison_id, "rater_slot": rater_slot}
    return f"judgment-slot-{sha256_json(identity)}"


def _arena_occurrences(arena: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    factorization = arena["factorization"]["matchings"]
    by_index = {
        int(row["matching_index_zero_based"]): [tuple(pair) for pair in row["model_pairs"]]
        for row in factorization
    }
    result: list[list[dict[str, Any]]] = []
    for slot in arena["abstract_task_schedule"]:
        occurrences: list[dict[str, Any]] = []
        for matching_index in slot["matching_indices_zero_based"]:
            for pair_index, pair in enumerate(by_index[int(matching_index)]):
                occurrences.append(
                    {
                        "matching_index_zero_based": int(matching_index),
                        "matching_pair_index_zero_based": pair_index,
                        "model_ids": tuple(sorted(pair)),
                    }
                )
        _require(len(occurrences) == 40, "arena occurrence materialization")
        result.append(occurrences)
    return result


def _arena_human_frame(
    arena: Mapping[str, Any],
    recipe_sha256: str,
) -> list[dict[str, Any]]:
    occurrences_by_task = _arena_occurrences(arena)
    factorization = arena["factorization"]["matchings"]
    ordered_pairs: list[tuple[str, str]] = []
    pair_targets: dict[tuple[str, str], int] = {}
    for matching in factorization:
        matching_index = int(matching["matching_index_zero_based"])
        for pair_index, raw_pair in enumerate(matching["model_pairs"]):
            pair = tuple(sorted(raw_pair))
            ordered_pairs.append(pair)
            pair_targets[pair] = 6 + int(
                (matching_index, pair_index) in ARENA_HUMAN_HIGH_TARGET_COORDINATES
            )
    _require(len(ordered_pairs) == 120 and len(set(ordered_pairs)) == 120, "arena pair order")
    _require(sum(pair_targets.values()) == 800, "arena sampling pair targets")

    source = 0
    task_start = 1
    pair_start = task_start + 80
    sink = pair_start + 120
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    pair_index = {pair: index for index, pair in enumerate(ordered_pairs)}
    occurrence_edges: list[list[tuple[dict[str, Any], _FlowEdge]]] = []
    for task_index, occurrences in enumerate(occurrences_by_task):
        task_node = task_start + task_index
        _add_flow_edge(graph, source, task_node, 10)
        tracked: list[tuple[dict[str, Any], _FlowEdge]] = []
        for occurrence in occurrences:
            pair = occurrence["model_ids"]
            edge = _add_flow_edge(graph, task_node, pair_start + pair_index[pair], 1)
            tracked.append((occurrence, edge))
        occurrence_edges.append(tracked)
    for pair, index in pair_index.items():
        _add_flow_edge(graph, pair_start + index, sink, pair_targets[pair])
    _require(_maximum_flow(graph, source, sink) == 800, "arena human-frame max flow")

    slots = arena["abstract_task_schedule"]
    rows: list[dict[str, Any]] = []
    for slot, tracked in zip(slots, occurrence_edges, strict=True):
        selected = [occurrence for occurrence, edge in tracked if edge.capacity == 0]
        _require(len(selected) == 10, "arena human comparisons per task")
        for occurrence in selected:
            identity = {
                "recipe_sha256": recipe_sha256,
                "track": "model_arena",
                "design_slot_ordinal": slot["design_slot_ordinal"],
                "source_matching_index_zero_based": occurrence[
                    "matching_index_zero_based"
                ],
                "source_matching_pair_index_zero_based": occurrence[
                    "matching_pair_index_zero_based"
                ],
                "model_ids": list(occurrence["model_ids"]),
            }
            comparison_id = _comparison_id(identity)
            rows.append(
                {
                    **identity,
                    "comparison_id": comparison_id,
                    "family": slot["family"],
                    "family_slot_ordinal": slot["family_slot_ordinal"],
                    "required_distinct_raters": 2,
                    "judgment_slot_ids": [
                        _judgment_slot_id(comparison_id, rater_slot)
                        for rater_slot in RATER_SLOTS
                    ],
                }
            )
    return rows


def _uplift_human_frame(
    model_ids: Sequence[str],
    uplift: Mapping[str, Any],
    recipe_sha256: str,
) -> list[dict[str, Any]]:
    models = list(model_ids)
    rows: list[dict[str, Any]] = []
    for slot in uplift["abstract_task_schedule"]:
        ordinal = int(slot["design_slot_ordinal"])
        selected_indices = sorted({(ordinal - 1 + delta) % 16 for delta in range(10)})
        _require(len(selected_indices) == 10, "uplift human models per task")
        third_group = set(slot["models_with_third_repetition"])
        for index in selected_indices:
            model_id = models[index]
            available = 2 + int(model_id in third_group)
            seed = (
                f"{SAMPLING_RECIPE_VERSION}|uplift|task={ordinal}|model={model_id}"
            ).encode()
            repetition = 1 + int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % available
            identity = {
                "recipe_sha256": recipe_sha256,
                "track": "epicure_uplift",
                "design_slot_ordinal": ordinal,
                "model_id": model_id,
                "source_generated_repetition_index_one_based": repetition,
            }
            comparison_id = _comparison_id(identity)
            rows.append(
                {
                    **identity,
                    "comparison_id": comparison_id,
                    "family": slot["family"],
                    "family_slot_ordinal": slot["family_slot_ordinal"],
                    "conditions": list(CONDITIONS),
                    "source_generated_repetitions_available": available,
                    "required_distinct_raters": 2,
                    "judgment_slot_ids": [
                        _judgment_slot_id(comparison_id, rater_slot)
                        for rater_slot in RATER_SLOTS
                    ],
                }
            )
    return rows


def _repeat_frame(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[tuple[int, str, int], list[tuple[str, str]]] = defaultdict(list)
    for row in comparisons:
        for rater_slot, judgment_slot_id in zip(
            RATER_SLOTS, row["judgment_slot_ids"], strict=True
        ):
            by_category[
                int(row["design_slot_ordinal"]), str(row["track"]), rater_slot
            ].append((str(row["comparison_id"]), str(judgment_slot_id)))
    categories = (
        ("model_arena", 1),
        ("model_arena", 2),
        ("epicure_uplift", 1),
        ("epicure_uplift", 2),
    )
    repeats: list[dict[str, Any]] = []
    for task_ordinal in range(1, 81):
        extra_category = (task_ordinal - 1) % 4
        for category_index, (track, rater_slot) in enumerate(categories):
            candidates = by_category[task_ordinal, track, rater_slot]
            _require(len(candidates) == 10, "repeat source category")
            count = 1 + int(category_index == extra_category)
            ranked = sorted(
                candidates,
                key=lambda item: hashlib.sha256(
                    f"{SAMPLING_RECIPE_VERSION}|concealed-repeat|{item[1]}".encode()
                ).hexdigest(),
            )
            for comparison_id, judgment_slot_id in ranked[:count]:
                identity = {
                    "recipe_version": SAMPLING_RECIPE_VERSION,
                    "source_judgment_slot_id": judgment_slot_id,
                    "repeat_ordinal_for_source": 1,
                }
                repeats.append(
                    {
                        "repeat_presentation_id": (
                            f"repeat-presentation-{sha256_json(identity)}"
                        ),
                        "source_comparison_id": comparison_id,
                        "source_judgment_slot_id": judgment_slot_id,
                        "design_slot_ordinal": task_ordinal,
                        "track": track,
                        "rater_slot": rater_slot,
                        "same_rater_as_source_required": True,
                    }
                )
    return repeats


def _distribution(values: Sequence[int]) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values).items())}


def _sampling_certificate(
    model_ids: Sequence[str],
    arena: Sequence[Mapping[str, Any]],
    uplift: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    models = list(model_ids)
    arena_tasks = Counter(int(row["design_slot_ordinal"]) for row in arena)
    arena_families = Counter(str(row["family"]) for row in arena)
    arena_models = Counter(model for row in arena for model in row["model_ids"])
    arena_pairs = Counter(tuple(row["model_ids"]) for row in arena)
    arena_family_models = Counter(
        (str(row["family"]), model) for row in arena for model in row["model_ids"]
    )
    arena_family_model_tasks: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in arena:
        for model in row["model_ids"]:
            arena_family_model_tasks[str(row["family"]), model].add(
                int(row["design_slot_ordinal"])
            )
    _require(len(arena) == 800 and set(arena_tasks.values()) == {10}, "arena sample size")
    _require(arena_families == Counter({family: 200 for family in FAMILIES}),
             "arena sample family count")
    _require(arena_models == Counter({model: 100 for model in models}),
             "arena sample model appearances")
    _require(len(arena_pairs) == 120 and Counter(arena_pairs.values()) == Counter({7: 80, 6: 40}),
             "arena sample pair balance")

    uplift_tasks = Counter(int(row["design_slot_ordinal"]) for row in uplift)
    uplift_families = Counter(str(row["family"]) for row in uplift)
    uplift_models = Counter(str(row["model_id"]) for row in uplift)
    _require(len(uplift) == 800 and set(uplift_tasks.values()) == {10}, "uplift sample size")
    _require(uplift_families == Counter({family: 200 for family in FAMILIES}),
             "uplift sample family count")
    _require(uplift_models == Counter({model: 50 for model in models}),
             "uplift sample model appearances")
    _require(
        all(
            1 <= int(row["source_generated_repetition_index_one_based"])
            <= int(row["source_generated_repetitions_available"])
            for row in uplift
        ),
        "uplift sample source membership",
    )

    comparisons = [*arena, *uplift]
    comparison_ids = [str(row["comparison_id"]) for row in comparisons]
    judgment_slot_ids = [
        str(slot_id) for row in comparisons for slot_id in row["judgment_slot_ids"]
    ]
    _require(len(set(comparison_ids)) == 1_600, "sample comparison ID uniqueness")
    _require(len(judgment_slot_ids) == 3_200 and len(set(judgment_slot_ids)) == 3_200,
             "primary judgment slot IDs")
    _require(len(repeats) == 400, "concealed repeat count")
    _require(len({row["repeat_presentation_id"] for row in repeats}) == 400,
             "concealed repeat ID uniqueness")
    repeat_tasks = Counter(int(row["design_slot_ordinal"]) for row in repeats)
    repeat_tracks = Counter(str(row["track"]) for row in repeats)
    repeat_raters = Counter(int(row["rater_slot"]) for row in repeats)
    _require(set(repeat_tasks.values()) == {5}, "concealed repeats per task")
    _require(repeat_tracks == Counter({"model_arena": 200, "epicure_uplift": 200}),
             "concealed repeat track balance")
    _require(repeat_raters == Counter({1: 200, 2: 200}), "concealed repeat rater balance")
    return {
        "arena": {
            "unique_comparisons": 800,
            "comparisons_per_task": 10,
            "comparisons_per_family": {family: 200 for family in FAMILIES},
            "global_model_appearance_distribution": _distribution(list(arena_models.values())),
            "exact_appearances_per_model": 100,
            "arithmetic_average_appearances_per_model_family": 25,
            "appearances_per_model_family": {
                family: {
                    model: arena_family_models[family, model] for model in models
                }
                for family in FAMILIES
            },
            "model_appearance_distribution_per_family": {
                family: _distribution(
                    [arena_family_models[family, model] for model in models]
                )
                for family in FAMILIES
            },
            "model_appearance_range_across_family_cells": [
                min(arena_family_models.values()),
                max(arena_family_models.values()),
            ],
            "unique_task_clusters_per_model_family": {
                family: {
                    model: len(arena_family_model_tasks[family, model]) for model in models
                }
                for family in FAMILIES
            },
            "unique_task_cluster_range_across_model_family_cells": [
                min(len(tasks) for tasks in arena_family_model_tasks.values()),
                max(len(tasks) for tasks in arena_family_model_tasks.values()),
            ],
            "global_pair_repetition_distribution": _distribution(list(arena_pairs.values())),
            "full_graph": {"nodes": 16, "distinct_edges": 120, "connected": True},
            "is_subset_of_40_generated_pairs_per_task": True,
            "exact_pair_counts_sha256": sha256_json(
                {" || ".join(pair): count for pair, count in sorted(arena_pairs.items())}
            ),
            "exact_selected_comparison_ids_sha256": sha256_json(sorted(comparison_ids[:800])),
        },
        "uplift": {
            "unique_comparisons": 800,
            "distinct_models_per_task": 10,
            "comparisons_per_family": {family: 200 for family in FAMILIES},
            "global_model_appearance_distribution": _distribution(list(uplift_models.values())),
            "exact_appearances_per_model": 50,
            "selected_repetition_exists_in_generation_schedule": True,
            "exact_selected_comparison_ids_sha256": sha256_json(sorted(comparison_ids[800:])),
        },
        "human_presentations": {
            "unique_comparisons": 1_600,
            "distinct_rater_slots_per_comparison": 2,
            "primary_judgment_slots": 3_200,
            "concealed_repeat_presentations": 400,
            "total_rating_presentations": 3_600,
            "repeats_per_task": 5,
            "repeat_presentations_per_track": {
                "model_arena": 200,
                "epicure_uplift": 200,
            },
            "repeat_presentations_per_abstract_rater_slot": {"1": 200, "2": 200},
            "exact_primary_judgment_slot_ids_sha256": sha256_json(sorted(judgment_slot_ids)),
            "exact_repeat_presentation_ids_sha256": sha256_json(
                sorted(str(row["repeat_presentation_id"]) for row in repeats)
            ),
        },
    }


def _materialize_sampling(
    model_ids: Sequence[str],
    arena_schedule: Mapping[str, Any],
    uplift_schedule: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    recipe = _sampling_recipe()
    recipe_sha256 = str(recipe["recipe_sha256"])
    arena = _arena_human_frame(arena_schedule, recipe_sha256)
    uplift = _uplift_human_frame(model_ids, uplift_schedule, recipe_sha256)
    repeats = _repeat_frame([*arena, *uplift])
    certificate = _sampling_certificate(model_ids, arena, uplift, repeats)
    summary = {
        "status": "blocked_outcome_blind_16_model_sampling_frame_not_authorized",
        "sampling_recipe": recipe,
        "balance_certificate": certificate,
        "materialization_contract": {
            "implementation": (
                "flavourbench.task_campaign_16_model_alternative."
                "materialize_human_sampling_frame"
            ),
            "compressed_frame_is_exact": True,
            "materialized_arena_comparisons": 800,
            "materialized_uplift_comparisons": 800,
            "materialized_primary_judgment_slots": 3_200,
            "materialized_concealed_repeat_presentations": 400,
            "actual_reviewer_assignments": 0,
        },
        "selection_timing_and_inputs": {
            "outcome_blind": True,
            "must_be_frozen_before_any_model_output_or_quality_outcome": True,
            "response_texts_used": 0,
            "model_scores_used": 0,
            "preference_labels_used": 0,
            "quality_observations_used": 0,
            "reviewer_identities_used": 0,
            "outcome_dependent_reselection_allowed": False,
        },
        "current_14_sampling_disposition": {
            "semantic_sha256": SOURCE_SPECS["current_14_sampling"].semantic_sha256,
            "source_design_semantic_sha256": SOURCE_SPECS["current_14_design"].semantic_sha256,
            "applicable_to_this_16_model_alternative": False,
            "reason": (
                "the existing coordinate recipe is content-bound to the 14-model K14 schedule; "
                "it is comparison evidence only and cannot be patched or reused for K16"
            ),
        },
        "authorized": False,
    }
    return summary, {"arena": arena, "uplift": uplift, "concealed_repeats": repeats}


def _inference_acceptance_mismatch(
    acceptance: Mapping[str, Any], sampling: Mapping[str, Any]
) -> dict[str, Any]:
    global_fit = acceptance["global_fit"]
    family_fit = acceptance["family_specific_fit"]
    pairwise = acceptance["pairwise_reporting"]
    simulation = acceptance["simulation_gate"]
    arena = sampling["balance_certificate"]["arena"]
    family_appearance_range = arena["model_appearance_range_across_family_cells"]
    family_cluster_range = arena[
        "unique_task_cluster_range_across_model_family_cells"
    ]
    return {
        "status": "no_go_successor_inference_and_power_contract_required",
        "official_v1_inherited": False,
        "official_fit_authorized": False,
        "pair_specific_intervals_authorized": False,
        "reason": (
            "the frozen v1 acceptance contract is content-bound to a different production "
            "study design and the 80-task/800-arena alternative also fails multiple numeric "
            "floors; arithmetic feasibility cannot be promoted to inference eligibility"
        ),
        "source_contract": {
            "semantic_sha256": SOURCE_SPECS["inference_acceptance"].semantic_sha256,
            "physical_sha256": SOURCE_SPECS["inference_acceptance"].physical_sha256,
            "bound_study_design_artifact_sha256": acceptance[
                "study_design_artifact_sha256"
            ],
            "bound_study_design_matches_this_alternative": False,
            "scope": acceptance["scope"],
            "current_development_pool_is_not_grandfathered": acceptance[
                "current_development_pool_is_not_grandfathered"
            ],
        },
        "validation_matrix": {
            "global_required_admitted_scored_tasks": {
                "required": global_fit["required_admitted_scored_tasks"],
                "alternative_design_capacity": 80,
                "admitted_now": 0,
                "design_capacity_meets_requirement": False,
                "acceptance_satisfied_now": False,
            },
            "global_required_admitted_scored_tasks_per_family": {
                "required": global_fit["required_admitted_scored_tasks_per_family"],
                "alternative_design_capacity": 20,
                "admitted_now": 0,
                "design_capacity_meets_requirement": False,
                "acceptance_satisfied_now": False,
            },
            "global_minimum_unique_comparisons_per_model": {
                "required": global_fit["minimum_unique_comparisons_per_model"],
                "outcome_blind_frame_capacity": arena["exact_appearances_per_model"],
                "collected_now": 0,
                "design_capacity_meets_numeric_requirement": True,
                "acceptance_satisfied_now": False,
                "note": "the K16 frame meets this one numeric floor exactly, globally only",
            },
            "global_minimum_unique_task_clusters_per_model_family": {
                "required": global_fit[
                    "minimum_unique_task_clusters_per_model_family"
                ],
                "outcome_blind_frame_range": family_cluster_range,
                "all_model_family_cells_meet_requirement": False,
                "acceptance_satisfied_now": False,
            },
            "family_fit_required_admitted_scored_tasks": {
                "required_per_family_fit": family_fit["required_admitted_scored_tasks"],
                "alternative_design_capacity_per_family": 20,
                "admitted_now": 0,
                "design_capacity_meets_requirement": False,
                "acceptance_satisfied_now": False,
            },
            "family_fit_minimum_unique_comparisons_per_model": {
                "required_per_family": family_fit[
                    "minimum_unique_comparisons_per_model"
                ],
                "arithmetic_average_per_model_family": arena[
                    "arithmetic_average_appearances_per_model_family"
                ],
                "outcome_blind_frame_range": family_appearance_range,
                "all_model_family_cells_meet_requirement": False,
                "acceptance_satisfied_now": False,
            },
            "family_fit_minimum_unique_task_clusters_per_model": {
                "required_per_family": family_fit[
                    "minimum_unique_task_clusters_per_model"
                ],
                "outcome_blind_frame_range": family_cluster_range,
                "all_model_family_cells_meet_requirement": False,
                "acceptance_satisfied_now": False,
            },
            "distinct_independent_raters_per_comparison": {
                "required": global_fit[
                    "minimum_distinct_independent_raters_per_comparison"
                ],
                "abstract_slots_per_comparison": 2,
                "actual_reviewers_assigned": 0,
                "acceptance_satisfied_now": False,
            },
            "bootstrap_connectivity_bias_coverage_and_error": {
                "required": True,
                "validated_for_this_alternative": False,
                "acceptance_satisfied_now": False,
            },
        },
        "pairwise_direct_support": {
            "required_shared_task_clusters_per_pair": pairwise[
                "minimum_shared_task_clusters_for_interval"
            ],
            "outcome_blind_frame_pair_repetition_range": [6, 7],
            "distinct_pairs_below_floor": 120,
            "distinct_pairs_at_or_above_floor": 0,
            "v1_below_floor_action": pairwise["below_floor_action"],
            "passes": False,
        },
        "simulation_contract": {
            "required_model_count": simulation["models"],
            "alternative_model_count": 16,
            "model_count_matches": True,
            "production_layout_required": simulation["production_layout_required"],
            "production_layout_matches_bound_v1_design": False,
            "minimum_datasets_per_scenario": simulation[
                "minimum_datasets_per_scenario"
            ],
            "datasets_executed_for_this_alternative": 0,
            "required_bootstrap_replicates": simulation["bootstrap_replicates"],
            "bootstrap_replicates_executed_for_this_alternative": 0,
            "validated_surrogate_permitted": simulation[
                "validated_surrogate_permitted"
            ],
            "passes": False,
        },
        "remediation_options_requiring_new_reviewed_content_addresses": {
            "retain_direct_pairwise_interval_support_floor": {
                "minimum_arena_comparisons": 1_200,
                "derivation": "120 K16 unordered pairs times 10 shared task clusters",
                "comparisons_per_model_if_every_pair_is_sampled_exactly_10_times": 150,
                "current_800_frame_reusable": False,
                "new_outcome_blind_allocation_power_cost_and_repeat_contract_required": True,
                "cures_160_task_and_40_per_family_fit_gates": False,
                "note": (
                    "1,200 is an arithmetic lower bound, not a certified fixed-per-task frame; "
                    "the replacement allocation must independently prove feasibility and balance"
                ),
            },
            "version_or_narrow_estimands": {
                "required_action": (
                    "adopt a successor inference-acceptance and power contract aligned to the "
                    "80/20 design, predeclare narrower estimands/reporting, and validate them "
                    "before collection"
                ),
                "pair_intervals_below_10_under_v1_remain_suppressed": True,
                "official_or_rank_authority_created_by_this_option_now": False,
            },
        },
    }


def _validation_matrix(
    kimi_entry: Mapping[str, Any],
    qwen: Mapping[str, Any],
    cohere_entry: Mapping[str, Any],
) -> dict[str, Any]:
    kimi_model = kimi_entry["model"]
    kimi_route = kimi_entry["execution_route"]
    kimi_backend = kimi_entry["backend_contract"]
    kimi_evidence = kimi_entry["contract_evidence"]
    qwen_identity = qwen["model_identity"]
    qwen_accounting = qwen["combined_reliability_accounting"]
    cohere_model = cohere_entry["model"]
    cohere_route = cohere_entry["execution_route"]
    cohere_backend = cohere_entry["backend_contract"]
    cohere_evidence = cohere_entry["contract_evidence"]
    common_eligibility = {
        "alternative_candidate_member": True,
        "official": False,
        "rank_eligible": False,
        "call_authorized": False,
        "quality_eligible": False,
    }
    return {
        "selection_rule": (
            "identity and route evidence only; no response quality, score, preference, or rank "
            "was inspected or used to select membership"
        ),
        "quality_based_choice_made": False,
        "rows": [
            {
                "model_id": "moonshotai/kimi-k3",
                "current_14_member": True,
                "identity": {
                    "status": "time_bounded_provider_managed_direct_identifier",
                    "logical_model_id": kimi_model["id"],
                    "canonical_model_slug": kimi_model["canonical_slug"],
                    "requested_model_id": kimi_backend["requested_model_id"],
                    "identity_basis": kimi_evidence["identity_basis"],
                    "catalog_snapshot_observed_at": "2026-08-01T00:00:00Z",
                    "catalog_details_path_contains_dated_slug": (
                        kimi_model.get("links", {}).get("details")
                    ),
                    "direct_requested_id_is_dated": False,
                    "immutable_served_revision_proven": False,
                    "continuity_or_version_stability_inferred": False,
                    "strength_relative_to_qwen": (
                        "stronger route provenance from the content-bound catalog and direct "
                        "response evidence, but not stronger proof of immutable served weights"
                    ),
                    "compatibility_artifact_sha256": kimi_evidence[
                        "source_artifact_sha256"
                    ],
                },
                "route": {
                    "status": kimi_evidence["contract_status"],
                    "selected_backend": kimi_route["selected_backend"],
                    "actual_provider": kimi_evidence["actual_provider"],
                    "fallback_used": kimi_route["fallback_used"],
                    "season_eligible_in_source_contract": kimi_backend["season_eligible"],
                },
                "cost": {
                    "status": "blocked_pending_full_schedule_reprice_and_reconciliation",
                    "source_cost_status": kimi_evidence["cost_status"],
                    "frozen_rate_card_is_not_final_cost_authority": True,
                    "candidate_15200_arm_cost_validated": False,
                },
                "eligibility": common_eligibility,
            },
            {
                "model_id": "qwen3.8-max",
                "current_14_member": False,
                "identity": {
                    "status": "time_bounded_mutable_alias_only",
                    "identity_kind": qwen_identity["identity_kind"],
                    "requested_model_id": qwen_identity["requested_model_id"],
                    "returned_model_ids": qwen_identity["returned_model_ids"],
                    "frozen_release": qwen_identity["frozen_release"],
                    "catalog_observed_at": qwen_identity["catalog_observed_at"],
                    "projection_recorded_at": qwen["recorded_at"],
                    "continuity_or_version_stability_inferred": False,
                    "evidence_scope": (
                        "only the recorded 2026-08-08 catalog and exploratory run observations"
                    ),
                },
                "route": {
                    "status": "verified_exploratory_post_freeze_only",
                    "provider": qwen_identity["provider"],
                    "automatic_provider_fallback": qwen_identity[
                        "automatic_provider_fallback"
                    ],
                    "season_eligible_in_source_projection": False,
                },
                "cost": {
                    "status": "blocked_unknown_not_free",
                    "provider_charge_available": qwen_accounting[
                        "provider_charge_available"
                    ],
                    "provider_cost_reconciled": qwen_accounting[
                        "provider_cost_reconciled"
                    ],
                    "recorded_zero_cost_means": qwen_accounting[
                        "recorded_zero_cost_means"
                    ],
                    "candidate_15200_arm_cost_validated": False,
                },
                "eligibility": common_eligibility,
            },
            {
                "model_id": "cohere/command-a-plus-05-2026",
                "current_14_member": False,
                "identity": {
                    "status": "authenticated_catalog_and_contract_smoke_bound",
                    "logical_model_id": cohere_model["id"],
                    "canonical_model_slug": cohere_model["canonical_slug"],
                    "requested_model_id": cohere_backend["requested_model_id"],
                    "catalog_sha256": cohere_backend["catalog_sha256"],
                    "catalog_entry_sha256": cohere_backend["catalog_entry_sha256"],
                    "compatibility_artifact_sha256": cohere_evidence[
                        "source_artifact_sha256"
                    ],
                },
                "route": {
                    "status": cohere_evidence["contract_status"],
                    "selected_backend": cohere_route["selected_backend"],
                    "actual_provider": cohere_evidence["actual_provider"],
                    "fallback_used": cohere_route["fallback_used"],
                    "season_eligible_in_source_contract": cohere_backend["season_eligible"],
                },
                "cost": {
                    "status": "blocked_separate_direct_provider_cost_governor_missing",
                    "source_cost_status": cohere_evidence["cost_status"],
                    "smoke_cost_status": "no_per_generation_cost_returned_by_provider",
                    "zero_public_rate_may_not_be_treated_as_reconciled_cost": True,
                    "separate_cost_governor_required": True,
                    "candidate_15200_arm_cost_validated": False,
                },
                "eligibility": common_eligibility,
            },
        ],
    }


def _missing_requirement(requirement_id: str, acceptance_evidence: str) -> dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "status": "missing",
        "satisfied_by_this_artifact": False,
        "acceptance_evidence": acceptance_evidence,
    }


def _remaining_requirements() -> dict[str, list[dict[str, Any]]]:
    return {
        "kimi_identity_and_cost": [
            _missing_requirement(
                "kimi_direct_revision_identity",
                (
                    "provider evidence that direct requested ID k3 resolves to an immutable "
                    "revision, or a dated immutable direct ID; the locally dated catalog details "
                    "path alone is a snapshot and is not proof of immutable serving"
                ),
            ),
            _missing_requirement(
                "kimi_full_block_cost_reconciliation",
                (
                    "a refreshed authoritative rate card, full-block worst-case bound, hard cap, "
                    "and actual-charge reconciliation despite unavailable per-generation charge"
                ),
            ),
        ],
        "qwen_cost_and_identity": [
            _missing_requirement(
                "qwen_alias_reverification_at_roster_freeze",
                (
                    "a newly content-addressed authenticated catalog observation immediately "
                    "before freeze, retaining identity_kind=mutable_alias and an explicit time"
                    " window"
                ),
            ),
            _missing_requirement(
                "qwen_per_call_requested_returned_identity_guard",
                (
                    "every production receipt binds requested qwen3.8-max to the returned ID, "
                    "fallback is disabled, and any drift invalidates rather than silently updates"
                ),
            ),
            _missing_requirement(
                "qwen_cost_authority",
                (
                    "provider price/charge authority, token accounting, worst-case forecast, "
                    "reservation rule, hard cap, and post-call reconciliation for its full block"
                ),
            ),
        ],
        "cohere_cost_governor": [
            _missing_requirement(
                "cohere_direct_lane_cost_governor",
                (
                    "a separately reviewed direct-provider governor with authoritative prices, "
                    "per-reservation worst-case bounds, concurrency-safe hard stop, and actual "
                    "charge reconciliation; provider-unreturned charge may not be recorded as zero"
                ),
            ),
        ],
        "provider_contracts_all_16": [
            _missing_requirement(
                "exact_production_route_contracts",
                (
                    "all 16 exact requested/returned identities, endpoints, regions, capabilities, "
                    "decoding, data/retention policy, no-fallback rules, and normal-finish "
                    "evidence "
                    "are frozen and pass the production contract"
                ),
            ),
            _missing_requirement(
                "route_drift_invalidation",
                "provider or identity drift fails closed and forces a new roster and schedule hash",
            ),
        ],
        "power_and_statistics": [
            _missing_requirement(
                "successor_inference_acceptance_contract",
                (
                    "a new reviewed content-addressed contract explicitly governs the 80-scored-"
                    "task K16 layout and either raises the arena human frame to at least the "
                    "1,200-comparison arithmetic lower bound for ten shared clusters per pair or "
                    "narrows/version-controls pairwise estimands after power validation; frozen "
                    "v1 remains non-applicable and cannot be inherited"
                ),
            ),
            _missing_requirement(
                "16_model_estimands_and_power",
                (
                    "predeclared primary/family estimands plus simulation of MDE, precision, "
                    "coverage, type-I error, and multiplicity for the K16 and selected human "
                    "sample; if v1 simulation thresholds are retained, run at least 2,000 "
                    "datasets per scenario and 5,000 bootstraps on the exact production layout"
                ),
            ),
            _missing_requirement(
                "missingness_ties_and_dependence",
                (
                    "validated handling of missing arms/ratings, retries, ties, task clustering, "
                    "rater dependence, separation, and reviewer dropout"
                ),
            ),
        ],
        "cost_and_funding": [
            _missing_requirement(
                "15200_arm_cost_envelope",
                (
                    "all 15,200 unique real arms are repriced against exact routes with funded "
                    "caps, reserves, rate limits, and reconciliation"
                ),
            ),
            _missing_requirement(
                "3600_presentation_human_budget",
                (
                    "funded compensation, training, calibration, adjudication, platform, tax, "
                    "withdrawal, retention, and contingency costs for 3,600 presentations"
                ),
            ),
        ],
        "ethics_and_human_operations": [
            _missing_requirement(
                "ethics_or_equivalent_determination",
                (
                    "recorded ethics/equivalent determination plus approved consent, withdrawal, "
                    "retention, privacy, identity, conflict-of-interest, and incident procedures"
                ),
            ),
            _missing_requirement(
                "reviewer_admission_and_assignment",
                (
                    "qualified reviewers accept terms, pass calibration, and are assigned under "
                    "the distinct-rater and concealed-repeat contract"
                ),
            ),
        ],
        "tasks_and_split": [
            _missing_requirement(
                "120_human_admitted_tasks",
                (
                    "120 rights-cleared, human-validated admissions reach exactly 30 per family; "
                    "public-source tasks remain contamination-limited, not private or clean"
                ),
            ),
            _missing_requirement(
                "pre_output_80_20_20_split",
                (
                    "an independently reviewed deterministic split binds 80 scored, 20 disjoint "
                    "development, and 20 rotation-reserve task IDs before any model output"
                ),
            ),
        ],
        "joint_activation_change": [
            _missing_requirement(
                "new_official_content_address",
                (
                    "a reviewed official contract adopts this or another design together with "
                    "importer, readiness, release, claim, power, cost, ethics, and task evidence; "
                    "the current 14-model design is never silently patched"
                ),
            ),
        ],
    }


def _build_body(
    roster: Mapping[str, Any],
    qwen: Mapping[str, Any],
    cohere_manifest: Mapping[str, Any],
    cohere_catalog: Mapping[str, Any],
    cohere_smoke: Mapping[str, Any],
    current_design: Mapping[str, Any],
    current_sampling: Mapping[str, Any],
    inference_acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    current_rows, kimi_entry = _verify_current_roster(roster)
    qwen_row = _verify_qwen_projection(qwen)
    cohere_entry = _verify_cohere_sources(cohere_manifest, cohere_catalog, cohere_smoke)
    _verify_current_design(current_design)
    _verify_current_sampling(current_sampling)
    _verify_inference_acceptance(inference_acceptance)
    cohere_model = cohere_entry["model"]
    cohere_row = {
        "model_id": COHERE_MODEL[0],
        "canonical_model_slug": COHERE_MODEL[1],
        "display_name": str(cohere_model["name"]),
        "identity_kind": "dated_authenticated_provider_model_id",
        "evidence_source_role": SOURCE_SPECS["cohere_manifest"].role,
    }
    panel_rows = [*current_rows, qwen_row, cohere_row]
    observed = tuple(
        (row["model_id"], row["canonical_model_slug"]) for row in panel_rows
    )
    _require(observed == EXPECTED_16_MODELS, "exact 16-model panel membership or order")
    model_ids = [row["model_id"] for row in panel_rows]
    arena = build_arena_schedule(model_ids)
    uplift = build_uplift_schedule(model_ids)
    sampling, _ = _materialize_sampling(model_ids, arena, uplift)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "artifact_role": "offline_roster_study_design_and_sampling_alternative_only",
        "audit_date": "2026-08-09",
        "source_commitments": [
            _source_commitment(SOURCE_SPECS[key])
            for key in (
                "current_14_roster",
                "qwen_projection",
                "cohere_manifest",
                "cohere_catalog",
                "cohere_smoke",
                "current_14_design",
                "current_14_sampling",
                "inference_acceptance",
            )
        ],
        "source_selection": {
            "cohere": {
                "audited_local_manifest_sequence": "v34 through v44",
                "selected_manifest_version": 44,
                "selected_semantic_sha256": SOURCE_SPECS["cohere_manifest"].semantic_sha256,
                "selected_physical_sha256": SOURCE_SPECS["cohere_manifest"].physical_sha256,
                "transitive_catalog_semantic_sha256": SOURCE_SPECS[
                    "cohere_catalog"
                ].semantic_sha256,
                "transitive_command_a_plus_smoke_semantic_sha256": SOURCE_SPECS[
                    "cohere_smoke"
                ].semantic_sha256,
                "selection_basis": (
                    "highest-numbered local direct-Cohere manifest in the audited sequence, "
                    "then independent physical, semantic, catalog-entry, route, and smoke checks"
                ),
                "downstream_quality_outputs_used": False,
            }
        },
        "non_activation": {
            "current_14_roster_or_schedule_modified": False,
            "current_14_design_semantic_sha256": SOURCE_SPECS[
                "current_14_design"
            ].semantic_sha256,
            "supersedes_current_design": False,
            "activates_candidate": False,
            "runtime_or_importer_change": False,
            "paper_or_result_change": False,
        },
        "prospective_task_bank_if_120_tasks_are_admitted": {
            "source_class": "licensed_public_human_written_questions",
            "total": 120,
            "families": {family: 30 for family in FAMILIES},
            "splits": {"scored": 80, "development": 20, "rotation_reserve": 20},
            "split_per_family": {"scored": 20, "development": 5, "rotation_reserve": 5},
            "development_disjoint_from_scored": True,
            "rotation_reserve_is_not_a_private_holdout": True,
            "task_ids_assigned_in_this_artifact": 0,
            "task_content_created_in_this_artifact": 0,
            "authorized": False,
        },
        "candidate_model_panel": {
            "definition": "exact_current_14_plus_qwen_3_8_max_plus_cohere_command_a_plus",
            "model_count": 16,
            "models": panel_rows,
            "selection_uses_quality_observations": False,
            "quality_observations_used": 0,
            "official": False,
            "rank_eligible": False,
            "calls_authorized": False,
            "quality_eligible": False,
            "frozen_for_confirmatory_collection": False,
            "membership_change_requires_full_rebuild": True,
        },
        "two_lane_assessment": {
            "immutable_confirmatory_roster": {
                "status": "blocked_roster_not_yet_defined",
                "purpose": (
                    "reproducible confirmatory inference under immutable endpoint identities"
                ),
                "qwen_3_8_max_eligible_now": False,
                "qwen_reason": (
                    "qwen3.8-max is a mutable alias observed on 2026-08-08; no continuity or "
                    "immutable served-revision claim is available"
                ),
                "kimi_k3_eligible_as_immutable_now": False,
                "kimi_reason": (
                    "the direct request is provider-managed k3; routed-v30 binds a dated catalog "
                    "snapshot and response evidence but does not prove immutable direct serving"
                ),
                "cohere_command_a_plus_identity_stronger": True,
                "cohere_identity_reason": (
                    "the requested provider ID command-a-plus-05-2026 is dated and bound to an "
                    "authenticated catalog entry and smoke, although season and cost gates remain"
                ),
                "official_model_count": None,
                "this_k16_content_address_reusable": False,
                "official": False,
                "rank_eligible": False,
            },
            "timestamped_current_frontier_observational_extension": {
                "status": "feasible_but_blocked",
                "purpose": (
                    "time-stamped operational observation of the exact 16 named routes without "
                    "reproducibility, confirmatory, leaderboard, or official claims"
                ),
                "qwen_3_8_max_may_appear": True,
                "qwen_alias_commitment": {
                    "identity_kind": "mutable_alias",
                    "catalog_observed_at": "2026-08-08T16:29:04Z",
                    "projection_recorded_at": "2026-08-08T21:24:15.897913Z",
                    "continuity_or_version_stability_inferred": False,
                    "future_reuse_requires_new_catalog_and_run_date_binding": True,
                },
                "k16_balanced_schedule_reusable_if_all_16_routes_are_retained": True,
                "reuse_scope": (
                    "the abstract K16 observational frame only, after a new pre-output source "
                    "freeze and content address; it is not an immutable confirmatory schedule"
                ),
                "official": False,
                "rank_eligible": False,
            },
            "missing_official_slot_consequence": {
                "example": "Qwen remains provisional and no immutable replacement is admitted",
                "resulting_model_count": 15,
                "existing_k16_schedule_reusable": False,
                "factorization_breaks": True,
                "mathematical_reason": (
                    "K15 has no perfect one-factorization, and a five-regular undirected graph "
                    "on 15 vertices is impossible because its degree sum would be odd"
                ),
                "estimands_unchanged": False,
                "why_estimands_change": (
                    "dropping or replacing a model changes the comparison population, pair set, "
                    "exposure weights, human frame, cost, power, and missingness assumptions"
                ),
                "required_action": (
                    "define the exact official membership and rebuild the schedule, sampling "
                    "frame, estimands, power, and cost under a new content address; an explicitly "
                    "modeled bye design would also be a new design"
                ),
            },
        },
        "validation_matrix": _validation_matrix(kimi_entry, qwen, cohere_entry),
        "study_design": {
            "model_arena": arena,
            "epicure_uplift": uplift,
            "generation_reliability_panel": {
                "split": "development",
                "task_slots": 20,
                "tasks_per_family": 5,
                "endpoint_count": 16,
                "conditions": list(CONDITIONS),
                "independent_generations_per_cell": 3,
                "retries_are_not_repetitions": True,
                "disjoint_from_scored_primary": True,
                "total_unique_response_arms": 1_920,
            },
            "prompt_sensitivity_audit": {
                "split": "development",
                "task_slots": 20,
                "tasks_per_family": 5,
                "endpoint_count": 8,
                "endpoint_selection": (
                    "eight endpoints predeclared from provider-and-model-family strata after an "
                    "official roster freeze and before any response or quality outcome"
                ),
                "endpoint_ids_assigned_in_this_artifact": 0,
                "condition": "epicure_off",
                "noncanonical_prompt_variants": 3,
                "canonical_baseline_source": "generation_reliability_panel",
                "total_new_unique_response_arms": 480,
                "development_only_nonranking": True,
            },
            "fixed_cell_count_without_effect_based_stopping": True,
            "power_validated": False,
        },
        "arithmetic": {
            "arena_battles": 3_200,
            "arena_response_arms": 6_400,
            "uplift_pairs": 3_200,
            "uplift_response_arms": 6_400,
            "primary_response_arms": 12_800,
            "disjoint_development_reliability_response_arms": 1_920,
            "prompt_sensitivity_new_response_arms": 480,
            "total_planned_unique_real_response_arms": 15_200,
            "power_or_precision_conclusion": "none_arithmetic_feasibility_only",
        },
        "human_sampling_frame": sampling,
        "inference_acceptance_mismatch": _inference_acceptance_mismatch(
            inference_acceptance, sampling
        ),
        "comparison_with_current_14_model_design": {
            "comparison_basis": "arithmetic_and_membership_only_not_quality",
            "quality_based_choice_made": False,
            "selected_design": None,
            "current_14": {
                "model_count": 14,
                "total_unique_real_response_arms": 13_300,
                "arena_battles": 2_800,
                "arena_response_arms": 5_600,
                "arena_appearances_per_model": 400,
                "arena_pair_repetition_range": [30, 31],
                "uplift_pairs": 2_800,
                "uplift_response_arms": 5_600,
                "uplift_pairs_per_model": 200,
                "uplift_pairs_per_model_family": 50,
                "reliability_response_arms": 1_680,
                "prompt_sensitivity_response_arms": 420,
                "prompt_sensitivity_endpoint_count": 7,
                "human_arena_comparisons": 800,
                "human_uplift_comparisons": 800,
                "human_primary_judgments": 3_200,
                "human_repeat_presentations": 400,
                "sampling_applicable_to_16_models": False,
            },
            "alternative_16": {
                "model_count": 16,
                "total_unique_real_response_arms": 15_200,
                "arena_battles": 3_200,
                "arena_response_arms": 6_400,
                "arena_appearances_per_model": 400,
                "arena_pair_repetition_range": [26, 27],
                "uplift_pairs": 3_200,
                "uplift_response_arms": 6_400,
                "uplift_pairs_per_model": 200,
                "uplift_pairs_per_model_family": 50,
                "reliability_response_arms": 1_920,
                "prompt_sensitivity_response_arms": 480,
                "prompt_sensitivity_endpoint_count": 8,
                "human_arena_comparisons": 800,
                "human_uplift_comparisons": 800,
                "human_primary_judgments": 3_200,
                "human_repeat_presentations": 400,
                "human_arena_appearances_per_model": 100,
                "human_uplift_appearances_per_model": 50,
            },
            "delta_16_minus_14": {
                "models": 2,
                "arena_battles": 400,
                "arena_response_arms": 800,
                "uplift_pairs": 400,
                "uplift_response_arms": 800,
                "reliability_response_arms": 240,
                "prompt_sensitivity_response_arms": 60,
                "total_unique_real_response_arms": 1_900,
                "human_primary_judgments": 0,
                "human_repeat_presentations": 0,
            },
        },
        "remaining_eligibility_requirements": _remaining_requirements(),
        "claim_boundary": {
            "activation_effect": "none",
            "official": False,
            "rank_eligible": False,
            "calls_authorized": False,
            "model_calls_authorized": False,
            "epicure_calls_authorized": False,
            "human_contact_authorized": False,
            "human_judgment_collection_authorized": False,
            "compensation_or_spend_authorized": False,
            "quality_eligible": False,
            "quality_observations_used": 0,
            "generation_calls_made_by_this_artifact": 0,
            "epicure_calls_made_by_this_artifact": 0,
            "human_judgments_made_by_this_artifact": 0,
            "tasks_admitted_by_this_artifact": 0,
            "research_result": False,
            "paper_or_public_claim_authorized": False,
        },
    }


def _load_sources(
    *,
    current_14_roster_path: Path,
    qwen_projection_path: Path,
    cohere_manifest_path: Path,
    cohere_catalog_path: Path,
    cohere_smoke_path: Path,
    current_14_design_path: Path,
    current_14_sampling_path: Path,
    inference_acceptance_path: Path,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _read_source(path, SOURCE_SPECS[key])
        for key, path in (
            ("current_14_roster", current_14_roster_path),
            ("qwen_projection", qwen_projection_path),
            ("cohere_manifest", cohere_manifest_path),
            ("cohere_catalog", cohere_catalog_path),
            ("cohere_smoke", cohere_smoke_path),
            ("current_14_design", current_14_design_path),
            ("current_14_sampling", current_14_sampling_path),
            ("inference_acceptance", inference_acceptance_path),
        )
    )


def build_alternative_candidate(
    *,
    current_14_roster_path: Path = DEFAULT_CURRENT_14_ROSTER,
    qwen_projection_path: Path = DEFAULT_QWEN_PROJECTION,
    cohere_manifest_path: Path = DEFAULT_COHERE_MANIFEST,
    cohere_catalog_path: Path = DEFAULT_COHERE_CATALOG,
    cohere_smoke_path: Path = DEFAULT_COHERE_SMOKE,
    current_14_design_path: Path = DEFAULT_CURRENT_14_DESIGN,
    current_14_sampling_path: Path = DEFAULT_CURRENT_14_SAMPLING,
    inference_acceptance_path: Path = DEFAULT_INFERENCE_ACCEPTANCE,
) -> dict[str, Any]:
    """Return the deterministic candidate after verifying every exact local source."""

    sources = _load_sources(
        current_14_roster_path=current_14_roster_path,
        qwen_projection_path=qwen_projection_path,
        cohere_manifest_path=cohere_manifest_path,
        cohere_catalog_path=cohere_catalog_path,
        cohere_smoke_path=cohere_smoke_path,
        current_14_design_path=current_14_design_path,
        current_14_sampling_path=current_14_sampling_path,
        inference_acceptance_path=inference_acceptance_path,
    )
    body = _build_body(*sources)
    document = {**body, "artifact_sha256": sha256_json(body)}
    _verify_document_body(document, body)
    return document


def _verify_document_body(document: Mapping[str, Any], expected_body: Mapping[str, Any]) -> None:
    recorded = document.get("artifact_sha256")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(isinstance(recorded, str) and recorded == sha256_json(body), "artifact digest")
    _require(body == expected_body, "candidate differs from exact source-derived design")
    _require(document.get("schema_version") == SCHEMA_VERSION, "candidate schema")
    _require(document.get("status") == STATUS, "candidate status")


def verify_alternative_candidate(
    document: Mapping[str, Any],
    *,
    current_14_roster_path: Path = DEFAULT_CURRENT_14_ROSTER,
    qwen_projection_path: Path = DEFAULT_QWEN_PROJECTION,
    cohere_manifest_path: Path = DEFAULT_COHERE_MANIFEST,
    cohere_catalog_path: Path = DEFAULT_COHERE_CATALOG,
    cohere_smoke_path: Path = DEFAULT_COHERE_SMOKE,
    current_14_design_path: Path = DEFAULT_CURRENT_14_DESIGN,
    current_14_sampling_path: Path = DEFAULT_CURRENT_14_SAMPLING,
    inference_acceptance_path: Path = DEFAULT_INFERENCE_ACCEPTANCE,
) -> None:
    """Fail unless the candidate exactly matches all bound evidence and arithmetic."""

    sources = _load_sources(
        current_14_roster_path=current_14_roster_path,
        qwen_projection_path=qwen_projection_path,
        cohere_manifest_path=cohere_manifest_path,
        cohere_catalog_path=cohere_catalog_path,
        cohere_smoke_path=cohere_smoke_path,
        current_14_design_path=current_14_design_path,
        current_14_sampling_path=current_14_sampling_path,
        inference_acceptance_path=inference_acceptance_path,
    )
    _verify_document_body(document, _build_body(*sources))


def materialize_human_sampling_frame(
    document: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Materialize the exact abstract comparison, judgment, and repeat coordinates."""

    verify_alternative_candidate(document)
    model_ids = [row["model_id"] for row in document["candidate_model_panel"]["models"]]
    design = document["study_design"]
    summary, frame = _materialize_sampling(
        model_ids, design["model_arena"], design["epicure_uplift"]
    )
    _require(summary == document["human_sampling_frame"], "sampling summary mismatch")
    return frame


def write_alternative_candidate(document: Mapping[str, Any], output_dir: Path) -> Path:
    """Atomically write one content-addressed blocked candidate."""

    verify_alternative_candidate(document)
    _require(not output_dir.is_symlink(), "output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(output_dir.is_dir() and not output_dir.is_symlink(), "invalid output directory")
    digest = str(document["artifact_sha256"])
    destination = output_dir / f"study-design-16-model-alternative-v1-candidate-{digest}.json"
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        _require(
            not destination.is_symlink() and destination.read_text(encoding="utf-8") == payload,
            "existing content-addressed candidate conflict",
        )
        return destination
    descriptor, temporary_name = tempfile.mkstemp(prefix=".study-design-16-", dir=output_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-14-roster", type=Path, default=DEFAULT_CURRENT_14_ROSTER)
    parser.add_argument("--qwen-projection", type=Path, default=DEFAULT_QWEN_PROJECTION)
    parser.add_argument("--cohere-manifest", type=Path, default=DEFAULT_COHERE_MANIFEST)
    parser.add_argument("--cohere-catalog", type=Path, default=DEFAULT_COHERE_CATALOG)
    parser.add_argument("--cohere-smoke", type=Path, default=DEFAULT_COHERE_SMOKE)
    parser.add_argument("--current-14-design", type=Path, default=DEFAULT_CURRENT_14_DESIGN)
    parser.add_argument("--current-14-sampling", type=Path, default=DEFAULT_CURRENT_14_SAMPLING)
    parser.add_argument(
        "--inference-acceptance", type=Path, default=DEFAULT_INFERENCE_ACCEPTANCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-candidate", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_alternative_candidate(
        current_14_roster_path=args.current_14_roster,
        qwen_projection_path=args.qwen_projection,
        cohere_manifest_path=args.cohere_manifest,
        cohere_catalog_path=args.cohere_catalog,
        cohere_smoke_path=args.cohere_smoke,
        current_14_design_path=args.current_14_design,
        current_14_sampling_path=args.current_14_sampling,
        inference_acceptance_path=args.inference_acceptance,
    )
    if args.write_candidate:
        print(write_alternative_candidate(document, args.output_dir))
    else:
        print(document["artifact_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
