"""Build an offline-only successor design for the public-source task campaign.

The builder reconciles immutable, already-recorded evidence.  It has no provider,
database, reviewer, consent, or deployment integration.  Its output is deliberately
blocked: it is an arithmetic design candidate, not an active study contract and not
authorization to admit tasks, contact people, generate responses, judge outputs, or
publish results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frontier_manifest import verify_manifest_content_address
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-season1-study-design-v6-candidate"
STATUS = "blocked_offline_successor_not_authorized"

FLAVOURBENCH_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FLAVOURBENCH_ROOT.parent

DEFAULT_CAMPAIGN = FLAVOURBENCH_ROOT / (
    "artifacts/season1/task-validation-campaign-v6/"
    "campaign-76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709.json"
)
DEFAULT_READINESS = FLAVOURBENCH_ROOT / (
    "artifacts/season1/task-validation-campaign-v6/"
    "readiness-449df377dd8de515a46a80d36dffc80f1734f3e86a60a53651675fb75c9d82c0.json"
)
DEFAULT_CONTAMINATION = FLAVOURBENCH_ROOT / (
    "artifacts/season1/task-validation-campaign-v6/contamination-replay-v2/"
    "contamination-replay-v2-"
    "2c7e2ead2e4e936e840d6b0fbc9bbf268c3237e908da903982a00a1af0f0b44d.json"
)
DEFAULT_LEGACY_DESIGN = FLAVOURBENCH_ROOT / (
    "contracts/season1/season1-study-design-v5.json"
)
DEFAULT_ROSTER = REPOSITORY_ROOT / "paper/flavourbench/provenance/routed-v30-manifest.json"
DEFAULT_QWEN_PROJECTION = REPOSITORY_ROOT / (
    "paper/flavourbench/provenance/qwencloud-exploratory-operational-projection.json"
)
DEFAULT_OUTPUT_DIR = FLAVOURBENCH_ROOT / "artifacts/season1/study-design-v6-candidate"

FAMILIES = ("substitution", "composition", "cookability", "evidence")
CONDITIONS = ("epicure_off", "epicure_on")

EXPECTED_MODELS: tuple[tuple[str, str], ...] = (
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


class SuccessorDesignError(RuntimeError):
    """The evidence or offline successor arithmetic failed closed."""


@dataclass(frozen=True)
class SourceSpec:
    role: str
    reference_path: str
    schema_version: str
    semantic_sha256: str
    physical_sha256: str
    address_kind: str = "artifact_sha256"


SOURCE_SPECS = {
    "campaign": SourceSpec(
        role="public_source_validation_campaign",
        reference_path=(
            "flavourbench/artifacts/season1/task-validation-campaign-v6/"
            "campaign-76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709.json"
        ),
        schema_version="flavourbench-public-source-task-validation-campaign-v1",
        semantic_sha256="76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709",
        physical_sha256="b514203aa1924a8661a7d393ec519071c0fff85ef6dfda894fcb6065cda27c69",
    ),
    "readiness": SourceSpec(
        role="campaign_technical_readiness",
        reference_path=(
            "flavourbench/artifacts/season1/task-validation-campaign-v6/"
            "readiness-449df377dd8de515a46a80d36dffc80f1734f3e86a60a53651675fb75c9d82c0.json"
        ),
        schema_version="flavourbench-public-source-task-campaign-readiness-v1",
        semantic_sha256="449df377dd8de515a46a80d36dffc80f1734f3e86a60a53651675fb75c9d82c0",
        physical_sha256="c64494828770dde9c9655d3531a44dea31b6454535191b1ef1ef672bc4382632",
    ),
    "contamination": SourceSpec(
        role="campaign_contamination_replay",
        reference_path=(
            "flavourbench/artifacts/season1/task-validation-campaign-v6/"
            "contamination-replay-v2/contamination-replay-v2-"
            "2c7e2ead2e4e936e840d6b0fbc9bbf268c3237e908da903982a00a1af0f0b44d.json"
        ),
        schema_version="flavourbench-task-validation-contamination-replay-v2",
        semantic_sha256="2c7e2ead2e4e936e840d6b0fbc9bbf268c3237e908da903982a00a1af0f0b44d",
        physical_sha256="9e520aa4f2384779efafb930d20cebe8f72b728197538adb43405943674abdb0",
    ),
    "legacy_design": SourceSpec(
        role="legacy_240_task_16_model_design",
        reference_path="flavourbench/contracts/season1/season1-study-design-v5.json",
        schema_version="flavourbench-season1-study-design-v5",
        semantic_sha256="7a63cfd6117338a3af16a422d5ee3458298fdc0ff2fd0abfe45fe851a7e54506",
        physical_sha256="57080b61171ad81d2d0d40307939ad9681db9a37452262b1baec4613d2b477cd",
    ),
    "roster": SourceSpec(
        role="current_14_model_routed_development_candidate",
        reference_path="paper/flavourbench/provenance/routed-v30-manifest.json",
        schema_version="flavourbench-routed-candidate-manifest-v1",
        semantic_sha256="e87a164d59bdd88eaf630c153755b5ec3c513e8b3770a17afac67037eb135910",
        physical_sha256="ac745d0ef19fbc9e17d38d4bd254e7688eee06a3280dbf9a3bbe032361b9d803",
        address_kind="content_address",
    ),
    "qwen_projection": SourceSpec(
        role="qwen_3_8_max_post_freeze_operational_projection",
        reference_path=(
            "paper/flavourbench/provenance/"
            "qwencloud-exploratory-operational-projection.json"
        ),
        schema_version="flavourbench-qwencloud-exploratory-operational-projection-v1",
        semantic_sha256="b2f7790b3eb18d1df083397ce02b5296c549e5ed3ddb3d3f32ea776db3ddca04",
        physical_sha256="9343a2959d3acf3079fb91b2bd7ff608af421532b826f7b98917c88b76a7f85c",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorDesignError(message)


def _physical_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SuccessorDesignError(f"source is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_source(path: Path, spec: SourceSpec) -> dict[str, Any]:
    _require(
        _physical_sha256(path) == spec.physical_sha256,
        f"{spec.role} physical digest mismatch",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SuccessorDesignError(f"invalid {spec.role} JSON") from error
    _require(isinstance(value, dict), f"{spec.role} must be a JSON object")
    _require(
        value.get("schema_version") == spec.schema_version,
        f"{spec.role} schema mismatch",
    )
    if spec.address_kind == "content_address":
        address = value.get("content_address")
        _require(
            isinstance(address, Mapping)
            and address.get("digest") == spec.semantic_sha256
            and verify_manifest_content_address(value),
            f"{spec.role} semantic address mismatch",
        )
    else:
        recorded = value.get("artifact_sha256")
        payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
        _require(
            recorded == spec.semantic_sha256 and sha256_json(payload) == spec.semantic_sha256,
            f"{spec.role} semantic address mismatch",
        )
    return value


def _verify_campaign(campaign: Mapping[str, Any]) -> None:
    target = campaign.get("target")
    observations = campaign.get("observations")
    boundary = campaign.get("claim_boundary")
    schedule = campaign.get("candidate_schedule")
    _require(campaign.get("status") == "frozen_instrument_no_human_ballots", "campaign status")
    _require(isinstance(target, Mapping), "campaign target is missing")
    _require(
        dict(target)
        == {
            "activation": (
                "fixed schedule order; stop only when human-confirmed quotas reach 30 in each "
                "family or the 180-candidate slate is exhausted"
            ),
            "candidate_slate": 180,
            "candidate_slate_per_scheduling_family": 45,
            "effect_based_or_model_result_based_stopping": False,
            "reserve_candidates": 60,
            "validated_tasks": 120,
            "validated_tasks_per_family": 30,
        },
        "campaign target differs from the frozen 180/120 contract",
    )
    _require(
        observations
        == {
            "adjudications": 0,
            "batch_audits": 0,
            "epicure_calls": 0,
            "human_ballots": 0,
            "model_calls": 0,
            "synthetic_tasks": 0,
        },
        "campaign contains observations or calls",
    )
    _require(isinstance(schedule, list) and len(schedule) == 180, "campaign slate size")
    candidate_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    for row in schedule:
        _require(isinstance(row, Mapping), "campaign candidate row is malformed")
        candidate_ids.add(str(row.get("candidate_id") or ""))
        family_counts[str(row.get("scheduling_family") or "")] += 1
        _require(row.get("rank_eligible") is False, "campaign candidate is rank eligible")
    _require("" not in candidate_ids and len(candidate_ids) == 180, "candidate IDs are not unique")
    _require(family_counts == Counter({family: 45 for family in FAMILIES}), "slate imbalance")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("human_validated_tasks") == 0
        and boundary.get("official_task_bank") is False
        and boundary.get("rank_eligible") is False
        and boundary.get("database_import_authorized") is False
        and boundary.get("contamination_free") is False,
        "campaign claim boundary is not fail closed",
    )


def _verify_readiness(readiness: Mapping[str, Any]) -> None:
    boundary = readiness.get("claim_boundary")
    gates = readiness.get("gates")
    _require(
        readiness.get("assessment_type")
        == "technical_readiness_not_human_release_authorization",
        "readiness authority boundary changed",
    )
    _require(
        readiness.get("overall")
        == "conditional_go_for_no_generation_campaign_after_adapter_review",
        "campaign readiness disposition changed",
    )
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("human_ballots") == 0
        and boundary.get("human_release_authority_exercised") is False
        and boundary.get("model_or_epicure_calls_authorized") is False
        and boundary.get("official") is False
        and boundary.get("rank_eligible") is False,
        "readiness claim boundary changed",
    )
    _require(isinstance(gates, list), "readiness gates are missing")
    decisions = {
        str(row.get("gate")): str(row.get("decision"))
        for row in gates
        if isinstance(row, Mapping)
    }
    _require(decisions.get("live_ballot_collection") == "no_go", "live ballots are not no-go")
    _require(decisions.get("official_task_bank") == "no_go", "official bank is not no-go")
    _require(
        decisions.get("contamination_free_claim") == "permanent_no_go",
        "public-source contamination-free claim is not permanently blocked",
    )
    _require(
        decisions.get("model_generation_and_ranking") == "no_go",
        "generation and ranking are not no-go",
    )


def _verify_contamination(contamination: Mapping[str, Any]) -> None:
    decision = contamination.get("decision")
    calibration = contamination.get("calibration")
    boundary = contamination.get("claim_boundary")
    _require(
        contamination.get("artifact_role") == "captured_contamination_method_no_go_successor",
        "contamination artifact role changed",
    )
    _require(
        isinstance(decision, Mapping)
        and decision.get("disposition") == "no_go"
        and decision.get("full_campaign_contamination_method_requirement_satisfied") is False,
        "contamination method is not no-go",
    )
    _require(
        isinstance(calibration, Mapping)
        and calibration.get("cases_observed") == 0
        and calibration.get("real_labeled_calibration_artifact_observed") is False,
        "contamination calibration evidence changed",
    )
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("contamination_free") is False
        and boundary.get("contamination_limited") is True
        and boundary.get("official_task_bank") is False
        and boundary.get("task_bank_import_authorized") is False
        and boundary.get("model_calls") == 0
        and boundary.get("epicure_calls") == 0,
        "contamination claim boundary changed",
    )


def _verify_legacy_design(design: Mapping[str, Any]) -> None:
    bank = design.get("task_bank")
    panel = design.get("model_panel")
    primary = design.get("primary_controlled_collection")
    robustness = design.get("validity_and_robustness")
    _require(
        design.get("status") == "prospective-design-superseding-v4-before-scored-collection",
        "legacy design status changed",
    )
    _require(isinstance(bank, Mapping), "legacy task bank is missing")
    _require(bank.get("total") == 240, "legacy task total changed")
    _require(
        bank.get("families") == {family: 60 for family in FAMILIES},
        "legacy family allocation changed",
    )
    _require(
        bank.get("splits") == {"development": 40, "private_reserve": 40, "scored": 160},
        "legacy task splits changed",
    )
    _require(bank.get("human_authored") is True and bank.get("hidden_until_release_cutoff") is True,
             "legacy task source assumptions changed")
    _require(isinstance(panel, Mapping) and panel.get("candidate_count") == 16, "legacy panel")
    _require(isinstance(primary, Mapping), "legacy primary schedule is missing")
    arena = primary.get("model_arena")
    uplift = primary.get("epicure_uplift")
    _require(
        isinstance(arena, Mapping)
        and arena.get("total_battles") == 3200
        and arena.get("endpoint_appearances") == 6400
        and arena.get("minimum_endpoint_appearances_per_model") == 400,
        "legacy arena schedule changed",
    )
    _require(
        isinstance(uplift, Mapping)
        and uplift.get("total_pairs") == 3200
        and uplift.get("minimum_pairs_per_model") == 200
        and uplift.get("minimum_pairs_per_model_family") == 50,
        "legacy uplift schedule changed",
    )
    _require(primary.get("total_model_response_arms") == 12800, "legacy primary arms changed")
    _require(isinstance(robustness, Mapping), "legacy robustness schedule is missing")
    reliability = robustness.get("generation_reliability_panel")
    sensitivity = robustness.get("prompt_sensitivity_audit")
    _require(
        isinstance(reliability, Mapping)
        and reliability.get("endpoint_count") == 16
        and reliability.get("task_count") == 20
        and reliability.get("total_panel_arms") == 1920
        and reliability.get("incremental_arms_beyond_primary") == 1280,
        "legacy reliability schedule changed",
    )
    _require(
        isinstance(sensitivity, Mapping)
        and sensitivity.get("endpoint_count") == 8
        and sensitivity.get("task_count") == 20
        and sensitivity.get("prompt_variants") == 3
        and sensitivity.get("total_response_arms") == 480,
        "legacy prompt-sensitivity schedule changed",
    )
    _require(
        robustness.get("total_planned_real_model_response_arms_including_robustness") == 14560,
        "legacy 14,560-arm total changed",
    )


def _model_rows(roster: Mapping[str, Any]) -> list[dict[str, str]]:
    models = roster.get("models")
    _require(isinstance(models, list), "candidate roster has no model list")
    rows: list[dict[str, str]] = []
    for entry in models:
        _require(isinstance(entry, Mapping), "candidate roster entry is malformed")
        model = entry.get("model")
        _require(isinstance(model, Mapping), "candidate roster entry has no model identity")
        rows.append(
            {
                "model_id": str(model.get("id") or ""),
                "canonical_model_slug": str(model.get("canonical_slug") or ""),
            }
        )
    return rows


def _verify_roster(roster: Mapping[str, Any]) -> list[dict[str, str]]:
    selection = roster.get("selection")
    governance = roster.get("governance")
    rows = _model_rows(roster)
    _require(roster.get("status") == "unranked_candidate", "roster is not an unranked candidate")
    _require(roster.get("official_results_authorised") is False, "roster authorizes results")
    _require(roster.get("generation_calls_made") == 0, "roster records generation calls")
    _require(roster.get("generation_spend_usd") == "0", "roster records generation spend")
    _require(isinstance(selection, Mapping), "roster selection is missing")
    _require(
        selection.get("model_count") == 14
        and selection.get("quality_observations_used") == 0
        and selection.get("performance_claim") == "none; inclusion is coverage, not a ranking",
        "roster selection boundary changed",
    )
    observed = tuple((row["model_id"], row["canonical_model_slug"]) for row in rows)
    _require(observed == EXPECTED_MODELS, "the exact 14-model roster changed")
    excluded = selection.get("excluded_lanes")
    _require(isinstance(excluded, list), "roster exclusions are missing")
    qwen37 = [
        row
        for row in excluded
        if isinstance(row, Mapping) and row.get("model_id") == "qwen/qwen3.7-max"
    ]
    _require(
        len(qwen37) == 1
        and qwen37[0].get("contract_status") == "failed_pre_generation"
        and qwen37[0].get("reason") == "exact_route_contract_failed",
        "Qwen 3.7 exclusion evidence changed",
    )
    _require(
        isinstance(governance, Mapping)
        and governance.get("official") is False
        and governance.get("rank_eligible") is False,
        "roster governance boundary changed",
    )
    return rows


def _verify_qwen_projection(projection: Mapping[str, Any]) -> None:
    identity = projection.get("model_identity")
    boundary = projection.get("claim_boundary")
    _require(
        projection.get("status") == "verified_exploratory_unranked_post_freeze_addendum",
        "Qwen projection status changed",
    )
    _require(
        isinstance(identity, Mapping)
        and identity.get("requested_model_id") == "qwen3.8-max"
        and identity.get("identity_kind") == "mutable_alias"
        and identity.get("frozen_release") is False
        and identity.get("provider") == "qwencloud-direct"
        and identity.get("automatic_provider_fallback") is False
        and identity.get("returned_model_ids") == ["qwen3.8-max"],
        "Qwen mutable-alias evidence changed",
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
        "Qwen claim boundary changed",
    )


def round_robin_matchings(model_ids: Sequence[str]) -> list[list[tuple[str, str]]]:
    """Return a one-factorization for an even number of unique model IDs."""

    players = list(model_ids)
    _require(len(players) >= 2 and len(players) % 2 == 0, "model count must be positive and even")
    _require(len(set(players)) == len(players) and "" not in players, "model IDs must be unique")
    matchings: list[list[tuple[str, str]]] = []
    for _ in range(len(players) - 1):
        pairs = [
            tuple(sorted((players[index], players[-1 - index])))
            for index in range(len(players) // 2)
        ]
        matchings.append(pairs)
        players = [players[0], players[-1], *players[1:-1]]
    observed_pairs = [pair for matching in matchings for pair in matching]
    expected_pairs = len(model_ids) * (len(model_ids) - 1) // 2
    _require(
        len(observed_pairs) == expected_pairs and len(set(observed_pairs)) == expected_pairs,
        "round-robin factorization does not cover every pair exactly once",
    )
    return matchings


def build_arena_schedule(model_ids: Sequence[str]) -> dict[str, Any]:
    """Build an abstract 80-slot, five-regular arena schedule."""

    matchings = round_robin_matchings(model_ids)
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
            slot_pairs: list[tuple[str, str]] = []
            for matching_index in matching_indices:
                matching_use[matching_index] += 1
                for pair in matchings[matching_index]:
                    pair_use[pair] += 1
                    model_use.update(pair)
                    slot_pairs.append(pair)
            _require(len(slot_pairs) == 35 and len(set(slot_pairs)) == 35, "arena slot imbalance")
            slot_degree: Counter[str] = Counter(model for pair in slot_pairs for model in pair)
            _require(set(slot_degree.values()) == {5}, "arena slot is not five-regular")
            slots.append(
                {
                    "design_slot_ordinal": ordinal,
                    "family": family,
                    "family_slot_ordinal": family_ordinal,
                    "matching_indices_zero_based": matching_indices,
                }
            )
    _require(set(model_use.values()) == {400}, "arena model appearance floor is not exact")
    _require(len(pair_use) == 91 and set(pair_use.values()) == {30, 31}, "arena pair balance")
    _require(set(matching_use.values()) == {30, 31}, "arena matching balance")
    return {
        "task_binding": (
            "abstract scored-slot ordinals only; bind 80 admitted task IDs after the split freeze; "
            "these rows are not task records"
        ),
        "task_slots": 80,
        "tasks_per_family": 20,
        "comparisons_per_task": 35,
        "endpoint_degree_per_task": 5,
        "total_battles": 2800,
        "total_response_arms": 5600,
        "endpoint_appearances_per_model": 400,
        "distinct_endpoint_pairs": 91,
        "pair_repetition_minimum": min(pair_use.values()),
        "pair_repetition_maximum": max(pair_use.values()),
        "factorization": {
            "matching_count": len(matchings),
            "pairs_per_matching": 7,
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
    """Build an abstract 80-slot uplift schedule with exact family balance."""

    models = list(model_ids)
    _require(len(models) == 14 and len(set(models)) == 14, "uplift design requires 14 models")
    totals: Counter[str] = Counter()
    family_totals: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    slots: list[dict[str, Any]] = []
    ordinal = 0
    for family in FAMILIES:
        for family_ordinal in range(1, 21):
            ordinal += 1
            third_group = models[:7] if family_ordinal % 2 else models[7:]
            repetitions = {model: 2 + int(model in third_group) for model in models}
            _require(sum(repetitions.values()) == 35, "uplift slot does not contain 35 pairs")
            _require(set(repetitions.values()) == {2, 3}, "uplift repetitions are not two or three")
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
    _require(set(totals.values()) == {200}, "uplift model totals are not exact")
    _require(
        all(set(counter.values()) == {50} for counter in family_totals.values()),
        "uplift family totals are not exact",
    )
    return {
        "task_binding": (
            "abstract scored-slot ordinals only; bind 80 admitted task IDs after the split freeze; "
            "these rows are not task records"
        ),
        "task_slots": 80,
        "tasks_per_family": 20,
        "conditions": list(CONDITIONS),
        "paired_repetitions_per_task": 35,
        "paired_repetitions_per_model_task": [2, 3],
        "paired_repetitions_per_model_family": 50,
        "paired_repetitions_per_model": 200,
        "total_pairs": 2800,
        "total_response_arms": 5600,
        "abstract_task_schedule": slots,
    }


def _source_commitment(spec: SourceSpec) -> dict[str, Any]:
    return {
        "role": spec.role,
        "reference_path": spec.reference_path,
        "schema_version": spec.schema_version,
        "semantic_sha256": spec.semantic_sha256,
        "physical_sha256": spec.physical_sha256,
    }


def _address(document: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return {**body, "artifact_sha256": sha256_json(body)}


def verify_successor_design(document: Mapping[str, Any]) -> None:
    recorded = document.get("artifact_sha256")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(
        isinstance(recorded, str) and recorded == sha256_json(body),
        "artifact digest mismatch",
    )
    _require(document.get("schema_version") == SCHEMA_VERSION, "successor schema mismatch")
    _require(document.get("status") == STATUS, "successor is not blocked")
    _require(
        document.get("source_commitments")
        == [
            _source_commitment(SOURCE_SPECS[key])
            for key in (
                "campaign",
                "readiness",
                "contamination",
                "legacy_design",
                "roster",
                "qwen_projection",
            )
        ],
        "successor source commitments changed",
    )
    bank = document.get("prospective_task_bank_if_120_tasks_are_admitted")
    _require(
        isinstance(bank, Mapping)
        and bank.get("total") == 120
        and bank.get("families") == {family: 30 for family in FAMILIES}
        and bank.get("splits")
        == {"scored": 80, "development": 20, "rotation_reserve": 20}
        and bank.get("split_per_family")
        == {"scored": 20, "development": 5, "rotation_reserve": 5}
        and bank.get("campaign_attrition_reserve_is_not_bank_split") is True
        and bank.get("rotation_reserve_is_not_a_private_holdout") is True
        and bank.get("task_ids_assigned_in_this_artifact") == 0
        and bank.get("task_content_created_in_this_artifact") == 0,
        "successor task-bank boundary changed",
    )
    panel = document.get("candidate_model_panel")
    expected_rows = [
        {"model_id": model_id, "canonical_model_slug": canonical}
        for model_id, canonical in EXPECTED_MODELS
    ]
    _require(
        isinstance(panel, Mapping)
        and panel.get("evidence_basis_only") is True
        and panel.get("official_roster") is False
        and panel.get("frozen_for_confirmatory_collection") is False
        and panel.get("quality_observations_used") == 0
        and panel.get("model_count") == 14
        and panel.get("models") == expected_rows
        and panel.get("replacement_or_addition_requires_full_rebuild") is True,
        "successor candidate panel changed",
    )
    model_ids = [row["model_id"] for row in expected_rows]
    primary = document.get("primary_schedule")
    _require(
        primary
        == {
            "model_arena": build_arena_schedule(model_ids),
            "epicure_uplift": build_uplift_schedule(model_ids),
            "total_response_arms": 11200,
            "fixed_cell_count_without_effect_based_stopping": True,
            "power_validated": False,
        },
        "successor primary schedule changed",
    )
    robustness = document.get("robustness_schedule")
    _require(isinstance(robustness, Mapping), "successor robustness schedule is missing")
    reliability = robustness.get("generation_reliability_panel")
    sensitivity = robustness.get("prompt_sensitivity_audit")
    _require(
        isinstance(reliability, Mapping)
        and reliability.get("split") == "development"
        and reliability.get("task_slots") == 20
        and reliability.get("tasks_per_family") == 5
        and reliability.get("endpoint_count") == 14
        and reliability.get("conditions") == list(CONDITIONS)
        and reliability.get("independent_generations_per_cell") == 3
        and reliability.get("retries_are_not_repetitions") is True
        and reliability.get("disjoint_from_scored_primary") is True
        and reliability.get("total_response_arms") == 1680,
        "successor reliability schedule changed",
    )
    _require(
        isinstance(sensitivity, Mapping)
        and sensitivity.get("split") == "development"
        and sensitivity.get("task_slots") == 20
        and sensitivity.get("tasks_per_family") == 5
        and sensitivity.get("endpoint_count") == 7
        and sensitivity.get("condition") == "epicure_off"
        and sensitivity.get("noncanonical_prompt_variants") == 3
        and sensitivity.get("canonical_baseline_source") == "generation_reliability_panel"
        and sensitivity.get("total_new_response_arms") == 420
        and sensitivity.get("development_only_nonranking") is True,
        "successor prompt-sensitivity schedule changed",
    )
    boundary = document.get("claim_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("activation_effect") == "none"
        and boundary.get("official") is False
        and boundary.get("rank_eligible") is False
        and boundary.get("task_bank_import_authorized") is False
        and boundary.get("human_contact_authorized") is False
        and boundary.get("model_or_epicure_calls_authorized") is False
        and boundary.get("quality_observations") == 0,
        "successor claim boundary is not fail closed",
    )
    totals = document.get("arithmetic")
    _require(
        isinstance(totals, Mapping)
        and totals.get("primary_response_arms") == 11200
        and totals.get("reliability_response_arms") == 1680
        and totals.get("prompt_sensitivity_response_arms") == 420
        and totals.get("total_planned_unique_real_response_arms") == 13300,
        "successor arm arithmetic changed",
    )
    qwen = document.get("qwen_3_8_max_eligibility")
    _require(
        isinstance(qwen, Mapping)
        and qwen.get("candidate_roster_member") is False
        and qwen.get("confirmatory_eligible") is False
        and qwen.get("identity_kind") == "mutable_alias"
        and qwen.get("silent_replacement_allowed") is False,
        "Qwen eligibility is not fail closed",
    )
    review = document.get("prospective_human_review_floor")
    _require(
        review
        == {
            "unique_comparisons": {"model_arena": 800, "epicure_uplift": 800},
            "independent_raters_per_comparison": 2,
            "primary_judgments": 3200,
            "concealed_repeat_rate": "0.125",
            "additional_repeat_presentations": 400,
            "total_rating_presentations": 3600,
            "authorized": False,
            "reviewers_contacted_by_this_artifact": 0,
        },
        "successor human-review floor changed",
    )


def build_successor_design(
    *,
    campaign_path: Path = DEFAULT_CAMPAIGN,
    readiness_path: Path = DEFAULT_READINESS,
    contamination_path: Path = DEFAULT_CONTAMINATION,
    legacy_design_path: Path = DEFAULT_LEGACY_DESIGN,
    roster_path: Path = DEFAULT_ROSTER,
    qwen_projection_path: Path = DEFAULT_QWEN_PROJECTION,
) -> dict[str, Any]:
    """Verify the six evidence inputs and return one deterministic blocked candidate."""

    campaign = _read_source(campaign_path, SOURCE_SPECS["campaign"])
    readiness = _read_source(readiness_path, SOURCE_SPECS["readiness"])
    contamination = _read_source(contamination_path, SOURCE_SPECS["contamination"])
    legacy = _read_source(legacy_design_path, SOURCE_SPECS["legacy_design"])
    roster = _read_source(roster_path, SOURCE_SPECS["roster"])
    qwen = _read_source(qwen_projection_path, SOURCE_SPECS["qwen_projection"])

    _verify_campaign(campaign)
    _verify_readiness(readiness)
    _verify_contamination(contamination)
    _verify_legacy_design(legacy)
    model_rows = _verify_roster(roster)
    _verify_qwen_projection(qwen)

    model_ids = [row["model_id"] for row in model_rows]
    arena = build_arena_schedule(model_ids)
    uplift = build_uplift_schedule(model_ids)

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "artifact_role": "offline_arithmetic_and_migration_candidate_only",
        "source_commitments": [
            _source_commitment(SOURCE_SPECS[key])
            for key in (
                "campaign",
                "readiness",
                "contamination",
                "legacy_design",
                "roster",
                "qwen_projection",
            )
        ],
        "reconciliation": {
            "legacy_design": {
                "task_source": "newly_human_authored_hidden",
                "task_total": 240,
                "tasks_per_family": 60,
                "splits": {"scored": 160, "development": 40, "private_reserve": 40},
                "model_count": 16,
                "primary_response_arms": 12800,
                "total_planned_real_response_arms": 14560,
                "paper_claim_class": "prospective_design_not_result",
            },
            "public_source_campaign": {
                "task_source": "licensed_public_human_written_questions",
                "candidate_slate": 180,
                "candidate_slate_per_scheduling_family": 45,
                "campaign_attrition_reserve_candidates": 60,
                "target_admitted_tasks": 120,
                "target_admitted_tasks_per_family": 30,
                "currently_admitted_tasks": 0,
                "human_ballots": 0,
                "official_task_bank": False,
                "contamination_claim": "contamination_limited_not_contamination_free",
            },
            "finding": (
                "The 14,560-arm value is internally correct only for the legacy 240-task, "
                "16-model contract. It cannot be carried into the 180-candidate/120-target, "
                "14-model campaign by relabeling. The campaign needs a synchronized successor "
                "contract, importer, power analysis, cost authorization, and claim update."
            ),
            "supersedes_current_contract": False,
            "changes_published_results": False,
        },
        "prospective_task_bank_if_120_tasks_are_admitted": {
            "source_class": "licensed_public_human_written_questions",
            "total": 120,
            "families": {family: 30 for family in FAMILIES},
            "splits": {"scored": 80, "development": 20, "rotation_reserve": 20},
            "split_per_family": {"scored": 20, "development": 5, "rotation_reserve": 5},
            "campaign_attrition_reserve_is_not_bank_split": True,
            "rotation_reserve_is_not_a_private_holdout": True,
            "split_assignment": (
                "deterministic content-addressed assignment after all 120 human admissions and "
                "before any model output; seed and algorithm require independent review"
            ),
            "task_ids_assigned_in_this_artifact": 0,
            "task_content_created_in_this_artifact": 0,
        },
        "candidate_model_panel": {
            "evidence_basis_only": True,
            "official_roster": False,
            "frozen_for_confirmatory_collection": False,
            "quality_observations_used": 0,
            "model_count": 14,
            "models": model_rows,
            "replacement_or_addition_requires_full_rebuild": True,
        },
        "primary_schedule": {
            "model_arena": arena,
            "epicure_uplift": uplift,
            "total_response_arms": 11200,
            "fixed_cell_count_without_effect_based_stopping": True,
            "power_validated": False,
        },
        "robustness_schedule": {
            "generation_reliability_panel": {
                "split": "development",
                "task_slots": 20,
                "tasks_per_family": 5,
                "endpoint_count": 14,
                "conditions": list(CONDITIONS),
                "independent_generations_per_cell": 3,
                "retries_are_not_repetitions": True,
                "disjoint_from_scored_primary": True,
                "total_response_arms": 1680,
            },
            "prompt_sensitivity_audit": {
                "split": "development",
                "task_slots": 20,
                "tasks_per_family": 5,
                "endpoint_count": 7,
                "endpoint_selection": (
                    "predeclared provider-and-model-family strata after the official roster "
                    "freeze and before any response or quality outcome"
                ),
                "condition": "epicure_off",
                "noncanonical_prompt_variants": 3,
                "canonical_baseline_source": "generation_reliability_panel",
                "total_new_response_arms": 420,
                "development_only_nonranking": True,
            },
        },
        "arithmetic": {
            "arena_battles": 2800,
            "arena_response_arms": 5600,
            "uplift_pairs": 2800,
            "uplift_response_arms": 5600,
            "primary_response_arms": 11200,
            "reliability_response_arms": 1680,
            "prompt_sensitivity_response_arms": 420,
            "total_planned_unique_real_response_arms": 13300,
            "power_or_precision_conclusion": "none_arithmetic_feasibility_only",
        },
        "prospective_human_review_floor": {
            "unique_comparisons": {"model_arena": 800, "epicure_uplift": 800},
            "independent_raters_per_comparison": 2,
            "primary_judgments": 3200,
            "concealed_repeat_rate": "0.125",
            "additional_repeat_presentations": 400,
            "total_rating_presentations": 3600,
            "authorized": False,
            "reviewers_contacted_by_this_artifact": 0,
        },
        "qwen_3_8_max_eligibility": {
            "requested_model_id": "qwen3.8-max",
            "provider": "qwencloud-direct",
            "identity_kind": "mutable_alias",
            "frozen_release": False,
            "post_freeze_operational_addendum_only": True,
            "candidate_roster_member": False,
            "confirmatory_eligible": False,
            "silent_replacement_allowed": False,
            "does_not_replace": [
                "the failed qwen/qwen3.7-max route",
                "any member of the current 14-model development roster",
            ],
            "required_before_future_roster_consideration": [
                "provider-issued immutable dated or versioned canonical model identity",
                "exact requested and returned identity binding with fallback disabled",
                "frozen endpoint and production contract pass",
                "approved data policy and complete price/cost authority",
                "pre-quality roster selection and a newly content-addressed schedule",
            ],
        },
        "blockers": [
            "the campaign has zero human ballots and zero admitted tasks",
            "the live ballot adapter and disagreement-only adjudication path are not approved",
            "the contamination replay is NO-GO with zero labeled calibration cases",
            "public-source tasks cannot support a contamination-free or private-holdout claim",
            "campaign-level rights review and exact task admission are incomplete",
            "the 80/20/20 split algorithm and seed are not independently reviewed or frozen",
            "the 14-model manifest is an unranked development candidate, not an official roster",
            "Qwen 3.8 Max is a mutable post-freeze alias and is not confirmatory eligible",
            "power, type-I error, coverage, missingness, and estimand validation are incomplete",
            "provider routes, production contracts, pricing, and funded model-call "
            "authority are absent",
            "human-study ethics/equivalent determination, monitored contact, funded compensation, "
            "participant acceptance, withdrawal, retention, identity, and COI gates remain "
            "required",
            "runtime importer, readiness logic, release gates, paper, and public status are "
            "still bound to the legacy design and must be changed together only after approval",
        ],
        "minimum_activation_patch_plan": [
            "obtain and freeze 120 genuine admissions with 30 human-confirmed tasks per family",
            "close the contamination and rights gates without claiming public tasks are private",
            "independently review and freeze the 80/20/20 split before model outputs",
            "select an exact versioned official roster without quality outcomes; rebuild if "
            "membership changes",
            "run and approve power, uncertainty, missingness, and cost-envelope validation for "
            "13,300 arms",
            "satisfy the separate human-study activation manifest and compensation/ethics "
            "operations",
            "implement a new importer and readiness/release contract behind fail-closed tests",
            "update manuscript, claim map, release gates, and public status in one reviewed "
            "evidence-bound change",
        ],
        "claim_boundary": {
            "activation_effect": "none",
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "paper_or_public_claim_authorized": False,
            "task_bank_import_authorized": False,
            "human_contact_authorized": False,
            "compensation_or_spend_authorized": False,
            "model_or_epicure_calls_authorized": False,
            "human_judgments": 0,
            "quality_observations": 0,
            "tasks_created": 0,
            "tasks_admitted": 0,
        },
    }
    addressed = _address(document)
    verify_successor_design(addressed)
    return addressed


def write_successor_design(document: Mapping[str, Any], output_dir: Path) -> Path:
    """Atomically write one content-addressed candidate after fail-closed verification."""

    verify_successor_design(document)
    _require(not output_dir.is_symlink(), "output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(output_dir.is_dir() and not output_dir.is_symlink(), "invalid output directory")
    digest = str(document["artifact_sha256"])
    destination = output_dir / f"study-design-v6-candidate-{digest}.json"
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if destination.exists():
        _require(
            not destination.is_symlink() and destination.read_text(encoding="utf-8") == payload,
            "existing candidate does not match deterministic output",
        )
        return destination
    descriptor, temporary_name = tempfile.mkstemp(prefix=".study-design-v6-", dir=output_dir)
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
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--contamination", type=Path, default=DEFAULT_CONTAMINATION)
    parser.add_argument("--legacy-design", type=Path, default=DEFAULT_LEGACY_DESIGN)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--qwen-projection", type=Path, default=DEFAULT_QWEN_PROJECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--write-candidate",
        action="store_true",
        help="write the blocked content-addressed candidate; otherwise verify and print its digest",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_successor_design(
        campaign_path=args.campaign,
        readiness_path=args.readiness,
        contamination_path=args.contamination,
        legacy_design_path=args.legacy_design,
        roster_path=args.roster,
        qwen_projection_path=args.qwen_projection,
    )
    if args.write_candidate:
        print(write_successor_design(document, args.output_dir))
    else:
        print(document["artifact_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
