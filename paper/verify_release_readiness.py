#!/usr/bin/env python3
"""Verify the compact FlavourBench release-readiness evidence package.

This verifier is intentionally standard-library-only and read-only.  It supports
both the repository layout and the compact arXiv-source layout.  It validates
the frozen evidence itself; it does not replay historical builders against a
newer application tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReleaseReadinessError(RuntimeError):
    """The release package does not satisfy its frozen evidence contract."""


@dataclass(frozen=True)
class ArtifactSpec:
    repo_path: str
    archive_path: str
    physical_sha256: str
    semantic_sha256: str


ARTIFACTS: dict[str, ArtifactSpec] = {
    "operational_leaderboard": ArtifactSpec(
        "paper/flavourbench/generated/operational-leaderboard/"
        "epicure-operational-leaderboard-"
        "c2f716d9822a12c99eb6e42be3b6ce98de96ebd15cf5e87a72d9c42310e8c727.json",
        "generated/operational-leaderboard/epicure-operational-leaderboard.json",
        "7b18a853914a7b28c4bb151120cc4281f58c46dc4537136add3afcb325926049",
        "c2f716d9822a12c99eb6e42be3b6ce98de96ebd15cf5e87a72d9c42310e8c727",
    ),
    "operational_source": ArtifactSpec(
        "paper/flavourbench/generated/frontier-study/high-resource/"
        "frontier-multirun-"
        "c0bd526a2776a25adfbd2c43b98b8f15c143a8cb93b957ba961d0e9efe626688.json",
        "generated/frontier-study/high-resource/frontier-multirun-provenance.json",
        "377a6afffab5c3b6072be8157fb08fb0a1d94e59900a69d68b39c5ac268c2252",
        "c0bd526a2776a25adfbd2c43b98b8f15c143a8cb93b957ba961d0e9efe626688",
    ),
    "qwen_operational_projection": ArtifactSpec(
        "flavourbench/artifacts/season1/current-quality-run/"
        "release-package-remediation-v1/"
        "qwencloud-exploratory-operational-projection-"
        "b2f7790b3eb18d1df083397ce02b5296c549e5ed3ddb3d3f32ea776db3ddca04.json",
        "provenance/qwencloud-exploratory-operational-projection.json",
        "9343a2959d3acf3079fb91b2bd7ff608af421532b826f7b98917c88b76a7f85c",
        "b2f7790b3eb18d1df083397ce02b5296c549e5ed3ddb3d3f32ea776db3ddca04",
    ),
    "k14_power": ArtifactSpec(
        "flavourbench/artifacts/season1/sampling-power-validation-v1-candidate/"
        "sampling-power-validation-v1-candidate-"
        "1077a8c237854c9ba96d7b03f22aa60e0562c5f8aeb986fd52e5e167a9694504.json",
        "contracts/release-readiness/k14-sampling-power-audit.json",
        "140419f71a43c0276e87350ad367e6cf2e896879d02ebfaf9c1131a39d1202f9",
        "1077a8c237854c9ba96d7b03f22aa60e0562c5f8aeb986fd52e5e167a9694504",
    ),
    "k14_power_predecessor": ArtifactSpec(
        "flavourbench/artifacts/season1/sampling-power-validation-v1-candidate/"
        "sampling-power-validation-v1-candidate-"
        "8cd9217f6e59b47de0f66f1fc6c45fe5766b8729d3f808ee83cfb2e5147c7a9d.json",
        "contracts/release-readiness/k14-sampling-power-predecessor.json",
        "5a91e9e9b18d7004313a80f0c1754c33ec8012797ee54709b6e6534c0937d421",
        "8cd9217f6e59b47de0f66f1fc6c45fe5766b8729d3f808ee83cfb2e5147c7a9d",
    ),
    "k14_sampling_v1": ArtifactSpec(
        "flavourbench/artifacts/season1/human-judgment-sampling-v1-candidate/"
        "human-judgment-sampling-v1-candidate-"
        "5a0b1bbeb20564c9e8fde78b958bbed723ee0cc3395c809267c3775adeed95f8.json",
        "contracts/release-readiness/k14-sampling-frame-v1.json",
        "6c7371daa9506cdf5dcee38c19ee48dd2938f36d259c1ee04b7c849c49977039",
        "5a0b1bbeb20564c9e8fde78b958bbed723ee0cc3395c809267c3775adeed95f8",
    ),
    "k14_sampling_v2": ArtifactSpec(
        "flavourbench/artifacts/season1/human-judgment-sampling-v2-candidate/"
        "human-judgment-sampling-v2-candidate-"
        "34e52469f335d65bc3369726b06ff226ca6b1df0f43121b42695bc79bb1a1dc2.json",
        "contracts/release-readiness/k14-sampling-frame-v2.json",
        "d753c003d86072de81b7663fae6327018649c062ed045b629baad1632973040f",
        "34e52469f335d65bc3369726b06ff226ca6b1df0f43121b42695bc79bb1a1dc2",
    ),
    "k14_design_v6": ArtifactSpec(
        "flavourbench/artifacts/season1/study-design-v6-candidate/"
        "study-design-v6-candidate-"
        "e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e.json",
        "contracts/release-readiness/k14-study-design-v6.json",
        "6affdc8f80e59476254834d8edc588c471a5bd7e86145e66448e4fb7b90118af",
        "e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e",
    ),
    "k16_design": ArtifactSpec(
        "flavourbench/artifacts/season1/study-design-16-model-alternative-v1-candidate/"
        "study-design-16-model-alternative-v1-candidate-"
        "675cdb81bcbd54cf3532025ae70069723d7e9843b0eeeb92f1ea38bee7c58278.json",
        "contracts/release-readiness/k16-alternative-design.json",
        "f26bac6aa12a6dfca4ab0ee8ff7f2a0814a2214e8f21f9292d09221ec5104740",
        "675cdb81bcbd54cf3532025ae70069723d7e9843b0eeeb92f1ea38bee7c58278",
    ),
    "k16_provisional": ArtifactSpec(
        "flavourbench/artifacts/season1/full-k16-arena-resolution-audit-v1-candidate/"
        "full-k16-arena-resolution-audit-v1-candidate-"
        "b59dfc07280f972b83e00c5699a0f28e3da61135f2da22cfaf4638a4d1391910.json",
        "contracts/release-readiness/k16-provisional-arena-audit.json",
        "82caec6dba223c089b972775236ea47e5ef380b1beffd437461b6160b58ff893",
        "b59dfc07280f972b83e00c5699a0f28e3da61135f2da22cfaf4638a4d1391910",
    ),
    "k16_invalidated": ArtifactSpec(
        "flavourbench/artifacts/season1/full-k16-arena-resolution-audit-v1-candidate/"
        "full-k16-arena-resolution-audit-v1-candidate-"
        "64f9f2f8afad51ded9f3be8b84d0c91c0259eae78c5b1086099dbb9f4f26eb1a.json",
        "contracts/release-readiness/k16-invalidated-arena-audit.json",
        "5685a70d168ba6dde901eb5481e80e9ad3574eb3e1de4131f84c4e53dff23920",
        "64f9f2f8afad51ded9f3be8b84d0c91c0259eae78c5b1086099dbb9f4f26eb1a",
    ),
    "k16_fresh": ArtifactSpec(
        "flavourbench/artifacts/season1/full-k16-arena-resolution-audit-v1-candidate/"
        "full-k16-arena-resolution-audit-v1-candidate-"
        "596ef6cd38132a351605ce0f734489262e372e1c148337a88238dbb466a12ddc.json",
        "contracts/release-readiness/k16-fresh-arena-audit.json",
        "b07fa02b5633c13d331ad5490c150a514f92aa4bb7abdce985b156aa9e8af17c",
        "596ef6cd38132a351605ce0f734489262e372e1c148337a88238dbb466a12ddc",
    ),
    "release_projection": ArtifactSpec(
        "paper/flavourbench/provenance/release-readiness-audit-v1-"
        "03e486c3814bfc836f2c2f4920c26df8f595a99e328853973d0b8f20ebddb24c.json",
        "provenance/release-readiness-audit-v1-"
        "03e486c3814bfc836f2c2f4920c26df8f595a99e328853973d0b8f20ebddb24c.json",
        "65c56e22e60bda6168c463b01a7ac698f58da7e640fe0cb9ead5ae7552dd7598",
        "03e486c3814bfc836f2c2f4920c26df8f595a99e328853973d0b8f20ebddb24c",
    ),
}

ACTIVATION_PATHS = {
    "repo": "protocol/human-study/human-study-activation-current-v1.json",
    "archive": "contracts/release-readiness/human-study-activation.json",
}
ACTIVATION_PHYSICAL_SHA256 = (
    "fbf7e05db339778102b7cb9a74e0f264102b49f480e8e2486f9a2f941af493a4"
)
HUMAN_MANIFEST_PATHS = {
    "repo": "protocol/human-study/HUMAN-STUDY-GO-PACKAGE-v3.sha256",
    "archive": "contracts/release-readiness/HUMAN-STUDY-GO-PACKAGE-v3.sha256",
}
HUMAN_MANIFEST_PHYSICAL_SHA256 = (
    "7273a74e1e75a2d2943cf7eef1fbc8d63d078cc1bfc2c19ff9a55e3eb4a78a82"
)

EXPECTED_K14_FAILURES = [
    "plausible_rater_dropout_overall_coverage",
    "outcome_dependent_missingness_overall_coverage",
    "outcome_dependent_missingness_absolute_bias",
    "calibrated_0_08_complete_family_power_at_0_08",
    "calibrated_0_08_complete_per_model_power",
    "calibrated_0_08_complete_per_model_precision",
    "calibrated_0_08_complete_arena_proxy_top_identification_at_50_elo",
    "calibrated_0_08_high_dependence_family_power_at_0_08",
    "calibrated_0_08_high_dependence_per_model_power",
    "calibrated_0_08_high_dependence_per_model_precision",
    "calibrated_0_08_high_dependence_arena_proxy_top_identification_at_50_elo",
    "reliability_track_precision",
]

EXPECTED_HUMAN_BLOCKERS = [
    "monitored_research_contact_missing",
    "research_lead_affiliation_disclosure_inconsistent",
    "funding_and_sponsorship_disclosure_incomplete",
    "funded_fair_compensation_authority_missing",
    "ethics_or_equivalent_determination_missing",
    "participant_owned_acceptance_operation_missing",
    "participant_withdrawal_operation_missing",
    "retention_and_deletion_operation_missing",
    "identity_and_conflict_controls_incomplete",
    "activation_approvals_incomplete",
]

_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\x00\r\n]+)$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseReadinessError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_regular(path: Path, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReleaseReadinessError(f"missing {label}") from error
    _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ReleaseReadinessError(f"cannot read {label}") from error


def _reject_constant(value: str) -> None:
    raise ReleaseReadinessError(f"non-finite JSON constant: {value}")


def _object_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseReadinessError(f"invalid {label} JSON") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _artifact_path(root: Path, layout: str, spec: ArtifactSpec) -> Path:
    return root / (spec.repo_path if layout == "repo" else spec.archive_path)


def _load_artifact(root: Path, layout: str, label: str) -> dict[str, Any]:
    spec = ARTIFACTS[label]
    data = _read_regular(_artifact_path(root, layout, spec), label)
    _require(
        hashlib.sha256(data).hexdigest() == spec.physical_sha256,
        f"{label} physical digest mismatch",
    )
    document = _strict_json(data, label)
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(
        document.get("artifact_sha256") == spec.semantic_sha256
        and _semantic_sha256(body) == spec.semantic_sha256,
        f"{label} semantic digest mismatch",
    )
    return document


def _require_false_or_zero(mapping: Mapping[str, Any], label: str) -> None:
    for key, value in mapping.items():
        if isinstance(value, bool):
            _require(value is False, f"{label}.{key} is unexpectedly true")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            _require(value == 0, f"{label}.{key} is unexpectedly nonzero")
        else:
            raise ReleaseReadinessError(f"{label}.{key} has an unsupported claim value")


def _verify_k14(document: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        document.get("status") == "blocked_no_go_exact_frame_power_not_validated",
        "K14 audit status changed",
    )
    checks = document.get("prespecified_candidate_acceptance", {}).get("checks")
    _require(isinstance(checks, list) and len(checks) == 50, "K14 check count changed")
    _require(
        all(isinstance(row, dict) and row.get("core") is True for row in checks),
        "K14 core-check designation changed",
    )
    identifiers = [row.get("check_id") for row in checks]
    _require(
        all(isinstance(value, str) for value in identifiers)
        and len(identifiers) == len(set(identifiers)),
        "K14 check identifiers are not unique strings",
    )
    failed = [row["check_id"] for row in checks if row.get("passed") is False]
    _require(failed == EXPECTED_K14_FAILURES, "K14 failed-check set or order changed")
    _require(
        sum(row.get("passed") is True for row in checks) == 38, "K14 pass count changed"
    )
    claim_boundary = document.get("claim_boundary")
    _require(isinstance(claim_boundary, dict), "K14 claim boundary is missing")
    _require_false_or_zero(claim_boundary, "K14 claim boundary")
    return {"core_checks": 50, "failed_core_checks": 12, "status": "NO-GO"}


def _verify_operational_leaderboard(
    document: Mapping[str, Any],
    source: Mapping[str, Any],
    qwen: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        document.get("schema_version")
        == "flavourbench-epicure-operational-leaderboard-v1"
        and document.get("status") == "public_release_candidate"
        and document.get("leaderboard_scope")
        == "epicure_grounded_automated_operational"
        and document.get("official_within_scope") is True,
        "Epicure operational leaderboard scope or status changed",
    )
    source_claim = document.get("source")
    _require(
        isinstance(source_claim, dict)
        and source_claim.get("semantic_sha256")
        == ARTIFACTS["operational_source"].semantic_sha256
        and source_claim.get("physical_sha256")
        == ARTIFACTS["operational_source"].physical_sha256
        and source_claim.get("execution_policy_sha256")
        == "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d"
        and source_claim.get("task_set_sha256")
        == "dc6a5ecea427ac2fec8198be9a852d1ca29d4033cd26bee9f240bd7c19fb2a92",
        "Epicure operational source commitment changed",
    )
    _require(
        source.get("status") == "verified_real_development_pilot"
        and source.get("quality_ranking") is False
        and source.get("synthetic_tasks") == 0,
        "Epicure operational source evidence boundary changed",
    )

    rows = document.get("rows")
    _require(
        isinstance(rows, list) and len(rows) == 16, "operational row count changed"
    )
    model_ids = [row.get("model_id") for row in rows if isinstance(row, dict)]
    _require(
        len(model_ids) == len(set(model_ids)) == 16,
        "operational model identifiers are not unique",
    )
    _require(
        {
            "moonshotai/kimi-k3",
            "cohere/command-a-plus-05-2026",
            "cohere/command-a-reasoning-08-2025",
        }
        <= set(model_ids),
        "Kimi or a Cohere route is missing from the operational panel",
    )
    expected_ranked_pairs = [
        (1, "cohere/command-a-plus-05-2026", 12, 12),
        (2, "anthropic/claude-opus-5", 8, 8),
        (3, "anthropic/claude-fable-5", 8, 7),
        (3, "anthropic/claude-sonnet-5", 8, 7),
        (3, "deepseek/deepseek-v4-flash-0731", 8, 7),
        (3, "google/gemini-3.6-flash", 8, 7),
        (3, "moonshotai/kimi-k3", 8, 7),
        (4, "google/gemini-3.1-pro-preview", 8, 6),
        (4, "nvidia/nemotron-3-ultra-550b-a55b", 8, 6),
        (4, "openai/gpt-5.6-sol-pro", 8, 6),
        (5, "cohere/command-a-reasoning-08-2025", 12, 8),
        (5, "x-ai/grok-4.5", 12, 8),
        (6, "minimax/minimax-m3", 12, 7),
        (7, "deepseek/deepseek-v4-pro", 8, 5),
        (7, "z-ai/glm-5.2", 8, 5),
        (8, "mistralai/mistral-medium-3-5", 16, 4),
    ]
    actual_ranked_pairs = [
        (
            row.get("operational_rank"),
            row.get("model_id"),
            row.get("scheduled_pairs"),
            row.get("verified_complete_pairs"),
        )
        for row in rows
        if isinstance(row, dict)
    ]
    _require(
        actual_ranked_pairs == expected_ranked_pairs, "operational ranking changed"
    )
    _require(
        all(
            row.get("automated_operational_rank_eligible") is True
            and row.get("culinary_quality_rank_eligible") is False
            and row.get("quality_judgments") == 0
            for row in rows
            if isinstance(row, dict)
        ),
        "operational and culinary-quality eligibility were conflated",
    )

    totals = document.get("totals")
    _require(
        isinstance(totals, dict)
        and totals.get("ranked_models") == 16
        and totals.get("qualified_models") == 4
        and totals.get("provisional_models") == 12
        and totals.get("scheduled_pairs") == 152
        and totals.get("complete_pairs") == 110
        and totals.get("epicure_calls") == 273
        and totals.get("epicure_successful_calls") == 207
        and totals.get("quality_judgments") == 0
        and totals.get("synthetic_tasks") == 0,
        "operational leaderboard totals changed",
    )
    boundary = document.get("claim_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("automated_operational_leaderboard_official") is True
        and boundary.get("culinary_quality_leaderboard_official") is False
        and boundary.get("human_preference_leaderboard_official") is False
        and boundary.get("epicure_uplift_leaderboard_official") is False
        and boundary.get("quality_judgments") == 0,
        "operational leaderboard claim boundary changed",
    )

    extensions = document.get("unranked_extensions")
    qwen_identity = qwen.get("model_identity")
    qwen_boundary = qwen.get("claim_boundary")
    _require(
        isinstance(extensions, list)
        and len(extensions) == 1
        and isinstance(extensions[0], dict)
        and extensions[0].get("model_id") == "qwen3.8-max"
        and extensions[0].get("completed_pairs") == 1
        and extensions[0].get("rank") is None
        and extensions[0].get("status") == "insufficient_comparable_evidence"
        and extensions[0].get("source_semantic_sha256")
        == ARTIFACTS["qwen_operational_projection"].semantic_sha256
        and isinstance(qwen_identity, dict)
        and qwen_identity.get("identity_kind") == "mutable_alias"
        and isinstance(qwen_boundary, dict)
        and qwen_boundary.get("official") is False
        and qwen_boundary.get("rank_eligible") is False,
        "Qwen display-only boundary changed",
    )
    return {
        "official_within_scope": True,
        "quality_judgments": 0,
        "ranked_models": 16,
        "scope": "epicure_grounded_automated_operational",
        "status": "GO",
        "unranked_qwen_extensions": 1,
    }


def _verify_k16_records(
    document: Mapping[str, Any], label: str
) -> list[Mapping[str, Any]]:
    _require(
        document.get("status")
        == "development_only_no_go_not_production_method_validation",
        f"{label} status changed",
    )
    records = document.get("dataset_records")
    _require(
        isinstance(records, list) and len(records) == 520,
        f"{label} record count changed",
    )
    record_hashes: list[str] = []
    for record in records:
        _require(isinstance(record, dict), f"{label} contains a non-object record")
        record_body = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        record_hash = record.get("record_sha256")
        _require(
            isinstance(record_hash, str)
            and record_hash == _semantic_sha256(record_body),
            f"{label} record digest mismatch",
        )
        record_hashes.append(record_hash)
    _require(
        len(record_hashes) == len(set(record_hashes)),
        f"{label} record hashes are not unique",
    )
    _require(
        document.get("record_set_sha256")
        == _semantic_sha256({"record_sha256s": sorted(record_hashes)}),
        f"{label} record-set digest mismatch",
    )
    conditions = [
        row
        for row in document.get("condition_results", [])
        if isinstance(row, dict) and row.get("stage") == "fresh_final_confirmation"
    ]
    _require(len(conditions) == 1, f"{label} final confirmation is not unique")
    condition = conditions[0]
    _require(
        condition.get("config_id") == "p80_r2"
        and condition.get("shift_elo") == 50.0
        and condition.get("bootstraps_per_dataset") == 200
        and condition.get("datasets") == 100,
        f"{label} final confirmation contract changed",
    )
    confirmation = [
        row
        for row in records
        if row.get("config_id") == condition["config_id"]
        and row.get("shift_elo") == condition["shift_elo"]
        and row.get("bootstrap_replicates") == condition["bootstraps_per_dataset"]
    ]
    _require(len(confirmation) == 100, f"{label} confirmation record count changed")
    return confirmation


def _simulation_identity(record: Mapping[str, Any]) -> tuple[str, str, float, int, int]:
    return (
        str(record["layout_sha256"]),
        str(record["config_id"]),
        float(record["shift_elo"]),
        int(record["dataset_index"]),
        int(record["dataset_seed"]),
    )


def _record_hashes(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["record_sha256"]) for row in records}


def _verify_k16_lineage(
    provisional: Mapping[str, Any],
    invalidated: Mapping[str, Any],
    fresh: Mapping[str, Any],
) -> dict[str, Any]:
    provisional_confirmation = _verify_k16_records(provisional, "K16 provisional audit")
    invalidated_confirmation = _verify_k16_records(invalidated, "K16 invalidated audit")
    fresh_confirmation = _verify_k16_records(fresh, "K16 fresh audit")

    _require(
        {int(row["dataset_index"]) for row in provisional_confirmation}
        == set(range(0, 100)),
        "K16 provisional confirmation indices changed",
    )
    _require(
        {int(row["dataset_index"]) for row in invalidated_confirmation}
        == set(range(40, 140)),
        "K16 invalidated confirmation indices changed",
    )
    _require(
        {int(row["dataset_index"]) for row in fresh_confirmation}
        == set(range(140, 240)),
        "K16 fresh confirmation indices changed",
    )
    _require(
        all(
            row.get("record_lineage") == "generated_fresh_successor_indices_140_239"
            for row in fresh_confirmation
        ),
        "K16 fresh confirmation lineage label changed",
    )

    provisional_identities = {
        _simulation_identity(row) for row in provisional_confirmation
    }
    invalidated_identities = {
        _simulation_identity(row) for row in invalidated_confirmation
    }
    fresh_identities = {_simulation_identity(row) for row in fresh_confirmation}
    _require(
        len(provisional_identities)
        == len(invalidated_identities)
        == len(fresh_identities)
        == 100,
        "K16 confirmation identities are not unique",
    )
    _require(
        len(provisional_identities.intersection(invalidated_identities)) == 60,
        "K16 predecessor identity overlap changed",
    )
    _require(
        fresh_identities.isdisjoint(
            provisional_identities.union(invalidated_identities)
        ),
        "K16 fresh confirmation overlaps prior observed identities",
    )

    provisional_records = provisional["dataset_records"]
    invalidated_records = invalidated["dataset_records"]
    fresh_records = fresh["dataset_records"]
    provisional_confirmation_hashes = _record_hashes(provisional_confirmation)
    invalidated_confirmation_hashes = _record_hashes(invalidated_confirmation)
    fresh_confirmation_hashes = _record_hashes(fresh_confirmation)
    common_provisional = (
        _record_hashes(provisional_records) - provisional_confirmation_hashes
    )
    common_invalidated = (
        _record_hashes(invalidated_records) - invalidated_confirmation_hashes
    )
    common_fresh = _record_hashes(fresh_records) - fresh_confirmation_hashes
    _require(
        len(common_provisional) == 420
        and common_provisional == common_invalidated == common_fresh,
        "K16 common 420-record lineage changed",
    )
    _require(
        fresh_confirmation_hashes.isdisjoint(
            provisional_confirmation_hashes.union(invalidated_confirmation_hashes)
        ),
        "K16 fresh confirmation reuses a predecessor record hash",
    )

    stage_lineage = fresh.get("stage_lineage")
    _require(isinstance(stage_lineage, dict), "K16 fresh stage lineage is missing")
    _require(
        stage_lineage.get("fresh_confirmation_dataset_indices") == [140, 239]
        and stage_lineage.get("fresh_confirmation_record_count") == 100
        and stage_lineage.get("fresh_confirmation_unique_simulation_identity_count")
        == 100
        and stage_lineage.get("fresh_confirmation_overlap_with_prior_observed_union")
        == 0
        and stage_lineage.get("prior_observed_union_unique_simulation_identity_count")
        == 520
        and stage_lineage.get("reused_record_count") == 420,
        "K16 fresh stage-lineage commitments changed",
    )

    conditions = [
        row
        for row in fresh["condition_results"]
        if row.get("stage") == "fresh_final_confirmation"
    ]
    condition = conditions[0]
    marginal = condition.get(
        "frozen_average_marginal_pair_power_uniform_peer_estimator"
    )
    simultaneous = condition.get("all_15_pointwise_intervals_positive_same_dataset")
    _require(
        isinstance(marginal, dict)
        and marginal.get("successes") == 97
        and marginal.get("trials") == 100
        and marginal.get("clopper_pearson_95_lower") == 0.914823947
        and condition.get("frozen_target_lower_bound_passes") is True,
        "K16 marginal-resolution result changed",
    )
    _require(
        isinstance(simultaneous, dict)
        and simultaneous.get("successes") == 67
        and simultaneous.get("trials") == 100
        and simultaneous.get("clopper_pearson_95_lower") == 0.568827249
        and condition.get("simultaneous_top_identification_lower_bound_passes")
        is False,
        "K16 simultaneous-top result changed",
    )
    production_gate = fresh.get("simulation_contract", {}).get(
        "production_gate_required_but_not_run"
    )
    _require(
        production_gate
        == {
            "bootstrap_replicates": 5000,
            "datasets_per_scenario": 2000,
            "nominal_bootstrap_refits": 80000000,
            "scenarios": 8,
        },
        "K16 production-gate commitment changed",
    )
    decision = fresh.get("decision")
    _require(
        isinstance(decision, dict)
        and decision.get("overall_verdict") == "NO-GO"
        and decision.get("frozen_50_elo_marginal_target_passed") is True
        and decision.get("simultaneous_top_resolution_passed") is False,
        "K16 release decision changed",
    )
    claim_boundary = fresh.get("claim_boundary")
    _require(isinstance(claim_boundary, dict), "K16 claim boundary is missing")
    _require(
        claim_boundary.get("development_only") is True, "K16 is not development-only"
    )
    for key, value in claim_boundary.items():
        if key != "development_only":
            _require(value is False, f"K16 claim boundary {key} is not false")
    return {
        "fresh_confirmation_records": 100,
        "fresh_dataset_indices": [140, 239],
        "fresh_identity_overlap_with_predecessors": 0,
        "marginal_50_elo_exact_95_lower": 0.914823947,
        "marginal_target_passed": True,
        "production_gate_completed": False,
        "simultaneous_top_exact_95_lower": 0.568827249,
        "simultaneous_top_target_passed": False,
        "status": "NO-GO",
    }


def _verify_k16_panel(document: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        document.get("status") == "blocked_offline_16_model_alternative_not_authorized",
        "K16 model-panel status changed",
    )
    panel = document.get("candidate_model_panel")
    _require(isinstance(panel, dict), "K16 candidate panel is missing")
    models = panel.get("models")
    _require(isinstance(models, list) and len(models) == 16, "K16 panel size changed")
    model_ids = [row.get("model_id") for row in models if isinstance(row, dict)]
    _require(
        len(model_ids) == len(set(model_ids)) == 16,
        "K16 panel model identifiers are not unique",
    )
    _require(
        model_ids.count("qwen3.8-max") == 1
        and model_ids.count("moonshotai/kimi-k3") == 1
        and model_ids.count("cohere/command-a-plus-05-2026") == 1,
        "K16 Qwen/Kimi/Cohere membership changed",
    )
    by_id = {row["model_id"]: row for row in models}
    _require(
        by_id["qwen3.8-max"].get("identity_kind") == "mutable_alias"
        and by_id["moonshotai/kimi-k3"].get("identity_kind")
        == "provider_managed_direct_identifier_immutability_unproven",
        "K16 Qwen or Kimi identity boundary changed",
    )
    _require(
        panel.get("official") is False
        and panel.get("rank_eligible") is False
        and panel.get("quality_eligible") is False
        and panel.get("quality_observations_used") == 0
        and panel.get("calls_authorized") is False
        and panel.get("frozen_for_confirmatory_collection") is False,
        "K16 panel authority boundary changed",
    )
    boundary = document.get("claim_boundary")
    _require(isinstance(boundary, dict), "K16 panel claim boundary is missing")
    for key, value in boundary.items():
        if key == "activation_effect":
            _require(value == "none", "K16 activation effect changed")
        elif isinstance(value, bool):
            _require(value is False, f"K16 panel claim {key} is true")
        elif isinstance(value, (int, float)):
            _require(value == 0, f"K16 panel claim {key} is nonzero")
        else:
            raise ReleaseReadinessError(
                f"K16 panel claim {key} has an unsupported value"
            )
    return {
        "includes_cohere_command_a_plus": True,
        "includes_kimi_k3": True,
        "includes_qwen_3_8_max": True,
        "model_count": 16,
        "official": False,
        "rank_eligible": False,
    }


def _verify_activation(root: Path, layout: str) -> dict[str, Any]:
    data = _read_regular(root / ACTIVATION_PATHS[layout], "human activation record")
    _require(
        hashlib.sha256(data).hexdigest() == ACTIVATION_PHYSICAL_SHA256,
        "human activation record digest mismatch",
    )
    document = _strict_json(data, "human activation record")
    _require(
        document.get("status") == "blocked"
        and document.get("consent_status") == "not_active"
        and document.get("participants_enrolled_before_activation") == 0
        and document.get("human_judgments_collected_before_activation") == 0
        and document.get("blockers") == EXPECTED_HUMAN_BLOCKERS,
        "human activation hold changed",
    )
    budget = document.get("human_evaluation_budget")
    _require(
        isinstance(budget, dict)
        and budget.get("EUR_micros") == 0
        and budget.get("USD_micros") == 0
        and budget.get("funded") is False,
        "human-study budget boundary changed",
    )
    boundary = document.get("claim_boundary")
    _require(isinstance(boundary, dict), "human activation claim boundary is missing")
    _require_false_or_zero(boundary, "human activation claim boundary")
    return {
        "blockers": 10,
        "funded": False,
        "human_judgments": 0,
        "participants_enrolled": 0,
        "status": "NO-GO",
    }


def _verify_human_manifest(root: Path, layout: str) -> dict[str, Any]:
    data = _read_regular(
        root / HUMAN_MANIFEST_PATHS[layout], "human v3 checksum manifest"
    )
    _require(
        hashlib.sha256(data).hexdigest() == HUMAN_MANIFEST_PHYSICAL_SHA256,
        "human v3 checksum manifest digest mismatch",
    )
    _require(
        data.endswith(b"\n"), "human v3 checksum manifest lacks a trailing newline"
    )
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ReleaseReadinessError(
            "human v3 checksum manifest is not UTF-8"
        ) from error
    _require(len(lines) == 61, "human v3 checksum manifest entry count changed")
    records: list[tuple[str, str]] = []
    for line in lines:
        match = _MANIFEST_LINE.fullmatch(line)
        _require(match is not None, "human v3 checksum manifest line is malformed")
        records.append((match.group(1), match.group(2)))
    members = [member for _, member in records]
    _require(
        len(members) == len(set(members)), "human v3 checksum manifest repeats a member"
    )
    verified = 0
    if layout == "repo":
        for expected, member in records:
            member_data = _read_regular(root / member, f"human v3 member {member}")
            _require(
                hashlib.sha256(member_data).hexdigest() == expected,
                f"human v3 member digest mismatch: {member}",
            )
            verified += 1
    return {
        "entry_count": 61,
        "entries_verified": verified,
        "full_member_verification_available": layout == "repo",
    }


def _verify_projection(
    projection: Mapping[str, Any],
    k14: Mapping[str, Any],
    k16: Mapping[str, Any],
    human: Mapping[str, Any],
    operational: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = projection.get("claim_boundary")
    _require(isinstance(boundary, dict), "release projection claim boundary is missing")
    _require_false_or_zero(
        {
            key: value
            for key, value in boundary.items()
            if key != "paper_disclosure_scope"
        },
        "release projection claim boundary",
    )
    _require(
        boundary.get("paper_disclosure_scope")
        == "methodological_readiness_and_release_hold_only",
        "release projection disclosure scope changed",
    )
    release = projection.get("release_decision")
    _require(
        isinstance(release, dict)
        and release.get("arxiv_preprint")
        == "TECHNICAL_BUILD_CANDIDATE_AS_RETROSPECTIVE_AUDIT_ONLY"
        and release.get("official_quality_leaderboard") == "NO-GO"
        and release.get("production_method_validation") == "NO-GO"
        and release.get("human_study_activation") == "NO-GO"
        and release.get("raw_response_release") == "NO-GO"
        and release.get("public_product_rank_rows") == 0,
        "release decision changed",
    )
    projected_k14 = projection.get("statistical_readiness", {}).get("k14_exact_frame")
    projected_k16 = projection.get("statistical_readiness", {}).get(
        "fresh_k16_arena_design_audit"
    )
    projected_human = projection.get("human_study_readiness")
    _require(
        isinstance(projected_k14, dict)
        and projected_k14.get("core_checks") == k14["core_checks"]
        and projected_k14.get("failed_core_checks") == k14["failed_core_checks"]
        and projected_k14.get("failed_check_ids") == EXPECTED_K14_FAILURES
        and projected_k14.get("status") == "NO-GO",
        "release projection K14 summary changed",
    )
    _require(
        isinstance(projected_k16, dict)
        and projected_k16.get("fresh_dataset_records")
        == k16["fresh_confirmation_records"]
        and projected_k16.get("fresh_identity_overlap_with_predecessors") == 0
        and projected_k16.get("production_gate_completed") is False
        and projected_k16.get("production_gate_nominal_refits") == 80000000
        and projected_k16.get("status") == "NO-GO",
        "release projection K16 summary changed",
    )
    _require(
        isinstance(projected_human, dict)
        and projected_human.get("human_judgments") == human["human_judgments"]
        and projected_human.get("participants_enrolled")
        == human["participants_enrolled"]
        and projected_human.get("funded_budget") is False
        and projected_human.get("blockers") == EXPECTED_HUMAN_BLOCKERS
        and projected_human.get("status") == "NO-GO",
        "release projection human-study summary changed",
    )
    return {
        "arxiv_preprint": "TECHNICAL_BUILD_CANDIDATE_AS_RETROSPECTIVE_AUDIT_ONLY",
        "automated_operational_leaderboard": "GO",
        "automated_operational_rank_rows": operational["ranked_models"],
        "official_quality_leaderboard": "NO-GO",
        "quality_public_product_rank_rows": 0,
        "raw_response_release": "NO-GO",
    }


def verify_release(root: Path, layout: str) -> dict[str, Any]:
    """Verify one repository or compact archive tree and return a safe summary."""

    _require(layout in {"repo", "archive"}, "unsupported layout")
    _require(root.is_dir(), "verification root is not a directory")
    documents = {label: _load_artifact(root, layout, label) for label in ARTIFACTS}

    operational = _verify_operational_leaderboard(
        documents["operational_leaderboard"],
        documents["operational_source"],
        documents["qwen_operational_projection"],
    )
    k14 = _verify_k14(documents["k14_power"])
    # Loading the other four K14 documents above verifies their exact physical and
    # semantic commitments.  Their historical builders are intentionally not
    # replayed against the newer application tree.
    k16_panel = _verify_k16_panel(documents["k16_design"])
    k16 = _verify_k16_lineage(
        documents["k16_provisional"],
        documents["k16_invalidated"],
        documents["k16_fresh"],
    )
    human = _verify_activation(root, layout)
    human_manifest = _verify_human_manifest(root, layout)
    release = _verify_projection(
        documents["release_projection"], k14, k16, human, operational
    )
    return {
        "artifacts_semantically_verified": len(ARTIFACTS),
        "human_go_package_v3": human_manifest,
        "human_study": human,
        "k14_statistical_readiness": k14,
        "k16_model_panel": k16_panel,
        "k16_statistical_readiness": k16,
        "layout": layout,
        "operational_leaderboard": operational,
        "release_decision": release,
        "schema_version": "flavourbench-release-readiness-verifier-v1",
        "status": "PASS",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--layout", choices=("repo", "archive"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = verify_release(arguments.root, arguments.layout)
    except ReleaseReadinessError as error:
        print(f"release-readiness verification failed: {error}", file=sys.stderr)
        return 1
    print(_canonical_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
