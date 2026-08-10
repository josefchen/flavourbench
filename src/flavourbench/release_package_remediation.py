"""Build append-only release corrections without invoking an external service."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .frontier_coverage_postrun import _uplift_identity_commitment
from .real_task_bank import sha256_json

HELD_REVIEW_ITEM_ID = "923c80138ebbda34c633b4db3ea35e57c5cab66d93b1d790ce35887ab726315c"


class ReleaseRemediationError(RuntimeError):
    """A release correction failed its integrity contract."""


def _physical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_addressed(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseRemediationError(f"not a regular artifact: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseRemediationError(f"invalid JSON artifact: {path}") from error
    if not isinstance(document, dict):
        raise ReleaseRemediationError(f"artifact is not an object: {path}")
    digest = document.get("artifact_sha256")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if not isinstance(digest, str) or sha256_json(body) != digest:
        raise ReleaseRemediationError(f"content address does not verify: {path}")
    return document


def _address(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _write_addressed(
    output_dir: Path,
    stem: str,
    document: Mapping[str, Any],
) -> Path:
    digest = str(document["artifact_sha256"])
    path = output_dir / f"{stem}-{digest}.json"
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if _read_addressed(path)["artifact_sha256"] != digest:
        raise ReleaseRemediationError(f"written artifact failed verification: {path}")
    return path


def _held_pair_policy_evidence(
    held_item: Mapping[str, Any], artifacts_root: Path
) -> tuple[dict[str, Any], dict[str, int]]:
    response_ids = {str(held_item[side]["response_artifact_sha256"]) for side in ("left", "right")}
    index = _index_target_records(artifacts_root, response_ids)
    arms: dict[str, Mapping[str, Any]] = {}
    for side in ("left", "right"):
        expected = held_item[side]
        response_id = str(expected["response_artifact_sha256"])
        _, response, canonical = index[response_id]
        if not canonical:
            raise ReleaseRemediationError("held normalized response is not self-addressed")
        source = response.get("source")
        model = response.get("model")
        task = response.get("task")
        provenance = response.get("provenance")
        if not all(isinstance(value, Mapping) for value in (source, model, task, provenance)):
            raise ReleaseRemediationError("held response provenance is incomplete")
        if (
            response.get("condition") != expected["condition"]
            or source.get("artifact_sha256") != expected["source_artifact_sha256"]
            or model.get("actual_model_id") != expected["actual_model_id"]
            or model.get("actual_provider") != expected["actual_provider"]
            or task.get("prompt_sha256") != held_item["prompt_sha256"]
            or task.get("public_id") != held_item["task_id"]
        ):
            raise ReleaseRemediationError("held response does not match its normalized arm")
        arms[side] = response

    left = arms["left"]
    right = arms["right"]

    def field(document: Mapping[str, Any], *path: str) -> Any:
        value: Any = document
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ReleaseRemediationError(f"held response is missing {'.'.join(path)}")
            value = value[key]
        return value

    exact_match_paths = {
        "task_id": ("task", "public_id"),
        "prompt_sha256": ("task", "prompt_sha256"),
        "requested_model_id": ("model", "requested_model_id"),
        "canonical_model_slug": ("model", "canonical_model_slug"),
        "actual_model_id": ("model", "actual_model_id"),
        "actual_provider": ("model", "actual_provider"),
        "provider_tag": ("model", "provider_tag"),
        "execution_policy_sha256": ("execution_policy_sha256",),
        "endpoint_execution_sha256": ("model", "endpoint_execution_sha256"),
        "system_prompt_sha256": ("provenance", "system_prompt_sha256"),
        "response_schema_sha256": ("provenance", "response_schema_sha256"),
        "epicure_tool_schema_sha256": ("provenance", "epicure_tool_schema_sha256"),
        "backend_contract_sha256": ("model", "backend_contract_sha256"),
    }
    matching: dict[str, Any] = {}
    for name, path in exact_match_paths.items():
        left_value = field(left, *path)
        right_value = field(right, *path)
        if left_value != right_value:
            raise ReleaseRemediationError(f"held pair differs on required field {name}")
        matching[name] = left_value
    left_decoding = field(left, "provenance", "decoding")
    right_decoding = field(right, "provenance", "decoding")
    if left_decoding != right_decoding:
        raise ReleaseRemediationError("held pair differs on decoding configuration")
    matching["decoding"] = left_decoding

    differing = {
        "condition": {side: arms[side]["condition"] for side in ("left", "right")},
        "manifest_sha256": {side: arms[side]["manifest_sha256"] for side in ("left", "right")},
        "protocol_bundle_sha256": {
            side: field(arms[side], "provenance", "protocol_bundle_sha256")
            for side in ("left", "right")
        },
        "work_item_id": {side: arms[side]["work_item_id"] for side in ("left", "right")},
        "source_artifact_sha256": {
            side: field(arms[side], "source", "artifact_sha256") for side in ("left", "right")
        },
        "response_artifact_sha256": {
            side: arms[side]["artifact_sha256"] for side in ("left", "right")
        },
    }
    if differing["protocol_bundle_sha256"]["left"] == differing["protocol_bundle_sha256"]["right"]:
        raise ReleaseRemediationError("held pair unexpectedly shares one protocol bundle")

    metrics = {
        "provider_calls": 0,
        "epicure_calls": 0,
        "successful_epicure_calls": 0,
        "cost_micros": 0,
    }
    for response in arms.values():
        cost = field(response, "cost")
        trace = field(response, "provenance", "mcp_trace_events")
        if not isinstance(cost, Mapping) or not isinstance(trace, list):
            raise ReleaseRemediationError("held response accounting is incomplete")
        generation_ids = cost.get("generation_ids")
        if not isinstance(generation_ids, list):
            raise ReleaseRemediationError("held response has no generation ledger")
        metrics["provider_calls"] += len(generation_ids)
        metrics["epicure_calls"] += len(trace)
        metrics["successful_epicure_calls"] += sum(
            isinstance(event, Mapping) and event.get("is_error") is False for event in trace
        )
        metrics["cost_micros"] += int(cost["actual_cost_micros"])
    evidence = {
        "source": "two_private_normalized_response_records",
        "matching_required_fields": matching,
        "condition_specific_or_collection_specific_fields": differing,
        "matching_required_fields_sha256": sha256_json(matching),
        "condition_specific_fields_sha256": sha256_json(differing),
        "evidence_boundary": (
            "Exact field equality is verified from the two raw normalized responses. "
            "The predecessor pair schema does not attest that the differing protocol bundles "
            "contain only the prespecified Epicure treatment delta."
        ),
    }
    return evidence, metrics


def _build_uplift_successor(predecessor: Mapping[str, Any], artifacts_root: Path) -> dict[str, Any]:
    items = predecessor.get("items")
    observed = predecessor.get("observed")
    boundary = predecessor.get("claim_boundary")
    if (
        predecessor.get("track") != "epicure_uplift"
        or not isinstance(items, list)
        or not isinstance(observed, Mapping)
        or not isinstance(boundary, Mapping)
    ):
        raise ReleaseRemediationError("uplift predecessor has an invalid shape")
    held = [item for item in items if item.get("review_item_id") == HELD_REVIEW_ITEM_ID]
    if len(held) != 1:
        raise ReleaseRemediationError("expected exactly one named uplift item")
    held_item = held[0]
    policy_evidence, removed_metrics = _held_pair_policy_evidence(held_item, artifacts_root)
    retained = [item for item in items if item.get("review_item_id") != HELD_REVIEW_ITEM_ID]
    if len(retained) != len(items) - 1:
        raise ReleaseRemediationError("uplift quarantine did not remove exactly one item")

    model_order = [str(value) for value in predecessor.get("model_order") or []]
    family_counts = Counter(str(item["task_family"]) for item in retained)
    model_counts = Counter(str(item["requested_model_id"]) for item in retained)
    left_on = sum(item["left"]["condition"] == "epicure_on" for item in retained)
    successor_observed = {
        "retained_strict_pairs": int(observed["retained_strict_pairs"]),
        "retained_high_resource_pairs": int(observed["retained_high_resource_pairs"]),
        "coverage_recovery_pairs_added": int(observed["coverage_recovery_pairs_added"]) - 1,
        "candidate_pairs": len(retained),
        "source_arms": 2 * len(retained),
        "unique_task_ids": len({str(item["task_id"]) for item in retained}),
        "distinct_tasks": len({str(item["task_id"]) for item in retained}),
        "candidate_pairs_by_family": {
            family: family_counts[family]
            for family in ("substitution", "composition", "cookability", "evidence")
        },
        "candidate_pairs_by_model": {model_id: model_counts[model_id] for model_id in model_order},
        "real_provider_calls": (
            int(observed["real_provider_calls"]) - removed_metrics["provider_calls"]
        ),
        "real_epicure_calls": int(observed["real_epicure_calls"])
        - removed_metrics["epicure_calls"],
        "successful_real_epicure_calls": (
            int(observed["successful_real_epicure_calls"])
            - removed_metrics["successful_epicure_calls"]
        ),
        "reviewed_source_cost_micros": (
            int(observed["reviewed_source_cost_micros"]) - removed_metrics["cost_micros"]
        ),
        "left_epicure_on": left_on,
        "right_epicure_on": len(retained) - left_on,
        "synthetic_arms": 0,
    }
    if (
        successor_observed["candidate_pairs"] != 186
        or successor_observed["source_arms"] != 372
        or successor_observed["coverage_recovery_pairs_added"] != 7
    ):
        raise ReleaseRemediationError("uplift successor does not reproduce the expected scope")

    payload = {
        "schema_version": "flavourbench-frontier-uplift-policy-hold-successor-v1",
        "artifact_role": "append_only_development_uplift_successor",
        "status": "verified_real_development_input_with_policy_hold",
        "track": "epicure_uplift",
        "source": {
            "predecessor_uplift_sha256": predecessor["artifact_sha256"],
            "historical_raw_artifacts_mutated": False,
            "supersession_kind": "selection_only_append_only_successor",
        },
        "selection_policy": {
            **dict(predecessor.get("selection_policy") or {}),
            "cross_source_pair_requires_explicit_allowed_policy_delta_attestation": True,
        },
        "policy_hold": {
            "review_item_id": HELD_REVIEW_ITEM_ID,
            "pair_key": held_item.get("pair_key"),
            "requested_model_id": held_item.get("requested_model_id"),
            "task_id": held_item.get("task_id"),
            "task_family": held_item.get("task_family"),
            "left_condition": held_item["left"]["condition"],
            "left_response_artifact_sha256": held_item["left"]["response_artifact_sha256"],
            "left_source_artifact_sha256": held_item["left"]["source_artifact_sha256"],
            "right_condition": held_item["right"]["condition"],
            "right_response_artifact_sha256": held_item["right"]["response_artifact_sha256"],
            "right_source_artifact_sha256": held_item["right"]["source_artifact_sha256"],
            "recorded_execution_policy_sha256": (
                "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d"
            ),
            "pair_policy_evidence": policy_evidence,
            "held_pair_operational_metrics": removed_metrics,
            "reason": (
                "The two raw arms record the same execution-policy digest, model, provider, "
                "task, and prompt. The normalized pair does not itself attest that every "
                "condition-specific protocol difference is an allowed Epicure treatment delta."
            ),
            "disposition": "excluded_from_uplift_fit_pending_formal_pair_policy_attestation",
            "raw_records_retained": True,
            "reliability_accounting_retained_in_source_collection": True,
        },
        "observed": successor_observed,
        "epicure": predecessor.get("epicure"),
        "identity_commitment_sha256": _uplift_identity_commitment(retained),
        "model_order": model_order,
        "model_contracts": predecessor.get("model_contracts"),
        "items": retained,
        "claim_boundary": {
            **dict(boundary),
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "prohibited_use": "quality or uplift ranking before admissible real judgments",
        },
    }
    return _address(payload)


def _build_coverage_successor(
    predecessor: Mapping[str, Any], uplift: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in predecessor.items()
        if key not in {"artifact_sha256", "schema_version", "artifact_role", "source", "uplift"}
    }
    source = dict(predecessor.get("source") or {})
    source.update(
        {
            "predecessor_coverage_sha256": predecessor["artifact_sha256"],
            "predecessor_corrected_uplift_sha256": source.get("corrected_uplift_sha256"),
            "corrected_uplift_sha256": uplift["artifact_sha256"],
            "historical_raw_artifacts_mutated": False,
        }
    )
    uplift_counts = dict(predecessor.get("uplift") or {})
    uplift_counts.update(
        {
            "pairs_added": uplift["observed"]["coverage_recovery_pairs_added"],
            "pairs_after": uplift["observed"]["candidate_pairs"],
            "pairs_on_policy_hold": 1,
        }
    )
    return _address(
        {
            "schema_version": "flavourbench-frontier-corrected-coverage-metrics-v3",
            "artifact_role": "append_only_development_coverage_successor",
            **payload,
            "source": source,
            "uplift": uplift_counts,
        }
    )


def _build_public_authorization(original_path: Path, original: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "flavourbench-human-pi-public-authorization-disclosure-v1",
        "artifact_role": "public_disclosure_superseding_environment_specific_governance_copy",
        "status": "public_metadata_projection",
        "human_principal": {
            "name": "Josef Chen",
            "role": "human principal investigator",
            "affiliation": "Independent Researcher",
        },
        "authorization": {
            "scope": (
                "Bounded V9 recovery of one already-paid response pair and release of "
                "never-started reservations"
            ),
            "provider_calls_authorized_by_this_public_record": 0,
            "epicure_calls_authorized_by_this_public_record": 0,
            "new_cost_authorized_by_this_public_record": 0,
            "recorded_decision": "approved",
            "basis": "human-principal instruction retained in the private governance record",
        },
        "superseded_public_copy": {
            "semantic_artifact_sha256": original["artifact_sha256"],
            "physical_file_sha256": _physical_sha256(original_path),
            "private_record_retained_unchanged": True,
            "included_in_public_source_archive": False,
            "reason_for_public_supersession": (
                "The private record contains environment-specific transport metadata that is "
                "irrelevant to the scientific claim."
            ),
        },
        "claim_boundary": {
            "cryptographic_signature": False,
            "independent_external_governance": False,
            "scientific_quality_review": False,
            "official_ranking_supported": False,
            "changes_original_record": False,
            "changes_recovery_receipt": False,
        },
    }
    return _address(payload)


def _index_target_records(
    artifacts_root: Path,
    targets: set[str],
) -> dict[str, tuple[Path, dict[str, Any], bool]]:
    candidates: defaultdict[str, list[tuple[Path, dict[str, Any], bool]]] = defaultdict(list)
    for path in sorted(artifacts_root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        digest = document.get("artifact_sha256")
        if digest in targets:
            body = {key: value for key, value in document.items() if key != "artifact_sha256"}
            candidates[digest].append((path, document, sha256_json(body) == digest))
    missing = sorted(targets - set(candidates))
    if missing:
        raise ReleaseRemediationError(
            f"{len(missing)} committed raw artifacts were not found: {missing[:3]}"
        )
    result: dict[str, tuple[Path, dict[str, Any], bool]] = {}
    for digest, matches in candidates.items():
        physical = {_physical_sha256(path) for path, _, _ in matches}
        if len(physical) != 1:
            raise ReleaseRemediationError(
                f"semantic artifact {digest} has conflicting physical records"
            )
        result[digest] = min(matches, key=lambda item: str(item[0]))
    return result


def _arm_references(
    arena: Mapping[str, Any], uplift: Mapping[str, Any]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    response_memberships: defaultdict[str, set[str]] = defaultdict(set)
    source_memberships: defaultdict[str, set[str]] = defaultdict(set)
    for track, document in (("model_arena", arena), ("epicure_uplift", uplift)):
        items = document.get("items")
        if not isinstance(items, list):
            raise ReleaseRemediationError(f"{track} items are missing")
        for item in items:
            for side in ("left", "right"):
                arm = item.get(side)
                if not isinstance(arm, Mapping):
                    raise ReleaseRemediationError(f"{track} item has no {side} arm")
                response_memberships[str(arm["response_artifact_sha256"])].add(track)
                source_memberships[str(arm["source_artifact_sha256"])].add(track)
    return dict(response_memberships), dict(source_memberships)


def _private_record_rows(
    *,
    artifacts_root: Path,
    record_class: str,
    memberships: Mapping[str, set[str]],
    index: Mapping[str, tuple[Path, dict[str, Any], bool]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for semantic_digest in sorted(memberships):
        path, _, canonical_body_match = index[semantic_digest]
        relative = path.relative_to(artifacts_root).as_posix()
        rows.append(
            {
                "record_class": record_class,
                "semantic_artifact_sha256": semantic_digest,
                "physical_file_sha256": _physical_sha256(path),
                "physical_file_bytes": path.stat().st_size,
                "embedded_semantic_address_matches_canonical_file_body": canonical_body_match,
                "private_locator": f"private-operational-artifacts/{relative}",
                "used_by_tracks": sorted(memberships[semantic_digest]),
                "availability": "private_operational_record_not_distributed",
                "included_in_arxiv_source": False,
            }
        )
    return rows


def _public_derivative_row(path: Path, document: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {
        "record_class": "public_derivative",
        "release_name": name,
        "semantic_artifact_sha256": document["artifact_sha256"],
        "physical_file_sha256": _physical_sha256(path),
        "physical_file_bytes": path.stat().st_size,
        "availability": "included_in_arxiv_source",
        "included_in_arxiv_source": True,
    }


def _build_input_commitment(
    *,
    artifacts_root: Path,
    arena_path: Path,
    arena: Mapping[str, Any],
    uplift_path: Path,
    uplift: Mapping[str, Any],
    coverage_path: Path,
    coverage: Mapping[str, Any],
    authorization_path: Path,
    authorization: Mapping[str, Any],
    original_authorization_path: Path,
    original_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    response_memberships, source_memberships = _arm_references(arena, uplift)
    targets = set(response_memberships) | set(source_memberships)
    index = _index_target_records(artifacts_root, targets)
    response_rows = _private_record_rows(
        artifacts_root=artifacts_root,
        record_class="normalized_response",
        memberships=response_memberships,
        index=index,
    )
    source_rows = _private_record_rows(
        artifacts_root=artifacts_root,
        record_class="provider_source_record",
        memberships=source_memberships,
        index=index,
    )
    private_rows = sorted(
        response_rows + source_rows,
        key=lambda row: (row["record_class"], row["semantic_artifact_sha256"]),
    )
    public_rows = [
        _public_derivative_row(arena_path, arena, "current-frontier-corrected-arena.json"),
        _public_derivative_row(uplift_path, uplift, "current-frontier-corrected-uplift.json"),
        _public_derivative_row(coverage_path, coverage, "current-frontier-corrected-coverage.json"),
        _public_derivative_row(
            authorization_path,
            authorization,
            "reasoning-effort-v9-public-authorization.json",
        ),
    ]
    payload = {
        "schema_version": "flavourbench-current-input-commitment-v1",
        "artifact_role": "public_hash_commitment_to_private_current_development_inputs",
        "status": "complete_commitment_raw_payload_private",
        "scope": {
            "model_arena_sha256": arena["artifact_sha256"],
            "epicure_uplift_sha256": uplift["artifact_sha256"],
            "coverage_sha256": coverage["artifact_sha256"],
            "arena_candidate_comparisons": arena["observed"]["candidate_comparisons"],
            "uplift_candidate_pairs": uplift["observed"]["candidate_pairs"],
            "quality_judgments": 0,
            "synthetic_arms": 0,
        },
        "hash_semantics": {
            "semantic_artifact_sha256": (
                "The embedded logical artifact identifier referenced by the review pools. "
                "Normalized responses are canonical self-addressed JSON; legacy source "
                "records use a run-level identifier and are committed separately by byte hash."
            ),
            "physical_file_sha256": "SHA-256 of the exact stored file bytes",
            "private_locator": (
                "Repository-relative operational locator; not a public path or archive member"
            ),
        },
        "observed": {
            "distinct_response_records": len(response_rows),
            "distinct_source_records": len(source_rows),
            "private_record_commitments": len(private_rows),
            "public_derivatives": len(public_rows),
            "missing_target_records": 0,
            "conflicting_physical_records": 0,
        },
        "private_input_set_sha256": sha256_json(private_rows),
        "private_inputs": private_rows,
        "public_derivatives": public_rows,
        "private_governance_record": {
            "semantic_artifact_sha256": original_authorization["artifact_sha256"],
            "physical_file_sha256": _physical_sha256(original_authorization_path),
            "availability": "private_governance_record_not_distributed",
            "included_in_arxiv_source": False,
            "public_supersession_sha256": authorization["artifact_sha256"],
        },
        "claim_boundary": {
            "raw_prompts_responses_or_tool_payloads_included": False,
            "private_bytes_independently_reconstructable_from_this_commitment": False,
            "membership_and_stored_byte_identity_committed": True,
            "official_ranking_supported": False,
        },
    }
    return _address(payload)


def build_release_remediations(
    *,
    uplift_predecessor_path: Path,
    coverage_predecessor_path: Path,
    arena_path: Path,
    artifacts_root: Path,
    original_authorization_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    uplift_predecessor = _read_addressed(uplift_predecessor_path)
    coverage_predecessor = _read_addressed(coverage_predecessor_path)
    arena = _read_addressed(arena_path)
    original_authorization = _read_addressed(original_authorization_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    uplift = _build_uplift_successor(uplift_predecessor, artifacts_root)
    uplift_path = _write_addressed(output_dir, "frontier-uplift-policy-hold-successor", uplift)
    coverage = _build_coverage_successor(coverage_predecessor, uplift)
    coverage_path = _write_addressed(
        output_dir, "frontier-coverage-policy-hold-successor", coverage
    )
    authorization = _build_public_authorization(original_authorization_path, original_authorization)
    authorization_path = _write_addressed(
        output_dir, "reasoning-effort-v9-public-authorization", authorization
    )
    commitment = _build_input_commitment(
        artifacts_root=artifacts_root,
        arena_path=arena_path,
        arena=arena,
        uplift_path=uplift_path,
        uplift=uplift,
        coverage_path=coverage_path,
        coverage=coverage,
        authorization_path=authorization_path,
        authorization=authorization,
        original_authorization_path=original_authorization_path,
        original_authorization=original_authorization,
    )
    commitment_path = _write_addressed(
        output_dir, "current-development-private-input-commitment", commitment
    )
    return {
        "uplift": uplift_path,
        "coverage": coverage_path,
        "authorization": authorization_path,
        "input_commitment": commitment_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uplift-predecessor", required=True, type=Path)
    parser.add_argument("--coverage-predecessor", required=True, type=Path)
    parser.add_argument("--arena", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--original-authorization", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = build_release_remediations(
        uplift_predecessor_path=args.uplift_predecessor,
        coverage_predecessor_path=args.coverage_predecessor,
        arena_path=args.arena,
        artifacts_root=args.artifacts_root,
        original_authorization_path=args.original_authorization,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
