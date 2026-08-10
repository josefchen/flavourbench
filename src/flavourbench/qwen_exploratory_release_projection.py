"""Publish a redacted QwenCloud operational projection from immutable run evidence.

This module performs no network operation. It verifies the two source runs, their
append-only journals, the successor recovery, and the governing ledger before
writing a content-addressed metadata projection. Raw prompts, answers, tool
arguments, tool results, and private human-governance records are not copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .frontier_manifest import verify_manifest_content_address
from .qwencloud_smoke_admission import (
    _live_source_sha256,
    load_ledger,
)
from .real_task_bank import sha256_json
from .run_journal import load_run_journal

SCHEMA_VERSION = "flavourbench-qwencloud-exploratory-operational-projection-v1"
MODEL_ID = "qwen3.8-max"
PROVIDER = "qwencloud-direct"

EXPECTED = {
    "predecessor_source_semantic": (
        "a9e863df14ef690fd194cb5689da3f3c947e615ac017d2264aff51b3b0d51a96"
    ),
    "predecessor_source_physical": (
        "ab8151db3ee65111ce460931a1e262d1f799748f53bbd2433a8e89e5e2f8e0d9"
    ),
    "predecessor_journal_physical": (
        "2db2e258607889d833cd3680ec573e6c49a6cd9bc882207cc76053b78b3b03f3"
    ),
    "predecessor_journal_head": (
        "04e94dce928307d7708dc361454fc375a55f773ca45acbc844a84a5820b7909e"
    ),
    "successor_source_semantic": (
        "124a38f1c9906dcb0fbea9fc21b8758f9b85ae5b1989dbe1dda00f761761459a"
    ),
    "successor_source_physical": (
        "52ccc3291f29d4fdab7da64ee219f04a343500715a4784616814b41ea6b8df56"
    ),
    "successor_journal_physical": (
        "e7c04218f1e910cee466e80d9ea9be917215db0a66a2e7022252d4acc6f9835d"
    ),
    "successor_journal_head": (
        "4e1e511c2a069221563235a0fad56ecfe475ec8bb65880835c95fcb0a17d1954"
    ),
    "recovery_semantic": (
        "7bb5f1392a2422437edc138b14940cd92736caa6bc6328acbf4b2dd73e8d479a"
    ),
    "recovery_physical": (
        "50d7cb73009916767e1cf0978cc48b25de074c715b7f378275d16a84d6ed94f5"
    ),
    "ledger_physical": (
        "2437b412ab18d90e82f199068a42eb5c49939e8cfff936393e0e21206e15cc63"
    ),
    "ledger_head": (
        "e29348d968cf757845140ab011030907c91e6394a79486d7e06b3d95e65c0375"
    ),
    "route_semantic": (
        "1e646c713945f9e492be99a49daae139cf8c6b799cbedeb5fc197d285771f0d4"
    ),
    "route_physical": (
        "cb93978316da6bb6c6350dfcfe0bb817626b94553c2c02ae4d2a511aac0d6f3e"
    ),
    "preflight_semantic": (
        "22580b7bd1c38039f9bbdfe2061974bf0944819ea4063a431381caf39d48b1d5"
    ),
    "preflight_physical": (
        "697bd701dafda3da6eb6fe85e55af2bd21d81b7e20201436d1a3271145471ced"
    ),
    "go_template_semantic": (
        "324fa0408946eb483a7103c305f4a394ebd6d24f6aee79643cd832500c57dea1"
    ),
    "go_template_physical": (
        "fc60f60a6a5dc0c133fdeafa16033f45f9fee6d145416366805651ae9d2af2e1"
    ),
    "authorization_semantic": (
        "93ac04e043027e333e283a966accb995ffb6b9255eb810b821495d2ae4f343cf"
    ),
    "authorization_physical": (
        "3ece94994c2873c38c9684aee2a375622f876e6dd9b44a6426d08c95b3bd1887"
    ),
}


class QwenProjectionError(RuntimeError):
    """Qwen operational evidence failed the public-projection contract."""


def _physical_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise QwenProjectionError(f"not a regular evidence file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_physical(path: Path, expected: str, label: str) -> None:
    if _physical_sha256(path) != expected:
        raise QwenProjectionError(f"{label} physical digest mismatch")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _physical_sha256(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QwenProjectionError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise QwenProjectionError(f"{label} must be a JSON object")
    return value


def _verify_addressed(
    path: Path,
    *,
    semantic: str,
    physical: str,
    label: str,
    ascii_canonical: bool = False,
) -> dict[str, Any]:
    _require_physical(path, physical, label)
    value = _read_json(path, label)
    digest = value.get("artifact_sha256")
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    calculated = _live_source_sha256(body) if ascii_canonical else sha256_json(body)
    if digest != semantic or calculated != semantic:
        raise QwenProjectionError(f"{label} semantic digest mismatch")
    return value


def _verify_route(path: Path) -> dict[str, Any]:
    _require_physical(path, EXPECTED["route_physical"], "route manifest")
    value = _read_json(path, "route manifest")
    address = value.get("content_address")
    if (
        not verify_manifest_content_address(value)
        or not isinstance(address, Mapping)
        or address.get("digest") != EXPECTED["route_semantic"]
    ):
        raise QwenProjectionError("route manifest content address mismatch")
    return value


def _event_counts(source: Mapping[str, Any]) -> Counter[str]:
    events = source.get("provider_attempt_events")
    if not isinstance(events, list):
        raise QwenProjectionError("source has no provider-attempt ledger")
    return Counter(
        str(event.get("event_type"))
        for event in events
        if isinstance(event, Mapping)
    )


def _results(source: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    value = source.get("results")
    if not isinstance(value, Mapping):
        raise QwenProjectionError("source results are not condition keyed")
    if any(not isinstance(item, Mapping) for item in value.values()):
        raise QwenProjectionError("source contains an invalid result arm")
    return value  # type: ignore[return-value]


def _generation_totals(results: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    generation_ids: set[str] = set()
    for result in results.values():
        metadata = result.get("generation_metadata")
        ids = result.get("generation_ids")
        if not isinstance(metadata, list) or not isinstance(ids, list):
            raise QwenProjectionError("result arm lacks generation accounting")
        for item in metadata:
            if not isinstance(item, Mapping):
                raise QwenProjectionError("generation metadata item is invalid")
            for field in totals:
                source_field = {
                    "prompt_tokens": "tokens_prompt",
                    "completion_tokens": "tokens_completion",
                    "reasoning_tokens": "reasoning_tokens",
                }[field]
                totals[field] += int(item.get(source_field) or 0)
            generation_ids.add(str(item.get("generation_id") or ""))
        if generation_ids.intersection({""}) or set(map(str, ids)) != {
            str(item.get("generation_id") or "") for item in metadata
        }:
            raise QwenProjectionError("generation IDs and metadata do not agree")
    totals["provider_generation_responses"] = len(generation_ids)
    return totals


def _commitment(
    role: str,
    *,
    physical_sha256: str,
    semantic_sha256: str | None = None,
    chain_head_sha256: str | None = None,
    distributed: bool,
    sensitivity: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "role": role,
        "physical_sha256": physical_sha256,
        "distributed_in_arxiv_source": distributed,
        "sensitivity": sensitivity,
    }
    if semantic_sha256 is not None:
        value["semantic_sha256"] = semantic_sha256
    if chain_head_sha256 is not None:
        value["chain_head_sha256"] = chain_head_sha256
    return value


def build_qwen_projection(
    *,
    predecessor_source_path: Path,
    predecessor_journal_path: Path,
    successor_source_path: Path,
    successor_journal_path: Path,
    recovery_path: Path,
    ledger_path: Path,
    route_path: Path,
    preflight_path: Path,
    go_template_path: Path,
    authorization_path: Path,
    output_dir: Path,
) -> Path:
    """Verify immutable evidence and write one deterministic public projection."""

    predecessor = _verify_addressed(
        predecessor_source_path,
        semantic=EXPECTED["predecessor_source_semantic"],
        physical=EXPECTED["predecessor_source_physical"],
        label="predecessor source",
        ascii_canonical=True,
    )
    successor = _verify_addressed(
        successor_source_path,
        semantic=EXPECTED["successor_source_semantic"],
        physical=EXPECTED["successor_source_physical"],
        label="successor source",
        ascii_canonical=True,
    )
    recovery = _verify_addressed(
        recovery_path,
        semantic=EXPECTED["recovery_semantic"],
        physical=EXPECTED["recovery_physical"],
        label="zero-call recovery",
    )
    route = _verify_route(route_path)
    preflight = _verify_addressed(
        preflight_path,
        semantic=EXPECTED["preflight_semantic"],
        physical=EXPECTED["preflight_physical"],
        label="successor preflight",
    )
    go_template = _verify_addressed(
        go_template_path,
        semantic=EXPECTED["go_template_semantic"],
        physical=EXPECTED["go_template_physical"],
        label="successor GO template",
    )
    authorization = _verify_addressed(
        authorization_path,
        semantic=EXPECTED["authorization_semantic"],
        physical=EXPECTED["authorization_physical"],
        label="private human-PI authorization",
    )

    _require_physical(
        predecessor_journal_path,
        EXPECTED["predecessor_journal_physical"],
        "predecessor journal",
    )
    predecessor_journal = load_run_journal(predecessor_journal_path)
    _require_physical(
        successor_journal_path,
        EXPECTED["successor_journal_physical"],
        "successor journal",
    )
    successor_journal = load_run_journal(successor_journal_path)
    _require_physical(ledger_path, EXPECTED["ledger_physical"], "exploratory ledger")
    ledger = load_ledger(ledger_path)

    predecessor_results = _results(predecessor)
    successor_results = _results(successor)
    predecessor_events = _event_counts(predecessor)
    successor_events = _event_counts(successor)
    totals = _generation_totals(successor_results)
    successor_traces = successor.get("mcp_trace_events")
    if not isinstance(successor_traces, list):
        raise QwenProjectionError("successor has no MCP trace")
    successful_tools = [
        str(trace.get("name") or "")
        for trace in successor_traces
        if isinstance(trace, Mapping) and trace.get("is_error") is False
    ]

    if (
        predecessor.get("status") != "failed_or_unreconciled"
        or set(predecessor_results) != {"epicure_off"}
        or predecessor_events
        != Counter(
            {
                "request_started": 5,
                "response_received": 4,
                "accounting_reconciled": 4,
                "request_rejected": 1,
                "mcp_session_started": 1,
                "mcp_session_attested": 1,
            }
        )
        or predecessor.get("mcp_trace_events") != []
        or predecessor_journal[-1].get("entry_sha256")
        != EXPECTED["predecessor_journal_head"]
        or len(predecessor_journal) != 18
    ):
        raise QwenProjectionError("predecessor reliability evidence changed")
    if (
        successor.get("status") != "complete_unpriced_budget_ceiling"
        or set(successor_results) != {"epicure_off", "epicure_on"}
        or any(result.get("finish_reason") != "stop" for result in successor_results.values())
        or any(result.get("actual_model_id") != MODEL_ID for result in successor_results.values())
        or any(result.get("actual_provider") != PROVIDER for result in successor_results.values())
        or any(int(result.get("retries") or 0) != 0 for result in successor_results.values())
        or successor.get("errors") != {}
        or successor_events
        != Counter(
            {
                "request_started": 6,
                "response_received": 6,
                "accounting_reconciled": 6,
                "mcp_call_started": 2,
                "mcp_call_completed": 2,
                "mcp_session_started": 1,
                "mcp_session_attested": 1,
            }
        )
        or totals
        != {
            "prompt_tokens": 12413,
            "completion_tokens": 14574,
            "reasoning_tokens": 9332,
            "provider_generation_responses": 6,
        }
        or successful_tools != ["list_targets", "flavour_correlations"]
        or successor_journal[-1].get("entry_sha256")
        != EXPECTED["successor_journal_head"]
        or len(successor_journal) != 28
    ):
        raise QwenProjectionError("successor operational evidence changed")

    successor_budget = successor.get("budget")
    predecessor_budget = predecessor.get("budget")
    recovery_counts = recovery.get("observed_counts")
    if (
        not isinstance(successor_budget, Mapping)
        or not isinstance(predecessor_budget, Mapping)
        or successor_budget.get("provider_cost_known") is not False
        or predecessor_budget.get("provider_cost_known") is not False
        or successor_budget.get("retained_exposure_usd") != "2"
        or predecessor_budget.get("retained_exposure_usd") != "2"
        or recovery.get("source_artifact_sha256")
        != EXPECTED["successor_source_semantic"]
        or recovery.get("journal_sha256") != EXPECTED["successor_journal_physical"]
        or recovery.get("provider_calls_made") is not False
        or recovery.get("epicure_calls_made") is not False
        or recovery.get("source_mutated") is not False
        or not isinstance(recovery_counts, Mapping)
        or recovery_counts.get("response_arms") != 2
        or recovery_counts.get("successful_real_epicure_calls") != 2
        or recovery_counts.get("synthetic_arms") != 0
        or recovery_counts.get("quality_comparisons_authorized") != 0
    ):
        raise QwenProjectionError("recovery or budget evidence changed")

    route_address = route["content_address"]
    if (
        successor.get("candidate_manifest_sha256") != EXPECTED["route_semantic"]
        or preflight.get("candidate_manifest_sha256") != EXPECTED["route_semantic"]
        or go_template.get("artifact_sha256") != EXPECTED["go_template_semantic"]
        or preflight.get("go_template_sha256") != EXPECTED["go_template_semantic"]
        or authorization.get("go_template_sha256") != EXPECTED["go_template_semantic"]
        or authorization.get("preflight_artifact_sha256")
        != EXPECTED["preflight_semantic"]
        or recovery.get("preflight_artifact_sha256") != EXPECTED["preflight_semantic"]
        or recovery.get("go_template_sha256") != EXPECTED["go_template_semantic"]
        or recovery.get("human_pi_authorization_sha256")
        != EXPECTED["authorization_semantic"]
        or route_address.get("digest") != EXPECTED["route_semantic"]
        or ledger[-1].get("entry_sha256") != EXPECTED["ledger_head"]
        or ledger[-1].get("safe_to_replay") is not False
        or ledger[-1].get("source_artifact_sha256")
        != EXPECTED["successor_source_semantic"]
    ):
        raise QwenProjectionError("execution authorization chain changed")

    identity_checks = (predecessor, successor)
    if any(
        source.get("requested_model_id") != MODEL_ID
        or source.get("requested_provider") != PROVIDER
        or source.get("model_identity_status")
        != "catalog_pinned_at_observation_not_a_frozen_model"
        or source.get("official") is not False
        or source.get("rank_eligible") is not False
        or source.get("research_result") is not False
        for source in identity_checks
    ):
        raise QwenProjectionError("model identity or admission boundary changed")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "public_redacted_operational_metadata_projection",
        "status": "verified_exploratory_unranked_post_freeze_addendum",
        "recorded_at": recovery["recorded_at"],
        "model_identity": {
            "display_name": "Qwen 3.8 Max",
            "requested_model_id": MODEL_ID,
            "returned_model_ids": [MODEL_ID],
            "provider": PROVIDER,
            "identity_kind": "mutable_alias",
            "catalog_observed_at": successor["catalog_observed_at"],
            "catalog_pinned_at_observation": True,
            "frozen_release": False,
            "automatic_provider_fallback": False,
        },
        "task_scope": {
            "task_id": successor["dataset_task_id"],
            "task_family": successor["category"],
            "prompt_distributed": False,
            "answer_text_distributed": False,
            "tool_arguments_or_results_distributed": False,
        },
        "predecessor_reliability_run": {
            "status": predecessor["status"],
            "delivered_response_arms": 1,
            "requested_response_arms": 2,
            "provider_requests": predecessor_events["request_started"],
            "provider_responses": predecessor_events["response_received"],
            "provider_rejections": predecessor_events["request_rejected"],
            "rejected_http_statuses": [400],
            "successful_real_epicure_calls": 0,
            "failure_interpretation": (
                "The Epicure-on arm stopped at the first tool round with HTTP 400. "
                "The recorded tool-choice diagnosis is an inference, not a provider attestation."
            ),
            "retained_budget_ceiling_usd": "2",
        },
        "successor_operational_run": {
            "status": successor["status"],
            "delivered_response_arms": 2,
            "completed_off_on_pairs": 1,
            "finish_reasons": ["stop", "stop"],
            "provider_requests": successor_events["request_started"],
            "provider_responses": successor_events["response_received"],
            "provider_rejections": successor_events["request_rejected"],
            "provider_retries": 0,
            "returned_stage_usage": {
                **totals,
                "normalized_result_reasoning_token_fields": 0,
            },
            "real_epicure_calls": 2,
            "successful_real_epicure_calls": 2,
            "epicure_tool_names": successful_tools,
            "synthetic_arms": 0,
            "retained_budget_ceiling_usd": "2",
        },
        "combined_reliability_accounting": {
            "provider_requests": 11,
            "provider_responses": 10,
            "provider_rejections": 1,
            "cumulative_retained_budget_ceiling_usd": "4",
            "provider_charge_available": False,
            "provider_cost_reconciled": False,
            "recorded_zero_cost_means": "unknown_not_free",
            "eligible_for_numeric_cost_plot": False,
        },
        "epicure_runtime": {
            "release_id": successor["epicure"]["release_id"],
            "application_sha256": successor["epicure"]["application_sha256"],
            "bundle_sha256": successor["epicure"]["bundle_sha256"],
            "lineage_status": "exploratory_unmatched_runtime",
        },
        "zero_call_recovery": {
            "decision": recovery["recovery_decision"],
            "root_cause_scope": "local_digest_validator_only",
            "provider_calls_made": False,
            "epicure_calls_made": False,
            "source_mutated": False,
            "safe_to_replay": False,
        },
        "source_commitments": [
            _commitment(
                "failed_predecessor_source",
                semantic_sha256=EXPECTED["predecessor_source_semantic"],
                physical_sha256=EXPECTED["predecessor_source_physical"],
                distributed=False,
                sensitivity="private_raw_prompt_answer_and_trace",
            ),
            _commitment(
                "failed_predecessor_journal",
                physical_sha256=EXPECTED["predecessor_journal_physical"],
                chain_head_sha256=EXPECTED["predecessor_journal_head"],
                distributed=False,
                sensitivity="private_append_only_execution_journal",
            ),
            _commitment(
                "successful_successor_source",
                semantic_sha256=EXPECTED["successor_source_semantic"],
                physical_sha256=EXPECTED["successor_source_physical"],
                distributed=False,
                sensitivity="private_raw_prompt_answer_and_trace",
            ),
            _commitment(
                "successful_successor_journal",
                physical_sha256=EXPECTED["successor_journal_physical"],
                chain_head_sha256=EXPECTED["successor_journal_head"],
                distributed=False,
                sensitivity="private_append_only_execution_journal",
            ),
            _commitment(
                "exploratory_budget_ledger",
                physical_sha256=EXPECTED["ledger_physical"],
                chain_head_sha256=EXPECTED["ledger_head"],
                distributed=False,
                sensitivity="private_append_only_governance_ledger",
            ),
            _commitment(
                "successor_route_manifest",
                semantic_sha256=EXPECTED["route_semantic"],
                physical_sha256=EXPECTED["route_physical"],
                distributed=False,
                sensitivity="non_distributed_execution_contract",
            ),
            _commitment(
                "successor_preflight",
                semantic_sha256=EXPECTED["preflight_semantic"],
                physical_sha256=EXPECTED["preflight_physical"],
                distributed=False,
                sensitivity="non_distributed_execution_contract",
            ),
            _commitment(
                "successor_go_template",
                semantic_sha256=EXPECTED["go_template_semantic"],
                physical_sha256=EXPECTED["go_template_physical"],
                distributed=False,
                sensitivity="private_governance_record",
            ),
            _commitment(
                "successor_human_pi_authorization",
                semantic_sha256=EXPECTED["authorization_semantic"],
                physical_sha256=EXPECTED["authorization_physical"],
                distributed=False,
                sensitivity="private_governance_record",
            ),
            _commitment(
                "zero_call_recovery",
                semantic_sha256=EXPECTED["recovery_semantic"],
                physical_sha256=EXPECTED["recovery_physical"],
                distributed=True,
                sensitivity="public_redacted_recovery_metadata",
            ),
        ],
        "claim_boundary": {
            "post_freeze_operational_addendum_only": True,
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "season_eligible": False,
            "quality_judgments": 0,
            "leaderboard_comparisons_authorized": 0,
            "included_in_current_uplift_pool": False,
            "included_in_current_model_arena_pool": False,
            "included_in_any_quality_fit": False,
            "changes_186_pair_uplift_count": False,
            "changes_915_comparison_arena_count": False,
            "supports_model_quality_claim": False,
            "supports_epicure_uplift_claim": False,
        },
    }
    addressed = {**payload, "artifact_sha256": sha256_json(payload)}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        "qwencloud-exploratory-operational-projection-"
        f"{addressed['artifact_sha256']}.json"
    )
    rendered = json.dumps(addressed, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(rendered, encoding="utf-8")
    written = _read_json(path, "written public projection")
    body = {key: value for key, value in written.items() if key != "artifact_sha256"}
    if written.get("artifact_sha256") != sha256_json(body):
        raise QwenProjectionError("written projection failed content-address verification")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-source", type=Path, required=True)
    parser.add_argument("--predecessor-journal", type=Path, required=True)
    parser.add_argument("--successor-source", type=Path, required=True)
    parser.add_argument("--successor-journal", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--go-template", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    path = build_qwen_projection(
        predecessor_source_path=args.predecessor_source,
        predecessor_journal_path=args.predecessor_journal,
        successor_source_path=args.successor_source,
        successor_journal_path=args.successor_journal,
        recovery_path=args.recovery,
        ledger_path=args.ledger,
        route_path=args.route,
        preflight_path=args.preflight,
        go_template_path=args.go_template,
        authorization_path=args.authorization,
        output_dir=args.output_dir,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
