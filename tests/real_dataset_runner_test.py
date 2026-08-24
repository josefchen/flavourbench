from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from flavourbench.execution_policy import (
    MATCHED_TOOL_ACCESS_PROTOCOL_V1,
    ExecutionPolicy,
    verify_policy_document,
)
from flavourbench.frontier_contract_runner import (
    AdmissionDenied,
    ArtifactExposure,
    ContractCandidate,
    load_candidate_manifest,
    select_candidates,
)
from flavourbench.frontier_contract_runner import (
    append_ledger_event as append_frontier_ledger_event,
)
from flavourbench.live_smoke import (
    _worst_case_cost_usd,
    endpoint_execution_contract_sha256,
)
from flavourbench.provider import system_prompt_sha256, system_prompt_text
from flavourbench.real_dataset_runner import (
    CONDITIONS,
    EXECUTION_CONFIRMATION,
    KNOWN_PRIOR_OPENROUTER_EXPOSURE_USD,
    OPENROUTER_PRICE_DRIFT_RESERVE_MULTIPLIER,
    RESOLVED_CONSERVATIVE_EXPOSURE_BASIS,
    SOURCE_INCIDENT_RESOLUTION_CONFIRMATION,
    DatasetSource,
    DatasetState,
    IntegrityError,
    SourceIncidentResolution,
    _finalize_source,
    _load_dataset_sources,
    _parser,
    _subprocess_command,
    append_dataset_ledger_event,
    build_balanced_work_items,
    build_dataset_plan,
    build_source_incident_resolution_payload,
    dataset_ledger_state,
    derive_pair_forecast,
    load_dataset_ledger,
    normalise_source_responses,
    record_no_id_source_incident_resolution,
    run_real_exploratory_dataset,
    select_balanced_tasks,
)
from flavourbench.tool_contract import required_tool_contract


def _candidate(index: int, *, expensive: bool = False) -> ContractCandidate:
    model_id = f"vendor/model-{index}"
    provider_name = f"Provider {index}"
    endpoint = {
        "name": f"{provider_name} fixed",
        "provider_name": provider_name,
        "tag": f"provider-{index}/fixed",
        "model_id": model_id,
        "quantization": "fp8",
        "context_length": 100_000,
        "max_completion_tokens": 8_192,
        "pricing": {
            "prompt": "0.001" if expensive else "0.000001",
            "completion": "0.002" if expensive else "0.000002",
            "internal_reasoning": "0",
        },
        "supported_parameters": [
            "max_tokens",
            "response_format",
            "seed",
            "structured_outputs",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ],
    }
    return ContractCandidate(
        slot_id=f"slot-{index}",
        model_id=model_id,
        canonical_model_slug=f"{model_id}-20260715",
        model_name=f"Model {index}",
        provider_tag=endpoint["tag"],
        provider_name=provider_name,
        endpoint_sha256=f"{index:064x}",
        endpoint_execution_sha256=endpoint_execution_contract_sha256(endpoint),
        endpoint=endpoint,
    )


def test_qwencloud_work_item_dispatches_only_to_direct_runner(tmp_path: Path) -> None:
    base = _candidate(91)
    backend_contract = {
        "schema_version": "flavourbench-qwencloud-direct-endpoint-contract-v1",
        "season_eligible": False,
        "rank_eligible": False,
    }
    endpoint = {
        **base.endpoint,
        "model_id": "qwen3.7-max-2026-06-08",
        "provider_name": "qwencloud-direct",
        "tag": "qwencloud-direct",
        "supported_parameters": [
            "max_tokens",
            "seed",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ],
    }
    qwen = replace(
        base,
        model_id="qwen3.7-max-2026-06-08",
        canonical_model_slug="qwen3.7-max-2026-06-08",
        provider_tag="qwencloud-direct",
        provider_name="qwencloud-direct",
        endpoint=endpoint,
        endpoint_sha256=hashlib.sha256(
            json.dumps(endpoint, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        endpoint_execution_sha256=endpoint_execution_contract_sha256(endpoint),
        execution_backend="qwencloud_direct",
        backend_contract=backend_contract,
        backend_contract_sha256=hashlib.sha256(
            json.dumps(backend_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        cost_accounting_policy="provider_usage_times_frozen_rate_card",
    )
    policy = ExecutionPolicy(
        final_response_mode="plain_text",
        matched_planning=False,
        evidence_protocol=MATCHED_TOOL_ACCESS_PROTOCOL_V1,
        intermediate_reasoning_effort=None,
        final_reasoning_effort=None,
        tool_catalog_bytes_bound=24_000,
    )
    work_item = _workload(candidates=[qwen], policy=policy)[0]
    forecast = derive_pair_forecast(work_item, policy=policy)
    command = _subprocess_command(
        work_item,
        forecast=forecast,
        source_directory=tmp_path / "source",
        manifest_path=tmp_path / "qwen-manifest.json",
    )

    assert command[1:3] == ["-m", "flavourbench.direct_qwencloud_pair"]
    assert "--route-manifest" in command
    assert "--plain-text-final" in command
    assert "--intermediate-reasoning-effort" not in command
    assert "--final-reasoning-effort" not in command


def test_mutable_qwen38_dispatch_requires_explicit_alias_flag_and_full_ceiling(
    tmp_path: Path,
) -> None:
    base = _candidate(92)
    backend_contract = {
        "schema_version": "flavourbench-qwencloud-direct-endpoint-contract-v1",
        "identity_kind": "mutable_alias",
        "catalog_pinned_at_observation": True,
        "model_identity_label": "catalog_pinned_at_observation_not_a_frozen_model",
        "mutable_alias_execution_requires_explicit_opt_in": True,
        "official": False,
        "season_eligible": False,
        "rank_eligible": False,
    }
    endpoint = {
        **base.endpoint,
        "model_id": "qwen3.8-max",
        "provider_name": "qwencloud-direct",
        "tag": "qwencloud-direct",
        "max_completion_tokens": None,
        "pricing": {
            "prompt": "0",
            "completion": "0",
            "internal_reasoning": "0",
            "request": "0",
            "provider_rate_known": False,
            "operational_reservation_ceiling_usd": "2",
            "zero_values_mean": "unknown_cost_not_free",
        },
        "supported_parameters": [
            "max_tokens",
            "seed",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ],
    }
    qwen = replace(
        base,
        model_id="qwen3.8-max",
        canonical_model_slug="qwen3.8-max",
        provider_tag="qwencloud-direct",
        provider_name="qwencloud-direct",
        endpoint=endpoint,
        endpoint_sha256=hashlib.sha256(
            json.dumps(endpoint, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        endpoint_execution_sha256=endpoint_execution_contract_sha256(endpoint),
        execution_backend="qwencloud_direct",
        backend_contract=backend_contract,
        backend_contract_sha256=hashlib.sha256(
            json.dumps(backend_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        cost_accounting_policy="provider_usage_with_unpriced_budget_ceiling",
    )
    policy = ExecutionPolicy(
        final_response_mode="plain_text",
        matched_planning=False,
        evidence_protocol=MATCHED_TOOL_ACCESS_PROTOCOL_V1,
        intermediate_reasoning_effort=None,
        final_reasoning_effort=None,
        tool_catalog_bytes_bound=24_000,
    )
    work_item = _workload(candidates=[qwen], policy=policy)[0]
    forecast = derive_pair_forecast(work_item, policy=policy)
    command = _subprocess_command(
        work_item,
        forecast=forecast,
        source_directory=tmp_path / "source",
        manifest_path=tmp_path / "qwen38-manifest.json",
    )

    assert forecast.forecast_usd == Decimal("2")
    assert command.count("--allow-mutable-alias-exploratory") == 1
    assert "--intermediate-reasoning-effort" not in command
    assert "--final-reasoning-effort" not in command


def _workload(
    *, candidates: list[ContractCandidate] | None = None, policy: ExecutionPolicy | None = None
):
    selected, registry_sha = select_balanced_tasks(tasks_per_family=3, seed="test-seed")
    return build_balanced_work_items(
        manifest_sha256="a" * 64,
        task_registry_digest=registry_sha,
        selected_tasks=selected,
        candidates=candidates or [_candidate(index) for index in range(12)],
        execution_policy=policy or ExecutionPolicy(),
        assignments_per_model=10,
    )


def _empty_state(*, prior: Decimal = KNOWN_PRIOR_OPENROUTER_EXPOSURE_USD) -> DatasetState:
    return DatasetState(
        prior_verified_exposure_usd=prior,
        prior_effective_exposure_usd=prior,
        prior_active_reservation_usd=Decimal(0),
        dataset_actual_cost_usd=Decimal(0),
        dataset_source_exposure_usd=Decimal(0),
        unresolved_dataset_source_reserve_usd=Decimal(0),
        sources={},
        responses={},
        ledger=(),
        reservations={},
        finalizations={},
        orphan_reservation_usd=Decimal(0),
    )


def _result(condition: str, candidate: ContractCandidate, *, reconciled: bool = True) -> dict:
    generation_id = f"gen-{condition}"
    trace = (
        [
            {
                "round_index": 0,
                "name": "find_pairings",
                "arguments": {"ingredients": ["test"]},
                "result": "bounded test evidence",
                "result_sha256": "b" * 64,
                "latency_ms": 2,
                "is_error": False,
            }
        ]
        if condition == "epicure_on"
        else []
    )
    return {
        "answer_markdown": "A practical, calibrated test answer.",
        "output_json": {
            "answer_markdown": "A practical, calibrated test answer.",
            "ingredient_mentions": [],
            "constraints_addressed": [],
            "uncertainties": ["Requires tasting."],
        },
        "actual_model_id": candidate.canonical_model_slug,
        "actual_provider": candidate.provider_name,
        "generation_id": generation_id,
        "generation_ids": [generation_id],
        "generation_metadata": [
            {
                "generation_id": generation_id,
                "cost_micros": 100,
                "reconciled": reconciled,
            }
        ],
        "decoding": {},
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "reasoning_tokens": 0,
        "cost_micros": 100,
        "cost_reconciled": reconciled,
        "latency_ms": 20,
        "retries": 0,
        "finish_reason": "stop",
        "tool_trace": trace,
    }


def _source(work_item, *, on_reconciled: bool = True) -> DatasetSource:
    policy = work_item.execution_policy
    candidate = work_item.candidate
    supported = sorted(candidate.endpoint["supported_parameters"])
    generation_endpoint_sha256 = "6" * 64
    required_tool = required_tool_contract(policy)
    protocol_bundle = {
        "schema_version": "flavourbench-live-development-protocol-v1",
        "core_protocol_bundle": {
            "tool_registry_sha256": "e" * 64,
            "model_smoke_registry_sha256": work_item.manifest_sha256,
        },
        "run_binding": {
            "candidate_manifest_sha256": work_item.manifest_sha256,
            "dataset_work_item_id": work_item.work_item_id,
            "dataset_task_id": work_item.task.public_id,
            "prompt_sha256": work_item.task.prompt_sha256,
            "category": work_item.task.family,
            "canonical_model_slug": candidate.canonical_model_slug,
            "provider_tag": candidate.provider_tag,
            "endpoint_contract_sha256": generation_endpoint_sha256,
            "execution_policy_sha256": work_item.execution_policy_sha256,
            "final_response_mode": policy.final_response_mode,
            "matched_planning": policy.matched_planning,
            "max_intermediate_tokens": policy.max_intermediate_tokens,
            "required_tool_contract_max_intermediate_tokens": (
                policy.required_tool_contract_max_intermediate_tokens
            ),
            "evidence_protocol": policy.evidence_protocol,
            "required_tool_contract_protocol": policy.required_tool_contract_protocol,
            "required_tool_contract_sha256": required_tool["content_address"]["digest"],
            "intermediate_reasoning_effort": policy.intermediate_reasoning_effort,
            "final_reasoning_effort": policy.final_reasoning_effort,
        },
    }
    protocol_bundle_sha256 = hashlib.sha256(
        json.dumps(protocol_bundle, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_artifact = {
        "dataset_work_item_id": work_item.work_item_id,
        "dataset_task_id": work_item.task.public_id,
        "candidate_manifest_sha256": work_item.manifest_sha256,
        "prompt": work_item.task.prompt,
        "prompt_sha256": work_item.task.prompt_sha256,
        "category": work_item.task.family,
        "requested_model_id": candidate.model_id,
        "requested_provider": candidate.provider_tag,
        "run_purpose": "epicure_on_off_pair",
        "run_id": "run-test",
        "model_contract": {
            "id": candidate.model_id,
            "canonical_slug": candidate.canonical_model_slug,
        },
        "endpoint_contract": dict(candidate.endpoint),
        "endpoint_execution_contract_sha256": work_item.endpoint_execution_sha256,
        "execution_policy": policy.document(),
        "execution_policy_sha256": policy.sha256,
        "decoding": {
            "temperature": policy.decoding_temperature,
            "top_p": policy.decoding_top_p,
            "seed": policy.decoding_seed,
            "max_output_tokens": policy.max_output_tokens,
            "max_tool_rounds": policy.max_tool_rounds,
            "max_tool_calls_per_round": policy.max_tool_calls_per_round,
            "max_tool_calls_total": policy.max_tool_calls_total,
            "max_tool_result_bytes": policy.max_tool_result_bytes,
            "max_cumulative_tool_result_bytes": policy.max_cumulative_tool_result_bytes,
            "max_provider_attempts": policy.max_provider_attempts,
            "parallel_tool_calls_enforcement": "bounded_sequential_execution",
        },
        "frozen_generation_contract": {
            "supported_parameters": supported,
            "decoding_parameters": {
                "max_tokens": policy.max_output_tokens,
                "temperature": policy.decoding_temperature,
                "top_p": policy.decoding_top_p,
                "seed": policy.decoding_seed,
            },
            "expected_actual_model_id": candidate.canonical_model_slug,
            "expected_actual_provider_slug": candidate.provider_name,
            "endpoint_contract_sha256": generation_endpoint_sha256,
            "final_response_mode": policy.final_response_mode,
            "matched_planning": policy.matched_planning,
            "intermediate_max_tokens": policy.max_intermediate_tokens,
            "required_tool_contract_max_intermediate_tokens": (
                policy.required_tool_contract_max_intermediate_tokens
            ),
            "evidence_protocol": policy.evidence_protocol,
            "required_tool_contract_protocol": policy.required_tool_contract_protocol,
            "required_tool_contract_sha256": required_tool["content_address"]["digest"],
            "intermediate_reasoning_effort": policy.intermediate_reasoning_effort,
            "final_reasoning_effort": policy.final_reasoning_effort,
            "protocol_bundle_sha256": protocol_bundle_sha256,
        },
        "protocol_bundle": protocol_bundle,
        "protocol_bundle_sha256": protocol_bundle_sha256,
        "system_prompt_sha256": {condition: "c" * 64 for condition in CONDITIONS},
        "response_schema_sha256": "d" * 64,
        "epicure": {"release_id": "exploratory-test"},
        "epicure_tool_schema_sha256": "e" * 64,
        "mcp_trace_events": [],
        "results": {
            "epicure_off": _result("epicure_off", candidate),
            "epicure_on": _result("epicure_on", candidate, reconciled=on_reconciled),
        },
        "errors": {},
    }
    exposure = ArtifactExposure(
        path=Path("source.json"),
        artifact_sha256="f" * 64,
        status="complete",
        requested_model_id=candidate.model_id,
        requested_provider=candidate.provider_tag,
        candidate_manifest_sha256=work_item.manifest_sha256,
        actual_cost_usd=Decimal("0.0002"),
        forecast_usd=Decimal("0.1"),
        admitted_cap_usd=Decimal("0.1"),
        exposure_usd=Decimal("0.0002"),
        exposure_basis="fully_reconciled_actual",
        contract_passed=False,
    )
    return DatasetSource(
        path=Path("source.json"),
        artifact_sha256="f" * 64,
        work_item_id=work_item.work_item_id,
        artifact=source_artifact,
        exposure=exposure,
    )


def _no_id_incident_source(work_item, *, path: Path) -> DatasetSource:
    source = _source(work_item)
    artifact = dict(source.artifact)
    artifact["results"] = {"epicure_on": artifact["results"]["epicure_on"]}
    artifact["errors"] = {"epicure_off": "ProviderError: OpenRouter returned no final choice"}
    artifact["status"] = "failed_or_unreconciled"
    artifact["official"] = False
    artifact["rank_eligible"] = False
    artifact["research_result"] = False
    artifact["budget"] = {
        "actual_cost_micros": 100,
        "all_generation_costs_reconciled": True,
        "cap_usd": "0.1",
        "forecast_worst_case_usd": "0.1",
    }
    artifact["run_journal"] = {
        "filename": "journal-test.jsonl",
        "sha256": "1" * 64,
        "head_entry_sha256": "2" * 64,
        "finalized": True,
    }
    artifact["provider_attempt_events"] = [
        {
            "arm_id": "run-test:epicure_off",
            "event_type": "request_started",
            "attempt_id": "attempt-no-id",
            "payload_sha256": "3" * 64,
            "request_key_sha256": "4" * 64,
            "phase": "final",
            "generation_id": "",
        },
        {
            "arm_id": "run-test:epicure_off",
            "event_type": "response_received",
            "attempt_id": "attempt-no-id",
            "payload_sha256": "3" * 64,
            "request_key_sha256": "4" * 64,
            "phase": "final",
            "http_status": 200,
            "generation_id": "",
        },
    ]
    exposure = replace(
        source.exposure,
        path=path,
        status="failed_or_unreconciled",
        actual_cost_usd=Decimal("0.0001"),
        forecast_usd=Decimal("0.1"),
        admitted_cap_usd=Decimal("0.1"),
        exposure_usd=Decimal("0.1"),
        exposure_basis="failed_or_unreconciled_full_admitted_allowance",
    )
    return DatasetSource(
        path=path,
        artifact_sha256=source.artifact_sha256,
        work_item_id=source.work_item_id,
        artifact=artifact,
        exposure=exposure,
    )


def test_execution_policy_is_content_addressed_and_exports_exact_settings() -> None:
    policy = ExecutionPolicy()
    document = policy.document()

    assert verify_policy_document(document)
    assert document["content_address"]["digest"] == policy.sha256
    assert policy.settings_environment()["FLAVOURBENCH_MAX_OUTPUT_TOKENS"] == "1000"
    assert policy.settings_environment()["FLAVOURBENCH_MAX_TOOL_ROUNDS"] == "4"
    tampered = json.loads(json.dumps(document))
    tampered["limits"]["max_tool_rounds"] = 8
    assert not verify_policy_document(tampered)


def test_plain_text_policy_is_versioned_without_changing_legacy_structured_hash() -> None:
    legacy = ExecutionPolicy(max_output_tokens=2400, max_tool_rounds=4)
    plain = replace(legacy, final_response_mode="plain_text")

    assert legacy.sha256 == "25118579cf2e66cc8e15134bff2685c913b61d25b5c71b6fadc6162b9ac67136"
    assert legacy.document()["schema_version"] == "flavourbench-real-execution-policy-v1"
    assert "final_response_mode" not in legacy.document()
    assert plain.document()["schema_version"] == "flavourbench-real-execution-policy-v2"
    assert plain.document()["final_response_mode"] == "plain_text"
    assert plain.sha256 != legacy.sha256


def test_evidence_boundary_protocol_is_separately_versioned() -> None:
    common = {
        "max_output_tokens": 4096,
        "max_intermediate_tokens": 1024,
        "final_response_mode": "plain_text",
        "matched_planning": True,
        "intermediate_reasoning_effort": "low",
        "final_reasoning_effort": "low",
    }
    v1 = ExecutionPolicy(evidence_protocol="matched_evidence_v1", **common)
    v2 = ExecutionPolicy(evidence_protocol="matched_evidence_v2", **common)

    assert v1.document()["schema_version"] == "flavourbench-real-execution-policy-v6"
    assert v2.document()["schema_version"] == "flavourbench-real-execution-policy-v7"
    assert v1.sha256 != v2.sha256
    assert v2.document()["evidence_protocol"] == "matched_evidence_v2"


def test_matched_tool_access_changes_only_tool_availability() -> None:
    policy = ExecutionPolicy(
        max_output_tokens=8192,
        max_intermediate_tokens=2048,
        final_response_mode="plain_text",
        matched_planning=False,
        evidence_protocol=MATCHED_TOOL_ACCESS_PROTOCOL_V1,
        intermediate_reasoning_effort="minimal",
        final_reasoning_effort="low",
    )

    document = policy.document()
    assert document["schema_version"] == "flavourbench-real-execution-policy-v8"
    assert document["matched_planning"] is False
    assert system_prompt_text(
        "epicure_off", "plain_text", MATCHED_TOOL_ACCESS_PROTOCOL_V1
    ) == system_prompt_text("epicure_on", "plain_text", MATCHED_TOOL_ACCESS_PROTOCOL_V1)
    assert system_prompt_sha256(
        "epicure_off", "plain_text", MATCHED_TOOL_ACCESS_PROTOCOL_V1
    ) == system_prompt_sha256("epicure_on", "plain_text", MATCHED_TOOL_ACCESS_PROTOCOL_V1)


def test_matched_tool_access_rejects_staged_planning() -> None:
    policy = ExecutionPolicy(
        final_response_mode="plain_text",
        matched_planning=True,
        evidence_protocol=MATCHED_TOOL_ACCESS_PROTOCOL_V1,
    )

    with pytest.raises(ValueError, match="prohibits staged planning"):
        policy.validate()


def test_collection_cli_freezes_provider_attempt_bound() -> None:
    arguments = _parser().parse_args(["--max-provider-attempts", "2"])

    assert arguments.max_provider_attempts == 2


def test_selection_and_ten_pair_panel_schedule_are_deterministic_and_balanced() -> None:
    first_tasks, first_registry = select_balanced_tasks(tasks_per_family=3, seed="deterministic")
    second_tasks, second_registry = select_balanced_tasks(tasks_per_family=3, seed="deterministic")
    assert [task.public_id for task in first_tasks] == [task.public_id for task in second_tasks]
    assert first_registry == second_registry
    assert Counter(task.family for task in first_tasks) == {
        "substitution": 3,
        "composition": 3,
        "cookability": 3,
        "evidence": 3,
    }

    work_items = _workload()
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    for work_item in work_items:
        by_model[work_item.candidate.model_id][work_item.task.family] += 1
    assert len(work_items) == 120
    assert all(sum(counts.values()) == 10 for counts in by_model.values())
    assert all(sorted(counts.values()) == [2, 2, 3, 3] for counts in by_model.values())
    assert Counter(item.task.family for item in work_items) == {
        "substitution": 30,
        "composition": 30,
        "cookability": 30,
        "evidence": 30,
    }
    assert len({item.candidate.model_id for item in work_items[:12]}) == 12


def test_policy_hash_changes_work_item_identity_and_forecast_uses_lean_caps() -> None:
    lean = ExecutionPolicy()
    tighter = ExecutionPolicy(
        max_output_tokens=800,
        max_tool_rounds=3,
        max_tool_result_bytes=8_192,
        max_cumulative_tool_result_bytes=24_576,
        max_tool_calls_per_round=4,
        max_tool_calls_total=10,
    )
    lean_item = _workload(candidates=[_candidate(1)], policy=lean)[0]
    tighter_item = _workload(candidates=[_candidate(1)], policy=tighter)[0]

    assert lean_item.work_item_id != tighter_item.work_item_id
    assert lean_item.execution_policy_sha256 == lean.sha256
    lean_forecast = derive_pair_forecast(lean_item, policy=lean)
    tighter_forecast = derive_pair_forecast(tighter_item, policy=tighter)
    assert lean_forecast.request_bound == 6
    assert tighter_forecast.request_bound == 5
    assert tighter_forecast.forecast_usd < lean_forecast.forecast_usd


def test_runner_and_delegated_live_smoke_share_the_same_pair_cost_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = ExecutionPolicy()
    work_item = _workload(candidates=[_candidate(1)], policy=policy)[0]
    runner_forecast = derive_pair_forecast(work_item, policy=policy)
    monkeypatch.setattr(
        "flavourbench.live_smoke.get_settings",
        lambda: SimpleNamespace(
            max_tool_calls_total=policy.max_tool_calls_total,
            max_tool_rounds=policy.max_tool_rounds,
            max_tool_calls_per_round=policy.max_tool_calls_per_round,
            max_cumulative_tool_result_bytes=policy.max_cumulative_tool_result_bytes,
            max_tool_result_bytes=policy.max_tool_result_bytes,
            max_output_tokens=policy.max_output_tokens,
        ),
    )
    delegated_forecast = _worst_case_cost_usd(
        dict(work_item.candidate.endpoint),
        prompt=work_item.task.prompt,
        include_tool_contract=False,
        execution_policy=policy,
    )
    assert (
        delegated_forecast * OPENROUTER_PRICE_DRIFT_RESERVE_MULTIPLIER
        == runner_forecast.forecast_usd
    )


def test_complete_block_plan_never_admits_an_unbalanced_prefix() -> None:
    work_items = _workload(candidates=[_candidate(1, expensive=True)])
    plan = build_dataset_plan(
        work_items,
        state=_empty_state(),
        policy=ExecutionPolicy(),
        cap_usd=Decimal("100"),
        admission_fraction=Decimal("0.85"),
    )

    assert len(plan) == 10
    decisions = {item["decision"] for item in plan}
    assert len(decisions) == 1
    assert next(iter(decisions)).startswith("block_complete_workload_")


def test_dataset_ledger_is_hash_chained_and_detects_tampering(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    reservation = append_dataset_ledger_event(
        ledger,
        {
            "event_type": "reservation_created",
            "work_item_id": "a" * 64,
            "reserved_usd": "0.5",
        },
        recorded_at="2026-07-15T00:00:00Z",
    )
    append_dataset_ledger_event(
        ledger,
        {
            "event_type": "source_artifact_recorded",
            "work_item_id": "a" * 64,
            "reservation_entry_sha256": reservation["entry_sha256"],
        },
        recorded_at="2026-07-15T00:01:00Z",
    )
    entries = load_dataset_ledger(ledger)
    reservations, finalizations = dataset_ledger_state(entries)
    assert set(reservations) == {"a" * 64}
    assert set(finalizations) == {"a" * 64}

    lines = ledger.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["reserved_usd"] = "50"
    lines[0] = json.dumps(tampered)
    ledger.write_text("\n".join(lines) + "\n")
    with pytest.raises(IntegrityError, match="digest mismatch"):
        load_dataset_ledger(ledger)


def test_per_response_artifacts_are_append_only_and_require_reconciled_cost(
    tmp_path: Path,
) -> None:
    work_item = _workload(candidates=[_candidate(1)])[0]
    source = _source(work_item)
    responses, issues = normalise_source_responses(
        source,
        work_item,
        response_directory=tmp_path,
    )

    assert not issues
    assert {response.condition for response in responses} == set(CONDITIONS)
    assert all(response.path.exists() for response in responses)
    assert all(len(response.path.stem.rsplit("-", 1)[-1]) == 64 for response in responses)
    on = next(response for response in responses if response.condition == "epicure_on")
    assert on.tool_used
    artifact = json.loads(on.path.read_text())
    assert artifact["official"] is False
    assert artifact["rank_eligible"] is False
    assert artifact["research_result"] is False
    assert artifact["execution_policy_sha256"] == work_item.execution_policy_sha256

    same, same_issues = normalise_source_responses(
        source,
        work_item,
        response_directory=tmp_path,
    )
    assert not same_issues
    assert [item.path for item in same] == [item.path for item in responses]

    partial_dir = tmp_path / "partial"
    partial, partial_issues = normalise_source_responses(
        _source(work_item, on_reconciled=False),
        work_item,
        response_directory=partial_dir,
    )
    assert [response.condition for response in partial] == ["epicure_off"]
    assert "epicure_on_cost_not_fully_reconciled" in partial_issues


def test_no_id_resolution_is_append_only_conservative_and_normalizes_only_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_item = _workload(candidates=[_candidate(1)])[0]
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_path = source_root / f"source-{'f' * 12}.json"
    source_path.write_text("{}\n")
    source = _no_id_incident_source(work_item, path=source_path)
    ledger_path = tmp_path / "ledger.jsonl"
    reservation = append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "reservation_created",
            "work_item_id": work_item.work_item_id,
            "reserved_usd": "0.1",
        },
        recorded_at="2026-07-15T00:00:00Z",
    )
    incident = append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "execution_incident",
            "work_item_id": work_item.work_item_id,
            "reservation_entry_sha256": reservation["entry_sha256"],
            "incident": "generation_cost_unreconciled_reservation_retained",
            "source_artifact_sha256": source.artifact_sha256,
        },
        recorded_at="2026-07-15T00:01:00Z",
    )
    payload = build_source_incident_resolution_payload(
        source=source,
        reservation=reservation,
        incident=incident,
        affected_condition="epicure_off",
    )
    assert payload["unidentified_response"] == {
        "generation_id_known": False,
        "generation_ids": [],
        "generation_id_was_inferred": False,
        "affected_condition": "epicure_off",
    }
    assert payload["cost"]["provider_reconciled_actual_cost_usd"] == "0.0001"
    assert payload["cost"]["conservative_budget_exposure_usd"] == "0.1"
    assert payload["cost"]["provider_cost_exact_for_unidentified_response"] is False
    assert payload["resolution"]["safe_to_replay"] is False
    assert payload["resolution"]["normalizable_conditions"] == ["epicure_on"]

    scan = SimpleNamespace(
        artifacts=(source.exposure,),
        actual_cost_usd=source.exposure.actual_cost_usd,
    )
    monkeypatch.setattr(
        "flavourbench.real_dataset_runner.scan_live_smoke_artifacts",
        lambda *_args, **_kwargs: scan,
    )
    monkeypatch.setattr(
        "flavourbench.real_dataset_runner._verify_live_artifact",
        lambda _path: (source.artifact, source.artifact_sha256),
    )
    resolution_root = tmp_path / "resolutions"
    resolution, resolution_event = record_no_id_source_incident_resolution(
        ledger_path=ledger_path,
        source_directory=source_root,
        source_corrections_directory=None,
        resolution_directory=resolution_root,
        reservation_entry_sha256=reservation["entry_sha256"],
        incident_entry_sha256=incident["entry_sha256"],
        affected_condition="epicure_off",
        confirmation=SOURCE_INCIDENT_RESOLUTION_CONFIRMATION,
    )
    assert resolution.path.exists()
    assert resolution.provider_reconciled_actual_cost_usd == Decimal("0.0001")
    assert resolution.conservative_budget_exposure_usd == Decimal("0.1")
    assert resolution.normalizable_conditions == ("epicure_on",)
    assert resolution_event["provider_cost_exact_for_unidentified_response"] is False
    entry_count = len(load_dataset_ledger(ledger_path))
    same_resolution, same_event = record_no_id_source_incident_resolution(
        ledger_path=ledger_path,
        source_directory=source_root,
        source_corrections_directory=None,
        resolution_directory=resolution_root,
        reservation_entry_sha256=reservation["entry_sha256"],
        incident_entry_sha256=incident["entry_sha256"],
        affected_condition="epicure_off",
        confirmation=SOURCE_INCIDENT_RESOLUTION_CONFIRMATION,
    )
    assert same_resolution.artifact_sha256 == resolution.artifact_sha256
    assert same_event["entry_sha256"] == resolution_event["entry_sha256"]
    assert len(load_dataset_ledger(ledger_path)) == entry_count

    scanned_sources, actual, exposure, unresolved, resolutions = _load_dataset_sources(
        source_root,
        corrections_directory=None,
        resolution_directory=resolution_root,
        ledger=load_dataset_ledger(ledger_path),
    )
    resolved_source = scanned_sources[work_item.work_item_id]
    assert actual == Decimal("0.0001")
    assert exposure == Decimal("0.1")
    assert unresolved == 0
    assert resolved_source.exposure.exposure_basis == (RESOLVED_CONSERVATIVE_EXPOSURE_BASIS)
    assert set(resolutions) == {work_item.work_item_id}

    finalization, responses, issues = _finalize_source(
        ledger_path=ledger_path,
        runner_run_id="resume-test",
        reservation=reservation,
        source=resolved_source,
        work_item=work_item,
        response_directory=tmp_path / "responses",
        incident_resolution=resolution,
    )
    assert [response.condition for response in responses] == ["epicure_on"]
    assert "epicure_off_missing_or_invalid" in issues
    assert finalization["all_generation_costs_reconciled"] is False
    assert finalization["provider_cost_exact"] is False
    assert finalization["source_actual_cost_usd"] == "0.0001"
    assert finalization["source_budget_exposure_usd"] == "0.1"
    assert finalization["source_incident_resolution_sha256"] == (resolution.artifact_sha256)
    assert finalization["response_conditions"] == ["epicure_on"]


def test_unresolved_source_can_be_quarantined_without_replay_and_keeps_full_reserve(
    tmp_path: Path,
) -> None:
    work_item = _workload()[0]
    source = _no_id_incident_source(work_item, path=tmp_path / "source.json")
    ledger_path = tmp_path / "ledger.jsonl"
    reservation = append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "reservation_created",
            "runner_run_id": "quarantine-test",
            "work_item_id": work_item.work_item_id,
            "reserved_usd": "0.1",
        },
    )
    incident = append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "execution_incident",
            "runner_run_id": "quarantine-test",
            "work_item_id": work_item.work_item_id,
            "reservation_entry_sha256": reservation["entry_sha256"],
            "incident": "generation_cost_unreconciled_reservation_retained",
            "source_artifact_sha256": source.artifact_sha256,
        },
    )

    with pytest.raises(IntegrityError, match="exact immutable incident"):
        _finalize_source(
            ledger_path=ledger_path,
            runner_run_id="quarantine-test",
            reservation=reservation,
            source=source,
            work_item=work_item,
            response_directory=tmp_path / "responses-without-incident",
        )

    finalization, responses, issues = _finalize_source(
        ledger_path=ledger_path,
        runner_run_id="quarantine-test",
        reservation=reservation,
        source=source,
        work_item=work_item,
        response_directory=tmp_path / "responses",
        unresolved_incident=incident,
    )

    assert [response.condition for response in responses] == ["epicure_on"]
    assert "epicure_off_missing_or_invalid" in issues
    assert finalization["source_budget_exposure_usd"] == "0.1"
    assert finalization["all_generation_costs_reconciled"] is False
    assert finalization["provider_cost_exact"] is False
    assert finalization["unresolved_full_admitted_allowance_retained"] is True
    assert finalization["safe_to_replay"] is False
    assert finalization["unresolved_delivery_incident_entry_sha256"] == incident[
        "entry_sha256"
    ]


def test_resolved_ordinal_94_plan_skips_93_recovers_one_and_admits_only_95_to_120(
    tmp_path: Path,
) -> None:
    work_items = _workload()
    sources = {item.work_item_id: _source(item) for item in work_items[:93]}
    incident_source = _no_id_incident_source(
        work_items[93],
        path=tmp_path / "source.json",
    )
    incident_source = replace(
        incident_source,
        exposure=replace(
            incident_source.exposure,
            exposure_basis=RESOLVED_CONSERVATIVE_EXPOSURE_BASIS,
        ),
    )
    sources[work_items[93].work_item_id] = incident_source
    reservation = {
        "entry_sha256": "6" * 64,
        "work_item_id": work_items[93].work_item_id,
        "reserved_usd": "0.1",
    }
    resolution = SourceIncidentResolution(
        path=tmp_path / "resolution.json",
        artifact_sha256="7" * 64,
        work_item_id=work_items[93].work_item_id,
        source_artifact_sha256=incident_source.artifact_sha256,
        reservation_entry_sha256=reservation["entry_sha256"],
        incident_entry_sha256="8" * 64,
        affected_condition="epicure_off",
        normalizable_conditions=("epicure_on",),
        provider_reconciled_actual_cost_usd=Decimal("0.0001"),
        conservative_budget_exposure_usd=Decimal("0.1"),
        ledger_event_sha256="9" * 64,
    )
    state = DatasetState(
        prior_verified_exposure_usd=KNOWN_PRIOR_OPENROUTER_EXPOSURE_USD,
        prior_effective_exposure_usd=KNOWN_PRIOR_OPENROUTER_EXPOSURE_USD,
        prior_active_reservation_usd=Decimal(0),
        dataset_actual_cost_usd=Decimal("0.0187"),
        dataset_source_exposure_usd=Decimal("0.1186"),
        unresolved_dataset_source_reserve_usd=Decimal(0),
        sources=sources,
        responses={},
        ledger=(),
        reservations={work_items[93].work_item_id: reservation},
        finalizations={item.work_item_id: {} for item in work_items[:93]},
        orphan_reservation_usd=Decimal(0),
        incident_resolutions={work_items[93].work_item_id: resolution},
    )
    plan = build_dataset_plan(
        work_items,
        state=state,
        policy=ExecutionPolicy(),
        cap_usd=Decimal("100"),
        admission_fraction=Decimal("0.85"),
    )
    assert [item["decision"] for item in plan[:93]] == ["skip_finalized_work_item"] * 93
    assert plan[93]["ordinal"] == 94
    assert plan[93]["decision"] == "recover_existing_reconciled_source"
    assert [item["ordinal"] for item in plan[94:]] == list(range(95, 121))
    assert {item["decision"] for item in plan[94:]} == {"admit_sequentially"}


def test_execution_requires_exact_confirmation() -> None:
    with pytest.raises(AdmissionDenied, match=EXECUTION_CONFIRMATION):
        # The validation occurs before any manifest or provider access.
        run_real_exploratory_dataset(
            manifest_path="absent.json",
            execute=True,
            confirmation="wrong",
        )


def _real_manifest_path() -> Path:
    return (
        Path(__file__).parents[1] / "artifacts/manifests/"
        "flavourbench-openrouter-unranked-"
        "aaf43f1bd770df5f120d79b66058cfad5092d5fb950e80bd24fac6d1d2e9acb5.json"
    )


def test_dry_run_makes_no_subprocess_and_counts_shared_frontier_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frontier_ledger = tmp_path / "frontier-ledger.jsonl"
    append_frontier_ledger_event(
        frontier_ledger,
        {
            "event_type": "reservation_created",
            "model_id": "test/prior",
            "provider_tag": "test/fixed",
            "manifest_sha256": "1" * 64,
            "reserved_usd": "1.1279388",
        },
        recorded_at="2026-07-15T00:00:00Z",
    )

    def forbidden_subprocess(*_args, **_kwargs):
        raise AssertionError("dry-run attempted a provider subprocess")

    monkeypatch.setattr("flavourbench.real_dataset_runner.subprocess.run", forbidden_subprocess)
    summary, summary_path = run_real_exploratory_dataset(
        manifest_path=_real_manifest_path(),
        prior_artifact_directory=tmp_path / "prior",
        prior_corrections_directory=tmp_path / "prior-corrections",
        source_directory=tmp_path / "sources",
        source_corrections_directory=tmp_path / "source-corrections",
        response_directory=tmp_path / "responses",
        ledger_path=tmp_path / "dataset-ledger.jsonl",
        global_budget_lock_path=frontier_ledger,
        summary_directory=tmp_path / "summaries",
    )

    assert summary_path.exists()
    assert summary["provider_calls_made"] is False
    assert summary["workload"]["expected_pair_count"] == 120
    assert summary["budget"]["active_frontier_ledger_reservations_usd"] == ("1.1279388")
    assert Decimal(summary["budget"]["final_total_exposure_usd"]) == (
        KNOWN_PRIOR_OPENROUTER_EXPOSURE_USD + Decimal("1.1279388")
    )
    assert {outcome["decision"] for outcome in summary["outcomes"]} == {"admit_sequentially"}


def test_active_dataset_reservation_is_never_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_candidate_manifest(
        _real_manifest_path(),
        expected_digest=("aaf43f1bd770df5f120d79b66058cfad5092d5fb950e80bd24fac6d1d2e9acb5"),
    )
    selected, registry_sha = select_balanced_tasks(tasks_per_family=3)
    policy = ExecutionPolicy()
    workload = build_balanced_work_items(
        manifest_sha256=manifest["content_address"]["digest"],
        task_registry_digest=registry_sha,
        selected_tasks=selected,
        candidates=select_candidates(manifest),
        execution_policy=policy,
        assignments_per_model=10,
    )
    dataset_ledger = tmp_path / "dataset-ledger.jsonl"
    append_dataset_ledger_event(
        dataset_ledger,
        {
            "event_type": "reservation_created",
            "work_item_id": workload[0].work_item_id,
            "reserved_usd": "0.5",
        },
        recorded_at="2026-07-15T00:00:00Z",
    )

    def forbidden_subprocess(*_args, **_kwargs):
        raise AssertionError("active reservation was replayed")

    monkeypatch.setattr("flavourbench.real_dataset_runner.subprocess.run", forbidden_subprocess)
    summary, _ = run_real_exploratory_dataset(
        manifest_path=_real_manifest_path(),
        prior_artifact_directory=tmp_path / "prior",
        prior_corrections_directory=tmp_path / "prior-corrections",
        source_directory=tmp_path / "sources",
        source_corrections_directory=tmp_path / "source-corrections",
        response_directory=tmp_path / "responses",
        ledger_path=dataset_ledger,
        global_budget_lock_path=tmp_path / "frontier-ledger.jsonl",
        summary_directory=tmp_path / "summaries",
        execution_policy=policy,
        execute=True,
        confirmation=EXECUTION_CONFIRMATION,
    )

    assert summary["provider_calls_made"] is False
    assert summary["budget"]["execution_blocked_by_complete_workload_policy"] is True
    assert any(
        outcome["decision"] == "block_active_reservation_without_source"
        for outcome in summary["outcomes"]
    )
