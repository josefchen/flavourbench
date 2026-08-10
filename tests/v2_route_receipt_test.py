from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

import flavourbench.reasoning_effort_sensitivity as sensitivity
from flavourbench.real_dataset_runner import (
    ResponseArtifact,
    append_dataset_ledger_event,
)
from flavourbench.run_journal import RunJournal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPOSITORY_ROOT / (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-sensitivity-v2-route-validation/"
    "reasoning-effort-v2-route-validation-plan-"
    "65b64747cbbecf116e3756f69bdbc7c0ccaf1a99a446d3352faa79b432e14e0f.json"
)
ASSETS_PATH = REPOSITORY_ROOT / (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-sensitivity-v2-route-validation/final-65b64747/"
    "reasoning-effort-v2-route-runner-assets-"
    "67ae96e905c36a1a6cdb2c208190bc6f9456dc5f4da9832cbcdc352e3f6ed805.json"
)


def _content_address(payload: dict) -> dict:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return {**unhashed, "artifact_sha256": sensitivity._sha256(unhashed)}


def _summary_content_address(payload: dict) -> dict:
    digest = sensitivity._sha256(payload)
    return {
        **payload,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }


def _replace_option(command: list[str], option: str, value: Path) -> None:
    command[command.index(option) + 1] = str(value)


def _isolated_assets(tmp_path: Path) -> tuple[dict, dict]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assets = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))
    assets.pop("artifact_sha256")
    for variant in assets["variants"]:
        run_root = tmp_path / "runs" / variant["variant_id"]
        variant["run_root"] = str(run_root)
        dry = variant["dry_run_command"]
        _replace_option(dry, "--source-directory", run_root / "source")
        _replace_option(dry, "--response-directory", run_root / "responses")
        _replace_option(dry, "--ledger", run_root / "ledger.jsonl")
        _replace_option(dry, "--summary-directory", run_root / "summaries")
        _replace_option(dry, "--route-validation-plan", PLAN_PATH)
        variant["manifest"] = str(REPOSITORY_ROOT / variant["manifest"])
        variant["live_command"] = [
            *dry,
            "--execute",
            "--confirm",
            "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET",
        ]
    return plan, _content_address(assets)


def _provider_event(
    *,
    arm_id: str,
    attempt_id: str,
    generation_id: str,
    event_type: str,
    model: str,
    provider: str,
) -> dict:
    metadata: dict = {}
    http_status = None
    recorded_generation_id = ""
    if event_type == "response_received":
        http_status = 200
        recorded_generation_id = generation_id
        metadata = {
            "response_model": model,
            "finish_reason": "stop",
            "native_finish_reason": "completed",
            "openrouter_cache_status": "",
            "cloudflare_cache_status": "MISS",
            "response_envelope": {
                "classification": "chat_completions",
                "accepted_chat_completion": True,
                "error_code": None,
                "error_type": None,
                "provider": provider,
            },
        }
    elif event_type == "accounting_reconciled":
        http_status = 200
        recorded_generation_id = generation_id
        metadata = {
            "generation_id": generation_id,
            "cost_micros": 10,
            "model": model,
            "provider": provider,
            "reconciled": True,
        }
    return {
        "arm_id": arm_id,
        "attempt_id": attempt_id,
        "attempt_index": 0,
        "error_type": "",
        "event_type": event_type,
        "generation_id": recorded_generation_id,
        "http_status": http_status,
        "metadata": metadata,
        "payload_sha256": sensitivity._sha256(
            {"attempt_id": attempt_id, "event_type": event_type}
        ),
        "phase": "final",
        "request_key_sha256": sensitivity._sha256({"attempt_id": attempt_id}),
    }


def _write_variant_records(
    *,
    plan: dict,
    assets: dict,
    variant_id: str,
    reused_generation_prefix: str | None = None,
) -> dict[str, ResponseArtifact]:
    asset = next(item for item in assets["variants"] if item["variant_id"] == variant_id)
    planned = next(
        item
        for item in plan["route_validation"]["work_items"]
        if item["variant_id"] == variant_id
    )
    run_root = Path(asset["run_root"])
    source_root = run_root / "source"
    response_root = run_root / "responses"
    summary_root = run_root / "summaries"
    source_root.mkdir(parents=True)
    response_root.mkdir(parents=True)
    summary_root.mkdir(parents=True)

    run_id = f"route-{variant_id}"
    generation_prefix = reused_generation_prefix or variant_id
    attempt_events: list[dict] = []
    results: dict[str, dict] = {}
    for index, condition in enumerate(("epicure_off", "epicure_on"), start=1):
        attempt_id = f"v2-{variant_id}-attempt-{index}"
        generation_id = f"v2-{generation_prefix}-generation-{index}"
        arm_id = f"{run_id}:{condition}"
        for event_type in (
            "request_started",
            "response_received",
            "accounting_reconciled",
        ):
            attempt_events.append(
                _provider_event(
                    arm_id=arm_id,
                    attempt_id=attempt_id,
                    generation_id=generation_id,
                    event_type=event_type,
                    model=planned["canonical_model_slug"],
                    provider=planned["actual_provider_name"],
                )
            )
        tool_trace = (
            [
                {
                    "arguments": {"ingredients": ["miso", "anchovy"]},
                    "is_error": False,
                    "latency_ms": 1,
                    "name": "find_pairings",
                    "result": "verified fixture result",
                    "result_sha256": sensitivity._sha256("verified fixture result"),
                    "round_index": 0,
                }
            ]
            if condition == "epicure_on"
            else []
        )
        results[condition] = {
            "actual_model_id": planned["canonical_model_slug"],
            "actual_provider": planned["actual_provider_name"],
            "answer_markdown": f"Complete real-output fixture for {condition}.",
            "finish_reason": "stop",
            "generation_id": generation_id,
            "generation_ids": [generation_id],
            "generation_metadata": [
                {
                    "generation_id": generation_id,
                    "cost_micros": 10,
                    "model": planned["canonical_model_slug"],
                    "provider": planned["actual_provider_name"],
                    "reconciled": True,
                }
            ],
            "intermediate_outputs": [],
            "cost_reconciled": True,
            "cost_micros": 10,
            "structured_output_valid": None,
            "tool_trace": tool_trace,
        }
    mcp_events = [
        {
            "arm_id": f"{run_id}:epicure_on",
            **results["epicure_on"]["tool_trace"][0],
        }
    ]

    journal = RunJournal.create(
        source_root,
        run_id=run_id,
        metadata={
            "candidate_manifest_sha256": asset["manifest_sha256"],
            "category": planned["task_family"],
            "contract_only": False,
            "dataset_task_id": planned["task_id"],
            "dataset_work_item_id": planned["work_item_id"],
            "epicure_conditions": ["epicure_off", "epicure_on"],
            "prompt_sha256": planned["prompt_sha256"],
            "requested_model_id": planned["model_id"],
            "requested_provider": planned["provider_endpoint"],
            "run_class": "engineering_live_smoke",
        },
    )
    for event in attempt_events:
        journal.append("provider_attempt", event)
    for event in mcp_events:
        journal.append("mcp_trace", event)
    descriptor = journal.finalize(
        {
            "status": "generation_complete",
            "condition_names": ["epicure_off", "epicure_on"],
            "error_keys": [],
            "generation_ids": sorted(
                result["generation_id"] for result in results.values()
            ),
            "actual_cost_micros": 20,
            "all_generation_costs_reconciled": True,
        }
    )

    source = {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": run_id,
        "status": "complete",
        "run_class": "engineering_live_smoke",
        "run_purpose": "epicure_on_off_pair",
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "candidate_manifest_sha256": asset["manifest_sha256"],
        "dataset_task_id": planned["task_id"],
        "dataset_work_item_id": planned["work_item_id"],
        "category": planned["task_family"],
        "prompt_sha256": planned["prompt_sha256"],
        "requested_model_id": planned["model_id"],
        "requested_provider": planned["provider_endpoint"],
        "endpoint_execution_contract_sha256": planned["endpoint_execution_sha256"],
        "execution_policy_sha256": planned["execution_policy_sha256"],
        "protocol_bundle": {
            "core_protocol_bundle": {
                "implementation_sha256": {
                    "provider.py": plan["source"]["provider_source_sha256"]
                }
            }
        },
        "epicure": {
            "release_id": "exploratory-unmatched-1790-runtime",
            "bundle_sha256": plan["epicure"]["bundle_sha256"],
            "application_sha256": plan["epicure"]["application_sha256"],
        },
        "epicure_tool_schema_sha256": plan["epicure"]["tool_schema_sha256"],
        "frozen_generation_contract": {
            "expected_epicure_bundle_sha256": plan["epicure"]["bundle_sha256"],
            "expected_epicure_application_sha256": plan["epicure"][
                "application_sha256"
            ],
            "expected_epicure_tool_schema_sha256": plan["epicure"][
                "tool_schema_sha256"
            ],
        },
        "budget": {
            "actual_cost_micros": 20,
            "all_generation_costs_reconciled": True,
        },
        "results": results,
        "errors": {},
        "provider_attempt_events": attempt_events,
        "mcp_trace_events": mcp_events,
        "incomplete_generation_metadata": [],
        "run_journal": descriptor.payload(),
    }
    source["artifact_sha256"] = sensitivity._sha256(source)
    source_path = source_root / (
        f"20260803T000000Z-{source['artifact_sha256'][:12]}.json"
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")

    response_records: dict[str, ResponseArtifact] = {}
    for condition in ("epicure_off", "epicure_on"):
        digest = sensitivity._sha256(
            {
                "work_item_id": planned["work_item_id"],
                "condition": condition,
                "variant": variant_id,
            }
        )
        path = response_root / (
            f"response-{planned['work_item_id']}-{condition}-{digest}.json"
        )
        path.write_text("{}", encoding="utf-8")
        response_records[str(path)] = ResponseArtifact(
            path=path,
            artifact_sha256=digest,
            work_item_id=planned["work_item_id"],
            condition=condition,
            task_id=planned["task_id"],
            task_family=planned["task_family"],
            model_id=planned["model_id"],
            provider_tag=planned["provider_endpoint"],
            source_artifact_sha256=source["artifact_sha256"],
            actual_cost_usd=Decimal("0.00001"),
            tool_used=condition == "epicure_on",
        )

    ledger_path = run_root / "ledger.jsonl"
    reservation = append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "reservation_created",
            "manifest_sha256": asset["manifest_sha256"],
            "model_id": planned["model_id"],
            "canonical_model_slug": planned["canonical_model_slug"],
            "provider_tag": planned["provider_endpoint"],
            "task_id": planned["task_id"],
            "task_family": planned["task_family"],
            "prompt_sha256": planned["prompt_sha256"],
            "endpoint_execution_sha256": planned["endpoint_execution_sha256"],
            "execution_policy_sha256": planned["execution_policy_sha256"],
            "reserved_usd": planned["worst_case_reserve_usd"],
            "work_item_id": planned["work_item_id"],
        },
    )
    finalization = append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "source_artifact_recorded",
            "reservation_entry_sha256": reservation["entry_sha256"],
            "source_artifact_sha256": source["artifact_sha256"],
            "source_artifact_filename": source_path.name,
            "source_status": "complete",
            "response_conditions": ["epicure_off", "epicure_on"],
            "response_artifact_sha256s": sorted(
                record.artifact_sha256 for record in response_records.values()
            ),
            "all_generation_costs_reconciled": True,
            "provider_cost_exact": True,
            "normalization_issues": [],
            "source_exposure_basis": "fully_reconciled_actual",
            "provider_reconciled_actual_cost_usd": "0.00002",
            "source_actual_cost_usd": "0.00002",
            "work_item_id": planned["work_item_id"],
        },
    )
    summary = _summary_content_address(
        {
            "schema_version": "flavourbench-real-exploratory-summary-v1",
            "mode": "execute",
            "provider_calls_made": True,
            "paid_subprocesses_started": 1,
            "workload": {
                "route_validation_override": {
                    "plan_sha256": plan["artifact_sha256"],
                    "route_cell_id": plan["route_validation"]["route_cell_id"],
                    "variant_id": variant_id,
                    "effective_fresh_work_item_id": planned["work_item_id"],
                    "model_id": planned["model_id"],
                    "task_id": planned["task_id"],
                    "quality_fit_eligible": False,
                }
            },
            "outcomes": [
                {
                    "work_item_id": planned["work_item_id"],
                    "decision": "pair_recorded",
                    "source_artifact_sha256": source["artifact_sha256"],
                    "response_artifact_sha256s": sorted(
                        record.artifact_sha256 for record in response_records.values()
                    ),
                    "subprocess_returncode": 0,
                }
            ],
            "ledger": {
                "entry_count": 2,
                "head_entry_sha256": finalization["entry_sha256"],
            },
        }
    )
    summary_path = summary_root / (
        f"real-exploratory-summary-{summary['content_address']['digest']}.json"
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return response_records


def _patch_response_verifier(
    monkeypatch: pytest.MonkeyPatch,
    response_records: dict[str, ResponseArtifact],
) -> None:
    def verify(path: Path) -> ResponseArtifact:
        return response_records[str(path)]

    monkeypatch.setattr(sensitivity, "_verify_response_artifact", verify)


def test_v2_receipt_builder_derives_strict_pass_from_immutable_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, assets = _isolated_assets(tmp_path)
    responses: dict[str, ResponseArtifact] = {}
    for variant_id in plan["route_validation"]["execution_order"]:
        responses.update(
            _write_variant_records(
                plan=plan,
                assets=assets,
                variant_id=variant_id,
            )
        )
    _patch_response_verifier(monkeypatch, responses)

    payload = sensitivity.build_v2_route_validation_audit(
        plan=plan,
        runner_assets=assets,
    )
    receipt = _content_address(payload)

    assert receipt["decision"] == "passed_all_predicates"
    assert receipt["counts"]["usable_pairs"] == 3
    assert receipt["counts"]["usable_arms"] == 6
    assert receipt["full_sensitivity_admission"]["authorized"] is True
    assert sensitivity.verify_v2_route_validation_pass_audit(receipt, plan)


def test_v2_receipt_builder_fails_duplicate_generation_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, assets = _isolated_assets(tmp_path)
    responses: dict[str, ResponseArtifact] = {}
    for variant_id in plan["route_validation"]["execution_order"]:
        responses.update(
            _write_variant_records(
                plan=plan,
                assets=assets,
                variant_id=variant_id,
                reused_generation_prefix="duplicate",
            )
        )
    _patch_response_verifier(monkeypatch, responses)

    payload = sensitivity.build_v2_route_validation_audit(
        plan=plan,
        runner_assets=assets,
    )
    receipt = _content_address(payload)

    assert receipt["decision"] == "failed_one_or_more_predicates"
    assert receipt["identifier_freshness_audit"]["duplicate_generation_ids"]
    assert receipt["full_sensitivity_admission"]["authorized"] is False
    assert not sensitivity.verify_v2_route_validation_pass_audit(receipt, plan)


def test_v2_receipt_builder_reports_not_executed_without_promoting_dry_runs(
    tmp_path: Path,
) -> None:
    plan, assets = _isolated_assets(tmp_path)

    payload = sensitivity.build_v2_route_validation_audit(
        plan=plan,
        runner_assets=assets,
    )
    receipt = _content_address(payload)

    assert receipt["decision"] == "not_executed"
    assert receipt["counts"]["attempted_pairs"] == 0
    assert receipt["full_sensitivity_admission"]["authorized"] is False
    assert not sensitivity.verify_v2_route_validation_pass_audit(receipt, plan)


def test_v2_receipt_builder_rejects_operator_mutation_of_runner_assets(
    tmp_path: Path,
) -> None:
    plan, assets = _isolated_assets(tmp_path)
    forged = copy.deepcopy(assets)
    forged["variants"][0]["fresh_work_item_id"] = "f" * 64

    with pytest.raises(sensitivity.SensitivityProtocolError, match="do not verify"):
        sensitivity.build_v2_route_validation_audit(
            plan=plan,
            runner_assets=forged,
        )
