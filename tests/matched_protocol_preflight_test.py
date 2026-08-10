from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.matched_protocol_preflight import (
    LIVE_PROTOCOL_SCHEMA_VERSION,
    PREFLIGHT_TASK_ID,
    ProtocolPreflightError,
    _write_artifact,
    build_plan,
    build_registry,
    promote_manifest,
    verify_registry_for_manifest,
)
from flavourbench.run_journal import write_fixture_journal

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / (
    "artifacts/season1/current-quality-run/manifest-v12-candidate/"
    "flavourbench-openrouter-unranked-"
    "4697f28ee4470ea0893327355d7f61d758790ee2e853ff499ea122e72d2d4417.json"
)
MANIFEST_SHA256 = "4697f28ee4470ea0893327355d7f61d758790ee2e853ff499ea122e72d2d4417"


def _live_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _result(entry: dict[str, object], lane: str) -> dict[str, object]:
    traces: list[dict[str, object]] = []
    if lane == "tool_contract":
        traces = [
            {
                "round_index": 0,
                "name": "find_pairings",
                "arguments": {"ingredients": ["pear", "white miso"]},
                "result": "fixture tool result",
                "result_sha256": hashlib.sha256(b"fixture tool result").hexdigest(),
                "latency_ms": 1,
                "is_error": False,
            }
        ]
    intermediate_outputs = (
        [
            {
                "phase": "tool_selection",
                "finish_reason": "tool_calls",
                "truncated": False,
                "tool_call_count": 1,
            }
        ]
        if lane == "tool_contract"
        else [
            {
                "phase": "planning",
                "finish_reason": "stop",
                "truncated": False,
            },
            {
                "phase": "evidence_decision" if lane == "epicure_off" else "tool_selection",
                "finish_reason": "stop" if lane == "epicure_off" else "tool_calls",
                "truncated": False,
            },
        ]
    )
    return {
        "answer_markdown": f"Verified fixture answer for {lane}.",
        "actual_model_id": entry["canonical_model_slug"],
        "actual_provider": entry["actual_provider_name"],
        "generation_ids": [f"fixture-{entry['ordinal']}-{lane}"],
        "finish_reason": "stop",
        "cost_reconciled": True,
        "cost_micros": 1,
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "reasoning_tokens": 0,
        "final_response_mode": "plain_text",
        "structured_output_requested": False,
        "structured_output_valid": None,
        "tool_trace": traces,
        "backend_tool_schema_sha256": "7" * 64,
        "intermediate_outputs": intermediate_outputs,
    }


def _receipt(tmp_path: Path, plan: dict[str, object], entry: dict[str, object]) -> Path:
    journal = write_fixture_journal(
        tmp_path,
        run_id=f"fixture-{entry['ordinal']}",
        events=[],
    )
    task = plan["task"]
    assert isinstance(task, dict)
    binding = {
        "candidate_manifest_sha256": plan["candidate_manifest_sha256"],
        "dataset_work_item_id": entry["work_item_id"],
        "dataset_task_id": PREFLIGHT_TASK_ID,
        "prompt_sha256": task["prompt_sha256"],
        "requested_model_id": entry["model_id"],
        "canonical_model_slug": entry["canonical_model_slug"],
        "provider_tag": entry["provider_tag"],
        "execution_policy_sha256": plan["execution_policy_sha256"],
        "evidence_protocol": plan["generation_protocol"]["evidence_protocol"],
        "required_tool_contract_protocol": "direct_tool_first_v1",
        "required_tool_contract_max_intermediate_tokens": plan["generation_protocol"][
            "required_tool_contract_max_intermediate_tokens"
        ],
        "required_tool_contract_sha256": plan["generation_protocol"][
            "required_tool_contract_sha256"
        ],
        "intermediate_reasoning_effort": plan["generation_protocol"][
            "intermediate_reasoning_effort"
        ],
        "final_reasoning_effort": plan["generation_protocol"][
            "final_reasoning_effort"
        ],
    }
    payload: dict[str, object] = {
        "schema_version": "flavourbench-live-smoke-v1",
        "status": "complete",
        "run_purpose": "epicure_on_off_pair",
        "candidate_manifest_sha256": plan["candidate_manifest_sha256"],
        "dataset_work_item_id": entry["work_item_id"],
        "dataset_task_id": PREFLIGHT_TASK_ID,
        "prompt_sha256": task["prompt_sha256"],
        "category": task["category"],
        "requested_model_id": entry["model_id"],
        "requested_provider": entry["provider_tag"],
        "endpoint_execution_contract_sha256": entry["endpoint_execution_sha256"],
        "official": False,
        "rank_eligible": False,
        "errors": {},
        "execution_policy": plan["execution_policy"],
        "frozen_generation_contract": {
            "evidence_protocol": plan["generation_protocol"]["evidence_protocol"],
            "required_tool_contract_protocol": "direct_tool_first_v1",
            "required_tool_contract_max_intermediate_tokens": plan[
                "generation_protocol"
            ]["required_tool_contract_max_intermediate_tokens"],
            "required_tool_contract_sha256": plan["generation_protocol"][
                "required_tool_contract_sha256"
            ],
            "final_response_mode": "plain_text",
            "matched_planning": True,
            "intermediate_reasoning_effort": plan["generation_protocol"][
                "intermediate_reasoning_effort"
            ],
            "final_reasoning_effort": plan["generation_protocol"][
                "final_reasoning_effort"
            ],
            "expected_actual_model_id": entry["canonical_model_slug"],
            "expected_actual_provider_slug": entry["actual_provider_name"],
        },
        "system_prompt_sha256": {
            "epicure_off": "1" * 64,
            "epicure_on": "1" * 64,
        },
        "protocol_bundle": {
            "schema_version": LIVE_PROTOCOL_SCHEMA_VERSION,
            "run_binding": binding,
            "core_protocol_bundle": {
                "implementation_sha256": plan["orchestration_source_sha256"],
                "release_inputs": {
                    "dependency_lock_sha256": "3" * 64,
                    "container_image_digest": "unresolved",
                },
            },
        },
        "required_tool_contract": plan["required_tool_contract"],
        "budget": {
            "all_generation_costs_reconciled": True,
            "actual_cost_micros": 3,
        },
        "results": {
            lane: _result(entry, lane)
            for lane in ("epicure_off", "epicure_on", "tool_contract")
        },
        "epicure": {
            "release_id": "exploratory-unmatched-1790-runtime",
            "bundle_sha256": "4" * 64,
            "application_sha256": "5" * 64,
        },
        "epicure_tool_schema_sha256": "6" * 64,
        "run_journal": journal.payload(),
    }
    payload["artifact_sha256"] = _live_sha256(payload)
    path = tmp_path / f"receipt-{entry['ordinal']}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_exact_preflight_registry_is_required_before_manifest_promotion(
    tmp_path: Path,
) -> None:
    plan = build_plan(
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    assert len(plan["entries"]) == 14
    assert plan["generation_protocol"]["intermediate_reasoning_effort"] == "low"
    assert plan["generation_protocol"]["max_intermediate_tokens"] == 4096
    assert plan["generation_protocol"][
        "required_tool_contract_max_intermediate_tokens"
    ] == 2048
    assert plan["execution_policy"]["limits"]["max_provider_attempts"] == 2
    plan_path = _write_artifact(tmp_path, "plan", plan)
    receipt_paths = [_receipt(tmp_path, plan, entry) for entry in plan["entries"]]

    with pytest.raises(ProtocolPreflightError, match="expected 14"):
        build_registry(plan_path=plan_path, artifact_paths=receipt_paths[:-1])

    registry = build_registry(plan_path=plan_path, artifact_paths=receipt_paths)
    registry_path = _write_artifact(tmp_path, "registry", registry)
    promoted = promote_manifest(
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
        registry_path=registry_path,
    )
    verified = verify_registry_for_manifest(
        registry_path=registry_path,
        manifest=promoted,
    )

    assert verified["all_required_routes_passed"] is True
    assert verified["synthetic_receipts"] == 0
    assert promoted["protocol_preflight"]["registry_sha256"] == (
        verified["artifact_sha256"]
    )
    assert promoted["content_address"]["digest"] != MANIFEST_SHA256


def test_preflight_rejects_condition_prompt_drift(tmp_path: Path) -> None:
    plan = build_plan(
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    entry = plan["entries"][0]
    receipt_path = _receipt(tmp_path, plan, entry)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["system_prompt_sha256"]["epicure_on"] = "9" * 64
    receipt.pop("artifact_sha256")
    receipt["artifact_sha256"] = _live_sha256(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProtocolPreflightError, match="share one system prompt"):
        from flavourbench.matched_protocol_preflight import validate_receipt

        validate_receipt(artifact_path=receipt_path, plan=plan)
