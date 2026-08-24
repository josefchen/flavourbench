from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from flavourbench.frontier_contract_runner import (
    EXECUTION_CONFIRMATION,
    NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION,
    AdmissionDenied,
    ContractPolicy,
    IntegrityError,
    active_ledger_reservations,
    append_ledger_event,
    build_plan,
    derive_contract_forecast,
    load_candidate_manifest,
    load_ledger,
    resolve_no_artifact_reservation,
    run_frontier_contracts,
    scan_live_smoke_artifacts,
    select_candidates,
    validate_ledger_artifact_links,
    validate_no_artifact_reconciliation,
    validate_no_artifact_reconciliation_v2,
    write_no_artifact_reconciliation,
)
from flavourbench.frontier_manifest import (
    ForecastPolicy,
    PanelSlot,
    build_candidate_manifest,
    write_content_addressed_manifest,
)


def _live_hash(value: object) -> str:
    canonical = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def _candidate_manifest(tmp_path: Path) -> tuple[Path, dict]:
    model_id = "vendor/model-v1"
    slot = PanelSlot(
        slot_id="frontier-test",
        cohort="closed_frontier",
        model_id=model_id,
        rationale="Test-only frozen model and endpoint.",
    )
    model = {
        "id": model_id,
        "canonical_slug": "vendor/model-v1-20260715",
        "name": "Vendor Model v1",
        "created": 1_784_000_000,
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tokenizer": "test",
        },
        "context_length": 200_000,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "supported_parameters": [
            "max_tokens",
            "response_format",
            "structured_outputs",
            "tool_choice",
            "tools",
        ],
        "top_provider": {"context_length": 200_000},
    }
    endpoint = {
        "name": "Vendor | Fixed",
        "provider_name": "Vendor",
        "tag": "vendor/fixed",
        "model_id": model_id,
        "quantization": "fp8",
        "context_length": 200_000,
        "max_completion_tokens": 10_000,
        "status": 0,
        "pricing": {
            "prompt": "0.000001",
            "completion": "0.000002",
            "internal_reasoning": "0.0000005",
        },
        "supported_parameters": [
            "max_tokens",
            "response_format",
            "structured_outputs",
            "tool_choice",
            "tools",
        ],
    }
    manifest = build_candidate_manifest(
        {"data": [model]},
        {model_id: {"data": {"id": model_id, "endpoints": [endpoint]}}},
        cap_usd="100",
        forecast_policy=ForecastPolicy(
            arms_per_model=1,
            max_generations_per_arm=1,
            max_prompt_tokens_per_generation=1_000,
            max_completion_tokens_per_generation=500,
            max_reasoning_tokens_per_generation=500,
        ),
        panel=[slot],
        requested_names=[],
        observed_at="2026-07-15T00:00:00Z",
    )
    path = write_content_addressed_manifest(manifest, tmp_path / "manifests")
    return path, manifest


def _write_live_artifact(
    root: Path,
    *,
    candidate: object,
    manifest_sha256: str | None,
    name: str,
    status: str = "complete",
    actual_cost_micros: int = 1_000,
    forecast_usd: str = "0.5",
    cap_usd: str = "0.6",
    reconciled: bool = True,
    contract_passed: bool = True,
    incomplete_cost_micros: int = 0,
    accounted_attempts: bool = False,
    pre_send_failure: bool = False,
    mcp_lifecycle: bool = False,
) -> tuple[Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    result_cost = actual_cost_micros - incomplete_cost_micros
    endpoint = dict(candidate.endpoint)
    endpoint_contract = {
        field: endpoint.get(field)
        for field in (
            "name",
            "provider_name",
            "tag",
            "model_id",
            "quantization",
            "context_length",
            "max_completion_tokens",
            "pricing",
            "supported_parameters",
            "uptime_last_1d",
        )
    }
    errors = {} if status == "complete" and contract_passed else {"tool_contract": "failed"}
    results = (
        {}
        if pre_send_failure
        else {
            "tool_contract": {
                "cost_micros": result_cost,
                "generation_metadata": [
                    {
                        "generation_id": f"gen-result-{name}",
                        "cost_micros": result_cost,
                        "reconciled": True,
                    }
                ]
                if accounted_attempts
                else [],
                "tool_trace": [
                    {
                        "name": "find_pairings",
                        "is_error": False,
                    }
                ]
                if contract_passed
                else [],
            }
        }
    )
    artifact = {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": f"run-{name}",
        "status": status,
        "official": False,
        "rank_eligible": False,
        "candidate_manifest_sha256": manifest_sha256,
        "requested_model_id": candidate.model_id,
        "requested_provider": candidate.provider_tag,
        "model_contract": {
            "id": candidate.model_id,
            "canonical_slug": candidate.canonical_model_slug,
        },
        "endpoint_contract": endpoint_contract,
        "results": results,
        "errors": errors,
        "incomplete_generation_metadata": [
            {
                "generation_id": f"gen-{name}",
                "cost_micros": incomplete_cost_micros,
                "reconciled": True,
            }
        ]
        if incomplete_cost_micros
        else [],
        "provider_attempt_events": [
            *(
                [
                    {
                        "attempt_id": f"mcp-session-{name}",
                        "event_type": "mcp_session_started",
                    },
                    {
                        "attempt_id": f"mcp-session-{name}",
                        "event_type": "mcp_session_attested",
                    },
                ]
                if mcp_lifecycle
                else []
            ),
            {
                "attempt_id": f"attempt-{name}",
                "event_type": "request_started",
            },
            {
                "attempt_id": f"attempt-{name}",
                "event_type": "response_received",
                "generation_id": f"gen-result-{name}",
            },
        ]
        if accounted_attempts
        else [],
        "budget": {
            "actual_cost_micros": actual_cost_micros,
            "forecast_worst_case_usd": forecast_usd,
            "cap_usd": cap_usd,
            "all_generation_costs_reconciled": reconciled,
        },
    }
    artifact["artifact_sha256"] = _live_hash(artifact)
    path = root / f"20260715T000000Z-{artifact['artifact_sha256'][:12]}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path, artifact


def _write_failed_kimi_rate_card_artifact(
    root: Path,
    *,
    candidate: object,
    manifest_sha256: str,
    missing_usage: bool = False,
) -> Path:
    _, artifact = _write_live_artifact(
        root.parent / "fixture-source",
        candidate=candidate,
        manifest_sha256=manifest_sha256,
        name="failed-kimi-rate-card",
    )
    result_metadata = {
        "generation_id": "gen-kimi-off",
        "cost_micros": 10_000,
        "reconciled": False,
        "accounting_basis": "frozen_rate_card_times_kimi_returned_usage",
        "billing_reconciliation_status": "provider_charge_unavailable",
        "tokens_prompt": 1_000,
        "tokens_completion": 500,
    }
    incomplete_metadata = {
        "generation_id": "gen-kimi-on",
        "cost_micros": 20_000,
        "reconciled": False,
        "accounting_basis": "frozen_rate_card_times_kimi_returned_usage",
        "billing_reconciliation_status": "provider_charge_unavailable",
        "tokens_prompt": 2_000,
        "tokens_completion": 1_000,
    }
    artifact.update(
        {
            "status": "failed_or_unreconciled",
            "execution_backend": "kimi_direct",
            "requested_provider": "kimi-code-direct",
            "results": {
                "epicure_off": {
                    "cost_micros": 10_000,
                    "cost_reconciled": False,
                    "cost_accounting_basis": (
                        "frozen_rate_card_times_kimi_returned_usage"
                    ),
                    "billing_reconciliation_status": "provider_charge_unavailable",
                    "generation_ids": ["gen-kimi-off"],
                    "generation_metadata": [result_metadata],
                }
            },
            "errors": {"epicure_on": "required Epicure tool call failed"},
            "incomplete_generation_metadata": [incomplete_metadata],
            "provider_attempt_events": [
                {
                    "attempt_id": "attempt-kimi-off",
                    "event_type": "request_started",
                },
                {
                    "attempt_id": "attempt-kimi-off",
                    "event_type": "response_received",
                    "generation_id": "gen-kimi-off",
                },
                {
                    "attempt_id": "attempt-kimi-on",
                    "event_type": "request_started",
                },
                {
                    "attempt_id": "attempt-kimi-on",
                    "event_type": "response_received",
                    "generation_id": (
                        "gen-kimi-on-without-usage" if missing_usage else "gen-kimi-on"
                    ),
                },
            ],
            "budget": {
                "actual_cost_micros": 30_000,
                "forecast_worst_case_usd": "0.5",
                "cap_usd": "0.6",
                "all_generation_costs_reconciled": False,
                # Older immutable artifacts wrote this false whenever an arm
                # failed, so the scanner must verify the records themselves.
                "all_generation_usage_rate_card_accounted": False,
                "accounting_basis": "provider_usage_times_frozen_rate_card",
                "provider_charge_available": False,
            },
        }
    )
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _live_hash(artifact)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"20260715T000000Z-{artifact['artifact_sha256'][:12]}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def _write_unpriced_qwen38_alias_artifact(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    accounting_basis = "qwencloud_returned_usage_with_full_unpriced_budget_ceiling"
    billing_status = "provider_rate_and_charge_unavailable"

    def result(condition: str) -> dict:
        generation_id = f"gen-qwen38-{condition}"
        return {
            "cost_micros": 0,
            "cost_reconciled": False,
            "cost_accounting_basis": accounting_basis,
            "billing_reconciliation_status": billing_status,
            "generation_ids": [generation_id],
            "generation_metadata": [
                {
                    "generation_id": generation_id,
                    "cost_micros": 0,
                    "reconciled": False,
                    "provider_cost_known": False,
                    "accounting_basis": accounting_basis,
                    "billing_reconciliation_status": billing_status,
                    "tokens_prompt": 100,
                    "tokens_completion": 50,
                }
            ],
        }

    artifact = {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": "run-qwen38-unpriced",
        "status": "complete_unpriced_budget_ceiling",
        "official": False,
        "rank_eligible": False,
        "execution_backend": "qwencloud_direct",
        "requested_model_id": "qwen3.8-max",
        "requested_provider": "qwencloud-direct",
        "requested_conditions": ["epicure_off", "epicure_on"],
        "mutable_alias_exploratory_opt_in": True,
        "candidate_manifest_sha256": "a" * 64,
        "backend_contract": {
            "identity_kind": "mutable_alias",
            "catalog_pinned_at_observation": True,
            "model_identity_label": "catalog_pinned_at_observation_not_a_frozen_model",
            "official": False,
            "season_eligible": False,
            "rank_eligible": False,
        },
        "results": {
            "epicure_off": result("off"),
            "epicure_on": result("on"),
        },
        "errors": {},
        "incomplete_generation_metadata": [],
        "budget": {
            "actual_cost_micros": 0,
            "forecast_worst_case_usd": "2",
            "cap_usd": "2",
            "all_generation_costs_reconciled": False,
            "all_generation_usage_accounted": True,
            "accounting_basis": "provider_usage_with_unpriced_budget_ceiling",
            "provider_rate_available": False,
            "provider_cost_known": False,
            "full_unpriced_budget_ceiling_retained": True,
        },
    }
    artifact["artifact_sha256"] = _live_hash(artifact)
    path = root / f"20260808T000000Z-{artifact['artifact_sha256'][:12]}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def test_unpriced_qwen38_alias_retains_full_ceiling_without_claiming_zero_cost(
    tmp_path: Path,
) -> None:
    _write_unpriced_qwen38_alias_artifact(tmp_path)
    scan = scan_live_smoke_artifacts(tmp_path)

    assert scan.actual_cost_usd == 0
    assert scan.exposure_usd == Decimal("2")
    assert scan.artifacts[0].exposure_basis == (
        "complete_unpriced_full_budget_ceiling_reserve"
    )


def _write_correction(
    root: Path,
    *,
    source: dict,
    source_path: Path,
    additional_cost_micros: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    original = source["budget"]["actual_cost_micros"]
    generation_id = "gen-corrected"
    correction = {
        "schema_version": "flavourbench-live-smoke-cost-correction-v1",
        "record_type": "superseding_cost_reconciliation",
        "created_at": "2026-07-15T00:01:00Z",
        "source": {
            "path": str(source_path),
            "artifact_sha256": source["artifact_sha256"],
            "run_id": source["run_id"],
            "requested_model_id": source["requested_model_id"],
            "candidate_manifest_sha256": source["candidate_manifest_sha256"],
        },
        "missing_generation_ids": [generation_id],
        "generation_metadata": [
            {
                "generation_id": generation_id,
                "cost_micros": additional_cost_micros,
                "reconciled": True,
            }
        ],
        "all_missing_generations_reconciled": True,
        "cost": {
            "original_recorded_cost_micros": original,
            "additional_cost_micros": additional_cost_micros,
            "corrected_total_cost_micros": original + additional_cost_micros,
        },
        "rank_eligible": False,
    }
    correction["artifact_sha256"] = _live_hash(correction)
    destination = root / (
        f"{source_path.stem}-cost-correction-{correction['artifact_sha256'][:12]}.json"
    )
    destination.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n")
    return destination


def test_manifest_selection_freezes_endpoint_and_derives_max_prices(tmp_path: Path) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    digest = manifest["content_address"]["digest"]
    loaded = load_candidate_manifest(path, expected_digest=digest)
    candidate = select_candidates(loaded, ["frontier-test"])[0]
    forecast = derive_contract_forecast(candidate)

    assert candidate.model_id == "vendor/model-v1"
    assert candidate.provider_tag == "vendor/fixed"
    assert forecast.price_envelope.prompt_usd_per_mtok == Decimal("1")
    assert forecast.price_envelope.completion_usd_per_mtok == Decimal("2")
    assert forecast.price_envelope.reasoning_usd_per_token == Decimal("0.0000005")
    assert forecast.actual_contract_request_bound == 9
    assert forecast.live_smoke_admission_request_bound == 19
    assert forecast.forecast_usd > 0


def test_manifest_tampering_and_unexpected_digest_are_rejected(tmp_path: Path) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    digest = manifest["content_address"]["digest"]
    with pytest.raises(IntegrityError, match="does not match expected"):
        load_candidate_manifest(path, expected_digest="0" * 64)

    tampered = json.loads(path.read_text())
    tampered["models"][0]["endpoint"]["tag"] = "vendor/changed"
    path.write_text(json.dumps(tampered))
    with pytest.raises(IntegrityError, match="invalid content address"):
        load_candidate_manifest(path, expected_digest=digest)


def test_failed_paid_artifact_keeps_full_allowance_and_applies_correction(
    tmp_path: Path,
) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    candidate = select_candidates(
        load_candidate_manifest(path, expected_digest=manifest["content_address"]["digest"])
    )[0]
    smoke = tmp_path / "live-smoke"
    corrections = tmp_path / "corrections"
    complete_path, _ = _write_live_artifact(
        smoke,
        candidate=candidate,
        manifest_sha256=None,
        name="complete",
        actual_cost_micros=20_000,
        cap_usd="0.6",
    )
    failed_path, failed = _write_live_artifact(
        smoke,
        candidate=candidate,
        manifest_sha256=manifest["content_address"]["digest"],
        name="failed",
        status="failed_or_unreconciled",
        actual_cost_micros=10_000,
        forecast_usd="0.5",
        cap_usd="0.6",
        reconciled=False,
        contract_passed=False,
        incomplete_cost_micros=4_000,
    )
    _write_correction(
        corrections,
        source=failed,
        source_path=failed_path,
        additional_cost_micros=5_000,
    )

    scan = scan_live_smoke_artifacts(smoke, corrections_directory=corrections)
    assert complete_path.exists()
    assert scan.actual_cost_usd == Decimal("0.035")
    assert scan.exposure_usd == Decimal("0.62")
    assert scan.failed_or_unreconciled_reserve_usd == Decimal("0.6")
    assert scan.correction_count == 1
    failed_exposure = next(item for item in scan.artifacts if item.status != "complete")
    assert failed_exposure.actual_cost_usd == Decimal("0.015")
    assert failed_exposure.exposure_basis == ("failed_or_unreconciled_full_admitted_allowance")


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    candidate = select_candidates(
        load_candidate_manifest(path, expected_digest=manifest["content_address"]["digest"])
    )[0]
    artifact_path, _ = _write_live_artifact(
        tmp_path / "smoke",
        candidate=candidate,
        manifest_sha256=None,
        name="tampered",
    )
    value = json.loads(artifact_path.read_text())
    value["budget"]["actual_cost_micros"] = 999_999
    artifact_path.write_text(json.dumps(value))
    with pytest.raises(IntegrityError, match="content address is invalid"):
        scan_live_smoke_artifacts(tmp_path / "smoke")


def test_failed_output_with_reconciled_paid_attempt_counts_actual_not_zero_or_cap(
    tmp_path: Path,
) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    candidate = select_candidates(
        load_candidate_manifest(path, expected_digest=manifest["content_address"]["digest"])
    )[0]
    _write_live_artifact(
        tmp_path / "smoke",
        candidate=candidate,
        manifest_sha256=manifest["content_address"]["digest"],
        name="accounted-failure",
        status="failed_or_unreconciled",
        actual_cost_micros=12_345,
        cap_usd="0.6",
        contract_passed=False,
        accounted_attempts=True,
    )
    scan = scan_live_smoke_artifacts(tmp_path / "smoke")
    assert scan.actual_cost_usd == Decimal("0.012345")
    assert scan.exposure_usd == Decimal("0.012345")
    assert scan.failed_or_unreconciled_reserve_usd == 0
    assert scan.artifacts[0].exposure_basis == ("failed_but_all_attempts_cost_reconciled_actual")


def test_failed_direct_kimi_with_usage_keeps_full_reserve_without_blocking_recovery(
    tmp_path: Path,
) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    candidate = select_candidates(
        load_candidate_manifest(path, expected_digest=manifest["content_address"]["digest"])
    )[0]
    _write_failed_kimi_rate_card_artifact(
        tmp_path / "smoke",
        candidate=candidate,
        manifest_sha256=manifest["content_address"]["digest"],
    )

    scan = scan_live_smoke_artifacts(tmp_path / "smoke")

    assert scan.actual_cost_usd == Decimal("0.03")
    assert scan.exposure_usd == Decimal("0.6")
    assert scan.failed_or_unreconciled_reserve_usd == 0
    assert scan.artifacts[0].exposure_basis == (
        "failed_rate_card_estimated_full_forecast_reserve"
    )


def test_failed_direct_kimi_with_unmatched_response_keeps_unresolved_reserve(
    tmp_path: Path,
) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    candidate = select_candidates(
        load_candidate_manifest(path, expected_digest=manifest["content_address"]["digest"])
    )[0]
    _write_failed_kimi_rate_card_artifact(
        tmp_path / "smoke",
        candidate=candidate,
        manifest_sha256=manifest["content_address"]["digest"],
        missing_usage=True,
    )

    scan = scan_live_smoke_artifacts(tmp_path / "smoke")

    assert scan.exposure_usd == Decimal("0.6")
    assert scan.failed_or_unreconciled_reserve_usd == Decimal("0.6")
    assert scan.artifacts[0].exposure_basis == (
        "failed_or_unreconciled_full_admitted_allowance"
    )


def test_mcp_lifecycle_events_do_not_create_phantom_provider_reserve(
    tmp_path: Path,
) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    candidate = select_candidates(
        load_candidate_manifest(path, expected_digest=manifest["content_address"]["digest"])
    )[0]
    _write_live_artifact(
        tmp_path / "smoke",
        candidate=candidate,
        manifest_sha256=manifest["content_address"]["digest"],
        name="accounted-mcp-failure",
        status="failed_or_unreconciled",
        actual_cost_micros=12_345,
        cap_usd="0.6",
        contract_passed=False,
        accounted_attempts=True,
        mcp_lifecycle=True,
    )

    scan = scan_live_smoke_artifacts(tmp_path / "smoke")
    assert scan.exposure_usd == Decimal("0.012345")
    assert scan.failed_or_unreconciled_reserve_usd == 0


def test_pre_send_contract_failure_releases_forecast_reserve(tmp_path: Path) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    candidate = select_candidates(
        load_candidate_manifest(path, expected_digest=manifest["content_address"]["digest"])
    )[0]
    _write_live_artifact(
        tmp_path / "smoke",
        candidate=candidate,
        manifest_sha256=manifest["content_address"]["digest"],
        name="pre-send",
        status="failed_or_unreconciled",
        actual_cost_micros=0,
        forecast_usd="5.1",
        cap_usd="5.1",
        reconciled=False,
        contract_passed=False,
        pre_send_failure=True,
    )
    scan = scan_live_smoke_artifacts(tmp_path / "smoke")
    assert scan.actual_cost_usd == 0
    assert scan.exposure_usd == 0
    assert scan.failed_or_unreconciled_reserve_usd == 0
    assert scan.artifacts[0].exposure_basis == ("failed_but_all_attempts_cost_reconciled_actual")


def test_hash_chained_ledger_tracks_and_releases_reservations(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    reservation = append_ledger_event(
        ledger,
        {
            "event_type": "reservation_created",
            "reserved_usd": "1.25",
            "model_id": "vendor/model-v1",
        },
        recorded_at="2026-07-15T00:00:00Z",
    )
    assert active_ledger_reservations(load_ledger(ledger)) == {
        reservation["entry_sha256"]: Decimal("1.25")
    }
    append_ledger_event(
        ledger,
        {
            "event_type": "artifact_recorded",
            "reservation_entry_sha256": reservation["entry_sha256"],
            "artifact_sha256": "a" * 64,
        },
        recorded_at="2026-07-15T00:01:00Z",
    )
    assert active_ledger_reservations(load_ledger(ledger)) == {}

    lines = ledger.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["reserved_usd"] = "0"
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")
    with pytest.raises(IntegrityError, match="digest mismatch"):
        load_ledger(ledger)


def _no_artifact_reconciliation_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict, dict]:
    manifest_path, manifest = _candidate_manifest(tmp_path)
    candidate = select_candidates(
        load_candidate_manifest(
            manifest_path, expected_digest=manifest["content_address"]["digest"]
        )
    )[0]
    smoke = tmp_path / "smoke"
    source_path, source = _write_live_artifact(
        smoke,
        candidate=candidate,
        manifest_sha256=manifest["content_address"]["digest"],
        name="pending-cost",
        actual_cost_micros=11_656,
        forecast_usd="0.5",
        cap_usd="0.6",
    )
    source_path.unlink()
    source["started_at"] = "2026-07-15T17:07:04.063202+00:00"
    source["completed_at"] = "2026-07-15T17:07:52.974264+00:00"
    source["budget"]["openrouter_key_after"] = {
        "usage_daily_usd": 57.414809377,
        "usage_monthly_usd": 80.551706585,
    }
    source.pop("artifact_sha256")
    source["artifact_sha256"] = _live_hash(source)
    source_path = smoke / f"20260715T170704Z-{source['artifact_sha256'][:12]}.json"
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")

    ledger = tmp_path / "frontier" / "ledger.jsonl"
    source_reservation = append_ledger_event(
        ledger,
        {
            "event_type": "reservation_created",
            "runner_run_id": "source-run",
            "manifest_sha256": manifest["content_address"]["digest"],
            "model_id": candidate.model_id,
            "provider_tag": candidate.provider_tag,
            "reserved_usd": "0.6",
        },
        recorded_at="2026-07-15T17:07:03.884730Z",
    )
    append_ledger_event(
        ledger,
        {
            "event_type": "artifact_recorded",
            "runner_run_id": "source-run",
            "reservation_entry_sha256": source_reservation["entry_sha256"],
            "manifest_sha256": manifest["content_address"]["digest"],
            "model_id": candidate.model_id,
            "provider_tag": candidate.provider_tag,
            "artifact_filename": source_path.name,
            "artifact_sha256": source["artifact_sha256"],
        },
        recorded_at="2026-07-15T17:07:53.047111Z",
    )
    reservation = append_ledger_event(
        ledger,
        {
            "event_type": "reservation_created",
            "runner_run_id": "incident-run",
            "manifest_sha256": manifest["content_address"]["digest"],
            "model_id": "nvidia/nemotron-test",
            "provider_tag": "together",
            "reserved_usd": "1.1279388",
        },
        recorded_at="2026-07-15T17:07:53.092036Z",
    )
    stdout_sha = "451031538ba3c318e6b34430ab6e1b407be905cf6fb8c4cfac1e306218c3d01c"
    incident = append_ledger_event(
        ledger,
        {
            "event_type": "execution_incident",
            "runner_run_id": "incident-run",
            "reservation_entry_sha256": reservation["entry_sha256"],
            "incident": "no_verifiable_artifact_reservation_retained",
            "subprocess_returncode": 1,
            "stdout_sha256": stdout_sha,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        },
        recorded_at="2026-07-15T17:07:53.759658Z",
    )
    record = {
        "schema_version": "flavourbench-frontier-no-artifact-reconciliation-v1",
        "record_type": "external_account_no_artifact_reconciliation",
        "created_at": "2026-07-15T17:12:00Z",
        "official": False,
        "rank_eligible": False,
        "provider_calls_made": False,
        "reservation": {
            "ledger_entry_sha256": reservation["entry_sha256"],
            "runner_run_id": "incident-run",
            "model_id": "nvidia/nemotron-test",
            "provider_tag": "together",
            "manifest_sha256": manifest["content_address"]["digest"],
        },
        "incident": {
            "ledger_entry_sha256": incident["entry_sha256"],
            "recorded_at": incident["recorded_at"],
            "subprocess_returncode": 1,
            "stdout_sha256": stdout_sha,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "stdout_match": {
            "safe_stdout_sha256": stdout_sha,
            "matched_error_message": (
                "RuntimeError: configured max output exceeds the endpoint completion limit"
            ),
            "failure_boundary": "before_openrouter_provider_instantiation",
            "source_symbol": "flavourbench.live_smoke.frozen_generation_contract",
        },
        "account_reconciliation": {
            "provider": "openrouter",
            "endpoint": "/api/v1/key",
            "currency": "USD",
            "before": {
                "captured_no_later_than": source["completed_at"],
                "usage_daily_usd": 57.414809377,
                "usage_monthly_usd": 80.551706585,
                "source_artifact_sha256": source["artifact_sha256"],
                "source_field": "budget.openrouter_key_after",
            },
            "after": {
                "request_started_at": "2026-07-15T17:08:51.167Z",
                "response_received_at": "2026-07-15T17:08:51.520Z",
                "usage_daily_usd": 57.426465201,
                "usage_monthly_usd": 80.563362409,
                "command_sha256": "1" * 64,
                "command_stdout_sha256": "2" * 64,
                "capture_request_record_sha256": "3" * 64,
                "capture_response_record_sha256": "4" * 64,
                "source_session_id": "test-session",
            },
            "delta": {
                "usage_daily_usd": "0.011655824",
                "usage_monthly_usd": "0.011655824",
            },
            "known_pending_cost": {
                "source_artifact_sha256": source["artifact_sha256"],
                "source_artifact_cost_micros": 11_656,
                "matching_rule": (
                    "account_delta_round_half_up_to_micros_equals_source_artifact_cost"
                ),
            },
            "unexplained_delta_usd": "0",
        },
        "conclusion": {
            "provider_generation_request_reached": False,
            "provider_generation_cost_usd": "0",
            "reservation_release_authorized": True,
            "basis": (
                "allow-listed pre-generation stdout plus account delta fully explained by the "
                "immediately preceding reconciled artifact"
            ),
        },
    }
    proof = write_no_artifact_reconciliation(record, ledger.parent / "reconciliations")
    return ledger, smoke, proof, reservation, record


def test_no_artifact_reservation_requires_content_addressed_external_proof(
    tmp_path: Path,
) -> None:
    ledger, smoke, proof, reservation, _record = _no_artifact_reconciliation_fixture(tmp_path)
    scan = scan_live_smoke_artifacts(smoke)
    verified = validate_no_artifact_reconciliation(
        proof,
        ledger_entries=load_ledger(ledger),
        artifact_scan=scan,
    )
    assert verified.reservation_entry_sha256 == reservation["entry_sha256"]
    assert verified.account_usage_delta_usd == Decimal("0.011655824")

    event = resolve_no_artifact_reservation(
        ledger_path=ledger,
        reconciliation_path=proof,
        live_smoke_directory=smoke,
        reconciliation_directory=proof.parent,
    )
    assert event["event_type"] == "no_artifact_reconciliation_recorded"
    assert event["provider_generation_cost_usd"] == "0"
    assert active_ledger_reservations(load_ledger(ledger)) == {}


def test_no_artifact_reconciliation_rejects_unexplained_usage_or_stdout(
    tmp_path: Path,
) -> None:
    ledger, smoke, _proof, _reservation, record = _no_artifact_reconciliation_fixture(tmp_path)
    bad_delta = json.loads(json.dumps(record))
    bad_delta["account_reconciliation"]["after"]["usage_daily_usd"] = 57.5
    bad_proof = write_no_artifact_reconciliation(bad_delta, ledger.parent / "bad-reconciliations")
    with pytest.raises(IntegrityError, match="daily/monthly usage deltas"):
        validate_no_artifact_reconciliation(
            bad_proof,
            ledger_entries=load_ledger(ledger),
            artifact_scan=scan_live_smoke_artifacts(smoke),
        )

    bad_stdout = json.loads(json.dumps(record))
    bad_stdout["incident"]["stdout_sha256"] = "f" * 64
    bad_stdout_proof = write_no_artifact_reconciliation(bad_stdout, ledger.parent / "bad-stdout")
    with pytest.raises(IntegrityError, match="incident evidence differs"):
        validate_no_artifact_reconciliation(
            bad_stdout_proof,
            ledger_entries=load_ledger(ledger),
            artifact_scan=scan_live_smoke_artifacts(smoke),
        )


def _no_delivery_v2_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    ledger = tmp_path / "ledger.jsonl"
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    reservation = append_ledger_event(
        ledger,
        {
            "event_type": "reservation_created",
            "runner_run_id": "run-v2-no-delivery",
            "manifest_sha256": "1" * 64,
            "model_id": "vendor/model-v2",
            "provider_tag": "provider-v2",
            "reserved_usd": "1.25",
            "campaign_id": "study-v2",
            "study_plan_sha256": "2" * 64,
            "admission_block_id": "3" * 64,
            "work_item_id": "4" * 64,
            "replay_permitted": False,
        },
    )
    record = {
        "schema_version": NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION,
        "record_role": "never_started_test_reservation_no_delivery_reconciliation",
        "reservation": {
            "ledger_entry_sha256": reservation["entry_sha256"],
            "runner_run_id": reservation["runner_run_id"],
            "model_id": reservation["model_id"],
            "provider_tag": reservation["provider_tag"],
            "manifest_sha256": reservation["manifest_sha256"],
            "study_plan_sha256": reservation["study_plan_sha256"],
            "admission_block_id": reservation["admission_block_id"],
            "work_item_id": reservation["work_item_id"],
            "reserved_usd": reservation["reserved_usd"],
        },
        "no_delivery_evidence": {
            "item_execution_started_events": 0,
            "provider_request_journals": 0,
            "provider_request_started_events": 0,
            "provider_response_received_events": 0,
            "source_artifacts": 0,
            "generation_ids": [],
            "mcp_trace_events": 0,
            "canonical_finalizations_before_reconciliation": 0,
            "evidence_snapshot": {
                "v8_tree_unchanged": True,
                "canonical_source_inventory_verified": True,
                "journal_inventory_verified": True,
                "v8_tree_snapshot_sha256": "5" * 64,
                "target_identity_scan_sha256": "6" * 64,
            },
        },
        "conclusion": {
            "delivery_attempted": False,
            "provider_generation_request_reached": False,
            "provider_generation_cost_usd": "0",
            "epicure_called": False,
            "reservation_release_authorized": True,
            "same_identifier_replay_permitted": False,
            "disposition": "release_never_started_no_delivery_reservation",
        },
        "provider_calls_made": False,
        "epicure_calls_made": False,
        "official": False,
        "rank_eligible": False,
    }
    proof = write_no_artifact_reconciliation(record, tmp_path / "reconciliations")
    return ledger, smoke, proof, reservation, record


def test_v2_no_delivery_reconciliation_releases_only_with_exact_zero_evidence(
    tmp_path: Path,
) -> None:
    ledger, smoke, proof, reservation, _ = _no_delivery_v2_fixture(tmp_path)
    verified = validate_no_artifact_reconciliation_v2(
        proof,
        ledger_entries=load_ledger(ledger),
    )
    event = append_ledger_event(
        ledger,
        {
            "event_type": "no_artifact_reconciliation_recorded",
            "runner_run_id": reservation["runner_run_id"],
            "reservation_entry_sha256": reservation["entry_sha256"],
            "model_id": reservation["model_id"],
            "provider_tag": reservation["provider_tag"],
            "manifest_sha256": reservation["manifest_sha256"],
            "study_plan_sha256": reservation["study_plan_sha256"],
            "admission_block_id": reservation["admission_block_id"],
            "work_item_id": reservation["work_item_id"],
            "reconciliation_filename": proof.name,
            "reconciliation_sha256": verified.artifact_sha256,
            "reconciliation_schema_version": NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION,
            "released_exposure_usd": reservation["reserved_usd"],
            "provider_generation_cost_usd": "0",
            "decision": "release_never_started_no_delivery_reservation_v2",
        },
    )
    assert event["event_type"] == "no_artifact_reconciliation_recorded"
    assert active_ledger_reservations(load_ledger(ledger)) == {}
    validate_ledger_artifact_links(
        load_ledger(ledger),
        scan_live_smoke_artifacts(smoke),
        reconciliation_directory=proof.parent,
    )


def test_v2_no_delivery_reconciliation_rejects_any_request_evidence(
    tmp_path: Path,
) -> None:
    ledger, _smoke, _proof, _reservation, record = _no_delivery_v2_fixture(tmp_path)
    record["no_delivery_evidence"]["provider_request_started_events"] = 1
    proof = write_no_artifact_reconciliation(record, tmp_path / "bad-v2")
    with pytest.raises(IntegrityError, match="every no-delivery boundary"):
        validate_no_artifact_reconciliation_v2(
            proof,
            ledger_entries=load_ledger(ledger),
        )


def test_plan_skips_passed_contract_and_stops_at_admission_ceiling(
    tmp_path: Path,
) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    loaded = load_candidate_manifest(path, expected_digest=manifest["content_address"]["digest"])
    candidate = select_candidates(loaded)[0]
    smoke = tmp_path / "smoke"
    _write_live_artifact(
        smoke,
        candidate=candidate,
        manifest_sha256=manifest["content_address"]["digest"],
        name="pass",
    )
    scan = scan_live_smoke_artifacts(smoke)
    plan = build_plan(
        loaded,
        [candidate],
        artifact_scan=scan,
        active_reservation_usd=Decimal(0),
        policy=ContractPolicy(),
        cap_usd=Decimal("100"),
    )
    assert plan[0]["decision"] == "skip_existing_contract_pass"

    empty = scan_live_smoke_artifacts(tmp_path / "empty")
    blocked = build_plan(
        loaded,
        [candidate],
        artifact_scan=empty,
        active_reservation_usd=Decimal("84"),
        policy=ContractPolicy(),
        cap_usd=Decimal("100"),
    )
    assert blocked[0]["decision"] == "stop_85_percent_admission_ceiling"


def test_default_plan_writes_content_addressed_summary_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _candidate_manifest(tmp_path)

    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("planning must not invoke live_smoke")

    monkeypatch.setattr(
        "flavourbench.frontier_contract_runner.subprocess.run", forbidden_subprocess
    )
    summary, summary_path = run_frontier_contracts(
        manifest_path=path,
        expected_manifest_sha256=manifest["content_address"]["digest"],
        live_smoke_directory=tmp_path / "smoke",
        corrections_directory=tmp_path / "corrections",
        ledger_path=tmp_path / "ledger.jsonl",
        summary_directory=tmp_path / "summaries",
    )

    assert summary["mode"] == "plan_no_provider_calls"
    assert summary["outcomes"][0]["decision"] == "admit_sequentially"
    assert summary_path.name.endswith(f"{summary['content_address']['digest']}.json")
    assert not (tmp_path / "ledger.jsonl").exists()


def test_execute_delegates_exact_manifest_route_digest_and_price_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    digest = manifest["content_address"]["digest"]
    candidate = select_candidates(load_candidate_manifest(path, expected_digest=digest))[0]
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        output_index = command.index("--output-dir") + 1
        artifact_path, _ = _write_live_artifact(
            Path(command[output_index]),
            candidate=candidate,
            manifest_sha256=digest,
            name="executed",
            actual_cost_micros=1_000,
            forecast_usd=command[command.index("--cap-usd") + 1],
            cap_usd=command[command.index("--cap-usd") + 1],
        )
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"artifact": str(artifact_path.resolve())}),
            stderr="",
        )

    monkeypatch.setattr("flavourbench.frontier_contract_runner.subprocess.run", fake_run)
    summary, _ = run_frontier_contracts(
        manifest_path=path,
        expected_manifest_sha256=digest,
        live_smoke_directory=tmp_path / "smoke",
        corrections_directory=tmp_path / "corrections",
        ledger_path=tmp_path / "ledger.jsonl",
        summary_directory=tmp_path / "summaries",
        execute=True,
        confirmation=EXECUTION_CONFIRMATION,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[command.index("--candidate-manifest-sha256") + 1] == digest
    assert command[command.index("--model-id") + 1] == candidate.model_id
    assert command[command.index("--provider-slug") + 1] == candidate.provider_tag
    assert command[command.index("--expected-canonical-model-slug") + 1] == (
        candidate.canonical_model_slug
    )
    assert command[command.index("--expected-endpoint-execution-sha256") + 1] == (
        candidate.endpoint_execution_sha256
    )
    assert "--contract-only" in command
    environment = observed["environment"]
    assert environment["FLAVOURBENCH_OPENROUTER_MAX_PROMPT_PRICE_PER_MTOK"] == "1"
    assert environment["FLAVOURBENCH_OPENROUTER_MAX_COMPLETION_PRICE_PER_MTOK"] == "2"
    assert summary["outcomes"][0]["decision"] == "contract_passed"
    entries = load_ledger(tmp_path / "ledger.jsonl")
    assert [entry["event_type"] for entry in entries] == [
        "reservation_created",
        "artifact_recorded",
    ]
    assert active_ledger_reservations(entries) == {}


def test_execute_requires_explicit_confirmation(tmp_path: Path) -> None:
    path, manifest = _candidate_manifest(tmp_path)
    with pytest.raises(AdmissionDenied, match="execution requires"):
        run_frontier_contracts(
            manifest_path=path,
            expected_manifest_sha256=manifest["content_address"]["digest"],
            live_smoke_directory=tmp_path / "smoke",
            corrections_directory=tmp_path / "corrections",
            ledger_path=tmp_path / "ledger.jsonl",
            summary_directory=tmp_path / "summaries",
            execute=True,
        )
