from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from flavourbench.direct_qwencloud_pair import run_pair
from flavourbench.qwencloud_catalog import (
    build_catalog_artifact,
    build_unranked_qwen38_alias_route_manifest,
    write_qwencloud_route_manifest,
)
from flavourbench.qwencloud_smoke_admission import (
    CONDITIONS,
    EPICURE_MCP_URL,
    EPICURE_PROVENANCE_URL,
    HUMAN_PI_CONFIRMATION,
    LIVE_CONFIRMATION,
    PREDECESSOR_FAILURE_ARTIFACT_SHA256,
    RESERVATION_CONFIRMATION,
    QwenCloudSmokeAdmissionError,
    _live_source_sha256,
    _sha256,
    _write_content_addressed,
    begin_execution,
    build_human_pi_authorization,
    build_smoke_binding,
    load_ledger,
    qwen38_smoke_execution_policy,
    reserve_and_write_template,
    terminalize_source,
    validate_ledger_state,
    verify_go_template,
    verify_human_pi_authorization,
    verify_preflight_artifact,
)

PROMPT = (
    "How to reduce the sweetness of a sauce?\n\n"
    "Often my sauces are a tad too sweet. I usually use a base of onions, then some "
    "vegetables and broth. Is there something I could add that would reduce the sense "
    "of sweetness?"
)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _content_artifact(payload: dict) -> dict:
    return {**payload, "artifact_sha256": _sha256(payload)}


def _route(
    tmp_path: Path,
    *,
    observed_at: str = "2026-08-08T16:29:04Z",
) -> tuple[Path, str]:
    catalog = build_catalog_artifact(
        catalog={
            "object": "list",
            "data": [
                {
                    "id": "qwen3.8-max",
                    "object": "model",
                    "created": 1,
                    "owned_by": "system",
                }
            ],
        },
        observed_at=observed_at,
        response_date="Sat, 08 Aug 2026 16:29:04 GMT",
    )
    catalog = _content_artifact(catalog)
    route = build_unranked_qwen38_alias_route_manifest(
        catalog_artifact=catalog,
        cap_usd="2",
        allow_mutable_alias_exploratory=True,
        tool_auto_successor_failure_sha256=(
            PREDECESSOR_FAILURE_ARTIFACT_SHA256
        ),
    )
    path = write_qwencloud_route_manifest(route, tmp_path / "route")
    return path, route["content_address"]["digest"]


def _task_validity(tmp_path: Path) -> tuple[Path, str]:
    prompt_sha = __import__("hashlib").sha256(PROMPT.encode()).hexdigest()
    payload = {
        "schema_version": "flavourbench-development-task-validity-v2",
        "tasks": [
            {
                "task_id": "fb-s0-composition-024",
                "task_sha256": "1" * 64,
                "family": "composition",
                "prompt": PROMPT,
                "prompt_sha256": prompt_sha,
                "source_url": "https://cooking.stackexchange.com/questions/137010",
                "source_license": "CC BY-SA 4.0",
                "surface_dependency_screen": {
                    "status": "pass",
                    "failure_reasons": [],
                },
                "confirmatory_eligible": False,
                "rank_eligible": False,
            }
        ],
    }
    artifact = _content_artifact(payload)
    path = _write_json(tmp_path / "task-validity.json", artifact)
    return path, artifact["artifact_sha256"]


def _pi_record(tmp_path: Path) -> tuple[Path, str, dict]:
    artifact = _content_artifact(
        {
            "schema_version": "flavourbench-reasoning-effort-independent-go-v4",
            "authorization_is_transparent_human_pi_record": True,
            "reviewer_identity": {
                "full_name": "Josef Chen",
                "role": "human_principal_investigator",
                "affiliation": "independent_research",
            },
        }
    )
    path = _write_json(tmp_path / "pi-record.json", artifact)
    return path, artifact["artifact_sha256"], artifact


def _frozen(tmp_path: Path):  # type: ignore[no-untyped-def]
    route_path, route_sha = _route(tmp_path)
    task_path, task_sha = _task_validity(tmp_path)
    pi_path, pi_sha, pi = _pi_record(tmp_path)
    binding = build_smoke_binding(
        route_manifest_path=route_path,
        expected_route_manifest_sha256=route_sha,
        task_validity_path=task_path,
        expected_task_validity_sha256=task_sha,
        cap_usd=Decimal("2"),
    )
    ledger = tmp_path / "governance" / "qwen-ledger.jsonl"
    reservation, template_path = reserve_and_write_template(
        binding=binding,
        ledger_path=ledger,
        output_directory=tmp_path / "governance",
        human_pi_identity_record_path=pi_path,
        expected_human_pi_identity_record_sha256=pi_sha,
        confirmation=RESERVATION_CONFIRMATION,
    )
    template_document = json.loads(template_path.read_text(encoding="utf-8"))
    template = verify_go_template(
        template_path,
        expected_sha256=template_document["artifact_sha256"],
    )
    return binding, ledger, reservation, template_path, template, pi


def _preflight(tmp_path: Path, template: dict) -> tuple[Path, dict]:
    model = template["model_identity"]
    task = template["task"]
    execution = template["execution"]
    reservation = template["reservation"]
    artifact = _content_artifact(
        {
            "schema_version": "flavourbench-direct-provider-preflight-v1",
            "status": "preflight_passed_no_provider_calls",
            "provider_calls_made": False,
            "epicure_attestation_performed": True,
            "epicure_mcp_url": EPICURE_MCP_URL,
            "epicure_provenance_url": EPICURE_PROVENANCE_URL,
            "official": False,
            "season_eligible": False,
            "rank_eligible": False,
            "research_result": False,
            "model_id": "qwen3.8-max",
            "canonical_model_slug": "qwen3.8-max",
            "provider_slug": "qwencloud-direct",
            "execution_backend": "qwencloud_direct",
            "candidate_manifest_sha256": model["route_manifest_sha256"],
            "endpoint_execution_sha256": model["endpoint_execution_sha256"],
            "backend_contract_sha256": model["backend_contract_sha256"],
            "execution_policy_sha256": execution["execution_policy_sha256"],
            "dataset_work_item_id": reservation["work_item_id"],
            "dataset_task_id": task["task_id"],
            "category": task["family"],
            "prompt_sha256": task["prompt_sha256"],
            "conditions": list(CONDITIONS),
            "cap_usd": "2",
            "forecast_worst_case_usd": "2",
            "full_unpriced_budget_ceiling_retained": True,
            "provider_cost_known": False,
            "reservation_entry_sha256": reservation["entry_sha256"],
            "go_template_sha256": template["artifact_sha256"],
            "model_identity_label": (
                "catalog_pinned_at_observation_not_a_frozen_model"
            ),
            "protocol_bundle_sha256": "2" * 64,
            "epicure_release_id": "exploratory-unmatched-1790-runtime",
            "epicure_bundle_sha256": "3" * 64,
            "epicure_application_sha256": "4" * 64,
            "epicure_tool_schema_sha256": "5" * 64,
        }
    )
    return _write_json(tmp_path / "preflight.json", artifact), artifact


def _authorization(
    tmp_path: Path,
    *,
    template: dict,
    preflight: dict,
    pi: dict,
) -> tuple[Path, dict]:
    payload = build_human_pi_authorization(
        template=template,
        preflight=preflight,
        standing_human_pi_record=pi,
        confirmation=HUMAN_PI_CONFIRMATION,
        recorded_at="2026-08-08T19:00:00Z",
    )
    path = _write_content_addressed(
        tmp_path,
        "human-pi-go",
        payload,
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return path, artifact


def _source(tmp_path: Path, template: dict) -> Path:
    payload = {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": template["execution"]["frozen_run_id"],
        "status": "complete_unpriced_budget_ceiling",
        "requested_model_id": "qwen3.8-max",
        "requested_provider": "qwencloud-direct",
        "execution_backend": "qwencloud_direct",
        "candidate_manifest_sha256": template["model_identity"][
            "route_manifest_sha256"
        ],
        "dataset_work_item_id": template["reservation"]["work_item_id"],
        "dataset_task_id": template["task"]["task_id"],
        "prompt_sha256": template["task"]["prompt_sha256"],
        "execution_policy_sha256": template["execution"][
            "execution_policy_sha256"
        ],
        "mutable_alias_exploratory_opt_in": True,
        "official": False,
        "rank_eligible": False,
        "requested_conditions": list(CONDITIONS),
        "budget": {
            "cap_usd": "2",
            "actual_cost_micros": 0,
            "provider_cost_known": False,
            "full_unpriced_budget_ceiling_retained": True,
            "retained_exposure_usd": "2",
            "zero_recorded_cost_means": "unknown_not_free",
        },
        "unicode_digest_probe": "crème brûlée",
    }
    artifact = {**payload, "artifact_sha256": _live_source_sha256(payload)}
    return _write_json(tmp_path / f"source-{artifact['artifact_sha256']}.json", artifact)


def test_qwen38_policy_uses_full_quality_caps_without_invented_reasoning() -> None:
    policy = qwen38_smoke_execution_policy()
    assert policy.max_output_tokens == 4096
    assert policy.max_intermediate_tokens == 4096
    assert policy.max_tool_rounds == 8
    assert policy.max_provider_attempts == 1
    assert policy.epicure_on_tool_required is True
    assert policy.intermediate_reasoning_effort is None
    assert policy.final_reasoning_effort is None
    assert policy.final_response_mode == "plain_text"


def test_reservation_and_go_template_are_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    binding, ledger, reservation, template_path, template, pi = _frozen(tmp_path)
    entries = load_ledger(ledger)
    state = validate_ledger_state(entries)
    assert len(entries) == 2
    assert reservation["reserved_usd"] == "2"
    assert reservation["full_ceiling_permanently_retained"] is True
    assert state.total_retained_exposure_usd == Decimal("2")
    assert template["reservation"]["entry_sha256"] == reservation["entry_sha256"]
    assert template["reservation"]["season_scope"] == "season1_unranked_development"
    assert template["reservation"]["season_budget_cap_usd"] == "2"
    assert template["model_identity"]["identity_kind"] == "mutable_alias"
    assert template["task"]["prompt_sha256"] == binding.task["prompt_sha256"]
    assert template["claim_boundary"]["leaderboard_comparisons_authorized"] == 0

    pi_path = _write_json(tmp_path / "pi-again.json", pi)
    second, second_template = reserve_and_write_template(
        binding=binding,
        ledger_path=ledger,
        output_directory=tmp_path / "governance",
        human_pi_identity_record_path=pi_path,
        expected_human_pi_identity_record_sha256=pi["artifact_sha256"],
        confirmation=RESERVATION_CONFIRMATION,
    )
    assert second["entry_sha256"] == reservation["entry_sha256"]
    assert second_template == template_path
    assert len(load_ledger(ledger)) == 2


def test_successor_reservation_append_only_corrects_cumulative_season_exposure(
    tmp_path: Path,
) -> None:
    _, ledger, _, _, _, _ = _frozen(tmp_path)
    route_path, route_sha = _route(
        tmp_path,
        observed_at="2026-08-08T16:30:04Z",
    )
    task_path, task_sha = _task_validity(tmp_path)
    pi_path, pi_sha, _ = _pi_record(tmp_path)
    successor = build_smoke_binding(
        route_manifest_path=route_path,
        expected_route_manifest_sha256=route_sha,
        task_validity_path=task_path,
        expected_task_validity_sha256=task_sha,
        cap_usd=Decimal("2"),
    )
    reservation, template_path = reserve_and_write_template(
        binding=successor,
        ledger_path=ledger,
        output_directory=tmp_path / "governance",
        human_pi_identity_record_path=pi_path,
        expected_human_pi_identity_record_sha256=pi_sha,
        confirmation=RESERVATION_CONFIRMATION,
    )
    template_document = json.loads(template_path.read_text(encoding="utf-8"))
    state = validate_ledger_state(load_ledger(ledger))

    assert state.total_retained_exposure_usd == Decimal("4")
    adjustment = state.scope_adjustments[reservation["entry_sha256"]]
    assert adjustment["season_retained_exposure_usd"] == "4"
    assert adjustment["season_budget_cap_usd"] == "4"
    assert template_document["reservation"]["scope_adjustment_entry_sha256"] == (
        adjustment["entry_sha256"]
    )
    assert template_document["reservation"]["season_retained_exposure_usd"] == "4"


def test_crash_after_start_retains_full_ceiling_and_permanently_blocks_replay(
    tmp_path: Path,
) -> None:
    _, ledger, reservation, _, template, pi = _frozen(tmp_path)
    preflight_path, preflight_document = _preflight(tmp_path, template)
    preflight = verify_preflight_artifact(
        preflight_path,
        expected_sha256=preflight_document["artifact_sha256"],
        template=template,
    )
    authorization_path, authorization_document = _authorization(
        tmp_path,
        template=template,
        preflight=preflight,
        pi=pi,
    )
    authorization = verify_human_pi_authorization(
        authorization_path,
        expected_sha256=authorization_document["artifact_sha256"],
        template=template,
        preflight=preflight,
    )
    started = begin_execution(
        ledger_path=ledger,
        template=template,
        preflight=preflight,
        authorization=authorization,
        confirmation=LIVE_CONFIRMATION,
    )
    assert started["provider_delivery_may_begin_after_this_fsync"] is True
    state = validate_ledger_state(load_ledger(ledger))
    assert state.total_retained_exposure_usd == Decimal("2")
    assert reservation["entry_sha256"] in state.starts
    assert not state.terminalizations

    with pytest.raises(QwenCloudSmokeAdmissionError, match="replay is permanently prohibited"):
        begin_execution(
            ledger_path=ledger,
            template=template,
            preflight=preflight,
            authorization=authorization,
            confirmation=LIVE_CONFIRMATION,
        )


def test_success_terminalizes_at_full_ceiling_not_zero(tmp_path: Path) -> None:
    _, ledger, reservation, _, template, pi = _frozen(tmp_path)
    preflight_path, preflight_document = _preflight(tmp_path, template)
    preflight = verify_preflight_artifact(
        preflight_path,
        expected_sha256=preflight_document["artifact_sha256"],
        template=template,
    )
    _, authorization = _authorization(
        tmp_path,
        template=template,
        preflight=preflight,
        pi=pi,
    )
    begin_execution(
        ledger_path=ledger,
        template=template,
        preflight=preflight,
        authorization=authorization,
        confirmation=LIVE_CONFIRMATION,
    )
    source = _source(tmp_path, template)
    terminal = terminalize_source(
        ledger_path=ledger,
        template=template,
        authorization=authorization,
        artifact_path=source,
    )
    assert terminal["retained_exposure_usd"] == "2"
    assert terminal["provider_reported_cost_micros"] == 0
    assert terminal["zero_recorded_cost_means"] == "unknown_not_free"
    state = validate_ledger_state(load_ledger(ledger))
    assert state.total_retained_exposure_usd == Decimal("2")
    assert reservation["entry_sha256"] in state.terminalizations


@pytest.mark.asyncio
async def test_direct_runner_starts_durably_before_call_and_never_delivers_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, ledger, reservation, template_path, template, pi = _frozen(tmp_path)
    preflight_path, preflight_document = _preflight(tmp_path, template)
    _, authorization = _authorization(
        tmp_path,
        template=template,
        preflight=preflight_document,
        pi=pi,
    )
    authorization_path = next(tmp_path.glob("human-pi-go-*.json"))
    source_path = _source(tmp_path, template)
    calls = 0

    async def fake_direct(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        state = validate_ledger_state(load_ledger(ledger))
        assert reservation["entry_sha256"] in state.starts
        return {
            "status": "complete_unpriced_budget_ceiling",
            "artifact": str(source_path),
            "artifact_sha256": json.loads(source_path.read_text())["artifact_sha256"],
        }

    monkeypatch.setattr(
        "flavourbench.direct_qwencloud_pair._run_direct_pair",
        fake_direct,
    )
    monkeypatch.setattr(
        "flavourbench.direct_qwencloud_pair.get_settings",
        lambda: SimpleNamespace(
            execution_mode="live",
            live_authorized=True,
            qwencloud_api_key="configured-not-real",
            mcp_url=EPICURE_MCP_URL,
        ),
    )
    args = SimpleNamespace(
        go_template=template_path,
        expected_go_template_sha256=template["artifact_sha256"],
        reservation_ledger=ledger,
        reservation_entry_sha256=reservation["entry_sha256"],
        candidate_manifest_sha256=binding.route_manifest_sha256,
        model_id="qwen3.8-max",
        provider_slug="qwencloud-direct",
        expected_canonical_model_slug="qwen3.8-max",
        expected_endpoint_execution_sha256=binding.candidate.endpoint_execution_sha256,
        expected_execution_policy_sha256=binding.execution_policy.sha256,
        dataset_work_item_id=binding.work_item_id,
        dataset_task_id=binding.task["task_id"],
        category="composition",
        prompt=binding.task["prompt"],
        cap_usd=Decimal("2"),
        condition=None,
        plain_text_final=True,
        evidence_protocol="matched_evidence_v2",
        require_epicure_call=True,
        sequential_arms=False,
        intermediate_reasoning_effort=None,
        final_reasoning_effort=None,
        allow_mutable_alias_exploratory=True,
        tool_catalog_bytes_bound=24_000,
        frozen_run_id="",
        frozen_attempt_slots=None,
        preflight_only=False,
        preflight=preflight_path,
        expected_preflight_sha256=preflight_document["artifact_sha256"],
        human_pi_authorization=authorization_path,
        expected_human_pi_authorization_sha256=authorization["artifact_sha256"],
        expected_epicure_release_id=preflight_document["epicure_release_id"],
        expected_epicure_bundle_sha256=preflight_document["epicure_bundle_sha256"],
        expected_epicure_application_sha256=preflight_document[
            "epicure_application_sha256"
        ],
        expected_epicure_tool_schema_sha256=preflight_document[
            "epicure_tool_schema_sha256"
        ],
        live_confirm=LIVE_CONFIRMATION,
    )
    result = await run_pair(args)
    assert result["retained_exposure_usd"] == "2"
    assert calls == 1

    with pytest.raises(QwenCloudSmokeAdmissionError, match="replay is permanently prohibited"):
        await run_pair(args)
    assert calls == 1
