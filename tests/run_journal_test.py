from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from flavourbench.provider import (
    OpenRouterProvider,
    ProviderError,
    UncertainDeliveryError,
)
from flavourbench.run_journal import (
    JournalIntegrityError,
    RunJournal,
    load_run_journal,
    recovery_state,
    scan_recovery_journals,
    verify_journal_descriptor,
)


def _attempt(event_type: str, *, generation_id: str = "") -> dict:
    return {
        "attempt_id": "attempt-1",
        "arm_id": "run:epicure_on",
        "request_key_sha256": "a" * 64,
        "phase": "tool_round_0",
        "attempt_index": 0,
        "event_type": event_type,
        "generation_id": generation_id,
        "http_status": 200 if event_type == "response_received" else None,
        "error_type": "",
        "payload_sha256": "b" * 64,
        "metadata": {},
    }


def test_journal_is_hash_chained_fsynced_and_content_addressed(tmp_path: Path) -> None:
    journal = RunJournal.create(
        tmp_path,
        run_id="run-1",
        metadata={"requested_model_id": "vendor/model", "prompt_sha256": "c" * 64},
    )
    journal.append("provider_attempt", _attempt("request_started"))
    journal.append(
        "mcp_trace",
        {
            "arm_id": "run:epicure_on",
            "round_index": 0,
            "name": "find_pairings",
            "arguments": {"ingredients": ["mint"]},
            "result": "bounded evidence",
            "result_sha256": "d" * 64,
            "latency_ms": 2,
            "is_error": False,
        },
    )
    descriptor = journal.finalize({"status": "complete"})

    entries = verify_journal_descriptor(tmp_path, descriptor.payload())
    assert [entry["event_type"] for entry in entries] == [
        "run_started",
        "provider_attempt",
        "mcp_trace",
        "run_finalized",
    ]
    assert descriptor.filename.endswith(f"{descriptor.sha256}.jsonl")
    assert (tmp_path / descriptor.filename).stat().st_mode & 0o077 == 0

    lines = (tmp_path / descriptor.filename).read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["phase"] = "changed"
    lines[1] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    (tmp_path / descriptor.filename).write_text("\n".join(lines) + "\n")
    with pytest.raises(JournalIntegrityError, match="digest mismatch"):
        load_run_journal(tmp_path / descriptor.filename)


def test_crash_journal_blocks_replay_and_exposes_generation_for_reconciliation(
    tmp_path: Path,
) -> None:
    journal = RunJournal.create(tmp_path, run_id="crash", metadata={"fixture": True})
    journal.append("provider_attempt", _attempt("request_started"))
    crashed = recovery_state(journal.path)
    assert crashed.uncertain_attempt_ids == ("attempt-1",)
    assert crashed.safe_to_replay is False
    assert crashed.recovery_action == "hold_uncertain_delivery_and_reconcile_provider_state"

    journal.append(
        "provider_attempt",
        _attempt("response_received", generation_id="gen-1"),
    )
    received = recovery_state(journal.path)
    assert received.generation_ids == ("gen-1",)
    assert received.unreconciled_generation_ids == ("gen-1",)
    assert received.safe_to_replay is False
    assert received.recovery_action == "reconcile_generation_ids_before_any_retry"

    reconciled = _attempt("accounting_reconciled", generation_id="gen-1")
    reconciled["metadata"] = {"reconciled": True, "cost_micros": 12}
    journal.append("provider_attempt", reconciled)
    state = recovery_state(journal.path)
    assert state.unreconciled_generation_ids == ()
    assert state.safe_to_replay is False


def test_valid_header_only_journal_proves_pre_provider_failure(tmp_path: Path) -> None:
    journal = RunJournal.create(
        tmp_path,
        run_id="preflight",
        metadata={"dataset_work_item_id": "f" * 64},
    )
    state = recovery_state(journal.path)
    assert state.generation_ids == ()
    assert state.uncertain_attempt_ids == ()
    assert state.safe_to_replay is True
    assert state.recovery_action == (
        "pre_provider_failure_safe_under_parent_reservation_policy"
    )


@pytest.mark.asyncio
async def test_provider_does_not_send_until_request_started_is_durable(
    tmp_path: Path,
) -> None:
    journal = RunJournal.create(tmp_path, run_id="barrier", metadata={"fixture": True})
    observed: list[str] = []

    def sink(event: object) -> None:
        payload = event.__dict__
        journal.append("provider_attempt", payload)
        observed.append(str(payload["event_type"]))

    provider = OpenRouterProvider(attempt_sink=sink)

    class FailingClient:
        async def post(self, *args: object, **kwargs: object) -> object:
            entries = load_run_journal(journal.path)
            assert entries[-1]["payload"]["event_type"] == "request_started"
            raise httpx.ConnectError("pre-send fixture")

    original = provider.client
    provider.client = FailingClient()  # type: ignore[assignment]
    try:
        with pytest.raises(ProviderError):
            await provider._post(  # noqa: SLF001
                {"model": "vendor/model"},
                "idempotency-key",
                arm_id="arm-1",
                phase="final",
            )
    finally:
        await original.aclose()
        await provider.accounting_client.aclose()
    assert observed == ["request_started", "pre_send_failure"]
    assert recovery_state(journal.path).safe_to_replay is True


@pytest.mark.asyncio
async def test_gateway_503_is_uncertain_and_is_never_replayed(tmp_path: Path) -> None:
    journal = RunJournal.create(tmp_path, run_id="gateway-503", metadata={"fixture": True})
    observed: list[str] = []

    def sink(event: object) -> None:
        payload = event.__dict__
        journal.append("provider_attempt", payload)
        observed.append(str(payload["event_type"]))

    provider = OpenRouterProvider(attempt_sink=sink)
    provider.settings = SimpleNamespace(max_provider_attempts=2)

    class Gateway503Client:
        calls = 0

        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            del args, kwargs
            self.calls += 1
            return httpx.Response(
                503,
                request=httpx.Request("POST", "https://gateway.example/v1/chat/completions"),
            )

    fake = Gateway503Client()
    original = provider.client
    provider.client = fake  # type: ignore[assignment]
    try:
        with pytest.raises(UncertainDeliveryError, match="possible upstream dispatch"):
            await provider._post(  # noqa: SLF001
                {"model": "vendor/model"},
                "idempotency-key",
                arm_id="arm-503",
                phase="final",
            )
    finally:
        await original.aclose()
        await provider.accounting_client.aclose()

    assert fake.calls == 1
    assert observed == ["request_started", "uncertain_delivery"]
    assert recovery_state(journal.path).safe_to_replay is False


def test_provider_error_envelope_classifier_never_retains_message_or_raw_body() -> None:
    classified = OpenRouterProvider.classify_response_envelope(
        {
            "error": {
                "code": 429,
                "type": "provider_error",
                "message": "sensitive upstream text",
                "metadata": {
                    "provider_name": "OpenAI",
                    "raw": "sensitive raw response",
                },
            }
        }
    )

    assert classified == {
        "classification": "openrouter_error_envelope",
        "accepted_chat_completion": False,
        "error_code": 429,
        "error_type": "provider_error",
        "provider": "OpenAI",
        "retryable": True,
    }
    assert "sensitive" not in json.dumps(classified)


@pytest.mark.parametrize(
    ("payload", "classification"),
    [
        (
            {"success": False, "result": None, "errors": [{"code": 1000}]},
            "gateway_api_envelope",
        ),
        ({"object": "response", "output": []}, "responses_api_schema_mismatch"),
        ({"unexpected": "shape"}, "unknown_non_chat_completion_envelope"),
        (
            {"id": "gen-1", "choices": [{"message": {"content": "ok"}}]},
            "chat_completions",
        ),
    ],
)
def test_provider_response_envelope_classifier_distinguishes_safe_shapes(
    payload: dict,
    classification: str,
) -> None:
    result = OpenRouterProvider.classify_response_envelope(payload)

    assert result["classification"] == classification
    assert result["accepted_chat_completion"] is (
        classification == "chat_completions"
    )


@pytest.mark.asyncio
async def test_http_200_error_envelope_is_rejected_and_safely_journaled() -> None:
    observed: list[dict] = []
    provider = OpenRouterProvider(attempt_sink=lambda event: observed.append(event.__dict__))
    provider.settings = SimpleNamespace(
        max_provider_attempts=1,
        openrouter_base_url="https://gateway.example/v1",
    )

    class ErrorEnvelopeClient:
        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            del args, kwargs
            return httpx.Response(
                200,
                json={
                    "error": {
                        "code": 400,
                        "type": "invalid_request_error",
                        "message": "must not enter the journal",
                        "metadata": {"provider_name": "OpenAI", "raw": "private"},
                    }
                },
                headers={"cf-aig-cache-status": "MISS"},
                request=httpx.Request(
                    "POST", "https://gateway.example/v1/chat/completions"
                ),
            )

    fake = ErrorEnvelopeClient()
    original = provider.client
    provider.client = fake  # type: ignore[assignment]
    try:
        with pytest.raises(ProviderError, match="openrouter_error_envelope code=400"):
            await provider._post(  # noqa: SLF001
                {"model": "vendor/model"},
                "idempotency-key",
                arm_id="arm-error",
                phase="planning",
            )
    finally:
        await original.aclose()
        await provider.accounting_client.aclose()

    assert [event["event_type"] for event in observed] == [
        "request_started",
        "request_rejected",
    ]
    rejected = observed[1]["metadata"]["response_envelope"]
    assert rejected["classification"] == "openrouter_error_envelope"
    assert rejected["error_code"] == 400
    assert rejected["retryable"] is False
    assert observed[1]["generation_id"] == ""
    assert "must not enter" not in json.dumps(observed)


@pytest.mark.asyncio
async def test_http_200_retryable_error_envelope_retries_without_generation_or_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict] = []
    sleeps: list[float] = []
    provider = OpenRouterProvider(attempt_sink=lambda event: observed.append(event.__dict__))
    provider.settings = SimpleNamespace(
        max_provider_attempts=2,
        openrouter_base_url="https://gateway.example/v1",
    )

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("flavourbench.provider.asyncio.sleep", record_sleep)

    class RetryThenCompletionClient:
        calls = 0

        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            del args, kwargs
            self.calls += 1
            request = httpx.Request(
                "POST", "https://gateway.example/v1/chat/completions"
            )
            if self.calls == 1:
                return httpx.Response(
                    200,
                    json={
                        "id": "must-not-be-treated-as-a-generation",
                        "error": {"code": 429, "message": "private"},
                    },
                    headers={"cf-aig-cache-status": "MISS", "Retry-After": "0"},
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "id": "gen-success",
                    "model": "vendor/model",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "ok"}}
                    ],
                },
                headers={"cf-aig-cache-status": "MISS"},
                request=request,
            )

    fake = RetryThenCompletionClient()
    original = provider.client
    provider.client = fake  # type: ignore[assignment]
    try:
        result = await provider._post(  # noqa: SLF001
            {"model": "vendor/model"},
            "idempotency-key",
            arm_id="arm-retry",
            phase="tool_round_0",
        )
    finally:
        await original.aclose()
        await provider.accounting_client.aclose()

    assert result["id"] == "gen-success"
    assert result["_flavourbench_retries"] == 1
    assert fake.calls == 2
    assert len(sleeps) == 1
    assert [event["event_type"] for event in observed] == [
        "request_started",
        "request_rejected",
        "retry_scheduled",
        "request_started",
        "response_received",
    ]
    assert observed[1]["generation_id"] == ""
    assert observed[1]["metadata"]["response_envelope"]["retryable"] is True
    assert observed[-1]["generation_id"] == "gen-success"
    assert {
        event["generation_id"]
        for event in observed
        if event["event_type"] == "response_received"
    } == {"gen-success"}


def test_journal_rejects_secret_bearing_payload(tmp_path: Path) -> None:
    journal = RunJournal.create(tmp_path, run_id="secret", metadata={"fixture": True})
    with pytest.raises(JournalIntegrityError, match="forbidden"):
        journal.append("provider_attempt", {"authorization": "Bearer secret"})


def test_scanner_routes_crash_journal_by_dataset_work_item(tmp_path: Path) -> None:
    selected = RunJournal.create(
        tmp_path,
        run_id="selected",
        metadata={"dataset_work_item_id": "a" * 64},
    )
    selected.append("provider_attempt", _attempt("request_started"))
    RunJournal.create(
        tmp_path,
        run_id="other",
        metadata={"dataset_work_item_id": "b" * 64},
    )
    states = scan_recovery_journals(tmp_path, dataset_work_item_id="a" * 64)
    assert [state.run_id for state in states] == ["selected"]
    assert states[0].safe_to_replay is False
