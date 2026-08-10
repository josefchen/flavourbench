from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from flavourbench.bedrock_auth import BedrockClients, BedrockLaneSettings
from flavourbench.bedrock_contract_smoke import (
    EXECUTION_CONFIRMATION,
    BedrockSmokeError,
    _assert_anthropic_use_case_ready,
    _load_epicure_tool_catalog,
    _safe_aws_error_details,
    _safe_response_sha256,
    _terminal_artifact,
    _tools_from_catalog,
    dry_run_plan,
    execute_contract_smoke,
    parser,
    project_epicure_tool_catalog,
)
from flavourbench.bedrock_manifest import assert_public_catalog_safe
from flavourbench.bedrock_provider import structured_output_config
from flavourbench.bedrock_smoke_ledger import (
    PROTECTED_ENTRY_FIELDS,
    BedrockSmokeLedger,
    BedrockSmokeLedgerError,
    sha256_json,
)
from flavourbench.mcp_client import tool_catalog_sha256

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / (
    "artifacts/bedrock/catalog/"
    "bedrock-catalog-bd78cad4246faff8cd72fd288dd268e856692eb00314a19f0e916bb9318144e6.json"
)
EVIDENCE = ROOT / "contracts/evidence/claude-haiku-4-5-global-2026-07-15-v10.json"
EVIDENCE_FILE_DIGEST = "048bac6da3f98e0625f3c5b3c7840f4bcfcbac0f6b05b056a1f8f79c07af6896"
MANIFEST_DIGEST = "13e55aa50acea7ac5ba06ccf055e4d19eadb01e7a92007b996bce41d5a8293f3"
MANIFEST = ROOT / f"artifacts/bedrock/contracts/bedrock-smoke-manifest-{MANIFEST_DIGEST}.json"
REAL_TOOL_DIGEST = "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd"
REAL_TOOL_FIXTURE = ROOT / f"contracts/epicure/tool-catalog-{REAL_TOOL_DIGEST}.json"
REAL_PROJECTED_TOOL_DIGEST = (
    "73ec3f96008b44e87524acde9cee4c247b333607e86453e062d17d4b26ce7d7b"
)
FAKE_SECRET = "bedrock-fake-secret-must-never-appear"
FAKE_ACCOUNT_ID = "123456" + "789012"

TOOL_CATALOG = [
    {
        "name": "find_pairings",
        "description": "Find culinary pairings for supplied ingredients",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ingredients": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["ingredients"],
            "additionalProperties": False,
        },
    }
]


def settings(*, hard_cap: str = "5000", smoke_cap: str = "5") -> BedrockLaneSettings:
    return BedrockLaneSettings.from_environ(
        {
            "FLAVOURBENCH_BEDROCK_ENABLED": "true",
            "FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED": "true",
            "FLAVOURBENCH_BEDROCK_CAP_USD": hard_cap,
            "FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_CAP_USD": smoke_cap,
            "FLAVOURBENCH_BEDROCK_STAGE": "contract_smoke",
            "FLAVOURBENCH_BEDROCK_PROFILE_SCOPE": "global",
            "AWS_REGION": "eu-west-1",
            "AWS_BEARER_TOKEN_BEDROCK": FAKE_SECRET,
        }
    )


def epicure_contract(tmp_path: Path) -> Path:
    path = tmp_path / "epicure-contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "flavourbench-epicure-runtime-contract-v1",
                "release_id": "fake-epicure-release",
                "bundle_sha256": "1" * 64,
                "application_sha256": "2" * 64,
                "tool_schema_sha256": tool_catalog_sha256(TOOL_CATALOG),
                "ingredient_count": 1790,
                "embedding_dimensions": 300,
                "status": "test_fixture",
                "rank_eligible": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def arguments(tmp_path: Path, epicure_path: Path) -> Any:
    tool_digest = tool_catalog_sha256(TOOL_CATALOG)
    tool_fixture = tmp_path / f"tool-catalog-{tool_digest}.json"
    tool_fixture.write_text(
        json.dumps(TOOL_CATALOG, sort_keys=True),
        encoding="utf-8",
    )
    return parser().parse_args(
        [
            "--catalog",
            str(CATALOG),
            "--evidence",
            str(EVIDENCE),
            "--epicure-contract",
            str(epicure_path),
            "--epicure-tool-catalog",
            str(tool_fixture),
            "--manifest",
            str(MANIFEST),
            "--expected-manifest-sha256",
            MANIFEST_DIGEST,
            "--output-dir",
            str(tmp_path / "output"),
            "--ledger",
            str(tmp_path / "output/ledger.jsonl"),
            "--execute",
            "--confirm",
            EXECUTION_CONFIRMATION,
        ]
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []
        self.responses = [
            self._final("aws-off", 100, 50, "Use compressed watermelon."),
            {
                "ResponseMetadata": {"RequestId": "aws-on-tool", "HTTPStatusCode": 200},
                "modelId": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                "stopReason": "tool_use",
                "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
                "metrics": {"latencyMs": 8},
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool-use-1",
                                    "name": "find_pairings",
                                    "input": {
                                        "ingredients": [
                                            "watermelon",
                                            "green olive",
                                            "mint",
                                        ]
                                    },
                                }
                            }
                        ],
                    }
                },
            },
            self._final("aws-on-final", 200, 50, "Use Epicure evidence, then taste."),
        ]

    @staticmethod
    def _final(
        request_id: str, input_tokens: int, output_tokens: int, answer: str
    ) -> dict[str, Any]:
        output = {
            "answer_markdown": answer,
            "ingredient_mentions": ["watermelon", "green olive", "mint"],
            "constraints_addressed": ["vegetarian", "no added sugar"],
            "uncertainties": ["final salt balance requires tasting"],
        }
        return {
            "ResponseMetadata": {"RequestId": request_id, "HTTPStatusCode": 200},
            "modelId": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": input_tokens + output_tokens,
            },
            "metrics": {"latencyMs": 11},
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": json.dumps(output, separators=(",", ":"))}],
                }
            },
        }

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def count_tokens(self, **kwargs: Any) -> dict[str, Any]:
        self.count_calls.append(kwargs)
        return {
            "ResponseMetadata": {
                "RequestId": f"count-{len(self.count_calls)}",
                "HTTPStatusCode": 200,
            },
            "inputTokens": 900,
        }


class FakeToolResult:
    structured = {"pairings": [{"ingredient": "green olive", "score": 0.82}]}
    text = ""
    latency_ms = 4
    is_error = False


class FakeMcp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> FakeMcp:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        return TOOL_CATALOG

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeToolResult:
        self.calls.append((name, arguments))
        return FakeToolResult()


def fake_attestation() -> dict[str, Any]:
    return {
        "release_id": "fake-epicure-release",
        "bundle_sha256": "1" * 64,
        "application_sha256": "2" * 64,
        "tool_schema_sha256": tool_catalog_sha256(TOOL_CATALOG),
        "ingredient_count": 1790,
        "embedding_dimensions": 300,
        "tool_count": 1,
    }


async def async_attestation() -> dict[str, Any]:
    return fake_attestation()


def test_dry_run_is_default_and_makes_zero_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    args.execute = False
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_ENABLED", "true")
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED", "false")
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_CAP_USD", "5000")
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_CAP_USD", "5")
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_PROFILE_SCOPE", "global")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    plan = dry_run_plan(args)

    assert plan["external_calls_made"] == 0
    assert plan["bedrock_inference_calls_made"] == 0
    assert plan["epicure_mcp_calls_made"] == 0
    assert plan["arm_reservations_micros"] == {
        "epicure_off": 22_240,
        "epicure_on": 66_720,
    }
    assert plan["rank_eligible"] is False
    assert plan["protocol"] == "bedrock_epicure_contract_smoke_v8"
    assert plan["decoding"] == {"temperature": 0.2, "top_p": None}
    assert plan["run_key"] == (
        "0f731f4983792492d08410a1e2e51fe3bd4e47ceb594dc095ab7401bc333b00f"
    )
    assert plan["run_key"] != (
        "72cba51c432fd2932b1bb4ed14684c13b1d4daab2499bde4c5061f6805cebbb3"
    )
    assert plan["execute_confirmation"] == (
        "RUN_BEDROCK_EPICURE_CONTRACT_SMOKE_V8_SAFE_RESPONSE_HASH"
    )
    assert not Path(args.ledger).exists()


@pytest.mark.asyncio
async def test_fake_e2e_is_content_addressed_private_and_idempotent(tmp_path: Path) -> None:
    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    runtime = FakeRuntime()
    mcp_instances: list[FakeMcp] = []

    def mcp_factory() -> FakeMcp:
        instance = FakeMcp()
        mcp_instances.append(instance)
        return instance

    first = await execute_contract_smoke(
        args,
        runtime_client=runtime,
        mcp_factory=mcp_factory,
        attestor=async_attestation,
        lane_settings=settings(),
    )
    second = await execute_contract_smoke(
        args,
        runtime_client=runtime,
        mcp_factory=mcp_factory,
        attestor=async_attestation,
        lane_settings=settings(),
    )

    assert len(runtime.calls) == 3
    assert len(runtime.count_calls) == 3
    assert first["summary_sha256"] == second["summary_sha256"]
    assert first["governed_exposure_usd"] == "0.00095"
    assert first["billing_actual_reconciliation_status"] == "not_reconciled"
    assert first["rank_eligible"] is False
    assert sum(len(instance.calls) for instance in mcp_instances) == 1
    assert runtime.calls[0]["modelId"].startswith("global.anthropic.claude-haiku")
    assert all(
        call["inferenceConfig"] == {"maxTokens": 2048, "temperature": 0.2}
        for call in runtime.calls
    )
    assert "toolConfig" not in runtime.calls[0]
    assert runtime.calls[1]["toolConfig"]["tools"][0]["toolSpec"]["strict"] is True
    assert all(
        call["modelId"] == "anthropic.claude-haiku-4-5-20251001-v1:0"
        for call in runtime.count_calls
    )
    assert "toolConfig" not in runtime.count_calls[0]["input"]["converse"]
    assert "toolConfig" in runtime.count_calls[1]["input"]["converse"]

    files = list((tmp_path / "output").rglob("*.json"))
    files.append(tmp_path / "output/ledger.jsonl")
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert FAKE_SECRET not in rendered
    assert FAKE_ACCOUNT_ID not in rendered
    for path in files[:-1]:
        assert_public_catalog_safe(json.loads(path.read_text(encoding="utf-8")))

    on_artifact = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in files
        if "bedrock-smoke-epicure_on-" in path.name
    )
    assert on_artifact["generation"]["request_ids"] == ["aws-on-tool", "aws-on-final"]
    assert on_artifact["schema_version"] == "flavourbench-bedrock-epicure-smoke-arm-v8"
    assert on_artifact["source"]["protocol"] == "bedrock_epicure_contract_smoke_v8"
    assert on_artifact["source"]["decoding"] == {
        "temperature": 0.2,
        "top_p": None,
    }
    assert on_artifact["complete_epicure_mcp_trace"][0]["result"]["pairings"]
    assert on_artifact["billing_actual_cost_micros"] is None
    assert on_artifact["rate_card_estimated_cost_micros"] == 600
    summary = json.loads(Path(first["summary"]).read_bytes())
    assert summary["schema_version"] == "flavourbench-bedrock-epicure-smoke-summary-v8"
    assert summary["protocol"] == "bedrock_epicure_contract_smoke_v8"
    assert summary["decoding"] == {"temperature": 0.2, "top_p": None}


def test_v8_runner_refuses_the_superseded_sampling_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    args.evidence = ROOT / (
        "contracts/evidence/claude-haiku-4-5-global-2026-07-15-v3.json"
    )
    args.expected_manifest_sha256 = (
        "1bbdb51e5eef5e9526ce9e608e23dba65f978e83ad6ba28f1d3b3dd762d61ff0"
    )
    args.manifest = ROOT / (
        "artifacts/bedrock/contracts/bedrock-smoke-manifest-"
        f"{args.expected_manifest_sha256}.json"
    )
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_ENABLED", "true")
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED", "false")
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_CAP_USD", "5000")
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_CAP_USD", "5")
    monkeypatch.setenv("FLAVOURBENCH_BEDROCK_PROFILE_SCOPE", "global")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    with pytest.raises(BedrockSmokeError, match="mutually exclusive"):
        dry_run_plan(args)

    assert not Path(args.ledger).exists()


@pytest.mark.asyncio
async def test_v8_preserves_prior_held_ledger_prefix_and_cumulative_exposure(
    tmp_path: Path,
) -> None:
    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    ledger = BedrockSmokeLedger(args.ledger)
    old_run = "1" * 64
    old_arm = "2" * 64
    old_reservation = "3" * 64
    ledger.reserve(
        settings=settings(),
        run_key=old_run,
        arm_id=old_arm,
        reservation_id=old_reservation,
        reservation_micros=15_840,
        payload={"fixture": "superseded_v1"},
    )
    for event_type, payload in (
        ("count_tokens_request_started", {"call_index": 1}),
        ("count_tokens_response_received", {"call_index": 1, "input_tokens": 113}),
        ("converse_request_started", {"call_index": 1}),
        (
            "converse_delivery_uncertain",
            {"call_index": 1, "aws_error_code": "ValidationException"},
        ),
        ("arm_artifact_recorded", {"status": "failed_uncertain"}),
        ("reservation_held_uncertain", {"reason": "superseded_v1_fixture"}),
    ):
        ledger.append(
            event_type,
            run_key=old_run,
            arm_id=old_arm,
            reservation_id=old_reservation,
            reservation_micros=15_840,
            payload=payload,
        )
    prefix = ledger.entries()

    result = await execute_contract_smoke(
        args,
        runtime_client=FakeRuntime(),
        mcp_factory=FakeMcp,
        attestor=async_attestation,
        lane_settings=settings(),
    )

    entries = ledger.entries()
    assert entries[:7] == prefix
    assert result["governed_exposure_usd"] == "0.01679"
    assert ledger.exposure().uncertain_held_reservations_micros == 15_840
    assert ledger.exposure().settled_rate_card_estimate_micros == 950


def test_restart_releases_pre_send_but_holds_ambiguous_delivery(tmp_path: Path) -> None:
    ledger = BedrockSmokeLedger(tmp_path / "ledger.jsonl")
    run_key = "a" * 64
    safe_arm = "b" * 64
    safe_reservation = "c" * 64
    ledger.reserve(
        settings=settings(),
        run_key=run_key,
        arm_id=safe_arm,
        reservation_id=safe_reservation,
        reservation_micros=10_000,
        payload={"fixture": "pre_send_crash"},
    )

    assert _terminal_artifact(
        ledger,
        run_key=run_key,
        arm_id=safe_arm,
        artifact_directory=tmp_path,
    ) is None
    assert ledger.next_attempt_index(run_key=run_key, arm_id=safe_arm) == 2
    assert ledger.exposure().governed_exposure_micros == 0

    counted_arm = "1" * 64
    counted_reservation = "2" * 64
    ledger.reserve(
        settings=settings(),
        run_key=run_key,
        arm_id=counted_arm,
        reservation_id=counted_reservation,
        reservation_micros=10_000,
        payload={"fixture": "count_tokens_completed_before_crash"},
    )
    ledger.append(
        "count_tokens_request_started",
        run_key=run_key,
        arm_id=counted_arm,
        reservation_id=counted_reservation,
        reservation_micros=10_000,
        payload={"call_index": 1, "count_tokens_payload_sha256": "3" * 64},
    )
    ledger.append(
        "count_tokens_response_received",
        run_key=run_key,
        arm_id=counted_arm,
        reservation_id=counted_reservation,
        reservation_micros=10_000,
        payload={"call_index": 1, "input_tokens": 900},
    )
    assert _terminal_artifact(
        ledger,
        run_key=run_key,
        arm_id=counted_arm,
        artifact_directory=tmp_path,
    ) is None
    assert ledger.exposure().governed_exposure_micros == 0

    uncertain_arm = "d" * 64
    uncertain_reservation = "e" * 64
    ledger.reserve(
        settings=settings(),
        run_key=run_key,
        arm_id=uncertain_arm,
        reservation_id=uncertain_reservation,
        reservation_micros=20_000,
        payload={"fixture": "request_boundary_crash"},
    )
    ledger.append(
        "converse_request_started",
        run_key=run_key,
        arm_id=uncertain_arm,
        reservation_id=uncertain_reservation,
        reservation_micros=20_000,
        payload={"call_index": 1, "request_payload_sha256": "f" * 64},
    )
    with pytest.raises(BedrockSmokeError, match="full reservation is held"):
        _terminal_artifact(
            ledger,
            run_key=run_key,
            arm_id=uncertain_arm,
            artifact_directory=tmp_path,
        )
    assert ledger.exposure().uncertain_held_reservations_micros == 20_000
    with pytest.raises(BedrockSmokeError, match="unresolved uncertain reservation"):
        _terminal_artifact(
            ledger,
            run_key=run_key,
            arm_id=uncertain_arm,
            artifact_directory=tmp_path,
        )


@pytest.mark.asyncio
async def test_stage_cap_fails_before_any_runtime_call(tmp_path: Path) -> None:
    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    runtime = FakeRuntime()

    with pytest.raises(BedrockSmokeLedgerError, match="not admitted"):
        await execute_contract_smoke(
            args,
            runtime_client=runtime,
            mcp_factory=FakeMcp,
            attestor=async_attestation,
            lane_settings=settings(hard_cap="0.01", smoke_cap="0.01"),
        )

    assert runtime.calls == []
    assert runtime.count_calls == []
    assert BedrockSmokeLedger(args.ledger).entries() == []


@pytest.mark.asyncio
async def test_live_factory_access_preflight_blocks_before_ledger_mcp_or_runtime(
    tmp_path: Path,
) -> None:
    class MissingControl:
        def get_use_case_for_model_access(self) -> dict[str, Any]:
            return {"formData": ""}

    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    runtime = FakeRuntime()
    mcp_created = False
    attestor_called = False

    def client_factory(_: BedrockLaneSettings) -> BedrockClients:
        return BedrockClients(
            control=MissingControl(),
            runtime=runtime,
            region="eu-west-1",
            auth_mode_hint="bedrock_bearer_token_env",
        )

    def mcp_factory() -> FakeMcp:
        nonlocal mcp_created
        mcp_created = True
        return FakeMcp()

    async def attestor() -> dict[str, Any]:
        nonlocal attestor_called
        attestor_called = True
        return fake_attestation()

    with pytest.raises(BedrockSmokeError, match="did not confirm"):
        await execute_contract_smoke(
            args,
            client_factory=client_factory,
            mcp_factory=mcp_factory,
            attestor=attestor,
            lane_settings=settings(),
        )

    assert runtime.calls == []
    assert runtime.count_calls == []
    assert mcp_created is False
    assert attestor_called is False
    assert not Path(args.ledger).exists()


def test_custom_or_identity_bearing_prompts_fail_before_execution(tmp_path: Path) -> None:
    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    args.prompt = f"customer account {FAKE_ACCOUNT_ID}"

    with pytest.raises(BedrockSmokeError, match="custom prompts are prohibited"):
        dry_run_plan(args)

    assert not Path(args.ledger).exists()


def test_reservation_ids_are_deterministic_but_attempt_specific() -> None:
    first = sha256_json({"run_key": "a", "arm_id": "b", "attempt_index": 1})
    second = sha256_json({"run_key": "a", "arm_id": "b", "attempt_index": 2})
    assert first != second


def test_real_epicure_catalog_fixture_and_projection_are_frozen_and_deterministic() -> None:
    raw = json.loads(REAL_TOOL_FIXTURE.read_bytes())
    original = copy.deepcopy(raw)

    frozen = _load_epicure_tool_catalog(
        REAL_TOOL_FIXTURE,
        expected_raw_sha256=REAL_TOOL_DIGEST,
    )
    first = project_epicure_tool_catalog(raw)
    second = project_epicure_tool_catalog(raw)

    assert raw == original
    assert len(raw) == 13
    assert len(first) == 1
    assert first[0]["name"] == "find_pairings"
    assert tool_catalog_sha256(raw) == REAL_TOOL_DIGEST
    assert tool_catalog_sha256(first) == REAL_PROJECTED_TOOL_DIGEST
    assert first == second == list(frozen.bedrock_tools)
    assert len(_tools_from_catalog(first)) == 1
    raw_rendered = json.dumps(raw, sort_keys=True)
    assert '"maxLength": 120' in raw_rendered
    assert '"maximum": 25' in raw_rendered
    rendered = json.dumps([tool["inputSchema"] for tool in first], sort_keys=True)
    for unsupported in (
        '"minLength"',
        '"maxLength"',
        '"minimum"',
        '"maximum"',
        '"maxItems"',
        '"default"',
        '"discriminator"',
        '"oneOf"',
        '"title"',
    ):
        assert unsupported not in rendered
    assert '"anyOf"' in rendered

    def assert_objects_closed(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                assert value.get("additionalProperties") is False
            for child in value.values():
                assert_objects_closed(child)
        elif isinstance(value, list):
            for child in value:
                assert_objects_closed(child)

    assert_objects_closed(first)
    for definition in _tools_from_catalog(first):
        definition.as_converse_tool()


def test_response_hash_falls_back_without_retaining_non_json_sdk_objects() -> None:
    canonical_digest, canonical_mode = _safe_response_sha256({"value": 1})
    projected_digest, projected_mode = _safe_response_sha256(
        {"decimal": Decimal("1.25"), "binary": b"private-response-bytes"}
    )

    assert len(canonical_digest) == len(projected_digest) == 64
    assert canonical_mode == "canonical_json"
    assert projected_mode == "sanitized_sdk_projection"
    assert canonical_digest != projected_digest


def test_superseding_evidence_freezes_sampling_constraint_and_official_price() -> None:
    evidence = json.loads(EVIDENCE.read_bytes())

    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == EVIDENCE_FILE_DIGEST
    assert evidence["capabilities"]["count_tokens"] is True
    assert evidence["inference_constraints"] == {
        "temperature_top_p_mutually_exclusive": True,
        "evidence_note": (
            "Protocol v8 freezes temperature=0.2, omits top_p, retains the "
            "2048-token ceiling and one-tool find_pairings projection, and hashes "
            "non-JSON SDK response members through a sanitized deterministic "
            "projection before journaling."
        ),
    }
    assert evidence["price"]["source_uri"] == (
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
        "AmazonBedrockFoundationModels/20260703085857/index.json"
    )
    assert evidence["price"]["offer_file_sha256"] == (
        "80e5caa7c2fc873f2d58cffbdcd3233added609f3497811217614b2cca381d26"
    )
    assert evidence["price"]["input_price_dimension"] == {
        "sku": "QAF6BJ2FZ92WV4RM",
        "usage_type": "EU-MP:EU_InputTokenCount_Global-Units",
        "price_per_million_usd": "1",
    }
    assert evidence["price"]["output_price_dimension"] == {
        "sku": "G47BMTZZRZZK23Q2",
        "usage_type": "EU-MP:EU_OutputTokenCount_Global-Units",
        "price_per_million_usd": "5",
    }


@pytest.mark.asyncio
async def test_mismatched_frozen_tool_fixture_fails_before_any_external_client(
    tmp_path: Path,
) -> None:
    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    fixture = Path(args.epicure_tool_catalog)
    fixture.write_text(
        json.dumps([{**TOOL_CATALOG[0], "description": "changed"}], sort_keys=True),
        encoding="utf-8",
    )
    runtime = FakeRuntime()
    attestor_called = False

    async def unexpected_attestor() -> dict[str, Any]:
        nonlocal attestor_called
        attestor_called = True
        return fake_attestation()

    with pytest.raises(BedrockSmokeError, match="differs from the runtime contract"):
        await execute_contract_smoke(
            args,
            runtime_client=runtime,
            mcp_factory=FakeMcp,
            attestor=unexpected_attestor,
            lane_settings=settings(),
        )

    assert attestor_called is False
    assert runtime.count_calls == []
    assert runtime.calls == []
    assert not Path(args.ledger).exists()


@pytest.mark.asyncio
async def test_count_tokens_over_bound_blocks_paid_converse_and_releases_reservation(
    tmp_path: Path,
) -> None:
    class OverBoundRuntime(FakeRuntime):
        def count_tokens(self, **kwargs: Any) -> dict[str, Any]:
            self.count_calls.append(kwargs)
            return {
                "ResponseMetadata": {
                    "RequestId": "count-over-bound",
                    "HTTPStatusCode": 200,
                },
                "inputTokens": 12_001,
            }

    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    runtime = OverBoundRuntime()

    with pytest.raises(BedrockSmokeError, match="immutable failure evidence"):
        await execute_contract_smoke(
            args,
            runtime_client=runtime,
            mcp_factory=FakeMcp,
            attestor=async_attestation,
            lane_settings=settings(),
        )

    assert len(runtime.count_calls) == 1
    assert runtime.calls == []
    ledger = BedrockSmokeLedger(args.ledger)
    entries = ledger.entries()
    assert [entry["event_type"] for entry in entries] == [
        "reservation_created",
        "count_tokens_request_started",
        "count_tokens_response_received",
        "arm_artifact_recorded",
        "reservation_released_pre_send",
    ]
    assert entries[2]["admitted_for_paid_converse"] is False
    assert ledger.exposure().governed_exposure_micros == 0
    artifact_path = next((tmp_path / "output/arms").glob("*-failure-*.json"))
    artifact = json.loads(artifact_path.read_bytes())
    assert artifact["status"] == "failed_pre_send"
    assert artifact["rate_card_estimated_cost_micros"] == 0
    assert artifact["cost_classification"] == "failure_pre_send_rate_card_zero"


@pytest.mark.asyncio
async def test_missing_converse_usage_is_held_as_uncertain_not_settled(
    tmp_path: Path,
) -> None:
    class MissingUsageRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.responses[0]["usage"].pop("inputTokens")

    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    runtime = MissingUsageRuntime()

    with pytest.raises(BedrockSmokeError, match="immutable failure evidence"):
        await execute_contract_smoke(
            args,
            runtime_client=runtime,
            mcp_factory=FakeMcp,
            attestor=async_attestation,
            lane_settings=settings(),
        )

    assert len(runtime.count_calls) == 1
    assert len(runtime.calls) == 1
    ledger = BedrockSmokeLedger(args.ledger)
    assert ledger.exposure().uncertain_held_reservations_micros == 22_240
    assert ledger.exposure().settled_rate_card_estimate_micros == 0
    assert ledger.entries()[-1]["event_type"] == "reservation_held_uncertain"


@pytest.mark.asyncio
async def test_delivered_invalid_response_settles_rate_card_and_blocks_replay(
    tmp_path: Path,
) -> None:
    class MaxTokensRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.responses[0]["stopReason"] = "max_tokens"

    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    runtime = MaxTokensRuntime()

    with pytest.raises(BedrockSmokeError, match="immutable failure evidence"):
        await execute_contract_smoke(
            args,
            runtime_client=runtime,
            mcp_factory=FakeMcp,
            attestor=async_attestation,
            lane_settings=settings(),
        )

    ledger = BedrockSmokeLedger(args.ledger)
    assert ledger.entries()[-1]["event_type"] == (
        "reservation_settled_rate_card_estimate"
    )
    assert ledger.exposure().uncertain_held_reservations_micros == 0
    assert ledger.exposure().settled_rate_card_estimate_micros == 350
    artifact_path = next((tmp_path / "output/arms").glob("*-failure-*.json"))
    artifact = json.loads(artifact_path.read_bytes())
    assert artifact["status"] == "failed_delivered_invalid_response"
    assert artifact["rate_card_estimated_cost_micros"] == 350
    assert artifact["billing_actual_cost_micros"] is None
    assert artifact["delivered_response_evidence"][0]["usage_complete"] is True

    with pytest.raises(BedrockSmokeError, match="paid replay is blocked"):
        await execute_contract_smoke(
            args,
            runtime_client=runtime,
            mcp_factory=FakeMcp,
            attestor=async_attestation,
            lane_settings=settings(),
        )

    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_explicit_provider_route_rejection_releases_but_blocks_same_run_replay(
    tmp_path: Path,
) -> None:
    botocore_exceptions = pytest.importorskip("botocore.exceptions")

    class RouteRejectedRuntime(FakeRuntime):
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(copy.deepcopy(kwargs))
            raise botocore_exceptions.ClientError(
                {
                    "Error": {
                        "Code": "ResourceNotFoundException",
                        "Message": "Model access prerequisites are not complete.",
                    },
                    "ResponseMetadata": {
                        "RequestId": "route-rejected",
                        "HTTPStatusCode": 404,
                    },
                },
                "Converse",
            )

    epicure_path = epicure_contract(tmp_path)
    args = arguments(tmp_path, epicure_path)
    runtime = RouteRejectedRuntime()

    with pytest.raises(BedrockSmokeError, match="immutable failure evidence"):
        await execute_contract_smoke(
            args,
            runtime_client=runtime,
            mcp_factory=FakeMcp,
            attestor=async_attestation,
            lane_settings=settings(),
        )

    ledger = BedrockSmokeLedger(args.ledger)
    assert ledger.entries()[-1]["event_type"] == (
        "reservation_released_service_rejection"
    )
    assert ledger.exposure().governed_exposure_micros == 0
    artifact_path = next((tmp_path / "output/arms").glob("*-failure-*.json"))
    artifact = json.loads(artifact_path.read_bytes())
    assert artifact["status"] == "failed_pre_generation_rejection"
    assert artifact["rate_card_estimated_cost_micros"] == 0
    assert artifact["billing_actual_cost_micros"] is None
    assert artifact["billing_actual_reconciliation_status"] == "not_reconciled"
    assert artifact["provider_error"]["aws_error_code"] == (
        "ResourceNotFoundException"
    )

    with pytest.raises(BedrockSmokeError, match="paid replay is blocked"):
        await execute_contract_smoke(
            args,
            runtime_client=runtime,
            mcp_factory=FakeMcp,
            attestor=async_attestation,
            lane_settings=settings(),
        )

    assert len(runtime.calls) == 1


def test_ledger_payload_cannot_override_protected_envelope_fields(tmp_path: Path) -> None:
    ledger = BedrockSmokeLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(BedrockSmokeLedgerError, match="protected fields"):
        ledger.reserve(
            settings=settings(),
            run_key="a" * 64,
            arm_id="b" * 64,
            reservation_id="c" * 64,
            reservation_micros=100_000,
            payload={"event_type": "not_a_reservation", "reservation_micros": 1},
        )
    assert ledger.entries() == []

    ledger.reserve(
        settings=settings(),
        run_key="a" * 64,
        arm_id="b" * 64,
        reservation_id="c" * 64,
        reservation_micros=100_000,
        payload={"fixture": "valid"},
    )
    with pytest.raises(BedrockSmokeLedgerError, match="protected fields"):
        ledger.append(
            "converse_request_started",
            run_key="a" * 64,
            arm_id="b" * 64,
            reservation_id="c" * 64,
            reservation_micros=100_000,
            payload={"schema_version": "forged"},
        )
    assert ledger.exposure().governed_exposure_micros == 100_000


@pytest.mark.parametrize("field", sorted(PROTECTED_ENTRY_FIELDS))
def test_every_ledger_envelope_field_is_protected(tmp_path: Path, field: str) -> None:
    ledger = BedrockSmokeLedger(tmp_path / "ledger.jsonl")

    with pytest.raises(BedrockSmokeLedgerError, match="protected fields"):
        ledger.reserve(
            settings=settings(),
            run_key="a" * 64,
            arm_id="b" * 64,
            reservation_id="c" * 64,
            reservation_micros=100_000,
            payload={field: "forged"},
        )

    assert ledger.entries() == []


def test_ledger_descriptor_is_one_locked_snapshot(tmp_path: Path) -> None:
    ledger = BedrockSmokeLedger(tmp_path / "ledger.jsonl")
    ledger.reserve(
        settings=settings(),
        run_key="a" * 64,
        arm_id="b" * 64,
        reservation_id="c" * 64,
        reservation_micros=100_000,
        payload={"fixture": "descriptor"},
    )

    descriptor = ledger.descriptor()
    entries = ledger.entries()

    assert descriptor["entry_count"] == 1
    assert descriptor["head_entry_sha256"] == entries[0]["entry_sha256"]
    assert descriptor["file_sha256"] == hashlib.sha256(ledger.path.read_bytes()).hexdigest()


def test_installed_sdk_accepts_count_and_converse_shapes_with_real_projected_tools() -> None:
    boto3 = pytest.importorskip("boto3")
    botocore_config = pytest.importorskip("botocore.config")
    botocore_stub = pytest.importorskip("botocore.stub")
    frozen = _load_epicure_tool_catalog(
        REAL_TOOL_FIXTURE,
        expected_raw_sha256=REAL_TOOL_DIGEST,
    )
    tool_config = {
        "tools": [
            definition.as_converse_tool()
            for definition in _tools_from_catalog(list(frozen.bedrock_tools))
        ],
        "toolChoice": {"auto": {}},
    }
    request = {
        "modelId": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "input": {
            "converse": {
                "messages": [
                    {"role": "user", "content": [{"text": "offline shape test"}]}
                ],
                "system": [{"text": "offline shape test"}],
                "toolConfig": tool_config,
            }
        },
    }
    converse_request = {
        "modelId": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "messages": [{"role": "user", "content": [{"text": "offline shape test"}]}],
        "system": [{"text": "offline shape test"}],
        "inferenceConfig": {"maxTokens": 2048, "temperature": 0.2},
        "outputConfig": structured_output_config(),
        "toolConfig": tool_config,
        "requestMetadata": {"flavourbench_protocol": "offline_shape_test"},
    }
    client = boto3.client(
        "bedrock-runtime",
        region_name="eu-west-1",
        aws_access_key_id="offline",
        aws_secret_access_key="offline",
        config=botocore_config.Config(retries={"max_attempts": 0}),
    )
    with botocore_stub.Stubber(client) as stubber:
        stubber.add_response("count_tokens", {"inputTokens": 900}, request)
        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "offline"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 900, "outputTokens": 1, "totalTokens": 901},
                "metrics": {"latencyMs": 1},
            },
            converse_request,
        )
        assert client.count_tokens(**request)["inputTokens"] == 900
        assert client.converse(**converse_request)["stopReason"] == "end_turn"


def test_aws_error_diagnostic_is_bounded_redacted_and_allowlisted() -> None:
    botocore_exceptions = pytest.importorskip("botocore.exceptions")
    raw_message = (
        f"Invalid model account {FAKE_ACCOUNT_ID}; temperature and top_p cannot both be set. "
        "Bearer secret-value-that-must-not-survive"
    )
    error = botocore_exceptions.ClientError(
        {
            "Error": {"Code": "ValidationException", "Message": raw_message},
            "ResponseMetadata": {
                "RequestId": "request-validation",
                "HTTPStatusCode": 400,
            },
        },
        "Converse",
    )

    details = _safe_aws_error_details(error)

    assert details["aws_error_code"] == "ValidationException"
    assert details["aws_http_status"] == 400
    assert details["aws_request_id"] == "request-validation"
    assert details["aws_error_message_sha256"] == hashlib.sha256(
        raw_message.encode("utf-8")
    ).hexdigest()
    assert details["aws_error_message_sanitized"] == "<diagnostic-redacted>"
    assert details["aws_error_sanitizer_version"] == "aws-error-redaction-v1"
    assert FAKE_ACCOUNT_ID not in json.dumps(details)

    useful_message = "temperature and top_p cannot both be set for Claude Haiku 4.5"
    useful_error = botocore_exceptions.ClientError(
        {
            "Error": {"Code": "ValidationException", "Message": useful_message},
            "ResponseMetadata": {"RequestId": "request-useful", "HTTPStatusCode": 400},
        },
        "Converse",
    )
    useful = _safe_aws_error_details(useful_error)
    assert useful["aws_error_message_sanitized"] == useful_message
    assert useful["aws_error_message_sha256"] == hashlib.sha256(
        useful_message.encode("utf-8")
    ).hexdigest()


def test_anthropic_access_preflight_discards_form_and_blocks_missing_prerequisite() -> None:
    botocore_exceptions = pytest.importorskip("botocore.exceptions")
    sensitive_form = "private-company-and-use-case-details"

    class ReadyControl:
        def get_use_case_for_model_access(self) -> dict[str, Any]:
            return {"formData": sensitive_form}

    assert _assert_anthropic_use_case_ready(ReadyControl()) is None

    class MissingControl:
        def get_use_case_for_model_access(self) -> dict[str, Any]:
            raise botocore_exceptions.ClientError(
                {
                    "Error": {
                        "Code": "ResourceNotFoundException",
                        "Message": (
                            "Model use case details have not been submitted for this account."
                        ),
                    },
                    "ResponseMetadata": {
                        "RequestId": "access-preflight",
                        "HTTPStatusCode": 404,
                    },
                },
                "GetUseCaseForModelAccess",
            )

    with pytest.raises(BedrockSmokeError, match="no inference reservation") as caught:
        _assert_anthropic_use_case_ready(MissingControl())

    assert sensitive_form not in str(caught.value)

    class MalformedControl:
        def get_use_case_for_model_access(self) -> dict[str, Any]:
            return {"formData": "short"}

    with pytest.raises(BedrockSmokeError, match="did not confirm"):
        _assert_anthropic_use_case_ready(MalformedControl())
