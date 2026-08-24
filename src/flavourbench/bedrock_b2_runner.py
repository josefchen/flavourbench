"""Fail-closed execution for a fully frozen, real Bedrock + Epicure B2 block.

The public execution boundary constructs the AWS and MCP clients itself.  Test
dependencies are accepted only by a private helper, so the CLI cannot inject an
adapter whose outputs could be mistaken for provider evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from .bedrock_auth import (
    BedrockConfigurationError,
    BedrockLaneSettings,
    create_boto3_clients,
)
from .bedrock_b2_manifest import (
    ALLOWED_EPICURE_TOOLS,
    EXECUTION_ADAPTER_ID,
    FINAL_RESPONSE_SCHEMA,
    OFF_SYSTEM_PROMPT,
    ON_SYSTEM_PROMPT,
    SCHEMA_VERSION,
    verify_b2_manifest_content_address,
)
from .bedrock_contract import LoadedSmokeContract, load_smoke_contract
from .bedrock_contract_smoke import (
    EpicureExecutor,
    LedgerRuntimeClient,
    _assert_anthropic_use_case_ready,
    _delivered_rate_card_cost_micros,
    _load_epicure_contract,
    _load_epicure_tool_catalog,
    _result_payload,
    _tools_from_catalog,
    attest_epicure_provenance_document,
)
from .bedrock_manifest import assert_public_catalog_safe
from .bedrock_provider import (
    BEDROCK_FINAL_SCHEMA,
    BedrockConverseProvider,
    BedrockGenerationSpec,
    BedrockInferenceConfig,
    project_bedrock_json_schema,
)
from .bedrock_smoke_ledger import BedrockSmokeLedger, BedrockSmokeLedgerError
from .execution_policy import assert_legacy_paid_cli_allowed
from .mcp_client import McpSession, tool_catalog_sha256

PROTOCOL = "flavourbench_bedrock_b2_common_task_v2"
ARTIFACT_SCHEMA = "flavourbench-bedrock-b2-arm-v2"
SUMMARY_SCHEMA = "flavourbench-bedrock-b2-summary-v2"
EXECUTION_CONFIRMATION = "RUN_FROZEN_BEDROCK_B2_REAL_INFERENCE_V2"


class BedrockB2RunnerError(RuntimeError):
    """The B2 execution boundary is missing, mutable, or ambiguous."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_file(directory: Path, filename: object, expected_sha256: object) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise BedrockB2RunnerError("frozen contract filename is unsafe")
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise BedrockB2RunnerError(f"missing frozen contract: {filename}")
    if not isinstance(expected_sha256, str) or _file_sha(path) != expected_sha256:
        raise BedrockB2RunnerError(f"frozen contract file digest differs: {filename}")
    return path


def _validate_execution_contract(manifest: Mapping[str, Any]) -> None:
    value = manifest.get("execution_contract")
    if not isinstance(value, Mapping):
        raise BedrockB2RunnerError("B2 manifest has no frozen execution contract")
    expected = {
        "adapter_id": EXECUTION_ADAPTER_ID,
        "off_system_prompt": OFF_SYSTEM_PROMPT,
        "off_system_prompt_sha256": hashlib.sha256(OFF_SYSTEM_PROMPT.encode()).hexdigest(),
        "on_system_prompt": ON_SYSTEM_PROMPT,
        "on_system_prompt_sha256": hashlib.sha256(ON_SYSTEM_PROMPT.encode()).hexdigest(),
        "decoding": {"temperature": "0.2", "top_p": None},
        "response_schema": FINAL_RESPONSE_SCHEMA,
        "response_schema_sha256": _sha(FINAL_RESPONSE_SCHEMA),
        "allowed_epicure_tools": list(ALLOWED_EPICURE_TOOLS),
        "tool_projection": "aws-draft-2020-12-supported-subset-v1",
        "max_tool_rounds": 8,
        "max_tool_calls_per_round": 4,
        "max_tool_calls_total": 16,
        "max_tool_result_bytes": 65_536,
        "max_cumulative_tool_result_bytes": 131_072,
        "retry_policy": "no-automatic-retry-after-paid-request-boundary",
        "delivery_policy": "fail-closed-hold-whole-block-on-ambiguous-delivery",
    }
    if dict(value) != expected or FINAL_RESPONSE_SCHEMA != BEDROCK_FINAL_SCHEMA:
        raise BedrockB2RunnerError("B2 execution contract differs from the reviewed adapter")


def load_frozen_manifest(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BedrockB2RunnerError("B2 manifest must be a regular file")
    try:
        value = json.loads(manifest_path.read_bytes())
    except json.JSONDecodeError as error:
        raise BedrockB2RunnerError("B2 manifest is invalid JSON") from error
    if not isinstance(value, dict) or not verify_b2_manifest_content_address(value):
        raise BedrockB2RunnerError("B2 manifest content address is invalid")
    digest = value["content_address"]["digest"]
    if digest != expected_sha256 or manifest_path.name != f"bedrock-b2-workload-{digest}.json":
        raise BedrockB2RunnerError("B2 manifest identity does not match the admission")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise BedrockB2RunnerError("legacy B2 manifests cannot cross the paid boundary")
    if value.get("official") is not False or value.get("rank_eligible") is not False:
        raise BedrockB2RunnerError("this runner accepts only the governed B2 compatibility block")
    if value.get("provider_calls_made") != 0:
        raise BedrockB2RunnerError("a frozen B2 manifest cannot contain prior provider calls")
    if value.get("stage") != "bedrock_b2_compatibility_pilot":
        raise BedrockB2RunnerError("refusing a non-B2 manifest")
    _validate_execution_contract(value)
    return value


def counterbalanced_arms(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms = manifest.get("arms")
    if not isinstance(arms, list):
        raise BedrockB2RunnerError("B2 manifest arms are invalid")
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for arm in arms:
        if not isinstance(arm, dict):
            raise BedrockB2RunnerError("B2 arm is invalid")
        key = (str(arm.get("canonical_model_id")), str(arm.get("task_id")))
        condition = str(arm.get("condition"))
        if condition in grouped.setdefault(key, {}):
            raise BedrockB2RunnerError("duplicate B2 arm condition")
        grouped[key][condition] = arm
    ordered: list[dict[str, Any]] = []
    for key in sorted(grouped):
        pair = grouped[key]
        if set(pair) != {"epicure_off", "epicure_on"}:
            raise BedrockB2RunnerError("every B2 model/task pair requires off and on arms")
        first = (
            "epicure_on"
            if int(_sha({"model": key[0], "task": key[1]})[0], 16) % 2
            else "epicure_off"
        )
        second = "epicure_off" if first == "epicure_on" else "epicure_on"
        ordered.extend((pair[first], pair[second]))
    return ordered


def _write_artifact(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    assert_public_catalog_safe(payload, path="$b2_artifact")
    digest = _sha(payload)
    document = {**payload, "artifact_sha256": digest}
    destination = directory / f"{prefix}-{digest}.json"
    rendered = _canonical(document) + b"\n"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    if destination.exists():
        if destination.read_bytes() != rendered:
            temporary.unlink()
            raise BedrockB2RunnerError("B2 artifact content-address conflict")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _load_model_contracts(
    manifest: Mapping[str, Any],
    *,
    endpoint_directory: Path,
    catalog_directory: Path,
    evidence_directory: Path,
) -> dict[str, LoadedSmokeContract]:
    loaded: dict[str, LoadedSmokeContract] = {}
    for model in manifest.get("models", []):
        if not isinstance(model, Mapping):
            raise BedrockB2RunnerError("B2 model contract is invalid")
        reference = model.get("endpoint_contract_reference")
        if not isinstance(reference, Mapping):
            raise BedrockB2RunnerError("B2 endpoint reference is invalid")
        endpoint = _regular_file(
            endpoint_directory, reference.get("filename"), reference.get("file_sha256")
        )
        try:
            wrapper = json.loads(endpoint.read_bytes())
        except json.JSONDecodeError as error:
            raise BedrockB2RunnerError("B2 endpoint wrapper is invalid JSON") from error
        if not isinstance(wrapper, Mapping):
            raise BedrockB2RunnerError("B2 endpoint wrapper is not an object")
        catalog = catalog_directory / str(wrapper.get("catalog_filename") or "")
        evidence = evidence_directory / str(wrapper.get("capability_price_evidence_filename") or "")
        contract = load_smoke_contract(
            manifest_path=endpoint,
            catalog_path=catalog,
            evidence_path=evidence,
            expected_manifest_sha256=str(wrapper.get("manifest_sha256") or ""),
        )
        canonical_id = str(model.get("canonical_model_id") or "")
        if (
            contract.contract.canonical_model_id != canonical_id
            or contract.contract.bedrock_target_id != model.get("bedrock_target_id")
            or list(contract.contract.expected_foundation_model_ids)
            != model.get("expected_foundation_model_ids")
            or contract.contract.sha256 != reference.get("endpoint_contract_sha256")
            or len(contract.contract.expected_foundation_model_ids) != 1
        ):
            raise BedrockB2RunnerError("B2 model identity differs from its frozen endpoint")
        loaded[canonical_id] = contract
    if len(loaded) != manifest.get("counts", {}).get("models"):
        raise BedrockB2RunnerError("B2 loaded model count differs from the manifest")
    return loaded


def _project_tools(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(tool.get("name") or ""): tool for tool in raw_tools}
    if any(name not in by_name for name in ALLOWED_EPICURE_TOOLS):
        raise BedrockB2RunnerError("frozen Epicure catalog lacks an allowed B2 tool")
    projected: list[dict[str, Any]] = []
    for name in ALLOWED_EPICURE_TOOLS:
        tool = copy.deepcopy(by_name[name])
        schema = tool.get("inputSchema")
        if not isinstance(schema, Mapping):
            raise BedrockB2RunnerError("Epicure tool has no input schema")
        tool["inputSchema"] = project_bedrock_json_schema(schema)
        projected.append(tool)
    for definition in _tools_from_catalog(projected):
        definition.as_converse_tool()
    return projected


def _assert_epicure_attestation(configured: Any, attested: Mapping[str, Any]) -> None:
    for field in (
        "release_id",
        "bundle_sha256",
        "application_sha256",
        "ingredient_count",
        "embedding_dimensions",
    ):
        if attested.get(field) != getattr(configured, field):
            raise BedrockB2RunnerError("live Epicure identity differs from the frozen contract")


def _micros(value: object) -> int:
    amount = Decimal(str(value)) * Decimal(1_000_000)
    return int(amount.quantize(Decimal("1"), rounding=ROUND_CEILING))


async def _execute_with_clients(
    *,
    manifest: dict[str, Any],
    expected_manifest_sha256: str,
    output_directory: Path,
    ledger_path: Path,
    settings: BedrockLaneSettings,
    runtime: Any,
    mcp_factory: Any,
    attestor: Any,
    endpoint_directory: Path,
    catalog_directory: Path,
    evidence_directory: Path,
    epicure_contract_directory: Path,
    tool_contract_directory: Path,
) -> Path:
    models = _load_model_contracts(
        manifest,
        endpoint_directory=endpoint_directory,
        catalog_directory=catalog_directory,
        evidence_directory=evidence_directory,
    )
    if any(
        loaded.contract.region != settings.region
        or loaded.contract.profile_scope != settings.profile_scope
        for loaded in models.values()
    ):
        raise BedrockB2RunnerError("configured Bedrock route differs from the frozen endpoints")
    contracts = manifest["contracts"]
    epicure_ref = contracts["epicure_lineage"]
    tool_ref = contracts["epicure_tool_catalog"]
    epicure_path = _regular_file(
        epicure_contract_directory, epicure_ref["filename"], epicure_ref["file_sha256"]
    )
    tool_path = _regular_file(
        tool_contract_directory, tool_ref["filename"], tool_ref["file_sha256"]
    )
    configured_epicure = _load_epicure_contract(epicure_path)
    frozen_tools = _load_epicure_tool_catalog(
        tool_path, expected_raw_sha256=configured_epicure.tool_schema_sha256
    )
    try:
        attested = await attestor()
    except Exception as error:
        raise BedrockB2RunnerError(
            f"Epicure provenance preflight failed before reservation: {type(error).__name__}"
        ) from error
    _assert_epicure_attestation(configured_epicure, attested)

    run_key = _sha({"protocol": PROTOCOL, "manifest_sha256": expected_manifest_sha256})
    block_id = _sha({"run_key": run_key, "scope": "whole_manifest"})
    reservation_id = _sha({"run_key": run_key, "reservation": "whole_manifest_v1"})
    reservation_micros = _micros(manifest["budget"]["whole_block_worst_case_usd"])
    ledger = BedrockSmokeLedger(ledger_path)
    if any(entry.get("run_key") == run_key for entry in ledger.entries()):
        raise BedrockB2RunnerError("B2 run already crossed admission; paid replay is forbidden")

    artifacts: list[dict[str, Any]] = []
    cost_micros = 0
    arm_directory = output_directory / "arms"
    async with mcp_factory() as mcp:
        raw_tools = await mcp.list_tools()
        if (
            raw_tools != list(frozen_tools.raw_tools)
            or tool_catalog_sha256(raw_tools) != configured_epicure.tool_schema_sha256
        ):
            raise BedrockB2RunnerError("live Epicure tool catalog differs from the frozen fixture")
        projected_tools = _project_tools(raw_tools)
        attested = {
            **dict(attested),
            "tool_schema_sha256": tool_catalog_sha256(raw_tools),
            "tool_count": len(raw_tools),
            "bedrock_tool_schema_sha256": tool_catalog_sha256(projected_tools),
        }
        assert_public_catalog_safe(attested, path="$b2_epicure_attestation")
        ledger.reserve(
            settings=settings,
            run_key=run_key,
            arm_id=block_id,
            reservation_id=reservation_id,
            reservation_micros=reservation_micros,
            payload={
                "protocol": PROTOCOL,
                "manifest_sha256": expected_manifest_sha256,
                "reservation_scope": "entire_manifest_before_first_provider_call",
            },
        )
        prompts = {str(task["task_id"]): str(task["prompt"]) for task in manifest["tasks"]}
        policy = manifest["forecast_policy"]
        execution = manifest["execution_contract"]
        for sequence, arm in enumerate(counterbalanced_arms(manifest), 1):
            arm_id = str(arm["arm_id"])
            condition = str(arm["condition"])
            loaded = models[str(arm["canonical_model_id"])]
            prompt = prompts[str(arm["task_id"])]
            journaled = LedgerRuntimeClient(
                runtime,
                ledger,
                run_key=run_key,
                arm_id=arm_id,
                reservation_id=reservation_id,
                reservation_micros=reservation_micros,
                expected_target_id=loaded.contract.bedrock_target_id,
                count_tokens_model_id=loaded.contract.expected_foundation_model_ids[0],
                max_input_tokens_per_call=int(policy["max_input_tokens_per_generation"]),
                max_converse_calls=(
                    int(policy["max_on_generations"])
                    if condition == "epicure_on"
                    else int(policy["max_off_generations"])
                ),
            )
            definitions = _tools_from_catalog(projected_tools) if condition == "epicure_on" else ()
            executor = EpicureExecutor(
                mcp=mcp,
                allowed_tools=frozenset(ALLOWED_EPICURE_TOOLS),
                max_result_bytes=int(execution["max_tool_result_bytes"]),
                max_cumulative_bytes=int(execution["max_cumulative_tool_result_bytes"]),
            )
            provider = BedrockConverseProvider(
                journaled,
                loaded.contract,
                tool_executor=executor if condition == "epicure_on" else None,
                max_tool_rounds=int(execution["max_tool_rounds"]),
                max_tool_calls_per_round=int(execution["max_tool_calls_per_round"]),
                max_tool_calls_total=int(execution["max_tool_calls_total"]),
            )
            base = {
                "schema_version": ARTIFACT_SCHEMA,
                "protocol": PROTOCOL,
                "run_key": run_key,
                "manifest_sha256": expected_manifest_sha256,
                "sequence": sequence,
                "arm": arm,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "execution_contract_sha256": _sha(execution),
                "endpoint_contract_sha256": loaded.contract.sha256,
                "attested_epicure_identity": attested,
                "official": False,
                "rank_eligible": False,
            }
            try:
                result = await provider.generate(
                    BedrockGenerationSpec(
                        arm_id=arm_id,
                        canonical_model_id=loaded.contract.canonical_model_id,
                        prompt=prompt,
                        system_prompt=ON_SYSTEM_PROMPT
                        if condition == "epicure_on"
                        else OFF_SYSTEM_PROMPT,
                        inference=BedrockInferenceConfig(
                            max_tokens=int(policy["max_output_tokens_per_generation"]),
                            temperature=0.2,
                        ),
                        tools=definitions,
                        request_metadata={
                            "flavourbench_protocol": PROTOCOL,
                            "flavourbench_condition": condition,
                        },
                    )
                )
                if (
                    result.identity.canonical_model_id != arm["canonical_model_id"]
                    or result.identity.requested_model_or_profile_id
                    != loaded.contract.bedrock_target_id
                    or result.identity.expected_foundation_model_ids
                    != loaded.contract.expected_foundation_model_ids
                    or result.provider_substitution
                    or result.identity.provider_substitution
                    or (condition == "epicure_on" and not result.tool_traces)
                    or (condition == "epicure_off" and result.tool_traces)
                    or not result.cost.estimate_complete
                    or result.cost.estimated_cost_micros is None
                ):
                    raise BedrockB2RunnerError(
                        "delivered arm violates identity, tool, or cost contract"
                    )
                cost_micros += result.cost.estimated_cost_micros
                payload = {
                    **base,
                    "status": "complete",
                    "generation": _result_payload(result),
                    "complete_epicure_mcp_trace": executor.traces
                    if condition == "epicure_on"
                    else [],
                    "count_tokens_preflight": {
                        "model_id": journaled.count_tokens_model_id,
                        "input_tokens": list(journaled.counted_input_tokens),
                        "free_api_attempts": journaled.count_tokens_attempt_count,
                    },
                }
            except Exception as error:
                delivered = _delivered_rate_card_cost_micros(loaded, journaled.response_evidence)
                if delivered is not None:
                    cost_micros += delivered
                payload = {
                    **base,
                    "status": "failed",
                    "failure": {
                        "class": type(error).__name__,
                        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                        "converse_calls_started": journaled.call_count,
                        "delivered_response_evidence": journaled.response_evidence,
                        "delivery_ambiguous": bool(journaled.call_count and delivered is None),
                    },
                    "complete_epicure_mcp_trace": executor.traces
                    if condition == "epicure_on"
                    else [],
                }
                path = _write_artifact(arm_directory, "bedrock-b2-arm-failure", payload)
                ledger.append(
                    "arm_artifact_recorded",
                    run_key=run_key,
                    arm_id=arm_id,
                    reservation_id=reservation_id,
                    reservation_micros=reservation_micros,
                    payload={
                        "artifact_filename": path.name,
                        "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                    },
                )
                terminal = (
                    "reservation_held_uncertain"
                    if journaled.call_count and delivered is None
                    else "reservation_settled_rate_card_estimate"
                )
                ledger.append(
                    terminal,
                    run_key=run_key,
                    arm_id=block_id,
                    reservation_id=reservation_id,
                    reservation_micros=reservation_micros,
                    payload={
                        "rate_card_estimated_cost_micros": cost_micros,
                        "failed_arm_id": arm_id,
                        "billing_actual_reconciliation_status": "uncertain_not_reconciled"
                        if terminal == "reservation_held_uncertain"
                        else "not_reconciled",
                    },
                )
                raise BedrockB2RunnerError(f"B2 stopped fail-closed after arm {arm_id}") from error
            path = _write_artifact(arm_directory, "bedrock-b2-arm", payload)
            record = {
                "arm_id": arm_id,
                "status": "complete",
                "artifact_filename": path.name,
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
            }
            artifacts.append(record)
            ledger.append(
                "arm_artifact_recorded",
                run_key=run_key,
                arm_id=arm_id,
                reservation_id=reservation_id,
                reservation_micros=reservation_micros,
                payload={key: value for key, value in record.items() if key != "arm_id"},
            )
    if cost_micros > reservation_micros:
        raise BedrockB2RunnerError("delivered rate-card estimate exceeded whole-block reservation")
    ledger.append(
        "reservation_settled_rate_card_estimate",
        run_key=run_key,
        arm_id=block_id,
        reservation_id=reservation_id,
        reservation_micros=reservation_micros,
        payload={
            "rate_card_estimated_cost_micros": cost_micros,
            "billing_actual_reconciliation_status": "not_reconciled",
            "completed_arms": len(artifacts),
        },
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "run_key": run_key,
        "manifest_sha256": expected_manifest_sha256,
        "official": False,
        "rank_eligible": False,
        "artifacts": artifacts,
        "counts": {"complete": len(artifacts), "failed": 0, "total": len(artifacts)},
        "rate_card_estimated_cost_micros": cost_micros,
        "billing_actual_reconciliation_status": "not_reconciled",
        "ledger": ledger.descriptor(),
    }
    return _write_artifact(output_directory / "summaries", "bedrock-b2-summary", summary)


async def execute_b2(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    output_directory: str | Path,
    ledger_path: str | Path,
    endpoint_directory: str | Path,
    catalog_directory: str | Path,
    evidence_directory: str | Path,
    epicure_contract_directory: str | Path,
    tool_contract_directory: str | Path,
    confirmation: str,
) -> Path:
    """Execute with only built-in AWS and Epicure adapters."""

    if confirmation != EXECUTION_CONFIRMATION:
        raise BedrockB2RunnerError("real B2 inference requires the exact execution confirmation")
    manifest = load_frozen_manifest(manifest_path, expected_manifest_sha256)
    settings = BedrockLaneSettings.from_environ()
    if not settings.enabled or not settings.live_authorized or settings.stage != "exploratory":
        raise BedrockB2RunnerError("B2 requires an explicitly authorized exploratory Bedrock lane")
    clients = create_boto3_clients(settings)
    _assert_anthropic_use_case_ready(clients.control)
    return await _execute_with_clients(
        manifest=manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        output_directory=Path(output_directory),
        ledger_path=Path(ledger_path),
        settings=settings,
        runtime=clients.runtime,
        mcp_factory=McpSession,
        attestor=attest_epicure_provenance_document,
        endpoint_directory=Path(endpoint_directory),
        catalog_directory=Path(catalog_directory),
        evidence_directory=Path(evidence_directory),
        epicure_contract_directory=Path(epicure_contract_directory),
        tool_contract_directory=Path(tool_contract_directory),
    )


def run(argv: Sequence[str] | None = None) -> int:
    assert_legacy_paid_cli_allowed("flavourbench-run-bedrock-b2")
    parser = argparse.ArgumentParser(description="Plan or run a frozen real Bedrock B2 workload")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/bedrock/b2/runs"))
    parser.add_argument("--ledger", type=Path, default=Path("artifacts/bedrock/b2/ledger.jsonl"))
    parser.add_argument(
        "--endpoint-directory", type=Path, default=Path("artifacts/bedrock/contracts")
    )
    parser.add_argument("--catalog-directory", type=Path, default=Path("artifacts/bedrock/catalog"))
    parser.add_argument("--evidence-directory", type=Path, default=Path("contracts/evidence"))
    parser.add_argument(
        "--epicure-contract-directory", type=Path, default=Path("contracts/epicure")
    )
    parser.add_argument("--tool-contract-directory", type=Path, default=Path("contracts/epicure"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    arguments = parser.parse_args(argv)
    try:
        manifest = load_frozen_manifest(arguments.manifest, arguments.expected_manifest_sha256)
    except BedrockB2RunnerError as error:
        raise SystemExit(str(error)) from error
    if not arguments.execute:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "provider_calls": 0,
                    "manifest_sha256": arguments.expected_manifest_sha256,
                    "counts": manifest["counts"],
                    "whole_block_worst_case_usd": manifest["budget"]["whole_block_worst_case_usd"],
                    "execution_adapter": EXECUTION_ADAPTER_ID,
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        path = asyncio.run(
            execute_b2(
                manifest_path=arguments.manifest,
                expected_manifest_sha256=arguments.expected_manifest_sha256,
                output_directory=arguments.output_directory,
                ledger_path=arguments.ledger,
                endpoint_directory=arguments.endpoint_directory,
                catalog_directory=arguments.catalog_directory,
                evidence_directory=arguments.evidence_directory,
                epicure_contract_directory=arguments.epicure_contract_directory,
                tool_contract_directory=arguments.tool_contract_directory,
                confirmation=arguments.confirm,
            )
        )
    except (BedrockB2RunnerError, BedrockConfigurationError, BedrockSmokeLedgerError) as error:
        raise SystemExit(str(error)) from error
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
