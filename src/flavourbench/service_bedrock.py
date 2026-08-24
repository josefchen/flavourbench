"""Service adapter for governed Bedrock Converse execution.

The adapter translates the worker's frozen arm contract into the isolated
Bedrock implementation. It journals every paid request boundary, keeps AWS
request identifiers hashed, and executes Epicure MCP tools on the private
network. OpenRouter substitution is deliberately outside this adapter: a
Bedrock arm either completes on its frozen route or fails unranked.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

import httpx

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .bedrock_contract import parse_bedrock_endpoint_contract
from .bedrock_manifest import BedrockEndpointContract
from .bedrock_provider import (
    BEDROCK_FINAL_SCHEMA,
    BedrockConverseProvider,
    BedrockGenerationSpec,
    BedrockInferenceConfig,
    BedrockProviderError,
    BedrockToolDefinition,
    BedrockToolExecution,
    _aws_error_code,
    _cost,
    _usage,
    project_bedrock_json_schema,
)
from .budget_policy import (
    provider_account_hard_cap_micros,
    provider_account_scope_sha256,
)
from .config import get_settings
from .mcp_client import McpSession
from .provider import (
    EPICURE_PROMPT,
    SYSTEM_PROMPT,
    AttemptSink,
    GenerationFailureResult,
    GenerationResult,
    GenerationSpec,
    ProviderAttemptEvent,
    ProviderError,
    ToolSink,
    ToolTrace,
    UncertainDeliveryError,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _arn_sha256(value: object) -> str:
    raw = str(value or "")
    if not raw.startswith("arn:"):
        raise ProviderError("Bedrock control plane returned an invalid ARN")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _foundation_model_id(value: object) -> str:
    raw = str(value or "")
    marker = "foundation-model/"
    return raw.split(marker, 1)[1] if marker in raw else raw


def _bedrock_control_plane_preflight(
    control: Any,
    contract: BedrockEndpointContract,
    spec: GenerationSpec,
) -> dict[str, Any]:
    """Resolve the frozen target through the same SDK credential before spend."""

    if contract.endpoint_kind == "inference_profile":
        operation = "get_inference_profile"
        response = control.get_inference_profile(
            inferenceProfileIdentifier=contract.bedrock_target_id
        )
        target_arn = response.get("inferenceProfileArn")
        model_rows = response.get("models")
        if not isinstance(model_rows, list):
            raise ProviderError("Bedrock inference profile returned no destination models")
        destination_arns = [
            row.get("modelArn")
            for row in model_rows
            if isinstance(row, Mapping) and row.get("modelArn")
        ]
        status = str(response.get("status") or "")
    elif contract.endpoint_kind == "foundation_model":
        operation = "get_foundation_model"
        response = control.get_foundation_model(modelIdentifier=contract.bedrock_target_id)
        details = response.get("modelDetails")
        if not isinstance(details, Mapping):
            raise ProviderError("Bedrock foundation-model identity response is malformed")
        target_arn = details.get("modelArn")
        destination_arns = [target_arn]
        lifecycle = details.get("modelLifecycle")
        status = str(lifecycle.get("status") or "") if isinstance(lifecycle, Mapping) else ""
    else:
        operation = "get_provisioned_model_throughput"
        response = control.get_provisioned_model_throughput(
            provisionedModelId=contract.bedrock_target_id
        )
        target_arn = response.get("provisionedModelArn")
        destination_arns = [
            response.get("foundationModelArn")
            or response.get("modelArn")
            or response.get("desiredModelArn")
        ]
        status = str(response.get("status") or "")
    metadata = response.get("ResponseMetadata")
    request_id = str(metadata.get("RequestId") or "") if isinstance(metadata, Mapping) else ""
    if not request_id:
        raise ProviderError("Bedrock control-plane receipt has no AWS request identity")
    target_arn_sha256 = _arn_sha256(target_arn)
    destination_arn_sha256s = sorted(_arn_sha256(value) for value in destination_arns if value)
    expected_destination_hashes = sorted(
        item.original_sha256 for item in contract.destination_model_arns
    )
    foundation_ids = sorted({_foundation_model_id(value) for value in destination_arns if value})
    if (
        target_arn_sha256 != contract.bedrock_target_arn.original_sha256
        or destination_arn_sha256s != expected_destination_hashes
        or foundation_ids != sorted(contract.expected_foundation_model_ids)
        or status.upper() not in {"ACTIVE", "LEGACY"}
        or spec.provider_credential_scope_sha256 != provider_account_scope_sha256("bedrock")
    ):
        raise ProviderError(
            "Bedrock control-plane identity differs from the frozen credential binding"
        )
    return {
        "schema_version": "flavourbench-bedrock-control-plane-receipt-v1",
        "operation": operation,
        "contract_sha256": contract.sha256,
        "client_region": contract.region,
        "target_arn_sha256": target_arn_sha256,
        "destination_model_arn_sha256s": destination_arn_sha256s,
        "foundation_model_ids": foundation_ids,
        "status": status.upper(),
        "aws_request_id_sha256": hashlib.sha256(request_id.encode()).hexdigest(),
        "credential_scope_sha256": spec.provider_credential_scope_sha256,
        "credential_binding_sha256": spec.provider_credential_binding_sha256,
    }


class _JournaledBedrockRuntime:
    """Synchronous boto3 proxy that durably journals each Converse request."""

    _safe_rejections = frozenset(
        {
            "AccessDeniedException",
            "ResourceNotFoundException",
            "ValidationException",
        }
    )

    def __init__(
        self,
        runtime: Any,
        contract: BedrockEndpointContract,
        spec: GenerationSpec,
        attempt_sink: AttemptSink | None,
    ) -> None:
        self.runtime = runtime
        self.contract = contract
        self.spec = spec
        self.attempt_sink = attempt_sink
        self.call_index = 0
        self.uncertain_delivery = False
        self.generation_ids: list[str] = []
        self.accounting: list[dict[str, Any]] = []

    def _emit(self, event: ProviderAttemptEvent) -> None:
        if self.attempt_sink is not None:
            self.attempt_sink(event)

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        call_index = self.call_index
        self.call_index += 1
        attempt_id = str(uuid.uuid4())
        request_key = hashlib.sha256(
            f"{self.spec.idempotency_key}:bedrock:{call_index}".encode()
        ).hexdigest()
        started = ProviderAttemptEvent(
            attempt_id=attempt_id,
            arm_id=self.spec.arm_id,
            request_key_sha256=request_key,
            phase=f"bedrock_converse_{call_index}",
            attempt_index=0,
            event_type="request_started",
            payload_sha256=_canonical_sha256(kwargs),
            metadata={
                "execution_backend": "bedrock",
                "target_sha256": hashlib.sha256(
                    self.contract.bedrock_target_id.encode()
                ).hexdigest(),
                "provider_account_authorization_envelope_sha256": (
                    self.spec.provider_account_authorization_envelope_sha256
                ),
                "provider_credential_binding_sha256": (
                    self.spec.provider_credential_binding_sha256
                ),
            },
        )
        self._emit(started)
        try:
            response = self.runtime.converse(**kwargs)
        except Exception as exc:
            code = _aws_error_code(exc)
            safe_rejection = code in self._safe_rejections
            self.uncertain_delivery = not safe_rejection
            self._emit(
                ProviderAttemptEvent(
                    **{
                        **started.__dict__,
                        "event_type": (
                            "request_rejected" if safe_rejection else "uncertain_delivery"
                        ),
                        "error_type": code or type(exc).__name__,
                    }
                )
            )
            raise
        if not isinstance(response, Mapping):
            self.uncertain_delivery = True
            raise BedrockProviderError("Bedrock returned a non-object response")

        response_metadata = response.get("ResponseMetadata")
        response_metadata = response_metadata if isinstance(response_metadata, Mapping) else {}
        request_id = str(response_metadata.get("RequestId") or "")
        request_id_sha256 = hashlib.sha256(request_id.encode()).hexdigest() if request_id else ""
        generation_id = (
            f"bedrock:{request_id_sha256}"
            if request_id_sha256
            else "bedrock-unverifiable:" + hashlib.sha256(attempt_id.encode()).hexdigest()
        )
        # Record provider acceptance before parsing any optional response field.
        # If usage or payload validation fails, failure reconciliation can still
        # retain the reservation and identify the accepted request.
        accounting: dict[str, Any] = {
            "generation_id": generation_id,
            "cost_micros": 0,
            "provider": "amazon-bedrock",
            "model": self.contract.canonical_model_id,
            "reconciled": False,
            "accounting_basis": "accepted_response_usage_unparsed",
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "provider_request_id_present": bool(request_id),
            "pricing_sha256": self.contract.price.sha256,
        }
        if request_id_sha256:
            accounting["aws_request_id_sha256"] = request_id_sha256
        self.generation_ids.append(generation_id)
        self.accounting.append(accounting)
        received = ProviderAttemptEvent(
            **{
                **started.__dict__,
                "event_type": "response_received",
                "generation_id": generation_id,
                "http_status": int(response_metadata.get("HTTPStatusCode") or 200),
                "payload_sha256": _canonical_sha256(response),
                "metadata": {
                    "execution_backend": "bedrock",
                    "provider_request_id_present": bool(request_id),
                    "identity_evidence": (
                        "frozen_catalog_target_plus_hashed_aws_request_id"
                        if request_id
                        else "accepted_response_without_provider_request_id"
                    ),
                    **({"aws_request_id_sha256": request_id_sha256} if request_id_sha256 else {}),
                },
            }
        )
        self._emit(received)
        usage = _usage(response)
        cost = _cost(usage, self.contract)
        accounting.update(
            {
                "generation_id": generation_id,
                "cost_micros": int(cost.estimated_cost_micros or 0),
                "provider": "amazon-bedrock",
                "model": self.contract.canonical_model_id,
                "reconciled": bool(cost.estimate_complete),
                "accounting_basis": "frozen_endpoint_rate_times_returned_usage",
                "tokens_prompt": usage.input_tokens,
                "tokens_completion": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "cache_write_input_tokens": usage.cache_write_input_tokens,
                "provider_request_id_present": bool(request_id),
                "pricing_sha256": cost.pricing_sha256,
                **({"aws_request_id_sha256": request_id_sha256} if request_id_sha256 else {}),
            }
        )
        self._emit(
            ProviderAttemptEvent(
                **{
                    **received.__dict__,
                    "event_type": "accounting_reconciled",
                    "metadata": accounting,
                }
            )
        )
        if not request_id:
            raise BedrockProviderError(
                "Bedrock accepted a response without an AWS request identifier"
            )
        return response

    def count_tokens(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.runtime.count_tokens(**kwargs)


class _EpicureBedrockExecutor:
    def __init__(
        self,
        *,
        mcp: McpSession,
        spec: GenerationSpec,
        attempt_sink: AttemptSink | None,
        tool_sink: ToolSink | None,
    ) -> None:
        self.mcp = mcp
        self.spec = spec
        self.attempt_sink = attempt_sink
        self.tool_sink = tool_sink
        self.settings = get_settings()
        self.round_index = 0
        self.call_index = 0
        self.tool_use_id = ""
        self.cumulative_result_bytes = 0
        self.error_repairs = 0
        self.traces: list[ToolTrace] = []

    def set_context(self, round_index: int, call_index: int, tool_use_id: str) -> None:
        self.round_index = round_index
        self.call_index = call_index
        self.tool_use_id = tool_use_id

    def _emit_attempt(self, event: ProviderAttemptEvent) -> None:
        if self.attempt_sink is not None:
            self.attempt_sink(event)

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> BedrockToolExecution:
        arguments_dict = dict(arguments)
        attempt_id = str(uuid.uuid4())
        request_key = hashlib.sha256(
            (
                f"{self.spec.idempotency_key}:mcp:{self.round_index}:{self.call_index}:{name}"
            ).encode()
        ).hexdigest()
        started = ProviderAttemptEvent(
            attempt_id=attempt_id,
            arm_id=self.spec.arm_id,
            request_key_sha256=request_key,
            phase=f"mcp_tool_{self.round_index}_{self.call_index}",
            attempt_index=0,
            event_type="mcp_call_started",
            payload_sha256=_canonical_sha256({"name": name, "arguments": arguments_dict}),
            metadata={"tool_name": name, "execution_backend": "bedrock"},
        )
        self._emit_attempt(started)
        call_started = time.monotonic()
        try:
            result = await self.mcp.call_tool(name, arguments_dict)
        except (RuntimeError, httpx.HTTPError) as exc:
            detail = f"Epicure MCP service error: {type(exc).__name__}"
            self._emit_attempt(
                ProviderAttemptEvent(
                    **{
                        **started.__dict__,
                        "event_type": "mcp_call_failed",
                        "error_type": type(exc).__name__,
                        "payload_sha256": hashlib.sha256(detail.encode()).hexdigest(),
                    }
                )
            )
            raise ProviderError("Epicure MCP service call failed") from exc

        result_text = result.text
        result_bytes = result_text.encode()
        self.cumulative_result_bytes += len(result_bytes)
        structured = dict(result.structured)
        if result.is_error:
            structured["flavourbench_error_kind"] = "tool_returned_error"
        trace = ToolTrace(
            round_index=self.round_index,
            name=name,
            arguments=arguments_dict,
            result=result_text,
            latency_ms=result.latency_ms or round((time.monotonic() - call_started) * 1000),
            is_error=result.is_error,
            call_index=self.call_index,
            tool_call_id=self.tool_use_id,
            structured_content=structured,
        )
        self.traces.append(trace)
        if self.tool_sink is not None:
            self.tool_sink(self.spec.arm_id, trace)
        self._emit_attempt(
            ProviderAttemptEvent(
                **{
                    **started.__dict__,
                    "event_type": "mcp_call_completed",
                    "payload_sha256": hashlib.sha256(result_bytes).hexdigest(),
                    "metadata": {
                        "tool_name": name,
                        "latency_ms": trace.latency_ms,
                        "is_error": result.is_error,
                        "execution_backend": "bedrock",
                    },
                }
            )
        )
        if self.cumulative_result_bytes > self.settings.max_cumulative_tool_result_bytes:
            raise ProviderError("cumulative Epicure tool evidence exceeded the frozen cap")
        if result.is_error:
            self.error_repairs += 1
            if self.error_repairs > 1:
                raise ProviderError("tool call remained invalid after one repair")

        if len(result_bytes) <= self.settings.max_tool_result_bytes:
            model_text = result_text
        else:
            bounded = result_bytes[: self.settings.max_tool_result_bytes]
            while bounded:
                try:
                    model_text = bounded.decode()
                    break
                except UnicodeDecodeError:
                    bounded = bounded[:-1]
            else:
                model_text = ""
            model_text += (
                "\n[FlavourBench truncated this tool result before returning it to "
                "the model; the complete trace remains in the audit record.]"
            )
        return BedrockToolExecution(content=model_text, is_error=result.is_error)


class BedrockServiceProvider:
    """GenerationSpec-compatible Bedrock backend for the asynchronous worker."""

    def __init__(
        self,
        attempt_sink: AttemptSink | None = None,
        tool_sink: ToolSink | None = None,
        *,
        runtime: Any | None = None,
        control: Any | None = None,
        lane_settings: BedrockLaneSettings | None = None,
    ) -> None:
        self.attempt_sink = attempt_sink
        self.tool_sink = tool_sink
        self._runtime = runtime
        self._control = control
        self._lane_settings = lane_settings
        self._accepted_by_arm: dict[str, _JournaledBedrockRuntime] = {}
        self._started_by_arm: dict[str, float] = {}
        self._executor_by_arm: dict[str, _EpicureBedrockExecutor] = {}
        self._backend_response_schema_by_arm: dict[str, str] = {}
        self._backend_tool_schema_by_arm: dict[str, str] = {}
        self._control_receipt_by_arm: dict[str, dict[str, Any]] = {}

    def _settings(self) -> BedrockLaneSettings:
        settings = self._lane_settings or BedrockLaneSettings.from_environ()
        if not settings.enabled or not settings.live_authorized:
            raise ProviderError("Bedrock execution is not explicitly authorized")
        if settings.stage != "season":
            raise ProviderError("the worker requires the Bedrock season execution stage")
        return settings

    def _clients(self, settings: BedrockLaneSettings) -> tuple[Any, Any]:
        if self._runtime is None or self._control is None:
            try:
                from botocore.config import Config
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ProviderError("Bedrock dependencies are not installed") from exc
            clients = create_boto3_clients(
                settings,
                client_config=Config(
                    connect_timeout=10,
                    read_timeout=300,
                    retries={"max_attempts": 0, "mode": "standard"},
                ),
            )
            if self._runtime is None:
                self._runtime = clients.runtime
            if self._control is None:
                self._control = clients.control
        return self._control, self._runtime

    def _emit_attempt(self, event: ProviderAttemptEvent) -> None:
        if self.attempt_sink is not None:
            self.attempt_sink(event)

    async def generate(self, spec: GenerationSpec) -> GenerationResult:
        if spec.execution_backend != "bedrock":
            raise ProviderError("Bedrock adapter received a non-Bedrock arm")
        if not spec.backend_contract_json:
            raise ProviderError("Bedrock arm has no frozen backend identity contract")
        try:
            contract = parse_bedrock_endpoint_contract(spec.backend_contract_json)
        except ValueError as exc:
            raise ProviderError("Bedrock backend identity contract is invalid") from exc
        lane_settings = self._settings()
        effective_stage_cap_micros = int(lane_settings.effective_stage_cap_usd * 1_000_000)
        if (
            contract.region != lane_settings.region
            or contract.profile_scope != lane_settings.profile_scope
            or contract.canonical_model_id != (spec.expected_actual_model_id or spec.model_id)
            or spec.provider_budget_cap_micros <= 0
            or spec.provider_budget_cap_micros > effective_stage_cap_micros
            or spec.provider_account_budget_cap_micros
            != provider_account_hard_cap_micros("bedrock")
            or spec.provider_account_budget_cap_micros > effective_stage_cap_micros
            or spec.provider_account_scope_sha256 != provider_account_scope_sha256("bedrock")
            or lane_settings.account_scope_sha256 != provider_account_scope_sha256("bedrock")
            or spec.contract_smoke_registry_sha256 != lane_settings.contract_smoke_evidence_sha256
            or len(spec.provider_authorization_envelope_sha256) != 64
            or len(spec.provider_account_authorization_envelope_sha256) != 64
            or len(spec.provider_credential_binding_sha256) != 64
            or spec.provider_credential_scope_sha256 != provider_account_scope_sha256("bedrock")
        ):
            raise ProviderError(
                "Bedrock runtime, smoke evidence, or provider budget does not match "
                "the frozen execution contract"
            )
        control, runtime = self._clients(lane_settings)
        control_receipt = _bedrock_control_plane_preflight(control, contract, spec)
        self._control_receipt_by_arm[spec.arm_id] = control_receipt
        self._emit_attempt(
            ProviderAttemptEvent(
                attempt_id=str(uuid.uuid4()),
                arm_id=spec.arm_id,
                request_key_sha256=hashlib.sha256(
                    f"{spec.idempotency_key}:bedrock-control-plane".encode()
                ).hexdigest(),
                phase="bedrock_control_plane_preflight",
                attempt_index=0,
                event_type="bedrock_control_plane_attested",
                payload_sha256=_canonical_sha256(control_receipt),
                metadata={"receipt": control_receipt},
            )
        )
        decoding = spec.decoding_parameters or {}
        if "max_tokens" not in decoding:
            raise ProviderError("Bedrock endpoint lacks a frozen output-token bound")
        if decoding.get("seed") is not None:
            raise ProviderError("Bedrock Converse does not accept the frozen seed parameter")
        inference = BedrockInferenceConfig(
            max_tokens=int(decoding["max_tokens"]),
            temperature=(
                float(decoding["temperature"]) if decoding.get("temperature") is not None else None
            ),
            top_p=(float(decoding["top_p"]) if decoding.get("top_p") is not None else None),
        )
        effective_decoding = {
            name: decoding.get(name, "provider_fixed_unsupported")
            for name in ("max_tokens", "temperature", "top_p", "seed")
        }
        journaled_runtime = _JournaledBedrockRuntime(
            runtime,
            contract,
            spec,
            self.attempt_sink,
        )
        self._accepted_by_arm[spec.arm_id] = journaled_runtime
        self._started_by_arm[spec.arm_id] = time.monotonic()
        self._backend_response_schema_by_arm[spec.arm_id] = _canonical_sha256(BEDROCK_FINAL_SCHEMA)
        self._backend_tool_schema_by_arm[spec.arm_id] = _canonical_sha256([])
        system_prompt = SYSTEM_PROMPT + (
            "\n\n" + EPICURE_PROMPT if spec.condition == "epicure_on" else ""
        )
        epicure_attestation: dict[str, Any] = {}
        executor: _EpicureBedrockExecutor | None = None
        tools: tuple[BedrockToolDefinition, ...] = ()
        started = time.monotonic()
        try:
            if spec.condition == "epicure_on":
                session_attempt_id = str(uuid.uuid4())
                session_key = hashlib.sha256(
                    f"{spec.idempotency_key}:mcp-session".encode()
                ).hexdigest()
                self._emit_attempt(
                    ProviderAttemptEvent(
                        attempt_id=session_attempt_id,
                        arm_id=spec.arm_id,
                        request_key_sha256=session_key,
                        phase="mcp_session",
                        attempt_index=0,
                        event_type="mcp_session_started",
                        payload_sha256=spec.expected_epicure_tool_schema_sha256,
                        metadata={
                            "protocol_bundle_sha256": spec.protocol_bundle_sha256,
                            "execution_backend": "bedrock",
                        },
                    )
                )
                async with McpSession() as mcp:
                    mcp_tools = await mcp.list_tools()
                    epicure_attestation = await mcp.attest_runtime(
                        expected={
                            "release_id": spec.expected_epicure_release_id,
                            "bundle_sha256": spec.expected_epicure_bundle_sha256,
                            "application_sha256": (spec.expected_epicure_application_sha256),
                            "tool_schema_sha256": (spec.expected_epicure_tool_schema_sha256),
                        },
                        tools=mcp_tools,
                    )
                    attestation_sha256 = _canonical_sha256(epicure_attestation)
                    self._emit_attempt(
                        ProviderAttemptEvent(
                            attempt_id=session_attempt_id,
                            arm_id=spec.arm_id,
                            request_key_sha256=session_key,
                            phase="mcp_attestation",
                            attempt_index=0,
                            event_type="mcp_session_attested",
                            payload_sha256=attestation_sha256,
                            metadata={
                                "attestation": epicure_attestation,
                                "attestation_sha256": attestation_sha256,
                                "protocol_bundle_sha256": spec.protocol_bundle_sha256,
                                "execution_backend": "bedrock",
                            },
                        )
                    )
                    tools = tuple(
                        BedrockToolDefinition(
                            name=str(tool["name"]),
                            description=str(tool.get("description") or tool["name"]),
                            input_schema=project_bedrock_json_schema(
                                tool.get("inputSchema") or {"type": "object"}
                            ),
                        )
                        for tool in mcp_tools
                        if isinstance(tool.get("name"), str)
                    )
                    self._backend_tool_schema_by_arm[spec.arm_id] = _canonical_sha256(
                        [tool.as_converse_tool() for tool in tools]
                    )
                    executor = _EpicureBedrockExecutor(
                        mcp=mcp,
                        spec=spec,
                        attempt_sink=self.attempt_sink,
                        tool_sink=self.tool_sink,
                    )
                    self._executor_by_arm[spec.arm_id] = executor
                    result = await BedrockConverseProvider(
                        journaled_runtime,
                        contract,
                        tool_executor=executor,
                        max_tool_rounds=get_settings().max_tool_rounds,
                        max_tool_calls_per_round=(get_settings().max_tool_calls_per_round),
                        max_tool_calls_total=get_settings().max_tool_calls_total,
                    ).generate(
                        BedrockGenerationSpec(
                            arm_id=spec.arm_id,
                            canonical_model_id=contract.canonical_model_id,
                            prompt=spec.prompt,
                            system_prompt=system_prompt,
                            inference=inference,
                            tools=tools,
                            request_metadata={
                                "flavourbench_protocol_sha256": (spec.protocol_bundle_sha256),
                            },
                        )
                    )
            else:
                result = await BedrockConverseProvider(
                    journaled_runtime,
                    contract,
                    max_tool_rounds=get_settings().max_tool_rounds,
                    max_tool_calls_per_round=get_settings().max_tool_calls_per_round,
                    max_tool_calls_total=get_settings().max_tool_calls_total,
                ).generate(
                    BedrockGenerationSpec(
                        arm_id=spec.arm_id,
                        canonical_model_id=contract.canonical_model_id,
                        prompt=spec.prompt,
                        system_prompt=system_prompt,
                        inference=inference,
                        request_metadata={
                            "flavourbench_protocol_sha256": spec.protocol_bundle_sha256,
                        },
                    )
                )
        except UncertainDeliveryError:
            raise
        except Exception as exc:
            if journaled_runtime.uncertain_delivery:
                raise UncertainDeliveryError(
                    "Bedrock delivery became uncertain; automatic fallback is forbidden"
                ) from exc
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"Bedrock generation failed: {type(exc).__name__}") from exc

        cost_micros = sum(
            int(item.get("cost_micros") or 0) for item in journaled_runtime.accounting
        )
        cost_reconciled = bool(journaled_runtime.accounting) and all(
            item.get("reconciled") is True for item in journaled_runtime.accounting
        )
        traces = executor.traces if executor is not None else []
        identity_metadata = {
            "identity_evidence": result.identity.identity_evidence,
            "requested_model_or_profile_id_sha256": hashlib.sha256(
                result.identity.requested_model_or_profile_id.encode()
            ).hexdigest(),
            "requested_model_or_profile_arn_sha256": (
                result.identity.requested_model_or_profile_arn_sha256
            ),
            "profile_scope_sha256": result.identity.profile_scope_sha256,
            "returned_model_ids": list(result.identity.returned_model_ids),
            "returned_model_id_sha256s": list(result.identity.returned_model_id_sha256s),
            "actual_execution_region": result.identity.actual_execution_region,
            "actual_foundation_model_id": (result.identity.actual_foundation_model_id),
            "control_plane_receipt": control_receipt,
        }
        generation_metadata = [
            {**item, **identity_metadata} for item in journaled_runtime.accounting
        ]
        return GenerationResult(
            answer_markdown=result.answer_markdown,
            output_json=dict(result.output_json),
            actual_model_id=contract.canonical_model_id,
            provider_slug="amazon-bedrock",
            generation_id=(
                journaled_runtime.generation_ids[-1] if journaled_runtime.generation_ids else ""
            ),
            generation_ids=list(journaled_runtime.generation_ids),
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            cost_micros=cost_micros,
            cost_reconciled=cost_reconciled,
            latency_ms=result.wall_clock_latency_ms or round((time.monotonic() - started) * 1000),
            retries=result.retries,
            finish_reason=result.finish_reason,
            tool_traces=list(traces),
            generation_metadata=generation_metadata,
            decoding_json=effective_decoding,
            epicure_attestation=epicure_attestation,
            backend_response_schema_sha256=result.response_schema_sha256,
            backend_tool_schema_sha256=result.tool_schema_sha256,
            cost_accounting_basis="frozen_rate_card_times_returned_usage",
            billing_reconciliation_status="pending_aws_billing_crosscheck",
        )

    async def reconcile_failure(
        self,
        spec: GenerationSpec,
        error: Exception,
    ) -> GenerationFailureResult | None:
        runtime = self._accepted_by_arm.get(spec.arm_id)
        if runtime is None or not runtime.generation_ids:
            return None
        accounting = list(runtime.accounting)
        control_receipt = self._control_receipt_by_arm.get(spec.arm_id, {})
        if control_receipt:
            accounting = [{**item, "control_plane_receipt": control_receipt} for item in accounting]
        return GenerationFailureResult(
            error=error,
            actual_model_id=spec.expected_actual_model_id or spec.model_id,
            provider_slug="amazon-bedrock",
            generation_id=runtime.generation_ids[-1],
            generation_ids=list(runtime.generation_ids),
            prompt_tokens=sum(int(item.get("tokens_prompt") or 0) for item in accounting),
            completion_tokens=sum(int(item.get("tokens_completion") or 0) for item in accounting),
            cost_micros=sum(int(item.get("cost_micros") or 0) for item in accounting),
            cost_reconciled=bool(accounting)
            and all(item.get("reconciled") is True for item in accounting),
            retries=0,
            generation_metadata=accounting,
            decoding_json={
                name: (spec.decoding_parameters or {}).get(name, "provider_fixed_unsupported")
                for name in ("max_tokens", "temperature", "top_p", "seed")
            },
            latency_ms=round(
                (time.monotonic() - self._started_by_arm.get(spec.arm_id, time.monotonic())) * 1000
            ),
            tool_traces=list(
                self._executor_by_arm.get(spec.arm_id).traces
                if spec.arm_id in self._executor_by_arm
                else []
            ),
            cost_accounting_basis="frozen_rate_card_times_returned_usage",
            billing_reconciliation_status="pending_aws_billing_crosscheck",
            backend_response_schema_sha256=(
                self._backend_response_schema_by_arm.get(spec.arm_id, "unresolved")
            ),
            backend_tool_schema_sha256=self._backend_tool_schema_by_arm.get(
                spec.arm_id, "unresolved"
            ),
        )

    async def aclose(self) -> None:
        return None
