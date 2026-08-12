"""Content-addressed execution policy shared by real FlavourBench runners."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .config import Settings, get_settings

POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v1"
PLAIN_TEXT_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v2"
MATCHED_EVIDENCE_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v3"
DIRECT_TOOL_CONTRACT_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v4"
SEPARATE_TOOL_CONTRACT_LIMIT_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v5"
SAFE_RETRY_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v6"
EVIDENCE_BOUNDARY_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v7"
MATCHED_TOOL_ACCESS_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v8"
REQUIRED_EPICURE_TREATMENT_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v9"
PORTABLE_TEXT_TOOL_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v10"
SELECTION_TEXT_POLICY_SCHEMA_VERSION = "flavourbench-real-execution-policy-v11"
DIRECT_TOOL_CONTRACT_PROTOCOL = "direct_tool_first_v1"
MATCHED_EVIDENCE_PROTOCOL_V1 = "matched_evidence_v1"
MATCHED_EVIDENCE_PROTOCOL_V2 = "matched_evidence_v2"
MATCHED_TOOL_ACCESS_PROTOCOL_V1 = "matched_tool_access_v1"
PORTABLE_TEXT_TOOL_PROTOCOL_V1 = "portable_text_tool_v1"
SELECTION_TEXT_PROTOCOL_V1 = "selection_text_v1"
MATCHED_EVIDENCE_PROTOCOLS = frozenset({MATCHED_EVIDENCE_PROTOCOL_V1, MATCHED_EVIDENCE_PROTOCOL_V2})
GOVERNED_EPICURE_PROTOCOLS = frozenset(
    {
        *MATCHED_EVIDENCE_PROTOCOLS,
        MATCHED_TOOL_ACCESS_PROTOCOL_V1,
        PORTABLE_TEXT_TOOL_PROTOCOL_V1,
        SELECTION_TEXT_PROTOCOL_V1,
    }
)


def assert_legacy_paid_cli_allowed(command: str) -> None:
    """Keep file-governed historical runners out of the production call plane."""

    if get_settings().environment == "production":
        raise RuntimeError(
            f"{command} is disabled in production; submit paid work through the "
            "PostgreSQL-governed API and worker"
        )


def _sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


@dataclass(frozen=True)
class ExecutionPolicy:
    max_output_tokens: int = 1_000
    max_tool_rounds: int = 4
    max_tool_result_bytes: int = 16_384
    max_cumulative_tool_result_bytes: int = 49_152
    max_tool_calls_per_round: int = 4
    max_tool_calls_total: int = 12
    max_provider_attempts: int = 1
    decoding_temperature: float = 0.2
    decoding_top_p: float = 0.95
    decoding_seed: int = 20_260_715
    tool_argument_repair_turns: int = 1
    approximate_non_user_prompt_bytes: int = 2_000
    conservative_bytes_per_token: int = 3
    pair_arm_scheduling: str = "concurrent"
    final_response_mode: str = "structured_json"
    max_intermediate_tokens: int = 700
    required_tool_contract_max_intermediate_tokens: int = 2_048
    matched_planning: bool = False
    evidence_protocol: str = "legacy_v6"
    intermediate_reasoning_effort: str | None = None
    final_reasoning_effort: str | None = None
    required_tool_contract_protocol: str = DIRECT_TOOL_CONTRACT_PROTOCOL
    tool_catalog_bytes_bound: int = 0
    epicure_on_tool_required: bool = False

    def validate(self) -> None:
        integer_fields = {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "decoding_temperature",
                "decoding_top_p",
                "pair_arm_scheduling",
                "final_response_mode",
                "matched_planning",
                "evidence_protocol",
                "intermediate_reasoning_effort",
                "final_reasoning_effort",
                "required_tool_contract_protocol",
                "tool_catalog_bytes_bound",
                "epicure_on_tool_required",
            }
        }
        for field, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"execution policy {field} must be a positive integer")
        if self.max_tool_rounds > 8:
            raise ValueError("execution policy permits at most eight Epicure tool rounds")
        if self.max_tool_calls_total < self.max_tool_calls_per_round:
            raise ValueError("total tool-call cap cannot be below the per-round cap")
        if self.max_output_tokens < 128 or self.max_output_tokens > 16_384:
            raise ValueError("execution policy output token bound is outside service limits")
        if self.max_intermediate_tokens > self.max_output_tokens:
            raise ValueError("intermediate token bound cannot exceed the final output bound")
        if self.required_tool_contract_max_intermediate_tokens > 8_192:
            raise ValueError("required-tool intermediate bound is outside service limits")
        if not 0 <= self.decoding_temperature <= 2:
            raise ValueError("execution policy temperature must be in [0, 2]")
        if not 0 < self.decoding_top_p <= 1:
            raise ValueError("execution policy top_p must be in (0, 1]")
        if self.pair_arm_scheduling not in {"concurrent", "sequential"}:
            raise ValueError("execution policy pair-arm scheduling is unsupported")
        if self.final_response_mode not in {"structured_json", "plain_text"}:
            raise ValueError("execution policy final-response mode is unsupported")
        if not isinstance(self.matched_planning, bool):
            raise ValueError("execution policy matched-planning flag must be boolean")
        if self.evidence_protocol not in {"legacy_v6", *GOVERNED_EPICURE_PROTOCOLS}:
            raise ValueError("execution policy evidence protocol is unsupported")
        reasoning_efforts = {None, "none", "minimal", "low", "medium", "high", "xhigh", "max"}
        if (
            self.intermediate_reasoning_effort not in reasoning_efforts
            or self.final_reasoning_effort not in reasoning_efforts
        ):
            raise ValueError("execution policy reasoning effort is unsupported")
        if self.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS and not self.matched_planning:
            raise ValueError("matched-evidence execution requires matched planning")
        if self.evidence_protocol == MATCHED_TOOL_ACCESS_PROTOCOL_V1 and self.matched_planning:
            raise ValueError("matched-tool-access execution prohibits staged planning")
        if self.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1 and self.matched_planning:
            raise ValueError("portable text-tool execution prohibits staged planning")
        if self.required_tool_contract_protocol != DIRECT_TOOL_CONTRACT_PROTOCOL:
            raise ValueError("execution policy required-tool contract protocol is unsupported")
        if not isinstance(self.epicure_on_tool_required, bool):
            raise ValueError("Epicure tool requirement must be boolean")
        if self.epicure_on_tool_required and self.evidence_protocol not in {
            *MATCHED_EVIDENCE_PROTOCOLS,
            PORTABLE_TEXT_TOOL_PROTOCOL_V1,
        }:
            raise ValueError(
                "required Epicure treatment is supported only by a matched-evidence protocol"
            )
        if (
            not isinstance(self.tool_catalog_bytes_bound, int)
            or isinstance(self.tool_catalog_bytes_bound, bool)
            or self.tool_catalog_bytes_bound < 0
        ):
            raise ValueError("execution policy tool-catalog byte bound is invalid")

    def unhashed_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "schema_version": (
                SELECTION_TEXT_POLICY_SCHEMA_VERSION
                if self.evidence_protocol == SELECTION_TEXT_PROTOCOL_V1
                else PORTABLE_TEXT_TOOL_POLICY_SCHEMA_VERSION
                if self.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
                else REQUIRED_EPICURE_TREATMENT_POLICY_SCHEMA_VERSION
                if self.epicure_on_tool_required
                else MATCHED_TOOL_ACCESS_POLICY_SCHEMA_VERSION
                if self.evidence_protocol == MATCHED_TOOL_ACCESS_PROTOCOL_V1
                else EVIDENCE_BOUNDARY_POLICY_SCHEMA_VERSION
                if self.evidence_protocol == MATCHED_EVIDENCE_PROTOCOL_V2
                else SAFE_RETRY_POLICY_SCHEMA_VERSION
                if self.evidence_protocol == MATCHED_EVIDENCE_PROTOCOL_V1
                else PLAIN_TEXT_POLICY_SCHEMA_VERSION
                if self.final_response_mode == "plain_text"
                else POLICY_SCHEMA_VERSION
            ),
            "limits": {
                "max_output_tokens": self.max_output_tokens,
                "max_tool_rounds": self.max_tool_rounds,
                "max_tool_result_bytes": self.max_tool_result_bytes,
                "max_cumulative_tool_result_bytes": (self.max_cumulative_tool_result_bytes),
                "max_tool_calls_per_round": self.max_tool_calls_per_round,
                "max_tool_calls_total": self.max_tool_calls_total,
                "max_provider_attempts": self.max_provider_attempts,
                "tool_argument_repair_turns": self.tool_argument_repair_turns,
            },
            "decoding": {
                "temperature": self.decoding_temperature,
                "top_p": self.decoding_top_p,
                "seed": self.decoding_seed,
            },
            "cost_forecast": {
                "approximate_non_user_prompt_bytes": (self.approximate_non_user_prompt_bytes),
                "conservative_bytes_per_token": self.conservative_bytes_per_token,
            },
            "pair_arm_scheduling": self.pair_arm_scheduling,
        }
        if (
            self.final_response_mode == "plain_text"
            or self.evidence_protocol in GOVERNED_EPICURE_PROTOCOLS
        ):
            payload["final_response_mode"] = self.final_response_mode
            payload["limits"]["max_intermediate_tokens"] = self.max_intermediate_tokens
            payload["limits"]["required_tool_contract_max_intermediate_tokens"] = (
                self.required_tool_contract_max_intermediate_tokens
            )
            payload["matched_planning"] = self.matched_planning
            payload["cost_forecast"]["tool_catalog_bytes_bound"] = self.tool_catalog_bytes_bound
        if self.evidence_protocol in GOVERNED_EPICURE_PROTOCOLS:
            payload["evidence_protocol"] = self.evidence_protocol
            payload["required_tool_contract_protocol"] = self.required_tool_contract_protocol
            payload["epicure_on_tool_required"] = self.epicure_on_tool_required
            payload["reasoning"] = {
                "intermediate_effort": self.intermediate_reasoning_effort,
                "final_effort": self.final_reasoning_effort,
                "exclude_from_provider_response": True,
            }
            payload["provider_retry_policy"] = {
                "maximum_attempts": self.max_provider_attempts,
                "retryable_failures": [
                    "connect_error_before_send",
                    "http_429_request_rejected",
                ],
                "ambiguous_delivery_retried": False,
                "backoff": (
                    "retry_after_or_exponential_0.4s_plus_sha256_idempotency_jitter_0_to_0.999s"
                ),
                "maximum_backoff_seconds": 30,
            }
        return payload

    @property
    def sha256(self) -> str:
        return _sha256(self.unhashed_payload())

    def document(self) -> dict[str, Any]:
        payload = self.unhashed_payload()
        digest = _sha256(payload)
        return {
            **payload,
            "content_address": {
                "algorithm": "sha256",
                "digest": digest,
                "uri": f"sha256:{digest}",
            },
        }

    def settings_environment(self) -> dict[str, str]:
        """Return exact service-setting overrides for a dedicated subprocess."""

        return {
            "FLAVOURBENCH_MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
            "FLAVOURBENCH_MAX_INTERMEDIATE_TOKENS": str(self.max_intermediate_tokens),
            "FLAVOURBENCH_MAX_TOOL_ROUNDS": str(self.max_tool_rounds),
            "FLAVOURBENCH_MAX_TOOL_RESULT_BYTES": str(self.max_tool_result_bytes),
            "FLAVOURBENCH_MAX_CUMULATIVE_TOOL_RESULT_BYTES": str(
                self.max_cumulative_tool_result_bytes
            ),
            "FLAVOURBENCH_MAX_TOOL_CALLS_PER_ROUND": str(self.max_tool_calls_per_round),
            "FLAVOURBENCH_MAX_TOOL_CALLS_TOTAL": str(self.max_tool_calls_total),
            "FLAVOURBENCH_MAX_PROVIDER_ATTEMPTS": str(self.max_provider_attempts),
            "FLAVOURBENCH_DECODING_TEMPERATURE": str(self.decoding_temperature),
            "FLAVOURBENCH_DECODING_TOP_P": str(self.decoding_top_p),
            "FLAVOURBENCH_DECODING_SEED": str(self.decoding_seed),
        }

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        pair_arm_scheduling: str,
        final_response_mode: str = "structured_json",
        matched_planning: bool = False,
        required_tool_contract_max_intermediate_tokens: int = 2_048,
        evidence_protocol: str = "legacy_v6",
        intermediate_reasoning_effort: str | None = None,
        final_reasoning_effort: str | None = None,
        required_tool_contract_protocol: str = DIRECT_TOOL_CONTRACT_PROTOCOL,
        tool_catalog_bytes_bound: int = 0,
        epicure_on_tool_required: bool = False,
        approximate_non_user_prompt_bytes: int = 2_000,
        conservative_bytes_per_token: int = 3,
    ) -> ExecutionPolicy:
        return cls(
            max_output_tokens=settings.max_output_tokens,
            max_tool_rounds=settings.max_tool_rounds,
            max_tool_result_bytes=settings.max_tool_result_bytes,
            max_cumulative_tool_result_bytes=(settings.max_cumulative_tool_result_bytes),
            max_tool_calls_per_round=settings.max_tool_calls_per_round,
            max_tool_calls_total=settings.max_tool_calls_total,
            max_provider_attempts=settings.max_provider_attempts,
            decoding_temperature=settings.decoding_temperature,
            decoding_top_p=settings.decoding_top_p,
            decoding_seed=settings.decoding_seed,
            approximate_non_user_prompt_bytes=approximate_non_user_prompt_bytes,
            conservative_bytes_per_token=conservative_bytes_per_token,
            pair_arm_scheduling=pair_arm_scheduling,
            final_response_mode=final_response_mode,
            max_intermediate_tokens=settings.max_intermediate_tokens,
            required_tool_contract_max_intermediate_tokens=(
                required_tool_contract_max_intermediate_tokens
            ),
            matched_planning=matched_planning,
            evidence_protocol=evidence_protocol,
            intermediate_reasoning_effort=intermediate_reasoning_effort,
            final_reasoning_effort=final_reasoning_effort,
            required_tool_contract_protocol=required_tool_contract_protocol,
            tool_catalog_bytes_bound=tool_catalog_bytes_bound,
            epicure_on_tool_required=epicure_on_tool_required,
        )


def verify_policy_document(document: object) -> bool:
    if not isinstance(document, dict):
        return False
    address = document.get("content_address")
    if not isinstance(address, dict):
        return False
    digest = address.get("digest")
    if address.get("algorithm") != "sha256" or address.get("uri") != f"sha256:{digest}":
        return False
    unhashed = dict(document)
    unhashed.pop("content_address", None)
    return isinstance(digest, str) and len(digest) == 64 and _sha256(unhashed) == digest
