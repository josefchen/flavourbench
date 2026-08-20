"""Sequential, budget-gated contract smokes for a frozen frontier manifest.

The runner is deliberately separate from the scored benchmark worker.  It can
only create permanently unranked ``live_smoke`` artifacts, admits one model at
a time, and treats an uncertain or failed run as having consumed its complete
per-run allowance until generation accounting proves otherwise.

Planning is the default and makes no provider calls.  Execution requires two
explicit switches and delegates each call to :mod:`flavourbench.live_smoke` so
the existing OpenRouter, Cloudflare AI Gateway, Epicure attestation, identity,
and generation-accounting checks remain the single implementation of record.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

from .config import get_settings
from .execution_policy import assert_legacy_paid_cli_allowed
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import CONFIRMATION as LIVE_SMOKE_CONFIRMATION
from .live_smoke import TOOL_CONTRACT_PROMPT, endpoint_execution_contract_sha256
from .run_journal import JournalIntegrityError, verify_journal_descriptor

CURRENT_CANDIDATE_MANIFEST_SHA256 = (
    "eb9e9b591d1695c38aeb79d65b59904d848b41dea449090eaeff8ebbed2138a2"
)
AUTHORIZED_TOTAL_CAP_USD = Decimal("200")
DEFAULT_ADMISSION_FRACTION = Decimal("0.85")
EXECUTION_CONFIRMATION = "RUN_SEQUENTIAL_UNRANKED_FRONTIER_CONTRACTS"
RUNNER_SCHEMA_VERSION = "flavourbench-frontier-contract-runner-v1"
LEDGER_SCHEMA_VERSION = "flavourbench-frontier-contract-ledger-v1"
SUMMARY_SCHEMA_VERSION = "flavourbench-frontier-contract-summary-v1"
LIVE_SMOKE_SCHEMA_VERSION = "flavourbench-live-smoke-v1"
COST_CORRECTION_SCHEMA_VERSION = "flavourbench-live-smoke-cost-correction-v1"
NO_ARTIFACT_RECONCILIATION_SCHEMA_VERSION = "flavourbench-frontier-no-artifact-reconciliation-v1"
NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION = "flavourbench-frontier-no-artifact-reconciliation-v2"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
RATE_CARD_ACCOUNTING_BASIS_BY_BACKEND = {
    "kimi_direct": "frozen_rate_card_times_kimi_returned_usage",
    "cohere_direct": "frozen_rate_card_times_cohere_returned_usage",
    "qwencloud_direct": "frozen_rate_card_times_qwencloud_returned_usage",
    "zai_coding_direct": "zai_coding_plan_subscription_quota",
}
RATE_CARD_PROVIDER_SLUG_BY_BACKEND = {
    "kimi_direct": "kimi-code-direct",
    "cohere_direct": "cohere-direct",
    "qwencloud_direct": "qwencloud-direct",
    "zai_coding_direct": "zai-coding-plan-direct",
}
QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS = (
    "qwencloud_returned_usage_with_full_unpriced_budget_ceiling"
)
QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS = "provider_rate_and_charge_unavailable"
PRE_GENERATION_STDOUT_MATCHES = {
    hashlib.sha256(
        b'{"status": "failed", "error": "RuntimeError: configured max output exceeds '
        b'the endpoint completion limit"}\n'
    ).hexdigest(): {
        "safe_stdout": (
            '{"status": "failed", "error": "RuntimeError: configured max output '
            'exceeds the endpoint completion limit"}\n'
        ),
        "failure_boundary": "before_openrouter_provider_instantiation",
        "source_symbol": "flavourbench.live_smoke.frozen_generation_contract",
    }
}


class ContractRunnerError(RuntimeError):
    """Base error for a rejected contract-runner operation."""


class IntegrityError(ContractRunnerError):
    """Raised when immutable input or ledger content fails verification."""


class AdmissionDenied(ContractRunnerError):
    """Raised when an execution request is not within the authorised envelope."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _live_smoke_sha256(value: object) -> str:
    """Match live_smoke's v1 canonicalisation, including ASCII escaping."""

    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise IntegrityError(f"{field} is not a decimal: {value!r}") from error
    if not parsed.is_finite() or parsed < 0:
        raise IntegrityError(f"{field} must be finite and non-negative")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"{label} must be a regular, non-symlink file: {path}")


def _timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{field} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise IntegrityError(f"{field} must be an RFC 3339 timestamp") from error
    if parsed.tzinfo is None:
        raise IntegrityError(f"{field} must include a UTC offset")
    return parsed.astimezone(UTC)


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IntegrityError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ContractPolicy:
    """Frozen bounds shared with the current ``live_smoke`` admission check."""

    max_tool_rounds: int = 8
    max_output_tokens: int = 1_800
    max_tool_result_bytes: int = 32_768
    approximate_non_user_prompt_bytes: int = 2_000
    conservative_bytes_per_token: int = 3

    def validate(self) -> None:
        for field, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.max_tool_rounds > 8:
            raise ValueError("FlavourBench permits at most eight Epicure tool rounds")


DEFAULT_CONTRACT_POLICY = ContractPolicy()


@dataclass(frozen=True)
class ContractCandidate:
    slot_id: str
    model_id: str
    canonical_model_slug: str
    model_name: str
    provider_tag: str
    provider_name: str
    endpoint_sha256: str
    endpoint_execution_sha256: str
    endpoint: Mapping[str, Any]
    execution_backend: str = "openrouter"
    backend_contract: Mapping[str, Any] = dataclass_field(default_factory=dict)
    backend_contract_sha256: str = "unfrozen"
    route_selection: Mapping[str, Any] = dataclass_field(default_factory=dict)
    cost_accounting_policy: str = "provider_generation_metadata"


@dataclass(frozen=True)
class PriceEnvelope:
    prompt_usd_per_token: Decimal
    completion_usd_per_token: Decimal
    reasoning_usd_per_token: Decimal
    request_usd: Decimal
    prompt_usd_per_mtok: Decimal
    completion_usd_per_mtok: Decimal

    def public_payload(self) -> dict[str, str]:
        return {key: _decimal_text(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ContractForecast:
    forecast_usd: Decimal
    price_envelope: PriceEnvelope
    prompt_tokens_per_request_bound: Decimal
    completion_tokens_per_request_bound: int
    actual_contract_request_bound: int
    live_smoke_admission_request_bound: int

    def public_payload(self) -> dict[str, Any]:
        return {
            "forecast_usd": _decimal_text(self.forecast_usd),
            "price_envelope": self.price_envelope.public_payload(),
            "prompt_tokens_per_request_bound": _decimal_text(self.prompt_tokens_per_request_bound),
            "completion_tokens_per_request_bound": self.completion_tokens_per_request_bound,
            "actual_contract_request_bound": self.actual_contract_request_bound,
            "live_smoke_admission_request_bound": self.live_smoke_admission_request_bound,
            "forecast_basis": (
                "mirrors the existing live_smoke worst-case admission calculation; it reserves "
                "the off/on plus contract envelope even though this runner requests contract-only"
            ),
        }


@dataclass(frozen=True)
class ArtifactExposure:
    path: Path
    artifact_sha256: str
    status: str
    requested_model_id: str
    requested_provider: str
    candidate_manifest_sha256: str | None
    actual_cost_usd: Decimal
    forecast_usd: Decimal
    admitted_cap_usd: Decimal
    exposure_usd: Decimal
    exposure_basis: str
    contract_passed: bool
    cost_correction_sha256: str | None = None

    def public_payload(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "artifact_sha256": self.artifact_sha256,
            "status": self.status,
            "requested_model_id": self.requested_model_id,
            "requested_provider": self.requested_provider,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "actual_cost_usd": _decimal_text(self.actual_cost_usd),
            "forecast_usd": _decimal_text(self.forecast_usd),
            "admitted_cap_usd": _decimal_text(self.admitted_cap_usd),
            "exposure_usd": _decimal_text(self.exposure_usd),
            "exposure_basis": self.exposure_basis,
            "contract_passed": self.contract_passed,
            "cost_correction_sha256": self.cost_correction_sha256,
        }


@dataclass(frozen=True)
class ArtifactScan:
    artifacts: tuple[ArtifactExposure, ...]
    actual_cost_usd: Decimal
    exposure_usd: Decimal
    failed_or_unreconciled_reserve_usd: Decimal
    correction_count: int = 0

    def public_payload(self) -> dict[str, Any]:
        return {
            "artifact_count": len(self.artifacts),
            "actual_cost_usd": _decimal_text(self.actual_cost_usd),
            "budget_exposure_usd": _decimal_text(self.exposure_usd),
            "failed_or_unreconciled_reserve_usd": _decimal_text(
                self.failed_or_unreconciled_reserve_usd
            ),
            "cost_correction_count": self.correction_count,
        }


@dataclass(frozen=True)
class CostCorrection:
    corrected_total_cost_micros: int
    artifact_sha256: str
    generation_ids: frozenset[str]
    all_reconciled: bool


@dataclass(frozen=True)
class NoArtifactReconciliation:
    path: Path
    artifact_sha256: str
    reservation_entry_sha256: str
    incident_entry_sha256: str
    known_pending_artifact_sha256: str
    account_usage_delta_usd: Decimal

    def public_payload(self) -> dict[str, str]:
        return {
            "filename": self.path.name,
            "artifact_sha256": self.artifact_sha256,
            "reservation_entry_sha256": self.reservation_entry_sha256,
            "incident_entry_sha256": self.incident_entry_sha256,
            "known_pending_artifact_sha256": self.known_pending_artifact_sha256,
            "account_usage_delta_usd": _decimal_text(self.account_usage_delta_usd),
        }


@dataclass(frozen=True)
class NoArtifactReconciliationV2:
    """A content-addressed proof that an atomic reservation was never delivered."""

    path: Path
    artifact_sha256: str
    reservation_entry_sha256: str
    study_plan_sha256: str
    admission_block_id: str
    work_item_id: str

    def public_payload(self) -> dict[str, str]:
        return {
            "filename": self.path.name,
            "artifact_sha256": self.artifact_sha256,
            "reservation_entry_sha256": self.reservation_entry_sha256,
            "study_plan_sha256": self.study_plan_sha256,
            "admission_block_id": self.admission_block_id,
            "work_item_id": self.work_item_id,
        }


def load_candidate_manifest(
    path: str | Path,
    *,
    expected_digest: str = CURRENT_CANDIDATE_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Load a candidate only when both its embedded and expected digests match."""

    manifest_path = Path(path)
    _require_regular_file(manifest_path, label="candidate manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IntegrityError(f"could not read candidate manifest {manifest_path}") from error
    if not isinstance(manifest, dict) or not verify_manifest_content_address(manifest):
        raise IntegrityError("candidate manifest has an invalid content address")
    digest = str((manifest.get("content_address") or {}).get("digest") or "")
    if expected_digest and digest != expected_digest:
        raise IntegrityError(
            f"candidate manifest digest {digest!r} does not match expected {expected_digest!r}"
        )
    if digest not in manifest_path.name:
        raise IntegrityError("candidate manifest filename does not contain its complete digest")
    if manifest.get("schema_version") not in {
        "flavourbench-openrouter-candidate-manifest-v1",
        "flavourbench-routed-candidate-manifest-v1",
    }:
        raise IntegrityError("unsupported candidate manifest schema")
    if manifest.get("status") != "unranked_candidate":
        raise IntegrityError("contract runner accepts only an unranked candidate manifest")
    if manifest.get("official_results_authorised") is not False:
        raise IntegrityError("candidate manifest unexpectedly authorises official results")
    if manifest.get("generation_calls_made") != 0:
        raise IntegrityError("candidate manifest discovery must contain zero generation calls")
    budget = manifest.get("budget")
    if not isinstance(budget, Mapping) or budget.get("within_cap") is not True:
        raise IntegrityError("candidate manifest has no valid bounded budget forecast")
    manifest_cap = _decimal(budget.get("cap_usd"), field="manifest budget.cap_usd")
    if manifest_cap > AUTHORIZED_TOTAL_CAP_USD:
        raise AdmissionDenied(
            f"manifest cap ${manifest_cap} exceeds the authorised ${AUTHORIZED_TOTAL_CAP_USD}"
        )
    return manifest


def select_candidates(
    manifest: Mapping[str, Any], selectors: Iterable[str] = ()
) -> list[ContractCandidate]:
    """Select exact frozen models/endpoints in manifest order."""

    entries = manifest.get("models")
    if not isinstance(entries, list) or not entries:
        raise IntegrityError("candidate manifest contains no model entries")
    candidates: list[ContractCandidate] = []
    seen_slots: set[str] = set()
    seen_models: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise IntegrityError(f"models[{index}] is not an object")
        slot = entry.get("slot")
        model = entry.get("model")
        endpoint = entry.get("endpoint")
        request_policy = entry.get("request_policy")
        if not all(isinstance(value, Mapping) for value in (slot, model, endpoint, request_policy)):
            raise IntegrityError(f"models[{index}] lacks a frozen slot/model/endpoint policy")
        provider_policy = request_policy.get("provider")
        if not isinstance(provider_policy, Mapping):
            raise IntegrityError(f"models[{index}] lacks its provider routing policy")
        slot_id = str(slot.get("slot_id") or "")
        model_id = str(model.get("id") or "")
        canonical = str(model.get("canonical_slug") or "")
        provider_tag = str(endpoint.get("tag") or "")
        provider_name = str(endpoint.get("provider_name") or "")
        execution_route = entry.get("execution_route") or {}
        if not isinstance(execution_route, Mapping):
            raise IntegrityError(f"models[{index}] has an invalid execution route")
        execution_backend = str(execution_route.get("selected_backend") or "openrouter")
        if execution_backend not in {
            "openrouter",
            "bedrock",
            "kimi_direct",
            "cohere_direct",
            "qwencloud_direct",
            "zai_coding_direct",
        }:
            raise IntegrityError(f"{model_id} has an unsupported execution backend")
        backend_contract = entry.get("backend_contract") or {}
        if not isinstance(backend_contract, Mapping):
            raise IntegrityError(f"{model_id} has an invalid backend contract")
        backend_contract_sha256 = str(entry.get("backend_contract_sha256") or "unfrozen")
        if execution_backend != "openrouter":
            if (
                len(backend_contract_sha256) != 64
                or any(character not in "0123456789abcdef" for character in backend_contract_sha256)
                or _sha256(backend_contract) != backend_contract_sha256
            ):
                raise IntegrityError(f"{model_id} backend contract is not content-bound")
        cost_accounting_policy = str(
            entry.get("cost_accounting_policy") or "provider_generation_metadata"
        )
        allowed_accounting = {
            "provider_generation_metadata",
            "provider_usage_times_frozen_rate_card",
            "provider_usage_with_unpriced_budget_ceiling",
            "bedrock_usage_times_frozen_rate_card",
            "provider_subscription_quota_no_marginal_price",
        }
        if cost_accounting_policy not in allowed_accounting:
            raise IntegrityError(f"{model_id} has an unsupported cost accounting policy")
        if (
            cost_accounting_policy == "provider_usage_with_unpriced_budget_ceiling"
            and execution_backend != "qwencloud_direct"
        ):
            raise IntegrityError(f"{model_id} unpriced ceiling policy is restricted to QwenCloud")
        if (
            cost_accounting_policy == "provider_subscription_quota_no_marginal_price"
            and execution_backend != "zai_coding_direct"
        ):
            raise IntegrityError(
                f"{model_id} subscription-quota policy is restricted to Z.ai Coding"
            )
        if not slot_id or slot_id in seen_slots:
            raise IntegrityError(f"duplicate or absent slot ID at models[{index}]")
        if not model_id or model_id in seen_models:
            raise IntegrityError(f"duplicate or absent model ID at models[{index}]")
        if not canonical or canonical.startswith("~"):
            raise IntegrityError(f"{model_id} has no stable canonical model slug")
        if endpoint.get("model_id") != model_id:
            raise IntegrityError(f"{model_id} endpoint model identity does not match")
        if provider_policy.get("only") != [provider_tag]:
            raise IntegrityError(f"{model_id} does not freeze exactly one endpoint tag")
        expected_data_policy = (
            "public_nonpersonal_contract_smoke_only"
            if execution_backend == "qwencloud_direct"
            else "deny"
        )
        expected_routing = {
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": expected_data_policy,
        }
        for key, expected in expected_routing.items():
            if provider_policy.get(key) != expected:
                raise IntegrityError(f"{model_id} routing policy {key} is not {expected!r}")
        if execution_backend == "qwencloud_direct":
            identity_kind = backend_contract.get("identity_kind")
            cost_reconciliation = backend_contract.get("cost_reconciliation")
            if (
                backend_contract.get("season_eligible") is not False
                or backend_contract.get("rank_eligible") is not False
                or backend_contract.get("official") is not False
                or backend_contract.get("data_policy") != "public_nonpersonal_contract_smoke_only"
                or identity_kind not in {"immutable_dated_release", "mutable_alias"}
                or (
                    identity_kind == "immutable_dated_release"
                    and (
                        cost_accounting_policy != "provider_usage_times_frozen_rate_card"
                        or cost_reconciliation != "provider_charge_unavailable"
                    )
                )
                or (
                    identity_kind == "mutable_alias"
                    and (
                        model_id != "qwen3.8-max"
                        or cost_accounting_policy != "provider_usage_with_unpriced_budget_ceiling"
                        or cost_reconciliation != "provider_rate_and_charge_unavailable"
                        or backend_contract.get("catalog_pinned_at_observation") is not True
                        or backend_contract.get("model_identity_label")
                        != "catalog_pinned_at_observation_not_a_frozen_model"
                        or backend_contract.get("mutable_alias_execution_requires_explicit_opt_in")
                        is not True
                    )
                )
            ):
                raise IntegrityError(
                    f"{model_id} QwenCloud route is not a fail-closed public smoke contract"
                )
        candidates.append(
            ContractCandidate(
                slot_id=slot_id,
                model_id=model_id,
                canonical_model_slug=canonical,
                model_name=str(model.get("name") or model_id),
                provider_tag=provider_tag,
                provider_name=provider_name,
                endpoint_sha256=_sha256(endpoint),
                endpoint_execution_sha256=endpoint_execution_contract_sha256(dict(endpoint)),
                endpoint=dict(endpoint),
                execution_backend=execution_backend,
                backend_contract=dict(backend_contract),
                backend_contract_sha256=backend_contract_sha256,
                route_selection=dict(execution_route),
                cost_accounting_policy=cost_accounting_policy,
            )
        )
        seen_slots.add(slot_id)
        seen_models.add(model_id)

    requested = tuple(dict.fromkeys(value.strip() for value in selectors if value.strip()))
    if not requested:
        return candidates
    by_selector: dict[str, ContractCandidate] = {}
    for candidate in candidates:
        by_selector[candidate.slot_id] = candidate
        by_selector[candidate.model_id] = candidate
        by_selector[candidate.canonical_model_slug] = candidate
    unknown = [selector for selector in requested if selector not in by_selector]
    if unknown:
        raise ContractRunnerError(f"selectors are absent from the frozen manifest: {unknown}")
    selected_ids = {by_selector[selector].model_id for selector in requested}
    return [candidate for candidate in candidates if candidate.model_id in selected_ids]


def _effective_rate(
    pricing: Mapping[str, Any],
    field: str,
    *,
    prompt_tokens_per_request: int,
    required: bool = False,
) -> Decimal:
    values: list[Decimal] = []
    if pricing.get(field) is not None:
        values.append(_decimal(pricing[field], field=f"endpoint pricing.{field}"))
    overrides = pricing.get("overrides") or []
    if not isinstance(overrides, list):
        raise IntegrityError("endpoint pricing.overrides must be a list")
    for index, override in enumerate(overrides):
        if not isinstance(override, Mapping) or override.get(field) is None:
            continue
        threshold = override.get("min_prompt_tokens")
        if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
            raise IntegrityError(
                f"endpoint pricing.overrides[{index}].min_prompt_tokens is invalid"
            )
        if prompt_tokens_per_request >= threshold:
            values.append(
                _decimal(
                    override[field],
                    field=f"endpoint pricing.overrides[{index}].{field}",
                )
            )
    if not values:
        if required:
            raise IntegrityError(f"endpoint has no price for required field {field}")
        return Decimal(0)
    return max(values)


def derive_contract_forecast(
    candidate: ContractCandidate,
    *,
    policy: ContractPolicy = DEFAULT_CONTRACT_POLICY,
    prompt: str = TOOL_CONTRACT_PROMPT,
) -> ContractForecast:
    """Derive endpoint price guards and a conservative one-run admission bound."""

    policy.validate()
    pricing = candidate.endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        raise IntegrityError(f"{candidate.model_id} endpoint has no pricing contract")
    base_prompt_tokens = Decimal(
        len(prompt.encode("utf-8")) + policy.approximate_non_user_prompt_bytes
    ) / Decimal(policy.conservative_bytes_per_token)
    tool_context_tokens = Decimal(policy.max_tool_result_bytes) / Decimal(
        policy.conservative_bytes_per_token
    )
    prompt_tokens = base_prompt_tokens + Decimal(policy.max_tool_rounds) * tool_context_tokens
    prompt_tokens_ceil = math.ceil(prompt_tokens)
    prompt_rate = _effective_rate(
        pricing,
        "prompt",
        prompt_tokens_per_request=prompt_tokens_ceil,
        required=True,
    )
    completion_rate = _effective_rate(
        pricing,
        "completion",
        prompt_tokens_per_request=prompt_tokens_ceil,
        required=True,
    )
    reasoning_rate = _effective_rate(
        pricing,
        "internal_reasoning",
        prompt_tokens_per_request=prompt_tokens_ceil,
    )
    request_rate = _effective_rate(
        pricing,
        "request",
        prompt_tokens_per_request=prompt_tokens_ceil,
    )
    actual_request_bound = policy.max_tool_rounds + 1
    # This intentionally mirrors live_smoke._worst_case_cost_usd today.  The
    # live smoke reserves one unaided request plus two complete tool loops even
    # in --contract-only mode.  Over-reserving is safe and makes the delegated
    # CLI accept exactly the same bound without changing its interface.
    admission_request_bound = 1 + actual_request_bound * 2
    total_prompt_tokens = Decimal(admission_request_bound) * prompt_tokens
    total_completion_tokens = Decimal(admission_request_bound * policy.max_output_tokens)
    forecast = (
        prompt_rate * total_prompt_tokens
        + completion_rate * total_completion_tokens
        + reasoning_rate * total_completion_tokens
        + request_rate * admission_request_bound
    )
    envelope = PriceEnvelope(
        prompt_usd_per_token=prompt_rate,
        completion_usd_per_token=completion_rate,
        reasoning_usd_per_token=reasoning_rate,
        request_usd=request_rate,
        prompt_usd_per_mtok=prompt_rate * Decimal(1_000_000),
        completion_usd_per_mtok=completion_rate * Decimal(1_000_000),
    )
    return ContractForecast(
        forecast_usd=forecast,
        price_envelope=envelope,
        prompt_tokens_per_request_bound=prompt_tokens,
        completion_tokens_per_request_bound=policy.max_output_tokens,
        actual_contract_request_bound=actual_request_bound,
        live_smoke_admission_request_bound=admission_request_bound,
    )


def _verify_live_artifact(path: Path) -> tuple[dict[str, Any], str]:
    _require_regular_file(path, label="live-smoke artifact")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IntegrityError(f"could not read live-smoke artifact {path}") from error
    if not isinstance(artifact, dict):
        raise IntegrityError(f"live-smoke artifact is not an object: {path}")
    digest = artifact.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise IntegrityError(f"live-smoke artifact has no complete SHA-256: {path}")
    unhashed = dict(artifact)
    unhashed.pop("artifact_sha256", None)
    if _live_smoke_sha256(unhashed) != digest:
        raise IntegrityError(f"live-smoke artifact content address is invalid: {path}")
    if not path.stem.endswith(digest[:12]):
        raise IntegrityError(f"live-smoke artifact filename does not match its digest: {path}")
    if artifact.get("schema_version") != LIVE_SMOKE_SCHEMA_VERSION:
        raise IntegrityError(f"unsupported live-smoke artifact schema: {path}")
    if artifact.get("official") is not False or artifact.get("rank_eligible") is not False:
        raise IntegrityError(
            f"frontier contract input unexpectedly claims rank eligibility: {path}"
        )
    journal_descriptor = artifact.get("run_journal")
    if journal_descriptor is not None:
        if not isinstance(journal_descriptor, Mapping):
            raise IntegrityError(f"live-smoke journal descriptor is invalid: {path}")
        try:
            journal_entries = verify_journal_descriptor(path.parent, journal_descriptor)
        except JournalIntegrityError as error:
            raise IntegrityError(f"live-smoke journal link is invalid: {path}") from error
        if journal_descriptor.get("run_id") != artifact.get("run_id"):
            raise IntegrityError(f"live-smoke artifact/journal run ID differs: {path}")
        journal_attempts = [
            entry.get("payload")
            for entry in journal_entries
            if entry.get("event_type") == "provider_attempt"
        ]
        journal_tools = [
            entry.get("payload")
            for entry in journal_entries
            if entry.get("event_type") == "mcp_trace"
        ]
        if journal_attempts != artifact.get("provider_attempt_events"):
            raise IntegrityError(f"live-smoke attempt journal differs from artifact: {path}")
        if journal_tools != artifact.get("mcp_trace_events"):
            raise IntegrityError(f"live-smoke MCP journal differs from artifact: {path}")
    return artifact, digest


def _artifact_contract_passed(artifact: Mapping[str, Any]) -> bool:
    results = artifact.get("results")
    errors = artifact.get("errors")
    if artifact.get("status") != "complete" or not isinstance(results, Mapping) or errors:
        return False
    contract = results.get("tool_contract")
    if not isinstance(contract, Mapping):
        return False
    traces = contract.get("tool_trace")
    return isinstance(traces, list) and any(
        isinstance(trace, Mapping)
        and trace.get("name") == "find_pairings"
        and trace.get("is_error") is False
        for trace in traces
    )


def _scan_cost_corrections(
    directory: str | Path | None,
    *,
    source_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, CostCorrection]:
    if directory is None:
        return {}
    root = Path(directory)
    if not root.exists():
        return {}
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError(f"cost-correction root must be a directory: {root}")
    corrections: dict[str, CostCorrection] = {}
    seen_digests: set[str] = set()
    for path in sorted(root.glob("*.json")):
        _require_regular_file(path, label="cost-correction artifact")
        try:
            correction = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise IntegrityError(f"could not read cost correction {path}") from error
        if not isinstance(correction, dict):
            raise IntegrityError(f"cost correction is not an object: {path}")
        digest = correction.get("artifact_sha256")
        unhashed = dict(correction)
        unhashed.pop("artifact_sha256", None)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or _live_smoke_sha256(unhashed) != digest
        ):
            raise IntegrityError(f"cost correction content address is invalid: {path}")
        if digest in seen_digests or not path.stem.endswith(digest[:12]):
            raise IntegrityError(f"duplicate or misnamed cost correction: {path}")
        seen_digests.add(digest)
        if correction.get("schema_version") != COST_CORRECTION_SCHEMA_VERSION:
            raise IntegrityError(f"unsupported cost-correction schema: {path}")
        if correction.get("record_type") != "superseding_cost_reconciliation":
            raise IntegrityError(f"unsupported cost-correction record type: {path}")
        if correction.get("rank_eligible") is not False:
            raise IntegrityError(f"cost correction unexpectedly claims rank eligibility: {path}")
        source = correction.get("source")
        cost = correction.get("cost")
        metadata = correction.get("generation_metadata")
        if not all(isinstance(value, Mapping) for value in (source, cost)) or not isinstance(
            metadata, list
        ):
            raise IntegrityError(f"cost correction lacks source/cost/metadata: {path}")
        source_digest = str(source.get("artifact_sha256") or "")
        source_artifact = source_artifacts.get(source_digest)
        if source_artifact is None:
            raise IntegrityError(f"cost correction refers to absent source {source_digest}: {path}")
        if source_digest in corrections:
            raise IntegrityError(f"more than one cost correction refers to {source_digest}")
        if source.get("run_id") != source_artifact.get("run_id"):
            raise IntegrityError(f"cost correction run identity mismatch: {path}")
        if source.get("requested_model_id") != source_artifact.get("requested_model_id"):
            raise IntegrityError(f"cost correction model identity mismatch: {path}")
        original = cost.get("original_recorded_cost_micros")
        additional = cost.get("additional_cost_micros")
        corrected = cost.get("corrected_total_cost_micros")
        for field, value in (
            ("original_recorded_cost_micros", original),
            ("additional_cost_micros", additional),
            ("corrected_total_cost_micros", corrected),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise IntegrityError(f"cost correction {field} is invalid: {path}")
        source_actual = (source_artifact.get("budget") or {}).get("actual_cost_micros")
        if original != source_actual or corrected != original + additional:
            raise IntegrityError(f"cost correction arithmetic/source cost mismatch: {path}")
        metadata_cost = 0
        generation_ids: set[str] = set()
        for index, generation in enumerate(metadata):
            if not isinstance(generation, Mapping):
                raise IntegrityError(f"cost correction metadata {index} is invalid: {path}")
            generation_id = str(generation.get("generation_id") or "")
            generation_cost = generation.get("cost_micros")
            if (
                not generation_id
                or generation_id in generation_ids
                or not isinstance(generation_cost, int)
                or isinstance(generation_cost, bool)
                or generation_cost < 0
            ):
                raise IntegrityError(f"cost correction generation {index} is invalid: {path}")
            generation_ids.add(generation_id)
            metadata_cost += generation_cost
        missing_ids = correction.get("missing_generation_ids")
        if (
            metadata_cost != additional
            or not isinstance(missing_ids, list)
            or set(missing_ids) != generation_ids
        ):
            raise IntegrityError(f"cost correction metadata total/IDs mismatch: {path}")
        corrections[source_digest] = CostCorrection(
            corrected_total_cost_micros=corrected,
            artifact_sha256=digest,
            generation_ids=frozenset(generation_ids),
            all_reconciled=correction.get("all_missing_generations_reconciled") is True,
        )
    return corrections


def _failed_artifact_costs_are_fully_accounted(
    artifact: Mapping[str, Any],
    correction: CostCorrection | None,
) -> bool:
    """Prove each attempted request was rejected pre-billing or reconciled by ID."""

    events = artifact.get("provider_attempt_events")
    if not isinstance(events, list):
        return False
    if not events:
        # The provider writes request_started synchronously before network I/O.
        # Therefore an empty durable event list, no generation evidence, and a
        # zero recorded cost prove the contract failed before any billable send.
        budget = artifact.get("budget") or {}
        results = artifact.get("results") or {}
        incomplete = artifact.get("incomplete_generation_metadata") or []
        return (
            isinstance(budget, Mapping)
            and budget.get("actual_cost_micros") == 0
            and isinstance(results, Mapping)
            and not results
            and isinstance(incomplete, list)
            and not incomplete
            and correction is None
        )
    attempts: dict[str, set[str]] = {}
    response_generations: dict[str, str] = {}
    for event in events:
        if not isinstance(event, Mapping):
            return False
        attempt_id = str(event.get("attempt_id") or "")
        event_type = str(event.get("event_type") or "")
        if not attempt_id or not event_type:
            return False
        # MCP lifecycle events share the durable attempt stream but are not
        # billable provider requests. Their cost is governed by the provider
        # generations that supplied or consumed the tool call, so they must
        # not be mistaken for accepted generations without accounting IDs.
        if event_type.startswith("mcp_"):
            continue
        attempts.setdefault(attempt_id, set()).add(event_type)
        if event_type == "response_received":
            generation_id = str(event.get("generation_id") or "")
            if not generation_id:
                return False
            response_generations[attempt_id] = generation_id
    accounted_generations: set[str] = set()
    metadata_groups: list[object] = [artifact.get("incomplete_generation_metadata") or []]
    results = artifact.get("results") or {}
    if not isinstance(results, Mapping):
        return False
    metadata_groups.extend(
        result.get("generation_metadata") or []
        for result in results.values()
        if isinstance(result, Mapping)
    )
    for group in metadata_groups:
        if not isinstance(group, list):
            return False
        for metadata in group:
            if not isinstance(metadata, Mapping) or metadata.get("reconciled") is not True:
                return False
            generation_id = str(metadata.get("generation_id") or "")
            if not generation_id:
                return False
            accounted_generations.add(generation_id)
    if correction is not None:
        if not correction.all_reconciled:
            return False
        accounted_generations.update(correction.generation_ids)
    safe_no_generation_terminals = {"pre_send_failure", "request_rejected"}
    unsafe_terminals = {"uncertain_delivery", "invalid_response"}
    for _attempt_id, event_types in attempts.items():
        if "request_started" not in event_types or event_types & unsafe_terminals:
            return False
        if "response_received" in event_types:
            if response_generations.get(attempt_id) not in accounted_generations:
                return False
        elif not event_types & safe_no_generation_terminals:
            return False
    return True


def _rate_card_artifact_result_is_accounted(
    result: Mapping[str, Any],
    accounting_basis: str = "frozen_rate_card_times_kimi_returned_usage",
) -> bool:
    metadata = result.get("generation_metadata")
    generation_ids = result.get("generation_ids")
    if (
        result.get("cost_reconciled") is not False
        or result.get("cost_accounting_basis") != accounting_basis
        or result.get("billing_reconciliation_status") != "provider_charge_unavailable"
        or not isinstance(metadata, list)
        or not isinstance(generation_ids, list)
        or not generation_ids
    ):
        return False
    seen_ids: set[str] = set()
    summed_cost = 0
    for item in metadata:
        if (
            not isinstance(item, Mapping)
            or item.get("reconciled") is not False
            or item.get("accounting_basis") != accounting_basis
            or item.get("billing_reconciliation_status") != "provider_charge_unavailable"
        ):
            return False
        generation_id = str(item.get("generation_id") or "")
        generation_cost = item.get("cost_micros")
        if (
            not generation_id
            or generation_id in seen_ids
            or not isinstance(generation_cost, int)
            or isinstance(generation_cost, bool)
            or generation_cost < 0
        ):
            return False
        seen_ids.add(generation_id)
        summed_cost += generation_cost
    return set(map(str, generation_ids)) == seen_ids and result.get("cost_micros") == summed_cost


def _rate_card_artifact_is_fully_accounted(artifact: Mapping[str, Any]) -> bool:
    """Accept usage-complete direct-provider records without calling them exact."""

    backend = str(artifact.get("execution_backend") or "")
    accounting_basis = RATE_CARD_ACCOUNTING_BASIS_BY_BACKEND.get(backend)

    requested_conditions = artifact.get("requested_conditions")
    if requested_conditions is None:
        expected_conditions = {"epicure_off", "epicure_on"}
    elif (
        not isinstance(requested_conditions, list)
        or not requested_conditions
        or len(set(map(str, requested_conditions))) != len(requested_conditions)
        or not set(map(str, requested_conditions)) <= {"epicure_off", "epicure_on"}
    ):
        return False
    else:
        expected_conditions = set(map(str, requested_conditions))
    if (
        artifact.get("status") != "complete_rate_card_estimated"
        or accounting_basis is None
        or artifact.get("errors") != {}
        or artifact.get("incomplete_generation_metadata") != []
    ):
        return False
    budget = artifact.get("budget")
    results = artifact.get("results")
    if (
        not isinstance(budget, Mapping)
        or budget.get("all_generation_costs_reconciled") is not False
        or budget.get("all_generation_usage_rate_card_accounted") is not True
        or budget.get("accounting_basis") != "provider_usage_times_frozen_rate_card"
        or not isinstance(results, Mapping)
        or set(results) != expected_conditions
    ):
        return False
    for result in results.values():
        if (
            not isinstance(result, Mapping)
            or result.get("cost_reconciled") is not False
            or result.get("cost_accounting_basis") != accounting_basis
            or result.get("billing_reconciliation_status") != "provider_charge_unavailable"
        ):
            return False
        metadata = result.get("generation_metadata")
        generation_ids = result.get("generation_ids")
        if not isinstance(metadata, list) or not isinstance(generation_ids, list):
            return False
        seen_ids: set[str] = set()
        summed_cost = 0
        for item in metadata:
            if (
                not isinstance(item, Mapping)
                or item.get("reconciled") is not False
                or item.get("accounting_basis") != accounting_basis
                or item.get("billing_reconciliation_status") != "provider_charge_unavailable"
            ):
                return False
            generation_id = str(item.get("generation_id") or "")
            generation_cost = item.get("cost_micros")
            if (
                not generation_id
                or generation_id in seen_ids
                or not isinstance(generation_cost, int)
                or isinstance(generation_cost, bool)
                or generation_cost < 0
            ):
                return False
            seen_ids.add(generation_id)
            summed_cost += generation_cost
        if set(map(str, generation_ids)) != seen_ids or result.get("cost_micros") != summed_cost:
            return False
    return True


def _unpriced_qwencloud_alias_is_fully_accounted(
    artifact: Mapping[str, Any],
) -> bool:
    """Verify usage while retaining the full ceiling for an unpriced alias."""

    budget = artifact.get("budget")
    results = artifact.get("results")
    contract = artifact.get("backend_contract")
    requested_conditions = artifact.get("requested_conditions")
    expected_conditions = (
        set(map(str, requested_conditions))
        if isinstance(requested_conditions, list)
        else {"epicure_off", "epicure_on"}
    )
    if (
        artifact.get("status") != "complete_unpriced_budget_ceiling"
        or artifact.get("execution_backend") != "qwencloud_direct"
        or artifact.get("requested_model_id") != "qwen3.8-max"
        or artifact.get("requested_provider") != "qwencloud-direct"
        or artifact.get("official") is not False
        or artifact.get("rank_eligible") is not False
        or artifact.get("mutable_alias_exploratory_opt_in") is not True
        or not isinstance(contract, Mapping)
        or contract.get("identity_kind") != "mutable_alias"
        or contract.get("catalog_pinned_at_observation") is not True
        or contract.get("model_identity_label")
        != "catalog_pinned_at_observation_not_a_frozen_model"
        or contract.get("official") is not False
        or contract.get("season_eligible") is not False
        or contract.get("rank_eligible") is not False
        or not isinstance(budget, Mapping)
        or budget.get("provider_rate_available") is not False
        or budget.get("provider_cost_known") is not False
        or budget.get("full_unpriced_budget_ceiling_retained") is not True
        or budget.get("all_generation_costs_reconciled") is not False
        or budget.get("all_generation_usage_accounted") is not True
        or budget.get("accounting_basis") != "provider_usage_with_unpriced_budget_ceiling"
        or not isinstance(results, Mapping)
        or set(results) != expected_conditions
        or not expected_conditions
        or not expected_conditions <= {"epicure_off", "epicure_on"}
    ):
        return False
    try:
        cap = _decimal(budget.get("cap_usd"), field="unpriced QwenCloud cap")
        forecast = _decimal(
            budget.get("forecast_worst_case_usd"),
            field="unpriced QwenCloud forecast",
        )
    except IntegrityError:
        return False
    if cap <= 0 or forecast != cap or budget.get("actual_cost_micros") != 0:
        return False
    for result in results.values():
        if not isinstance(result, Mapping):
            return False
        metadata = result.get("generation_metadata")
        generation_ids = result.get("generation_ids")
        if (
            result.get("cost_reconciled") is not False
            or result.get("cost_micros") != 0
            or result.get("cost_accounting_basis") != QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS
            or result.get("billing_reconciliation_status") != QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS
            or not isinstance(metadata, list)
            or not isinstance(generation_ids, list)
            or not generation_ids
        ):
            return False
        seen: set[str] = set()
        for generation in metadata:
            if (
                not isinstance(generation, Mapping)
                or generation.get("reconciled") is not False
                or generation.get("cost_micros") != 0
                or generation.get("provider_cost_known") is not False
                or generation.get("accounting_basis") != QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS
                or generation.get("billing_reconciliation_status")
                != QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS
            ):
                return False
            generation_id = str(generation.get("generation_id") or "")
            if not generation_id or generation_id in seen:
                return False
            seen.add(generation_id)
        if set(map(str, generation_ids)) != seen:
            return False
    return True


def _failed_rate_card_artifact_is_fully_accounted(
    artifact: Mapping[str, Any],
) -> bool:
    """Prove a failed direct-provider artifact has usage for every paid response.

    The direct APIs expose generation IDs and token usage but not a charged-cost
    lookup. A failed arm therefore cannot be called provider-reconciled. It can,
    however, be proved complete for rate-card accounting when every delivered
    response has one strictly typed usage record. The scanner retains the full
    admitted allowance for such an artifact, so this classification can never
    release budget on the strength of an estimate.
    """

    budget = artifact.get("budget")
    results = artifact.get("results")
    errors = artifact.get("errors")
    incomplete = artifact.get("incomplete_generation_metadata")
    events = artifact.get("provider_attempt_events")
    backend = str(artifact.get("execution_backend") or "")
    accounting_basis = RATE_CARD_ACCOUNTING_BASIS_BY_BACKEND.get(backend)
    provider_slug = RATE_CARD_PROVIDER_SLUG_BY_BACKEND.get(backend)
    if (
        artifact.get("status") != "failed_or_unreconciled"
        or accounting_basis is None
        or provider_slug is None
        or artifact.get("requested_provider") != provider_slug
        or not isinstance(budget, Mapping)
        or budget.get("all_generation_costs_reconciled") is not False
        or budget.get("accounting_basis") != "provider_usage_times_frozen_rate_card"
        or budget.get("provider_charge_available") is not False
        or not isinstance(results, Mapping)
        or not isinstance(errors, Mapping)
        or not errors
        or not isinstance(incomplete, list)
        or not isinstance(events, list)
    ):
        return False

    attempts: dict[str, set[str]] = {}
    response_generations: dict[str, str] = {}
    for event in events:
        if not isinstance(event, Mapping):
            return False
        event_type = str(event.get("event_type") or "")
        if event_type.startswith("mcp_"):
            continue
        attempt_id = str(event.get("attempt_id") or "")
        if not attempt_id or not event_type:
            return False
        attempts.setdefault(attempt_id, set()).add(event_type)
        if event_type == "response_received":
            generation_id = str(event.get("generation_id") or "")
            if not generation_id or attempt_id in response_generations:
                return False
            response_generations[attempt_id] = generation_id

    safe_no_generation_terminals = {"pre_send_failure", "request_rejected"}
    unsafe_terminals = {"uncertain_delivery", "invalid_response"}
    for _attempt_id, event_types in attempts.items():
        if "request_started" not in event_types or event_types & unsafe_terminals:
            return False
        if "response_received" not in event_types and not (
            event_types & safe_no_generation_terminals
        ):
            return False

    metadata_by_id: dict[str, Mapping[str, Any]] = {}
    summed_cost_micros = 0

    def record_metadata(item: object) -> bool:
        nonlocal summed_cost_micros
        if (
            not isinstance(item, Mapping)
            or item.get("reconciled") is not False
            or item.get("accounting_basis") != accounting_basis
            or item.get("billing_reconciliation_status") != "provider_charge_unavailable"
        ):
            return False
        generation_id = str(item.get("generation_id") or "")
        generation_cost = item.get("cost_micros")
        prompt_tokens = item.get("tokens_prompt")
        completion_tokens = item.get("tokens_completion")
        if (
            not generation_id
            or generation_id in metadata_by_id
            or not isinstance(generation_cost, int)
            or isinstance(generation_cost, bool)
            or generation_cost < 0
            or not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens < 0
            or not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            return False
        metadata_by_id[generation_id] = item
        summed_cost_micros += generation_cost
        return True

    for result in results.values():
        if not isinstance(result, dict) or not _rate_card_artifact_result_is_accounted(
            result, accounting_basis
        ):
            return False
        metadata = result.get("generation_metadata")
        assert isinstance(metadata, list)
        if any(not record_metadata(item) for item in metadata):
            return False
    if any(not record_metadata(item) for item in incomplete):
        return False

    actual_cost_micros = budget.get("actual_cost_micros")
    return (
        bool(metadata_by_id)
        and set(response_generations.values()) == set(metadata_by_id)
        and isinstance(actual_cost_micros, int)
        and not isinstance(actual_cost_micros, bool)
        and actual_cost_micros == summed_cost_micros
    )


def scan_live_smoke_artifacts(
    directory: str | Path,
    *,
    corrections_directory: str | Path | None = None,
) -> ArtifactScan:
    """Verify and conservatively account every immutable live-smoke artifact."""

    root = Path(directory)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise IntegrityError(f"live-smoke artifact root must be a directory: {root}")
    raw_artifacts: list[tuple[Path, dict[str, Any], str]] = []
    seen_digests: set[str] = set()
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        artifact, digest = _verify_live_artifact(path)
        if digest in seen_digests:
            raise IntegrityError(f"duplicate live-smoke artifact digest: {digest}")
        seen_digests.add(digest)
        raw_artifacts.append((path, artifact, digest))
    corrections = _scan_cost_corrections(
        corrections_directory,
        source_artifacts={digest: artifact for _, artifact, digest in raw_artifacts},
    )
    artifacts: list[ArtifactExposure] = []
    for path, artifact, digest in raw_artifacts:
        budget = artifact.get("budget")
        results = artifact.get("results")
        if not isinstance(budget, Mapping) or not isinstance(results, Mapping):
            raise IntegrityError(f"live-smoke artifact lacks budget/results objects: {path}")
        actual_micros = budget.get("actual_cost_micros")
        if (
            not isinstance(actual_micros, int)
            or isinstance(actual_micros, bool)
            or actual_micros < 0
        ):
            raise IntegrityError(f"actual_cost_micros is invalid: {path}")
        result_micros = 0
        for condition, result in results.items():
            if not isinstance(result, Mapping):
                raise IntegrityError(f"result {condition!r} is not an object: {path}")
            result_cost = result.get("cost_micros") or 0
            if not isinstance(result_cost, int) or isinstance(result_cost, bool) or result_cost < 0:
                raise IntegrityError(f"result cost_micros is invalid: {path}")
            result_micros += result_cost
        incomplete_micros = 0
        incomplete_metadata = artifact.get("incomplete_generation_metadata") or []
        if not isinstance(incomplete_metadata, list):
            raise IntegrityError(f"incomplete_generation_metadata is not a list: {path}")
        for index, generation in enumerate(incomplete_metadata):
            if not isinstance(generation, Mapping):
                raise IntegrityError(
                    f"incomplete generation metadata {index} is not an object: {path}"
                )
            generation_cost = generation.get("cost_micros") or 0
            if (
                not isinstance(generation_cost, int)
                or isinstance(generation_cost, bool)
                or generation_cost < 0
            ):
                raise IntegrityError(
                    f"incomplete generation cost_micros is invalid at {index}: {path}"
                )
            incomplete_micros += generation_cost
        if result_micros + incomplete_micros != actual_micros:
            raise IntegrityError(
                "artifact budget cost does not equal result plus incomplete-generation "
                f"costs: {path}"
            )
        corrected_cost = corrections.get(digest)
        effective_actual_micros = (
            corrected_cost.corrected_total_cost_micros if corrected_cost else actual_micros
        )
        actual = Decimal(effective_actual_micros) / Decimal(1_000_000)
        forecast = _decimal(budget.get("forecast_worst_case_usd"), field=f"{path.name} forecast")
        admitted_cap = _decimal(budget.get("cap_usd"), field=f"{path.name} cap")
        errors = artifact.get("errors")
        reconciled_complete = (
            artifact.get("status") == "complete"
            and budget.get("all_generation_costs_reconciled") is True
            and isinstance(errors, Mapping)
            and not errors
        )
        if reconciled_complete:
            exposure = actual
            basis = "fully_reconciled_actual"
        elif _rate_card_artifact_is_fully_accounted(artifact):
            # Provider usage proves the estimate inputs, but Kimi exposes no
            # per-generation charged amount. Retain the entire pre-admitted
            # forecast as budget exposure and keep the estimate out of exact
            # cost rankings.
            exposure = max(actual, forecast, admitted_cap)
            basis = "complete_rate_card_estimated_full_forecast_reserve"
        elif _unpriced_qwencloud_alias_is_fully_accounted(artifact):
            # A zero cost field is explicitly unknown, not free.  Preserve the
            # complete allowance while allowing the real response/tool trace
            # to be retained as permanently exploratory evidence.
            exposure = max(forecast, admitted_cap)
            basis = "complete_unpriced_full_budget_ceiling_reserve"
        elif corrected_cost is None and _failed_rate_card_artifact_is_fully_accounted(artifact):
            # The response failed benchmark normalization, but every delivered
            # direct-Kimi generation has an ID and returned usage. Preserve the
            # failure for reliability and retain the full allowance because the
            # provider exposes no charged-cost lookup.
            exposure = max(actual, forecast, admitted_cap)
            basis = "failed_rate_card_estimated_full_forecast_reserve"
        elif _failed_artifact_costs_are_fully_accounted(artifact, corrected_cost):
            exposure = actual
            basis = "failed_but_all_attempts_cost_reconciled_actual"
        else:
            # A failed request can still have reached a billable provider.  The
            # complete admitted allowance remains charged until reconciled.
            exposure = max(actual, forecast, admitted_cap)
            basis = "failed_or_unreconciled_full_admitted_allowance"
        manifest_digest = artifact.get("candidate_manifest_sha256")
        if manifest_digest is not None and (
            not isinstance(manifest_digest, str) or len(manifest_digest) != 64
        ):
            raise IntegrityError(f"candidate manifest digest is malformed: {path}")
        artifacts.append(
            ArtifactExposure(
                path=path,
                artifact_sha256=digest,
                status=str(artifact.get("status") or ""),
                requested_model_id=str(artifact.get("requested_model_id") or ""),
                requested_provider=str(artifact.get("requested_provider") or ""),
                candidate_manifest_sha256=manifest_digest,
                actual_cost_usd=actual,
                forecast_usd=forecast,
                admitted_cap_usd=admitted_cap,
                exposure_usd=exposure,
                exposure_basis=basis,
                contract_passed=_artifact_contract_passed(artifact),
                cost_correction_sha256=(corrected_cost.artifact_sha256 if corrected_cost else None),
            )
        )
    actual_total = sum((item.actual_cost_usd for item in artifacts), Decimal(0))
    exposure_total = sum((item.exposure_usd for item in artifacts), Decimal(0))
    uncertain = sum(
        (
            item.exposure_usd
            for item in artifacts
            if item.exposure_basis == "failed_or_unreconciled_full_admitted_allowance"
        ),
        Decimal(0),
    )
    return ArtifactScan(
        tuple(artifacts),
        actual_total,
        exposure_total,
        uncertain,
        correction_count=len(corrections),
    )


def write_no_artifact_reconciliation(
    record: Mapping[str, Any], output_directory: str | Path
) -> Path:
    """Write a secret-free, content-addressed external accounting record.

    This function performs no network or provider operation. The caller must
    supply already captured evidence; link validation occurs before a ledger
    reservation can be released.
    """

    payload = dict(record)
    if "content_address" in payload:
        raise IntegrityError("reconciliation input must not supply a content address")
    if payload.get("schema_version") not in {
        NO_ARTIFACT_RECONCILIATION_SCHEMA_VERSION,
        NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION,
    }:
        raise IntegrityError("unsupported no-artifact reconciliation schema")
    if _contains_forbidden_key(
        payload,
        {
            "api_key",
            "authorization",
            "cloudflare_ai_gateway_token",
            "command",
            "environment",
            "headers",
            "mcp_token",
            "openrouter_api_key",
            "password",
            "secret",
        },
    ):
        raise IntegrityError("reconciliation record contains a forbidden field")
    digest = _sha256(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"no-artifact-reconciliation-{digest}.json"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise IntegrityError(f"refusing to overwrite conflicting reconciliation: {destination}")
        return destination
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=root,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    destination.chmod(0o644)
    directory_descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination


def _load_no_artifact_reconciliation(path: Path) -> tuple[dict[str, Any], str]:
    _require_regular_file(path, label="no-artifact reconciliation")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IntegrityError(f"could not read no-artifact reconciliation {path}") from error
    if not isinstance(record, dict):
        raise IntegrityError(f"no-artifact reconciliation is not an object: {path}")
    if record.get("schema_version") not in {
        NO_ARTIFACT_RECONCILIATION_SCHEMA_VERSION,
        NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION,
    }:
        raise IntegrityError(f"unsupported no-artifact reconciliation schema: {path}")
    address = record.get("content_address")
    if not isinstance(address, Mapping):
        raise IntegrityError(f"reconciliation has no content address: {path}")
    digest = _require_sha256(address.get("digest"), field="reconciliation digest")
    unhashed = dict(record)
    unhashed.pop("content_address", None)
    if (
        address.get("algorithm") != "sha256"
        or address.get("uri") != f"sha256:{digest}"
        or _sha256(unhashed) != digest
        or path.name != f"no-artifact-reconciliation-{digest}.json"
    ):
        raise IntegrityError(f"reconciliation content address is invalid: {path}")
    if _contains_forbidden_key(
        record,
        {
            "api_key",
            "authorization",
            "cloudflare_ai_gateway_token",
            "command",
            "environment",
            "headers",
            "mcp_token",
            "openrouter_api_key",
            "password",
            "secret",
        },
    ):
        raise IntegrityError(f"reconciliation contains a forbidden field: {path}")
    if (
        record.get("official") is not False
        or record.get("rank_eligible") is not False
        or record.get("provider_calls_made") is not False
    ):
        raise IntegrityError(f"reconciliation makes an invalid execution claim: {path}")
    return record, digest


def validate_no_artifact_reconciliation_v2(
    path: str | Path,
    *,
    ledger_entries: Sequence[Mapping[str, Any]],
) -> NoArtifactReconciliationV2:
    """Validate an immutable proof that a reserved work item was never delivered.

    V2 is intentionally narrower than an account-delta inference.  It applies
    only when the frozen executor's durable boundaries prove zero starts,
    journals, provider requests, sources, generations, and MCP events for the
    target work item.  The producing recovery protocol is responsible for
    re-reading and hashing those local sources before this proof is written.
    """

    record_path = Path(path)
    record, digest = _load_no_artifact_reconciliation(record_path)
    if record.get("schema_version") != NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION:
        raise IntegrityError("unexpected no-delivery reconciliation schema")
    reservation_ref = record.get("reservation")
    execution = record.get("no_delivery_evidence")
    conclusion = record.get("conclusion")
    if not all(isinstance(value, Mapping) for value in (reservation_ref, execution, conclusion)):
        raise IntegrityError("V2 reconciliation lacks reservation or no-delivery evidence")
    reservation_id = _require_sha256(
        reservation_ref.get("ledger_entry_sha256"), field="V2 reservation reference"
    )
    reservations = {
        str(entry.get("entry_sha256") or ""): entry
        for entry in ledger_entries
        if entry.get("event_type") == "reservation_created"
    }
    reservation = reservations.get(reservation_id)
    if reservation is None:
        raise IntegrityError("V2 reconciliation reservation is absent")
    identity = {
        "runner_run_id": reservation.get("runner_run_id"),
        "model_id": reservation.get("model_id"),
        "provider_tag": reservation.get("provider_tag"),
        "manifest_sha256": reservation.get("manifest_sha256"),
        "study_plan_sha256": reservation.get("study_plan_sha256"),
        "admission_block_id": reservation.get("admission_block_id"),
        "work_item_id": reservation.get("work_item_id"),
        "reserved_usd": reservation.get("reserved_usd"),
    }
    if any(reservation_ref.get(field) != value for field, value in identity.items()):
        raise IntegrityError("V2 reconciliation reservation identity differs")
    exact_zero_counts = {
        "item_execution_started_events": 0,
        "provider_request_journals": 0,
        "provider_request_started_events": 0,
        "provider_response_received_events": 0,
        "source_artifacts": 0,
        "generation_ids": [],
        "mcp_trace_events": 0,
        "canonical_finalizations_before_reconciliation": 0,
    }
    if any(execution.get(key) != value for key, value in exact_zero_counts.items()):
        raise IntegrityError("V2 reconciliation does not prove every no-delivery boundary")
    snapshot = execution.get("evidence_snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("v8_tree_unchanged") is not True
        or snapshot.get("canonical_source_inventory_verified") is not True
        or snapshot.get("journal_inventory_verified") is not True
        or not isinstance(snapshot.get("v8_tree_snapshot_sha256"), str)
        or len(str(snapshot["v8_tree_snapshot_sha256"])) != 64
        or not isinstance(snapshot.get("target_identity_scan_sha256"), str)
        or len(str(snapshot["target_identity_scan_sha256"])) != 64
    ):
        raise IntegrityError("V2 reconciliation evidence snapshot is incomplete")
    expected_conclusion = {
        "delivery_attempted": False,
        "provider_generation_request_reached": False,
        "provider_generation_cost_usd": "0",
        "epicure_called": False,
        "reservation_release_authorized": True,
        "same_identifier_replay_permitted": False,
        "disposition": "release_never_started_no_delivery_reservation",
    }
    if dict(conclusion) != expected_conclusion:
        raise IntegrityError("V2 reconciliation conclusion differs")
    if (
        record.get("official") is not False
        or record.get("rank_eligible") is not False
        or record.get("provider_calls_made") is not False
        or record.get("epicure_calls_made") is not False
    ):
        raise IntegrityError("V2 reconciliation makes an invalid execution claim")
    related = [
        entry
        for entry in ledger_entries
        if entry.get("event_type") in {"artifact_recorded", "no_artifact_reconciliation_recorded"}
        and entry.get("reservation_entry_sha256") == reservation_id
    ]
    v2_events = [
        entry
        for entry in related
        if entry.get("event_type") == "no_artifact_reconciliation_recorded"
        and entry.get("reconciliation_sha256") == digest
    ]
    if related and (len(related) != 1 or len(v2_events) != 1):
        raise IntegrityError("V2 reservation was finalized by a different or duplicate event")
    return NoArtifactReconciliationV2(
        path=record_path,
        artifact_sha256=digest,
        reservation_entry_sha256=reservation_id,
        study_plan_sha256=str(identity["study_plan_sha256"] or ""),
        admission_block_id=str(identity["admission_block_id"] or ""),
        work_item_id=str(identity["work_item_id"] or ""),
    )


def validate_no_artifact_reconciliation(
    path: str | Path,
    *,
    ledger_entries: Sequence[Mapping[str, Any]],
    artifact_scan: ArtifactScan,
) -> NoArtifactReconciliation:
    """Prove one no-artifact reservation had no unaccounted generation cost."""

    record_path = Path(path)
    record, digest = _load_no_artifact_reconciliation(record_path)
    if record.get("schema_version") != NO_ARTIFACT_RECONCILIATION_SCHEMA_VERSION:
        raise IntegrityError("unexpected V1 no-artifact reconciliation schema")
    reservation_ref = record.get("reservation")
    incident_ref = record.get("incident")
    stdout_match = record.get("stdout_match")
    account = record.get("account_reconciliation")
    if not all(
        isinstance(value, Mapping)
        for value in (reservation_ref, incident_ref, stdout_match, account)
    ):
        raise IntegrityError("reconciliation lacks reservation/incident/account evidence")

    by_digest = {str(entry.get("entry_sha256") or ""): entry for entry in ledger_entries}
    reservation_id = _require_sha256(
        reservation_ref.get("ledger_entry_sha256"), field="reservation reference"
    )
    incident_id = _require_sha256(
        incident_ref.get("ledger_entry_sha256"), field="incident reference"
    )
    reservation = by_digest.get(reservation_id)
    incident = by_digest.get(incident_id)
    if reservation is None or reservation.get("event_type") != "reservation_created":
        raise IntegrityError("reconciliation reservation reference is not a reservation")
    if incident is None or incident.get("event_type") != "execution_incident":
        raise IntegrityError("reconciliation incident reference is not an incident")
    if (
        incident.get("reservation_entry_sha256") != reservation_id
        or incident.get("incident") != "no_verifiable_artifact_reservation_retained"
        or incident.get("runner_run_id") != reservation.get("runner_run_id")
    ):
        raise IntegrityError("reconciliation incident does not bind the reservation")
    identity = {
        "runner_run_id": reservation.get("runner_run_id"),
        "model_id": reservation.get("model_id"),
        "provider_tag": reservation.get("provider_tag"),
        "manifest_sha256": reservation.get("manifest_sha256"),
    }
    if any(reservation_ref.get(field) != value for field, value in identity.items()):
        raise IntegrityError("reconciliation reservation identity differs from the ledger")
    if (
        incident_ref.get("recorded_at") != incident.get("recorded_at")
        or incident_ref.get("subprocess_returncode") != incident.get("subprocess_returncode")
        or incident_ref.get("stdout_sha256") != incident.get("stdout_sha256")
        or incident_ref.get("stderr_sha256") != incident.get("stderr_sha256")
    ):
        raise IntegrityError("reconciliation incident evidence differs from the ledger")
    stdout_sha = _require_sha256(incident.get("stdout_sha256"), field="incident stdout digest")
    expected_match = PRE_GENERATION_STDOUT_MATCHES.get(stdout_sha)
    if (
        expected_match is None
        or incident.get("subprocess_returncode") != 1
        or incident.get("stderr_sha256") != EMPTY_SHA256
        or stdout_match.get("safe_stdout_sha256") != stdout_sha
        or stdout_match.get("matched_error_message")
        != "RuntimeError: configured max output exceeds the endpoint completion limit"
        or stdout_match.get("failure_boundary") != expected_match["failure_boundary"]
        or stdout_match.get("source_symbol") != expected_match["source_symbol"]
    ):
        raise IntegrityError("incident stdout is not an allow-listed pre-generation failure")

    before = account.get("before")
    after = account.get("after")
    delta = account.get("delta")
    pending = account.get("known_pending_cost")
    if not all(isinstance(value, Mapping) for value in (before, after, delta, pending)):
        raise IntegrityError("reconciliation account evidence is incomplete")
    if (
        account.get("provider") != "openrouter"
        or account.get("endpoint") != "/api/v1/key"
        or account.get("currency") != "USD"
        or account.get("unexplained_delta_usd") != "0"
    ):
        raise IntegrityError("reconciliation account scope or conclusion is invalid")

    source_digest = _require_sha256(
        pending.get("source_artifact_sha256"), field="pending source artifact"
    )
    if before.get("source_artifact_sha256") != source_digest:
        raise IntegrityError("before snapshot and pending cost cite different artifacts")
    exposures = {item.artifact_sha256: item for item in artifact_scan.artifacts}
    source_exposure = exposures.get(source_digest)
    if source_exposure is None:
        raise IntegrityError("pending-cost source artifact is absent")
    source_artifact, verified_source_digest = _verify_live_artifact(source_exposure.path)
    source_budget = source_artifact.get("budget")
    if (
        verified_source_digest != source_digest
        or source_artifact.get("status") != "complete"
        or not isinstance(source_budget, Mapping)
        or source_budget.get("all_generation_costs_reconciled") is not True
        or source_exposure.exposure_basis != "fully_reconciled_actual"
    ):
        raise IntegrityError("pending-cost source is not a fully reconciled artifact")
    source_cost_micros = source_budget.get("actual_cost_micros")
    if (
        not isinstance(source_cost_micros, int)
        or isinstance(source_cost_micros, bool)
        or source_cost_micros <= 0
        or pending.get("source_artifact_cost_micros") != source_cost_micros
    ):
        raise IntegrityError("pending source artifact cost is invalid")
    source_events = [
        entry
        for entry in ledger_entries
        if entry.get("event_type") == "artifact_recorded"
        and entry.get("artifact_sha256") == source_digest
    ]
    if len(source_events) != 1 or source_events[0].get("entry_sha256") != reservation.get(
        "previous_entry_sha256"
    ):
        raise IntegrityError("pending source was not immediately before the reservation")

    artifact_after = source_budget.get("openrouter_key_after")
    if not isinstance(artifact_after, Mapping):
        raise IntegrityError("pending source has no OpenRouter postflight snapshot")
    for metric in ("usage_daily_usd", "usage_monthly_usd"):
        if before.get(metric) != artifact_after.get(metric):
            raise IntegrityError(f"before account snapshot {metric} differs from its artifact")
    if before.get("source_field") != "budget.openrouter_key_after":
        raise IntegrityError("before snapshot source field is not explicit")

    source_completed = _timestamp(
        source_artifact.get("completed_at"), field="source artifact completed_at"
    )
    reservation_time = _timestamp(reservation.get("recorded_at"), field="reservation time")
    incident_time = _timestamp(incident.get("recorded_at"), field="incident time")
    before_upper = _timestamp(
        before.get("captured_no_later_than"), field="before capture upper bound"
    )
    after_started = _timestamp(after.get("request_started_at"), field="after capture request time")
    after_received = _timestamp(
        after.get("response_received_at"), field="after capture response time"
    )
    if not (
        before_upper == source_completed
        and source_completed <= reservation_time <= incident_time <= after_started
        and after_started <= after_received <= incident_time + timedelta(minutes=15)
    ):
        raise IntegrityError("reconciliation timestamps do not bracket the incident")
    for field in (
        "command_sha256",
        "command_stdout_sha256",
        "capture_request_record_sha256",
        "capture_response_record_sha256",
    ):
        _require_sha256(after.get(field), field=f"after capture {field}")
    if not isinstance(after.get("source_session_id"), str) or not after.get("source_session_id"):
        raise IntegrityError("after snapshot has no external capture session reference")

    daily_delta = _decimal(after.get("usage_daily_usd"), field="after daily usage") - _decimal(
        before.get("usage_daily_usd"), field="before daily usage"
    )
    monthly_delta = _decimal(
        after.get("usage_monthly_usd"), field="after monthly usage"
    ) - _decimal(before.get("usage_monthly_usd"), field="before monthly usage")
    if daily_delta < 0 or monthly_delta < 0 or daily_delta != monthly_delta:
        raise IntegrityError("OpenRouter daily/monthly usage deltas do not agree")
    if (
        _decimal(delta.get("usage_daily_usd"), field="recorded daily delta") != daily_delta
        or _decimal(delta.get("usage_monthly_usd"), field="recorded monthly delta") != monthly_delta
    ):
        raise IntegrityError("recorded account deltas are incorrect")
    rounded_delta_micros = int(
        (daily_delta * Decimal(1_000_000)).to_integral_value(rounding=ROUND_HALF_UP)
    )
    if (
        rounded_delta_micros != source_cost_micros
        or pending.get("matching_rule")
        != "account_delta_round_half_up_to_micros_equals_source_artifact_cost"
    ):
        raise IntegrityError("account delta is not fully explained by the pending source cost")
    conclusion = record.get("conclusion")
    if not isinstance(conclusion, Mapping) or conclusion != {
        "provider_generation_request_reached": False,
        "provider_generation_cost_usd": "0",
        "reservation_release_authorized": True,
        "basis": (
            "allow-listed pre-generation stdout plus account delta fully explained by the "
            "immediately preceding reconciled artifact"
        ),
    }:
        raise IntegrityError("reconciliation conclusion is not the governed no-cost conclusion")
    return NoArtifactReconciliation(
        path=record_path,
        artifact_sha256=digest,
        reservation_entry_sha256=reservation_id,
        incident_entry_sha256=incident_id,
        known_pending_artifact_sha256=source_digest,
        account_usage_delta_usd=daily_delta,
    )


def _ledger_entry_digest(entry: Mapping[str, Any]) -> str:
    unhashed = dict(entry)
    unhashed.pop("entry_sha256", None)
    return _sha256(unhashed)


def _contains_forbidden_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


def load_ledger(path: str | Path) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    _require_regular_file(ledger_path, label="frontier contract ledger")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, raw_line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise IntegrityError(f"blank line in ledger at line {line_number}")
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise IntegrityError(f"invalid ledger JSON at line {line_number}") from error
        if not isinstance(entry, dict):
            raise IntegrityError(f"ledger line {line_number} is not an object")
        if entry.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise IntegrityError(f"unsupported ledger schema at line {line_number}")
        if entry.get("sequence") != line_number:
            raise IntegrityError(f"ledger sequence mismatch at line {line_number}")
        if entry.get("previous_entry_sha256") != previous:
            raise IntegrityError(f"ledger hash chain mismatch at line {line_number}")
        digest = entry.get("entry_sha256")
        if not isinstance(digest, str) or digest != _ledger_entry_digest(entry):
            raise IntegrityError(f"ledger entry digest mismatch at line {line_number}")
        entries.append(entry)
        previous = digest
    return entries


def append_ledger_event(
    path: str | Path,
    event: Mapping[str, Any],
    *,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Append one hash-chained allow-listed event and fsync it before return."""

    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_ledger(ledger_path)
    forbidden = {
        "api_key",
        "authorization",
        "cloudflare_ai_gateway_token",
        "mcp_token",
        "stdout",
        "stderr",
        "environment",
    }
    if _contains_forbidden_key(event, forbidden):
        raise IntegrityError("ledger event includes a forbidden secret-bearing field")
    protected = {
        "schema_version",
        "sequence",
        "recorded_at",
        "previous_entry_sha256",
        "entry_sha256",
    }
    if protected.intersection(event):
        raise IntegrityError("ledger event attempts to override hash-chain fields")
    entry = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sequence": len(entries) + 1,
        "recorded_at": recorded_at or _utc_now(),
        "previous_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        **dict(event),
    }
    if entry.get("event_type") not in {
        "reservation_created",
        "artifact_recorded",
        "execution_incident",
        "no_artifact_reconciliation_recorded",
    }:
        raise IntegrityError("unsupported frontier contract ledger event type")
    entry["entry_sha256"] = _ledger_entry_digest(entry)
    line = _canonical(entry) + b"\n"
    descriptor = os.open(ledger_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError("short append while writing frontier contract ledger")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return entry


def active_ledger_reservations(entries: Sequence[Mapping[str, Any]]) -> dict[str, Decimal]:
    reservations: dict[str, Decimal] = {}
    finalized: set[str] = set()
    for entry in entries:
        event_type = entry.get("event_type")
        if event_type == "reservation_created":
            reservation_id = str(entry.get("entry_sha256") or "")
            if reservation_id in reservations:
                raise IntegrityError(f"duplicate reservation entry: {reservation_id}")
            reservations[reservation_id] = _decimal(
                entry.get("reserved_usd"), field=f"reservation {reservation_id}"
            )
        elif event_type in {
            "artifact_recorded",
            "no_artifact_reconciliation_recorded",
        }:
            reservation_id = str(entry.get("reservation_entry_sha256") or "")
            if reservation_id not in reservations:
                raise IntegrityError(
                    f"artifact ledger event refers to unknown reservation {reservation_id}"
                )
            if reservation_id in finalized:
                raise IntegrityError(f"reservation was finalized twice: {reservation_id}")
            finalized.add(reservation_id)
    return {
        reservation_id: amount
        for reservation_id, amount in reservations.items()
        if reservation_id not in finalized
    }


def validate_ledger_artifact_links(
    entries: Sequence[Mapping[str, Any]],
    artifact_scan: ArtifactScan,
    *,
    reconciliation_directory: str | Path | None = None,
) -> None:
    """Refuse reservation releases without immutable generation or no-cost proof."""

    reservations = {
        str(entry.get("entry_sha256") or ""): entry
        for entry in entries
        if entry.get("event_type") == "reservation_created"
    }
    artifacts = {item.artifact_sha256: item for item in artifact_scan.artifacts}
    recorded_artifacts: set[str] = set()
    for entry in entries:
        if entry.get("event_type") != "artifact_recorded":
            continue
        reservation_id = str(entry.get("reservation_entry_sha256") or "")
        reservation = reservations.get(reservation_id)
        if reservation is None:
            raise IntegrityError(f"artifact record has no reservation: {reservation_id}")
        artifact_digest = str(entry.get("artifact_sha256") or "")
        artifact = artifacts.get(artifact_digest)
        if artifact is None:
            raise IntegrityError(
                f"ledger releases reservation {reservation_id} but artifact is absent"
            )
        if artifact_digest in recorded_artifacts:
            raise IntegrityError(f"artifact was recorded more than once: {artifact_digest}")
        recorded_artifacts.add(artifact_digest)
        if entry.get("artifact_filename") != artifact.path.name:
            raise IntegrityError(f"ledger artifact filename mismatch: {artifact_digest}")
        identity_fields = {
            "model_id": artifact.requested_model_id,
            "provider_tag": artifact.requested_provider,
            "manifest_sha256": artifact.candidate_manifest_sha256,
        }
        for field, artifact_value in identity_fields.items():
            if entry.get(field) != artifact_value or reservation.get(field) != artifact_value:
                raise IntegrityError(
                    f"ledger {field} does not match artifact/reservation {artifact_digest}"
                )
    reconciliation_events = [
        entry
        for entry in entries
        if entry.get("event_type") == "no_artifact_reconciliation_recorded"
    ]
    if reconciliation_events and reconciliation_directory is None:
        raise IntegrityError("ledger has no-artifact releases but no reconciliation root")
    recorded_reconciliations: set[str] = set()
    for entry in reconciliation_events:
        reservation_id = str(entry.get("reservation_entry_sha256") or "")
        reservation = reservations.get(reservation_id)
        if reservation is None:
            raise IntegrityError(f"no-artifact reconciliation has no reservation: {reservation_id}")
        digest = _require_sha256(
            entry.get("reconciliation_sha256"), field="ledger reconciliation digest"
        )
        if digest in recorded_reconciliations:
            raise IntegrityError(f"reconciliation was recorded more than once: {digest}")
        recorded_reconciliations.add(digest)
        filename = entry.get("reconciliation_filename")
        if filename != f"no-artifact-reconciliation-{digest}.json":
            raise IntegrityError("ledger reconciliation filename is not content-addressed")
        path = Path(str(reconciliation_directory)) / str(filename)
        record, _ = _load_no_artifact_reconciliation(path)
        if record.get("schema_version") == NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION:
            verified_v2 = validate_no_artifact_reconciliation_v2(
                path,
                ledger_entries=entries,
            )
            if (
                verified_v2.artifact_sha256 != digest
                or verified_v2.reservation_entry_sha256 != reservation_id
                or entry.get("reconciliation_schema_version")
                != NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION
                or entry.get("study_plan_sha256") != verified_v2.study_plan_sha256
                or entry.get("admission_block_id") != verified_v2.admission_block_id
                or entry.get("work_item_id") != verified_v2.work_item_id
                or entry.get("released_exposure_usd") != reservation.get("reserved_usd")
                or entry.get("model_id") != reservation.get("model_id")
                or entry.get("provider_tag") != reservation.get("provider_tag")
                or entry.get("manifest_sha256") != reservation.get("manifest_sha256")
                or entry.get("provider_generation_cost_usd") != "0"
                or entry.get("decision") != "release_never_started_no_delivery_reservation_v2"
            ):
                raise IntegrityError("ledger V2 reconciliation event does not match its proof")
            continue
        verified = validate_no_artifact_reconciliation(
            path,
            ledger_entries=entries,
            artifact_scan=artifact_scan,
        )
        if (
            verified.artifact_sha256 != digest
            or verified.reservation_entry_sha256 != reservation_id
            or entry.get("incident_entry_sha256") != verified.incident_entry_sha256
            or entry.get("known_pending_artifact_sha256") != verified.known_pending_artifact_sha256
            or entry.get("released_exposure_usd") != reservation.get("reserved_usd")
            or entry.get("model_id") != reservation.get("model_id")
            or entry.get("provider_tag") != reservation.get("provider_tag")
        ):
            raise IntegrityError("ledger reconciliation event does not match its proof")


def resolve_no_artifact_reservation(
    *,
    ledger_path: str | Path,
    reconciliation_path: str | Path,
    live_smoke_directory: str | Path,
    corrections_directory: str | Path | None = None,
    reconciliation_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Append a zero-cost finalization only after strict offline proof validation."""

    ledger = Path(ledger_path)
    proof_path = Path(reconciliation_path)
    proof_root = (
        Path(reconciliation_directory)
        if reconciliation_directory is not None
        else proof_path.parent
    )
    if proof_path.parent.resolve() != proof_root.resolve():
        raise IntegrityError("reconciliation proof is outside its governed directory")
    with _exclusive_runner_lock(ledger):
        entries = load_ledger(ledger)
        scan = scan_live_smoke_artifacts(
            live_smoke_directory,
            corrections_directory=corrections_directory,
        )
        validate_ledger_artifact_links(
            entries,
            scan,
            reconciliation_directory=proof_root,
        )
        proof = validate_no_artifact_reconciliation(
            proof_path,
            ledger_entries=entries,
            artifact_scan=scan,
        )
        active = active_ledger_reservations(entries)
        reserved = active.get(proof.reservation_entry_sha256)
        if reserved is None:
            raise IntegrityError("reconciled reservation is not active")
        reservation = next(
            entry
            for entry in entries
            if entry.get("entry_sha256") == proof.reservation_entry_sha256
        )
        event = append_ledger_event(
            ledger,
            {
                "event_type": "no_artifact_reconciliation_recorded",
                "runner_run_id": reservation.get("runner_run_id"),
                "reservation_entry_sha256": proof.reservation_entry_sha256,
                "incident_entry_sha256": proof.incident_entry_sha256,
                "model_id": reservation.get("model_id"),
                "provider_tag": reservation.get("provider_tag"),
                "manifest_sha256": reservation.get("manifest_sha256"),
                "reconciliation_filename": proof.path.name,
                "reconciliation_sha256": proof.artifact_sha256,
                "known_pending_artifact_sha256": (proof.known_pending_artifact_sha256),
                "account_usage_delta_usd": _decimal_text(proof.account_usage_delta_usd),
                "released_exposure_usd": _decimal_text(reserved),
                "provider_generation_cost_usd": "0",
                "decision": "release_pre_generation_no_cost_reservation",
            },
        )
        final_entries = load_ledger(ledger)
        validate_ledger_artifact_links(
            final_entries,
            scan,
            reconciliation_directory=proof_root,
        )
        if proof.reservation_entry_sha256 in active_ledger_reservations(final_entries):
            raise IntegrityError("reconciliation event did not finalize its reservation")
        return event


@contextmanager
def _exclusive_runner_lock(ledger_path: Path) -> Iterable[TextIO]:
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _existing_contract_pass(
    scan: ArtifactScan,
    *,
    manifest_sha256: str,
    candidate: ContractCandidate,
) -> ArtifactExposure | None:
    matches = [
        item
        for item in scan.artifacts
        if item.candidate_manifest_sha256 == manifest_sha256
        and item.requested_model_id == candidate.model_id
        and item.requested_provider == candidate.provider_tag
        and item.contract_passed
    ]
    return matches[-1] if matches else None


def _safe_process_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _extract_artifact_path(stdout: str, output_directory: Path) -> Path | None:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("artifact"), str):
        return None
    candidate = Path(value["artifact"]).resolve()
    root = output_directory.resolve()
    if root != candidate.parent:
        raise IntegrityError("delegated live_smoke returned an artifact outside its output root")
    return candidate


def _postflight_contract(
    path: Path,
    *,
    manifest_sha256: str,
    candidate: ContractCandidate,
    corrections_directory: str | Path | None = None,
) -> tuple[ArtifactExposure, list[str]]:
    artifact, digest = _verify_live_artifact(path)
    issues: list[str] = []
    if artifact.get("candidate_manifest_sha256") != manifest_sha256:
        issues.append("candidate_manifest_sha256_mismatch")
    if artifact.get("requested_model_id") != candidate.model_id:
        issues.append("requested_model_id_mismatch")
    if artifact.get("requested_provider") != candidate.provider_tag:
        issues.append("requested_provider_mismatch")
    model = artifact.get("model_contract")
    if not isinstance(model, Mapping):
        issues.append("missing_model_contract")
    else:
        if model.get("id") != candidate.model_id:
            issues.append("model_contract_id_mismatch")
        if model.get("canonical_slug") != candidate.canonical_model_slug:
            issues.append("canonical_model_slug_mismatch")
    endpoint = artifact.get("endpoint_contract")
    frozen_endpoint_fields = (
        "model_id",
        "provider_name",
        "tag",
        "quantization",
        "context_length",
        "max_completion_tokens",
        "pricing",
        "supported_parameters",
    )
    if not isinstance(endpoint, Mapping):
        issues.append("missing_endpoint_contract")
    else:
        for field in frozen_endpoint_fields:
            actual = endpoint.get(field)
            expected = candidate.endpoint.get(field)
            if field == "supported_parameters":
                actual = sorted(actual or [])
                expected = sorted(expected or [])
            if actual != expected:
                issues.append(f"endpoint_{field}_mismatch")
    scan = scan_live_smoke_artifacts(
        path.parent,
        corrections_directory=corrections_directory,
    )
    exposure = next(item for item in scan.artifacts if item.artifact_sha256 == digest)
    if not exposure.contract_passed:
        issues.append("tool_contract_not_passed")
    return exposure, issues


def _write_content_addressed_summary(
    summary: Mapping[str, Any], output_directory: str | Path
) -> Path:
    payload = dict(summary)
    digest = _sha256(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"frontier-contract-summary-{digest}.json"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise IntegrityError(f"refusing to overwrite conflicting summary: {destination}")
        return destination
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=root,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def _subprocess_command(
    candidate: ContractCandidate,
    *,
    manifest_sha256: str,
    forecast: ContractForecast,
    output_directory: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "flavourbench.live_smoke",
        "--confirm",
        LIVE_SMOKE_CONFIRMATION,
        "--cap-usd",
        _decimal_text(forecast.forecast_usd),
        "--model-id",
        candidate.model_id,
        "--provider-slug",
        candidate.provider_tag,
        "--prompt",
        TOOL_CONTRACT_PROMPT,
        "--category",
        "evidence",
        "--contract-only",
        "--output-dir",
        str(output_directory.resolve()),
        "--candidate-manifest-sha256",
        manifest_sha256,
        "--expected-canonical-model-slug",
        candidate.canonical_model_slug,
        "--expected-endpoint-execution-sha256",
        candidate.endpoint_execution_sha256,
    ]


def build_plan(
    manifest: Mapping[str, Any],
    candidates: Sequence[ContractCandidate],
    *,
    artifact_scan: ArtifactScan,
    active_reservation_usd: Decimal,
    policy: ContractPolicy,
    cap_usd: Decimal,
    admission_fraction: Decimal = DEFAULT_ADMISSION_FRACTION,
) -> list[dict[str, Any]]:
    manifest_sha = str(manifest["content_address"]["digest"])
    admission_ceiling = cap_usd * admission_fraction
    exposure = artifact_scan.exposure_usd + active_reservation_usd
    planned: list[dict[str, Any]] = []
    for candidate in candidates:
        forecast = derive_contract_forecast(candidate, policy=policy)
        passed = _existing_contract_pass(
            artifact_scan,
            manifest_sha256=manifest_sha,
            candidate=candidate,
        )
        base = {
            "slot_id": candidate.slot_id,
            "model_id": candidate.model_id,
            "canonical_model_slug": candidate.canonical_model_slug,
            "provider_tag": candidate.provider_tag,
            "provider_name": candidate.provider_name,
            "endpoint_sha256": candidate.endpoint_sha256,
            "forecast": forecast.public_payload(),
        }
        if passed is not None:
            planned.append(
                {
                    **base,
                    "decision": "skip_existing_contract_pass",
                    "existing_artifact_sha256": passed.artifact_sha256,
                }
            )
            continue
        projected = exposure + forecast.forecast_usd
        if projected > cap_usd:
            planned.append(
                {
                    **base,
                    "decision": "stop_hard_cap",
                    "exposure_before_usd": _decimal_text(exposure),
                    "projected_exposure_usd": _decimal_text(projected),
                }
            )
            break
        if projected > admission_ceiling:
            planned.append(
                {
                    **base,
                    "decision": "stop_85_percent_admission_ceiling",
                    "exposure_before_usd": _decimal_text(exposure),
                    "projected_exposure_usd": _decimal_text(projected),
                }
            )
            break
        planned.append(
            {
                **base,
                "decision": "admit_sequentially",
                "exposure_before_usd": _decimal_text(exposure),
                "projected_exposure_usd": _decimal_text(projected),
            }
        )
        # Planning reserves forecasts in sequence so it cannot over-admit a
        # whole panel on the assumption that every earlier run will be cheap.
        exposure = projected
    return planned


def run_frontier_contracts(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str = CURRENT_CANDIDATE_MANIFEST_SHA256,
    selectors: Iterable[str] = (),
    live_smoke_directory: str | Path = "artifacts/live-smoke",
    corrections_directory: str | Path | None = "artifacts/corrections",
    ledger_path: str | Path = "artifacts/frontier-contract/ledger.jsonl",
    reconciliation_directory: str | Path = ("artifacts/frontier-contract/reconciliations"),
    summary_directory: str | Path = "artifacts/frontier-contract/summaries",
    cap_usd: Decimal = AUTHORIZED_TOTAL_CAP_USD,
    admission_fraction: Decimal = DEFAULT_ADMISSION_FRACTION,
    execute: bool = False,
    confirmation: str = "",
    process_timeout_seconds: int = 3_600,
) -> tuple[dict[str, Any], Path]:
    """Plan or execute exact-endpoint contract smokes one model at a time."""

    if cap_usd <= 0 or cap_usd > AUTHORIZED_TOTAL_CAP_USD:
        raise AdmissionDenied(
            f"cap must be positive and at most the authorised ${AUTHORIZED_TOTAL_CAP_USD}"
        )
    if admission_fraction <= 0 or admission_fraction > DEFAULT_ADMISSION_FRACTION:
        raise AdmissionDenied(f"admission fraction must be in (0, {DEFAULT_ADMISSION_FRACTION}]")
    if execute and confirmation != EXECUTION_CONFIRMATION:
        raise AdmissionDenied(
            f"execution requires --confirm {EXECUTION_CONFIRMATION}; planning needs no confirmation"
        )
    manifest = load_candidate_manifest(
        manifest_path,
        expected_digest=expected_manifest_sha256,
    )
    candidates = select_candidates(manifest, selectors)
    if execute and any(
        candidate.execution_backend == "qwencloud_direct" for candidate in candidates
    ):
        raise AdmissionDenied(
            "QwenCloud execution must use the dataset runner's frozen policy and direct "
            "provider journal; this contract runner is planning-only for that backend"
        )
    settings = get_settings()
    policy = ContractPolicy(
        max_tool_rounds=settings.max_tool_rounds,
        max_output_tokens=settings.max_output_tokens,
        max_tool_result_bytes=settings.max_tool_result_bytes,
    )
    manifest_sha = str(manifest["content_address"]["digest"])
    smoke_root = Path(live_smoke_directory)
    ledger = Path(ledger_path)
    run_id = str(uuid.uuid4())
    started_at = _utc_now()
    outcomes: list[dict[str, Any]] = []

    with _exclusive_runner_lock(ledger):
        initial_scan = scan_live_smoke_artifacts(
            smoke_root,
            corrections_directory=corrections_directory,
        )
        initial_ledger = load_ledger(ledger)
        validate_ledger_artifact_links(
            initial_ledger,
            initial_scan,
            reconciliation_directory=reconciliation_directory,
        )
        initial_active = active_ledger_reservations(initial_ledger)
        initial_active_total = sum(initial_active.values(), Decimal(0))
        plan = build_plan(
            manifest,
            candidates,
            artifact_scan=initial_scan,
            active_reservation_usd=initial_active_total,
            policy=policy,
            cap_usd=cap_usd,
            admission_fraction=admission_fraction,
        )

        if not execute:
            outcomes = plan
        else:
            smoke_root.mkdir(parents=True, exist_ok=True)
            for candidate in candidates:
                # Execution replans after every completed artifact.  This
                # releases unused reservations only after verified accounting
                # and avoids stopping early merely because the no-call plan
                # reserves all remaining worst cases at once.
                current_scan = scan_live_smoke_artifacts(
                    smoke_root,
                    corrections_directory=corrections_directory,
                )
                current_ledger = load_ledger(ledger)
                validate_ledger_artifact_links(
                    current_ledger,
                    current_scan,
                    reconciliation_directory=reconciliation_directory,
                )
                active = active_ledger_reservations(current_ledger)
                active_total = sum(active.values(), Decimal(0))
                planned = build_plan(
                    manifest,
                    [candidate],
                    artifact_scan=current_scan,
                    active_reservation_usd=active_total,
                    policy=policy,
                    cap_usd=cap_usd,
                    admission_fraction=admission_fraction,
                )[0]
                decision = planned["decision"]
                if decision != "admit_sequentially":
                    outcomes.append(planned)
                    if decision.startswith("stop_"):
                        break
                    continue
                forecast = derive_contract_forecast(candidate, policy=policy)
                exposure_before = current_scan.exposure_usd + active_total
                projected = exposure_before + forecast.forecast_usd
                if projected > cap_usd or projected > cap_usd * admission_fraction:
                    outcomes.append(
                        {
                            **planned,
                            "decision": "stop_revalidated_budget_ceiling",
                            "exposure_before_usd": _decimal_text(exposure_before),
                            "projected_exposure_usd": _decimal_text(projected),
                        }
                    )
                    break
                reservation = append_ledger_event(
                    ledger,
                    {
                        "event_type": "reservation_created",
                        "runner_run_id": run_id,
                        "manifest_sha256": manifest_sha,
                        "model_id": candidate.model_id,
                        "canonical_model_slug": candidate.canonical_model_slug,
                        "provider_tag": candidate.provider_tag,
                        "endpoint_sha256": candidate.endpoint_sha256,
                        "reserved_usd": _decimal_text(forecast.forecast_usd),
                        "exposure_before_usd": _decimal_text(exposure_before),
                        "derived_max_price": {
                            "prompt_usd_per_mtok": _decimal_text(
                                forecast.price_envelope.prompt_usd_per_mtok
                            ),
                            "completion_usd_per_mtok": _decimal_text(
                                forecast.price_envelope.completion_usd_per_mtok
                            ),
                        },
                    },
                )
                command = _subprocess_command(
                    candidate,
                    manifest_sha256=manifest_sha,
                    forecast=forecast,
                    output_directory=smoke_root,
                )
                environment = os.environ.copy()
                environment["FLAVOURBENCH_OPENROUTER_MAX_PROMPT_PRICE_PER_MTOK"] = _decimal_text(
                    forecast.price_envelope.prompt_usd_per_mtok
                )
                environment["FLAVOURBENCH_OPENROUTER_MAX_COMPLETION_PRICE_PER_MTOK"] = (
                    _decimal_text(forecast.price_envelope.completion_usd_per_mtok)
                )
                try:
                    completed = subprocess.run(
                        command,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=process_timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    append_ledger_event(
                        ledger,
                        {
                            "event_type": "execution_incident",
                            "runner_run_id": run_id,
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "incident": "subprocess_timeout_uncertain_delivery",
                            "timeout_seconds": process_timeout_seconds,
                            "output_sha256": _safe_process_hash(str(error.output or "")),
                        },
                    )
                    outcomes.append(
                        {
                            **planned,
                            "decision": "execution_timeout_reservation_retained",
                            "reservation_entry_sha256": reservation["entry_sha256"],
                        }
                    )
                    break
                artifact_path = _extract_artifact_path(completed.stdout, smoke_root)
                if artifact_path is None or not artifact_path.exists():
                    append_ledger_event(
                        ledger,
                        {
                            "event_type": "execution_incident",
                            "runner_run_id": run_id,
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "incident": "no_verifiable_artifact_reservation_retained",
                            "subprocess_returncode": completed.returncode,
                            "stdout_sha256": _safe_process_hash(completed.stdout),
                            "stderr_sha256": _safe_process_hash(completed.stderr),
                        },
                    )
                    outcomes.append(
                        {
                            **planned,
                            "decision": "no_artifact_reservation_retained",
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "subprocess_returncode": completed.returncode,
                        }
                    )
                    break
                exposure, issues = _postflight_contract(
                    artifact_path,
                    manifest_sha256=manifest_sha,
                    candidate=candidate,
                    corrections_directory=corrections_directory,
                )
                recorded = append_ledger_event(
                    ledger,
                    {
                        "event_type": "artifact_recorded",
                        "runner_run_id": run_id,
                        "reservation_entry_sha256": reservation["entry_sha256"],
                        "manifest_sha256": manifest_sha,
                        "model_id": candidate.model_id,
                        "provider_tag": candidate.provider_tag,
                        "artifact_filename": artifact_path.name,
                        "artifact_sha256": exposure.artifact_sha256,
                        "artifact_status": exposure.status,
                        "artifact_exposure_usd": _decimal_text(exposure.exposure_usd),
                        "postflight_issues": issues,
                        "subprocess_returncode": completed.returncode,
                        "stdout_sha256": _safe_process_hash(completed.stdout),
                        "stderr_sha256": _safe_process_hash(completed.stderr),
                    },
                )
                outcomes.append(
                    {
                        **planned,
                        "decision": (
                            "contract_passed"
                            if not issues and completed.returncode == 0
                            else "contract_failed_or_mismatched"
                        ),
                        "reservation_entry_sha256": reservation["entry_sha256"],
                        "artifact_ledger_entry_sha256": recorded["entry_sha256"],
                        "artifact_sha256": exposure.artifact_sha256,
                        "artifact_filename": artifact_path.name,
                        "artifact_exposure_usd": _decimal_text(exposure.exposure_usd),
                        "postflight_issues": issues,
                        "subprocess_returncode": completed.returncode,
                    }
                )

        final_scan = scan_live_smoke_artifacts(
            smoke_root,
            corrections_directory=corrections_directory,
        )
        final_ledger = load_ledger(ledger)
        validate_ledger_artifact_links(
            final_ledger,
            final_scan,
            reconciliation_directory=reconciliation_directory,
        )
        final_active = active_ledger_reservations(final_ledger)
        final_active_total = sum(final_active.values(), Decimal(0))
        final_exposure = final_scan.exposure_usd + final_active_total
        if final_exposure > cap_usd:
            raise AdmissionDenied(
                f"verified budget exposure ${final_exposure} exceeds cap ${cap_usd}; "
                "no further calls are permitted"
            )
        summary: dict[str, Any] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "runner_run_id": run_id,
            "mode": "execute" if execute else "plan_no_provider_calls",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "manifest": {
                "filename": Path(manifest_path).name,
                "sha256": manifest_sha,
                "observed_at": manifest.get("observed_at"),
                "selected_model_count": len(candidates),
            },
            "budget": {
                "currency": "USD",
                "authorised_hard_cap_usd": _decimal_text(cap_usd),
                "admission_fraction": _decimal_text(admission_fraction),
                "admission_ceiling_usd": _decimal_text(cap_usd * admission_fraction),
                "initial_artifacts": initial_scan.public_payload(),
                "initial_active_ledger_reservations_usd": _decimal_text(initial_active_total),
                "final_artifacts": final_scan.public_payload(),
                "final_active_ledger_reservations_usd": _decimal_text(final_active_total),
                "final_total_exposure_usd": _decimal_text(final_exposure),
                "remaining_hard_cap_usd": _decimal_text(cap_usd - final_exposure),
            },
            "contract_policy": asdict(policy),
            "outcomes": outcomes,
            "ledger": {
                "filename": ledger.name,
                "entry_count": len(final_ledger),
                "head_entry_sha256": (final_ledger[-1]["entry_sha256"] if final_ledger else None),
            },
            "secret_handling": {
                "credentials_recorded": False,
                "subprocess_output_recorded": False,
                "subprocess_output_sha256_only": execute,
            },
            "limitations": [
                "All artifacts are permanently unranked engineering contract smokes.",
                "A passing tool contract is not a culinary preference result.",
                "Failed or unreconciled runs retain their full admitted allowance.",
                (
                    "The lock serializes this runner; direct live_smoke calls outside it cannot "
                    "be locked."
                ),
                "Planning reserves runs in manifest order but makes no provider calls or spend.",
            ],
        }
        summary_path = _write_content_addressed_summary(summary, summary_directory)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    return written, summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or sequentially execute exact-endpoint, unranked frontier contract smokes. "
            "The default is a no-provider-call plan."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--expected-manifest-sha256",
        default=CURRENT_CANDIDATE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Manifest slot ID, model ID, or canonical slug; repeat to select several",
    )
    parser.add_argument("--live-smoke-directory", default="artifacts/live-smoke")
    parser.add_argument("--corrections-directory", default="artifacts/corrections")
    parser.add_argument("--ledger", default="artifacts/frontier-contract/ledger.jsonl")
    parser.add_argument(
        "--reconciliation-directory",
        default="artifacts/frontier-contract/reconciliations",
    )
    parser.add_argument("--summary-directory", default="artifacts/frontier-contract/summaries")
    parser.add_argument("--cap-usd", type=Decimal, default=AUTHORIZED_TOTAL_CAP_USD)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--process-timeout-seconds", type=int, default=3_600)
    return parser


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-run-frontier-contracts")
    arguments = _parser().parse_args()
    try:
        summary, path = run_frontier_contracts(
            manifest_path=arguments.manifest,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
            selectors=arguments.model,
            live_smoke_directory=arguments.live_smoke_directory,
            corrections_directory=arguments.corrections_directory,
            ledger_path=arguments.ledger,
            reconciliation_directory=arguments.reconciliation_directory,
            summary_directory=arguments.summary_directory,
            cap_usd=arguments.cap_usd,
            execute=arguments.execute,
            confirmation=arguments.confirm,
            process_timeout_seconds=arguments.process_timeout_seconds,
        )
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
    print(
        json.dumps(
            {
                "status": "planned" if not arguments.execute else "finished",
                "provider_calls_made": arguments.execute,
                "summary": str(path.resolve()),
                "summary_sha256": summary["content_address"]["digest"],
                "budget": summary["budget"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
