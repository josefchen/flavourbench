"""Run the preregistered powered FlavourBench panel with resumable evidence.

The runner shares provider clients across cells, publishes one immutable
content-addressed response per model/task, writes provider-attempt events before
network I/O, and enforces a process-wide spend ceiling.  Its default mode is a
read-only preflight; live calls require an explicit confirmation string.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .budget_policy import provider_account_hard_cap_micros, provider_account_scope_sha256
from .config import get_settings
from .direct_kimi_pair import _credential_free_binding, _rate_card
from .epicure_native_powered_plan import (
    EPICURE_TOOL_SCHEMA_SHA256,
    MODEL_COUNT,
    PLAN_SCHEMA_VERSION,
    REPEAT_SCHEMA_VERSION,
    powered_execution_policy,
    verify_plan,
    verify_repeat_panel,
)
from .epicure_native_taskset_v2 import score_answer, verify_taskset
from .execution_policy import ExecutionPolicy
from .frontier_contract_runner import (
    ContractCandidate,
    load_candidate_manifest,
    select_candidates,
)
from .live_smoke import build_live_protocol_bundle, frozen_generation_contract
from .provider import (
    GenerationFailureResult,
    GenerationResult,
    GenerationSpec,
    OpenRouterProvider,
    ProviderAttemptEvent,
)
from .service_cohere import CohereDirectProvider
from .service_kimi import KimiDirectProvider
from .tool_contract import required_tool_contract

RUNNER_SCHEMA_VERSION = "flavourbench-powered-runner-v1"
RESPONSE_SCHEMA_VERSION = "flavourbench-powered-response-v1"
PREFLIGHT_SCHEMA_VERSION = "flavourbench-powered-runner-preflight-v1"
CONFIRMATION = "RUN_POWERED_FLAVOURBENCH_V1"
SECRET_KEYS = {
    "OPENROUTER_API_KEY": "FLAVOURBENCH_OPENROUTER_API_KEY",
    "KIMI_API_KEY": "FLAVOURBENCH_KIMI_API_KEY",
    "COHERE_API_KEY": "FLAVOURBENCH_COHERE_API_KEY",
}
SECRET_PATTERN = re.compile(
    r"(?:sk-(?:[A-Za-z0-9._-]{8,})|hf_[A-Za-z0-9]{8,}|Bearer\s+\S+)", re.IGNORECASE
)


class PoweredRunnerError(RuntimeError):
    """The powered run cannot safely continue."""


class BudgetExhausted(PoweredRunnerError):
    """The next cell cannot be admitted inside the frozen spend cap."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PoweredRunnerError(f"{label} must be a regular, non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PoweredRunnerError(f"could not parse {label}") from error
    if not isinstance(value, dict):
        raise PoweredRunnerError(f"{label} must be a JSON object")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_error(error: BaseException) -> str:
    rendered = SECRET_PATTERN.sub("[credential redacted]", str(error))
    return rendered[:1000]


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise PoweredRunnerError(f"invalid decimal in {field}") from error
    if not parsed.is_finite() or parsed < 0:
        raise PoweredRunnerError(f"invalid decimal in {field}")
    return parsed


def _semantic_valid(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == _sha256(payload))


def _write_content_addressed(
    document: Mapping[str, Any], *, directory: Path, filename_prefix: str
) -> Path:
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    digest = _sha256(payload)
    payload["artifact_sha256"] = digest
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{filename_prefix}-{digest}.json"
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise PoweredRunnerError("content-addressed response conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_secret_file(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise PoweredRunnerError("credential source must be a regular, non-symlink file")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in SECRET_KEYS:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            values[SECRET_KEYS[name]] = value
    missing = sorted(set(SECRET_KEYS.values()) - set(values))
    if missing:
        raise PoweredRunnerError(
            "credential source lacks required OpenRouter, Kimi, or Cohere key material"
        )
    return values


def configure_live_environment(
    *,
    secret_file: Path,
    plan: Mapping[str, Any],
    execution_policy: ExecutionPolicy | None = None,
) -> None:
    secrets = _read_secret_file(secret_file)
    execution = plan["execution"]
    policy = execution_policy or powered_execution_policy()
    if execution.get("execution_policy_sha256") != policy.sha256:
        raise PoweredRunnerError("plan execution policy differs from runner policy")
    overrides = {
        **secrets,
        **policy.settings_environment(),
        "FLAVOURBENCH_ENVIRONMENT": "development",
        "FLAVOURBENCH_SERVICE_ROLE": "worker",
        "FLAVOURBENCH_EXECUTION_MODE": "live",
        "FLAVOURBENCH_LIVE_AUTHORIZED": "true",
        "FLAVOURBENCH_EPICURE_RELEASE_ID": str(execution["epicure_release_id"]),
        "FLAVOURBENCH_EPICURE_BUNDLE_SHA256": str(execution["epicure_bundle_sha256"]),
        "FLAVOURBENCH_EPICURE_APPLICATION_SHA256": str(execution["epicure_application_sha256"]),
        "FLAVOURBENCH_EPICURE_TOOL_SCHEMA_SHA256": str(execution["epicure_tool_schema_sha256"]),
    }
    for name, value in overrides.items():
        os.environ[name] = value
    get_settings.cache_clear()
    settings = get_settings()
    if (
        settings.execution_mode != "live"
        or settings.live_authorized is not True
        or not settings.openrouter_api_key
        or not settings.kimi_api_key
        or not settings.cohere_api_key
    ):
        raise PoweredRunnerError("live provider settings did not validate")


@dataclass(frozen=True)
class PlannedCell:
    cell_id: str
    panel: str
    candidate: ContractCandidate
    task: Mapping[str, Any]

    @property
    def arm_id(self) -> str:
        return f"powered-{self.cell_id[:32]}"


def _cell_id(
    *, plan_sha256: str, panel: str, candidate: ContractCandidate, task: Mapping[str, Any]
) -> str:
    return _sha256(
        {
            "schema_version": "flavourbench-powered-cell-v1",
            "plan_sha256": plan_sha256,
            "panel": panel,
            "slot_id": candidate.slot_id,
            "model_id": candidate.model_id,
            "task_id": task["task_id"],
            "prompt_sha256": task["prompt_sha256"],
        }
    )


def build_cells(
    *,
    plan: Mapping[str, Any],
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    candidates: Sequence[ContractCandidate],
    phase: str,
) -> list[PlannedCell]:
    if phase not in {"pilot", "primary", "repeat", "all"}:
        raise PoweredRunnerError("unsupported run phase")
    plan_sha256 = str(plan["artifact_sha256"])
    pilot_task_id = str(plan["execution"]["pilot"]["task_id"])
    primary_by_id = {str(task["task_id"]): task for task in taskset["tasks"]}
    if pilot_task_id not in primary_by_id:
        raise PoweredRunnerError("frozen pilot task is absent from task set")
    cells: list[PlannedCell] = []
    for candidate in candidates:
        primary_tasks = sorted(
            taskset["tasks"],
            key=lambda task: (
                0 if task["task_id"] == pilot_task_id else 1,
                hashlib.sha256(
                    (
                        plan_sha256
                        + "\0primary\0"
                        + candidate.model_id
                        + "\0"
                        + str(task["task_id"])
                    ).encode()
                ).hexdigest(),
            ),
        )
        if phase == "pilot":
            primary_tasks = [primary_by_id[pilot_task_id]]
        elif phase == "repeat":
            primary_tasks = []
        for task in primary_tasks:
            cells.append(
                PlannedCell(
                    cell_id=_cell_id(
                        plan_sha256=plan_sha256,
                        panel="primary",
                        candidate=candidate,
                        task=task,
                    ),
                    panel="primary",
                    candidate=candidate,
                    task=task,
                )
            )
        if phase in {"repeat", "all"}:
            repeated = sorted(
                repeat_panel["tasks"],
                key=lambda task: hashlib.sha256(
                    (
                        plan_sha256
                        + "\0repeat\0"
                        + candidate.model_id
                        + "\0"
                        + str(task["task_id"])
                    ).encode()
                ).hexdigest(),
            )
            for task in repeated:
                cells.append(
                    PlannedCell(
                        cell_id=_cell_id(
                            plan_sha256=plan_sha256,
                            panel="repeat",
                            candidate=candidate,
                            task=task,
                        ),
                        panel="repeat",
                        candidate=candidate,
                        task=task,
                    )
                )
    expected = {
        "pilot": len(candidates),
        "primary": 640 * len(candidates),
        "repeat": 64 * len(candidates),
        "all": 704 * len(candidates),
    }[phase]
    if len(cells) != expected or len({cell.cell_id for cell in cells}) != expected:
        raise PoweredRunnerError("powered cell schedule is incomplete or duplicated")
    return cells


class AttemptJournal:
    """Durably record each safe provider lifecycle event before returning."""

    def __init__(self, path: Path, *, plan_sha256: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.plan_sha256 = plan_sha256
        self.by_arm: dict[str, list[str]] = defaultdict(list)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                payload = dict(record)
                recorded = str(payload.pop("event_sha256", ""))
                if recorded != _sha256(payload) or payload.get("plan_sha256") != plan_sha256:
                    raise PoweredRunnerError("attempt journal failed integrity validation")
                self.by_arm[str(payload["event"]["arm_id"])].append(recorded)
        self.handle = path.open("a", encoding="utf-8")

    def __call__(self, event: ProviderAttemptEvent) -> None:
        payload: dict[str, Any] = {
            "schema_version": "flavourbench-powered-attempt-event-v1",
            "plan_sha256": self.plan_sha256,
            "recorded_at": _utc_now(),
            "event": asdict(event),
        }
        digest = _sha256(payload)
        payload["event_sha256"] = digest
        self.handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.by_arm[event.arm_id].append(digest)

    def close(self) -> None:
        self.handle.close()


class BudgetTracker:
    def __init__(self, *, cap_micros: int, initial_spend_micros: int = 0) -> None:
        self.cap_micros = cap_micros
        self.spent_micros = initial_spend_micros
        self.reserved_micros = 0
        self.lock = asyncio.Lock()
        self.exhausted = False

    async def acquire(self, reserve_micros: int) -> None:
        async with self.lock:
            if (
                self.exhausted
                or self.spent_micros + self.reserved_micros + reserve_micros > self.cap_micros
            ):
                self.exhausted = True
                raise BudgetExhausted("powered run budget cap reached before cell admission")
            self.reserved_micros += reserve_micros

    async def settle(self, *, reserve_micros: int, actual_micros: int) -> None:
        async with self.lock:
            self.reserved_micros -= reserve_micros
            self.spent_micros += actual_micros
            if self.spent_micros > self.cap_micros:
                self.exhausted = True
                raise BudgetExhausted("provider-reported spend exceeded the powered run cap")


def _reserve_micros(
    candidate: ContractCandidate,
    predecessor_release: Mapping[str, Any],
    *,
    max_output_tokens: int = 2_048,
) -> int:
    historical = [
        int(observation["cost_micros"])
        for observation in predecessor_release.get("observations", [])
        if observation.get("condition") == "epicure_off"
        and observation.get("model_id") == candidate.model_id
        and isinstance(observation.get("cost_micros"), int)
    ]
    observed_max = max(historical, default=0)
    pricing = candidate.endpoint.get("pricing")
    pricing = pricing if isinstance(pricing, Mapping) else {}
    prompt = _decimal(pricing.get("prompt", 0), field="prompt price") * 4096
    completion = (
        _decimal(pricing.get("completion", 0), field="completion price") * max_output_tokens
    )
    reasoning = (
        _decimal(pricing.get("internal_reasoning", 0), field="reasoning price") * max_output_tokens
    )
    request = _decimal(pricing.get("request", 0), field="request price")
    price_envelope = int(
        ((prompt + completion + reasoning + request) * Decimal(1_000_000)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    return max(1000, observed_max, price_envelope)


def build_generation_spec(
    *,
    cell: PlannedCell,
    plan: Mapping[str, Any],
    manifest_sha256: str,
    taskset: Mapping[str, Any],
    reserve_micros: int,
    execution_policy: ExecutionPolicy | None = None,
) -> tuple[GenerationSpec, dict[str, Any]]:
    candidate = cell.candidate
    model = {
        "id": candidate.model_id,
        "canonical_slug": candidate.canonical_model_slug,
        "name": candidate.model_name,
    }
    endpoint = dict(candidate.endpoint)
    policy = execution_policy or powered_execution_policy()
    roster_rows = [
        row
        for row in plan.get("roster", {}).get("models", [])
        if row.get("model_id") == candidate.model_id
    ]
    if len(roster_rows) != 1:
        raise PoweredRunnerError("candidate is not uniquely bound in the frozen roster")
    route_effort = roster_rows[0].get("final_reasoning_effort")
    if route_effort == "provider_fixed":
        route_effort = None
    if route_effort is not None:
        policy = replace(policy, final_reasoning_effort=str(route_effort))
    route_max_output_tokens = roster_rows[0].get("final_max_output_tokens")
    if route_max_output_tokens is not None:
        if (
            not isinstance(route_max_output_tokens, int)
            or isinstance(route_max_output_tokens, bool)
            or route_max_output_tokens <= 0
            or route_max_output_tokens > policy.max_output_tokens
        ):
            raise PoweredRunnerError("route-specific response ceiling is invalid")
        policy = replace(policy, max_output_tokens=route_max_output_tokens)
    backend = candidate.execution_backend
    generation_contract = frozen_generation_contract(
        model,
        endpoint,
        max_output_tokens=policy.max_output_tokens,
        decoding_temperature=policy.decoding_temperature,
        decoding_top_p=policy.decoding_top_p,
        decoding_seed=policy.decoding_seed,
    )
    if generation_contract.get("decoding_parameters", {}).get("max_tokens") != (
        policy.max_output_tokens
    ):
        raise PoweredRunnerError(
            "configured request ceiling differs from the explicit execution policy"
        )
    provenance = dict(taskset["epicure_provenance"])
    protocol_bundle, protocol_bundle_sha256 = build_live_protocol_bundle(
        candidate_manifest_sha256=manifest_sha256,
        dataset_work_item_id=cell.cell_id,
        dataset_task_id=str(cell.task["task_id"]),
        prompt=str(cell.task["prompt"]),
        category=str(cell.task["family"]),
        model=model,
        endpoint=endpoint,
        generation_contract=generation_contract,
        execution_policy=policy,
        provenance=provenance,
        tool_schema_sha256=EPICURE_TOOL_SCHEMA_SHA256,
        run_purpose=f"powered_model_only_{cell.panel}",
        final_response_mode=policy.final_response_mode,
        selected_conditions=("epicure_off",),
    )
    scope_sha256 = provider_account_scope_sha256(backend)
    backend_contract_sha256 = (
        candidate.backend_contract_sha256
        if backend != "openrouter"
        else candidate.endpoint_execution_sha256
    )
    smoke_sha256 = str(
        (candidate.route_selection.get("evidence") or {}).get("compatibility_artifact_sha256")
        or manifest_sha256
    )
    generation_contract.update(
        {
            "arm_id": cell.arm_id,
            "battle_id": cell.cell_id,
            "prompt": str(cell.task["prompt"]),
            "category": str(cell.task["family"]),
            "model_id": candidate.model_id,
            "model_name": candidate.model_name,
            "provider_slug": candidate.provider_tag,
            "condition": "epicure_off",
            "idempotency_key": cell.cell_id,
            "final_response_mode": policy.final_response_mode,
            "matched_planning": policy.matched_planning,
            "intermediate_max_tokens": policy.max_intermediate_tokens,
            "required_tool_contract_max_intermediate_tokens": (
                policy.required_tool_contract_max_intermediate_tokens
            ),
            "evidence_protocol": policy.evidence_protocol,
            "intermediate_reasoning_effort": policy.intermediate_reasoning_effort,
            "final_reasoning_effort": policy.final_reasoning_effort,
            "required_tool_contract_protocol": policy.required_tool_contract_protocol,
            "required_tool_contract_sha256": required_tool_contract(policy)["content_address"][
                "digest"
            ],
            "epicure_on_tool_required": policy.epicure_on_tool_required,
            "protocol_bundle_sha256": protocol_bundle_sha256,
            "expected_epicure_release_id": provenance["release_id"],
            "expected_epicure_bundle_sha256": provenance["bundle_sha256"],
            "expected_epicure_application_sha256": provenance["application_sha256"],
            "expected_epicure_tool_schema_sha256": EPICURE_TOOL_SCHEMA_SHA256,
            "execution_backend": backend,
            "rate_card_json": _rate_card(endpoint),
            "backend_contract_json": dict(candidate.backend_contract),
            "provider_budget_cap_micros": reserve_micros,
            "provider_account_budget_cap_micros": provider_account_hard_cap_micros(backend),
            "provider_account_scope_sha256": scope_sha256,
            "provider_authorization_envelope_sha256": _credential_free_binding(
                manifest_sha256=str(plan["artifact_sha256"]),
                backend_contract_sha256=backend_contract_sha256,
                scope_sha256=scope_sha256,
                kind="powered_cell_budget",
            ),
            "provider_account_authorization_envelope_sha256": _credential_free_binding(
                manifest_sha256=str(plan["artifact_sha256"]),
                backend_contract_sha256=backend_contract_sha256,
                scope_sha256=scope_sha256,
                kind="powered_account_budget",
            ),
            "provider_credential_binding_sha256": _credential_free_binding(
                manifest_sha256=str(plan["artifact_sha256"]),
                backend_contract_sha256=backend_contract_sha256,
                scope_sha256=scope_sha256,
                kind="powered_credential_binding",
            ),
            "provider_credential_scope_sha256": scope_sha256,
            "contract_smoke_registry_sha256": smoke_sha256,
        }
    )
    return GenerationSpec(**generation_contract), protocol_bundle


def _generation_payload(result: GenerationResult) -> dict[str, Any]:
    return {
        "answer_markdown": result.answer_markdown,
        "actual_model_id": result.actual_model_id,
        "actual_provider": result.provider_slug,
        "generation_id": result.generation_id,
        "generation_ids": result.generation_ids,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "cost_micros": result.cost_micros,
        "cost_reconciled": result.cost_reconciled,
        "latency_ms": result.latency_ms,
        "retries": result.retries,
        "finish_reason": result.finish_reason,
        "generation_metadata": result.generation_metadata,
        "decoding": result.decoding_json,
        "cost_accounting_basis": result.cost_accounting_basis,
        "billing_reconciliation_status": result.billing_reconciliation_status,
        "backend_response_schema_sha256": result.backend_response_schema_sha256,
        "backend_tool_schema_sha256": result.backend_tool_schema_sha256,
    }


def _failure_payload(result: GenerationFailureResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "cost_micros": 0,
            "cost_reconciled": False,
            "generation_ids": [],
            "generation_metadata": [],
            "billing_reconciliation_status": "no_accepted_request_identified",
        }
    return {
        "actual_model_id": result.actual_model_id,
        "actual_provider": result.provider_slug,
        "generation_id": result.generation_id,
        "generation_ids": result.generation_ids,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_micros": result.cost_micros,
        "cost_reconciled": result.cost_reconciled,
        "latency_ms": result.latency_ms,
        "retries": result.retries,
        "generation_metadata": result.generation_metadata,
        "decoding": result.decoding_json,
        "cost_accounting_basis": result.cost_accounting_basis,
        "billing_reconciliation_status": result.billing_reconciliation_status,
    }


def _task_reference_payload(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen task reference without assuming one task format."""
    payload: dict[str, Any] = {}
    if "expected_choice" in task:
        payload["expected_choice"] = task["expected_choice"]
    if "optimal_selection" in task:
        payload["optimal_selection"] = task["optimal_selection"]
    if not payload:
        raise PoweredRunnerError("powered task has no frozen scoring reference")
    return payload


class ProviderPool:
    def __init__(self, *, attempt_sink: AttemptJournal) -> None:
        self.openrouter = OpenRouterProvider(attempt_sink=attempt_sink)
        self.kimi = KimiDirectProvider(attempt_sink=attempt_sink)
        self.cohere = CohereDirectProvider(attempt_sink=attempt_sink)

    def get(self, backend: str) -> Any:
        if backend == "openrouter":
            return self.openrouter
        if backend == "kimi_direct":
            return self.kimi
        if backend == "cohere_direct":
            return self.cohere
        raise PoweredRunnerError(f"powered panel has unsupported backend: {backend}")

    async def close(self) -> None:
        await asyncio.gather(self.openrouter.aclose(), self.kimi.aclose(), self.cohere.aclose())


class RequestPacer:
    """Enforce a shared backend start-to-start request interval."""

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds < 0:
            raise PoweredRunnerError("request pacing interval cannot be negative")
        self.interval_seconds = interval_seconds
        self.lock = asyncio.Lock()
        self.next_start = 0.0

    async def wait(self) -> None:
        if self.interval_seconds == 0:
            return
        async with self.lock:
            delay = self.next_start - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = time.monotonic() + self.interval_seconds


class PoweredRun:
    def __init__(
        self,
        *,
        plan: Mapping[str, Any],
        taskset: Mapping[str, Any],
        repeat_panel: Mapping[str, Any],
        manifest_sha256: str,
        predecessor_release: Mapping[str, Any],
        output_directory: Path,
        global_concurrency: int,
        per_model_concurrency: int,
        score_function: Callable[[Mapping[str, Any], str], dict[str, Any]] = score_answer,
        execution_policy: ExecutionPolicy | None = None,
    ) -> None:
        self.plan = plan
        self.taskset = taskset
        self.repeat_panel = repeat_panel
        self.manifest_sha256 = manifest_sha256
        self.predecessor_release = predecessor_release
        self.output_directory = output_directory
        self.global_semaphore = asyncio.Semaphore(global_concurrency)
        self.per_model_concurrency = per_model_concurrency
        collection = plan.get("execution", {}).get("collection_concurrency", {})
        per_backend = collection.get("per_model_by_backend", {})
        if not isinstance(per_backend, Mapping) or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in per_backend.values()
        ):
            raise PoweredRunnerError("per-backend concurrency contract is invalid")
        self.per_model_concurrency_by_backend = {
            str(backend): int(value) for backend, value in per_backend.items()
        }
        per_model = collection.get("per_model_by_model_id", {})
        if not isinstance(per_model, Mapping) or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in per_model.values()
        ):
            raise PoweredRunnerError("per-model concurrency contract is invalid")
        self.per_model_concurrency_by_model_id = {
            str(model_id): int(value) for model_id, value in per_model.items()
        }
        self.score_function = score_function
        self.execution_policy = execution_policy or powered_execution_policy()
        pacing = plan.get("execution", {}).get("minimum_request_interval_seconds_by_backend", {})
        if not isinstance(pacing, Mapping):
            raise PoweredRunnerError("backend request pacing contract is invalid")
        self.backend_pacers = {
            str(backend): RequestPacer(float(interval)) for backend, interval in pacing.items()
        }
        self.journal = AttemptJournal(
            output_directory / "attempts" / "provider-attempts.jsonl",
            plan_sha256=str(plan["artifact_sha256"]),
        )
        existing_spend = self._existing_spend()
        cap_micros = int(Decimal(str(plan["budget"]["hard_cap"])) * 1_000_000)
        self.budget = BudgetTracker(cap_micros=cap_micros, initial_spend_micros=existing_spend)
        self.completed_this_session = 0
        self.failed_this_session = 0
        self.skipped_existing = 0
        self.progress_lock = asyncio.Lock()

    def _response_directory(self, cell: PlannedCell) -> Path:
        return self.output_directory / "responses" / cell.panel / cell.candidate.slot_id

    def _existing_response(self, cell: PlannedCell) -> dict[str, Any] | None:
        paths = list(self._response_directory(cell).glob(f"response-{cell.cell_id}-*.json"))
        if not paths:
            return None
        if len(paths) != 1:
            raise PoweredRunnerError(f"cell {cell.cell_id} has multiple response artifacts")
        document = _load_json(paths[0], label="powered response")
        if (
            not _semantic_valid(document)
            or document.get("schema_version") != RESPONSE_SCHEMA_VERSION
            or document.get("cell_id") != cell.cell_id
            or document.get("plan_sha256") != self.plan["artifact_sha256"]
        ):
            raise PoweredRunnerError(f"cell {cell.cell_id} response failed integrity validation")
        return document

    def _existing_spend(self) -> int:
        total = 0
        for path in (self.output_directory / "responses").glob("*/*/response-*.json"):
            document = _load_json(path, label="existing powered response")
            if (
                not _semantic_valid(document)
                or document.get("plan_sha256") != self.plan["artifact_sha256"]
            ):
                raise PoweredRunnerError("existing powered response has invalid provenance")
            total += int((document.get("generation") or {}).get("cost_micros") or 0)
        return total

    async def _record_progress(self, *, failed: bool) -> None:
        async with self.progress_lock:
            self.completed_this_session += 1
            self.failed_this_session += int(failed)
            if self.completed_this_session % 100 == 0:
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "completed_this_session": self.completed_this_session,
                            "failed_this_session": self.failed_this_session,
                            "skipped_existing": self.skipped_existing,
                            "spend_usd": f"{self.budget.spent_micros / 1_000_000:.6f}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    async def execute_cell(self, cell: PlannedCell, provider_pool: ProviderPool) -> None:
        if self._existing_response(cell) is not None:
            self.skipped_existing += 1
            return
        reserve = _reserve_micros(
            cell.candidate,
            self.predecessor_release,
            max_output_tokens=self.execution_policy.max_output_tokens,
        )
        await self.budget.acquire(reserve)
        actual_cost = 0
        failed = False
        try:
            spec, protocol_bundle = build_generation_spec(
                cell=cell,
                plan=self.plan,
                manifest_sha256=self.manifest_sha256,
                taskset=self.taskset,
                reserve_micros=reserve,
                execution_policy=self.execution_policy,
            )
            started_at = _utc_now()
            started = time.monotonic()
            provider = provider_pool.get(cell.candidate.execution_backend)
            error: BaseException | None = None
            result: GenerationResult | None = None
            failure: GenerationFailureResult | None = None
            async with self.global_semaphore:
                try:
                    pacer = self.backend_pacers.get(cell.candidate.execution_backend)
                    if pacer is not None:
                        await pacer.wait()
                    result = await provider.generate(spec)
                except Exception as caught:
                    error = caught
                    try:
                        failure = await provider.reconcile_failure(spec, caught)
                    except Exception as reconciliation_error:
                        error = PoweredRunnerError(
                            f"{type(caught).__name__}; accounting reconciliation failed: "
                            f"{type(reconciliation_error).__name__}"
                        )
            if result is not None:
                generation = _generation_payload(result)
                scoring = self.score_function(cell.task, result.answer_markdown)
                actual_cost = int(result.cost_micros)
                status = "completed" if result.finish_reason in {"stop", "end_turn"} else "failed"
                failed = status != "completed"
                if failed:
                    scoring = {
                        **scoring,
                        **({"correct": False} if "correct" in scoring else {}),
                        **({"optimal": False} if "optimal" in scoring else {}),
                        **({"score_bps": 0} if "score_bps" in scoring else {}),
                        "score": 0,
                    }
            else:
                generation = _failure_payload(failure)
                actual_cost = int(generation.get("cost_micros") or 0)
                scoring = self.score_function(cell.task, "")
                status = "failed"
                failed = True
            artifact: dict[str, Any] = {
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                "status": status,
                "recorded_at": _utc_now(),
                "started_at": started_at,
                "wall_time_ms": round((time.monotonic() - started) * 1000),
                "plan_sha256": self.plan["artifact_sha256"],
                "manifest_sha256": self.manifest_sha256,
                "taskset_sha256": self.taskset["artifact_sha256"],
                "repeat_panel_sha256": self.repeat_panel["artifact_sha256"],
                "cell_id": cell.cell_id,
                "panel": cell.panel,
                "arm_id": cell.arm_id,
                "slot_id": cell.candidate.slot_id,
                "model_id": cell.candidate.model_id,
                "model_name": cell.candidate.model_name,
                "canonical_model_slug": cell.candidate.canonical_model_slug,
                "execution_backend": cell.candidate.execution_backend,
                "provider_route": cell.candidate.provider_tag,
                "endpoint_execution_sha256": cell.candidate.endpoint_execution_sha256,
                "backend_contract_sha256": cell.candidate.backend_contract_sha256,
                "task_id": cell.task["task_id"],
                "original_task_id": cell.task.get("original_task_id"),
                "family": cell.task["family"],
                "prompt_sha256": cell.task["prompt_sha256"],
                **_task_reference_payload(cell.task),
                "protocol_bundle": protocol_bundle,
                "protocol_bundle_sha256": spec.protocol_bundle_sha256,
                "attempt_event_sha256s": list(self.journal.by_arm.get(cell.arm_id, [])),
                "generation": generation,
                "scoring": scoring,
                "error": (
                    {"type": type(error).__name__, "message": _safe_error(error)}
                    if error is not None
                    else None
                ),
                "budget": {
                    "reserved_micros": reserve,
                    "actual_micros": actual_cost,
                    "global_cap_micros": self.budget.cap_micros,
                },
            }
            _write_content_addressed(
                artifact,
                directory=self._response_directory(cell),
                filename_prefix=f"response-{cell.cell_id}",
            )
            await self._record_progress(failed=failed)
        finally:
            await self.budget.settle(reserve_micros=reserve, actual_micros=actual_cost)

    async def execute(self, cells: Sequence[PlannedCell]) -> dict[str, Any]:
        by_model: dict[str, list[PlannedCell]] = defaultdict(list)
        for cell in cells:
            by_model[cell.candidate.model_id].append(cell)
        provider_pool = ProviderPool(attempt_sink=self.journal)

        async def worker(queue: asyncio.Queue[PlannedCell]) -> None:
            while not self.budget.exhausted:
                try:
                    cell = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await self.execute_cell(cell, provider_pool)
                except BudgetExhausted:
                    return
                finally:
                    queue.task_done()

        tasks: list[asyncio.Task[None]] = []
        for model_cells in by_model.values():
            queue: asyncio.Queue[PlannedCell] = asyncio.Queue()
            for cell in model_cells:
                queue.put_nowait(cell)
            backend = model_cells[0].candidate.execution_backend
            model_id = model_cells[0].candidate.model_id
            model_concurrency = self.per_model_concurrency_by_model_id.get(
                model_id,
                self.per_model_concurrency_by_backend.get(backend, self.per_model_concurrency),
            )
            for _ in range(min(model_concurrency, len(model_cells))):
                tasks.append(asyncio.create_task(worker(queue)))
        try:
            await asyncio.gather(*tasks)
        finally:
            await provider_pool.close()
            self.journal.close()
        return {
            "schema_version": "flavourbench-powered-run-session-v1",
            "status": "budget_exhausted" if self.budget.exhausted else "phase_complete",
            "plan_sha256": self.plan["artifact_sha256"],
            "scheduled_cells": len(cells),
            "completed_this_session": self.completed_this_session,
            "failed_this_session": self.failed_this_session,
            "skipped_existing": self.skipped_existing,
            "spend_micros": self.budget.spent_micros,
            "budget_cap_micros": self.budget.cap_micros,
            "finished_at": _utc_now(),
        }


def validate_inputs(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    taskset_path: Path,
    repeat_panel_path: Path,
    plan_path: Path,
    predecessor_release_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[ContractCandidate],
]:
    manifest = load_candidate_manifest(manifest_path, expected_digest=manifest_sha256)
    taskset = _load_json(taskset_path, label="powered task set")
    repeat = _load_json(repeat_panel_path, label="powered repeat panel")
    plan = _load_json(plan_path, label="powered analysis plan")
    predecessor = _load_json(predecessor_release_path, label="predecessor release")
    if (
        not verify_taskset(taskset)
        or not verify_repeat_panel(repeat, source_taskset=taskset)
        or not verify_plan(plan)
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or repeat.get("schema_version") != REPEAT_SCHEMA_VERSION
    ):
        raise PoweredRunnerError("powered input verification failed")
    inputs = plan["inputs"]
    exact = {
        "manifest": (
            manifest_sha256,
            _sha256_file(manifest_path),
            inputs["route_manifest"],
        ),
        "taskset": (
            taskset["artifact_sha256"],
            _sha256_file(taskset_path),
            inputs["hidden_taskset"],
        ),
        "repeat": (
            repeat["artifact_sha256"],
            _sha256_file(repeat_panel_path),
            inputs["repeat_panel"],
        ),
        "predecessor": (
            predecessor["artifact_sha256"],
            _sha256_file(predecessor_release_path),
            inputs["predecessor_release"],
        ),
    }
    for label, (semantic, physical, recorded) in exact.items():
        if recorded["semantic_sha256"] != semantic or recorded["physical_sha256"] != physical:
            raise PoweredRunnerError(f"plan {label} pin differs from exact input")
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT:
        raise PoweredRunnerError("powered manifest does not contain exactly 20 candidates")
    return manifest, taskset, repeat, plan, predecessor, candidates


async def _async_run(args: argparse.Namespace) -> None:
    manifest, taskset, repeat, plan, predecessor, candidates = validate_inputs(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_semantic_sha256,
        taskset_path=args.taskset,
        repeat_panel_path=args.repeat_panel,
        plan_path=args.plan,
        predecessor_release_path=args.predecessor_release,
    )
    configure_live_environment(secret_file=args.secrets_env_file, plan=plan)
    cells = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase=args.phase,
    )
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    if args.preflight_only:
        document = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "preflight_passed_no_provider_calls",
            "plan_sha256": plan["artifact_sha256"],
            "manifest_sha256": args.manifest_semantic_sha256,
            "taskset_sha256": taskset["artifact_sha256"],
            "repeat_panel_sha256": repeat["artifact_sha256"],
            "models": len(candidates),
            "scheduled_cells": len(cells),
            "required_credentials_present": sorted(SECRET_KEYS),
            "provider_clients_constructed": False,
            "provider_calls_made": 0,
        }
        document["artifact_sha256"] = _sha256(document)
        print(json.dumps(document, sort_keys=True))
        return
    if args.confirm != CONFIRMATION:
        raise PoweredRunnerError(f"live execution requires --confirm {CONFIRMATION}")
    runner = PoweredRun(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        manifest_sha256=args.manifest_semantic_sha256,
        predecessor_release=predecessor,
        output_directory=args.output_directory,
        global_concurrency=args.global_concurrency,
        per_model_concurrency=args.per_model_concurrency,
    )
    result = await runner.execute(cells)
    print(json.dumps(result, sort_keys=True))


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predecessor-release", type=Path, required=True)
    parser.add_argument("--secrets-env-file", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "primary", "repeat", "all"), default="pilot")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--global-concurrency", type=int, default=40)
    parser.add_argument("--per-model-concurrency", type=int, default=3)
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.global_concurrency <= 80:
        raise PoweredRunnerError("global concurrency must be in [1, 80]")
    if not 1 <= args.per_model_concurrency <= 8:
        raise PoweredRunnerError("per-model concurrency must be in [1, 8]")
    if args.max_cells is not None and args.max_cells <= 0:
        raise PoweredRunnerError("max cells must be positive")
    asyncio.run(_async_run(args))


if __name__ == "__main__":
    run()
