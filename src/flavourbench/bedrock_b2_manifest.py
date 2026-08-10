"""Freeze the provider-free Bedrock B2 common-task workload.

This module only reads local, frozen contracts and the authored task inventory.
It has no AWS/OpenRouter client and cannot perform inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .bedrock_manifest import assert_public_catalog_safe
from .tasks import CandidateTask, candidate_tasks

SCHEMA_VERSION = "flavourbench-bedrock-b2-workload-manifest-v2"
FAMILIES = ("composition", "cookability", "evidence", "substitution")
CONDITIONS = ("epicure_off", "epicure_on")
DEFAULT_TASK_IDS = (
    "sub-001",
    "sub-002",
    "comp-001",
    "comp-002",
    "cook-001",
    "cook-002",
    "evid-001",
    "evid-002",
)
MAX_B2_BLOCK_USD = Decimal("100")
EXECUTION_ADAPTER_ID = "bedrock-converse-epicure-bounded-v1"
OFF_SYSTEM_PROMPT = (
    "You are completing a blinded culinary benchmark without Epicure. Use only your own "
    "reasoning. Return only the required structured answer object. Do not claim that an "
    "external culinary tool or database was consulted."
)
ON_SYSTEM_PROMPT = (
    "You are completing a blinded culinary benchmark with Epicure. Call at least one relevant "
    "Epicure tool before answering. Use its evidence, distinguish evidence from culinary "
    "judgment and tasting uncertainty, and return only the required structured answer object."
)
ALLOWED_EPICURE_TOOLS = (
    "compare_on_axis",
    "pairing_score",
    "find_pairings",
    "flavour_correlations",
    "neighbors",
    "morph",
    "list_targets",
    "list_factors",
    "ingredient_on_factor",
    "pareto_navigate",
    "closest_mode",
    "where_on_atlas",
)
FINAL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_markdown": {"type": "string"},
        "ingredient_mentions": {"type": "array", "items": {"type": "string"}},
        "constraints_addressed": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "answer_markdown",
        "ingredient_mentions",
        "constraints_addressed",
        "uncertainties",
    ],
    "additionalProperties": False,
}
_FORBIDDEN_MODEL_MARKER = re.compile(
    r"(?:^|[/_.:-])(?:mock|fixture|test)(?:$|[/_.:-])", re.IGNORECASE
)


class BedrockB2ManifestError(ValueError):
    """A B2 workload cannot be frozen without an exact, safe contract."""


class BedrockB2BudgetExceeded(BedrockB2ManifestError):
    """The whole B2 workload exceeds its immutable admission ceiling."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _money(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise BedrockB2ManifestError(f"{field} must be a decimal") from error
    if not result.is_finite() or result < 0:
        raise BedrockB2ManifestError(f"{field} must be finite and non-negative")
    return result


def _money_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _load_json_file(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path)
    if resolved.is_symlink() or not resolved.is_file():
        raise BedrockB2ManifestError(f"missing {label}: {resolved}")
    try:
        value = json.loads(resolved.read_bytes())
    except json.JSONDecodeError as error:
        raise BedrockB2ManifestError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(value, dict):
        raise BedrockB2ManifestError(f"{label} must be a JSON object: {resolved}")
    assert_public_catalog_safe(value, path=f"${label}")
    return resolved, value, _file_sha256(resolved)


@dataclass(frozen=True)
class B2ForecastPolicy:
    """Whole-block upper bounds; token prices are per one million tokens."""

    max_off_generations: int = 1
    max_on_generations: int = 9
    max_input_tokens_per_generation: int = 8_000
    max_output_tokens_per_generation: int = 2_048

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BedrockB2ManifestError(f"{name} must be a positive integer")


DEFAULT_FORECAST_POLICY = B2ForecastPolicy()


def _validate_tasks(tasks: Sequence[CandidateTask]) -> list[CandidateTask]:
    if len(tasks) != 8:
        raise BedrockB2ManifestError("B2 requires exactly eight common tasks")
    task_ids = [task.public_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise BedrockB2ManifestError("duplicate task IDs are forbidden")
    prompts = [task.prompt_sha256 for task in tasks]
    if len(set(prompts)) != len(prompts):
        raise BedrockB2ManifestError("duplicate task prompts are forbidden")
    coverage = Counter(task.family for task in tasks)
    if coverage != Counter({family: 2 for family in FAMILIES}):
        raise BedrockB2ManifestError("B2 requires exactly two tasks from each of four families")
    return sorted(tasks, key=lambda item: (item.family, item.public_id))


def select_tasks(task_ids: Sequence[str]) -> list[CandidateTask]:
    inventory = {task.public_id: task for task in candidate_tasks()}
    if len(set(task_ids)) != len(task_ids):
        raise BedrockB2ManifestError("duplicate task IDs are forbidden")
    missing = sorted(set(task_ids) - inventory.keys())
    if missing:
        raise BedrockB2ManifestError(f"unknown task IDs: {', '.join(missing)}")
    return _validate_tasks([inventory[task_id] for task_id in task_ids])


def _endpoint_manifest(document: Mapping[str, Any]) -> Mapping[str, Any]:
    embedded = document.get("endpoint_manifest")
    if isinstance(embedded, Mapping):
        claimed = document.get("manifest_sha256")
        payload = dict(document)
        payload.pop("manifest_sha256", None)
        if not isinstance(claimed, str) or _sha256(payload) != claimed:
            raise BedrockB2ManifestError("endpoint wrapper has an invalid content digest")
        return embedded
    return document


def _load_endpoint_contracts(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    if not paths:
        raise BedrockB2ManifestError("at least one endpoint contract is required")
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path, document, file_digest = _load_json_file(raw_path, label="endpoint contract")
        endpoint_manifest = _endpoint_manifest(document)
        if (
            endpoint_manifest.get("official") is not False
            or endpoint_manifest.get("rank_eligible") is not False
        ):
            raise BedrockB2ManifestError("B2 endpoint contracts must remain unofficial/unranked")
        contracts = endpoint_manifest.get("contracts")
        if not isinstance(contracts, list) or not contracts:
            raise BedrockB2ManifestError(f"endpoint contract contains no model contracts: {path}")
        endpoint_manifest_digest = _sha256(endpoint_manifest)
        for contract in contracts:
            if not isinstance(contract, Mapping):
                raise BedrockB2ManifestError("endpoint model contract must be an object")
            model_id = str(contract.get("canonical_model_id") or "")
            if not model_id:
                raise BedrockB2ManifestError("endpoint model contract has no canonical model ID")
            if _FORBIDDEN_MODEL_MARKER.search(model_id):
                raise BedrockB2ManifestError(f"mock/fixture model ID is forbidden: {model_id}")
            if model_id in seen:
                raise BedrockB2ManifestError(f"duplicate canonical model: {model_id}")
            if not all(
                contract.get(field) is True
                for field in (
                    "supports_converse",
                    "supports_tool_use",
                    "supports_structured_output",
                )
            ):
                raise BedrockB2ManifestError(f"incomplete endpoint capability contract: {model_id}")
            price = contract.get("price")
            if not isinstance(price, Mapping):
                raise BedrockB2ManifestError(f"missing price contract: {model_id}")
            input_price = _money(price.get("input_per_million_usd"), field="input price")
            output_price = _money(price.get("output_per_million_usd"), field="output price")
            seen.add(model_id)
            models.append(
                {
                    "canonical_model_id": model_id,
                    "bedrock_target_id": contract.get("bedrock_target_id"),
                    "expected_foundation_model_ids": contract.get("expected_foundation_model_ids"),
                    "endpoint_contract_reference": {
                        "filename": path.name,
                        "file_sha256": file_digest,
                        "endpoint_manifest_sha256": endpoint_manifest_digest,
                        "endpoint_contract_sha256": _sha256(contract),
                    },
                    "price_contract": {
                        "input_per_million_usd": _money_text(input_price),
                        "output_per_million_usd": _money_text(output_price),
                        "price_sha256": price.get("price_sha256"),
                    },
                }
            )
    return sorted(models, key=lambda item: item["canonical_model_id"])


def _local_contract_reference(path: str | Path, *, label: str) -> dict[str, str]:
    resolved = Path(path)
    if resolved.is_symlink() or not resolved.is_file():
        raise BedrockB2ManifestError(f"missing {label}: {resolved}")
    try:
        document = json.loads(resolved.read_bytes())
    except json.JSONDecodeError as error:
        raise BedrockB2ManifestError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(document, dict | list):
        raise BedrockB2ManifestError(f"{label} must be a JSON object or array: {resolved}")
    assert_public_catalog_safe(document, path=f"${label}")
    return {
        "filename": resolved.name,
        "file_sha256": _file_sha256(resolved),
        "document_sha256": _sha256(document),
    }


def build_b2_manifest(
    *,
    endpoint_contract_paths: Sequence[str | Path],
    epicure_contract_path: str | Path,
    tool_contract_path: str | Path,
    tasks: Sequence[CandidateTask],
    forecast_policy: B2ForecastPolicy = DEFAULT_FORECAST_POLICY,
    cap_usd: str | Decimal = "100",
    frozen_at: str | None = None,
) -> dict[str, Any]:
    """Build a complete Cartesian B2 plan without contacting any provider."""

    selected_tasks = _validate_tasks(tasks)
    models = _load_endpoint_contracts(endpoint_contract_paths)
    cap = _money(cap_usd, field="B2 cap")
    if cap > MAX_B2_BLOCK_USD:
        raise BedrockB2ManifestError("B2 cap cannot exceed USD 100")
    epicure_reference = _local_contract_reference(
        epicure_contract_path, label="Epicure lineage contract"
    )
    tool_reference = _local_contract_reference(tool_contract_path, label="Epicure tool contract")

    total = Decimal("0")
    for model in models:
        price = model["price_contract"]
        per_generation = (
            Decimal(forecast_policy.max_input_tokens_per_generation)
            * Decimal(price["input_per_million_usd"])
            + Decimal(forecast_policy.max_output_tokens_per_generation)
            * Decimal(price["output_per_million_usd"])
        ) / Decimal("1000000")
        per_pair = per_generation * Decimal(
            forecast_policy.max_off_generations + forecast_policy.max_on_generations
        )
        model_total = per_pair * Decimal(len(selected_tasks))
        model["worst_case_cost_usd"] = _money_text(model_total)
        total += model_total
    if total > cap or total > MAX_B2_BLOCK_USD:
        raise BedrockB2BudgetExceeded(
            f"whole-block forecast ${_money_text(total)} exceeds cap ${_money_text(cap)}"
        )

    arms = [
        {
            "arm_id": _sha256(
                {
                    "model": model["canonical_model_id"],
                    "task": task.public_id,
                    "condition": condition,
                }
            ),
            "canonical_model_id": model["canonical_model_id"],
            "endpoint_contract_sha256": model["endpoint_contract_reference"][
                "endpoint_contract_sha256"
            ],
            "task_id": task.public_id,
            "prompt_sha256": task.prompt_sha256,
            "family": task.family,
            "condition": condition,
        }
        for model in models
        for task in selected_tasks
        for condition in CONDITIONS
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "frozen_at": frozen_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "stage": "bedrock_b2_compatibility_pilot",
        "official": False,
        "rank_eligible": False,
        "provider_calls_made": 0,
        "common_task_design": True,
        "contracts": {
            "epicure_lineage": epicure_reference,
            "epicure_tool_catalog": tool_reference,
        },
        "forecast_policy": asdict(forecast_policy),
        "execution_contract": {
            "adapter_id": EXECUTION_ADAPTER_ID,
            "off_system_prompt": OFF_SYSTEM_PROMPT,
            "off_system_prompt_sha256": hashlib.sha256(
                OFF_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "on_system_prompt": ON_SYSTEM_PROMPT,
            "on_system_prompt_sha256": hashlib.sha256(ON_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "decoding": {"temperature": "0.2", "top_p": None},
            "response_schema": FINAL_RESPONSE_SCHEMA,
            "response_schema_sha256": _sha256(FINAL_RESPONSE_SCHEMA),
            "allowed_epicure_tools": list(ALLOWED_EPICURE_TOOLS),
            "tool_projection": "aws-draft-2020-12-supported-subset-v1",
            "max_tool_rounds": 8,
            "max_tool_calls_per_round": 4,
            "max_tool_calls_total": 16,
            "max_tool_result_bytes": 65_536,
            "max_cumulative_tool_result_bytes": 131_072,
            "retry_policy": "no-automatic-retry-after-paid-request-boundary",
            "delivery_policy": "fail-closed-hold-whole-block-on-ambiguous-delivery",
        },
        "budget": {
            "currency": "USD",
            "cap_usd": _money_text(cap),
            "whole_block_worst_case_usd": _money_text(total),
            "reservation_scope": "entire_manifest_before_first_provider_call",
        },
        "tasks": [
            {
                "task_id": task.public_id,
                "family": task.family,
                "prompt": task.prompt,
                "prompt_sha256": task.prompt_sha256,
                "split": task.split,
                "review_status": task.review_status,
            }
            for task in selected_tasks
        ],
        "models": models,
        "arms": arms,
        "counts": {
            "models": len(models),
            "tasks": len(selected_tasks),
            "arms": len(arms),
            "arms_per_model": len(selected_tasks) * len(CONDITIONS),
        },
    }
    digest = _sha256(payload)
    document = {
        **payload,
        "content_address": {"algorithm": "sha256-canonical-json", "digest": digest},
    }
    assert_public_catalog_safe(document, path="$bedrock_b2_manifest")
    return document


def verify_b2_manifest_content_address(document: Mapping[str, Any]) -> bool:
    address = document.get("content_address")
    if not isinstance(address, Mapping) or address.get("algorithm") != "sha256-canonical-json":
        return False
    payload = dict(document)
    payload.pop("content_address", None)
    return address.get("digest") == _sha256(payload)


def write_b2_manifest(document: Mapping[str, Any], output_directory: str | Path) -> Path:
    if not verify_b2_manifest_content_address(document):
        raise BedrockB2ManifestError("refusing to write an invalid content address")
    digest = str(document["content_address"]["digest"])
    destination = Path(output_directory) / f"bedrock-b2-workload-{digest}.json"
    rendered = _canonical_json(document) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    if destination.exists():
        if destination.read_bytes() != rendered:
            temporary.unlink()
            raise BedrockB2ManifestError("content-addressed B2 manifest conflicts")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the provider-free Bedrock B2 workload")
    parser.add_argument("--endpoint-contract", action="append", type=Path, required=True)
    parser.add_argument("--epicure-contract", type=Path, required=True)
    parser.add_argument("--tool-contract", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--cap-usd", default="100")
    parser.add_argument("--frozen-at")
    parser.add_argument(
        "--output-directory", type=Path, default=Path("artifacts/bedrock/b2/manifests")
    )
    arguments = parser.parse_args(argv)
    manifest = build_b2_manifest(
        endpoint_contract_paths=arguments.endpoint_contract,
        epicure_contract_path=arguments.epicure_contract,
        tool_contract_path=arguments.tool_contract,
        tasks=select_tasks(arguments.task_id or DEFAULT_TASK_IDS),
        cap_usd=arguments.cap_usd,
        frozen_at=arguments.frozen_at,
    )
    path = write_b2_manifest(manifest, arguments.output_directory)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
