"""Assemble a dated, unranked registry of current provider and Epicure contracts.

The registry deliberately contains no culinary quality measurement. It combines
content-addressed contract receipts from several provider lanes and preserves a
failed exact route as a compatibility result. It cannot be consumed as a season
manifest or a leaderboard input.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-current-route-registry-v1"

REGISTRY_ORDER = (
    ("openai/gpt-5.6-sol-pro", "season1", "OpenAI GPT-5.6 Sol (pro mode)"),
    ("anthropic/claude-fable-5", "refresh", "Anthropic Claude Fable 5"),
    ("anthropic/claude-opus-5", "season1", "Anthropic Claude Opus 5"),
    ("anthropic/claude-sonnet-5", "season1", "Anthropic Claude Sonnet 5"),
    ("google/gemini-3.1-pro-preview", "season1", "Google Gemini 3.1 Pro Preview"),
    ("google/gemini-3.6-flash", "refresh", "Google Gemini 3.6 Flash"),
    ("x-ai/grok-4.5", "refresh", "xAI Grok 4.5"),
    ("command-a-plus-05-2026", "cohere", "Cohere Command A+"),
    ("moonshotai/kimi-k3", "season1", "MoonshotAI Kimi K3"),
    ("z-ai/glm-5.2", "season1", "Z.AI GLM 5.2"),
    ("deepseek/deepseek-v4-pro", "season1", "DeepSeek V4 Pro"),
    (
        "deepseek/deepseek-v4-flash-0731",
        "refresh",
        "DeepSeek V4 Flash 0731",
    ),
    ("qwen/qwen3.7-max", "refresh", "Qwen 3.7 Max"),
    ("minimax/minimax-m3", "alternate", "MiniMax M3"),
    (
        "nvidia/nemotron-3-ultra-550b-a55b",
        "refresh",
        "NVIDIA Nemotron 3 Ultra",
    ),
    (
        "mistralai/mistral-medium-3-5",
        "refresh",
        "Mistral Medium 3.5",
    ),
)


class CurrentRouteRegistryError(RuntimeError):
    """The current-route registry could not be verified or assembled."""


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CurrentRouteRegistryError(f"input must be a regular file: {path}")
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CurrentRouteRegistryError(f"invalid JSON input: {path}") from error
    if not isinstance(document, dict):
        raise CurrentRouteRegistryError(f"expected a JSON object: {path}")
    return document


def _verified(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    document = _load_object(path)
    recorded = str(document.get("artifact_sha256") or "")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    actual = sha256_json(payload)
    if recorded != actual or (expected_sha256 is not None and actual != expected_sha256):
        raise CurrentRouteRegistryError(f"content address does not verify: {path}")
    return document


def _atomic_write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"current-route-registry-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise CurrentRouteRegistryError("content-addressed registry conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _trace_passed(artifact: Mapping[str, Any]) -> bool:
    trace = artifact.get("complete_epicure_trace")
    return bool(
        isinstance(trace, list)
        and len(trace) == 1
        and isinstance(trace[0], Mapping)
        and trace[0].get("name") == "find_pairings"
        and trace[0].get("is_error") is False
        and str(trace[0].get("result_sha256") or "")
    )


def _passed_record(
    *,
    display_name: str,
    model_id: str,
    canonical_slug: str,
    provider_endpoint: str,
    actual_provider: str,
    identity_basis: str,
    artifact: Mapping[str, Any],
    artifact_path: Path,
    artifact_sha256: str,
    cost_usd: str | None,
    cost_status: str,
    response_returned_model: bool,
) -> dict[str, Any]:
    provider_calls = int(artifact.get("real_provider_calls") or artifact.get("provider_calls") or 0)
    epicure_calls = int(artifact.get("real_epicure_calls") or 0)
    output = artifact.get("output_json")
    if (
        artifact.get("status") != "smoke_passed"
        or provider_calls != 2
        or epicure_calls != 1
        or not _trace_passed(artifact)
        or not isinstance(output, Mapping)
        or not str(output.get("answer_markdown") or "").strip()
    ):
        raise CurrentRouteRegistryError(f"passed contract receipt is incomplete: {model_id}")
    return {
        "display_name": display_name,
        "requested_model_id": model_id,
        "canonical_model_slug": canonical_slug,
        "provider_endpoint": provider_endpoint,
        "actual_provider": actual_provider,
        "identity_basis": identity_basis,
        "response_returned_model": response_returned_model,
        "contract_status": "passed_unranked",
        "provider_calls": provider_calls,
        "epicure_calls": epicure_calls,
        "structured_final_contract": True,
        "normal_finish_recorded": artifact.get("finish_reason") in {None, "stop"},
        "wall_clock_latency_ms": int(artifact.get("wall_clock_latency_ms") or 0),
        "cost_usd": cost_usd,
        "cost_status": cost_status,
        "source_artifact_path": str(artifact_path),
        "source_artifact_sha256": artifact_sha256,
        "quality_observations": 0,
        "rankable_comparisons": 0,
        "official": False,
        "rank_eligible": False,
    }


def _summary_index(summary_path: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    summary = _verified(summary_path)
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list):
        raise CurrentRouteRegistryError("contract summary has no artifact list")
    indexed: dict[str, tuple[dict[str, Any], Path]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise CurrentRouteRegistryError("contract summary contains a non-object artifact")
        model_id = str(item.get("requested_endpoint_id") or "")
        path = Path(str(item.get("artifact_path") or ""))
        expected = str(item.get("artifact_sha256") or "")
        if not model_id or model_id in indexed:
            raise CurrentRouteRegistryError("contract summary model IDs are missing or duplicated")
        indexed[model_id] = (_verified(path, expected), path)
    return indexed


def build_registry(
    *,
    season1_panel_path: Path,
    refresh_summary_path: Path,
    alternate_summary_path: Path,
    frontier_bundle_path: Path,
) -> dict[str, Any]:
    season1 = _verified(season1_panel_path)
    if season1.get("status") != "contract_qualified_candidate_not_season_frozen":
        raise CurrentRouteRegistryError("unexpected Season 1 panel status")
    season1_models = season1.get("models")
    if not isinstance(season1_models, list):
        raise CurrentRouteRegistryError("Season 1 panel has no model list")
    season1_index = {
        str(item.get("requested_endpoint_id") or ""): item
        for item in season1_models
        if isinstance(item, Mapping)
    }
    refresh_index = _summary_index(refresh_summary_path)
    alternate_index = _summary_index(alternate_summary_path)
    frontier = _verified(frontier_bundle_path)
    frontier_models = frontier.get("models")
    if not isinstance(frontier_models, list):
        raise CurrentRouteRegistryError("frontier evidence bundle has no model list")
    cohere_index = {
        str(item.get("requested_model_id") or ""): item
        for item in frontier_models
        if isinstance(item, Mapping) and item.get("provider_lane") == "cohere_direct"
    }

    records: list[dict[str, Any]] = []
    source_artifacts: set[str] = set()
    for model_id, source, display_name in REGISTRY_ORDER:
        if source == "season1":
            item = season1_index.get(model_id)
            if not isinstance(item, Mapping):
                raise CurrentRouteRegistryError(f"Season 1 receipt is missing: {model_id}")
            path = Path(str(item.get("smoke_artifact_path") or ""))
            expected = str(item.get("smoke_artifact_sha256") or "")
            artifact = _verified(path, expected)
            record = _passed_record(
                display_name=display_name,
                model_id=model_id,
                canonical_slug=str(item.get("canonical_model_slug") or ""),
                provider_endpoint=str(item.get("provider_endpoint") or ""),
                actual_provider=str(item.get("actual_provider") or ""),
                identity_basis=str(item.get("identity_basis") or ""),
                artifact=artifact,
                artifact_path=path,
                artifact_sha256=expected,
                cost_usd=(str(item["cost_usd"]) if item.get("cost_usd") is not None else None),
                cost_status=str(item.get("cost_status") or ""),
                response_returned_model=True,
            )
        elif source in {"refresh", "alternate"}:
            indexed = refresh_index if source == "refresh" else alternate_index
            pair = indexed.get(model_id)
            if pair is None:
                raise CurrentRouteRegistryError(f"refresh receipt is missing: {model_id}")
            artifact, path = pair
            expected = str(artifact.get("artifact_sha256") or "")
            if artifact.get("status") == "smoke_passed":
                record = _passed_record(
                    display_name=display_name,
                    model_id=model_id,
                    canonical_slug=str(artifact.get("canonical_slug") or ""),
                    provider_endpoint=str(artifact.get("requested_provider_slug") or ""),
                    actual_provider=str(artifact.get("returned_provider_name") or ""),
                    identity_basis="reconciled_generation_metadata_model_and_provider",
                    artifact=artifact,
                    artifact_path=path,
                    artifact_sha256=expected,
                    cost_usd=str(artifact.get("cost_usd") or "0"),
                    cost_status="openrouter_generation_metadata_reconciled",
                    response_returned_model=True,
                )
            else:
                record = {
                    "display_name": display_name,
                    "requested_model_id": model_id,
                    "canonical_model_slug": str(artifact.get("canonical_slug") or ""),
                    "provider_endpoint": str(artifact.get("requested_provider_slug") or ""),
                    "actual_provider": None,
                    "identity_basis": "catalog_and_exact_failed_request",
                    "response_returned_model": False,
                    "contract_status": "failed_pre_generation",
                    "failure_type": str(artifact.get("error_type") or "unknown"),
                    "provider_calls": 0,
                    "epicure_calls": 0,
                    "structured_final_contract": False,
                    "normal_finish_recorded": False,
                    "wall_clock_latency_ms": 0,
                    "cost_usd": None,
                    "cost_status": "no_reconciled_cost",
                    "source_artifact_path": str(path),
                    "source_artifact_sha256": expected,
                    "quality_observations": 0,
                    "rankable_comparisons": 0,
                    "official": False,
                    "rank_eligible": False,
                }
        elif source == "cohere":
            item = cohere_index.get(model_id)
            if not isinstance(item, Mapping):
                raise CurrentRouteRegistryError(f"Cohere receipt is missing: {model_id}")
            path = Path(str(item.get("artifact_path") or ""))
            expected = str(item.get("artifact_sha256") or "")
            artifact = _verified(path, expected)
            record = _passed_record(
                display_name=display_name,
                model_id=model_id,
                canonical_slug=str(item.get("canonical_model_id") or ""),
                provider_endpoint="cohere-direct",
                actual_provider=str(item.get("actual_provider") or ""),
                identity_basis=str(item.get("verification") or ""),
                artifact=artifact,
                artifact_path=path,
                artifact_sha256=expected,
                cost_usd=None,
                cost_status=str(item.get("cost_status") or ""),
                response_returned_model=bool(item.get("response_returned_model")),
            )
        else:  # pragma: no cover - constant table is closed
            raise CurrentRouteRegistryError(f"unknown registry source: {source}")
        source_sha = str(record["source_artifact_sha256"])
        if source_sha in source_artifacts:
            raise CurrentRouteRegistryError("source contract artifacts must be unique")
        source_artifacts.add(source_sha)
        records.append(record)

    if len(records) != len(REGISTRY_ORDER):
        raise CurrentRouteRegistryError("current route registry is incomplete")
    known_cost = sum(
        (Decimal(str(item["cost_usd"])) for item in records if item["cost_usd"] is not None),
        Decimal(0),
    )
    passed = [item for item in records if item["contract_status"] == "passed_unranked"]
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": "2026-08-01",
        "scope": "current_route_qualification_not_quality_measurement",
        "status": "current_registry_with_one_failed_exact_route",
        "selection_basis": (
            "Current or newly omitted model-family coverage, selected without culinary "
            "quality observations. Registry order is not a quality order."
        ),
        "counts": {
            "models": len(records),
            "contract_passed": len(passed),
            "contract_failed": len(records) - len(passed),
            "real_provider_generations_in_passed_receipts": sum(
                int(item["provider_calls"]) for item in passed
            ),
            "real_epicure_calls_in_passed_receipts": sum(
                int(item["epicure_calls"]) for item in passed
            ),
            "quality_observations": 0,
            "rankable_comparisons": 0,
        },
        "known_reconciled_cost_usd": format(known_cost, "f"),
        "cost_scope": (
            "Passed receipt costs where the provider returned price metadata. "
            "Cohere cost and pre-receipt execution incidents are excluded."
        ),
        "sources": {
            "season1_candidate_panel_sha256": str(season1["artifact_sha256"]),
            "refresh_summary_sha256": str(_verified(refresh_summary_path)["artifact_sha256"]),
            "alternate_summary_sha256": str(_verified(alternate_summary_path)["artifact_sha256"]),
            "frontier_bundle_sha256": str(frontier["artifact_sha256"]),
        },
        "models": records,
        "interpretation": {
            "proves_endpoint_compatibility": True,
            "proves_culinary_quality": False,
            "supports_leaderboard": False,
            "supports_epicure_uplift": False,
        },
        "official": False,
        "rank_eligible": False,
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season1-panel", type=Path, required=True)
    parser.add_argument("--refresh-summary", type=Path, required=True)
    parser.add_argument("--alternate-summary", type=Path, required=True)
    parser.add_argument("--frontier-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = build_registry(
        season1_panel_path=arguments.season1_panel,
        refresh_summary_path=arguments.refresh_summary,
        alternate_summary_path=arguments.alternate_summary,
        frontier_bundle_path=arguments.frontier_bundle,
    )
    path = _atomic_write(arguments.output_dir, payload)
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "counts": payload["counts"],
                "known_reconciled_cost_usd": payload["known_reconciled_cost_usd"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
