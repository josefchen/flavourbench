"""Render the verified current-route registry for the FlavourBench manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .current_route_registry import SCHEMA_VERSION
from .real_task_bank import sha256_json


class CurrentFrontierPaperAssetError(RuntimeError):
    """The current-route paper assets failed their publication boundary checks."""


def _load_registry(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CurrentFrontierPaperAssetError("invalid current-route registry") from error
    if not isinstance(document, dict):
        raise CurrentFrontierPaperAssetError("current-route registry must be an object")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if document.get("artifact_sha256") != sha256_json(payload):
        raise CurrentFrontierPaperAssetError("current-route registry hash does not verify")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("official") is not False
        or document.get("rank_eligible") is not False
    ):
        raise CurrentFrontierPaperAssetError("current-route registry has an invalid status")
    counts = document.get("counts")
    models = document.get("models")
    if not isinstance(counts, Mapping) or not isinstance(models, list):
        raise CurrentFrontierPaperAssetError("current-route registry is incomplete")
    if (
        int(counts.get("models") or 0) != len(models)
        or counts.get("quality_observations") != 0
        or counts.get("rankable_comparisons") != 0
    ):
        raise CurrentFrontierPaperAssetError("current-route registry contains quality data")
    for model in models:
        if (
            not isinstance(model, Mapping)
            or int(model.get("quality_observations") or 0) != 0
            or int(model.get("rankable_comparisons") or 0) != 0
            or model.get("official") is not False
            or model.get("rank_eligible") is not False
        ):
            raise CurrentFrontierPaperAssetError("current-route model crossed claim boundary")
    return document


def _tex_text(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _short_provider(model: Mapping[str, Any]) -> str:
    provider = str(model.get("actual_provider") or "")
    endpoint = str(model.get("provider_endpoint") or "")
    if endpoint == "cohere-direct":
        return "Cohere direct"
    if provider:
        suffixes = {
            "openai/flex": " flex",
            "google-ai-studio/flex": " flex",
            "xai/zdr": " ZDR",
            "moonshotai/mxfp4": " MXFP4",
            "deepinfra/fp4": " FP4",
            "cloudflare": "",
            "morph": "",
            "together": "",
            "mistral": "",
            "alibaba/fp8": " FP8",
            "anthropic": "",
        }
        return provider + suffixes.get(endpoint, "")
    return endpoint


def _paper_display_name(model: Mapping[str, Any]) -> str:
    """Use compact labels in the paper table without changing registry identity."""
    compact_names = {
        "OpenAI GPT-5.6 Sol (pro mode)": "GPT-5.6 Sol (pro mode)",
        "Anthropic Claude Fable 5": "Claude Fable 5",
        "Anthropic Claude Opus 5": "Claude Opus 5",
        "Anthropic Claude Sonnet 5": "Claude Sonnet 5",
        "Google Gemini 3.1 Pro Preview": "Gemini 3.1 Pro (preview)",
        "Google Gemini 3.6 Flash": "Gemini 3.6 Flash",
        "xAI Grok 4.5": "Grok 4.5",
        "Cohere Command A+": "Command A+",
        "MoonshotAI Kimi K3": "Kimi K3",
        "Z.AI GLM 5.2": "GLM 5.2",
        "NVIDIA Nemotron 3 Ultra": "Nemotron 3 Ultra",
    }
    display_name = str(model["display_name"])
    return compact_names.get(display_name, display_name)


def _render_table(registry: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for model in registry["models"]:
        passed = model["contract_status"] == "passed_unranked"
        status = "pass" if passed else "failed"
        canonical = str(model["canonical_model_slug"])
        if not canonical:
            canonical = str(model["requested_model_id"])
        rows.append(
            " & ".join(
                [
                    _tex_text(_paper_display_name(model)),
                    rf"\path{{{canonical}}}",
                    _tex_text(_short_provider(model)),
                    status,
                    str(int(model["provider_calls"])),
                    str(int(model["epicure_calls"])),
                    "0",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{@{}p{0.16\textwidth} X p{0.11\textwidth} l r r r@{}}",
            r"\toprule",
            (
                r"Model & Canonical provider identity & Fixed provider & Contract & "
                r"Gen. & MCP & Quality $n$ \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            "",
        ]
    )


def _render_macros(registry: Mapping[str, Any]) -> str:
    counts = registry["counts"]
    cost = float(registry["known_reconciled_cost_usd"])
    return "\n".join(
        [
            rf"\newcommand{{\CurrentRouteModelCount}}{{{int(counts['models'])}}}",
            rf"\newcommand{{\CurrentRoutePassCount}}{{{int(counts['contract_passed'])}}}",
            rf"\newcommand{{\CurrentRouteFailCount}}{{{int(counts['contract_failed'])}}}",
            rf"\newcommand{{\CurrentRouteGenerationCount}}{{{int(counts['real_provider_generations_in_passed_receipts'])}}}",
            rf"\newcommand{{\CurrentRouteEpicureCallCount}}{{{int(counts['real_epicure_calls_in_passed_receipts'])}}}",
            rf"\newcommand{{\CurrentRouteKnownCost}}{{\${cost:.3f}}}",
            rf"\newcommand{{\CurrentRouteRegistryHash}}{{{registry['artifact_sha256']}}}",
            "",
        ]
    )


def _write_csv(registry: Mapping[str, Any], output: Path) -> None:
    fieldnames = [
        "display_name",
        "requested_model_id",
        "canonical_model_slug",
        "provider_endpoint",
        "actual_provider",
        "contract_status",
        "provider_calls",
        "epicure_calls",
        "quality_observations",
        "rankable_comparisons",
        "cost_usd",
        "cost_status",
        "source_artifact_sha256",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(registry["models"])


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    registry = _load_registry(arguments.registry)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "current-frontier-contract-table.tex").write_text(
        _render_table(registry), encoding="utf-8"
    )
    (arguments.output_dir / "current-frontier-macros.tex").write_text(
        _render_macros(registry), encoding="utf-8"
    )
    _write_csv(registry, arguments.output_dir / "current-frontier-contract-table.csv")


if __name__ == "__main__":
    run()
