"""Render the verified frontier contract bundle as a compact LaTeX audit table."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .frontier_evidence import EXPECTED_MODELS, SCHEMA_VERSION
from .real_task_bank import sha256_json

DISPLAY_NAMES = {
    "anthropic/claude-opus-5": "Claude Opus 5",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "command-a-plus-05-2026": "Command A+",
    "command-a-reasoning-08-2025": "Command A Reasoning",
    "k3": "K3",
}
PROVIDER_NAMES = {
    "openrouter": "OpenRouter, Anthropic",
    "cohere_direct": "Cohere direct",
    "kimi_code_direct": "Kimi Code direct",
}
IDENTITY_LABELS = {
    "generation_accounting_model_and_provider": "returned model + provider",
    "response_returned_exact_catalog_model": "returned exact model",
    "authenticated_catalog_and_exact_request_only": "catalog + exact request",
}


class FrontierPaperAssetError(RuntimeError):
    """The frontier paper table could not be derived from verified evidence."""


def load_bundle(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise FrontierPaperAssetError("invalid frontier evidence bundle") from error
    if not isinstance(document, dict):
        raise FrontierPaperAssetError("frontier evidence bundle must be an object")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if document.get("artifact_sha256") != sha256_json(payload):
        raise FrontierPaperAssetError("frontier evidence content address does not verify")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("status") != "all_selected_contract_smokes_passed"
        or document.get("official") is not False
        or document.get("rank_eligible") is not False
    ):
        raise FrontierPaperAssetError("frontier evidence bundle has an invalid release status")
    models = document.get("models")
    if not isinstance(models, list) or {
        str(item.get("requested_model_id") or "") for item in models if isinstance(item, Mapping)
    } != set(EXPECTED_MODELS):
        raise FrontierPaperAssetError("frontier evidence model set is incomplete")
    return document


def render_table(bundle: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for item in bundle["models"]:
        model_id = str(item["requested_model_id"])
        cost = item.get("cost_usd")
        rendered_cost = f"\\${float(cost):.4f}" if cost is not None else "not returned"
        rows.append(
            " & ".join(
                [
                    DISPLAY_NAMES[model_id],
                    PROVIDER_NAMES[str(item["provider_lane"])],
                    IDENTITY_LABELS[str(item["verification"])],
                    f"{int(item['real_provider_calls'])}",
                    f"{int(item['real_epicure_calls'])}",
                    rendered_cost,
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{@{}l l X r r r@{}}",
            r"\toprule",
            r"Model & Route & Identity evidence & Gen. & MCP & Known cost \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            "",
        ]
    )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    bundle = load_bundle(arguments.bundle)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render_table(bundle), encoding="utf-8")


if __name__ == "__main__":
    run()
