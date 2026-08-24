from __future__ import annotations

import json
from pathlib import Path

from flavourbench.frontier_contract_runner import select_candidates
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v16/manifest").glob("*.json"))


def test_refreshed_manifest_is_exact_and_unranked() -> None:
    document = json.loads(MANIFEST.read_text())
    assert verify_manifest_content_address(document)
    assert document["official_results_authorised"] is False
    assert document["generation_calls_made"] == 0
    assert len(select_candidates(document)) == 20
    assert document["route_refresh"]["calibration_used_as_primary_data"] is False


def test_failed_routes_are_replaced_without_fallback() -> None:
    document = json.loads(MANIFEST.read_text())
    expected = {
        "google/gemini-3.6-flash": "google-vertex/us",
        "nvidia/nemotron-3-ultra-550b-a55b": "baseten/fp4",
    }
    by_id = {row["model"]["id"]: row for row in document["models"]}
    for model_id, tag in expected.items():
        row = by_id[model_id]
        assert row["endpoint"]["tag"] == tag
        assert row["request_policy"]["provider"]["only"] == [tag]
        assert row["execution_route"]["generation_time_automatic_fallback"] is False
    assert by_id["meta-llama/llama-4-maverick"]["endpoint"]["tag"] == "digitalocean"


def test_prior_tencent_route_remains_exact() -> None:
    document = json.loads(MANIFEST.read_text())
    by_id = {row["model"]["id"]: row for row in document["models"]}
    row = by_id["tencent/hy3"]
    assert row["endpoint"]["tag"] == "atlas-cloud/fp8"
    assert row["request_policy"]["provider"]["only"] == ["atlas-cloud/fp8"]
