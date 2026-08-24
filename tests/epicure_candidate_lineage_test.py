from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_candidate_lineage import verify_candidate_lineage_audit

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/season1/epicure-lineage/candidate-source-trace"


def _artifact() -> dict:
    paths = sorted(ARTIFACT_DIR.glob("epicure-candidate-training-lineage-audit-*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_candidate_source_trace_fails_closed_on_exact_lineage() -> None:
    document = _artifact()
    assert verify_candidate_lineage_audit(document)
    assert document["deployed_runtime_payload"]["embeddings_sha256"] == (
        "b27fe776a59b59d703ae24170ccfcf384b89a753fd358307d085514a4fde6f69"
    )
    assert document["candidate_private_source"]["candidate_job"]["seed"] == 42
    assert document["exact_lineage_test"]["candidate_and_deployed_statistics_match"] is False
    assert document["exact_lineage_test"]["decision"] == "candidate_rejected_as_exact_lineage"
    assert document["rank_eligible"] is False
    assert document["redistributable"] is False


def test_candidate_trace_contains_no_model_quality_observations() -> None:
    document = _artifact()
    assert document["provider_calls_made"] == 0
    assert document["epicure_network_calls_made"] == 0
    assert document["synthetic_observations"] == 0
