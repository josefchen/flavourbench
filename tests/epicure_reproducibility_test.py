from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.epicure_reproducibility import (
    ReproducibilityEvidenceError,
    build_authority_record,
    verify_evidence_chain,
)

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / "artifacts/season1/epicure-lineage"
INVENTORY = LINEAGE / (
    "epicure-recovered-runtime-inventory-"
    "70d00d933aa1340841a82a9637de8b75de380f8aeba2179beab419fb6542ab5f.json"
)
MANIFEST = LINEAGE / "reproducibility" / (
    "epicure-exact-runtime-manifest-"
    "a37e5d25f9c5f7a1ec32708b17e0301bbd88248b4c0aeacecf89579106d8edf5.json"
)
RECEIPT = LINEAGE / "reproducibility" / (
    "epicure-private-offline-rebuild-receipt-"
    "35854e9f50f8f3756ab480a6ded012e15bb2e2cc4673948923068cc9deb88255.json"
)
LOCK = ROOT / "contracts/epicure/reproducibility/runtime-linux-x86_64-cp312.lock"
SBOM = ROOT / "contracts/epicure/reproducibility/runtime-linux-x86_64-cp312.cdx.json"
VERIFIER = ROOT / "contracts/epicure/reproducibility/rebuild_verifier.py"
RECIPE = ROOT / "contracts/epicure/reproducibility/PRIVATE-REBUILD.md"
DOCKERFILE = ROOT / "contracts/epicure/reproducibility/Dockerfile.runtime"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authoritative_private_rebuild_matches_recovered_runtime() -> None:
    evidence = verify_evidence_chain(
        recovered_inventory_path=INVENTORY,
        runtime_manifest_path=MANIFEST,
        rebuild_receipt_path=RECEIPT,
    )

    receipt = evidence["receipt"]
    assert receipt["runtime_manifest"]["data"]["sha256"] == (
        "98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1"
    )
    assert receipt["runtime_manifest"]["source"]["sha256"] == (
        "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313"
    )
    observed = receipt["offline_rebuild"]["observed_runtime_environment"]["integrity"]
    rebuilt = receipt["offline_rebuild"]["rebuilt_runtime_environment"]["integrity"]
    assert observed["runtime_payload_environment_sha256"] == rebuilt[
        "runtime_payload_environment_sha256"
    ]
    assert _file_sha256(LOCK) == receipt["runtime_manifest"]["dependency_lock"]["sha256"]
    assert _file_sha256(SBOM) == receipt["runtime_manifest"]["sbom"]["sha256"]
    implementation = receipt["verification_implementation"]
    assert _file_sha256(VERIFIER) == implementation["script_sha256"]
    assert _file_sha256(RECIPE) == implementation["recipe_sha256"]
    assert _file_sha256(DOCKERFILE) == implementation["dockerfile_sha256"]


def test_authority_record_advances_only_evidenced_gates() -> None:
    evidence = verify_evidence_chain(
        recovered_inventory_path=INVENTORY,
        runtime_manifest_path=MANIFEST,
        rebuild_receipt_path=RECEIPT,
    )
    record = build_authority_record(evidence=evidence, historical_non_authoritative=[])

    assert all(record["gates_advanced"].values())
    assert not any(record["gates_still_closed"].values())
    assert record["rank_eligible"] is False
    assert record["redistributable"] is False


def test_receipt_mutation_fails_content_address_check(tmp_path: Path) -> None:
    document = json.loads(RECEIPT.read_text(encoding="utf-8"))
    document["rank_eligible"] = True
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReproducibilityEvidenceError, match="content address"):
        verify_evidence_chain(
            recovered_inventory_path=INVENTORY,
            runtime_manifest_path=MANIFEST,
            rebuild_receipt_path=mutated,
        )
