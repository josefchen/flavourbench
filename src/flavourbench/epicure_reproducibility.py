"""Verify and index Epicure's private runtime reconstruction evidence.

The supplement advances only dependency, SBOM, and private rebuild gates. It
cannot supersede unresolved training lineage, data rights, public payload, OCI,
or independent-reproduction gates in the recovered runtime inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

INVENTORY_SCHEMA = "epicure-recovered-runtime-inventory-v2"
MANIFEST_SCHEMA = "epicure-exact-runtime-manifest-v1"
RECEIPT_SCHEMA = "epicure-private-offline-rebuild-receipt-v1"
AUTHORITY_SCHEMA = "epicure-runtime-reconstruction-authority-v1"


class ReproducibilityEvidenceError(RuntimeError):
    """A reconstruction artifact is malformed or crosses its evidence boundary."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _verified_document(path: Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReproducibilityEvidenceError(f"artifact must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ReproducibilityEvidenceError(f"artifact has the wrong schema: {path}")
    digest = value.get("artifact_sha256")
    unhashed = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if not isinstance(digest, str) or digest != _sha256(unhashed):
        raise ReproducibilityEvidenceError(f"artifact content address is invalid: {path}")
    return value


def verify_evidence_chain(
    *,
    recovered_inventory_path: Path,
    runtime_manifest_path: Path,
    rebuild_receipt_path: Path,
) -> dict[str, Any]:
    """Verify the supplement matches the existing opaque runtime identity."""

    inventory = _verified_document(recovered_inventory_path, INVENTORY_SCHEMA)
    manifest = _verified_document(runtime_manifest_path, MANIFEST_SCHEMA)
    receipt = _verified_document(rebuild_receipt_path, RECEIPT_SCHEMA)
    embedded_manifest = receipt.get("runtime_manifest")
    manifest_unhashed = {
        key: value for key, value in manifest.items() if key != "artifact_sha256"
    }
    if embedded_manifest != manifest_unhashed:
        raise ReproducibilityEvidenceError(
            "rebuild receipt does not embed the exact runtime manifest"
        )
    if manifest["data"]["sha256"] != inventory["bundle"]["sha256"]:
        raise ReproducibilityEvidenceError("runtime manifest data differs from inventory")
    if manifest["source"]["sha256"] != inventory["application"]["sha256"]:
        raise ReproducibilityEvidenceError("runtime manifest source differs from inventory")
    recovered = receipt.get("recovered_inventory")
    if not isinstance(recovered, Mapping) or recovered.get("artifact_sha256") != inventory[
        "artifact_sha256"
    ]:
        raise ReproducibilityEvidenceError("receipt does not bind the recovered inventory")

    gates = receipt.get("release_gates")
    required_true = {
        "exact_source_and_data_manifest",
        "hash_locked_platform_runtime",
        "machine_readable_sbom",
        "private_offline_runtime_rebuild",
    }
    required_false = {
        "independent_reproduction",
        "immutable_oci_identity",
        "training_lineage_recovered",
        "payload_rights_attested",
        "public_redistributable_payload",
    }
    if not isinstance(gates, Mapping):
        raise ReproducibilityEvidenceError("receipt has no release-gate object")
    if any(gates.get(gate) is not True for gate in required_true):
        raise ReproducibilityEvidenceError("receipt overstates a resolved runtime gate")
    if any(gates.get(gate) is not False for gate in required_false):
        raise ReproducibilityEvidenceError("receipt crosses an unresolved release gate")
    if receipt.get("rank_eligible") is not False or receipt.get("redistributable") is not False:
        raise ReproducibilityEvidenceError("receipt must remain unranked and non-redistributable")
    implementation = receipt.get("verification_implementation")
    if not isinstance(implementation, Mapping) or any(
        not isinstance(implementation.get(field), str)
        or len(implementation[field]) != 64
        for field in ("script_sha256", "recipe_sha256", "dockerfile_sha256")
    ):
        raise ReproducibilityEvidenceError("receipt does not bind its verification materials")
    if implementation.get("dockerfile_base_image_content_pinned") is not False:
        raise ReproducibilityEvidenceError("receipt overstates the Docker base-image identity")

    rebuild = receipt.get("offline_rebuild")
    if not isinstance(rebuild, Mapping):
        raise ReproducibilityEvidenceError("receipt has no offline-rebuild evidence")
    install = rebuild.get("dependency_install")
    if not isinstance(install, Mapping) or not all(
        install.get(field) is True
        for field in (
            "all_locked_versions_matched",
            "physical_runtime_payload_manifests_match_observed_environment",
        )
    ):
        raise ReproducibilityEvidenceError("offline rebuild did not match the observed runtime")
    observed = rebuild.get("observed_runtime_environment", {}).get("integrity", {})
    rebuilt = rebuild.get("rebuilt_runtime_environment", {}).get("integrity", {})
    if observed.get("runtime_payload_environment_sha256") != rebuilt.get(
        "runtime_payload_environment_sha256"
    ):
        raise ReproducibilityEvidenceError("observed and rebuilt runtime payloads differ")
    return {
        "inventory": inventory,
        "manifest": manifest,
        "receipt": receipt,
    }


def build_authority_record(
    *,
    evidence: Mapping[str, Any],
    historical_non_authoritative: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    inventory = evidence["inventory"]
    manifest = evidence["manifest"]
    receipt = evidence["receipt"]
    return {
        "schema_version": AUTHORITY_SCHEMA,
        "record_role": "authoritative_runtime_reconstruction_supplement_pointer",
        "opaque_runtime_identity": {
            "runtime_id": inventory["runtime_id"],
            "bundle_sha256": inventory["bundle"]["sha256"],
            "application_sha256": inventory["application"]["sha256"],
            "recovered_inventory_sha256": inventory["artifact_sha256"],
        },
        "authoritative_supplement": {
            "runtime_manifest_sha256": manifest["artifact_sha256"],
            "private_offline_rebuild_receipt_sha256": receipt["artifact_sha256"],
            "dependency_lock_sha256": manifest["dependency_lock"]["sha256"],
            "sbom_sha256": manifest["sbom"]["sha256"],
            "observed_and_rebuilt_runtime_payload_sha256": receipt["offline_rebuild"][
                "observed_runtime_environment"
            ]["integrity"]["runtime_payload_environment_sha256"],
        },
        "historical_non_authoritative": list(historical_non_authoritative),
        "gates_advanced": {
            "exact_source_and_data_manifest": True,
            "hash_locked_observed_runtime_dependencies": True,
            "cyclonedx_sbom": True,
            "private_offline_runtime_rebuild": True,
        },
        "gates_still_closed": {
            "training_lineage_recovered": False,
            "payload_rights_attested": False,
            "public_redistributable_payload": False,
            "immutable_oci_identity": False,
            "independent_reproduction": False,
            "clean_signed_application_release": False,
        },
        "status": "private_runtime_reconstruction_verified_release_blocked",
        "rank_eligible": False,
        "redistributable": False,
        "provider_calls_made": False,
        "epicure_network_calls_made": False,
        "interpretation_rule": (
            "Use the authoritative supplement for dependency, SBOM, and private rebuild claims. "
            "Continue using the recovered inventory for the opaque runtime identity. Do not use "
            "either artifact to claim recovered training lineage, public redistribution rights, "
            "an immutable OCI release, independent reproduction, or official rank eligibility."
        ),
    }


def _write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = _sha256(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"epicure-runtime-reconstruction-authority-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise ReproducibilityEvidenceError("content-addressed authority conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-inventory", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--rebuild-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    evidence = verify_evidence_chain(
        recovered_inventory_path=arguments.recovered_inventory,
        runtime_manifest_path=arguments.runtime_manifest,
        rebuild_receipt_path=arguments.rebuild_receipt,
    )
    historical = [
        {
            "artifact_sha256": "0735af49c5bee697267b909bad17679c0ade587fa3f3dd8d9eb02a4888308412",
            "role": "prospective_dependency_runtime_manifest",
            "reason": "resolver selected versions newer than the observed loopback environment",
        },
        {
            "artifact_sha256": "602f9d31788157f469298ef50e97a20b31999935680e1092f67197a487724c76",
            "role": "prospective_dependency_rebuild_receipt",
            "reason": (
                "verified a clean environment but did not match or physically bind the observed "
                "loopback dependency environment"
            ),
        },
        {
            "artifact_sha256": "4230d7a5bb1a4008e638d7f2199c86554972266bf3c26bdde0d1bfbca3bbe584",
            "role": "private_rebuild_receipt_without_verifier_source_binding",
            "reason": (
                "the runtime evidence was correct, but the receipt did not bind the verifier, "
                "rebuild recipe, and Dockerfile sources"
            ),
        },
    ]
    payload = build_authority_record(
        evidence=evidence,
        historical_non_authoritative=historical,
    )
    path = _write(arguments.output_dir, payload)
    document = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(path.resolve()),
                "artifact_sha256": document["artifact_sha256"],
                "status": document["status"],
                "rank_eligible": False,
                "redistributable": False,
                "provider_calls_made": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
