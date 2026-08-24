"""Create an immutable correction chain between two recovered Epicure inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_lineage_inventory import verify_inventory

SCHEMA_VERSION = "epicure-recovered-runtime-inventory-correction-v1"


class LineageCorrectionError(RuntimeError):
    """The two inventories do not form the expected parser-only correction."""


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LineageCorrectionError(f"inventory must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_inventory(value):
        raise LineageCorrectionError(f"inventory does not verify: {path}")
    return value


def build_correction(
    *,
    parser_defective_inventory: Mapping[str, Any],
    authoritative_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that the authoritative inventory differs only in the first Git status record."""

    if not verify_inventory(parser_defective_inventory) or not verify_inventory(
        authoritative_inventory
    ):
        raise LineageCorrectionError("one or both inventory content addresses do not verify")
    identity_fields = {
        "runtime_id": lambda value: value.get("runtime_id"),
        "bundle_sha256": lambda value: (value.get("bundle") or {}).get("sha256"),
        "application_sha256": lambda value: (value.get("application") or {}).get("sha256"),
        "tool_schema_sha256": lambda value: (value.get("tool_contract") or {}).get(
            "semantic_sha256"
        ),
        "runtime_attestation_sha256": lambda value: (
            value.get("runtime_attestation") or {}
        ).get("response_sha256"),
    }
    identity = {name: getter(authoritative_inventory) for name, getter in identity_fields.items()}
    if any(
        getter(parser_defective_inventory) != identity[name]
        for name, getter in identity_fields.items()
    ):
        raise LineageCorrectionError("the inventories change the recovered runtime identity")
    old_dirty = (parser_defective_inventory.get("application") or {}).get("git", {}).get(
        "dirty_files"
    )
    new_dirty = (authoritative_inventory.get("application") or {}).get("git", {}).get(
        "dirty_files"
    )
    if (
        not isinstance(old_dirty, list)
        or not isinstance(new_dirty, list)
        or not old_dirty
        or not new_dirty
    ):
        raise LineageCorrectionError("the inventories lack Git dirty-file records")
    expected_old = {
        "bytes": None,
        "git_status": "M ",
        "path": "ockerfile",
        "sha256": None,
    }
    expected_new = {
        "bytes": 1376,
        "git_status": " M",
        "path": "Dockerfile",
        "sha256": "af17c9d344147c82cb7e00e426ebfd1d028ba60abaeb21a1cac0e955708e1461",
    }
    if old_dirty[0] != expected_old or new_dirty[0] != expected_new:
        raise LineageCorrectionError("the expected Dockerfile parser correction is absent")
    old_normalized = json.loads(json.dumps(parser_defective_inventory))
    new_normalized = json.loads(json.dumps(authoritative_inventory))
    old_normalized.pop("artifact_sha256", None)
    new_normalized.pop("artifact_sha256", None)
    old_normalized["application"]["git"]["dirty_files"][0] = expected_new
    if old_normalized != new_normalized:
        raise LineageCorrectionError("the inventories differ beyond the parser correction")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_role": "immutable_parser_correction_and_authoritative_inventory_pointer",
        "parser_defective_inventory_sha256": parser_defective_inventory["artifact_sha256"],
        "authoritative_inventory_sha256": authoritative_inventory["artifact_sha256"],
        "authoritative_status": "authoritative_recovered_runtime_inventory",
        "parser_defective_status": "retained_non_authoritative_historical_record",
        "correction": {
            "component": "application.git.dirty_files[0]",
            "cause": "text-mode strip removed the leading porcelain status-space",
            "before": expected_old,
            "after": expected_new,
            "other_inventory_fields_changed": 0,
        },
        "runtime_identity": identity,
        "interpretation_rule": (
            "Use the authoritative inventory for current prose, release packaging, and new "
            "plans. Preserve the parser-defective inventory only where an immutable historical "
            "record already binds it."
        ),
        "provider_calls_made": False,
        "epicure_calls_made": False,
    }


def verify_correction(document: object) -> bool:
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return isinstance(digest, str) and len(digest) == 64 and _sha256(unhashed) == digest


def _write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = _sha256(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"epicure-lineage-inventory-correction-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise LineageCorrectionError("content-addressed correction conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as file:
        temporary = Path(file.name)
        file.write(rendered)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-defective-inventory", type=Path, required=True)
    parser.add_argument("--authoritative-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = build_correction(
        parser_defective_inventory=_load(arguments.parser_defective_inventory),
        authoritative_inventory=_load(arguments.authoritative_inventory),
    )
    path = _write(arguments.output_dir, payload)
    document = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(path.resolve()),
                "artifact_sha256": document["artifact_sha256"],
                "authoritative_inventory_sha256": document[
                    "authoritative_inventory_sha256"
                ],
                "provider_calls_made": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
