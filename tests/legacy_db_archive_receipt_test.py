from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import Draft202012Validator, FormatChecker

PROJECTS_ROOT = Path(__file__).resolve().parents[4]
EPICURE_ROOT = PROJECTS_ROOT / "epicure"
SERVICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    EPICURE_ROOT / "deployment" / "flavourbench" / "seal_legacy_db_archive.py"
)
SCHEMA_PATH = (
    EPICURE_ROOT
    / "deployment"
    / "flavourbench"
    / "legacy-db-offline-archive-v1.schema.json"
)
LEGACY_PROOF_PATH = (
    SERVICE_ROOT
    / "artifacts"
    / "migration-proofs"
    / "legacy-0001-upgrade-bridge-proof-"
    "81dde02834fd6d5096245fdb9e23b74fd3b7d07a56977596d26e63fe5641fb74.json"
)
BRIDGE_SCRIPT_PATH = SERVICE_ROOT / "scripts" / "upgrade_legacy_0001_postgresql.py"
FROZEN_DUMP_PATH = (
    EPICURE_ROOT
    / ".private"
    / "flavourbench-backups"
    / "reef-cluster-pre-0032-20260808T223552Z.dump"
)


def _load_archive_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("seal_legacy_db_archive", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_write(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    path.chmod(0o600)


def _build_volume_tar(path: Path) -> None:
    root = path.parent / "pgdata"
    (root / "global").mkdir(parents=True)
    (root / "PG_VERSION").write_text("16\n", encoding="utf-8")
    (root / "global" / "pg_control").write_bytes(b"test-control")
    with tarfile.open(path, mode="w") as archive:
        archive.add(root / "PG_VERSION", arcname="./PG_VERSION")
        archive.add(root / "global", arcname="./global")
    path.chmod(0o600)


def _synthetic_inputs(tmp_path: Path, module: ModuleType) -> dict[str, Path]:
    frozen = tmp_path / "frozen.dump"
    fresh = tmp_path / "fresh.dump"
    volume = tmp_path / "legacy-volume.tar"
    proof = tmp_path / "proof.json"
    bridge = tmp_path / "bridge.py"
    _private_write(frozen, b"PGDMP-frozen-test")
    _private_write(fresh, b"PGDMP-fresh-test")
    _private_write(bridge, b"# reviewed bridge test\n")
    _build_volume_tar(volume)

    module.FROZEN_DUMP_SHA256 = _sha256(frozen)
    module.BRIDGE_SCRIPT_SHA256 = _sha256(bridge)
    proof_body = {
        "schema_version": "flavourbench-legacy-0001-upgrade-proof-v3",
        "source_backup_sha256": module.FROZEN_DUMP_SHA256,
        "bridge_script_sha256": module.BRIDGE_SCRIPT_SHA256,
        "start_revision": module.SOURCE_REVISION,
        "final_revision": module.SUCCESSOR_SCHEMA_REVISION,
        "preflight": {"content_root_sha256": module.SOURCE_CONTENT_ROOT_SHA256},
        "final": {"content_root_sha256": module.SOURCE_CONTENT_ROOT_SHA256},
    }
    semantic = hashlib.sha256(module._canonical_bytes(proof_body)).hexdigest()
    module.LEGACY_PROOF_SEMANTIC_SHA256 = semantic
    _private_write(
        proof,
        json.dumps({**proof_body, "semantic_sha256": semantic}, sort_keys=True).encode(),
    )
    module.LEGACY_PROOF_PHYSICAL_SHA256 = _sha256(proof)
    return {
        "frozen_dump": frozen,
        "fresh_dump": fresh,
        "volume_tar": volume,
        "legacy_proof": proof,
        "bridge_script": bridge,
    }


def test_archive_schema_and_reviewed_evidence_pins_are_exact() -> None:
    module = _load_archive_module()
    assert _sha256(SCHEMA_PATH) == module.SCHEMA_PHYSICAL_SHA256
    assert module.SOURCE_VOLUME == "epicure_flavourbench-db-data"
    assert module.SUCCESSOR_VOLUME == "epicure_flavourbench-db-data-v2"
    assert _sha256(LEGACY_PROOF_PATH) == module.LEGACY_PROOF_PHYSICAL_SHA256
    assert _sha256(BRIDGE_SCRIPT_PATH) == module.BRIDGE_SCRIPT_SHA256
    if FROZEN_DUMP_PATH.exists():
        assert _sha256(FROZEN_DUMP_PATH) == module.FROZEN_DUMP_SHA256


def test_archive_receipt_is_schema_valid_content_addressed_and_append_only(
    tmp_path: Path,
) -> None:
    module = _load_archive_module()
    inputs = _synthetic_inputs(tmp_path, module)
    receipt = module.build_receipt(
        **inputs,
        rollback_api_image_id="sha256:" + "1" * 64,
        rollback_api_image_tag=(
            "epicure-flavourbench-api:rollback-pre-v2-20260809T000000Z"
        ),
        created_at="2026-08-09T00:00:00Z",
    )
    assert receipt["separation"] == {
        "source_volume_offline": True,
        "source_volume_mounted_by_successor": False,
        "legacy_rows_imported_into_successor": False,
        "sql_downgrade_authorized": False,
    }
    assert set(receipt["generation_activity"].values()) == {0}
    assert receipt["rollback_api_image"] == {
        "image_id": "sha256:" + "1" * 64,
        "local_tag": "epicure-flavourbench-api:rollback-pre-v2-20260809T000000Z",
        "local_only": True,
        "pull_policy": "never",
        "tag_resolution_verified_before_mutation": True,
    }

    output = tmp_path / "receipts"
    output.mkdir(mode=0o700)
    path, semantic, physical = module.write_receipt(receipt, output)
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    embedded = wrapper.pop("semantic_sha256")
    assert embedded == semantic
    assert hashlib.sha256(module._canonical_bytes(wrapper)).hexdigest() == semantic
    assert _sha256(path) == physical
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_example = json.loads(json.dumps({**wrapper, "semantic_sha256": embedded}))
    schema_example["archive_members"]["frozen_source_dump"]["sha256"] = (
        "f81b637903a25514a14cefb7363dd9e32b2b4659e65e10b43e25ca9ec8a5bd08"
    )
    schema_example["archive_members"]["legacy_upgrade_proof"].update(
        {
            "sha256": "527487c8fdbc664a4b94006a51afe77b81f5292cf07746e347817cfbf6a1e151",
            "semantic_sha256": (
                "81dde02834fd6d5096245fdb9e23b74fd3b7d07a56977596d26e63fe5641fb74"
            ),
        }
    )
    schema_example["archive_members"]["legacy_bridge_script"]["sha256"] = (
        "5c94535e064171a2714e16e49940628f030cf3c541239b6bd3b9bdafb747fb78"
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(
        schema_example
    )
    with pytest.raises(module.ArchiveReceiptError, match="refusing to overwrite"):
        module.write_receipt(receipt, output)


def test_archive_receipt_rejects_nonprivate_or_nonself_contained_inputs(
    tmp_path: Path,
) -> None:
    module = _load_archive_module()
    inputs = _synthetic_inputs(tmp_path, module)
    inputs["fresh_dump"].chmod(0o644)
    with pytest.raises(module.ArchiveReceiptError, match="mode 0600"):
        module.build_receipt(
            **inputs,
            rollback_api_image_id="sha256:" + "1" * 64,
            rollback_api_image_tag=(
                "epicure-flavourbench-api:rollback-pre-v2-20260809T000000Z"
            ),
            created_at="2026-08-09T00:00:00Z",
        )

    inputs["fresh_dump"].chmod(0o600)
    unsafe_tar = inputs["volume_tar"]
    link_target = tmp_path / "link-target"
    link_target.write_text("target", encoding="utf-8")
    link = tmp_path / "pg-link"
    link.symlink_to(link_target)
    with tarfile.open(unsafe_tar, mode="w") as archive:
        archive.add(link, arcname="./PG_VERSION", recursive=False)
        archive.add(link, arcname="./global/pg_control", recursive=False)
    unsafe_tar.chmod(0o600)
    with pytest.raises(module.ArchiveReceiptError, match="not self-contained"):
        module.build_receipt(
            **inputs,
            rollback_api_image_id="sha256:" + "1" * 64,
            rollback_api_image_tag=(
                "epicure-flavourbench-api:rollback-pre-v2-20260809T000000Z"
            ),
            created_at="2026-08-09T00:00:00Z",
        )


def test_receipt_writer_rejects_public_directory(tmp_path: Path) -> None:
    module = _load_archive_module()
    inputs = _synthetic_inputs(tmp_path, module)
    receipt = module.build_receipt(
        **inputs,
        rollback_api_image_id="sha256:" + "1" * 64,
        rollback_api_image_tag=(
            "epicure-flavourbench-api:rollback-pre-v2-20260809T000000Z"
        ),
        created_at="2026-08-09T00:00:00Z",
    )
    output = tmp_path / "public-receipts"
    output.mkdir(mode=0o755)
    with pytest.raises(module.ArchiveReceiptError, match="group/world accessible"):
        module.write_receipt(receipt, output)


def test_archive_receipt_rejects_unpinned_rollback_image(tmp_path: Path) -> None:
    module = _load_archive_module()
    inputs = _synthetic_inputs(tmp_path, module)
    with pytest.raises(module.ArchiveReceiptError, match="content digest"):
        module.build_receipt(
            **inputs,
            rollback_api_image_id="latest",
            rollback_api_image_tag=(
                "epicure-flavourbench-api:rollback-pre-v2-20260809T000000Z"
            ),
            created_at="2026-08-09T00:00:00Z",
        )
    with pytest.raises(module.ArchiveReceiptError, match="naming contract"):
        module.build_receipt(
            **inputs,
            rollback_api_image_id="sha256:" + "1" * 64,
            rollback_api_image_tag="epicure-flavourbench-api:latest",
            created_at="2026-08-09T00:00:00Z",
        )
