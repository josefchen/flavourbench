from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import Draft202012Validator, FormatChecker

PROJECTS_ROOT = Path(__file__).resolve().parents[4]
DEPLOYMENT_ROOT = PROJECTS_ROOT / "epicure" / "deployment" / "flavourbench"
SCRIPT_PATH = DEPLOYMENT_ROOT / "seal_legacy_db_recovery_binding.py"
SCHEMA_PATH = DEPLOYMENT_ROOT / "legacy-db-offline-recovery-binding-v1.schema.json"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("offline_recovery_binding", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_receipt(path: Path, body: dict) -> None:
    body["semantic_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _fixtures(tmp_path: Path, module: ModuleType) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_id = "sha256:" + "1" * 64
    tag = "epicure-flavourbench-api:rollback-pre-v2-20260809T002959Z"
    rootfs = tmp_path / "rollback-api-rootfs-export.tar"
    image = tmp_path / "rollback-api-image.tar"
    rootfs.write_bytes(b"rootfs")
    image.write_bytes(b"image")
    rootfs.chmod(0o600)
    image.chmod(0o600)
    offline = tmp_path / "offline.json"
    _write_receipt(
        offline,
        {
            "schema_version": module.OFFLINE_SCHEMA_VERSION,
            "receipt_schema_physical_sha256": module.OFFLINE_SCHEMA_PHYSICAL_SHA256,
            "source_revision": "0001_initial",
            "source_content_root_sha256": module.EXPECTED_CONTENT_ROOT_SHA256,
            "rollback_api_image": {"image_id": image_id, "local_tag": tag},
            "separation": {"source_volume_offline": True},
        },
    )
    recovery = tmp_path / "recovery.json"
    _write_receipt(
        recovery,
        {
            "schema_version": module.RECOVERY_SCHEMA_VERSION,
            "receipt_schema_physical_sha256": module.RECOVERY_SCHEMA_PHYSICAL_SHA256,
            "construction": "paused_container_export_clean_import_isolated_restore-v1",
            "imported_image": {"image_id": image_id, "local_tag": tag},
            "rootfs_export": {
                "sha256": hashlib.sha256(rootfs.read_bytes()).hexdigest(),
                "size_bytes": rootfs.stat().st_size,
            },
            "saved_image": {
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "size_bytes": image.stat().st_size,
                "docker_load_verified": True,
            },
            "isolated_restore_smoke": {"status": "pass"},
            "generation_activity": {
                "provider_calls": 0,
                "mcp_tool_calls": 0,
                "model_outputs": 0,
            },
        },
    )
    return offline, recovery, rootfs, image


def test_binding_schema_and_cross_receipt_contract(tmp_path: Path) -> None:
    module = _load_module()
    assert hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest() == (
        module.SCHEMA_PHYSICAL_SHA256
    )
    offline, recovery, rootfs, image = _fixtures(tmp_path, module)
    binding = module.build_binding(
        offline_receipt_path=offline,
        recovery_receipt_path=recovery,
        rootfs_export_path=rootfs,
        saved_image_path=image,
        created_at="2026-08-09T04:00:00Z",
    )
    wrapper = {
        **binding,
        "semantic_sha256": hashlib.sha256(module._canonical_bytes(binding)).hexdigest(),
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(wrapper)
    assert binding["binding"]["source_volume_offline"] is True
    assert binding["binding"]["saved_image_archive_bound"] is True


def test_binding_rejects_archive_drift_and_false_offline_claim(tmp_path: Path) -> None:
    module = _load_module()
    offline, recovery, rootfs, image = _fixtures(tmp_path, module)
    image.write_bytes(b"changed")
    image.chmod(0o600)
    with pytest.raises(module.BindingError, match="differ"):
        module.build_binding(
            offline_receipt_path=offline,
            recovery_receipt_path=recovery,
            rootfs_export_path=rootfs,
            saved_image_path=image,
            created_at="2026-08-09T04:00:00Z",
        )

    offline, recovery, rootfs, image = _fixtures(tmp_path / "second", module)
    payload = json.loads(offline.read_text(encoding="utf-8"))
    payload.pop("semantic_sha256")
    payload["separation"]["source_volume_offline"] = False
    _write_receipt(offline, payload)
    with pytest.raises(module.BindingError, match="offline"):
        module.build_binding(
            offline_receipt_path=offline,
            recovery_receipt_path=recovery,
            rootfs_export_path=rootfs,
            saved_image_path=image,
            created_at="2026-08-09T04:00:00Z",
        )
