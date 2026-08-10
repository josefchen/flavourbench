from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from flavourbench.epicure_lineage_inventory import (
    LineageInventoryError,
    build_inventory,
    capture_local_runtime_attestation,
    verify_inventory,
    write_inventory,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "mcp"
    source = root / "src/epicure_mcp"
    data = root / "data"
    scripts = root / "scripts"
    source.mkdir(parents=True)
    data.mkdir()
    scripts.mkdir()
    (source / "server.py").write_text("VALUE = 1\n", encoding="utf-8")
    (data / "embeddings.csv").write_text(
        "node_id,name,dim_0,dim_1\n1,tomato,0.1,0.2\n2,basil,0.3,0.4\n",
        encoding="utf-8",
    )
    (data / "ingredient_list.csv").write_text("name\ntomato\nbasil\n", encoding="utf-8")
    for name in (
        "Dockerfile",
        "LICENSE",
        "PRIVACY.md",
        "README.md",
        "SECURITY.md",
        "SUPPORT.md",
        "TERMS.md",
        "pyproject.toml",
    ):
        (root / name).write_text(f"{name}\n", encoding="utf-8")
    (scripts / "build_data.py").write_text("pass\n", encoding="utf-8")
    (scripts / "verify_data.py").write_text("pass\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")

    tool = tmp_path / ("tool-catalog-" + "a" * 64 + ".json")
    tool.write_text(json.dumps({"tools": []}), encoding="utf-8")
    # Mirror the runtime canonical bundle/application algorithm for the fixture.
    def manifest_digest(version: str, manifest_root: Path, paths: list[Path]) -> str:
        entries = [
            {
                "path": path.relative_to(manifest_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(paths, key=lambda item: item.relative_to(manifest_root).as_posix())
        ]
        canonical = json.dumps(
            {"manifest_version": version, "entries": entries},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    bundle = manifest_digest(
        "epicure-data-bundle-manifest-v1",
        data,
        [path for path in data.rglob("*") if path.is_file()],
    )
    app = manifest_digest(
        "epicure-python-source-manifest-v1",
        source,
        [path for path in source.rglob("*.py") if path.is_file()],
    )
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": "epicure-runtime-provenance-v1",
                "release_id": "exploratory-unmatched-1790-runtime",
                "bundle_sha256": bundle,
                "application_sha256": app,
                "ingredient_count": 2,
                "embedding_dimensions": 2,
            }
        ),
        encoding="utf-8",
    )
    return root, tool, attestation


def test_inventory_recovers_identity_but_does_not_invent_release(tmp_path: Path) -> None:
    root, tool, attestation = _fixture(tmp_path)
    payload = build_inventory(
        mcp_root=root,
        tool_contract_path=tool,
        runtime_attestation_path=attestation,
    )
    path = write_inventory(tmp_path / "out", payload)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert verify_inventory(document)
    assert document["bundle"]["ingredient_count"] == 2
    assert document["bundle"]["embedding_dimensions"] == 2
    assert document["runtime_attestation"]["matches_recovered_checkout"] is True
    assert document["training_lineage"]["status"] == "not_recovered"
    assert document["rights"]["redistributable_payload"] is False
    assert document["rank_eligible"] is False
    assert document["redistributable"] is False
    assert document["application"]["git"]["clean"] is True


def test_inventory_rejects_mismatched_runtime_attestation(tmp_path: Path) -> None:
    root, tool, attestation = _fixture(tmp_path)
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["bundle_sha256"] = "f" * 64
    attestation.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(LineageInventoryError, match="differs"):
        build_inventory(
            mcp_root=root,
            tool_contract_path=tool,
            runtime_attestation_path=attestation,
        )


def test_inventory_rejects_secret_shaped_attestation(tmp_path: Path) -> None:
    root, tool, attestation = _fixture(tmp_path)
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["token"] = "must-not-be-archived"
    attestation.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(LineageInventoryError, match="secret"):
        build_inventory(
            mcp_root=root,
            tool_contract_path=tool,
            runtime_attestation_path=attestation,
        )


def test_dirty_inventory_preserves_leading_path_and_rename_target(tmp_path: Path) -> None:
    root, tool, _ = _fixture(tmp_path)
    (root / "Dockerfile").write_text("changed Dockerfile\n", encoding="utf-8")
    _git(
        root,
        "mv",
        "src/epicure_mcp/server.py",
        "src/epicure_mcp/runtime.py",
    )

    payload = build_inventory(
        mcp_root=root,
        tool_contract_path=tool,
        runtime_attestation_path=None,
    )
    dirty = payload["application"]["git"]["dirty_files"]
    by_path = {record["path"]: record for record in dirty}

    assert "Dockerfile" in by_path
    assert "ockerfile" not in by_path
    rename = by_path[
        "src/epicure_mcp/server.py -> src/epicure_mcp/runtime.py"
    ]
    assert "R" in rename["git_status"]
    assert rename["bytes"] is not None
    assert len(rename["sha256"]) == 64


def test_capture_local_runtime_attestation_is_content_addressed(tmp_path: Path) -> None:
    payload = {
        "schema_version": "epicure-runtime-provenance-v1",
        "release_id": "exploratory-unmatched-1790-runtime",
        "bundle_sha256": "b" * 64,
        "application_sha256": "a" * 64,
        "ingredient_count": 1790,
        "embedding_dimensions": 300,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            assert self.path == "/provenance"
            assert self.headers["Authorization"] == "Bearer fixture-token"
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        path = capture_local_runtime_attestation(
            provenance_url=f"http://127.0.0.1:{server.server_port}/provenance",
            bearer_token="fixture-token",
            output_dir=tmp_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert "fixture-token" not in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_capture_rejects_non_loopback_runtime(tmp_path: Path) -> None:
    with pytest.raises(LineageInventoryError, match="loopback"):
        capture_local_runtime_attestation(
            provenance_url="https://example.test/provenance",
            bearer_token=None,
            output_dir=tmp_path,
        )
