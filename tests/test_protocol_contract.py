from __future__ import annotations

import hashlib
from pathlib import Path

from flavourbench.database import EXPECTED_SCHEMA_REVISION
from flavourbench.protocol_contract import build_protocol_bundle


def test_protocol_bundle_declares_the_runtime_schema_head() -> None:
    bundle, _ = build_protocol_bundle(
        tool_registry_sha256="1" * 64,
        epicure_release_id="protocol-contract-test",
        epicure_bundle_sha256="2" * 64,
        epicure_application_sha256="3" * 64,
        analysis_plan_sha256="4" * 64,
    )
    filename = f"{EXPECTED_SCHEMA_REVISION}.py"
    revision_path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / filename
    expected_sha256 = hashlib.sha256(revision_path.read_bytes()).hexdigest()
    release_inputs = bundle["release_inputs"]

    assert release_inputs["alembic_head"] == EXPECTED_SCHEMA_REVISION
    assert release_inputs["alembic_head_sha256"] == expected_sha256
    assert release_inputs["alembic_revisions_sha256"][filename] == expected_sha256
