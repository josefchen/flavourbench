from __future__ import annotations

import pytest

from flavourbench.season0_epicure_release import EpicureFreezeError, build_intervention


def _provenance(count: int = 1790) -> dict[str, object]:
    return {
        "release_id": "runtime-release",
        "bundle_sha256": "a" * 64,
        "application_sha256": "b" * 64,
        "ingredient_count": count,
        "embedding_dimensions": 300,
    }


def _tools() -> list[dict[str, object]]:
    return [{"name": "find_pairings", "inputSchema": {"type": "object"}}]


def test_build_intervention_binds_opaque_runtime_without_ground_truth_claim() -> None:
    value = build_intervention(
        _provenance(), _tools(), _tools(), observed_at="2026-07-16T00:00:00Z"
    )
    assert value["runtime"]["ingredient_count"] == 1790
    assert value["lineage_statement"]["public_release_match"] is False
    assert value["lineage_statement"]["similarity_as_ground_truth"] is False


def test_build_intervention_refuses_wrong_bundle_size() -> None:
    with pytest.raises(EpicureFreezeError, match="1,790"):
        build_intervention(
            _provenance(100), _tools(), _tools(), observed_at="2026-07-16T00:00:00Z"
        )
