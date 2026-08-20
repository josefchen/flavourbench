from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_selection_route_manifest_v65 import (
    MODEL_ID,
    build,
    verify_manifest,
)
from flavourbench.frontier_contract_runner import select_candidates

ROOT = Path(__file__).resolve().parents[1]
SOURCE = next((ROOT / "benchmark/powered-v61/manifest").glob("*.json"))
PERMISSION = next((ROOT / "benchmark/powered-v65/governance").glob("*.json"))


def test_v65_adds_only_the_finite_glm53_route() -> None:
    source = json.loads(SOURCE.read_text())
    document = build(source_path=SOURCE, permission_path=PERMISSION)
    assert verify_manifest(document)
    assert document["models"][:-1] == source["models"]
    candidates = select_candidates(document)
    assert len(candidates) == 27
    glm = candidates[-1]
    assert glm.model_id == MODEL_ID
    assert glm.canonical_model_slug == "glm-5.3"
    assert glm.execution_backend == "zai_coding_direct"
    assert glm.provider_tag == "zai-coding-plan-direct"
    assert (
        glm.backend_contract["limited_run_permission"]["permanent_running_function_authorized"]
        is False
    )
