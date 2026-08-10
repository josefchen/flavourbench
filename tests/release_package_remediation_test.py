from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flavourbench.release_package_remediation import (
    HELD_REVIEW_ITEM_ID,
    build_release_remediations,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
POSTRUN = ARTIFACTS / "season1/current-quality-run/frontier-coverage-v4-postrun"
ARENA = POSTRUN / (
    "frontier-corrected-development-arena-"
    "234f5b5e3364f0e0f2fddc0f23d47d1d670df509c5707e35cb713183264c5c5e.json"
)
UPLIFT = POSTRUN / (
    "frontier-corrected-development-uplift-"
    "1a280655dc7693f3e0f9c6a5b9666449362d0b3dcebc6609411b72d12ed450a1.json"
)
COVERAGE = POSTRUN / (
    "frontier-corrected-development-coverage-"
    "68863208cf4ffc4772ee55378e3ce82a66988b8933e3ef1ced34edf290afa695.json"
)
ORIGINAL_AUTHORIZATION = ARTIFACTS / (
    "season1/current-quality-run/"
    "reasoning-effort-v8-live-incident-recovery-v9/governance/"
    "v9-recovery-independent-go-"
    "bd73868acace85eb409ff21c5ecc619f4e9300f2e62c0db64dca8f595a20c09b.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_release_remediation_is_append_only_and_complete(tmp_path: Path) -> None:
    inputs = (ARENA, UPLIFT, COVERAGE, ORIGINAL_AUTHORIZATION)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    outputs = build_release_remediations(
        uplift_predecessor_path=UPLIFT,
        coverage_predecessor_path=COVERAGE,
        arena_path=ARENA,
        artifacts_root=ARTIFACTS,
        original_authorization_path=ORIGINAL_AUTHORIZATION,
        output_dir=tmp_path,
    )
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}

    uplift = _read(outputs["uplift"])
    assert uplift["observed"]["candidate_pairs"] == 186
    assert uplift["observed"]["source_arms"] == 372
    assert uplift["observed"]["coverage_recovery_pairs_added"] == 7
    assert HELD_REVIEW_ITEM_ID not in {
        item["review_item_id"] for item in uplift["items"]
    }
    assert uplift["policy_hold"]["review_item_id"] == HELD_REVIEW_ITEM_ID
    assert uplift["claim_boundary"]["quality_judgments"] == 0

    coverage = _read(outputs["coverage"])
    assert coverage["uplift"]["pairs_after"] == 186
    assert coverage["uplift"]["pairs_on_policy_hold"] == 1
    assert coverage["model_arena"]["comparisons_after"] == 915

    commitment = _read(outputs["input_commitment"])
    assert commitment["observed"] == {
        "conflicting_physical_records": 0,
        "distinct_response_records": 379,
        "distinct_source_records": 193,
        "missing_target_records": 0,
        "private_record_commitments": 572,
        "public_derivatives": 4,
    }
    assert all(not row["included_in_arxiv_source"] for row in commitment["private_inputs"])
    serialized = json.dumps(commitment, sort_keys=True)
    assert "/home/" not in serialized
    assert "remy-simpc4" not in serialized

    authorization = _read(outputs["authorization"])
    assert authorization["human_principal"]["name"] == "Josef Chen"
    assert authorization["claim_boundary"]["cryptographic_signature"] is False
    assert (
        authorization["superseded_public_copy"]["semantic_artifact_sha256"]
        == "bd73868acace85eb409ff21c5ecc619f4e9300f2e62c0db64dca8f595a20c09b"
    )
