"""Frozen runtime binding for the campaign-v6 rights replay.

The published v1 replay proves source-snapshot integrity and performs a local
prompt-risk scan.  It does not implement the five-method contamination
contract (exact, fuzzy, n-gram, semantic, and web), so this module deliberately
defines no contamination-auditor authorization plan.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

TASK_VALIDATION_V6_CAMPAIGN_SHA256 = (
    "76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709"
)
TASK_VALIDATION_V1_REPLAY_SHA256 = (
    "89f6dede2826e27bcd69eb764e32bd7a203b371f0098831c78c1077383383157"
)
TASK_VALIDATION_V1_REPLAY_PHYSICAL_SHA256 = (
    "ced66727597192342ddb978f7f48153a8fe82b0d5808d17f89c6ce42aabaaab9"
)
TASK_VALIDATION_FORMAL_CONTAMINATION_METHODS = (
    "exact",
    "fuzzy",
    "ngram",
    "semantic",
    "web",
)
TASK_VALIDATION_RIGHTS_SAMPLE_SEED_SHA256 = (
    "aae16a208727c7f64b4a89607929c783891afc580f7def0b6fdb4d6849ea49f7"
)
TASK_VALIDATION_RIGHTS_SAMPLE_IDS = (
    "25edd53f-82c2-5ebb-b058-807487e6db8b",
    "976ab5c3-0a48-55c5-8373-ba377f63caf5",
    "17a39314-e09e-5f38-9cb2-768f3b7c643c",
    "fb42473c-ea53-521a-a0f5-9a72efb6c250",
    "6735309d-3f6e-5e18-9b62-4a6d817c9e6a",
    "e89c6c7f-b3ea-5e5c-92d5-aff512690230",
    "5982509e-5a77-55d0-b283-d95a427f0c63",
    "423b3c67-2c8b-52e9-9b45-e8d8ad0dca05",
    "a1c64071-58ee-5a78-a1dc-e32e853ea572",
    "210d72af-ea8f-54ba-b231-5ced2e448195",
    "04b7528e-f430-5461-9bf8-4ea7e682549f",
    "5968388e-b7e9-5d10-b4df-e59394ec5d04",
    "8abdb882-8b8b-5792-aa99-5aaa0c6f7dcb",
    "dc041268-8129-5ffb-a280-af204c0c2c31",
    "a2c0e720-12b2-5005-be8c-8385ae81d3d5",
    "e13f4620-45a7-5a00-be5c-ef429f0674b5",
    "848c2474-0a5c-5468-a21e-59d168196e15",
    "9c622e63-5e68-598d-96e5-c8d76f6eb3ed",
    "310a6f75-6487-53c9-b26d-7e8a4d470573",
    "be6124d6-2d96-57f9-aa48-2d6af4eee159",
    "4f132dc2-f3ac-5bfa-8d2c-02cc99925ba7",
    "1c1ddc8f-48b7-5a13-af06-493865b5e0d0",
    "a4e901d3-ea6c-532c-9fd5-0323648f1b1b",
    "39e1c00d-26be-54ff-81a8-39216ffc42b7",
)
TASK_VALIDATION_RIGHTS_ANOMALY_IDS = ("210d72af-ea8f-54ba-b231-5ced2e448195",)
TASK_VALIDATION_RIGHTS_REQUIRED_IDS = (
    "04b7528e-f430-5461-9bf8-4ea7e682549f",
    "17a39314-e09e-5f38-9cb2-768f3b7c643c",
    "1c1ddc8f-48b7-5a13-af06-493865b5e0d0",
    "210d72af-ea8f-54ba-b231-5ced2e448195",
    "25edd53f-82c2-5ebb-b058-807487e6db8b",
    "310a6f75-6487-53c9-b26d-7e8a4d470573",
    "39e1c00d-26be-54ff-81a8-39216ffc42b7",
    "423b3c67-2c8b-52e9-9b45-e8d8ad0dca05",
    "4f132dc2-f3ac-5bfa-8d2c-02cc99925ba7",
    "5968388e-b7e9-5d10-b4df-e59394ec5d04",
    "5982509e-5a77-55d0-b283-d95a427f0c63",
    "6735309d-3f6e-5e18-9b62-4a6d817c9e6a",
    "848c2474-0a5c-5468-a21e-59d168196e15",
    "8abdb882-8b8b-5792-aa99-5aaa0c6f7dcb",
    "976ab5c3-0a48-55c5-8373-ba377f63caf5",
    "9c622e63-5e68-598d-96e5-c8d76f6eb3ed",
    "a1c64071-58ee-5a78-a1dc-e32e853ea572",
    "a2c0e720-12b2-5005-be8c-8385ae81d3d5",
    "a4e901d3-ea6c-532c-9fd5-0323648f1b1b",
    "be6124d6-2d96-57f9-aa48-2d6af4eee159",
    "dc041268-8129-5ffb-a280-af204c0c2c31",
    "e13f4620-45a7-5a00-be5c-ef429f0674b5",
    "e89c6c7f-b3ea-5e5c-92d5-aff512690230",
    "fb42473c-ea53-521a-a0f5-9a72efb6c250",
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rights_audit_plan() -> dict[str, Any]:
    """Return a fresh copy of the only v1 replay-backed authorization plan."""

    return {
        "schema_version": "flavourbench-task-validation-batch-audit-plan-v2",
        "campaign_sha256": TASK_VALIDATION_V6_CAMPAIGN_SHA256,
        "audit_kind": "rights",
        "sample_seed_commitment_sha256": TASK_VALIDATION_RIGHTS_SAMPLE_SEED_SHA256,
        "sample_candidate_ids": list(TASK_VALIDATION_RIGHTS_SAMPLE_IDS),
        "anomaly_or_hit_candidate_ids": list(TASK_VALIDATION_RIGHTS_ANOMALY_IDS),
        "required_candidate_ids": list(TASK_VALIDATION_RIGHTS_REQUIRED_IDS),
        "automated_evidence_sha256": TASK_VALIDATION_V1_REPLAY_SHA256,
        "automated_evidence_verified": True,
        "automated_evidence_scope": "rights_snapshot_integrity_and_local_prompt_risk_only",
        "rights_snapshot_integrity_verified": True,
        "local_prompt_risk_replay_verified": True,
        "contamination_campaign_coverage_verified": False,
        "formal_contamination_methods_required": list(TASK_VALIDATION_FORMAL_CONTAMINATION_METHODS),
        "model_outputs_available": False,
        "rank_eligible": False,
    }


TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256 = canonical_sha256(rights_audit_plan())
