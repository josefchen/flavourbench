from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flavourbench.epicure_selection_powered_plan_v74 import verify_plan as verify_plan_v74
from flavourbench.epicure_selection_powered_plan_v75 import verify_plan as verify_plan_v75
from flavourbench.epicure_selection_powered_plan_v76 import _as_v67
from flavourbench.epicure_selection_powered_plan_v76 import verify_plan as verify_plan_v76
from flavourbench.epicure_selection_powered_plan_v77 import _as_v76
from flavourbench.epicure_selection_powered_plan_v77 import verify_plan as verify_plan_v77
from flavourbench.epicure_selection_powered_plan_v78 import _as_v77
from flavourbench.epicure_selection_powered_plan_v78 import verify_plan as verify_plan_v78
from flavourbench.epicure_selection_route_manifest_v54 import DEEPSEEK_PRO_MODEL_ID
from flavourbench.epicure_selection_route_manifest_v73 import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "benchmark/powered-v65/manifest"
    / (
        "flavourbench-frontier-refresh-27-"
        "978db714e4f2ef348022c8f1afce214c076a02a3ec1b8b7a2cf7677810dac725.json"
    )
)
MANIFEST = (
    ROOT
    / "benchmark/powered-v73/manifest"
    / (
        "flavourbench-frontier-refresh-27-"
        "993e6ad499f7a26f52baf25cc22c52e8a8938703365dc1814f0c17b156d73368.json"
    )
)
PANEL_1 = (
    ROOT
    / "benchmark/powered-v74/plan"
    / (
        "epicure-selection-analysis-plan-"
        "55fef84440b6dda3db7ad44ba3947a3b2d81f127c2ad81fce7e6fbd38c8df6c0.json"
    )
)
PANEL_2 = (
    ROOT
    / "benchmark/powered-v75/plan"
    / (
        "epicure-selection-analysis-plan-"
        "48d9d8f12d6da1910621d15ea7f26750119745ecde3b94fb8dfb3ec5c382cf75.json"
    )
)
JOINT = (
    ROOT
    / "benchmark/powered-v76/plan"
    / (
        "epicure-selection-joint-analysis-plan-"
        "32cc023f3589e6d950fe533625fda1c8793f208e72c8b8194661f465733575e0.json"
    )
)
JOINT_PREDECESSOR = (
    ROOT
    / "benchmark/powered-v67/plan"
    / (
        "epicure-selection-joint-analysis-plan-"
        "d88e7e297174a217dbaa9a118967fd0624c744499e136ea6c658d3cebe25b748.json"
    )
)
JOINT_PRICE_LINEAGE = (
    ROOT
    / "benchmark/powered-v77/plan"
    / (
        "epicure-selection-joint-analysis-plan-"
        "9a6a835f956b276f98f4aeba6e20e218a5c2567e599d67602b50d990b9a4d091.json"
    )
)
JOINT_COMPLETE_CASE = (
    ROOT
    / "benchmark/powered-v78/plan"
    / (
        "epicure-selection-joint-analysis-plan-"
        "d7a85e423f04fa9710e8ad906fe992b4fc74f3ece638bd0e527cea47d22a2bd0.json"
    )
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _physical(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v73_refreshes_only_the_deepseek_execution_contract() -> None:
    source = _load(SOURCE)
    manifest = _load(MANIFEST)
    assert _physical(MANIFEST) == (
        "f25577f233de6c2f9d7a9685f1f726035f18c342b7aa400e8d39012415b4c56d"
    )
    assert verify_manifest(manifest)
    before = {row["model"]["id"]: row for row in source["models"]}
    after = {row["model"]["id"]: row for row in manifest["models"]}
    assert {
        model_id: row for model_id, row in before.items() if model_id != DEEPSEEK_PRO_MODEL_ID
    } == {model_id: row for model_id, row in after.items() if model_id != DEEPSEEK_PRO_MODEL_ID}
    deepseek = after[DEEPSEEK_PRO_MODEL_ID]
    assert deepseek["endpoint"]["tag"] == "gmicloud/fp8"
    assert deepseek["request_policy"]["provider"]["data_collection"] == "deny"
    refresh = manifest["deepseek_gmicloud_contract_refresh_v73"]
    assert refresh["compatibility_probe"]["scored_as_benchmark_evidence"] is False
    assert refresh["complete_two_panel_blocks_required"] is True
    assert refresh["cross_contract_response_pooling"] is False


def test_v74_v75_freeze_complete_non_pooled_blocks() -> None:
    panel_1 = _load(PANEL_1)
    panel_2 = _load(PANEL_2)
    assert _physical(PANEL_1) == (
        "ab8bb1a6cc294bc02246b06bb6c76b1b76cda3cc099481f3aae671fe210fd3c7"
    )
    assert _physical(PANEL_2) == (
        "6dff278f5870d80c36fd0ab6249270a233d0fdba15db4653a25e4146b6466733"
    )
    assert verify_plan_v74(panel_1)
    assert verify_plan_v75(panel_2)
    for document, key in (
        (panel_1, "deepseek_gmicloud_contract_refresh_v74"),
        (panel_2, "deepseek_gmicloud_contract_refresh_v75"),
    ):
        contract = document["execution"][key]
        assert contract["replacement_primary_cells_per_model"] == 640
        assert contract["replacement_repeat_cells_per_model"] == 64
        assert contract["selective_failed_cell_retry"] is False
        assert contract["superseded_responses_used"] is False


def test_v76_freezes_joint_inference_before_deepseek_quality() -> None:
    document = _load(JOINT)
    predecessor = _load(JOINT_PREDECESSOR)
    assert _physical(JOINT) == ("664013931bec65ddc3c5e3b7c581735c17231f93630454197c4e5d50944cbfd9")
    assert verify_plan_v76(document)
    assert _as_v67(document) == predecessor
    rules = document["source_rules"]
    assert rules["refreshed_deepseek_prior_responses_used"] is False
    assert rules["refreshed_deepseek_cross_contract_pooling"] is False
    assert rules["refreshed_deepseek_included_without_quality_scores_or_selections"] is True


def test_v77_freezes_price_only_deepseek_source_lineage() -> None:
    document = _load(JOINT_PRICE_LINEAGE)
    predecessor = _load(JOINT)
    assert _physical(JOINT_PRICE_LINEAGE) == (
        "38cc784966bb683877b49765f89585b87a866732a15f510ec8f424f6b340a591"
    )
    assert verify_plan_v77(document)
    assert _as_v76(document) == predecessor
    rules = document["source_rules"]
    assert rules["deepseek_cross_contract_allowed_roster_differences"] == ["endpoint_sha256"]
    assert rules["deepseek_cross_contract_difference_class"] == "price_metadata_only"
    assert rules["deepseek_endpoint_execution_sha256_unchanged"] is True
    assert rules["coverage_and_parseability_inspected_before_source_freeze"] is True
    assert rules["quality_scores_and_model_selections_inspected_for_source_decision"] is False


def test_v78_freezes_a_no_dnf_complete_case_leaderboard() -> None:
    document = _load(JOINT_COMPLETE_CASE)
    predecessor = _load(JOINT_PRICE_LINEAGE)
    assert _physical(JOINT_COMPLETE_CASE) == (
        "da753ef8731ddea0123040f5bff9a1b7cd0b32c5e96dc70a57c123d0aecd04e7"
    )
    assert verify_plan_v78(document)
    assert _as_v77(document) == predecessor
    eligibility = document["eligibility"]
    assert len(eligibility["ranked_model_ids"]) == 26
    assert eligibility["coverage_diagnostic_model_ids"] == ["anthropic/claude-fable-5"]
    assert eligibility["requires_every_scheduled_primary_and_repeat_cell"] is True
    assert eligibility["dnf_rows_emitted"] is False
    assert eligibility["quality_scores_or_model_selections_used"] is False
    assert document["roster"]["pairwise_hypotheses"] == 325
