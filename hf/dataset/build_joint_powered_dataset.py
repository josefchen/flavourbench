"""Build the two-panel FlavourBench Hugging Face dataset from frozen raw cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from build_powered_dataset import (
    PoweredDatasetBuildError,
    _jsonl,
    _load,
    _physical,
    _provider_attempt_documents,
    _response_documents,
    _semantic_valid,
    _write_atomic,
)

from flavourbench.epicure_selection_powered_plan_v44 import verify_plan as verify_plan_v44
from flavourbench.epicure_selection_powered_plan_v45 import verify_plan as verify_plan_v45
from flavourbench.epicure_selection_powered_plan_v46 import verify_plan as verify_plan_v46
from flavourbench.epicure_selection_powered_plan_v47 import verify_plan as verify_plan_v47
from flavourbench.epicure_selection_powered_plan_v48 import verify_plan as verify_joint_plan
from flavourbench.epicure_selection_powered_plan_v49 import verify_plan as verify_plan_v49
from flavourbench.epicure_selection_powered_plan_v50 import verify_plan as verify_plan_v50
from flavourbench.epicure_selection_powered_plan_v51 import (
    verify_plan as verify_joint_plan_v51,
)
from flavourbench.epicure_selection_powered_plan_v52 import verify_plan as verify_plan_v52
from flavourbench.epicure_selection_powered_plan_v53 import (
    verify_plan as verify_joint_plan_v53,
)
from flavourbench.epicure_selection_powered_plan_v54 import verify_plan as verify_plan_v54
from flavourbench.epicure_selection_powered_plan_v55 import verify_plan as verify_plan_v55
from flavourbench.epicure_selection_powered_plan_v56 import (
    verify_plan as verify_joint_plan_v56,
)
from flavourbench.epicure_selection_powered_plan_v58 import verify_plan as verify_plan_v58
from flavourbench.epicure_selection_powered_plan_v59 import verify_plan as verify_plan_v59
from flavourbench.epicure_selection_powered_plan_v60 import (
    verify_plan as verify_joint_plan_v60,
)
from flavourbench.epicure_selection_powered_plan_v62 import verify_plan as verify_plan_v62
from flavourbench.epicure_selection_powered_plan_v63 import verify_plan as verify_plan_v63
from flavourbench.epicure_selection_powered_plan_v64 import (
    verify_plan as verify_joint_plan_v64,
)
from flavourbench.epicure_selection_powered_plan_v65 import verify_plan as verify_plan_v65
from flavourbench.epicure_selection_powered_plan_v66 import verify_plan as verify_plan_v66
from flavourbench.epicure_selection_powered_plan_v67 import (
    verify_plan as verify_joint_plan_v67,
)
from flavourbench.epicure_selection_powered_plan_v74 import verify_plan as verify_plan_v74
from flavourbench.epicure_selection_powered_plan_v75 import verify_plan as verify_plan_v75
from flavourbench.epicure_selection_powered_plan_v76 import (
    verify_plan as verify_joint_plan_v76,
)
from flavourbench.epicure_selection_repeat_panel_replication_v1 import (
    verify_repeat_panel as verify_repeat_panel_replication_2,
)
from flavourbench.epicure_selection_repeat_panel_v2 import verify_repeat_panel
from flavourbench.epicure_selection_route_manifest_v45 import FABLE_MODEL_ID, QWEN_MODEL_ID
from flavourbench.epicure_selection_route_manifest_v52 import (
    DEEPSEEK_FLASH_MODEL_ID,
    LUNA_MODEL_ID,
)
from flavourbench.epicure_selection_route_manifest_v54 import (
    REPLACEMENT_MODEL_IDS as COVERAGE_REPAIR_MODEL_IDS,
)
from flavourbench.epicure_selection_route_manifest_v57 import DEEPSEEK_PRO_MODEL_ID
from flavourbench.epicure_selection_route_manifest_v65 import MODEL_ID as GLM53_MODEL_ID
from flavourbench.epicure_selection_taskset_replication_v1 import (
    verify_taskset as verify_taskset_replication_2,
)
from flavourbench.epicure_selection_taskset_v2 import verify_taskset

TABLE_ORDER = (
    "models",
    "tasks",
    "primary_observations",
    "repeat_observations",
    "provider_attempt_events",
    "leaderboard",
    "pairwise_comparisons",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _physical(path),
    }


def _require_pin(
    parent: Mapping[str, Any],
    label: str,
    document: Mapping[str, Any],
    path: Path,
) -> None:
    if parent["inputs"][label] != _pin(document, path):
        raise PoweredDatasetBuildError(f"joint dataset {label} pin differs")


def _require_response_commitment(
    release: Mapping[str, Any], label: str, documents: Sequence[Mapping[str, Any]]
) -> None:
    artifacts = sorted(str(row["artifact_sha256"]) for row in documents)
    if len(set(artifacts)) != len(artifacts):
        raise PoweredDatasetBuildError(f"{label} reuses a response artifact")
    observed = {
        "count": len(documents),
        "artifact_set_sha256": hashlib.sha256(_canonical(artifacts)).hexdigest(),
        "spend_micros": sum(
            int((row.get("generation") or {}).get("cost_micros") or 0) for row in documents
        ),
    }
    if release["inputs"][label] != observed:
        raise PoweredDatasetBuildError(f"{label} differs from the release commitment")


def _annotated_tasks(
    panel_1: Mapping[str, Any], panel_2: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {**dict(task), "panel_index": panel_index}
        for panel_index, document in ((1, panel_1), (2, panel_2))
        for task in document["tasks"]
    ]


def _tables(
    *,
    release: Mapping[str, Any],
    joint_plan: Mapping[str, Any],
    panel_1_plan: Mapping[str, Any],
    panel_2_plan: Mapping[str, Any],
    panel_1_taskset: Mapping[str, Any],
    panel_2_taskset: Mapping[str, Any],
    panel_1_repeat: Mapping[str, Any],
    panel_2_repeat: Mapping[str, Any],
    primary_documents: Sequence[Mapping[str, Any]],
    repeat_documents: Sequence[Mapping[str, Any]],
    attempt_documents: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    analysis_by_model = {str(row["model_id"]): dict(row) for row in release["analysis"]["models"]}
    replication_by_model = {
        str(row["model_id"]): dict(row)
        for row in release["analysis"]["panel_replication"]["models"]
    }
    if set(replication_by_model) != set(analysis_by_model):
        raise PoweredDatasetBuildError("panel replication roster differs from analysis")
    panel_1_routes = {str(row["model_id"]): row for row in panel_1_plan["roster"]["models"]}
    panel_2_routes = {str(row["model_id"]): row for row in panel_2_plan["roster"]["models"]}
    if set(panel_1_routes) != set(analysis_by_model) or set(panel_2_routes) != set(
        analysis_by_model
    ):
        raise PoweredDatasetBuildError("panel route rosters differ from analysis")
    model_rows = []
    for route in joint_plan["roster"]["models"]:
        model_id = str(route["model_id"])
        model_rows.append(
            {
                **analysis_by_model[model_id],
                "canonical_model_slug": route["canonical_model_slug"],
                "execution_backend": route["execution_backend"],
                "provider_tag": route["provider_tag"],
                "provider_name": route["provider_name"],
                "endpoint_execution_sha256": route["endpoint_execution_sha256"],
                "panel_1_route": {
                    key: panel_1_routes[model_id][key]
                    for key in (
                        "execution_backend",
                        "provider_name",
                        "provider_tag",
                        "endpoint_execution_sha256",
                    )
                },
                "panel_2_route": {
                    key: panel_2_routes[model_id][key]
                    for key in (
                        "execution_backend",
                        "provider_name",
                        "provider_tag",
                        "endpoint_execution_sha256",
                    )
                },
                "panel_replication": replication_by_model[model_id],
                "response_source": (
                    "powered-v76-two-panel-glm53-plus-refreshed-deepseek-complete-block-lineage"
                    if verify_joint_plan_v76(joint_plan)
                    else "powered-v67-two-panel-glm53-limited-run-complete-block-lineage"
                    if verify_joint_plan_v67(joint_plan)
                    else "powered-v64-two-panel-deepseek-gmicloud-complete-block-lineage"
                    if verify_joint_plan_v64(joint_plan)
                    else "powered-v60-two-panel-deepseek-complete-block-repair-lineage"
                    if verify_joint_plan_v60(joint_plan)
                    else "powered-v56-two-panel-complete-coverage-repair-lineage"
                    if verify_joint_plan_v56(joint_plan)
                    else "powered-v53-two-panel-frozen-response-lineage"
                ),
            }
        )
    leaderboard = sorted(
        model_rows,
        key=lambda row: (int(row["point_estimate_rank"]), str(row["model_id"])),
    )
    return {
        "models": model_rows,
        "tasks": _annotated_tasks(panel_1_taskset, panel_2_taskset),
        "primary_observations": [dict(row) for row in primary_documents],
        "repeat_observations": [dict(row) for row in repeat_documents],
        "provider_attempt_events": [dict(row) for row in attempt_documents],
        "leaderboard": leaderboard,
        "pairwise_comparisons": [dict(row) for row in release["analysis"]["pairwise_comparisons"]],
    }


def _expected_files(
    *,
    release: Mapping[str, Any],
    joint_plan: Mapping[str, Any],
    panel_1_plan: Mapping[str, Any],
    panel_2_plan: Mapping[str, Any],
    panel_1_taskset: Mapping[str, Any],
    panel_2_taskset: Mapping[str, Any],
    panel_1_repeat: Mapping[str, Any],
    panel_2_repeat: Mapping[str, Any],
    primary_documents: Sequence[Mapping[str, Any]],
    repeat_documents: Sequence[Mapping[str, Any]],
    attempt_documents: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    tables = _tables(
        release=release,
        joint_plan=joint_plan,
        panel_1_plan=panel_1_plan,
        panel_2_plan=panel_2_plan,
        panel_1_taskset=panel_1_taskset,
        panel_2_taskset=panel_2_taskset,
        panel_1_repeat=panel_1_repeat,
        panel_2_repeat=panel_2_repeat,
        primary_documents=primary_documents,
        repeat_documents=repeat_documents,
        attempt_documents=attempt_documents,
    )
    files = {f"{name}.jsonl": _jsonl(tables[name]) for name in TABLE_ORDER}
    manifest: dict[str, Any] = {
        "schema_version": "flavourbench-hf-powered-dataset-manifest-v4-two-panel-routes",
        "release_artifact_sha256": release["artifact_sha256"],
        "joint_plan_artifact_sha256": joint_plan["artifact_sha256"],
        "panel_plan_artifact_sha256s": [
            panel_1_plan["artifact_sha256"],
            panel_2_plan["artifact_sha256"],
        ],
        "taskset_artifact_sha256s": [
            panel_1_taskset["artifact_sha256"],
            panel_2_taskset["artifact_sha256"],
        ],
        "repeat_panel_artifact_sha256s": [
            panel_1_repeat["artifact_sha256"],
            panel_2_repeat["artifact_sha256"],
        ],
        "independence_unit": "anchor_ingredient",
        "panel_count": joint_plan["design"]["panel_count"],
        "scheduled_primary_tasks_per_model": joint_plan["design"][
            "scheduled_primary_tasks_per_model"
        ],
        "scheduled_repeat_tasks_per_model": joint_plan["design"][
            "scheduled_repeat_tasks_per_model"
        ],
        "unique_anchor_clusters": joint_plan["design"]["unique_anchor_clusters"],
        "shared_anchor_clusters": joint_plan["design"]["shared_anchor_clusters"],
        "files": [
            {
                "name": name,
                "rows": len(tables[name.removesuffix(".jsonl")]),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in files.items()
        ],
    }
    manifest["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    files["DATA_MANIFEST.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return files


def build(
    *,
    release_path: Path,
    joint_plan_path: Path,
    panel_1_plan_path: Path,
    panel_1_taskset_path: Path,
    panel_1_repeat_path: Path,
    panel_1_run: Path,
    panel_1_base_completion_runs: Sequence[Path] = (),
    panel_1_source_plan_v44_path: Path,
    panel_1_qwen_plan_v45_path: Path,
    panel_1_qwen_run: Path,
    panel_1_fable_run: Path | None,
    panel_1_prior_plan_v50_path: Path | None = None,
    panel_1_coverage_repair_run: Path | None = None,
    panel_1_coverage_completion_runs: Sequence[Path] = (),
    panel_1_prior_plan_v55_path: Path | None = None,
    panel_1_deepseek_repair_run: Path | None = None,
    panel_1_prior_plan_v62_path: Path | None = None,
    panel_1_prior_plan_v65_path: Path | None = None,
    panel_1_glm53_run: Path | None = None,
    panel_1_glm53_repair_runs: Sequence[Path] = (),
    panel_2_plan_path: Path,
    panel_2_taskset_path: Path,
    panel_2_repeat_path: Path,
    panel_2_run: Path,
    panel_2_base_completion_runs: Sequence[Path] = (),
    panel_2_source_plan_v49_path: Path,
    panel_2_luna_run: Path,
    panel_2_deepseek_flash_run: Path,
    panel_2_prior_plan_v52_path: Path | None = None,
    panel_2_coverage_repair_run: Path | None = None,
    panel_2_coverage_completion_runs: Sequence[Path] = (),
    panel_2_prior_plan_v54_path: Path | None = None,
    panel_2_deepseek_repair_run: Path | None = None,
    panel_2_prior_plan_v63_path: Path | None = None,
    panel_2_prior_plan_v66_path: Path | None = None,
    panel_2_glm53_run: Path | None = None,
    panel_2_glm53_repair_runs: Sequence[Path] = (),
    output: Path,
    check: bool,
) -> None:
    release = _load(release_path)
    joint_plan = _load(joint_plan_path)
    panel_1_plan = _load(panel_1_plan_path)
    panel_1_taskset = _load(panel_1_taskset_path)
    panel_1_repeat = _load(panel_1_repeat_path)
    source_plan_v44 = _load(panel_1_source_plan_v44_path)
    qwen_plan_v45 = _load(panel_1_qwen_plan_v45_path)
    panel_1_prior_plan_v50 = (
        _load(panel_1_prior_plan_v50_path) if panel_1_prior_plan_v50_path is not None else None
    )
    panel_1_prior_plan_v55 = (
        _load(panel_1_prior_plan_v55_path) if panel_1_prior_plan_v55_path is not None else None
    )
    panel_1_prior_plan_v62 = (
        _load(panel_1_prior_plan_v62_path) if panel_1_prior_plan_v62_path is not None else None
    )
    panel_1_prior_plan_v65 = (
        _load(panel_1_prior_plan_v65_path) if panel_1_prior_plan_v65_path is not None else None
    )
    panel_2_plan = _load(panel_2_plan_path)
    panel_2_source_plan_v49 = _load(panel_2_source_plan_v49_path)
    panel_2_prior_plan_v52 = (
        _load(panel_2_prior_plan_v52_path) if panel_2_prior_plan_v52_path is not None else None
    )
    panel_2_prior_plan_v54 = (
        _load(panel_2_prior_plan_v54_path) if panel_2_prior_plan_v54_path is not None else None
    )
    panel_2_prior_plan_v63 = (
        _load(panel_2_prior_plan_v63_path) if panel_2_prior_plan_v63_path is not None else None
    )
    panel_2_prior_plan_v66 = (
        _load(panel_2_prior_plan_v66_path) if panel_2_prior_plan_v66_path is not None else None
    )
    panel_2_taskset = _load(panel_2_taskset_path)
    panel_2_repeat = _load(panel_2_repeat_path)
    if (
        not _semantic_valid(release)
        or release.get("schema_version") != "flavourbench-selection-powered-joint-release-v1"
        or release.get("status") != "final_complete"
        or not (
            verify_joint_plan(joint_plan)
            or verify_joint_plan_v51(joint_plan)
            or verify_joint_plan_v53(joint_plan)
            or verify_joint_plan_v56(joint_plan)
            or verify_joint_plan_v60(joint_plan)
            or verify_joint_plan_v64(joint_plan)
            or verify_joint_plan_v67(joint_plan)
            or verify_joint_plan_v76(joint_plan)
        )
        or not (
            verify_plan_v47(panel_1_plan)
            or verify_plan_v50(panel_1_plan)
            or verify_plan_v55(panel_1_plan)
            or verify_plan_v58(panel_1_plan)
            or verify_plan_v62(panel_1_plan)
            or verify_plan_v65(panel_1_plan)
            or verify_plan_v74(panel_1_plan)
        )
        or not (
            verify_plan_v46(panel_2_plan)
            or verify_plan_v49(panel_2_plan)
            or verify_plan_v52(panel_2_plan)
            or verify_plan_v54(panel_2_plan)
            or verify_plan_v59(panel_2_plan)
            or verify_plan_v63(panel_2_plan)
            or verify_plan_v66(panel_2_plan)
            or verify_plan_v75(panel_2_plan)
        )
        or not verify_plan_v44(source_plan_v44)
        or not verify_plan_v45(qwen_plan_v45)
        or not verify_plan_v49(panel_2_source_plan_v49)
        or not verify_taskset(panel_1_taskset)
        or not verify_repeat_panel(panel_1_repeat, taskset=panel_1_taskset)
        or not verify_taskset_replication_2(panel_2_taskset)
        or not verify_repeat_panel_replication_2(panel_2_repeat, taskset=panel_2_taskset)
    ):
        raise PoweredDatasetBuildError("joint powered dataset inputs failed verification")
    for label, document, path in (
        ("panel_1_plan", panel_1_plan, panel_1_plan_path),
        ("panel_1_taskset", panel_1_taskset, panel_1_taskset_path),
        ("panel_1_repeat_panel", panel_1_repeat, panel_1_repeat_path),
        ("panel_2_plan", panel_2_plan, panel_2_plan_path),
        ("panel_2_taskset", panel_2_taskset, panel_2_taskset_path),
        ("panel_2_repeat_panel", panel_2_repeat, panel_2_repeat_path),
    ):
        _require_pin(joint_plan, label, document, path)
        if release["inputs"][label] != _pin(document, path):
            raise PoweredDatasetBuildError(f"release {label} pin differs")
    if release["inputs"]["joint_plan"] != _pin(joint_plan, joint_plan_path):
        raise PoweredDatasetBuildError("release joint-plan pin differs")
    if joint_plan.get("schema_version") in {
        "flavourbench-selection-powered-joint-analysis-plan-v67",
        "flavourbench-selection-powered-joint-analysis-plan-v76",
    }:
        lineage = release["inputs"].get("response_lineage") or {}
        is_v76 = (
            joint_plan.get("schema_version")
            == "flavourbench-selection-powered-joint-analysis-plan-v76"
        )
        response_selection_rule = (
            "first_completed_parseable_response_in_frozen_source_directory_order"
            if is_v76
            else "first_completed_response_in_frozen_source_directory_order"
        )
        if (
            lineage.get("glm53_response_selection_rule") != response_selection_rule
            or lineage.get("glm53_failed_response_artifacts_preserved") is not True
            or lineage.get("glm53_failed_response_artifacts_used_as_score_data") is not False
            or lineage.get("glm53_panel_1_source_directory_count")
            != 1 + len(panel_1_glm53_repair_runs)
            or lineage.get("glm53_panel_2_source_directory_count")
            != 1 + len(panel_2_glm53_repair_runs)
            or lineage.get("panel_1_coverage_response_selection_rule")
            != (response_selection_rule if panel_1_coverage_completion_runs else None)
            or lineage.get("panel_2_coverage_response_selection_rule")
            != (response_selection_rule if panel_2_coverage_completion_runs else None)
            or lineage.get("panel_1_coverage_completion_source_directory_count")
            != 1 + len(panel_1_coverage_completion_runs)
            or lineage.get("panel_2_coverage_completion_source_directory_count")
            != 1 + len(panel_2_coverage_completion_runs)
            or lineage.get("panel_1_coverage_failed_response_artifacts_preserved")
            is not bool(panel_1_coverage_completion_runs)
            or lineage.get("panel_2_coverage_failed_response_artifacts_preserved")
            is not bool(panel_2_coverage_completion_runs)
            or lineage.get("panel_1_coverage_failed_response_artifacts_used_as_score_data")
            is not False
            or lineage.get("panel_2_coverage_failed_response_artifacts_used_as_score_data")
            is not False
            or (
                is_v76
                and lineage.get("panel_1_base_completion_source_directory_count")
                != 1 + len(panel_1_base_completion_runs)
            )
            or (
                is_v76
                and lineage.get("panel_2_base_completion_source_directory_count")
                != 1 + len(panel_2_base_completion_runs)
            )
        ):
            raise PoweredDatasetBuildError("release completion-overlay lineage differs")
    if panel_1_plan["inputs"]["plan_v44_predecessor"] != _pin(
        source_plan_v44, panel_1_source_plan_v44_path
    ) or panel_1_plan["inputs"]["plan_v45_qwen_source"] != _pin(
        qwen_plan_v45, panel_1_qwen_plan_v45_path
    ):
        raise PoweredDatasetBuildError("panel 1 composite source pin differs")
    joint_v56 = verify_joint_plan_v56(joint_plan)
    joint_v60 = verify_joint_plan_v60(joint_plan)
    joint_v64 = verify_joint_plan_v64(joint_plan)
    joint_v67 = verify_joint_plan_v67(joint_plan)
    joint_v76 = verify_joint_plan_v76(joint_plan)
    if joint_v56 != (verify_plan_v55(panel_1_plan) and verify_plan_v54(panel_2_plan)):
        raise PoweredDatasetBuildError("v56 joint and panel plan versions differ")
    if joint_v60 != (verify_plan_v58(panel_1_plan) and verify_plan_v59(panel_2_plan)):
        raise PoweredDatasetBuildError("v60 joint and panel plan versions differ")
    if joint_v64 != (verify_plan_v62(panel_1_plan) and verify_plan_v63(panel_2_plan)):
        raise PoweredDatasetBuildError("v64 joint and panel plan versions differ")
    if joint_v67 != (verify_plan_v65(panel_1_plan) and verify_plan_v66(panel_2_plan)):
        raise PoweredDatasetBuildError("v67 joint and panel plan versions differ")
    if joint_v76 != (verify_plan_v74(panel_1_plan) and verify_plan_v75(panel_2_plan)):
        raise PoweredDatasetBuildError("v76 joint and panel plan versions differ")
    if (
        verify_plan_v58(panel_1_plan)
        or verify_plan_v62(panel_1_plan)
        or verify_plan_v65(panel_1_plan)
        or verify_plan_v74(panel_1_plan)
    ):
        if (
            panel_1_prior_plan_v50 is None
            or panel_1_prior_plan_v50_path is None
            or panel_1_coverage_repair_run is None
            or panel_1_prior_plan_v55 is None
            or panel_1_prior_plan_v55_path is None
            or panel_1_deepseek_repair_run is None
            or panel_1_fable_run is not None
            or not verify_plan_v50(panel_1_prior_plan_v50)
            or not verify_plan_v55(panel_1_prior_plan_v55)
            or panel_1_plan["inputs"]["plan_v55_predecessor"]
            != _pin(panel_1_prior_plan_v55, panel_1_prior_plan_v55_path)
            or panel_1_prior_plan_v55["inputs"]["plan_v50_predecessor"]
            != _pin(panel_1_prior_plan_v50, panel_1_prior_plan_v50_path)
        ):
            raise PoweredDatasetBuildError("panel 1 DeepSeek-repair source pin differs")
        if (verify_plan_v65(panel_1_plan) or verify_plan_v74(panel_1_plan)) and (
            panel_1_prior_plan_v62 is None
            or panel_1_prior_plan_v62_path is None
            or panel_1_glm53_run is None
            or not verify_plan_v62(panel_1_prior_plan_v62)
            or panel_1_plan["inputs"]["plan_v62_predecessor"]
            != _pin(panel_1_prior_plan_v62, panel_1_prior_plan_v62_path)
        ):
            raise PoweredDatasetBuildError("panel 1 GLM-5.3 source pin differs")
        if verify_plan_v74(panel_1_plan) and (
            panel_1_prior_plan_v65 is None
            or panel_1_prior_plan_v65_path is None
            or not verify_plan_v65(panel_1_prior_plan_v65)
            or panel_1_plan["inputs"]["plan_v65_predecessor"]
            != _pin(panel_1_prior_plan_v65, panel_1_prior_plan_v65_path)
            or panel_1_prior_plan_v65["inputs"]["plan_v62_predecessor"]
            != _pin(panel_1_prior_plan_v62, panel_1_prior_plan_v62_path)
        ):
            raise PoweredDatasetBuildError("panel 1 v74 predecessor chain differs")
    elif verify_plan_v55(panel_1_plan):
        if (
            panel_1_prior_plan_v50 is None
            or panel_1_prior_plan_v50_path is None
            or panel_1_coverage_repair_run is None
            or panel_1_prior_plan_v55 is not None
            or panel_1_deepseek_repair_run is not None
            or panel_1_fable_run is not None
            or not verify_plan_v50(panel_1_prior_plan_v50)
            or panel_1_plan["inputs"]["plan_v50_predecessor"]
            != _pin(panel_1_prior_plan_v50, panel_1_prior_plan_v50_path)
        ):
            raise PoweredDatasetBuildError("panel 1 coverage-repair source pin differs")
    elif any(
        value is not None
        for value in (
            panel_1_prior_plan_v50,
            panel_1_coverage_repair_run,
            panel_1_prior_plan_v55,
            panel_1_deepseek_repair_run,
            panel_1_prior_plan_v62,
            panel_1_prior_plan_v65,
            panel_1_glm53_run,
        )
    ):
        raise PoweredDatasetBuildError("panel 1 coverage-repair sources require v55")
    elif verify_plan_v50(panel_1_plan) and panel_1_fable_run is None:
        raise PoweredDatasetBuildError("panel 1 v50 requires a Fable response directory")

    if (
        verify_plan_v52(panel_2_plan)
        or verify_plan_v54(panel_2_plan)
        or verify_plan_v59(panel_2_plan)
        or verify_plan_v63(panel_2_plan)
        or verify_plan_v66(panel_2_plan)
        or verify_plan_v75(panel_2_plan)
    ) and panel_2_plan["inputs"].get("plan_v49_predecessor") != _pin(
        panel_2_source_plan_v49, panel_2_source_plan_v49_path
    ):
        raise PoweredDatasetBuildError("panel 2 composite source pin differs")
    if (
        verify_plan_v59(panel_2_plan)
        or verify_plan_v63(panel_2_plan)
        or verify_plan_v66(panel_2_plan)
        or verify_plan_v75(panel_2_plan)
    ):
        if (
            panel_2_prior_plan_v52 is None
            or panel_2_prior_plan_v52_path is None
            or panel_2_coverage_repair_run is None
            or panel_2_prior_plan_v54 is None
            or panel_2_prior_plan_v54_path is None
            or panel_2_deepseek_repair_run is None
            or not verify_plan_v52(panel_2_prior_plan_v52)
            or not verify_plan_v54(panel_2_prior_plan_v54)
            or panel_2_plan["inputs"]["plan_v54_predecessor"]
            != _pin(panel_2_prior_plan_v54, panel_2_prior_plan_v54_path)
            or panel_2_prior_plan_v54["inputs"]["plan_v52_predecessor"]
            != _pin(panel_2_prior_plan_v52, panel_2_prior_plan_v52_path)
        ):
            raise PoweredDatasetBuildError("panel 2 DeepSeek-repair source pin differs")
        if (verify_plan_v66(panel_2_plan) or verify_plan_v75(panel_2_plan)) and (
            panel_2_prior_plan_v63 is None
            or panel_2_prior_plan_v63_path is None
            or panel_2_glm53_run is None
            or not verify_plan_v63(panel_2_prior_plan_v63)
            or panel_2_plan["inputs"]["plan_v63_predecessor"]
            != _pin(panel_2_prior_plan_v63, panel_2_prior_plan_v63_path)
        ):
            raise PoweredDatasetBuildError("panel 2 GLM-5.3 source pin differs")
        if verify_plan_v75(panel_2_plan) and (
            panel_2_prior_plan_v66 is None
            or panel_2_prior_plan_v66_path is None
            or not verify_plan_v66(panel_2_prior_plan_v66)
            or panel_2_plan["inputs"]["plan_v66_predecessor"]
            != _pin(panel_2_prior_plan_v66, panel_2_prior_plan_v66_path)
            or panel_2_prior_plan_v66["inputs"]["plan_v63_predecessor"]
            != _pin(panel_2_prior_plan_v63, panel_2_prior_plan_v63_path)
        ):
            raise PoweredDatasetBuildError("panel 2 v75 predecessor chain differs")
    elif verify_plan_v54(panel_2_plan):
        if (
            panel_2_prior_plan_v52 is None
            or panel_2_prior_plan_v52_path is None
            or panel_2_coverage_repair_run is None
            or panel_2_prior_plan_v54 is not None
            or panel_2_deepseek_repair_run is not None
            or not verify_plan_v52(panel_2_prior_plan_v52)
            or panel_2_plan["inputs"]["plan_v52_predecessor"]
            != _pin(panel_2_prior_plan_v52, panel_2_prior_plan_v52_path)
        ):
            raise PoweredDatasetBuildError("panel 2 coverage-repair source pin differs")
    elif any(
        value is not None
        for value in (
            panel_2_prior_plan_v52,
            panel_2_coverage_repair_run,
            panel_2_prior_plan_v54,
            panel_2_deepseek_repair_run,
            panel_2_prior_plan_v63,
            panel_2_prior_plan_v66,
            panel_2_glm53_run,
        )
    ):
        raise PoweredDatasetBuildError("panel 2 coverage-repair sources require v54")

    model_ids = [str(row["model_id"]) for row in joint_plan["roster"]["models"]]
    panel_1_sources = {
        model_id: (panel_1_run, *panel_1_base_completion_runs) for model_id in model_ids
    }
    panel_1_sources[QWEN_MODEL_ID] = panel_1_qwen_run
    if (
        verify_plan_v58(panel_1_plan)
        or verify_plan_v62(panel_1_plan)
        or verify_plan_v65(panel_1_plan)
        or verify_plan_v74(panel_1_plan)
    ):
        assert panel_1_coverage_repair_run is not None
        assert panel_1_deepseek_repair_run is not None
        for model_id in COVERAGE_REPAIR_MODEL_IDS:
            panel_1_sources[model_id] = (
                panel_1_coverage_repair_run,
                *panel_1_coverage_completion_runs,
            )
        panel_1_sources[DEEPSEEK_PRO_MODEL_ID] = panel_1_deepseek_repair_run
        if verify_plan_v65(panel_1_plan) or verify_plan_v74(panel_1_plan):
            assert panel_1_glm53_run is not None
            panel_1_sources[GLM53_MODEL_ID] = (
                panel_1_glm53_run,
                *panel_1_glm53_repair_runs,
            )
    elif verify_plan_v55(panel_1_plan):
        assert panel_1_coverage_repair_run is not None
        for model_id in COVERAGE_REPAIR_MODEL_IDS:
            panel_1_sources[model_id] = (
                panel_1_coverage_repair_run,
                *panel_1_coverage_completion_runs,
            )
    elif verify_plan_v50(panel_1_plan):
        assert panel_1_fable_run is not None
        panel_1_sources[FABLE_MODEL_ID] = panel_1_fable_run
    panel_2_sources = {
        model_id: (panel_2_run, *panel_2_base_completion_runs) for model_id in model_ids
    }
    if (
        verify_plan_v59(panel_2_plan)
        or verify_plan_v63(panel_2_plan)
        or verify_plan_v66(panel_2_plan)
        or verify_plan_v75(panel_2_plan)
    ):
        assert panel_2_coverage_repair_run is not None
        assert panel_2_deepseek_repair_run is not None
        panel_2_sources[LUNA_MODEL_ID] = panel_2_luna_run
        panel_2_sources[DEEPSEEK_FLASH_MODEL_ID] = panel_2_deepseek_flash_run
        for model_id in COVERAGE_REPAIR_MODEL_IDS:
            panel_2_sources[model_id] = (
                panel_2_coverage_repair_run,
                *panel_2_coverage_completion_runs,
            )
        panel_2_sources[DEEPSEEK_PRO_MODEL_ID] = panel_2_deepseek_repair_run
        if verify_plan_v66(panel_2_plan) or verify_plan_v75(panel_2_plan):
            assert panel_2_glm53_run is not None
            panel_2_sources[GLM53_MODEL_ID] = (
                panel_2_glm53_run,
                *panel_2_glm53_repair_runs,
            )
    elif verify_plan_v54(panel_2_plan):
        assert panel_2_coverage_repair_run is not None
        panel_2_sources[LUNA_MODEL_ID] = panel_2_luna_run
        panel_2_sources[DEEPSEEK_FLASH_MODEL_ID] = panel_2_deepseek_flash_run
        for model_id in COVERAGE_REPAIR_MODEL_IDS:
            panel_2_sources[model_id] = (
                panel_2_coverage_repair_run,
                *panel_2_coverage_completion_runs,
            )
    elif verify_plan_v52(panel_2_plan):
        panel_2_sources[LUNA_MODEL_ID] = panel_2_luna_run
        panel_2_sources[DEEPSEEK_FLASH_MODEL_ID] = panel_2_deepseek_flash_run
    primary_1 = _response_documents(
        panel="primary",
        final_plan=panel_1_plan,
        task_ids=[str(task["task_id"]) for task in panel_1_taskset["tasks"]],
        source_directories=panel_1_sources,
    )
    repeat_1 = _response_documents(
        panel="repeat",
        final_plan=panel_1_plan,
        task_ids=[str(task["task_id"]) for task in panel_1_repeat["tasks"]],
        source_directories=panel_1_sources,
    )
    primary_2 = _response_documents(
        panel="primary",
        final_plan=panel_2_plan,
        task_ids=[str(task["task_id"]) for task in panel_2_taskset["tasks"]],
        source_directories=panel_2_sources,
    )
    repeat_2 = _response_documents(
        panel="repeat",
        final_plan=panel_2_plan,
        task_ids=[str(task["task_id"]) for task in panel_2_repeat["tasks"]],
        source_directories=panel_2_sources,
    )
    for label, documents in (
        ("panel_1_primary", primary_1),
        ("panel_1_repeat", repeat_1),
        ("panel_2_primary", primary_2),
        ("panel_2_repeat", repeat_2),
    ):
        _require_response_commitment(release, label, documents)
    primary_documents = primary_1 + primary_2
    repeat_documents = repeat_1 + repeat_2
    design = joint_plan["design"]
    expected_primary = len(model_ids) * int(design["scheduled_primary_tasks_per_model"])
    expected_repeat = len(model_ids) * int(design["scheduled_repeat_tasks_per_model"])
    if len(primary_documents) != expected_primary or len(repeat_documents) != expected_repeat:
        raise PoweredDatasetBuildError("joint dataset response grid is incomplete")
    journal_sources: dict[str, Path] = {}
    for panel_label, sources in (("panel1", panel_1_sources), ("panel2", panel_2_sources)):
        for model_id, directory_value in sources.items():
            directories = (
                (directory_value,) if isinstance(directory_value, Path) else tuple(directory_value)
            )
            for source_index, directory in enumerate(directories):
                journal_sources[f"{panel_label}:{model_id}:{source_index}"] = directory
    attempt_documents = _provider_attempt_documents(
        response_documents=primary_documents + repeat_documents,
        source_directories=journal_sources,
    )
    files = _expected_files(
        release=release,
        joint_plan=joint_plan,
        panel_1_plan=panel_1_plan,
        panel_2_plan=panel_2_plan,
        panel_1_taskset=panel_1_taskset,
        panel_2_taskset=panel_2_taskset,
        panel_1_repeat=panel_1_repeat,
        panel_2_repeat=panel_2_repeat,
        primary_documents=primary_documents,
        repeat_documents=repeat_documents,
        attempt_documents=attempt_documents,
    )
    if check:
        drift = [
            name
            for name, payload in files.items()
            if not (output / name).is_file() or (output / name).read_bytes() != payload
        ]
        if drift:
            raise PoweredDatasetBuildError(f"joint dataset differs: {', '.join(drift)}")
        print(f"OK: {len(files)} joint dataset files")
        return
    for name, payload in files.items():
        _write_atomic(output / name, payload)
    print(f"Wrote {len(files)} joint dataset files to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--joint-plan", type=Path, required=True)
    parser.add_argument("--panel-1-plan", type=Path, required=True)
    parser.add_argument("--panel-1-taskset", type=Path, required=True)
    parser.add_argument("--panel-1-repeat-panel", type=Path, required=True)
    parser.add_argument("--panel-1-run-directory", type=Path, required=True)
    parser.add_argument("--panel-1-base-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-1-source-plan-v44", type=Path, required=True)
    parser.add_argument("--panel-1-qwen-plan-v45", type=Path, required=True)
    parser.add_argument("--panel-1-qwen-run-directory", type=Path, required=True)
    parser.add_argument("--panel-1-fable-run-directory", type=Path)
    parser.add_argument("--panel-1-prior-plan-v50", type=Path)
    parser.add_argument("--panel-1-coverage-repair-run-directory", type=Path)
    parser.add_argument("--panel-1-coverage-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-1-prior-plan-v55", type=Path)
    parser.add_argument("--panel-1-deepseek-repair-run-directory", type=Path)
    parser.add_argument("--panel-1-prior-plan-v62", type=Path)
    parser.add_argument("--panel-1-prior-plan-v65", type=Path)
    parser.add_argument("--panel-1-glm53-run-directory", type=Path)
    parser.add_argument("--panel-1-glm53-repair-run-directory", type=Path, action="append")
    parser.add_argument("--panel-2-plan", type=Path, required=True)
    parser.add_argument("--panel-2-taskset", type=Path, required=True)
    parser.add_argument("--panel-2-repeat-panel", type=Path, required=True)
    parser.add_argument("--panel-2-run-directory", type=Path, required=True)
    parser.add_argument("--panel-2-base-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-2-source-plan-v49", type=Path, required=True)
    parser.add_argument("--panel-2-luna-run-directory", type=Path, required=True)
    parser.add_argument("--panel-2-deepseek-flash-run-directory", type=Path, required=True)
    parser.add_argument("--panel-2-prior-plan-v52", type=Path)
    parser.add_argument("--panel-2-coverage-repair-run-directory", type=Path)
    parser.add_argument("--panel-2-coverage-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-2-prior-plan-v54", type=Path)
    parser.add_argument("--panel-2-deepseek-repair-run-directory", type=Path)
    parser.add_argument("--panel-2-prior-plan-v63", type=Path)
    parser.add_argument("--panel-2-prior-plan-v66", type=Path)
    parser.add_argument("--panel-2-glm53-run-directory", type=Path)
    parser.add_argument("--panel-2-glm53-repair-run-directory", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(
        release_path=args.release,
        joint_plan_path=args.joint_plan,
        panel_1_plan_path=args.panel_1_plan,
        panel_1_taskset_path=args.panel_1_taskset,
        panel_1_repeat_path=args.panel_1_repeat_panel,
        panel_1_run=args.panel_1_run_directory,
        panel_1_base_completion_runs=(args.panel_1_base_completion_run_directory or ()),
        panel_1_source_plan_v44_path=args.panel_1_source_plan_v44,
        panel_1_qwen_plan_v45_path=args.panel_1_qwen_plan_v45,
        panel_1_qwen_run=args.panel_1_qwen_run_directory,
        panel_1_fable_run=args.panel_1_fable_run_directory,
        panel_1_prior_plan_v50_path=args.panel_1_prior_plan_v50,
        panel_1_coverage_repair_run=args.panel_1_coverage_repair_run_directory,
        panel_1_coverage_completion_runs=(args.panel_1_coverage_completion_run_directory or ()),
        panel_1_prior_plan_v55_path=args.panel_1_prior_plan_v55,
        panel_1_deepseek_repair_run=args.panel_1_deepseek_repair_run_directory,
        panel_1_prior_plan_v62_path=args.panel_1_prior_plan_v62,
        panel_1_prior_plan_v65_path=args.panel_1_prior_plan_v65,
        panel_1_glm53_run=args.panel_1_glm53_run_directory,
        panel_1_glm53_repair_runs=args.panel_1_glm53_repair_run_directory or (),
        panel_2_plan_path=args.panel_2_plan,
        panel_2_taskset_path=args.panel_2_taskset,
        panel_2_repeat_path=args.panel_2_repeat_panel,
        panel_2_run=args.panel_2_run_directory,
        panel_2_base_completion_runs=(args.panel_2_base_completion_run_directory or ()),
        panel_2_source_plan_v49_path=args.panel_2_source_plan_v49,
        panel_2_luna_run=args.panel_2_luna_run_directory,
        panel_2_deepseek_flash_run=args.panel_2_deepseek_flash_run_directory,
        panel_2_prior_plan_v52_path=args.panel_2_prior_plan_v52,
        panel_2_coverage_repair_run=args.panel_2_coverage_repair_run_directory,
        panel_2_coverage_completion_runs=(args.panel_2_coverage_completion_run_directory or ()),
        panel_2_prior_plan_v54_path=args.panel_2_prior_plan_v54,
        panel_2_deepseek_repair_run=args.panel_2_deepseek_repair_run_directory,
        panel_2_prior_plan_v63_path=args.panel_2_prior_plan_v63,
        panel_2_prior_plan_v66_path=args.panel_2_prior_plan_v66,
        panel_2_glm53_run=args.panel_2_glm53_run_directory,
        panel_2_glm53_repair_runs=args.panel_2_glm53_repair_run_directory or (),
        output=args.output,
        check=args.check,
    )


if __name__ == "__main__":
    main()
