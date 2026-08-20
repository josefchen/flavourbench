"""Restore the two-panel powered response lineages from Hugging Face JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from restore_powered_runs import (
    PoweredRunRestoreError,
    _canonical,
    _load_jsonl,
    _load_release,
    _write_no_replace,
)

from flavourbench.epicure_selection_route_manifest_v54 import (
    REPLACEMENT_MODEL_IDS as COVERAGE_REPAIR_MODEL_IDS,
)

JOINT_SCHEMA = "flavourbench-selection-powered-joint-release-v1"


def _source_plans(
    release: Mapping[str, Any],
) -> tuple[
    str,
    str,
    str | None,
    str | None,
    str | None,
    str | None,
    str,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    lineage = release["inputs"]["response_lineage"]
    fable_value = lineage.get("panel_1_fable_replacement_plan_sha256")
    fable = str(fable_value) if fable_value is not None else None
    panel_1_coverage_value = lineage.get("panel_1_coverage_repair_plan_sha256")
    panel_1_coverage = str(panel_1_coverage_value) if panel_1_coverage_value is not None else None
    panel_1_deepseek_value = lineage.get("panel_1_deepseek_repair_plan_sha256")
    panel_1_deepseek = str(panel_1_deepseek_value) if panel_1_deepseek_value is not None else None
    panel_1_glm_value = lineage.get("glm53_panel_1_plan_sha256")
    panel_1_glm = str(panel_1_glm_value) if panel_1_glm_value is not None else None
    panel_2_plan = str(lineage["panel_2_plan_sha256"])
    panel_2_base = str(lineage.get("panel_2_base_plan_sha256") or panel_2_plan)
    replacement_value = lineage.get("panel_2_replacement_plan_sha256")
    panel_2_replacement = str(replacement_value) if replacement_value is not None else None
    panel_2_coverage_value = lineage.get("panel_2_coverage_repair_plan_sha256")
    panel_2_coverage = str(panel_2_coverage_value) if panel_2_coverage_value is not None else None
    panel_2_deepseek_value = lineage.get("panel_2_deepseek_repair_plan_sha256")
    panel_2_deepseek = str(panel_2_deepseek_value) if panel_2_deepseek_value is not None else None
    panel_2_glm_value = lineage.get("glm53_panel_2_plan_sha256")
    panel_2_glm = str(panel_2_glm_value) if panel_2_glm_value is not None else None
    values = (
        str(lineage["panel_1_base_plan_sha256"]),
        str(lineage["panel_1_qwen_replacement_plan_sha256"]),
        fable,
        panel_1_coverage,
        panel_1_deepseek,
        panel_1_glm,
        panel_2_base,
        panel_2_replacement,
        panel_2_coverage,
        panel_2_deepseek,
        panel_2_glm,
    )
    present = [value for value in values if value is not None]
    if any(len(value) != 64 for value in present) or len(set(present)) != len(present):
        raise PoweredRunRestoreError("joint response plan lineage is malformed")
    if (
        lineage.get("panel_1_superseded_qwen_responses_used") is not False
        or (
            fable is not None
            and lineage.get("panel_1_superseded_fable_responses_used") is not False
        )
        or lineage.get("panel_2_reuses_panel_1_responses") is not False
        or (
            panel_2_replacement is not None
            and lineage.get("panel_2_superseded_route_responses_used") is not False
        )
        or (
            panel_1_coverage is not None
            and lineage.get("panel_1_superseded_coverage_route_responses_used") is not False
        )
        or (
            panel_2_coverage is not None
            and lineage.get("panel_2_superseded_coverage_route_responses_used") is not False
        )
        or (
            panel_1_deepseek is not None
            and lineage.get("panel_1_superseded_deepseek_route_responses_used") is not False
        )
        or (
            panel_2_deepseek is not None
            and lineage.get("panel_2_superseded_deepseek_route_responses_used") is not False
        )
        or (panel_1_glm is None) != (panel_2_glm is None)
        or (
            panel_1_glm is not None
            and (
                lineage.get("glm53_limited_run_model_ids") != ["z-ai/glm-5.3"]
                or lineage.get("glm53_finite_cli_only") is not True
                or lineage.get("glm53_standing_service") is not False
                or lineage.get("glm53_automatic_fallback") is not False
            )
        )
    ):
        raise PoweredRunRestoreError("joint response lineage permits pooling or reuse")
    return values


def _require_response_commitment(
    release: Mapping[str, Any], label: str, rows: list[dict[str, Any]]
) -> None:
    artifacts = sorted(str(row["artifact_sha256"]) for row in rows)
    observed = {
        "count": len(rows),
        "artifact_set_sha256": hashlib.sha256(_canonical(artifacts)).hexdigest(),
        "spend_micros": sum(
            int((row.get("generation") or {}).get("cost_micros") or 0) for row in rows
        ),
    }
    if len(set(artifacts)) != len(artifacts) or release["inputs"][label] != observed:
        raise PoweredRunRestoreError(f"{label} differs from the release commitment")


def _destination(
    *,
    row: Mapping[str, Any],
    panel_1_base_plan: str,
    panel_1_qwen_plan: str,
    panel_1_fable_plan: str | None,
    panel_1_coverage_plan: str | None,
    panel_1_coverage_model_ids: set[str],
    panel_1_deepseek_plan: str | None,
    panel_1_deepseek_model_ids: set[str],
    panel_1_glm_plan: str | None,
    panel_1_glm_model_ids: set[str],
    panel_2_base_plan: str,
    panel_2_replacement_plan: str | None,
    panel_2_replacement_model_ids: set[str],
    panel_2_coverage_plan: str | None,
    panel_2_coverage_model_ids: set[str],
    panel_2_deepseek_plan: str | None,
    panel_2_deepseek_model_ids: set[str],
    panel_2_glm_plan: str | None,
    panel_2_glm_model_ids: set[str],
    panel_1_run: Path,
    panel_1_qwen_run: Path,
    panel_1_fable_run: Path | None,
    panel_1_coverage_repair_run: Path | None,
    panel_1_deepseek_repair_run: Path | None,
    panel_1_glm_run: Path | None,
    panel_2_run: Path,
    panel_2_luna_run: Path | None,
    panel_2_deepseek_flash_run: Path | None,
    panel_2_coverage_repair_run: Path | None,
    panel_2_deepseek_repair_run: Path | None,
    panel_2_glm_run: Path | None,
) -> Path:
    plan_sha256 = str(row.get("plan_sha256", ""))
    if plan_sha256 == panel_1_base_plan:
        root = panel_1_run
    elif plan_sha256 == panel_1_qwen_plan:
        root = panel_1_qwen_run
    elif panel_1_fable_plan is not None and plan_sha256 == panel_1_fable_plan:
        if panel_1_fable_run is None:
            raise PoweredRunRestoreError("Fable response lineage has no destination")
        root = panel_1_fable_run
    elif panel_1_coverage_plan is not None and plan_sha256 == panel_1_coverage_plan:
        if (
            str(row.get("model_id", "")) not in panel_1_coverage_model_ids
            or panel_1_coverage_repair_run is None
        ):
            raise PoweredRunRestoreError("unexpected panel-1 coverage-repair response")
        root = panel_1_coverage_repair_run
    elif panel_1_deepseek_plan is not None and plan_sha256 == panel_1_deepseek_plan:
        if (
            str(row.get("model_id", "")) not in panel_1_deepseek_model_ids
            or panel_1_deepseek_repair_run is None
        ):
            raise PoweredRunRestoreError("unexpected panel-1 DeepSeek-repair response")
        root = panel_1_deepseek_repair_run
    elif panel_1_glm_plan is not None and plan_sha256 == panel_1_glm_plan:
        if str(row.get("model_id", "")) not in panel_1_glm_model_ids or panel_1_glm_run is None:
            raise PoweredRunRestoreError("unexpected panel-1 GLM-5.3 response")
        root = panel_1_glm_run
    elif plan_sha256 == panel_2_base_plan:
        root = panel_2_run
    elif panel_2_replacement_plan is not None and plan_sha256 == panel_2_replacement_plan:
        model_id = str(row.get("model_id", ""))
        if model_id == "openai/gpt-5.6-luna-pro" and model_id in panel_2_replacement_model_ids:
            if panel_2_luna_run is None:
                raise PoweredRunRestoreError("Luna replacement has no destination")
            root = panel_2_luna_run
        elif (
            model_id == "deepseek/deepseek-v4-flash-0731"
            and model_id in panel_2_replacement_model_ids
        ):
            if panel_2_deepseek_flash_run is None:
                raise PoweredRunRestoreError("DeepSeek Flash replacement has no destination")
            root = panel_2_deepseek_flash_run
        else:
            raise PoweredRunRestoreError("unexpected panel-2 replacement model")
    elif panel_2_coverage_plan is not None and plan_sha256 == panel_2_coverage_plan:
        if (
            str(row.get("model_id", "")) not in panel_2_coverage_model_ids
            or panel_2_coverage_repair_run is None
        ):
            raise PoweredRunRestoreError("unexpected panel-2 coverage-repair response")
        root = panel_2_coverage_repair_run
    elif panel_2_deepseek_plan is not None and plan_sha256 == panel_2_deepseek_plan:
        if (
            str(row.get("model_id", "")) not in panel_2_deepseek_model_ids
            or panel_2_deepseek_repair_run is None
        ):
            raise PoweredRunRestoreError("unexpected panel-2 DeepSeek-repair response")
        root = panel_2_deepseek_repair_run
    elif panel_2_glm_plan is not None and plan_sha256 == panel_2_glm_plan:
        if str(row.get("model_id", "")) not in panel_2_glm_model_ids or panel_2_glm_run is None:
            raise PoweredRunRestoreError("unexpected panel-2 GLM-5.3 response")
        root = panel_2_glm_run
    else:
        raise PoweredRunRestoreError("response uses a plan outside the joint lineage")
    panel = str(row.get("panel", ""))
    slot_id = str(row.get("slot_id", ""))
    cell_id = str(row.get("cell_id", ""))
    artifact = str(row.get("artifact_sha256", ""))
    if panel not in {"primary", "repeat"}:
        raise PoweredRunRestoreError(f"unexpected response panel: {panel}")
    if not slot_id or len(cell_id) != 64 or len(artifact) != 64:
        raise PoweredRunRestoreError("response identity is malformed")
    return root / "responses" / panel / slot_id / f"response-{cell_id}-{artifact}.json"


def _validate_grid(
    release: Mapping[str, Any],
    primary: list[dict[str, Any]],
    repeat: list[dict[str, Any]],
    *,
    panel_1_base_plan: str,
    panel_1_qwen_plan: str,
    panel_1_fable_plan: str | None,
    panel_2_base_plan: str,
    panel_2_replacement_plan: str | None,
    panel_2_replacement_model_ids: set[str],
    panel_1_coverage_plan: str | None = None,
    panel_1_coverage_model_ids: set[str] | None = None,
    panel_1_deepseek_plan: str | None = None,
    panel_1_deepseek_model_ids: set[str] | None = None,
    panel_1_glm_plan: str | None = None,
    panel_1_glm_model_ids: set[str] | None = None,
    panel_2_coverage_plan: str | None = None,
    panel_2_coverage_model_ids: set[str] | None = None,
    panel_2_deepseek_plan: str | None = None,
    panel_2_deepseek_model_ids: set[str] | None = None,
    panel_2_glm_plan: str | None = None,
    panel_2_glm_model_ids: set[str] | None = None,
) -> None:
    panel_1_coverage_model_ids = panel_1_coverage_model_ids or set()
    panel_1_deepseek_model_ids = panel_1_deepseek_model_ids or set()
    panel_1_glm_model_ids = panel_1_glm_model_ids or set()
    panel_2_coverage_model_ids = panel_2_coverage_model_ids or set()
    panel_2_deepseek_model_ids = panel_2_deepseek_model_ids or set()
    panel_2_glm_model_ids = panel_2_glm_model_ids or set()
    models = {str(row["model_id"]) for row in release["analysis"]["models"]}
    design = release["analysis"]["design"]
    tasks_per_panel = int(design["primary_tasks_per_panel"])
    repeats_per_panel = int(design["repeat_tasks_per_panel"])
    expected_primary = len(models) * tasks_per_panel * 2
    expected_repeat = len(models) * repeats_per_panel * 2
    if len(primary) != expected_primary or len(repeat) != expected_repeat:
        raise PoweredRunRestoreError("joint downloaded response cardinality failed")
    if any(row.get("panel") != "primary" for row in primary) or any(
        row.get("panel") != "repeat" for row in repeat
    ):
        raise PoweredRunRestoreError("joint response table panel assignment failed")

    plan_by_model: dict[str, Counter[str]] = {model_id: Counter() for model_id in models}
    identities: set[tuple[str, str, str]] = set()
    for row in primary + repeat:
        model_id = str(row.get("model_id", ""))
        plan_sha256 = str(row.get("plan_sha256", ""))
        panel = str(row["panel"])
        task_id = str(row.get("task_id", ""))
        allowed_plans = {panel_1_base_plan, panel_1_qwen_plan, panel_2_base_plan}
        if panel_1_fable_plan is not None:
            allowed_plans.add(panel_1_fable_plan)
        if panel_1_coverage_plan is not None:
            allowed_plans.add(panel_1_coverage_plan)
        if panel_1_deepseek_plan is not None:
            allowed_plans.add(panel_1_deepseek_plan)
        if panel_1_glm_plan is not None:
            allowed_plans.add(panel_1_glm_plan)
        if panel_2_replacement_plan is not None:
            allowed_plans.add(panel_2_replacement_plan)
        if panel_2_coverage_plan is not None:
            allowed_plans.add(panel_2_coverage_plan)
        if panel_2_deepseek_plan is not None:
            allowed_plans.add(panel_2_deepseek_plan)
        if panel_2_glm_plan is not None:
            allowed_plans.add(panel_2_glm_plan)
        if model_id not in models or plan_sha256 not in allowed_plans:
            raise PoweredRunRestoreError("joint response model or plan is outside the release")
        identity = (panel, model_id, task_id)
        if identity in identities:
            raise PoweredRunRestoreError("joint response task identity is duplicated")
        identities.add(identity)
        plan_by_model[model_id][f"{panel}:{plan_sha256}"] += 1

    qwen_models = {
        model_id
        for model_id, counts in plan_by_model.items()
        if counts[f"primary:{panel_1_qwen_plan}"] == tasks_per_panel
        and counts[f"repeat:{panel_1_qwen_plan}"] == repeats_per_panel
    }
    if len(qwen_models) != 1:
        raise PoweredRunRestoreError("joint Qwen replacement lineage is not exactly one model")
    qwen_model = next(iter(qwen_models))
    fable_model = None
    if panel_1_fable_plan is not None:
        fable_models = {
            model_id
            for model_id, counts in plan_by_model.items()
            if counts[f"primary:{panel_1_fable_plan}"] == tasks_per_panel
            and counts[f"repeat:{panel_1_fable_plan}"] == repeats_per_panel
        }
        if len(fable_models) != 1 or qwen_model in fable_models:
            raise PoweredRunRestoreError("joint Fable replacement lineage is not exactly one model")
        fable_model = next(iter(fable_models))
    observed_panel_1_coverage: set[str] = set()
    if panel_1_coverage_plan is not None:
        observed_panel_1_coverage = {
            model_id
            for model_id, counts in plan_by_model.items()
            if counts[f"primary:{panel_1_coverage_plan}"] == tasks_per_panel
            and counts[f"repeat:{panel_1_coverage_plan}"] == repeats_per_panel
        }
        expected_panel_1_coverage = panel_1_coverage_model_ids - panel_1_deepseek_model_ids
        if observed_panel_1_coverage != expected_panel_1_coverage:
            raise PoweredRunRestoreError("panel-1 coverage-repair lineage differs from release")
    observed_panel_2_replacements: set[str] = set()
    observed_panel_1_deepseek: set[str] = set()
    if panel_1_deepseek_plan is not None:
        observed_panel_1_deepseek = {
            model_id
            for model_id, counts in plan_by_model.items()
            if counts[f"primary:{panel_1_deepseek_plan}"] == tasks_per_panel
            and counts[f"repeat:{panel_1_deepseek_plan}"] == repeats_per_panel
        }
        if observed_panel_1_deepseek != panel_1_deepseek_model_ids:
            raise PoweredRunRestoreError("panel-1 DeepSeek-repair lineage differs from release")
    observed_panel_1_glm: set[str] = set()
    if panel_1_glm_plan is not None:
        observed_panel_1_glm = {
            model_id
            for model_id, counts in plan_by_model.items()
            if counts[f"primary:{panel_1_glm_plan}"] == tasks_per_panel
            and counts[f"repeat:{panel_1_glm_plan}"] == repeats_per_panel
        }
        if observed_panel_1_glm != panel_1_glm_model_ids:
            raise PoweredRunRestoreError("panel-1 GLM-5.3 lineage differs from release")
    if panel_2_replacement_plan is not None:
        observed_panel_2_replacements = {
            model_id
            for model_id, counts in plan_by_model.items()
            if counts[f"primary:{panel_2_replacement_plan}"] == tasks_per_panel
            and counts[f"repeat:{panel_2_replacement_plan}"] == repeats_per_panel
        }
        if observed_panel_2_replacements != panel_2_replacement_model_ids:
            raise PoweredRunRestoreError("panel-2 replacement lineage differs from release")
    observed_panel_2_coverage: set[str] = set()
    if panel_2_coverage_plan is not None:
        observed_panel_2_coverage = {
            model_id
            for model_id, counts in plan_by_model.items()
            if counts[f"primary:{panel_2_coverage_plan}"] == tasks_per_panel
            and counts[f"repeat:{panel_2_coverage_plan}"] == repeats_per_panel
        }
        expected_panel_2_coverage = panel_2_coverage_model_ids - panel_2_deepseek_model_ids
        if observed_panel_2_coverage != expected_panel_2_coverage:
            raise PoweredRunRestoreError("panel-2 coverage-repair lineage differs from release")
    observed_panel_2_deepseek: set[str] = set()
    if panel_2_deepseek_plan is not None:
        observed_panel_2_deepseek = {
            model_id
            for model_id, counts in plan_by_model.items()
            if counts[f"primary:{panel_2_deepseek_plan}"] == tasks_per_panel
            and counts[f"repeat:{panel_2_deepseek_plan}"] == repeats_per_panel
        }
        if observed_panel_2_deepseek != panel_2_deepseek_model_ids:
            raise PoweredRunRestoreError("panel-2 DeepSeek-repair lineage differs from release")
    observed_panel_2_glm: set[str] = set()
    if panel_2_glm_plan is not None:
        observed_panel_2_glm = {
            model_id
            for model_id, counts in plan_by_model.items()
            if counts[f"primary:{panel_2_glm_plan}"] == tasks_per_panel
            and counts[f"repeat:{panel_2_glm_plan}"] == repeats_per_panel
        }
        if observed_panel_2_glm != panel_2_glm_model_ids:
            raise PoweredRunRestoreError("panel-2 GLM-5.3 lineage differs from release")
    for model_id, counts in plan_by_model.items():
        panel_1_plan = (
            panel_1_glm_plan
            if model_id in observed_panel_1_glm
            else panel_1_deepseek_plan
            if model_id in observed_panel_1_deepseek
            else panel_1_coverage_plan
            if model_id in observed_panel_1_coverage
            else panel_1_qwen_plan
            if model_id == qwen_model
            else panel_1_fable_plan
            if model_id == fable_model
            else panel_1_base_plan
        )
        assert panel_1_plan is not None
        panel_2_plan = (
            panel_2_glm_plan
            if model_id in observed_panel_2_glm
            else panel_2_deepseek_plan
            if model_id in observed_panel_2_deepseek
            else panel_2_coverage_plan
            if model_id in observed_panel_2_coverage
            else panel_2_replacement_plan
            if model_id in observed_panel_2_replacements
            else panel_2_base_plan
        )
        assert panel_2_plan is not None
        expected = Counter(
            {
                f"primary:{panel_1_plan}": tasks_per_panel,
                f"repeat:{panel_1_plan}": repeats_per_panel,
                f"primary:{panel_2_plan}": tasks_per_panel,
                f"repeat:{panel_2_plan}": repeats_per_panel,
            }
        )
        if counts != expected:
            raise PoweredRunRestoreError(f"joint response grid differs for {model_id}")


def restore(
    *,
    release_path: Path,
    primary_path: Path,
    repeat_path: Path,
    panel_1_run: Path,
    panel_1_qwen_run: Path,
    panel_2_run: Path,
    check: bool,
    panel_1_fable_run: Path | None = None,
    panel_1_coverage_repair_run: Path | None = None,
    panel_1_deepseek_repair_run: Path | None = None,
    panel_1_glm_run: Path | None = None,
    panel_2_luna_run: Path | None = None,
    panel_2_deepseek_flash_run: Path | None = None,
    panel_2_coverage_repair_run: Path | None = None,
    panel_2_deepseek_repair_run: Path | None = None,
    panel_2_glm_run: Path | None = None,
) -> dict[str, Any]:
    release = _load_release(release_path)
    if release.get("schema_version") != JOINT_SCHEMA:
        raise PoweredRunRestoreError("release is not the joint powered release")
    primary = _load_jsonl(primary_path)
    repeat = _load_jsonl(repeat_path)
    (
        panel_1_base_plan,
        panel_1_qwen_plan,
        panel_1_fable_plan,
        panel_1_coverage_plan,
        panel_1_deepseek_plan,
        panel_1_glm_plan,
        panel_2_base_plan,
        panel_2_replacement_plan,
        panel_2_coverage_plan,
        panel_2_deepseek_plan,
        panel_2_glm_plan,
    ) = _source_plans(release)
    lineage = release["inputs"]["response_lineage"]
    panel_1_coverage_model_ids = set(
        str(value) for value in lineage.get("panel_1_coverage_repair_model_ids", [])
    )
    panel_2_replacement_model_ids = set(
        str(value) for value in lineage.get("panel_2_replacement_model_ids", [])
    )
    panel_2_coverage_model_ids = set(
        str(value) for value in lineage.get("panel_2_coverage_repair_model_ids", [])
    )
    panel_1_deepseek_model_ids = set(
        str(value) for value in lineage.get("panel_1_deepseek_repair_model_ids", [])
    )
    panel_2_deepseek_model_ids = set(
        str(value) for value in lineage.get("panel_2_deepseek_repair_model_ids", [])
    )
    panel_1_glm_model_ids = set(
        str(value) for value in lineage.get("glm53_limited_run_model_ids", [])
    )
    panel_2_glm_model_ids = set(panel_1_glm_model_ids)
    expected_coverage_models = set(COVERAGE_REPAIR_MODEL_IDS)
    if (panel_1_coverage_plan is not None) != (
        panel_1_coverage_model_ids == expected_coverage_models
    ) or (panel_2_coverage_plan is not None) != (
        panel_2_coverage_model_ids == expected_coverage_models
    ):
        raise PoweredRunRestoreError("coverage-repair plan and model lineage differ")
    expected_deepseek_models = {"deepseek/deepseek-v4-pro-0813"}
    if (panel_1_deepseek_plan is not None) != (
        panel_1_deepseek_model_ids == expected_deepseek_models
    ) or (panel_2_deepseek_plan is not None) != (
        panel_2_deepseek_model_ids == expected_deepseek_models
    ):
        raise PoweredRunRestoreError("DeepSeek-repair plan and model lineage differ")
    expected_glm_models = {"z-ai/glm-5.3"}
    if (panel_1_glm_plan is not None) != (panel_1_glm_model_ids == expected_glm_models) or (
        panel_2_glm_plan is not None
    ) != (panel_2_glm_model_ids == expected_glm_models):
        raise PoweredRunRestoreError("GLM-5.3 plan and model lineage differ")
    panel_1_plans = {panel_1_base_plan, panel_1_qwen_plan}
    if panel_1_fable_plan is not None:
        panel_1_plans.add(panel_1_fable_plan)
    if panel_1_coverage_plan is not None:
        panel_1_plans.add(panel_1_coverage_plan)
    if panel_1_deepseek_plan is not None:
        panel_1_plans.add(panel_1_deepseek_plan)
    if panel_1_glm_plan is not None:
        panel_1_plans.add(panel_1_glm_plan)
    panel_2_plans = {panel_2_base_plan}
    if panel_2_replacement_plan is not None:
        panel_2_plans.add(panel_2_replacement_plan)
    if panel_2_coverage_plan is not None:
        panel_2_plans.add(panel_2_coverage_plan)
    if panel_2_deepseek_plan is not None:
        panel_2_plans.add(panel_2_deepseek_plan)
    if panel_2_glm_plan is not None:
        panel_2_plans.add(panel_2_glm_plan)
    for label, rows in (
        (
            "panel_1_primary",
            [row for row in primary if str(row.get("plan_sha256")) in panel_1_plans],
        ),
        (
            "panel_1_repeat",
            [row for row in repeat if str(row.get("plan_sha256")) in panel_1_plans],
        ),
        (
            "panel_2_primary",
            [row for row in primary if str(row.get("plan_sha256")) in panel_2_plans],
        ),
        (
            "panel_2_repeat",
            [row for row in repeat if str(row.get("plan_sha256")) in panel_2_plans],
        ),
    ):
        _require_response_commitment(release, label, rows)
    _validate_grid(
        release,
        primary,
        repeat,
        panel_1_base_plan=panel_1_base_plan,
        panel_1_qwen_plan=panel_1_qwen_plan,
        panel_1_fable_plan=panel_1_fable_plan,
        panel_2_base_plan=panel_2_base_plan,
        panel_2_replacement_plan=panel_2_replacement_plan,
        panel_2_replacement_model_ids=panel_2_replacement_model_ids,
        panel_1_coverage_plan=panel_1_coverage_plan,
        panel_1_coverage_model_ids=panel_1_coverage_model_ids,
        panel_1_deepseek_plan=panel_1_deepseek_plan,
        panel_1_deepseek_model_ids=panel_1_deepseek_model_ids,
        panel_1_glm_plan=panel_1_glm_plan,
        panel_1_glm_model_ids=panel_1_glm_model_ids,
        panel_2_coverage_plan=panel_2_coverage_plan,
        panel_2_coverage_model_ids=panel_2_coverage_model_ids,
        panel_2_deepseek_plan=panel_2_deepseek_plan,
        panel_2_deepseek_model_ids=panel_2_deepseek_model_ids,
        panel_2_glm_plan=panel_2_glm_plan,
        panel_2_glm_model_ids=panel_2_glm_model_ids,
    )

    outcomes = Counter()
    for row in primary + repeat:
        destination = _destination(
            row=row,
            panel_1_base_plan=panel_1_base_plan,
            panel_1_qwen_plan=panel_1_qwen_plan,
            panel_1_fable_plan=panel_1_fable_plan,
            panel_1_coverage_plan=panel_1_coverage_plan,
            panel_1_coverage_model_ids=panel_1_coverage_model_ids,
            panel_1_deepseek_plan=panel_1_deepseek_plan,
            panel_1_deepseek_model_ids=panel_1_deepseek_model_ids,
            panel_1_glm_plan=panel_1_glm_plan,
            panel_1_glm_model_ids=panel_1_glm_model_ids,
            panel_2_base_plan=panel_2_base_plan,
            panel_2_replacement_plan=panel_2_replacement_plan,
            panel_2_replacement_model_ids=panel_2_replacement_model_ids,
            panel_2_coverage_plan=panel_2_coverage_plan,
            panel_2_coverage_model_ids=panel_2_coverage_model_ids,
            panel_2_deepseek_plan=panel_2_deepseek_plan,
            panel_2_deepseek_model_ids=panel_2_deepseek_model_ids,
            panel_2_glm_plan=panel_2_glm_plan,
            panel_2_glm_model_ids=panel_2_glm_model_ids,
            panel_1_run=panel_1_run,
            panel_1_qwen_run=panel_1_qwen_run,
            panel_1_fable_run=panel_1_fable_run,
            panel_1_coverage_repair_run=panel_1_coverage_repair_run,
            panel_1_deepseek_repair_run=panel_1_deepseek_repair_run,
            panel_1_glm_run=panel_1_glm_run,
            panel_2_run=panel_2_run,
            panel_2_luna_run=panel_2_luna_run,
            panel_2_deepseek_flash_run=panel_2_deepseek_flash_run,
            panel_2_coverage_repair_run=panel_2_coverage_repair_run,
            panel_2_deepseek_repair_run=panel_2_deepseek_repair_run,
            panel_2_glm_run=panel_2_glm_run,
        )
        payload = (json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        if check:
            if destination.is_symlink() or not destination.is_file():
                raise PoweredRunRestoreError(f"restored response is missing: {destination}")
            if destination.read_bytes() != payload:
                raise PoweredRunRestoreError(f"restored response differs: {destination}")
            outcomes["verified"] += 1
        else:
            outcomes[_write_no_replace(destination, payload)] += 1
    return {
        "status": "verified" if check else "restored",
        "models": len(release["analysis"]["models"]),
        "primary_responses": len(primary),
        "repeat_responses": len(repeat),
        "files": dict(sorted(outcomes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--panel-1-run", type=Path, required=True)
    parser.add_argument("--panel-1-qwen-run", type=Path, required=True)
    parser.add_argument("--panel-1-fable-run", type=Path)
    parser.add_argument("--panel-1-coverage-repair-run", type=Path)
    parser.add_argument("--panel-1-deepseek-repair-run", type=Path)
    parser.add_argument("--panel-1-glm53-run", type=Path)
    parser.add_argument("--panel-2-run", type=Path, required=True)
    parser.add_argument("--panel-2-luna-run", type=Path)
    parser.add_argument("--panel-2-deepseek-flash-run", type=Path)
    parser.add_argument("--panel-2-coverage-repair-run", type=Path)
    parser.add_argument("--panel-2-deepseek-repair-run", type=Path)
    parser.add_argument("--panel-2-glm53-run", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            restore(
                release_path=args.release,
                primary_path=args.primary,
                repeat_path=args.repeat,
                panel_1_run=args.panel_1_run,
                panel_1_qwen_run=args.panel_1_qwen_run,
                panel_2_run=args.panel_2_run,
                panel_2_luna_run=args.panel_2_luna_run,
                panel_2_deepseek_flash_run=args.panel_2_deepseek_flash_run,
                check=args.check,
                panel_1_fable_run=args.panel_1_fable_run,
                panel_1_coverage_repair_run=args.panel_1_coverage_repair_run,
                panel_1_deepseek_repair_run=args.panel_1_deepseek_repair_run,
                panel_1_glm_run=args.panel_1_glm53_run,
                panel_2_coverage_repair_run=args.panel_2_coverage_repair_run,
                panel_2_deepseek_repair_run=args.panel_2_deepseek_repair_run,
                panel_2_glm_run=args.panel_2_glm53_run,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
