from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _address(document: dict[str, object]) -> dict[str, object]:
    document["artifact_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return document


def _load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module():
    _load_module("restore_powered_runs", "hf/dataset/restore_powered_runs.py")
    return _load_module("restore_joint_powered_runs", "hf/dataset/restore_joint_powered_runs.py")


def _response(
    *, model_id: str, panel: str, task_id: str, plan_sha256: str, slot_id: str
) -> dict[str, object]:
    identity = hashlib.sha256(f"{model_id}:{panel}:{task_id}:{plan_sha256}".encode()).hexdigest()
    return _address(
        {
            "arm_id": identity,
            "cell_id": identity,
            "model_id": model_id,
            "panel": panel,
            "plan_sha256": plan_sha256,
            "slot_id": slot_id,
            "task_id": task_id,
        }
    )


def test_joint_restore_splits_base_qwen_and_second_panel(tmp_path: Path) -> None:
    module = _module()
    base_plan = "1" * 64
    qwen_plan = "2" * 64
    panel_2_plan = "3" * 64
    release = _address(
        {
            "schema_version": "flavourbench-selection-powered-joint-release-v1",
            "status": "final_complete",
            "inputs": {
                "response_lineage": {
                    "panel_1_base_plan_sha256": base_plan,
                    "panel_1_qwen_replacement_plan_sha256": qwen_plan,
                    "panel_1_superseded_qwen_responses_used": False,
                    "panel_2_plan_sha256": panel_2_plan,
                    "panel_2_reuses_panel_1_responses": False,
                }
            },
            "analysis": {
                "design": {
                    "primary_tasks_per_panel": 2,
                    "repeat_tasks_per_panel": 1,
                },
                "models": [{"model_id": "model/a"}, {"model_id": "qwen/model"}],
            },
        }
    )
    release_path = tmp_path / "release.json"
    primary = []
    repeat = []
    for model_id, first_plan, slot in (
        ("model/a", base_plan, "slot-a"),
        ("qwen/model", qwen_plan, "slot-q"),
    ):
        for index in range(2):
            primary.append(
                _response(
                    model_id=model_id,
                    panel="primary",
                    task_id=f"p1-{index}",
                    plan_sha256=first_plan,
                    slot_id=slot,
                )
            )
            primary.append(
                _response(
                    model_id=model_id,
                    panel="primary",
                    task_id=f"p2-{index}",
                    plan_sha256=panel_2_plan,
                    slot_id=slot,
                )
            )
        repeat.append(
            _response(
                model_id=model_id,
                panel="repeat",
                task_id="r1",
                plan_sha256=first_plan,
                slot_id=slot,
            )
        )
        repeat.append(
            _response(
                model_id=model_id,
                panel="repeat",
                task_id="r2",
                plan_sha256=panel_2_plan,
                slot_id=slot,
            )
        )
    panel_1_plans = {base_plan, qwen_plan}

    def commitment(rows: list[dict[str, object]]) -> dict[str, object]:
        artifacts = sorted(str(row["artifact_sha256"]) for row in rows)
        return {
            "count": len(rows),
            "artifact_set_sha256": hashlib.sha256(_canonical(artifacts)).hexdigest(),
            "spend_micros": 0,
        }

    release["inputs"].update(
        {
            "panel_1_primary": commitment(
                [row for row in primary if str(row["plan_sha256"]) in panel_1_plans]
            ),
            "panel_1_repeat": commitment(
                [row for row in repeat if str(row["plan_sha256"]) in panel_1_plans]
            ),
            "panel_2_primary": commitment(
                [row for row in primary if row["plan_sha256"] == panel_2_plan]
            ),
            "panel_2_repeat": commitment(
                [row for row in repeat if row["plan_sha256"] == panel_2_plan]
            ),
        }
    )
    release.pop("artifact_sha256")
    _address(release)
    release_path.write_text(json.dumps(release), encoding="utf-8")
    primary_path = tmp_path / "primary.jsonl"
    repeat_path = tmp_path / "repeat.jsonl"
    primary_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in primary))
    repeat_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in repeat))
    panel_1_run = tmp_path / "panel-1"
    qwen_run = tmp_path / "qwen"
    panel_2_run = tmp_path / "panel-2"

    result = module.restore(
        release_path=release_path,
        primary_path=primary_path,
        repeat_path=repeat_path,
        panel_1_run=panel_1_run,
        panel_1_qwen_run=qwen_run,
        panel_2_run=panel_2_run,
        check=False,
    )
    assert result == {
        "files": {"created": 12},
        "models": 2,
        "primary_responses": 8,
        "repeat_responses": 4,
        "status": "restored",
    }
    assert len(list(panel_1_run.rglob("response-*.json"))) == 3
    assert len(list(qwen_run.rglob("response-*.json"))) == 3
    assert len(list(panel_2_run.rglob("response-*.json"))) == 6
    checked = module.restore(
        release_path=release_path,
        primary_path=primary_path,
        repeat_path=repeat_path,
        panel_1_run=panel_1_run,
        panel_1_qwen_run=qwen_run,
        panel_2_run=panel_2_run,
        check=True,
    )
    assert checked["files"] == {"verified": 12}


def test_v67_restore_routes_finite_glm_cells_to_separate_panel_directories(
    tmp_path: Path,
) -> None:
    module = _module()
    panel_1_base = "1" * 64
    panel_1_qwen = "2" * 64
    panel_1_glm = "3" * 64
    panel_2_base = "4" * 64
    panel_2_glm = "5" * 64
    glm_model = "z-ai/glm-5.3"
    models = ["model/base", "qwen/model", glm_model]
    primary: list[dict[str, object]] = []
    repeat: list[dict[str, object]] = []
    for index, model_id in enumerate(models):
        first_plan = (
            panel_1_glm
            if model_id == glm_model
            else panel_1_qwen
            if model_id == "qwen/model"
            else panel_1_base
        )
        second_plan = panel_2_glm if model_id == glm_model else panel_2_base
        for panel, destination in (("primary", primary), ("repeat", repeat)):
            destination.extend(
                (
                    _response(
                        model_id=model_id,
                        panel=panel,
                        task_id=f"{panel}-panel-1",
                        plan_sha256=first_plan,
                        slot_id=f"slot-{index}",
                    ),
                    _response(
                        model_id=model_id,
                        panel=panel,
                        task_id=f"{panel}-panel-2",
                        plan_sha256=second_plan,
                        slot_id=f"slot-{index}",
                    ),
                )
            )

    def commitment(rows: list[dict[str, object]]) -> dict[str, object]:
        artifacts = sorted(str(row["artifact_sha256"]) for row in rows)
        return {
            "count": len(rows),
            "artifact_set_sha256": hashlib.sha256(_canonical(artifacts)).hexdigest(),
            "spend_micros": 0,
        }

    panel_1_plans = {panel_1_base, panel_1_qwen, panel_1_glm}
    panel_2_plans = {panel_2_base, panel_2_glm}
    release = _address(
        {
            "schema_version": "flavourbench-selection-powered-joint-release-v1",
            "status": "final_complete",
            "inputs": {
                "response_lineage": {
                    "panel_1_base_plan_sha256": panel_1_base,
                    "panel_1_qwen_replacement_plan_sha256": panel_1_qwen,
                    "panel_1_superseded_qwen_responses_used": False,
                    "panel_2_plan_sha256": panel_2_base,
                    "panel_2_base_plan_sha256": panel_2_base,
                    "panel_2_reuses_panel_1_responses": False,
                    "panel_2_superseded_route_responses_used": False,
                    "glm53_limited_run_model_ids": [glm_model],
                    "glm53_panel_1_plan_sha256": panel_1_glm,
                    "glm53_panel_2_plan_sha256": panel_2_glm,
                    "glm53_finite_cli_only": True,
                    "glm53_standing_service": False,
                    "glm53_automatic_fallback": False,
                },
                "panel_1_primary": commitment(
                    [row for row in primary if str(row["plan_sha256"]) in panel_1_plans]
                ),
                "panel_1_repeat": commitment(
                    [row for row in repeat if str(row["plan_sha256"]) in panel_1_plans]
                ),
                "panel_2_primary": commitment(
                    [row for row in primary if str(row["plan_sha256"]) in panel_2_plans]
                ),
                "panel_2_repeat": commitment(
                    [row for row in repeat if str(row["plan_sha256"]) in panel_2_plans]
                ),
            },
            "analysis": {
                "design": {"primary_tasks_per_panel": 1, "repeat_tasks_per_panel": 1},
                "models": [{"model_id": model_id} for model_id in models],
            },
        }
    )
    release_path = tmp_path / "release.json"
    primary_path = tmp_path / "primary.jsonl"
    repeat_path = tmp_path / "repeat.jsonl"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    primary_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in primary))
    repeat_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in repeat))
    destinations = {
        "panel_1_run": tmp_path / "panel-1-base",
        "panel_1_qwen_run": tmp_path / "panel-1-qwen",
        "panel_1_glm_run": tmp_path / "panel-1-glm",
        "panel_2_run": tmp_path / "panel-2-base",
        "panel_2_glm_run": tmp_path / "panel-2-glm",
    }
    result = module.restore(
        release_path=release_path,
        primary_path=primary_path,
        repeat_path=repeat_path,
        check=False,
        **destinations,
    )
    assert result["files"] == {"created": 12}
    assert len(list(destinations["panel_1_glm_run"].rglob("response-*.json"))) == 2
    assert len(list(destinations["panel_2_glm_run"].rglob("response-*.json"))) == 2
    checked = module.restore(
        release_path=release_path,
        primary_path=primary_path,
        repeat_path=repeat_path,
        check=True,
        **destinations,
    )
    assert checked["files"] == {"verified": 12}


def test_joint_grid_requires_one_complete_fable_replacement() -> None:
    module = _module()
    base_plan = "1" * 64
    qwen_plan = "2" * 64
    fable_plan = "3" * 64
    panel_2_plan = "4" * 64
    models = (
        ("model/a", base_plan, "slot-a"),
        ("qwen/model", qwen_plan, "slot-q"),
        ("anthropic/fable", fable_plan, "slot-f"),
    )
    release = {
        "analysis": {
            "design": {"primary_tasks_per_panel": 1, "repeat_tasks_per_panel": 1},
            "models": [{"model_id": model_id} for model_id, _, _ in models],
        }
    }
    primary = []
    repeat = []
    for model_id, first_plan, slot in models:
        primary.extend(
            (
                _response(
                    model_id=model_id,
                    panel="primary",
                    task_id="p1",
                    plan_sha256=first_plan,
                    slot_id=slot,
                ),
                _response(
                    model_id=model_id,
                    panel="primary",
                    task_id="p2",
                    plan_sha256=panel_2_plan,
                    slot_id=slot,
                ),
            )
        )
        repeat.extend(
            (
                _response(
                    model_id=model_id,
                    panel="repeat",
                    task_id="r1",
                    plan_sha256=first_plan,
                    slot_id=slot,
                ),
                _response(
                    model_id=model_id,
                    panel="repeat",
                    task_id="r2",
                    plan_sha256=panel_2_plan,
                    slot_id=slot,
                ),
            )
        )
    module._validate_grid(
        release,
        primary,
        repeat,
        panel_1_base_plan=base_plan,
        panel_1_qwen_plan=qwen_plan,
        panel_1_fable_plan=fable_plan,
        panel_2_base_plan=panel_2_plan,
        panel_2_replacement_plan=None,
        panel_2_replacement_model_ids=set(),
    )
    broken = [dict(row) for row in primary]
    next(row for row in broken if row["model_id"] == "anthropic/fable" and row["task_id"] == "p1")[
        "plan_sha256"
    ] = base_plan
    try:
        module._validate_grid(
            release,
            broken,
            repeat,
            panel_1_base_plan=base_plan,
            panel_1_qwen_plan=qwen_plan,
            panel_1_fable_plan=fable_plan,
            panel_2_base_plan=panel_2_plan,
            panel_2_replacement_plan=None,
            panel_2_replacement_model_ids=set(),
        )
    except module.PoweredRunRestoreError:
        pass
    else:
        raise AssertionError("an incomplete Fable replacement block was accepted")


def test_joint_grid_requires_both_complete_panel_2_route_replacements() -> None:
    module = _module()
    base_plan = "1" * 64
    qwen_plan = "2" * 64
    panel_2_base = "3" * 64
    panel_2_replacement = "4" * 64
    replacements = {
        "openai/gpt-5.6-luna-pro",
        "deepseek/deepseek-v4-flash-0731",
    }
    models = ("model/a", "qwen/model", *sorted(replacements))
    release = {
        "analysis": {
            "design": {"primary_tasks_per_panel": 1, "repeat_tasks_per_panel": 1},
            "models": [{"model_id": model_id} for model_id in models],
        }
    }
    primary = []
    repeat = []
    for index, model_id in enumerate(models):
        panel_1_plan = qwen_plan if model_id == "qwen/model" else base_plan
        panel_2_plan = panel_2_replacement if model_id in replacements else panel_2_base
        for panel, task_id, destination in (
            ("primary", "p1", primary),
            ("repeat", "r1", repeat),
        ):
            destination.extend(
                (
                    _response(
                        model_id=model_id,
                        panel=panel,
                        task_id=task_id,
                        plan_sha256=panel_1_plan,
                        slot_id=f"slot-{index}",
                    ),
                    _response(
                        model_id=model_id,
                        panel=panel,
                        task_id=f"{task_id}-panel-2",
                        plan_sha256=panel_2_plan,
                        slot_id=f"slot-{index}",
                    ),
                )
            )
    module._validate_grid(
        release,
        primary,
        repeat,
        panel_1_base_plan=base_plan,
        panel_1_qwen_plan=qwen_plan,
        panel_1_fable_plan=None,
        panel_2_base_plan=panel_2_base,
        panel_2_replacement_plan=panel_2_replacement,
        panel_2_replacement_model_ids=replacements,
    )
    broken = [dict(row) for row in primary]
    next(
        row
        for row in broken
        if row["model_id"] == "openai/gpt-5.6-luna-pro" and row["task_id"] == "p1-panel-2"
    )["plan_sha256"] = panel_2_base
    try:
        module._validate_grid(
            release,
            broken,
            repeat,
            panel_1_base_plan=base_plan,
            panel_1_qwen_plan=qwen_plan,
            panel_1_fable_plan=None,
            panel_2_base_plan=panel_2_base,
            panel_2_replacement_plan=panel_2_replacement,
            panel_2_replacement_model_ids=replacements,
        )
    except module.PoweredRunRestoreError:
        pass
    else:
        raise AssertionError("an incomplete panel-2 replacement block was accepted")


def test_v56_restore_routes_both_complete_coverage_repair_blocks(tmp_path: Path) -> None:
    module = _module()
    panel_1_base = "1" * 64
    panel_1_qwen = "2" * 64
    panel_1_coverage = "3" * 64
    panel_2_base = "4" * 64
    panel_2_replacement = "5" * 64
    panel_2_coverage = "6" * 64
    qwen_model = "qwen/qwen3.8-max"
    panel_2_replacements = {
        "openai/gpt-5.6-luna-pro",
        "deepseek/deepseek-v4-flash-0731",
    }
    coverage_models = set(module.COVERAGE_REPAIR_MODEL_IDS)
    models = [
        "model/base",
        qwen_model,
        *sorted(panel_2_replacements),
        *sorted(coverage_models),
    ]
    primary: list[dict[str, object]] = []
    repeat: list[dict[str, object]] = []
    for index, model_id in enumerate(models):
        first_plan = (
            panel_1_coverage
            if model_id in coverage_models
            else panel_1_qwen
            if model_id == qwen_model
            else panel_1_base
        )
        second_plan = (
            panel_2_coverage
            if model_id in coverage_models
            else panel_2_replacement
            if model_id in panel_2_replacements
            else panel_2_base
        )
        for panel, destination in (("primary", primary), ("repeat", repeat)):
            destination.extend(
                (
                    _response(
                        model_id=model_id,
                        panel=panel,
                        task_id=f"{panel}-panel-1",
                        plan_sha256=first_plan,
                        slot_id=f"slot-{index}",
                    ),
                    _response(
                        model_id=model_id,
                        panel=panel,
                        task_id=f"{panel}-panel-2",
                        plan_sha256=second_plan,
                        slot_id=f"slot-{index}",
                    ),
                )
            )

    def commitment(rows: list[dict[str, object]]) -> dict[str, object]:
        artifacts = sorted(str(row["artifact_sha256"]) for row in rows)
        return {
            "count": len(rows),
            "artifact_set_sha256": hashlib.sha256(_canonical(artifacts)).hexdigest(),
            "spend_micros": 0,
        }

    panel_1_plans = {panel_1_base, panel_1_qwen, panel_1_coverage}
    panel_2_plans = {panel_2_base, panel_2_replacement, panel_2_coverage}
    release = _address(
        {
            "schema_version": "flavourbench-selection-powered-joint-release-v1",
            "status": "final_complete",
            "inputs": {
                "response_lineage": {
                    "panel_1_base_plan_sha256": panel_1_base,
                    "panel_1_qwen_replacement_plan_sha256": panel_1_qwen,
                    "panel_1_fable_replacement_plan_sha256": None,
                    "panel_1_coverage_repair_plan_sha256": panel_1_coverage,
                    "panel_1_coverage_repair_model_ids": list(module.COVERAGE_REPAIR_MODEL_IDS),
                    "panel_1_superseded_qwen_responses_used": False,
                    "panel_1_superseded_fable_responses_used": False,
                    "panel_1_superseded_coverage_route_responses_used": False,
                    "panel_2_plan_sha256": panel_2_coverage,
                    "panel_2_base_plan_sha256": panel_2_base,
                    "panel_2_replacement_plan_sha256": panel_2_replacement,
                    "panel_2_replacement_model_ids": sorted(panel_2_replacements),
                    "panel_2_coverage_repair_plan_sha256": panel_2_coverage,
                    "panel_2_coverage_repair_model_ids": list(module.COVERAGE_REPAIR_MODEL_IDS),
                    "panel_2_reuses_panel_1_responses": False,
                    "panel_2_superseded_route_responses_used": False,
                    "panel_2_superseded_coverage_route_responses_used": False,
                },
                "panel_1_primary": commitment(
                    [row for row in primary if str(row["plan_sha256"]) in panel_1_plans]
                ),
                "panel_1_repeat": commitment(
                    [row for row in repeat if str(row["plan_sha256"]) in panel_1_plans]
                ),
                "panel_2_primary": commitment(
                    [row for row in primary if str(row["plan_sha256"]) in panel_2_plans]
                ),
                "panel_2_repeat": commitment(
                    [row for row in repeat if str(row["plan_sha256"]) in panel_2_plans]
                ),
            },
            "analysis": {
                "design": {"primary_tasks_per_panel": 1, "repeat_tasks_per_panel": 1},
                "models": [{"model_id": model_id} for model_id in models],
            },
        }
    )
    release_path = tmp_path / "release.json"
    primary_path = tmp_path / "primary.jsonl"
    repeat_path = tmp_path / "repeat.jsonl"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    primary_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in primary))
    repeat_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in repeat))
    destinations = {
        "panel_1_run": tmp_path / "panel-1-base",
        "panel_1_qwen_run": tmp_path / "panel-1-qwen",
        "panel_1_coverage_repair_run": tmp_path / "panel-1-coverage",
        "panel_2_run": tmp_path / "panel-2-base",
        "panel_2_luna_run": tmp_path / "panel-2-luna",
        "panel_2_deepseek_flash_run": tmp_path / "panel-2-deepseek",
        "panel_2_coverage_repair_run": tmp_path / "panel-2-coverage",
    }
    result = module.restore(
        release_path=release_path,
        primary_path=primary_path,
        repeat_path=repeat_path,
        check=False,
        **destinations,
    )
    assert result["files"] == {"created": len(models) * 4}
    assert len(list(destinations["panel_1_coverage_repair_run"].rglob("response-*.json"))) == (
        len(coverage_models) * 2
    )
    assert len(list(destinations["panel_2_coverage_repair_run"].rglob("response-*.json"))) == (
        len(coverage_models) * 2
    )
    assert len(list(destinations["panel_2_luna_run"].rglob("response-*.json"))) == 2
    assert len(list(destinations["panel_2_deepseek_flash_run"].rglob("response-*.json"))) == 2


def test_v60_restore_routes_deepseek_after_coverage_repairs(tmp_path: Path) -> None:
    module = _module()
    panel_1_base = "1" * 64
    panel_1_qwen = "2" * 64
    panel_1_coverage = "3" * 64
    panel_1_deepseek = "4" * 64
    panel_2_base = "5" * 64
    panel_2_replacement = "6" * 64
    panel_2_coverage = "7" * 64
    panel_2_deepseek = "8" * 64
    deepseek_model = "deepseek/deepseek-v4-pro-0813"
    qwen_model = "qwen/qwen3.8-max"
    panel_2_replacements = {
        "openai/gpt-5.6-luna-pro",
        "deepseek/deepseek-v4-flash-0731",
    }
    coverage_models = set(module.COVERAGE_REPAIR_MODEL_IDS)
    models = list(
        dict.fromkeys(
            [
                "model/base",
                qwen_model,
                *sorted(panel_2_replacements),
                *sorted(coverage_models),
            ]
        )
    )
    primary: list[dict[str, object]] = []
    repeat: list[dict[str, object]] = []
    for index, model_id in enumerate(models):
        first_plan = (
            panel_1_deepseek
            if model_id == deepseek_model
            else panel_1_coverage
            if model_id in coverage_models
            else panel_1_qwen
            if model_id == qwen_model
            else panel_1_base
        )
        second_plan = (
            panel_2_deepseek
            if model_id == deepseek_model
            else panel_2_coverage
            if model_id in coverage_models
            else panel_2_replacement
            if model_id in panel_2_replacements
            else panel_2_base
        )
        for panel, destination in (("primary", primary), ("repeat", repeat)):
            destination.extend(
                (
                    _response(
                        model_id=model_id,
                        panel=panel,
                        task_id=f"{panel}-panel-1",
                        plan_sha256=first_plan,
                        slot_id=f"slot-{index}",
                    ),
                    _response(
                        model_id=model_id,
                        panel=panel,
                        task_id=f"{panel}-panel-2",
                        plan_sha256=second_plan,
                        slot_id=f"slot-{index}",
                    ),
                )
            )

    def commitment(rows: list[dict[str, object]]) -> dict[str, object]:
        artifacts = sorted(str(row["artifact_sha256"]) for row in rows)
        return {
            "count": len(rows),
            "artifact_set_sha256": hashlib.sha256(_canonical(artifacts)).hexdigest(),
            "spend_micros": 0,
        }

    panel_1_plans = {panel_1_base, panel_1_qwen, panel_1_coverage, panel_1_deepseek}
    panel_2_plans = {
        panel_2_base,
        panel_2_replacement,
        panel_2_coverage,
        panel_2_deepseek,
    }
    release = _address(
        {
            "schema_version": "flavourbench-selection-powered-joint-release-v1",
            "status": "final_complete",
            "inputs": {
                "response_lineage": {
                    "panel_1_base_plan_sha256": panel_1_base,
                    "panel_1_qwen_replacement_plan_sha256": panel_1_qwen,
                    "panel_1_fable_replacement_plan_sha256": None,
                    "panel_1_coverage_repair_plan_sha256": panel_1_coverage,
                    "panel_1_coverage_repair_model_ids": list(module.COVERAGE_REPAIR_MODEL_IDS),
                    "panel_1_deepseek_repair_plan_sha256": panel_1_deepseek,
                    "panel_1_deepseek_repair_model_ids": [deepseek_model],
                    "panel_1_superseded_qwen_responses_used": False,
                    "panel_1_superseded_fable_responses_used": False,
                    "panel_1_superseded_coverage_route_responses_used": False,
                    "panel_1_superseded_deepseek_route_responses_used": False,
                    "panel_2_plan_sha256": panel_2_deepseek,
                    "panel_2_base_plan_sha256": panel_2_base,
                    "panel_2_replacement_plan_sha256": panel_2_replacement,
                    "panel_2_replacement_model_ids": sorted(panel_2_replacements),
                    "panel_2_coverage_repair_plan_sha256": panel_2_coverage,
                    "panel_2_coverage_repair_model_ids": list(module.COVERAGE_REPAIR_MODEL_IDS),
                    "panel_2_deepseek_repair_plan_sha256": panel_2_deepseek,
                    "panel_2_deepseek_repair_model_ids": [deepseek_model],
                    "panel_2_reuses_panel_1_responses": False,
                    "panel_2_superseded_route_responses_used": False,
                    "panel_2_superseded_coverage_route_responses_used": False,
                    "panel_2_superseded_deepseek_route_responses_used": False,
                },
                "panel_1_primary": commitment(
                    [row for row in primary if str(row["plan_sha256"]) in panel_1_plans]
                ),
                "panel_1_repeat": commitment(
                    [row for row in repeat if str(row["plan_sha256"]) in panel_1_plans]
                ),
                "panel_2_primary": commitment(
                    [row for row in primary if str(row["plan_sha256"]) in panel_2_plans]
                ),
                "panel_2_repeat": commitment(
                    [row for row in repeat if str(row["plan_sha256"]) in panel_2_plans]
                ),
            },
            "analysis": {
                "design": {"primary_tasks_per_panel": 1, "repeat_tasks_per_panel": 1},
                "models": [{"model_id": model_id} for model_id in models],
            },
        }
    )
    release_path = tmp_path / "release.json"
    primary_path = tmp_path / "primary.jsonl"
    repeat_path = tmp_path / "repeat.jsonl"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    primary_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in primary))
    repeat_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in repeat))
    destinations = {
        "panel_1_run": tmp_path / "panel-1-base",
        "panel_1_qwen_run": tmp_path / "panel-1-qwen",
        "panel_1_coverage_repair_run": tmp_path / "panel-1-coverage",
        "panel_1_deepseek_repair_run": tmp_path / "panel-1-deepseek",
        "panel_2_run": tmp_path / "panel-2-base",
        "panel_2_luna_run": tmp_path / "panel-2-luna",
        "panel_2_deepseek_flash_run": tmp_path / "panel-2-deepseek-flash",
        "panel_2_coverage_repair_run": tmp_path / "panel-2-coverage",
        "panel_2_deepseek_repair_run": tmp_path / "panel-2-deepseek-pro",
    }
    result = module.restore(
        release_path=release_path,
        primary_path=primary_path,
        repeat_path=repeat_path,
        check=False,
        **destinations,
    )
    assert result["files"] == {"created": len(models) * 4}
    assert len(list(destinations["panel_1_deepseek_repair_run"].rglob("response-*.json"))) == 2
    assert len(list(destinations["panel_2_deepseek_repair_run"].rglob("response-*.json"))) == 2
    assert (
        len(list(destinations["panel_1_coverage_repair_run"].rglob("response-*.json")))
        == (len(coverage_models) - 1) * 2
    )
    assert (
        len(list(destinations["panel_2_coverage_repair_run"].rglob("response-*.json")))
        == (len(coverage_models) - 1) * 2
    )
