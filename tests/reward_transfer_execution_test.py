from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flavourbench.reward_transfer import (
    DEFAULT_PRIMARY_TASKS,
    RewardTransferError,
    create_evaluation_gate,
    crossed_seed_anchor_bootstrap,
    file_sha256,
    load_plan,
    matched_anchor_sign_flip,
    read_jsonl,
    semantic_sha256,
    stratified_point_estimate,
    verify_evaluation_gate,
    verify_scored_run,
)


def _fake_training_run(
    checkpoints: Path,
    *,
    condition: str,
    seed: int,
    protocol_hash: str,
    base_model: dict[str, object],
) -> None:
    directory = checkpoints / condition / f"seed-{seed}"
    directory.mkdir(parents=True)
    adapter = directory / "adapter_model.safetensors"
    adapter.write_bytes(f"{condition}:{seed}".encode())
    manifest = {
        "schema_version": "flavourbench-reward-transfer-training-run-v1",
        "status": "confirmatory_adapter_complete",
        "protocol_artifact_sha256": protocol_hash,
        "condition": condition,
        "seed": seed,
        "base_model": base_model,
        "git_commit": "1" * 40,
        "duration_seconds": 1.0,
        "smoke_max_steps": None,
        "train_metrics": {"epoch": 3.0, "train_loss": 1.0},
        "validation_metrics": {"epoch": 3.0, "eval_loss": 1.0},
        "software": {"torch": "test"},
        "hardware": {"device": "test"},
        "files": [
            {
                "path": adapter.name,
                "bytes": adapter.stat().st_size,
                "sha256": file_sha256(adapter),
            }
        ],
    }
    manifest["artifact_sha256"] = semantic_sha256(manifest)
    (directory / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_evaluation_gate_requires_and_verifies_all_six_runs(tmp_path: Path) -> None:
    plan = load_plan()
    checkpoints = tmp_path / "checkpoints"
    for condition in ("sft_format_control", "sft_epicure_optimum"):
        for seed in plan["seeds"]:
            _fake_training_run(
                checkpoints,
                condition=condition,
                seed=int(seed),
                protocol_hash=plan["artifact_sha256"],
                base_model=plan["base_model"],
            )
    output = tmp_path / "evaluation-gate.json"
    gate = create_evaluation_gate(checkpoints=checkpoints, output=output)
    assert gate["status"] == "confirmatory_evaluation_unlocked"
    assert len(gate["runs"]) == 6
    assert verify_evaluation_gate(output, checkpoints=checkpoints) == gate

    adapter = checkpoints / "sft_epicure_optimum/seed-20260826/adapter_model.safetensors"
    adapter.write_bytes(b"tampered")
    with pytest.raises(RewardTransferError, match="byte count differs|hash differs"):
        verify_evaluation_gate(output, checkpoints=checkpoints)


def test_crossed_inference_is_deterministic_and_preserves_paired_seeds() -> None:
    families = [
        family
        for family in ("substitution", "pairing", "constraint")
        for _panel in ("panel_1", "panel_2")
        for _task in range(2)
    ]
    panels = [
        panel
        for _family in ("substitution", "pairing", "constraint")
        for panel in ("panel_1", "panel_2")
        for _task in range(2)
    ]
    differences = np.full((3, 12), 4.0)
    first = crossed_seed_anchor_bootstrap(
        differences,
        families,
        panels,
        resamples=500,
        seed=17,
    )
    second = crossed_seed_anchor_bootstrap(
        differences,
        families,
        panels,
        resamples=500,
        seed=17,
    )
    assert first == second
    assert first[0] == 4.0
    assert first[1] == pytest.approx([4.0, 4.0])
    observed, p_value = matched_anchor_sign_flip(
        differences,
        families,
        panels,
        resamples=5_000,
        seed=19,
    )
    assert observed == 4.0
    assert p_value < 0.01
    assert stratified_point_estimate(differences[0], families, panels) == 4.0


def test_scored_run_recomputes_raw_completions() -> None:
    tasks = read_jsonl(DEFAULT_PRIMARY_TASKS)
    rows = []
    for task in tasks:
        selection = str(task["optimal_selection"])
        rows.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "panel": task["source_panel"],
                "completion": f"FINAL_SELECTION: {','.join(selection)}",
                "observed_selection": selection,
                "parseable": True,
                "score_bps": 10_000,
                "score": 100.0,
                "optimal": True,
            }
        )
    verify_scored_run(tasks, rows)
    rows[0]["score"] = 99.0
    with pytest.raises(RewardTransferError, match="scored row differs"):
        verify_scored_run(tasks, rows)
