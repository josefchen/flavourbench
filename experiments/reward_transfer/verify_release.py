#!/usr/bin/env python3
"""Verify and reconstruct the released Epicure reward-transfer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from flavourbench.reward_transfer import (
    TRAINED_CONDITIONS,
    RewardTransferError,
    canonical_bytes,
    crossed_seed_anchor_bootstrap,
    evaluation_rng_seed,
    file_sha256,
    matched_anchor_sign_flip,
    read_jsonl,
    summarize_run,
    verify_content_addressed,
    verify_scored_run,
)

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPOSITORY / "hf/dataset/data-analysis"
DEFAULT_LAB_DATA = REPOSITORY / "hf/dataset/data-lab"
DEFAULT_PUBLIC_TASKS = REPOSITORY / "hf/dataset/data-complete-core/tasks.jsonl"
DEFAULT_PLAN = REPOSITORY / "contracts/reward-transfer/reward-transfer-plan-v2.json"
EXPECTED_FILES = {
    "reward-transfer-evaluation-gate.json",
    "reward-transfer-primary-analysis.json",
    "reward-transfer-public-analysis.json",
    "reward-transfer-training-manifests.jsonl",
    "reward-transfer-evaluation-manifests.jsonl",
    "reward-transfer-primary-responses.jsonl",
    "reward-transfer-public-responses.jsonl",
}


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RewardTransferError(f"release file is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RewardTransferError(f"release JSON is not an object: {path}")
    return value


def _close(actual: float, expected: object, *, label: str) -> None:
    if not np.isclose(actual, float(expected), rtol=0, atol=1e-12):
        raise RewardTransferError(f"reconstructed value differs: {label}")


def _condition(analysis: Mapping[str, Any], condition: str) -> Mapping[str, Any]:
    rows = analysis.get("condition_summaries")
    if not isinstance(rows, list):
        raise RewardTransferError("released condition summaries are absent")
    matches = [
        row for row in rows if isinstance(row, Mapping) and row.get("condition") == condition
    ]
    if len(matches) != 1:
        raise RewardTransferError(f"released condition summary differs: {condition}")
    return matches[0]


def _canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _verify_file_manifest(dataset: Path, release: Mapping[str, Any]) -> None:
    records = release.get("files")
    if not isinstance(records, list):
        raise RewardTransferError("release file manifest is absent")
    names = [str(record.get("name")) for record in records if isinstance(record, Mapping)]
    if len(names) != len(records) or len(names) != len(set(names)) or set(names) != EXPECTED_FILES:
        raise RewardTransferError("release file roster differs")
    for record in records:
        path = dataset / str(record["name"])
        if path.is_symlink() or not path.is_file():
            raise RewardTransferError(f"released member is unavailable: {path}")
        payload = path.read_bytes()
        if len(payload) != int(record["bytes"]):
            raise RewardTransferError(f"released byte count differs: {path.name}")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise RewardTransferError(f"released file hash differs: {path.name}")
        if path.suffix == ".jsonl" and payload.count(b"\n") != int(record["rows"]):
            raise RewardTransferError(f"released row count differs: {path.name}")


def _verify_training_manifests(
    dataset: Path, plan: Mapping[str, Any]
) -> dict[tuple[str, int], str]:
    wrappers = read_jsonl(dataset / "reward-transfer-training-manifests.jsonl")
    expected = {
        (condition, int(seed)) for condition in TRAINED_CONDITIONS for seed in plan["seeds"]
    }
    observed: dict[tuple[str, int], str] = {}
    for wrapper in wrappers:
        manifest = wrapper.get("manifest")
        if not isinstance(manifest, dict):
            raise RewardTransferError("training manifest wrapper differs")
        verify_content_addressed(manifest, label="released training manifest")
        identity = (str(manifest.get("condition")), int(manifest.get("seed")))
        if identity in observed:
            raise RewardTransferError(f"duplicate released training manifest: {identity}")
        if (
            manifest.get("schema_version") != "flavourbench-reward-transfer-training-run-v1"
            or manifest.get("status") != "confirmatory_adapter_complete"
            or manifest.get("protocol_artifact_sha256") != plan["artifact_sha256"]
            or manifest.get("base_model") != plan["base_model"]
        ):
            raise RewardTransferError(f"released training manifest contract differs: {identity}")
        observed[identity] = str(manifest["artifact_sha256"])
    if set(observed) != expected:
        raise RewardTransferError("released training-run roster differs")
    return observed


def _evaluation_manifests(
    dataset: Path, release: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, int | None], dict[str, Any]]]:
    wrappers = read_jsonl(dataset / "reward-transfer-evaluation-manifests.jsonl")
    masters: dict[str, dict[str, Any]] = {}
    runs: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    for wrapper in wrappers:
        split = str(wrapper.get("evaluation_split"))
        manifest_type = wrapper.get("manifest_type")
        manifest = wrapper.get("manifest")
        if split not in {"primary", "public"} or not isinstance(manifest, dict):
            raise RewardTransferError("evaluation manifest wrapper differs")
        verify_content_addressed(manifest, label=f"released {split} evaluation manifest")
        if manifest.get("protocol_artifact_sha256") != plan["artifact_sha256"]:
            raise RewardTransferError(f"released {split} protocol binding differs")
        if manifest_type == "master":
            if split in masters:
                raise RewardTransferError(f"duplicate released evaluation master: {split}")
            masters[split] = manifest
        elif manifest_type == "run":
            seed_value = manifest.get("training_seed")
            seed = int(seed_value) if seed_value is not None else None
            identity = (split, str(manifest.get("condition")), seed)
            if identity in runs:
                raise RewardTransferError(f"duplicate released evaluation run: {identity}")
            runs[identity] = manifest
        else:
            raise RewardTransferError("evaluation manifest type differs")
    if set(masters) != {"primary", "public"} or len(runs) != 14:
        raise RewardTransferError("released evaluation manifest roster differs")
    for split, field in (
        ("primary", "primary_evaluation_artifact_sha256"),
        ("public", "public_evaluation_artifact_sha256"),
    ):
        master = masters[split]
        if (
            master.get("artifact_sha256") != release[field]
            or master.get("schema_version") != "flavourbench-reward-transfer-evaluation-manifest-v1"
            or master.get("status") != "complete"
            or master.get("split") != split
        ):
            raise RewardTransferError(f"released evaluation master differs: {split}")
        declared = {
            (
                split,
                str(record.get("condition")),
                int(record["training_seed"]) if record.get("training_seed") is not None else None,
            ): str(record.get("run_artifact_sha256"))
            for record in master.get("runs", [])
            if isinstance(record, Mapping)
        }
        actual = {
            identity: str(manifest["artifact_sha256"])
            for identity, manifest in runs.items()
            if identity[0] == split
        }
        if declared != actual:
            raise RewardTransferError(f"released evaluation run bindings differ: {split}")
    return masters, runs


def _reconstruct_split(
    *,
    split: str,
    dataset: Path,
    task_path: Path,
    tasks: list[dict[str, Any]],
    analysis: Mapping[str, Any],
    master: Mapping[str, Any],
    run_manifests: Mapping[tuple[str, str, int | None], Mapping[str, Any]],
    training_hashes: Mapping[tuple[str, int], str],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if file_sha256(task_path) != master["task_file_sha256"]:
        raise RewardTransferError(f"released task hash differs: {split}")
    responses = read_jsonl(dataset / f"reward-transfer-{split}-responses.jsonl")
    expected_identities = [("pretrained_base", None)] + [
        (condition, int(seed)) for condition in TRAINED_CONDITIONS for seed in plan["seeds"]
    ]
    if len(responses) != len(tasks) * len(expected_identities) or any(
        row.get("evaluation_split") != split for row in responses
    ):
        raise RewardTransferError(f"released response roster differs: {split}")
    rows_by_run: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for condition, seed in expected_identities:
        rows = [
            {key: value for key, value in row.items() if key != "evaluation_split"}
            for row in responses
            if row.get("evaluation_split") == split
            and row.get("condition") == condition
            and row.get("training_seed") == seed
        ]
        verify_scored_run(tasks, rows)
        manifest = run_manifests[(split, condition, seed)]
        if len(rows) != int(manifest["rows"]):
            raise RewardTransferError(f"released evaluated row count differs: {split}/{condition}")
        payload = _canonical_jsonl(rows)
        if (
            len(payload) != int(manifest["output"]["bytes"])
            or hashlib.sha256(payload).hexdigest() != manifest["output"]["sha256"]
        ):
            raise RewardTransferError(f"released run output differs: {split}/{condition}/{seed}")
        expected_adapter = None if seed is None else training_hashes[(condition, seed)]
        if manifest.get("adapter_run_artifact_sha256") != expected_adapter:
            raise RewardTransferError(
                f"released adapter binding differs: {split}/{condition}/{seed}"
            )
        rows_by_run[(condition, seed)] = rows

    task_ids = [str(task["task_id"]) for task in tasks]
    seeds = [int(seed) for seed in plan["seeds"]]
    families = [str(task["family"]) for task in tasks]
    panels = [str(task.get("source_panel") or task.get("release_panel")) for task in tasks]
    matrices: dict[str, np.ndarray] = {}
    for condition in TRAINED_CONDITIONS:
        matrix = []
        for seed in seeds:
            by_id = {str(row["task_id"]): row for row in rows_by_run[(condition, seed)]}
            matrix.append([float(by_id[task_id]["score"]) for task_id in task_ids])
            summary = summarize_run(rows_by_run[(condition, seed)])
            expected_runs = _condition(analysis, condition)["runs"]
            expected_run = next(row for row in expected_runs if row["training_seed"] == seed)
            if {"training_seed": seed, **summary} != expected_run:
                raise RewardTransferError(
                    f"released condition summary differs: {split}/{condition}/{seed}"
                )
        matrices[condition] = np.asarray(matrix, dtype=np.float64)

    base_summary = summarize_run(rows_by_run[("pretrained_base", None)])
    if {"training_seed": None, **base_summary} != _condition(analysis, "pretrained_base")["runs"][
        0
    ]:
        raise RewardTransferError(f"released base summary differs: {split}")

    differences = matrices["sft_epicure_optimum"] - matrices["sft_format_control"]
    contrast_key = "confirmatory_contrast" if split == "primary" else "replication_contrast"
    contrast = analysis[contrast_key]
    label = str(contrast["label"])
    expected_bootstrap_seed = evaluation_rng_seed(
        str(plan["artifact_sha256"]), f"{label}:crossed-bootstrap"
    )
    expected_sign_flip_seed = evaluation_rng_seed(
        str(plan["artifact_sha256"]), f"{label}:anchor-sign-flip"
    )
    if (
        contrast["bootstrap"]["rng_seed"] != expected_bootstrap_seed
        or contrast["sign_flip"]["rng_seed"] != expected_sign_flip_seed
        or contrast["bootstrap"]["resamples"] != 50_000
        or contrast["sign_flip"]["resamples"] != 100_000
    ):
        raise RewardTransferError(f"released inference contract differs: {split}")
    point, interval = crossed_seed_anchor_bootstrap(
        differences,
        families,
        panels,
        resamples=50_000,
        seed=expected_bootstrap_seed,
    )
    tested_point, p_value = matched_anchor_sign_flip(
        differences,
        families,
        panels,
        resamples=100_000,
        seed=expected_sign_flip_seed,
    )
    _close(point, contrast["estimate_points"], label=f"{split} effect")
    _close(tested_point, contrast["estimate_points"], label=f"{split} tested effect")
    _close(interval[0], contrast["confidence_interval_95"][0], label=f"{split} CI low")
    _close(interval[1], contrast["confidence_interval_95"][1], label=f"{split} CI high")
    _close(p_value, contrast["two_sided_sign_flip_p"], label=f"{split} sign-flip p")
    seed_effects = [
        mean(row.tolist())
        for row in matrices["sft_epicure_optimum"] - matrices["sft_format_control"]
    ]
    for index, value in enumerate(seed_effects):
        _close(
            value,
            contrast["training_seed_estimates_points"][index],
            label=f"{split} seed effect {seeds[index]}",
        )
    return {
        "effect_points": point,
        "interval_95": interval,
        "p_value": p_value,
        "responses": len(responses),
        "tasks": len(tasks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-directory", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lab-data", type=Path, default=DEFAULT_LAB_DATA)
    parser.add_argument("--public-tasks", type=Path, default=DEFAULT_PUBLIC_TASKS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()

    plan = _load_json(args.plan)
    verify_content_addressed(plan, label="reward-transfer plan")
    release = _load_json(args.dataset_directory / "reward-transfer-release-manifest.json")
    verify_content_addressed(release, label="reward-transfer release")
    if (
        release.get("schema_version") != "flavourbench-reward-transfer-release-v1"
        or release.get("status") != "complete"
        or release.get("protocol_artifact_sha256") != plan["artifact_sha256"]
    ):
        raise RewardTransferError("reward-transfer release contract differs")
    _verify_file_manifest(args.dataset_directory, release)
    gate = _load_json(args.dataset_directory / "reward-transfer-evaluation-gate.json")
    verify_content_addressed(gate, label="released evaluation gate")
    if gate.get("artifact_sha256") != release["evaluation_gate_artifact_sha256"]:
        raise RewardTransferError("released evaluation gate binding differs")
    training_hashes = _verify_training_manifests(args.dataset_directory, plan)
    masters, run_manifests = _evaluation_manifests(args.dataset_directory, release, plan)
    primary = _load_json(args.dataset_directory / "reward-transfer-primary-analysis.json")
    public = _load_json(args.dataset_directory / "reward-transfer-public-analysis.json")
    for split, analysis, field in (
        ("primary", primary, "primary_analysis_artifact_sha256"),
        ("public", public, "public_analysis_artifact_sha256"),
    ):
        verify_content_addressed(analysis, label=f"released {split} analysis")
        if analysis.get("artifact_sha256") != release[field]:
            raise RewardTransferError(f"released analysis binding differs: {split}")
    if (
        public.get("primary_analysis_artifact_sha256") != primary["artifact_sha256"]
        or masters["public"].get("primary_analysis_artifact_sha256") != primary["artifact_sha256"]
    ):
        raise RewardTransferError("public replication does not bind the primary analysis")

    primary_result = _reconstruct_split(
        split="primary",
        dataset=args.dataset_directory,
        task_path=args.lab_data / "evaluation_tasks.jsonl",
        tasks=read_jsonl(args.lab_data / "evaluation_tasks.jsonl"),
        analysis=primary,
        master=masters["primary"],
        run_manifests=run_manifests,
        training_hashes=training_hashes,
        plan=plan,
    )
    public_result = _reconstruct_split(
        split="public",
        dataset=args.dataset_directory,
        task_path=args.public_tasks,
        tasks=read_jsonl(args.public_tasks),
        analysis=public,
        master=masters["public"],
        run_manifests=run_manifests,
        training_hashes=training_hashes,
        plan=plan,
    )
    print(
        json.dumps(
            {
                "primary": primary_result,
                "public": public_result,
                "status": "PASS",
                "training_runs": len(training_hashes),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
