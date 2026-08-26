"""Fail-closed execution and inference helpers for the Epicure transfer study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .lab import PRIMARY_FAMILIES, validate_tasks
from .selection_response_parser_v3 import score_answer_v3

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = REPOSITORY / "contracts/reward-transfer/reward-transfer-plan-v2.json"
DEFAULT_AUDIT = REPOSITORY / "experiments/reward_transfer/data-audit.json"
DEFAULT_CHECKPOINTS = REPOSITORY / "experiments/reward_transfer/checkpoints"
DEFAULT_RESULTS = REPOSITORY / "experiments/reward_transfer/results"
DEFAULT_PRIMARY_TASKS = REPOSITORY / "hf/dataset/data-lab/evaluation_tasks.jsonl"
DEFAULT_PUBLIC_TASKS = REPOSITORY / "hf/dataset/data-complete-core/tasks.jsonl"
TRAINED_CONDITIONS = ("sft_format_control", "sft_epicure_optimum")
EXPECTED_STRATA = tuple(
    (family, panel)
    for family in PRIMARY_FAMILIES
    for panel in ("panel_1", "panel_2")
)


class RewardTransferError(RuntimeError):
    """A frozen study artifact or execution invariant differs from its contract."""


def canonical_bytes(value: object) -> bytes:
    """Serialize JSON deterministically for semantic content addressing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RewardTransferError(f"not a regular JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RewardTransferError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise RewardTransferError(f"non-object JSON row at {path}:{line_number}")
        rows.append(row)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write one human-readable JSON document."""

    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(path, b"".join(canonical_bytes(dict(row)) + b"\n" for row in rows))


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise RewardTransferError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_content_addressed(document: Mapping[str, Any], *, label: str) -> str:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    if not recorded or recorded != semantic_sha256(payload):
        raise RewardTransferError(f"{label} semantic hash differs")
    return recorded


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    verify_content_addressed(plan, label="reward-transfer protocol")
    if plan.get("schema_version") != "flavourbench-reward-transfer-plan-v2":
        raise RewardTransferError("reward-transfer protocol schema differs")
    if plan.get("status") != "prospective_protocol_frozen_before_any_transfer_outcome":
        raise RewardTransferError("reward-transfer protocol is not prospective")
    if tuple(plan.get("seeds", ())) != (20260824, 20260825, 20260826):
        raise RewardTransferError("frozen training seeds differ")
    return plan


def _regular_manifest_member(root: Path, relative_text: str) -> Path:
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RewardTransferError(f"unsafe manifest path: {relative_text}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise RewardTransferError(f"manifest member is unavailable or unsafe: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise RewardTransferError(f"manifest member escapes its root: {relative_text}")
    return path


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_all_finite(item) for item in value)
    return False


def verify_training_run(
    directory: Path,
    *,
    plan: Mapping[str, Any],
    condition: str,
    seed: int,
) -> dict[str, Any]:
    """Verify one final adapter and its content-addressed training manifest."""

    manifest_path = directory / "run-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RewardTransferError(f"training run is incomplete: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_content_addressed(manifest, label=f"training run {condition}/{seed}")
    expected = {
        "schema_version": "flavourbench-reward-transfer-training-run-v1",
        "status": "confirmatory_adapter_complete",
        "protocol_artifact_sha256": plan["artifact_sha256"],
        "condition": condition,
        "seed": seed,
        "smoke_max_steps": None,
        "base_model": plan["base_model"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RewardTransferError(f"training run {condition}/{seed} differs at {key}")
    if not _all_finite(manifest):
        raise RewardTransferError(f"training run {condition}/{seed} contains non-finite data")
    if float(manifest.get("duration_seconds", 0.0)) <= 0:
        raise RewardTransferError(f"training run {condition}/{seed} has no positive duration")
    for metric_group in ("train_metrics", "validation_metrics"):
        metrics = manifest.get(metric_group)
        if not isinstance(metrics, Mapping) or float(metrics.get("epoch", -1.0)) != 3.0:
            raise RewardTransferError(
                f"training run {condition}/{seed} did not complete three epochs"
            )

    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RewardTransferError(f"training run {condition}/{seed} has no file manifest")
    recorded_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise RewardTransferError(f"training run {condition}/{seed} has a malformed file row")
        relative = str(record.get("path", ""))
        if not relative or relative in recorded_paths:
            raise RewardTransferError(f"training run {condition}/{seed} repeats a file path")
        recorded_paths.add(relative)
        path = _regular_manifest_member(directory, relative)
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise RewardTransferError(f"training file byte count differs: {path}")
        if file_sha256(path) != str(record.get("sha256", "")):
            raise RewardTransferError(f"training file hash differs: {path}")
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "run-manifest.json"
    }
    if actual_paths != recorded_paths:
        raise RewardTransferError(f"training run {condition}/{seed} has unbound files")
    if "adapter_model.safetensors" not in recorded_paths:
        raise RewardTransferError(f"training run {condition}/{seed} has no final adapter")
    return manifest


def create_evaluation_gate(
    *,
    checkpoints: Path = DEFAULT_CHECKPOINTS,
    output: Path = DEFAULT_RESULTS / "evaluation-gate.json",
    plan_path: Path = DEFAULT_PLAN,
    audit_path: Path = DEFAULT_AUDIT,
    primary_tasks_path: Path = DEFAULT_PRIMARY_TASKS,
    public_tasks_path: Path = DEFAULT_PUBLIC_TASKS,
) -> dict[str, Any]:
    """Unlock confirmatory evaluation only after all six frozen adapters verify."""

    plan = load_plan(plan_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verify_content_addressed(audit, label="reward-transfer data audit")
    if audit.get("status") != "pass":
        raise RewardTransferError("reward-transfer data audit did not pass")
    if audit.get("lab_dataset_artifact_sha256") != plan["dataset"][
        "lab_dataset_artifact_sha256"
    ]:
        raise RewardTransferError("data audit does not bind the frozen lab dataset")
    expected_task_hashes = {
        "primary": plan["dataset"]["evaluation_tasks_sha256"],
        "public": plan["dataset"]["official_replication_tasks_sha256"],
    }
    actual_task_hashes = {
        "primary": file_sha256(primary_tasks_path),
        "public": file_sha256(public_tasks_path),
    }
    if actual_task_hashes != expected_task_hashes:
        raise RewardTransferError("evaluation task bytes differ from the frozen protocol")

    runs: list[dict[str, Any]] = []
    git_commits: set[str] = set()
    software: set[bytes] = set()
    hardware: set[bytes] = set()
    for condition in TRAINED_CONDITIONS:
        for seed in plan["seeds"]:
            directory = checkpoints / condition / f"seed-{seed}"
            manifest = verify_training_run(
                directory,
                plan=plan,
                condition=condition,
                seed=int(seed),
            )
            git_commit = str(manifest.get("git_commit") or "")
            if not git_commit:
                raise RewardTransferError(f"training run {condition}/{seed} has no git commit")
            git_commits.add(git_commit)
            software.add(canonical_bytes(manifest["software"]))
            hardware.add(canonical_bytes(manifest["hardware"]))
            runs.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "run_artifact_sha256": manifest["artifact_sha256"],
                    "git_commit": git_commit,
                    "duration_seconds": manifest["duration_seconds"],
                    "train_loss": manifest["train_metrics"]["train_loss"],
                    "validation_loss": manifest["validation_metrics"]["eval_loss"],
                }
            )
    if len(git_commits) != 1 or len(software) != 1 or len(hardware) != 1:
        raise RewardTransferError(
            "training runs do not share one code, software, and hardware state"
        )
    gate: dict[str, Any] = {
        "schema_version": "flavourbench-reward-transfer-evaluation-gate-v1",
        "status": "confirmatory_evaluation_unlocked",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_artifact_sha256": plan["artifact_sha256"],
        "data_audit_artifact_sha256": audit["artifact_sha256"],
        "task_file_sha256": actual_task_hashes,
        "training_git_commit": next(iter(git_commits)),
        "runs": runs,
        "outcome_access_statement": (
            "All six final adapters verified before any held-out model completion was generated."
        ),
    }
    gate["artifact_sha256"] = semantic_sha256(gate)
    write_json(output, gate)
    return gate


def verify_evaluation_gate(
    path: Path,
    *,
    checkpoints: Path = DEFAULT_CHECKPOINTS,
    plan_path: Path = DEFAULT_PLAN,
) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    verify_content_addressed(gate, label="reward-transfer evaluation gate")
    if gate.get("schema_version") != "flavourbench-reward-transfer-evaluation-gate-v1":
        raise RewardTransferError("evaluation gate schema differs")
    if gate.get("status") != "confirmatory_evaluation_unlocked":
        raise RewardTransferError("confirmatory evaluation is not unlocked")
    plan = load_plan(plan_path)
    if gate.get("protocol_artifact_sha256") != plan["artifact_sha256"]:
        raise RewardTransferError("evaluation gate binds a different protocol")
    expected = {
        (condition, int(seed))
        for condition in TRAINED_CONDITIONS
        for seed in plan["seeds"]
    }
    records = gate.get("runs")
    if not isinstance(records, list) or {
        (str(row.get("condition")), int(row.get("seed", -1)))
        for row in records
        if isinstance(row, Mapping)
    } != expected:
        raise RewardTransferError("evaluation gate does not contain exactly six runs")
    for row in records:
        condition = str(row["condition"])
        seed = int(row["seed"])
        manifest = verify_training_run(
            checkpoints / condition / f"seed-{seed}",
            plan=plan,
            condition=condition,
            seed=seed,
        )
        if row.get("run_artifact_sha256") != manifest["artifact_sha256"]:
            raise RewardTransferError(f"evaluation gate run hash differs: {condition}/{seed}")
    return gate


def evaluation_rng_seed(protocol_hash: str, label: str) -> int:
    """Derive an outcome-independent RNG seed from the frozen protocol."""

    digest = hashlib.sha256(f"{protocol_hash}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _validate_contrast_arrays(
    differences: np.ndarray,
    families: Sequence[str],
    panels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(differences, dtype=np.float64)
    family_array = np.asarray(families, dtype=object)
    panel_array = np.asarray(panels, dtype=object)
    if array.ndim != 2 or array.shape[1] != len(family_array) or len(panel_array) != len(
        family_array
    ):
        raise RewardTransferError("contrast arrays have incompatible shapes")
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise RewardTransferError("contrast array is empty or non-finite")
    counts = Counter(zip(family_array.tolist(), panel_array.tolist(), strict=True))
    if set(counts) != set(EXPECTED_STRATA) or len(set(counts.values())) != 1:
        raise RewardTransferError("contrast tasks are not balanced over six frozen strata")
    return array, family_array, panel_array


def stratified_point_estimate(
    values: Sequence[float], families: Sequence[str], panels: Sequence[str]
) -> float:
    array, family_array, panel_array = _validate_contrast_arrays(
        np.asarray(values, dtype=np.float64)[None, :], families, panels
    )
    return float(
        np.mean(
            [
                array[0, (family_array == family) & (panel_array == panel)].mean()
                for family, panel in EXPECTED_STRATA
            ]
        )
    )


def crossed_seed_anchor_bootstrap(
    differences: np.ndarray,
    families: Sequence[str],
    panels: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Resample matched training seeds and anchors within the six frozen strata."""

    if resamples <= 0:
        raise RewardTransferError("bootstrap resamples must be positive")
    array, family_array, panel_array = _validate_contrast_arrays(
        differences, families, panels
    )
    point = stratified_point_estimate(array.mean(axis=0), families, panels)
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        width = min(1000, resamples - start)
        seed_draws = rng.integers(0, array.shape[0], size=(width, array.shape[0]))
        chunk = np.zeros(width, dtype=np.float64)
        for family, panel in EXPECTED_STRATA:
            indices = np.flatnonzero((family_array == family) & (panel_array == panel))
            task_draws = rng.integers(0, len(indices), size=(width, len(indices)))
            matrix = array[:, indices]
            sampled = matrix[
                seed_draws[:, :, None],
                task_draws[:, None, :],
            ]
            chunk += sampled.mean(axis=(1, 2)) / len(EXPECTED_STRATA)
        draws[start : start + width] = chunk
    low, high = np.quantile(draws, (0.025, 0.975), method="linear")
    return point, [float(low), float(high)]


def matched_anchor_sign_flip(
    differences: np.ndarray,
    families: Sequence[str],
    panels: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Two-sided sign flip on per-anchor differences averaged over matched seeds."""

    if resamples <= 0:
        raise RewardTransferError("sign-flip resamples must be positive")
    array, family_array, panel_array = _validate_contrast_arrays(
        differences, families, panels
    )
    anchor_differences = array.mean(axis=0)
    observed = stratified_point_estimate(anchor_differences, families, panels)
    weights = np.zeros(len(anchor_differences), dtype=np.float64)
    for family, panel in EXPECTED_STRATA:
        indices = np.flatnonzero((family_array == family) & (panel_array == panel))
        weights[indices] = 1.0 / (len(EXPECTED_STRATA) * len(indices))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, resamples, 2000):
        width = min(2000, resamples - start)
        signs = rng.integers(0, 2, size=(width, len(anchor_differences)), dtype=np.int8)
        signs = signs * 2 - 1
        null = (signs * anchor_differences[None, :] * weights[None, :]).sum(axis=1)
        exceed += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    return observed, float((exceed + 1) / (resamples + 1))


def verify_scored_run(
    tasks: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Recompute every parser and reward outcome from raw completion text."""

    validate_tasks(tasks)
    task_by_id = {str(task["task_id"]): task for task in tasks}
    row_ids = [str(row.get("task_id")) for row in rows]
    if len(rows) != len(tasks) or len(set(row_ids)) != len(row_ids) or set(row_ids) != set(
        task_by_id
    ):
        raise RewardTransferError("evaluated run does not cover every task exactly once")
    for row in rows:
        task = task_by_id[str(row["task_id"])]
        scoring = score_answer_v3(task, str(row.get("completion", "")))
        expected = {
            "family": task["family"],
            "panel": task.get("source_panel") or task.get("release_panel"),
            "observed_selection": scoring["observed_selection"],
            "parseable": scoring["parseable"],
            "score_bps": scoring["score_bps"],
            "score": scoring["score"],
            "optimal": scoring["optimal"],
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise RewardTransferError(f"scored row differs at {row['task_id']}:{key}")


def load_verified_evaluation(
    directory: Path,
    *,
    split: str,
    gate_path: Path = DEFAULT_RESULTS / "evaluation-gate.json",
    checkpoints: Path = DEFAULT_CHECKPOINTS,
    plan_path: Path = DEFAULT_PLAN,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, int | None], list[dict[str, Any]]],
]:
    """Verify a sealed seven-run evaluation and return its task-aligned rows."""

    if split not in {"primary", "public"}:
        raise RewardTransferError(f"unknown reward-transfer split: {split}")
    plan = load_plan(plan_path)
    gate = verify_evaluation_gate(gate_path, checkpoints=checkpoints, plan_path=plan_path)
    master_path = directory / "evaluation-manifest.json"
    if master_path.is_symlink() or not master_path.is_file():
        raise RewardTransferError(f"evaluation manifest is unavailable: {master_path}")
    master = json.loads(master_path.read_text(encoding="utf-8"))
    verify_content_addressed(master, label=f"{split} evaluation manifest")
    tasks_path = DEFAULT_PRIMARY_TASKS if split == "primary" else DEFAULT_PUBLIC_TASKS
    tasks = read_jsonl(tasks_path)
    expected_task_hash = str(gate["task_file_sha256"][split])
    expected_master = {
        "schema_version": "flavourbench-reward-transfer-evaluation-manifest-v1",
        "status": "complete",
        "split": split,
        "protocol_artifact_sha256": plan["artifact_sha256"],
        "evaluation_gate_artifact_sha256": gate["artifact_sha256"],
        "task_file_sha256": expected_task_hash,
        "tasks": len(tasks),
    }
    for key, value in expected_master.items():
        if master.get(key) != value:
            raise RewardTransferError(f"{split} evaluation manifest differs at {key}")
    if file_sha256(tasks_path) != expected_task_hash:
        raise RewardTransferError(f"{split} task bytes differ from the evaluation gate")

    expected_runs = {("pretrained_base", None)} | {
        (condition, int(seed))
        for condition in TRAINED_CONDITIONS
        for seed in plan["seeds"]
    }
    run_records = master.get("runs")
    if not isinstance(run_records, list) or len(run_records) != len(expected_runs):
        raise RewardTransferError(f"{split} evaluation does not declare seven runs")
    declared = {
        (str(row.get("condition")), row.get("training_seed"))
        for row in run_records
        if isinstance(row, Mapping)
    }
    if declared != expected_runs:
        raise RewardTransferError(f"{split} evaluation run identities differ")

    rows_by_run: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for record in run_records:
        condition = str(record["condition"])
        raw_seed = record["training_seed"]
        seed = int(raw_seed) if raw_seed is not None else None
        manifest_path = _regular_manifest_member(directory, str(record["manifest"]))
        output_path = _regular_manifest_member(directory, str(record["output"]))
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_content_addressed(
            run_manifest,
            label=f"{split} evaluation run {condition}/{seed}",
        )
        if run_manifest.get("artifact_sha256") != record.get("run_artifact_sha256"):
            raise RewardTransferError(f"{split} evaluation run hash differs: {condition}/{seed}")
        adapter_hash = None
        if seed is not None:
            training = verify_training_run(
                checkpoints / condition / f"seed-{seed}",
                plan=plan,
                condition=condition,
                seed=seed,
            )
            adapter_hash = training["artifact_sha256"]
        expected_run = {
            "schema_version": "flavourbench-reward-transfer-evaluation-run-v1",
            "status": "complete",
            "split": split,
            "condition": condition,
            "training_seed": seed,
            "protocol_artifact_sha256": plan["artifact_sha256"],
            "evaluation_gate_artifact_sha256": gate["artifact_sha256"],
            "task_file_sha256": expected_task_hash,
            "adapter_run_artifact_sha256": adapter_hash,
            "decoding": plan["decoding"],
            "rows": len(tasks),
        }
        for key, value in expected_run.items():
            if run_manifest.get(key) != value:
                raise RewardTransferError(
                    f"{split} evaluation run {condition}/{seed} differs at {key}"
                )
        output_record = run_manifest.get("output")
        if (
            not isinstance(output_record, Mapping)
            or output_record.get("name") != output_path.name
            or int(output_record.get("bytes", -1)) != output_path.stat().st_size
            or output_record.get("sha256") != file_sha256(output_path)
        ):
            raise RewardTransferError(f"{split} evaluation output binding differs: {output_path}")
        rows = read_jsonl(output_path)
        if any(
            row.get("condition") != condition or row.get("training_seed") != seed for row in rows
        ):
            raise RewardTransferError(f"{split} evaluation contains mixed run rows")
        verify_scored_run(tasks, rows)
        rows_by_run[(condition, seed)] = rows
    return master, tasks, rows_by_run


def summarize_run(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RewardTransferError("cannot summarize an empty run")
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    parseable = np.asarray([bool(row["parseable"]) for row in rows], dtype=bool)
    families = [str(row["family"]) for row in rows]
    panels = [str(row["panel"]) for row in rows]
    per_family = []
    for family in PRIMARY_FAMILIES:
        indices = np.asarray([value == family for value in families], dtype=bool)
        parsed_indices = indices & parseable
        per_family.append(
            {
                "family": family,
                "tasks": int(indices.sum()),
                "score_unconditional": float(scores[indices].mean()),
                "parse_rate": float(parseable[indices].mean()),
                "score_conditional_on_parse": (
                    float(scores[parsed_indices].mean()) if parsed_indices.any() else None
                ),
                "exact_optimum_rate": float(
                    np.mean([bool(row["optimal"]) for row in rows if row["family"] == family])
                ),
            }
        )
    return {
        "tasks": len(rows),
        "score_unconditional": stratified_point_estimate(scores, families, panels),
        "parse_rate": float(parseable.mean()),
        "score_conditional_on_parse": float(scores[parseable].mean()) if parseable.any() else None,
        "exact_optimum_rate": float(np.mean([bool(row["optimal"]) for row in rows])),
        "per_family": per_family,
    }
