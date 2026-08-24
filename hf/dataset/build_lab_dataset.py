"""Build contamination-separated FlavourBench training, validation, and transfer tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "data-lab"
DEFAULT_PANEL_1 = (
    REPOSITORY / "benchmark/powered-v44/taskset/"
    "epicure-selection-taskset-a33bf28db372090015118371417b0e8ed1254f416d03d2c2c5816a6a752beb41.json"
)
DEFAULT_PANEL_2 = (
    REPOSITORY / "benchmark/powered-v45/taskset/"
    "epicure-selection-taskset-925ba9d1d4be9c2b7a1e9956ecd6c18d34ffcad22eee28522f16892922c91e3f.json"
)
DEFAULT_OFFICIAL_PLAN = (
    REPOSITORY / "benchmark/powered-v84/plan/"
    "epicure-selection-joint-analysis-plan-2ba71c793c8d4b97eed863ee83fd770b429fdefdffebdeafb241672f634ee507.json"
)

PRIMARY_FAMILIES = ("substitution", "pairing", "constraint")
SCHEMA_VERSION = "flavourbench-lab-dataset-v2"
SPLIT_SALT = "flavourbench-lab-anchor-split-v2"
DPO_MIN_MARGIN_BPS = 500
TRAIN_PER_STRATUM = 45
VALIDATION_PER_STRATUM = 12
EVALUATION_PER_STRATUM = 14


class LabDatasetBuildError(RuntimeError):
    """The lab dataset violates its disjoint-anchor or content-integrity contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(dict(row)) + b"\n" for row in rows)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LabDatasetBuildError(f"source is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LabDatasetBuildError(f"source is not a JSON object: {path}")
    payload = dict(value)
    recorded = str(payload.pop("artifact_sha256", ""))
    if not recorded or hashlib.sha256(_canonical(payload)).hexdigest() != recorded:
        raise LabDatasetBuildError(f"source semantic digest failed: {path}")
    return value


def _hash_key(value: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}|{value}".encode()).hexdigest()


def _primary_split(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for family in PRIMARY_FAMILIES:
        for panel in ("panel_1", "panel_2"):
            stratum = sorted(
                (
                    dict(row)
                    for row in rows
                    if row["family"] == family and row["source_panel"] == panel
                ),
                key=lambda row: (_hash_key(str(row["anchor_ingredient"])), str(row["task_id"])),
            )
            if len(stratum) != 71:
                raise LabDatasetBuildError(f"{family}/{panel} does not contain 71 lab tasks")
            if TRAIN_PER_STRATUM + VALIDATION_PER_STRATUM + EVALUATION_PER_STRATUM != len(stratum):
                raise LabDatasetBuildError("primary split cardinalities do not exhaust a stratum")
            split_by_id: dict[str, str] = {}
            boundaries = (
                ("evaluation", EVALUATION_PER_STRATUM),
                ("validation", VALIDATION_PER_STRATUM),
                ("train", TRAIN_PER_STRATUM),
            )
            offset = 0
            for split, count in boundaries:
                for row in stratum[offset : offset + count]:
                    split_by_id[str(row["task_id"])] = split
                offset += count
            for row in stratum:
                split = split_by_id[str(row["task_id"])]
                transformed = {
                    **row,
                    "source_split": row.get("split"),
                    "split": split,
                    "lab_split": split,
                    "official_leaderboard_eligible": False,
                    "official_test_anchor_overlap": False,
                    "official_test_task_overlap": False,
                }
                {"train": train, "validation": validation, "evaluation": evaluation}[split].append(
                    transformed
                )
    train.sort(key=lambda row: (PRIMARY_FAMILIES.index(str(row["family"])), str(row["task_id"])))
    validation.sort(
        key=lambda row: (PRIMARY_FAMILIES.index(str(row["family"])), str(row["task_id"]))
    )
    evaluation.sort(
        key=lambda row: (PRIMARY_FAMILIES.index(str(row["family"])), str(row["task_id"]))
    )
    return train, validation, evaluation


def _supplemental_split(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    anchors = sorted({str(row["anchor_ingredient"]) for row in rows}, key=_hash_key)
    validation_anchors = set(anchors[: round(len(anchors) * 0.2)])
    output = []
    for row in rows:
        split = "validation" if str(row["anchor_ingredient"]) in validation_anchors else "train"
        output.append(
            {
                **dict(row),
                "source_split": row.get("split"),
                "split": split,
                "lab_split": split,
                "track": "supplemental_cultural_composition",
                "official_leaderboard_eligible": False,
                "official_test_anchor_overlap": False,
                "official_test_task_overlap": False,
            }
        )
    return sorted(output, key=lambda row: (str(row["split"]), str(row["task_id"])))


def _render(selection: str) -> str:
    return "FINAL_SELECTION: " + ",".join(selection)


def _sft(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": row["task_id"],
            "family": row["family"],
            "anchor_ingredient": row["anchor_ingredient"],
            "prompt": row["prompt"],
            "completion": _render(str(row["optimal_selection"])),
            "reward": 1.0,
            "optimal_margin_bps": row["optimal_margin_bps"],
            "split": row["split"],
        }
        for row in rows
    ]


def _dpo(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        ranked_actions = sorted(
            (
                (str(selection), int(score))
                for selection, score in row["selection_scores_bps"].items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        candidate_pairs = sorted(
            (
                (chosen, rejected, chosen_bps, rejected_bps)
                for chosen, chosen_bps in ranked_actions
                for rejected, rejected_bps in ranked_actions
                if chosen_bps - rejected_bps >= DPO_MIN_MARGIN_BPS
            ),
            key=lambda item: (-(item[2] - item[3]), -item[2], item[0], item[1]),
        )
        if len(candidate_pairs) < 4:
            raise LabDatasetBuildError(f"reward map lacks DPO resolution: {row['task_id']}")
        selected_indices = (0, len(candidate_pairs) // 3, 2 * len(candidate_pairs) // 3, -1)
        selected_pairs = [candidate_pairs[index] for index in selected_indices]
        if len({(pair[0], pair[1]) for pair in selected_pairs}) != 4:
            raise LabDatasetBuildError(f"DPO quantiles collide for {row['task_id']}")
        for pair_index, pair in enumerate(selected_pairs, start=1):
            chosen_selection, rejected_selection, chosen_bps, rejected_bps = pair
            output.append(
                {
                    "task_id": row["task_id"],
                    "pair_id": f"{row['task_id']}-p{pair_index}",
                    "family": row["family"],
                    "anchor_ingredient": row["anchor_ingredient"],
                    "prompt": row["prompt"],
                    "chosen": _render(chosen_selection),
                    "rejected": _render(rejected_selection),
                    "chosen_reward": chosen_bps / 10_000,
                    "rejected_reward": rejected_bps / 10_000,
                    "reward_margin": (chosen_bps - rejected_bps) / 10_000,
                    "split": row["split"],
                }
            )
    return output


def _grpo(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": row["task_id"],
            "family": row["family"],
            "anchor_ingredient": row["anchor_ingredient"],
            "prompt": row["prompt"],
            "choices": row["choices"],
            "selection_scores_bps": row["selection_scores_bps"],
            "split": row["split"],
        }
        for row in rows
    ]


def build_files(
    *,
    panel_1_path: Path,
    panel_2_path: Path,
    official_plan_path: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    panel_documents = (("panel_1", _load_json(panel_1_path)), ("panel_2", _load_json(panel_2_path)))
    official_plan = _load_json(official_plan_path)
    official_ids: set[str] = set()
    official_anchors: set[str] = set()
    for panel, document in panel_documents:
        task_index = {str(row["task_id"]): row for row in document["tasks"]}
        panel_contract = official_plan.get("common_core", {}).get("panels", {}).get(panel, {})
        selected = panel_contract.get("selected_task_ids_by_family", {})
        if set(selected) != set(PRIMARY_FAMILIES) or any(
            not isinstance(selected[family], list) or len(selected[family]) != 89
            for family in PRIMARY_FAMILIES
        ):
            raise LabDatasetBuildError(f"official plan has an invalid {panel} selection")
        panel_ids = {str(task_id) for family in PRIMARY_FAMILIES for task_id in selected[family]}
        if len(panel_ids) != 267 or not panel_ids <= set(task_index):
            raise LabDatasetBuildError(f"official plan and {panel} taskset differ")
        official_ids.update(panel_ids)
        official_anchors.update(
            str(task_index[task_id]["anchor_ingredient"]) for task_id in panel_ids
        )
    if len(official_ids) != 534 or len(official_anchors) != 534:
        raise LabDatasetBuildError("official plan is not the 534-anchor common core")

    candidates: list[dict[str, Any]] = []
    for panel, document in panel_documents:
        tasks = document.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 640:
            raise LabDatasetBuildError(f"{panel} source does not contain 640 tasks")
        candidates.extend({"source_panel": panel, **dict(row)} for row in tasks)

    disjoint = [
        row
        for row in candidates
        if str(row["task_id"]) not in official_ids
        and str(row["anchor_ingredient"]) not in official_anchors
    ]
    primary = [row for row in disjoint if row["family"] in PRIMARY_FAMILIES]
    supplemental = [row for row in disjoint if row["family"] == "cultural_composition"]
    if len(primary) != 426 or Counter(row["family"] for row in primary) != {
        family: 142 for family in PRIMARY_FAMILIES
    }:
        raise LabDatasetBuildError("disjoint primary development pool differs")
    if len(supplemental) != 283:
        raise LabDatasetBuildError("disjoint supplemental development pool differs")

    train_tasks, validation_tasks, evaluation_tasks = _primary_split(primary)
    if len(train_tasks) != 270 or len(validation_tasks) != 72 or len(evaluation_tasks) != 84:
        raise LabDatasetBuildError("primary train/validation/evaluation cardinality differs")
    train_anchors = {str(row["anchor_ingredient"]) for row in train_tasks}
    validation_anchors = {str(row["anchor_ingredient"]) for row in validation_tasks}
    evaluation_anchors = {str(row["anchor_ingredient"]) for row in evaluation_tasks}
    if (
        train_anchors & validation_anchors
        or train_anchors & evaluation_anchors
        or validation_anchors & evaluation_anchors
        or (train_anchors | validation_anchors | evaluation_anchors) & official_anchors
    ):
        raise LabDatasetBuildError("anchor leakage detected across lab splits")

    supplemental_tasks = _supplemental_split(supplemental)
    files = {
        "train_tasks.jsonl": _jsonl(train_tasks),
        "validation_tasks.jsonl": _jsonl(validation_tasks),
        "evaluation_tasks.jsonl": _jsonl(evaluation_tasks),
        "sft_train.jsonl": _jsonl(_sft(train_tasks)),
        "sft_validation.jsonl": _jsonl(_sft(validation_tasks)),
        "dpo_train.jsonl": _jsonl(_dpo(train_tasks)),
        "dpo_validation.jsonl": _jsonl(_dpo(validation_tasks)),
        "grpo_train.jsonl": _jsonl(_grpo(train_tasks)),
        "grpo_validation.jsonl": _jsonl(_grpo(validation_tasks)),
        "supplemental_cultural_composition.jsonl": _jsonl(supplemental_tasks),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "preregistered_transfer_maps_not_official_leaderboard_test",
        "split_salt": SPLIT_SALT,
        "official_test": {
            "tasks": len(official_ids),
            "anchors": len(official_anchors),
            "plan_semantic_sha256": official_plan["artifact_sha256"],
            "plan_physical_sha256": hashlib.sha256(official_plan_path.read_bytes()).hexdigest(),
        },
        "sources": {
            "panel_1_semantic_sha256": panel_documents[0][1]["artifact_sha256"],
            "panel_2_semantic_sha256": panel_documents[1][1]["artifact_sha256"],
        },
        "counts": {
            "train_tasks": len(train_tasks),
            "validation_tasks": len(validation_tasks),
            "evaluation_tasks": len(evaluation_tasks),
            "sft_train": len(train_tasks),
            "sft_validation": len(validation_tasks),
            "dpo_train": len(train_tasks) * 4,
            "dpo_validation": len(validation_tasks) * 4,
            "grpo_train": len(train_tasks),
            "grpo_validation": len(validation_tasks),
            "supplemental_cultural_composition": len(supplemental_tasks),
        },
        "contract": {
            "primary_families": list(PRIMARY_FAMILIES),
            "train_validation_evaluation_anchor_overlap": 0,
            "development_official_anchor_overlap": 0,
            "direct_optimization_on_official_test_forbidden": True,
            "direct_optimization_on_evaluation_split_forbidden": True,
            "evaluation_split_is_public_not_secret": True,
            "dpo_minimum_reward_margin_bps": DPO_MIN_MARGIN_BPS,
        },
        "files": [
            {
                "name": name,
                "bytes": len(payload),
                "rows": payload.count(b"\n"),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in sorted(files.items())
        ],
    }
    manifest["artifact_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    return files, manifest


def _write_atomic(path: Path, payload: bytes) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FlavourBench lab training dataset")
    parser.add_argument("--panel-1", type=Path, default=DEFAULT_PANEL_1)
    parser.add_argument("--panel-2", type=Path, default=DEFAULT_PANEL_2)
    parser.add_argument("--official-plan", type=Path, default=DEFAULT_OFFICIAL_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files, manifest = build_files(
        panel_1_path=args.panel_1,
        panel_2_path=args.panel_2,
        official_plan_path=args.official_plan,
    )
    expected = {**files, "DATA_MANIFEST.json": _json_bytes(manifest)}
    if args.check:
        observed = {path.name for path in args.output.iterdir() if path.is_file()}
        if observed != set(expected):
            raise LabDatasetBuildError("lab dataset file inventory differs")
        for name, payload in expected.items():
            if (args.output / name).read_bytes() != payload:
                raise LabDatasetBuildError(f"lab dataset output differs: {name}")
        print(f"OK: lab dataset {manifest['artifact_sha256']}")
        return
    for name, payload in expected.items():
        _write_atomic(args.output / name, payload)
    print(f"built lab dataset {manifest['artifact_sha256']} at {args.output}")


if __name__ == "__main__":
    main()
