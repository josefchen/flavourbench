"""Build the label-permutation repeat panel for task replication 2."""

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

from .epicure_selection_powered_plan import _permuted_score_map
from .epicure_selection_taskset_replication_v1 import verify_taskset
from .epicure_selection_taskset_v1 import ALL_SELECTION_KEYS, FAMILIES, LABELS, SELECTION_SIZE
from .epicure_selection_taskset_v2 import CONCRETE_SELECTION_EXAMPLE, PROMPT_PROTOCOL

SCHEMA_VERSION = "flavourbench-selection-powered-repeat-panel-v3-replication-2"
SELECTION_SEED = "flavourbench-selection-repeat-replication-2-20260815"
REPEAT_TASK_COUNT = 64
REPEATS_PER_FAMILY = 16


class SelectionRepeatReplicationError(RuntimeError):
    """The replication-2 repeat panel failed verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionRepeatReplicationError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionRepeatReplicationError("repeat input is not a JSON object")
    return value


def _selection_key(*parts: str) -> str:
    return hashlib.sha256((SELECTION_SEED + "\0" + "\0".join(parts)).encode()).hexdigest()


def _render_prompt(prefix: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {choices[label].replace('_', ' ')}" for label in LABELS)
    prompt = (
        f"{prefix}\n\nCandidates:\n{rendered}\n\n"
        f"Choose exactly {SELECTION_SIZE} distinct labels. Return exactly one line beginning "
        "with the marker `FINAL_SELECTION:`, followed by the three selected A-through-H labels "
        "separated by commas and ordered alphabetically."
    )
    if CONCRETE_SELECTION_EXAMPLE.search(prompt):
        raise SelectionRepeatReplicationError("repeat prompt contains a concrete answer")
    return prompt


def build_repeat_panel(
    taskset: Mapping[str, Any], *, taskset_physical_sha256: str
) -> dict[str, Any]:
    if not verify_taskset(taskset):
        raise SelectionRepeatReplicationError("repeat panel requires replication task set")
    repeats: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_tasks = [task for task in taskset["tasks"] if task["family"] == family]
        selected = sorted(
            family_tasks,
            key=lambda task: _selection_key(family, str(task["task_id"])),
        )[:REPEATS_PER_FAMILY]
        for index, task in enumerate(selected, start=1):
            shift = 1 + int(_selection_key("shift", str(task["task_id"])), 16) % 7
            original = dict(task["choices"])
            permuted = {
                label: original[LABELS[(position - shift) % len(LABELS)]]
                for position, label in enumerate(LABELS)
            }
            scores = _permuted_score_map(
                original_choices=original,
                new_choices=permuted,
                scores=task["selection_scores_bps"],
            )
            optimum = max(scores, key=scores.__getitem__)
            prefix = str(task["prompt"]).split("\n\nCandidates:\n", 1)[0]
            prompt = _render_prompt(prefix, permuted)
            repeats.append(
                {
                    "task_id": f"repeat-rep2-{family}-{index:02d}",
                    "original_task_id": task["task_id"],
                    "family": family,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_protocol": PROMPT_PROTOCOL,
                    "choices": permuted,
                    "selection_size": SELECTION_SIZE,
                    "selection_scores_bps": scores,
                    "optimal_selection": optimum,
                    "original_optimal_selection": task["optimal_selection"],
                    "permutation_shift": shift,
                    "oracle_reference_sha256": task["oracle_reference_sha256"],
                }
            )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_replication_2_model_execution",
        "prompt_protocol": PROMPT_PROTOCOL,
        "source_taskset_artifact_sha256": taskset["artifact_sha256"],
        "source_taskset_physical_sha256": taskset_physical_sha256,
        "source_task_set_sha256": taskset["task_set_sha256"],
        "selection_seed": SELECTION_SEED,
        "counts": {
            "tasks": REPEAT_TASK_COUNT,
            "tasks_per_family": REPEATS_PER_FAMILY,
            "provider_calls_per_model": REPEAT_TASK_COUNT,
        },
        "purpose": {
            "primary": "ingredient-set agreement after answer-label permutation",
            "secondary": "score stability and label-position sensitivity",
            "excluded_from_primary_score": True,
        },
        "tasks": repeats,
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_repeat_panel(document, taskset=taskset):
        raise SelectionRepeatReplicationError("constructed repeat panel failed verification")
    return document


def verify_repeat_panel(
    document: Mapping[str, Any], *, taskset: Mapping[str, Any] | None = None
) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    tasks = document.get("tasks")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("prompt_protocol") != PROMPT_PROTOCOL
        or recorded != _sha256(payload)
        or not isinstance(tasks, list)
        or len(tasks) != REPEAT_TASK_COUNT
        or len({task.get("task_id") for task in tasks}) != REPEAT_TASK_COUNT
        or len({task.get("original_task_id") for task in tasks}) != REPEAT_TASK_COUNT
        or Counter(task.get("family") for task in tasks)
        != Counter({family: REPEATS_PER_FAMILY for family in FAMILIES})
    ):
        return False
    if not all(
        task.get("prompt_protocol") == PROMPT_PROTOCOL
        and not CONCRETE_SELECTION_EXAMPLE.search(str(task.get("prompt") or ""))
        and hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
        == task.get("prompt_sha256")
        and task.get("permutation_shift") in set(range(1, 8))
        and set(task.get("choices") or {}) == set(LABELS)
        and set(task.get("selection_scores_bps") or {}) == set(ALL_SELECTION_KEYS)
        and sum(value == 10_000 for value in task["selection_scores_bps"].values()) == 1
        and task["selection_scores_bps"].get(task.get("optimal_selection")) == 10_000
        for task in tasks
    ):
        return False
    if taskset is None:
        return True
    if not verify_taskset(taskset):
        return False
    if document.get("source_taskset_artifact_sha256") != taskset.get(
        "artifact_sha256"
    ) or document.get("source_task_set_sha256") != taskset.get("task_set_sha256"):
        return False
    originals = {task["task_id"]: task for task in taskset["tasks"]}
    for repeat in tasks:
        original = originals.get(repeat["original_task_id"])
        if original is None:
            return False
        original_label = {value: label for label, value in original["choices"].items()}
        for key, score in repeat["selection_scores_bps"].items():
            ingredients = [repeat["choices"][label] for label in key]
            source_key = "".join(sorted(original_label[value] for value in ingredients))
            if score != original["selection_scores_bps"][source_key]:
                return False
    return True


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-repeat-panel-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionRepeatReplicationError("content-addressed repeat-panel conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    taskset = _load(args.taskset)
    document = build_repeat_panel(taskset, taskset_physical_sha256=_sha256_file(args.taskset))
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
