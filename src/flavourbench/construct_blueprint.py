from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    candidates = (Path.cwd().resolve(), Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / "contracts/season1/season1-construct-blueprint-v1.json").is_file():
            return candidate
    return candidates[-1]


BLUEPRINT_PATH = _project_root() / "contracts/season1/season1-construct-blueprint-v1.json"


class ConstructBlueprintError(ValueError):
    """A proposed task bank does not measure the frozen culinary construct."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_blueprint() -> dict[str, Any]:
    value = json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Season 1 construct blueprint is not a JSON object")
    embedded = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    observed = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if embedded != observed:
        raise RuntimeError("Season 1 construct blueprint content address is invalid")
    return value


BLUEPRINT = _load_blueprint()
BLUEPRINT_SHA256 = str(BLUEPRINT["artifact_sha256"])
DIFFICULTY_TIERS = frozenset(BLUEPRINT["difficulty_tiers"])
CONSTRUCT_CELLS = {
    family: tuple(specification["construct_cells"])
    for family, specification in BLUEPRINT["families"].items()
}


def validate_task_binding(
    *,
    family: str,
    construct_blueprint_sha256: str,
    construct_cell_id: str,
    difficulty_tier: str,
) -> None:
    if construct_blueprint_sha256 != BLUEPRINT_SHA256:
        raise ConstructBlueprintError("task is not bound to the frozen construct blueprint")
    if family not in CONSTRUCT_CELLS or construct_cell_id not in CONSTRUCT_CELLS[family]:
        raise ConstructBlueprintError("task construct cell is not valid for its family")
    if difficulty_tier not in DIFFICULTY_TIERS:
        raise ConstructBlueprintError("task difficulty tier is not frozen")


def validate_candidate_binding(
    candidate: dict[str, Any],
    *,
    family: str,
    construct_cell_id: str,
    difficulty_tier: str,
) -> None:
    if not (
        candidate.get("construct_blueprint_sha256") == BLUEPRINT_SHA256
        and candidate.get("construct_cell_id") == construct_cell_id
        and candidate.get("difficulty_tier") == difficulty_tier
        and candidate.get("family") == family
        and construct_cell_id in (candidate.get("subskills") or [])
    ):
        raise ConstructBlueprintError(
            "task candidate and confirmatory construct bindings differ"
        )


def _normalized_words(prompt: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", prompt).casefold()
    return tuple(re.findall(r"[a-z0-9]+", value))


def _trigrams(prompt: str) -> set[tuple[str, ...]]:
    words = _normalized_words(prompt)
    width = min(3, len(words))
    return {
        tuple(words[index : index + width])
        for index in range(max(1, len(words) - width + 1))
    }


def _jaccard(first: set[tuple[str, ...]], second: set[tuple[str, ...]]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def validate_confirmatory_bank(tasks: Sequence[Any]) -> None:
    if len(tasks) != int(BLUEPRINT["task_count"]):
        raise ConstructBlueprintError("construct blueprint requires exactly 240 tasks")

    cell_counts: Counter[tuple[str, str]] = Counter()
    cell_split_counts: Counter[tuple[str, str, str]] = Counter()
    cell_difficulty_counts: Counter[tuple[str, str, str]] = Counter()
    family_difficulty_counts: Counter[tuple[str, str]] = Counter()
    executable_by_family: Counter[str] = Counter()
    executable_by_cell: Counter[tuple[str, str]] = Counter()
    author_counts: Counter[str] = Counter()
    author_family_counts: Counter[tuple[str, str]] = Counter()
    author_cell_counts: Counter[tuple[str, str, str]] = Counter()
    normalized_prompts: set[tuple[str, ...]] = set()
    prompt_features: list[tuple[str, set[tuple[str, ...]]]] = []

    for task in tasks:
        family = str(task.family.value)
        split = str(task.split.value)
        cell = str(task.construct_cell_id)
        difficulty = str(task.difficulty_tier)
        validate_task_binding(
            family=family,
            construct_blueprint_sha256=str(task.construct_blueprint_sha256),
            construct_cell_id=cell,
            difficulty_tier=difficulty,
        )
        cell_counts[(family, cell)] += 1
        cell_split_counts[(family, cell, split)] += 1
        cell_difficulty_counts[(family, cell, difficulty)] += 1
        family_difficulty_counts[(family, difficulty)] += 1
        author = str(task.human_author_id)
        author_counts[author] += 1
        author_family_counts[(author, family)] += 1
        author_cell_counts[(author, family, cell)] += 1
        if task.validator_contract.objective_scope == "executable_subset":
            executable_by_family[family] += 1
            executable_by_cell[(family, cell)] += 1
        normalized = _normalized_words(str(task.prompt))
        if normalized in normalized_prompts:
            raise ConstructBlueprintError("normalized duplicate prompts are prohibited")
        normalized_prompts.add(normalized)
        prompt_features.append((str(task.public_id), _trigrams(str(task.prompt))))

    cell_quota = int(BLUEPRINT["cell_quotas"]["tasks_per_family_cell"])
    difficulty_cell_quota = int(
        BLUEPRINT["cell_quotas"]["tasks_per_family_cell_difficulty"]
    )
    split_quotas = BLUEPRINT["cell_quotas"]["splits_per_family_cell"]
    for family, cells in CONSTRUCT_CELLS.items():
        for cell in cells:
            if cell_counts[(family, cell)] != cell_quota:
                raise ConstructBlueprintError(
                    f"{family}/{cell} requires exactly {cell_quota} tasks"
                )
            for split, expected in split_quotas.items():
                if cell_split_counts[(family, cell, split)] != int(expected):
                    raise ConstructBlueprintError(
                        f"{family}/{cell}/{split} violates its frozen quota"
                    )
            for difficulty in DIFFICULTY_TIERS:
                if (
                    cell_difficulty_counts[(family, cell, difficulty)]
                    != difficulty_cell_quota
                ):
                    raise ConstructBlueprintError(
                        f"{family}/{cell}/{difficulty} violates its frozen quota"
                    )
        for difficulty, specification in BLUEPRINT["difficulty_tiers"].items():
            if family_difficulty_counts[(family, difficulty)] != int(
                specification["tasks_per_family"]
            ):
                raise ConstructBlueprintError(
                    f"{family}/{difficulty} violates its frozen difficulty quota"
                )

    authorship = BLUEPRINT["authorship"]
    if len(author_counts) < int(authorship["minimum_distinct_authors"]):
        raise ConstructBlueprintError("task bank has insufficient author diversity")
    if max(author_counts.values(), default=0) > int(authorship["maximum_tasks_per_author"]):
        raise ConstructBlueprintError("one author exceeds the total task concentration cap")
    if max(author_family_counts.values(), default=0) > int(
        authorship["maximum_tasks_per_author_family"]
    ):
        raise ConstructBlueprintError("one author exceeds a family concentration cap")
    if max(author_cell_counts.values(), default=0) > int(
        authorship["maximum_tasks_per_author_cell"]
    ):
        raise ConstructBlueprintError("one author exceeds a construct-cell concentration cap")

    validation = BLUEPRINT["objective_validation"]
    if sum(executable_by_family.values()) < int(
        validation["minimum_executable_subset_tasks"]
    ):
        raise ConstructBlueprintError("task bank misses the executable-validator coverage floor")
    for family, cells in CONSTRUCT_CELLS.items():
        if executable_by_family[family] < int(
            validation["minimum_executable_subset_tasks_per_family"]
        ):
            raise ConstructBlueprintError(
                f"{family} misses the executable-validator coverage floor"
            )
        for cell in cells:
            if executable_by_cell[(family, cell)] < int(
                validation["minimum_executable_subset_tasks_per_family_cell"]
            ):
                raise ConstructBlueprintError(
                    f"{family}/{cell} misses the executable-validator coverage floor"
                )

    threshold = float(BLUEPRINT["near_duplicate_policy"]["maximum_pairwise_jaccard"])
    for index, (first_id, first) in enumerate(prompt_features):
        for second_id, second in prompt_features[index + 1 :]:
            if _jaccard(first, second) > threshold:
                raise ConstructBlueprintError(
                    f"near-duplicate prompts exceed the threshold: {first_id}, {second_id}"
                )
