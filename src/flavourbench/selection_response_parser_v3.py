"""Unambiguously decode FlavourBench selections from labels or exact ingredient names."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .epicure_selection_taskset_v1 import SELECTION_SIZE

PARSER_SCHEMA_VERSION = "flavourbench-selection-set-parser-v3"
_MARKER = re.compile(r"FINAL_SELECTION\s*:\s*", flags=re.IGNORECASE)
_LABEL_TRIPLE = re.compile(
    r"^\s*([A-H])\s*,\s*([A-H])\s*,\s*([A-H])"
    r"\s*(?:[`*_]+\s*)?(?:<\|close\|>response\s*)?$",
    flags=re.IGNORECASE,
)


def _normal_name(value: str) -> str:
    return " ".join(value.replace("_", " ").casefold().split())


def _marker_segments(answer_markdown: str) -> tuple[str, ...]:
    matches = tuple(_MARKER.finditer(answer_markdown))
    segments: list[str] = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(answer_markdown)
        segment = answer_markdown[match.end() : stop].splitlines()[0]
        segments.append(segment.strip())
    return tuple(segments)


def parse_final_selection_v3(task: Mapping[str, Any], answer_markdown: str) -> str | None:
    """Return one unique three-label selection, or fail closed on ambiguity."""

    choices = task.get("choices") or {}
    if set(choices) != set("ABCDEFGH"):
        return None
    name_to_label = {_normal_name(str(name)): str(label) for label, name in choices.items()}
    if len(name_to_label) != len(choices):
        return None
    candidates: set[str] = set()
    for segment in _marker_segments(answer_markdown):
        label_match = _LABEL_TRIPLE.fullmatch(segment)
        if label_match:
            labels = tuple(value.upper() for value in label_match.groups())
            if len(set(labels)) == SELECTION_SIZE:
                candidates.add("".join(sorted(labels)))
            continue
        rendered = segment.strip().strip("`*_").strip()
        names = tuple(_normal_name(value) for value in rendered.split(","))
        if len(names) != SELECTION_SIZE or any(name not in name_to_label for name in names):
            continue
        labels = tuple(name_to_label[name] for name in names)
        if len(set(labels)) == SELECTION_SIZE:
            candidates.add("".join(sorted(labels)))
    return next(iter(candidates)) if len(candidates) == 1 else None


def score_answer_v3(task: Mapping[str, Any], answer_markdown: str) -> dict[str, Any]:
    observed = parse_final_selection_v3(task, answer_markdown)
    score_bps = int(task["selection_scores_bps"].get(observed, 0)) if observed else 0
    return {
        "observed_selection": observed,
        "optimal_selection": task["optimal_selection"],
        "parseable": observed is not None,
        "score_bps": score_bps,
        "score": score_bps / 100,
        "optimal": score_bps == 10_000,
    }
