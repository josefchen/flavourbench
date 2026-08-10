"""Frozen prospective Season 1 design constants.

Retrospective Season 0 modules retain their historical 12-model, 120-task
identities. Production lifecycle checks import this module so the prospective
protocol cannot silently drift back to the pilot design.
"""

from __future__ import annotations

SEASON_MODEL_COUNT = 16
CONFIRMATORY_TASK_COUNT = 240
CONFIRMATORY_TASKS_PER_FAMILY = 60
SEASON_TASK_SPLIT_COUNTS = {
    "scored": 160,
    "development": 40,
    "private_reserve": 40,
}
SEASON_TASK_SPLIT_COUNTS_PER_FAMILY = {
    "scored": 40,
    "development": 10,
    "private_reserve": 10,
}

SEASON_SLOT_ROLE_COUNTS = {
    "closed_family": 4,
    "open_weight": 8,
    "efficiency": 2,
    "reasoning": 2,
}

if sum(SEASON_SLOT_ROLE_COUNTS.values()) != SEASON_MODEL_COUNT:  # pragma: no cover
    raise RuntimeError("Season 1 slot-role counts do not match the model count")

if sum(SEASON_TASK_SPLIT_COUNTS.values()) != CONFIRMATORY_TASK_COUNT:  # pragma: no cover
    raise RuntimeError("Season 1 split counts do not match the task count")

if any(
    count * 4 != SEASON_TASK_SPLIT_COUNTS[split]
    for split, count in SEASON_TASK_SPLIT_COUNTS_PER_FAMILY.items()
):  # pragma: no cover
    raise RuntimeError("Season 1 split-by-family counts are inconsistent")
