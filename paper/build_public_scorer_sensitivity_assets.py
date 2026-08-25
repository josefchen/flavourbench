#!/usr/bin/env python3
"""Rescore the complete core with three immutable public Epicure checkpoints.

This is an explicitly post-collection sensitivity analysis.  It keeps the
published prompts, candidate sets, constraints, and model selections fixed,
then replaces only the embedding used to construct each task's 56-action
reward map.  It therefore tests scorer dependence conditional on the released
tasks; it is not a new preregistered benchmark run or an external validation of
culinary quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import struct
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flavourbench.epicure_selection_common_core_analysis_v1 import (
    load_complete_common_core,
)
from flavourbench.epicure_selection_complete_core_plan_v84 import (
    selected_task_ids,
    verify_plan,
)
from flavourbench.epicure_selection_complete_core_sources_v1 import source_graph
from flavourbench.epicure_selection_route_manifest_v57 import DEEPSEEK_PRO_MODEL_ID

SCHEMA_VERSION = "flavourbench-public-epicure-scorer-sensitivity-v1"
PLAN_PATH = Path(
    "benchmark/powered-v84/plan/epicure-selection-joint-analysis-plan-"
    "2ba71c793c8d4b97eed863ee83fd770b429fdefdffebdeafb241672f634ee507.json"
)
FAMILIES = ("substitution", "pairing", "constraint")
PANELS = ("panel_1", "panel_2")
LABELS = tuple("ABCDEFGH")
SELECTION_KEYS = tuple("".join(value) for value in itertools.combinations(LABELS, 3))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260826
VOCAB_SHA256 = "5a10278cd71eeb66051d23ef8621b917a60ecd493d4580f60d324623191f8005"
INGREDIENT_TAGS_SHA256 = "8f52e83a072069f436ab7d851ed0251e775da92afc46e0deaa61d49d91014772"
PUBLIC_EPICURE_MCP_COMMIT = "14ddf04aba81a76b75efa6554041f6bff48992c6"
INGREDIENT_TAGS_URL = (
    "https://raw.githubusercontent.com/KAIKAKU-AI/epicure-mcp/"
    f"{PUBLIC_EPICURE_MCP_COMMIT}/data/ingredient_tags.csv"
)

CHECKPOINTS: dict[str, dict[str, str]] = {
    "cooc": {
        "label": "Epicure-Cooc",
        "repo_id": "Kaikaku/epicure-cooc",
        "revision": "03edd311adde6e39a2eb6f9f3fa78f7396be6b53",
        "embeddings_sha256": ("5ce3bdd03deb8cc8e30bd2a574e04a997b9990422d9203dac4f148949b701a87"),
        "config_sha256": ("8fd5ed3de8295d33240b4fbf16a3a84788c916bcb09ab450434b8b0a1dfd6853"),
    },
    "core": {
        "label": "Epicure-Core",
        "repo_id": "Kaikaku/epicure-core",
        "revision": "d31ebb5af8e92bbaf5cb67381d5006d4ea8368b7",
        "embeddings_sha256": ("58c965532709e415cc00098ea24ad153ca5e02f2ffc88d6b7287c36308e34120"),
        "config_sha256": ("0edfdd497a9828d86bd3490294512bc7e94fdee085fe8b1a5f1811e4b9245a99"),
    },
    "chem": {
        "label": "Epicure-Chem",
        "repo_id": "Kaikaku/epicure-chem",
        "revision": "2461ef3fbafab36d2b1111187a3df98721146861",
        "embeddings_sha256": ("66679efdd10502f3ba52bc18c379a3617a11c8b30afc29327a286b1c5b2a22b9"),
        "config_sha256": ("423b70aa8b1253d8fd401207f4e8359f5411a242d6627b6879cda672bf146b6c"),
    },
}


class PublicScorerSensitivityError(RuntimeError):
    """A bound input or derived scorer-sensitivity result is inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PublicScorerSensitivityError(f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicScorerSensitivityError(f"JSON input is not an object: {path}")
    return value


def _discover_checkpoint(
    name: str,
    checkpoint_root: Path | None,
    *,
    allow_network_download: bool,
) -> Path:
    record = CHECKPOINTS[name]
    if checkpoint_root is not None:
        candidates = (
            checkpoint_root / name,
            checkpoint_root / f"epicure-{name}",
            checkpoint_root,
        )
    else:
        candidates = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--Kaikaku--epicure-{name}"
            / "snapshots"
            / record["revision"],
        )
    for candidate in candidates:
        if all(
            (candidate / filename).is_file()
            for filename in (
                "embeddings.safetensors",
                "vocab.json",
                "config.json",
            )
        ):
            return candidate
    if checkpoint_root is not None:
        raise PublicScorerSensitivityError(
            f"immutable {record['repo_id']} revision {record['revision']} is unavailable "
            f"under explicit checkpoint root {checkpoint_root}"
        )
    if allow_network_download:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise PublicScorerSensitivityError(
                "huggingface-hub is required for --allow-network-download"
            ) from error
        downloaded = [
            Path(
                hf_hub_download(
                    repo_id=record["repo_id"],
                    filename=filename,
                    revision=record["revision"],
                )
            )
            for filename in ("embeddings.safetensors", "vocab.json", "config.json")
        ]
        directories = {path.parent.resolve() for path in downloaded}
        if len(directories) != 1:
            raise PublicScorerSensitivityError(
                f"downloaded {record['repo_id']} files do not share one immutable snapshot"
            )
        directory = directories.pop()
        if all((directory / path.name).is_file() for path in downloaded):
            return directory
    raise PublicScorerSensitivityError(
        f"immutable {record['repo_id']} revision {record['revision']} is unavailable; "
        "download embeddings.safetensors, vocab.json, and config.json with `hf download` "
        "or pass --checkpoint-root, or opt in with --allow-network-download"
    )


def _resolve_ingredient_tags(
    *,
    root: Path,
    explicit_path: Path | None,
    allow_network_download: bool,
) -> Path:
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        if not candidate.is_file():
            raise PublicScorerSensitivityError(
                f"explicit ingredient-tags input does not exist: {candidate}"
            )
        if _file_sha256(candidate) != INGREDIENT_TAGS_SHA256:
            raise PublicScorerSensitivityError(
                f"explicit ingredient-tags hash differs from {INGREDIENT_TAGS_SHA256}: {candidate}"
            )
        return candidate

    sibling = root.parent / "epicure" / "epicure-mcp" / "data" / "ingredient_tags.csv"
    if sibling.is_file() and _file_sha256(sibling) == INGREDIENT_TAGS_SHA256:
        return sibling.resolve()

    cache = (
        Path.home()
        / ".cache"
        / "flavourbench"
        / "public-scorer"
        / PUBLIC_EPICURE_MCP_COMMIT
        / "ingredient_tags.csv"
    )
    if cache.is_file() and _file_sha256(cache) == INGREDIENT_TAGS_SHA256:
        return cache.resolve()
    if not allow_network_download:
        raise PublicScorerSensitivityError(
            "the immutable ingredient-tags input is unavailable; pass --ingredient-tags "
            "or opt in with --allow-network-download"
        )

    request = urllib.request.Request(  # noqa: S310
        INGREDIENT_TAGS_URL,
        headers={"User-Agent": "FlavourBench-public-scorer-rebuild/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = response.read()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != INGREDIENT_TAGS_SHA256:
        raise PublicScorerSensitivityError(f"downloaded ingredient-tags hash differs: {observed}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    return cache.resolve()


def _read_safetensors(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_length_bytes = handle.read(8)
        if len(header_length_bytes) != 8:
            raise PublicScorerSensitivityError(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        header = json.loads(handle.read(header_length))
        if set(header) != {"embeddings"}:
            raise PublicScorerSensitivityError(f"unexpected safetensors keys: {path}")
        specification = header["embeddings"]
        if specification.get("dtype") != "F32" or specification.get("shape") != [1790, 300]:
            raise PublicScorerSensitivityError(f"unexpected embedding tensor metadata: {path}")
        start, stop = map(int, specification["data_offsets"])
        handle.seek(8 + header_length + start)
        payload = handle.read(stop - start)
    matrix = np.frombuffer(payload, dtype="<f4").reshape(1790, 300).copy()
    if not np.isfinite(matrix).all():
        raise PublicScorerSensitivityError(f"non-finite public embedding value: {path}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise PublicScorerSensitivityError(f"zero-norm public embedding row: {path}")
    return matrix / norms


def _load_checkpoint(
    name: str, directory: Path
) -> tuple[dict[str, int], np.ndarray, dict[str, Any]]:
    record = CHECKPOINTS[name]
    files = {
        "embeddings.safetensors": record["embeddings_sha256"],
        "vocab.json": VOCAB_SHA256,
        "config.json": record["config_sha256"],
    }
    for filename, expected in files.items():
        observed = _file_sha256(directory / filename)
        if observed != expected:
            raise PublicScorerSensitivityError(
                f"{record['repo_id']} {filename} hash differs: {observed}"
            )
    vocab_value = _load_json(directory / "vocab.json")
    if (
        len(vocab_value) != 1790
        or set(vocab_value.values()) != set(range(1790))
        or any(
            not isinstance(key, str) or not isinstance(value, int)
            for key, value in vocab_value.items()
        )
    ):
        raise PublicScorerSensitivityError(f"malformed public vocabulary: {directory}")
    config = _load_json(directory / "config.json")
    if (
        config.get("schema") != name
        or config.get("vocab_size") != 1790
        or config.get("d_model") != 300
    ):
        raise PublicScorerSensitivityError(f"public checkpoint config differs: {directory}")
    return vocab_value, _read_safetensors(directory / "embeddings.safetensors"), config


def _load_ingredient_tags(path: Path) -> dict[str, dict[str, Any]]:
    if _file_sha256(path) != INGREDIENT_TAGS_SHA256:
        raise PublicScorerSensitivityError("ingredient-tags hash differs from the task compiler")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                nova: float | None = float(row["nova_level"])
                if not math.isfinite(nova):
                    nova = None
            except (TypeError, ValueError):
                nova = None
            name = str(row["name"])
            rows[name] = {
                "vegan": str(row["is_vegan"]).lower() == "true",
                "vegetarian": str(row["is_vegetarian"]).lower() == "true",
                "nova": nova,
            }
    if len(rows) != 1790:
        raise PublicScorerSensitivityError("ingredient-tags vocabulary cardinality differs")
    return rows


def _mean_pairwise_cosine(matrix: np.ndarray) -> float:
    return float(
        np.mean(
            [
                matrix[left] @ matrix[right]
                for left, right in itertools.combinations(range(len(matrix)), 2)
            ]
        )
    )


def _public_score_map(
    task: Mapping[str, Any],
    *,
    vocab: Mapping[str, int],
    embeddings: np.ndarray,
    tags: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    family = str(task["family"])
    anchor = embeddings[int(vocab[str(task["anchor_ingredient"])])]
    choices = task["choices"]
    raw: dict[str, float | None] = {}
    for selection in SELECTION_KEYS:
        names = [str(choices[label]) for label in selection]
        vectors = embeddings[[int(vocab[name]) for name in names]]
        anchor_similarity = float(np.mean(vectors @ anchor))
        coherence = _mean_pairwise_cosine(vectors)
        if family == "substitution":
            utility: float | None = 0.8 * anchor_similarity + 0.2 * coherence
        elif family == "pairing":
            utility = 0.65 * anchor_similarity + 0.35 * coherence
        elif family == "constraint":
            reference = task["oracle_reference"]
            variant = str(reference["constraint_variant"])
            maximum_nova = int(reference["maximum_nova"])
            invalid = any(
                tags[name]["nova"] is None
                or float(tags[name]["nova"]) > maximum_nova
                or tags[name][variant] is not True
                for name in names
            )
            utility = None if invalid else 0.7 * anchor_similarity + 0.3 * coherence
        else:
            raise PublicScorerSensitivityError(f"unsupported complete-core family: {family}")
        raw[selection] = utility
    valid = [value for value in raw.values() if value is not None and math.isfinite(value)]
    if len(valid) < 4 or max(valid) - min(valid) < 1e-6:
        raise PublicScorerSensitivityError(f"public scorer lacks a usable range: {task['task_id']}")
    low, high = min(valid), max(valid)
    scores = {
        key: (
            0
            if value is None
            else int(round(10_000 * max(0.0, min(1.0, (value - low) / (high - low)))))
        )
        for key, value in raw.items()
    }
    if max(scores.values()) != 10_000:
        raise PublicScorerSensitivityError(f"public score map has no optimum: {task['task_id']}")
    return scores


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _point_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(-array, kind="stable")
    ranks = np.empty(len(array), dtype=int)
    ranks[order] = np.arange(1, len(array) + 1)
    return ranks


def _pair_order_agreement(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    triangle = np.triu_indices(len(left_array), k=1)
    return float(
        np.mean(
            np.sign(left_array[triangle[0]] - left_array[triangle[1]])
            == np.sign(right_array[triangle[0]] - right_array[triangle[1]])
        )
    )


def _balanced_scores(matrix: np.ndarray, strata: Sequence[str]) -> np.ndarray:
    labels = np.asarray(strata, dtype=object)
    parts = [matrix[:, labels == stratum].mean(axis=1) for stratum in sorted(set(strata))]
    if len(parts) != 6 or any(
        len(np.flatnonzero(labels == stratum)) != 89 for stratum in set(strata)
    ):
        raise PublicScorerSensitivityError("the complete core is not six balanced 89-task strata")
    return np.stack(parts, axis=1).mean(axis=1)


def _bootstrap_indices(strata: Sequence[str], replicates: int, seed: int) -> list[np.ndarray]:
    labels = np.asarray(strata, dtype=object)
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    for stratum in sorted(set(strata)):
        positions = np.flatnonzero(labels == stratum)
        if len(positions) != 89:
            raise PublicScorerSensitivityError("bootstrap stratum cardinality differs")
        output.append(positions[rng.integers(0, len(positions), size=(replicates, len(positions)))])
    if len(output) != 6:
        raise PublicScorerSensitivityError("bootstrap stratum count differs")
    return output


def _bootstrap_scores(matrix: np.ndarray, draws: Sequence[np.ndarray]) -> np.ndarray:
    output = np.zeros((draws[0].shape[0], matrix.shape[0]), dtype=float)
    for indices in draws:
        output += matrix[:, indices].mean(axis=2).T / len(draws)
    return output


def _bootstrap_agreement(
    original: np.ndarray,
    alternative: np.ndarray,
    *,
    release_leader_index: int,
) -> dict[str, Any]:
    if original.shape != alternative.shape or original.ndim != 2:
        raise PublicScorerSensitivityError("bootstrap score matrices differ")
    replicates, model_count = original.shape
    original_order = np.argsort(-original, axis=1, kind="stable")
    alternative_order = np.argsort(-alternative, axis=1, kind="stable")
    original_ranks = np.empty_like(original_order)
    alternative_ranks = np.empty_like(alternative_order)
    rows = np.arange(replicates)[:, None]
    original_ranks[rows, original_order] = np.arange(1, model_count + 1)
    alternative_ranks[rows, alternative_order] = np.arange(1, model_count + 1)
    squared = np.square(original_ranks - alternative_ranks).sum(axis=1)
    rho = 1.0 - 6.0 * squared / (model_count * (model_count**2 - 1))
    pair = np.zeros(replicates, dtype=float)
    triangle = np.triu_indices(model_count, k=1)
    for start in range(0, replicates, 1000):
        stop = min(replicates, start + 1000)
        original_sign = np.sign(
            original[start:stop, triangle[0]] - original[start:stop, triangle[1]]
        )
        alternative_sign = np.sign(
            alternative[start:stop, triangle[0]] - alternative[start:stop, triangle[1]]
        )
        pair[start:stop] = np.mean(original_sign == alternative_sign, axis=1)
    original_leader = np.argmax(original, axis=1)
    alternative_leader = np.argmax(alternative, axis=1)
    return {
        "replicates": replicates,
        "rank_spearman": {
            "median": float(np.median(rho)),
            "percentile_95_interval": list(map(float, np.quantile(rho, [0.025, 0.975]))),
        },
        "pair_order_agreement": {
            "median": float(np.median(pair)),
            "percentile_95_interval": list(map(float, np.quantile(pair, [0.025, 0.975]))),
        },
        "same_sampled_point_leader_rate": float(np.mean(original_leader == alternative_leader)),
        "release_point_leader_rate_under_alternative": float(
            np.mean(alternative_leader == release_leader_index)
        ),
        "release_point_leader_rate_under_original": float(
            np.mean(original_leader == release_leader_index)
        ),
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise PublicScorerSensitivityError("cannot serialize an empty CSV")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def _companion(name: str, payload: bytes, records: int) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(payload),
        "format": "jsonl" if name.endswith(".jsonl") else "csv",
        "records": records,
        "newline_count": payload.count(b"\n"),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _task_map_agreement(
    original: Mapping[str, int], alternative: Mapping[str, int]
) -> dict[str, Any]:
    if set(original) != set(SELECTION_KEYS) or set(alternative) != set(SELECTION_KEYS):
        raise PublicScorerSensitivityError("score-map action keys differ")
    left = np.asarray([original[key] for key in SELECTION_KEYS], dtype=float)
    right = np.asarray([alternative[key] for key in SELECTION_KEYS], dtype=float)
    original_optima = {key for key in SELECTION_KEYS if original[key] == int(np.max(left))}
    alternative_optima = {key for key in SELECTION_KEYS if alternative[key] == int(np.max(right))}
    return {
        "rank_spearman": _spearman(left, right),
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "optimum_set_agreement": original_optima == alternative_optima,
        "original_optimum_count": len(original_optima),
        "alternative_optimum_count": len(alternative_optima),
        "original_chance_score": float(np.mean(left) / 100.0),
        "alternative_chance_score": float(np.mean(right) / 100.0),
    }


def _summarize_map_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rho = np.asarray([float(row["rank_spearman"]) for row in rows])
    pearson = np.asarray([float(row["pearson"]) for row in rows])
    return {
        "tasks": len(rows),
        "rank_spearman": {
            "median": float(np.median(rho)),
            "interquartile_interval": list(map(float, np.quantile(rho, [0.25, 0.75]))),
            "percentile_95_interval": list(map(float, np.quantile(rho, [0.025, 0.975]))),
        },
        "pearson": {
            "mean": float(np.mean(pearson)),
            "median": float(np.median(pearson)),
        },
        "exact_optimum_set_agreement_rate": float(
            np.mean([bool(row["optimum_set_agreement"]) for row in rows])
        ),
        "mean_absolute_chance_score_shift": float(
            np.mean(
                [
                    abs(
                        float(row["alternative_chance_score"]) - float(row["original_chance_score"])
                    )
                    for row in rows
                ]
            )
        ),
    }


def build_analysis(
    *,
    root: Path,
    plan_path: Path,
    ingredient_tags_path: Path,
    checkpoint_root: Path | None,
    allow_network_download: bool,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = root.resolve()
    plan_path = plan_path.resolve()
    ingredient_tags_path = ingredient_tags_path.resolve()
    plan = _load_json(plan_path)
    if not verify_plan(plan):
        raise PublicScorerSensitivityError("complete-core analysis plan failed verification")
    graph = source_graph(root)
    task_ids_1, task_ids_2 = selected_task_ids(plan)
    allowed = {DEEPSEEK_PRO_MODEL_ID: frozenset({"endpoint_sha256"})}
    panel_1 = load_complete_common_core(
        panel="primary",
        plan=graph.panel_1_plan,
        taskset=graph.panel_1_taskset,
        repeat_panel=graph.panel_1_repeat,
        task_ids=task_ids_1,
        model_sources=graph.panel_1_model_sources,
        allowed_source_roster_differences=allowed,
    )
    panel_2 = load_complete_common_core(
        panel="primary",
        plan=graph.panel_2_plan,
        taskset=graph.panel_2_taskset,
        repeat_panel=graph.panel_2_repeat,
        task_ids=task_ids_2,
        model_sources=graph.panel_2_model_sources,
        allowed_source_roster_differences=allowed,
    )
    if panel_1.model_ids != panel_2.model_ids:
        raise PublicScorerSensitivityError("panel model rosters differ")
    task_by_panel = {
        "panel_1": {str(task["task_id"]): task for task in graph.panel_1_taskset["tasks"]},
        "panel_2": {str(task["task_id"]): task for task in graph.panel_2_taskset["tasks"]},
    }
    tasks = [
        *(task_by_panel["panel_1"][task_id] for task_id in task_ids_1),
        *(task_by_panel["panel_2"][task_id] for task_id in task_ids_2),
    ]
    panels = ["panel_1"] * len(task_ids_1) + ["panel_2"] * len(task_ids_2)
    strata = [f"{panel}:{task['family']}" for panel, task in zip(panels, tasks, strict=True)]
    original_matrix = np.concatenate([panel_1.scores, panel_2.scores], axis=1)
    selections = np.asarray(
        [
            (*panel_1.selections[index], *panel_2.selections[index])
            for index in range(len(panel_1.model_ids))
        ],
        dtype=object,
    )
    if original_matrix.shape != (27, 534) or selections.shape != (27, 534):
        raise PublicScorerSensitivityError("complete-core response matrix shape differs")
    if any(selection not in SELECTION_KEYS for selection in selections.flat):
        raise PublicScorerSensitivityError("complete-core selection lies outside the action set")

    tags = _load_ingredient_tags(ingredient_tags_path)
    required_ingredients = {
        name
        for task in tasks
        for name in (str(task["anchor_ingredient"]), *map(str, task["choices"].values()))
    }
    original_maps = [
        {key: int(task["selection_scores_bps"][key]) for key in SELECTION_KEYS} for task in tasks
    ]
    original_scores = _balanced_scores(original_matrix, strata)
    original_ranks = _point_ranks(original_scores)
    release_leader_index = int(np.argmax(original_scores))
    bootstrap_draws = _bootstrap_indices(strata, bootstrap_replicates, BOOTSTRAP_SEED)
    original_bootstrap = _bootstrap_scores(original_matrix, bootstrap_draws)

    score_map_rows: list[dict[str, Any]] = []
    task_agreement_rows: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    checkpoint_results: list[dict[str, Any]] = []
    checkpoint_inputs: list[dict[str, Any]] = []
    for name, checkpoint in CHECKPOINTS.items():
        directory = _discover_checkpoint(
            name,
            checkpoint_root,
            allow_network_download=allow_network_download,
        )
        vocab, embeddings, config = _load_checkpoint(name, directory)
        missing = sorted(required_ingredients - set(vocab))
        if missing:
            raise PublicScorerSensitivityError(
                f"{checkpoint['repo_id']} misses {len(missing)} benchmark ingredients"
            )
        maps = [
            _public_score_map(task, vocab=vocab, embeddings=embeddings, tags=tags) for task in tasks
        ]
        alternative_matrix = np.empty_like(original_matrix)
        for model_index in range(len(panel_1.model_ids)):
            for task_index, selection in enumerate(selections[model_index]):
                alternative_matrix[model_index, task_index] = (
                    maps[task_index][str(selection)] / 100.0
                )
        alternative_scores = _balanced_scores(alternative_matrix, strata)
        alternative_ranks = _point_ranks(alternative_scores)
        alternative_bootstrap = _bootstrap_scores(alternative_matrix, bootstrap_draws)

        agreements: list[dict[str, Any]] = []
        for panel, task, original_map, alternative_map in zip(
            panels, tasks, original_maps, maps, strict=True
        ):
            agreement = _task_map_agreement(original_map, alternative_map)
            agreements.append(agreement)
            task_agreement_rows.append(
                {
                    "checkpoint": name,
                    "repo_id": checkpoint["repo_id"],
                    "revision": checkpoint["revision"],
                    "release_panel": panel,
                    "task_id": task["task_id"],
                    "family": task["family"],
                    **agreement,
                }
            )
            score_map_rows.append(
                {
                    "checkpoint": name,
                    "release_panel": panel,
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "selection_scores_bps": alternative_map,
                }
            )
        for model_index, model_id in enumerate(panel_1.model_ids):
            family_scores = {
                family: float(
                    alternative_matrix[
                        model_index,
                        np.asarray([task["family"] == family for task in tasks], dtype=bool),
                    ].mean()
                )
                for family in FAMILIES
            }
            leaderboard_rows.append(
                {
                    "checkpoint": name,
                    "repo_id": checkpoint["repo_id"],
                    "revision": checkpoint["revision"],
                    "model_id": model_id,
                    "model_name": panel_1.model_names[model_index],
                    "score": float(alternative_scores[model_index]),
                    "point_rank": int(alternative_ranks[model_index]),
                    "original_score": float(original_scores[model_index]),
                    "original_point_rank": int(original_ranks[model_index]),
                    **{f"{family}_score": family_scores[family] for family in FAMILIES},
                }
            )
        checkpoint_results.append(
            {
                "checkpoint": name,
                "label": checkpoint["label"],
                "repo_id": checkpoint["repo_id"],
                "revision": checkpoint["revision"],
                "ingredient_coverage": {
                    "required": len(required_ingredients),
                    "covered": len(required_ingredients),
                    "missing": 0,
                },
                "task_score_map_agreement": _summarize_map_agreement(agreements),
                "leaderboard_agreement": {
                    "rank_spearman": _spearman(original_scores, alternative_scores),
                    "pair_order_agreement": _pair_order_agreement(
                        original_scores, alternative_scores
                    ),
                    "mean_absolute_score_shift": float(
                        np.mean(np.abs(alternative_scores - original_scores))
                    ),
                    "maximum_absolute_score_shift": float(
                        np.max(np.abs(alternative_scores - original_scores))
                    ),
                    "original_point_leader_model_id": panel_1.model_ids[release_leader_index],
                    "alternative_point_leader_model_id": panel_1.model_ids[
                        int(np.argmax(alternative_scores))
                    ],
                    "point_leader_preserved": bool(
                        int(np.argmax(alternative_scores)) == release_leader_index
                    ),
                },
                "stratified_anchor_bootstrap": _bootstrap_agreement(
                    original_bootstrap,
                    alternative_bootstrap,
                    release_leader_index=release_leader_index,
                ),
            }
        )
        checkpoint_inputs.append(
            {
                "checkpoint": name,
                "label": checkpoint["label"],
                "repo_id": checkpoint["repo_id"],
                "revision": checkpoint["revision"],
                "embeddings_sha256": checkpoint["embeddings_sha256"],
                "vocab_sha256": VOCAB_SHA256,
                "config_sha256": checkpoint["config_sha256"],
                "schema": config["schema"],
                "vocab_size": config["vocab_size"],
                "embedding_dimensions": config["d_model"],
            }
        )

    companion_payloads = {
        "complete-core-public-scorer-score-maps.jsonl": _jsonl_bytes(score_map_rows),
        "complete-core-public-scorer-task-agreement.csv": _csv_bytes(task_agreement_rows),
        "complete-core-public-scorer-leaderboard.csv": _csv_bytes(leaderboard_rows),
    }
    companion_manifest = [
        _companion(
            name,
            payload,
            len(score_map_rows)
            if name.endswith(".jsonl")
            else (len(task_agreement_rows) if "task-agreement" in name else len(leaderboard_rows)),
        )
        for name, payload in companion_payloads.items()
    ]
    analysis: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "post_collection_fixed_task_public_scorer_sensitivity",
        "analysis_timing": {
            "prespecified_primary_analysis": False,
            "posthoc_sensitivity": True,
            "model_provider_calls_added": 0,
            "model_responses_changed": False,
            "task_prompts_or_candidate_sets_changed": False,
        },
        "inputs": {
            "builder_physical_sha256": _file_sha256(Path(__file__).resolve()),
            "complete_core_plan_semantic_sha256": plan["artifact_sha256"],
            "complete_core_plan_physical_sha256": _file_sha256(plan_path),
            "panel_1_taskset_semantic_sha256": graph.panel_1_taskset["artifact_sha256"],
            "panel_1_taskset_physical_sha256": _file_sha256(graph.panel_1_taskset_path),
            "panel_2_taskset_semantic_sha256": graph.panel_2_taskset["artifact_sha256"],
            "panel_2_taskset_physical_sha256": _file_sha256(graph.panel_2_taskset_path),
            "ingredient_tags_sha256": INGREDIENT_TAGS_SHA256,
            "ingredient_tags_source_commit": PUBLIC_EPICURE_MCP_COMMIT,
            "ingredient_tags_source_url": INGREDIENT_TAGS_URL,
            "panel_1_response_artifact_set_sha256": _sha256(
                list(panel_1.response_artifact_sha256s)
            ),
            "panel_2_response_artifact_set_sha256": _sha256(
                list(panel_2.response_artifact_sha256s)
            ),
            "public_checkpoints": checkpoint_inputs,
        },
        "design": {
            "models": len(panel_1.model_ids),
            "tasks": len(tasks),
            "model_task_cells": int(original_matrix.size),
            "panels": 2,
            "families": list(FAMILIES),
            "tasks_per_panel_family": 89,
            "actions_per_task": len(SELECTION_KEYS),
            "public_checkpoint_count": len(CHECKPOINTS),
            "required_ingredient_count": len(required_ingredients),
            "missing_public_checkpoint_ingredients": 0,
            "score_rule": (
                "recompute all 56 within-task min-max-normalized rewards from each public "
                "embedding while preserving the released action set, constraint metadata, "
                "and observed model selection"
            ),
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "seed": BOOTSTRAP_SEED,
                "resampling_unit": "task anchor",
                "stratification": "panel by family",
                "interval": "descriptive percentile 95 percent",
            },
        },
        "original_point_leader_model_id": panel_1.model_ids[release_leader_index],
        "checkpoint_results": checkpoint_results,
        "all_public_checkpoint_point_estimates_preserve_release_leader": all(
            bool(row["leaderboard_agreement"]["point_leader_preserved"])
            for row in checkpoint_results
        ),
        "minimum_model_rank_spearman": min(
            float(row["leaderboard_agreement"]["rank_spearman"]) for row in checkpoint_results
        ),
        "maximum_model_rank_spearman": max(
            float(row["leaderboard_agreement"]["rank_spearman"]) for row in checkpoint_results
        ),
        "minimum_pair_order_agreement": min(
            float(row["leaderboard_agreement"]["pair_order_agreement"])
            for row in checkpoint_results
        ),
        "maximum_pair_order_agreement": max(
            float(row["leaderboard_agreement"]["pair_order_agreement"])
            for row in checkpoint_results
        ),
        "companion_files": companion_manifest,
        "claim_boundary": (
            "This post-hoc analysis tests reward-map dependence conditional on the released "
            "534 prompts, candidate sets, constraints, and model selections. The original "
            "runtime selected those candidate sets. The analysis does not validate the scorer "
            "against human culinary judgments, recover the original runtime's training lineage, "
            "or show what a newly compiled public-checkpoint benchmark would rank."
        ),
    }
    analysis["artifact_sha256"] = _sha256(analysis)
    return analysis, companion_payloads


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _write_tex(directory: Path, analysis: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    results = analysis["checkpoint_results"]
    lines = [
        "% Generated by build_public_scorer_sensitivity_assets.py; do not edit.",
        rf"\newcommand{{\FBPublicScorerMinRho}}{{{float(analysis['minimum_model_rank_spearman']):.3f}}}",
        rf"\newcommand{{\FBPublicScorerMaxRho}}{{{float(analysis['maximum_model_rank_spearman']):.3f}}}",
        rf"\newcommand{{\FBPublicScorerMinPair}}"
        rf"{{{100 * float(analysis['minimum_pair_order_agreement']):.1f}\%}}",
        rf"\newcommand{{\FBPublicScorerMaxPair}}"
        rf"{{{100 * float(analysis['maximum_pair_order_agreement']):.1f}\%}}",
        rf"\newcommand{{\FBPublicScorerReplicates}}{{{int(results[0]['stratified_anchor_bootstrap']['replicates']):,}}}",
        rf"\newcommand{{\FBPublicScorerArtifact}}{{{analysis['artifact_sha256']}}}",
    ]
    (directory / "complete-core-public-scorer-macros.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    table = [
        r"\begin{tabular}{@{}l r r r l@{}}",
        r"\toprule",
        r"Reward map & Task-map $\rho$ & Model-rank $\rho$ & Pair order & Point leader \\",
        r"\midrule",
    ]
    for row in results:
        maps = row["task_score_map_agreement"]
        leaderboard = row["leaderboard_agreement"]
        leader = str(leaderboard["alternative_point_leader_model_id"])
        leader_label = "Grok 4.6" if leader == "x-ai/grok-4.6" else leader.rsplit("/", 1)[-1]
        table.append(
            f"{_latex_escape(str(row['label']))} & "
            f"{float(maps['rank_spearman']['median']):.3f} & "
            f"{float(leaderboard['rank_spearman']):.3f} & "
            f"{100 * float(leaderboard['pair_order_agreement']):.1f}\\% & "
            f"{_latex_escape(leader_label)} \\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (directory / "complete-core-public-scorer-table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )


def _plot(directory: Path, analysis: Mapping[str, Any], leaderboard_payload: bytes) -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1787714400")
    rows = list(csv.DictReader(io.StringIO(leaderboard_payload.decode())))
    results = analysis["checkpoint_results"]
    checkpoint_order = [str(row["checkpoint"]) for row in results]
    labels = {str(row["checkpoint"]): str(row["label"]).replace("Epicure-", "") for row in results}
    original_ranks = {
        str(row["model_id"]): int(row["original_point_rank"])
        for row in rows
        if row["checkpoint"] == checkpoint_order[0]
    }
    top_ids = [
        model_id
        for model_id, rank in sorted(original_ranks.items(), key=lambda item: item[1])
        if rank <= 10
    ]
    short = {
        "x-ai/grok-4.6": "Grok 4.6",
        "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
        "openai/gpt-5.6-sol-pro": "GPT-5.6 Sol",
        "meta/muse-spark-1.2": "Muse Spark 1.2",
        "openai/gpt-5.6-terra-pro": "GPT-5.6 Terra",
        "anthropic/claude-fable-5": "Claude Fable 5",
        "openai/gpt-5.6-luna-pro": "GPT-5.6 Luna",
        "anthropic/claude-opus-5": "Claude Opus 5",
        "qwen/qwen3.8-2.4t-a95b": "Qwen 3.8 A95B",
        "moonshotai/kimi-k3": "Kimi K3",
    }
    rank_lookup = {
        (str(row["checkpoint"]), str(row["model_id"])): int(row["point_rank"]) for row in rows
    }
    highlight = {
        "x-ai/grok-4.6": "#A83D34",
        "google/gemini-3.1-pro-preview": "#356C64",
    }
    neutral = ["#4E6173", "#6A7478", "#7D8585", "#8C8175", "#77717C", "#68746B"]
    paper, ink, muted, rule = "#F6F7F5", "#171A18", "#68706C", "#DDE1DE"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": paper,
            "figure.facecolor": paper,
            "axes.edgecolor": rule,
            "axes.labelcolor": muted,
            "xtick.color": ink,
            "ytick.color": ink,
            "axes.titlecolor": ink,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), constrained_layout=True)
    x_labels = ["Original", *[labels[name] for name in checkpoint_order]]
    x = np.arange(len(x_labels))
    for index, model_id in enumerate(top_ids):
        y = [
            original_ranks[model_id],
            *[rank_lookup[(name, model_id)] for name in checkpoint_order],
        ]
        color = highlight.get(model_id, neutral[index % len(neutral)])
        width = 1.8 if model_id in highlight else 1.1
        alpha = 0.95 if model_id in highlight else 0.72
        axes[0].plot(
            x,
            y,
            marker="o",
            linewidth=width,
            markersize=4,
            color=color,
            alpha=alpha,
        )
        axes[0].text(
            x[-1] + 0.07,
            y[-1],
            short.get(model_id, model_id.rsplit("/", 1)[-1]),
            fontsize=7,
            va="center",
        )
    axes[0].set_xticks(x, x_labels)
    axes[0].set_ylim(15.5, 0.5)
    axes[0].set_yticks(range(1, 16, 2))
    axes[0].set_ylabel("Point rank (lower is better)")
    axes[0].set_xlim(-0.1, len(x_labels) - 0.45)
    axes[0].grid(axis="y", color=rule, linewidth=0.8)
    axes[0].set_title("Top-ten rank trajectories", loc="left", fontweight="bold")

    names = [str(row["label"]).replace("Epicure-", "") for row in results]
    rho = [float(row["leaderboard_agreement"]["rank_spearman"]) for row in results]
    pair = [float(row["leaderboard_agreement"]["pair_order_agreement"]) for row in results]
    y = np.arange(len(names))
    axes[1].scatter(rho, y - 0.10, color="#A83D34", s=52, label="Model-rank correlation")
    axes[1].scatter(pair, y + 0.10, color="#356C64", s=52, label="Pair-order agreement")
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.82, 1.005)
    axes[1].grid(axis="x", color=rule, linewidth=0.8)
    axes[1].set_xlabel("Agreement with original reward map")
    axes[1].set_title("Global order under public checkpoints", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    figure.suptitle(
        "Is the leaderboard tied to one Epicure embedding?",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=ink,
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / "complete-core-public-scorer-sensitivity"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "FlavourBench", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _write_outputs(
    *,
    analysis: Mapping[str, Any],
    companions: Mapping[str, bytes],
    generated_directory: Path,
    figure_directory: Path,
    dataset_directory: Path | None,
    dataset_figure_directory: Path | None,
) -> None:
    payload = json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    generated_directory.mkdir(parents=True, exist_ok=True)
    (generated_directory / "complete-core-public-scorer-sensitivity.json").write_text(
        payload, encoding="utf-8"
    )
    for name, value in companions.items():
        (generated_directory / name).write_bytes(value)
    _write_tex(generated_directory, analysis)
    leaderboard = companions["complete-core-public-scorer-leaderboard.csv"]
    _plot(figure_directory, analysis, leaderboard)
    if dataset_directory is not None:
        dataset_directory.mkdir(parents=True, exist_ok=True)
        (dataset_directory / "complete-core-public-scorer-sensitivity.json").write_text(
            payload, encoding="utf-8"
        )
        for name, value in companions.items():
            (dataset_directory / name).write_bytes(value)
    if dataset_figure_directory is not None:
        _plot(dataset_figure_directory, analysis, leaderboard)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--ingredient-tags", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--allow-network-download",
        action="store_true",
        help=(
            "download the exact hash-pinned public checkpoint and ingredient-tag inputs "
            "when they are absent from local caches"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--generated-directory", type=Path, required=True)
    parser.add_argument("--figure-directory", type=Path, required=True)
    parser.add_argument("--dataset-directory", type=Path)
    parser.add_argument("--dataset-figure-directory", type=Path)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        raise PublicScorerSensitivityError("at least 1,000 bootstrap replicates are required")
    root = args.root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    ingredient_tags_path = _resolve_ingredient_tags(
        root=root,
        explicit_path=args.ingredient_tags,
        allow_network_download=args.allow_network_download,
    )
    analysis, companions = build_analysis(
        root=root,
        plan_path=plan_path,
        ingredient_tags_path=ingredient_tags_path,
        checkpoint_root=args.checkpoint_root,
        allow_network_download=args.allow_network_download,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    _write_outputs(
        analysis=analysis,
        companions=companions,
        generated_directory=args.generated_directory,
        figure_directory=args.figure_directory,
        dataset_directory=args.dataset_directory,
        dataset_figure_directory=args.dataset_figure_directory,
    )
    print(
        "built public-scorer sensitivity analysis "
        f"{analysis['artifact_sha256']} from 3 immutable checkpoints and 14,418 selections"
    )


if __name__ == "__main__":
    main()
