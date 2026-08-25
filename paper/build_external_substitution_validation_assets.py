#!/usr/bin/env python3
"""Validate public Epicure substitution geometry on Recipe1MSubs labels.

The analysis plan is a separate, hash-bound input. Raw Recipe1MSubs files are
downloaded from Meta's public host when requested, verified byte-for-byte, and
never redistributed. Only aggregate statistics and a figure are emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import pickletools
import shutil
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from build_public_scorer_sensitivity_assets import (  # noqa: E402
    CHECKPOINTS,
    INGREDIENT_TAGS_SHA256,
    INGREDIENT_TAGS_URL,
    PUBLIC_EPICURE_MCP_COMMIT,
    VOCAB_SHA256,
    _discover_checkpoint,
    _file_sha256,
    _load_checkpoint,
    _resolve_ingredient_tags,
)

SCHEMA_VERSION = "flavourbench-external-substitution-validation-v1"
PROTOCOL_PATH = Path("paper/protocols/external_substitution_validation_v1.json")
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20260826
MINIMUM_GROUP_CANDIDATES = 20
PICKLE_UNSAFE_OPCODES = frozenset(
    {
        "GLOBAL",
        "STACK_GLOBAL",
        "REDUCE",
        "BUILD",
        "OBJ",
        "INST",
        "NEWOBJ",
        "NEWOBJ_EX",
        "EXT1",
        "EXT2",
        "EXT4",
        "PERSID",
        "BINPERSID",
    }
)
RECIPE1MSUBS: dict[str, dict[str, Any]] = {
    "train": {
        "filename": "train_comments_subs.pkl",
        "url": "https://dl.fbaipublicfiles.com/gismo/train_comments_subs.pkl",
        "sha256": "b4d58e6c6c6747671e531812393397acf75c02d2a9ce00c2edcc7794bf567c81",
        "records": 49_044,
    },
    "test": {
        "filename": "test_comments_subs.pkl",
        "url": "https://dl.fbaipublicfiles.com/gismo/test_comments_subs.pkl",
        "sha256": "0758382cae59697958dd2a5e509c15f8d0417ea1a9e079fd75792779f1ce6174",
        "records": 10_747,
    },
}


class ExternalSubstitutionValidationError(RuntimeError):
    """A bound input or derived validation result is inconsistent."""


class _PrimitiveUnpickler(pickle.Unpickler):
    """Reject class construction and persistent references in external pickles."""

    def find_class(self, module: str, name: str) -> Any:
        raise ExternalSubstitutionValidationError(
            f"external pickle attempted class resolution: {module}.{name}"
        )

    def persistent_load(self, pid: object) -> Any:
        raise ExternalSubstitutionValidationError(
            f"external pickle attempted persistent loading: {pid!r}"
        )


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExternalSubstitutionValidationError(f"JSON input is not an object: {path}")
    return value


def _resolve_recipe1msubs(
    split: str,
    *,
    explicit_directory: Path | None,
    allow_network_download: bool,
) -> Path:
    record = RECIPE1MSUBS[split]
    filename = str(record["filename"])
    if explicit_directory is not None:
        candidate = explicit_directory.expanduser().resolve() / filename
        if not candidate.is_file():
            raise ExternalSubstitutionValidationError(
                f"explicit Recipe1MSubs input is missing: {candidate}"
            )
    else:
        candidate = (
            Path.home()
            / ".cache"
            / "flavourbench"
            / "recipe1msubs"
            / str(record["sha256"])
            / filename
        )
    if candidate.is_file() and _file_sha256(candidate) == record["sha256"]:
        return candidate
    if explicit_directory is not None:
        raise ExternalSubstitutionValidationError(
            f"explicit Recipe1MSubs {split} hash differs from {record['sha256']}"
        )
    if not allow_network_download:
        raise ExternalSubstitutionValidationError(
            f"Recipe1MSubs {split} is unavailable; pass --recipe1msubs-directory "
            "or opt in with --allow-network-download"
        )
    request = urllib.request.Request(
        str(record["url"]),
        headers={"User-Agent": "FlavourBench-external-validation/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = response.read()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != record["sha256"]:
        raise ExternalSubstitutionValidationError(
            f"downloaded Recipe1MSubs {split} hash differs: {observed}"
        )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(payload)
    return candidate


def _restricted_pickle_load(handle: BinaryIO) -> object:
    return _PrimitiveUnpickler(handle).load()


def _load_recipe1msubs(path: Path, *, split: str) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    observed_unsafe = [
        (position, operation.name, argument)
        for operation, argument, position in pickletools.genops(payload)
        if operation.name in PICKLE_UNSAFE_OPCODES
    ]
    if observed_unsafe:
        raise ExternalSubstitutionValidationError(
            f"Recipe1MSubs {split} contains unsafe pickle opcodes: {observed_unsafe[:3]}"
        )
    with io.BytesIO(payload) as handle:
        value = _restricted_pickle_load(handle)
    expected = int(RECIPE1MSUBS[split]["records"])
    if not isinstance(value, list) or len(value) != expected:
        observed = len(value) if isinstance(value, list) else type(value)
        raise ExternalSubstitutionValidationError(
            f"Recipe1MSubs {split} record count differs: {observed}"
        )
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != {"id", "ingredients", "subs"}:
            raise ExternalSubstitutionValidationError(
                f"Recipe1MSubs {split} row {index} has unexpected fields"
            )
        recipe_id = row["id"]
        ingredients = row["ingredients"]
        substitution = row["subs"]
        if (
            not isinstance(recipe_id, str)
            or not isinstance(ingredients, list)
            or not isinstance(substitution, tuple)
            or len(substitution) != 2
            or not all(isinstance(item, str) and item for item in substitution)
            or not all(
                isinstance(group, list)
                and group
                and all(isinstance(item, str) and item for item in group)
                for group in ingredients
            )
        ):
            raise ExternalSubstitutionValidationError(
                f"Recipe1MSubs {split} row {index} is malformed"
            )
        rows.append(row)
    return rows


def _load_food_groups(path: Path) -> dict[str, str]:
    if _file_sha256(path) != INGREDIENT_TAGS_SHA256:
        raise ExternalSubstitutionValidationError("ingredient-tags hash differs")
    groups: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = str(row["name"])
            group = str(row["food_group"])
            if not name or not group or name in groups:
                raise ExternalSubstitutionValidationError("ingredient-tags rows are malformed")
            groups[name] = group
    if len(groups) != 1790:
        raise ExternalSubstitutionValidationError("ingredient-tags cardinality differs")
    return groups


def _average_rank(scores: np.ndarray, target_index: int) -> float:
    target = float(scores[target_index])
    return float(1 + np.sum(scores > target) + 0.5 * (np.sum(scores == target) - 1))


def _rank_metrics(ranks: Sequence[float], candidate_counts: Sequence[int]) -> dict[str, float]:
    rank_values = np.asarray(ranks, dtype=float)
    counts = np.asarray(candidate_counts, dtype=int)
    if len(rank_values) == 0 or len(rank_values) != len(counts):
        raise ExternalSubstitutionValidationError("rank metric inputs are empty or misaligned")
    return {
        "mean_reciprocal_rank": float(np.mean(1.0 / rank_values)),
        "hit_at_1": float(np.mean(rank_values <= 1)),
        "hit_at_3": float(np.mean(rank_values <= 3)),
        "hit_at_10": float(np.mean(rank_values <= 10)),
        "median_rank": float(np.median(rank_values)),
        "mean_rank": float(np.mean(rank_values)),
        "analytic_chance_mrr": float(
            np.mean(
                [
                    sum(1.0 / rank for rank in range(1, int(count) + 1)) / int(count)
                    for count in counts
                ]
            )
        ),
        "analytic_chance_hit_at_1": float(np.mean(1.0 / counts)),
        "analytic_chance_hit_at_3": float(np.mean(np.minimum(3, counts) / counts)),
        "analytic_chance_hit_at_10": float(np.mean(np.minimum(10, counts) / counts)),
    }


def _cluster_summary(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row["source"])].append(float(row["within_group_percentile"]))
    sources = sorted(grouped)
    source_means = np.asarray([np.mean(grouped[source]) for source in sources], dtype=float)
    estimate = float(np.mean(source_means))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=float)
    null_bootstrap = np.empty(bootstrap_replicates, dtype=float)
    residuals = source_means - estimate
    chunk = 2_000
    for start in range(0, bootstrap_replicates, chunk):
        stop = min(start + chunk, bootstrap_replicates)
        indices = rng.integers(0, len(sources), size=(stop - start, len(sources)))
        bootstrap[start:stop] = source_means[indices].mean(axis=1)
        null_bootstrap[start:stop] = residuals[indices].mean(axis=1)
    observed_difference = estimate - 0.5
    p_value = float(
        (1 + np.sum(null_bootstrap >= observed_difference)) / (bootstrap_replicates + 1)
    )
    return {
        "unique_sources": len(sources),
        "unique_pairs": len(pair_rows),
        "equal_source_mean_rank_percentile": estimate,
        "percentile_95_interval": list(map(float, np.quantile(bootstrap, [0.025, 0.975]))),
        "chance_value": 0.5,
        "difference_from_chance": observed_difference,
        "one_sided_null_centered_cluster_bootstrap_p": p_value,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
    }


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, (total - index) * float(p_values[key]))
        adjusted[key] = min(1.0, running)
    return adjusted


def _evaluate_checkpoint(
    *,
    checkpoint: str,
    vocab: Mapping[str, int],
    embeddings: np.ndarray,
    food_groups: Mapping[str, str],
    pairs: Sequence[tuple[str, str]],
    train_pairs: frozenset[tuple[str, str]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(vocab) != set(food_groups):
        raise ExternalSubstitutionValidationError("checkpoint and food-group vocabularies differ")
    names = [name for name, _ in sorted(vocab.items(), key=lambda item: item[1])]
    group_names: dict[str, list[str]] = defaultdict(list)
    for name in names:
        group_names[food_groups[name]].append(name)

    rows: list[dict[str, Any]] = []
    for source, target in pairs:
        target_group = food_groups[target]
        within_names = [name for name in group_names[target_group] if name != source]
        if len(within_names) < MINIMUM_GROUP_CANDIDATES:
            continue
        source_vector = embeddings[int(vocab[source])]
        within_indices = np.asarray([int(vocab[name]) for name in within_names], dtype=int)
        within_scores = embeddings[within_indices] @ source_vector
        target_within_index = within_names.index(target)
        within_rank = _average_rank(within_scores, target_within_index)
        within_percentile = float((len(within_names) - within_rank) / (len(within_names) - 1))

        full_scores = embeddings @ source_vector
        full_scores = full_scores.copy()
        full_scores[int(vocab[source])] = -np.inf
        full_rank = _average_rank(full_scores, int(vocab[target]))
        rows.append(
            {
                "checkpoint": checkpoint,
                "source": source,
                "target": target,
                "target_food_group": target_group,
                "pair_seen_in_recipe1msubs_train": (source, target) in train_pairs,
                "within_group_candidates": len(within_names),
                "within_group_rank": within_rank,
                "within_group_percentile": within_percentile,
                "full_vocabulary_candidates": len(vocab) - 1,
                "full_vocabulary_rank": full_rank,
            }
        )
    if len(rows) != len(pairs):
        raise ExternalSubstitutionValidationError(
            f"food-group eligibility unexpectedly removed pairs: {len(rows)} of {len(pairs)}"
        )

    novel_rows = [row for row in rows if not row["pair_seen_in_recipe1msubs_train"]]
    all_cluster = _cluster_summary(
        rows,
        bootstrap_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    novel_cluster = _cluster_summary(
        novel_rows,
        bootstrap_replicates=bootstrap_replicates,
        seed=bootstrap_seed + 1,
    )
    full_metrics = _rank_metrics(
        [float(row["full_vocabulary_rank"]) for row in rows],
        [int(row["full_vocabulary_candidates"]) for row in rows],
    )
    group_counts = np.asarray([int(row["within_group_candidates"]) for row in rows])
    return (
        {
            "checkpoint": checkpoint,
            "label": CHECKPOINTS[checkpoint]["label"],
            "primary_all_unique_pairs": all_cluster,
            "sensitivity_novel_unique_pairs": novel_cluster,
            "full_vocabulary_retrieval": full_metrics,
            "within_group_candidate_count": {
                "minimum": int(np.min(group_counts)),
                "median": float(np.median(group_counts)),
                "maximum": int(np.max(group_counts)),
            },
        },
        rows,
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ExternalSubstitutionValidationError("cannot serialize an empty CSV")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def _companion(name: str, payload: bytes, records: int, *, format_name: str) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(payload),
        "format": format_name,
        "records": records,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _format_interval(result: Mapping[str, Any]) -> str:
    low, high = result["percentile_95_interval"]
    return (
        f"{float(result['equal_source_mean_rank_percentile']):.3f} "
        f"[{float(low):.3f}, {float(high):.3f}]"
    )


def _render_table(results: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Checkpoint & Percentile [95\% CI] & Novel pairs & Hit@10 \\",
        r"\midrule",
    ]
    for row in results:
        primary = row["primary_all_unique_pairs"]
        novel = row["sensitivity_novel_unique_pairs"]
        retrieval = row["full_vocabulary_retrieval"]
        lines.append(
            f"{str(row['label']).removeprefix('Epicure-')} & {_format_interval(primary)} & "
            f"{float(novel['equal_source_mean_rank_percentile']):.3f} & "
            f"{float(retrieval['hit_at_10']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines).encode()


def _render_macros(coverage: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> bytes:
    best = max(
        results,
        key=lambda row: float(row["primary_all_unique_pairs"]["equal_source_mean_rank_percentile"]),
    )
    best_percentile = 100 * float(
        best["primary_all_unique_pairs"]["equal_source_mean_rank_percentile"]
    )
    lines = [
        "% Auto-generated by build_external_substitution_validation_assets.py.",
        rf"\newcommand{{\FBExternalSubRawTest}}{{{int(coverage['raw_test_records']):,}}}",
        rf"\newcommand{{\FBExternalSubMappedEvents}}{{{int(coverage['mapped_test_events']):,}}}",
        rf"\newcommand{{\FBExternalSubPairs}}{{{int(coverage['unique_mapped_test_pairs']):,}}}",
        rf"\newcommand{{\FBExternalSubNovelPairs}}{{{int(coverage['novel_unique_pairs']):,}}}",
        rf"\newcommand{{\FBExternalSubSources}}{{{int(coverage['unique_source_ingredients']):,}}}",
        rf"\newcommand{{\FBExternalSubBestLabel}}{{{best['label']}}}",
        rf"\newcommand{{\FBExternalSubBestPercentile}}{{{best_percentile:.1f}\%}}",
        "",
    ]
    return "\n".join(lines).encode()


def _render_figure(results: Sequence[Mapping[str, Any]]) -> tuple[bytes, bytes]:
    labels = [str(row["label"]) for row in results]
    estimates = np.asarray(
        [
            float(row["primary_all_unique_pairs"]["equal_source_mean_rank_percentile"])
            for row in results
        ]
    )
    intervals = np.asarray(
        [row["primary_all_unique_pairs"]["percentile_95_interval"] for row in results],
        dtype=float,
    )
    novel = np.asarray(
        [
            float(row["sensitivity_novel_unique_pairs"]["equal_source_mean_rank_percentile"])
            for row in results
        ]
    )
    positions = np.arange(len(labels))[::-1]
    figure, axis = plt.subplots(figsize=(6.9, 2.55))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.axvline(0.5, color="#8f9692", linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates)),
        fmt="o",
        markersize=6,
        color="#202421",
        ecolor="#202421",
        elinewidth=1.5,
        capsize=3,
        label="All mapped test pairs",
        zorder=3,
    )
    axis.scatter(
        novel,
        positions - 0.16,
        s=34,
        facecolors="white",
        edgecolors="#b34032",
        linewidths=1.4,
        label="Pairs unseen in Recipe1MSubs train",
        zorder=4,
    )
    axis.set_yticks(positions, labels)
    axis.set_xlim(0.45, 0.85)
    axis.set_xticks(np.arange(0.5, 0.851, 0.05))
    axis.set_xlabel("Equal-source within-food-group rank percentile")
    axis.grid(axis="x", color="#e2e5e2", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    figure.tight_layout()

    pdf_handle = io.BytesIO()
    figure.savefig(
        pdf_handle,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "External substitution validation",
            "Author": "FlavourBench",
            "Creator": "FlavourBench",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    png_handle = io.BytesIO()
    figure.savefig(
        png_handle,
        format="png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "FlavourBench"},
    )
    plt.close(figure)
    return pdf_handle.getvalue(), png_handle.getvalue()


def build_analysis(
    *,
    root: Path,
    protocol_path: Path,
    recipe1msubs_directory: Path | None,
    ingredient_tags_path: Path | None,
    checkpoint_root: Path | None,
    allow_network_download: bool,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[str, bytes], tuple[bytes, bytes]]:
    root = root.resolve()
    protocol_path = protocol_path.resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != "flavourbench-external-substitution-validation-plan-v1":
        raise ExternalSubstitutionValidationError("external validation protocol differs")
    if protocol.get("status") != "analysis_protocol_fixed_before_public_checkpoint_scoring":
        raise ExternalSubstitutionValidationError("external validation protocol is not fixed")

    train_path = _resolve_recipe1msubs(
        "train",
        explicit_directory=recipe1msubs_directory,
        allow_network_download=allow_network_download,
    )
    test_path = _resolve_recipe1msubs(
        "test",
        explicit_directory=recipe1msubs_directory,
        allow_network_download=allow_network_download,
    )
    train = _load_recipe1msubs(train_path, split="train")
    test = _load_recipe1msubs(test_path, split="test")
    train_pairs = frozenset(tuple(row["subs"]) for row in train)

    resolved_tags = _resolve_ingredient_tags(
        root=root,
        explicit_path=ingredient_tags_path,
        allow_network_download=allow_network_download,
    )
    food_groups = _load_food_groups(resolved_tags)

    checkpoint_inputs: list[dict[str, Any]] = []
    loaded: dict[str, tuple[dict[str, int], np.ndarray]] = {}
    for checkpoint in ("cooc", "core", "chem"):
        directory = _discover_checkpoint(
            checkpoint,
            checkpoint_root,
            allow_network_download=allow_network_download,
        )
        vocab, embeddings, _ = _load_checkpoint(checkpoint, directory)
        loaded[checkpoint] = (vocab, embeddings)
        record = CHECKPOINTS[checkpoint]
        checkpoint_inputs.append(
            {
                "checkpoint": checkpoint,
                "label": record["label"],
                "repo_id": record["repo_id"],
                "revision": record["revision"],
                "embeddings_sha256": record["embeddings_sha256"],
                "vocab_sha256": VOCAB_SHA256,
                "config_sha256": record["config_sha256"],
            }
        )
    vocabulary = loaded["cooc"][0]
    if any(vocab != vocabulary for vocab, _ in loaded.values()):
        raise ExternalSubstitutionValidationError("public checkpoint vocabularies differ")

    source_mapped = sum(str(row["subs"][0]) in vocabulary for row in test)
    target_mapped = sum(str(row["subs"][1]) in vocabulary for row in test)
    mapped_events = [
        row
        for row in test
        if str(row["subs"][0]) in vocabulary
        and str(row["subs"][1]) in vocabulary
        and str(row["subs"][0]) != str(row["subs"][1])
    ]
    pairs = sorted({(str(row["subs"][0]), str(row["subs"][1])) for row in mapped_events})
    ineligible = [
        pair
        for pair in pairs
        if len(
            [
                name
                for name, group in food_groups.items()
                if group == food_groups[pair[1]] and name != pair[0]
            ]
        )
        < MINIMUM_GROUP_CANDIDATES
    ]
    if ineligible:
        raise ExternalSubstitutionValidationError(
            f"predeclared food-group threshold excludes mapped pairs: {ineligible[:5]}"
        )
    novel_pairs = [pair for pair in pairs if pair not in train_pairs]
    coverage = {
        "raw_train_records": len(train),
        "raw_test_records": len(test),
        "source_mapped_test_events": source_mapped,
        "target_mapped_test_events": target_mapped,
        "mapped_test_events": len(mapped_events),
        "mapped_test_event_rate": len(mapped_events) / len(test),
        "unique_mapped_test_pairs": len(pairs),
        "unique_source_ingredients": len({source for source, _ in pairs}),
        "unique_target_ingredients": len({target for _, target in pairs}),
        "novel_unique_pairs": len(novel_pairs),
        "novel_unique_pair_rate": len(novel_pairs) / len(pairs),
        "exact_manual_aliases": 0,
    }

    results: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(("cooc", "core", "chem")):
        vocab, embeddings = loaded[checkpoint]
        result, rows = _evaluate_checkpoint(
            checkpoint=checkpoint,
            vocab=vocab,
            embeddings=embeddings,
            food_groups=food_groups,
            pairs=pairs,
            train_pairs=train_pairs,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=BOOTSTRAP_SEED + 10 * index,
        )
        results.append(result)
        pair_rows.extend(rows)

    raw_p_values = {
        str(row["checkpoint"]): float(
            row["primary_all_unique_pairs"]["one_sided_null_centered_cluster_bootstrap_p"]
        )
        for row in results
    }
    adjusted = _holm_adjust(raw_p_values)
    for row in results:
        primary = row["primary_all_unique_pairs"]
        primary["holm_adjusted_p"] = adjusted[str(row["checkpoint"])]
        primary["holm_reject_at_familywise_0_05"] = adjusted[str(row["checkpoint"])] < 0.05

    summary_rows = [
        {
            "checkpoint": row["checkpoint"],
            "label": row["label"],
            "unique_pairs": row["primary_all_unique_pairs"]["unique_pairs"],
            "unique_sources": row["primary_all_unique_pairs"]["unique_sources"],
            "equal_source_within_group_percentile": row["primary_all_unique_pairs"][
                "equal_source_mean_rank_percentile"
            ],
            "ci_95_low": row["primary_all_unique_pairs"]["percentile_95_interval"][0],
            "ci_95_high": row["primary_all_unique_pairs"]["percentile_95_interval"][1],
            "holm_adjusted_p": row["primary_all_unique_pairs"]["holm_adjusted_p"],
            "novel_unique_pairs": row["sensitivity_novel_unique_pairs"]["unique_pairs"],
            "novel_unique_sources": row["sensitivity_novel_unique_pairs"]["unique_sources"],
            "novel_equal_source_within_group_percentile": row["sensitivity_novel_unique_pairs"][
                "equal_source_mean_rank_percentile"
            ],
            "full_vocabulary_mrr": row["full_vocabulary_retrieval"]["mean_reciprocal_rank"],
            "full_vocabulary_hit_at_1": row["full_vocabulary_retrieval"]["hit_at_1"],
            "full_vocabulary_hit_at_3": row["full_vocabulary_retrieval"]["hit_at_3"],
            "full_vocabulary_hit_at_10": row["full_vocabulary_retrieval"]["hit_at_10"],
            "full_vocabulary_chance_mrr": row["full_vocabulary_retrieval"]["analytic_chance_mrr"],
            "full_vocabulary_chance_hit_at_10": row["full_vocabulary_retrieval"][
                "analytic_chance_hit_at_10"
            ],
        }
        for row in results
    ]
    summary_csv = _csv_bytes(summary_rows)
    table = _render_table(results)
    macros = _render_macros(coverage, results)
    companions = {
        "complete-core-external-substitution-validation.csv": summary_csv,
        "complete-core-external-substitution-validation-table.tex": table,
        "complete-core-external-substitution-validation-macros.tex": macros,
    }
    companion_records = [
        _companion(
            name,
            payload,
            len(summary_rows) if name.endswith(".csv") else 1,
            format_name=("csv" if name.endswith(".csv") else "tex"),
        )
        for name, payload in companions.items()
    ]
    figure = _render_figure(results)
    figure_records = [
        {
            "name": f"complete-core-external-substitution-validation.{extension}",
            "bytes": len(payload),
            "format": extension,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for extension, payload in zip(("pdf", "png"), figure, strict=True)
    ]
    analysis: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "post_collection_label_independent_convergent_validation",
        "analysis_timing": {
            "protocol_fixed_before_public_checkpoint_scores_were_inspected": True,
            "prespecified_primary_flavourbench_analysis": False,
            "model_provider_calls_added": 0,
            "model_responses_changed": False,
            "flavourbench_tasks_or_scores_changed": False,
        },
        "inputs": {
            "protocol_filename": protocol_path.name,
            "protocol_physical_sha256": _file_sha256(protocol_path),
            "builder_physical_sha256": _file_sha256(Path(__file__).resolve()),
            "checkpoint_loader_physical_sha256": _file_sha256(
                Path(__file__).resolve().parent / "build_public_scorer_sensitivity_assets.py"
            ),
            "ingredient_tags_sha256": INGREDIENT_TAGS_SHA256,
            "ingredient_tags_source_commit": PUBLIC_EPICURE_MCP_COMMIT,
            "ingredient_tags_source_url": INGREDIENT_TAGS_URL,
            "recipe1msubs": {
                split: {
                    "filename": record["filename"],
                    "url": record["url"],
                    "sha256": record["sha256"],
                    "records": record["records"],
                }
                for split, record in RECIPE1MSUBS.items()
            },
            "public_checkpoints": checkpoint_inputs,
        },
        "design": {
            "mapping": "exact canonical token equality",
            "manual_aliases": 0,
            "deduplication_unit": "directed source-target pair",
            "candidate_universe": "public Epicure vocabulary excluding source",
            "primary_negative_control": (
                "all candidates in the observed target food group excluding source"
            ),
            "minimum_food_group_candidates": MINIMUM_GROUP_CANDIDATES,
            "primary_estimand": "equal-source mean within-food-group rank percentile",
            "chance_value": 0.5,
            "uncertainty_unit": "source ingredient",
            "bootstrap_replicates": bootstrap_replicates,
            "multiplicity": "Holm correction across three public checkpoint tests",
        },
        "coverage": coverage,
        "checkpoint_results": results,
        "all_three_primary_tests_reject_after_holm": all(
            bool(row["primary_all_unique_pairs"]["holm_reject_at_familywise_0_05"])
            for row in results
        ),
        "minimum_primary_equal_source_percentile": min(
            float(row["primary_all_unique_pairs"]["equal_source_mean_rank_percentile"])
            for row in results
        ),
        "minimum_novel_pair_equal_source_percentile": min(
            float(row["sensitivity_novel_unique_pairs"]["equal_source_mean_rank_percentile"])
            for row in results
        ),
        "companion_files": companion_records,
        "figure_files": figure_records,
        "raw_external_rows_redistributed": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    analysis["artifact_sha256"] = _sha256(analysis)
    return analysis, companions, figure


def _write_outputs(
    *,
    analysis: Mapping[str, Any],
    companions: Mapping[str, bytes],
    figure: tuple[bytes, bytes],
    generated_directory: Path,
    figure_directory: Path,
    dataset_directory: Path,
    dataset_figure_directory: Path,
) -> None:
    generated_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    dataset_directory.mkdir(parents=True, exist_ok=True)
    dataset_figure_directory.mkdir(parents=True, exist_ok=True)
    analysis_name = "complete-core-external-substitution-validation.json"
    payload = json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    for directory in (generated_directory, dataset_directory):
        (directory / analysis_name).write_text(payload, encoding="utf-8")
    for name, data in companions.items():
        (generated_directory / name).write_bytes(data)
        if name.endswith(".csv"):
            (dataset_directory / name).write_bytes(data)
    pdf, png = figure
    figure_name = "complete-core-external-substitution-validation"
    (figure_directory / f"{figure_name}.pdf").write_bytes(pdf)
    (figure_directory / f"{figure_name}.png").write_bytes(png)
    shutil.copyfile(
        figure_directory / f"{figure_name}.pdf",
        dataset_figure_directory / f"{figure_name}.pdf",
    )
    shutil.copyfile(
        figure_directory / f"{figure_name}.png",
        dataset_figure_directory / f"{figure_name}.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--recipe1msubs-directory", type=Path)
    parser.add_argument("--ingredient-tags", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--allow-network-download", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--generated-directory", type=Path, required=True)
    parser.add_argument("--figure-directory", type=Path, required=True)
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--dataset-figure-directory", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.bootstrap_replicates < 1_000:
        raise ExternalSubstitutionValidationError("at least 1000 bootstrap replicates are required")
    root = arguments.root.resolve()
    protocol = arguments.protocol
    if not protocol.is_absolute():
        protocol = root / protocol
    analysis, companions, figure = build_analysis(
        root=root,
        protocol_path=protocol,
        recipe1msubs_directory=arguments.recipe1msubs_directory,
        ingredient_tags_path=arguments.ingredient_tags,
        checkpoint_root=arguments.checkpoint_root,
        allow_network_download=arguments.allow_network_download,
        bootstrap_replicates=arguments.bootstrap_replicates,
    )
    _write_outputs(
        analysis=analysis,
        companions=companions,
        figure=figure,
        generated_directory=arguments.generated_directory,
        figure_directory=arguments.figure_directory,
        dataset_directory=arguments.dataset_directory,
        dataset_figure_directory=arguments.dataset_figure_directory,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": analysis["artifact_sha256"],
                "coverage": analysis["coverage"],
                "checkpoint_results": analysis["checkpoint_results"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
