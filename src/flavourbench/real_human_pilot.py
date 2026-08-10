"""Build a bounded quality report from the real anonymous human pilot.

The historical review session is not paper-eligible: its consent document was
inactive, it contains one self-attested rater, and it used a selected public
development pool.  Discarding the records would nevertheless hide useful
engineering evidence.  This module therefore exposes task-screen failures,
completion-clean preference counts, and paired rubric diagnostics while
hard-coding the non-publication boundary into the content-addressed artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .anonymous_review_report import (
    ACCEPTED_FINAL_FINISH_REASONS,
    DEFAULT_OUTPUT,
    _uplift_outcome,
    _wilson_interval,
    attach_candidate_evidence,
    attach_consent_evidence,
    attach_replacement_candidate_evidence,
    canonical_bytes,
    control_plane_snapshot,
)
from .anonymous_review_report import (
    build_report as build_containment_report,
)
from .expert_calibration import TASK_SCOPE_QUARANTINE
from .expert_review import RUBRIC_DIMENSIONS, TASK_FAMILIES

SCHEMA_VERSION = "flavourbench-real-human-pilot-quality-v1"
DEFAULT_PILOT_OUTPUT = DEFAULT_OUTPUT / "real-human-pilot-v1"
BOOTSTRAP_DRAWS = 20_000


class RealHumanPilotError(RuntimeError):
    """The real-human pilot evidence failed closed validation."""


def _finish_clean(row: Mapping[str, Any]) -> bool:
    return all(
        row.get(f"{side}_status") == "complete"
        and str(row.get(f"{side}_finish_reason") or "").lower()
        in ACCEPTED_FINAL_FINISH_REASONS
        for side in ("left", "right")
    )


def _model_id(row: Mapping[str, Any]) -> str:
    raw = row.get("model_ids")
    if not isinstance(raw, list):
        raise RealHumanPilotError("review record has no model identity array")
    identities = sorted({str(value) for value in raw if isinstance(value, str) and value})
    if len(identities) != 1:
        raise RealHumanPilotError("uplift pair must contain one exact model identity")
    return identities[0]


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise RealHumanPilotError("cannot calculate a quantile over an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed_material: str
) -> list[float] | None:
    if not values:
        return None
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    generator = random.Random(seed)
    sample_size = len(values)
    means = [
        sum(values[generator.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return [round(_quantile(means, 0.025), 4), round(_quantile(means, 0.975), 4)]


def _preference_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(_uplift_outcome(row) for row in rows)
    effective_n = outcomes["epicure_win"] + outcomes["tie"] + outcomes["epicure_loss"]
    tie_adjusted_successes = outcomes["epicure_win"] + 0.5 * outcomes["tie"]
    return {
        "pairs": len(rows),
        "epicure_wins": outcomes["epicure_win"],
        "ties": outcomes["tie"],
        "epicure_losses": outcomes["epicure_loss"],
        "both_bad": outcomes["both_bad"],
        "effective_n_excluding_both_bad": effective_n,
        "tie_adjusted_epicure_preference_share": (
            round(tie_adjusted_successes / effective_n, 4) if effective_n else None
        ),
        "quasi_binomial_wilson_95": _wilson_interval(tie_adjusted_successes, effective_n),
    }


def _paired_rubric_deltas(
    rows: Sequence[Mapping[str, Any]], *, seed_prefix: str
) -> dict[str, Any]:
    by_dimension: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        rubric = row.get("rubric")
        if not isinstance(rubric, Mapping):
            continue
        on_side = next(
            (side for side in ("left", "right") if row.get(f"{side}_condition") == "epicure_on"),
            None,
        )
        off_side = next(
            (
                side
                for side in ("left", "right")
                if row.get(f"{side}_condition") == "epicure_off"
            ),
            None,
        )
        if on_side is None or off_side is None:
            raise RealHumanPilotError("uplift pair has no unique Epicure on/off orientation")
        on_rubric = rubric.get(on_side)
        off_rubric = rubric.get(off_side)
        if not isinstance(on_rubric, Mapping) or not isinstance(off_rubric, Mapping):
            continue
        for dimension in RUBRIC_DIMENSIONS:
            on_score = on_rubric.get(dimension)
            off_score = off_rubric.get(dimension)
            if isinstance(on_score, int) and isinstance(off_score, int):
                by_dimension[dimension].append(float(on_score - off_score))

    return {
        dimension: {
            "paired_scores": len(values),
            "mean_on_minus_off": round(statistics.fmean(values), 4) if values else None,
            "median_on_minus_off": round(statistics.median(values), 4) if values else None,
            "task_resample_bootstrap_95": _bootstrap_mean_interval(
                values,
                seed_material=f"{seed_prefix}:{dimension}",
            ),
        }
        for dimension, values in sorted(by_dimension.items())
    }


def _group_diagnostics(
    rows: Sequence[Mapping[str, Any]], *, key: str, seed_prefix: str
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        label = _model_id(row) if key == "model" else str(row.get("category") or "missing")
        grouped[label].append(row)
    return {
        label: {
            "preference": _preference_summary(group),
            "rubric_deltas": _paired_rubric_deltas(
                group,
                seed_prefix=f"{seed_prefix}:{key}:{label}",
            ),
            "provisional": True,
        }
        for label, group in sorted(grouped.items())
    }


def build_pilot_report(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    containment = build_containment_report(snapshot)
    records = snapshot.get("records")
    if not isinstance(records, list):
        raise RealHumanPilotError("review records are unavailable")
    primary = [row for row in records if row.get("mode") == "primary"]
    if any(row.get("track") != "epicure_uplift" for row in primary):
        raise RealHumanPilotError("historical human pilot must contain only uplift pairs")
    clean = [row for row in primary if _finish_clean(row)]
    valid_rows = [row for row in primary if row.get("task_validity") == "valid"]
    scope_disagreements = [
        row for row in valid_rows if str(row.get("task_public_id")) in TASK_SCOPE_QUARANTINE
    ]
    post_scope_agreements = len(valid_rows) - len(scope_disagreements)
    historical_pool_sha256 = str(
        containment["source_pool"]["historical_review_session_pool_sha256"]
    )
    containment_sha256 = hashlib.sha256(canonical_bytes(containment)).hexdigest()

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "restricted_real_human_pilot_diagnostic",
        "scope": "development_quality_assurance_not_benchmark_results",
        "source_containment_report_sha256": containment_sha256,
        "source_pool_sha256": historical_pool_sha256,
        "observed_at": containment["observed_at"],
        "real_data_inventory": {
            "candidate_pairs": containment["quality_funnel"]["candidate_pairs"],
            "paid_source_arms": containment["source_pool"]["paid_source_arms"],
            "real_provider_calls": containment["source_pool"]["provider_calls"],
            "real_epicure_calls": containment["source_pool"]["epicure_calls"],
            "synthetic_arms": containment["source_pool"]["synthetic_arms"],
            "primary_human_judgments": len(primary),
            "finish_clean_primary_judgments": len(clean),
            "distinct_models": len({_model_id(row) for row in primary}),
            "distinct_task_families": len({str(row.get("category")) for row in primary}),
        },
        "task_validity_diagnostic": {
            "answer_blind_task_screen": True,
            "self_attested_external_raters": 1,
            "primary_tasks_screened": len(primary),
            "rater_valid": len(valid_rows),
            "rater_minor_issue": sum(row.get("task_validity") == "minor_issue" for row in primary),
            "rater_invalid": sum(row.get("task_validity") == "invalid" for row in primary),
            "later_governance_scope_disagreements": len(scope_disagreements),
            "post_scope_agreements": post_scope_agreements,
            "post_scope_agreement_rate": (
                round(post_scope_agreements / len(valid_rows), 4) if valid_rows else None
            ),
            "post_scope_agreement_wilson_95": _wilson_interval(
                post_scope_agreements, len(valid_rows)
            ),
            "interpretation": (
                "The old blind screen admitted specialist-scope tasks later quarantined by the "
                "governance screen. Season 1 therefore requires two independent solutions, "
                "adjudication, and separate validator/contamination reviews."
            ),
        },
        "completion_diagnostic": {
            "accepted_finish_reasons": sorted(ACCEPTED_FINAL_FINISH_REASONS),
            "finish_clean_pairs": len(clean),
            "affected_pairs": len(primary) - len(clean),
            "non_normal_arms": containment["completion_audit"]["non_normal_response_arms"],
            "replacement_pool": containment["completion_audit"]["replacement_candidate"],
        },
        "finish_clean_uplift": {
            "overall": {
                "preference": _preference_summary(clean),
                "rubric_deltas": _paired_rubric_deltas(
                    clean,
                    seed_prefix=f"{historical_pool_sha256}:overall",
                ),
            },
            "by_family": _group_diagnostics(
                clean,
                key="family",
                seed_prefix=historical_pool_sha256,
            ),
            "by_exact_model": _group_diagnostics(
                clean,
                key="model",
                seed_prefix=historical_pool_sha256,
            ),
        },
        "coverage": {
            "families": {
                family: sum(str(row.get("category")) == family for row in clean)
                for family in TASK_FAMILIES
            },
            "model_allocation_is_balanced": False,
            "model_quality_ranking_permitted": False,
        },
        "claim_boundary": {
            "real_model_outputs": True,
            "real_epicure_calls": True,
            "real_human_judgments": True,
            "synthetic_observations": 0,
            "consent_active_at_collection": False,
            "credential_verified_raters": 0,
            "independent_rater_consensus": False,
            "selected_public_development_tasks": True,
            "post_hoc_finish_clean_sensitivity": True,
            "paper_use": False,
            "research_release_use": False,
            "official_leaderboard_use": False,
            "model_ranking_use": False,
            "causal_epicure_claim": False,
        },
        "next_evidence_step": {
            "pool_sha256": containment["completion_audit"]["replacement_candidate"][
                "artifact_sha256"
            ],
            "pairs": containment["completion_audit"]["replacement_candidate"]["pairs"],
            "required_change": (
                "Collect new prospective judgments under active consent on the finish-clean, "
                "scope-filtered replacement pool; do not relabel this historical session."
            ),
        },
    }


def _write_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(canonical_bytes(report)).hexdigest()
    document = {**report, "artifact_sha256": digest}
    rendered = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"real-human-pilot-quality-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise RealHumanPilotError("content-addressed pilot report conflicts with disk")
    else:
        with tempfile.NamedTemporaryFile(
            "w", dir=output_dir, delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    destination.chmod(0o600)
    markdown = output_dir / f"real-human-pilot-quality-{digest}.md"
    preference = report["finish_clean_uplift"]["overall"]["preference"]
    validity = report["task_validity_diagnostic"]
    markdown_text = "\n".join(
        (
            "# Real-human pilot quality diagnostic",
            "",
            f"Artifact SHA-256: `{digest}`",
            "",
            "This is restricted operational evidence, not a leaderboard or paper result.",
            "",
            "## Observed real evidence",
            "",
            "- Human primary judgments: "
            f"`{report['real_data_inventory']['primary_human_judgments']}`",
            "- Finish-clean judgments: "
            f"`{report['real_data_inventory']['finish_clean_primary_judgments']}`",
            f"- Paid real source arms: `{report['real_data_inventory']['paid_source_arms']}`",
            f"- Synthetic observations: `{report['real_data_inventory']['synthetic_arms']}`",
            "",
            "## Task-screen diagnostic",
            "",
            "The rater marked "
            f"`{validity['rater_valid']}` / `{validity['primary_tasks_screened']}` "
            "tasks valid before seeing answers. The later governance scope screen disagreed on "
            f"`{validity['later_governance_scope_disagreements']}` tasks, leaving a diagnostic "
            f"agreement rate of `{validity['post_scope_agreement_rate']}`.",
            "",
            "## Finish-clean uplift diagnostic",
            "",
            "Epicure-on received "
            f"`{preference['epicure_wins']}` wins, `{preference['ties']}` ties, "
            f"and `{preference['epicure_losses']}` losses; `{preference['both_bad']}` pairs were "
            "both-bad. These selected, one-rater counts are hypothesis-generating only.",
            "",
        )
    )
    if markdown.exists():
        if markdown.read_text(encoding="utf-8") != markdown_text:
            raise RealHumanPilotError("content-addressed pilot markdown conflicts with disk")
    else:
        markdown.write_text(markdown_text, encoding="utf-8")
    markdown.chmod(0o600)
    return destination, markdown


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PILOT_OUTPUT)
    arguments = parser.parse_args(argv)
    snapshot = control_plane_snapshot()
    attach_candidate_evidence(snapshot)
    attach_replacement_candidate_evidence(snapshot)
    attach_consent_evidence(snapshot)
    report = build_pilot_report(snapshot)
    paths = _write_report(report, arguments.output_dir.resolve())
    print(
        json.dumps(
            {
                "json": str(paths[0]),
                "markdown": str(paths[1]),
                "artifact_sha256": paths[0].stem.rsplit("-", 1)[-1],
                "claim_boundary": report["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
