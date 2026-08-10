"""Deterministic publication figure for terminal real FlavourBench evidence.

The input is the content-addressed, no-call aggregate produced by
``flavourbench.evidence_aggregate``.  This module deliberately refuses partial
checkpoints, unverifiable digests, non-terminal runner summaries, or aggregates
with human judgments.  Its output compares *operational evidence* in frozen
manifest order and never computes a preference, quality, or uplift ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .evidence_aggregate import SCHEMA_VERSION as AGGREGATE_SCHEMA_VERSION
from .execution_policy import ExecutionPolicy
from .frontier_contract_runner import IntegrityError

FIGURE_SCHEMA_VERSION = "flavourbench-real-operational-figure-v1"
EXPECTED_MODEL_COUNT = 12
PROTOCOL_V1_MAX_OUTPUT_TOKENS = 1_000

PAIR_COMPLETE = "#579DFF"
PAIR_PARTIAL = "#F6C85F"
PAIR_FAILED = "#D5D7DD"
EPICURE_OFF = "#59616D"
EPICURE_ON = "#2F6BFF"
COST = "#A76000"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _verify_content_address(value: Mapping[str, Any], *, expected_digest: str) -> None:
    address = value.get("content_address")
    if not isinstance(address, Mapping):
        raise IntegrityError("aggregate has no content address")
    digest = address.get("digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or address.get("algorithm") != "sha256"
        or address.get("uri") != f"sha256:{digest}"
        or digest != expected_digest
    ):
        raise IntegrityError("aggregate digest does not match the pinned terminal digest")
    unhashed = dict(value)
    unhashed.pop("content_address", None)
    if _sha256(unhashed) != digest:
        raise IntegrityError("aggregate content address is invalid")


def load_terminal_aggregate(path: Path, *, expected_digest: str) -> dict[str, Any]:
    """Load and strictly verify one terminal real-evidence aggregate."""

    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"aggregate must be a regular, non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"aggregate is not valid readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError("aggregate root must be a JSON object")
    _verify_content_address(value, expected_digest=expected_digest)

    if value.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        raise IntegrityError("unsupported real-evidence aggregate schema")
    if value.get("collection_state") != "complete":
        raise IntegrityError("refusing to render a non-terminal collection checkpoint")
    if value.get("official") is not False or value.get("rank_eligible") is not False:
        raise IntegrityError("operational figure input must remain exploratory and unranked")
    verification = value.get("verification")
    if not isinstance(verification, Mapping):
        raise IntegrityError("aggregate verification block is missing")
    if (
        verification.get("all_checks_passed") is not True
        or verification.get("checkpoint_stable_during_aggregation") is not True
        or verification.get("terminal_runner_summary_verified") is not True
    ):
        raise IntegrityError("terminal aggregate verification is incomplete")

    workload = value.get("workload")
    progress = value.get("progress")
    judgments = value.get("human_judgments")
    models = value.get("models")
    if not all(
        isinstance(item, Mapping) for item in (workload, progress, judgments)
    ) or not isinstance(models, list):
        raise IntegrityError("aggregate workload, progress, judgments, or models are missing")
    if (
        int(workload.get("model_count", -1)) != EXPECTED_MODEL_COUNT
        or len(models) != EXPECTED_MODEL_COUNT
    ):
        raise IntegrityError("terminal operational figure requires the frozen 12-model panel")
    if workload.get("execution_policy_sha256") != ExecutionPolicy().sha256:
        raise IntegrityError("aggregate does not use the frozen Protocol v1 execution policy")
    expected_pairs = int(workload.get("expected_pairs", -1))
    expected_arms = int(workload.get("expected_arms", -1))
    if expected_pairs <= 0 or expected_arms != expected_pairs * 2:
        raise IntegrityError("aggregate pair/arm workload is inconsistent")
    if (
        int(progress.get("finalized_pairs", -1)) != expected_pairs
        or int(progress.get("active_journals", -1)) != 0
    ):
        raise IntegrityError("terminal aggregate still contains unfinished collection state")
    statuses = progress.get("pair_status_counts")
    if not isinstance(statuses, Mapping) or (
        sum(int(statuses.get(key, 0)) for key in ("complete", "partial", "failed"))
        != expected_pairs
        or int(statuses.get("in_progress", 0)) != 0
        or int(statuses.get("pending", 0)) != 0
    ):
        raise IntegrityError("terminal pair outcomes do not partition the workload")
    if int(judgments.get("public", -1)) != 0 or int(judgments.get("expert", -1)) != 0:
        raise IntegrityError("this operational figure is defined only for n=0 judgments")
    if any(
        judgments.get(field) is not None
        for field in (
            "preference_estimate",
            "bradley_terry_rating",
            "epicure_uplift_estimate",
        )
    ):
        raise IntegrityError("operational aggregate unexpectedly contains a comparative estimate")
    return value


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IntegrityError(f"figure input field {key!r} is not a non-negative integer")
    return value


def _condition(model: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    conditions = model.get("conditions")
    if not isinstance(conditions, Mapping) or not isinstance(conditions.get(name), Mapping):
        raise IntegrityError(f"model condition {name!r} is missing")
    return conditions[name]


def _condition_view(slice_value: Mapping[str, Any]) -> dict[str, Any]:
    arms = slice_value.get("arms")
    latency = slice_value.get("latency_ms")
    if not isinstance(arms, Mapping) or not isinstance(latency, Mapping):
        raise IntegrityError("condition arms or latency summary is missing")
    attempted = _integer(arms, "provider_attempted")
    normalized = _integer(arms, "normalized_responses")
    latency_n = _integer(latency, "n")
    p50 = latency.get("p50")
    if not 0 <= normalized <= attempted:
        raise IntegrityError("normalized-response count exceeds attempted-arm count")
    if latency_n != normalized:
        raise IntegrityError("survivor latency n must equal normalized-response n")
    if p50 is not None and (not isinstance(p50, int) or isinstance(p50, bool) or p50 < 0):
        raise IntegrityError("latency p50 must be null or a non-negative integer")
    if (normalized == 0) != (p50 is None):
        raise IntegrityError("latency p50 nullability does not match its survivor denominator")
    return {
        "attempted": attempted,
        "normalized": normalized,
        "normalized_fraction": normalized / attempted if attempted else None,
        "latency_p50_ms": p50,
        "latency_n": latency_n,
        "latency_basis": "normalized_response_survivors_only",
    }


def build_figure_payload(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Build a renderer-neutral, non-ranked operational figure view model."""

    source_address = aggregate.get("content_address")
    workload = aggregate.get("workload")
    cost = aggregate.get("cost")
    models = aggregate.get("models")
    if not all(isinstance(value, Mapping) for value in (source_address, workload, cost)):
        raise IntegrityError("aggregate source, workload, or cost block is missing")
    if not isinstance(models, list):
        raise IntegrityError("aggregate models are missing")

    rows: list[dict[str, Any]] = []
    model_ids: set[str] = set()
    for display_order, raw_model in enumerate(models, start=1):
        if not isinstance(raw_model, Mapping):
            raise IntegrityError("model row is not a JSON object")
        model_id = str(raw_model.get("model_id") or "")
        display_name = str(raw_model.get("display_name") or "")
        if not model_id or not display_name or model_id in model_ids:
            raise IntegrityError("model IDs/names must be present and unique")
        model_ids.add(model_id)
        off_slice = _condition(raw_model, "epicure_off")
        on_slice = _condition(raw_model, "epicure_on")
        off_pairs = off_slice.get("pairs")
        on_pairs = on_slice.get("pairs")
        on_tools = on_slice.get("tools")
        off_cost = off_slice.get("cost")
        on_cost = on_slice.get("cost")
        if not all(
            isinstance(value, Mapping)
            for value in (off_pairs, on_pairs, on_tools, off_cost, on_cost)
        ):
            raise IntegrityError("model pair/tool/cost evidence is missing")
        pair_fields = ("expected", "attempted", "complete", "partial", "failed")
        if any(_integer(off_pairs, field) != _integer(on_pairs, field) for field in pair_fields):
            raise IntegrityError("off/on pair-status denominators disagree")
        expected = _integer(off_pairs, "expected")
        attempted = _integer(off_pairs, "attempted")
        complete = _integer(off_pairs, "complete")
        partial = _integer(off_pairs, "partial")
        failed = _integer(off_pairs, "failed")
        if attempted != expected or complete + partial + failed != expected:
            raise IntegrityError("terminal model pair outcomes do not partition expected pairs")
        successful_calls = _integer(on_tools, "successful_calls")
        error_calls = _integer(on_tools, "error_calls")
        tool_calls = _integer(on_tools, "calls")
        if successful_calls + error_calls != tool_calls:
            raise IntegrityError("Epicure tool success/error counts do not equal all calls")
        normalized_trace_successes = _integer(on_tools, "normalized_trace_successful_calls")
        normalized_trace_errors = _integer(on_tools, "normalized_trace_error_calls")
        normalized_trace_calls = _integer(on_tools, "normalized_trace_calls")
        if normalized_trace_successes + normalized_trace_errors != normalized_trace_calls:
            raise IntegrityError(
                "normalized Epicure tool success/error counts do not equal trace calls"
            )
        known_cost_micros = _integer(off_cost, "actual_cost_micros") + _integer(
            on_cost, "actual_cost_micros"
        )
        rows.append(
            {
                "display_order": display_order,
                "order_basis": "frozen_manifest_not_performance",
                "model_id": model_id,
                "display_name": display_name,
                "pairs": {
                    "expected": expected,
                    "complete": complete,
                    "partial": partial,
                    "failed": failed,
                },
                "epicure_off": _condition_view(off_slice),
                "epicure_on": _condition_view(on_slice),
                "epicure_tools": {
                    "successful_calls": successful_calls,
                    "error_calls": error_calls,
                    "total_calls": tool_calls,
                    "normalized_trace_successful_calls": normalized_trace_successes,
                    "normalized_trace_error_calls": normalized_trace_errors,
                    "normalized_trace_total_calls": normalized_trace_calls,
                    "denominator": "all source-journal MCP calls, including failed arms",
                },
                "known_id_cost_micros": known_cost_micros,
                "known_id_cost_usd": f"{known_cost_micros / 1_000_000:.6f}",
            }
        )

    dataset_cost_micros = _integer(cost, "dataset_actual_cost_micros")
    if sum(row["known_id_cost_micros"] for row in rows) != dataset_cost_micros:
        raise IntegrityError("model known-ID costs do not reconcile to dataset actual cost")
    overall = aggregate.get("overall_by_condition")
    if not isinstance(overall, Mapping) or not isinstance(overall.get("epicure_on"), Mapping):
        raise IntegrityError("overall Epicure-on evidence is missing")
    overall_tools = overall["epicure_on"].get("tools")
    if not isinstance(overall_tools, Mapping):
        raise IntegrityError("overall Epicure tool evidence is missing")
    tool_total_fields = {
        "successful_calls": "successful_calls",
        "error_calls": "error_calls",
        "total_calls": "calls",
        "normalized_trace_successful_calls": "normalized_trace_successful_calls",
        "normalized_trace_error_calls": "normalized_trace_error_calls",
        "normalized_trace_total_calls": "normalized_trace_calls",
    }
    tool_totals = {
        output_key: sum(int(row["epicure_tools"][output_key]) for row in rows)
        for output_key in tool_total_fields
    }
    for output_key, aggregate_key in tool_total_fields.items():
        if tool_totals[output_key] != _integer(overall_tools, aggregate_key):
            raise IntegrityError(f"model Epicure tool subtotal {output_key} does not reconcile")
    no_id_incidents = _integer(cost, "resolved_no_id_incident_count")
    no_id_increment = str(cost.get("resolved_no_id_exposure_increment_usd") or "0")
    if no_id_incidents and (
        cost.get("provider_cost_exact_for_all_attempts") is not False
        or cost.get("all_provider_attempts_have_generation_ids") is not False
    ):
        raise IntegrityError("no-ID incident cannot be presented as exact provider spend")

    payload = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "source": {
            "aggregate_sha256": source_address["digest"],
            "aggregate_schema_version": aggregate.get("schema_version"),
            "manifest_sha256": workload.get("manifest_sha256"),
            "execution_policy_sha256": workload.get("execution_policy_sha256"),
            "terminal_runner_summary_sha256": aggregate.get("verification", {}).get(
                "terminal_runner_summary_sha256"
            ),
        },
        "figure_contract": {
            "analytical_question": (
                "What operational outcomes were observed for each frozen frontier model "
                "under Protocol v1?"
            ),
            "takeaway": (
                "Real model and Epicure execution is evidenced, but model-dependent "
                "attrition and zero judgments preclude any quality or uplift ranking."
            ),
            "family": "composition plus aligned descriptive measures",
            "variant": "manifest-order stacked bars and condition availability glyphs",
            "row_count": len(rows),
            "row_order": "frozen manifest; never sorted by an outcome",
            "palette_policy": "hard two-root cap plus neutrals",
            "non_color_distinction": (
                "direct labels, C/P/F and O/E symbols, outlines, and exact fractions"
            ),
            "headline": "NOT A QUALITY LEADERBOARD",
        },
        "protocol": {
            "label": "Protocol v1",
            "max_output_tokens": PROTOCOL_V1_MAX_OUTPUT_TOKENS,
            "attrition_warning": (
                "The 1,000-token final-response ceiling is associated with severe, "
                "model-dependent invalid-JSON attrition; all displayed answer metrics are "
                "descriptive compatibility/reliability evidence."
            ),
        },
        "judgments": {
            "public": 0,
            "expert": 0,
            "quality_estimate": None,
            "preference_estimate": None,
            "epicure_uplift_estimate": None,
        },
        "totals": {
            "expected_pairs": workload.get("expected_pairs"),
            "expected_arms": workload.get("expected_arms"),
            "known_id_generation_cost_usd": str(cost.get("dataset_actual_cost_usd")),
            "conservative_dataset_source_exposure_usd": str(
                cost.get("dataset_source_budget_exposure_usd")
            ),
            "no_id_incident_count": no_id_incidents,
            "no_id_exposure_increment_usd": no_id_increment,
            "no_id_increment_is_provider_spend": False,
            "epicure_journal_calls": tool_totals["total_calls"],
            "epicure_journal_successful_calls": tool_totals["successful_calls"],
            "epicure_journal_error_calls": tool_totals["error_calls"],
            "normalized_survivor_trace_calls": tool_totals["normalized_trace_total_calls"],
            "normalized_survivor_trace_successful_calls": tool_totals[
                "normalized_trace_successful_calls"
            ],
            "normalized_survivor_trace_error_calls": tool_totals["normalized_trace_error_calls"],
        },
        "encodings": {
            "pair_complete": {"color": PAIR_COMPLETE, "symbol": "C"},
            "pair_partial": {"color": PAIR_PARTIAL, "symbol": "P"},
            "pair_failed": {"color": PAIR_FAILED, "symbol": "F"},
            "epicure_off": {"color": EPICURE_OFF, "symbol": "O", "fill": "open"},
            "epicure_on": {"color": EPICURE_ON, "symbol": "E", "fill": "solid"},
            "known_id_cost": {"color": COST, "zero_baseline": True},
        },
        "rows": rows,
        "metric_notes": {
            "pair_outcomes": "complete/partial/failed finalized pairs over all scheduled pairs",
            "normalized_availability": (
                "normalized response artifacts over provider-attempted arms; not correctness"
            ),
            "tool_calls": (
                "successful/error Epicure MCP calls over all journaled calls, including "
                "arms without normalized final answers"
            ),
            "cost": (
                "exact OpenRouter generation-metadata cost for identified generations; "
                "the no-ID conservative increment is not allocated to any model"
            ),
            "latency": (
                "p50 end-to-end latency among normalized-response survivors only, with n; "
                "different prompt subsets and attrition make cross-model speed comparisons invalid"
            ),
        },
    }
    return payload


def content_address_figure(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    digest = _sha256(value)
    value["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    return value


def render_csv(payload: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    fields = [
        "display_order",
        "model_id",
        "display_name",
        "pairs_expected",
        "pairs_complete",
        "pairs_partial",
        "pairs_failed",
        "off_attempted",
        "off_normalized",
        "on_attempted",
        "on_normalized",
        "tool_successful_calls",
        "tool_error_calls",
        "tool_total_calls",
        "normalized_trace_successful_calls",
        "normalized_trace_error_calls",
        "normalized_trace_total_calls",
        "known_id_cost_micros",
        "known_id_cost_usd",
        "off_latency_p50_ms",
        "off_latency_n",
        "on_latency_p50_ms",
        "on_latency_n",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in payload["rows"]:
        writer.writerow(
            {
                "display_order": row["display_order"],
                "model_id": row["model_id"],
                "display_name": row["display_name"],
                "pairs_expected": row["pairs"]["expected"],
                "pairs_complete": row["pairs"]["complete"],
                "pairs_partial": row["pairs"]["partial"],
                "pairs_failed": row["pairs"]["failed"],
                "off_attempted": row["epicure_off"]["attempted"],
                "off_normalized": row["epicure_off"]["normalized"],
                "on_attempted": row["epicure_on"]["attempted"],
                "on_normalized": row["epicure_on"]["normalized"],
                "tool_successful_calls": row["epicure_tools"]["successful_calls"],
                "tool_error_calls": row["epicure_tools"]["error_calls"],
                "tool_total_calls": row["epicure_tools"]["total_calls"],
                "normalized_trace_successful_calls": row["epicure_tools"][
                    "normalized_trace_successful_calls"
                ],
                "normalized_trace_error_calls": row["epicure_tools"][
                    "normalized_trace_error_calls"
                ],
                "normalized_trace_total_calls": row["epicure_tools"][
                    "normalized_trace_total_calls"
                ],
                "known_id_cost_micros": row["known_id_cost_micros"],
                "known_id_cost_usd": row["known_id_cost_usd"],
                "off_latency_p50_ms": row["epicure_off"]["latency_p50_ms"],
                "off_latency_n": row["epicure_off"]["latency_n"],
                "on_latency_p50_ms": row["epicure_on"]["latency_p50_ms"],
                "on_latency_n": row["epicure_on"]["latency_n"],
            }
        )
    return output.getvalue()


def _tex_escape(value: object) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _seconds(milliseconds: int | None) -> str:
    return "---" if milliseconds is None else f"{milliseconds / 1_000:.1f}s"


def render_tex(payload: Mapping[str, Any]) -> str:
    """Render one compact, publication-oriented TikZ figure."""

    rows = payload["rows"]
    maximum_cost = max((int(row["known_id_cost_micros"]) for row in rows), default=0)
    commands: list[str] = [
        "% Generated by flavourbench.operational_figure; do not edit by hand.",
        f"% Figure manifest sha256: {payload['content_address']['digest']}",
        f"% Source aggregate sha256: {payload['source']['aggregate_sha256']}",
        "\\begin{figure*}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1mm,y=1mm]",
        "\\fill[FailBG] (0,111) rectangle (180,118);",
        "\\node[anchor=west,font=\\sffamily\\bfseries\\small,text=Ink] at (2,114.5) "
        "{NOT A QUALITY LEADERBOARD};",
        "\\node[anchor=east,font=\\sffamily\\scriptsize,text=Ink] at (178,114.5) "
        "{Protocol v1: 1,000-token ceiling $\\cdot$ public n=0 $\\cdot$ expert n=0};",
        "\\node[anchor=west,font=\\sffamily\\bfseries\\scriptsize,text=Ink] "
        "at (0,106) {Model (manifest order)};",
        "\\node[anchor=west,font=\\sffamily\\bfseries\\scriptsize,text=Ink] "
        "at (42,106) {Pair outcome composition};",
        "\\node[anchor=west,font=\\sffamily\\bfseries\\scriptsize,text=Ink] "
        "at (79,106) {Normalized availability};",
        "\\node[anchor=west,font=\\sffamily\\bfseries\\scriptsize,text=Ink] "
        "at (116,106) {Tool S/T};",
        "\\node[anchor=west,font=\\sffamily\\bfseries\\scriptsize,text=Ink] "
        "at (130,106) {Known-ID cost};",
        "\\node[anchor=west,font=\\sffamily\\bfseries\\scriptsize,text=Ink] "
        "at (157,106) {Survivor p50 (n)};",
        "\\node[anchor=west,font=\\sffamily\\tiny,text=Ink!68] at (42,102.5) "
        "{C/P/F; denominator shown in labels};",
        "\\node[anchor=west,font=\\sffamily\\tiny,text=Ink!68] at (79,102.5) "
        "{O=off, E=Epicure available};",
        "\\node[anchor=west,font=\\sffamily\\tiny,text=Ink!68] at (157,102.5) "
        "{off / on; normalized only};",
        "\\draw[Ink!18] (0,100.5) -- (180,100.5);",
    ]
    y = 96.5
    for index, row in enumerate(rows):
        if index % 2:
            commands.append(f"\\fill[NeutralBG] (0,{y - 3.25:.2f}) rectangle (180,{y + 3.25:.2f});")
        commands.append(
            f"\\node[anchor=west,font=\\sffamily\\scriptsize,text=Ink] at (0,{y:.2f}) "
            f"{{{_tex_escape(row['display_name'])}}};"
        )

        pairs = row["pairs"]
        expected = max(1, int(pairs["expected"]))
        pair_x = 42.0
        pair_width = 32.0
        cursor = pair_x
        pair_styles = (
            ("complete", "ReefBlue!72", "C"),
            ("partial", "WarnBG", "P"),
            ("failed", "RouteBG", "F"),
        )
        for key, style, symbol in pair_styles:
            count = int(pairs[key])
            width = pair_width * count / expected
            if width > 0:
                commands.append(
                    f"\\filldraw[fill={style},draw=Ink!35,line width=.25pt] "
                    f"({cursor:.3f},{y - 1.55:.2f}) rectangle "
                    f"({cursor + width:.3f},{y + 1.55:.2f});"
                )
                if width >= 4.0:
                    text_color = "white" if key == "complete" else "Ink"
                    commands.append(
                        f"\\node[font=\\sffamily\\bfseries\\tiny,text={text_color}] at "
                        f"({cursor + width / 2:.3f},{y:.2f}) {{{symbol}{count}}};"
                    )
                cursor += width
        commands.append(
            f"\\node[anchor=west,font=\\ttfamily\\tiny,text=Ink!72] at (42,{y - 2.55:.2f}) "
            f"{{C{pairs['complete']} P{pairs['partial']} F{pairs['failed']} / "
            f"{pairs['expected']}}};"
        )

        for offset, condition, symbol, style, fill in (
            (1.25, "epicure_off", "O", "Ink!62", "white"),
            (-1.25, "epicure_on", "E", "ReefBlue", "ReefBlue!78"),
        ):
            view = row[condition]
            bar_x = 85.0
            bar_width = 15.0
            attempted = int(view["attempted"])
            normalized = int(view["normalized"])
            fraction = normalized / attempted if attempted else 0
            commands.extend(
                [
                    f"\\node[anchor=east,font=\\sffamily\\bfseries\\tiny,text=Ink] at "
                    f"(83.5,{y + offset:.2f}) {{{symbol}}};",
                    f"\\filldraw[fill=white,draw=Ink!25,line width=.25pt] "
                    f"({bar_x:.2f},{y + offset - 0.72:.2f}) rectangle "
                    f"({bar_x + bar_width:.2f},{y + offset + 0.72:.2f});",
                ]
            )
            if fraction:
                commands.append(
                    f"\\filldraw[fill={fill},draw={style},line width=.25pt] "
                    f"({bar_x:.2f},{y + offset - 0.72:.2f}) rectangle "
                    f"({bar_x + bar_width * fraction:.3f},{y + offset + 0.72:.2f});"
                )
            commands.append(
                f"\\node[anchor=west,font=\\ttfamily\\tiny,text=Ink] at "
                f"(101.2,{y + offset:.2f}) {{{normalized}/{attempted}}};"
            )

        tools = row["epicure_tools"]
        commands.append(
            f"\\node[anchor=west,font=\\ttfamily\\scriptsize,text=Ink] at (116,{y:.2f}) "
            f"{{{tools['successful_calls']}/{tools['total_calls']}}};"
        )
        if int(tools["error_calls"]):
            commands.append(
                f"\\node[anchor=west,font=\\sffamily\\tiny,text=Alert] at (116,{y - 2.15:.2f}) "
                f"{{{tools['error_calls']} error}};"
            )

        cost_fraction = int(row["known_id_cost_micros"]) / maximum_cost if maximum_cost else 0
        commands.extend(
            [
                f"\\fill[RouteBG] (130,{y - 0.85:.2f}) rectangle (145,{y + 0.85:.2f});",
                f"\\fill[Alert!68] (130,{y - 0.85:.2f}) rectangle "
                f"({130 + 15 * cost_fraction:.3f},{y + 0.85:.2f});",
                f"\\node[anchor=west,font=\\ttfamily\\tiny,text=Ink] at (146,{y:.2f}) "
                f"{{\\${float(row['known_id_cost_usd']):.4f}}};",
            ]
        )

        off = row["epicure_off"]
        on = row["epicure_on"]
        commands.extend(
            [
                f"\\node[anchor=west,font=\\ttfamily\\tiny,text=Ink!74] at (157,{y + 1.25:.2f}) "
                f"{{O {_seconds(off['latency_p50_ms'])} (n={off['latency_n']})}};",
                f"\\node[anchor=west,font=\\ttfamily\\tiny,text=ReefBlue] at (157,{y - 1.25:.2f}) "
                f"{{E {_seconds(on['latency_p50_ms'])} (n={on['latency_n']})}};",
            ]
        )
        commands.append(f"\\draw[Ink!10] (0,{y - 3.5:.2f}) -- (180,{y - 3.5:.2f});")
        y -= 7.45

    totals = payload["totals"]
    commands.extend(
        [
            "\\node[anchor=west,font=\\sffamily\\tiny,text=Ink] at (0,5.8) "
            "{C=both arms normalized; P=one; F=neither. Availability is normalization, "
            "not correctness.};",
            "\\node[anchor=west,font=\\sffamily\\tiny,text=Ink] at (0,2.7) "
            "{Latency is p50 among normalized survivors only; n is shown and cross-model "
            "speed comparison is invalid.};",
            f"\\node[anchor=east,font=\\ttfamily\\tiny,text=Ink] at (180,5.8) "
            f"{{Known-ID cost: \\${_tex_escape(totals['known_id_generation_cost_usd'])}}};",
            f"\\node[anchor=east,font=\\ttfamily\\tiny,text=Alert] at (180,2.7) "
            "{Conservative source exposure: \\$"
            f"{_tex_escape(totals['conservative_dataset_source_exposure_usd'])} "
            "(+\\$"
            f"{_tex_escape(totals['no_id_exposure_increment_usd'])} no-ID; "
            "not provider spend)};",
            "\\end{tikzpicture}",
            "\\caption{Real OpenRouter $\\times$ Epicure Protocol~v1 operational evidence "
            "in frozen manifest order. Pair composition, normalized-output availability, "
            "journaled Epicure calls, and known-generation cost retain all scheduled or "
            "attempted denominators. Tool S/T includes all "
            f"{totals['epicure_journal_successful_calls']}/"
            f"{totals['epicure_journal_calls']} journaled call outcomes; normalized-answer "
            f"survivors contain {totals['normalized_survivor_trace_successful_calls']}/"
            f"{totals['normalized_survivor_trace_calls']}. Latency is the median among "
            "normalized-response "
            "survivors only, with its sample size shown; model-dependent attrition, "
            "different candidate-prompt subsets, and the 1,000-token response ceiling "
            "preclude speed or quality comparison. Known-ID cost comes from reconciled "
            "OpenRouter generation metadata. The conservative no-ID increment is budget "
            "exposure, not exact provider spend and is not allocated to a model. There "
            "are zero public and zero expert judgments: this is not a quality leaderboard "
            "and no Epicure-uplift estimate exists.}",
            "\\label{fig:real-operational}",
            "\\end{figure*}",
            "",
        ]
    )
    return "\n".join(commands)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def write_staged_outputs(
    payload: Mapping[str, Any],
    *,
    output_directory: Path,
) -> dict[str, Path]:
    digest = str(payload["content_address"]["digest"])
    json_path = output_directory / f"real-operational-figure-{digest}.json"
    csv_path = output_directory / f"real-operational-figure-{digest}.csv"
    tex_path = output_directory / f"real-operational-figure-{digest}.tex"
    outputs = {
        "json": (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        "csv": render_csv(payload).encode("utf-8"),
        "tex": render_tex(payload).encode("utf-8"),
    }
    for path, content in (
        (json_path, outputs["json"]),
        (csv_path, outputs["csv"]),
        (tex_path, outputs["tex"]),
    ):
        if path.exists() and path.read_bytes() != content:
            raise IntegrityError(f"refusing to overwrite conflicting figure artifact: {path}")
        if not path.exists():
            _atomic_write(path, content)
    return {"json": json_path, "csv": csv_path, "tex": tex_path}


def publish_outputs(
    staged: Mapping[str, Path],
    *,
    destinations: Mapping[str, Sequence[Path]],
) -> None:
    for kind, paths in destinations.items():
        source = staged[kind]
        for destination in paths:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as handle:
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, handle)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            if temporary.read_bytes() != source.read_bytes():
                temporary.unlink(missing_ok=True)
                raise IntegrityError(f"staged publication copy drifted: {destination}")
            os.replace(temporary, destination)
            destination.chmod(0o644)


def _parser() -> argparse.ArgumentParser:
    evaluation_root = Path(__file__).resolve().parents[3]
    workspace_root = evaluation_root.parents[1] / "epicure"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=evaluation_root / "flavourbench" / "artifacts" / "figures",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--paper-json",
        type=Path,
        default=evaluation_root
        / "paper"
        / "flavourbench"
        / "data"
        / "real_operational_figure.json",
    )
    parser.add_argument(
        "--paper-csv",
        type=Path,
        default=evaluation_root / "paper" / "flavourbench" / "data" / "real_operational_figure.csv",
    )
    parser.add_argument(
        "--paper-tex",
        type=Path,
        default=evaluation_root
        / "paper"
        / "flavourbench"
        / "figures"
        / "real_operational_evidence.tex",
    )
    parser.add_argument(
        "--web-json",
        type=Path,
        default=workspace_root
        / "epicure-webapp"
        / "lib"
        / "flavourbench-real-operational-figure.json",
    )
    return parser


def run() -> None:
    arguments = _parser().parse_args()
    aggregate = load_terminal_aggregate(
        arguments.aggregate.resolve(),
        expected_digest=arguments.expected_aggregate_sha256,
    )
    figure = content_address_figure(build_figure_payload(aggregate))
    staged = write_staged_outputs(
        figure,
        output_directory=arguments.output_directory.resolve(),
    )
    if arguments.publish:
        publish_outputs(
            staged,
            destinations={
                "json": (arguments.paper_json, arguments.web_json),
                "csv": (arguments.paper_csv,),
                "tex": (arguments.paper_tex,),
            },
        )
    print(
        json.dumps(
            {
                "source_aggregate_sha256": figure["source"]["aggregate_sha256"],
                "figure_sha256": figure["content_address"]["digest"],
                "model_count": len(figure["rows"]),
                "human_judgments": 0,
                "ranked": False,
                "published": bool(arguments.publish),
                "outputs": {key: str(path.resolve()) for key, path in staged.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
