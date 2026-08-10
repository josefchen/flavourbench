"""Generate compact vector figures for the retrospective FlavourBench pilot paper.

The output is LaTeX/TikZ, not a rasterized dashboard.  Every plotted value is also
written to CSV and bound to a provenance manifest.  This module deliberately does
not emit a ranked family heatmap, a cost frontier, or a definitive leaderboard.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .real_task_bank import sha256_json
from .season0_analysis import _panel_uplift_summary
from .season0_arm_corrections import validate_arm_interpretation_correction
from .season0_completion_corrections import (
    ValidatedCompletionInterpretationCorrection,
    validate_completion_interpretation_correction,
)

IMAGE_DEPENDENT_TASK_ID = "fb-s0-cookability-003"


class PilotAssetError(RuntimeError):
    """A pilot-paper input or figure invariant failed."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise PilotAssetError(f"expected a JSON object: {path}")
    return value


def _verify(document: Mapping[str, Any], label: str) -> str:
    claimed = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise PilotAssetError(f"{label} artifact hash mismatch")
    return actual


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latex(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _tex_integer(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise PilotAssetError(f"refusing empty plot data: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _short_model_name(name: str) -> str:
    for prefix in ("Anthropic ", "OpenAI ", "Google ", "Mistral ", "Alibaba "):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _flow_data(
    analysis: Mapping[str, Any],
    task_bank: Mapping[str, Any],
    review_queue: Mapping[str, Any],
    comparison_manifest: Mapping[str, Any],
    curation_audits: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    operational = analysis["operational_metrics"].values()
    success = sum(int(row["success"]) for row in operational)
    failed = sum(int(row["failed"]) for row in operational)
    reviews = review_queue.get("items")
    if not isinstance(reviews, list) or len(reviews) != len(task_bank["tasks"]):
        raise PilotAssetError("task review queue does not cover the task bank")
    completed_reviews = sum(item.get("review") is not None for item in reviews)
    if len(curation_audits) != 2:
        raise PilotAssetError("task flow requires both frozen curation audits")
    task_inputs = {
        str(row.get("curation_audit_sha256"))
        for row in task_bank.get("input_artifacts", [])
        if isinstance(row, Mapping)
    }
    audit_hashes = {_verify(audit, "curation audit") for audit in curation_audits}
    if audit_hashes != task_inputs:
        raise PilotAssetError("curation audits do not match the frozen task bank")
    candidate_count = sum(int(audit["counts"]["candidates"]) for audit in curation_audits)
    strict_consensus_count = sum(
        sum(int(value) for value in audit["counts"]["strict_consensus_by_family"].values())
        for audit in curation_audits
    )
    if candidate_count != 979 or strict_consensus_count != 462:
        raise PilotAssetError("task curation flow changed")
    consensus = [
        row
        for row in analysis["comparison_consensus"]
        if row.get("primary_consensus_available") is True
    ]
    by_track = {
        track: sum(row.get("track") == track for row in consensus)
        for track in ("model_arena", "epicure_uplift")
    }
    comparison_counts = comparison_manifest.get("counts")
    if not isinstance(comparison_counts, Mapping):
        raise PilotAssetError("comparison manifest has no count reconciliation")
    planned = int(comparison_counts.get("comparisons") or 0)
    failed_arm_exclusions = int(comparison_counts.get("failed_arm_exclusions") or 0)
    identity_leak_exclusions = int(comparison_counts.get("identity_leak_exclusions") or 0)
    source_admitted = int(comparison_counts.get("judgable") or 0)
    source_dual_answer = planned - failed_arm_exclusions
    if source_dual_answer - identity_leak_exclusions != source_admitted:
        raise PilotAssetError("comparison exclusion flow does not reconcile")
    correction_count = int(analysis["arm_validation"]["completion_interpretation_correction_count"])
    collector_accepted = int(analysis["arm_validation"]["collector_accepted_arms"])
    collector_rejected = int(analysis["arm_validation"]["collector_rejected_arms"])
    effective_admitted = int(analysis["counts"]["effective_judgable_comparisons"])
    completion_pair_exclusions = int(
        analysis["counts"]["incomplete_final_response_comparison_exclusions"]
    )
    if (
        collector_accepted + collector_rejected != int(analysis["counts"]["scored_arms"])
        or collector_accepted - correction_count != success
        or collector_rejected + correction_count != failed
        or source_admitted - completion_pair_exclusions != effective_admitted
    ):
        raise PilotAssetError("completion reinterpretation flow does not reconcile")
    rows: list[dict[str, object]] = [
        {"panel": "task", "stage": "source_candidates", "count": candidate_count},
        {
            "panel": "task",
            "stage": "strict_llm_curator_agreement",
            "count": strict_consensus_count,
        },
        {"panel": "task", "stage": "selected_tasks", "count": len(task_bank["tasks"])},
        {
            "panel": "task",
            "stage": "qualified_human_reviewed",
            "count": completed_reviews,
        },
        {
            "panel": "evaluation",
            "stage": "response_arms",
            "count": analysis["counts"]["scored_arms"],
        },
        {
            "panel": "evaluation",
            "stage": "collector_accepted_arms",
            "count": collector_accepted,
        },
        {
            "panel": "evaluation",
            "stage": "collector_rejected_arms",
            "count": collector_rejected,
        },
        {
            "panel": "evaluation",
            "stage": "completion_reclassified_arms",
            "count": correction_count,
        },
        {"panel": "evaluation", "stage": "effective_complete_arms", "count": success},
        {"panel": "evaluation", "stage": "effective_failed_arms", "count": failed},
        {"panel": "evaluation", "stage": "planned_comparisons", "count": planned},
        {"panel": "evaluation", "stage": "failed_arm_exclusions", "count": failed_arm_exclusions},
        {
            "panel": "evaluation",
            "stage": "source_dual_answer_comparisons",
            "count": source_dual_answer,
        },
        {
            "panel": "evaluation",
            "stage": "identity_leak_exclusions",
            "count": identity_leak_exclusions,
        },
        {
            "panel": "evaluation",
            "stage": "source_judging_admitted_comparisons",
            "count": source_admitted,
        },
        {
            "panel": "evaluation",
            "stage": "completion_pair_exclusions",
            "count": completion_pair_exclusions,
        },
        {
            "panel": "evaluation",
            "stage": "effective_judging_admitted_comparisons",
            "count": effective_admitted,
        },
        {
            "panel": "evaluation",
            "stage": "no_primary_consensus",
            "count": effective_admitted - len(consensus),
        },
        {"panel": "evaluation", "stage": "primary_consensus", "count": len(consensus)},
        {"panel": "evaluation", "stage": "model_arena_consensus", "count": by_track["model_arena"]},
        {
            "panel": "evaluation",
            "stage": "paired_tool_consensus",
            "count": by_track["epicure_uplift"],
        },
    ]
    if success + failed != int(analysis["counts"]["scored_arms"]):
        raise PilotAssetError("arm flow does not reconcile")
    if len(consensus) != int(analysis["counts"]["consensus_available"]):
        raise PilotAssetError("consensus flow does not reconcile")
    if planned != int(analysis["counts"]["comparison_manifest_rows"]):
        raise PilotAssetError("analysis and comparison manifest planning counts differ")
    if effective_admitted != int(analysis["counts"]["consensus_rows"]):
        raise PilotAssetError("analysis and comparison manifest admitted counts differ")
    return rows


def _flow_figure(rows: Sequence[Mapping[str, object]]) -> str:
    values = {(row["panel"], row["stage"]): int(row["count"]) for row in rows}
    candidate_n = values[("task", "source_candidates")]
    strict_n = values[("task", "strict_llm_curator_agreement")]
    task_n = values[("task", "selected_tasks")]
    reviewed = values[("task", "qualified_human_reviewed")]
    arms = values[("evaluation", "response_arms")]
    collector_accepted = values[("evaluation", "collector_accepted_arms")]
    collector_rejected = values[("evaluation", "collector_rejected_arms")]
    completion_reclassified = values[("evaluation", "completion_reclassified_arms")]
    success = values[("evaluation", "effective_complete_arms")]
    failed = values[("evaluation", "effective_failed_arms")]
    planned = values[("evaluation", "planned_comparisons")]
    source_admitted = values[("evaluation", "source_judging_admitted_comparisons")]
    completion_pair_exclusions = values[("evaluation", "completion_pair_exclusions")]
    admitted = values[("evaluation", "effective_judging_admitted_comparisons")]
    no_consensus = values[("evaluation", "no_primary_consensus")]
    consensus = values[("evaluation", "primary_consensus")]
    arena = values[("evaluation", "model_arena_consensus")]
    uplift = values[("evaluation", "paired_tool_consensus")]
    arms_text = _tex_integer(arms)
    success_text = _tex_integer(success)
    failed_text = _tex_integer(failed)
    planned_text = _tex_integer(planned)
    collector_accepted_text = _tex_integer(collector_accepted)
    collector_rejected_text = _tex_integer(collector_rejected)
    completion_reclassified_text = _tex_integer(completion_reclassified)
    source_admitted_text = _tex_integer(source_admitted)
    completion_pair_exclusions_text = _tex_integer(completion_pair_exclusions)
    admitted_text = _tex_integer(admitted)
    no_consensus_text = _tex_integer(no_consensus)
    consensus_text = _tex_integer(consensus)
    return rf"""
\begin{{tikzpicture}}[x=1cm,y=1cm,>=latex,every node/.style={{font=\fontsize{{8.0}}{{9.0}}\selectfont}}]
  \node[anchor=west,font=\bfseries\scriptsize] at (0,3.35) {{(a) Task sample}};
  \node[draw=black!55,minimum width=1.48cm,minimum height=0.88cm,align=center] (source) at (0.85,2.35)
    {{source\\candidates\\$n={candidate_n}$}};
  \node[draw=black!55,minimum width=1.72cm,minimum height=0.88cm,align=center] (curated) at (2.75,2.35)
    {{strict LLM-curator\\agreement\\$n={strict_n}$}};
  \node[draw=black!55,minimum width=1.62cm,minimum height=0.88cm,align=center] (selected) at (4.82,2.35)
    {{deterministic\\selection\\$n={task_n}$}};
  \node[draw=black!55,minimum width=1.72cm,minimum height=0.88cm,align=center] (review) at (6.85,2.35)
    {{qualified human\\item reviews\\${reviewed}/{task_n}$}};
  \draw[->,black!65] (source) -- (curated);
  \draw[->,black!65] (curated) -- (selected);
  \draw[->,black!65] (selected) -- (review);

  \node[anchor=west,font=\bfseries\scriptsize] at (7.60,3.35) {{(b) Completion audit}};
  \node[draw=black!55,text width=1.35cm,minimum height=0.80cm,inner sep=2pt,align=center] (arms) at (8.55,2.35)
    {{${arms_text}$\\response arms}};
  \node[draw=black!55,text width=1.85cm,minimum height=0.92cm,inner sep=2pt,align=center] (collector) at (11.05,2.35)
    {{collector accepted ${collector_accepted_text}$\\rejected ${collector_rejected_text}$}};
  \node[draw=FBBlue,text width=1.90cm,minimum height=0.92cm,inner sep=2pt,align=center] (audit) at (13.80,2.35)
    {{finish-reason audit\\${completion_reclassified_text}$ reclassified}};
  \node[draw=black!55,text width=1.85cm,minimum height=0.92cm,inner sep=2pt,align=center] (valid) at (16.35,2.35)
    {{${success_text}$ complete\\${failed_text}$ failed}};
  \draw[->,black!65] (arms) -- (collector);
  \draw[->,black!65] (collector) -- (audit);
  \draw[->,black!65] (audit) -- (valid);

  \node[draw=black!55,text width=1.35cm,minimum height=0.76cm,inner sep=2pt,align=center] (pairs) at (1.05,0.55)
    {{${planned_text}$\\planned pairs}};
  \node[draw=black!55,text width=1.90cm,minimum height=0.82cm,inner sep=2pt,align=center] (source) at (4.00,0.55)
    {{${source_admitted_text}$ originally\\judge-admitted}};
  \node[draw=FBBlue,text width=2.00cm,minimum height=0.82cm,inner sep=2pt,align=center] (excluded) at (7.15,0.55)
    {{${completion_pair_exclusions_text}$ later excluded\\for incomplete arm}};
  \node[draw=black!55,text width=1.90cm,minimum height=0.82cm,inner sep=2pt,align=center] (admitted) at (10.30,0.55)
    {{${admitted_text}$ effective\\judge cohort}};
  \node[draw=black!55,text width=1.82cm,minimum height=0.82cm,inner sep=2pt,align=center] (consensus) at (14.55,1.05)
    {{${consensus_text}$ consensus\\{arena} endpoint; {uplift} paired}};
  \node[draw=black!55,text width=1.82cm,minimum height=0.72cm,inner sep=2pt,align=center] (no-consensus) at (14.55,0.02)
    {{${no_consensus_text}$ without\\primary consensus}};
  \draw[->,black!65] (pairs) -- (source);
  \draw[->,black!65] (source) -- (excluded);
  \draw[->,black!65] (excluded) -- (admitted);
  \draw[->,black!65] (admitted.east) -- (consensus.west);
  \draw[->,black!65] (admitted.east) -- (no-consensus.west);
  \draw[->,black!50,densely dashed] (valid.south) to[out=250,in=70] (admitted.north);
\end{{tikzpicture}}
"""


def _system_architecture_figure() -> str:
    """Render the implemented execution boundary as a plain vector schematic."""

    return r"""
\begin{tikzpicture}[
  x=1cm,y=1cm,>=latex,
  every node/.style={font=\fontsize{7.8}{8.8}\selectfont},
  box/.style={draw=black!62,fill=white,minimum height=0.82cm,align=center},
  flow/.style={->,draw=black!68,line width=0.55pt},
  evidence/.style={->,draw=FBBlue,densely dashed,line width=0.75pt}
]
  \node[anchor=west,font=\bfseries\scriptsize] at (0,5.90) {(a) Admission, evidence, and release};
  \node[box,minimum width=2.02cm] (participant) at (1.10,4.45) {participant or\\research user};
  \node[box,minimum width=2.10cm] (web) at (3.75,4.45) {same-origin web\\and API proxy};
  \node[box,minimum width=2.20cm] (api) at (6.50,4.45) {FlavourBench API\\admission and reveal};
  \node[box,minimum width=2.35cm] (store) at (9.30,4.45) {PostgreSQL\\jobs and evidence};
  \node[box,minimum width=2.25cm] (snapshot) at (13.20,4.45) {versioned snapshot\\private or released};
  \draw[flow,<->] (participant) -- (web);
  \draw[flow,<->] (web) -- (api);
  \draw[flow] (api) -- (store);
  \node[anchor=south,fill=white,inner sep=0.8pt,
    font=\fontsize{7.8}{8.8}\selectfont] at (7.90,4.88) {admit / enqueue};
  \draw[evidence] (store) -- (snapshot);
  \node[anchor=south,fill=white,inner sep=0.8pt,
    font=\fontsize{7.8}{8.8}\selectfont] at (11.25,4.88) {digest-bound snapshot};
  \draw[flow] (snapshot.north) -- ++(0,0.72) -| (api.north);
  \node[fill=white,inner sep=1pt] at (9.85,5.35)
    {poll / blinded result / identity reveal};

  \node[anchor=west,font=\bfseries\scriptsize] at (0,3.05) {(b) Private asynchronous execution};
  \draw[black!42,densely dashed] (2.65,0.55) rectangle (15.35,2.82);
  \node[anchor=north west,fill=white,inner sep=1pt,text=black!65] at (2.78,2.78)
    {private execution boundary};
  \node[box,minimum width=2.35cm] (mcp) at (4.10,1.55) {Epicure MCP\\read-only tools};
  \node[box,minimum width=2.55cm] (worker) at (9.30,1.55)
    {asynchronous worker\\frozen run contract};
  \node[draw=black!62,fill=white,minimum width=1.20cm,minimum height=0.64cm,
    inner sep=1pt,align=center] (route) at (11.50,1.55) {route\\selector};
  \node[box,minimum width=2.18cm] (bedrock) at (14.10,2.05)
    {Amazon Bedrock\\fixed endpoint};
  \node[box,minimum width=2.18cm] (openrouter) at (14.10,1.05)
    {OpenRouter\\fixed provider route};
  \draw[flow,<->] (mcp) -- (worker);
  \node[fill=white,inner sep=1pt,text=black!65] at (6.70,1.93) {tool request / result};
  \draw[flow] (worker) -- (route);
  \draw[flow] (route) |- (bedrock.west);
  \draw[flow] (route) |- (openrouter.west);
  \draw[flow] ([xshift=-0.28cm]store.south) --
    node[pos=0.43,left=2pt,fill=white,inner sep=0.8pt]{claim job}
    ([xshift=-0.28cm]worker.north);
  \draw[evidence] ([xshift=0.28cm]worker.north) --
    node[pos=0.57,right=2pt,fill=white,inner sep=0.8pt,align=left]
      {record output\\and trace}
    ([xshift=0.28cm]store.south);
  \node[anchor=west,text=black!68] at (0.12,0.12)
    {Dashed blue arrows denote digest-bound evidence; solid arrows denote requests or control.};
\end{tikzpicture}
"""


def _effective_comparison_manifest(
    comparison_manifest: Mapping[str, Any],
    completion_interpretation: ValidatedCompletionInterpretationCorrection,
) -> dict[str, Any]:
    """Apply the derived completion policy without mutating the frozen manifest."""

    effective = copy.deepcopy(dict(comparison_manifest))
    corrected_ids = set(completion_interpretation.arm_ids)
    encountered: set[str] = set()
    for comparison in effective.get("comparisons", []):
        if not isinstance(comparison, dict):
            raise PilotAssetError("comparison manifest contains a non-object row")
        for side in ("left", "right"):
            arm = comparison.get(side)
            if not isinstance(arm, dict):
                raise PilotAssetError("comparison manifest contains an invalid arm")
            arm_id = str(arm.get("arm_id") or "")
            if arm_id not in corrected_ids:
                continue
            encountered.add(arm_id)
            arm["source_status"] = arm.get("status")
            arm["status"] = "failed"
            arm["completion_interpretation"] = "incomplete_final_response"
    if encountered != corrected_ids:
        raise PilotAssetError("completion correction is not represented by the pair manifest")
    return effective


def _study_design_figure() -> str:
    """Render the measurement pipeline and its separate reported outcomes."""

    return r"""
\begin{tikzpicture}[
  x=1cm,y=1cm,>=latex,
  every node/.style={font=\fontsize{7.6}{8.6}\selectfont},
  box/.style={draw=black!62,fill=white,minimum height=1.22cm,align=center,inner sep=2.5pt},
  flow/.style={->,draw=black!68,line width=0.60pt},
  inference/.style={->,draw=FBBlue,line width=0.75pt}
]
  \node[anchor=west,font=\bfseries\scriptsize] at (0,5.12)
    {Measurement pipeline};
  \node[box,text width=2.10cm] (source) at (1.20,3.55)
    {human-sourced tasks\\constraints; reference};
  \node[box,text width=2.10cm] (validity) at (3.78,3.55)
    {prompt-only review\\validity sealed before\\answers are shown};

  \draw[black!55,fill=white] (5.22,2.55) rectangle (9.02,4.55);
  \node[anchor=north,font=\bfseries] at (7.12,4.43) {blinded matched arms};
  \node[draw=black!42,fill=black!3,minimum width=3.34cm,minimum height=0.45cm,
    align=center,inner sep=1pt] at (7.11,3.72)
    {arena: \(m_1+E\) versus \(m_2+E\)};
  \node[draw=black!42,fill=black!3,minimum width=3.34cm,minimum height=0.45cm,
    align=center,inner sep=1pt] at (7.11,3.10)
    {uplift: \(m_1\) versus \(m_1+E\)};
  \node[anchor=north,text=black!65] at (7.11,2.43) {left/right randomized};

  \node[box,text width=2.10cm] (ballots) at (10.48,3.55)
    {independent ballots\\preference; rubric\\failure tags};
  \node[box,text width=2.58cm] (fit) at (13.45,3.55)
    {family-aware BT\\paired ordinal model\\clustered intervals};

  \node[anchor=west,font=\bfseries\scriptsize] at (16.05,5.12)
    {Reported separately};
  \node[draw=black!62,fill=white,text width=3.28cm,minimum height=0.82cm,
    align=center,inner sep=2pt] (quality) at (18.15,4.45)
    {\textbf{quality}\\preference, rubric, intervals, \(n\)};
  \node[draw=black!62,fill=white,text width=3.28cm,minimum height=0.82cm,
    align=center,inner sep=2pt] (reliability) at (18.15,3.45)
    {\textbf{reliability}\\completion, tool success, failures};
  \node[draw=black!62,fill=white,text width=3.28cm,minimum height=0.82cm,
    align=center,inner sep=2pt] (efficiency) at (18.15,2.45)
    {\textbf{cost and latency}\\known-USD coverage stated};

  \draw[flow] (source) -- (validity);
  \draw[flow] (validity) -- (5.22,3.55);
  \draw[flow] (9.02,3.55) -- (ballots);
  \draw[inference] (ballots) -- (fit);
  \draw[inference] (fit.east) -- ++(0.45,0) |- (quality.west);
  \draw[flow] (fit.east) -- ++(0.45,0) |- (reliability.west);
  \draw[flow] (fit.east) -- ++(0.45,0) |- (efficiency.west);

  \node[draw=black!32,fill=black!2,text=black!72,text width=19.25cm,
    minimum height=0.68cm,align=center,inner sep=2pt] at (10.03,1.08)
    {Every assigned arm enters the reliability, latency, and cost denominators. Preference fitting
     requires two complete responses and an eligible ballot.};
\end{tikzpicture}
"""


def _endpoint_manifest_table(model_manifest: Mapping[str, Any]) -> str:
    models = model_manifest.get("models")
    if not isinstance(models, list) or len(models) != 12:
        raise PilotAssetError("endpoint appendix requires the frozen 12-model manifest")
    rows = sorted(models, key=lambda row: str(row.get("season_model_id") or ""))
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}r p{0.23\textwidth} X@{}}",
        r"\toprule",
        r"& Endpoint label & Route identity (OpenRouter: request $\rightarrow$ observed @ provider) \\",
        r"\midrule",
    ]
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        season_model_id = str(row.get("season_model_id") or "")
        if not season_model_id or season_model_id in seen_ids:
            raise PilotAssetError("model manifest has a missing or duplicate season-model ID")
        seen_ids.add(season_model_id)
        provider = str(row.get("provider") or "")
        request_id = str(row.get("requested_endpoint_id") or "")
        canonical_id = str(row.get("canonical_model_id") or "")
        provider_slug = str(row.get("provider_slug") or "")
        display_name = _short_model_name(str(row.get("display_name") or ""))
        if not display_name or not request_id or not canonical_id:
            raise PilotAssetError(f"incomplete endpoint manifest row: {season_model_id}")
        if provider == "bedrock":
            route = rf"Bedrock: \path{{{_latex(request_id)}}}"
        elif provider == "openrouter" and provider_slug:
            route = (
                rf"\path{{{_latex(request_id)}}} $\rightarrow$ "
                rf"\path{{{_latex(canonical_id)}}} @ \path{{{_latex(provider_slug)}}}"
            )
        else:
            raise PilotAssetError(f"unsupported frozen provider: {provider}")
        lines.append(rf"{index:02d} & {_latex(display_name)} & {route} \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return "\n".join(lines)


def _measurement_data(analysis: Mapping[str, Any]) -> list[dict[str, object]]:
    def wilson(numerator: int, denominator: int) -> tuple[float, float]:
        if denominator <= 0:
            raise PilotAssetError("measurement interval has a zero denominator")
        z = 1.959963984540054
        proportion = numerator / denominator
        denominator_term = 1.0 + z**2 / denominator
        centre = (proportion + z**2 / (2 * denominator)) / denominator_term
        radius = (
            z
            * math.sqrt(proportion * (1 - proportion) / denominator + z**2 / (4 * denominator**2))
            / denominator_term
        )
        return max(0.0, centre - radius), min(1.0, centre + radius)

    rows: list[dict[str, object]] = []
    judges = analysis["judge_diagnostics"]["judges"]
    names = {
        "judge-anthropic-haiku-4-5": "Claude Haiku 4.5",
        "judge-anthropic-sonnet-4-6": "Claude Sonnet 4.6",
        "judge-mistral-devstral-2-123b": "Devstral 2 123B",
        "judge-alibaba-qwen3-next-80b": "Qwen3 Next 80B",
    }
    for judge_id in names:
        row = judges[judge_id]
        planned = int(row["planned_comparisons"])
        completed = planned - int(row.get("incomplete_comparisons") or 0)
        consistent = int(row.get("orientation_consistent") or 0)
        disagreements = int(row.get("orientation_disagreement") or 0)
        nonself_consistent = consistent - int(row.get("self_judgments") or 0)
        if completed != consistent + disagreements:
            raise PilotAssetError(f"judge completion counts do not reconcile: {judge_id}")
        if not 0 <= nonself_consistent <= consistent:
            raise PilotAssetError(f"judge non-self counts do not reconcile: {judge_id}")
        for metric, numerator, denominator in (
            ("both_orientations_complete", completed, planned),
            ("agreement_given_completion", consistent, completed),
            ("eligible_vote_yield", nonself_consistent, planned),
        ):
            lower, upper = wilson(numerator, denominator)
            rows.append(
                {
                    "panel": "judge",
                    "label": names[judge_id],
                    "metric": metric,
                    "rate": numerator / denominator,
                    "numerator": numerator,
                    "denominator": denominator,
                    "wilson_lower_95": lower,
                    "wilson_upper_95": upper,
                }
            )
    primary_n = int(analysis["counts"]["consensus_available"])
    judgable = int(analysis["counts"]["consensus_rows"])
    family_diag = analysis["judge_family_balanced_sensitivity"]["diagnostics"]
    verbosity = analysis["verbosity_diagnostics"]
    diagnostics = [
        {
            "panel": "diagnostic",
            "label": "Primary consensus",
            "rate": primary_n / judgable,
            "numerator": primary_n,
            "denominator": judgable,
        },
        {
            "panel": "diagnostic",
            "label": "Cross-family-admitted consensus",
            "rate": family_diag["coverage"],
            "numerator": family_diag["consensus_available"],
            "denominator": family_diag["rows"],
        },
        {
            "panel": "diagnostic",
            "label": "Longer answer selected",
            "rate": verbosity["preferred_longer_rate_among_unequal"],
            "numerator": verbosity["preferred_longer"],
            "denominator": verbosity["unequal_length_preferences"],
        },
    ]
    for row in diagnostics:
        lower, upper = wilson(int(row["numerator"]), int(row["denominator"]))
        row["wilson_lower_95"] = lower
        row["wilson_upper_95"] = upper
    rows.extend(diagnostics)
    return rows


def _bar_panel(
    rows: Sequence[Mapping[str, object]],
    *,
    x0: float,
    y0: float,
    width: float,
    row_gap: float,
) -> list[str]:
    lines: list[str] = []
    for index, row in enumerate(rows):
        y = y0 - index * row_gap
        rate = float(row["rate"])
        bar_end = x0 + width * rate
        label = _latex(row["label"])
        numerator = int(row["numerator"])
        denominator = int(row["denominator"])
        lines.extend(
            [
                rf"\node[anchor=east] at ({x0 - 0.16:.3f},{y:.3f}) {{{label}}};",
                rf"\draw[black!25,line width=0.5pt] ({x0:.3f},{y:.3f}) -- ({x0 + width:.3f},{y:.3f});",
                rf"\draw[FBBlue,line width=3.2pt] ({x0:.3f},{y:.3f}) -- ({bar_end:.3f},{y:.3f});",
                rf"\node[anchor=west] at ({x0 + width + 0.12:.3f},{y:.3f}) "
                rf"{{{100 * rate:.1f}\%\;({numerator}/{denominator})}};",
            ]
        )
    for tick in (0.0, 0.5, 1.0):
        x = x0 + width * tick
        lines.append(
            rf"\node[anchor=north,text=black!65] at ({x:.3f},{y0 - len(rows) * row_gap + 0.18:.3f}) {{{int(100 * tick)}\%}};"
        )
    return lines


def _measurement_figure(rows: Sequence[Mapping[str, object]]) -> str:
    judge_rows = [row for row in rows if row["panel"] == "judge"]
    diagnostic_rows = [row for row in rows if row["panel"] == "diagnostic"]
    judge_names = list(dict.fromkeys(str(row["label"]) for row in judge_rows))
    by_judge_metric = {(str(row["label"]), str(row["metric"])): row for row in judge_rows}
    plot_left, plot_right = 2.2, 6.75

    def x(rate: float) -> float:
        return plot_left + rate * (plot_right - plot_left)

    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,every node/.style={font=\fontsize{8.0}{9.0}\selectfont}]",
        r"\node[anchor=west,font=\bfseries\scriptsize] at (0,6.68) {(a) Judge-pair eligibility decomposition};",
        r"\fill[FBBlue] (0.12,6.16) circle (1.45pt);",
        r"\node[anchor=west] at (0.28,6.16) {complete pair};",
        r"\draw[black,line width=0.65pt,fill=white] (2.64,6.10) rectangle (2.76,6.22);",
        r"\node[anchor=west] at (2.86,6.16) {agreement};",
        r"\fill[black] (5.20,6.09) -- (5.34,6.09) -- (5.27,6.23) -- cycle;",
        r"\node[anchor=west] at (5.41,6.16) {eligible vote};",
    ]
    for tick in (0.0, 0.5, 1.0):
        tick_x = x(tick)
        tick_anchor = {0.0: "north west", 0.5: "north", 1.0: "north east"}[tick]
        lines.extend(
            [
                rf"\draw[black!18] ({tick_x:.3f},3.02) -- ({tick_x:.3f},5.73);",
                rf"\node[anchor={tick_anchor},text=black!65] at ({tick_x:.3f},2.98) {{{int(100 * tick)}\%}};",
            ]
        )
    metric_offsets = {
        "both_orientations_complete": 0.11,
        "agreement_given_completion": 0.0,
        "eligible_vote_yield": -0.11,
    }
    for index, judge_name in enumerate(judge_names):
        y = 5.45 - index * 0.64
        lines.append(rf"\node[anchor=east] at (2.02,{y:.3f}) {{{_latex(judge_name)}}};")
        for metric, y_offset in metric_offsets.items():
            row = by_judge_metric[(judge_name, metric)]
            point_x = x(float(row["rate"]))
            point_y = y + y_offset
            if metric == "both_orientations_complete":
                lines.append(rf"\fill[FBBlue] ({point_x:.3f},{point_y:.3f}) circle (1.45pt);")
            elif metric == "agreement_given_completion":
                lines.append(
                    rf"\draw[black,line width=0.65pt,fill=white] "
                    rf"({point_x - 0.06:.3f},{point_y - 0.06:.3f}) rectangle "
                    rf"({point_x + 0.06:.3f},{point_y + 0.06:.3f});"
                )
            else:
                lines.append(
                    rf"\fill[black] ({point_x - 0.07:.3f},{point_y - 0.06:.3f}) -- "
                    rf"({point_x + 0.07:.3f},{point_y - 0.06:.3f}) -- "
                    rf"({point_x:.3f},{point_y + 0.08:.3f}) -- cycle;"
                )
    lines.append(
        r"\node[anchor=west,font=\bfseries\scriptsize] at (0,2.42) {(b) Realized cohort diagnostics};"
    )
    diag_left, diag_right = 2.45, 4.85
    for index, row in enumerate(diagnostic_rows):
        y = 1.72 - index * 0.64
        rate = float(row["rate"])
        end = diag_left + rate * (diag_right - diag_left)
        lines.extend(
            [
                rf"\node[anchor=east] at ({diag_left - 0.15:.3f},{y:.3f}) "
                rf"{{{_latex(row['label'])}}};",
                rf"\draw[black!25,line width=0.5pt] ({diag_left:.3f},{y:.3f}) -- "
                rf"({diag_right:.3f},{y:.3f});",
                rf"\draw[FBBlue,line width=2.6pt] ({diag_left:.3f},{y:.3f}) -- ({end:.3f},{y:.3f});",
                rf"\node[anchor=west] at ({diag_right + 0.10:.3f},{y:.3f}) "
                rf"{{{100 * rate:.1f}\%}};",
                rf"\node[anchor=west,text=black!70] at (5.95,{y:.3f}) "
                rf"{{{int(row['numerator'])}/{int(row['denominator'])}}};",
            ]
        )
    for tick in (0.0, 0.5, 1.0):
        tick_x = diag_left + tick * (diag_right - diag_left)
        lines.append(
            rf"\node[anchor=north,text=black!65] at ({tick_x:.3f},-0.10) "
            rf"{{{int(100 * tick)}\%}};"
        )
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _model_data(analysis: Mapping[str, Any]) -> list[dict[str, object]]:
    leaderboard = {row["season_model_id"]: row for row in analysis["model_leaderboard"]}
    bootstrap = analysis["arena_task_cluster_bootstrap"]["models"]
    rows: list[dict[str, object]] = []
    for model_id in sorted(leaderboard):
        fitted = leaderboard[model_id]
        boot = bootstrap[model_id]
        complete_separation = (
            int(fitted["wins"]) == 0 and int(fitted["ties"]) == 0 and int(fitted["losses"]) > 0
        ) or (int(fitted["losses"]) == 0 and int(fitted["ties"]) == 0 and int(fitted["wins"]) > 0)
        rows.append(
            {
                "season_model_id": model_id,
                "model": fitted["display_name"],
                "bootstrap_median": boot["rating_median"],
                "bootstrap_lower": boot["rating_interval_lower"],
                "bootstrap_upper": boot["rating_interval_upper"],
                "highest_resample_fraction": boot["rank_one_probability"],
                "primary_rating": fitted["rating"],
                "primary_lower": fitted["rating_lower"],
                "primary_upper": fitted["rating_upper"],
                "wins": fitted["wins"],
                "ties": fitted["ties"],
                "losses": fitted["losses"],
                "n": fitted["comparisons"],
                "complete_separation": complete_separation,
            }
        )
    return rows


def _primary_model_table(rows: Sequence[Mapping[str, object]]) -> str:
    ordered = list(rows)
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}X r c c r@{}}",
        r"\toprule",
        r"Endpoint & BT diagnostic & Sandwich 95\% CI & W/T/L & $n$ \\",
        r"\midrule",
    ]
    for row in ordered:
        name = _latex(_short_model_name(str(row["model"])))
        if row["complete_separation"]:
            name += r"\textsuperscript{\dag}"
            rating = "not estimable"
            interval = "unbounded"
        else:
            rating = f"{float(row['primary_rating']):.0f}"
            interval = f"[{float(row['primary_lower']):.0f}, {float(row['primary_upper']):.0f}]"
        wtl = f"{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}"
        lines.append(rf"{name} & {rating} & {interval} & {wtl} & {int(row['n'])} \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return "\n".join(lines)


def _model_figure(rows: Sequence[Mapping[str, object]]) -> str:
    finite_rows = [row for row in rows if not row["complete_separation"]]
    separated_rows = [row for row in rows if row["complete_separation"]]
    if not finite_rows:
        raise PilotAssetError("model uncertainty figure has no finite shared-scale rows")

    plot_left, plot_right = 5.55, 12.85
    centered_bounds = [
        float(row[key]) - 1000.0
        for row in finite_rows
        for key in ("bootstrap_lower", "bootstrap_upper")
    ]
    tick_step = 1000
    low = math.floor(min(centered_bounds) / tick_step) * tick_step
    high = math.ceil(max(centered_bounds) / tick_step) * tick_step
    if not low < high:
        raise PilotAssetError("model interval axis has no range")
    if any(value < low or value > high for value in centered_bounds):
        raise PilotAssetError("model interval would be clipped")
    y_top = 7.25
    y_bottom = 1.85
    gap = (y_top - y_bottom) / max(1, len(finite_rows) - 1)

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,every node/.style={font=\fontsize{8.0}{9.0}\selectfont}]",
        rf"\node[anchor=east,font=\bfseries\scriptsize] at ({plot_left - 0.18:.3f},8.0) {{Endpoint}};",
        rf"\node[font=\bfseries\scriptsize] at ({(plot_left + plot_right) / 2:.3f},8.0) {{task-bootstrap BT score (shifted; shared scale)}};",
        r"\node[font=\bfseries\scriptsize] at (13.65,8.0) {row W/T/L};",
        r"\node[font=\bfseries\scriptsize] at (14.65,8.0) {$n$};",
        r"\node[font=\bfseries\scriptsize] at (15.75,8.0) {highest-resample frac.};",
    ]
    for tick in range(int(low), int(high) + tick_step, tick_step):
        tx = x(float(tick))
        lines.extend(
            [
                rf"\draw[black!17] ({tx:.3f},1.55) -- ({tx:.3f},7.62);",
                rf"\node[anchor=north,text=black!65] at ({tx:.3f},1.48) {{{tick}}};",
            ]
        )
    ref_x = x(0.0)
    lines.append(rf"\draw[black!55,densely dashed] ({ref_x:.3f},1.55) -- ({ref_x:.3f},7.62);")
    for index, row in enumerate(finite_rows):
        y = y_top - index * gap
        name = _latex(_short_model_name(str(row["model"])))
        wtl = f"{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}"
        highest_fraction = float(row["highest_resample_fraction"])
        lines.extend(
            [
                rf"\node[anchor=east] at ({plot_left - 0.18:.3f},{y:.3f}) {{{name}}};",
                rf"\node at (13.65,{y:.3f}) {{{wtl}}};",
                rf"\node at (14.65,{y:.3f}) {{{int(row['n'])}}};",
                rf"\node at (15.75,{y:.3f}) {{{100 * highest_fraction:.1f}\%}};",
            ]
        )
        lower = x(float(row["bootstrap_lower"]) - 1000.0)
        upper = x(float(row["bootstrap_upper"]) - 1000.0)
        median = x(float(row["bootstrap_median"]) - 1000.0)
        lines.extend(
            [
                rf"\draw[black!70,line width=0.7pt] ({lower:.3f},{y:.3f}) -- ({upper:.3f},{y:.3f});",
                rf"\draw[black!70,line width=0.7pt] ({lower:.3f},{y - 0.07:.3f}) -- ({lower:.3f},{y + 0.07:.3f});",
                rf"\draw[black!70,line width=0.7pt] ({upper:.3f},{y - 0.07:.3f}) -- ({upper:.3f},{y + 0.07:.3f});",
                rf"\fill[FBBlue] ({median:.3f},{y:.3f}) circle (1.6pt);",
            ]
        )

    if separated_rows:
        lines.extend(
            [
                r"\draw[black!38,line width=0.45pt] (0.05,1.08) -- (16.10,1.08);",
                r"\node[anchor=west,font=\bfseries\scriptsize,text=black!70] at (0.05,0.86) {Complete separation (reported outside the shared axis)};",
            ]
        )
        for index, row in enumerate(separated_rows):
            y = 0.34 - index * 0.48
            name = _latex(_short_model_name(str(row["model"]))) + r"\textsuperscript{*}"
            wtl = f"{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}"
            highest_fraction = float(row["highest_resample_fraction"])
            median = float(row["bootstrap_median"]) - 1000.0
            lower = float(row["bootstrap_lower"]) - 1000.0
            upper = float(row["bootstrap_upper"]) - 1000.0
            lines.extend(
                [
                    rf"\node[anchor=east] at ({plot_left - 0.18:.3f},{y:.3f}) {{{name}}};",
                    rf"\node[anchor=west,align=left,text width=6.4cm,text=black!72] "
                    rf"at ({plot_left:.3f},{y:.3f}) {{unpenalized BT: not estimable\\"
                    rf"ridge bootstrap: {median:.0f} [{lower:.0f}, {upper:.0f}]}};",
                    rf"\node at (13.65,{y:.3f}) {{{wtl}}};",
                    rf"\node at (14.65,{y:.3f}) {{{int(row['n'])}}};",
                    rf"\node at (15.75,{y:.3f}) {{{100 * highest_fraction:.1f}\%}};",
                ]
            )

    footnote_y = -0.28 - max(0, len(separated_rows) - 1) * 0.48
    lines.extend(
        [
            rf"\node[anchor=west,text=black!70] at (0.05,{footnote_y:.3f}) "
            r"{All rows have $n<100$. Intervals resample tasks. \textsuperscript{*}The unpenalized fit is unbounded; its ridge sensitivity does not set the shared scale.};",
            r"\end{tikzpicture}",
        ]
    )
    return "\n".join(lines)


def _standardized_uplift_sensitivities(
    analysis: Mapping[str, Any],
) -> dict[str, object]:
    score_rows: list[tuple[str, str, float]] = []
    for row in analysis["comparison_consensus"]:
        if row.get("track") != "epicure_uplift":
            continue
        choice = row.get("primary_consensus_choice")
        if choice in (None, "both_bad"):
            continue
        if choice == "tie":
            score = 0.5
        elif choice in ("left", "right"):
            score = 1.0 if row[choice]["condition"] == "epicure_on" else 0.0
        else:
            raise PilotAssetError(f"unexpected uplift consensus choice: {choice}")
        score_rows.append((str(row["task_id"]), str(row["season_model_id"]), score))

    scores_by_task: dict[str, list[tuple[str, float]]] = defaultdict(list)
    scores_by_model: dict[str, list[float]] = defaultdict(list)
    for task_id, model_id, score in score_rows:
        scores_by_task[task_id].append((model_id, score))
        scores_by_model[model_id].append(score)

    task_ids = sorted(scores_by_task)
    model_ids = sorted(scores_by_model)
    primary = analysis["panel_uplift"]
    if len(score_rows) != int(primary["valid_comparisons"]):
        raise PilotAssetError("uplift sensitivity population differs from the primary cohort")
    if len(task_ids) != int(primary["task_clusters"]):
        raise PilotAssetError("uplift sensitivity task count differs from the primary cohort")
    if len(model_ids) != len(analysis["uplift_leaderboard"]):
        raise PilotAssetError("uplift sensitivity omits a frozen endpoint")

    cell_weighted = statistics.fmean(score for _, _, score in score_rows)
    task_weighted = statistics.fmean(
        statistics.fmean(score for _, score in scores_by_task[task_id]) for task_id in task_ids
    )
    model_weighted = statistics.fmean(
        statistics.fmean(scores_by_model[model_id]) for model_id in model_ids
    )
    if not math.isclose(
        cell_weighted,
        float(primary["task_cluster_win_share"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PilotAssetError("declared cell-weighted estimate does not reproduce")

    seed = 20260722
    replicates = 5000
    generator = np.random.default_rng(seed)
    task_bootstraps = np.empty(replicates, dtype=float)
    model_bootstraps: list[float] = []
    task_means = {
        task_id: statistics.fmean(score for _, score in rows)
        for task_id, rows in scores_by_task.items()
    }
    for index in range(replicates):
        sampled = generator.choice(task_ids, size=len(task_ids), replace=True)
        sampled_task_ids = [str(task_id) for task_id in sampled]
        task_bootstraps[index] = statistics.fmean(
            task_means[task_id] for task_id in sampled_task_ids
        )
        sampled_model_scores: dict[str, list[float]] = defaultdict(list)
        for task_id in sampled_task_ids:
            for model_id, score in scores_by_task[task_id]:
                sampled_model_scores[model_id].append(score)
        if all(sampled_model_scores.get(model_id) for model_id in model_ids):
            model_bootstraps.append(
                statistics.fmean(
                    statistics.fmean(sampled_model_scores[model_id]) for model_id in model_ids
                )
            )
    if len(model_bootstraps) < int(0.98 * replicates):
        raise PilotAssetError("too few complete equal-model task-bootstrap replicates")

    return {
        "cell_weighted_estimate": cell_weighted,
        "equal_task_estimate": task_weighted,
        "equal_task_interval_lower": float(np.quantile(task_bootstraps, 0.025)),
        "equal_task_interval_upper": float(np.quantile(task_bootstraps, 0.975)),
        "equal_model_estimate": model_weighted,
        "equal_model_interval_lower": float(np.quantile(model_bootstraps, 0.025)),
        "equal_model_interval_upper": float(np.quantile(model_bootstraps, 0.975)),
        "comparisons": len(score_rows),
        "task_clusters": len(task_ids),
        "models": len(model_ids),
        "cells_per_task_min": min(len(rows) for rows in scores_by_task.values()),
        "cells_per_task_max": max(len(rows) for rows in scores_by_task.values()),
        "cells_per_model_min": min(len(rows) for rows in scores_by_model.values()),
        "cells_per_model_max": max(len(rows) for rows in scores_by_model.values()),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "equal_model_complete_replicates": len(model_bootstraps),
    }


def _uplift_data(
    analysis: Mapping[str, Any], standardized: Mapping[str, object]
) -> list[dict[str, object]]:
    primary = analysis["panel_uplift"]
    family_balanced = analysis["judge_family_balanced_sensitivity"]["panel_uplift"]
    comparison_rows = analysis["comparison_consensus"]

    judge_families = analysis["judge_family_balanced_sensitivity"]["diagnostics"]["judge_families"]
    retained_choices: dict[str, str] = {}
    primary_uplift_choices: dict[str, str] = {}
    for comparison in comparison_rows:
        if comparison.get("track") != "epicure_uplift":
            continue
        comparison_id = str(comparison["comparison_id"])
        primary_choice = comparison.get("primary_consensus_choice")
        if comparison.get("primary_consensus_available") is True:
            if primary_choice not in {"left", "right", "tie", "both_bad"}:
                raise PilotAssetError("primary uplift choice is invalid")
            primary_uplift_choices[comparison_id] = str(primary_choice)

        by_family: dict[str, list[str]] = defaultdict(list)
        for vote in comparison.get("consistent_judge_votes", []):
            if not isinstance(vote, Mapping) or vote.get("self_judgment") is True:
                continue
            judge_id = str(vote.get("judge_id") or "")
            family = judge_families.get(judge_id)
            choice = str(vote.get("choice") or "")
            if family is None or choice not in {"left", "right", "tie", "both_bad"}:
                raise PilotAssetError("cross-family uplift vote is invalid")
            by_family[str(family)].append(choice)
        family_votes = [
            choices[0] for choices in by_family.values() if choices and len(set(choices)) == 1
        ]
        if len(family_votes) < 2:
            continue
        counts = Counter(family_votes)
        choice, count = counts.most_common(1)[0]
        if count >= len(family_votes) // 2 + 1:
            retained_choices[comparison_id] = choice

    retained_ids = set(retained_choices)
    primary_ids = set(primary_uplift_choices)
    if not retained_ids < primary_ids:
        raise PilotAssetError("cross-family cohort is not a strict subset of primary uplift")
    changed = sum(
        retained_choices[comparison_id] != primary_uplift_choices[comparison_id]
        for comparison_id in retained_ids
    )
    if changed != 0:
        raise PilotAssetError("a retained cross-family uplift choice changed")

    def _epicure_outcomes(choices: Mapping[str, str], comparison_ids: set[str]) -> Counter[str]:
        outcomes: Counter[str] = Counter()
        rows_by_id = {
            str(row["comparison_id"]): row
            for row in comparison_rows
            if str(row.get("comparison_id") or "") in comparison_ids
        }
        for comparison_id in comparison_ids:
            row = rows_by_id[comparison_id]
            choice = choices[comparison_id]
            if choice == "tie":
                outcomes["tie"] += 1
            elif choice == "both_bad":
                outcomes["both_bad"] += 1
            else:
                chosen_condition = row[choice]["condition"]
                outcomes["epicure_win" if chosen_condition == "epicure_on" else "unaided_win"] += 1
        return outcomes

    retained_outcomes = _epicure_outcomes(retained_choices, retained_ids)
    excluded_ids = primary_ids - retained_ids
    excluded_outcomes = _epicure_outcomes(primary_uplift_choices, excluded_ids)
    if sum(retained_outcomes.values()) != len(retained_ids) or sum(
        excluded_outcomes.values()
    ) != len(excluded_ids):
        raise PilotAssetError("cross-family uplift selection diagnostics do not reconcile")
    defective_rows = [
        row for row in comparison_rows if row.get("task_id") == IMAGE_DEPENDENT_TASK_ID
    ]
    defective_consensus = [
        row for row in defective_rows if row.get("primary_consensus_available") is True
    ]
    if len(defective_consensus) > len(defective_rows):
        raise PilotAssetError("image-dependent task sensitivity population is invalid")
    leave_one_out = _panel_uplift_summary(
        [row for row in comparison_rows if row.get("task_id") != IMAGE_DEPENDENT_TASK_ID],
        None,
    )
    rows = [
        {
            "group": "weighting",
            "label": "Cell weighted (primary)",
            "estimate": primary["task_cluster_win_share"],
            "lower": primary["task_cluster_interval_lower"],
            "upper": primary["task_cluster_interval_upper"],
            "wins": primary["epicure_wins"],
            "ties": primary["ties"],
            "losses": primary["unaided_wins"],
            "n": primary["valid_comparisons"],
            "task_clusters": primary["task_clusters"],
            "weighting": "retrospective primary; observed endpoint-task cells",
            "interval_type": "observed-task resampling",
        },
        {
            "group": "weighting",
            "label": "Equal task means",
            "estimate": standardized["equal_task_estimate"],
            "lower": standardized["equal_task_interval_lower"],
            "upper": standardized["equal_task_interval_upper"],
            "wins": primary["epicure_wins"],
            "ties": primary["ties"],
            "losses": primary["unaided_wins"],
            "n": primary["valid_comparisons"],
            "task_clusters": primary["task_clusters"],
            "weighting": "equal observed task means",
            "interval_type": "observed-task resampling",
        },
        {
            "group": "weighting",
            "label": "Equal endpoint means",
            "estimate": standardized["equal_model_estimate"],
            "lower": standardized["equal_model_interval_lower"],
            "upper": standardized["equal_model_interval_upper"],
            "wins": primary["epicure_wins"],
            "ties": primary["ties"],
            "losses": primary["unaided_wins"],
            "n": primary["valid_comparisons"],
            "task_clusters": primary["task_clusters"],
            "weighting": "equal observed endpoint means",
            "interval_type": "observed-task resampling",
        },
        {
            "group": "item",
            "label": "Primary, item removed",
            "estimate": leave_one_out["task_cluster_win_share"],
            "lower": leave_one_out["task_cluster_interval_lower"],
            "upper": leave_one_out["task_cluster_interval_upper"],
            "wins": leave_one_out["epicure_wins"],
            "ties": leave_one_out["ties"],
            "losses": leave_one_out["unaided_wins"],
            "n": leave_one_out["valid_comparisons"],
            "task_clusters": leave_one_out["task_clusters"],
            "weighting": "observed endpoint-task cells",
            "interval_type": "observed-task resampling",
        },
        {
            "group": "admission",
            "label": "Cross-family-admitted subset",
            "estimate": family_balanced["task_cluster_win_share"],
            "lower": family_balanced["task_cluster_interval_lower"],
            "upper": family_balanced["task_cluster_interval_upper"],
            "wins": family_balanced["epicure_wins"],
            "ties": family_balanced["ties"],
            "losses": family_balanced["unaided_wins"],
            "n": family_balanced["valid_comparisons"],
            "task_clusters": family_balanced["task_clusters"],
            "weighting": (
                "observed endpoint-task cells; strict subset admitted by cross-family rule; "
                "no retained primary choice changed"
            ),
            "interval_type": "observed-task resampling",
            "primary_valid_comparisons": len(primary_ids),
            "retained_primary_choice_changes": changed,
            "excluded_primary_comparisons": len(excluded_ids),
            "excluded_epicure_wins": excluded_outcomes["epicure_win"],
            "excluded_ties": excluded_outcomes["tie"],
            "excluded_unaided_wins": excluded_outcomes["unaided_win"],
        },
    ]
    diagnostic_defaults: dict[str, object] = {
        "primary_valid_comparisons": "",
        "retained_primary_choice_changes": "",
        "excluded_primary_comparisons": "",
        "excluded_epicure_wins": "",
        "excluded_ties": "",
        "excluded_unaided_wins": "",
    }
    for row in rows:
        for key, value in diagnostic_defaults.items():
            row.setdefault(key, value)
        lower = float(row["lower"])
        estimate = float(row["estimate"])
        upper = float(row["upper"])
        if not 0 <= lower <= estimate <= upper <= 1:
            raise PilotAssetError(f"invalid probability interval for {row['label']}")
    return rows


def _uplift_figure(rows: Sequence[Mapping[str, object]]) -> str:
    plot_left, plot_right = 3.15, 7.6
    low, high = -0.15, 0.15
    centered_bounds = [float(row[key]) - 0.5 for row in rows for key in ("lower", "upper")]
    if any(value < low or value > high for value in centered_bounds):
        raise PilotAssetError("uplift interval would be clipped")
    y_top, gap = 3.92, 0.78

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,every node/.style={font=\fontsize{8.0}{9.0}\selectfont}]",
        rf"\node[anchor=east,font=\bfseries\scriptsize] at ({plot_left - 0.16:.3f},4.58) {{Analysis cohort}};",
        rf"\node[align=center,font=\bfseries\scriptsize] "
        rf"at ({(plot_left + plot_right) / 2:.3f},4.64) "
        r"{Enabled-condition preference\\share minus 0.5};",
    ]
    for tick in (-0.1, 0.0, 0.1):
        tx = x(tick)
        lines.extend(
            [
                rf"\draw[black!17] ({tx:.3f},0.22) -- ({tx:.3f},4.24);",
                rf"\node[anchor=north,text=black!65] at ({tx:.3f},0.18) "
                rf"{{{tick:+.1f}}};",
            ]
        )
    null_x = x(0.0)
    lines.append(rf"\draw[black!65,densely dashed] ({null_x:.3f},0.22) -- ({null_x:.3f},4.24);")
    for index, row in enumerate(rows):
        y = y_top - index * gap
        label = _latex(row["label"])
        lower = x(float(row["lower"]) - 0.5)
        upper = x(float(row["upper"]) - 0.5)
        estimate = x(float(row["estimate"]) - 0.5)
        wtl = f"{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}"
        annotation = rf"{wtl}; $n={int(row['n'])}$; {int(row['task_clusters'])} tasks"
        lines.extend(
            [
                rf"\node[anchor=east] at ({plot_left - 0.16:.3f},{y:.3f}) {{{label}}};",
                rf"\draw[black!75,line width=0.75pt] ({lower:.3f},{y:.3f}) -- ({upper:.3f},{y:.3f});",
                rf"\draw[black!75,line width=0.75pt] ({lower:.3f},{y - 0.075:.3f}) -- ({lower:.3f},{y + 0.075:.3f});",
                rf"\draw[black!75,line width=0.75pt] ({upper:.3f},{y - 0.075:.3f}) -- ({upper:.3f},{y + 0.075:.3f});",
                rf"\fill[FBBlue] ({estimate:.3f},{y:.3f}) circle (1.8pt);",
                rf"\node[anchor=east,text=black!70] at ({plot_left - 0.16:.3f},{y - 0.22:.3f}) "
                rf"{{{annotation}}};",
            ]
        )
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _operational_data(
    analysis: Mapping[str, Any], comparison_manifest: Mapping[str, Any]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    paired_by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pairs": 0, "off_success": 0, "on_success": 0}
    )
    matrix = {
        "both_success": 0,
        "on_failed_off_success": 0,
        "off_failed_on_success": 0,
        "both_failed": 0,
    }
    for comparison in comparison_manifest["comparisons"]:
        if comparison.get("track") != "epicure_uplift":
            continue
        arms = {comparison[side]["condition"]: comparison[side] for side in ("left", "right")}
        if set(arms) != {"epicure_off", "epicure_on"}:
            raise PilotAssetError("paired comparison does not contain one arm per condition")
        off_arm = arms["epicure_off"]
        on_arm = arms["epicure_on"]
        if off_arm["season_model_id"] != on_arm["season_model_id"]:
            raise PilotAssetError("paired comparison crosses model endpoints")
        model_id = str(off_arm["season_model_id"])
        off_success = off_arm["status"] == "success"
        on_success = on_arm["status"] == "success"
        paired_by_model[model_id]["pairs"] += 1
        paired_by_model[model_id]["off_success"] += int(off_success)
        paired_by_model[model_id]["on_success"] += int(on_success)
        if off_success and on_success:
            matrix["both_success"] += 1
        elif off_success:
            matrix["on_failed_off_success"] += 1
        elif on_success:
            matrix["off_failed_on_success"] += 1
        else:
            matrix["both_failed"] += 1

    rows: list[dict[str, object]] = []
    operational = analysis["operational_metrics"]
    for model_id in sorted(operational):
        metric = operational[model_id]
        paired = paired_by_model[model_id]
        if paired["pairs"] != 120:
            raise PilotAssetError(f"paired condition denominator changed for {model_id}")
        expected_off = round(120 * (1.0 - float(metric["end_to_end_failure_rate_epicure_off"])))
        expected_on = round(120 * (1.0 - float(metric["end_to_end_failure_rate_epicure_on"])))
        if paired["off_success"] != expected_off or paired["on_success"] != expected_on:
            raise PilotAssetError(f"condition success counts do not reconcile for {model_id}")
        breakdown = metric["failure_breakdown"]
        reconciled_terminal_failures = int(breakdown.get("model_behavior_failure") or 0)
        provider_failures = int(breakdown.get("provider_pre_inference_failure") or 0)
        uncertain = int(breakdown.get("uncertain_delivery") or 0)
        if int(
            metric["success"]
        ) + reconciled_terminal_failures + provider_failures + uncertain != int(metric["arms"]):
            raise PilotAssetError(f"failure classes do not reconcile for {model_id}")
        tool_success = metric.get("tool_success_rate")
        tool_calls = int(metric["tool_calls"])
        tool_successes = None if tool_success is None else round(float(tool_success) * tool_calls)
        rows.append(
            {
                "season_model_id": model_id,
                "model": metric["display_name"],
                "provider": metric["provider"],
                "off_success": paired["off_success"],
                "off_attempts": paired["pairs"],
                "on_success": paired["on_success"],
                "on_attempts": paired["pairs"],
                "reconciled_terminal_failures": reconciled_terminal_failures,
                "provider_pre_inference_failures": provider_failures,
                "uncertain_delivery": uncertain,
                "on_tool_use_rate": metric["epicure_on_tool_use_rate"],
                "tool_calls": tool_calls,
                "tool_successes": tool_successes,
                "tool_success_rate": tool_success,
                "latency_median_seconds": float(metric["latency_median_ms"]) / 1000.0,
                "latency_p95_seconds": float(metric["latency_p95_ms"]) / 1000.0,
                "attributed_cost_usd": metric["attributed_cost_usd"],
                "cost_unattributed_arms": metric["cost_unattributed_arms"],
            }
        )

    if sum(matrix.values()) != 120 * len(rows):
        raise PilotAssetError("paired condition attrition matrix does not reconcile")
    matrix_rows = [{"outcome": outcome, "count": count} for outcome, count in matrix.items()]
    return rows, matrix_rows


def _paired_reliability_data(
    comparison_manifest: Mapping[str, Any],
) -> list[dict[str, object]]:
    by_task: dict[str, list[int]] = defaultdict(list)
    off_success_total = 0
    on_success_total = 0
    outcome_counts = {
        "both_success": 0,
        "off_only_success": 0,
        "on_only_success": 0,
        "neither_success": 0,
    }
    for comparison in comparison_manifest["comparisons"]:
        if comparison.get("track") != "epicure_uplift":
            continue
        arms = {comparison[side]["condition"]: comparison[side] for side in ("left", "right")}
        if set(arms) != {"epicure_off", "epicure_on"}:
            raise PilotAssetError("paired reliability row has invalid conditions")
        off_success = int(arms["epicure_off"]["status"] == "success")
        on_success = int(arms["epicure_on"]["status"] == "success")
        task_id = str(comparison["task_id"])
        by_task[task_id].append(on_success - off_success)
        off_success_total += off_success
        on_success_total += on_success
        if off_success and on_success:
            outcome_counts["both_success"] += 1
        elif off_success:
            outcome_counts["off_only_success"] += 1
        elif on_success:
            outcome_counts["on_only_success"] += 1
        else:
            outcome_counts["neither_success"] += 1

    if len(by_task) != 120 or any(len(values) != 12 for values in by_task.values()):
        raise PilotAssetError("paired reliability task structure changed")
    task_differences = np.asarray(
        [statistics.fmean(by_task[task_id]) for task_id in sorted(by_task)],
        dtype=float,
    )
    seed = 20260723
    replicates = 5000
    rng = np.random.default_rng(seed)
    sampled = rng.choice(task_differences, size=(replicates, len(task_differences)), replace=True)
    bootstrap = sampled.mean(axis=1)
    estimate = float(task_differences.mean())
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    if sum(outcome_counts.values()) != sum(len(values) for values in by_task.values()):
        raise PilotAssetError("paired reliability totals do not reconcile")
    return [
        {
            "estimand": ("tool_available_minus_tool_unavailable_realized_success_proportion"),
            "attempted_cells": sum(len(values) for values in by_task.values()),
            "task_clusters": len(task_differences),
            "off_success": off_success_total,
            "on_success": on_success_total,
            **outcome_counts,
            "realized_success_proportion_difference": estimate,
            "lower_95_task_bootstrap": float(lower),
            "upper_95_task_bootstrap": float(upper),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
        }
    ]


def _preference_score_completion_ranges(
    analysis: Mapping[str, Any],
    comparison_manifest: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Compute bounded-score completion sensitivities for missing paired outcomes."""

    primary = analysis["panel_uplift"]
    wins = int(primary["epicure_wins"])
    ties = int(primary["ties"])
    losses = int(primary["unaided_wins"])
    both_bad = int(primary["both_bad"])
    judged = int(primary["valid_comparisons"])
    no_consensus = int(primary["no_consensus"])
    if wins + ties + losses != judged or both_bad < 0 or no_consensus < 0:
        raise PilotAssetError("paired preference cohort does not reconcile")

    observed_score = wins + ties / 2
    planned_population = int(
        sum(
            comparison.get("track") == "epicure_uplift"
            for comparison in comparison_manifest["comparisons"]
        )
    )
    if planned_population != 1440:
        raise PilotAssetError("frozen planned paired population changed")
    planned_unresolved = planned_population - judged
    planned_lower = observed_score / planned_population
    planned_upper = (observed_score + planned_unresolved) / planned_population
    planned_tipping_mean = (0.5 * planned_population - observed_score) / planned_unresolved

    admitted_population = judged + no_consensus
    admitted_lower = observed_score / admitted_population
    admitted_upper = (observed_score + no_consensus) / admitted_population
    admitted_tipping_mean = (0.5 * admitted_population - observed_score) / no_consensus

    reliability = _paired_reliability_data(comparison_manifest)[0]
    both_success = int(reliability["both_success"])
    off_only = int(reliability["off_only_success"])
    on_only = int(reliability["on_only_success"])
    neither = int(reliability["neither_success"])
    attempted = int(reliability["attempted_cells"])
    unresolved_success_pairs = both_success - judged
    failure_aware_population = attempted - neither
    failure_aware_score = observed_score + on_only
    failure_aware_lower = failure_aware_score / failure_aware_population
    failure_aware_upper = (
        failure_aware_score + unresolved_success_pairs
    ) / failure_aware_population
    failure_aware_tipping_mean = (
        0.5 * failure_aware_population - failure_aware_score
    ) / unresolved_success_pairs

    rows = [
        {
            "analysis": "planned_cell_score_completion",
            "population": planned_population,
            "judged_consensuses": judged,
            "deterministic_tool_available_wins_from_one_sided_success": 0,
            "deterministic_tool_unavailable_wins_from_one_sided_success": 0,
            "unresolved_scores": planned_unresolved,
            "excluded_dual_failures": 0,
            "observed_favourable_score": observed_score,
            "lower_bound": planned_lower,
            "upper_bound": planned_upper,
            "missing_score_mean_at_neutrality": planned_tipping_mean,
            "assumption": (
                "each paired cell without a primary consensus, including target "
                "failures and exclusions, may take any tie-adjusted tool-available "
                "preference score in [0,1]"
            ),
        },
        {
            "analysis": "admitted_pair_score_completion",
            "population": admitted_population,
            "judged_consensuses": judged,
            "deterministic_tool_available_wins_from_one_sided_success": 0,
            "deterministic_tool_unavailable_wins_from_one_sided_success": 0,
            "unresolved_scores": no_consensus,
            "excluded_dual_failures": 0,
            "observed_favourable_score": observed_score,
            "lower_bound": admitted_lower,
            "upper_bound": admitted_upper,
            "missing_score_mean_at_neutrality": admitted_tipping_mean,
            "assumption": (
                "each non-consensus admitted comparison may take any tie-adjusted "
                "tool-available preference score in [0,1]"
            ),
        },
        {
            "analysis": "failure_aware_score_completion",
            "population": failure_aware_population,
            "judged_consensuses": judged,
            "deterministic_tool_available_wins_from_one_sided_success": on_only,
            "deterministic_tool_unavailable_wins_from_one_sided_success": off_only,
            "unresolved_scores": unresolved_success_pairs,
            "excluded_dual_failures": neither,
            "observed_favourable_score": failure_aware_score,
            "lower_bound": failure_aware_lower,
            "upper_bound": failure_aware_upper,
            "missing_score_mean_at_neutrality": failure_aware_tipping_mean,
            "assumption": (
                "one-sided target success beats target failure; dual failures are "
                "excluded; each unresolved dual-success pair may score in [0,1]"
            ),
        },
    ]
    if not all(
        0 <= float(row["lower_bound"]) <= float(row["upper_bound"]) <= 1
        and 0 <= float(row["missing_score_mean_at_neutrality"]) <= 1
        for row in rows
    ):
        raise PilotAssetError("paired preference identification bounds are invalid")
    return rows


def _operational_error_data(
    arm_paths: Sequence[Path],
    completion_interpretation: ValidatedCompletionInterpretationCorrection,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    str,
]:
    terminal_order = (
        "incomplete_final_response",
        "second_recorded_mcp_error_arm_stop",
        "tool_round_cap",
        "tool_fanout_contract",
        "provider_overload",
        "empty_final_answer",
        "read_timeout",
        "cost_reconciliation",
    )
    terminal: dict[str, dict[str, int]] = {
        category: {"epicure_on": 0, "epicure_off": 0} for category in terminal_order
    }
    trace_categories: dict[str, dict[str, object]] = {
        category: {
            "events": 0,
            "arms": set(),
            "successful_final_arms": set(),
            "failed_final_arms": set(),
        }
        for category in (
            "ingredient_or_entity_resolution",
            "unsupported_controlled_vocabulary",
            "schema_or_numeric_list_constraint",
        )
    }
    artifact_hashes: list[str] = []
    arm_ids: set[str] = set()
    tool_using_arms: set[str] = set()
    error_arms: set[str] = set()
    error_arms_with_success: set[str] = set()
    trace_events = 0
    trace_successes = 0
    trace_errors = 0
    second_error_stop_arms = 0
    second_error_same_round = 0
    second_error_same_round_zero = 0
    effective_failures = 0
    corrected_ids = set(completion_interpretation.arm_ids)

    for path in sorted(arm_paths):
        arm = _load(path)
        artifact_hashes.append(_verify(arm, f"scored arm {path.name}"))
        arm_id = str(arm["arm_id"])
        if arm_id in arm_ids:
            raise PilotAssetError(f"duplicate scored arm identity: {arm_id}")
        arm_ids.add(arm_id)
        if arm.get("synthetic") is not False:
            raise PilotAssetError("operational error analysis refuses generated fixtures")
        condition = str(arm["condition"])
        source_status = str(arm["status"])
        status = "failed" if arm_id in corrected_ids else source_status
        second_error_stop = False
        if condition not in {"epicure_on", "epicure_off"}:
            raise PilotAssetError(f"unknown scored-arm condition: {condition}")
        if status != "success":
            effective_failures += 1
            error = str(arm.get("error") or "")
            if arm_id in corrected_ids:
                category = "incomplete_final_response"
            elif error == "tool arguments remained invalid after one repair":
                category = "second_recorded_mcp_error_arm_stop"
                second_error_stop = True
            elif "exhausted the Epicure tool-round cap" in error:
                category = "tool_round_cap"
            elif "tool fan-out violated the frozen contract" in error:
                category = "tool_fanout_contract"
            elif "rejected the request: Overloaded" in error:
                category = "provider_overload"
            elif "returned an empty final answer" in error:
                category = "empty_final_answer"
            elif error.startswith("Read timeout on endpoint URL:"):
                category = "read_timeout"
            elif "cost did not reconcile" in error:
                category = "cost_reconciliation"
            else:
                raise PilotAssetError(f"unclassified target-arm failure: {error}")
            terminal[category][condition] += 1

        result = arm.get("result")
        trace = result.get("tool_trace", []) if isinstance(result, dict) else []
        if not isinstance(trace, list):
            raise PilotAssetError(f"scored arm has an invalid tool trace: {arm_id}")
        if trace:
            tool_using_arms.add(arm_id)
        error_rounds: list[int] = []
        for event in trace:
            if not isinstance(event, dict) or not isinstance(event.get("is_error"), bool):
                raise PilotAssetError(f"tool trace event is malformed: {arm_id}")
            trace_events += 1
            if event["is_error"] is False:
                trace_successes += 1
                continue
            trace_errors += 1
            round_index = event.get("round_index")
            if not isinstance(round_index, int) or round_index < 0:
                raise PilotAssetError(f"tool trace error lacks a valid round index: {arm_id}")
            error_rounds.append(round_index)
            error_arms.add(arm_id)
            if status == "success":
                error_arms_with_success.add(arm_id)
            error_text = str(event.get("result") or "")
            tool_name = str(event.get("name") or "")
            if "Could not resolve" in error_text:
                trace_category = "ingredient_or_entity_resolution"
            elif "Unknown axis" in error_text or (
                tool_name == "morph" and "Unknown " in error_text
            ):
                trace_category = "unsupported_controlled_vocabulary"
            elif "validation error" in error_text:
                trace_category = "schema_or_numeric_list_constraint"
            else:
                raise PilotAssetError(f"unclassified MCP error boundary: {error_text}")
            category_values = trace_categories[trace_category]
            category_values["events"] = int(category_values["events"]) + 1
            cast_arms = category_values["arms"]
            cast_successes = category_values["successful_final_arms"]
            cast_failures = category_values["failed_final_arms"]
            assert isinstance(cast_arms, set)
            assert isinstance(cast_successes, set)
            assert isinstance(cast_failures, set)
            cast_arms.add(arm_id)
            (cast_successes if status == "success" else cast_failures).add(arm_id)

        if second_error_stop:
            if len(error_rounds) < 2:
                raise PilotAssetError("second-error stop arm has fewer than two error traces")
            second_error_stop_arms += 1
            if error_rounds[-2] == error_rounds[-1]:
                second_error_same_round += 1
                if error_rounds[-1] == 0:
                    second_error_same_round_zero += 1

    if len(arm_ids) != 2880:
        raise PilotAssetError("operational error analysis requires all 2,880 target arms")
    terminal_rows = [
        {
            "category": category,
            "epicure_on": terminal[category]["epicure_on"],
            "epicure_off": terminal[category]["epicure_off"],
            "total": terminal[category]["epicure_on"] + terminal[category]["epicure_off"],
        }
        for category in terminal_order
    ]
    if sum(int(row["total"]) for row in terminal_rows) != effective_failures:
        raise PilotAssetError("target-arm failure taxonomy does not reconcile")

    trace_rows: list[dict[str, object]] = []
    for category, values in trace_categories.items():
        category_arms = values["arms"]
        successful_arms = values["successful_final_arms"]
        failed_arms = values["failed_final_arms"]
        assert isinstance(category_arms, set)
        assert isinstance(successful_arms, set)
        assert isinstance(failed_arms, set)
        trace_rows.append(
            {
                "category": category,
                "error_events": int(values["events"]),
                "distinct_arms": len(category_arms),
                "successful_final_arms": len(successful_arms),
                "failed_final_arms": len(failed_arms),
            }
        )
    if sum(int(row["error_events"]) for row in trace_rows) != trace_errors:
        raise PilotAssetError("MCP failure taxonomy does not reconcile")
    summary_rows = [
        {"metric": "tool_using_arms", "count": len(tool_using_arms)},
        {"metric": "trace_events", "count": trace_events},
        {"metric": "successful_trace_events", "count": trace_successes},
        {"metric": "error_trace_events", "count": trace_errors},
        {"metric": "arms_with_error_trace", "count": len(error_arms)},
        {
            "metric": "error_trace_arms_with_successful_final_answer",
            "count": len(error_arms_with_success),
        },
    ]
    if trace_successes + trace_errors != trace_events:
        raise PilotAssetError("MCP trace summary does not reconcile")
    stop_rule_rows = [
        {"metric": "arms_stopped_on_second_recorded_mcp_error", "count": second_error_stop_arms},
        {"metric": "both_errors_in_same_tool_round", "count": second_error_same_round},
        {"metric": "both_errors_in_round_zero", "count": second_error_same_round_zero},
    ]
    if not (second_error_same_round_zero <= second_error_same_round <= second_error_stop_arms):
        raise PilotAssetError("arm-wide MCP stop-rule audit does not reconcile")
    arm_set_sha256 = sha256_json({"artifact_sha256s": sorted(artifact_hashes)})
    return terminal_rows, trace_rows, summary_rows, stop_rule_rows, arm_set_sha256


def _reliability_figure(
    reliability_rows: Sequence[Mapping[str, object]],
    terminal_rows: Sequence[Mapping[str, object]],
) -> str:
    if len(reliability_rows) != 1:
        raise PilotAssetError("reliability figure requires one paired estimand")
    reliability = reliability_rows[0]
    attempted = int(reliability["attempted_cells"])
    off_failures = attempted - int(reliability["off_success"])
    on_failures = attempted - int(reliability["on_success"])
    failure_difference = -float(reliability["realized_success_proportion_difference"])
    difference_lower = -float(reliability["upper_95_task_bootstrap"])
    difference_upper = -float(reliability["lower_95_task_bootstrap"])
    if attempted != 1440:
        raise PilotAssetError("paired condition denominator changed")

    condition_rows = (
        ("Unaided", off_failures / attempted, off_failures, "black!52"),
        ("Epicure enabled", on_failures / attempted, on_failures, "FBBlue"),
    )
    terminal_labels = {
        "incomplete_final_response": "Incomplete final response",
        "second_recorded_mcp_error_arm_stop": "Second recorded MCP error",
        "provider_overload": "Provider overload",
        "empty_final_answer": "Empty final answer",
        "read_timeout": "Read timeout",
        "tool_round_cap": "Tool-round cap",
        "tool_fanout_contract": "Fan-out cap",
        "cost_reconciliation": "Cost reconciliation",
    }
    by_category = {str(row["category"]): row for row in terminal_rows}
    plot_order = tuple(terminal_labels)
    if set(by_category) != set(plot_order):
        raise PilotAssetError("terminal failure categories do not match figure contract")

    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,every node/.style={font=\fontsize{7.8}{8.8}\selectfont}]",
        r"\node[anchor=west,font=\bfseries\scriptsize] at (0,5.65) {(a) Failure by condition};",
        r"\node[anchor=west,text=black!65] at (0,5.28) {All 1,440 matched endpoint--task cells};",
    ]
    a_left, a_width, a_max = 1.55, 5.25, 0.26
    for tick in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        tick_x = a_left + a_width * tick / a_max
        lines.extend(
            [
                rf"\draw[black!16] ({tick_x:.3f},1.15) -- ({tick_x:.3f},4.82);",
                rf"\node[anchor=north,text=black!65] at ({tick_x:.3f},1.10) "
                rf"{{{100 * tick:.0f}\%}};",
            ]
        )
    for index, (label, rate, count, color) in enumerate(condition_rows):
        y = 4.52 - index * 0.78
        end_x = a_left + a_width * rate / a_max
        lines.extend(
            [
                rf"\node[anchor=east] at ({a_left - 0.14:.3f},{y:.3f}) {{{label}}};",
                rf"\draw[{color},line width=5.2pt] ({a_left:.3f},{y:.3f}) -- ({end_x:.3f},{y:.3f});",
                rf"\node[anchor=west] at ({end_x + 0.12:.3f},{y:.3f}) "
                rf"{{{100 * rate:.1f}\% ({count}/{attempted})}};",
            ]
        )
    diff_y = 2.36
    diff_x = a_left + a_width * failure_difference / a_max
    lower_x = a_left + a_width * difference_lower / a_max
    upper_x = a_left + a_width * difference_upper / a_max
    lines.extend(
        [
            rf"\node[anchor=east,align=right] at ({a_left - 0.14:.3f},{diff_y:.3f}) "
            r"{Paired failure-risk\\difference};",
            rf"\draw[black!78,line width=0.8pt] ({lower_x:.3f},{diff_y:.3f}) -- "
            rf"({upper_x:.3f},{diff_y:.3f});",
            rf"\draw[black!78,line width=0.8pt] ({lower_x:.3f},{diff_y - 0.09:.3f}) -- "
            rf"({lower_x:.3f},{diff_y + 0.09:.3f});",
            rf"\draw[black!78,line width=0.8pt] ({upper_x:.3f},{diff_y - 0.09:.3f}) -- "
            rf"({upper_x:.3f},{diff_y + 0.09:.3f});",
            rf"\fill[FBBlue] ({diff_x:.3f},{diff_y:.3f}) circle (1.9pt);",
            rf"\node[anchor=north] at ({diff_x:.3f},{diff_y - 0.16:.3f}) "
            rf"{{+{100 * failure_difference:.1f} pp "
            rf"[{100 * difference_lower:.1f}, {100 * difference_upper:.1f}]}};",
            r"\node[anchor=west,font=\bfseries\scriptsize] at (7.35,5.65) {(b) Mutually exclusive terminal classes};",
            r"\fill[FBBlue] (12.25,5.28) rectangle (12.43,5.40);",
            r"\node[anchor=west] at (12.52,5.34) {Epicure enabled};",
            r"\fill[black!52] (14.55,5.28) rectangle (14.73,5.40);",
            r"\node[anchor=west] at (14.82,5.34) {Unaided};",
        ]
    )

    b_left, b_width, b_max = 11.30, 5.25, 180
    for tick in (0, 45, 90, 135, 180):
        tick_x = b_left + b_width * tick / b_max
        lines.extend(
            [
                rf"\draw[black!14] ({tick_x:.3f},0.73) -- ({tick_x:.3f},4.96);",
                rf"\node[anchor=north,text=black!65] at ({tick_x:.3f},0.68) {{{tick}}};",
            ]
        )
    for index, category in enumerate(plot_order):
        row = by_category[category]
        y = 4.70 - index * 0.49
        on_count = int(row["epicure_on"])
        off_count = int(row["epicure_off"])
        on_x = b_left + b_width * on_count / b_max
        off_x = b_left + b_width * off_count / b_max
        label = terminal_labels[category]
        lines.extend(
            [
                rf"\node[anchor=east] at ({b_left - 0.16:.3f},{y:.3f}) {{{label}}};",
                rf"\draw[FBBlue,line width=3.3pt] ({b_left:.3f},{y + 0.085:.3f}) -- "
                rf"({on_x:.3f},{y + 0.085:.3f});",
                rf"\node[anchor=west,text=FBBlue] at ({on_x + 0.07:.3f},{y + 0.085:.3f}) "
                rf"{{{on_count}}};",
                rf"\draw[black!52,line width=3.3pt] ({b_left:.3f},{y - 0.085:.3f}) -- "
                rf"({off_x:.3f},{y - 0.085:.3f});",
                rf"\node[anchor=west,text=black!65] at ({off_x + 0.07:.3f},{y - 0.085:.3f}) "
                rf"{{{off_count}}};",
            ]
        )
    lines.extend(
        [
            r"\node[anchor=east,text=black!65] at (16.55,0.18) {terminal failures ($n=217$)};",
            r"\end{tikzpicture}",
        ]
    )
    return "\n".join(lines)


def _operational_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}X r r r r r r r r r@{}}",
        r"\toprule",
        r"Endpoint & \shortstack{Unavailable\\completed} & "
        r"\shortstack{Available\\completed} & "
        r"\shortstack{Failures, both\\conditions R/P/U} & "
        r"\shortstack{Available-arm\\tool use} & "
        r"\shortstack{Tool ok\\success/total} & \shortstack{Median\\s} & "
        r"\shortstack{p95\\s} & \shortstack{Attributed\\USD} & "
        r"\shortstack{Unattr.\\arms} \\",
        r"\midrule",
    ]
    for row in rows:
        tool_successes = row["tool_successes"]
        tool_success_text = (
            "--" if tool_successes is None else f"{int(tool_successes)}/{int(row['tool_calls'])}"
        )
        failures = (
            f"{int(row['reconciled_terminal_failures'])}/"
            f"{int(row['provider_pre_inference_failures'])}/"
            f"{int(row['uncertain_delivery'])}"
        )
        lines.append(
            rf"{_latex(_short_model_name(str(row['model'])))} & "
            rf"{int(row['off_success'])}/120 & {int(row['on_success'])}/120 & "
            rf"{failures} & {100 * float(row['on_tool_use_rate']):.1f}\% & "
            rf"{tool_success_text} & {float(row['latency_median_seconds']):.1f} & "
            rf"{float(row['latency_p95_seconds']):.1f} & "
            rf"{float(row['attributed_cost_usd']):.2f} & "
            rf"{int(row['cost_unattributed_arms'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return "\n".join(lines)


def generate_pilot_assets(
    *,
    analysis: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    task_bank: Mapping[str, Any],
    review_queue: Mapping[str, Any],
    comparison_manifest: Mapping[str, Any],
    curation_audits: Sequence[Mapping[str, Any]],
    scored_arm_paths: Sequence[Path],
    arm_interpretation_correction: dict[str, Any],
    completion_interpretation_correction: dict[str, Any],
    scored_arm_dir: Path,
    output_dir: Path,
    input_paths: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    analysis_sha = _verify(analysis, "analysis")
    model_manifest_sha = _verify(model_manifest, "model manifest")
    task_sha = _verify(task_bank, "task bank")
    review_sha = _verify(review_queue, "task review queue")
    comparison_sha = _verify(comparison_manifest, "comparison manifest")
    if analysis.get("task_bank_artifact_sha256") != task_sha:
        raise PilotAssetError("analysis and task bank are not bound")
    if analysis.get("model_manifest_artifact_sha256") != model_manifest_sha:
        raise PilotAssetError("analysis and model manifest are not bound")
    if review_queue.get("task_set_sha256") != task_bank.get("task_set_sha256"):
        raise PilotAssetError("task review queue does not match the task set")
    if analysis.get("comparison_manifest_artifact_sha256") != comparison_sha:
        raise PilotAssetError("analysis and comparison manifest are not bound")
    if analysis.get("synthetic_arms") != 0 or analysis.get("synthetic_judgments") != 0:
        raise PilotAssetError("pilot figures refuse synthetic observations")
    interpretation = validate_arm_interpretation_correction(
        correction=arm_interpretation_correction,
        arms_dir=scored_arm_dir,
    )
    if (
        interpretation is None
        or analysis.get("arm_interpretation_correction_artifact_sha256")
        != interpretation.artifact_sha256
        or analysis.get("arm_interpretation_correction_count") != len(interpretation.arm_ids)
    ):
        raise PilotAssetError("analysis is not bound to the active arm correction")
    completion_interpretation = validate_completion_interpretation_correction(
        correction=completion_interpretation_correction,
        arms_dir=scored_arm_dir,
    )
    if analysis.get(
        "completion_interpretation_correction_artifact_sha256"
    ) != completion_interpretation.artifact_sha256 or analysis.get(
        "completion_interpretation_correction_count"
    ) != len(completion_interpretation.arm_ids):
        raise PilotAssetError("analysis is not bound to the active completion correction")
    effective_comparison_manifest = _effective_comparison_manifest(
        comparison_manifest,
        completion_interpretation,
    )

    flow_rows = _flow_data(
        analysis,
        task_bank,
        review_queue,
        comparison_manifest,
        curation_audits,
    )
    measurement_rows = _measurement_data(analysis)
    model_rows = _model_data(analysis)
    uplift_sensitivities = _standardized_uplift_sensitivities(analysis)
    uplift_rows = _uplift_data(analysis, uplift_sensitivities)
    operational_rows, attrition_rows = _operational_data(analysis, effective_comparison_manifest)
    reliability_rows = _paired_reliability_data(effective_comparison_manifest)
    preference_bound_rows = _preference_score_completion_ranges(
        analysis,
        effective_comparison_manifest,
    )
    (
        terminal_failure_rows,
        mcp_error_rows,
        mcp_summary_rows,
        mcp_stop_rule_rows,
        scored_arm_set_sha256,
    ) = _operational_error_data(scored_arm_paths, completion_interpretation)

    outputs = {
        "study_design_figure": output_dir / "pilot-study-design.tex",
        "system_architecture_figure": output_dir / "pilot-system-architecture.tex",
        "model_primary_table": output_dir / "pilot-model-primary-table.tex",
        "operational_table": output_dir / "pilot-operational-table.tex",
        "endpoint_manifest_table": output_dir / "pilot-endpoint-manifest.tex",
        "flow_data": output_dir / "pilot-flow.csv",
        "measurement_data": output_dir / "pilot-measurement-integrity.csv",
        "model_data": output_dir / "pilot-model-uncertainty.csv",
        "uplift_data": output_dir / "pilot-epicure-robustness.csv",
        "operational_data": output_dir / "pilot-operational-table.csv",
        "attrition_data": output_dir / "pilot-condition-attrition.csv",
        "reliability_data": output_dir / "pilot-condition-reliability.csv",
        "preference_bounds_data": output_dir / "pilot-preference-bounds.csv",
        "terminal_failure_data": output_dir / "pilot-terminal-failures.csv",
        "mcp_error_data": output_dir / "pilot-mcp-errors.csv",
        "mcp_summary_data": output_dir / "pilot-mcp-summary.csv",
        "mcp_stop_rule_data": output_dir / "pilot-mcp-stop-rule.csv",
    }
    _write(outputs["study_design_figure"], _study_design_figure())
    _write(outputs["system_architecture_figure"], _system_architecture_figure())
    _write(outputs["model_primary_table"], _primary_model_table(model_rows))
    _write(outputs["operational_table"], _operational_table(operational_rows))
    _write(outputs["endpoint_manifest_table"], _endpoint_manifest_table(model_manifest))
    _write_csv(outputs["flow_data"], flow_rows)
    _write_csv(outputs["measurement_data"], measurement_rows)
    _write_csv(outputs["model_data"], model_rows)
    _write_csv(outputs["uplift_data"], uplift_rows)
    _write_csv(outputs["operational_data"], operational_rows)
    _write_csv(outputs["attrition_data"], attrition_rows)
    _write_csv(outputs["reliability_data"], reliability_rows)
    _write_csv(outputs["preference_bounds_data"], preference_bound_rows)
    _write_csv(outputs["terminal_failure_data"], terminal_failure_rows)
    _write_csv(outputs["mcp_error_data"], mcp_error_rows)
    _write_csv(outputs["mcp_summary_data"], mcp_summary_rows)
    _write_csv(outputs["mcp_stop_rule_data"], mcp_stop_rule_rows)

    provenance = {
        "schema_version": "flavourbench-pilot-publication-assets-v9",
        "status": "retrospective_pilot_not_official_benchmark",
        "analysis_artifact_sha256": analysis_sha,
        "model_manifest_artifact_sha256": model_manifest_sha,
        "task_bank_artifact_sha256": task_sha,
        "task_review_queue_artifact_sha256": review_sha,
        "comparison_manifest_artifact_sha256": comparison_sha,
        "arm_interpretation_correction_artifact_sha256": (interpretation.artifact_sha256),
        "arm_interpretation_correction_count": len(interpretation.arm_ids),
        "completion_interpretation_correction_artifact_sha256": (
            completion_interpretation.artifact_sha256
        ),
        "completion_interpretation_correction_count": len(completion_interpretation.arm_ids),
        "curation_audit_artifact_sha256": sorted(
            _verify(audit, "curation audit") for audit in curation_audits
        ),
        "generator_sha256": _file_sha256(Path(__file__)),
        "correction_validator_sha256": _file_sha256(
            Path(__file__).with_name("season0_arm_corrections.py")
        ),
        "completion_correction_validator_sha256": _file_sha256(
            Path(__file__).with_name("season0_completion_corrections.py")
        ),
        "input_file_sha256": {
            key: _file_sha256(path) for key, path in sorted((input_paths or {}).items())
        },
        "uplift_weighting_sensitivities": uplift_sensitivities,
        "paired_condition_attrition": {
            str(row["outcome"]): int(row["count"]) for row in attrition_rows
        },
        "paired_condition_reliability": reliability_rows[0],
        "paired_preference_score_completion_sensitivities": preference_bound_rows,
        "scored_arm_set_sha256": scored_arm_set_sha256,
        "target_terminal_failure_taxonomy": terminal_failure_rows,
        "mcp_error_taxonomy": mcp_error_rows,
        "mcp_trace_summary": mcp_summary_rows,
        "mcp_arm_wide_stop_rule_audit": mcp_stop_rule_rows,
        "output_sha256": {key: _file_sha256(path) for key, path in sorted(outputs.items())},
    }
    provenance["artifact_sha256"] = sha256_json(provenance)
    provenance_path = output_dir / "pilot-figure-provenance.json"
    _write(provenance_path, json.dumps(provenance, indent=2, sort_keys=True))
    outputs["provenance"] = provenance_path
    return {key: str(path) for key, path in outputs.items()}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--task-review-queue", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--curation-audit", type=Path, action="append", required=True)
    parser.add_argument("--scored-arm-dir", type=Path, required=True)
    parser.add_argument("--arm-interpretation-correction", type=Path, required=True)
    parser.add_argument("--completion-interpretation-correction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "analysis": args.analysis,
        "model_manifest": args.model_manifest,
        "task_bank": args.task_bank,
        "task_review_queue": args.task_review_queue,
        "comparison_manifest": args.comparison_manifest,
        "arm_interpretation_correction": args.arm_interpretation_correction,
        "completion_interpretation_correction": (args.completion_interpretation_correction),
    }
    for index, path in enumerate(args.curation_audit, start=1):
        paths[f"curation_audit_{index}"] = path
    result = generate_pilot_assets(
        analysis=_load(args.analysis),
        model_manifest=_load(args.model_manifest),
        task_bank=_load(args.task_bank),
        review_queue=_load(args.task_review_queue),
        comparison_manifest=_load(args.comparison_manifest),
        curation_audits=[_load(path) for path in args.curation_audit],
        scored_arm_paths=sorted(args.scored_arm_dir.glob("*.json")),
        arm_interpretation_correction=_load(args.arm_interpretation_correction),
        completion_interpretation_correction=_load(args.completion_interpretation_correction),
        scored_arm_dir=args.scored_arm_dir,
        output_dir=args.output_dir,
        input_paths=paths,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
