from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

try:
    from lab_api import SpaceLabError, score_completion, score_payload
except ModuleNotFoundError:  # repository-root imports used by CI
    from hf.space.lab_api import SpaceLabError, score_completion, score_payload

HERE = Path(__file__).resolve().parent
BUNDLE_PATH = Path(
    os.environ.get(
        "FLAVOURBENCH_BUNDLE",
        HERE / "data-complete-core" / "flavourbench-complete-core-space.json",
    )
)

BLUE = "#1769AA"
CHARCOAL = "#262B33"

CSS = """
:root {
  --fb-accent: #1769AA;
  --fb-accent-soft: #EAF3FA;
  --fb-ink: #262B33;
  --fb-muted: #657180;
  --fb-paper: #F6F8FA;
  --fb-panel: #FFFFFF;
  --fb-rule: #D9E0E7;
  --fb-code: #EEF2F5;
}
.dark {
  --fb-accent-soft: #102B3E;
  --fb-ink: #EAF0F4;
  --fb-muted: #A7B2BC;
  --fb-paper: #11171D;
  --fb-panel: #182129;
  --fb-rule: #30404D;
  --fb-code: #202C35;
}
body, .gradio-container {
  background: var(--fb-paper) !important;
  color: var(--fb-ink) !important;
  font-family: "Geist", "Avenir Next", system-ui, sans-serif !important;
}
.gradio-container { max-width: 1460px !important; }
.fb-shell { max-width: 1360px; margin: 0 auto; }
.fb-hero {
  display: grid;
  grid-template-columns: minmax(0, .9fr) minmax(520px, 1.1fr);
  gap: 54px;
  padding: 48px 8px 34px;
  border-bottom: 1px solid var(--fb-rule);
}
.fb-kicker {
  color: var(--fb-accent);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.fb-hero h1 {
  color: var(--fb-ink);
  font-size: clamp(46px, 6vw, 78px);
  letter-spacing: -.06em;
  line-height: .96;
  margin: 14px 0 18px;
  max-width: 760px;
}
.fb-dek {
  color: var(--fb-muted);
  font-size: 19px;
  line-height: 1.48;
  margin: 0;
  max-width: 640px;
}
.fb-byline {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 22px;
  margin: 16px 0 0;
}
.fb-byline span {
  color: var(--fb-ink);
  display: flex;
  flex-direction: column;
  font-size: 14px;
  font-weight: 650;
  letter-spacing: .01em;
}
.fb-byline small {
  color: var(--fb-muted);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: .04em;
  margin-top: 2px;
}
.fb-stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(92px, 1fr));
  gap: 22px;
  margin-top: 31px;
}
.fb-stat { border-top: 2px solid var(--fb-rule); padding-top: 11px; }
.fb-stat strong {
  color: var(--fb-ink);
  display: block;
  font-size: 26px;
  letter-spacing: -.04em;
  line-height: 1;
}
.fb-stat span {
  color: var(--fb-muted);
  display: block;
  font-size: 11px;
  margin-top: 7px;
  text-transform: uppercase;
  letter-spacing: .07em;
}
.fb-frontier {
  align-self: end;
  background: var(--fb-panel);
  border: 1px solid var(--fb-rule);
  border-top: 4px solid var(--fb-accent);
  border-radius: 8px;
  padding: 20px 22px 16px;
}
.fb-frontier-head {
  align-items: baseline;
  display: flex;
  justify-content: space-between;
  margin-bottom: 13px;
}
.fb-frontier-head strong { font-size: 14px; }
.fb-frontier-head span { color: var(--fb-muted); font-size: 12px; }
.fb-forest-row {
  align-items: center;
  display: grid;
  grid-template-columns: 150px 1fr 42px 38px;
  gap: 10px;
  min-height: 26px;
}
.fb-model {
  color: var(--fb-ink);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fb-axis { height: 13px; position: relative; }
.fb-axis::before {
  background: var(--fb-rule);
  content: "";
  height: 1px;
  left: 0;
  position: absolute;
  right: 0;
  top: 6px;
}
.fb-ci {
  border-top: 2px solid var(--fb-muted);
  height: 1px;
  position: absolute;
  top: 5px;
}
.fb-ci::before, .fb-ci::after {
  background: var(--fb-muted);
  content: "";
  height: 7px;
  position: absolute;
  top: -4px;
  width: 1px;
}
.fb-ci::before { left: 0; }
.fb-ci::after { right: 0; }
.fb-point {
  background: var(--fb-accent);
  height: 11px;
  position: absolute;
  top: 1px;
  width: 3px;
}
.fb-number {
  color: var(--fb-ink);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  text-align: right;
}
.fb-group {
  color: var(--fb-muted);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10px;
  text-align: right;
}
.fb-section { margin: 27px 0 8px; }
.fb-section h2 {
  color: var(--fb-ink);
  font-size: 30px;
  letter-spacing: -.035em;
  margin: 0 0 4px;
}
.fb-section p { color: var(--fb-muted); margin: 0; max-width: 70ch; }
.fb-metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin: 8px 0 16px;
}
.fb-metric {
  background: var(--fb-panel);
  border: 1px solid var(--fb-rule);
  border-radius: 8px;
  padding: 16px;
}
.fb-metric small {
  color: var(--fb-muted);
  display: block;
  font-size: 11px;
  letter-spacing: .05em;
  text-transform: uppercase;
}
.fb-metric strong { color: var(--fb-ink); display: block; font-size: 23px; margin-top: 5px; }
.fb-evidence {
  background: var(--fb-accent-soft);
  border-left: 4px solid var(--fb-accent);
  border-radius: 0 8px 8px 0;
  color: var(--fb-ink);
  line-height: 1.48;
  padding: 17px 19px;
}
.fb-evidence code, .fb-hash {
  background: var(--fb-code);
  color: var(--fb-muted);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
}
.fb-method {
  display: grid;
  grid-template-columns: 1.15fr .85fr;
  gap: 28px;
}
.fb-footer {
  border-top: 1px solid var(--fb-rule);
  color: var(--fb-muted);
  font-size: 12px;
  margin-top: 36px;
  padding: 18px 8px 26px;
}
.gradio-dataframe, .block { border-radius: 8px !important; }
@media (max-width: 980px) {
  .fb-hero, .fb-method { grid-template-columns: 1fr; }
  .fb-hero { gap: 28px; padding-top: 34px; }
}
@media (max-width: 700px) {
  .fb-hero h1 { font-size: 48px; }
  .fb-stats, .fb-metric-grid { grid-template-columns: repeat(2, 1fr); }
  .fb-forest-row { grid-template-columns: 105px 1fr 36px 30px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


class SpaceDataError(RuntimeError):
    """The public explorer bundle is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _load_bundle() -> dict[str, Any]:
    if BUNDLE_PATH.is_symlink() or not BUNDLE_PATH.is_file():
        raise FileNotFoundError(
            f"Complete-core Space bundle not found at {BUNDLE_PATH}. Set FLAVOURBENCH_BUNDLE."
        )
    value = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = str(payload.pop("artifact_sha256", ""))
    if (
        recorded != hashlib.sha256(_canonical(payload)).hexdigest()
        or value.get("schema_version") != "flavourbench-complete-core-space-bundle-v1"
        or value.get("status") != "final_complete_common_core"
    ):
        raise SpaceDataError("complete-core Space bundle failed verification")
    return value


BUNDLE = _load_bundle()
MODELS = BUNDLE["models"]
TASKS = BUNDLE["tasks"]
LAB_TASKS = BUNDLE.get("lab_tasks", [])
PAIRWISE = BUNDLE["pairwise_comparisons"]
MODEL_COUNT = len(MODELS)
TASK_COUNT = len(TASKS)
PAIR_COUNT = len(PAIRWISE)
DESIGN = BUNDLE["design"]
PANEL_COUNT = int(DESIGN.get("panel_count", 1))
INDEPENDENT_CLUSTER_COUNT = int(DESIGN.get("unique_anchor_clusters", TASK_COUNT))
PRIMARY_COUNT = MODEL_COUNT * TASK_COUNT
MODEL_BY_NAME = {str(row["model_name"]): row for row in MODELS}
MODEL_BY_ID = {str(row["model_id"]): row for row in MODELS}
TASK_BY_ID = {str(row["task_id"]): row for row in TASKS}
LAB_TASK_BY_ID = {str(row["task_id"]): row for row in LAB_TASKS}
if set(TASK_BY_ID) & set(LAB_TASK_BY_ID):
    raise SpaceDataError("official and training task IDs overlap")
OBSERVATIONS = {
    (str(row["model_id"]), str(row["task_id"])): row for row in BUNDLE["primary_observations"]
}
PAIR_INDEX: dict[tuple[str, str], dict[str, Any]] = {}
for _row in PAIRWISE:
    PAIR_INDEX[(str(_row["left_model_id"]), str(_row["right_model_id"]))] = _row


def _rank_key(row: dict[str, Any]) -> tuple[bool, int, str]:
    rank = row.get("point_estimate_rank")
    return rank is None, int(rank or 10_000), str(row["model_id"])


DISPLAY_MODELS = sorted(MODELS, key=_rank_key)
MODEL_NAMES = [str(row["model_name"]) for row in DISPLAY_MODELS]
TASK_LABEL_TO_ID = {
    f"{row['task_id']} | {str(row['family']).replace('_', ' ')}": str(row["task_id"])
    for row in TASKS
}


def _completion_diagnostic(model_id: str) -> dict[str, Any]:
    family_rows: dict[str, list[dict[str, Any]]] = {}
    for task_id, task in TASK_BY_ID.items():
        family = str(task["family"])
        family_rows.setdefault(family, []).append(OBSERVATIONS[(model_id, task_id)])
    conditional_family_scores: dict[str, float] = {}
    completed_by_family: dict[str, int] = {}
    scheduled_by_family: dict[str, int] = {}
    for family, rows in family_rows.items():
        completed = [
            row
            for row in rows
            if row["status"] == "completed" and bool(row.get("scoring", {}).get("parseable", True))
        ]
        scheduled_by_family[family] = len(rows)
        completed_by_family[family] = len(completed)
        conditional_family_scores[family] = (
            sum(float(row["scoring"]["score"]) for row in completed) / len(completed)
            if completed
            else 0.0
        )
    completed = sum(completed_by_family.values())
    return {
        "scheduled": len(TASK_BY_ID),
        "completed": completed,
        "failed": len(TASK_BY_ID) - completed,
        "completion_rate": completed / len(TASK_BY_ID),
        "conditional_family_scores": conditional_family_scores,
        "completed_by_family": completed_by_family,
        "scheduled_by_family": scheduled_by_family,
        "conditional_equal_family_score": sum(conditional_family_scores.values())
        / len(conditional_family_scores),
    }


def _short(value: str) -> str:
    return (
        value.replace("GPT-5.6 ", "5.6 ")
        .replace("Claude ", "")
        .replace("DeepSeek ", "DS ")
        .replace("Command ", "Cmd ")
    )


def _frontier_html() -> str:
    rows = []
    for model in DISPLAY_MODELS[:10]:
        score = float(model["flavourbench_score"])
        lower, upper = (float(value) for value in model["score_simultaneous_95_ci"])
        left = max(0.0, min(100.0, lower))
        right = max(left, min(100.0, upper))
        group = model.get("statistical_rank_group")
        rows.append(
            "<div class='fb-forest-row'>"
            f"<div class='fb-model' title='{html.escape(str(model['model_name']))}'>"
            f"{html.escape(_short(str(model['model_name'])))}</div>"
            "<div class='fb-axis'>"
            f"<span class='fb-ci' style='left:{left:.3f}%;width:{right - left:.3f}%'></span>"
            f"<span class='fb-point' style='left:{score:.3f}%'></span>"
            "</div>"
            f"<div class='fb-number'>{score:.1f}</div>"
            f"<div class='fb-group'>G{group if group is not None else '-'}</div>"
            "</div>"
        )
    return "".join(rows)


def _hero_html() -> str:
    inference = BUNDLE["analysis"]["inference"]
    panel_phrase = f" across {PANEL_COUNT} collection panels" if PANEL_COUNT > 1 else ""
    return f"""
    <div class="fb-shell fb-hero">
      <section>
        <div class="fb-kicker">Executable culinary evaluation</div>
        <h1>{TASK_COUNT:,} decisions.<br>No model judge.</h1>
        <p class="fb-dek">Executable score maps rank {MODEL_COUNT} frontier endpoints{panel_phrase}
        with anchor-clustered uncertainty and inspectable responses.</p>
        <div class="fb-byline">
          <span>Josef Chen<small>Independent Researcher</small></span>
          <span>Erim Hayretci<small>Imperial College London</small></span>
        </div>
        <div class="fb-stats">
          <div class="fb-stat"><strong>{MODEL_COUNT}</strong><span>models</span></div>
          <div class="fb-stat"><strong>{TASK_COUNT}</strong><span>tasks</span></div>
          <div class="fb-stat"><strong>{PRIMARY_COUNT:,}</strong><span>primary cells</span></div>
          <div class="fb-stat"><strong>{inference["pairwise_hypotheses"]}</strong><span>paired tests</span></div>
        </div>
      </section>
      <section class="fb-frontier" aria-label="Score forest with simultaneous intervals">
        <div class="fb-frontier-head"><strong>FlavourBench Score</strong><span>Top 10, simultaneous 95%</span></div>
        {_frontier_html()}
      </section>
    </div>
    """


def _leaderboard_frame() -> pd.DataFrame:
    rows = []
    for model in DISPLAY_MODELS:
        ci = model["score_simultaneous_95_ci"]
        rank_ci = model["bootstrap_rank_95_interval"]
        rows.append(
            {
                "Rank": model["point_estimate_rank"],
                "Group": model["statistical_rank_group"],
                "Model": model["model_name"],
                "Score": round(float(model["flavourbench_score"]), 2),
                "Simultaneous 95%": f"{ci[0]:.2f} to {ci[1]:.2f}",
                "Rank 95%": f"{rank_ci[0]} to {rank_ci[1]}",
                "Cells": f"{model['coverage']['valid_scored']}/{TASK_COUNT}",
                "Backend": model["execution_backend"],
            }
        )
    return pd.DataFrame(rows)


def _model_detail(model_name: str) -> tuple[str, pd.DataFrame]:
    model = MODEL_BY_NAME[model_name]
    rank_interval = model["bootstrap_rank_95_interval"]
    coverage = model["coverage"]
    panel_replication = model["panel_replication"]
    summary = f"""
    <div class="fb-metric-grid">
      <div class="fb-metric"><small>FlavourBench Score</small><strong>{model["flavourbench_score"]:.2f}</strong></div>
      <div class="fb-metric"><small>Complete cells</small><strong>{coverage["valid_scored"]}/{coverage["scheduled"]}</strong></div>
      <div class="fb-metric"><small>Point rank</small><strong>#{model["point_estimate_rank"]}</strong></div>
      <div class="fb-metric"><small>Statistical group</small><strong>G{model["statistical_rank_group"]}</strong></div>
      <div class="fb-metric"><small>Bootstrap rank</small><strong>{rank_interval[0]}-{rank_interval[1]}</strong></div>
      <div class="fb-metric"><small>Backend</small><strong>{html.escape(str(model["execution_backend"]))}</strong></div>
    </div>
    <div class="fb-evidence">
      <strong>Identical evidence for every model.</strong> This score uses all {TASK_COUNT} common-core
      cells: {coverage["valid_scored_per_family"]["substitution"]} substitution,
      {coverage["valid_scored_per_family"]["pairing"]} pairing, and
      {coverage["valid_scored_per_family"]["constraint"]} constraint tasks. Panel scores are
      {float(panel_replication["panel_1"]):.2f} and
      {float(panel_replication["panel_2"]):.2f}
      ({float(panel_replication["difference"]):+.2f}).
    </div>
    """
    family_rows = [
        {
            "Family": family.replace("_", " ").title(),
            "Score": round(float(score), 3),
            "Cells": coverage["valid_scored_per_family"][family],
        }
        for family, score in model["family_scores"].items()
    ]
    chance = model["chance_comparison"]
    chance_score = round(float(chance["exact_chance_score"]), 3)
    family_rows.append(
        {"Family": "Exact chance baseline", "Score": chance_score, "Cells": TASK_COUNT}
    )
    return summary, pd.DataFrame(family_rows)


def _task_detail(
    model_name: str, task_label: str
) -> tuple[str, str, dict[str, str], pd.DataFrame, str, str]:
    model = MODEL_BY_NAME[model_name]
    task_id = TASK_LABEL_TO_ID[task_label]
    task = TASK_BY_ID[task_id]
    observation = OBSERVATIONS[(str(model["model_id"]), task_id)]
    scoring = observation["scoring"]
    observed = scoring.get("observed_selection")
    optimum = str(task["optimal_selection"])
    observed_ingredients = [task["choices"][label] for label in observed] if observed else []
    optimum_ingredients = [task["choices"][label] for label in optimum]
    if observation.get("status") != "completed" or not bool(scoring.get("parseable")):
        raise SpaceDataError("common-core observation is not release-valid")
    status = f"""
    <div class="fb-evidence">
      <strong>{html.escape(model_name)}</strong> selected
      <code>{html.escape(str(observed))}</code> and scored
      <strong>{float(scoring["score"]):.2f}</strong>. The optimum is
      <code>{html.escape(optimum)}</code>.
      <br>Observed: {html.escape(", ".join(observed_ingredients))}
      <br>Optimum: {html.escape(", ".join(optimum_ingredients))}
    </div>
    """
    ranked = sorted(
        task["selection_scores_bps"].items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    score_rows = [
        {
            "Selection": selection,
            "Ingredients": ", ".join(task["choices"][label] for label in selection),
            "Score": int(score) / 100,
            "Role": (
                "model selection"
                if selection == observed
                else "optimum"
                if selection == optimum
                else ""
            ),
        }
        for selection, score in ranked[:12]
    ]
    provenance = (
        f"Response SHA-256: `{observation['artifact_sha256']}`  \n"
        f"Actual model: `{observation.get('actual_model_id')}`  \n"
        f"Provider: `{observation.get('actual_provider')}`  \n"
        f"Prompt SHA-256: `{task['prompt_sha256']}`"
    )
    answer = str(observation.get("answer_excerpt") or "No answer was recorded.")
    if observation.get("answer_truncated"):
        answer += "\n\n[Excerpt truncated. The full response is in the dataset.]"
    return (
        status,
        str(task["prompt"]),
        dict(task["choices"]),
        pd.DataFrame(score_rows),
        answer,
        provenance,
    )


def _pair_detail(left_name: str, right_name: str) -> str:
    left = str(MODEL_BY_NAME[left_name]["model_id"])
    right = str(MODEL_BY_NAME[right_name]["model_id"])
    if left == right:
        return "<div class='fb-evidence'>Choose two different models.</div>"
    row = PAIR_INDEX.get((left, right))
    sign = 1.0
    if row is None:
        row = PAIR_INDEX[(right, left)]
        sign = -1.0
    difference = sign * float(row["mean_difference"])
    interval = [sign * float(value) for value in row["bootstrap_95_ci"]]
    interval.sort()
    verdict = "distinguishable after Holm correction" if row["holm_significant"] else "not resolved"
    return f"""
    <div class="fb-evidence">
      <strong>{html.escape(left_name)}</strong> minus <strong>{html.escape(right_name)}</strong>:
      <strong>{difference:+.3f} points</strong> (bootstrap 95% {interval[0]:+.3f} to {interval[1]:+.3f}).
      The comparison is <strong>{verdict}</strong> across all {PAIR_COUNT} tests
      (shared valid tasks: <code>{row.get("shared_valid_tasks", TASK_COUNT)}</code>).
      <br>Holm p = <code>{float(row["holm_p"]):.4g}</code>, paired Cohen dz =
      <code>{row.get("cohen_dz")}</code>.
    </div>
    """


def _score_completion_api(task_id: str, completion: str) -> dict[str, Any]:
    """Named Gradio endpoint for one released-map reward lookup."""

    return score_completion(TASK_BY_ID, task_id, completion)


def _score_submission_api(payload: str) -> dict[str, Any]:
    """Named Gradio endpoint for a complete JSON/JSONL response artifact."""

    report, _ = score_payload(TASKS, payload)
    return report


def _training_reward_api(task_id: str, completion: str) -> dict[str, Any]:
    """Named endpoint for a development-map reward used during training."""

    result = score_completion(LAB_TASK_BY_ID, task_id, completion)
    task = LAB_TASK_BY_ID[task_id]
    return {
        **result,
        "track": "development_training",
        "split": task["lab_split"],
        "family": task["family"],
        "official_leaderboard_eligible": False,
    }


def _score_upload(
    artifact_path: str | None,
    model_name: str,
    disclosure: str,
) -> tuple[str, pd.DataFrame, str | None]:
    """Score one upload without publishing it or changing the leaderboard."""

    if not artifact_path:
        raise gr.Error("Choose a JSON or JSONL response artifact first.")
    source = Path(artifact_path)
    if source.is_symlink() or not source.is_file():
        raise gr.Error("The upload is not a regular file.")
    if source.stat().st_size > 16 * 1024 * 1024:
        raise gr.Error("The upload exceeds 16 MiB.")
    try:
        report, per_task = score_payload(TASKS, source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SpaceLabError) as error:
        raise gr.Error(str(error)) from error

    label = " ".join(str(model_name or "").split())[:160] or "Unnamed model"
    disclosure = " ".join(str(disclosure or "Not disclosed").split())[:240]
    report["submission"] = {
        "model_name": label,
        "method_disclosure": disclosure,
        "scored_at_utc": datetime.now(UTC).isoformat(),
        "published_to_leaderboard": False,
    }
    report.pop("artifact_sha256", None)
    report["artifact_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    coverage = report["coverage"]
    if report["comparable"]:
        score = float(report["flavourbench_score"])
        summary = (
            f"### {label}: {score:.2f}\n\n"
            f"**Comparable lab score.** All {coverage['tasks']} tasks were present and parseable. "
            "This result is not added to the public leaderboard automatically."
        )
    else:
        diagnostic = report.get("diagnostic_valid_score")
        diagnostic_text = f"{float(diagnostic):.2f}" if diagnostic is not None else "unavailable"
        summary = (
            f"### {label}: no FlavourBench Score issued\n\n"
            f"Valid coverage is **{coverage['valid']}/{coverage['tasks']}** "
            f"({coverage['fraction_valid']:.1%}); {coverage['missing']} missing and "
            f"{coverage['invalid']} invalid. The valid-only diagnostic is {diagnostic_text}, "
            "but it is not a comparable leaderboard score."
        )

    rows = pd.DataFrame(per_task)[
        ["task_id", "family", "status", "observed_selection", "score", "optimal"]
    ].rename(
        columns={
            "task_id": "Task",
            "family": "Family",
            "status": "Status",
            "observed_selection": "Selection",
            "score": "Score",
            "optimal": "Optimal",
        }
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="flavourbench-report-",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        report_path = handle.name
    return summary, rows, report_path


theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EAF3FA",
        c100="#D5E8F5",
        c200="#A9D0E9",
        c300="#75B3DA",
        c400="#4292C6",
        c500=BLUE,
        c600="#12588F",
        c700="#104873",
        c800="#103C5D",
        c900="#10334E",
        c950="#081E30",
    ),
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Geist", weights=(400, 500, 600, 700)),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono", weights=(400, 600)),
).set(
    body_background_fill="#F6F8FA",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_label_text_weight="600",
    button_primary_background_fill=BLUE,
    button_primary_background_fill_hover="#12588F",
)


with gr.Blocks(title="FlavourBench | Executable culinary evaluation") as demo:
    gr.HTML(_hero_html())
    with gr.Tabs():
        with gr.Tab("Leaderboard"):
            gr.HTML(
                """
                <div class="fb-section">
                  <h2>The complete common-core leaderboard</h2>
                  <p>Every model is scored on the same 534 tasks. Point ranks sit beside statistical groups and simultaneous intervals.</p>
                </div>
                """
            )
            gr.Dataframe(
                value=_leaderboard_frame(),
                interactive=False,
                wrap=True,
                show_search="filter",
                show_row_numbers=False,
                column_widths=[55, 55, 245, 80, 180, 110, 125, 120],
            )
            gr.Markdown(
                "A point rank orders the observed scores. A statistical group keeps models together "
                "when the shared-task evidence does not separate them after multiplicity control."
            )

        with gr.Tab("Model profile"):
            gr.HTML(
                """
                <div class="fb-section">
                  <h2>Family profile and panel replication</h2>
                  <p>Inspect where a model earns its score and how its estimate moves across the two independently compiled panels.</p>
                </div>
                """
            )
            model_selector = gr.Dropdown(
                choices=MODEL_NAMES,
                value=MODEL_NAMES[0],
                label="Model",
                filterable=True,
            )
            model_summary = gr.HTML(_model_detail(MODEL_NAMES[0])[0])
            family_table = gr.Dataframe(
                value=_model_detail(MODEL_NAMES[0])[1],
                interactive=False,
                wrap=True,
                show_row_numbers=False,
            )
            model_selector.change(
                _model_detail,
                inputs=model_selector,
                outputs=[model_summary, family_table],
            )

        with gr.Tab("Task lens"):
            gr.HTML(
                """
                <div class="fb-section">
                  <h2>Open one scored decision</h2>
                  <p>Read the prompt, the model answer, and the top of the precomputed reward surface.</p>
                </div>
                """
            )
            with gr.Row():
                task_model = gr.Dropdown(
                    choices=MODEL_NAMES,
                    value=MODEL_NAMES[0],
                    label="Model",
                    filterable=True,
                    scale=1,
                )
                task_selector = gr.Dropdown(
                    choices=list(TASK_LABEL_TO_ID),
                    value=next(iter(TASK_LABEL_TO_ID)),
                    label="Task",
                    filterable=True,
                    scale=2,
                )
                inspect_task = gr.Button("Inspect", variant="primary", scale=0)
            initial = _task_detail(MODEL_NAMES[0], next(iter(TASK_LABEL_TO_ID)))
            task_status = gr.HTML(initial[0])
            prompt = gr.Textbox(value=initial[1], label="Exact prompt", lines=13, interactive=False)
            choices = gr.JSON(value=initial[2], label="Candidates")
            score_map = gr.Dataframe(
                value=initial[3],
                label="Top 12 of 56 frozen selections",
                interactive=False,
                wrap=True,
                show_row_numbers=False,
            )
            answer = gr.Markdown(value=initial[4], label="Model response")
            provenance = gr.Markdown(value=initial[5], label="Provenance")
            inspect_task.click(
                _task_detail,
                inputs=[task_model, task_selector],
                outputs=[task_status, prompt, choices, score_map, answer, provenance],
            )

        with gr.Tab("Evaluate your model"):
            gr.HTML(
                f"""
                <div class="fb-section">
                  <h2>Bring a checkpoint or endpoint</h2>
                  <p>Run the prompts in your own environment, upload the response artifact, and score it here. Credentials and model weights never enter this Space.</p>
                </div>
                <div class="fb-evidence">
                  <strong>Comparable means complete.</strong> A lab score requires one valid answer for every one of the {TASK_COUNT} released tasks. Partial runs receive coverage and per-task diagnostics only.
                </div>
                """
            )
            with gr.Row():
                lab_name = gr.Textbox(
                    label="Model or experiment name",
                    placeholder="lab/model-name · checkpoint · decoding policy",
                    scale=2,
                )
                lab_disclosure = gr.Dropdown(
                    choices=[
                        "Base model; no FlavourBench training",
                        "Fine-tuned without FlavourBench training data",
                        "Fine-tuned with FlavourBench lab data or reward",
                        "Other; disclose in the artifact",
                    ],
                    value="Base model; no FlavourBench training",
                    label="Method disclosure",
                    scale=2,
                )
            lab_upload = gr.File(
                label="Responses (.jsonl or .json)",
                file_types=[".jsonl", ".json"],
                type="filepath",
            )
            score_upload = gr.Button("Score artifact", variant="primary")
            lab_summary = gr.Markdown()
            lab_rows = gr.Dataframe(
                interactive=False,
                wrap=True,
                show_search="filter",
                show_row_numbers=False,
                label="Per-task results",
            )
            lab_report = gr.File(label="Download content-addressed report")
            score_upload.click(
                _score_upload,
                inputs=[lab_upload, lab_name, lab_disclosure],
                outputs=[lab_summary, lab_rows, lab_report],
                api_name="score_uploaded_submission",
            )
            gr.Markdown(
                """
The accepted JSONL contract is one object per task:

```json
{"task_id":"...","status":"completed","response":"FINAL_SELECTION: A,B,C"}
```

For automation, use the named `score_completion`, `score_submission`, and `training_reward`
endpoints shown under **Use via API** in the Space footer. `training_reward` accepts only the 426
development task IDs; it cannot score against the leaderboard by accident. The source repository
also provides a local runner, scorer, schemas, and TRL recipes. Local reward lookup remains the
recommended path for high-throughput training.
                """
            )

        with gr.Tab("Pairwise evidence"):
            gr.HTML(
                f"""
                <div class="fb-section">
                  <h2>Is the score gap resolved?</h2>
                  <p>Query any paired contrast from the {PAIR_COUNT}-test family.</p>
                </div>
                """
            )
            with gr.Row():
                left_model = gr.Dropdown(
                    choices=MODEL_NAMES,
                    value=MODEL_NAMES[0],
                    label="First model",
                    filterable=True,
                )
                right_model = gr.Dropdown(
                    choices=MODEL_NAMES,
                    value=MODEL_NAMES[1],
                    label="Second model",
                    filterable=True,
                )
                compare = gr.Button("Compare", variant="primary", scale=0)
            pair_result = gr.HTML(_pair_detail(MODEL_NAMES[0], MODEL_NAMES[1]))
            compare.click(_pair_detail, inputs=[left_model, right_model], outputs=pair_result)

        with gr.Tab("Method and download"):
            gr.HTML(
                f"""
                <div class="fb-section">
                  <h2>One metric, complete evidence</h2>
                  <p>The Space makes no model or provider calls. Its lab API performs deterministic lookups against the released reward maps.</p>
                </div>
                <div class="fb-method">
                  <div>
                    <h3>Scoring contract</h3>
                    <p>Every task exposes eight candidates and all 56 three-item scores. The
                    FlavourBench Score is the equal-family mean across substitution, pairing, and
                    constraint tasks. The release uses a complete 27-by-534 matrix: every ranked
                    model contributes one valid response to every scored task.</p>
                    <h3>Inference</h3>
                    <p>Results use {INDEPENDENT_CLUSTER_COUNT:,} ingredient-anchor clusters,
                    50,000 ingredient-anchor cluster bootstraps, simultaneous score bands,
                    100,000 cluster sign flips, Holm correction, exact-chance tests, bootstrap rank
                    intervals, and an independently compiled second panel.</p>
                  </div>
                  <aside class="fb-evidence">
                    <strong>Exact release</strong><br>
                    <span class="fb-hash">{BUNDLE["release_artifact_sha256"]}</span><br><br>
                    {MODEL_COUNT} models<br>{TASK_COUNT} tasks<br>{INDEPENDENT_CLUSTER_COUNT:,} anchor clusters<br>{PRIMARY_COUNT:,} complete scored responses<br>{BUNDLE["analysis"]["resolved_pair_count"]}/{PAIR_COUNT} resolved pairs
                  </aside>
                </div>
                """
            )
            gr.Markdown(
                """
```bash
git clone https://github.com/josefchen/flavourbench.git
cd flavourbench
pip install -e '.[dev]'
make -C paper -f Makefile.powered verify
make -C paper -f Makefile.powered arxiv
```

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) | [Dataset](https://huggingface.co/datasets/josefchen/flavourbench) | [Source](https://github.com/josefchen/flavourbench)
                """
            )

    gr.api(
        _score_completion_api,
        api_name="score_completion",
        api_description="Score one completion against one released FlavourBench task.",
        queue=False,
    )
    gr.api(
        _score_submission_api,
        api_name="score_submission",
        api_description="Score a complete response artifact supplied as JSON or JSON Lines text.",
        queue=False,
    )
    gr.api(
        _training_reward_api,
        api_name="training_reward",
        api_description="Return an Epicure-derived dense reward for a development task completion.",
        queue=False,
    )

    gr.HTML(
        """
        <div class="fb-shell fb-footer">
          FlavourBench | Josef Chen, Independent Researcher · Erim Hayretci, Imperial College London
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
