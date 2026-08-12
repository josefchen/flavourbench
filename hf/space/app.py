from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE_PATH = Path(
    os.environ.get(
        "FLAVOURBENCH_BUNDLE",
        HERE / "data-powered" / "flavourbench-powered-space.json",
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
  color: var(--fb-ink);
  font-size: 14px;
  font-weight: 650;
  letter-spacing: .01em;
  margin: 16px 0 0;
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
  grid-template-columns: repeat(4, 1fr);
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
            f"Powered Space bundle not found at {BUNDLE_PATH}. Set FLAVOURBENCH_BUNDLE."
        )
    value = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = str(payload.pop("artifact_sha256", ""))
    if (
        recorded != hashlib.sha256(_canonical(payload)).hexdigest()
        or value.get("schema_version") != "flavourbench-powered-space-bundle-v1"
        or value.get("status") != "final_complete"
    ):
        raise SpaceDataError("powered Space bundle failed verification")
    return value


BUNDLE = _load_bundle()
MODELS = BUNDLE["models"]
TASKS = BUNDLE["tasks"]
PAIRWISE = BUNDLE["pairwise_comparisons"]
MODEL_BY_NAME = {str(row["model_name"]): row for row in MODELS}
MODEL_BY_ID = {str(row["model_id"]): row for row in MODELS}
TASK_BY_ID = {str(row["task_id"]): row for row in TASKS}
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
    return f"""
    <div class="fb-shell fb-hero">
      <section>
        <div class="fb-kicker">Executable culinary evaluation</div>
        <h1>640 decisions.<br>No model judge.</h1>
        <p class="fb-dek">Executable score maps rank 20 frontier endpoints with shared-task
        uncertainty and inspectable responses.</p>
        <p class="fb-byline">Josef Chen · Jakub Radzikowski · Erim Hayretci</p>
        <div class="fb-stats">
          <div class="fb-stat"><strong>{len(MODELS)}</strong><span>models</span></div>
          <div class="fb-stat"><strong>{len(TASKS)}</strong><span>tasks</span></div>
          <div class="fb-stat"><strong>{len(MODELS) * len(TASKS):,}</strong><span>primary cells</span></div>
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
        repeat = model.get("repeatability") or {}
        rows.append(
            {
                "Rank": model.get("point_estimate_rank") or "DNF",
                "Group": model.get("statistical_rank_group") or "DNF",
                "Model": model["model_name"],
                "Score": round(float(model["flavourbench_score"]), 2),
                "Simultaneous 95%": f"{ci[0]:.2f} to {ci[1]:.2f}",
                "Completed": f"{model['availability']['completed']}/640",
                "Repeat Jaccard": (
                    round(float(repeat["mean_ingredient_set_jaccard"]), 3) if repeat else None
                ),
                "Route": model["provider_name"],
            }
        )
    return pd.DataFrame(rows)


def _model_detail(model_name: str) -> tuple[str, pd.DataFrame]:
    model = MODEL_BY_NAME[model_name]
    repeat = model.get("repeatability") or {}
    rank_interval = model.get("bootstrap_rank_95_interval") or [None, None]
    summary = f"""
    <div class="fb-metric-grid">
      <div class="fb-metric"><small>FlavourBench Score</small><strong>{model["flavourbench_score"]:.2f}</strong></div>
      <div class="fb-metric"><small>Statistical group</small><strong>G{model.get("statistical_rank_group") or "-"}</strong></div>
      <div class="fb-metric"><small>Bootstrap rank</small><strong>{rank_interval[0]}-{rank_interval[1]}</strong></div>
      <div class="fb-metric"><small>Repeat Jaccard</small><strong>{float(repeat.get("mean_ingredient_set_jaccard", 0)):.3f}</strong></div>
    </div>
    """
    family_rows = [
        {
            "Family": family.replace("_", " ").title(),
            "Score": round(float(score), 3),
        }
        for family, score in model["family_scores"].items()
    ]
    chance = model["chance_comparison"]
    family_rows.append(
        {
            "Family": "Exact chance baseline",
            "Score": round(float(chance["exact_chance_score"]), 3),
        }
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
    status = f"""
    <div class="fb-evidence">
      <strong>{html.escape(model_name)}</strong> selected
      <code>{html.escape(str(observed or "no valid selection"))}</code> and scored
      <strong>{float(scoring["score"]):.2f}</strong>. The optimum is
      <code>{html.escape(optimum)}</code>.
      <br>Observed: {html.escape(", ".join(observed_ingredients) or "none")}
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
      The comparison is <strong>{verdict}</strong> across all 190 tests.
      <br>Holm p = <code>{float(row["holm_p"]):.4g}</code>, paired Cohen dz =
      <code>{row.get("cohen_dz")}</code>.
    </div>
    """


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
                  <h2>The powered frontier panel</h2>
                  <p>Point ranks are shown beside statistical groups and simultaneous intervals.</p>
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
                  <h2>Family profile and repeatability</h2>
                  <p>Inspect where a model earns its score and whether its selection survives relabeling.</p>
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

        with gr.Tab("Pairwise evidence"):
            gr.HTML(
                """
                <div class="fb-section">
                  <h2>Is the score gap resolved?</h2>
                  <p>Query any paired contrast from the 190-test family.</p>
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
                  <p>The Space makes no model or provider calls.</p>
                </div>
                <div class="fb-method">
                  <div>
                    <h3>Scoring contract</h3>
                    <p>Every task exposes eight candidates and all 56 three-item scores. The
                    FlavourBench Score is the equal-family mean over 640 tasks. Invalid and failed
                    responses remain in the denominator at zero.</p>
                    <h3>Inference</h3>
                    <p>Results use 50,000 family-stratified shared-task bootstraps, simultaneous
                    score bands, 100,000 sign flips, Holm correction, exact-chance tests, and 64
                    label-permuted repeats per model.</p>
                  </div>
                  <aside class="fb-evidence">
                    <strong>Exact release</strong><br>
                    <span class="fb-hash">{BUNDLE["release_artifact_sha256"]}</span><br><br>
                    20 models<br>640 tasks<br>12,800 primary responses<br>1,280 repeats
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
make -C paper -f Makefile.powered analysis
make -C paper -f Makefile.powered arxiv
```

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) | [Dataset](https://huggingface.co/datasets/josefchen/flavourbench) | [Source](https://github.com/josefchen/flavourbench)
                """
            )

    gr.HTML(
        """
        <div class="fb-shell fb-footer">
          FlavourBench | Josef Chen · Jakub Radzikowski · Erim Hayretci
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
