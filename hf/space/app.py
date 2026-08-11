from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
RELEASE_PATH = Path(
    os.environ.get("FLAVOURBENCH_RELEASE", HERE / "data" / "epicure-native-release.json")
)

BRAND = {
    "blue": "#1769AA",
    "gold": "#E6A11A",
    "teal": "#168C7A",
    "red": "#C75450",
    "charcoal": "#262B33",
    "paper": "#F7F8FA",
}

CSS = """
:root {
  --fb-blue: #1769AA;
  --fb-blue-soft: #EAF3FA;
  --fb-gold: #E6A11A;
  --fb-teal: #168C7A;
  --fb-red: #C75450;
  --fb-ink: #262B33;
  --fb-muted: #66707D;
  --fb-paper: #F7F8FA;
  --fb-panel: #FFFFFF;
  --fb-rule: #DDE3EA;
  --fb-code: #EEF2F5;
}
.dark {
  --fb-blue-soft: #112B3E;
  --fb-ink: #ECF1F5;
  --fb-muted: #A7B2BC;
  --fb-paper: #11171D;
  --fb-panel: #182129;
  --fb-rule: #30404D;
  --fb-code: #202C35;
}
body, .gradio-container {
  background: var(--fb-paper) !important;
  color: var(--fb-ink) !important;
  font-family: "Geist", "Inter", "Avenir Next", system-ui, sans-serif !important;
}
.gradio-container { max-width: 1440px !important; }
.fb-shell { max-width: 1320px; margin: 0 auto; }
.fb-kicker {
  color: var(--fb-blue);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.fb-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(420px, .95fr);
  gap: 42px;
  padding: 54px 8px 38px;
  border-bottom: 1px solid var(--fb-rule);
}
.fb-hero h1 {
  color: var(--fb-ink);
  font-size: clamp(52px, 7vw, 92px);
  letter-spacing: -.065em;
  line-height: .92;
  margin: 14px 0 24px;
}
.fb-dek {
  color: var(--fb-muted);
  font-size: 20px;
  line-height: 1.5;
  margin: 0;
  max-width: 720px;
}
.fb-dek strong { color: var(--fb-ink); font-weight: 650; }
.fb-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 18px 30px;
  margin-top: 34px;
}
.fb-stat { min-width: 118px; }
.fb-stat strong {
  color: var(--fb-ink);
  display: block;
  font-size: 29px;
  letter-spacing: -.04em;
  line-height: 1;
}
.fb-stat span {
  color: var(--fb-muted);
  display: block;
  font-size: 12px;
  margin-top: 7px;
  text-transform: uppercase;
  letter-spacing: .08em;
}
.fb-frontier {
  align-self: end;
  background: var(--fb-panel);
  border: 1px solid var(--fb-rule);
  border-top: 4px solid var(--fb-blue);
  padding: 22px 22px 17px;
}
.fb-frontier-head {
  align-items: baseline;
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
}
.fb-frontier-head strong { font-size: 14px; }
.fb-frontier-head span { color: var(--fb-muted); font-size: 12px; }
.fb-rail-row {
  align-items: center;
  display: grid;
  grid-template-columns: 116px 1fr 34px;
  gap: 10px;
  margin: 10px 0;
}
.fb-rail-label {
  color: var(--fb-ink);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fb-rail {
  background: var(--fb-code);
  height: 8px;
  position: relative;
}
.fb-rail-base { background: var(--fb-blue); height: 8px; left: 0; position: absolute; }
.fb-rail-uplift { background: var(--fb-gold); height: 8px; position: absolute; }
.fb-rail-score {
  color: var(--fb-muted);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  text-align: right;
}
.fb-note {
  border-left: 3px solid var(--fb-gold);
  color: var(--fb-muted);
  font-size: 13px;
  line-height: 1.45;
  margin-top: 17px;
  padding-left: 12px;
}
.fb-section-title { margin: 30px 0 6px; }
.fb-section-title h2 {
  color: var(--fb-ink);
  font-size: 31px;
  letter-spacing: -.035em;
  margin: 5px 0 2px;
}
.fb-section-title p { color: var(--fb-muted); margin: 0; }
.fb-pair-status {
  align-items: stretch;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin: 8px 0 16px;
}
.fb-condition {
  background: var(--fb-panel);
  border: 1px solid var(--fb-rule);
  padding: 16px;
}
.fb-condition.good { border-top: 4px solid var(--fb-teal); }
.fb-condition.bad { border-top: 4px solid var(--fb-red); }
.fb-condition.unknown { border-top: 4px solid var(--fb-muted); }
.fb-condition small {
  color: var(--fb-muted);
  display: block;
  font-size: 11px;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.fb-condition strong { display: block; font-size: 22px; margin: 7px 0 4px; }
.fb-condition code, .fb-hash {
  background: var(--fb-code);
  color: var(--fb-muted);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
}
.fb-method-grid {
  display: grid;
  grid-template-columns: 1.25fr .75fr;
  gap: 28px;
}
.fb-callout {
  background: var(--fb-blue-soft);
  border-left: 4px solid var(--fb-blue);
  padding: 20px;
}
.fb-callout h3 { margin-top: 0; }
.fb-footer {
  border-top: 1px solid var(--fb-rule);
  color: var(--fb-muted);
  font-size: 12px;
  margin-top: 40px;
  padding: 20px 8px 36px;
}
.tabs { border-bottom: 1px solid var(--fb-rule) !important; }
.tab-nav button { font-weight: 600 !important; }
.tab-nav button.selected { color: var(--fb-blue) !important; }
button.primary { background: var(--fb-blue) !important; border-color: var(--fb-blue) !important; }
.gradio-dataframe, .block { border-radius: 8px !important; }
@media (max-width: 920px) {
  .fb-hero, .fb-method-grid { grid-template-columns: 1fr; }
  .fb-hero { gap: 28px; padding-top: 34px; }
  .fb-frontier { min-width: 0; }
}
@media (max-width: 620px) {
  .fb-hero h1 { font-size: 52px; }
  .fb-pair-status { grid-template-columns: 1fr; }
  .fb-rail-row { grid-template-columns: 92px 1fr 30px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


def _load_release() -> dict[str, Any]:
    if not RELEASE_PATH.is_file():
        raise FileNotFoundError(
            f"Release not found at {RELEASE_PATH}. Set FLAVOURBENCH_RELEASE to its path."
        )
    return json.loads(RELEASE_PATH.read_text(encoding="utf-8"))


RELEASE = _load_release()
MODELS = RELEASE["leaderboard"]["models"]
TASKS = RELEASE["tasks"]
MODEL_BY_NAME = {model["display_name"]: model for model in MODELS}
TASK_BY_ID = {task["task_id"]: task for task in TASKS}
OBSERVATION_INDEX = {
    (row["model_id"], row["task_id"], row["condition"]): row for row in RELEASE["observations"]
}


def _short_name(display_name: str) -> str:
    return display_name.split(": ", 1)[-1].replace(" Pro", "")


def _frontier_html() -> str:
    rows = []
    for model in MODELS[:12]:
        off = float(model["conditions"]["epicure_off"]["accuracy_percent"])
        on = float(model["conditions"]["epicure_on"]["accuracy_percent"])
        rows.append(
            "<div class='fb-rail-row'>"
            f"<div class='fb-rail-label' title='{html.escape(model['display_name'])}'>"
            f"{html.escape(_short_name(model['display_name']))}</div>"
            "<div class='fb-rail'>"
            f"<span class='fb-rail-base' style='width:{off:.3f}%'></span>"
            f"<span class='fb-rail-uplift' style='left:{off:.3f}%;width:{max(on - off, 0):.3f}%'></span>"
            "</div>"
            f"<div class='fb-rail-score'>{off:.0f}</div>"
            "</div>"
        )
    return "".join(rows)


def _hero_html() -> str:
    counts = RELEASE["counts"]
    return f"""
    <div class="fb-shell fb-hero">
      <section>
        <div class="fb-kicker">Frontier culinary reasoning benchmark · 20 endpoints</div>
        <h1>Culinary reasoning without a model judge.</h1>
        <p class="fb-dek">FlavourBench scores 20 current language-model endpoints against answer
        keys compiled by Epicure.
        <strong>Blue measures Model only. Gold measures the gain from Epicure.</strong>
        Open any pair to inspect the prompt, answers, tool trace, and hashes.</p>
        <div class="fb-stats">
          <div class="fb-stat"><strong>{counts["models"]}</strong><span>models</span></div>
          <div class="fb-stat"><strong>{counts["tasks"]}</strong><span>tasks</span></div>
          <div class="fb-stat"><strong>{counts["assigned_pairs"]}</strong><span>matched pairs</span></div>
          <div class="fb-stat"><strong>{counts["observed_response_arms"]:,}</strong><span>observed arms</span></div>
        </div>
      </section>
      <section class="fb-frontier" aria-label="FlavourBench Score and Epicure Gain">
        <div class="fb-frontier-head"><strong>FlavourBench Score plus Epicure Gain</strong><span>Top 12 · percent correct</span></div>
        {_frontier_html()}
        <div class="fb-note">One task equals 3.125 percentage points. Read adjacent rows as a
        close score group, then inspect the underlying pairs.</div>
      </section>
    </div>
    """


def _leaderboard_frame() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        off = model["conditions"]["epicure_off"]
        on = model["conditions"]["epicure_on"]
        observed = int(off["normal_completions"]) + int(on["normal_completions"])
        rows.append(
            {
                "Rank": model["rank"],
                "Model": model["display_name"],
                "Model only": f"{off['accuracy_percent']:.3g}%",
                "Model + Epicure": f"{on['accuracy_percent']:.3g}%",
                "Epicure Gain": f"+{model['uplift_percentage_points']:.3g} pp",
                "Observed arms": f"{observed}/64",
                "Backend": model["execution_backend"],
            }
        )
    return pd.DataFrame(rows)


def _model_detail(model_name: str) -> tuple[str, pd.DataFrame]:
    model = MODEL_BY_NAME[model_name]
    off = model["conditions"]["epicure_off"]
    on = model["conditions"]["epicure_on"]
    summary = f"""
    <div class="fb-pair-status">
      <div class="fb-condition good">
        <small>FlavourBench Score</small>
        <strong>{off["accuracy_percent"]:.3g}%</strong>
        <span>Wilson 95%: {off["wilson_95"][0] * 100:.1f}% to {off["wilson_95"][1] * 100:.1f}%</span>
      </div>
      <div class="fb-condition good">
        <small>Model + Epicure accuracy</small>
        <strong>{on["accuracy_percent"]:.3g}%</strong>
        <span>Epicure Gain: +{model["uplift_percentage_points"]:.3g} percentage points</span>
      </div>
    </div>
    """
    family_rows = []
    for family in RELEASE["leaderboard"]["design"]["families"]:
        family_rows.append(
            {
                "Family": family.title(),
                "Model only": f"{off['family_accuracy'][family] * 100:.1f}%",
                "Model + Epicure": f"{on['family_accuracy'][family] * 100:.1f}%",
                "Change": (
                    f"{(on['family_accuracy'][family] - off['family_accuracy'][family]) * 100:+.1f} pp"
                ),
            }
        )
    return summary, pd.DataFrame(family_rows)


def _observation(model_id: str, task_id: str, condition: str) -> dict[str, Any]:
    return OBSERVATION_INDEX.get(
        (model_id, task_id, condition),
        {
            "answer_markdown": "No observation was recorded.",
            "correct": False,
            "parseable_normal_completion": False,
            "latency_ms": None,
            "response_artifact_sha256": "unavailable",
            "source_status": "unavailable",
            "tool_trace": [],
        },
    )


def _status_card(label: str, row: dict[str, Any]) -> str:
    observed = bool(row.get("parseable_normal_completion"))
    if not observed:
        css_class = "unknown"
        status = "Unavailable"
    elif row.get("correct"):
        css_class = "good"
        status = "Correct"
    else:
        css_class = "bad"
        status = "Incorrect"
    latency = row.get("latency_ms")
    latency_text = f"{latency:,} ms" if isinstance(latency, int) else "not observed"
    artifact = html.escape(str(row.get("response_artifact_sha256", "unavailable"))[:16])
    return f"""
    <div class="fb-condition {css_class}">
      <small>{html.escape(label)}</small>
      <strong>{status}</strong>
      <span>Choice {html.escape(str(row.get("observed_choice") or "none"))} · {latency_text}</span><br>
      <code>{artifact}...</code>
    </div>
    """


def _pair_detail(
    model_name: str, task_id: str
) -> tuple[str, str, dict[str, str], str, str, list[Any], str]:
    model = MODEL_BY_NAME[model_name]
    task = TASK_BY_ID[task_id]
    off = _observation(model["model_id"], task_id, "epicure_off")
    on = _observation(model["model_id"], task_id, "epicure_on")
    status = (
        "<div class='fb-pair-status'>"
        + _status_card("Model only", off)
        + _status_card("Model + Epicure", on)
        + "</div>"
    )
    reference = json.dumps(
        {
            "call": task["reference_tool_call"],
            "result": task["reference_tool_result"],
            "result_sha256": task["reference_tool_result_sha256"],
        },
        indent=2,
        ensure_ascii=False,
    )
    provenance = (
        f"Task family: `{task['family']}`  \n"
        f"Scoring family: `{task['scoring_family']}`  \n"
        f"Prompt SHA-256: `{task['prompt_sha256']}`  \n"
        f"Release artifact: `{RELEASE['artifact_sha256']}`"
    )
    return (
        status,
        task["prompt"],
        task["choices"],
        str(off.get("answer_markdown") or "No answer recorded."),
        str(on.get("answer_markdown") or "No answer recorded."),
        on.get("tool_trace") or [],
        reference + "\n\n" + provenance,
    )


def _task_label(task: dict[str, Any]) -> str:
    return f"{task['task_id']} · {task['family']}"


MODEL_NAMES = list(MODEL_BY_NAME)
TASK_LABEL_TO_ID = {_task_label(task): task["task_id"] for task in TASKS}


def _pair_from_label(
    model_name: str, task_label: str
) -> tuple[str, str, dict[str, str], str, str, list[Any], str]:
    return _pair_detail(model_name, TASK_LABEL_TO_ID[task_label])


theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EAF3FA",
        c100="#D5E8F5",
        c200="#A9D0E9",
        c300="#75B3DA",
        c400="#4292C6",
        c500=BRAND["blue"],
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
    body_background_fill=BRAND["paper"],
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_label_text_weight="600",
    button_primary_background_fill=BRAND["blue"],
    button_primary_background_fill_hover="#12588F",
)


with gr.Blocks(title="FlavourBench · Frontier culinary reasoning benchmark") as demo:
    gr.HTML(_hero_html())

    with gr.Tabs():
        with gr.Tab("Leaderboard"):
            gr.HTML(
                """
                <div class="fb-section-title">
                  <div class="fb-kicker">Automated exact-choice track</div>
                  <h2>The complete public benchmark</h2>
                  <p>Rank follows FlavourBench Score. Model + Epicure and Epicure Gain show the matched intervention.</p>
                </div>
                """
            )
            gr.Dataframe(
                value=_leaderboard_frame(),
                interactive=False,
                wrap=True,
                show_search="filter",
                show_row_numbers=False,
                column_widths=[55, 260, 95, 95, 105, 110, 110],
            )
            gr.Markdown(
                "**Reading the table.** Rank follows Model only accuracy over this 32-task release; "
                "it is not a claim about general model quality. Equal scores follow the release's "
                "frozen tie-break rules. Use Pair Lens before interpreting small differences."
            )

        with gr.Tab("Model fingerprint"):
            gr.HTML(
                """
                <div class="fb-section-title">
                  <div class="fb-kicker">Family profile</div>
                  <h2>Where does Epicure change the model?</h2>
                  <p>Compare substitution, composition, cookability, and evidence tasks.</p>
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

        with gr.Tab("Pair Lens"):
            gr.HTML(
                """
                <div class="fb-section-title">
                  <div class="fb-kicker">Model-task evidence</div>
                  <h2>Open the score</h2>
                  <p>Inspect both answers, the exact Epicure reference, the observed trace, and hashes.</p>
                </div>
                """
            )
            with gr.Row():
                pair_model = gr.Dropdown(
                    choices=MODEL_NAMES,
                    value=MODEL_NAMES[0],
                    label="Model",
                    filterable=True,
                    scale=1,
                )
                pair_task = gr.Dropdown(
                    choices=list(TASK_LABEL_TO_ID),
                    value=next(iter(TASK_LABEL_TO_ID)),
                    label="Task",
                    filterable=True,
                    scale=2,
                )
                inspect_button = gr.Button("Inspect pair", variant="primary", scale=0)

            initial_pair = _pair_from_label(MODEL_NAMES[0], next(iter(TASK_LABEL_TO_ID)))
            pair_status = gr.HTML(initial_pair[0])
            prompt = gr.Textbox(
                value=initial_pair[1],
                label="Exact prompt",
                lines=12,
                interactive=False,
            )
            choices = gr.JSON(value=initial_pair[2], label="Choices")
            with gr.Row():
                off_answer = gr.Markdown(value=initial_pair[3], label="Model only answer")
                on_answer = gr.Markdown(value=initial_pair[4], label="Model + Epicure answer")
            tool_trace = gr.JSON(value=initial_pair[5], label="Observed Epicure trace")
            reference = gr.Textbox(
                value=initial_pair[6],
                label="Reference operation and provenance",
                lines=16,
                interactive=False,
            )
            inspect_button.click(
                _pair_from_label,
                inputs=[pair_model, pair_task],
                outputs=[
                    pair_status,
                    prompt,
                    choices,
                    off_answer,
                    on_answer,
                    tool_trace,
                    reference,
                ],
            )

        with gr.Tab("Method and download"):
            gr.HTML(
                f"""
                <div class="fb-section-title">
                  <div class="fb-kicker">Reproduce, cite, extend</div>
                  <h2>One release, five public tables</h2>
                  <p>The Space reads a content-addressed JSON release and makes no provider calls.</p>
                </div>
                <div class="fb-method-grid">
                  <div>
                    <h3>Scoring contract</h3>
                    <p><strong>FlavourBench Score</strong> is Model only exact-choice accuracy over
                    all 32 tasks. <strong>Model + Epicure</strong> uses the same endpoint-task cells
                    with one named Epicure operation. <strong>Epicure Gain</strong> is the matched
                    percentage-point change and does not affect rank.</p>
                    <p>Tasks cover substitution, composition, cookability, and evidence. Every expected
                    answer is derived from a fixed read-only Epicure operation.</p>
                    <h3>Public records</h3>
                    <p>Downloadable configs cover models, tasks, observations, paired outcomes, and
                    leaderboard rows. Response and result hashes connect every table.</p>
                  </div>
                  <aside class="fb-callout">
                    <h3>Exact release</h3>
                    <p><span class="fb-hash">{RELEASE["artifact_sha256"]}</span></p>
                    <p>{RELEASE["counts"]["models"]} models · {RELEASE["counts"]["tasks"]} tasks ·
                    {RELEASE["counts"]["assigned_arms"]:,} assigned arms</p>
                    <p>Track: {html.escape(RELEASE["track"])}</p>
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
python -I paper/reproduce_epicure_native.py \\
  --release paper/generated/epicure-native/epicure-native-release.json
```

**Project:** [paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf)
· [dataset](https://huggingface.co/datasets/josefchen/flavourbench)
· [source](https://github.com/josefchen/flavourbench)

**Rights note:** tasks and authored metadata are CC BY 4.0. Model responses and third-party
materials retain the boundaries in
[`LICENSES.md`](https://github.com/josefchen/flavourbench/blob/main/LICENSES.md). This explorer is a
public research preview.
                """
            )

    gr.HTML(
        """
        <div class="fb-shell fb-footer">
          FlavourBench · Executable culinary evaluation · Public automated benchmark
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
