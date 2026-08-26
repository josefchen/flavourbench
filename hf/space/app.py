from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import shlex
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

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

RUST = "#A83D34"
INK = "#161817"
PAPER_URL = "https://arxiv.org/abs/2608.20574"
DATASET_URL = "https://huggingface.co/datasets/josefchen/flavourbench"
SOURCE_URL = "https://github.com/josefchen/flavourbench"
DATASET_RESULTS_URL = (
    "https://huggingface.co/datasets/josefchen/flavourbench/resolve/main/"
    "data-complete-core/leaderboard.jsonl?download=true"
)
SUBMISSION_GUIDE_URL = (
    "https://github.com/josefchen/flavourbench/blob/main/docs/submitting-results.md"
)
REWARD_TRANSFER_PROTOCOL_URL = (
    "https://github.com/josefchen/flavourbench/blob/main/docs/reward-transfer-study.md"
)
REWARD_TRANSFER_DATA_URL = (
    "https://huggingface.co/datasets/josefchen/flavourbench/tree/main/data-analysis"
)
REWARD_TRANSFER_VERIFY_URL = (
    "https://github.com/josefchen/flavourbench/blob/main/"
    "experiments/reward_transfer/verify_release.py"
)
REWARD_TRANSFER_FIGURE_URL = (
    "https://huggingface.co/datasets/josefchen/flavourbench/resolve/main/"
    "assets/complete-core-reward-transfer.png"
)
SUBMIT_RESULT_URL = (
    "https://github.com/josefchen/flavourbench/issues/new?template=flavourbench-result.yml"
)

LAB_LOGO_FILES = {
    "xAI": "xai.svg",
    "Google": "google.svg",
    "OpenAI": "openai.svg",
    "Meta": "meta.svg",
    "Anthropic": "anthropic.svg",
    "Qwen": "qwen.svg",
    "Kimi": "kimi.svg",
    "DeepSeek": "deepseek.svg",
    "Tencent": "tencent.svg",
    "MiniMax": "minimax.svg",
}


def _asset_url(path: Path) -> str:
    """Serve one verified release asset through Gradio's allowlisted file route."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Space asset not found: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to((HERE / "assets").resolve()):
        raise ValueError(f"Space asset is outside the public asset directory: {path}")
    return f"/gradio_api/file={quote(str(resolved), safe='/')}"


def _font_data_url(path: Path) -> str:
    """Inline the small launch-font subsets before Gradio lays out the page."""

    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".woff2":
        raise FileNotFoundError(f"Space font not found: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to((HERE / "assets/fonts").resolve()):
        raise ValueError(f"Space font is outside the public font directory: {path}")
    return "data:font/woff2;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


LAB_LOGO_URLS = {
    lab: _asset_url(HERE / "assets" / "providers" / filename)
    for lab, filename in LAB_LOGO_FILES.items()
}
ARCHITECTURE_URL = _asset_url(HERE / "assets" / "executable-judge.svg")
FONT_URLS = {
    "lato_regular": _font_data_url(HERE / "assets/fonts/Lato-Regular.woff2"),
    "lato_semibold": _font_data_url(HERE / "assets/fonts/Lato-Semibold.woff2"),
    "lato_bold": _font_data_url(HERE / "assets/fonts/Lato-Bold.woff2"),
    "lato_black": _font_data_url(HERE / "assets/fonts/Lato-Black.woff2"),
    "mono_regular": _font_data_url(HERE / "assets/fonts/DejaVuSansMono-Regular.woff2"),
    "mono_bold": _font_data_url(HERE / "assets/fonts/DejaVuSansMono-Bold.woff2"),
}
FONT_CSS = f"""
@font-face {{
  font-family: "Lato";
  font-style: normal;
  font-weight: 400;
  font-display: block;
  src: url("{FONT_URLS["lato_regular"]}") format("woff2");
}}
@font-face {{
  font-family: "Lato";
  font-style: normal;
  font-weight: 600;
  font-display: block;
  src: url("{FONT_URLS["lato_semibold"]}") format("woff2");
}}
@font-face {{
  font-family: "Lato";
  font-style: normal;
  font-weight: 700;
  font-display: block;
  src: url("{FONT_URLS["lato_bold"]}") format("woff2");
}}
@font-face {{
  font-family: "Lato";
  font-style: normal;
  font-weight: 900;
  font-display: block;
  src: url("{FONT_URLS["lato_black"]}") format("woff2");
}}
@font-face {{
  font-family: "DejaVu Sans Mono";
  font-style: normal;
  font-weight: 400;
  font-display: block;
  src: url("{FONT_URLS["mono_regular"]}") format("woff2");
}}
@font-face {{
  font-family: "DejaVu Sans Mono";
  font-style: normal;
  font-weight: 700;
  font-display: block;
  src: url("{FONT_URLS["mono_bold"]}") format("woff2");
}}
"""
CSS = """
:root {
  --fb-accent: #A83D34;
  --fb-accent-soft: #F1DFDC;
  --fb-ink: #161817;
  --fb-muted: #68706C;
  --fb-line: #56605B;
  --fb-paper: #F6F7F5;
  --fb-paper-raised: #FBFCFA;
  --fb-rule: #DDE1DE;
  --fb-code: #ECEFEC;
  --fb-z-nav: 20;
}
.dark {
  --fb-accent: #EF796D;
  --fb-accent-soft: #422723;
  --fb-ink: #F0EFE9;
  --fb-muted: #A9ACA3;
  --fb-line: #B1B7B3;
  --fb-paper: #171815;
  --fb-paper-raised: #20211E;
  --fb-rule: #41433D;
  --fb-code: #2A2B27;
}
html { scroll-behavior: smooth; }
body, .gradio-container {
  background: var(--fb-paper) !important;
  color: var(--fb-ink) !important;
  font-family: "Lato", "Avenir Next", system-ui, sans-serif !important;
}
#huggingface-space-header {
  background: var(--fb-paper-raised) !important;
  background-image: none !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  display: none !important;
}
#huggingface-space-header a {
  font-family: "Lato", "Avenir Next", system-ui, sans-serif !important;
}
.gradio-container {
  max-width: none !important;
  overflow: visible !important;
  padding: 0 !important;
}
.gradio-container > main { padding: 0 !important; }
.fb-shell {
  box-sizing: border-box;
  margin: 0 auto;
  max-width: 1440px;
  padding-left: clamp(22px, 4vw, 62px);
  padding-right: clamp(22px, 4vw, 62px);
}
.fb-masthead {
  align-items: center;
  display: flex;
  justify-content: space-between;
  padding-bottom: 7px;
  padding-top: 18px;
}
.fb-masthead-brand {
  align-items: baseline;
  display: flex;
  gap: 14px;
  min-width: 0;
}
.fb-masthead-brand strong {
  color: var(--fb-ink);
  font-size: 15px;
  font-weight: 900;
  letter-spacing: -.03em;
}
.fb-masthead-brand span {
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 9px;
  letter-spacing: .05em;
  text-transform: uppercase;
}
.fb-masthead nav { display: flex; gap: 22px; }
.fb-masthead a {
  color: var(--fb-muted) !important;
  font-size: 12px;
  font-weight: 700;
  text-decoration: none;
}
.fb-masthead a:hover { color: var(--fb-accent) !important; }
.fb-hero {
  display: grid;
  gap: clamp(40px, 4vw, 64px);
  grid-template-columns: minmax(560px, 1.05fr) minmax(500px, .95fr);
  padding-bottom: clamp(28px, 3vw, 36px);
  padding-top: clamp(24px, 3vw, 38px);
}
.fb-hero h1 {
  color: var(--fb-ink);
  font-size: clamp(56px, 5vw, 72px);
  font-weight: 900;
  letter-spacing: -.065em;
  line-height: .91;
  margin: 18px 0 23px;
  max-width: 740px;
}
.fb-dek {
  color: var(--fb-muted);
  font-size: clamp(17px, 1.6vw, 21px);
  line-height: 1.46;
  margin: 0;
  max-width: 570px;
}
.fb-stats {
  display: grid;
  gap: 20px;
  grid-template-columns: repeat(4, minmax(88px, 1fr));
  margin-top: 34px;
}
.fb-stat strong {
  color: var(--fb-ink);
  display: block;
  font-size: 25px;
  font-weight: 900;
  letter-spacing: -.04em;
  line-height: 1;
}
.fb-stat span {
  color: var(--fb-muted);
  display: block;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  letter-spacing: .06em;
  margin-top: 7px;
  text-transform: uppercase;
}
.fb-frontier {
  align-self: end;
  min-width: 0;
}
.fb-frontier-head {
  align-items: baseline;
  display: flex;
  gap: 16px;
  justify-content: space-between;
  margin-bottom: 15px;
}
.fb-frontier-head strong { font-size: 15px; font-weight: 620; }
.fb-mobile-label { display: none; }
.fb-frontier-head span {
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  text-align: right;
}
.fb-frontier-note {
  align-items: baseline;
  display: flex;
  gap: 14px;
  justify-content: space-between;
  margin-top: 10px;
}
.fb-frontier-note span,
.fb-frontier-note strong {
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 9px;
  line-height: 1.45;
}
.fb-frontier-note span { color: var(--fb-muted); }
.fb-frontier-note strong { color: var(--fb-ink); font-weight: 600; text-align: right; }
.fb-forest-row {
  align-items: center;
  display: grid;
  gap: 12px;
  grid-template-columns: 22px minmax(170px, 210px) 1fr 47px;
  min-height: 36px;
}
.fb-place {
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
}
.fb-model {
  align-items: center;
  color: var(--fb-ink);
  display: flex;
  font-size: 12px;
  font-weight: 700;
  gap: 8px;
  min-width: 0;
}
.fb-model-mark {
  background: transparent;
  box-sizing: border-box;
  flex: 0 0 20px;
  height: 20px;
  object-fit: contain;
  width: 20px;
}
.dark .fb-model-mark {
  background: #F6F7F5;
  border-radius: 50%;
  padding: 2px;
}
.fb-model-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fb-axis {
  height: 10px;
  position: relative;
}
.fb-axis::before {
  background: var(--fb-rule);
  content: "";
  height: 2px;
  left: 0;
  position: absolute;
  right: 0;
  top: 4px;
}
.fb-bar {
  background: var(--fb-line);
  height: 4px;
  left: 0;
  position: absolute;
  top: 3px;
}
.fb-point {
  background: var(--fb-ink);
  border-radius: 50%;
  height: 10px;
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  width: 10px;
}
.fb-forest-row:first-of-type .fb-bar,
.fb-forest-row:first-of-type .fb-point { background: var(--fb-accent); }
.fb-forest-row:first-of-type .fb-model,
.fb-forest-row:first-of-type .fb-number { color: var(--fb-accent); }
.fb-number {
  color: var(--fb-ink);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 12px;
  font-weight: 600;
  text-align: right;
}
.fb-chart-foot {
  color: var(--fb-muted);
  display: flex;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 9px;
  justify-content: space-between;
  margin: 9px 59px 0 266px;
}
.tab-wrapper {
  background: color-mix(in srgb, var(--fb-paper) 94%, transparent) !important;
  backdrop-filter: blur(12px);
  border: 0 !important;
  padding-bottom: 0 !important;
  position: sticky !important;
  top: 0;
  z-index: var(--fb-z-nav);
}
.tab-container[role="tablist"] {
  margin: 0 auto !important;
  max-width: 1316px !important;
  padding-left: clamp(22px, 4vw, 62px) !important;
  padding-right: clamp(22px, 4vw, 62px) !important;
}
.tab-container[role="tablist"]::after { display: none !important; }
.tab-container[role="tablist"] button {
  border: 0 !important;
  color: var(--fb-muted) !important;
  font-size: 13px !important;
  padding: 15px 0 13px !important;
  margin-right: 30px !important;
}
.tab-container[role="tablist"] button.selected {
  color: var(--fb-ink) !important;
}
.tab-container[role="tablist"] button.selected::after {
  background: var(--fb-accent) !important;
  height: 2px !important;
}
.overflow-menu { display: none !important; }
.tab-container.visually-hidden { display: none !important; }
.tabitem {
  box-sizing: border-box;
  margin: 0 auto !important;
  max-width: 1440px !important;
  padding: 0 clamp(22px, 4vw, 62px) !important;
}
.fb-section { margin: 36px 0 18px; }
.fb-section h2 {
  color: var(--fb-ink);
  font-size: clamp(30px, 3vw, 45px);
  font-weight: 900;
  letter-spacing: -.045em;
  line-height: 1.02;
  margin: 0 0 9px;
}
.fb-section p {
  color: var(--fb-muted);
  font-size: 15px;
  line-height: 1.5;
  margin: 0;
  max-width: 72ch;
}
.fb-lab-path {
  align-items: baseline;
  display: grid;
  gap: 14px clamp(28px, 5vw, 72px);
  grid-template-columns: minmax(380px, 1.15fr) minmax(280px, .85fr);
  margin: 10px 0 30px;
}
.fb-lab-path strong { color: var(--fb-ink); font-size: 18px; font-weight: 900; }
.fb-lab-path i { color: var(--fb-accent); font-style: normal; padding: 0 7px; }
.fb-lab-path span { color: var(--fb-muted); font-size: 13px; line-height: 1.5; }
.fb-choice-grid {
  display: grid;
  column-gap: 28px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 8px 0 20px;
}
.fb-data-heading {
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .06em;
  margin: 22px 0 8px;
  text-transform: uppercase;
}
.fb-choice {
  align-items: center;
  border-bottom: 1px solid var(--fb-rule);
  display: flex;
  gap: 13px;
  min-width: 0;
  padding: 12px 20px 12px 0;
}
.fb-choice:nth-child(even) {
  padding-left: 0;
}
.fb-choice:nth-last-child(-n+2) { border-bottom: 0; }
.fb-choice-label {
  align-items: center;
  background: var(--fb-ink);
  border-radius: 50%;
  color: var(--fb-paper);
  display: inline-flex;
  flex: 0 0 30px;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 12px;
  font-weight: 700;
  height: 30px;
  justify-content: center;
}
.fb-choice-name {
  color: var(--fb-ink);
  font-size: 14px;
  font-weight: 700;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fb-metric-grid {
  display: grid;
  gap: 20px clamp(22px, 4vw, 52px);
  grid-template-columns: repeat(3, 1fr);
  margin: 10px 0 22px;
}
.fb-metric { min-width: 0; padding: 12px 0 14px; }
.fb-metric:nth-child(3n+2), .fb-metric:nth-child(3n+3) {
  padding-left: 0;
}
.fb-metric small {
  color: var(--fb-muted);
  display: block;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  letter-spacing: .05em;
  text-transform: uppercase;
}
.fb-metric strong {
  color: var(--fb-ink);
  display: block;
  font-size: 24px;
  font-weight: 900;
  margin-top: 5px;
  white-space: nowrap;
}
.fb-evidence {
  color: var(--fb-ink);
  line-height: 1.52;
  padding: 12px 0;
}
.fb-evidence strong:first-child { color: var(--fb-accent); }
.fb-evidence code, .fb-hash {
  background: var(--fb-code);
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 11px;
  overflow-wrap: anywhere;
}
.fb-method {
  display: grid;
  gap: clamp(30px, 6vw, 86px);
  grid-template-columns: 1.25fr .75fr;
}
.fb-method h3 { font-size: 19px; margin: 23px 0 6px; }
.fb-method p { color: var(--fb-muted); line-height: 1.55; }
.fb-method-visual {
  align-items: start;
  display: grid;
  gap: clamp(34px, 5vw, 72px);
  grid-template-columns: minmax(340px, .82fr) minmax(440px, 1.18fr);
}
.fb-architecture {
  display: block;
  height: auto;
  width: 100%;
}
.fb-command-note {
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  line-height: 1.5;
}
.fb-table-wrap {
  overflow-x: auto;
  width: 100%;
}
.fb-leader-tools {
  align-items: end;
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(220px, 1fr) auto auto;
  padding: 4px 0 18px;
}
.fb-search-label {
  color: var(--fb-muted);
  display: block;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  letter-spacing: .05em;
  text-transform: uppercase;
}
.fb-search-label input {
  background: transparent !important;
  border: 0 !important;
  border-bottom: 1px solid var(--fb-rule) !important;
  color: var(--fb-ink) !important;
  display: block;
  font-family: "Lato", "Avenir Next", system-ui, sans-serif;
  font-size: 15px;
  margin-top: 5px;
  min-height: 34px;
  padding: 2px 0;
  width: 100%;
}
.fb-filter-set { display: flex; }
.fb-filter-button {
  background: transparent;
  border: 1px solid var(--fb-rule);
  border-radius: 0;
  color: var(--fb-muted);
  cursor: pointer;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  min-height: 36px;
  padding: 0 12px;
  text-transform: uppercase;
}
.fb-filter-button + .fb-filter-button { border-left: 0; }
.fb-filter-button[aria-pressed="true"] {
  background: var(--fb-ink);
  border-color: var(--fb-ink);
  color: var(--fb-paper);
}
.fb-leader-meta {
  align-items: center;
  display: flex;
  gap: 16px;
  justify-content: flex-end;
  min-height: 36px;
}
.fb-metric-rail {
  align-items: center;
  display: grid;
  gap: 10px 18px;
  grid-template-columns: auto auto 1fr;
  padding: 0 0 16px;
}
.fb-metric-rail-label,
.fb-metric-note {
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 9px;
}
.fb-metric-rail-label {
  letter-spacing: .06em;
  text-transform: uppercase;
}
.fb-metric-switch { display: flex; gap: 18px; }
.fb-metric-switch button {
  background: transparent;
  border: 0;
  border-bottom: 2px solid transparent;
  color: var(--fb-muted);
  cursor: pointer;
  font-family: "Lato", "Avenir Next", system-ui, sans-serif;
  font-size: 12px;
  font-weight: 700;
  padding: 7px 0 5px;
  white-space: nowrap;
}
.fb-metric-switch button[aria-pressed="true"] {
  border-bottom-color: var(--fb-accent);
  color: var(--fb-ink);
}
.fb-metric-switch button:hover { color: var(--fb-accent); }
.fb-metric-note { justify-self: end; text-align: right; }
.fb-result-count {
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  white-space: nowrap;
}
.fb-text-link,
.fb-action-link {
  color: var(--fb-ink) !important;
  font-size: 12px;
  font-weight: 700;
  text-decoration: underline;
  text-decoration-color: var(--fb-rule);
  text-underline-offset: 4px;
}
.fb-text-link:hover,
.fb-action-link:hover { color: var(--fb-accent) !important; text-decoration-color: currentColor; }
.fb-empty-row td {
  color: var(--fb-muted);
  padding: 28px 0 !important;
}
.fb-release-line {
  align-items: center;
  display: grid;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  gap: 18px;
  grid-template-columns: repeat(3, minmax(0, auto)) 1fr;
  padding: 12px 0 0;
}
.fb-release-line span { color: var(--fb-muted); }
.fb-release-line strong { color: var(--fb-ink); font-weight: 700; }
.fb-release-line a { justify-self: end; }
.fb-insight-layout {
  display: grid;
  gap: clamp(34px, 5vw, 72px);
  grid-template-columns: minmax(640px, 1.4fr) minmax(260px, .6fr);
}
.fb-panel-label {
  color: var(--fb-muted);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .06em;
  margin-bottom: 9px;
  text-transform: uppercase;
}
.fb-ci-row {
  align-items: center;
  display: grid;
  gap: 12px;
  grid-template-columns: 24px minmax(180px, 240px) 1fr 48px;
  min-height: 42px;
}
.fb-ci-track { height: 16px; position: relative; }
.fb-ci-track::before {
  background: var(--fb-rule);
  content: "";
  height: 1px;
  left: 0;
  position: absolute;
  right: 0;
  top: 8px;
}
.fb-ci-band {
  background: var(--fb-line);
  height: 3px;
  position: absolute;
  top: 7px;
}
.fb-ci-band::before,
.fb-ci-band::after {
  background: var(--fb-line);
  content: "";
  height: 9px;
  position: absolute;
  top: -3px;
  width: 1px;
}
.fb-ci-band::before { left: 0; }
.fb-ci-band::after { right: 0; }
.fb-ci-point {
  background: var(--fb-ink);
  border-radius: 50%;
  height: 9px;
  position: absolute;
  top: 4px;
  transform: translateX(-50%);
  width: 9px;
}
.fb-ci-row:first-child .fb-ci-band,
.fb-ci-row:first-child .fb-ci-band::before,
.fb-ci-row:first-child .fb-ci-band::after,
.fb-ci-row:first-child .fb-ci-point { background: var(--fb-accent); }
.fb-ci-axis-labels {
  color: var(--fb-muted);
  display: flex;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 9px;
  justify-content: space-between;
  margin: 9px 60px 0 280px;
}
.fb-resolution-row {
  padding: 0 0 25px;
}
.fb-resolution-row strong {
  color: var(--fb-accent);
  display: block;
  font-size: clamp(30px, 3vw, 42px);
  font-weight: 900;
  letter-spacing: -.05em;
  line-height: 1;
}
.fb-resolution-row span {
  color: var(--fb-muted);
  display: block;
  font-size: 13px;
  line-height: 1.4;
  margin-top: 6px;
}
.fb-resolution-note { color: var(--fb-muted); font-size: 13px; line-height: 1.52; margin: 18px 0 0; }
.fb-transfer-layout {
  align-items: start;
  display: grid;
  gap: clamp(34px, 6vw, 88px);
  grid-template-columns: minmax(260px, .58fr) minmax(600px, 1.42fr);
  margin-top: 10px;
}
.fb-transfer-layout > * { min-width: 0; }
.fb-transfer-primary strong {
  color: var(--fb-accent);
  display: block;
  font-size: clamp(58px, 7vw, 92px);
  font-weight: 900;
  letter-spacing: -.07em;
  line-height: .9;
}
.fb-transfer-primary > span {
  color: var(--fb-ink);
  display: block;
  font-size: 17px;
  font-weight: 700;
  line-height: 1.3;
  margin-top: 12px;
  max-width: 24ch;
}
.fb-transfer-primary p {
  color: var(--fb-muted);
  font-size: 13px;
  line-height: 1.52;
  margin: 12px 0 0;
  max-width: 35ch;
}
.fb-transfer-facts {
  border-top: 1px solid var(--fb-rule);
  display: grid;
  gap: 20px 28px;
  grid-template-columns: repeat(4, 1fr);
  margin-top: 24px;
  padding-top: 18px;
}
.fb-transfer-fact strong {
  color: var(--fb-ink);
  display: block;
  font-size: 19px;
  font-weight: 900;
}
.fb-transfer-fact span {
  color: var(--fb-muted);
  display: block;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 9px;
  line-height: 1.45;
  margin-top: 5px;
  text-transform: uppercase;
}
.fb-table.fb-transfer-table { font-size: 11px; min-width: 640px; }
.fb-transfer-table th,
.fb-transfer-table td { padding-right: 8px; }
.fb-transfer-table td:last-child { color: var(--fb-accent); font-weight: 700; }
.fb-transfer-figure { margin: 38px 0 0; }
.fb-transfer-figure img {
  display: block;
  height: auto;
  max-width: 1120px;
  width: 100%;
}
.fb-transfer-figure figcaption {
  color: var(--fb-muted);
  font-size: 12px;
  line-height: 1.5;
  margin-top: 10px;
  max-width: 84ch;
}
.fb-transfer-boundary {
  border-top: 1px solid var(--fb-rule);
  display: grid;
  gap: clamp(30px, 6vw, 90px);
  grid-template-columns: 1fr 1fr;
  margin-top: 34px;
  padding-top: 22px;
}
.fb-transfer-boundary h3 { color: var(--fb-ink); font-size: 19px; margin: 0 0 6px; }
.fb-transfer-boundary p { color: var(--fb-muted); font-size: 13px; line-height: 1.55; margin: 0; }
.fb-transfer-links { display: flex; flex-wrap: wrap; gap: 12px 24px; margin-top: 22px; }
.fb-family-insight { margin-top: 42px; }
.fb-family-insight h3 { color: var(--fb-ink); font-size: 22px; margin: 0 0 4px; }
.fb-family-insight > p { color: var(--fb-muted); font-size: 13px; margin: 0 0 14px; }
.fb-family-table { min-width: 760px; }
.fb-family-short { display: none; }
.fb-family-model {
  align-items: center;
  display: flex;
  font-family: "Lato", "Avenir Next", system-ui, sans-serif;
  font-weight: 700;
  gap: 9px;
}
.fb-family-value {
  align-items: center;
  display: grid;
  gap: 9px;
  grid-template-columns: 44px 1fr;
}
.fb-family-mini-axis { height: 10px; position: relative; }
.fb-family-mini-axis::before {
  background: var(--fb-rule);
  content: "";
  height: 1px;
  left: 0;
  position: absolute;
  right: 0;
  top: 5px;
}
.fb-family-dot {
  background: var(--fb-ink);
  border-radius: 50%;
  height: 7px;
  position: absolute;
  top: 2px;
  transform: translateX(-50%);
  width: 7px;
}
.fb-family-best .fb-family-dot { background: var(--fb-accent); height: 9px; top: 1px; width: 9px; }
.fb-family-best > span { color: var(--fb-accent); font-weight: 700; }
.fb-publish-path {
  align-items: center;
  display: grid;
  gap: 28px;
  grid-template-columns: 1fr auto;
  margin-top: 24px;
  padding: 10px 0;
}
.fb-publish-path h3 { font-size: 19px; margin: 0 0 5px; }
.fb-publish-path p { color: var(--fb-muted); font-size: 13px; line-height: 1.5; margin: 0; max-width: 72ch; }
.fb-action-link {
  border: 1px solid var(--fb-ink);
  display: inline-flex;
  padding: 11px 14px;
  text-decoration: none;
  white-space: nowrap;
}
.fb-action-link:hover { border-color: var(--fb-accent); }
.fb-table {
  border: 0 !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  color: var(--fb-ink);
  font-family: "DejaVu Sans Mono", ui-monospace, monospace;
  font-size: 12px;
  min-width: 860px;
  width: 100%;
}
.fb-table tr { border: 0 !important; }
.fb-table--family { min-width: 100%; }
.fb-table--score { min-width: 760px; }
.fb-table--score td:nth-child(2) { white-space: nowrap; }
.fb-table caption {
  height: 1px;
  overflow: hidden;
  position: absolute;
  width: 1px;
}
.fb-table th {
  border-left: 0 !important;
  border-right: 0 !important;
  border-top: 0 !important;
  border-bottom: 1px solid var(--fb-rule) !important;
  color: var(--fb-muted);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .05em;
  padding: 10px 12px 11px 0;
  text-align: left;
  text-transform: uppercase;
}
.fb-table td {
  border-left: 0 !important;
  border-right: 0 !important;
  border-top: 0 !important;
  border-bottom: 1px solid var(--fb-rule);
  padding: 11px 12px 11px 0;
  vertical-align: top;
}
.fb-table .fb-score-cell { font-weight: 600; }
.fb-table-model {
  align-items: center;
  display: flex;
  font-family: "Lato", "Avenir Next", system-ui, sans-serif;
  font-weight: 700;
  gap: 9px;
  min-width: 0;
}
.fb-table-model .fb-model-mark { flex-basis: 18px; height: 18px; width: 18px; }
.fb-table-model span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fb-score-head-short { display: none; }
.fb-table--leaderboard.fb-family-view .fb-overall-only { display: none; }
.fb-table tbody tr:first-child .fb-rank-cell,
.fb-table tbody tr:first-child .fb-score-cell { color: var(--fb-accent); }
.fb-table tbody tr:hover { background: color-mix(in srgb, var(--fb-ink) 3%, transparent); }
.fb-table .fb-selected-row td { color: var(--fb-accent); font-weight: 600; }
.fb-footer {
  color: var(--fb-muted);
  display: flex;
  font-size: 12px;
  justify-content: space-between;
  margin-top: 54px;
  padding-bottom: 30px;
  padding-top: 18px;
}
.fb-footer-main,
.fb-footer-links {
  align-items: center;
  display: flex;
  gap: 20px;
}
.fb-footer-main strong { color: var(--fb-ink); }
.fb-footer a { color: var(--fb-ink); text-decoration: none; }
.fb-footer a:hover { color: var(--fb-accent); text-decoration: underline; }
.gradio-container footer a, .gradio-container footer button { color: var(--fb-muted) !important; }
.gradio-container footer .built-with,
.gradio-container footer .divider,
.gradio-container footer button.settings { display: none !important; }
.gradio-container footer button.show-api img { display: none !important; }
.block, .form, .gradio-dataframe, .gradio-json, .gradio-textbox {
  background: transparent !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
.block:not(.gradio-dataframe):not(.gradio-json):not(.gradio-textbox) {
  border-color: transparent !important;
}
button.primary {
  background: var(--fb-ink) !important;
  border: 1px solid var(--fb-ink) !important;
  border-radius: 0 !important;
  color: var(--fb-paper) !important;
  min-height: 44px !important;
  white-space: nowrap !important;
}
button.primary:hover { background: var(--fb-accent) !important; border-color: var(--fb-accent) !important; }
button.primary:active,
.fb-filter-button:active,
.fb-metric-switch button:active,
.fb-action-link:active { transform: translateY(1px); }
.fb-download { max-width: 320px !important; }
.fb-download a,
.fb-download button {
  background: transparent !important;
  border: 1px solid var(--fb-ink) !important;
  border-radius: 0 !important;
  color: var(--fb-ink) !important;
}
input, textarea, select {
  background: var(--fb-paper-raised) !important;
  border-color: var(--fb-rule) !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
input:focus, textarea:focus, select:focus, button:focus-visible, a:focus-visible {
  outline: 2px solid var(--fb-accent) !important;
  outline-offset: 2px !important;
}
table { border-collapse: collapse !important; }
th { background: var(--fb-paper) !important; }
pre, code { border-radius: 0 !important; }
.prose pre,
.prose pre code,
.prose pre span {
  background: var(--fb-code) !important;
  color: var(--fb-ink) !important;
}
.gradio-dataframe table {
  border-left: 0 !important;
  border-right: 0 !important;
  font-family: "DejaVu Sans Mono", ui-monospace, monospace !important;
}
.gradio-dataframe th,
.gradio-dataframe td {
  border-left: 0 !important;
  border-right: 0 !important;
}
@media (max-width: 1180px) {
  .fb-hero { grid-template-columns: 1fr; }
  .fb-frontier { max-width: 820px; }
  .fb-method { grid-template-columns: 1fr; }
  .fb-method-visual { grid-template-columns: minmax(300px, .85fr) minmax(390px, 1.15fr); }
  .fb-insight-layout { grid-template-columns: 1fr; }
  .fb-transfer-layout { grid-template-columns: 1fr; }
  .fb-transfer-primary p { max-width: 62ch; }
  .fb-resolution { display: grid; grid-template-columns: repeat(3, 1fr); }
  .fb-resolution-row { padding-right: 18px; }
  .fb-resolution-row + .fb-resolution-row { padding-left: 18px; }
  .fb-resolution-note { grid-column: 1 / -1; }
}
@media (max-width: 720px) {
  .fb-hero { gap: 38px; padding-bottom: 32px; }
  .fb-hero h1 { font-size: clamp(52px, 15vw, 72px); }
  .fb-stats { gap: 16px 10px; grid-template-columns: repeat(2, 1fr); }
  .fb-forest-row { gap: 8px; grid-template-columns: 19px minmax(122px, 164px) 1fr 42px; }
  .fb-model { font-size: 11px; }
  .fb-model-mark { flex-basis: 18px; height: 18px; width: 18px; }
  .fb-chart-foot { margin-left: 163px; margin-right: 50px; }
  .fb-chart-foot span:nth-child(2) { display: none; }
  .fb-masthead-brand span { display: none; }
  .fb-masthead nav { gap: 16px; }
  .fb-masthead nav a:nth-child(2) { display: none; }
  .fb-desktop-label { display: none; }
  .fb-mobile-label { display: inline; }
  .fb-frontier .fb-forest-row:nth-child(n+5) { display: none; }
  .fb-frontier-note { justify-content: flex-end; }
  .fb-frontier-note span { display: none; }
  .fb-lab-path { grid-template-columns: 1fr; }
  .fb-choice-grid { grid-template-columns: 1fr; }
  .fb-choice,
  .fb-choice:nth-child(even) { border-left: 0; padding-left: 0; }
  .fb-choice:nth-last-child(-n+2) { border-bottom: 1px solid var(--fb-rule); }
  .fb-choice:last-child { border-bottom: 0; }
  .fb-table--score { min-width: 100%; }
  .fb-table--score th:nth-child(2),
  .fb-table--score td:nth-child(2) { display: none; }
  .fb-table--leaderboard { min-width: 100%; table-layout: fixed; }
  .fb-table--leaderboard th:nth-child(4),
  .fb-table--leaderboard td:nth-child(4),
  .fb-table--leaderboard th:nth-child(5),
  .fb-table--leaderboard td:nth-child(5),
  .fb-table--leaderboard th:nth-child(6),
  .fb-table--leaderboard td:nth-child(6),
  .fb-table--leaderboard th:nth-child(7),
  .fb-table--leaderboard td:nth-child(7) { display: none; }
  .fb-table--leaderboard th:nth-child(1),
  .fb-table--leaderboard td:nth-child(1) { width: 12%; }
  .fb-table--leaderboard th:nth-child(2),
  .fb-table--leaderboard td:nth-child(2) { width: 64%; }
  .fb-table--leaderboard th:nth-child(3),
  .fb-table--leaderboard td:nth-child(3) { width: 24%; }
  .fb-score-head-long { display: none; }
  .fb-score-head-short { display: inline; }
  .fb-table--leaderboard td:nth-child(2) {
    overflow: hidden;
    padding-left: 6px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .fb-table--leaderboard th:nth-child(2) { padding-left: 6px; }
  .fb-family-table { min-width: 100%; table-layout: fixed; }
  .fb-family-table th,
  .fb-family-table td { font-size: 9px; padding-right: 4px; }
  .fb-family-table th:first-child,
  .fb-family-table td:first-child { width: 46%; }
  .fb-family-model { gap: 5px; min-width: 0; }
  .fb-family-model .fb-model-mark { flex-basis: 15px; height: 15px; width: 15px; }
  .fb-family-model span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .fb-family-value { display: block; }
  .fb-family-mini-axis { display: none; }
  .fb-family-long { display: none; }
  .fb-family-short { display: inline; }
  .fb-transfer-facts { grid-template-columns: repeat(2, 1fr); }
  .fb-transfer-boundary { grid-template-columns: 1fr; }
  .fb-ci-axis-labels span:nth-child(2) { display: none; }
  .fb-metric-grid { grid-template-columns: repeat(2, 1fr); }
  .fb-metric:nth-child(3n+2), .fb-metric:nth-child(3n+3) { padding-left: 0; }
  .fb-metric:nth-child(even) { padding-left: 0; }
  .fb-metric strong { font-size: 21px; }
  .fb-method-visual { grid-template-columns: 1fr; }
  .fb-leader-tools { align-items: stretch; grid-template-columns: 1fr; gap: 11px; }
  .fb-filter-set { width: 100%; }
  .fb-filter-button { flex: 1; }
  .fb-leader-meta { justify-content: space-between; }
  .fb-metric-rail { display: block; }
  .fb-metric-rail-label { display: block; margin-bottom: 4px; }
  .fb-metric-switch {
    gap: 19px;
    overflow-x: auto;
    padding-bottom: 4px;
    scrollbar-width: none;
  }
  .fb-metric-switch::-webkit-scrollbar { display: none; }
  .fb-metric-note { display: block; margin-top: 8px; text-align: left; }
  .fb-release-line { grid-template-columns: repeat(2, 1fr); }
  .fb-release-line a { justify-self: start; }
  .fb-ci-row { gap: 8px; grid-template-columns: 20px minmax(128px, 168px) 1fr 43px; }
  .fb-ci-axis-labels { margin-left: 196px; margin-right: 51px; }
  .fb-resolution { grid-template-columns: 1fr; }
  .fb-resolution-row + .fb-resolution-row { border-left: 0; padding-left: 0; }
  .fb-publish-path { align-items: start; grid-template-columns: 1fr; }
  .fb-action-link { justify-content: center; }
  .tab-container[role="tablist"] { overflow-x: auto !important; }
  .tab-container[role="tablist"] button { margin-right: 22px !important; white-space: nowrap !important; }
  .fb-footer { display: block; }
  .fb-footer-main,
  .fb-footer-links { align-items: flex-start; flex-direction: column; gap: 6px; }
  .fb-footer-links { margin-top: 14px; }
}
@media (prefers-reduced-transparency: reduce) {
  .tab-wrapper { backdrop-filter: none; background: var(--fb-paper) !important; }
}
@media (max-width: 430px) {
  .fb-forest-row { grid-template-columns: 18px minmax(112px, 140px) 1fr 39px; }
  .fb-chart-foot { margin-left: 148px; margin-right: 47px; }
  .fb-ci-row { grid-template-columns: 18px minmax(108px, 138px) 1fr 40px; }
  .fb-ci-row .fb-model-mark { display: none; }
  .fb-ci-axis-labels { margin-left: 164px; margin-right: 48px; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""
HEAD = f"<style>{FONT_CSS}{CSS}</style>"


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
STABILITY = BUNDLE["stability_analysis"]
REWARD_TRANSFER = BUNDLE.get("reward_transfer")
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
LAB_TASK_LABEL_TO_ID = {
    f"{row['task_id']} | {str(row['family']).replace('_', ' ')} | {row['lab_split']}": str(
        row["task_id"]
    )
    for row in LAB_TASKS
}


def _lab_name(row: dict[str, Any]) -> str:
    model_name = str(row["model_name"])
    prefixes = (
        ("SpaceXAI:", "xAI"),
        ("Anthropic:", "Anthropic"),
        ("Claude ", "Anthropic"),
        ("MoonshotAI:", "Kimi"),
        ("ByteDance Seed:", "ByteDance"),
        ("Thinking Machines:", "Thinking Machines"),
        ("MiniMax:", "MiniMax"),
        ("DeepSeek:", "DeepSeek"),
        ("OpenAI:", "OpenAI"),
        ("Google:", "Google"),
        ("Meta:", "Meta"),
        ("Qwen:", "Qwen"),
        ("Tencent:", "Tencent"),
        ("Z.ai:", "Z.ai"),
        ("NVIDIA:", "NVIDIA"),
        ("Cohere:", "Cohere"),
        ("Mistral:", "Mistral"),
    )
    for prefix, lab in prefixes:
        if model_name.startswith(prefix):
            return lab
    return str(row.get("provider_name") or model_name.split(":", 1)[0])


def _model_label(model_name: str) -> str:
    label = model_name.split(":", 1)[-1].strip()
    return (
        label.replace("GPT-5.6 ", "5.6 ")
        .replace("Claude ", "")
        .replace("DeepSeek ", "")
        .replace("Command ", "")
    )


_seen_labs: set[str] = set()
LAB_CHAMPIONS: list[dict[str, Any]] = []
for _model in DISPLAY_MODELS:
    _lab = _lab_name(_model)
    if _lab not in _seen_labs:
        _seen_labs.add(_lab)
        LAB_CHAMPIONS.append(_model)

LEADERBOARD_METRICS = (
    ("overall", "Overall", "FlavourBench Score", "Score"),
    ("substitution", "Substitution", "Substitution score", "Sub"),
    ("pairing", "Pairing", "Pairing score", "Pair"),
    ("constraint", "Constraints", "Constraint score", "Rules"),
)


def _leaderboard_metric_score(model: dict[str, Any], metric: str) -> float:
    if metric == "overall":
        return float(model["flavourbench_score"])
    return float(model["family_scores"][metric])


LEADERBOARD_RANKS: dict[tuple[str, str], int] = {}
LEADERBOARD_CHAMPIONS: set[tuple[str, str]] = set()
for _metric, _, _, _ in LEADERBOARD_METRICS:
    _ordered = sorted(
        DISPLAY_MODELS,
        key=lambda row, metric=_metric: (
            -_leaderboard_metric_score(row, metric),
            str(row["model_id"]),
        ),
    )
    _previous_score: float | None = None
    _rank = 0
    _metric_labs: set[str] = set()
    for _position, _model in enumerate(_ordered, start=1):
        _score = _leaderboard_metric_score(_model, _metric)
        if _previous_score is None or _score != _previous_score:
            _rank = _position
            _previous_score = _score
        _model_id = str(_model["model_id"])
        LEADERBOARD_RANKS[(_metric, _model_id)] = _rank
        _lab = _lab_name(_model)
        if _lab not in _metric_labs:
            _metric_labs.add(_lab)
            LEADERBOARD_CHAMPIONS.add((_metric, _model_id))

LEADERBOARD_JS = """
if (element.dataset.fbReady !== "true") {
  element.dataset.fbReady = "true";
  const search = element.querySelector("[data-fb-search]");
  const rows = Array.from(element.querySelectorAll("tbody tr[data-model]"));
  const modeButtons = Array.from(element.querySelectorAll("[data-fb-mode]"));
  const metricButtons = Array.from(element.querySelectorAll("[data-fb-metric]"));
  const count = element.querySelector("[data-fb-count]");
  const empty = element.querySelector("[data-fb-empty]");
  const table = element.querySelector("[data-fb-leaderboard]");
  const body = table?.querySelector("tbody");
  const scoreHead = table?.querySelector("[data-fb-score-head]");
  const scoreHeadLong = scoreHead?.querySelector("[data-fb-score-long]");
  const scoreHeadShort = scoreHead?.querySelector("[data-fb-score-short]");
  const metricNote = element.querySelector("[data-fb-metric-note]");
  let mode = "all";
  let metric = "overall";

  const dataKey = (prefix) =>
    prefix + metric.charAt(0).toUpperCase() + metric.slice(1);

  const apply = () => {
    const rankKey = dataKey("rank");
    const scoreKey = dataKey("score");
    const championKey = dataKey("champion");
    rows.sort((left, right) => {
      const rankGap = Number(left.dataset[rankKey]) - Number(right.dataset[rankKey]);
      return rankGap || left.dataset.search.localeCompare(right.dataset.search);
    });
    for (const row of rows) body?.insertBefore(row, empty);

    const query = (search?.value || "").trim().toLocaleLowerCase();
    let visible = 0;
    for (const row of rows) {
      const matchesText = !query || row.dataset.search.includes(query);
      const matchesMode = mode === "all" || row.dataset[championKey] === "true";
      row.hidden = !(matchesText && matchesMode);
      const rankCell = row.querySelector("[data-fb-rank]");
      const scoreCell = row.querySelector("[data-fb-score]");
      if (rankCell) rankCell.textContent = String(row.dataset[rankKey]).padStart(2, "0");
      if (scoreCell) scoreCell.textContent = Number(row.dataset[scoreKey]).toFixed(2);
      if (!row.hidden) visible += 1;
    }
    table?.classList.toggle("fb-family-view", metric !== "overall");
    if (count) count.textContent = `${visible} model${visible === 1 ? "" : "s"}`;
    if (empty) empty.hidden = visible !== 0;
  };

  search?.addEventListener("input", apply);
  for (const button of modeButtons) {
    button.addEventListener("click", () => {
      mode = button.dataset.fbMode;
      for (const peer of modeButtons) {
        peer.setAttribute("aria-pressed", String(peer === button));
      }
      apply();
    });
  }
  for (const button of metricButtons) {
    button.addEventListener("click", () => {
      metric = button.dataset.fbMetric;
      for (const peer of metricButtons) {
        peer.setAttribute("aria-pressed", String(peer === button));
      }
      if (scoreHeadLong) scoreHeadLong.textContent = button.dataset.fbScoreLabel;
      if (scoreHeadShort) scoreHeadShort.textContent = button.dataset.fbScoreShort;
      if (metricNote) metricNote.textContent = button.dataset.fbNote;
      apply();
    });
  }
  apply();
}
"""


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


def _frontier_html() -> str:
    axis_floor = 55.0
    axis_ceiling = 66.0
    rows = []
    for place, model in enumerate(LAB_CHAMPIONS[:10], start=1):
        score = float(model["flavourbench_score"])
        position = max(0.0, min(100.0, (score - axis_floor) / (axis_ceiling - axis_floor) * 100))
        full_name = str(model["model_name"])
        lab = _lab_name(model)
        label = _model_label(full_name)
        logo_url = LAB_LOGO_URLS.get(lab)
        logo = (
            f"<img class='fb-model-mark' src='{logo_url}' alt='' loading='eager'>"
            if logo_url
            else ""
        )
        rows.append(
            "<div class='fb-forest-row'>"
            f"<div class='fb-place'>{place:02d}</div>"
            f"<div class='fb-model' title='{html.escape(full_name)}'>"
            f"{logo}<span class='fb-model-text'>{html.escape(lab)} · {html.escape(label)}</span>"
            "</div>"
            "<div class='fb-axis'>"
            f"<span class='fb-bar' style='width:{position:.3f}%'></span>"
            f"<span class='fb-point' style='left:{position:.3f}%'></span>"
            "</div>"
            f"<div class='fb-number'>{score:.1f}</div>"
            "</div>"
        )
    return "".join(rows) + (
        f"<div class='fb-chart-foot'><span>{axis_floor:.0f}</span>"
        f"<span>FlavourBench Score</span><span>{axis_ceiling:.0f}</span></div>"
    )


def _hero_html() -> str:
    inference = BUNDLE["analysis"]["inference"]
    leading_group_count = sum(int(model["statistical_rank_group"]) == 1 for model in DISPLAY_MODELS)
    return f"""
    <header class="fb-shell fb-masthead">
      <div class="fb-masthead-brand">
        <strong>FlavourBench</strong>
        <span>Executable culinary benchmark</span>
      </div>
      <nav aria-label="Research resources">
        <a href="{PAPER_URL}" target="_blank" rel="noreferrer">Paper</a>
        <a href="{DATASET_URL}" target="_blank" rel="noreferrer">Dataset</a>
        <a href="{SOURCE_URL}" target="_blank" rel="noreferrer">GitHub</a>
      </nav>
    </header>
    <div class="fb-shell fb-hero">
      <section>
        <h1>Which AI knows food best?</h1>
        <p class="fb-dek">Epicure scores every legal answer first. Then {MODEL_COUNT} frontier endpoints face the same {TASK_COUNT} food decisions.</p>
        <div class="fb-stats">
          <div class="fb-stat"><strong>{MODEL_COUNT}</strong><span>frontier endpoints</span></div>
          <div class="fb-stat"><strong>{TASK_COUNT}</strong><span>shared tasks</span></div>
          <div class="fb-stat"><strong>{PRIMARY_COUNT:,}</strong><span>scored answers</span></div>
          <div class="fb-stat"><strong>{inference["pairwise_hypotheses"]}</strong><span>model pairs</span></div>
        </div>
      </section>
      <section class="fb-frontier" aria-label="Best FlavourBench model from each leading lab">
        <div class="fb-frontier-head"><strong><span class="fb-desktop-label">Best model from each lab</span><span class="fb-mobile-label">Top lab leaders</span></strong><span>Score / 100, focused 55 to 66 axis</span></div>
        {_frontier_html()}
        <div class="fb-frontier-note"><span>Overall point estimates</span><strong>{leading_group_count} models share the leading statistical group</strong></div>
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
                "Model": model["model_name"],
                "Score ↑": round(float(model["flavourbench_score"]), 2),
                "Simultaneous 95%": f"{ci[0]:.2f} to {ci[1]:.2f}",
                "Group": model["statistical_rank_group"],
                "Rank 95%": f"{rank_ci[0]} to {rank_ci[1]}",
                "Cells": f"{model['coverage']['valid_scored']}/{TASK_COUNT}",
            }
        )
    return pd.DataFrame(rows)


def _leaderboard_html() -> str:
    family_task_counts = {
        family: sum(str(task["family"]) == family for task in TASKS)
        for family in ("substitution", "pairing", "constraint")
    }
    metric_controls = []
    for metric, label, score_label, score_short in LEADERBOARD_METRICS:
        note = (
            "Equal-family mean. Open Insights for uncertainty and significance."
            if metric == "overall"
            else f"{family_task_counts[metric]} {label.lower()} tasks. Family views rank point scores only."
        )
        metric_controls.append(
            f"<button type='button' data-fb-metric='{metric}' "
            f"data-fb-score-label='{html.escape(score_label, quote=True)}' "
            f"data-fb-score-short='{html.escape(score_short, quote=True)}' "
            f"data-fb-note='{html.escape(note, quote=True)}' "
            f"aria-pressed='{str(metric == 'overall').lower()}'>{html.escape(label)}</button>"
        )

    rows = []
    for model in DISPLAY_MODELS:
        ci = model["score_simultaneous_95_ci"]
        rank_ci = model["bootstrap_rank_95_interval"]
        model_name = str(model["model_name"])
        lab = _lab_name(model)
        label = _model_label(model_name)
        model_id = str(model["model_id"])
        search_value = html.escape(f"{model_name} {lab}".lower(), quote=True)
        metric_attributes = " ".join(
            (
                f"data-score-{metric}='{_leaderboard_metric_score(model, metric):.8f}' "
                f"data-rank-{metric}='{LEADERBOARD_RANKS[(metric, model_id)]}' "
                f"data-champion-{metric}='{str((metric, model_id) in LEADERBOARD_CHAMPIONS).lower()}'"
            )
            for metric, _, _, _ in LEADERBOARD_METRICS
        )
        logo_url = LAB_LOGO_URLS.get(lab)
        logo = (
            f"<img class='fb-model-mark' src='{logo_url}' alt='' loading='lazy'>"
            if logo_url
            else ""
        )
        rows.append(
            f"<tr data-model='{html.escape(model_id, quote=True)}' "
            f"data-search='{search_value}' data-lab='{html.escape(lab, quote=True)}' "
            f"{metric_attributes}>"
            f"<td class='fb-rank-cell' data-fb-rank>{int(model['point_estimate_rank']):02d}</td>"
            f"<td><div class='fb-table-model' title='{html.escape(model_name, quote=True)}'>"
            f"{logo}<span>{html.escape(lab)} · {html.escape(label)}</span></div></td>"
            f"<td class='fb-score-cell' data-fb-score>{float(model['flavourbench_score']):.2f}</td>"
            f"<td class='fb-overall-only'>{float(ci[0]):.2f} to {float(ci[1]):.2f}</td>"
            f"<td class='fb-overall-only'>G{model['statistical_rank_group']}</td>"
            f"<td class='fb-overall-only'>{rank_ci[0]} to {rank_ci[1]}</td>"
            f"<td>{model['coverage']['valid_scored']}/{TASK_COUNT}</td>"
            "</tr>"
        )
    release_id = html.escape(str(BUNDLE["release_artifact_sha256"])[:12])
    return (
        f"""
    <div class="fb-leader-tools" role="search" aria-label="Filter leaderboard">
      <label class="fb-search-label">Find a model
        <input type="search" placeholder="Model or lab" autocomplete="off" data-fb-search>
      </label>
      <div class="fb-filter-set" aria-label="Leaderboard scope">
        <button type="button" class="fb-filter-button" data-fb-mode="all" aria-pressed="true">All {MODEL_COUNT}</button>
        <button type="button" class="fb-filter-button" data-fb-mode="champions" aria-pressed="false">Lab leaders</button>
      </div>
      <div class="fb-leader-meta">
        <span class="fb-result-count" data-fb-count aria-live="polite">{MODEL_COUNT} models</span>
        <a class="fb-text-link" href="{DATASET_RESULTS_URL}" target="_blank" rel="noreferrer">Download JSONL</a>
      </div>
    </div>
    <div class="fb-metric-rail">
      <span class="fb-metric-rail-label">Rank by</span>
      <div class="fb-metric-switch" role="group" aria-label="Leaderboard score view">
        {"".join(metric_controls)}
      </div>
      <span class="fb-metric-note" data-fb-metric-note>Equal-family mean. Open Insights for uncertainty and significance.</span>
    </div>
    <div class="fb-table-wrap">
      <table class="fb-table fb-table--leaderboard" data-fb-leaderboard>
        <caption>Complete FlavourBench common-core leaderboard</caption>
        <thead><tr>
          <th scope="col">Rank</th><th scope="col">Model</th><th scope="col" data-fb-score-head><span class="fb-score-head-long" data-fb-score-long>FlavourBench Score</span><span class="fb-score-head-short" data-fb-score-short>Score</span></th>
          <th scope="col" class="fb-overall-only">Simultaneous 95%</th><th scope="col" class="fb-overall-only">Group</th>
          <th scope="col" class="fb-overall-only">Rank 95%</th><th scope="col">Cells</th>
        </tr></thead>
        <tbody>"""
        + "".join(rows)
        + f"""
          <tr class="fb-empty-row" data-fb-empty hidden><td colspan="7">No model matches this search.</td></tr>
        </tbody>
      </table>
    </div>
    <div class="fb-release-line" aria-label="Release provenance">
      <span><strong>Release</strong> {release_id}</span>
      <span><strong>Matrix</strong> {MODEL_COUNT} × {TASK_COUNT}</span>
      <span><strong>Coverage</strong> {PRIMARY_COUNT:,}/{PRIMARY_COUNT:,}</span>
      <a class="fb-text-link" href="https://huggingface.co/datasets/josefchen/flavourbench" target="_blank" rel="noreferrer">Open dataset</a>
    </div>
    """
    )


def _axis_position(value: float, floor: float, ceiling: float) -> float:
    return max(0.0, min(100.0, (value - floor) / (ceiling - floor) * 100.0))


def _insights_html() -> str:
    axis_floor = 55.0
    axis_ceiling = 70.0
    score_rows = []
    for model in DISPLAY_MODELS[:10]:
        model_name = str(model["model_name"])
        lab = _lab_name(model)
        label = _model_label(model_name)
        score = float(model["flavourbench_score"])
        ci_low, ci_high = (float(value) for value in model["score_simultaneous_95_ci"])
        low = _axis_position(ci_low, axis_floor, axis_ceiling)
        high = _axis_position(ci_high, axis_floor, axis_ceiling)
        point = _axis_position(score, axis_floor, axis_ceiling)
        logo_url = LAB_LOGO_URLS.get(lab)
        logo = (
            f"<img class='fb-model-mark' src='{logo_url}' alt='' loading='lazy'>"
            if logo_url
            else ""
        )
        aria = html.escape(
            f"{model_name}: score {score:.2f}, simultaneous 95 percent interval "
            f"{ci_low:.2f} to {ci_high:.2f}",
            quote=True,
        )
        score_rows.append(
            "<div class='fb-ci-row'>"
            f"<div class='fb-place'>{int(model['point_estimate_rank']):02d}</div>"
            f"<div class='fb-model' title='{html.escape(model_name, quote=True)}'>"
            f"{logo}<span class='fb-model-text'>{html.escape(lab)} · {html.escape(label)}</span>"
            "</div>"
            f"<div class='fb-ci-track' role='img' aria-label='{aria}'>"
            f"<span class='fb-ci-band' style='left:{low:.3f}%;width:{high - low:.3f}%'></span>"
            f"<span class='fb-ci-point' style='left:{point:.3f}%'></span>"
            "</div>"
            f"<div class='fb-number'>{score:.2f}</div>"
            "</div>"
        )

    family_models = LAB_CHAMPIONS[:8]
    families = (
        ("substitution", "Substitution"),
        ("pairing", "Pairing"),
        ("constraint", "Constraint"),
    )
    family_maxima = {
        family: max(float(model["family_scores"][family]) for model in family_models)
        for family, _ in families
    }
    family_rows = []
    for model in family_models:
        model_name = str(model["model_name"])
        lab = _lab_name(model)
        label = _model_label(model_name)
        logo_url = LAB_LOGO_URLS.get(lab)
        logo = (
            f"<img class='fb-model-mark' src='{logo_url}' alt='' loading='lazy'>"
            if logo_url
            else ""
        )
        cells = []
        for family, _ in families:
            score = float(model["family_scores"][family])
            position = _axis_position(score, 40.0, 75.0)
            best_class = " fb-family-best" if score == family_maxima[family] else ""
            cells.append(
                f"<td class='{best_class.strip()}'>"
                f"<div class='fb-family-value{best_class}'><span>{score:.1f}</span>"
                "<span class='fb-family-mini-axis' aria-hidden='true'>"
                f"<i class='fb-family-dot' style='left:{position:.3f}%'></i></span></div></td>"
            )
        family_rows.append(
            "<tr>"
            f"<td><div class='fb-family-model' title='{html.escape(model_name, quote=True)}'>"
            f"{logo}<span>{html.escape(lab)} · {html.escape(label)}</span></div></td>"
            + "".join(cells)
            + "</tr>"
        )

    leading_group_count = sum(int(model["statistical_rank_group"]) == 1 for model in DISPLAY_MODELS)
    resolved = int(BUNDLE["analysis"]["resolved_pair_count"])
    chance = float(DISPLAY_MODELS[0]["chance_comparison"]["exact_chance_score"])
    leader_gap = float(
        DISPLAY_MODELS[0]["chance_comparison"].get(
            "mean_difference", float(DISPLAY_MODELS[0]["flavourbench_score"]) - chance
        )
    )
    stability_rows = []
    for row in STABILITY["task_count_stability"]:
        rank = row["metrics"]["rank_spearman"]
        top_five = row["metrics"]["top_5_overlap"]
        leader = row["metrics"]["top_1_preserved"]
        stability_rows.append(
            "<tr>"
            f"<td>{int(row['tasks'])}</td>"
            f"<td>{float(rank['median']):.3f}</td>"
            f"<td>{float(rank['p2_5']):.3f} to {float(rank['p97_5']):.3f}</td>"
            f"<td>{float(top_five['median']) * 100:.0f}%</td>"
            f"<td>{float(leader['mean']) * 100:.1f}%</td>"
            "</tr>"
        )
    variance = STABILITY["variance_partition"]
    generalizability = float(variance["relative_decision_generalizability_at_534_tasks"])
    tasks_for_g_90 = int(variance["estimated_balanced_tasks_for_relative_g_0_90"])
    return f"""
    <div class="fb-insight-layout">
      <section>
        <div class="fb-panel-label">Top 10 / simultaneous 95% bands / focused 55 to 70 axis</div>
        <div class="fb-ci-plot">{"".join(score_rows)}</div>
        <div class="fb-ci-axis-labels"><span>{axis_floor:.0f}</span><span>FlavourBench Score</span><span>{axis_ceiling:.0f}</span></div>
      </section>
      <aside>
        <div class="fb-panel-label">What the evidence resolves</div>
        <div class="fb-resolution">
          <div class="fb-resolution-row"><strong>{leading_group_count}</strong><span>models share the leading statistical group</span></div>
          <div class="fb-resolution-row"><strong>{resolved}/{PAIR_COUNT}</strong><span>paired gaps remain significant after Holm control</span></div>
          <div class="fb-resolution-row"><strong>+{leader_gap:.1f}</strong><span>points above the analytically computed random-choice baseline ({chance:.1f})</span></div>
          <p class="fb-resolution-note">Point estimates give the order. The score bands and pairwise tests show which gaps this release can defend.</p>
        </div>
      </aside>
    </div>
    <section class="fb-family-insight">
      <h3>Why 534 tasks?</h3>
      <p>The crossed design's descriptive relative-decision generalizability is
      <strong>{generalizability:.3f}</strong>; the same variance model estimates
      <strong>{tasks_for_g_90}</strong> balanced tasks for 0.90. The table below repeatedly takes
      score-blind, balanced subsets and compares them with the complete point order.</p>
      <div class="fb-table-wrap">
        <table class="fb-table">
          <caption>5,000 family-by-panel stratified subsets at each non-complete task count</caption>
          <thead><tr>
            <th scope="col">Tasks</th>
            <th scope="col">Median rank ρ</th>
            <th scope="col">Empirical 95%</th>
            <th scope="col">Top-five overlap</th>
            <th scope="col">Point leader kept</th>
          </tr></thead>
          <tbody>{"".join(stability_rows)}</tbody>
        </table>
      </div>
      <p class="fb-resolution-note">This is a precision diagnostic relative to the complete
      release, not a post-hoc power claim. The point leader remains unstable in smaller subsets;
      the simultaneous score bands and rank intervals remain the inferential result.</p>
    </section>
    <section class="fb-family-insight">
      <h3>Where the leading labs differ</h3>
      <p>Scores are out of 100. Red marks each column leader.</p>
      <div class="fb-table-wrap">
        <table class="fb-table fb-family-table">
          <caption>Family scores for the eight leading lab champions</caption>
          <thead><tr>
            <th scope="col">Lab champion</th>
            <th scope="col"><span class="fb-family-long">Substitution</span><span class="fb-family-short">Sub</span></th>
            <th scope="col"><span class="fb-family-long">Pairing</span><span class="fb-family-short">Pair</span></th>
            <th scope="col"><span class="fb-family-long">Constraint</span><span class="fb-family-short">Rules</span></th>
          </tr></thead>
          <tbody>{"".join(family_rows)}</tbody>
        </table>
      </div>
    </section>
    """


def _reward_transfer_html() -> str:
    if not isinstance(REWARD_TRANSFER, dict) or REWARD_TRANSFER.get("status") != "complete":
        return """
        <p class="fb-resolution-note">The verified reward-transfer release is not present in this
        local Space bundle. Rebuild the bundle from the released analysis artifacts.</p>
        """
    primary = REWARD_TRANSFER["primary"]
    replication = REWARD_TRANSFER["public_replication"]
    base_model = REWARD_TRANSFER["base_model"]

    def score(block: dict[str, Any], condition: str) -> float:
        return float(block["scores"][condition])

    primary_low, primary_high = (float(value) for value in primary["confidence_interval_95"])
    replication_low, replication_high = (
        float(value) for value in replication["confidence_interval_95"]
    )
    release_id = html.escape(str(REWARD_TRANSFER["release_artifact_sha256"]))
    model_id = html.escape(str(base_model["id"]))
    revision = html.escape(str(base_model["revision"])[:12])
    figure_record = REWARD_TRANSFER.get("figure")
    figure_url = REWARD_TRANSFER_FIGURE_URL
    if isinstance(figure_record, dict) and str(figure_record.get("data_url", "")).startswith(
        "data:image/png;base64,"
    ):
        figure_url = str(figure_record["data_url"])
    figure_url = html.escape(figure_url, quote=True)
    return f"""
    <div class="fb-transfer-layout">
      <section class="fb-transfer-primary">
        <strong>+{float(primary["effect_points"]):.2f}</strong>
        <span>points beyond format-matched supervision</span>
        <p>Primary estimate on {int(primary["tasks"])} predeclared, anchor-disjoint maps.
        The 95% interval is {primary_low:.2f} to {primary_high:.2f}; the matched sign-flip
        p-value is {float(primary["two_sided_sign_flip_p"]):.6f}. That p-value tests held-out
        anchors conditional on the three matched seed pairs.</p>
      </section>
      <section>
        <div class="fb-table-wrap">
          <table class="fb-table fb-transfer-table">
            <caption>Controlled reward-transfer scores</caption>
            <thead><tr>
              <th scope="col">Evaluation split</th>
              <th scope="col">Base</th>
              <th scope="col">Format control</th>
              <th scope="col">Epicure SFT</th>
              <th scope="col">Treatment effect, 95% CI</th>
            </tr></thead>
            <tbody>
              <tr>
                <td>Unseen primary ({int(primary["tasks"])})</td>
                <td>{score(primary, "pretrained_base"):.2f}</td>
                <td>{score(primary, "sft_format_control"):.2f}</td>
                <td>{score(primary, "sft_epicure_optimum"):.2f}</td>
                <td>+{float(primary["effect_points"]):.2f} [{primary_low:.2f}, {primary_high:.2f}]</td>
              </tr>
              <tr>
                <td>Public replication ({int(replication["tasks"])})</td>
                <td>{score(replication, "pretrained_base"):.2f}</td>
                <td>{score(replication, "sft_format_control"):.2f}</td>
                <td>{score(replication, "sft_epicure_optimum"):.2f}</td>
                <td>+{float(replication["effect_points"]):.2f} [{replication_low:.2f}, {replication_high:.2f}]</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="fb-transfer-facts">
          <div class="fb-transfer-fact"><strong>{len(REWARD_TRANSFER["training_seeds"])}</strong><span>matched seeds</span></div>
          <div class="fb-transfer-fact"><strong>{int(REWARD_TRANSFER["training_tasks"])}</strong><span>training maps</span></div>
          <div class="fb-transfer-fact"><strong>{int(REWARD_TRANSFER["validation_tasks"])}</strong><span>validation maps</span></div>
          <div class="fb-transfer-fact"><strong>{int(REWARD_TRANSFER["response_rows"]):,}</strong><span>released generations</span></div>
        </div>
      </section>
    </div>
    <figure class="fb-transfer-figure">
      <img src="{figure_url}" loading="lazy" alt="Scores for the base, format-control SFT, and Epicure-optimum SFT conditions on the unseen primary split and public replication split, with confidence intervals for the treatment effect.">
      <figcaption>One pinned {model_id} checkpoint ({revision}) and the same three training seeds in both SFT conditions. The control sees the same prompts and output format; only the target portfolios differ.</figcaption>
    </figure>
    <div class="fb-transfer-boundary">
      <section>
        <h3>What this result establishes</h3>
        <p>Epicure-optimal supervision increases agreement with unseen Epicure reward maps beyond
        format-matched supervision. Every matched-seed effect is positive, and the result repeats
        on all {int(replication["tasks"])} public maps.</p>
      </section>
      <section>
        <h3>Where the claim stops</h3>
        <p>This is controlled SFT transfer within the Epicure construct. It does not establish
        human taste, general model quality, or reinforcement-learning improvement.</p>
      </section>
    </div>
    <div class="fb-transfer-links">
      <a class="fb-text-link" href="{REWARD_TRANSFER_PROTOCOL_URL}" target="_blank" rel="noreferrer">Read the protocol and result</a>
      <a class="fb-text-link" href="{REWARD_TRANSFER_DATA_URL}" target="_blank" rel="noreferrer">Open all released generations</a>
      <a class="fb-text-link" href="{REWARD_TRANSFER_VERIFY_URL}" target="_blank" rel="noreferrer">Inspect the offline verifier</a>
      <span class="fb-hash">release {release_id}</span>
    </div>
    """


def _model_detail(model_name: str) -> tuple[str, pd.DataFrame]:
    model = MODEL_BY_NAME[model_name]
    rank_interval = model["bootstrap_rank_95_interval"]
    coverage = model["coverage"]
    panel_replication = model["panel_replication"]
    score_interval = model["score_simultaneous_95_ci"]
    summary = f"""
    <div class="fb-metric-grid">
      <div class="fb-metric"><small>FlavourBench Score</small><strong>{model["flavourbench_score"]:.2f}</strong></div>
      <div class="fb-metric"><small>Complete cells</small><strong>{coverage["valid_scored"]}/{coverage["scheduled"]}</strong></div>
      <div class="fb-metric"><small>Point rank</small><strong>#{model["point_estimate_rank"]}</strong></div>
      <div class="fb-metric"><small>Statistical group</small><strong>G{model["statistical_rank_group"]}</strong></div>
      <div class="fb-metric"><small>Bootstrap rank</small><strong>{rank_interval[0]}-{rank_interval[1]}</strong></div>
      <div class="fb-metric"><small>95% score band</small><strong>{float(score_interval[0]):.1f}-{float(score_interval[1]):.1f}</strong></div>
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
        {"Family": "Random legal choice (exact)", "Score": chance_score, "Cells": TASK_COUNT}
    )
    return summary, pd.DataFrame(family_rows)


def _family_table_html(frame: pd.DataFrame) -> str:
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['Family']))}</td>"
            f"<td class='fb-score-cell'>{float(record['Score']):.3f}</td>"
            f"<td>{int(record['Cells'])}</td>"
            "</tr>"
        )
    return (
        "<div class='fb-data-heading'>Score by task family</div>"
        "<div class='fb-table-wrap'><table class='fb-table fb-table--family'>"
        "<caption>FlavourBench score by task family</caption>"
        "<thead><tr><th scope='col'>Family</th><th scope='col'>Score</th>"
        "<th scope='col'>Cells</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _model_detail_ui(model_name: str) -> tuple[str, str]:
    summary, frame = _model_detail(model_name)
    return summary, _family_table_html(frame)


def _choices_html(choices: dict[str, str]) -> str:
    rows = []
    for label, ingredient in choices.items():
        rows.append(
            "<div class='fb-choice'>"
            f"<span class='fb-choice-label'>{html.escape(str(label))}</span>"
            f"<span class='fb-choice-name'>{html.escape(str(ingredient))}</span>"
            "</div>"
        )
    return (
        "<div class='fb-data-heading'>Eight candidates</div>"
        "<div class='fb-choice-grid'>" + "".join(rows) + "</div>"
    )


def _score_table_html(frame: pd.DataFrame) -> str:
    rows = []
    for record in frame.to_dict(orient="records"):
        selected = " fb-selected-row" if record["Role"] else ""
        rows.append(
            f"<tr class='{selected.strip()}'>"
            f"<td>{html.escape(str(record['Selection']))}</td>"
            f"<td>{html.escape(str(record['Ingredients']))}</td>"
            f"<td class='fb-score-cell'>{float(record['Score']):.2f}</td>"
            f"<td>{html.escape(str(record['Role']))}</td>"
            "</tr>"
        )
    return (
        "<div class='fb-data-heading'>Top 12 of 56 scored selections</div>"
        "<div class='fb-table-wrap'><table class='fb-table fb-table--score'>"
        "<caption>Highest-scoring legal ingredient selections</caption>"
        "<thead><tr><th scope='col'>Selection</th><th scope='col'>Ingredients</th>"
        "<th scope='col'>Score</th><th scope='col'>Role</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


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
    score_rows = []
    for selection, score in ranked[:12]:
        roles = []
        if selection == observed:
            roles.append("model selection")
        if selection == optimum:
            roles.append("optimum")
        score_rows.append(
            {
                "Selection": selection,
                "Ingredients": ", ".join(task["choices"][label] for label in selection),
                "Score": int(score) / 100,
                "Role": ", ".join(roles),
            }
        )
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


def _task_detail_ui(model_name: str, task_label: str) -> tuple[str, str, str, str, str, str]:
    status, prompt, choices, score_map, answer, provenance = _task_detail(model_name, task_label)
    return (
        status,
        prompt,
        _choices_html(choices),
        _score_table_html(score_map),
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
    cohen_dz = row.get("cohen_dz")
    cohen_text = f"{float(cohen_dz):+.3f}" if cohen_dz is not None else "not available"
    return f"""
    <div class="fb-evidence">
      <strong>{html.escape(left_name)}</strong> minus <strong>{html.escape(right_name)}</strong>:
      <strong>{difference:+.3f} points</strong> (bootstrap 95% {interval[0]:+.3f} to {interval[1]:+.3f}).
      The comparison is <strong>{verdict}</strong> across all {PAIR_COUNT} tests
      (shared valid tasks: <code>{row.get("shared_valid_tasks", TASK_COUNT)}</code>).
      <br>Holm p = <code>{float(row["holm_p"]):.4g}</code>, paired Cohen dz =
      <code>{cohen_text}</code>.
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


def _command_preview(
    runtime: str,
    model: str,
    base_url: str,
    api_key_env: str,
    scope: str,
) -> str:
    """Render a copyable lab command without receiving model credentials."""

    model = " ".join(str(model or "").split()) or "your-exact-model-id"
    base_url = " ".join(str(base_url or "").split()) or "https://your-endpoint.example/v1"
    api_key_env = "".join(
        character for character in str(api_key_env or "") if character.isalnum() or character == "_"
    )
    api_key_env = api_key_env or "LAB_MODEL_API_KEY"
    smoke = " --limit 12" if scope.startswith("12-task") else ""
    backend = "transformers" if runtime.startswith("Local") else "openai-compatible"
    route = ""
    if backend == "openai-compatible":
        route = f" \\\n  --base-url {shlex.quote(base_url)} \\\n  --api-key-env {shlex.quote(api_key_env)}"
    command = (
        'python -m pip install "epicure-flavourbench @ '
        'git+https://github.com/josefchen/flavourbench.git"\n\n'
        f"flavourbench run \\\n  --backend {backend} \\\n  --model {shlex.quote(model)}"
        f"{route} \\\n  --responses responses.jsonl \\\n  --report flavourbench-report.json \\\n  --resume{smoke}"
    )
    note = (
        f"Set `{api_key_env}` in your shell before running. The key stays on your machine."
        if backend == "openai-compatible"
        else "The checkpoint runs locally through Transformers. No endpoint credential is used."
    )
    return f"```bash\n{command}\n```\n\n{note}"


def _reward_preview(task_label: str, completion: str) -> str:
    """Score one development completion for the interactive reward demonstration."""

    task_id = LAB_TASK_LABEL_TO_ID[task_label]
    try:
        result = _training_reward_api(task_id, completion)
    except SpaceLabError as error:
        return f"<div class='fb-evidence'><strong>Not parseable.</strong> {html.escape(str(error))}</div>"
    selection = html.escape(str(result.get("selection") or result.get("observed_selection") or ""))
    return f"""
    <div class="fb-evidence">
      <strong>Reward {float(result["reward"]):.4f}</strong>, score {float(result["score"]):.2f}, selection <code>{selection}</code><br>
      This map belongs to the anchor-disjoint development track. It cannot alter the public leaderboard.
    </div>
    """


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


def _score_upload_ui(
    artifact_path: str | None,
    model_name: str,
    disclosure: str,
) -> tuple[Any, Any, Any]:
    """Reveal upload results only after the scorer has produced them."""

    summary, rows, report_path = _score_upload(artifact_path, model_name, disclosure)
    return (
        gr.update(value=summary, visible=True),
        gr.update(value=rows, visible=True),
        gr.update(value=report_path, visible=True),
    )


theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#F8ECE9",
        c100="#F1DDD8",
        c200="#E7BCB5",
        c300="#D98F84",
        c400="#C96357",
        c500=RUST,
        c600="#98352D",
        c700="#7C2E28",
        c800="#662A25",
        c900="#562722",
        c950="#2F120F",
    ),
    neutral_hue="stone",
    font=(
        gr.themes.Font("Lato"),
        gr.themes.Font("Avenir Next"),
        gr.themes.Font("Arial"),
        gr.themes.Font("sans-serif"),
    ),
    font_mono=(
        gr.themes.Font("DejaVu Sans Mono"),
        gr.themes.Font("monospace"),
    ),
).set(
    body_background_fill="#F6F7F5",
    block_background_fill="#F6F7F5",
    block_border_width="0px",
    block_label_text_weight="600",
    button_primary_background_fill=INK,
    button_primary_background_fill_hover=RUST,
)


with gr.Blocks(title="FlavourBench | Which AI knows food best?") as demo:
    gr.HTML(_hero_html(), elem_id="fb-hero-block")
    with gr.Tabs():
        with gr.Tab("Leaders"):
            gr.HTML(
                f"""
                <div class="fb-section">
                  <h2>The complete leaderboard</h2>
                  <p>All {MODEL_COUNT} models face the same {TASK_COUNT} tasks. Change the score view, filter the field, then inspect the uncertainty behind the overall rank.</p>
                </div>
                """
            )
            gr.HTML(_leaderboard_html(), js_on_load=LEADERBOARD_JS)
            gr.Markdown(
                "**Read the score first.** The focused chart above uses point estimates. "
                "The table adds simultaneous 95% bands, bootstrap rank intervals, and multiplicity-controlled groups. "
                "A score of 100 means selecting Epicure's optimum on every task."
            )

        with gr.Tab("Insights"):
            gr.HTML(
                """
                <div class="fb-section">
                  <h2>Ranks need error bars</h2>
                  <p>The point order is real, but not every adjacent gap is resolved. Simultaneous bands and paired tests show where the evidence separates models.</p>
                </div>
                """
            )
            gr.HTML(_insights_html())

        with gr.Tab("Transfer"):
            gr.HTML(
                """
                <div class="fb-section">
                  <h2>Can the benchmark teach a model?</h2>
                  <p>A preregistered, three-seed SFT study isolates Epicure supervision from output-format learning, then tests the same adapters on unseen reward maps.</p>
                </div>
                """
            )
            gr.HTML(_reward_transfer_html())

        with gr.Tab("Profiles"):
            gr.HTML(
                """
                <div class="fb-section">
                  <h2>Where each model wins</h2>
                  <p>Break the headline score into substitution, pairing, and constraint performance, then compare the two collection panels.</p>
                </div>
                """
            )
            model_selector = gr.Dropdown(
                choices=MODEL_NAMES,
                value=MODEL_NAMES[0],
                label="Model",
                filterable=True,
            )
            initial_model_summary, initial_family_table = _model_detail_ui(MODEL_NAMES[0])
            model_summary = gr.HTML(initial_model_summary)
            family_table = gr.HTML(initial_family_table)
            model_selector.change(
                _model_detail_ui,
                inputs=model_selector,
                outputs=[model_summary, family_table],
                api_visibility="private",
            )

        with gr.Tab("Inspect"):
            gr.HTML(
                """
                <div class="fb-section">
                  <h2>Open one scored decision</h2>
                  <p>Every answer is traceable to the exact prompt, model response, and precomputed 56-choice reward surface.</p>
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
                inspect_task = gr.Button("Inspect task", variant="primary", scale=0)
            initial = _task_detail_ui(MODEL_NAMES[0], next(iter(TASK_LABEL_TO_ID)))
            task_status = gr.HTML(initial[0])
            prompt = gr.Textbox(value=initial[1], label="Exact prompt", lines=11, interactive=False)
            choices = gr.HTML(initial[2])
            score_map = gr.HTML(initial[3])
            answer = gr.Markdown(value=initial[4], label="Model response")
            provenance = gr.Markdown(value=initial[5], label="Content hashes and route")
            inspect_task.click(
                _task_detail_ui,
                inputs=[task_model, task_selector],
                outputs=[task_status, prompt, choices, score_map, answer, provenance],
                api_visibility="private",
            )

        with gr.Tab("Run your model"):
            gr.HTML(
                f"""
                <div class="fb-section">
                  <h2>Run FlavourBench on any model</h2>
                  <p>Your endpoint key or checkpoint stays in your environment. The open runner fetches the exact task set, resumes interrupted jobs, and writes a verifiable report.</p>
                </div>
                <div class="fb-lab-path">
                  <strong>Endpoint / checkpoint <i>→</i> {TASK_COUNT} shared tasks <i>→</i> verifiable report</strong>
                  <span>Keys stay local. Start with 12 tasks, resume without repeated calls, then publish the exact responses and route.</span>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    runtime = gr.Radio(
                        choices=[
                            "Hosted OpenAI-compatible endpoint",
                            "Local Transformers checkpoint",
                        ],
                        value="Hosted OpenAI-compatible endpoint",
                        label="Runtime",
                    )
                    run_model = gr.Textbox(
                        value="your-exact-model-id",
                        label="Model ID or checkpoint",
                    )
                    run_scope = gr.Radio(
                        choices=["12-task smoke test", f"Full {TASK_COUNT}-task benchmark"],
                        value="12-task smoke test",
                        label="Run size",
                    )
                with gr.Column(scale=1):
                    run_base_url = gr.Textbox(
                        value="https://your-endpoint.example/v1",
                        label="Base URL for hosted endpoints",
                    )
                    run_key_env = gr.Textbox(
                        value="LAB_MODEL_API_KEY",
                        label="Local environment variable containing the key",
                    )
                    generate_command = gr.Button("Build command", variant="primary")
            initial_command = _command_preview(
                "Hosted OpenAI-compatible endpoint",
                "your-exact-model-id",
                "https://your-endpoint.example/v1",
                "LAB_MODEL_API_KEY",
                "12-task smoke test",
            )
            run_command = gr.Markdown(initial_command)
            generate_command.click(
                _command_preview,
                inputs=[runtime, run_model, run_base_url, run_key_env, run_scope],
                outputs=run_command,
                api_visibility="private",
            )

            gr.HTML(
                """
                <div class="fb-section">
                  <h2>Try the training reward</h2>
                  <p>Paste one completion and query an anchor-disjoint development map. This is the same deterministic reward used by the local GRPO recipe.</p>
                </div>
                """
            )
            initial_lab_label = next(iter(LAB_TASK_LABEL_TO_ID))
            with gr.Row():
                reward_task = gr.Dropdown(
                    choices=list(LAB_TASK_LABEL_TO_ID),
                    value=initial_lab_label,
                    label="Development task",
                    filterable=True,
                    scale=2,
                )
                reward_completion = gr.Textbox(
                    value="FINAL_SELECTION: A,B,C",
                    label="Model completion",
                    scale=2,
                )
                score_reward = gr.Button("Score answer", variant="primary", scale=0)
            reward_result = gr.HTML(_reward_preview(initial_lab_label, "FINAL_SELECTION: A,B,C"))
            score_reward.click(
                _reward_preview,
                inputs=[reward_task, reward_completion],
                outputs=reward_result,
                api_visibility="private",
            )

            gr.HTML(
                f"""
                <div class="fb-section">
                  <h2>Score a completed run</h2>
                  <p>Upload one JSON or JSONL response per task. Complete runs receive a FlavourBench Score; partial runs receive diagnostics only.</p>
                </div>
                <div class="fb-evidence"><strong>Comparable means complete.</strong> All {TASK_COUNT} responses must be present and parseable. Uploads are scored in-session and never added to the public leaderboard automatically.</div>
                """
            )
            with gr.Row():
                lab_name = gr.Textbox(
                    label="Model or experiment name",
                    placeholder="lab/model-name, checkpoint, decoding policy",
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
                height=150,
            )
            score_upload = gr.Button("Score complete artifact", variant="primary")
            lab_summary = gr.Markdown(visible=False)
            lab_rows = gr.Dataframe(
                interactive=False,
                wrap=True,
                show_search="filter",
                show_row_numbers=False,
                label="Per-task results",
                visible=False,
            )
            lab_report = gr.DownloadButton(
                "Download content-addressed report",
                visible=False,
                size="md",
                elem_classes="fb-download",
            )
            score_upload.click(
                _score_upload_ui,
                inputs=[lab_upload, lab_name, lab_disclosure],
                outputs=[lab_summary, lab_rows, lab_report],
                api_visibility="private",
            )
            gr.HTML(
                f"""
                <div class="fb-publish-path">
                  <div>
                    <h3>Publish a result</h3>
                    <p>Submit the complete report, raw responses, exact route, decoding settings, and training disclosure. Maintainers verify the {TASK_COUNT}-task matrix before any leaderboard update.</p>
                  </div>
                  <a class="fb-action-link" href="{SUBMIT_RESULT_URL}" target="_blank" rel="noreferrer">Open result submission</a>
                </div>
                """
            )
            upload_api = gr.Button(visible=False)
            upload_api.click(
                _score_upload,
                inputs=[lab_upload, lab_name, lab_disclosure],
                outputs=[lab_summary, lab_rows, lab_report],
                api_name="score_uploaded_submission",
            )
            gr.Markdown(
                f"""
```json
{{"task_id":"...","status":"completed","response":"FINAL_SELECTION: A,B,C"}}
```

The named `score_completion`, `score_submission`, and `training_reward` endpoints appear under
**Use via API**. For high-throughput RL, use the local reward lookup and the runnable SFT, DPO,
and GRPO recipes in the source repository. The full publication contract is in the
[submission guide]({SUBMISSION_GUIDE_URL}).
                """
            )

        with gr.Tab("Compare"):
            gr.HTML(
                f"""
                <div class="fb-section">
                  <h2>Does the gap hold up?</h2>
                  <p>Query any of the {PAIR_COUNT} paired model contrasts on the same tasks, with Holm control across the full comparison family.</p>
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
            compare.click(
                _pair_detail,
                inputs=[left_model, right_model],
                outputs=pair_result,
                api_visibility="private",
            )

        with gr.Tab("Method"):
            gr.HTML(
                f"""
                <div class="fb-section">
                  <h2>One lookup, repeated {TASK_COUNT} times</h2>
                  <p>The Space makes no provider calls. It reads released reward maps and returns deterministic scores.</p>
                </div>
                <div class="fb-method-visual">
                  <img class="fb-architecture" src="{ARCHITECTURE_URL}" alt="A culinary task branches to a model choosing three ingredients and Epicure scoring all 56 legal choices before the benchmark aggregates the result.">
                  <div>
                    <h3>Scoring contract</h3>
                    <p>A task score ranges from 0 to 100 on its released Epicure map. The complete
                    release is a {MODEL_COUNT} by {TASK_COUNT} matrix with one valid response in every cell.</p>
                    <h3>Inference</h3>
                    <p>Results use {INDEPENDENT_CLUSTER_COUNT:,} ingredient-anchor clusters,
                    50,000 shared cluster bootstraps, simultaneous score bands, 100,000 cluster
                    sign flips, Holm correction, exact tests against a random legal choice, bootstrap rank intervals,
                    and an independently compiled second panel.</p>
                    <h3>Training boundary</h3>
                    <p>The 342 optimizer-facing SFT, DPO, and GRPO maps use anchors that do not
                    occur in the 84-task transfer split or the {TASK_COUNT}-task leaderboard.
                    Training cannot query either evaluation map through the reward endpoint.</p>
                    <aside class="fb-evidence">
                      <strong>Content-addressed release</strong><br>
                      <span class="fb-hash">{BUNDLE["release_artifact_sha256"]}</span><br><br>
                      {MODEL_COUNT} endpoints<br>{TASK_COUNT} tasks<br>{INDEPENDENT_CLUSTER_COUNT:,} anchor clusters<br>{PRIMARY_COUNT:,} complete answers<br>{BUNDLE["analysis"]["resolved_pair_count"]}/{PAIR_COUNT} resolved pairs
                    </aside>
                  </div>
                </div>
                """
            )
            gr.Markdown(
                """
```bash
git clone https://github.com/josefchen/flavourbench.git
cd flavourbench
python -m pip install -e '.[dev]'
pytest -q tests/lab_cli_test.py tests/hf_lab_space_api_test.py
```
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
          <div class="fb-footer-main">
            <strong>FlavourBench</strong>
            <span>Josef Chen, Independent Researcher</span>
            <span>Erim Hayretci, Imperial College London</span>
          </div>
          <nav class="fb-footer-links" aria-label="Project resources">
            <a href="https://arxiv.org/abs/2608.20574">Paper</a>
            <a href="https://huggingface.co/datasets/josefchen/flavourbench">Dataset</a>
            <a href="https://github.com/josefchen/flavourbench">Source</a>
          </nav>
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(
        theme=theme,
        css=CSS,
        head=HEAD,
        allowed_paths=[str(HERE / "assets")],
    )
