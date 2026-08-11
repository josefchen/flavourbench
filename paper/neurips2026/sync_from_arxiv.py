#!/usr/bin/env python3
"""Render the anonymous NeurIPS manuscript from the canonical paper body."""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent / "main.tex"
TARGET = HERE / "main.tex"
CANONICAL_REFERENCES = HERE.parent / "references.bib"
TARGET_REFERENCES = HERE / "references.bib"


PREAMBLE = r"""\documentclass{article}

\PassOptionsToPackage{numbers,sort&compress}{natbib}
\usepackage[eandd]{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{xurl}
\usepackage{seqsplit}
\usepackage{hyperref}
\usepackage[capitalise,noabbrev]{cleveref}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}

\definecolor{FBBlue}{HTML}{1769AA}
\definecolor{FBGold}{HTML}{E6A11A}
\definecolor{FBTeal}{HTML}{168C7A}
\definecolor{FBRed}{HTML}{C75450}
\definecolor{FBCharcoal}{HTML}{262B33}
\definecolor{FBPaper}{HTML}{F7F8FA}
\hypersetup{
  colorlinks=true,
  linkcolor=FBBlue,
  citecolor=FBBlue,
  urlcolor=FBBlue,
  pdftitle={FlavourBench: What Language Models Know and What Epicure Adds},
  pdfauthor={Anonymous Authors},
  pdfsubject={An executable culinary benchmark with a matched Epicure intervention},
  pdfkeywords={language models, benchmarks, tool use, culinary reasoning, executable evaluation}
}
\setlist{leftmargin=*,nosep}

\newcommand{\system}{\textsc{FlavourBench}}
\newcommand{\epicure}{\textsc{Epicure}}
\newcommand{\sha}[1]{\texttt{\seqsplit{#1}}}
\input{../generated/epicure-native/epicure-native-macros.tex}

\title{\system{}: What Language Models Know and What Epicure Adds}
\author{Anonymous Authors}

\begin{document}
\raggedbottom
\maketitle
"""


ENDING = r"""
\clearpage
\label{references-start}
{\small
\bibliographystyle{plainnat}
\bibliography{references}
}

\clearpage
\input{checklist-answers.tex}

\end{document}
"""


def _pop_figure(body: str, label: str) -> tuple[str, str]:
    label_position = body.index(rf"\label{{{label}}}")
    start = body.rfind(r"\begin{figure*}", 0, label_position)
    if start < 0:
        raise RuntimeError(f"figure start missing: {label}")
    end = body.index(r"\end{figure*}", label_position) + len(r"\end{figure*}")
    return body[:start] + body[end:], body[start:end]


def render() -> str:
    source = CANONICAL.read_text(encoding="utf-8")
    abstract_start = source.index(r"\begin{abstract}")
    abstract_end = source.index(r"\end{abstract}", abstract_start) + len(r"\end{abstract}")
    abstract = source[abstract_start:abstract_end]
    body_start = source.index(r"\section{Introduction}")
    bibliography_start = source.index(r"\bibliographystyle", body_start)
    body_end = source.rfind(r"\clearpage", body_start, bibliography_start)
    if body_end < 0:
        body_end = bibliography_start
    body = source[body_start:body_end]

    case_start = body.index(r"\section{Real prompts, answers, and tool calls}")
    case_end = body.index(r"\section{Using the score as a training signal}")
    case_section = body[case_start:case_end].strip()
    body = body[:case_start] + body[case_end:]

    appendix_figures: list[str] = []
    for label in ("fig:heatmap", "fig:matrix", "fig:latency"):
        body, figure = _pop_figure(body, label)
        appendix_figures.append(figure.strip())

    appendix_marker = r"\appendix"
    appendix_position = body.index(appendix_marker) + len(appendix_marker)
    appendix_material = (
        "\n\n"
        + case_section
        + "\n\n\\section{Additional result figures}\n\n"
        + "\n\n".join(appendix_figures)
    )
    body = body[:appendix_position] + appendix_material + body[appendix_position:]

    body = body.replace(
        r"{generated/operational-analysis/",
        r"{../generated/operational-analysis/",
    )
    body = body.replace(
        r"{generated/epicure-native/",
        r"{../generated/epicure-native/",
    )
    body = body.replace(
        r"{figures/epicure-native/",
        r"{../figures/epicure-native/",
    )
    return PREAMBLE + "\n" + abstract + "\n\n" + body.rstrip() + "\n" + ENDING


def render_references() -> str:
    source = CANONICAL_REFERENCES.read_text(encoding="utf-8")
    identity = "author       = {Radzikowski, Jakub and Chen," + " Josef},"
    if source.count(identity) != 2:
        raise RuntimeError("expected exactly two Epicure self-citations")
    return source.replace(identity, "author       = {{Anonymous}},")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    expected_references = render_references()
    if args.check:
        if (
            TARGET.read_text(encoding="utf-8") != expected
            or TARGET_REFERENCES.read_text(encoding="utf-8") != expected_references
        ):
            raise SystemExit("anonymous manuscript drifted from the canonical paper")
    else:
        TARGET.write_text(expected, encoding="utf-8")
        TARGET_REFERENCES.write_text(expected_references, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
