#!/usr/bin/env python3
"""Verify the anonymous ICLR 2027 manuscript and its authoritative data bindings."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
STAGE = BUILD / "stage"
PDF = BUILD / "flavourbench-iclr2027-anonymous.pdf"
LOG = STAGE / "main.log"

OFFICIAL_HASHES = {
    "iclr2027_conference.sty": "797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6",
    "iclr2027_conference.bst": "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5",
    "natbib.sty": "88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d",
    "fancyhdr.sty": "b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea",
}
PRIVATE_HOME = "/" + "home/" + "remy-" + "simpc4"
FORBIDDEN = re.compile(
    r"josefchen|erim[ -]hayretci|imperial college london|"
    r"arxiv\.org/abs/2608\.20574|" + re.escape(PRIVATE_HOME),
    re.IGNORECASE,
)
SECRET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])sk-(?:or-v1|kimi|sp|ws)-[A-Za-z0-9._-]{12,}"
    r"|(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{28,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|\b[0-9a-f]{32}\.[A-Za-z0-9_-]{16,}\b"
    r"|BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"
)


class VerificationError(RuntimeError):
    """The ICLR package violates a submission or release invariant."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*command: str, cwd: Path = HERE) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def _page_text(page: int) -> str:
    return _run("pdftotext", "-f", str(page), "-l", str(page), str(PDF), "-")


def _find_page(phrase: str, pages: int) -> int:
    for page in range(1, pages + 1):
        if phrase in " ".join(_page_text(page).split()):
            return page
    raise VerificationError(f"PDF section not found: {phrase}")


def _verify_template_and_source() -> None:
    for name, expected in OFFICIAL_HASHES.items():
        source = HERE / name
        staged = STAGE / name
        if source.is_symlink() or not source.is_file() or _sha(source) != expected:
            raise VerificationError(f"official template hash differs: {name}")
        if staged.is_symlink() or not staged.is_file() or _sha(staged) != expected:
            raise VerificationError(f"staged template hash differs: {name}")
    if _sha(HERE / "main.tex") != _sha(STAGE / "main.tex"):
        raise VerificationError("staged manuscript differs from source")
    if _sha(ROOT / "paper/references.bib") != _sha(STAGE / "references.bib"):
        raise VerificationError("staged bibliography differs from canonical bibliography")

    source = (HERE / "main.tex").read_text(encoding="utf-8")
    if re.search(r"(?m)^[ \t]*\\iclrfinalcopy\b", source):
        raise VerificationError("camera-ready switch is active")
    if "\\usepackage{iclr2027_conference,times}" not in source:
        raise VerificationError("official ICLR 2027 style is not loaded")
    if re.search(r"\\usepackage(?:\[[^]]*\])?\{geometry\}|\\textwidth|\\textheight", source):
        raise VerificationError("manuscript overrides official page geometry")
    if "\\subsection*{AI use statement}" not in source:
        raise VerificationError("mandatory AI-use statement is missing")
    if re.search(r"\b(?:TODO|TBD|PLACEHOLDER|FIXME)\b|\[tasks with", source, re.IGNORECASE):
        raise VerificationError("submission source contains a placeholder")
    if FORBIDDEN.search(source) or SECRET_PATTERN.search(source):
        raise VerificationError("submission source contains identity or credential material")


def _verify_pdf() -> tuple[int, int, int]:
    if PDF.is_symlink() or not PDF.is_file():
        raise VerificationError("anonymous PDF is missing")
    info = _run("pdfinfo", str(PDF))
    fields = {
        key.strip(): value.strip()
        for line in info.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    pages = int(fields.get("Pages", "0"))
    if fields.get("Author") != "Anonymous Authors":
        raise VerificationError("PDF author metadata is not anonymous")
    if fields.get("Page size") != "612 x 792 pts (letter)":
        raise VerificationError("PDF is not US Letter")
    if not (12 <= pages <= 30) or PDF.stat().st_size >= 50 * 1024 * 1024:
        raise VerificationError("PDF page or file-size envelope differs")

    text = _run("pdftotext", str(PDF), "-")
    normalized = " ".join(text.split())
    if FORBIDDEN.search(normalized) or SECRET_PATTERN.search(normalized):
        raise VerificationError("anonymous PDF contains identity or credential material")
    for phrase in (
        "An executable evaluator gives a benchmark a stable object to measure",
        "27 frontier endpoints",
        "534 substitution, pairing, and constraint tasks",
        "14418 complete model–task observations",
        "does not identify a unique best endpoint",
        "13.30 points over a format- and label-matched control",
        "The replication gain is 11.73 points",
        "AI USE STATEMENT",
        "The anonymous supplement contains the exact task maps",
    ):
        if phrase.lower() not in normalized.lower():
            raise VerificationError(f"PDF is missing required content: {phrase}")

    conclusion_page = _find_page(
        "evaluates open-ended culinary decisions without asking one language model", pages
    )
    ai_page = _find_page("Generative AI tools assisted with methodology feedback", pages)
    appendix_page = _find_page("Complete leaderboard on the common 534-task core", pages)
    if conclusion_page > 9 or ai_page != conclusion_page + 1:
        raise VerificationError("main text exceeds the strict nine-page limit")
    if appendix_page <= ai_page:
        raise VerificationError("appendix does not follow disclosures and references")

    fonts = _run("pdffonts", str(PDF)).splitlines()[2:]
    font_pattern = re.compile(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$")
    font_flags = [font_pattern.search(line) for line in fonts]
    if not fonts or any(match is None or match.group(1) != "yes" for match in font_flags):
        raise VerificationError("PDF contains an unembedded font")
    return pages, conclusion_page, ai_page


def _verify_log() -> None:
    if LOG.is_symlink() or not LOG.is_file():
        raise VerificationError("LaTeX log is missing")
    log = LOG.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"Overfull \\hbox",
        r"Overfull \\vbox",
        r"undefined references",
        r"Citation .* undefined",
        r"There were undefined citations",
        r"Label\(s\) may have changed",
    )
    if any(re.search(pattern, log, re.IGNORECASE) for pattern in patterns):
        raise VerificationError("LaTeX log contains overflow or unresolved references")


def _verify_release() -> dict[str, object]:
    releases = sorted(
        (ROOT / "paper/generated/complete-core").glob("flavourbench-complete-core-release-*.json")
    )
    if len(releases) != 1:
        raise VerificationError("expected one content-addressed complete-core release")
    receipt = json.loads(
        _run(
            sys.executable,
            str(ROOT / "paper/verify_complete_core_release.py"),
            "--release",
            str(releases[0]),
            cwd=ROOT,
        )
    )
    if receipt.get("status") != "PASS" or receipt.get("models") != 27:
        raise VerificationError("authoritative complete-core release failed verification")
    dataset_receipt = _run(
        sys.executable,
        str(ROOT / "hf/dataset/verify_complete_core_dataset.py"),
        "--dataset-directory",
        str(ROOT / "hf/dataset/data-complete-core"),
        cwd=ROOT,
    )
    if not dataset_receipt.startswith("OK: 27 models, 534 tasks, 14,418 complete responses"):
        raise VerificationError("anonymous-dataset source failed verification")
    return receipt


def _verify_reward_transfer() -> dict[str, object]:
    receipt = json.loads(
        _run(
            sys.executable,
            str(ROOT / "experiments/reward_transfer/verify_release.py"),
            cwd=ROOT,
        )
    )
    primary = receipt.get("primary")
    public = receipt.get("public")
    if (
        receipt.get("status") != "PASS"
        or not isinstance(primary, dict)
        or not isinstance(public, dict)
        or primary.get("tasks") != 84
        or public.get("tasks") != 534
        or abs(float(primary.get("effect_points", 0)) - 13.300238095238095) > 1e-12
        or abs(float(public.get("effect_points", 0)) - 11.728370786516855) > 1e-12
    ):
        raise VerificationError("reward-transfer release failed reconstruction")
    return receipt


def main() -> None:
    _verify_template_and_source()
    pages, conclusion_page, ai_page = _verify_pdf()
    _verify_log()
    release = _verify_release()
    reward_transfer = _verify_reward_transfer()
    print(
        json.dumps(
            {
                "ai_statement_page": ai_page,
                "conclusion_page": conclusion_page,
                "main_text_pages": conclusion_page,
                "models": release["models"],
                "pairwise_rows": release["pairwise_rows"],
                "pdf_pages": pages,
                "reward_transfer_primary_effect": reward_transfer["primary"]["effect_points"],
                "reward_transfer_public_effect": reward_transfer["public"]["effect_points"],
                "status": "PASS",
                "tasks_per_model": release["tasks_per_model"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
