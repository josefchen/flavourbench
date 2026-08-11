"""Verify the anonymous Epicure-native NeurIPS submission."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
BUILD = HERE / "build"
PDF = BUILD / "flavourbench-neurips2026.pdf"
LOG = BUILD / "main.log"
REFERENCES = HERE / "references.bib"
RELEASE = PAPER / "generated/epicure-native/epicure-native-release.json"
REPLAY = PAPER / "reproduce_epicure_native.py"

EXPECTED_ASSETS = (
    PAPER / "generated/epicure-native/epicure-native-macros.tex",
    PAPER / "generated/epicure-native/epicure-native-leaderboard-table.tex",
    PAPER / "generated/epicure-native/epicure-native-route-table.tex",
    PAPER / "generated/epicure-native/epicure-native-family-table.tex",
    PAPER / "generated/epicure-native/epicure-native-case-studies.tex",
    PAPER / "figures/epicure-native/frontier-score-forest.pdf",
    PAPER / "figures/epicure-native/frontier-score-dumbbell.pdf",
    PAPER / "figures/epicure-native/frontier-family-heatmap.pdf",
    PAPER / "figures/epicure-native/frontier-paired-outcome-matrix.pdf",
    PAPER / "figures/epicure-native/frontier-latency-uplift.pdf",
    PAPER / "figures/epicure-native/frontier-social-summary.pdf",
)

SECRET_PATTERN = re.compile(
    r"(?:sk-(?:or-v1|kimi|sp|ws)-[A-Za-z0-9._-]{20,}|"
    r"(?:AKIA|ASIA)[0-9A-Z]{16}|[0-9a-f]{32}\.[A-Za-z0-9_-]{16,}|"
    r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY)"
)
AUTHOR_IDENTITY = "Josef" + " Chen"
BIBLIOGRAPHY_IDENTITY = "Chen," + " Josef"


class VerificationError(RuntimeError):
    """The anonymous package does not match the benchmark release."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON number: {value}")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _read_release() -> dict[str, Any]:
    value = json.loads(
        RELEASE.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_constant,
    )
    if not isinstance(value, dict):
        raise VerificationError("release root is not an object")
    payload = dict(value)
    recorded = str(payload.pop("artifact_sha256", ""))
    if hashlib.sha256(_canonical(payload)).hexdigest() != recorded:
        raise VerificationError("release content address does not verify")
    return value


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=HERE,
        check=True,
        text=True,
        capture_output=True,
    )


def main() -> int:
    for path in (PDF, LOG, REFERENCES, RELEASE, REPLAY, *EXPECTED_ASSETS):
        if path.is_symlink() or not path.is_file():
            raise VerificationError(f"required regular file missing: {path}")

    release = _read_release()
    leaderboard = release.get("leaderboard") or {}
    models = release.get("models") or []
    tasks = release.get("tasks") or []
    observations = release.get("observations") or []
    ranked = leaderboard.get("models") or []
    model_ids = {str(model.get("model_id") or "") for model in models}
    models_by_id = {str(model.get("model_id") or ""): model for model in models}
    required_models = {
        "moonshotai/kimi-k3",
        "qwen/qwen3.8-max",
        "cohere/command-a-plus-05-2026",
        "cohere/command-a-reasoning-08-2025",
    }
    if not (
        release.get("release_status") == "complete_public_automated_leaderboard"
        and release.get("human_judgments") == 0
        and len(models) == 20
        and len(tasks) == 32
        and len(observations) == 1_280
        and leaderboard.get("status") == "complete_automated_leaderboard"
        and leaderboard.get("official_track") is True
        and [row.get("rank") for row in ranked] == list(range(1, 21))
        and required_models <= model_ids
    ):
        raise VerificationError("release does not contain the complete ranked 20-model panel")
    route_expectations = {
        "moonshotai/kimi-k3": ("kimi_direct", "kimi-code-direct", "k3", 64),
        "cohere/command-a-plus-05-2026": (
            "cohere_direct",
            "cohere-direct",
            "command-a-plus-05-2026",
            64,
        ),
        "cohere/command-a-reasoning-08-2025": (
            "cohere_direct",
            "cohere-direct",
            "command-a-reasoning-08-2025",
            64,
        ),
    }
    for model_id, (
        backend,
        provider,
        returned_id,
        expected_arms,
    ) in route_expectations.items():
        observed = [
            row
            for row in observations
            if row.get("model_id") == model_id and row.get("response_artifact_sha256") is not None
        ]
        if (
            models_by_id[model_id].get("execution_backend") != backend
            or len(observed) != expected_arms
            or {row.get("actual_provider") for row in observed} != {provider}
            or {row.get("actual_model_id") for row in observed} != {returned_id}
        ):
            raise VerificationError(f"direct route fidelity failed: {model_id}")

    routed_expectations = {
        "qwen/qwen3.8-max": (
            "Alibaba",
            "qwen/qwen3.8-max-20260803",
            60,
        ),
        "z-ai/glm-5.2": ("CoreWeave", "z-ai/glm-5.2-20260616", 63),
    }
    for model_id, (provider, returned_id, expected_arms) in routed_expectations.items():
        observed = [
            row
            for row in observations
            if row.get("model_id") == model_id and row.get("response_artifact_sha256") is not None
        ]
        if (
            models_by_id[model_id].get("execution_backend") != "openrouter"
            or len(observed) != expected_arms
            or {row.get("actual_provider") for row in observed} != {provider}
            or {row.get("actual_model_id") for row in observed} != {returned_id}
        ):
            raise VerificationError(f"routed identity fidelity failed: {model_id}")

    _run(sys.executable, "sync_from_arxiv.py", "--check")
    replay = subprocess.run(
        [sys.executable, "-I", str(REPLAY), "--release", str(RELEASE)],
        cwd=PAPER,
        check=True,
        text=True,
        capture_output=True,
    )
    receipt = json.loads(replay.stdout)
    if receipt.get("status") != "verified" or receipt.get("models") != 20:
        raise VerificationError("offline replay receipt is not complete")

    text = _run("pdftotext", str(PDF), "-").stdout
    if "Anonymous Authors" in text or AUTHOR_IDENTITY in text:
        raise VerificationError("anonymous PDF contains an author identity")
    normalized_text = " ".join(text.split())
    for phrase in (
        "An Executable Benchmark for Culinary Reasoning Without a Model Judge",
        "The FlavourBench Score leaderboard",
        "Using the score as a training signal",
        "Command A Plus",
        "Command A Reasoning",
        "Qwen 3.8 Max",
        "Kimi K3",
    ):
        if phrase not in normalized_text:
            raise VerificationError(f"anonymous PDF is missing: {phrase}")

    appendix_page = None
    for page in range(1, 16):
        page_text = _run(
            "pdftotext",
            "-f",
            str(page),
            "-l",
            str(page),
            str(PDF),
            "-",
        ).stdout
        if "Real prompts, answers, and tool calls" in page_text:
            appendix_page = page
            break
    if appendix_page is None or appendix_page > 10:
        raise VerificationError("NeurIPS main paper exceeds the nine-page content limit")

    log = LOG.read_text(encoding="utf-8", errors="replace")
    if re.search(r"undefined references|Citation .* undefined|Overfull \\hbox", log):
        raise VerificationError("LaTeX log contains unresolved references or overflow")
    checklist = (HERE / "checklist-answers.tex").read_text(encoding="utf-8")
    if r"\answerTODO" in checklist or r"\justificationTODO" in checklist:
        raise VerificationError("NeurIPS checklist still contains TODOs")

    for path in (
        HERE / "main.tex",
        REFERENCES,
        HERE / "checklist-answers.tex",
        PAPER / "main.tex",
        PAPER / "README.md",
        REPLAY,
        RELEASE,
    ):
        if SECRET_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
            raise VerificationError(f"credential-like text found in release file: {path}")
    references_text = REFERENCES.read_text(encoding="utf-8")
    if AUTHOR_IDENTITY in references_text or BIBLIOGRAPHY_IDENTITY in references_text:
        raise VerificationError("anonymous bibliography contains an author identity")

    print(
        json.dumps(
            {
                "status": "PASS",
                "models": 20,
                "tasks": 32,
                "observations": 1_280,
                "content_pages": appendix_page - 1,
                "release_artifact_sha256": release["artifact_sha256"],
                "top_model": ranked[0]["model_id"],
                "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
