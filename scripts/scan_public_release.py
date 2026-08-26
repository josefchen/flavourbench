from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "local_home_path": re.compile(r"/" + r"home/|remy-" + r"simpc4"),
    "provider_key": re.compile(
        r"(?<![A-Za-z0-9_])sk-(?:or-v1|kimi|sp|ws)-[A-Za-z0-9._-]{12,}"
        r"|(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{28,}"
    ),
    "hugging_face_token": re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "compound_api_key": re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_-]{16,}\b"),
    "private_key_marker": re.compile("BEGIN (?:" + "RSA|OPENSSH|EC" + ") PRIVATE KEY"),
}
LOCAL_PATH_ASSERTION_FILES = {
    "tests/qwen_exploratory_release_projection_test.py",
    "tests/release_package_remediation_test.py",
}
SYNTHETIC_SECRET_FIXTURE_FILES = {
    "tests/research_release_test.py",
    "tests/season0_release_test.py",
}


def _candidate_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _allowed(path: str, line: str, category: str) -> bool:
    lowered = line.lower()
    if category == "local_home_path":
        return path in LOCAL_PATH_ASSERTION_FILES and "not in serialized" in lowered
    if category == "provider_key":
        return path.startswith("tests/") and (
            "not-real" in lowered or "fake" in lowered or "not-a-public-handle" in lowered
        )
    if category == "aws_access_key":
        return path in SYNTHETIC_SECRET_FIXTURE_FILES
    return False


def main() -> None:
    findings: list[tuple[str, int, str]] = []
    for absolute_path in _candidate_paths():
        relative = absolute_path.relative_to(ROOT).as_posix()
        try:
            text = absolute_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for category, pattern in PATTERNS.items():
                if pattern.search(line) and not _allowed(relative, line, category):
                    findings.append((relative, line_number, category))
    if findings:
        for path, line_number, category in findings:
            print(f"{path}:{line_number}: {category}")
        raise SystemExit(f"Public release refused: {len(findings)} potential private values")
    print(f"OK: scanned {len(_candidate_paths())} public files with no unapproved private values")


if __name__ == "__main__":
    main()
