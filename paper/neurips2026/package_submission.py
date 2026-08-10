"""Build the deterministic anonymous FlavourBench NeurIPS supplement."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
EVALUATION_ROOT = PAPER.parent.parent / "flavourbench"
BUILD = HERE / "build"
PDF = BUILD / "flavourbench-neurips2026.pdf"
EPOCH = 1_786_217_820
SECRET_PATTERN = re.compile(
    rb"(?:sk-(?:or-v1|kimi|sp|ws)-[A-Za-z0-9._-]{20,}|"
    rb"(?:AKIA|ASIA)[0-9A-Z]{16}|[0-9a-f]{32}\.[A-Za-z0-9_-]{16,}|"
    rb"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY|" + rb"/hom" + rb"e/|remy-" + rb"simpc4)"
)
AUTHOR_IDENTITY = b"Josef" + b" Chen"
BIBLIOGRAPHY_IDENTITY = b"Chen," + b" Josef"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _files() -> dict[Path, Path]:
    mapping = {
        HERE / "main.tex": Path("paper/flavourbench/neurips2026/main.tex"),
        HERE / "checklist-answers.tex": Path(
            "paper/flavourbench/neurips2026/checklist-answers.tex"
        ),
        HERE / "references.bib": Path("paper/flavourbench/neurips2026/references.bib"),
        HERE / "prepare_checklist.py": Path(
            "paper/flavourbench/neurips2026/prepare_checklist.py"
        ),
        HERE / "verify_submission.py": Path(
            "paper/flavourbench/neurips2026/verify_submission.py"
        ),
        HERE / "package_submission.py": Path(
            "paper/flavourbench/neurips2026/package_submission.py"
        ),
        HERE / "sync_from_arxiv.py": Path(
            "paper/flavourbench/neurips2026/sync_from_arxiv.py"
        ),
        HERE / "Makefile": Path("paper/flavourbench/neurips2026/Makefile"),
        HERE / "README.md": Path("paper/flavourbench/neurips2026/README.md"),
        HERE / "ANONYMOUS-LICENSES.md": Path(
            "paper/flavourbench/neurips2026/ANONYMOUS-LICENSES.md"
        ),
        HERE / "neurips_2026.sty": Path(
            "paper/flavourbench/neurips2026/neurips_2026.sty"
        ),
        HERE / "neurips_2026.tex": Path(
            "paper/flavourbench/neurips2026/neurips_2026.tex"
        ),
        HERE / "checklist.tex": Path("paper/flavourbench/neurips2026/checklist.tex"),
        PAPER / "main.tex": Path("paper/flavourbench/main.tex"),
        PAPER / "README.md": Path("paper/flavourbench/README.md"),
        PAPER / "reproduce_epicure_native.py": Path("reproduce_epicure_native.py"),
        PAPER / "build_epicure_native_assets.py": Path(
            "paper/flavourbench/build_epicure_native_assets.py"
        ),
        PAPER / "generated/epicure-native/epicure-native-release.json": Path(
            "generated/epicure-native/epicure-native-release.json"
        ),
        EVALUATION_ROOT / "src/flavourbench/__init__.py": Path(
            "flavourbench/src/flavourbench/__init__.py"
        ),
        EVALUATION_ROOT / "src/flavourbench/epicure_native_taskset.py": Path(
            "flavourbench/src/flavourbench/epicure_native_taskset.py"
        ),
        EVALUATION_ROOT / "src/flavourbench/epicure_native_leaderboard.py": Path(
            "flavourbench/src/flavourbench/epicure_native_leaderboard.py"
        ),
        EVALUATION_ROOT / "src/flavourbench/epicure_native_release.py": Path(
            "flavourbench/src/flavourbench/epicure_native_release.py"
        ),
        EVALUATION_ROOT / "src/flavourbench/frontier_manifest.py": Path(
            "flavourbench/src/flavourbench/frontier_manifest.py"
        ),
        EVALUATION_ROOT / "tests/epicure_native_taskset_test.py": Path(
            "flavourbench/tests/epicure_native_taskset_test.py"
        ),
        EVALUATION_ROOT / "pyproject.toml": Path("flavourbench/pyproject.toml"),
        EVALUATION_ROOT / "requirements.lock": Path("flavourbench/requirements.lock"),
    }
    for path in sorted((PAPER / "generated/epicure-native").glob("epicure-native-*")):
        if path.is_file() and path.name != "epicure-native-release.json":
            mapping[path] = (
                Path("paper/flavourbench/generated/epicure-native") / path.name
            )
    for path in sorted((PAPER / "figures/epicure-native").glob("frontier-*.pdf")):
        if path.is_file():
            mapping[path] = (
                Path("paper/flavourbench/figures/epicure-native") / path.name
            )
    return mapping


def _safe_write(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"conflicting generated artifact: {path}")
        return
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _source_manifest(entries: dict[Path, bytes]) -> bytes:
    return "".join(
        f"{_sha(payload)}  {path.as_posix()}\n"
        for path, payload in sorted(
            entries.items(), key=lambda item: item[0].as_posix()
        )
    ).encode()


def _zip(entries: dict[Path, bytes]) -> bytes:
    timestamp = datetime.fromtimestamp(EPOCH, tz=UTC).timetuple()[:6]
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path, payload in sorted(
            entries.items(), key=lambda item: item[0].as_posix()
        ):
            info = zipfile.ZipInfo(path.as_posix(), date_time=timestamp)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return output.getvalue()


def _verify(payload: bytes) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="flavourbench-neurips-check-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise RuntimeError("supplement contains duplicate members")
            archive.extractall(root)
        manifest = root / "SOURCE_MANIFEST.sha256"
        subprocess.run(
            ["sha256sum", "--check", str(manifest)],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        replay = subprocess.run(
            [
                sys.executable,
                "-I",
                "reproduce_epicure_native.py",
                "--release",
                "generated/epicure-native/epicure-native-release.json",
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        receipt = json.loads(replay.stdout)
        if receipt.get("status") != "verified" or receipt.get("models") != 20:
            raise RuntimeError("supplement replay failed")
        return receipt


def main() -> int:
    if PDF.is_symlink() or not PDF.is_file():
        raise RuntimeError("verified anonymous PDF is missing")
    mapping = _files()
    entries: dict[Path, bytes] = {}
    for source, destination in mapping.items():
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"supplement source missing: {source}")
        if destination in entries:
            raise RuntimeError(f"duplicate supplement destination: {destination}")
        payload = source.read_bytes()
        if source == PAPER / "main.tex":
            payload = payload.replace(
                b"pdfauthor={" + AUTHOR_IDENTITY + b"}", b"pdfauthor={Anonymous}"
            )
            payload = payload.replace(
                b"\\author{" + AUTHOR_IDENTITY + b"\\\\\\small Independent Researcher}",
                b"\\author{Anonymous}",
            )
        entries[destination] = payload
    entries[Path("paper/flavourbench/references.bib")] = (
        HERE / "references.bib"
    ).read_bytes()
    contaminated = [
        path.as_posix()
        for path, payload in entries.items()
        if SECRET_PATTERN.search(payload)
        or AUTHOR_IDENTITY in payload
        or BIBLIOGRAPHY_IDENTITY in payload
    ]
    if contaminated:
        raise RuntimeError(
            "supplement contains a local path or credential marker: "
            + ", ".join(contaminated)
        )
    entries[Path("SOURCE_MANIFEST.sha256")] = _source_manifest(entries)
    supplement = _zip(entries)
    receipt = _verify(supplement)

    BUILD.mkdir(parents=True, exist_ok=True)
    pdf_payload = PDF.read_bytes()
    pdf_sha = _sha(pdf_payload)
    supplement_sha = _sha(supplement)
    pdf_name = f"flavourbench-neurips2026-{pdf_sha}.pdf"
    supplement_name = f"flavourbench-neurips2026-supplement-{supplement_sha}.zip"
    _safe_write(BUILD / pdf_name, pdf_payload)
    _safe_write(BUILD / supplement_name, supplement)

    upload = {
        "schema_version": "flavourbench-neurips2026-upload-v2",
        "anonymous": True,
        "paper": {"name": pdf_name, "sha256": pdf_sha, "bytes": len(pdf_payload)},
        "supplement": {
            "name": supplement_name,
            "sha256": supplement_sha,
            "bytes": len(supplement),
        },
        "release_artifact_sha256": receipt["release_artifact_sha256"],
        "models": 20,
        "tasks": 32,
        "observations": 1_280,
    }
    upload_payload = (
        json.dumps(upload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    upload_sha = _sha(upload_payload)
    upload_name = f"flavourbench-neurips2026-upload-{upload_sha}.json"
    _safe_write(BUILD / upload_name, upload_payload)
    manifest = (
        f"{pdf_sha}  {pdf_name}\n"
        f"{supplement_sha}  {supplement_name}\n"
        f"{upload_sha}  {upload_name}\n"
    ).encode()
    (BUILD / "SUBMISSION-MANIFEST.sha256").write_bytes(manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "pdf": pdf_name,
                "supplement": supplement_name,
                "upload_manifest": upload_name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
