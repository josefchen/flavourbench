#!/usr/bin/env python3
"""Build deterministic anonymous ICLR source and supplementary archives."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
STAGE = BUILD / "stage"
PDF = BUILD / "flavourbench-iclr2027-anonymous.pdf"
SOURCE_ARCHIVE = BUILD / "flavourbench-iclr2027-anonymous-source.tar.gz"
SUPPLEMENT_ARCHIVE = BUILD / "flavourbench-iclr2027-anonymous-supplement.zip"
OUTER_MANIFEST = BUILD / "ICLR2027-MANIFEST.sha256"
SOURCE_DATE_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "1787616000"))
ZIP_TIME = time.gmtime(SOURCE_DATE_EPOCH)[:6]

PRIVATE_HOME = b"/" + b"home/" + b"remy-" + b"simpc4"
FORBIDDEN_IDENTITY = re.compile(
    rb"josefchen|erim[ -]hayretci|imperial college london|"
    rb"arxiv\.org/abs/2608\.20574|" + re.escape(PRIVATE_HOME),
    re.IGNORECASE,
)
SECRET_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9_])sk-(?:or-v1|kimi|sp|ws)-[A-Za-z0-9._-]{12,}"
    rb"|(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{28,}"
    rb"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    rb"|\b[0-9a-f]{32}\.[A-Za-z0-9_-]{16,}\b"
    rb"|BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"
)


class PackageError(RuntimeError):
    """The anonymous package is incomplete or unsafe."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan(path: Path) -> None:
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sample = overlap + chunk
            if FORBIDDEN_IDENTITY.search(sample):
                raise PackageError(f"identity leak in anonymous payload: {path}")
            if SECRET_PATTERN.search(sample):
                raise PackageError(f"credential-like value in anonymous payload: {path}")
            overlap = sample[-512:]


def _regular_files(directory: Path) -> list[Path]:
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise PackageError(f"symlink refused: {path}")
        if path.is_file():
            result.append(path)
    return result


def _anonymous_lab_module() -> Path:
    """Copy the lab SDK while neutralizing its unused public-repository default."""

    source = ROOT / "src/flavourbench/lab.py"
    marker = b'DEFAULT_DATASET_REPO = "josefchen/flavourbench"'
    replacement = b'DEFAULT_DATASET_REPO = "anonymous/flavourbench"'
    payload = source.read_bytes()
    if payload.count(marker) != 1:
        raise PackageError("lab SDK public-repository marker differs")
    destination = BUILD / "anonymous-support/lab.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload.replace(marker, replacement))
    return destination


def _source_payload() -> list[tuple[Path, str]]:
    if not STAGE.is_dir():
        raise PackageError("build stage is missing; run make verify first")
    allowed_top = {
        "main.tex",
        "references.bib",
        "iclr2027_conference.sty",
        "iclr2027_conference.bst",
        "natbib.sty",
        "fancyhdr.sty",
    }
    result: list[tuple[Path, str]] = []
    for name in sorted(allowed_top):
        path = STAGE / name
        if not path.is_file() or path.is_symlink():
            raise PackageError(f"source file missing: {path}")
        result.append((path, name))
    for directory in ("generated", "figures"):
        for path in _regular_files(STAGE / directory):
            result.append((path, path.relative_to(STAGE).as_posix()))
    result.extend(
        (
            (HERE / "source-package/README.md", "README.md"),
            (HERE / "source-package/Makefile", "Makefile"),
        )
    )
    return result


def _supplement_payload(source: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    result = [
        (HERE / "supplement/README.md", "README.md"),
        (HERE / "supplement/LICENSES.md", "LICENSES.md"),
        (
            HERE / "supplement/requirements-reward-transfer.txt",
            "requirements-reward-transfer.txt",
        ),
        (HERE / "supplement/rebuild_summary.py", "code/rebuild_summary.py"),
        (ROOT / "hf/dataset/verify_complete_core_dataset.py", "code/verify_dataset.py"),
        (
            ROOT / "paper/protocols/external_substitution_validation_v1.json",
            "protocols/external_substitution_validation_v1.json",
        ),
        (
            ROOT / "contracts/reward-transfer/reward-transfer-plan-v2.json",
            "contracts/reward-transfer/reward-transfer-plan-v2.json",
        ),
        (
            ROOT / "experiments/reward_transfer/data-audit.json",
            "experiments/reward_transfer/data-audit.json",
        ),
        *[
            (
                ROOT / f"experiments/reward_transfer/{name}",
                f"experiments/reward_transfer/{name}",
            )
            for name in (
                "analyze.py",
                "audit_data.py",
                "evaluate.py",
                "release_results.py",
                "train_sft.py",
                "unlock_evaluation.py",
                "verify_release.py",
            )
        ],
        *[
            (ROOT / f"src/flavourbench/{name}", f"src/flavourbench/{name}")
            for name in (
                "__init__.py",
                "reward_transfer.py",
                "selection_response_parser_v3.py",
            )
        ],
        (_anonymous_lab_module(), "src/flavourbench/lab.py"),
    ]
    for source_directory, target_directory in (
        (ROOT / "hf/dataset/data-complete-core", "data/complete-core"),
        (ROOT / "hf/dataset/data-analysis", "data/analysis"),
        (ROOT / "hf/dataset/data-lab", "data/lab"),
    ):
        for path in _regular_files(source_directory):
            result.append(
                (path, f"{target_directory}/{path.relative_to(source_directory).as_posix()}")
            )
    for path in _regular_files(ROOT / "hf/dataset/data-lab"):
        relative = path.relative_to(ROOT / "hf/dataset/data-lab").as_posix()
        result.append((path, f"hf/dataset/data-lab/{relative}"))
    result.append(
        (
            ROOT / "hf/dataset/data-complete-core/tasks.jsonl",
            "hf/dataset/data-complete-core/tasks.jsonl",
        )
    )
    for name in (
        "reward-transfer-evaluation-gate.json",
        "reward-transfer-primary-analysis.json",
        "reward-transfer-public-analysis.json",
        "reward-transfer-training-manifests.jsonl",
        "reward-transfer-evaluation-manifests.jsonl",
        "reward-transfer-primary-responses.jsonl",
        "reward-transfer-public-responses.jsonl",
        "reward-transfer-release-manifest.json",
    ):
        result.append(
            (
                ROOT / f"hf/dataset/data-analysis/{name}",
                f"hf/dataset/data-analysis/{name}",
            )
        )
    result.extend((path, f"paper-source/{name}") for path, name in source)
    return result


def _validate_payload(payload: list[tuple[Path, str]]) -> None:
    names = [name for _, name in payload]
    if len(names) != len(set(names)):
        raise PackageError("archive contains duplicate names")
    for path, _ in payload:
        if path.is_symlink() or not path.is_file():
            raise PackageError(f"payload path is not a regular file: {path}")
        _scan(path)


def _manifest_bytes(payload: list[tuple[Path, str]]) -> bytes:
    lines = [f"{_sha(path)}  {name}" for path, name in sorted(payload, key=lambda item: item[1])]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _tar_source(payload: list[tuple[Path, str]]) -> None:
    root_name = "flavourbench-iclr2027-anonymous-source"
    manifest = _manifest_bytes(payload)
    with SOURCE_ARCHIVE.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=SOURCE_DATE_EPOCH) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path, name in sorted(payload, key=lambda item: item[1]):
                    info = archive.gettarinfo(str(path), arcname=f"{root_name}/{name}")
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = SOURCE_DATE_EPOCH
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
                info = tarfile.TarInfo(f"{root_name}/MANIFEST.sha256")
                info.size = len(manifest)
                info.mtime = SOURCE_DATE_EPOCH
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(manifest))


def _zip_supplement(payload: list[tuple[Path, str]]) -> None:
    root_name = "flavourbench-iclr2027-anonymous-supplement"
    manifest = _manifest_bytes(payload)
    with zipfile.ZipFile(
        SUPPLEMENT_ARCHIVE,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path, name in sorted(payload, key=lambda item: item[1]):
            info = zipfile.ZipInfo(f"{root_name}/{name}", date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        info = zipfile.ZipInfo(f"{root_name}/MANIFEST.sha256", date_time=ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest)


def main() -> None:
    if not PDF.is_file() or PDF.is_symlink():
        raise PackageError("verified anonymous PDF is missing")
    source = _source_payload()
    supplement = _supplement_payload(source)
    _validate_payload(source)
    _validate_payload(supplement)
    _tar_source(source)
    _zip_supplement(supplement)
    files = [PDF, SOURCE_ARCHIVE, SUPPLEMENT_ARCHIVE]
    OUTER_MANIFEST.write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    print(
        f"OK: PDF={PDF.stat().st_size} bytes, source={SOURCE_ARCHIVE.stat().st_size} bytes, "
        f"supplement={SUPPLEMENT_ARCHIVE.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
