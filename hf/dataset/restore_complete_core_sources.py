from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_VERIFIER_PATH = Path(__file__).resolve().with_name("verify_complete_core_dataset.py")
_VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "flavourbench_complete_core_dataset_verifier", _VERIFIER_PATH
)
if _VERIFIER_SPEC is None or _VERIFIER_SPEC.loader is None:
    raise RuntimeError("complete-core dataset verifier could not be loaded")
_VERIFIER = importlib.util.module_from_spec(_VERIFIER_SPEC)
_VERIFIER_SPEC.loader.exec_module(_VERIFIER)
verify_dataset = _VERIFIER.verify_dataset


class CompleteCoreRestoreError(RuntimeError):
    """The selected source-response restoration failed."""


def _payload(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _destination(repository: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CompleteCoreRestoreError("response source path escapes the repository")
    if (
        not candidate.parts
        or candidate.parts[0] != "benchmark"
        or "responses" not in candidate.parts
        or "primary" not in candidate.parts
        or not candidate.name.startswith("response-")
        or candidate.suffix != ".json"
    ):
        raise CompleteCoreRestoreError("response source path is outside the allowed namespace")
    destination = (repository / candidate).resolve()
    try:
        destination.relative_to(repository.resolve())
    except ValueError as exc:
        raise CompleteCoreRestoreError("response destination escapes the repository") from exc
    return destination


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore selected FlavourBench response sources")
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--restore", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    verify_dataset(args.dataset_directory)
    restored = 0
    for line in (args.dataset_directory / "primary_observations.jsonl").read_bytes().splitlines():
        row = json.loads(line)
        document = row["response"]
        artifact = str(document["artifact_sha256"])
        expected_name = f"response-{document['cell_id']}-{artifact}.json"
        destination = _destination(args.repository, str(row["source_path"]))
        if destination.name != expected_name:
            raise CompleteCoreRestoreError("response filename is not content addressed")
        payload = _payload(document)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file():
                raise CompleteCoreRestoreError(f"response destination is unsafe: {destination}")
            if json.loads(destination.read_text(encoding="utf-8")) != document:
                raise CompleteCoreRestoreError(f"response destination conflicts: {destination}")
        elif args.check:
            raise CompleteCoreRestoreError(f"response source is absent: {destination}")
        else:
            _write_new(destination, payload)
            restored += 1
    action = "verified" if args.check else f"restored {restored} and verified"
    print(f"OK: {action} 14,418 selected source responses")


if __name__ == "__main__":
    main()
