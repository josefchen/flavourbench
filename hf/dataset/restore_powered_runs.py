"""Restore the clean powered response lineages from Hugging Face JSONL.

The public Git repository omits the large per-response directory trees.  The
Hugging Face dataset stores the same content-addressed response documents as two
JSONL tables.  This utility verifies those documents and recreates the exact
directory shape consumed by the frozen statistical analysis, without making any
network, model, or Epicure call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class PoweredRunRestoreError(RuntimeError):
    """The downloaded powered response tables failed closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _semantic_valid(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    stated = str(payload.pop("artifact_sha256", ""))
    return len(stated) == 64 and hashlib.sha256(_canonical(payload)).hexdigest() == stated


def _load_release(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoweredRunRestoreError(f"cannot read release: {path}") from exc
    if not isinstance(value, dict) or not _semantic_valid(value):
        raise PoweredRunRestoreError("release semantic verification failed")
    if value.get("status") != "final_complete":
        raise PoweredRunRestoreError("release is not final_complete")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PoweredRunRestoreError(f"response table is not a regular file: {path}")
    output: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict) or not _semantic_valid(value):
                    raise PoweredRunRestoreError(
                        f"response semantic verification failed at {path}:{line_number}"
                    )
                output.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoweredRunRestoreError(f"cannot parse response table: {path}") from exc
    return output


def _destination(
    *,
    row: Mapping[str, Any],
    base_models: set[str],
    deepseek_model: str | None,
    cohere_models: set[str],
    frontier_models: set[str],
    successor_models: set[str],
    base_run: Path,
    deepseek_run: Path | None,
    cohere_run: Path | None,
    frontier_run: Path | None,
    successor_run: Path | None,
) -> Path:
    model_id = str(row.get("model_id", ""))
    if model_id in base_models:
        root = base_run
    elif deepseek_model is not None and model_id == deepseek_model:
        if deepseek_run is None:
            raise PoweredRunRestoreError("DeepSeek run directory is missing")
        root = deepseek_run
    elif model_id in cohere_models:
        if cohere_run is None:
            raise PoweredRunRestoreError("Cohere run directory is missing")
        root = cohere_run
    elif model_id in frontier_models:
        if frontier_run is None:
            raise PoweredRunRestoreError("frontier run directory is missing")
        root = frontier_run
    elif model_id in successor_models:
        if successor_run is None:
            raise PoweredRunRestoreError("successor run directory is missing")
        root = successor_run
    else:
        raise PoweredRunRestoreError(f"response model is outside the final roster: {model_id}")
    panel = str(row.get("panel", ""))
    if panel not in {"primary", "repeat"}:
        raise PoweredRunRestoreError(f"unexpected response panel: {panel}")
    slot_id = str(row.get("slot_id", ""))
    cell_id = str(row.get("cell_id", ""))
    artifact = str(row.get("artifact_sha256", ""))
    if not slot_id or len(cell_id) != 64 or len(artifact) != 64:
        raise PoweredRunRestoreError("response identity is malformed")
    return root / "responses" / panel / slot_id / f"response-{cell_id}-{artifact}.json"


def _write_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return "existing"
        raise PoweredRunRestoreError(f"conflicting restored response: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise PoweredRunRestoreError(f"response appeared during restore: {path}") from exc
        path.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "created"


def restore(
    *,
    release_path: Path,
    primary_path: Path,
    repeat_path: Path,
    base_run: Path,
    deepseek_run: Path | None,
    cohere_run: Path | None,
    check: bool,
    frontier_run: Path | None = None,
    successor_run: Path | None = None,
) -> dict[str, Any]:
    release = _load_release(release_path)
    primary = _load_jsonl(primary_path)
    repeat = _load_jsonl(repeat_path)
    if any(row.get("panel") != "primary" for row in primary) or any(
        row.get("panel") != "repeat" for row in repeat
    ):
        raise PoweredRunRestoreError("response table panel assignment failed")

    lineage = release["inputs"]["model_response_sources"]
    if lineage.get("schema_version") == "flavourbench-single-fresh-response-source-v1":
        base_models = {str(value) for value in lineage["model_ids"]}
        cohere_models = set()
        frontier_models = set()
        successor_models = set()
        deepseek_model = None
        if len(base_models) != 26:
            raise PoweredRunRestoreError("v44 source lineage cardinality failed")
    elif "deepseek_model_ids" in lineage:
        base_models = {str(value) for value in lineage["base_model_ids"]}
        cohere_models = {str(value) for value in lineage["cohere_model_ids"]}
        frontier_models = {str(value) for value in lineage["frontier_model_ids"]}
        deepseek_models = {str(value) for value in lineage["deepseek_model_ids"]}
        successor_models = {str(value) for value in lineage["successor_model_ids"]}
        if (
            len(base_models) != 16
            or len(cohere_models) != 2
            or len(frontier_models) != 6
            or len(deepseek_models) != 1
            or len(successor_models) != 1
        ):
            raise PoweredRunRestoreError("v42 source lineage cardinality failed")
        deepseek_model = next(iter(deepseek_models))
    elif "frontier_model_ids" in lineage:
        base_models = {str(value) for value in lineage["base_model_ids"]}
        cohere_models = {str(value) for value in lineage["cohere_model_ids"]}
        frontier_models = {str(value) for value in lineage["frontier_model_ids"]}
        successor_models = {str(value) for value in lineage["successor_model_ids"]}
        deepseek_model = None
        if (
            len(base_models) != 16
            or len(cohere_models) != 2
            or len(frontier_models) != 7
            or len(successor_models) != 1
        ):
            raise PoweredRunRestoreError("v39 source lineage cardinality failed")
    elif "successor_model_ids" in lineage:
        base_models = {str(value) for value in lineage["base_model_ids"]}
        cohere_models = {str(value) for value in lineage["cohere_model_ids"]}
        frontier_models = set()
        successor_models = {str(value) for value in lineage["successor_model_ids"]}
        deepseek_model = None
        if len(base_models) != 16 or len(cohere_models) != 2 or len(successor_models) != 8:
            raise PoweredRunRestoreError("v38 source lineage cardinality failed")
    else:
        base_models = {str(value) for value in lineage["base_models"]}
        deepseek_model = str(lineage["deepseek_model_id"])
        cohere_models = {str(value) for value in lineage["cohere_model_ids"]}
        frontier_models = set()
        successor_models = set()
        if len(base_models) != 17 or len(cohere_models) != 2:
            raise PoweredRunRestoreError("release source lineage cardinality failed")

    roster = base_models | cohere_models | frontier_models | successor_models
    if deepseek_model is not None:
        roster.add(deepseek_model)
    expected_primary = len(roster) * 640
    expected_repeat = len(roster) * 64
    if len(primary) != expected_primary or len(repeat) != expected_repeat:
        raise PoweredRunRestoreError("downloaded response table cardinality failed")

    counts = Counter((str(row["model_id"]), str(row["panel"])) for row in primary + repeat)
    expected = Counter(
        {
            **{(model_id, "primary"): 640 for model_id in roster},
            **{(model_id, "repeat"): 64 for model_id in roster},
        }
    )
    if counts != expected:
        raise PoweredRunRestoreError("response model/panel grid is incomplete")
    identities = {
        (str(row["panel"]), str(row["model_id"]), str(row["task_id"])) for row in primary + repeat
    }
    if len(identities) != expected_primary + expected_repeat:
        raise PoweredRunRestoreError("response task identities are not unique")

    outcomes = Counter()
    for row in primary + repeat:
        destination = _destination(
            row=row,
            base_models=base_models,
            deepseek_model=deepseek_model,
            cohere_models=cohere_models,
            frontier_models=frontier_models,
            successor_models=successor_models,
            base_run=base_run,
            deepseek_run=deepseek_run,
            cohere_run=cohere_run,
            frontier_run=frontier_run,
            successor_run=successor_run,
        )
        payload = (json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        if check:
            if destination.is_symlink() or not destination.is_file():
                raise PoweredRunRestoreError(f"restored response is missing: {destination}")
            if destination.read_bytes() != payload:
                raise PoweredRunRestoreError(f"restored response differs: {destination}")
            outcomes["verified"] += 1
        else:
            outcomes[_write_no_replace(destination, payload)] += 1
    return {
        "status": "verified" if check else "restored",
        "primary_responses": len(primary),
        "repeat_responses": len(repeat),
        "models": len(roster),
        "files": dict(sorted(outcomes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--deepseek-run", type=Path)
    parser.add_argument("--cohere-run", type=Path)
    parser.add_argument("--frontier-run", type=Path)
    parser.add_argument("--successor-run", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            restore(
                release_path=args.release,
                primary_path=args.primary,
                repeat_path=args.repeat,
                base_run=args.base_run,
                deepseek_run=args.deepseek_run,
                cohere_run=args.cohere_run,
                frontier_run=args.frontier_run,
                successor_run=args.successor_run,
                check=args.check,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
