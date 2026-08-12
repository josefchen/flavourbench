from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flavourbench.epicure_selection_powered_analysis import load_panel
from flavourbench.epicure_selection_powered_plan import verify_repeat_panel
from flavourbench.epicure_selection_powered_plan_v31 import verify_plan as verify_v31_plan
from flavourbench.epicure_selection_powered_plan_v33 import verify_plan as verify_v33_plan
from flavourbench.epicure_selection_powered_plan_v35 import verify_plan as verify_v35_plan
from flavourbench.epicure_selection_taskset_v1 import verify_taskset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "data-powered"
TABLE_ORDER = (
    "models",
    "tasks",
    "primary_observations",
    "repeat_observations",
    "leaderboard",
    "pairwise_comparisons",
)


class PoweredDatasetBuildError(RuntimeError):
    """The powered Hugging Face export failed closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _semantic_valid(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == hashlib.sha256(_canonical(payload)).hexdigest())


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PoweredDatasetBuildError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PoweredDatasetBuildError(f"input is not a JSON object: {path}")
    return value


def _physical(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_pin(
    *,
    document: Mapping[str, Any],
    path: Path,
    pin: Mapping[str, Any],
    verifier: Any,
    label: str,
) -> None:
    if (
        not verifier(document)
        or document.get("artifact_sha256") != pin.get("semantic_sha256")
        or _physical(path) != pin.get("physical_sha256")
    ):
        raise PoweredDatasetBuildError(f"{label} plan binding failed")


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _response_documents(
    *,
    panel: str,
    final_plan: Mapping[str, Any],
    task_ids: Sequence[str],
    source_directories: Mapping[str, Path],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    expected_tasks = set(task_ids)
    for roster_row in final_plan["roster"]["models"]:
        model_id = str(roster_row["model_id"])
        directory = source_directories[model_id]
        paths = sorted(
            (directory / "responses" / panel / str(roster_row["slot_id"])).glob("response-*.json")
        )
        rows = [_load(path) for path in paths]
        if (
            len(rows) != len(task_ids)
            or {str(row.get("task_id")) for row in rows} != expected_tasks
            or any(str(row.get("model_id")) != model_id for row in rows)
            or any(not _semantic_valid(row) for row in rows)
        ):
            raise PoweredDatasetBuildError(f"{panel} response block failed for {model_id}")
        output.extend(sorted(rows, key=lambda row: str(row["task_id"])))
    return output


def _tables(
    *,
    release: Mapping[str, Any],
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    plan: Mapping[str, Any],
    primary_documents: Sequence[Mapping[str, Any]],
    repeat_documents: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    analysis_by_model = {str(row["model_id"]): row for row in release["analysis"]["models"]}
    source_lineage = release["inputs"]["model_response_sources"]
    base_models = set(source_lineage["base_models"])
    deepseek_model = str(source_lineage["deepseek_model_id"])
    cohere_models = set(source_lineage["cohere_model_ids"])
    model_rows = []
    for route in plan["roster"]["models"]:
        model_id = str(route["model_id"])
        if model_id in base_models:
            source = "powered-v31-base"
        elif model_id == deepseek_model:
            source = "powered-v33-clean-deepseek"
        elif model_id in cohere_models:
            source = "powered-v35-clean-cohere"
        else:
            raise PoweredDatasetBuildError(f"model has no response lineage: {model_id}")
        model_rows.append(
            {
                **analysis_by_model[model_id],
                "canonical_model_slug": route["canonical_model_slug"],
                "execution_backend": route["execution_backend"],
                "provider_tag": route["provider_tag"],
                "provider_name": route["provider_name"],
                "endpoint_execution_sha256": route["endpoint_execution_sha256"],
                "response_source": source,
            }
        )
    leaderboard = sorted(
        model_rows,
        key=lambda row: (
            row["point_estimate_rank"] is None,
            row["point_estimate_rank"] or 10_000,
            row["model_id"],
        ),
    )
    return {
        "models": model_rows,
        "tasks": list(taskset["tasks"]),
        "primary_observations": [dict(row) for row in primary_documents],
        "repeat_observations": [dict(row) for row in repeat_documents],
        "leaderboard": leaderboard,
        "pairwise_comparisons": list(release["analysis"]["pairwise_comparisons"]),
    }


def _expected_files(
    *,
    release: Mapping[str, Any],
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    plan: Mapping[str, Any],
    primary_documents: Sequence[Mapping[str, Any]],
    repeat_documents: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    tables = _tables(
        release=release,
        taskset=taskset,
        repeat_panel=repeat_panel,
        plan=plan,
        primary_documents=primary_documents,
        repeat_documents=repeat_documents,
    )
    files = {f"{name}.jsonl": _jsonl(tables[name]) for name in TABLE_ORDER}
    manifest: dict[str, Any] = {
        "schema_version": "flavourbench-hf-powered-dataset-manifest-v1",
        "release_artifact_sha256": release["artifact_sha256"],
        "plan_artifact_sha256": plan["artifact_sha256"],
        "taskset_artifact_sha256": taskset["artifact_sha256"],
        "repeat_panel_artifact_sha256": repeat_panel["artifact_sha256"],
        "files": [
            {
                "name": name,
                "rows": len(tables[name.removesuffix(".jsonl")]),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in files.items()
        ],
    }
    manifest["artifact_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    files["DATA_MANIFEST.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return files


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build(
    *,
    release_path: Path,
    taskset_path: Path,
    repeat_panel_path: Path,
    plan_path: Path,
    base_plan_path: Path,
    deepseek_plan_path: Path,
    base_run: Path,
    deepseek_run: Path,
    cohere_run: Path,
    output: Path,
    check: bool,
) -> None:
    release = _load(release_path)
    taskset = _load(taskset_path)
    repeat_panel = _load(repeat_panel_path)
    plan = _load(plan_path)
    base_plan = _load(base_plan_path)
    deepseek_plan = _load(deepseek_plan_path)
    if (
        not _semantic_valid(release)
        or release.get("status") != "final_complete"
        or not verify_taskset(taskset)
        or not verify_repeat_panel(repeat_panel, taskset=taskset)
        or not verify_v35_plan(plan)
    ):
        raise PoweredDatasetBuildError("powered release inputs failed verification")
    _verify_pin(
        document=base_plan,
        path=base_plan_path,
        pin=plan["inputs"]["plan_v31_predecessor"],
        verifier=verify_v31_plan,
        label="base",
    )
    _verify_pin(
        document=deepseek_plan,
        path=deepseek_plan_path,
        pin=plan["inputs"]["plan_v33_predecessor"],
        verifier=verify_v33_plan,
        label="DeepSeek",
    )
    deepseek_model = str(plan["execution"]["deepseek_route_recovery"]["model_id"])
    cohere_models = tuple(
        str(value) for value in plan["execution"]["cohere_route_successor"]["successor_model_ids"]
    )
    source_directories = {str(row["model_id"]): base_run for row in plan["roster"]["models"]}
    source_plans: dict[str, tuple[Path, Mapping[str, Any]]] = {
        str(row["model_id"]): (base_run, base_plan) for row in plan["roster"]["models"]
    }
    source_directories[deepseek_model] = deepseek_run
    source_plans[deepseek_model] = (deepseek_run, deepseek_plan)
    for model_id in cohere_models:
        source_directories[model_id] = cohere_run
        source_plans[model_id] = (cohere_run, plan)
    primary = load_panel(
        run_directory=base_run,
        panel="primary",
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat_panel,
        model_sources=source_plans,
    )
    repeat = load_panel(
        run_directory=base_run,
        panel="repeat",
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat_panel,
        model_sources=source_plans,
    )
    primary_documents = _response_documents(
        panel="primary",
        final_plan=plan,
        task_ids=primary.task_ids,
        source_directories=source_directories,
    )
    repeat_documents = _response_documents(
        panel="repeat",
        final_plan=plan,
        task_ids=repeat.task_ids,
        source_directories=source_directories,
    )
    if (
        release["inputs"]["primary_responses"]["artifact_set_sha256"]
        != hashlib.sha256(_canonical(list(primary.response_artifact_sha256s))).hexdigest()
        or release["inputs"]["repeat_responses"]["artifact_set_sha256"]
        != hashlib.sha256(_canonical(list(repeat.response_artifact_sha256s))).hexdigest()
    ):
        raise PoweredDatasetBuildError("release response commitments differ from exact files")
    expected = _expected_files(
        release=release,
        taskset=taskset,
        repeat_panel=repeat_panel,
        plan=plan,
        primary_documents=primary_documents,
        repeat_documents=repeat_documents,
    )
    if check:
        mismatches = [
            name
            for name, payload in expected.items()
            if not (output / name).is_file() or (output / name).read_bytes() != payload
        ]
        if mismatches:
            raise PoweredDatasetBuildError(f"generated dataset mismatch: {', '.join(mismatches)}")
        print(f"OK: {len(expected) - 1} powered tables match {release['artifact_sha256']}")
        return
    for name, payload in expected.items():
        _write_atomic(output / name, payload)
    print(f"Wrote {len(expected) - 1} powered tables to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full powered FlavourBench HF dataset")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--deepseek-plan", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--deepseek-run", type=Path, required=True)
    parser.add_argument("--cohere-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(
        release_path=args.release,
        taskset_path=args.taskset,
        repeat_panel_path=args.repeat_panel,
        plan_path=args.plan,
        base_plan_path=args.base_plan,
        deepseek_plan_path=args.deepseek_plan,
        base_run=args.base_run,
        deepseek_run=args.deepseek_run,
        cohere_run=args.cohere_run,
        output=args.output,
        check=args.check,
    )


if __name__ == "__main__":
    main()
