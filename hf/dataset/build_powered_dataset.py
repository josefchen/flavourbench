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
from flavourbench.epicure_selection_powered_plan_v38 import verify_plan as verify_v38_plan
from flavourbench.epicure_selection_powered_plan_v39 import verify_plan as verify_v39_plan
from flavourbench.epicure_selection_powered_plan_v42 import verify_plan as verify_v42_plan
from flavourbench.epicure_selection_taskset_v1 import verify_taskset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "data-powered"
TABLE_ORDER = (
    "models",
    "tasks",
    "primary_observations",
    "repeat_observations",
    "provider_attempt_events",
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


def _attempt_event_valid(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("event_sha256", ""))
    return bool(recorded and recorded == hashlib.sha256(_canonical(payload)).hexdigest())


def _provider_attempt_documents(
    *,
    response_documents: Sequence[Mapping[str, Any]],
    source_directories: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Return every provider event referenced by the exported response cells."""
    required: dict[str, tuple[str, str]] = {}
    for response in response_documents:
        arm_id = str(response["arm_id"])
        plan_sha256 = str(response["plan_sha256"])
        hashes = response.get("attempt_event_sha256s")
        if not isinstance(hashes, list) or not hashes:
            raise PoweredDatasetBuildError(f"response has no provider-attempt lineage: {arm_id}")
        for value in hashes:
            event_sha256 = str(value)
            if event_sha256 in required:
                raise PoweredDatasetBuildError(
                    f"provider-attempt event is referenced by multiple responses: {event_sha256}"
                )
            required[event_sha256] = (arm_id, plan_sha256)

    observed: dict[str, dict[str, Any]] = {}
    for directory in sorted(set(source_directories.values()), key=str):
        journal = directory / "attempts/provider-attempts.jsonl"
        if journal.is_symlink() or not journal.is_file():
            raise PoweredDatasetBuildError(f"provider-attempt journal is missing: {journal}")
        for line in journal.read_bytes().splitlines():
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PoweredDatasetBuildError(
                    f"provider-attempt journal contains invalid JSON: {journal}"
                ) from exc
            if not isinstance(document, dict) or not _attempt_event_valid(document):
                raise PoweredDatasetBuildError(
                    f"provider-attempt event failed its semantic hash: {journal}"
                )
            event_sha256 = str(document["event_sha256"])
            if event_sha256 not in required:
                continue
            if event_sha256 in observed:
                raise PoweredDatasetBuildError(
                    f"duplicate provider-attempt event across journals: {event_sha256}"
                )
            arm_id, plan_sha256 = required[event_sha256]
            event = document.get("event")
            if (
                document.get("plan_sha256") != plan_sha256
                or not isinstance(event, Mapping)
                or event.get("arm_id") != arm_id
            ):
                raise PoweredDatasetBuildError(
                    f"provider-attempt event differs from its response binding: {event_sha256}"
                )
            observed[event_sha256] = document

    missing = sorted(set(required) - set(observed))
    if missing:
        raise PoweredDatasetBuildError(
            f"provider-attempt lineage is incomplete: {len(missing)} events missing"
        )
    return [observed[event_sha256] for event_sha256 in sorted(observed)]


def _tables(
    *,
    release: Mapping[str, Any],
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    plan: Mapping[str, Any],
    primary_documents: Sequence[Mapping[str, Any]],
    repeat_documents: Sequence[Mapping[str, Any]],
    provider_attempt_documents: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    analysis_by_model = {str(row["model_id"]): row for row in release["analysis"]["models"]}
    source_lineage = release["inputs"]["model_response_sources"]
    if "deepseek_model_ids" in source_lineage:
        base_models = set(source_lineage["base_model_ids"])
        cohere_models = set(source_lineage["cohere_model_ids"])
        frontier_models = set(source_lineage["frontier_model_ids"])
        deepseek_models = set(source_lineage["deepseek_model_ids"])
        successor_models = set(source_lineage["successor_model_ids"])
        deepseek_model = None
    elif "frontier_model_ids" in source_lineage:
        base_models = set(source_lineage["base_model_ids"])
        cohere_models = set(source_lineage["cohere_model_ids"])
        frontier_models = set(source_lineage["frontier_model_ids"])
        successor_models = set(source_lineage["successor_model_ids"])
        deepseek_model = None
        deepseek_models = set()
    elif "successor_model_ids" in source_lineage:
        base_models = set(source_lineage["base_model_ids"])
        cohere_models = set(source_lineage["cohere_model_ids"])
        frontier_models = set()
        successor_models = set(source_lineage["successor_model_ids"])
        deepseek_model = None
        deepseek_models = set()
    else:
        base_models = set(source_lineage["base_models"])
        deepseek_model = str(source_lineage["deepseek_model_id"])
        cohere_models = set(source_lineage["cohere_model_ids"])
        frontier_models = set()
        successor_models = set()
        deepseek_models = {deepseek_model}
    model_rows = []
    for route in plan["roster"]["models"]:
        model_id = str(route["model_id"])
        if model_id in base_models:
            source = "powered-v31-base"
        elif model_id in deepseek_models:
            source = (
                "powered-v39-deepseek-repair"
                if "deepseek_model_ids" in source_lineage
                else "powered-v33-clean-deepseek"
            )
        elif model_id in cohere_models:
            source = "powered-v35-clean-cohere"
        elif model_id in successor_models:
            source = (
                "powered-v42-fable-complete-block"
                if "deepseek_model_ids" in source_lineage
                else (
                    "powered-v39-deepseek-repair"
                    if frontier_models
                    else "powered-v38-frontier-refresh"
                )
            )
        elif model_id in frontier_models:
            source = "powered-v38-frontier-refresh"
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
        "provider_attempt_events": [dict(row) for row in provider_attempt_documents],
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
    provider_attempt_documents: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    tables = _tables(
        release=release,
        taskset=taskset,
        repeat_panel=repeat_panel,
        plan=plan,
        primary_documents=primary_documents,
        repeat_documents=repeat_documents,
        provider_attempt_documents=provider_attempt_documents,
    )
    files = {f"{name}.jsonl": _jsonl(tables[name]) for name in TABLE_ORDER}
    manifest: dict[str, Any] = {
        "schema_version": "flavourbench-hf-powered-dataset-manifest-v2",
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
    deepseek_plan_path: Path | None,
    cohere_plan_path: Path | None,
    frontier_plan_path: Path | None,
    base_run: Path,
    deepseek_run: Path | None,
    cohere_run: Path,
    frontier_run: Path | None,
    successor_run: Path | None,
    output: Path,
    check: bool,
) -> None:
    release = _load(release_path)
    taskset = _load(taskset_path)
    repeat_panel = _load(repeat_panel_path)
    plan = _load(plan_path)
    base_plan = _load(base_plan_path)
    plan_is_v42 = verify_v42_plan(plan)
    plan_is_v39 = verify_v39_plan(plan)
    plan_is_v38 = verify_v38_plan(plan)
    deepseek_plan = _load(deepseek_plan_path) if deepseek_plan_path is not None else None
    cohere_plan = _load(cohere_plan_path) if cohere_plan_path is not None else None
    frontier_plan = _load(frontier_plan_path) if frontier_plan_path is not None else None
    if (
        not _semantic_valid(release)
        or release.get("status") != "final_complete"
        or not verify_taskset(taskset)
        or not verify_repeat_panel(repeat_panel, taskset=taskset)
        or not (plan_is_v42 or plan_is_v39 or plan_is_v38 or verify_v35_plan(plan))
    ):
        raise PoweredDatasetBuildError("powered release inputs failed verification")
    if plan_is_v42:
        if (
            cohere_plan_path is None
            or cohere_plan is None
            or frontier_plan_path is None
            or frontier_plan is None
            or deepseek_plan_path is None
            or deepseek_plan is None
            or frontier_run is None
            or deepseek_run is None
            or successor_run is None
        ):
            raise PoweredDatasetBuildError(
                "v42 export requires exact Cohere/v38/v39 plans and all successor runs"
            )
        _verify_pin(
            document=base_plan,
            path=base_plan_path,
            pin=plan["inputs"]["retained_base_response_source_plan"],
            verifier=verify_v31_plan,
            label="base",
        )
        _verify_pin(
            document=cohere_plan,
            path=cohere_plan_path,
            pin=plan["inputs"]["retained_cohere_response_source_plan"],
            verifier=verify_v35_plan,
            label="Cohere",
        )
        _verify_pin(
            document=frontier_plan,
            path=frontier_plan_path,
            pin=plan["inputs"]["plan_v38_predecessor"],
            verifier=verify_v38_plan,
            label="v38 frontier",
        )
        _verify_pin(
            document=deepseek_plan,
            path=deepseek_plan_path,
            pin=plan["inputs"]["plan_v39_predecessor"],
            verifier=verify_v39_plan,
            label="v39 DeepSeek",
        )
        successor = plan["execution"]["frontier_refresh_successor"]
        base_models = {str(value) for value in successor["retained_base_model_ids"]}
        cohere_models = {str(value) for value in successor["retained_cohere_model_ids"]}
        frontier_models = {str(value) for value in successor["retained_v38_new_model_ids"]}
        deepseek_models = {str(value) for value in successor["retained_v39_new_model_ids"]}
        successor_models = {str(value) for value in successor["rerun_model_ids"]}
        groups = (base_models, cohere_models, frontier_models, deepseek_models, successor_models)
        roster = {str(row["model_id"]) for row in plan["roster"]["models"]}
        if (
            tuple(map(len, groups)) != (16, 2, 6, 1, 1)
            or any(
                left & right for index, left in enumerate(groups) for right in groups[index + 1 :]
            )
            or set().union(*groups) != roster
        ):
            raise PoweredDatasetBuildError("v42 response-source partition failed")
        source_directories = {model_id: base_run for model_id in base_models}
        source_directories.update({model_id: cohere_run for model_id in cohere_models})
        source_directories.update({model_id: frontier_run for model_id in frontier_models})
        source_directories.update({model_id: deepseek_run for model_id in deepseek_models})
        source_directories.update({model_id: successor_run for model_id in successor_models})
        source_plans = {model_id: (base_run, base_plan) for model_id in base_models}
        source_plans.update({model_id: (cohere_run, cohere_plan) for model_id in cohere_models})
        source_plans.update(
            {model_id: (frontier_run, frontier_plan) for model_id in frontier_models}
        )
        source_plans.update(
            {model_id: (deepseek_run, deepseek_plan) for model_id in deepseek_models}
        )
        source_plans.update({model_id: (successor_run, plan) for model_id in successor_models})
    elif plan_is_v39:
        if (
            cohere_plan_path is None
            or cohere_plan is None
            or frontier_plan_path is None
            or frontier_plan is None
            or frontier_run is None
            or successor_run is None
        ):
            raise PoweredDatasetBuildError(
                "v39 export requires exact Cohere/v38 plans and frontier/successor runs"
            )
        _verify_pin(
            document=base_plan,
            path=base_plan_path,
            pin=plan["inputs"]["retained_base_response_source_plan"],
            verifier=verify_v31_plan,
            label="base",
        )
        _verify_pin(
            document=cohere_plan,
            path=cohere_plan_path,
            pin=plan["inputs"]["retained_cohere_response_source_plan"],
            verifier=verify_v35_plan,
            label="Cohere",
        )
        _verify_pin(
            document=frontier_plan,
            path=frontier_plan_path,
            pin=plan["inputs"]["plan_v38_predecessor"],
            verifier=verify_v38_plan,
            label="v38 frontier",
        )
        successor = plan["execution"]["frontier_refresh_successor"]
        base_models = {str(value) for value in successor["retained_base_model_ids"]}
        cohere_models = {str(value) for value in successor["retained_cohere_model_ids"]}
        frontier_models = {str(value) for value in successor["retained_v38_new_model_ids"]}
        successor_models = {str(value) for value in successor["rerun_model_ids"]}
        source_directories = {model_id: base_run for model_id in base_models}
        source_directories.update({model_id: cohere_run for model_id in cohere_models})
        source_directories.update({model_id: frontier_run for model_id in frontier_models})
        source_directories.update({model_id: successor_run for model_id in successor_models})
        source_plans = {model_id: (base_run, base_plan) for model_id in base_models}
        source_plans.update({model_id: (cohere_run, cohere_plan) for model_id in cohere_models})
        source_plans.update(
            {model_id: (frontier_run, frontier_plan) for model_id in frontier_models}
        )
        source_plans.update({model_id: (successor_run, plan) for model_id in successor_models})
    elif plan_is_v38:
        if cohere_plan_path is None or cohere_plan is None or successor_run is None:
            raise PoweredDatasetBuildError(
                "v38 export requires the exact Cohere plan and successor run"
            )
        _verify_pin(
            document=base_plan,
            path=base_plan_path,
            pin=plan["inputs"]["retained_base_response_source_plan"],
            verifier=verify_v31_plan,
            label="base",
        )
        _verify_pin(
            document=cohere_plan,
            path=cohere_plan_path,
            pin=plan["inputs"]["retained_cohere_response_source_plan"],
            verifier=verify_v35_plan,
            label="Cohere",
        )
        successor = plan["execution"]["frontier_refresh_successor"]
        base_models = {str(value) for value in successor["retained_base_model_ids"]}
        cohere_models = {str(value) for value in successor["retained_cohere_model_ids"]}
        successor_models = {str(value) for value in successor["new_model_ids"]}
        source_directories = {model_id: base_run for model_id in base_models}
        source_directories.update({model_id: cohere_run for model_id in cohere_models})
        source_directories.update({model_id: successor_run for model_id in successor_models})
        source_plans = {model_id: (base_run, base_plan) for model_id in base_models}
        source_plans.update({model_id: (cohere_run, cohere_plan) for model_id in cohere_models})
        source_plans.update({model_id: (successor_run, plan) for model_id in successor_models})
    else:
        if deepseek_plan_path is None or deepseek_plan is None or deepseek_run is None:
            raise PoweredDatasetBuildError("v35 export requires the exact DeepSeek plan and run")
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
            str(value)
            for value in plan["execution"]["cohere_route_successor"]["successor_model_ids"]
        )
        source_directories = {str(row["model_id"]): base_run for row in plan["roster"]["models"]}
        source_plans = {
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
    provider_attempt_documents = _provider_attempt_documents(
        response_documents=(*primary_documents, *repeat_documents),
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
        provider_attempt_documents=provider_attempt_documents,
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
    parser.add_argument("--deepseek-plan", type=Path)
    parser.add_argument("--cohere-plan", type=Path)
    parser.add_argument("--frontier-plan", type=Path)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--deepseek-run", type=Path)
    parser.add_argument("--cohere-run", type=Path, required=True)
    parser.add_argument("--frontier-run", type=Path)
    parser.add_argument("--successor-run", type=Path)
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
        cohere_plan_path=args.cohere_plan,
        frontier_plan_path=args.frontier_plan,
        base_run=args.base_run,
        deepseek_run=args.deepseek_run,
        cohere_run=args.cohere_run,
        frontier_run=args.frontier_run,
        successor_run=args.successor_run,
        output=args.output,
        check=args.check,
    )


if __name__ == "__main__":
    main()
