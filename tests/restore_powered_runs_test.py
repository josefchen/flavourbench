from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hf.dataset.restore_powered_runs import PoweredRunRestoreError, _write_no_replace, restore


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _address(document: dict[str, object]) -> dict[str, object]:
    document["artifact_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return document


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = {f"base-{index:02d}" for index in range(17)}
    deepseek = "deepseek"
    cohere = {"cohere-a", "cohere-r"}
    release = _address(
        {
            "status": "final_complete",
            "inputs": {
                "model_response_sources": {
                    "base_models": sorted(base),
                    "deepseek_model_id": deepseek,
                    "cohere_model_ids": sorted(cohere),
                }
            },
        }
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    primary: list[dict[str, object]] = []
    repeat: list[dict[str, object]] = []
    for model_index, model_id in enumerate(sorted(base | {deepseek} | cohere)):
        for panel, size, output in (("primary", 640, primary), ("repeat", 64, repeat)):
            for task_index in range(size):
                cell = hashlib.sha256(f"{panel}:{model_id}:{task_index}".encode()).hexdigest()
                output.append(
                    _address(
                        {
                            "schema_version": "flavourbench-powered-response-v1",
                            "panel": panel,
                            "model_id": model_id,
                            "slot_id": f"slot-{model_index:02d}",
                            "task_id": f"task-{task_index:03d}",
                            "cell_id": cell,
                        }
                    )
                )
    primary_path = tmp_path / "primary.jsonl"
    repeat_path = tmp_path / "repeat.jsonl"
    _write_jsonl(primary_path, primary)
    _write_jsonl(repeat_path, repeat)
    return release_path, primary_path, repeat_path


def _v42_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    groups = {
        "base_model_ids": [f"base-{index:02d}" for index in range(16)],
        "cohere_model_ids": ["cohere-a", "cohere-r"],
        "frontier_model_ids": [f"frontier-{index:02d}" for index in range(6)],
        "deepseek_model_ids": ["deepseek-v4-pro"],
        "successor_model_ids": ["fable-5"],
    }
    roster = {model_id for values in groups.values() for model_id in values}
    release = _address(
        {
            "status": "final_complete",
            "inputs": {
                "model_response_sources": {
                    "schema_version": "flavourbench-selection-composite-response-sources-v7",
                    **groups,
                }
            },
        }
    )
    release_path = tmp_path / "release-v42.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    primary: list[dict[str, object]] = []
    repeat: list[dict[str, object]] = []
    for model_index, model_id in enumerate(sorted(roster)):
        for panel, size, output in (("primary", 640, primary), ("repeat", 64, repeat)):
            for task_index in range(size):
                cell = hashlib.sha256(f"{panel}:{model_id}:{task_index}".encode()).hexdigest()
                output.append(
                    _address(
                        {
                            "schema_version": "flavourbench-powered-response-v1",
                            "panel": panel,
                            "model_id": model_id,
                            "slot_id": f"slot-{model_index:02d}",
                            "task_id": f"task-{task_index:03d}",
                            "cell_id": cell,
                        }
                    )
                )
    primary_path = tmp_path / "primary-v42.jsonl"
    repeat_path = tmp_path / "repeat-v42.jsonl"
    _write_jsonl(primary_path, primary)
    _write_jsonl(repeat_path, repeat)
    return release_path, primary_path, repeat_path


def test_restores_complete_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release, primary, repeat = _inputs(tmp_path)
    roots = [tmp_path / name for name in ("base", "deepseek", "cohere")]
    destinations: set[Path] = set()

    def record_write(path: Path, payload: bytes) -> str:
        assert payload.endswith(b"\n")
        destinations.add(path)
        return "created"

    monkeypatch.setattr("hf.dataset.restore_powered_runs._write_no_replace", record_write)
    summary = restore(
        release_path=release,
        primary_path=primary,
        repeat_path=repeat,
        base_run=roots[0],
        deepseek_run=roots[1],
        cohere_run=roots[2],
        check=False,
    )
    assert summary["files"] == {"created": 14_080}
    assert len(destinations) == 14_080
    assert any(path.is_relative_to(roots[0]) for path in destinations)
    assert any(path.is_relative_to(roots[1]) for path in destinations)
    assert any(path.is_relative_to(roots[2]) for path in destinations)


def test_no_replace_writer_is_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "response.json"
    assert _write_no_replace(destination, b"evidence\n") == "created"
    assert _write_no_replace(destination, b"evidence\n") == "existing"
    with pytest.raises(PoweredRunRestoreError, match="conflicting"):
        _write_no_replace(destination, b"different\n")


def test_restores_v42_five_source_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, primary, repeat = _v42_inputs(tmp_path)
    roots = {
        "base": tmp_path / "base",
        "deepseek": tmp_path / "deepseek",
        "cohere": tmp_path / "cohere",
        "frontier": tmp_path / "frontier",
        "successor": tmp_path / "successor",
    }
    destinations: set[Path] = set()

    def record_write(path: Path, payload: bytes) -> str:
        assert payload.endswith(b"\n")
        destinations.add(path)
        return "created"

    monkeypatch.setattr("hf.dataset.restore_powered_runs._write_no_replace", record_write)
    summary = restore(
        release_path=release,
        primary_path=primary,
        repeat_path=repeat,
        base_run=roots["base"],
        deepseek_run=roots["deepseek"],
        cohere_run=roots["cohere"],
        frontier_run=roots["frontier"],
        successor_run=roots["successor"],
        check=False,
    )
    assert summary["models"] == 26
    assert summary["files"] == {"created": 18_304}
    assert len(destinations) == 18_304
    assert all(any(path.is_relative_to(root) for path in destinations) for root in roots.values())


def test_rejects_incomplete_grid(tmp_path: Path) -> None:
    release, primary, repeat = _inputs(tmp_path)
    rows = repeat.read_text(encoding="utf-8").splitlines()
    repeat.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(PoweredRunRestoreError, match="cardinality"):
        restore(
            release_path=release,
            primary_path=primary,
            repeat_path=repeat,
            base_run=tmp_path / "base",
            deepseek_run=tmp_path / "deepseek",
            cohere_run=tmp_path / "cohere",
            check=False,
        )
