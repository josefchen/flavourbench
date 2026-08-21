"""Command-line interface for evaluating and training with FlavourBench."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .lab import (
    DEFAULT_DATASET_REPO,
    DEFAULT_EVALUATION_FILE,
    LabValidationError,
    load_hub_tasks,
    read_json_records,
    reward_bps,
    score_submission,
    validate_tasks,
    verify_report,
    write_jsonl,
)
from .lab_runner import run_openai_compatible, run_transformers


def _json_write(path: Path, value: object) -> None:
    if path.is_symlink():
        raise LabValidationError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
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


def _tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.tasks:
        rows = read_json_records(args.tasks)
        validate_tasks(rows)
        return rows
    return load_hub_tasks(
        repo_id=args.dataset_repo,
        filename=args.dataset_file,
        revision=args.revision,
    )


def _print_report(report: dict[str, Any], output: Path) -> None:
    coverage = report["coverage"]
    print(f"report: {output}")
    print(
        "coverage: "
        f"{coverage['valid']}/{coverage['tasks']} valid "
        f"({coverage['fraction_valid']:.1%})"
    )
    if report["comparable"]:
        score = float(report["flavourbench_score"])
        interval = report.get("inference") and report["inference"]["confidence_interval_95"]
        suffix = f" (95% CI {interval[0]:.2f}-{interval[1]:.2f})" if interval else ""
        print(f"FlavourBench Score: {score:.2f}{suffix}")
    else:
        diagnostic = report.get("diagnostic_valid_score")
        rendered = f"{diagnostic:.2f}" if diagnostic is not None else "unavailable"
        print("FlavourBench Score: not issued (run is incomplete or contains invalid answers)")
        print(f"diagnostic valid-only score: {rendered}")


def _score(args: argparse.Namespace) -> int:
    tasks = _tasks(args)
    responses = read_json_records(args.responses)
    report = score_submission(
        tasks,
        responses,
        include_inference=not args.no_inference,
        bootstrap_resamples=args.bootstrap_resamples,
        sign_flip_resamples=args.sign_flip_resamples,
        seed=args.seed,
    )
    _json_write(args.output, report)
    if args.per_task:
        write_jsonl(args.per_task, report["per_task"])
    _print_report(report, args.output)
    return 0 if report["comparable"] or args.allow_partial else 2


def _reward(args: argparse.Namespace) -> int:
    tasks = _tasks(args)
    task_by_id = {str(task["task_id"]): task for task in tasks}
    task = task_by_id.get(args.task_id)
    if task is None:
        raise LabValidationError(f"unknown task_id: {args.task_id}")
    score_bps = reward_bps(task, args.completion)
    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "reward": score_bps / 10_000,
                "score": score_bps / 100,
                "score_bps": score_bps,
            },
            sort_keys=True,
        )
    )
    return 0


def _template(args: argparse.Namespace) -> int:
    tasks = _tasks(args)
    rows = [
        {
            "schema_version": "flavourbench-lab-response-v1",
            "task_id": task["task_id"],
            "status": "completed",
            "response": "FINAL_SELECTION: A,B,C",
        }
        for task in tasks
    ]
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} response rows to {args.output}")
    return 0


def _verify_report(args: argparse.Namespace) -> int:
    rows = read_json_records(args.report)
    if len(rows) != 1:
        raise LabValidationError("report input must contain exactly one JSON object")
    artifact = verify_report(rows[0])
    print(f"OK: FlavourBench lab report {artifact}")
    return 0


def _smoke_tasks(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise LabValidationError("--limit must be positive")
    if limit >= len(tasks):
        raise LabValidationError("--limit must be smaller than the full task set")
    families = ("substitution", "pairing", "constraint")
    if {str(task["family"]) for task in tasks} != set(families):
        raise LabValidationError("model runs require the three-family primary task track")
    by_family = {
        family: [task for task in tasks if task["family"] == family] for family in families
    }
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        for family in families:
            if offset < len(by_family[family]) and len(selected) < limit:
                selected.append(by_family[family][offset])
        offset += 1
    return selected


def _run(args: argparse.Namespace) -> int:
    tasks = _tasks(args)
    if {str(task["family"]) for task in tasks} != {
        "substitution",
        "pairing",
        "constraint",
    }:
        raise LabValidationError("model runs require the three-family primary task track")
    run_tasks = tasks
    if args.limit is not None:
        run_tasks = _smoke_tasks(tasks, args.limit)

    merged: dict[str, dict[str, Any]] = {}
    if args.resume and args.responses.exists():
        task_ids = {str(task["task_id"]) for task in tasks}
        for row in read_json_records(args.responses):
            task_id = str(row.get("task_id") or "")
            if task_id not in task_ids:
                raise LabValidationError(f"resume artifact contains unknown task_id: {task_id}")
            if task_id in merged:
                raise LabValidationError(f"resume artifact duplicates task_id: {task_id}")
            merged[task_id] = row
    pending = [
        task
        for task in run_tasks
        if merged.get(str(task["task_id"]), {}).get("status") != "completed"
    ]

    def checkpoint(row: dict[str, Any]) -> None:
        merged[str(row["task_id"])] = row
        write_jsonl(
            args.responses,
            (merged[task_id] for task in tasks if (task_id := str(task["task_id"])) in merged),
        )

    new_rows: list[dict[str, Any]] = []
    if pending and args.backend == "openai-compatible":
        try:
            extra_body = json.loads(args.extra_body) if args.extra_body else None
        except json.JSONDecodeError as error:
            raise LabValidationError("--extra-body is not valid JSON") from error
        if extra_body is not None and not isinstance(extra_body, dict):
            raise LabValidationError("--extra-body must decode to a JSON object")
        new_rows = asyncio.run(
            run_openai_compatible(
                pending,
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                concurrency=args.concurrency,
                timeout_seconds=args.timeout,
                max_tokens=args.max_tokens,
                temperature=None if args.omit_temperature else args.temperature,
                max_attempts=args.max_attempts,
                extra_body=extra_body,
                on_result=checkpoint,
            )
        )
    elif pending:
        new_rows = run_transformers(
            pending,
            model=args.model,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
        )
    merged.update({str(row["task_id"]): row for row in new_rows})
    rows = [merged[str(task["task_id"])] for task in tasks if str(task["task_id"]) in merged]
    write_jsonl(args.responses, rows)

    report = score_submission(
        tasks,
        rows,
        include_inference=not args.no_inference,
        bootstrap_resamples=args.bootstrap_resamples,
        sign_flip_resamples=args.sign_flip_resamples,
        seed=args.seed,
    )
    _json_write(args.report, report)
    _print_report(report, args.report)
    return 0 if report["comparable"] or args.limit is not None else 2


def _add_task_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", type=Path, help="Local tasks JSON/JSONL; otherwise use the Hub")
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-file", default=DEFAULT_EVALUATION_FILE)
    parser.add_argument("--revision", help="Immutable Hub commit or tag")


def _add_inference(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-inference", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=50_000)
    parser.add_argument("--sign-flip-resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260821)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flavourbench",
        description="Evaluate any model against released Epicure reward maps.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score a JSONL response artifact")
    _add_task_source(score)
    _add_inference(score)
    score.add_argument("responses", type=Path)
    score.add_argument("--output", type=Path, default=Path("flavourbench-report.json"))
    score.add_argument("--per-task", type=Path)
    score.add_argument("--allow-partial", action="store_true")
    score.set_defaults(handler=_score)

    reward_parser = subparsers.add_parser("reward", help="Score one completion")
    _add_task_source(reward_parser)
    reward_parser.add_argument("--task-id", required=True)
    reward_parser.add_argument("--completion", required=True)
    reward_parser.set_defaults(handler=_reward)

    template = subparsers.add_parser("template", help="Write a response JSONL template")
    _add_task_source(template)
    template.add_argument("--output", type=Path, default=Path("responses.jsonl"))
    template.set_defaults(handler=_template)

    verify = subparsers.add_parser("verify-report", help="Verify a content-addressed lab report")
    verify.add_argument("report", type=Path)
    verify.set_defaults(handler=_verify_report)

    run = subparsers.add_parser("run", help="Run and score a model checkpoint or endpoint")
    _add_task_source(run)
    _add_inference(run)
    run.add_argument("--backend", choices=("openai-compatible", "transformers"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--responses", type=Path, default=Path("responses.jsonl"))
    run.add_argument("--report", type=Path, default=Path("flavourbench-report.json"))
    run.add_argument("--limit", type=int, help="Smoke-test only; never yields an official score")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--base-url", default="https://api.openai.com/v1")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--timeout", type=float, default=180.0)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--omit-temperature", action="store_true")
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--extra-body", help="Additional endpoint parameters as a JSON object")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--trust-remote-code", action="store_true")
    run.set_defaults(handler=_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        status = int(args.handler(args))
    except LabValidationError as error:
        parser.exit(2, f"flavourbench: error: {error}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
