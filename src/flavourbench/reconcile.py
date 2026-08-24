"""Append-only cost corrections for incomplete live-smoke generations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .execution_policy import assert_legacy_paid_cli_allowed
from .provider import OpenRouterProvider

CONFIRMATION = "RECONCILE_OPENROUTER_GENERATIONS"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def missing_generation_ids(artifact: dict[str, Any]) -> list[str]:
    received = {
        str(event.get("generation_id") or "")
        for event in artifact.get("provider_attempt_events") or []
        if event.get("event_type") == "response_received"
    }
    accounted = {
        str(metadata.get("generation_id") or "")
        for result in (artifact.get("results") or {}).values()
        for metadata in result.get("generation_metadata") or []
    }
    accounted.update(
        str(metadata.get("generation_id") or "")
        for metadata in artifact.get("incomplete_generation_metadata") or []
    )
    return sorted(received - accounted - {""})


def _verify_source(artifact: dict[str, Any]) -> str:
    claimed = str(artifact.get("artifact_sha256") or "")
    unhashed = dict(artifact)
    unhashed.pop("artifact_sha256", None)
    if len(claimed) != 64 or _sha256(unhashed) != claimed:
        raise RuntimeError("source live-smoke artifact failed its content hash")
    return claimed


async def reconcile_artifact(source: Path, output_dir: Path) -> Path:
    artifact = json.loads(source.read_text())
    if not isinstance(artifact, dict):
        raise RuntimeError("source live-smoke artifact is invalid")
    source_sha256 = _verify_source(artifact)
    missing = missing_generation_ids(artifact)
    if not missing:
        raise RuntimeError("source artifact has no unreconciled received generations")

    provider = OpenRouterProvider()
    try:
        metadata = [await provider._generation_cost(item) for item in missing]  # noqa: SLF001
    finally:
        await provider.aclose()
    additional_cost_micros = sum(int(item.get("cost_micros") or 0) for item in metadata)
    original_cost_micros = int((artifact.get("budget") or {}).get("actual_cost_micros") or 0)
    correction: dict[str, Any] = {
        "schema_version": "flavourbench-live-smoke-cost-correction-v1",
        "record_type": "superseding_cost_reconciliation",
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(source.resolve()),
            "artifact_sha256": source_sha256,
            "run_id": artifact.get("run_id"),
            "requested_model_id": artifact.get("requested_model_id"),
            "candidate_manifest_sha256": artifact.get("candidate_manifest_sha256"),
        },
        "missing_generation_ids": missing,
        "generation_metadata": metadata,
        "all_missing_generations_reconciled": all(
            bool(item.get("reconciled")) for item in metadata
        ),
        "cost": {
            "original_recorded_cost_micros": original_cost_micros,
            "additional_cost_micros": additional_cost_micros,
            "corrected_total_cost_micros": original_cost_micros + additional_cost_micros,
        },
        "rank_eligible": False,
        "note": "The immutable source is not rewritten; this record supersedes only its cost.",
    }
    correction["artifact_sha256"] = _sha256(correction)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (
        f"{source.stem}-cost-correction-{correction['artifact_sha256'][:12]}.json"
    )
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite correction artifact: {destination}")
    destination.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n")
    destination.chmod(0o644)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/corrections"))
    parser.add_argument("--confirm", required=True)
    return parser


async def _run(args: argparse.Namespace) -> list[Path]:
    if args.confirm != CONFIRMATION:
        raise RuntimeError(f"pass --confirm {CONFIRMATION}")
    outputs = []
    for source in args.sources:
        outputs.append(await reconcile_artifact(source, args.output_dir))
    return outputs


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-reconcile-live-smoke")
    try:
        outputs = asyncio.run(_run(_parser().parse_args()))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(1) from exc
    print(json.dumps({"status": "complete", "artifacts": [str(path) for path in outputs]}))


if __name__ == "__main__":
    run()
