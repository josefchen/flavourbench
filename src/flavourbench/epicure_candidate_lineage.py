"""Recover and falsify a candidate training lineage for the Epicure runtime.

The deployed 1,790-row payload was imported into ``epicure-mcp`` from a
private ``epicure-data`` checkout.  This audit binds the import commit and the
surviving candidate run metadata, then compares the candidate export manifest
with the exact deployed matrix.  A source trail is useful evidence, but a
failed matrix comparison must never be promoted into an exact lineage claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "epicure-candidate-training-lineage-audit-v1"
MCP_IMPORT_COMMIT = "177b506a829afca3a85084cca0cd1590d67caef5"
CANDIDATE_SOURCE_COMMIT = "a5ee8f954e1375f706512a9f925983bfa38d0b1e"
CANDIDATE_EXPORTER_COMMIT = "773ed0bf4ddb10229a9d5edd9eda4985e434a6cf"
CANDIDATE_RUN_ID = "mp2v_ml_cooc_dense_ctx7_wpn100_007c"
CANDIDATE_MANIFEST_PATH = "deploy/payload/embeddings_manifest.json"


class CandidateLineageError(RuntimeError):
    """The candidate source trail is incomplete or internally inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object) -> str:
    return _sha256_bytes(_canonical(value))


def _git(repo: Path, *arguments: str) -> bytes:
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        raise CandidateLineageError(f"not a Git checkout: {repo}")
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise CandidateLineageError(
            f"Git evidence lookup failed in {repo}: {' '.join(arguments)}"
        ) from exc


def _git_file(repo: Path, revision: str, path: str) -> bytes:
    return _git(repo, "show", f"{revision}:{path}")


def _git_commit(repo: Path, revision: str) -> dict[str, str]:
    rendered = _git(
        repo,
        "show",
        "-s",
        "--format=%H%x00%aI%x00%an%x00%s",
        revision,
    ).decode("utf-8").rstrip("\n")
    fields = rendered.split("\x00")
    if len(fields) != 4:
        raise CandidateLineageError("Git commit metadata is malformed")
    return dict(zip(("commit", "authored_at", "author", "subject"), fields, strict=True))


def _payload_statistics(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    dimensions = [column for column in frame.columns if column.startswith("dim_")]
    if len(frame) != 1790 or len(dimensions) != 300:
        raise CandidateLineageError("deployed embedding matrix has an unexpected shape")
    matrix = frame[dimensions].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise CandidateLineageError("deployed embedding matrix contains a zero vector")
    normalized = matrix / norms
    similarities = (normalized @ normalized.T)[np.triu_indices(len(frame), 1)]
    return {
        "n_ingredients": len(frame),
        "n_dims": len(dimensions),
        "n_cosine_pairs": len(similarities),
        "similarity_min": round(float(similarities.min()), 6),
        "similarity_max": round(float(similarities.max()), 6),
        "similarity_mean": round(float(similarities.mean()), 6),
    }


def _candidate_job(jobs: object) -> Mapping[str, Any]:
    if not isinstance(jobs, list):
        raise CandidateLineageError("candidate job registry is not an array")
    matches = [
        item
        for item in jobs
        if isinstance(item, Mapping) and item.get("run_id") == CANDIDATE_RUN_ID
    ]
    if len(matches) != 1:
        raise CandidateLineageError("candidate run is absent or duplicated in the job registry")
    return matches[0]


def _stats_match(candidate: Mapping[str, Any], deployed: Mapping[str, Any]) -> bool:
    fields = (
        "n_ingredients",
        "n_dims",
        "n_cosine_pairs",
        "similarity_min",
        "similarity_max",
        "similarity_mean",
    )
    return all(candidate.get(field) == deployed.get(field) for field in fields)


def build_candidate_lineage_audit(
    *,
    mcp_repo: Path,
    source_repo: Path,
) -> dict[str, Any]:
    deployed_path = mcp_repo / "data" / "embeddings.csv"
    if deployed_path.is_symlink() or not deployed_path.is_file():
        raise CandidateLineageError("deployed embedding payload is unavailable")
    deployed_bytes = deployed_path.read_bytes()
    imported_bytes = _git_file(
        mcp_repo,
        MCP_IMPORT_COMMIT,
        "data/embeddings.csv",
    )
    if deployed_bytes != imported_bytes:
        raise CandidateLineageError("current payload differs from the recorded MCP import commit")

    manifest_bytes = _git_file(source_repo, "origin/NLG", CANDIDATE_MANIFEST_PATH)
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, Mapping):
        raise CandidateLineageError("candidate export manifest is not an object")
    if (
        manifest.get("git_sha") != CANDIDATE_SOURCE_COMMIT
        or manifest.get("git_dirty") is not True
        or manifest.get("params", {}).get("run_id") != CANDIDATE_RUN_ID
    ):
        raise CandidateLineageError("candidate export manifest identity is inconsistent")

    jobs_path = "multi_language/jobs_ml_007c.json"
    jobs_bytes = _git_file(source_repo, CANDIDATE_SOURCE_COMMIT, jobs_path)
    job = _candidate_job(json.loads(jobs_bytes))
    training_paths = (
        "models/train.py",
        "models/metapath2vec_train.py",
        "models/graph_loader.py",
        "models/skipgram.py",
        "models/utils.py",
        "models/walkers.py",
    )
    training_sources = [
        {
            "path": path,
            "sha256": _sha256_bytes(_git_file(source_repo, CANDIDATE_SOURCE_COMMIT, path)),
        }
        for path in training_paths
    ]
    exporter_path = "deploy/deploy_payload.py"
    exporter_bytes = _git_file(
        source_repo,
        CANDIDATE_EXPORTER_COMMIT,
        exporter_path,
    )
    graph_provenance_path = "multi_language/data/graph/_provenance.json"
    graph_provenance_bytes = _git_file(
        source_repo,
        "origin/NLG",
        graph_provenance_path,
    )
    graph_provenance = json.loads(graph_provenance_bytes)
    if CANDIDATE_RUN_ID not in json.dumps(graph_provenance, sort_keys=True):
        raise CandidateLineageError("graph provenance does not name the candidate run")

    deployed_stats = _payload_statistics(deployed_path)
    candidate_stats = manifest.get("output_stats")
    if not isinstance(candidate_stats, Mapping):
        raise CandidateLineageError("candidate manifest has no output statistics")
    exact_statistics_match = _stats_match(candidate_stats, deployed_stats)

    extra_args = job.get("extra_args")
    if not isinstance(extra_args, list) or "--seed" in extra_args:
        raise CandidateLineageError("candidate job's seed contract is not the expected default")

    return {
        "schema_version": SCHEMA_VERSION,
        "record_role": "candidate_source_trace_with_exact_lineage_falsification",
        "deployed_runtime_payload": {
            "mcp_import_commit": _git_commit(mcp_repo, MCP_IMPORT_COMMIT),
            "embeddings_sha256": _sha256_bytes(deployed_bytes),
            "embeddings_bytes": len(deployed_bytes),
            "statistics": deployed_stats,
            "unchanged_since_import_commit": True,
        },
        "candidate_private_source": {
            "repository": "KAIKAKU-AI/epicure-data",
            "visibility": "private_at_audit_time",
            "candidate_source_commit": _git_commit(source_repo, CANDIDATE_SOURCE_COMMIT),
            "candidate_export_manifest": {
                "path": CANDIDATE_MANIFEST_PATH,
                "sha256": _sha256_bytes(manifest_bytes),
                "created_at": manifest.get("created_at"),
                "recorded_git_sha": manifest.get("git_sha"),
                "recorded_git_dirty": manifest.get("git_dirty"),
                "input_embedding_sha256": manifest.get("inputs", {})
                .get("embeddings", {})
                .get("sha256"),
                "output_statistics": dict(candidate_stats),
            },
            "candidate_job": {
                "registry_path": jobs_path,
                "registry_sha256": _sha256_bytes(jobs_bytes),
                "run_id": job.get("run_id"),
                "model": job.get("model"),
                "variant": job.get("variant"),
                "gpu": job.get("gpu"),
                "epochs": job.get("epochs"),
                "extra_args": extra_args,
                "seed": 42,
                "seed_basis": "job omits --seed; the content-bound training CLI default is 42",
            },
            "training_sources": training_sources,
            "candidate_exporter": {
                "path": exporter_path,
                "sha256": _sha256_bytes(exporter_bytes),
                "first_committed_in": _git_commit(source_repo, CANDIDATE_EXPORTER_COMMIT),
                "relationship_to_manifest": (
                    "The export manifest records the immediately preceding source commit with "
                    "git_dirty=true; the named exporter first appears in the next commit. Its "
                    "hash is source-trace evidence, not proof of the exact dirty-worktree bytes."
                ),
            },
            "graph_provenance": {
                "path": graph_provenance_path,
                "sha256": _sha256_bytes(graph_provenance_bytes),
                "source_buckets": graph_provenance.get("source"),
                "key_files": graph_provenance.get("key_files"),
            },
        },
        "exact_lineage_test": {
            "candidate_and_deployed_statistics_match": exact_statistics_match,
            "candidate_output_statistics": dict(candidate_stats),
            "deployed_output_statistics": deployed_stats,
            "candidate_input_embedding_artifact_available_in_checkout": False,
            "candidate_export_was_recorded_from_dirty_worktree": True,
            "decision": (
                "candidate_rejected_as_exact_lineage"
                if not exact_statistics_match
                else "manual_exact_matrix_verification_still_required"
            ),
        },
        "gates_advanced": {
            "mcp_import_commit_recovered": True,
            "candidate_run_id_and_job_recovered": True,
            "candidate_training_code_recovered": True,
            "candidate_seed_and_graph_provenance_recovered": True,
        },
        "gates_still_closed": {
            "exact_training_lineage_recovered": False,
            "candidate_input_embedding_artifact_available": False,
            "clean_source_revision_recovered": False,
            "payload_rights_attested": False,
            "public_redistributable_payload": False,
            "independent_end_to_end_reproduction": False,
        },
        "status": "candidate_source_trace_recovered_exact_lineage_rejected",
        "rank_eligible": False,
        "redistributable": False,
        "provider_calls_made": 0,
        "epicure_network_calls_made": 0,
        "synthetic_observations": 0,
        "interpretation_rule": (
            "The surviving private repository recovers a plausible run ID, job, seed, training "
            "implementation, and graph source for the payload-import period. Its recorded export "
            "statistics differ from the exact deployed matrix, and the export was made from a "
            "dirty worktree whose input embedding is absent. The candidate therefore narrows the "
            "search but does not establish exact training lineage or redistribution rights."
        ),
    }


def verify_candidate_lineage_audit(document: Mapping[str, Any]) -> bool:
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    gates = document.get("gates_still_closed")
    test = document.get("exact_lineage_test")
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and isinstance(digest, str)
        and digest == _sha256(unhashed)
        and document.get("status")
        == "candidate_source_trace_recovered_exact_lineage_rejected"
        and document.get("rank_eligible") is False
        and document.get("redistributable") is False
        and isinstance(test, Mapping)
        and test.get("candidate_and_deployed_statistics_match") is False
        and test.get("decision") == "candidate_rejected_as_exact_lineage"
        and isinstance(gates, Mapping)
        and not any(gates.values())
    )


def _write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = _sha256(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"epicure-candidate-training-lineage-audit-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise CandidateLineageError("content-addressed candidate-lineage conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-repo", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = build_candidate_lineage_audit(
        mcp_repo=arguments.mcp_repo.resolve(),
        source_repo=arguments.source_repo.resolve(),
    )
    path = _write(arguments.output_dir.resolve(), payload)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not verify_candidate_lineage_audit(document):
        raise CandidateLineageError("written candidate-lineage artifact does not verify")
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": document["artifact_sha256"],
                "status": document["status"],
                "rank_eligible": False,
                "provider_calls_made": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
