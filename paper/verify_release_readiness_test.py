from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import verify_release_readiness as verifier


def _current_layout() -> tuple[Path, str]:
    directory = Path(__file__).resolve().parent
    if (directory / "contracts" / "release-readiness").is_dir():
        return directory, "archive"
    archive_root = directory.parents[1]
    if (archive_root / "contracts" / "release-readiness").is_dir():
        return archive_root, "archive"
    return directory.parents[1], "repo"


class ReleaseReadinessVerifierTest(unittest.TestCase):
    def test_current_release_tree_verifies(self) -> None:
        root, layout = _current_layout()
        result = verifier.verify_release(root, layout)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["artifacts_semantically_verified"], 13)
        self.assertEqual(result["k14_statistical_readiness"]["failed_core_checks"], 12)
        self.assertFalse(
            result["k16_statistical_readiness"]["simultaneous_top_target_passed"]
        )
        self.assertEqual(
            result["release_decision"]["quality_public_product_rank_rows"], 0
        )
        self.assertEqual(
            result["release_decision"]["automated_operational_rank_rows"], 16
        )
        self.assertEqual(
            result["release_decision"]["automated_operational_leaderboard"], "GO"
        )
        self.assertEqual(result["operational_leaderboard"]["status"], "GO")
        self.assertEqual(result["operational_leaderboard"]["ranked_models"], 16)
        self.assertEqual(result["operational_leaderboard"]["quality_judgments"], 0)

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with self.assertRaises(verifier.ReleaseReadinessError):
            verifier._strict_json(b'{"same":1,"same":2}', "duplicate fixture")

    def test_artifact_semantic_digest_excludes_only_self_hash(self) -> None:
        body = {"schema_version": "fixture-v1", "status": "NO-GO"}
        semantic = hashlib.sha256(verifier._canonical_bytes(body)).hexdigest()
        document = {**body, "artifact_sha256": semantic}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = verifier._strict_json(path.read_bytes(), "semantic fixture")
        loaded_body = {
            key: value for key, value in loaded.items() if key != "artifact_sha256"
        }
        self.assertEqual(verifier._semantic_sha256(loaded_body), semantic)


if __name__ == "__main__":
    unittest.main()
