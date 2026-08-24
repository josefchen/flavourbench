import json
from pathlib import Path

from flavourbench.current_frontier_correction_assets import (
    _read_verified,
    build_summary,
    write_assets,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/season1/current-quality-run"


def test_correction_assets_close_every_quarantine_denominator(tmp_path: Path) -> None:
    summary = build_summary(
        arena=_read_verified(
            RUN
            / "frontier-model-arena-review-pool-quarantine-v1"
            / (
                "frontier-model-arena-review-pool-"
                "407e7fc6413e6d009c942eb51d9603d7cb958f0f282ffe90e1dc8ff28c3b6ac3.json"
            )
        ),
        strict_uplift=_read_verified(
            RUN
            / "frontier-strict-review-pool-quarantine-v1"
            / (
                "frontier-multirun-review-pool-"
                "0da4c58326a936daef3d9e6ac606cfb5abaff2e9d93784754c56a302c662f38c.json"
            )
        ),
        high_uplift=_read_verified(
            RUN
            / "frontier-high-resource-review-pool-quarantine-v1"
            / (
                "frontier-multirun-review-pool-"
                "cd47055d12e6360a1ad0bfaa73fe4b2cef5bd1f5666150968bdfeeaf9eca024c.json"
            )
        ),
        quarantine=_read_verified(
            RUN
            / "task-quarantine-v1"
            / (
                "current-frontier-task-quarantine-"
                "e095c45ed27b0639a8eefae13a028c653fdea493999e095c2a757818ebbb7a15.json"
            )
        ),
        coverage=_read_verified(
            RUN
            / "frontier-coverage-repair-v1"
            / (
                "frontier-coverage-repair-"
                "45ffc02f56b16b04f2fb4ce51c3561ddb99bd0cad55bf3a7c5162107b2085857.json"
            )
        ),
    )

    assert summary["tasks"] == {
        "gross": 24,
        "excluded": 4,
        "retained": 20,
        "quarantined_task_ids": [
            "fb-s0-composition-006",
            "fb-s0-composition-008",
            "fb-s0-composition-009",
            "fb-s0-cookability-003",
        ],
    }
    assert summary["arena_comparisons"] == {
        "gross": 1024,
        "excluded": 148,
        "retained": 876,
    }
    assert summary["uplift_pairs"]["gross"] == 211
    assert summary["uplift_pairs"]["excluded"] == 32
    assert summary["uplift_pairs"]["retained"] == 179
    assert summary["uplift_pairs"]["minimum_per_model"] == 6
    assert summary["uplift_pairs"]["maximum_per_model"] == 17
    assert summary["uplift_pairs"]["models_below_eight"] == 3
    assert summary["dependence"]["maximum_reuse"] == 13
    assert summary["coverage"]["missing_cells"] == 94
    assert summary["coverage"]["scheduled_real_arms"] == 25
    assert summary["coverage"]["calls_completed"] is False

    paths = write_assets(summary, tmp_path)
    assert all(path.is_file() for path in paths.values())
    assert "Matched uplift pairs & 211 & 32 & 179" in paths["table"].read_text()
    assert json.loads(paths["summary"].read_text())["coverage"]["calls_completed"] is False
