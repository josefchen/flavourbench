#!/usr/bin/env python3
"""Verify all frozen adapters and create the confirmatory evaluation gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from flavourbench.reward_transfer import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_RESULTS,
    create_evaluation_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "evaluation-gate.json")
    args = parser.parse_args()
    gate = create_evaluation_gate(checkpoints=args.checkpoints, output=args.output)
    print(f"{args.output} {gate['artifact_sha256']}")


if __name__ == "__main__":
    main()
