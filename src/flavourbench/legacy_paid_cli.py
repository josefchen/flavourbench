"""Production-safe command boundaries for frozen, file-governed paid runners."""

from __future__ import annotations

from collections.abc import Sequence

from .execution_policy import assert_legacy_paid_cli_allowed


def run_season0_collection(argv: Sequence[str] | None = None) -> None:
    assert_legacy_paid_cli_allowed("flavourbench-run-season0-collection")
    from .season0_collection import run

    run(argv)


def run_season0_judging(argv: Sequence[str] | None = None) -> None:
    assert_legacy_paid_cli_allowed("flavourbench-run-season0-judging")
    from .season0_judging import run

    run(argv)


def run_season0_judgment_recovery(argv: Sequence[str] | None = None) -> None:
    assert_legacy_paid_cli_allowed("flavourbench-recover-season0-throttles")
    from .season0_judgment_recovery import run

    run(argv)
