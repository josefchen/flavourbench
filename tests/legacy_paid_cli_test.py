from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import pytest

from flavourbench import execution_policy, legacy_paid_cli


@pytest.mark.parametrize(
    ("runner", "command"),
    [
        (
            legacy_paid_cli.run_season0_collection,
            "flavourbench-run-season0-collection",
        ),
        (
            legacy_paid_cli.run_season0_judging,
            "flavourbench-run-season0-judging",
        ),
        (
            legacy_paid_cli.run_season0_judgment_recovery,
            "flavourbench-recover-season0-throttles",
        ),
    ],
)
def test_frozen_paid_runner_wrappers_check_policy_before_import(
    monkeypatch: pytest.MonkeyPatch,
    runner: Callable[[Sequence[str] | None], None],
    command: str,
) -> None:
    observed: list[str] = []

    def deny(value: str) -> None:
        observed.append(value)
        raise RuntimeError("denied")

    monkeypatch.setattr(legacy_paid_cli, "assert_legacy_paid_cli_allowed", deny)
    with pytest.raises(RuntimeError, match="denied"):
        runner([])
    assert observed == [command]


def test_legacy_paid_policy_rejects_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution_policy,
        "get_settings",
        lambda: SimpleNamespace(environment="production"),
    )
    with pytest.raises(RuntimeError, match="PostgreSQL-governed API and worker"):
        execution_policy.assert_legacy_paid_cli_allowed("historical-command")
