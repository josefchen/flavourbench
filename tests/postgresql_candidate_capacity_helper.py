from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psycopg
from psycopg import sql


def _wait_for_peer(directory: Path) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if len(tuple(directory.glob("ready-*"))) >= 2:
            return
        time.sleep(0.01)
    raise RuntimeError("PostgreSQL candidate-capacity process barrier timed out")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--barrier-directory", type=Path, required=True)
    args = parser.parse_args()

    args.barrier_directory.mkdir(parents=True, exist_ok=True)
    (args.barrier_directory / f"ready-{args.ordinal}").write_text("ready\n", encoding="utf-8")
    _wait_for_peer(args.barrier_directory)
    status = "inserted"
    try:
        with psycopg.connect(args.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} (campaign_sha256, candidate_id, event_type) "
                        "VALUES ('capacity-campaign', 'capacity-candidate', 'blind_ballot')"
                    ).format(sql.Identifier(args.table))
                )
    except psycopg.errors.RaiseException as exc:
        if "task-validation candidate event capacity is already sealed" not in str(exc):
            raise
        status = "sealed"
    print(json.dumps({"status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
