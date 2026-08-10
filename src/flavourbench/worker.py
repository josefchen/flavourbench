from __future__ import annotations

import asyncio
import socket
import time

from .catalog import fetch_openrouter_catalog, sync_catalog
from .config import get_settings
from .database import database_readiness, init_database, session_scope
from .engine import recover_stale_jobs, redact_expired, run_worker_once
from .models import Incident, RunEvent
from .seed import seed_database


async def serve() -> None:
    settings = get_settings()
    init_database()
    seed_database()
    with session_scope() as session:
        database_readiness(session, expected_role="flavourbench_worker")
    worker_id = f"{socket.gethostname()}-worker"
    last_maintenance = 0.0
    last_catalog_sync = 0.0
    while True:
        now = time.monotonic()
        if now - last_maintenance >= 60:
            with session_scope() as session:
                recover_stale_jobs(session)
                redact_expired(session)
            last_maintenance = now
        if (settings.catalog_sync_enabled or settings.execution_mode == "live") and (
            now - last_catalog_sync >= settings.catalog_sync_hours * 3600
        ):
            try:
                items = await fetch_openrouter_catalog()
                with session_scope() as session:
                    counts = sync_catalog(session, items)
                    session.add(
                        RunEvent(
                            entity_type="catalog",
                            entity_id="openrouter",
                            event_type="catalog_synchronized",
                            payload_json=counts,
                        )
                    )
            except Exception as exc:
                with session_scope() as session:
                    session.add(
                        Incident(
                            severity="medium",
                            code="catalog_sync_failed",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                    )
            last_catalog_sync = now
        handled = await run_worker_once(worker_id)
        if not handled:
            await asyncio.sleep(settings.worker_poll_seconds)


def run() -> None:
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
