from __future__ import annotations

import asyncio
import json

from sqlalchemy import select

from .arena import create_battle
from .config import get_settings
from .database import init_database, session_scope
from .engine import run_worker_once
from .models import Battle, ResponseArm, Season, SeasonModel, Task
from .schemas import BattleCreate
from .seed import seed_database

SMOKE_MODELS = {
    "flavourbench/mock-efficient-a",
    "flavourbench/mock-efficient-b",
    "flavourbench/mock-mistral-open",
}


async def smoke() -> dict:
    settings = get_settings()
    if settings.execution_mode != "mock":
        raise RuntimeError("the bundled smoke command is mock-only")
    init_database()
    seed_database()
    battle_ids = []
    with session_scope() as session:
        season = session.scalar(select(Season).where(Season.slug == settings.default_season_slug))
        if season is None:
            raise RuntimeError("Season 0 is unavailable")
        tasks = session.scalars(
            select(Task).where(Task.season_id == season.id).order_by(Task.public_id).limit(3)
        ).all()
        slots = session.scalars(select(SeasonModel).where(SeasonModel.season_id == season.id)).all()
        original_eligibility = {slot.id: slot.eligible for slot in slots}
        for slot in slots:
            slot.eligible = slot.model_id in SMOKE_MODELS
        session.flush()
        for index, task in enumerate(tasks):
            battle = create_battle(
                session,
                BattleCreate(
                    prompt=task.prompt,
                    category=task.family,
                    research_consent=False,
                    client_nonce=f"mock-smoke-{task.prompt_sha256[:24]}-{index}",
                ),
                pseudonym="e" * 64,
            )
            battle_ids.append(battle.id)
        for slot in slots:
            slot.eligible = original_eligibility[slot.id]
    while await run_worker_once("mock-smoke-worker"):
        pass
    results = []
    with session_scope() as session:
        for battle_id in battle_ids:
            battle = session.get(Battle, battle_id)
            arms = session.scalars(
                select(ResponseArm).where(ResponseArm.battle_id == battle_id)
            ).all()
            results.append(
                {
                    "battle_id": battle_id,
                    "status": battle.status if battle else "missing",
                    "models": sorted({arm.model_id for arm in arms}),
                    "conditions": sorted({arm.condition for arm in arms}),
                }
            )
    return {"mode": "mock", "ranked": False, "battles": results}


def run() -> None:
    print(json.dumps(asyncio.run(smoke()), sort_keys=True))


if __name__ == "__main__":
    run()
