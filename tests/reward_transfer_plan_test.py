from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = REPOSITORY / "contracts/reward-transfer/reward-transfer-plan-v1.json"
LAB_DATA = REPOSITORY / "hf/dataset/data-lab"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _rows(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (LAB_DATA / name).read_text().splitlines()]


def test_reward_transfer_plan_is_semantically_bound() -> None:
    plan = json.loads(PLAN.read_text())
    recorded = plan.pop("artifact_sha256")
    assert hashlib.sha256(_canonical(plan)).hexdigest() == recorded
    assert plan["status"] == "prospective_protocol_frozen_before_training"
    assert plan["confirmatory"]["family_size"] == 6
    assert len(plan["base_models"]) * len(plan["treatments"]) == 6
    assert plan["seeds"] == [20260824, 20260825, 20260826]


def test_transfer_split_is_balanced_and_anchor_disjoint() -> None:
    train = _rows("train_tasks.jsonl")
    validation = _rows("validation_tasks.jsonl")
    evaluation = _rows("evaluation_tasks.jsonl")
    assert Counter((row["family"], row["source_panel"]) for row in train) == Counter(
        {
            (family, panel): 45
            for family in ("substitution", "pairing", "constraint")
            for panel in ("panel_1", "panel_2")
        }
    )
    assert Counter((row["family"], row["source_panel"]) for row in validation) == Counter(
        {
            (family, panel): 12
            for family in ("substitution", "pairing", "constraint")
            for panel in ("panel_1", "panel_2")
        }
    )
    assert Counter((row["family"], row["source_panel"]) for row in evaluation) == Counter(
        {
            (family, panel): 14
            for family in ("substitution", "pairing", "constraint")
            for panel in ("panel_1", "panel_2")
        }
    )
    anchor_sets = [
        {str(row["anchor_ingredient"]) for row in split}
        for split in (train, validation, evaluation)
    ]
    assert not anchor_sets[0] & anchor_sets[1]
    assert not anchor_sets[0] & anchor_sets[2]
    assert not anchor_sets[1] & anchor_sets[2]


def test_optimizer_configs_do_not_expose_transfer_rows() -> None:
    card = (REPOSITORY / "hf/dataset/README.md").read_text()
    transfer_path = "data-lab/evaluation_tasks.jsonl"
    assert card.count(transfer_path) == 1
    for config in ("sft", "dpo", "grpo"):
        section = card.split(f"- config_name: {config}\n", 1)[1].split("- config_name:", 1)[0]
        assert transfer_path not in section


def test_training_recipes_bind_the_seed() -> None:
    for method in ("sft", "dpo", "grpo"):
        source = (REPOSITORY / f"examples/lab/train_{method}.py").read_text()
        assert 'SEED = int(os.environ.get("SEED", "20260824"))' in source
        assert "seed=SEED" in source
        assert "data_seed=SEED" in source
        assert "full_determinism=True" in source
