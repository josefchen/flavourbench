from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = REPOSITORY / "contracts/reward-transfer/reward-transfer-plan-v2.json"
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
    assert plan["status"] == "prospective_protocol_frozen_before_any_transfer_outcome"
    assert plan["inference"]["family_size"] == 1
    assert plan["inference"]["confirmatory_contrast"] == (
        "sft_epicure_optimum_minus_sft_format_control"
    )
    assert plan["base_model"]["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
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
    for config in ("sft", "sft_format_control", "dpo", "grpo"):
        section = card.split(f"- config_name: {config}\n", 1)[1].split("- config_name:", 1)[0]
        assert transfer_path not in section


def test_format_control_matches_prompts_and_label_marginals_without_optima() -> None:
    reward = _rows("sft_train.jsonl") + _rows("sft_validation.jsonl")
    control = _rows("sft_format_control_train.jsonl") + _rows("sft_format_control_validation.jsonl")
    reward_by_id = {str(row["task_id"]): row for row in reward}
    control_by_id = {str(row["task_id"]): row for row in control}
    assert set(reward_by_id) == set(control_by_id)
    for task_id, reward_row in reward_by_id.items():
        control_row = control_by_id[task_id]
        assert control_row["prompt"] == reward_row["prompt"]
        assert control_row["completion"] != reward_row["completion"]
        assert control_row["control_is_optimal"] is False
        assert int(control_row["control_reward_bps"]) < 10_000

    tasks = {
        str(row["task_id"]): row
        for name in ("train_tasks.jsonl", "validation_tasks.jsonl")
        for row in _rows(name)
    }
    for split in ("train", "validation"):
        for family in ("substitution", "pairing", "constraint"):
            for panel in ("panel_1", "panel_2"):
                ids = {
                    task_id
                    for task_id, row in tasks.items()
                    if row["split"] == split
                    and row["family"] == family
                    and row["source_panel"] == panel
                }
                assert Counter(reward_by_id[task_id]["completion"] for task_id in ids) == Counter(
                    control_by_id[task_id]["completion"] for task_id in ids
                )


def test_training_recipes_bind_the_seed() -> None:
    for method in ("sft", "dpo", "grpo"):
        source = (REPOSITORY / f"examples/lab/train_{method}.py").read_text()
        assert 'SEED = int(os.environ.get("SEED", "20260824"))' in source
        assert "seed=SEED" in source
        assert "data_seed=SEED" in source
        assert "full_determinism=True" in source
