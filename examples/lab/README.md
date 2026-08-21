# FlavourBench lab training recipes

These scripts train only on the released development reward maps. Their 426 ingredient anchors
are disjoint from the 534-anchor leaderboard, and the train and validation partitions are also
anchor-disjoint. Do not substitute `data-complete-core/tasks.jsonl`: optimizing on the leaderboard
map turns evaluation into memorization.

Each script is a PEP 723 program and can run locally with `uv run` or directly on Hugging Face
Jobs. Set `OUTPUT_MODEL` to a Hub repository you control; Jobs are ephemeral, so every recipe
pushes checkpoints and the final adapter to that repository.

## SFT

```bash
hf jobs uv run \
  --flavor a10g-small \
  --timeout 3h \
  --secrets HF_TOKEN \
  --env BASE_MODEL=Qwen/Qwen3-0.6B \
  --env OUTPUT_MODEL=your-org/flavourbench-sft \
  https://raw.githubusercontent.com/josefchen/flavourbench/main/examples/lab/train_sft.py
```

## DPO

```bash
hf jobs uv run \
  --flavor a10g-small \
  --timeout 3h \
  --secrets HF_TOKEN \
  --env BASE_MODEL=Qwen/Qwen3-0.6B \
  --env OUTPUT_MODEL=your-org/flavourbench-dpo \
  https://raw.githubusercontent.com/josefchen/flavourbench/main/examples/lab/train_dpo.py
```

## GRPO

```bash
hf jobs uv run \
  --flavor a10g-large \
  --timeout 4h \
  --secrets HF_TOKEN \
  --env BASE_MODEL=Qwen/Qwen3-0.6B \
  --env OUTPUT_MODEL=your-org/flavourbench-grpo \
  https://raw.githubusercontent.com/josefchen/flavourbench/main/examples/lab/train_grpo.py
```

The examples default to private output repositories. Add `--env PUBLIC_MODEL=1` only when the
trained adapter and its base-model licence are ready for public release. Trackio receives training
and validation metrics through the Transformers integration.

SFT defaults to demonstrations whose optimum beats the runner-up by at least 500 basis points
(five FlavourBench points). Override this with `--env MIN_OPTIMAL_MARGIN_BPS=0` to retain all 342
training rows. DPO already enforces a five-point minimum gap for every chosen/rejected pair. GRPO
uses the full dense reward surface, including near-ties as continuous rather than binary signal.
