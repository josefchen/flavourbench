---
title: FlavourBench
emoji: 🍲
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: other
datasets:
- josefchen/flavourbench
---

# FlavourBench

**Josef Chen · Erim Hayretci**<br>
Josef Chen, Independent Researcher · Erim Hayretci, Imperial College London

This Space explores the final FlavourBench complete common core: 27 frontier endpoints, 534
identical tasks per model, 14,418 valid scored responses, and all 351 paired comparisons.

Epicure scores all 56 legal three-ingredient portfolios before a model is called. The Space lets
you inspect the resulting leaderboard without relying on an LLM judge. It includes:

- the full score table with simultaneous 95% intervals and bootstrap rank intervals;
- statistical rank groups and all Holm-adjusted pairwise comparisons;
- family profiles and score replication across two independently compiled panels; and
- exact prompts, candidate lists, Epicure score maps, model answers, and response hashes.

Every ranked endpoint has one valid response for every task. The Space bundle is content-addressed
and makes no provider calls.

## Evaluate your own model

The **Evaluate your model** tab scores a complete JSON or JSONL response artifact without receiving
model credentials or weights. A public FlavourBench lab score is issued only when all 534 responses
are present and parseable; incomplete runs retain coverage and per-task diagnostics. Uploads are not
automatically added to the official leaderboard.

Each JSONL row follows this compact contract:

```json
{"task_id":"...","status":"completed","response":"FINAL_SELECTION: A,B,C"}
```

The Space also exposes named Gradio endpoints:

- `/score_completion` performs one deterministic reward-map lookup;
- `/score_submission` scores JSON or JSON Lines supplied as text;
- `/training_reward` scores only the 426 non-leaderboard development maps; and
- `/score_uploaded_submission` powers the file-upload interface.

Use **Use via API** in the running Space for generated Python, JavaScript, and curl examples. The
local SDK is preferable for high-throughput RL because it avoids network latency and Space rate
limits. The Space exposes released FlavourBench reward maps, not the private Epicure corpus or a
general arbitrary-task generation service.

## Train on separate reward maps

The linked dataset now includes anchor-disjoint development splits for SFT, DPO, and GRPO. These
training maps do not reuse any task ID or ingredient anchor from the 534-task official test set.
Runnable LoRA recipes and the local dense reward function live in the source repository under
`examples/lab`.

Josef Chen is a Cohere Labs Catalyst Grant recipient. This acknowledgement does not imply Cohere
endorsement of FlavourBench, Epicure, the protocol, or any model ranking.

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) |
[Dataset](https://huggingface.co/datasets/josefchen/flavourbench) |
[Source](https://github.com/josefchen/flavourbench)

## Citation

```bibtex
@article{chen2026flavourbench,
  title  = {FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth},
  author = {Chen, Josef and Hayretci, Erim},
  year   = {2026}
}
```
