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
