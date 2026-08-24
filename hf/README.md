# Hugging Face launch kit

This directory contains the two Hugging Face surfaces for FlavourBench, the executable culinary
reasoning benchmark for frontier language models, scored without a human or model judge.

- [`space`](space): the branded leaderboard, statistical insights, evidence inspector, runner,
  upload scorer, comparison view, and reward API;
- [`dataset`](dataset): the complete evaluation matrix, pairwise evidence, official task maps, and
  anchor-disjoint SFT, DPO, and GRPO views; and
- [`../docs/huggingface-space-plan.md`](../docs/huggingface-space-plan.md): the product and visual
  publication specification.

Neither component makes model-provider or remote Epicure calls. Both read content-addressed
release artifacts. Provider credentials and model weights stay in the lab's own environment.
