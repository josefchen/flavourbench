# Hugging Face launch kit

This directory contains the two Hugging Face surfaces for FlavourBench, the executable culinary
reasoning benchmark for frontier language models, scored without a human or model judge.

- [`space`](space): a Gradio evidence explorer built around the model-task Pair Lens;
- [`dataset`](dataset): a multi-configuration dataset card, deterministic exporter, and generated
  JSON Lines tables; and
- [`../docs/huggingface-space-plan.md`](../docs/huggingface-space-plan.md): the product and visual
  design brief.

Neither component makes model, provider, or Epicure calls. Both read the checked-in public
release artifact.
