# Hugging Face launch kit

This directory contains a self-contained public explorer and a dataset export surface for the
FlavourBench Epicure-native release.

- [`space`](space): a Gradio evidence explorer built around the model-task Pair Lens;
- [`dataset`](dataset): a multi-configuration dataset card, deterministic exporter, and generated
  JSON Lines tables; and
- [`../docs/huggingface-space-plan.md`](../docs/huggingface-space-plan.md): the product and visual
  design brief.

Neither component makes model, provider, or Epicure calls. Both read the checked-in public
release artifact.
