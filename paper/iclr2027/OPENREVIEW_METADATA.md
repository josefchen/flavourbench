# OpenReview metadata for ICLR 2027

This file is not included in either anonymous archive. Both authors must approve it before the
abstract deadline.

## Title

FlavourBench: Executable Culinary Reward Maps for Language Model Evaluation and Post-Training

## Authors and affiliations

1. Josef Chen, Independent Researcher
2. Erim Hayretci, Imperial College London

OpenReview profiles, verified emails, conflicts, and affiliations are authoritative. Correct the
profiles rather than adding affiliation text to the anonymous PDF.

## Abstract

Open-ended language-model evaluation often substitutes another model or a small preference panel
for a missing answer key. We introduce FlavourBench, which instead compiles dense answer maps from
a versioned culinary environment. Each task asks for a three-ingredient portfolio from eight
candidates; before inference, Epicure scores all 56 portfolios. We evaluate 27 frontier endpoints
on the same 534 substitution, pairing, and constraint tasks, yielding 14,418 complete model-task
observations. Anchor-cluster bootstraps and multiplicity-controlled paired tests resolve 101 of 351
model contrasts. Grok 4.6 has the largest point estimate (65.1), but the corrected evidence does
not identify a unique best endpoint. The ranking replicates across independently compiled panels
and remains similar under alternative metrics, task filters, family weights, and three public
Epicure checkpoints. We then run a preregistered, three-seed post-training study. LoRA SFT of a
pinned Qwen3-0.6B checkpoint on 270 Epicure-optimal answers improves its score on 84
anchor-disjoint maps by 13.30 points over a format- and label-matched control (95% CI 6.52 to
20.29, p = 0.000170), and the effect replicates on all 534 public maps. The replication gain is
11.73 points (95% CI 8.98 to 14.54). On the primary split, both trained arms parse every response
while the format control does not improve on the base model. The release contains prompts,
exhaustive reward maps, raw responses, training and evaluation manifests, statistical plans, code,
and an offline verifier.

## Suggested keywords

language model evaluation; executable benchmarks; reward maps; culinary reasoning; post-training;
uncertainty quantification; reproducibility

## Suggested subject areas

Choose the closest labels available in the live form, prioritizing language-model evaluation,
reinforcement learning, uncertainty quantification, and applications of machine learning. Do not
select a subject merely to increase reviewer exposure.
