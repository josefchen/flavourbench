# OpenReview metadata for ICLR 2027

This file is not included in either anonymous archive. Both authors must approve it before the
abstract deadline.

## Title

FlavourBench: Executable Culinary Reward Maps for Language Model Evaluation

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
not identify a unique best endpoint. Rankings replicate across two independently compiled panels
and remain similar under alternative score definitions, task filters, family weights, and three
public Epicure checkpoints. On 1,469 mapped Recipe1MSubs substitutions, those public checkpoints
place the human-observed target at mean within-food-group percentiles of 0.780 to 0.806, with
source-clustered intervals above chance. The anonymous release contains prompts, exhaustive reward
maps, raw responses, route identities, statistical plans, and an offline verifier, together with
disjoint maps for prospective reward-based training studies.

## Suggested keywords

language model evaluation; executable benchmarks; reward maps; culinary reasoning; uncertainty
quantification; reinforcement learning environments; reproducibility

## Suggested subject areas

Choose the closest labels available in the live form, prioritizing language-model evaluation,
reinforcement learning, uncertainty quantification, and applications of machine learning. Do not
select a subject merely to increase reviewer exposure.
