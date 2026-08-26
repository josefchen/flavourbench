# arXiv v2 metadata

The live arXiv record is still version 1. Use this metadata with
`build/flavourbench-arxiv-source.tar.gz` when replacing it. Confirm that the arXiv preview and
metadata show exactly two authors before finalizing the replacement.

## Title

FlavourBench: Executable Culinary Reward Maps for Language Model Evaluation and Post-Training

## Authors

1. Josef Chen, Independent Researcher
2. Erim Hayretci, Imperial College London

## Abstract

Open-ended language-model evaluation often substitutes another model or a small preference panel
for a missing answer key. We introduce FlavourBench, which instead compiles dense answer maps from
a versioned culinary environment. Each task asks for a three-ingredient portfolio from eight
candidates; before inference, Epicure scores all 56 portfolios. We evaluate 27 frontier endpoints
on the same 534 substitution, pairing, and constraint tasks, yielding 14,418 complete model-task
observations. Anchor-cluster bootstraps and multiplicity-controlled paired tests resolve 101 of 351
model contrasts. Grok 4.6 has the largest point estimate at 65.1, but the corrected evidence does
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

## Comments

18 pages, 8 figures. Evaluation of 27 frontier language-model endpoints on 534 identical tasks per
model, comprising 14,418 scored model-task cells. Adds reward-map sensitivity, selection and metric
robustness, held-out Recipe1MSubs substitution validation, and a preregistered controlled
reward-transfer study. Code, dataset, and interactive leaderboard links remain unchanged.

## Subjects

Keep the current subjects unless arXiv moderation advises otherwise: Artificial Intelligence
(cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), and Software Engineering (cs.SE).

## Upload checks

1. Run `make -C paper -f Makefile.powered arxiv`.
2. Run `(cd paper/build && sha256sum --check ARTIFACTS.sha256)`.
3. Upload `paper/build/flavourbench-arxiv-source.tar.gz` as a replacement, not as a new paper.
4. Replace the title, abstract, and comments with the text above.
5. Confirm the preview is 18 pages and lists Josef Chen and Erim Hayretci in that order.
6. After announcement, confirm the live record exposes version 2 and that its PDF title, abstract,
   author list, code link, dataset link, and leaderboard link agree with this repository.
