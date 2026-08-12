---
pretty_name: "FlavourBench: Executable Culinary Reasoning"
license: other
language:
- en
task_categories:
- text-classification
- question-answering
tags:
- leaderboard
- llm-evaluation
- culinary
- reproducibility
- frontier-models
- executable-ground-truth
size_categories:
- 10K<n<100K
configs:
- config_name: models
  data_files:
  - split: train
    path: data-powered/models.jsonl
- config_name: tasks
  data_files:
  - split: test
    path: data-powered/tasks.jsonl
- config_name: primary_observations
  data_files:
  - split: test
    path: data-powered/primary_observations.jsonl
- config_name: repeat_observations
  data_files:
  - split: test
    path: data-powered/repeat_observations.jsonl
- config_name: leaderboard
  data_files:
  - split: test
    path: data-powered/leaderboard.jsonl
- config_name: pairwise_comparisons
  data_files:
  - split: test
    path: data-powered/pairwise_comparisons.jsonl
---

# FlavourBench: Executable Culinary Reasoning

**Josef Chen · Jakub Radzikowski · Erim Hayretci**

FlavourBench evaluates 20 frontier language-model endpoints on 640 culinary decisions with an
executable answer surface. Each task asks for a three-ingredient portfolio from eight candidates.
Before any evaluated model is called, Epicure scores all 56 portfolios. A response therefore earns
graded credit from a fixed table rather than from another language model.

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) ·
[Interactive explorer](https://huggingface.co/spaces/josefchen/flavourbench) ·
[Source and reproduction](https://github.com/josefchen/flavourbench)

## Release at a glance

| Config | Rows | Unit |
|---|---:|---|
| `models` | 20 | One exact evaluated route and statistical summary |
| `tasks` | 640 | One task with eight candidates and all 56 frozen scores |
| `primary_observations` | 12,800 | One model--task response; failures remain as zeroes |
| `repeat_observations` | 1,280 | One label-permuted repeat response |
| `leaderboard` | 20 | One model score, uncertainty interval, and rank group |
| `pairwise_comparisons` | 190 | One paired contrast with Holm-adjusted inference |

The FlavourBench Score is the equal-family mean of the 640 task scores. The four equally weighted
families are substitution, pairing, dietary constraints, and regional composition. Invalid or
failed responses remain in the denominator at zero. A model must complete at least 608 primary
tasks to be rank-eligible.

## Statistical outputs

The release reports 50,000 family-stratified shared-task bootstrap replicates, simultaneous 95%
score bands, all 190 paired model contrasts with 100,000 sign-flip resamples and Holm correction,
exact-chance tests, bootstrap rank intervals, statistical rank groups, and label-permutation
repeatability. Point ranks and rank groups are both included because a numerical ordering can be
finer than the data support.

## What is included

- exact prompts, candidate labels, ingredient names, and all frozen portfolio scores;
- complete primary and repeat response records, including answer text, parsed selection, score,
  completion state, exact route identity, latency, token use, cost, and content hashes;
- route-level model metadata and clean-source lineage for the 17 base blocks, the DeepSeek rerun,
  and the two Cohere reruns;
- the complete leaderboard and every multiplicity-adjusted pairwise comparison; and
- a manifest binding every table by row count, byte size, and SHA-256.

## Interpretation

Epicure is the benchmark environment, not a contestant. A score of 100 means that a model selected
Epicure's optimum on every task. It does not mean that Epicure has been assigned 100% accuracy.
FlavourBench measures agreement with this published culinary reward surface. External culinary
validity is a separate scientific question, just as simulator fidelity is separate from agent
performance in an embodied benchmark.

The primary release is fully automated; no human or model judge determines the leaderboard. The
fixed tasks, common response format, intention-to-evaluate accounting, and shared-task inference
make the release useful for comparing systems on this domain. It should not be read as a universal
ranking of general intelligence or food safety.

## Reproduction

The repository contains the task compiler, route manifests, frozen analysis plans, scorer,
statistical analysis, paper asset builder, and this dataset exporter. The response files on this
dataset are the large evidence layer omitted from GitHub.

```bash
git clone https://github.com/josefchen/flavourbench.git
cd flavourbench
pip install -e '.[dev]'

hf download josefchen/flavourbench --repo-type dataset \
  --include 'data-powered/*' --local-dir hf-release

python -I hf/dataset/restore_powered_runs.py \
  --release paper/generated/powered/flavourbench-powered-release-<sha256>.json \
  --primary hf-release/data-powered/primary_observations.jsonl \
  --repeat hf-release/data-powered/repeat_observations.jsonl \
  --base-run benchmark/powered-v31/run \
  --deepseek-run benchmark/powered-v33/run \
  --cohere-run benchmark/powered-v35/run

make -C paper -f Makefile.powered analysis assets arxiv
```

The restore step accepts only the complete 12,800 + 1,280 content-addressed response grid and
refuses conflicts. Re-running it with `--check` verifies every restored file byte-for-byte.

## Citation

```bibtex
@article{chen2026flavourbench,
  title   = {FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth},
  author  = {Chen, Josef and Radzikowski, Jakub and Hayretci, Erim},
  year    = {2026}
}
```

## Licensing and rights

Benchmark prompts, choices, identifiers, authored metadata, derived tables, and original figures
are released under CC BY 4.0. Model responses are research records and do not grant rights in
provider software, weights, services, or marks. The underlying Epicure ingredient data and
embeddings are not redistributed. See
[`LICENSES.md`](https://github.com/josefchen/flavourbench/blob/main/LICENSES.md) for the
component-level boundary.
