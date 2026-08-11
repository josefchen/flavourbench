---
pretty_name: "FlavourBench: Executable Culinary Evaluation of Frontier Language Models"
license: other
language:
- en
task_categories:
- text-classification
- question-answering
tags:
- leaderboard
- llm-evaluation
- tool-use
- culinary
- reproducibility
- frontier-models
size_categories:
- 1K<n<10K
configs:
- config_name: models
  data_files:
  - split: train
    path: data/models.jsonl
- config_name: tasks
  data_files:
  - split: test
    path: data/tasks.jsonl
- config_name: observations
  data_files:
  - split: test
    path: data/observations.jsonl
- config_name: paired_outcomes
  data_files:
  - split: test
    path: data/paired_outcomes.jsonl
- config_name: leaderboard
  data_files:
  - split: test
    path: data/leaderboard.jsonl
---

# FlavourBench: Executable Culinary Evaluation of Frontier Language Models

FlavourBench evaluates 20 frontier language-model endpoints against executable culinary answer
keys computed before evaluation. This dataset contains the complete 32-task release used by the
paper and leaderboard, including both Model only and Model + Epicure conditions.

It is the machine-readable evidence layer for evaluating culinary reasoning across the current
frontier panel; it is not a general model-intelligence or food-safety ranking.

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) ·
[Interactive explorer](https://huggingface.co/spaces/josefchen/flavourbench) ·
[Source and reproduction](https://github.com/josefchen/flavourbench)

## Configurations

| Config | Rows | Unit |
|---|---:|---|
| `models` | 20 | One evaluated model route |
| `tasks` | 32 | One exact-choice Epicure-native task |
| `observations` | 1,280 | One assigned model-task-condition arm |
| `paired_outcomes` | 640 | One matched Model only/Model + Epicure pair |
| `leaderboard` | 20 | One scored model summary |

## Key fields

- `task_id`, `model_id`, and `condition` form the observation key.
- `correct` records exact-choice correctness.
- `parseable_normal_completion` distinguishes observed answers from endpoint or protocol failures.
- `response_artifact_sha256`, `prompt_sha256`, and reference-result hashes preserve lineage.
- `paired_outcome` is one of `both_correct`, `off_only`, `on_only`, `neither`, or `incomplete`.
- `epicure_benchmark_score` stores the FlavourBench Score: Model only exact-choice accuracy.
- `rank` is the shared score rank; `release_order` preserves the source artifact's deterministic
  row order without treating an assisted metric as evidence that tied scores differ.
- `uplift_percentage_points` stores the secondary matched change with Epicure available. It is an
  integration diagnostic and does not enter score rank.

## Intended use

Use the dataset to reproduce the public automated leaderboard, inspect task-level errors, study the
secondary model-Epicure interaction, or build alternative visualizations. The primary leaderboard
uses only `epicure_benchmark_score`; equal scores share a displayed score rank. The 32-task release
should not be used as a general ranking of model intelligence. One task changes the score by 3.125
points, and leading Wilson intervals overlap.

## Availability and missingness

The release preserves assigned arms even when an endpoint did not produce a parseable normal
completion. This is essential for distinguishing service availability from demonstrated model
capability. In particular, a zero score with zero observed responses is not evidence that the
underlying model has zero capability.

## Licensing and rights

Benchmark prompts, choices, identifiers, authored metadata, derived tables, and original figures
are released under CC BY 4.0. Model responses are included as research records but do not grant
rights in provider software, weights, services, or marks. The underlying Epicure ingredient data,
embeddings, and unrestricted payloads are not redistributed. See the repository
[`LICENSES.md`](https://github.com/josefchen/flavourbench/blob/main/LICENSES.md) for the complete
component-level boundary.

## Reproduction

```bash
python hf/dataset/build_dataset.py --check
python -I paper/reproduce_epicure_native.py \
  --release paper/generated/epicure-native/epicure-native-release.json
```

The exporter verifies the release semantic hash and every generated table byte-for-byte.
