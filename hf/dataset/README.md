---
pretty_name: "FlavourBench: Executable Culinary Ground Truth"
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
    path: data-complete-core/models.jsonl
- config_name: tasks
  data_files:
  - split: test
    path: data-complete-core/tasks.jsonl
- config_name: primary_observations
  data_files:
  - split: test
    path: data-complete-core/primary_observations.jsonl
- config_name: leaderboard
  data_files:
  - split: test
    path: data-complete-core/leaderboard.jsonl
- config_name: pairwise_comparisons
  data_files:
  - split: test
    path: data-complete-core/pairwise_comparisons.jsonl
---

# FlavourBench: executable culinary ground truth

**Josef Chen · Jakub Radzikowski · Erim Hayretci**<br>
Independent Researchers

FlavourBench ranks 27 language-model endpoints on an identical set of 534 culinary selection
tasks. Each task asks for three ingredients from eight candidates. Epicure scores all 56 legal
portfolios before any model is called, so each response is evaluated against a fixed score map
rather than a human or model judge.

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) ·
[Interactive leaderboard](https://huggingface.co/spaces/josefchen/flavourbench) ·
[Source](https://github.com/josefchen/flavourbench)

## Dataset contents

| Config | Rows | Unit |
|---|---:|---|
| `models` | 27 | Model score, uncertainty, exact route, family scores, and panel replication |
| `tasks` | 534 | Prompt, candidates, all 56 Epicure scores, family, anchor, and panel |
| `primary_observations` | 14,418 | One selected source response for each model-task cell |
| `leaderboard` | 27 | Point rank, rank group, score interval, and route |
| `pairwise_comparisons` | 351 | One paired contrast with sign-flip and Holm-adjusted inference |

`primary_observations` stores the original content-addressed response document, the exact source
path, the release panel, and parser-v3 scoring used by the final analysis. The selected matrix is
complete: every model has one valid, parseable response for every ranked task.

## Score and inference

The FlavourBench Score is the equal-weight mean of substitution, pairing, and constraint scores.
Each family contains 178 tasks. A task score lies between 0 and 100 and is determined by the
position of the chosen portfolio on that task's Epicure score map.

The release uses 534 ingredient-anchor clusters, 50,000 shared cluster-bootstrap replicates,
simultaneous 95% score intervals, 100,000 cluster sign flips for all model pairs, Holm correction,
exact-chance tests, and bootstrap rank intervals. There are 101 Holm-significant pairwise
differences. The leading intervals overlap, so the release does not claim one statistically unique
top model.

## Common-core construction

The original two panels contained 640 tasks each. For every panel and included family, the analysis
retains tasks with a completed parser-v3-valid response from all 27 models, orders those tasks by a
fixed hash rule, and takes the first 89. The selection rule uses status and parseability only. It
does not inspect scores or selected ingredients. The resulting common core has 267 tasks per panel
and 534 tasks overall.

Regional composition remains in the full development evidence but is not part of this ranked
common-core estimand because it could not supply the same balanced all-model cell.

## Integrity

`data-complete-core/DATA_MANIFEST.json` binds each table by byte size and SHA-256. It also binds the
final statistical release and frozen analysis plan.

- Dataset manifest: `54331a825da40ab90c8e13fc971d47fe4eb94e5f23cf87ec70131fe5e3807e05`
- Release semantic ID: `0a20655c97aa1363c2266e247f3dd03b759d0f80bca9154c6619c5549b2fac99`
- Release file SHA-256: `709452f8cf54ebc1947f2a3c24e6ee19580be1c115ba3a9effbac441de556db4`
- Analysis plan semantic ID: `2ba71c793c8d4b97eed863ee83fd770b429fdefdffebdeafb241672f634ee507`

Verify the downloaded export without provider access:

```bash
python3 -I hf/dataset/verify_complete_core_dataset.py \
  --dataset-directory hf-release/data-complete-core
```

To reconstruct the ignored source-response tree for a full analysis rebuild, run the checked
restorer and then the paper targets:

```bash
python3 -I hf/dataset/restore_complete_core_sources.py \
  --dataset-directory hf-release/data-complete-core \
  --repository . \
  --restore
make -C paper -f Makefile.powered analysis assets verify
```

These commands make no provider calls.

## Interpretation and rights

A score of 100 means that a model chose Epicure's optimum on every task. Epicure is the reference
environment and is not a ranked model. These scores measure culinary portfolio selection under the
released reward surface, not general intelligence, food safety, or sensory preference.

Prompts, candidate sets, derived tables, and original figures are CC BY 4.0. Model responses retain
their provider terms. The underlying Epicure ingredient data and embeddings are not redistributed.
See the repository [rights boundary](https://github.com/josefchen/flavourbench/blob/main/LICENSES.md).

## Citation

```bibtex
@article{chen2026flavourbench,
  title  = {FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth},
  author = {Chen, Josef and Radzikowski, Jakub and Hayretci, Erim},
  year   = {2026}
}
```
