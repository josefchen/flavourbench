<div align="center">

# FlavourBench

### Ranking frontier language models with executable culinary ground truth

[![Public release integrity](https://github.com/josefchen/flavourbench/actions/workflows/ci.yml/badge.svg)](https://github.com/josefchen/flavourbench/actions/workflows/ci.yml)

**Josef Chen** · **Erim Hayretci**<br>
Josef Chen, Independent Researcher · Erim Hayretci, Imperial College London

**27 models · 534 identical tasks per model · 14,418 scored responses · 351 paired tests**

[Paper](https://arxiv.org/abs/2608.20574) · [PDF](paper/build/flavourbench.pdf) · [Leaderboard](https://huggingface.co/spaces/josefchen/flavourbench) · [Dataset](https://huggingface.co/datasets/josefchen/flavourbench) · [arXiv source](paper/build/flavourbench-arxiv-source.tar.gz)

</div>

FlavourBench measures culinary decision quality without a human panel or an LLM judge. Each task
presents eight ingredients and asks a model to choose three. Before any model is called, Epicure
scores all 56 legal portfolios. The model's choice receives a continuous score from 0 to 100 on
that fixed reward surface.

The ranked release uses an identical 534-task common core for every endpoint. It contains two
independently compiled panels and three balanced families: substitution, pairing, and culinary
constraints. Every one of the 14,418 model-task cells is complete and parseable. Failed calls,
content filters, and model-specific task subsets do not enter this leaderboard.

## Result

Grok 4.6 has the highest point estimate, followed by Gemini 3.1 Pro and GPT-5.6 Sol Pro. The leading
simultaneous confidence intervals overlap, so the data do not identify one statistically unique
winner. Point ranks and rank groups answer different questions and both are reported.

| Rank | Model | Score | Simultaneous 95% CI | Bootstrap rank 95% | Group |
|---:|---|---:|---:|---:|:---:|
| 1 | Grok 4.6 | 65.07 | 60.98-69.15 | 1-5 | 1 |
| 2 | Gemini 3.1 Pro Preview | 64.95 | 60.80-69.10 | 1-6 | 1 |
| 3 | GPT-5.6 Sol Pro | 64.23 | 60.09-68.37 | 1-8 | 1 |
| 4 | Muse Spark 1.2 | 63.75 | 59.63-67.88 | 1-10 | 1 |
| 5 | GPT-5.6 Terra Pro | 63.67 | 59.53-67.81 | 1-11 | 1 |
| 6 | Claude Fable 5 | 63.36 | 59.18-67.54 | 2-13 | 1 |
| 7 | GPT-5.6 Luna Pro | 62.64 | 58.52-66.77 | 4-14 | 1 |
| 8 | Claude Opus 5 | 62.50 | 58.39-66.61 | 3-15 | 1 |
| 9 | Qwen3.8 2.4T A95B | 62.08 | 57.93-66.24 | 4-17 | 1 |
| 10 | Kimi K3 | 62.05 | 57.92-66.18 | 5-16 | 1 |
| 11 | Gemini 3.6 Flash | 61.98 | 57.72-66.24 | 4-17 | 1 |
| 12 | DeepSeek V4 Pro 0813 | 61.95 | 57.80-66.11 | 5-17 | 1 |
| 13 | Qwen3.8 Max | 61.50 | 57.36-65.65 | 7-18 | 1 |
| 14 | Hy3 | 61.48 | 57.29-65.67 | 6-19 | 1 |
| 15 | MiniMax M3 | 60.94 | 56.78-65.10 | 8-20 | 1 |
| 16 | GLM-5.3 | 60.57 | 56.36-64.79 | 8-20 | 1 |
| 17 | Muse Glimmer 30B | 59.89 | 55.86-63.91 | 12-21 | 2 |
| 18 | Seed 2.1 Turbo | 59.72 | 55.44-64.01 | 12-22 | 2 |
| 19 | Inkling | 59.61 | 55.44-63.78 | 12-22 | 2 |
| 20 | Claude Sonnet 5 | 59.52 | 55.39-63.65 | 13-22 | 2 |
| 21 | GLM 5.2 | 58.45 | 54.19-62.71 | 16-23 | 2 |
| 22 | Nemotron 3.5 Lightning | 57.39 | 53.23-61.56 | 19-25 | 2 |
| 23 | Command A | 56.71 | 52.48-60.94 | 20-25 | 2 |
| 24 | DeepSeek V4 Flash 0731 | 55.43 | 51.12-59.75 | 22-26 | 2 |
| 25 | Mistral Large 3 2512 | 55.42 | 51.35-59.48 | 22-26 | 2 |
| 26 | Llama 4 Maverick | 53.65 | 49.61-57.70 | 24-26 | 3 |
| 27 | Command R+ (08-2024) | 47.86 | 43.73-51.98 | 27-27 | 3 |

All models score above their taskwise exact-chance baselines after Holm correction. Of the 351
pairwise model contrasts, 101 remain significant after familywise correction. Each contrast uses
the same 534 paired cells. Cross-panel agreement is 0.885 by Pearson correlation and 0.804 by
Spearman correlation.

## The FlavourBench Score

For task \(t\), Epicure supplies a score for each legal three-item portfolio. The chosen portfolio
is normalized to a 0-100 scale between that task's worst and best portfolio. A model's final score
is the equal-weight mean of its substitution, pairing, and constraint means across both panels.

A score of 100 means that the model chose Epicure's optimum on every task. Epicure is the reference
environment, not a contestant. The benchmark measures alignment with a published culinary reward
surface; it does not claim to rank general intelligence or sensory taste.

## Run your own model

The public lab kit evaluates an OpenAI-compatible endpoint, a vLLM server, a local Transformers
checkpoint, or an existing JSONL response artifact. It never asks for provider keys in a result
file, and it refuses to issue a comparable score unless every task is present and parseable.

```bash
pip install 'epicure-flavourbench @ git+https://github.com/josefchen/flavourbench.git'

export LAB_MODEL_API_KEY='...'
flavourbench run \
  --backend openai-compatible \
  --base-url https://your-endpoint.example/v1 \
  --api-key-env LAB_MODEL_API_KEY \
  --model your-exact-model-id \
  --responses responses.jsonl \
  --report report.json
```

For a local or Hub checkpoint:

```bash
pip install 'epicure-flavourbench[transformers] @ git+https://github.com/josefchen/flavourbench.git'
flavourbench run --backend transformers --model your-org/your-model
```

Already have outputs? Each JSONL row needs only a task ID and answer:

```json
{"task_id":"fb-executable-substitution-136","response":"FINAL_SELECTION: A,F,H"}
```

```bash
flavourbench score responses.jsonl --output report.json
flavourbench verify-report report.json
```

See the complete [lab evaluation contract](docs/lab-kit.md).

## Train with Epicure rewards

The Hugging Face dataset now exposes three training-ready configurations:

| Config | Train | Validation | Interface |
|---|---:|---:|---|
| `sft` | 270 | 72 | Optimal demonstrations with optimum margins |
| `dpo` | 1,080 | 288 | Deterministic preferences with gaps of at least 5 points |
| `grpo` | 270 | 72 | Prompt plus complete local reward map |

All 426 lab anchors are disjoint from the 534 leaderboard anchors. Train, validation, and the
84-task predeclared transfer split are mutually anchor-disjoint and balanced by family and source
panel. The transfer maps are public rather than hidden, so the protocol relies on declared split
discipline; optimizer-facing configs omit them. The [runnable Hugging Face Jobs recipes](examples/lab)
cover SFT, DPO, and GRPO with LoRA, Trackio, evaluation, checkpointing, and Hub persistence.
Training on either the transfer split or the public leaderboard maps is outside the protocol.

The prospective [Epicure reward-transfer study](docs/reward-transfer-study.md) freezes two base
checkpoints, three training methods, three seeds, six multiplicity-controlled contrasts, inference
effort scaling, and candidate-order robustness before any transfer result is inspected.

The Hugging Face Space also exposes named Gradio API endpoints for official evaluation and a
separate `training_reward` endpoint restricted to the 342 train/validation maps. Local
reward lookup remains the recommended path for high-throughput RL.

## Statistical design

- 27 models and exactly 534 valid scored tasks per model
- 178 tasks per family, split evenly across two independently compiled panels
- 534 ingredient-anchor clusters, the unit used for uncertainty
- 50,000 shared cluster-bootstrap replicates
- simultaneous 95% score intervals and bootstrap rank intervals
- 100,000 cluster sign flips for each of 351 paired contrasts
- Holm correction across the full pairwise family
- exact taskwise chance tests for every model
- crossed-design relative-decision generalizability of 0.936
- 5,000 balanced, score-blind subsamples at five smaller task counts

The common core was selected using completion and parseability only. No score or observed selection
was inspected during task selection. This removes the missing-cell ambiguity that affected earlier
development releases while retaining a fixed, shared estimand for all models.

The generalizability model estimates 329 balanced tasks for relative reliability 0.90. At 270
tasks, the median rank correlation with the complete point order is 0.952, but the point leader is
preserved in only 46.1% of subsets. The score is stable before the exact winner is; this is why the
release shows simultaneous intervals and bootstrap rank ranges beside the point order.

## Reproduce the release

Python 3.12 is recommended.

```bash
git clone https://github.com/josefchen/flavourbench.git
cd flavourbench
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

make ci

make -C paper -f Makefile.powered verify
make -C paper -f Makefile.powered arxiv
cd paper/build
sha256sum --check ARTIFACTS.sha256
```

The compact release verifier has no provider dependency:

```bash
python3 -I paper/verify_complete_core_release.py \
  --release paper/generated/complete-core/flavourbench-complete-core-release-0a20655c97aa1363c2266e247f3dd03b759d0f80bca9154c6619c5549b2fac99.json
```

The Hugging Face dataset carries the 14,418 selected source responses and the exact task records:

```bash
hf download josefchen/flavourbench --repo-type dataset \
  --include 'data-complete-core/*' --local-dir hf-release

python3 -I hf/dataset/verify_complete_core_dataset.py \
  --dataset-directory hf-release/data-complete-core

python3 -I hf/dataset/restore_complete_core_sources.py \
  --dataset-directory hf-release/data-complete-core \
  --repository . \
  --restore
```

No reproduction command calls a model provider.

## Release artifacts

| Artifact | SHA-256 |
|---|---|
| Statistical release | `709452f8cf54ebc1947f2a3c24e6ee19580be1c115ba3a9effbac441de556db4` |
| Release semantic ID | `0a20655c97aa1363c2266e247f3dd03b759d0f80bca9154c6619c5549b2fac99` |
| Analysis plan file | `17ac5aea6eb25a0c0af440124849c926fdcafaf36956fd2e676f2c70ca80faa6` |
| Lab training dataset | `257dfaf17c4f529f2f9b538c0c0b7d7d8ea030262f75ecf06284b61658a64137` |
| Task-count stability analysis | `4b359ac51db465c7a3f49fb5567a624b1ce3ad6280d309f31545e17ff2797026` |
| Final PDF | `aec6b96193e046519e27234278eb33c8f52dce852dfd1d79f5fd5e4cf0ee23ed` |
| arXiv source tarball | `3bab563508060a73ae657ec7a3ee681568dc9b72862d08ce0ff6a2c08a80fb41` |

## Repository map

| Path | Contents |
|---|---|
| [`src/flavourbench`](src/flavourbench) | Task construction, route contracts, response parsing, scoring, and inference |
| [`benchmark`](benchmark) | Frozen task sets, route manifests, analysis plans, and compact evidence |
| [`paper`](paper) | Manuscript, figures, tables, PDF, and arXiv source package |
| [`hf/dataset`](hf/dataset) | Deterministic benchmark and training-dataset builders |
| [`hf/space`](hf/space) | Leaderboard, evidence explorer, upload scorer, and reward API |
| [`examples/lab`](examples/lab) | Runnable SFT, DPO, and GRPO recipes |
| [`tests`](tests) | Statistical, route, integrity, and publication tests |

## Citation

```bibtex
@article{chen2026flavourbench,
  title  = {FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth},
  author = {Chen, Josef and Hayretci, Erim},
  journal = {arXiv preprint arXiv:2608.20574},
  year   = {2026},
  eprint = {2608.20574},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2608.20574}
}
```

Prompts, candidate sets, derived tables, and original figures are released under CC BY 4.0.
Provider responses retain their source terms. See [`LICENSES.md`](LICENSES.md) for the component
rights boundary and [`SECURITY.md`](SECURITY.md) for credential handling.
Original software is Apache-2.0. Josef Chen is a Cohere Labs Catalyst Grant recipient; this
acknowledgement does not imply Cohere endorsement of the benchmark, methods, or rankings.
