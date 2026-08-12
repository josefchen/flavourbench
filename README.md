<div align="center">

# FlavourBench

### Ranking frontier language models with executable culinary ground truth

**Josef Chen · Jakub Radzikowski · Erim Hayretci**

**20 models · 640 tasks · 12,800 primary responses · 1,280 repeats · 190 paired tests**

[Paper](paper/build/flavourbench.pdf) · [Leaderboard](https://huggingface.co/spaces/josefchen/flavourbench) · [Dataset](https://huggingface.co/datasets/josefchen/flavourbench) · [Reproduce](#reproduce-the-release)

</div>

FlavourBench is an automated benchmark for culinary reasoning. Every model solves the same 640
three-of-eight ingredient-selection tasks. Before model execution, a versioned Epicure runtime
scores all 56 possible portfolios for each task, producing a fixed continuous reward surface.
Models never see Epicure at inference time, and no language model or human grades their answers.

The public release covers current frontier systems from OpenAI, Anthropic, Google, xAI, Alibaba,
Moonshot, Zhipu, DeepSeek, Meta, MiniMax, NVIDIA, Mistral, Tencent, and Cohere. It ships the complete
response grid, repeat panel, route identities, statistical analysis, paper, and provider-free
reproduction path.

## Result

Gemini 3.1 Pro Preview has the highest point estimate at **72.90**, followed by Grok 4.5 at
**72.40** and GPT-5.6 Sol Pro at **71.99**. The leading confidence bands overlap, so the release
reports both point ranks and statistical rank groups instead of manufacturing a unique winner.

| Rank | Model | FlavourBench Score | Simultaneous 95% band | Group | Repeat agreement |
|---:|---|---:|---:|:---:|---:|
| 1 | Gemini 3.1 Pro Preview | 72.90 | 69.71–76.10 | 1 | 80.8% |
| 2 | Grok 4.5 | 72.40 | 69.24–75.56 | 1 | 78.7% |
| 3 | GPT-5.6 Sol Pro | 71.99 | 68.75–75.23 | 1 | 86.3% |
| 4 | GPT-5.6 Terra Pro | 71.48 | 68.27–74.70 | 1 | 79.8% |
| 5 | Qwen3.8 Max | 71.41 | 68.10–74.71 | 1 | 79.5% |
| 6 | Claude Opus 5 | 70.76 | 67.55–73.96 | 1 | 74.8% |
| 7 | Kimi K3 | 70.31 | 67.08–73.54 | 1 | 74.2% |
| 8 | Gemini 3.6 Flash | 70.08 | 66.87–73.28 | 1 | 79.5% |
| 9 | Claude Sonnet 5 | 69.44 | 66.19–72.68 | 1 | 79.1% |
| 10 | GPT-5.6 Luna Pro | 69.16 | 65.92–72.40 | 2 | 71.9% |
| 11 | DeepSeek V4 Pro | 69.03 | 65.66–72.39 | 2 | 70.6% |
| 12 | Tencent HY3 | 68.89 | 65.60–72.18 | 2 | 52.0% |
| 13 | GLM 5.2 | 66.88 | 63.57–70.18 | 2 | 66.6% |
| 14 | MiniMax M3 | 66.73 | 63.41–70.05 | 2 | 70.5% |
| 15 | Nemotron 3.5 Lightning | 65.15 | 61.71–68.60 | 2 | 75.6% |
| 16 | DeepSeek V4 Flash | 62.69 | 59.25–66.12 | 3 | 55.8% |
| 17 | Mistral Large 3 | 61.40 | 57.96–64.84 | 3 | 64.1% |
| 18 | Llama 4 Maverick | 60.17 | 56.72–63.63 | 3 | 52.7% |
| 19 | Cohere Command A | 58.11 | 54.90–61.32 | 4 | 57.2% |
| 20 | Cohere Command R+ | 36.67 | 33.73–39.61 | 5 | 23.0% |

All 20 systems complete the preregistered eligibility floor. All 20 score above the taskwise exact
chance baseline after Holm correction, and **102 of 190** paired model contrasts remain significant
after multiplicity correction. Scores use 50,000 family-stratified shared-task bootstrap replicates;
pairwise tests use 100,000 shared-task sign flips. Label-permuted repeats measure response stability.

![FlavourBench leaderboard with simultaneous confidence bands](paper/figures/powered/powered-leaderboard-forest.png)

## What the score means

For each task, Epicure assigns a utility to every legal three-ingredient portfolio. The model's
chosen portfolio is placed on a 0–100 task scale between the task's worst and best portfolio. The
**FlavourBench Score** is the equal-family mean across substitution, pairing, dietary constraints,
and regional composition. Invalid, failed, or unparseable responses remain in the denominator at
zero.

Epicure is the executable environment, not a contestant. A score of 100 means that a model selected
Epicure's optimum on every task; it does not mean that Epicure was awarded 100% by itself. The
benchmark measures decision quality against a published, versioned culinary reward surface.

## Why it is useful

- **Dense, deterministic supervision.** Every task has 56 scored actions, not one brittle answer key.
- **No judge-model circularity.** The leaderboard does not depend on another LLM's preferences.
- **Common-task inference.** Models are compared on the same tasks with paired uncertainty.
- **Modern routes.** Qwen3.8 Max, Kimi K3, Cohere Command A, GPT-5.6 Sol/Terra/Luna, and the rest of
  the panel are measured in the same clean lineage with provider fallback disabled.
- **Training-ready rewards.** The exhaustive action maps support reward modeling, reranking, search,
  and held-out reinforcement-learning experiments without leaking the public test set.
- **Auditability.** Prompts, candidate sets, scores, raw responses, failures, routes, seeds, and
  content hashes are distributed as explicit tables.

## Reproduce the release

Python 3.12 is recommended. The compact statistical release is checked into GitHub and verifies
without network access:

```bash
git clone https://github.com/josefchen/flavourbench.git
cd flavourbench
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

python3 -I paper/reproduce_powered_release.py \
  --release paper/generated/powered/flavourbench-powered-release-4bd53e485cd48dbb65b86c9ddd3ff4ad26e34069dd2e163a5b67d2bb47d4f7df.json

make -C paper -f Makefile.powered arxiv
cd paper/build && sha256sum --check ARTIFACTS.sha256
```

To reconstruct the release from every raw response, download the larger evidence layer from
Hugging Face and restore the three content-addressed run directories:

```bash
hf download josefchen/flavourbench --repo-type dataset \
  --include 'data-powered/*' --local-dir hf-release

python3 -I hf/dataset/restore_powered_runs.py \
  --release paper/generated/powered/flavourbench-powered-release-4bd53e485cd48dbb65b86c9ddd3ff4ad26e34069dd2e163a5b67d2bb47d4f7df.json \
  --primary hf-release/data-powered/primary_observations.jsonl \
  --repeat hf-release/data-powered/repeat_observations.jsonl \
  --base-run benchmark/powered-v31/run \
  --deepseek-run benchmark/powered-v33/run \
  --cohere-run benchmark/powered-v35/run

make -C paper -f Makefile.powered analysis assets arxiv
```

The restore path validates the complete 12,800 + 1,280 grid and refuses conflicts. No reproduction
command calls a model provider.

## Repository map

| Path | Contents |
|---|---|
| [`src/flavourbench`](src/flavourbench) | Task construction, route contracts, runners, scoring, and inference |
| [`benchmark`](benchmark) | Content-addressed task sets, plans, manifests, and compact run evidence |
| [`paper`](paper) | Manuscript, generated tables and figures, final PDF, and arXiv source tarball |
| [`hf/dataset`](hf/dataset) | Six-table Hugging Face dataset exporter and strict run restorer |
| [`hf/space`](hf/space) | Interactive evidence explorer and compact verified bundle |
| [`tests`](tests) | Statistical, route, integrity, reproduction, and publication tests |

## Scientific scope

FlavourBench ranks models on constrained culinary selection under the released Epicure reward
surface. It is not a universal intelligence, food-safety, sensory-quality, or full-recipe benchmark.
Endpoint results bind the exact routes and collection period in the release. The paper reports
unresolved top groups where the data do not support a sharper claim.

## Citation

The arXiv identifier will be added after submission. Until then, cite the paper and powered release
`4bd53e485cd48dbb65b86c9ddd3ff4ad26e34069dd2e163a5b67d2bb47d4f7df`:

```bibtex
@article{chen2026flavourbench,
  title   = {FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth},
  author  = {Chen, Josef and Radzikowski, Jakub and Hayretci, Erim},
  year    = {2026}
}
```

## Licensing and security

Benchmark prompts, choices, identifiers, authored metadata, derived tables, and original figures
are CC BY 4.0. Model responses retain the provider and research-use boundaries in
[`LICENSES.md`](LICENSES.md). Credentials, private databases, and unrestricted Epicure source data
must never be committed; see [`SECURITY.md`](SECURITY.md).
