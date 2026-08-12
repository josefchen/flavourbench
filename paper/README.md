# FlavourBench

**FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth**

FlavourBench is an executable 20-model benchmark of culinary reasoning across current frontier
language-model endpoints. A versioned Epicure runtime generates 640 three-of-eight selection
tasks and dense score maps across four balanced families:

- substitution portfolios;
- ingredient pairing;
- dietary and processing constraints; and
- regional composition.

Every model answers the same 640 tasks without access to Epicure. Epicure exhaustively scores all
56 candidate portfolios before model execution. The release contains 12,800 primary responses
and 1,280 label-permuted repeats.

## Score

The public model rank uses the **FlavourBench Score**:

```text
FlavourBench Score = equal-family mean of frozen per-task portfolio scores
```

Each task contributes a continuous value from 0 to 100. Missing, abnormal, or unparseable
responses score zero. Every family contributes equally. No human or LLM judge is used; Epicure
computes the complete reward surface before any evaluated endpoint is called.

## Panel

The release covers GPT-5.6 Sol/Terra/Luna; Claude Opus/Sonnet 5; Gemini 3.1 Pro and 3.6 Flash;
Grok 4.5; Llama 4 Maverick; Kimi K3; Qwen 3.8 Max; GLM 5.2; DeepSeek V4 Pro and Flash;
MiniMax M3; Nemotron 3.5 Lightning; Mistral Large 3; Tencent HY 3; Cohere Command A; and Cohere
Command R+ (08-2024). Every scored block uses a pinned OpenRouter provider endpoint with fallback
disabled. Superseded route-calibration responses are retained separately and never pooled.

## Build and replay

From this directory:

```bash
make -f Makefile.powered verify
make -f Makefile.powered paper
make -f Makefile.powered arxiv
```

`paper` and `arxiv` compile the checked, content-addressed tables and figures and therefore work
from a clean Git checkout. `verify` additionally regenerates those assets and requires the raw
response layer restored from the Hugging Face dataset. The dependency-free compact replay is:

```bash
python3 -I reproduce_powered_release.py \
  --release generated/powered/flavourbench-powered-release-<sha256>.json
```

It checks the release content address, complete 20-model result, all 190 pairwise rows, 20 repeat
rows, and the committed leaderboard and pairwise table hashes without provider access. The raw
response release on Hugging Face contains every prompt and response; the statistical release,
tables, figures, and manuscript are generated from the same frozen analysis.

## Outputs

- `build/flavourbench.pdf`: canonical paper PDF.
- `build/flavourbench-arxiv-source.tar.gz`: self-contained arXiv source archive.
- `build/ARTIFACTS.sha256`: PDF and archive checksums.
- `generated/powered/`: statistical release, tables, case studies, and macros.
- `figures/powered/`: vector leaderboard, family, pairwise, and repeatability figures.

The source archive contains the manuscript, generated tables, vector figures, references,
licenses, and an exact source manifest. The larger machine-readable response release is
distributed through the linked Hugging Face dataset. Credentials, internal deployment material,
and superseded exploratory analyses are excluded.
