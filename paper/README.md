# FlavourBench

**FlavourBench: Executable Culinary Evaluation of Frontier Language Models**

FlavourBench is an executable 20-model benchmark of culinary reasoning across current frontier
language-model endpoints. A versioned Epicure runtime generates
32 four-choice tasks and their exact answer keys across four balanced families:

- nearest-neighbour substitution;
- exact ingredient-pairing affinity;
- flavour-axis comparison; and
- cuisine-direction projection.

Every model answers every task twice: model only, then with one read-only call to a named Epicure
operation. The model-only arm produces the ranking score. The assisted arm is a secondary
integration diagnostic. Together they produce 640 matched model-task pairs and 1,280 responses.

## Score

The public model rank uses the **FlavourBench Score**:

```text
FlavourBench Score = 100 × Model only correct answers / 32
```

Because all four families contain eight tasks, this is also exact accuracy over the common
32-task panel. Missing, abnormal, or unparseable responses score zero. Equal scores share a score
rank; model name orders rows within a tie. Assisted results never break score ties.

The paired diagnostic also records:

```text
Epicure Gain = accuracy(Model + Epicure) − accuracy(Model only)
```

No human or LLM judge is used. Epicure computes the answer key before any model call. Because the
assisted prompt names the answer-generating operation, its near-ceiling accuracy is expected and
must not be read as a second leaderboard.

## Panel

The release covers GPT-5.6 Sol/Terra/Luna; Claude Fable/Opus/Sonnet 5; Gemini 3.1 Pro and 3.6
Flash; Grok 4.5; Kimi K3; Qwen 3.8 Max; GLM 5.2; DeepSeek V4 Pro and Flash; MiniMax M3;
Nemotron 3 Ultra; Mistral Large 2512; Tencent HY 3; and Cohere Command A Plus and Command A
Reasoning. Kimi and both Cohere models use direct provider routes. Other models use exact
no-fallback OpenRouter routes recorded in the release.

The direct-route replay contract requires 64 observed arms each for Cohere Command A Plus and
Command A Reasoning, with returned IDs `command-a-plus-05-2026` and
`command-a-reasoning-08-2025`. A missing, substituted, or rerouted Cohere model fails verification.

## Build and replay

From this directory:

```bash
make verify
make paper
make arxiv
```

The dependency-free replay is:

```bash
python3 -I reproduce_epicure_native.py \
  --release generated/epicure-native/epicure-native-release.json
```

It checks the content address and full 20 × 32 × 2 observation grid, then recomputes every score,
family result, completion value, matched change, and source order. The paper displays shared
score-only ranks. Tables and vector figures are built from the same release by
`build_epicure_native_assets.py`.

## Outputs

- `build/flavourbench.pdf`: canonical paper PDF.
- `build/flavourbench-arxiv-source.tar.gz`: self-contained arXiv source archive.
- `build/ARTIFACTS.sha256`: PDF and archive checksums.
- `generated/epicure-native/`: public release, tables, case studies, and macros.
- `figures/epicure-native/`: vector score, interval, family, paired-diagnostic, response-time, and
  16:9 social figures.

The source archive contains the manuscript, public 1,280-arm release, scorer/replay code, task
compiler, tests, figures, and licenses. It excludes credentials, internal deployment material,
unrelated governance material, and superseded exploratory analyses.
