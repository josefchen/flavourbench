# FlavourBench positioning

## Category

**FlavourBench is the executable culinary benchmark for frontier language models.**

It tests ingredient substitution, composition, cookability, and cultural association against
answer keys compiled by a versioned culinary runtime. No human or model judge determines the
leaderboard. A matched second condition checks execution when the same model can call Epicure; it
does not affect rank.

This is the canonical category sentence. Use it consistently in search descriptions, launch posts,
repository summaries, paper metadata, and the first screen of the benchmark Space.

## Naming architecture

| Name | Role | Canonical description |
|---|---|---|
| **FlavourBench** | Benchmark and leaderboard | Executable culinary benchmark for frontier language models |
| **FlavourBench Score** | Official ranking metric | Model-only exact-choice accuracy on the fixed task panel |
| **Epicure Gain** | Secondary diagnostic | Matched change when the same endpoint receives the named Epicure operation; never a rank input |
| **Epicure** | Culinary runtime and operator layer | Versioned system that compiles the answer keys and powers the assisted condition |
| **Epicure Explorer** | Operator demonstration | Interactive view of the culinary representation and operators behind FlavourBench |

FlavourBench is the benchmark brand. Epicure is the underlying system. Do not call the leaderboard
the “Epicure benchmark,” because that hides the independent benchmark identity and makes the score
sound like a product test.

## Message hierarchy

1. **Executable ground truth:** the answer key exists before an evaluated model is called.
2. **Frontier comparison:** 20 current endpoints receive the same 32 tasks and missing-response rule.
3. **Matched execution diagnostic:** 640 model-task pairs expose named-operation integration failures.
4. **Auditability:** prompts, routes, responses, completion states, bounded traces, and hashes replay offline.

The short launch hook is:

> How well do frontier models reason about flavour?

The proof line is:

> 20 current endpoints. 32 executable culinary tasks. 640 matched Model only / Model + Epicure pairs.

## Surface roles

| Surface | Job | Primary call to action |
|---|---|---|
| Paper | Establish the construct, method, evidence, and limitations | Cite and reproduce |
| FlavourBench Space | Make every rank and paired outcome inspectable | Explore the leaderboard and Pair Lens |
| FlavourBench dataset | Provide machine-readable benchmark evidence | Load, audit, and build analyses |
| GitHub repository | Supply exact replay, source, checksums, and release history | Reproduce the result |
| Epicure Explorer | Demonstrate the operator layer behind the answer keys | Explore Epicure, then continue to FlavourBench |

## Claims boundary

Use:

- “the executable culinary benchmark for frontier language models”;
- “20 current language-model endpoints” for this dated release;
- “no human or model judge scores the primary leaderboard”;
- “FlavourBench Score is the only ranking metric, and equal scores share a score rank”;
- “the same endpoint and task are measured with and without Epicure”; and
- “the complete leaderboard reconstructs from one content-addressed artifact.”

Do not use:

- “the largest food benchmark” or “the first culinary benchmark” without a formal systematic review;
- “a general ranking of cooking ability or model intelligence”;
- “human culinary truth,” “food-safety certification,” or “cultural-authenticity certification”;
- “20 independent foundation models” when the measured objects are hosted endpoints; or
- “tool-use benchmark” without explaining that the operation is specified rather than discovered.

## Citation-oriented release loop

1. Keep each benchmark release immutable and give it a stable version, artifact hash, and dated model routes.
2. Put the paper, dataset, Space, and replay command one click apart; each surface should cite the others.
3. Publish task-level records so later papers can analyze calibration, model families, failures, and tool gain.
4. Add larger hidden panels as new releases rather than silently changing the current 32-task snapshot.
5. Invite independently reproduced result artifacts, clearly separated from author-run routes.
6. After an arXiv identifier exists, add the same BibTeX block and canonical URL to every public surface.

## Canonical links

- Paper: <https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf>
- Source and replay: <https://github.com/josefchen/flavourbench>
- Benchmark Space: <https://huggingface.co/spaces/josefchen/flavourbench>
- Dataset: <https://huggingface.co/datasets/josefchen/flavourbench>
- Epicure Explorer: <https://huggingface.co/spaces/Kaikaku/epicure-explorer>
