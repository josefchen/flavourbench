# FlavourBench positioning

## Category

**FlavourBench is an executable culinary benchmark for frontier language models.**

Each task asks a model to select three ingredients from eight. Epicure assigns a score to all 56
legal portfolios before the evaluated model runs, so scoring is deterministic and does not depend
on a human panel or an LLM judge. The benchmark covers substitution, pairing, and constrained
composition.

Use this description across the paper, repository, Hugging Face pages, and research outreach.

## What is being measured

FlavourBench measures how closely an endpoint's selections agree with a fixed, released culinary
reward map. It does not measure universal taste, food safety, complete recipe writing, or general
intelligence.

The primary reward map comes from a content-addressed Epicure runtime whose exact training run and
seed were not recovered. Three immutable public Epicure checkpoints provide a post-collection
reward-map sensitivity analysis. That analysis preserves broad model ordering but remains
conditional on the released prompts and candidate sets.

Recipe1MSubs supplies a separate label-independent check on the substitution geometry of those
public checkpoints. Its user-observed substitution labels were not used to fit Epicure, although
its underlying recipes share Recipe1M ancestry with part of Epicure's corpus. It does not validate
the unrecovered primary runtime or the full pairing and constraint reward.

## Naming

| Name | Role | Description |
|---|---|---|
| **FlavourBench** | Benchmark and leaderboard | Executable culinary evaluation of language-model endpoints |
| **FlavourBench Score** | Primary metric | Equal-family mean of within-task 0 to 100 portfolio scores |
| **Epicure** | Reference environment | Fixed function used to construct candidate reward maps |
| **Public-scorer sensitivity** | Robustness analysis | Rescore of fixed model choices under Cooc, Core, and Chem checkpoints |
| **External substitution check** | Convergent validation | Held-out Recipe1MSubs labels evaluated against public checkpoint geometry |
| **Lab track** | Training and evaluation kit | Anchor-disjoint SFT, DPO, GRPO, validation, and transfer splits |

Epicure is not a leaderboard contestant. A score of 100 means selecting Epicure's optimum on every
task, not that Epicure has 100% culinary accuracy.

## Evidence hierarchy

1. **Shared executable tasks.** Every ranked endpoint has one valid response on the same 534 tasks.
2. **Paired inference.** Simultaneous confidence bands, rank intervals, and Holm-controlled paired
   tests quantify uncertainty rather than treating point ranks as definitive.
3. **Replication and sensitivity.** Two independently compiled panels, score-blind subsampling,
   leave-one-endpoint checks, score-definition checks, and three public reward maps test dependence
   on design choices.
4. **External labels.** Public checkpoint substitution geometry is tested on 1,469 unique
   held-out Recipe1MSubs pairs, with source-clustered intervals and a novel-pair sensitivity set.
5. **Reconstruction.** Prompts, reward maps, responses, routes, hashes, analysis plans, code, and an
   offline verifier reconstruct the published result.

The current proof line is:

> 27 endpoints. 534 shared tasks each. 14,418 scored decisions. 351 paired comparisons.

The statistical qualifier belongs near any rank claim:

> Grok 4.6 has the largest point estimate; overlapping simultaneous intervals do not establish a
> unique best endpoint.

## Claims boundary

Use:

- "an executable culinary benchmark for frontier language models";
- "27 dated language-model endpoints in the current release";
- "no human or model judge scores the leaderboard";
- "a fixed, released culinary reward map";
- "all ranked endpoints face the same 534 tasks";
- "the full leaderboard reconstructs from content-addressed artifacts"; and
- "three public Epicure checkpoints preserve the same point leader in a fixed-task sensitivity
  analysis"; and
- "public Epicure checkpoints rank held-out, human-observed substitutions above same-food-group
  chance alternatives."

Do not use:

- "the first" or "the largest" without a documented systematic review;
- "culinary ground truth," "human culinary truth," or "universal taste";
- "statistically significant winner" when leading simultaneous intervals overlap;
- "27 independent foundation models" when the observations are dated endpoints;
- "human validated" or "cooking validated";
- public-checkpoint sensitivity alone as evidence of external construct validity;
- "corpus-independent validation" for the Recipe1MSubs analysis; or
- any claim that a model's culinary rank establishes its general model quality.

## Surface roles

| Surface | Purpose | Main action |
|---|---|---|
| Paper | Define the construct, design, evidence, and limitations | Read and cite |
| Hugging Face Space | Inspect ranks, intervals, tasks, and model profiles | Explore or score a run |
| Hugging Face dataset | Load task, response, robustness, and training records | Analyze or train |
| GitHub repository | Rebuild the release and run a new endpoint | Reproduce or evaluate |
| Lab track | Test whether Epicure rewards improve held-out culinary decisions | Run the protocol |

## Release discipline

1. Keep each release immutable and bind it to semantic and physical hashes.
2. Put the paper, dataset, Space, source, and exact replay command one click apart.
3. Publish task-level records and uncertainty, not only a sorted score table.
4. Add new models or task panels through a new versioned release.
5. Separate independently submitted runs from author-run routes.
6. Keep the same title, author list, claims, citation, and artifact identifiers on every surface.

## Canonical links

- Paper: <https://arxiv.org/abs/2608.20574>
- Source: <https://github.com/josefchen/flavourbench>
- Benchmark Space: <https://huggingface.co/spaces/josefchen/flavourbench>
- Dataset: <https://huggingface.co/datasets/josefchen/flavourbench>
- Epicure Explorer: <https://huggingface.co/spaces/Kaikaku/epicure-explorer>
