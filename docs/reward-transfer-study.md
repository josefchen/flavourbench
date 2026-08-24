# Epicure reward-transfer study

This protocol asks the question a model lab actually cares about: does optimization against
Epicure reward improve decisions on ingredient anchors that the optimizer never saw?

The study is prospective. Its task split, treatments, seeds, outcomes, and multiplicity family are
committed before training. A later result artifact must cite the exact protocol commit and the
content hash of every input and checkpoint.

## Design

| Component | Prespecified choice |
|---|---|
| Base checkpoints | `Qwen/Qwen3-0.6B`; `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Treatments | Unmodified base, LoRA SFT, LoRA DPO, LoRA GRPO |
| Training seeds | 20260824, 20260825, 20260826 |
| Training maps | 270 tasks: 45 per family-by-panel stratum |
| Model selection | Validation score on 72 tasks: 12 per stratum |
| Primary evaluation | 84 tasks: 14 per stratum, used once after selection |
| Primary outcome | Equal-family FlavourBench Score on the 84-task transfer split |
| Primary contrast | Each trained treatment minus its unmodified base checkpoint |
| Multiplicity family | Six treatment contrasts; Holm control at familywise 0.05 |

All 426 lab anchors are mutually exclusive across train, validation, and transfer evaluation. They
are also disjoint from the 534 leaderboard anchors. The public transfer maps are not cryptographic
secrets; this is a declared held-out protocol. Training code must load only the `sft`, `dpo`, or
`grpo` configurations, which contain train and validation splits but no transfer rows.

## Inference

Each trained condition is evaluated with greedy decoding on all 84 transfer tasks. The treatment
effect is the mean, across three seeds, of the task-paired score difference from the base model.
Uncertainty uses a hierarchical bootstrap: resample the 84 ingredient-anchor tasks within the six
family-by-panel strata, then resample training seeds. The confirmatory p-value uses sign flips of
the per-task mean treatment difference. Holm correction covers the three training methods for both
base checkpoints. Report the raw and adjusted p-values, the 95% interval, standardized paired
effect, per-family effects, and every task-level response.

A treatment is a confirmed transfer improvement only if its adjusted p-value is below 0.05 and the
95% interval for the paired score difference excludes zero. Point gains without both conditions
are estimates, not confirmed improvements.

## Secondary analyses

Secondary results do not alter the primary claim:

1. **Inference effort without Epicure.** For each base checkpoint, sample 1, 4, and 16 independent
   completions per transfer task at temperature 0.7, map the modal valid portfolio to a score, and
   report the paired curve. This tests whether extra inference alone substitutes for reward
   optimization.
2. **Candidate-order robustness.** Evaluate the base and best validation-selected treatment under
   five deterministic label permutations per transfer task. Invert the labels before scoring and
   report mean absolute score drift and exact selection consistency.
3. **Data scaling.** Train the best validation-selected method with 25%, 50%, and 100% of each
   training stratum, preserving the same validation and transfer maps.
4. **Family transfer.** Report substitution, pairing, and constraint effects separately with
   intervals labelled exploratory.

## Execution contract

- Do not inspect transfer outcomes to select checkpoints, prompts, hyperparameters, or methods.
- Keep parse and transport failures in the response artifact. A run is analyzed only after every
  failed cell is retried under the same decoding contract; no task is dropped selectively.
- Record the exact base revision, dependency versions, GPU, seed, duration, Trackio run, adapter
  revision, and SHA-256 of every response and report.
- Publish unsuccessful treatments alongside successful ones. They are part of the six-contrast
  family, not hidden exploratory runs.
- Do not submit a trained checkpoint to the 534-task public leaderboard as a contamination-free
  result. A lab-facing follow-up leaderboard requires a new server-side panel with unseen anchors.

The machine-readable companion is
[`contracts/reward-transfer/reward-transfer-plan-v1.json`](../contracts/reward-transfer/reward-transfer-plan-v1.json).
