# Epicure reward-transfer study

This confirmatory study asks whether Epicure supervision teaches a model anything beyond the
required answer syntax. It compares reward SFT with a format-matched control on ingredient anchors
that appear in neither training nor validation.

The protocol was frozen before generating any transfer outcome. The machine-readable contract is
[`reward-transfer-plan-v2.json`](../contracts/reward-transfer/reward-transfer-plan-v2.json), with
semantic SHA-256
`9256abd79a522898da08406d780a7bf1c06fa50cd156ba1381df42d15bc9e7ec`.

## The comparison

| Component | Frozen choice |
|---|---|
| Base checkpoint | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` |
| Descriptive reference | Unmodified base model |
| Confirmatory control | LoRA SFT on task-mismatched but format- and label-matched completions |
| Confirmatory treatment | LoRA SFT on each task's Epicure-optimal completion |
| Training seeds | 20260824, 20260825, 20260826 |
| Training maps | 270 tasks, 45 per family-by-panel stratum |
| Validation maps | 72 tasks, used for monitoring rather than checkpoint selection |
| Primary evaluation | 84 transfer tasks, 14 per stratum, opened after all six adapters finish |
| Primary outcome | Equal-family score with unparseable model completions retained at zero |
| Primary contrast | Epicure SFT minus format-control SFT, paired within seed and task |
| Public replication | The 534 public leaderboard maps, evaluated once after the primary analysis |

Every train, validation, transfer, and leaderboard anchor is disjoint. Exact candidate sets and
prompt hashes are also disjoint across the three lab splits. The public transfer maps are a declared
holdout rather than a cryptographic secret; the training programs never load them.

## Why the control matters

Ordinary reward SFT also teaches the model to emit `FINAL_SELECTION: A,B,C`. A base-versus-SFT
gain therefore mixes culinary reward learning with output-format learning. The control breaks that
confound.

Within each family and source panel, the builder rotates optimal A--H portfolios onto different
prompts. It chooses the first deterministic rotation with no accidental target optimum. Reward SFT
and control SFT consequently have identical prompts, row counts, completion lengths, and portfolio-
label histograms. Only the alignment between a prompt and its Epicure target changes. The control's
mean training-map score is 28.38, compared with the exact random-portfolio mean of 31.56.

## Training and decoding

Both SFT conditions use all 270 rows, three fixed epochs, LoRA rank 16, learning rate
$10^{-4}$, effective batch size 16, cosine decay, 10% warmup, completion-only loss, and the final
checkpoint. No validation-selected checkpoint is used. Qwen thinking is disabled through the chat
template; evaluation uses greedy decoding and at most 64 new tokens under the same prompt contract
for every condition.

Transport failures are retried without changing decoding. Unparseable completions are model
outcomes, not missing data: they remain in the primary score at zero. Parse rate and score
conditional on parsing are reported separately, so syntax gains cannot masquerade as reward gains.

## Inference

The primary effect is computed for every transfer task and seed, then averaged over the three
matched seeds. A 50,000-draw crossed bootstrap resamples training seeds and ingredient anchors
within the six family-by-panel strata. A two-sided 100,000-draw sign-flip test operates on the
per-anchor seed-mean differences. There is one confirmatory contrast, so no multiplicity correction
is required. The study reports the point effect, 95% interval, raw $p$-value, each seed, all task
scores, and the three family effects.

At 84 paired tasks, a normal-approximation 80%-power sensitivity spans roughly 4.6, 6.1, 7.6, and
9.2 score points when the paired-difference standard deviation is 15, 20, 25, or 30. The 534-task
public replication reduces those values to approximately 1.8, 2.4, 3.0, and 3.6 points. These are
design sensitivities, not assumptions about the outcome.

An effect is statistically resolved when the two-sided $p$-value is below .05 and the 95% interval
excludes zero. A gain of at least three points is additionally labelled practically material. Null
and negative seed results remain in the release.

The execution path is fail-closed and content-addressed:

```bash
python experiments/reward_transfer/unlock_evaluation.py
python experiments/reward_transfer/evaluate.py --split primary
python experiments/reward_transfer/analyze.py --split primary
python experiments/reward_transfer/evaluate.py --split public
python experiments/reward_transfer/analyze.py --split public
```

The unlock step verifies every byte named by all six final-adapter manifests, their shared frozen
protocol, code revision, software stack, hardware, and the audited task hashes. Each analysis RNG
seed is derived from the protocol hash and a fixed analysis label rather than selected after seeing
an outcome. Public-map generation refuses to start until a content-addressed primary analysis
exists. The analyzer reparses and rescores every raw completion before inference.

## Claim boundary

A positive result would show that Epicure-aligned supervision transfers to unseen Epicure maps
beyond answer-format training. It would not establish better human taste, cooked-food quality,
general language-model ability, or reinforcement-learning improvement. DPO and GRPO remain
runnable lab recipes, but they are not part of this single powered confirmatory contrast.

The earlier v1 plan compared six trained conditions only with their unmodified bases. It was
superseded before training because those contrasts did not isolate format learning. Its hash remains
in the v2 contract so the protocol change is auditable without cluttering the paper.
