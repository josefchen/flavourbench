# Reasoning-effort sensitivity: governed family-block execution

This is a development-only sensitivity study. It is not an official model
ranking and must not enter the Season 0 leaderboard.

## Frozen inputs

- Executor plan: `artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2/plan/reasoning-effort-task-wave-plan-v2-7c22c5bac5063158bb46627a2253a686b2c0d8fb2ea70b7340b0ee53d1eb9d74.json`
- Plan SHA-256: `7c22c5bac5063158bb46627a2253a686b2c0d8fb2ea70b7340b0ee53d1eb9d74`
- Human protocol: `artifacts/season1/current-quality-run/reasoning-effort-human-protocol-v1/reasoning-effort-human-protocol-bafcd88ec629b389550e573b68ad9874a342d462681b18e6492c83891d81f258.json`
- Human protocol SHA-256: `bafcd88ec629b389550e573b68ad9874a342d462681b18e6492c83891d81f258`
- Bound admission preflight: `artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2/bound-preflight/reasoning-effort-bound-admission-preflight-v2-5bef4adc2caa34a26fce0192784d95ce232099283cf5f17434f94d6358c581a8.json`
- Bound preflight SHA-256: `5bef4adc2caa34a26fce0192784d95ce232099283cf5f17434f94d6358c581a8`

The pre-protocol preflight is intentionally non-executable. Only the bound
preflight above verifies the plan, the independently frozen human protocol,
336 response-arm coordinates, and 240 comparison cells together.

## Design and admission

The 24 tasks are ordered into six atomic blocks. Every block contains four
task waves—one substitution, one composition, one cookability, and one
evidence task—and therefore 28 matched Epicure off/on pairs (56 real response
arms). A budget stop can occur only between these balanced blocks.

The first block reserves USD 17.56585256. Against the source-reconstructed
USD 48.01944682666666666666666666 baseline, its conservative projected
exposure is USD 65.58529938666666666666666666. New blocks stop at USD 85;
the hard cap remains USD 100.

One coordinator reservation covers the entire block. There are no pair-level
reservations. The shared frontier budget lock remains held until all 28 pairs
have terminal dispositions. Completed reconciled sources replace the block
reserve with actual cost. A provider request without a reconciled immutable
source retains the full block reserve and stops execution. Identifiers are
never replayed or replaced.

## Governance-review command

Run from the `flavourbench` directory only after governance explicitly clears
the three frozen artifacts and the live environment is present. Do not place
credentials on the command line.

```bash
.venv/bin/python -m flavourbench.reasoning_effort_full_study_executor_v1 \
  --plan artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2/plan/reasoning-effort-task-wave-plan-v2-7c22c5bac5063158bb46627a2253a686b2c0d8fb2ea70b7340b0ee53d1eb9d74.json \
  --human-protocol artifacts/season1/current-quality-run/reasoning-effort-human-protocol-v1/reasoning-effort-human-protocol-bafcd88ec629b389550e573b68ad9874a342d462681b18e6492c83891d81f258.json \
  --bound-preflight artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2/bound-preflight/reasoning-effort-bound-admission-preflight-v2-5bef4adc2caa34a26fce0192784d95ce232099283cf5f17434f94d6358c581a8.json \
  --confirm RUN_REASONING_EFFORT_V2_ONE_COMPLETE_FAMILY_BLOCK \
  --max-new-family-blocks 1
```

The command first re-verifies all frozen inputs and the source-backed budget,
then performs six zero-generation catalog GETs. It sends no provider
completion or Epicure request unless every endpoint still matches its frozen
semantic contract and the whole block remains admissible.

## Claim boundary

The tasks are real, non-synthetic development tasks, but they are explicitly
not confirmatory. The study estimates endpoint-specific low-to-high reasoning
effects for Sonnet, Gemini, and DeepSeek. Gemini's omitted provider default is
a secondary medium-effort dose level. The 24 tasks—not arms, ballots, or reused
responses—are the independent clusters. Epicure's
`exploratory-unmatched-1790-runtime` payload is not yet independently
reconstructable, so this study cannot cure the separate lineage hold.
