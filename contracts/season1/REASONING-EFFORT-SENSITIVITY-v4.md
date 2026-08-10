# Reasoning-effort sensitivity v4

Status: frozen, verified, unexecuted, non-ranking.

This package replaces the unusable v1–v3 reasoning-effort attempts with a
fail-closed design. It makes no provider or Epicure calls. Full execution is
blocked until every exact endpoint passes a separately reviewed six-pair route
gate.

## Frozen design

- Endpoints: Claude Sonnet 5 on Anthropic, Gemini 3.6 Flash on Google AI
  Studio/Flex, and DeepSeek V4 Flash 0731 on DeepInfra FP4.
- Tasks: eight real development tasks, two per culinary family, drawn from the
  already frozen v27 and v29 panels. No quarantined or synthetic task is used.
- Conditions: Epicure off and Epicure on.
- Effort: 23 reconstructed complete explicit-low pairs plus one immutable low
  failure; 24 new provider-default pairs; 24 new explicit-high pairs.
- New full-study exposure: 48 matched pairs, 96 response arms. The six-pair
  route gate adds 12 diagnostic arms that never enter a quality fit.
- Ordering: exactly 12 blocks default-first and 12 high-first, balanced within
  each panel before outcomes exist.

Provider-default means the reasoning field is absent from every request.
Explicit-high means the field is present and set to `high`. The route gate must
reconstruct these predicates from immutable journals, verify the exact returned
model and provider, reconcile every generation cost, observe zero tool calls in
Epicure-off arms, and observe at least one successful real tool call in every
Epicure-on arm. A single failure closes the remaining suffix; replay is barred.

## Budget and admission

The six-pair gate reserves at most $3.71850672. The 48-pair study reserves at
most $23.26794993333333333333333334. Total new worst-case exposure is
$26.98645665333333333333333334; projected global exposure is
$80.26262584666666666666666667 against the current $85 admission ceiling and
$100 hard cap. A fresh global audit is still required before each pair.

Only the six-pair route gate has a callable paid-execution command. The 48-pair
study command remains unimplemented. A route PASS permits a fresh zero-call
study preflight and review of a separate executor; it does not authorize or
start the study. The route confirmation token is
`RUN_EXACT_REASONING_EFFORT_V4_ROUTE_GATE_6_PAIRS`. The reserved but inactive
full-study token is `RUN_EXACT_REASONING_EFFORT_V4_48_NEW_PAIRS`.

## Freeze and verify

Run from the evaluation-paper repository with the locked Python environment:

```bash
flavourbench/.venv/bin/python -m flavourbench.reasoning_effort_sensitivity_v4 freeze \
  --repo-root . \
  --output-dir flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4
```

Then pass the six exact generated JSON paths to:

```bash
flavourbench/.venv/bin/python -m flavourbench.reasoning_effort_sensitivity_v4 verify \
  --repo-root . \
  --history HISTORY.json --baseline BASELINE.json --route-plan ROUTE.json \
  --study-plan STUDY.json --runner-assets RUNNER.json --preflight PREFLIGHT.json
```

No provider credential, MCP endpoint, or network access is required for either
command. Future paid execution additionally requires fresh OpenRouter and
Cloudflare gateway credentials, reachable private Epicure MCP, exact frozen
endpoint availability, an unchanged source bundle, no active/orphan budget
reservation, and a passing global budget audit.

## Six-pair route gate

The route-gate implementation is
`src/flavourbench/reasoning_effort_route_gate_v4.py`. It accepts only the exact
history, baseline, route-plan, study-plan, runner-assets, and preflight hashes
frozen above. It also reopens the stopped coverage ledger and accepts its
zero-dollar Cohere retirement only when closure `3cb144abd1162447...` and the
source-reconstructing stopped-run audit both verify. Any other orphan or active
reservation blocks admission.

Run the following from the `flavourbench` directory. This is a zero-call
preflight; it neither reads provider credentials nor contacts Epicure:

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.reasoning_effort_route_gate_v4 \
  --repo-root .. plan \
  --history artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/reasoning-effort-v4-history-audit-308ac12ebdf375d83337d55a98a0c5aef055f6cb9b26d74795bf09d14b80b386.json \
  --baseline artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/reasoning-effort-v4-low-baseline-audit-1fce54a13e2f844ae7a5d6b2d6f97eee4a8f37d58520d026c25ebe31cb2970e6.json \
  --route-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/reasoning-effort-v4-route-gate-plan-2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352.json \
  --study-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/reasoning-effort-v4-study-plan-733977cc3eac48316244adcf9beb726824505173b9fe52140cb664ad35d348c0.json \
  --runner-assets artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/reasoning-effort-v4-runner-assets-f4516e382422add2a0a68b17857e7b724090e6b49542158cc2927b6cb8be6ebf.json \
  --preflight artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/reasoning-effort-v4-preflight-a7396f64a4db08dc1eef8425b59eb61f21836bdc5a8c572f12748f6ee3e239f7.json \
  --budget-audit artifacts/season1/current-quality-run/frontier-budget-audits/frontier-global-budget-ec179b7889834d2c6c92343acfb332e907a22600531333e9f0e1f7d7708a241d.json \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/explicit_low \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/provider_default \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/explicit_high \
  --supplemental-run-root artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1 \
  --supplemental-run-root artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1/v2 \
  --supplemental-run-root artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1/v3 \
  --retired-zero-reservation-closure artifacts/season1/current-quality-run/frontier-coverage-continuation-v2/frontier-coverage-orphan-closure-3cb144abd1162447e3e64ba0b703ea09d9ead595d141e2dbf1ffb0103d27e370.json \
  --retired-zero-reservation-audit artifacts/season1/current-quality-run/frontier-coverage-continuation-v2/frontier-coverage-stopped-run-audit-b0990b3b8869325771433cccd8a390a0e48038cf07637cac7ee244a39e9ca4d5.json \
  --source-directory artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/source \
  --ledger artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/ledger.jsonl \
  --global-budget-lock-path artifacts/frontier-contract/ledger.jsonl \
  --global-artifact-directory artifacts/live-smoke \
  --global-corrections-directory artifacts/corrections \
  --global-reconciliation-directory artifacts/frontier-contract/reconciliations \
  --cap-usd 100 --admission-fraction 0.85 \
  --output-directory artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate
```

The current preflight is
`reasoning-effort-v4-route-gate-execution-plan-f8555a1463ef4c07c4b1cca99406bd7a6871d0a9164aad991e6e1121210699ac.json`.
It starts zero calls. It reconstructs $46.47831132666666666666666666 of
current conservative exposure and projects $50.19681804666666666666666666
after the complete $3.71850672 gate, below both the $85 admission ceiling and
the $100 hard cap.

Paid execution uses the identical arguments, replaces `plan` with `execute`,
and appends exactly:

```text
--confirm RUN_EXACT_REASONING_EFFORT_V4_ROUTE_GATE_6_PAIRS
```

The executor takes the shared frontier lock and route-local ledger lock,
re-audits all sources before each one-pair reservation, and processes the six
pairs in their frozen order. A reservation without a verifiable source is never
replayed. A failed source closes the remaining suffix. The execution receipt is
then passed to the `audit` subcommand, followed by `close` and `verify`. The
verifier reopens the receipt, ledger, six live artifacts, six finalized journal
chains, request-semantics projections, provider identities, generation-cost
metadata, and complete MCP traces. A self-consistent summary cannot pass.

## Statistical boundary

The primary unit is the fixed model-by-task block. Aggregate default/high and
low contrasts use paired randomization/sign inference with Holm correction
across three aggregate effort contrasts. Resampling must respect task and model
dependence. With 24 blocks, the test is powered only for a large, consistent
directional shift (81.1% power when the true same-direction probability is
0.8). Model-specific cells (eight blocks) and family cells (six blocks) are
descriptive with intervals. Results cannot rank frontier models, generalize to
unmeasured models or tasks, or enter the primary leaderboard.

The historical low journals preserve request-payload hashes rather than raw
request bodies. Their returned model/provider identities, final responses,
costs, tool traces, artifact hashes, and journal chains are reconstructable;
their original routing-control bodies are not. This limitation is explicit in
the low-baseline audit and is why fresh default/high route qualification is
mandatory.
