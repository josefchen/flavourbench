# Reasoning-effort route gate v5

Status: frozen and zero-call preflights passed. No v5 generation or Epicure
call has been made.

## Why v4 stopped

The immutable v4 incident hash resolves exactly to:

```text
current endpoint execution contract differs from the frozen manifest
```

The stopped journal contains only `run_started` and
`openrouter_key_status`. The exception was raised after three OpenRouter
account/catalog GETs, before Epicure attestation, provider construction, or a
chat-completion request. It therefore incurred no generation cost and made no
MCP call. Its $0.6765315 reservation remains conservative exposure, and every
v4 identifier remains permanently closed by closure `807aa054...`.

A subsequent content-addressed catalog snapshot found one raw difference from
the frozen Gemini endpoint: `pricing.input_cache_write` changed from
`0.00000004166666666666667` to `0.0000000416666666666667`. These are not equal
Decimals. Their difference is 3e-23 USD/token, or 3e-17 USD/MTok. This
field-level observation is consistent with the proven hash mismatch, but the
exact in-flight endpoint payload was not persisted by v4; the audit does not
retroactively admit that run.

V5 retains each raw endpoint contract and separately freezes a semantic
contract. Price strings are parsed as Decimals and quantized at 1e-15 USD per
raw unit, equivalent to 1e-9 USD/MTok for token prices. Maximum rounding is
5e-10 USD/MTok, far below the 1e-6 USD generation-accounting resolution.
Provider, model, context, completion limit, capabilities, or any price change
outside this fixed quantum fails before reservation. Tests confirm that a
material price change and removal of `reasoning_effort` both fail.

## Frozen evidence

- Endpoint snapshot: `ce46706dd7c2cb0605c3dd5abc34f36714f09a6074e155b18298393f14a38262`
- V4 pre-request audit: `691bc2b19a36b49984907f15fe0890577b4b25aa7251b6edf6ac1d6960694915`
- V5 route plan: `0481ecd9c8260967275e18a72d4ed265352d35ca2254f554ba55053bc61bb71c`
- Gemini zero-call preflight: `10395305ebf102cfaaa5da701475a2acbe38527486b25c5948d40d49f5d7c189`
- Sonnet zero-call preflight: `595fb093bd424a883df8be03305edc65839bb82917d333892a828a0fad15522b`

The plan accepts the two source-reconstructed DeepSeek pairs from v4 without
replay. Gemini default/high and Sonnet default/high use four fresh work-item,
run, arm, and attempt pools, all disjoint from the complete v4 closure.

Gemini and Sonnet have separate ledgers, source directories, confirmation
tokens, receipts, audits, and closures. A failure closes only that endpoint;
it does not close or block the peer endpoint. The aggregate gate passes only
if both endpoint audits pass alongside the two preserved DeepSeek pairs.

## Budget

| Exposure | USD |
|---|---:|
| Current conservative exposure, including the retained v4 orphan | 47.19792482666666666666666666 |
| Two fresh Gemini pairs | 1.353063 |
| Two fresh Sonnet pairs | 2.297448 |
| Four-pair v5 worst case | 3.650511 |
| Projected after all v5 pairs | 50.84843582666666666666666666 |
| Admission ceiling / hard cap | 85 / 100 |

Each endpoint command takes the shared budget lock, makes a fresh two-GET
metadata attestation before each reservation, reserves one pair at a time,
and reconciles actual generation cost from source metadata. The old orphan is
never released or replayed.

## Exact paid commands

Run from the `flavourbench` directory in the same validated live environment
used by the v4 gate. Either command may run first. They share a global lock and
cannot execute paid requests concurrently.

Sonnet:

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.reasoning_effort_route_gate_v5 \
  --repo-root .. execute-endpoint \
  --route-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/route-gate/reasoning-effort-v5-route-gate-plan-0481ecd9c8260967275e18a72d4ed265352d35ca2254f554ba55053bc61bb71c.json \
  --endpoint sonnet \
  --endpoint-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/sonnet \
  --peer-endpoint-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/gemini \
  --v4-receipt artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/reasoning-effort-v4-route-gate-execution-receipt-172f4a08003656371de69c0907975f83761597338b159031b16052417d575852.json \
  --global-budget-lock artifacts/frontier-contract/ledger.jsonl \
  --confirm RUN_EXACT_REASONING_EFFORT_V5_SONNET_2_PAIRS
```

Gemini:

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.reasoning_effort_route_gate_v5 \
  --repo-root .. execute-endpoint \
  --route-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/route-gate/reasoning-effort-v5-route-gate-plan-0481ecd9c8260967275e18a72d4ed265352d35ca2254f554ba55053bc61bb71c.json \
  --endpoint gemini \
  --endpoint-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/gemini \
  --peer-endpoint-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/sonnet \
  --v4-receipt artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/reasoning-effort-v4-route-gate-execution-receipt-172f4a08003656371de69c0907975f83761597338b159031b16052417d575852.json \
  --global-budget-lock artifacts/frontier-contract/ledger.jsonl \
  --confirm RUN_EXACT_REASONING_EFFORT_V5_GEMINI_2_PAIRS
```

Each command prints the exact content-addressed receipt path. Pass that path to
`audit-endpoint`, then pass the resulting audit to `close-endpoint`. Do this
independently for Gemini and Sonnet. Once both closures exist, run
`aggregate-audit` followed by `aggregate-close`. The aggregate closure alone
can authorize a fresh zero-call preflight for the 48-pair sensitivity study;
it never starts that study and contributes no quality observation itself.

## Verification

```bash
.venv/bin/ruff check \
  src/flavourbench/reasoning_effort_route_gate_v5.py \
  tests/reasoning_effort_route_gate_v5_test.py

PYTHONPATH=src .venv/bin/pytest -q \
  tests/reasoning_effort_route_gate_v4_test.py \
  tests/reasoning_effort_route_gate_v5_test.py
```

Current result: 19 passed, Ruff clean.
