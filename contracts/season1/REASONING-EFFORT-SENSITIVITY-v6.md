# Reasoning-effort route gate v6

Status: the Sonnet-only successor is frozen, but execution is quarantined. Its
zero-call preflight was admissible against the exposure observed at freeze
time. A concurrent coverage-repair run subsequently began changing global
exposure and ledgers, so that preflight is no longer an execution authority.
Do not run the paid command below until a terminal coverage closure and a
superseding, transactionally safe budget admission are frozen.

## Closed v5 failure

Sonnet v5 failed after the provider returned nine tool calls in the required
first tool-selection turn. The frozen client accepted at most six calls per
round, so it raised:

```text
ProviderError: provider tool-call fan-out (9) exceeded the per-round cap (6)
```

The exception hash is
`fb1dc9d8ef84ec08a83e22fa9f74700732f7981c473f54d32884b6e81d0cb007`.
The source and hash-chained journal establish the boundary precisely: five
provider requests were accepted and reconciled, one Epicure session was
started and attested, and no MCP call started. Complete provider cost was
$0.061742. The v5 ledger recorded $0.033475 because it finalized the completed
off arm but not the incomplete on arm; the final v5 receipt correctly retained
the full source-derived cost. No provider, endpoint, or Epicure service outage
is inferred from this incident.

The v5 default and high identifiers remain permanently closed. V6 neither
replays nor relabels them.

## V6 protocol change

V6 changes one client-side acceptance rule:

| Field | V5 | V6 |
|---|---:|---:|
| Maximum tool calls in one selection turn | 6 | 13 |
| Maximum tool calls over the arm | 12 | 13 |

Thirteen equals the frozen Epicure catalog size, permitting at most one full
catalog sweep. Provider request fields, the pinned Anthropic endpoint, task,
prompt, Epicure release, tool schema, number of tool rounds, output limits,
retry policy, and the 65,536-byte cumulative tool-result cap are unchanged.
The existing $1.148724 per-pair reserve therefore remains conservative: the
number of provider turns is unchanged and the cumulative tool-result envelope
does not grow.

The bridge audit source-reconstructs the two preserved DeepSeek v4 cells and
the two passed Gemini v5 cells. Their maximum observed per-turn fan-outs were
below six and their total tool calls were below twelve. The client fan-out
limits were absent from all provider request contracts. The 13/13 widening is
therefore non-binding for those four route cells. This is a route-compatibility
bridge only; it does not convert their outputs into quality observations.

## Frozen packet

- V5 failure audit:
  `f308f5b5fc57ce6f1d9c52b0e0f21f653843b96fd3ae295c21fe6ba7a3320c34`
- Fan-out bridge audit:
  `9d389ac19fff5a57d801c3ee076f38276793413c260b47cf135601ab441a81f4`
- V6 manifest:
  `052a214a4d1358ca80aba3612949c3aa1177907924a0f0c62df588424301eef1`
- V6 route plan:
  `905f41ba1cd50915d6aa8fc11f5f582930e045e9dc0586dae98549ad21fa6a2c`
- Zero-call execution plan:
  `4091db95f115d79aa454821aa0700941284dd1a8795e787c5db7d2a405121d54`

Both v6 work items have fresh route-cell, work-item, run, arm, and attempt
identifiers. The freeze validates disjointness against the complete DeepSeek
v4, Gemini v5, and Sonnet v5 closures. Each on arm preallocates 13 MCP slots in
each of three possible rounds, while the runtime policy independently enforces
13 calls total.

## Budget

The baseline at freeze time was the final Gemini v5 receipt, which already
included the complete Sonnet v5 source cost. These figures are historical
freeze values, not current admission values while coverage repair is active.

| Exposure | USD |
|---|---:|
| Current conservative exposure | 47.32616982666666666666666666 |
| Two fresh Sonnet v6 pairs, worst case | 2.297448 |
| Projected exposure | 49.62361782666666666666666666 |
| Admission ceiling / hard cap | 85 / 100 |

The worker takes the shared budget lock, attests endpoint metadata before each
reservation, reserves one pair at a time, reconciles every generation from the
source artifact, and stops permanently on the first failed cell.

## Withdrawn paid command

Do **not** run this command from the historical preflight. It is retained so
the frozen executor interface is inspectable. A superseding admission packet
must bind terminal coverage source/closure evidence and the global ledger
before this or any successor command may be authorized.

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.reasoning_effort_route_gate_v6 \
  --repo-root .. execute \
  --route-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v6/route-gate/reasoning-effort-v6-sonnet-route-gate-plan-905f41ba1cd50915d6aa8fc11f5f582930e045e9dc0586dae98549ad21fa6a2c.json \
  --root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v6/sonnet \
  --baseline-receipt artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/gemini/receipts/reasoning-effort-v5-gemini-receipt-157e3aaeb8faf02830c927ddbe035dcb7414900cf900c59bd23db57bf918b803.json \
  --global-budget-lock artifacts/frontier-contract/ledger.jsonl \
  --confirm RUN_EXACT_REASONING_EFFORT_V6_SONNET_2_PAIRS
```

The command prints the content-addressed receipt. Audit that exact receipt,
then close its identifiers:

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.reasoning_effort_route_gate_v6 \
  --repo-root .. audit \
  --route-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v6/route-gate/reasoning-effort-v6-sonnet-route-gate-plan-905f41ba1cd50915d6aa8fc11f5f582930e045e9dc0586dae98549ad21fa6a2c.json \
  --root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v6/sonnet \
  --receipt PATH_PRINTED_BY_EXECUTE

PYTHONPATH=src .venv/bin/python -m flavourbench.reasoning_effort_route_gate_v6 \
  --repo-root .. close \
  --route-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v6/route-gate/reasoning-effort-v6-sonnet-route-gate-plan-905f41ba1cd50915d6aa8fc11f5f582930e045e9dc0586dae98549ad21fa6a2c.json \
  --audit PATH_PRINTED_BY_AUDIT
```

## Scientific boundary

The route gate uses one development task and has no blinded judgments. It can
establish fixed-route execution, reasoning-request semantics, real Epicure tool
use, identity integrity, and cost reconciliation. It cannot estimate whether
default or high effort improves answer quality.

The passed Gemini and preserved DeepSeek cells may later support separate,
model-specific paired sensitivity estimates once a frozen multi-task study and
blinded judgments exist. Such estimates should report model-specific effects
and intervals without pooling disconnected comparison graphs. No aggregate
cross-model reasoning-effort effect is supported by the route gate.

## Verification

```bash
.venv/bin/ruff check \
  src/flavourbench/reasoning_effort_route_gate_v6.py \
  tests/reasoning_effort_route_gate_v6_test.py

PYTHONPATH=src .venv/bin/pytest -q \
  tests/reasoning_effort_route_gate_v4_test.py \
  tests/reasoning_effort_route_gate_v5_test.py \
  tests/reasoning_effort_route_gate_v6_test.py
```
