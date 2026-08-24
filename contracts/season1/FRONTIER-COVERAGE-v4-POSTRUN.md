# Frontier coverage v4 postrun and primary on-arm recovery

This runbook covers the source reconstruction after the two terminal v4 phases and the
blocked successor plan. Every command below is offline. None sends a provider request or
opens an Epicure MCP session.

## Observed evidence boundary

The corrected development arena contains 915 candidate comparisons over 192 distinct
Epicure-on response coordinates. Four coordinates are singletons. Seventy-three of 480
model-pair-by-family cells have no support: 3 composition, 20 cookability, 27 evidence,
and 23 substitution. The corrected uplift pool contains 187 matched pairs (374 arms).

These are development review inputs, not quality judgments. The comparison rows are not
independent: inference must cluster by task and response. Failed cells remain in
reliability reporting but do not enter preference or uplift fitting.

The rejected 100-arm off/on draft remains permanently blocked at
`frontier-coverage-residual-v5-plan-e3f96fb63a17e36d43fd33635b43b473ccc477a9e3457c809ceca2b439a8c59d.json`.
It must not be executed.

## Primary successor

The successor freezes 50 fresh Epicure-on arms under one existing content-addressed
policy (`579bef8d…`): three tool rounds, six potential calls per round, twelve total tool
calls, 8,192 intermediate tokens, and 8,192 final tokens. It contains no Epicure-off arm
and no diagnostic 13-by-13 or 16K override. Endpoints are isolated into sixteen batches.

Observed support remains 73 holes. If and only if all 50 new arms are source-reconstructed
usable under the exact policy, the projection is 1,281 comparisons, 242 response
coordinates, and zero support holes. This is a connectivity projection, not an observed
quality result or a family-precision claim.

Primary plan:

`artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/frontier-coverage-primary-on-v5-plan-f79850aaa6a9b256340c2932ae376e6887e387b7bded6ce2ffd06d7caa3dc308.json`

## Cohere continuation gate

The direct Cohere adapter now preserves provider-native assistant content blocks and tool
plans across staged turns while publishing only text blocks. It also replays tool calls
with the exact result ID. Five isolated adapter tests and a separate in-process projection
contract pass without network traffic.

Route gate:

`artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/route-gate/frontier-coverage-primary-cohere-route-gate-a89c319e32ba169645173809b1019a51b549dfdc22cab75f06c4d5718cb8f918.json`

## Rebased budget and admission

The preflight binds the terminal v4 preflight, both v4 phase audits, the complete Sonnet
v6 route-plan/receipt/audit/closure chain, both v6 raw sources, the v6 aggregate closure,
the primary plan, and the Cohere code gate.

- Historical exposure: $47.32616982666666666666666666
- v4 phase 1 actual: $0.424886
- v4 phase 2 actual: $0.024079
- Sonnet v6 actual: $0.244312
- Rebased current exposure: $48.01944682666666666666666666
- Primary worst case: $34.25976628333333333333333333
- Projected exposure: $82.27921310999999999999999999
- Headroom to the $85 admission ceiling: $2.72078689000000000000000001

Budget fit does not grant admission. Independent governance must issue GO, and every
endpoint reservation requires a fresh source and ledger rebase. Only one endpoint batch
may be active. Partial anchor completion cannot be represented as closed support.

Preflight:

`artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/preflight/frontier-coverage-primary-preflight-4b0be120e32f5f8e448742a1411ed48cccf64f0af29c359a28cf0f6a1eaa1797.json`

## Offline verification

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.frontier_coverage_v4_postrun freeze-cohere-gate
PYTHONPATH=src .venv/bin/python -m flavourbench.frontier_coverage_v4_postrun freeze-primary-preflight \
  --cohere-gate artifacts/season1/current-quality-run/frontier-coverage-primary-on-v5/route-gate/frontier-coverage-primary-cohere-route-gate-a89c319e32ba169645173809b1019a51b549dfdc22cab75f06c4d5718cb8f918.json
PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_service_cohere.py \
  tests/frontier_coverage_v4_postrun_test.py
```

There is intentionally no paid execution command in this runbook before governance GO.
