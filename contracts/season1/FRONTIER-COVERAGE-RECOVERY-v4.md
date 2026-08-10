# Frontier coverage recovery v4

## Scope and immutable parent outcome

The v2/v3 continuation is permanently closed. Its receipt
`788ae734229a27aa7efeade9c4c60f95d106e0efda9b7408122cbdab64e1a0d9`,
closure
`b18eb9eb94bdc6d251cb5f33c5c94b64795c18fa50e56bacc233d38e6cc144c4`,
and source-reconstructed audit
`4ac6e792d06f832e9e0215a47b28087f6b0ad91bd5de745b7e5f683c2eabc008`
remain authoritative. They record one complete Nemotron substitution cell, one
incomplete GLM substitution cell, and seven cells that were never reserved.
Recovery v4 neither reopens those identifiers nor supersedes their reliability
outcomes.

The frozen v4 plan is
`730b426cfa5b7481446b4618166a2e6f75107c52ec26243283ef10ccbe01c0b8`.
It contains eight new development-only matched pairs and zero synthetic arms.
All use alternate, non-quarantined tasks for which the model had no recorded
provider exposure at freeze time.

## Failure isolation and execution order

Phase 1 contains the seven previously untouched cells, under fresh work, arm,
run, and attempt identifiers. The first six cells are non-GLM cells. The
previously untouched GLM composition cell is seventh. Each cell is finalized
or receives its own terminal no-replay incident; failure in one cell does not
stop or invalidate later unrelated cells.

Phase 2 contains a separate GLM substitution replacement. It has its own
confirmation and ledger and cannot start until a source-reconstructing phase-1
audit proves that all seven phase-1 cells received terminal dispositions.
Success in either phase cannot erase the original GLM failure. Neither phase
is eligible for an official preference or uplift fit.

The exact worst-case reservations are:

- phase 1: USD 3.763091066666666666666666667;
- phase 2: USD 0.3419004;
- both phases: USD 4.104991466666666666666666667.

The current zero-call admission receipt is
`c2cf6aa4d6397f6034114dfb9ead0b446895256a7a84705fbd3a55c70d742268`.
It records current shared exposure of
USD 47.32616982666666666666666666 and projected exposure of
USD 51.43116129333333333333333333 against the USD 85 admission ceiling and
USD 100 hard cap. It made no provider or MCP calls. The earlier receipts
`62a87d8b73b98a6773a4ca0fae730e30f66af95fc5032bc9b702807e9ffcfeb9`
and
`de093d7b6fa34f2932d6d02669dde669262f5b501a0bd61662ad4916a6000e72`
and
`9df5c9e7c5195dfc69258af4be09ffab97f28aef99e9c6a8f692404089ca3914`
and
`5b219789f1d3862b0652ba573d449fb905ab851db2eb2bdda4c2a5e938b88592`
are stale and must not be used for execution.

The first invocation under preflight `5b219789...` stopped at the local
hash-chain guard because the recovery reservation payload supplied the
ledger-owned `schema_version` field. The guard rejected the event before it
could append. The phase directory contained only the zero-byte lock file:
there was no ledger, reservation, source directory, response directory,
journal, receipt, closure, or audit. No provider subprocess or MCP boundary
was reached, and no frozen work, run, arm, or attempt identifier was started.
The corrected code leaves all protected hash-chain fields to the ledger. The
current preflight embeds this pre-reservation failure inventory, and a
regression test appends the corrected payload to an isolated temporary ledger.

The current receipt includes the reasoning-effort v4 route-gate root. Its
Gemini reservation
`db19e86ac60a9fa9d0c34a7787b7b383e4aa2b3ec30eec4006628ffd7e8a4e26`
remains charged at USD 0.6765315. It is not released or subtracted. The
reservation is classified as terminal no-replay only after independently
rebuilding audit
`c90617d7b6a8cab918bf0f50f7190f8ad8f49badb5ce036c7c9fa716d7d9a959`,
verifying closure
`807aa054e7f0aaaa770630adae7696bba8fc24251d7ed2b08082b46a0edfde87`,
checking the exact incident and error hashes, and verifying that the orphan's
journal contains only `run_started` and `openrouter_key_status`. There is no
provider request, generation, MCP session, or MCP tool call for that work
item. Execution repeats these checks and every bound ledger-head check. If any
evidence or concurrent governed ledger changes, the receipt fails closed;
rerun the no-call preflight rather than bypassing it.

The receipt also reconstructs both reasoning-effort v5 endpoint runs from
their raw sources, ledgers, endpoint attestations, receipts, audits, and
closures. Gemini passed both pairs. Sonnet failed its first pair after a real
provider tool-call fan-out exceeded the frozen cap; its second pair was not
started. Closure `99c19496...` permanently closes both Sonnet identifier
pools, and closure `44ba45a5...` closes both Gemini pools. Aggregate closure
`e6ce615d...` records that the sensitivity gate failed, the full sensitivity
study is not admitted, every v5 identifier is closed, and replay is forbidden.
That failed diagnostic gate does not block these separate recovery cells.

Budget accounting uses the complete reconciled source costs: USD 0.061742 for
Sonnet and USD 0.066503 for Gemini, or USD 0.128245 total. The v5 aggregate
quality audit reports only USD 0.099978 because it excludes USD 0.028267 from
the failed Sonnet arm. This recovery preflight retains that excluded cost in
shared exposure; it does not use the narrower aggregate figure as its budget
basis.

## Reproduce the no-call freeze

Run from the `flavourbench` directory. This command performs no provider or MCP
call.

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.frontier_coverage_recovery_v4 freeze \
  --project-root . \
  --output-root artifacts/season1/current-quality-run/frontier-coverage-recovery-v4 \
  --parent-root artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1 \
  --quarantine artifacts/season1/current-quality-run/task-quarantine-v1/current-frontier-task-quarantine-e095c45ed27b0639a8eefae13a028c653fdea493999e095c2a757818ebbb7a15.json \
  --exposure-root artifacts/season1/current-quality-run
```

The command must reproduce plan hash
`730b426cfa5b7481446b4618166a2e6f75107c52ec26243283ef10ccbe01c0b8`.

## Refresh the zero-call preflight

Credential checks below test presence only. They do not contact a provider.
The active-expert variable is removed because it is unrelated to this run and
may contain a non-JSON shell representation.

```bash
set -a
source "${EPICURE_ROOT:?set EPICURE_ROOT}/.env"
set +a
export FLAVOURBENCH_OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?missing OpenRouter credential}"
export FLAVOURBENCH_COHERE_API_KEY="${COHERE_API_KEY:?missing Cohere credential}"
export FLAVOURBENCH_OPENROUTER_BASE_URL=https://gateway.ai.cloudflare.com/v1/f6f9ba40f69c8bfd8905206c45477027/epicure-prod/openrouter/v1
export FLAVOURBENCH_CLOUDFLARE_AI_GATEWAY_TOKEN="${CLOUDFLARE_AI_GATEWAY_TOKEN:?missing gateway token}"
export FLAVOURBENCH_MCP_URL=http://127.0.0.1:18082/mcp
export FLAVOURBENCH_MCP_TOKEN="${MCP_API_TOKEN:?missing MCP token}"
unset FLAVOURBENCH_ACTIVE_EXPERT_CONSENT_SHA256S

PYTHONPATH=src .venv/bin/python -m flavourbench.frontier_coverage_recovery_v4 preflight \
  --project-root . \
  --output-root artifacts/season1/current-quality-run/frontier-coverage-recovery-v4 \
  --plan artifacts/season1/current-quality-run/frontier-coverage-recovery-v4/frontier-coverage-recovery-v4-plan-730b426cfa5b7481446b4618166a2e6f75107c52ec26243283ef10ccbe01c0b8.json \
  --parent-root artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1 \
  --quarantine artifacts/season1/current-quality-run/task-quarantine-v1/current-frontier-task-quarantine-e095c45ed27b0639a8eefae13a028c653fdea493999e095c2a757818ebbb7a15.json \
  --exposure-root artifacts/season1/current-quality-run \
  --budget-audit artifacts/season1/current-quality-run/frontier-budget-audits/frontier-global-budget-ec179b7889834d2c6c92343acfb332e907a22600531333e9f0e1f7d7708a241d.json \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/explicit_low \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/provider_default \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/explicit_high \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate \
  --reasoning-v4-route-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/reasoning-effort-v4-route-gate-plan-2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352.json \
  --reasoning-v4-receipt artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/reasoning-effort-v4-route-gate-execution-receipt-172f4a08003656371de69c0907975f83761597338b159031b16052417d575852.json \
  --reasoning-v4-audit artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/reasoning-effort-v4-route-gate-audit-c90617d7b6a8cab918bf0f50f7190f8ad8f49badb5ce036c7c9fa716d7d9a959.json \
  --reasoning-v4-closure artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/reasoning-effort-v4-route-gate-closure-807aa054e7f0aaaa770630adae7696bba8fc24251d7ed2b08082b46a0edfde87.json \
  --reasoning-v4-ledger artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/ledger.jsonl \
  --reasoning-v4-journal artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/source/.flavourbench-live-smoke-journal-19125098-99b0-58af-b87b-a6260a9c5bd3.inprogress.jsonl \
  --reasoning-v4-source-directory artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4/route-gate/source \
  --reasoning-v5-route-plan artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/route-gate/reasoning-effort-v5-route-gate-plan-0481ecd9c8260967275e18a72d4ed265352d35ca2254f554ba55053bc61bb71c.json \
  --reasoning-v5-endpoint-snapshot artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/endpoint-snapshot/reasoning-effort-v5-endpoint-snapshot-ce46706dd7c2cb0605c3dd5abc34f36714f09a6074e155b18298393f14a38262.json \
  --reasoning-v5-sonnet-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/sonnet \
  --reasoning-v5-sonnet-receipt artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/sonnet/receipts/reasoning-effort-v5-sonnet-receipt-6b54b77c744016dd17714b25f7f0e2795600fb204d02e37f11165451a35de7a6.json \
  --reasoning-v5-sonnet-audit artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/sonnet/audits/reasoning-effort-v5-sonnet-audit-4c5e4a6fb796f9791fbf5e1889d3a09fd52fa3a1e67f0e50a0dbb6daeba49feb.json \
  --reasoning-v5-sonnet-closure artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/sonnet/closures/reasoning-effort-v5-sonnet-closure-99c194969edabe33ebfb942c1bf053515c871c953c65b8b8372400c4b245f068.json \
  --reasoning-v5-gemini-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/gemini \
  --reasoning-v5-gemini-receipt artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/gemini/receipts/reasoning-effort-v5-gemini-receipt-157e3aaeb8faf02830c927ddbe035dcb7414900cf900c59bd23db57bf918b803.json \
  --reasoning-v5-gemini-audit artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/gemini/audits/reasoning-effort-v5-gemini-audit-63da19f18b9c2f3104d6ef775969cc1a0c8750ef5bcacf03cf9f6bfdd0223f23.json \
  --reasoning-v5-gemini-closure artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/gemini/closures/reasoning-effort-v5-gemini-closure-44ba45a5c967744ffb9d9b107511a3104c4b0dfd6d708c8fdfddcf28c5ce0c04.json \
  --reasoning-v5-aggregate-audit artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/aggregate/reasoning-effort-v5-aggregate-audit-30271cb2108274271700be203d0eb3c7efde53875ca927c5425021ba27c32a35.json \
  --reasoning-v5-aggregate-closure artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v5/aggregate/reasoning-effort-v5-aggregate-closure-e6ce615dbe15c29ae8066990371f7512b532eea2689ddb83fd5917b059d8859b.json
```

Use only the newest admissible content-addressed preflight whose external
ledger heads have not changed.

## Execute and audit phase 1

Execution is the only provider/MCP-capable command. The local Epicure service
must already be available at port 18082. Environment variables from the
preflight section remain required, along with the usual live-run controls.

```bash
export FLAVOURBENCH_ENVIRONMENT=development
export FLAVOURBENCH_ROLE=worker
export FLAVOURBENCH_EXECUTION_MODE=live
export FLAVOURBENCH_LIVE_AUTHORIZED=true
export FLAVOURBENCH_OPENROUTER_ACCOUNTING_BASE_URL=https://openrouter.ai/api/v1
export FLAVOURBENCH_EPICURE_PROVENANCE_URL=http://127.0.0.1:18082/provenance

PYTHONPATH=src .venv/bin/python -m flavourbench.frontier_coverage_recovery_v4 execute \
  --project-root . \
  --output-root artifacts/season1/current-quality-run/frontier-coverage-recovery-v4 \
  --preflight artifacts/season1/current-quality-run/frontier-coverage-recovery-v4/frontier-coverage-recovery-v4-preflight-c2cf6aa4d6397f6034114dfb9ead0b446895256a7a84705fbd3a55c70d742268.json \
  --phase untouched_recovery \
  --confirm RUN_EXACT_V4_UNTOUCHED_RECOVERY_14_REAL_ARMS
```

Audit the emitted receipt and closure without provider calls:

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.frontier_coverage_recovery_v4 audit \
  --project-root . \
  --output-root artifacts/season1/current-quality-run/frontier-coverage-recovery-v4 \
  --preflight artifacts/season1/current-quality-run/frontier-coverage-recovery-v4/frontier-coverage-recovery-v4-preflight-<CURRENT-ADMISSIBLE-SHA256>.json \
  --receipt artifacts/season1/current-quality-run/frontier-coverage-recovery-v4/frontier-coverage-recovery-v4-untouched_recovery-receipt-<SHA256>.json \
  --execution-closure artifacts/season1/current-quality-run/frontier-coverage-recovery-v4/frontier-coverage-recovery-v4-untouched_recovery-closure-<SHA256>.json
```

The phase-1 audit must say `passed_complete_phase_disposition` and report seven
terminally dispositioned cells. Usable complete cells are reported separately;
a provider failure can lower that count without invalidating already complete
unrelated cells.

## Execute and audit the isolated GLM replacement

Only after the phase-1 audit passes:

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.frontier_coverage_recovery_v4 execute \
  --project-root . \
  --output-root artifacts/season1/current-quality-run/frontier-coverage-recovery-v4 \
  --preflight artifacts/season1/current-quality-run/frontier-coverage-recovery-v4/frontier-coverage-recovery-v4-preflight-<CURRENT-ADMISSIBLE-SHA256>.json \
  --phase glm_specific_replacement \
  --recovery-audit artifacts/season1/current-quality-run/frontier-coverage-recovery-v4/frontier-coverage-recovery-v4-untouched_recovery-audit-<SHA256>.json \
  --confirm RUN_EXACT_V4_GLM_REPLACEMENT_2_REAL_ARMS
```

Audit phase 2 with the same audit command, substituting the GLM receipt and
closure paths. Do not rerun a closed phase or reuse any v4 identifier.

## Offline verification

```bash
.venv/bin/ruff check \
  src/flavourbench/frontier_coverage_recovery_v4.py \
  tests/frontier_coverage_recovery_v4_test.py
PYTHONPATH=src .venv/bin/pytest -q tests/frontier_coverage_recovery_v4_test.py
```
