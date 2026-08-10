# Frontier coverage stopped-run disposition

The coverage-repair v1 ledger is immutable and remains fail-closed. Its GLM
5.2 and Mistral Medium sources are retained as partial real runs; the Cohere
reservation is retained with its original incident and an appended retirement
event. No record is deleted, rewritten, or replayed.

The stopped-run audit distinguishes three failure classes:

- GLM 5.2: a local six-call-per-round safety gate rejected a ten-call fan-out
  after four real Epicure calls. The valid Epicure-off response remains a real
  arena arm; the incomplete cell is not an uplift pair.
- Mistral Medium: the Epicure-on tool turn returned HTTP 200 with native and
  normalized finish reason `length` at 8,192 completion tokens. No Epicure tool
  ran in that arm. The valid off response remains an arena arm; the incomplete
  cell is not an uplift pair.
- Cohere Direct: the incident stdout is exactly reconstructed from the missing
  prefixed-credential exception. The credential check precedes manifest load
  and provider construction. There is no source, journal, response, cost, or
  provider delivery for that work item. The original work item is nevertheless
  retired permanently.

The v2 continuation contains only the six untouched non-Cohere cells. Each is
moved to an alternate surface-screened development task and receives new cell,
work, run, arm, and attempt identifiers in a separate namespace and ledger.
The old orphan therefore cannot block this separate development-only schedule.

The v3 replacement plan is distinct from both schedules. It freezes alternate,
previously unseen model-task cells for GLM composition, Mistral cookability,
and Cohere cookability. These are post-failure sensitivity observations, not
missing-at-random replacements: they remain ineligible for an official fit and
cannot erase the original reliability failures. The plan is executable only
through the separate fail-closed continuation executor described below.

For every future `cohere_direct` cell,
`FLAVOURBENCH_COHERE_API_KEY` must be present before a reservation can be
appended. An unprefixed `COHERE_API_KEY` is deliberately ignored. Use
`append_guarded_continuation_reservation`; no v2/v3 executor may append a
reservation directly.

The original audit, closure, and plan commands are local and make zero provider
or MCP calls. Their authoritative artifacts are content-addressed under
`artifacts/season1/current-quality-run/frontier-coverage-continuation-v2/`.

## Executable continuation boundary

The paid executor accepts only v2 plan `e9f4375f8976...` and v3 plan
`3baff4ae405b...`. It reconstructs both plans from their original task dossier,
v1 materialization, route manifests, stopped-run evidence, orphan closure, and
v4 response-envelope qualification. Nine endpoint-task cells and 18 real arms
are frozen. The six v2 cells and three v3 replacement cells use separate
append-only ledgers and reservation namespaces under:

`artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1/`

Cells run sequentially. The prefixed `FLAVOURBENCH_COHERE_API_KEY` gate is
checked before the first reservation and again under the shared budget mutex
before every cell. An unprefixed variable is not read by the executor. The
current zero-call preflight reserves a worst-case $4.4862581067 for the 18 arms
and projects total shared exposure of $50.9645694333, below the $85 admission
ceiling and $100 hard cap.

An incomplete cell permanently stops the schedule. A reservation without a
source is never submitted again. A complete source awaiting ledger
finalization may be recovered without a provider call; this is not a replay.
The terminal closure retires every planned work, run, and attempt identifier,
including identifiers in cells that were not reached.

Journal evidence distinguishes failure classes. A finalized journal with no
provider attempt, or only safe pre-send/rejection terminals and no generation,
is recorded as a pre-request or safe provider rejection. An `uncertain_delivery`
event, an unterminated `request_started`, or an unreconciled generation is
recorded as uncertain delivery. No journal is recorded as unknown, not as proof
of non-delivery. All classes remain no-replay under the parent continuation
policy.

### Zero-call preflight

Run from the `flavourbench` directory after placing the Cohere credential in
the prefixed process variable. This command performs no provider or MCP call.

```bash
set -a
source "${EPICURE_ROOT:?set EPICURE_ROOT}/.env"
set +a
export FLAVOURBENCH_COHERE_API_KEY="${COHERE_API_KEY:?missing Cohere credential}"
unset FLAVOURBENCH_ACTIVE_EXPERT_CONSENT_SHA256S

.venv/bin/python -m flavourbench.frontier_coverage_continuation_executor preflight \
  --project-root . \
  --output-root artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1 \
  --v2-plan artifacts/season1/current-quality-run/frontier-coverage-continuation-v2/frontier-coverage-continuation-plan-e9f4375f8976ec7468d436ff1ade21642d6746a6eca1722f4355cdd96be19646.json \
  --v3-plan artifacts/season1/current-quality-run/frontier-coverage-continuation-v2/frontier-coverage-replacement-plan-3baff4ae405b0dbe4eb5168a5a088b29cb9438c86b01ce3d5a5be670839d14ee.json \
  --v1-materialization artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/frontier-coverage-materialization-eb27d59a5ec474f3b7975ea4649217182054f92d4b40bd6efbf3f1e4567b029f.json \
  --task-validity artifacts/season1/task-validity/development-v2/development-task-validity-v2-86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json \
  --route-manifest artifacts/season1/current-quality-run/manifest-v29-high-resource/flavourbench-routed-unranked-f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json \
  --route-manifest artifacts/season1/current-quality-run/manifest-v42-high-resource-cohere-direct/flavourbench-cohere-unranked-fd28d55f78056d4d668a8f610a8de63228f7aabdc05fdfb5bfa4389d837d8a22.json \
  --stopped-audit artifacts/season1/current-quality-run/frontier-coverage-continuation-v2/frontier-coverage-stopped-run-audit-b0990b3b8869325771433cccd8a390a0e48038cf07637cac7ee244a39e9ca4d5.json \
  --orphan-closure artifacts/season1/current-quality-run/frontier-coverage-continuation-v2/frontier-coverage-orphan-closure-3cb144abd1162447e3e64ba0b703ea09d9ead595d141e2dbf1ffb0103d27e370.json \
  --v1-run-root artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1 \
  --budget-audit artifacts/season1/current-quality-run/frontier-budget-audits/frontier-global-budget-ec179b7889834d2c6c92343acfb332e907a22600531333e9f0e1f7d7708a241d.json \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/explicit_low \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/provider_default \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/explicit_high \
  --v4-route-plan artifacts/season1/current-quality-run/response-envelope-route-v4/response-envelope-route-v4-plan-a3ef7434064415c93ab78fe818339e0466b100bee01e10e67cbdf1e4d848a4d6.json \
  --v4-route-audit artifacts/season1/current-quality-run/response-envelope-route-v4/response-envelope-route-v4-audit-70fb6f9389885059f0ddf9bb6868ffe846ebcd48df67644a34075b9043dd32c3.json \
  --v4-route-closure artifacts/season1/current-quality-run/response-envelope-route-v4/response-envelope-route-v4-closure-dfb54062b304b31c52f69a9698d6ffeda39f38f7bdf749d60fc9554f0d15078c.json
```

### Exact paid command

The command below is the only admitted paid suffix. The local Epicure service
must already be available at port 18082. It retains OpenRouter accounting on
the direct OpenRouter endpoint while routing generation through the existing
Cloudflare AI Gateway.

```bash
export FLAVOURBENCH_ENVIRONMENT=development
export FLAVOURBENCH_ROLE=worker
export FLAVOURBENCH_EXECUTION_MODE=live
export FLAVOURBENCH_LIVE_AUTHORIZED=true
export FLAVOURBENCH_OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
export FLAVOURBENCH_OPENROUTER_BASE_URL=https://gateway.ai.cloudflare.com/v1/f6f9ba40f69c8bfd8905206c45477027/epicure-prod/openrouter/v1
export FLAVOURBENCH_OPENROUTER_ACCOUNTING_BASE_URL=https://openrouter.ai/api/v1
export FLAVOURBENCH_CLOUDFLARE_AI_GATEWAY_TOKEN="$CLOUDFLARE_AI_GATEWAY_TOKEN"
export FLAVOURBENCH_MCP_URL=http://127.0.0.1:18082/mcp
export FLAVOURBENCH_EPICURE_PROVENANCE_URL=http://127.0.0.1:18082/provenance
export FLAVOURBENCH_MCP_TOKEN="$MCP_API_TOKEN"

.venv/bin/python -m flavourbench.frontier_coverage_continuation_executor execute \
  --project-root . \
  --output-root artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1 \
  --preflight artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1/frontier-coverage-continuation-preflight-<SHA256>.json \
  --confirm RUN_EXACT_V2_V3_COVERAGE_CONTINUATION_18_REAL_ARMS
```

The `<SHA256>` placeholder must be replaced by the exact admissible preflight
emitted immediately before execution. Do not regenerate or resume after a
closure exists.

### Source-reconstructing post-run audit

The executor emits a receipt and terminal closure even when the first cell
fails. The audit reopens both ledgers, every source, normalized response, and
run journal; verifies routes, tasks, Epicure identity, frozen attempt slots,
generation/accounting bijections, MCP traces, source hashes, and closure heads;
and writes a content-addressed pass or failed-closed decision.

```bash
.venv/bin/python -m flavourbench.frontier_coverage_continuation_executor audit \
  --project-root . \
  --output-root artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1 \
  --preflight artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1/frontier-coverage-continuation-preflight-<SHA256>.json \
  --receipt artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1/frontier-coverage-continuation-receipt-<SHA256>.json \
  --execution-closure artifacts/season1/current-quality-run/frontier-coverage-continuation-execution-v1/frontier-coverage-continuation-closure-<SHA256>.json
```

Neither a complete continuation nor a passing audit authorizes an official
fit. These observations can close comparison-graph diagnostics only. The v1
failures remain in reliability reporting, and the v3 replacements remain a
post-failure, non-missing-at-random stratum.
