# Frontier coverage repair v1

This development-only repair requests exactly 25 missing real response arms in 13 frozen
endpoint-task cells. Twelve cells request a complete Epicure-off/on pair. The DeepSeek v4 Pro
composition cell requests only `epicure_off`; its existing content-addressed `epicure_on` arm is
not replayed.

The materializer binds the corrected task dossier, corrected arena, two exact route manifests,
the high-resource execution policy, and the local Epicure release, bundle, application, and tool
schema hashes. It also rechecks the frozen global budget audit and the three conservatively retained
reasoning-sensitivity sources while holding the shared frontier-ledger lock.

Its count field `planned_provider_work_items` describes the 13 frozen cells that would be submitted
only after a successor route gate passes. `provider_calls_executed_by_materialization` remains zero:
materialization is a read-only preflight, not a provider invocation.

## Zero-call preflight

Run from the `flavourbench` directory:

```bash
.venv/bin/python -m flavourbench.frontier_coverage_repair_executor \
  --schedule artifacts/season1/current-quality-run/frontier-coverage-repair-v1/frontier-coverage-repair-45ffc02f56b16b04f2fb4ce51c3561ddb99bd0cad55bf3a7c5162107b2085857.json \
  --arena artifacts/season1/current-quality-run/frontier-model-arena-review-pool-quarantine-v1/frontier-model-arena-review-pool-407e7fc6413e6d009c942eb51d9603d7cb958f0f282ffe90e1dc8ff28c3b6ac3.json \
  --task-validity artifacts/season1/task-validity/development-v2/development-task-validity-v2-86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json \
  --route-manifest artifacts/season1/current-quality-run/manifest-v29-high-resource/flavourbench-routed-unranked-f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json \
  --route-manifest artifacts/season1/current-quality-run/manifest-v42-high-resource-cohere-direct/flavourbench-cohere-unranked-fd28d55f78056d4d668a8f610a8de63228f7aabdc05fdfb5bfa4389d837d8a22.json \
  --project-root .. \
  --budget-audit artifacts/season1/current-quality-run/frontier-budget-audits/frontier-global-budget-ec179b7889834d2c6c92343acfb332e907a22600531333e9f0e1f7d7708a241d.json \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/explicit_low \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/provider_default \
  --supplemental-run-root artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/runs/explicit_high \
  --response-envelope-route-plan artifacts/season1/current-quality-run/response-envelope-route-v4/response-envelope-route-v4-plan-a3ef7434064415c93ab78fe818339e0466b100bee01e10e67cbdf1e4d848a4d6.json \
  --response-envelope-route-audit artifacts/season1/current-quality-run/response-envelope-route-v4/response-envelope-route-v4-audit-70fb6f9389885059f0ddf9bb6868ffe846ebcd48df67644a34075b9043dd32c3.json \
  --response-envelope-route-closure artifacts/season1/current-quality-run/response-envelope-route-v4/response-envelope-route-v4-closure-dfb54062b304b31c52f69a9698d6ffeda39f38f7bdf749d60fc9554f0d15078c.json \
  --output-directory artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1
```

The current preflight emits materialization
`eb27d59a5ec474f3b7975ea4649217182054f92d4b40bd6efbf3f1e4567b029f` and admissible dry-run plan
`3f4d1a8135232bb4097b64b6a4c8dae17b20faeb97a65ca5e824a5d3163e5fae`. It schedules 13 fresh
work items and 25 real arms, starts zero subprocesses, and makes zero provider and zero Epicure
calls. The reconciled projected exposure is $53.27616919333333333333333333 under the shared $85
admission ceiling. Execution still requires the separate `--execute` confirmation shown below.

## V4 source-reconstructed route gate

The successor protocol admits at most one fresh, sequential Epicure-off/on diagnostic pair on
`deepseek/deepseek-v4-flash-0731` at the exact `deepinfra/fp4` endpoint. It freezes the complete
work, run, arm, provider-attempt, and possible MCP-attempt ID pool before sending request bytes.
The route is fixed with `allow_fallbacks=false`, `require_parameters=true`,
`data_collection=deny`, an exact endpoint-only list, and endpoint-price ceilings. Both
intermediate and final requests use explicit low reasoning effort. The diagnostic has a $0.05
reserve and remains permanently unranked.

A v4 PASS cannot be manufactured from a self-consistent summary. The verifier reopens the exact
live artifact and finalized journal, verifies the journal hash chain, and reconstructs request
routing, reasoning and tool semantics; retry chains; response envelopes; generation-to-accounting
identity; cost; full MCP traces; and identifier freshness. It also requires substantive non-empty
final responses. The audit and permanent closure must both bind the same execution receipt. A
missing, stale, tampered, or failed layer blocks coverage before any subprocess starts.

The v4 pair is compatibility evidence only. Its prompt, answers, and tool evidence are never copied
into the coverage repair, a preference fit, or a leaderboard. A PASS permits only creation of a
fresh zero-call executable coverage plan. Never reuse a v1, v2, or v3 identifier or output.

The one authorized invocation is closed. The source-reconstructed PASS audit is
`70fb6f9389885059f0ddf9bb6868ffe846ebcd48df67644a34075b9043dd32c3`; its permanent closure is
`dfb54062b304b31c52f69a9698d6ffeda39f38f7bdf749d60fc9554f0d15078c`. Seven accepted DeepInfra
chat completions produced both matched arms and seven real Epicure calls, five of which returned
successful tool results. All seven generation costs reconcile to $0.002899. The earlier failed
audit `489aa7bf16491e9b305145eb1c1cd651638c16424fbf444eea63488c210df2ce` remains immutable. The PASS
records a content-addressed post-execution verifier correction for the historical live-artifact
JSON serializer and inclusion of retained MCP error results; no provider or MCP call was replayed.

The v4 lifecycle is:

```text
python -m flavourbench.response_envelope_route_v4 refresh-catalog ...
python -m flavourbench.response_envelope_route_v4 freeze ...
python -m flavourbench.response_envelope_route_v4 execute ...   # one pair maximum
python -m flavourbench.response_envelope_route_v4 audit ...
python -m flavourbench.response_envelope_route_v4 close ...
python -m flavourbench.response_envelope_route_v4 verify ...
```

After the verified PASS and closure exist, the coverage preflight adds all three content-addressed
gate artifacts:

```text
--response-envelope-route-plan <v4-plan-SHA256.json>
--response-envelope-route-audit <v4-PASS-audit-SHA256.json>
--response-envelope-route-closure <v4-closure-SHA256.json>
--execute
--confirm RUN_EXACT_COVERAGE_REPAIR_25_REAL_ARMS
```

The explicit `--execute` suffix remains separately governed and is not part of v4 route
qualification. After a future run, rerun the complete preflight command with the same three
gate-artifact arguments and no
`--execute`. It deterministically rescans the append-only source, response, and ledger directories
and emits a new content-addressed materialization/execution-plan pair without provider calls.

## Interpretation and downstream integration

The materializer exposes the present unsupported model-pair-by-family cells as composition 17,
cookability 27, evidence 27, and substitution 23. Completing the repair projects these graph holes
to zero because every endpoint gains a shared real anchor in each affected family. This is a graph
connectivity repair, not adequate evidence for family-specific rankings: the weakest endpoint-family
cells still have only one common task.

## Deterministic post-run paper inputs

Once all 13 repair cells are finalized and all 25 new normalized response arms exist, run the
following read-only materializer. It verifies the corrected (post-quarantine) strict and
high-resource uplift pools, the corrected model arena, all 524 historical normalized responses in
the nine contributing v27--v44 run directories, the frozen schedule and route identities, the
coverage ledger, every source/response content address, model/provider identity, exact Epicure
provenance, and provider-generation non-overlap.

```bash
.venv/bin/python -m flavourbench.frontier_coverage_postrun \
  --schedule artifacts/season1/current-quality-run/frontier-coverage-repair-v1/frontier-coverage-repair-45ffc02f56b16b04f2fb4ce51c3561ddb99bd0cad55bf3a7c5162107b2085857.json \
  --arena-base artifacts/season1/current-quality-run/frontier-model-arena-review-pool-quarantine-v1/frontier-model-arena-review-pool-407e7fc6413e6d009c942eb51d9603d7cb958f0f282ffe90e1dc8ff28c3b6ac3.json \
  --strict-base artifacts/season1/current-quality-run/frontier-strict-review-pool-quarantine-v1/frontier-multirun-review-pool-0da4c58326a936daef3d9e6ac606cfb5abaff2e9d93784754c56a302c662f38c.json \
  --high-base artifacts/season1/current-quality-run/frontier-high-resource-review-pool-quarantine-v1/frontier-multirun-review-pool-cd47055d12e6360a1ad0bfaa73fe4b2cef5bd1f5666150968bdfeeaf9eca024c.json \
  --task-validity artifacts/season1/task-validity/development-v2/development-task-validity-v2-86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json \
  --route-manifest artifacts/season1/current-quality-run/manifest-v29-high-resource/flavourbench-routed-unranked-f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json \
  --route-manifest artifacts/season1/current-quality-run/manifest-v42-high-resource-cohere-direct/flavourbench-cohere-unranked-fd28d55f78056d4d668a8f610a8de63228f7aabdc05fdfb5bfa4389d837d8a22.json \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v27-eight-pairs/responses \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v28-replenishment/responses \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v29-high-resource/responses \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v30-floor-replenishment/responses \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v32-floor-replenishment/responses \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v33-mistral-floor/responses \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v42-cohere-direct/responses \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v43-cohere-direct/responses \
  --historical-response-directory artifacts/season1/current-quality-run/pilot-v44-cohere-direct/responses \
  --output-directory artifacts/season1/current-quality-run/frontier-coverage-corrected-paper-inputs-v1
```

The command makes zero provider and zero Epicure calls. It fails closed unless the repair is
exactly complete; partial output is not written. On success it emits five content-addressed JSON
artifacts: corrected strict uplift, corrected high-resource uplift, corrected model arena,
coverage metrics (including paper macro values), and a bundle binding all four files and their
physical hashes. Repeating the command over unchanged inputs yields the same content addresses.

The expected structural result is 86 strict uplift pairs, 106 high-resource uplift pairs, 192
combined uplift pairs, 197 unique arena answers, and 1,043 arena comparisons. Unsupported
model-pair-by-family cells fall from 94 to zero. These values are conditional on successful,
identity-clean execution and are verified rather than hard-coded by the materializer.

Existing v27--v44 raw records, aggregates, pools, judgments, tables, and figures are never
overwritten or silently reinterpreted. Paper rendering must consume the new bundle explicitly.
Closing 94 graph holes with one shared anchor task per repaired endpoint-family cell is a
connectivity result only; it remains insufficient evidence for family-specific rankings.
