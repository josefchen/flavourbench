# Reasoning-effort sensitivity v2 route gate

> **Closed historical gate.** The explicit-low pair ran on 2026-08-03 and failed on an HTTP-200
> OpenRouter error envelope with embedded code 429. Audit SHA-256
> `481303eefacc872701d6a09aa9baeefe887027f655ccdc114ab881c8a16ff821` closes every v2
> work-item ID. None of the commands in this document may be executed. The corrected live runbook
> is `REASONING-EFFORT-SENSITIVITY-v3-ROUTE-GATE.md`.

The v1 route check yielded no usable pair. Its work-item IDs are closed and must not be replayed.
The v2 gate tests one model and task cell before any 72-arm sensitivity collection:

- model: `openai/gpt-5.6-sol-pro` on the frozen `openai/flex` endpoint;
- task: `fb-s0-substitution-003`;
- effort settings: explicit low, omitted provider default, and explicit high;
- workload: three matched Epicure-off/on pairs, six intended arms, and zero synthetic arms.

The content-addressed plan is
`reasoning-effort-v2-route-validation-plan-65b64747cbbecf116e3756f69bdbc7c0ccaf1a99a446d3352faa79b432e14e0f.json`.
It binds provider source SHA-256
`2782f577c6f70d4dbe0d6bdbafcca9d8013c71f9f681ce9b49cc1cea853b7422` and corrected Epicure
lineage inventory SHA-256
`70d00d933aa1340841a82a9637de8b75de380f8aeba2179beab419fb6542ab5f`.
It also binds lineage-correction SHA-256
`d739a1b08be79c8a116ea86687ef8f4c983fe8cc9c312257f85ef107e32e90e7` and v1 closed-identifier
inventory SHA-256
`707b0a3333efad66f77a587a8d275b3bb56a55d84b4c8bb12d02f75869d34bd6`.

The three fresh work-item IDs are:

1. `5340eb725ce68efdcc266ee023f951d23944541733db270b2a2f33f15f790590`;
2. `d0d076e88bc468410eb77ef499744a78d2e2bf6f382c112dff58d561e9480b3f`;
3. `70bafea998ae008712662174e7936e19d1556f3cb7e431ec347ac06558105713`.

## Admission rule

All six arms must finish with non-empty answers classified as chat completions. Every response must
have a generation ID, reconciled accounting, and the exact frozen model and provider identities.
Each Epicure-on arm must complete at least one real tool call; every Epicure-off arm must contain no
tool call. Runtime release, bundle, application, and semantic tool hashes must match. Truncation,
substitution, caching, ambiguous retries, unknown HTTP-200 envelopes, or unresolved cost blocks the
full study. One failed predicate closes all v2 IDs and requires a newly frozen v3 plan.
The pass audit must also show that every work-item, provider-attempt, and generation identifier is
unique and does not overlap the six closed v1 work IDs, 12 v1 attempt IDs, or six v1 generation IDs.

The gate reserves at most USD 4.78568. Projected conservative exposure after execution is USD
50.93887432666666666666666666 against the USD 85 admission ceiling. This budget calculation does
not authorize a provider call. The plan records zero provider calls and zero Epicure calls.

The diagnostic outputs never enter the quality fit. Passing the gate only permits materialization
of the prespecified 36-pair, 72-arm development sensitivity. Official ranking remains separately
blocked by Epicure release and redistribution gates.

## Exact operator sequence

The executable asset is
`reasoning-effort-v2-route-runner-assets-67ae96e905c36a1a6cdb2c208190bc6f9456dc5f4da9832cbcdc352e3f6ed805.json`.
Its three dry-run arrays completed successfully and reported `provider_calls_made: false`.

From the evaluation repository root, load the existing secret environment without printing it:

```bash
set -a
. "${EPICURE_ROOT:?set EPICURE_ROOT}/.env"
set +a
unset FLAVOURBENCH_ACTIVE_EXPERT_CONSENT_SHA256S
export PYTHONPATH=flavourbench/src
export FLAVOURBENCH_ENVIRONMENT=development
export FLAVOURBENCH_SERVICE_ROLE=worker
export FLAVOURBENCH_EXECUTION_MODE=live
export FLAVOURBENCH_LIVE_AUTHORIZED=true
export FLAVOURBENCH_OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
export FLAVOURBENCH_OPENROUTER_BASE_URL=https://gateway.ai.cloudflare.com/v1/f6f9ba40f69c8bfd8905206c45477027/epicure-prod/openrouter/v1
export FLAVOURBENCH_OPENROUTER_ACCOUNTING_BASE_URL=https://openrouter.ai/api/v1
export FLAVOURBENCH_CLOUDFLARE_AI_GATEWAY_TOKEN="$CLOUDFLARE_AI_GATEWAY_TOKEN"
export FLAVOURBENCH_OPENROUTER_TIMEOUT_SECONDS=300
export FLAVOURBENCH_MCP_URL=http://127.0.0.1:18082/mcp
export FLAVOURBENCH_EPICURE_PROVENANCE_URL=http://127.0.0.1:18082/provenance
export FLAVOURBENCH_MCP_TOKEN="$MCP_API_TOKEN"
export FLAVOURBENCH_EPICURE_RELEASE_ID=exploratory-unmatched-1790-runtime
export FLAVOURBENCH_EPICURE_BUNDLE_SHA256=98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1
export FLAVOURBENCH_EPICURE_APPLICATION_SHA256=be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313
export FLAVOURBENCH_EPICURE_TOOL_SCHEMA_SHA256=666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd
```

Bind the asset path once:

```bash
ASSETS=flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v2-route-validation/final-65b64747/reasoning-effort-v2-route-runner-assets-67ae96e905c36a1a6cdb2c208190bc6f9456dc5f4da9832cbcdc352e3f6ed805.json
```

The original array extraction below is retained only to document the corrected shell semantics.
It uses `mapfile` and array expansion; it must not be run because the v2 IDs are closed:

```bash
for variant in explicit_low provider_default explicit_high; do
  mapfile -d '' -t cmd < <(
    jq -j --arg v "$variant" \
      '.variants[] | select(.variant_id == $v) | .dry_run_command[] + "\u0000"' \
      "$ASSETS"
  )
  (( ${#cmd[@]} > 0 ))
  # Historical only. Do not execute: "${cmd[@]}"
done
```

The obsolete paid extraction is shown without execution so no closed v2 ID can be replayed:

```bash
variant=explicit_low
mapfile -d '' -t closed_v2_command < <(
  jq -j --arg v "$variant" \
    '.variants[] | select(.variant_id == $v) | .live_command[] + "\u0000"' \
    "$ASSETS"
)
(( ${#closed_v2_command[@]} > 0 ))
# Deliberately no array expansion: this command is permanently closed.
```

The explicit-low paid command was executed once and failed; the other two are closed without
execution. The repository contains a strict
`verify_v2_route_validation_pass_audit` verifier and a source-derived receipt builder. A
hand-authored PASS document is not acceptable. After each paid command, derive a new receipt with:

```bash
PLAN=flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v2-route-validation/reasoning-effort-v2-route-validation-plan-65b64747cbbecf116e3756f69bdbc7c0ccaf1a99a446d3352faa79b432e14e0f.json
AUDIT_DIR=flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v2-route-validation/final-65b64747/audits

PYTHONPATH=flavourbench/src flavourbench/.venv/bin/python \
  -m flavourbench.reasoning_effort_sensitivity audit-v2-route \
  --plan "$PLAN" \
  --runner-assets "$ASSETS" \
  --output-dir "$AUDIT_DIR"
```

The builder makes no provider or MCP calls and accepts no predicate values from the operator. It
verifies the exact runner assets and manifests, then derives the decision from content-addressed
execute summaries and source/response artifacts plus hash-chained journals and ledgers. Before all
three variants execute successfully, the expected decisions are `not_executed` or
`awaiting_remaining_route_variants`; continue only when no attempted variant reports a predicate
failure. Any `failed_one_or_more_predicates` receipt closes all three v2 work IDs.

A valid final PASS is a file named
`reasoning-effort-v2-route-validation-audit-<artifact_sha256>.json` whose content address verifies
and whose command result reports all of the following:

- `decision: passed_all_predicates`;
- `strict_pass_verifies: true`;
- `attempted_pairs: 3` and `usable_pairs: 3`;
- `full_sensitivity_authorized: true`;
- `provider_calls_made_by_builder: false` and `epicure_calls_made_by_builder: false`.

Inside the artifact, all 11 frozen predicate records must have `status: passed`, `passed: true`,
and non-empty immutable evidence hashes. The receipt must link exactly three source and summary
digests, six response artifacts, all fresh provider-attempt and generation IDs, exact generation
accounting, exact OpenAI model/provider identity, three successful Epicure-on treatments, no
Epicure-off call, no non-chat HTTP-200 envelope, no cache/ambiguity/truncation, and conservative
post-route exposure at or below USD 85. The strict verifier rejects any PASS that omits or changes
one of these fields. Until this exact PASS exists, full sensitivity and coverage execution remain
blocked even if all three paid commands exit successfully.
