# Reasoning-effort sensitivity v3 route gate

## Status and scope

> **Closed on 2026-08-03. Do not execute any v3 live command.** Both explicit-low arms exhausted
> the frozen two-attempt allowance with HTTP-200 error envelopes carrying code 429. The
> authoritative corrected audit is
> `aa66b52d784d813251f7506bbff3eff287f6a94c206fe0550b081ad34a37fb78`. Closure
> `290713a8758e9dcabd8567ed086425390537a121385a7ed6c956845d8d3ca1fb` permanently closes all
> v3 work-item and attempt IDs and does not authorize v4.

The v2 explicit-low route pair failed on an OpenRouter error envelope returned with HTTP 200 and
embedded code 429. The source-derived v2 audit is
`481303eefacc872701d6a09aa9baeefe887027f655ccdc114ab881c8a16ff821`. The immutable closure
`bf56182669498177a20d2b851ef9e89e2b0be0906e2c66b7dfaab432fcc25099` closes all three planned
v2 work-item IDs, five observed provider-attempt IDs, and four observed generation IDs.

The repaired provider adapter records an allowlisted error envelope as `request_rejected`, never as
`response_received`. Codes 408, 429, 502, and 503 may receive one bounded retry. A retry uses a new
attempt ID. The rejected envelope has no generation ID and receives no generation-cost lookup.
Unknown, non-transient, cached, or malformed envelopes remain terminal.

The content-addressed v3 plan is
`reasoning-effort-v3-route-validation-plan-be2f9d19c2565df76988318b91aa8963d216ec24691446aee8c49b8737f57a56.json`.
It binds:

- provider source SHA-256
  `21cfe7a0305376572cc60986af4d3612a480b5de789cdbca18ff72ee9d7c2d12`;
- safe-envelope contract SHA-256
  `ba5b771e18af15f3d2d4e9e472a258e47e982e293001c2b010fc984e291ab866`;
- Epicure lineage inventory SHA-256
  `70d00d933aa1340841a82a9637de8b75de380f8aeba2179beab419fb6542ab5f`;
- Epicure bundle SHA-256
  `98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1`;
- Epicure application SHA-256
  `be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313`;
- Epicure semantic tool-schema SHA-256
  `666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd`.

The gate remains diagnostic and unranked. It contains three matched Epicure-off/on pairs for
`openai/gpt-5.6-sol-pro`, task `fb-s0-substitution-003`, under explicit low, provider default, and
explicit high reasoning effort. Its six outputs cannot enter a quality fit.

The current conservative exposure is USD 47.74842099333333333333333333. The gate reserves at most
USD 4.785679999999999999999999998, projecting USD 52.53410099333333333333333333 against the USD 85
admission ceiling. The plan and materialization performed no provider or MCP calls.

## Exact environment

Run from the evaluation repository root. Load the existing secret environment without printing it:

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

Bind the immutable paths:

```bash
PLAN=flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v3-route-validation/reasoning-effort-v3-route-validation-plan-be2f9d19c2565df76988318b91aa8963d216ec24691446aee8c49b8737f57a56.json
ASSETS=flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v3-route-validation/final-be2f9d19/reasoning-effort-v3-route-runner-assets-aa2d631e73355d03f4f68709981e7d6995922158d9b057cac5bed29ad02a1844.json
AUDIT_DIR=flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v3-route-validation/final-be2f9d19/audits
```

## No-call verification

Run every dry command as an actual Bash array. `mapfile` preserves argument boundaries. Do not pipe
the NUL stream to an empty `xargs` invocation because that only prints the arguments:

```bash
for variant in explicit_low provider_default explicit_high; do
  mapfile -d '' -t cmd < <(
    jq -j --arg v "$variant" \
      '.variants[] | select(.variant_id == $v) | .dry_run_command[] + "\u0000"' \
      "$ASSETS"
  )
  (( ${#cmd[@]} > 0 ))
  "${cmd[@]}"
done
```

All three frozen dry commands have completed successfully with `provider_calls_made: false`.

## Closed route result

The original audit
`78f8209bdde679dafaa5cea45fe541fddd0cd2e78795c6e2a486ae6b7f2d8455` correctly failed the
gate but overstated retry-safety failures. It required both terminal second-attempt rejections to
schedule an impossible third attempt. The corrected source-derived audit reports:

- four planning requests across Epicure-off and Epicure-on;
- four safe allowlisted 429 rejection envelopes;
- two first-attempt rejections followed by fresh second attempts;
- two terminal second-attempt rejections at the frozen limit;
- zero unsafe rejections, accepted generations, cost lookups, or identified generation cost;
- zero usable pairs and no authorization for the full sensitivity study.

This correction does not change the route outcome. Neither arm reached a chat completion. The
redacted envelope locates failure at Cloudflare/OpenRouter route admission but cannot distinguish
gateway, router, or fixed upstream-provider throttling. The complete USD
1.595226666666666666666666666 pair allowance remains conservatively retained.

## Safe next command

Only local closure verification is permitted. This command performs no provider or MCP call:

```bash
CLOSURE=flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v3-route-validation/reasoning-effort-v3-route-closure-290713a8758e9dcabd8567ed086425390537a121385a7ed6c956845d8d3ca1fb.json
PYTHONPATH=flavourbench/src flavourbench/.venv/bin/python -c \
  'import json,sys; from pathlib import Path; from flavourbench.reasoning_effort_route_recovery import verify_v3_route_closure; document=json.loads(Path(sys.argv[1]).read_text()); assert verify_v3_route_closure(document); print(document["artifact_sha256"])' \
  "$CLOSURE"
```

Do not freeze or run v4 without a separately governed endpoint requalification or a different fixed
route. A future gate must use new identifiers and must bind that new availability evidence.
