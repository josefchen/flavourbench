# Reasoning-effort sensitivity v1 operator record

This runbook records the exact local runtime and runner invocation used for the paid v1 smoke. The
three v1 work items are now closed after failed source-backed executions. Do not execute them again.
Any subsequent smoke must use a newly frozen `frontier-reasoning-effort-sensitivity-v2` plan,
manifests, policies, and work-item IDs.

## Local Epicure runtime

```bash
set -a
. "${EPICURE_ROOT:?set EPICURE_ROOT}/.env"
set +a
cd "${EPICURE_MCP_ROOT:?set EPICURE_MCP_ROOT}"
PORT=18082 .venv/bin/python -m epicure_mcp.server
```

The authenticated provenance capture command is recorded in
`EPICURE-MCP-1790-R1-REPRODUCIBILITY.md`. Its response SHA-256 is
`825d087713ef525a1643a81bb9b94c26f3be64794ac84b4699d1cc8380922220`.

## Frozen v1 artifacts

- plan: `reasoning-effort-sensitivity-plan-4fddf823b552f930d807e02c0e9bfb706c5a15020b93244cf9e82d541a177097.json`;
- runner assets: `reasoning-effort-runner-assets-6df8e8774a1363ed2bfce4ef37e282370f6b893f306e292552769c3ead96dced.json`;
- paid-smoke audit: `reasoning-effort-smoke-audit-da645b267bb52e6eab248d608be181eba0fc14548a0a7d64d952a3091f6ce840.json`.

All are under
`flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/`.

## Historical execution environment

The token values are read from the existing private environment and never printed or serialized.

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

The runner-assets document contains argument arrays rather than shell strings. Before the paid
smoke, each `dry_run_command` was executed unchanged. Each paid invocation appended only the
recorded `smoke_command_suffix`, one variant at a time. No full command was run.

## Immutable outcome and audit

```bash
PYTHONPATH=flavourbench/src flavourbench/.venv/bin/python \
  -m flavourbench.reasoning_effort_sensitivity audit-smokes \
  --plan flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/reasoning-effort-sensitivity-plan-4fddf823b552f930d807e02c0e9bfb706c5a15020b93244cf9e82d541a177097.json \
  --runner-assets flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1/reasoning-effort-runner-assets-6df8e8774a1363ed2bfce4ef37e282370f6b893f306e292552769c3ead96dced.json \
  --output-dir flavourbench/artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v1
```

The audit decision is `blocked_before_full_study_due_smoke_failures`: zero of three pairs is usable,
the identifiable generation cost is USD 0.140822, and USD 4.78568 remains conservatively exposed.
Five HTTP-200 non-chat-completion envelopes share digest `16d1f38e...`. Their historical raw bodies
were intentionally not retained, so their exact upstream envelope class is indeterminate.
