# Season 1 arena method-validation runbook

This job validates the statistical method on simulated outcomes. It does not
call a language model, generate benchmark answers, create votes, or modify any
model score. The publication gate accepts only the final verified aggregate;
partial shards remain `required_not_yet_executed`.

## Freeze the deterministic worklist

Run from the evaluation-paper root:

```bash
flavourbench/.venv/bin/python -m flavourbench.season1_arena_distributed \
  freeze-manifest \
  --shard-size 1 \
  --output flavourbench/artifacts/season1/method-validation/distributed-manifest.json
```

The manifest contains 16,000 content-addressed one-dataset shards. Re-running
the command against unchanged code and contracts produces the same bytes and
manifest digest.

## One production-image measurement on Modal

Install the pinned controller client without changing the worker image:

```bash
flavourbench/.venv/bin/pip install 'modal==1.5.3'
```

Then run exactly one shard. The authorization amount must exceed the frozen
ten-minute CPU-and-requested-memory function bound. It excludes any Modal image
build charge, tax, credit, or invoice adjustment.

```bash
modal run flavourbench/src/flavourbench/season1_arena_modal.py::measure \
  --manifest flavourbench/artifacts/season1/method-validation/distributed-manifest.json \
  --output-directory flavourbench/artifacts/season1/method-validation/modal \
  --maximum-authorized-usd 0.02
```

Each worker verifies the policy, layout, source files, lockfile, Dockerfile,
Python version, machine architecture, and numerical dependencies before the
first bootstrap replicate. The result and telemetry are separate
content-addressed artifacts so measured runtime cannot alter deterministic
statistical output.

## Full campaign

Do not run the full command while `protocol/study.yaml` keeps
`compute.modal_enabled: false` or `budget.modal_cap_usd: 0`. After a positive
non-fungible Modal authorization and workspace hard budget are recorded, build
a zero-cloud-spend admission artifact first. This command fails under the
current zero cap and therefore runs before the Modal CLI can build an image.

```bash
flavourbench/.venv/bin/python -m flavourbench.season1_arena_distributed \
  admit-modal-full \
  --manifest flavourbench/artifacts/season1/method-validation/distributed-manifest.json \
  --measurement /absolute/path/to/telemetry-<sha256>.json \
  --governance-study protocol/study.yaml \
  --maximum-authorized-usd <positive-governed-cap> \
  --workspace-hard-budget-usd <confirmed-workspace-cap> \
  --output /absolute/path/to/modal-admission.json
```

Only after admission succeeds may the Modal command be invoked:

```bash
export FLAVOURBENCH_MODAL_FULL_RUN_AUTHORIZED=I_AUTHORIZE_THE_BOUNDED_SEASON1_MONTE_CARLO
export FLAVOURBENCH_MODAL_WORKSPACE_BUDGET_CONFIRMED=I_CONFIRMED_THE_MODAL_WORKSPACE_HARD_BUDGET

modal run flavourbench/src/flavourbench/season1_arena_modal.py::full \
  --manifest flavourbench/artifacts/season1/method-validation/distributed-manifest.json \
  --output-directory flavourbench/artifacts/season1/method-validation/modal \
  --measurement /absolute/path/to/telemetry-<sha256>.json \
  --governance-study protocol/study.yaml \
  --admission /absolute/path/to/modal-admission.json
```

The controller admits at most 64 shards per wave, writes every completed shard
atomically, resumes by exact shard digest, and stops admission at 85% of the
bounded cap. Modal automatic retries are disabled; a retry is a new governed
attempt. The external workspace budget is the absolute spending stop.

## Provider-neutral execution and aggregation

Any CPU scheduler can execute an individual manifest shard inside the frozen
`flavourbench/Dockerfile` image:

```bash
export FLAVOURBENCH_SEASON1_PRODUCTION_SHARD_AUTHORIZED=I_AUTHORIZE_ONE_SEASON1_METHOD_VALIDATION_SHARD

python -m flavourbench.season1_arena_distributed run-shard \
  --manifest /artifacts/distributed-manifest.json \
  --shard-sha256 <sha256> \
  --output-directory /artifacts/results \
  --provider <scheduler-name>
```

Aggregation is always safe to run. It returns
`required_not_yet_executed`, null acceptance, and no pass claim until every
expected shard is present and verified.

```bash
python -m flavourbench.season1_arena_distributed aggregate \
  --manifest /artifacts/distributed-manifest.json \
  --output-directory /artifacts/results \
  --output /artifacts/distributed-aggregate.json
```
