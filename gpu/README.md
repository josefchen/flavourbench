# FlavourBench GPU deployment scaffold

This directory contains inert-by-default deployment templates for real open-weight model inference
on Modal and Lambda On-Demand Cloud. It does not replace the FlavourBench worker. The worker remains
the only Epicure MCP orchestrator: it sends the same tool schemas to every model, executes Epicure
calls itself, records the complete trace, and returns tool evidence to the model. GPU endpoints never
receive Epicure credentials and never call MCP directly.

No file in this directory launches compute during import, validation, planning, or cloud-init
rendering. A provider mutation requires all three controls:

1. a manifest with both `controls.mutations_authorized=true` and an approved authorization ticket;
2. the explicit `--apply` option; and
3. `FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED=I_UNDERSTAND_THIS_CREATES_GPU_SPEND`.

The checked-in examples keep authorization false and therefore cannot be deployed, even if someone
adds `--apply`. Current Gate A authorization remains zero.

## What is pinned

The two examples pin the same development model workload:

- Qwen 2.5 0.5B Instruct at Git commit
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- `model.safetensors` SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
- the model chat-template hash;
- the vLLM 0.25.1 amd64 container manifest digest;
- Modal SDK 1.5.2;
- dtype, context limit, parallelism, parser, structured-output backend, concurrency, seed, and
  request decoding configuration.

This tiny model is a deployment and contract-test fixture, not a proposed Season 0 competitor. The
Modal manifest is completely immutable. The Lambda workload is immutable, while three
account-specific infrastructure identifiers remain conspicuous `REPLACE_...` values; the live gate
rejects them. Resolve the current Lambda image ID, SSH key, and SSH-only firewall ruleset only in a
new, governed manifest.

Each endpoint runs the shared gateway in [common/gateway.py](common/gateway.py). It:

- exposes only `/v1/models`, `/v1/chat/completions`, `/healthz`, and an authenticated
  `/flavourbench/manifest`;
- requires a FlavourBench generation ID on every completion;
- rejects the wrong model or decoding settings;
- injects `parallel_tool_calls=false`;
- disables streaming and vLLM request logging;
- returns `X-FlavourBench-Deployment-SHA256` on every model response; and
- reports actual runtime/GPU attestation without exposing credentials.

## Exact zero-cost commands

Run these from this directory. They make no provider API call and create no cloud resource:

```bash
python3 common/manifest.py validate manifests/example-modal-qwen2.5-0.5b.json
python3 common/manifest.py validate manifests/example-lambda-qwen2.5-0.5b.json

python3 modal/deploy.py plan \
  --manifest manifests/example-modal-qwen2.5-0.5b.json

python3 lambda_cloud/controller.py plan \
  --manifest manifests/example-lambda-qwen2.5-0.5b.json \
  --max-runtime-minutes 30

python3 lambda_cloud/controller.py render-cloud-init \
  --manifest manifests/example-lambda-qwen2.5-0.5b.json \
  --max-runtime-minutes 30 >/tmp/flavourbench-cloud-init-dry-run.yaml
```

`make check` performs syntax, JSON, and manifest-integrity checks. `make plan` runs both no-cost
plans. Installing the pinned local deployment client also creates no cloud resource:

```bash
python3 -m venv .venv
.venv/bin/pip install --require-virtualenv --no-deps modal==1.5.2
```

Do not use a raw `modal deploy` or Lambda launch call for this benchmark. The controllers are the
enforcement boundary.

## Modal topology

[modal/app.py](modal/app.py) defines one authenticated Modal Server per immutable model manifest.
It uses a digest-pinned vLLM image, bounded autoscaling, fixed routing/compute regions, and an
authenticated Modal Secret. The secret must contain independently generated
`FLAVOURBENCH_GATEWAY_API_KEY` and `FLAVOURBENCH_INTERNAL_VLLM_API_KEY`; add `HF_TOKEN` only for a
governed gated model. Never put these values in a manifest.

For an authorized manifest, configure the Modal workspace hard budget and the season Environment
budget before deployment. The database governor remains primary: reserve worst-case GPU seconds
before admitting work, stop admission at 85%, drain at 95%, and stop the App at 100%. Keep quality
scores separate from cold-start, latency, and allocated infrastructure cost.

The mutation shape, shown only for operator review, is:

```bash
export FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED=I_UNDERSTAND_THIS_CREATES_GPU_SPEND
python3 modal/deploy.py deploy \
  --manifest /absolute/path/to/authorized-modal-manifest.json \
  --authorization-ticket GATE-A-APPROVED-TICKET \
  --apply
```

The checked-in manifest refuses this command.

## Lambda topology and external hard stop

Lambda is treated as capacity, not a per-generation API. The controller verifies the frozen
instance type, exact price, and region capacity before launch. Cloud-init starts the same
digest-pinned vLLM workload behind the shared gateway. Both services bind to localhost; the worker
must reach port 8000 through a managed SSH tunnel. The Lambda firewall should permit only SSH from
the benchmark worker's fixed egress `/32`. Do not expose vLLM on `0.0.0.0:8000`.

The termination watchdog runs on independent, non-Lambda infrastructure. This is essential because
shutting down Linux on a Lambda VM does not terminate the billable instance. The watchdog owns a
mode-0600 lease containing the instance ID, manifest hash, absolute deadline, frozen price, and hard
cost cap. It calls Lambda's `instance-operations/terminate` endpoint when any hard condition fires.

Production launch is intentionally two-phase:

1. Start the watchdog under systemd, Compose, or another restartable supervisor. With `--loop`, it
   writes a fresh `<lease>.watchdog-ready` heartbeat before any VM exists.
2. The launch controller checks that the heartbeat is under 30 seconds old, belongs to a live PID,
   and matches the manifest. It refuses to launch otherwise. While armed without a lease, the
   watchdog also polls for instances tagged with the manifest hash. If a launch response is lost
   and no lease appears within 30 seconds, it terminates the orphan through the Cloud API.

Operator command shape for an authorized manifest:

```bash
export LAMBDA_API_KEY='stored-in-the-capacity-controller-secret-store'
export FLAVOURBENCH_GPU_MUTATIONS_AUTHORIZED=I_UNDERSTAND_THIS_CREATES_GPU_SPEND

python3 lambda_cloud/watchdog.py \
  --lease /var/lib/flavourbench/lambda/epoch-001.lease.json \
  --manifest /absolute/path/to/authorized-lambda-manifest.json \
  --authorization-ticket GATE-A-APPROVED-TICKET \
  --apply --loop

python3 lambda_cloud/controller.py launch \
  --manifest /absolute/path/to/authorized-lambda-manifest.json \
  --lease /var/lib/flavourbench/lambda/epoch-001.lease.json \
  --max-runtime-minutes 30 \
  --authorization-ticket GATE-A-APPROVED-TICKET \
  --apply
```

The checked-in manifest refuses both commands. Supervising the watchdog and persisting its lease on
non-ephemeral storage are acceptance requirements, not optional operations advice. The Lambda API
key has broad account authority; keep it outside the GPU VM in a dedicated workspace/controller.

## Turning a fixture into a scored deployment

1. Resolve Gate A model-lineage, privacy, budget, account, conflict-disclosure, and reviewer
   decisions.
2. Copy—not edit—the example into the frozen season-manifest directory.
3. Replace the development model with an eligible model and pin model, tokenizer, code, weights,
   container, chat-template, parser, GPU, region, and price identities.
4. Run tools/structured-output contract tests and an unranked Epicure-on/off smoke test.
5. Set the approved ticket, monetary cap, and authorization booleans.
6. Recompute `spec_sha256` over the JSON object with that field omitted, then validate it:

   ```bash
   python3 common/manifest.py hash /absolute/path/to/new-manifest.json
   python3 common/manifest.py validate /absolute/path/to/new-manifest.json
   ```

7. Freeze and sign the manifest. Never redeploy an App or rebuild a Lambda VM inside a scored
   collection window; create a new deployment profile instead.
8. Before every collection block, compare `/v1/models`, `/flavourbench/manifest`, the response model,
   and the attestation response header to the frozen profile. Any mismatch is an incident and the
   generation is exploratory only.

## Official primary documentation

- Modal [Endpoints](https://modal.com/docs/guide/endpoints),
  [Servers](https://modal.com/docs/guide/servers),
  [GPU selection and `H100!`](https://modal.com/docs/guide/gpu),
  [runtime identity variables](https://modal.com/docs/guide/environment_variables),
  [budgets](https://modal.com/docs/guide/budgets),
  [pricing](https://modal.com/pricing), and
  [endpoint privacy/retention](https://modal.com/docs/guide/security).
- Lambda [Cloud API](https://docs.lambda.ai/public-cloud/cloud-api/),
  [instance lifecycle and termination warning](https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/),
  [firewalls](https://docs.lambda.ai/public-cloud/firewalls/),
  [access controls](https://docs.lambda.ai/public-cloud/access-security/),
  [vLLM deployment](https://docs.lambda.ai/education/large-language-models/deploying-nemotron-3-nano/),
  and [current instance pricing](https://lambda.ai/instances).
- vLLM [OpenAI-compatible serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
  and [structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/).
