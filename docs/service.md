# FlavourBench service

FlavourBench is the isolated API, worker, storage model, and scoring package for
Epicure's culinary LLM arena. Local service development defaults to deterministic
fixtures; the retrospective Season 0 analysis pipeline rejects fixture, mock, synthetic, and
placeholder evidence at every publication boundary. Live provider calls require all
of the following:

- `FLAVOURBENCH_EXECUTION_MODE=live`
- `FLAVOURBENCH_LIVE_AUTHORIZED=true`
- a non-zero, frozen season budget in the database
- Bedrock authorization and, only for frozen OpenRouter routes, an OpenRouter API key

The public web application talks only to the API. The worker owns model and MCP
execution and claims durable PostgreSQL jobs with `FOR UPDATE SKIP LOCKED`.

```bash
alembic upgrade head
flavourbench-seed
flavourbench-api
# separate process
flavourbench-worker
# unranked three-battle engineering check
flavourbench-smoke
```

## Retrospective Season 0 pilot

The current paper pipeline is separate from the historical engineering runners below. It binds:

- 120 licensed human-origin culinary questions, 30 per family, with zero generated tasks;
- 12 exact model/provider endpoints and two conditions, for 2,880 real target arms;
- a real 1,790-ingredient Epicure MCP snapshot with 13 read-only tools;
- 720 balanced Model Arena comparisons and 1,440 paired Epicure-uplift contrasts; and
- four real Bedrock automated judges run in original and swapped orientation.

The frozen task bank is `data/season0/frozen/season0-real-task-bank-1ce969bdee4124fa44bab46a04feda2a0ebeddf4d37c49c0264b48b3833a4313.json`.
The frozen model manifest is `artifacts/season0/manifests/season0-model-manifest-3919def66686b4bd939c94cdd89659f63ae2afbbf03288413129e2ea8d6b83d2.json`.
The active retrospective analysis refuses synthetic records, incomplete dense inputs, unbound hashes, or an
over-cap workload, and withholds ratings rather than fitting a disconnected preference graph.
Human preference and independent chef validation are not claimed. This pilot is not an official
quality leaderboard or a confirmatory estimate of Epicure uplift.

## Public task-validity calibration

The deployed non-ranking calibration packet contains 40 real human-authored public tasks, ten per
family, under digest
`c45023aee6cf8ff91437c08c16ae20498b2d025e9a0155aedd44898de1d7fbb1`. Each task requires three
independent prompt-blind validity decisions. A valid reviewer must then author a task-specific
criterion pack after the source reference is revealed. Unanimous valid records finish without an
extra step; every non-unanimous record goes to a distinct fourth adjudicator. Model outputs are
never shown during this campaign.

The live status artifact reports 40 queued tasks, 120 required source reviews, zero completed
reviews, and zero independently validated tasks at the 2 August 2026 cutoff. The packet itself is
an instrument, not human-validity evidence, and its tasks remain ineligible for confirmatory
ranking. Qualified invited validators use `/flavourbench/task-review`; the administrator status
route is `GET /v1/admin/development-tasks/status`.

The separate no-generation OpenRouter catalog audit at
`artifacts/season1/current-quality-run/catalog-audit-v2/` binds the frozen 14-route development
manifest. At the 2 August 2026 cutoff, all 14 stable model IDs, dated canonical releases, exact
provider tags, provider names, and required tool and structured-output parameters matched. The
audit used 15 public catalog requests, made no provider generations or Epicure calls, cost USD 0,
and contributes no quality observation.

Reproduce the audit without credentials or paid inference:

```bash
flavourbench-audit-current-catalog \
  --manifest artifacts/season1/current-quality-run/manifest-v13-evidence-boundary/flavourbench-openrouter-unranked-12f411f86c67af5555036851713290bcaf04e1d725bada5af937839753e7db54.json \
  --output-dir artifacts/season1/current-quality-run/catalog-audit-v2
```

## Current exact-frontier real-call corpus

The dated v27, v28, v29, v30, v32, and v33 collections contain 296 finalized matched pairs across
14 exact endpoints and 28 distinct human-authored tasks. They preserve 480 normalized response
arms, 1,739 provider generation identifiers, and 454 real Epicure calls, of which 350 succeeded.
The collections contain no synthetic task, response, or tool-trace record. Their conservative
total exposure is USD 53.707711. Each endpoint has between 8 and 19 complete pairs.

The 191 complete pairs define the blinded Epicure-uplift workload. The 196 valid Epicure-on
answers that share a task with at least one other endpoint define a separate model-arena workload
with 843 same-task comparisons on one connected 14-model graph. The canonical pool digest is
`547114fe0f47520296e527177e9d53b635d9899beae785a26da681288fecb5e0`; its public receipt digest is
`46799f3a7cb63ec742b6b07c01bb64fbbdd9e9f30975413cb5587987442f0857`. Compose projects this exact
pool into PostgreSQL before the API starts. The rows remain development-only and rank-ineligible
until real blinded ballots are submitted.

## Prospective Season 1

The public-source validation campaign is frozen under digest
`76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709`. It targets 120 admitted
human-written tasks, 30 per family, from a fixed 180-candidate slate. The 60 extra candidates form
an attrition reserve. The slate binds candidate bundle `b13ab30b...`, assignment `631932c...`,
and acquisition receipt `847a95f...`. It contains zero source-answer requests, zero model calls,
zero Epicure calls, and zero human ballots.

Each candidate receives two prompt-only, qualification-matched culinary reviews. Unanimous valid
records are admitted after both validators countersign the merged criterion-pack hash. A third
person adjudicates only a decision, label, or pack disagreement. Rights and contamination checks
run over the complete sealed bank and reserve; two campaign auditors inspect seed-committed
samples plus every anomaly or hit. The two validators, a disagreement adjudicator, and the two
campaign auditors require five people under maximum cross-task reuse. Eight are recommended.
Season-specific person commitments enforce within-task separation without publishing raw identity
handles.

External volunteers whose identity and qualifications are checked privately use the verified
task-validator or calibrated-expert pathways. Their published identities remain pseudonymous. The
`anonymous_external_rater` pathway is a separate self-attested, unverified cohort and cannot support
an independent-expert claim. External enrollment is currently disabled: the candidate consent at
`../protocol/consent/EXPERT-CONSENT-v1-DRAFT.md` has no approved monitored research contact, the
institutional determination is unresolved, and reviewer withdrawal and identity-record retention
still require an operational sign-off. The content-addressed activation contract is recorded at
`../protocol/consent/REVIEWER-CONSENT-ACTIVATION.md`.

The full procedure is in
`../protocol/FLAVOURBENCH-PUBLIC-SOURCE-TASK-VALIDATION-v1.md`. The source questions are public,
so the permitted claim is contamination-limited, not contamination-free. The current runtime bank
importer and `season1-study-design-v5.json` still encode the superseded 240-task, six-person
admission design. They must fail closed until a reviewed v6 adapter and synchronized study contract
replace those assumptions. No prospective generation may begin in the interim.

The primary controlled collection contains 3,200 model-arena battles and 3,200 paired Epicure
uplift battles, or 12,800 model response arms. The independent culinary-review floor is 800
unique comparisons per track, each rated by two distinct reviewers (3,200 primary judgments),
with a 12.5% concealed repeat schedule for reliability. Live public traffic is a separate,
observational cohort. Stopping, estimands, exclusions, and separate reporting of preference,
reliability, tool success, cost, and latency are fixed before collection.

Release also requires a post-generation item audit of every anomaly-flagged task and a
seed-committed random quarter of the bank, with two new independent auditors per task and zero
unresolved material defects. A nested 20-task, three-generation reliability panel adds 1,280
arms beyond the primary schedule; a development-only three-prompt sensitivity audit adds 480.
The full plan therefore contains 14,560 real model arms. A separate blinded study runs 24
cookability outputs twice in real kitchens (48 executions) to test whether rubric cookability
tracks completion, time, deviations, yield, and acceptability. These robustness and
construct-validity results remain separate from the primary ranking.
The row-level completion-artifact contract is
`contracts/season1/season1-validity-robustness-evidence-v1.md` (SHA-256
`37aa50d8dba05410aaa7862d4e2c55674fa992b5cb415ea3ae2b6ece95357056`). It requires
reproducible task, cell, arm, auditor, correction, and kitchen-execution records rather than
accepting aggregate assertions.

The current-route registry contains 16 exact endpoints across Bedrock, OpenRouter, and direct
providers. Fifteen routes passed and contribute 30 real provider generations and 15 real Epicure
calls under the production structured-output contract. Qwen 3.7 Max failed before generation on
its only exact compatible route. Those calls establish compatibility only: they contain zero
prospective quality observations and are not rank-eligible.

The preregistered paired estimator and its identifiable two-endpoint Bradley-Terry subproblem have
a reproducible precollection Monte Carlo check at
`contracts/season1/method-validation/season1-statistical-method-validation-0b4345e523fdaa97d1b406cd1f2165540d0f9ad338bb49f3ac656da73e3c1933.json`.
It uses 2,000 datasets per scenario, obtains nominal coverage and type I error, and contains zero
scored model observations. `flavourbench-build-research-release` and
`flavourbench-verify-research-release` provide the deterministic signed archive boundary for the
four canonical Season 1 analysis cells. The archive boundary cannot pass until the real frozen run
and independent reviews exist. A leaderboard-ready archive must additionally contain the four
content-addressed v5 robustness artifacts and bind their exact digests in its signed manifest.

The older invitation-only original-task intake is implemented at `POST /v1/task-contributions`.
It remains inactive for this public-source campaign. Admission uses a
privately verified identity handle to derive a season-specific commitment; the raw handle is not
stored. Before the route is enabled, the invitation holder must accept the exact content-addressed
task-contributor protocol through `POST /v1/task-contributions/protocol-acceptance`. Superseded or
arbitrary protocol hashes fail closed. The acceptance governs task authorship and redistribution
only; it does not admit an output rater or resolve the human-subjects determination. Each submitted
candidate binds the acceptance event, protocol version, and protocol digest into its immutable
record. Qualified independent reviewers use
`GET /v1/expert/task-candidates/next` and
the blind-validity, reconciliation, adjudication, validator-review, and contamination-review
endpoints. Before bank import, an author can use the receipt-bound, idempotent
`POST /v1/task-contributions/{candidate_id}/withdrawal` route. Withdrawal is append-only,
serializes against review and import, removes the candidate from assignment, and cannot be
overridden by bank assembly. After import, corrections use the task-challenge and retirement
ledger. Its current bank import verifies the candidate hash, six-person role separation,
qualification, evidence receipts, review-history digest, labeled detector calibration, and task
lifecycle against the append-only ledger. This is why it cannot import the v6 public-source
campaign. Submission or approval alone never makes an item rank-eligible. The original-contributor protocol is
`../protocol/FLAVOURBENCH-TASK-CONTRIBUTOR-v2.md`.

## Historical frozen frontier contract smokes

The current 12-model OpenRouter panel is a content-addressed, unranked candidate
manifest. Plan its exact-endpoint tool contracts without making provider calls:

```bash
flavourbench-run-frontier-contracts \
  --manifest artifacts/manifests/flavourbench-openrouter-unranked-eb9e9b591d1695c38aeb79d65b59904d848b41dea449090eaeff8ebbed2138a2.json
```

The plan verifies every manifest and artifact hash, applies append-only cost
corrections, includes paid failed generations, and reserves the panel in manifest
order against the authorised USD 100 hard cap and USD 85 admission ceiling.
Execution additionally requires `--execute` and
`--confirm RUN_SEQUENTIAL_UNRANKED_FRONTIER_CONTRACTS`. It delegates exactly one
contract at a time to `flavourbench-live-smoke`, passes the manifest digest into
the artifact, freezes the endpoint tag with fallbacks disabled, derives endpoint
price guards, and fsyncs a hash-chained reservation ledger before any request.
These artifacts can never enter a leaderboard.

## Governed real exploratory dataset

The real-dataset runner uses the immutable 12-model manifest, exact endpoint
tags, and real Epicure on/off arms. Its default workload is 10 paired candidate
task assignments per model (120 pairs / 240 arms). A deterministic 12-task pool
is selected across all four families; the rotated 2/2/3/3 assignment gives 30
model-task pairs per family across the complete panel.

The default command is a dry plan and makes no OpenRouter generation calls:

```bash
flavourbench-run-real-dataset \
  --manifest artifacts/manifests/flavourbench-openrouter-unranked-aaf43f1bd770df5f120d79b66058cfad5092d5fb950e80bd24fac6d1d2e9acb5.json
```

The default content-addressed pilot policy is deliberately lean and uniform:
1,000 output tokens, four Epicure rounds, 16,384 bytes per tool result,
49,152 cumulative tool-result bytes, four calls per round, 12 calls total, one
provider attempt, temperature 0.2, top-p 0.95, and seed 20260715. CLI flags or
the matching `FLAVOURBENCH_DATASET_*` environment variables can freeze another
policy. The policy digest is part of every work-item, source, response, ledger,
and summary identity.

Paid execution is unavailable unless the complete balanced 120-pair block's
worst-case reserve, all verified prior costs, and every active shared frontier
reservation remain at or below the USD 85 admission ceiling. It also requires
the normal live-service environment, `--execute`, and the exact acknowledgement:

```bash
flavourbench-run-real-dataset \
  --manifest artifacts/manifests/flavourbench-openrouter-unranked-aaf43f1bd770df5f120d79b66058cfad5092d5fb950e80bd24fac6d1d2e9acb5.json \
  --execute \
  --confirm RUN_SEQUENTIAL_UNRANKED_REAL_DATASET
```

Work items run one at a time. The two arms within a pair run concurrently to
avoid a fixed Epicure-off/Epicure-on order confound. Before either paid arm, the
subprocess verifies the canonical model, endpoint execution hash, and policy
hash. Every generation is reconciled through OpenRouter accounting. Responses
are written separately as append-only, content-addressed artifacts only after
their costs reconcile. A hash-chained fsynced ledger reserves each pair before
the call; an active reservation without a fully accounted source is never
replayed on resume.

All records from this runner are explicitly exploratory, unranked, non-official,
and not approved research results. They contain no human preference labels and
cannot be used for Bradley-Terry rankings.

Build an incremental, no-call evidence checkpoint from the immutable real
OpenRouter + Epicure records at any time with:

```bash
flavourbench-aggregate-real-evidence
```

The aggregator verifies every source and response content address, every live
run journal and hash chain, the dataset ledger, frozen manifest, workload links,
generation accounting, and all prior runner summaries. It reports the exact
model × task-family × condition cube for response normalization, the
acknowledgement-only constraint validator, tool calls, latency, tokens, and
actual reconciled cost. These operational measures are never converted into a
quality, preference, or uplift score. Re-running against unchanged inputs is
byte-identical. `--publish` copies the aggregate to the paper and webapp only
after all 120 pairs are finalized; interim checkpoints are rejected.

## Isolated Amazon Bedrock lane (no-call scaffold)

The Bedrock lane is isolated from the active OpenRouter runner. It is not wired
into `provider.py`, does not change existing run artifacts, and makes no request
merely by being imported. Install its optional SDK dependency with:

```bash
.venv/bin/pip install -e '.[dev,bedrock]'
```

Authentication is delegated to boto3. The SDK recognizes
`AWS_BEARER_TOKEN_BEDROCK`; when that variable is absent, it uses the standard
boto3 credential provider chain. FlavourBench never accepts the credential as
an application argument and never includes its value in settings, logs,
manifests, request metadata, or provenance.

For this workspace, load the protected environment without displaying it:

```bash
set -a
source "${EPICURE_ROOT:?set EPICURE_ROOT}/.env"
set +a
```

Live use additionally requires these explicit controls:

```bash
export FLAVOURBENCH_BEDROCK_ENABLED=true
export FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED=true
export FLAVOURBENCH_BEDROCK_CAP_USD=5000
export FLAVOURBENCH_BEDROCK_STAGE=contract_smoke
export FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_CAP_USD=5
```

`FLAVOURBENCH_BEDROCK_HARD_CAP_USD` remains accepted as a compatibility alias.
If both cap variables are present, they must parse to the same decimal value or
configuration fails closed.

`AWS_REGION` is mandatory. It is the real Bedrock control/runtime ingress
Region and is never silently defaulted. Cross-Region routing is a separate,
hashed contract field: for example, `AWS_REGION=eu-west-1` with
`FLAVOURBENCH_BEDROCK_PROFILE_SCOPE=global` means the client uses the
`eu-west-1` service endpoint while the frozen `global.*` inference-profile ID,
redacted profile ARN plus original-ARN SHA-256, and destination model ARNs
describe Bedrock's routing scope. Raw AWS account IDs are never persisted in a
public catalog or provenance record.

The USD 5,000 value is a hard authorization ceiling, not an instruction to
spend it. B0 catalog discovery has a paid ceiling of USD 0. B1 contract smoke
is independently capped at USD 5 inside the cumulative authorization and
applies the same 85% admission stop, 95% drain, and 100% hard-stop rules. A
transactional worker integration must reserve worst-case cost before calling
the runtime. Advancing to `exploratory` or `season` additionally requires the
immutable B1 contract-smoke evidence digest in
`FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_EVIDENCE_SHA256`.

The following first prints a redacted, zero-network plan. The second command is
the exact B0 catalog preflight: it performs only `ListFoundationModels`,
`ListInferenceProfiles`, and `ListProvisionedModelThroughputs`, writes a
content-addressed catalog, and makes zero inference calls.

```bash
flavourbench-bedrock-preflight

flavourbench-bedrock-preflight \
  --discover \
  --confirm DISCOVER_BEDROCK_CATALOG_WITHOUT_INFERENCE \
  --output-dir artifacts/bedrock/catalog
```

No model becomes eligible from discovery alone. A frozen endpoint contract
must separately attest Converse, strict tool use, structured output, price-card
source, canonical model mapping, and all destination model ARNs. Bedrock is the
primary route. An OpenRouter fallback is permitted only for the identical
canonical model and is always persisted with `provider_substitution=true`,
`rank_eligible=false`, and an unpooled provider-substitution group. Invalid
Bedrock output or uncertain delivery never triggers fallback.

Converse provenance records AWS request IDs, ingress Region, profile ID,
redacted profile ARN plus hash, destination ARNs, returned model identifiers
when AWS provides them, latency, usage, stop reason, retries, schema/tool
hashes, and cost provenance. Token-rate
costs are labelled frozen rate-card estimates; they are never represented as
actual AWS charges unless independently reconciled against AWS billing data.

### B1 Bedrock + Epicure contract smoke

B1 now has a complete, real, unranked Claude Haiku 4.5 pair through the global
Bedrock inference profile. The Anthropic use-case form was submitted with
`PutUseCaseForModelAccess` (HTTP 201), read back without retaining its contents,
and the control plane reported agreement `AVAILABLE`, authorization
`AUTHORIZED`, entitlement `AVAILABLE`, and Region `AVAILABLE` before inference.

Protocol v8 uses manifest
`13e55aa50acea7ac5ba06ccf055e4d19eadb01e7a92007b996bce41d5a8293f3`
and run key
`9d2098b7a81afc309918312f9fdeff59d6dd4ac938a1260469c3e05d8be73157`.
The Epicure-off arm completed one HTTP-200 Bedrock turn (369 input, 860 output
tokens; 8,823 ms service latency). The Epicure-on arm completed two HTTP-200
turns (2,957 input, 1,971 output tokens; 29,646 ms combined service latency),
with one successful real `find_pairings` call to the private Epicure MCP between
them. Both arms ended with `end_turn` and valid normalized structured output.
There were no retries or provider substitutions.

The pair's frozen-rate-card estimate is USD 0.017481: USD 0.004669 off and USD
0.012812 on. AWS billing actual remains null until independently reconciled, so
these figures are estimates rather than claimed charges. The summary semantic
digest is
`46353f27c009c817867e9e9a4c90195ca4820c8690c39dd8d7e19524b72a92e2`
(physical SHA-256
`56567f99377fded86f443e1170fb29938b5163913f6756ed68ca2d6ea03e4c69`).
The 85-entry append-only execution ledger has physical SHA-256
`0a6a808bb58e89f05e3565df88689238424a3c69c4fcde5fdedf4b138bea7ace`.
All B1 artifacts remain `official=false` and `rank_eligible=false`: one
engineering pair is proof of the route and tool loop, not a quality or uplift
result.

Earlier protocol identities remain terminal and must not be replayed. They
include pre-generation request/access rejections, one delivered truncated
answer, and exploratory schema/grammar/journaling failures. Immutable holds are
retained where delivery or billing remained uncertain. Current governed
Bedrock exposure is USD 0.213568: USD 0.032608 in settled rate-card estimates
plus USD 0.180960 in conservative historical holds. It is not pooled with the
OpenRouter cap.

Dry-run remains the default and creates no AWS or MCP client. The current
manifest can be inspected without crossing a provider boundary:

```bash
.venv/bin/python -m flavourbench.bedrock_contract_smoke \
  --catalog artifacts/bedrock/catalog/bedrock-catalog-bd78cad4246faff8cd72fd288dd268e856692eb00314a19f0e916bb9318144e6.json \
  --evidence contracts/evidence/claude-haiku-4-5-global-2026-07-15-v10.json \
  --epicure-contract contracts/epicure/exploratory-unmatched-1790-runtime.json \
  --epicure-tool-catalog contracts/epicure/tool-catalog-666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd.json \
  --manifest artifacts/bedrock/contracts/bedrock-smoke-manifest-13e55aa50acea7ac5ba06ccf055e4d19eadb01e7a92007b996bce41d5a8293f3.json \
  --expected-manifest-sha256 13e55aa50acea7ac5ba06ccf055e4d19eadb01e7a92007b996bce41d5a8293f3
```

The live CLI calls `GetUseCaseForModelAccess` before ledger creation, MCP,
`CountTokens`, or `Converse`, and discards returned form contents.

Before each arm, a locked hash-chain ledger reserves the rate-card worst case.
Before every paid `Converse`, the free `CountTokens` API checks the exact
messages, system content, and projected strict tool configuration against the
12,000-token per-call reservation using the in-Region foundation-model ID. Its
result and the subsequent `Converse` request boundary are separately fsynced.
Count failure or an over-bound result prevents the paid call. Ambiguous paid
delivery holds the full reservation and blocks replay; clean restart recovery
reuses completed artifacts or releases only a provably pre-send reservation.
The frozen raw 13-tool Epicure catalog retains its complete MCP constraints at
hash `666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd`;
a deterministic one-tool Bedrock projection is recorded separately at hash
`73ec3f96008b44e87524acde9cee4c247b333607e86453e062d17d4b26ce7d7b`.
The Epicure-on arm must produce a real MCP
tool trace. The two arms use the same authored non-public, non-PII prompt,
bounded tokens and tool rounds, and remain unranked. Nova 2 Lite is not used for
this smoke because the frozen B1 contract requires native Bedrock
structured-output evidence in addition to Converse and tool use.

AWS references: [Bedrock API-key use](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html),
[Converse](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html),
[CountTokens](https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html),
[structured output](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html),
[client-side tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-client-side.html), and
[inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html).

### GPT-5.6 on the Bedrock Mantle Responses endpoint

GPT-5.6 Sol, Terra, and Luna are Mantle-only frontier contracts and do not use
the Converse catalog above. Their isolated implementation is in
`bedrock_mantle.py` with separate routing in `bedrock_mantle_routing.py`.
Mantle uses the OpenAI-compatible Responses API at the model-specific
`https://bedrock-mantle.{region}.api.aws/openai/v1` base URL. There is no
`global.*` profile: Sol is restricted to `us-east-1` or `us-east-2`; Terra and
Luna may additionally use `us-west-2`.

This command is credential-free and makes zero provider or inference calls:

```bash
flavourbench-bedrock-mantle-plan
```

It reports the three canonical FlavourBench bindings, AWS-documented Mantle
model IDs (`openai.gpt-5.6-sol`, `openai.gpt-5.6-terra`, and
`openai.gpt-5.6-luna`), allowed ingress Regions, and the remaining freeze
gates. Before inference, each documented ID must also be confirmed in a
content-addressed Mantle `/models` snapshot, then paired with model-specific
function-tool, `text.format` `json_schema`, request-ID, and pricing evidence. A
canonical contract can be checked and its worst-case reservation forecast
without network access:

```bash
flavourbench-bedrock-mantle-plan \
  --contract artifacts/bedrock/mantle/contracts/gpt-5.6-sol.json
```

The provider always sends `store=false`, reconstructs tool-loop context
client-side, and executes Epicure through the caller's private client-side
executor. It never registers Epicure as a server-owned MCP connector. Every
round retains canonical request/response JSON plus hashes, Responses IDs, AWS
request IDs, returned model identity, usage, latency, and the complete hashed
tool trace. The Mantle module never reads or copies
`AWS_BEARER_TOKEN_BEDROCK`; a caller injects an already authenticated
transport, and authentication headers are never included in provenance.
An exploratory contract may use `client_validation_only`: this omits the
unproven native `text.format` request, instructs the model to emit the frozen
JSON shape, validates it locally, and remains ineligible for ranking. Native
`responses_json_schema` is required for season eligibility.

Budget admission is mandatory before the first request. The worker-facing
budget controller must transactionally reserve the worst case at non-cache
rates and enforce the existing Bedrock hard cap. Because Responses exposes no
request parameter that enforces an input-token ceiling, admission reserves the
entire remaining 272K model context for every permitted response round; a
tokenizer estimate therefore cannot under-reserve a call. Contract-smoke runs
must use correspondingly small response/tool-round bounds. Only failures proved to be
`not_sent` may retry, at most twice. Ambiguous delivery is never retried and
holds the full reservation. Only an explicit pre-inference route rejection may
trigger an exact-same-canonical OpenRouter fallback, which remains
`provider_substitution=true`, unranked, and unpooled. Responses usage is priced
only as a frozen rate-card estimate until independently reconciled with AWS
billing.

AWS references: [Mantle Responses](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html),
[client-side tools](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-client-side.html), and
[GPT-5.6 Regions](https://aws.amazon.com/about-aws/whats-new/2026/07/openai-gpt-sol-terra/).

## Service boundaries

- FastAPI admits battles, hides identities, records idempotent votes, serves
  catalog/leaderboard reads, and exposes token-protected governance operations.
- The worker is the only process that calls OpenRouter or Epicure MCP. It runs
  both arms concurrently, persists complete tool traces, rejects model/provider
  substitutions, and reconciles generation costs.
- PostgreSQL owns admission counters and durable jobs. Redis is not required.
- The Next.js same-origin proxy is the only public caller. It verifies origin
  and Turnstile, then derives a season-scoped HMAC pseudonym without storing a
  raw IP address.

## Season lifecycle

The service deliberately cannot turn seeded fixtures into an official season
by configuration alone. Admin operations must, in order:

1. synchronize the OpenRouter catalog;
2. record tool/structured-output/provider-policy smoke evidence;
3. assemble the exact 240-task ID/hash map from sealed human candidates and two independently
   verified approval events per task;
4. freeze a unique 16-model 4/8/2/2 manifest with public Epicure lineage and a
   non-zero budget authorization reference;
5. create one-time, qualification-scoped expert invitations; and
6. admit a prospective season through Gate A, privacy, security, and reviewer
   access references.

Research exports include only consented battles that have a separate approved
release-review event. Invitation tokens are returned once and stored only as
SHA-256 digests.

## Terminal real-data operational figure

`flavourbench.operational_figure` turns the verified terminal real-exploratory
aggregate into one renderer-neutral, content-addressed view model plus
deterministic CSV and TikZ outputs. The same JSON powers the accessible web
figure. It preserves the frozen manifest order and reports pair outcome
composition, off/on normalization availability, complete Epicure MCP call
outcomes, known-ID generation cost, and survivor-only p50 latency with its
sample size. It never computes or sorts by a model score.

Rendering is fail closed. The command requires the exact terminal aggregate
digest and rejects partial collection, an unverified terminal runner summary,
the wrong Protocol v1 policy hash, any non-terminal pair, any public or expert
judgment, cost drift, or a latency denominator that omits a normalized response.
No paper or web destination is changed unless `--publish` is supplied.

```bash
.venv/bin/python -m flavourbench.operational_figure \
  --aggregate artifacts/aggregates/real-exploratory-evidence-<sha256>.json \
  --expected-aggregate-sha256 <sha256> \
  --publish
```

The figure labels itself `NOT A QUALITY LEADERBOARD`. Protocol v1's 1,000-token
ceiling and model-dependent attrition are visible. Model cost bars allocate only
identified, provider-reconciled OpenRouter generation metadata; any conservative
HTTP-200/no-choice no-ID increment stays separate and is never labelled provider
spend. Latency is descriptive only for normalized-response survivors, with `n`
shown and an explicit cross-model comparison warning.

## Verification

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/pytest -q
```
