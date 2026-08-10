# Reasoning-effort sensitivity protocol v1

## Configuration disclosure

The exact-frontier development records froze `low` for both the intermediate reasoning phases and
the normalized final response. This is verified in the content-addressed routed manifest
`e87a164d59bdd88eaf630c153755b5ec3c513e8b3770a17afac67037eb135910`; it is not a retrospective
interpretation of model behavior.

Any report using those records must state the two explicit `low` settings. It must not describe the
records as provider-default or maximum-compute frontier performance.

## Prespecified sensitivity

The development-only sensitivity uses the same exact endpoint contracts for:

- `openai/gpt-5.6-sol-pro`;
- `anthropic/claude-sonnet-5`;
- `deepseek/deepseek-v4-flash-0731`.

Every endpoint supports explicit low and high effort on its frozen route. The subset spans a closed
reasoning model, a second closed frontier family, and an efficient open-weight family while fitting
under the existing shared OpenRouter admission envelope. Model selection uses no FlavourBench
quality outcome.

Each model receives the same four real-human, non-suspect anchors: one substitution, one
composition, one cookability, and one evidence-interpretation task. The anchor schedule is supplied
by the separate task-quarantine and coverage-repair audit and is content-bound before planning.
Synthetic tasks and synthetic response arms are prohibited.

Three contemporaneous variants are collected:

1. explicit low, rerun rather than silently reusing an earlier trajectory;
2. provider default, implemented by omitting the reasoning-effort parameter;
3. explicit high.

Each variant has matched Epicure-off and Epicure-on arms. Every on arm must contain at least one
successful call to the exact attested Epicure runtime. Provider fallback and identity substitution
are prohibited. Failures remain in reliability denominators and do not receive preference ballots.

Before the 36-pair collection, one paid pair per effort variant is run sequentially. The full study
is admitted only if all six arms finish, every identity matches, each on arm completes a real
Epicure tool call, and provider accounting is complete. A failed smoke is immutable, retains its
full conservative reservation, and blocks the study until a revised protocol is frozen under new
work-item identities. It is never replayed in place.

Any provider-adapter or response-envelope code change requires a content-addressed v2 study. The
v1 work-item IDs remain closed. Future HTTP-200 responses that are not chat-completions objects are
classified fail-closed as provider-error, gateway, Responses-schema, or unknown envelopes. Only
error code, error type, and provider identity metadata may enter the journal; raw provider messages
and response bodies remain excluded.

The complete design contains 36 on/off pairs and 72 response arms. Its order is a
content-addressed permutation. Raw records are append-only and cannot supersede the original low
records.

## Cost and execution gates

`flavourbench.reasoning_effort_sensitivity` binds the latest content-addressed global budget audit,
computes a worst-case reserve from each frozen endpoint, and blocks before provider I/O if the
projected total exceeds the 85% admission ceiling or 100% hard stop. A passing cost calculation is
necessary but not sufficient. Development collection also requires an exact provider endpoint
contract for every model, a content-addressed Epicure bundle, application and semantic tool schema,
and a live private-runtime attestation that matches the recovered checkout. The runner then verifies
the same identities again before generation.

Rank eligibility, payload redistributability, a clean signed application release, dependency locks,
and public reconstruction are separate officialization gates. They do not block a clearly labelled,
non-ranking development sensitivity when the exact private runtime is attested. The current runtime
passes the development collection identity gate but fails those officialization gates. No resulting
record may be promoted into an official fit until they are resolved. It remains forbidden to
populate any result field with simulated outputs.

## Analysis boundary

The prespecified contrasts are provider-default minus explicit-low, explicit-high minus
explicit-low, and the effort-by-Epicure interaction within model. Quality outcomes require new
blinded human judgments. Inference uses paired task contrasts and task-cluster resampling; response
reuse is never treated as independent evidence. Operational completion, treatment success, cost,
latency, and tool-call counts are reported separately.

This study is a configuration and coverage sensitivity. It cannot enter an official quality fit or
repair missing official family support until the task, human-review, Epicure-release, and
officialization gates are independently cleared.
