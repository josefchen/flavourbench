# Manuscript wording for reasoning effort and sensitivity status

## Methods disclosure

All exact-frontier records analyzed in this study were generated with low intermediate reasoning
effort and low final reasoning effort. They do not represent provider-default or maximum-compute
performance. Reasoning effort is therefore part of the evaluated system configuration, not an
uncontrolled model attribute.

## Sensitivity outcome

We planned a contemporaneous sensitivity study with explicit low effort, omitted provider-default
effort, and explicit high effort. Each setting paired an Epicure-off arm with an Epicure-on arm.
Before collecting the prespecified 36 pairs, we ran one real route-check pair at each setting. The
three checks comprised six intended response arms and issued 11 provider requests. None produced a
usable matched pair, and no model-issued Epicure tool call completed successfully. One off arm
returned a final answer, but every pair failed. We therefore estimated no reasoning-effort or
effort-by-Epicure effect. These records did not enter a leaderboard, preference fit, or quality
analysis.

Five failed requests returned HTTP 200 with the same retained response digest. The data-minimization
policy preserved safe transport metadata and the digest, but not the raw response bodies or provider
messages. The retained record cannot distinguish an OpenRouter error object, a Cloudflare gateway
object, or an upstream schema mismatch. We report the failure class as indeterminate rather than
assigning it to a model or provider.

The v1 checks incurred USD 0.140822 in identifiable generation cost and retained USD 4.78568 in
conservative exposure. The subsequent v2 explicit-low pair identified USD 0.097758 in generation
cost but recovered no usable pair after an HTTP 200 error envelope carrying code 429 interrupted
the treatment route. Its accounting remained incomplete, so the admitted allowance was retained.

A fresh v3 gate classified HTTP 200 error envelopes before generation accounting. Both conditions
returned code 429, received one bounded retry with fresh attempt identifiers, and returned code 429
again. The corrected audit records four safe rejections, two retried and two terminal, zero unsafe
rejections, zero accepted generations, zero generation-cost lookups, and zero identified generation
cost. All v3 identifiers are closed and no v4 replay is authorized. The full 72-arm study remains
blocked until separately governed endpoint requalification or a fixed-route change supplies new
evidence. Route-check outputs are diagnostic and will not be reused in any quality analysis.
