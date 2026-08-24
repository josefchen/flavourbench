# Submit a FlavourBench result

FlavourBench accepts complete, reproducible model runs. A submission proposes a result for the
next versioned release. It never rewrites a published leaderboard.

## Required artifacts

Run all 534 official tasks with the released runner and keep both outputs:

```bash
flavourbench run \
  --backend openai-compatible \
  --base-url https://your-endpoint.example/v1 \
  --api-key-env LAB_MODEL_API_KEY \
  --model your-exact-model-id \
  --responses responses.jsonl \
  --report flavourbench-report.json \
  --resume
```

Submit:

1. `responses.jsonl`, with one completed and parseable response for every official task.
2. `flavourbench-report.json`, produced from that exact response file.
3. An immutable public URL for both artifacts, such as a Hugging Face dataset commit.
4. Exact model, provider, route, endpoint, region, and model-version identifiers.
5. Temperature, top-p, token limit, reasoning-effort setting, and any other decoding controls.
6. A disclosure of FlavourBench, Epicure, or related culinary data used in training or tuning.

Never include API keys, access tokens, private model weights, or customer data.

## Review contract

The maintainer checks that:

- the task-set semantic hash matches the current release;
- all 534 task IDs occur exactly once and every response is parseable;
- the offline verifier reproduces the submitted score and family scores;
- the response-set and report hashes match the submitted files;
- the route and generation settings are sufficient to rerun the evaluation; and
- the training disclosure is complete.

An accepted entry is labeled `author-run` unless a maintainer or third party independently reruns
the same route. Independently rerun entries may additionally be labeled `reproduced`.

## Open the review

Use the [FlavourBench result template](https://github.com/josefchen/flavourbench/issues/new?template=flavourbench-result.yml).
The issue is the review record; large artifacts should remain in the immutable dataset revision
linked from it.
