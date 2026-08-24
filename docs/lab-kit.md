# FlavourBench lab kit

The lab kit has two jobs: evaluate a checkpoint without modifying the benchmark, and optimize a
checkpoint on a separate set of Epicure reward maps. Those jobs use different task IDs and
different ingredient anchors.

## What is public

- A provider-neutral evaluator and strict response parser
- The 534-task published evaluation map used for the paper
- 270 training maps, 72 validation maps, and an 84-task predeclared transfer split
- SFT demonstrations, DPO preference pairs, and GRPO-ready dense rewards
- A Hugging Face Space UI and named Gradio API endpoints
- Content hashes for every dataset table and evaluation report

The public reward function is a lookup over a task's 56 precomputed Epicure scores. It is not the
private Epicure ingredient corpus or a general endpoint for creating arbitrary new culinary tasks.

## Install

Python 3.12 is recommended.

```bash
pip install 'epicure-flavourbench @ git+https://github.com/josefchen/flavourbench.git'
flavourbench --help
```

For local checkpoint execution:

```bash
pip install 'epicure-flavourbench[transformers] @ git+https://github.com/josefchen/flavourbench.git'
```

## Evaluate an endpoint

Any OpenAI-compatible chat-completions endpoint works, including a lab's internal gateway or a
vLLM server. The credential is read from the named environment variable and is never serialized.

```bash
export LAB_MODEL_API_KEY='...'
flavourbench run \
  --backend openai-compatible \
  --base-url https://your-endpoint.example/v1 \
  --api-key-env LAB_MODEL_API_KEY \
  --model exact-checkpoint-or-route-id \
  --responses responses.jsonl \
  --report report.json \
  --resume
```

Use `--limit 8` for a transport smoke test. A limited run cannot produce a comparable score.
Provider-specific parameters can be supplied with `--extra-body`, but that object cannot replace
the model ID or messages.

## Evaluate a local checkpoint

```bash
flavourbench run \
  --backend transformers \
  --model your-org/your-checkpoint \
  --batch-size 4 \
  --responses responses.jsonl \
  --report report.json
```

A vLLM deployment should use the OpenAI-compatible route instead; it gives the lab control over
tensor parallelism, quantization, and scheduling without changing FlavourBench.

## Score existing outputs

The smallest accepted JSONL row is:

```json
{"task_id":"fb-executable-substitution-136","response":"FINAL_SELECTION: A,F,H"}
```

`completion`, `answer`, and `answer_markdown` are accepted aliases. Duplicate or unknown task IDs
fail closed. A failed row may be retained with `"status":"failed"`, but it prevents the report
from receiving a comparable score.

```bash
flavourbench template --output responses.jsonl
flavourbench score responses.jsonl \
  --output report.json \
  --per-task per-task.jsonl
flavourbench verify-report report.json
```

The report contains:

- the equal-family FlavourBench Score, but only at 100% valid coverage;
- family scores and coverage;
- a 50,000-resample 95% bootstrap interval for a complete run;
- the exact expected score of a uniformly random three-item portfolio;
- a 100,000-resample paired sign-flip test against that chance baseline; and
- semantic hashes of the task and response artifacts.

This is a single-model confidence interval. An official multi-model release also needs shared
bootstrap draws, simultaneous intervals, pairwise tests, and multiplicity correction.

## Use one reward

```python
from flavourbench.lab import load_hub_tasks, reward

tasks = load_hub_tasks(filename="data-lab/train_tasks.jsonl")
value = reward(tasks[0], "FINAL_SELECTION: A,B,C")  # float in [0, 1]
```

Or from the command line:

```bash
flavourbench reward \
  --dataset-file data-lab/train_tasks.jsonl \
  --task-id fb-executable-rep2-substitution-002 \
  --completion 'FINAL_SELECTION: A,B,C'
```

## Train

```python
from datasets import load_dataset

sft = load_dataset("josefchen/flavourbench", "sft")
dpo = load_dataset("josefchen/flavourbench", "dpo")
grpo = load_dataset("josefchen/flavourbench", "grpo")
```

SFT rows include `optimal_margin_bps`, which lets a lab restrict demonstrations to clear optima.
Each DPO pair has a reward gap of at least 500 basis points (five FlavourBench points); the export
does not turn near-ties into preference labels.

The scripts in [`examples/lab`](../examples/lab) are directly runnable on Hugging Face Jobs. They
use LoRA, a validation split, Trackio, checkpoint retention, and `push_to_hub=True`. The GRPO
script passes the row's `choices` and `selection_scores_bps` columns to
`flavourbench.lab.trl_reward`; reward calculation is local and deterministic.

## Space reward API

The existing FlavourBench Space hosts the released-map API. Gradio generates the API documentation
and client code from named endpoints.

```python
from gradio_client import Client

client = Client("josefchen/flavourbench")
result = client.predict(
    task_id="fb-executable-substitution-136",
    completion="FINAL_SELECTION: A,F,H",
    api_name="/score_completion",
)
```

`/score_submission` accepts a JSON string containing a complete response array and returns the
same coverage-gated report as the upload interface. The API is useful for integration tests and
small experiments. High-throughput RL should use the local reward maps to avoid network latency
and Space rate limits.

The separate `/training_reward` endpoint accepts IDs only from the 270-task training and 72-task
validation partitions. It omits the 84 transfer-evaluation maps:

```python
result = client.predict(
    task_id="fb-executable-rep2-substitution-002",
    completion="FINAL_SELECTION: A,B,C",
    api_name="/training_reward",
)
assert result["official_leaderboard_eligible"] is False
```

This separation prevents an RL loop from silently optimizing on either the transfer split or the
public leaderboard map.

Hugging Face can therefore host the released FlavourBench reward service. It should not be
described as the full Epicure API: arbitrary task generation would require the separate Epicure
runtime and its private data boundary.

## Contamination and official submissions

The paper's 534 task maps become visible when the dataset is public. They remain appropriate for
reproducing the paper and evaluating checkpoints that did not train on them. Fine-tuning directly
on those maps invalidates a prospective leaderboard claim.

The Space scores uploaded artifacts but does not silently add them to the official table. A future
official submission must bind the exact checkpoint, route, decoding settings, dataset-exposure
declaration, task-set hash, and complete response artifact. Models trained after publication need
a newly generated server-side test panel with disjoint anchors.

## Rights

Original software is Apache-2.0. FlavourBench task prompts, candidate layouts, metadata, and
derived training views are CC BY 4.0. Provider responses keep their provider terms. The private
Epicure ingredient data and embeddings are not redistributed. See [`LICENSES.md`](../LICENSES.md).
