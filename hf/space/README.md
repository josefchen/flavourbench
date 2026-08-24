---
title: FlavourBench
emoji: 🍲
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: other
datasets:
- josefchen/flavourbench
tags:
- leaderboard
- modality:text
- judge:auto
- submission:manual
- test:public
- reproducibility
- culinary
---

# FlavourBench

![Which AI knows food best? FlavourBench leaderboard](./assets/flavourbench-leaderboard.svg)

**Pick 3 ingredients from 8. Epicure scores all 56 legal portfolios first. Then every model faces
the same 534 decisions.**

The Space is both the public scorebook and a working benchmark interface:

- **Leaders** ranks all 27 endpoints with simultaneous intervals and statistical groups.
- **Insights** shows score bands, pairwise resolution, and task-family fingerprints.
- **Profiles** breaks each score into substitution, pairing, and constraint performance.
- **Inspect** opens the exact prompt, answer, 56-choice score map, route, and content hashes.
- **Run your model** builds a copyable endpoint or local-checkpoint command, demonstrates one dense
  training reward, and scores a complete JSONL run.
- **Compare** queries any of the 351 shared-task pairwise contrasts.

No model judge runs behind the interface. The Space performs deterministic lookups against the
released reward maps and makes no model-provider calls.

## Run from your own environment

```bash
python -m pip install "epicure-flavourbench @ git+https://github.com/josefchen/flavourbench.git"

export LAB_MODEL_API_KEY='...'
flavourbench run \
  --backend openai-compatible \
  --base-url https://your-endpoint.example/v1 \
  --api-key-env LAB_MODEL_API_KEY \
  --model your-exact-model-id \
  --responses responses.jsonl \
  --report flavourbench-report.json \
  --resume
```

Add `--limit 12` for a balanced smoke test. The runner checkpoints each answer and resumes without
repeating completed calls. Credentials and model weights stay in your environment.

The accepted response contract is one JSON object per line:

```json
{"task_id":"...","status":"completed","response":"FINAL_SELECTION: A,B,C"}
```

A comparable score requires one valid answer for all 534 tasks. Partial runs still receive
per-task and coverage diagnostics. Uploads are never added to the official leaderboard
automatically.

Verified complete runs can be proposed through the
[result submission form](https://github.com/josefchen/flavourbench/issues/new?template=flavourbench-result.yml).
The [submission contract](https://github.com/josefchen/flavourbench/blob/main/docs/submitting-results.md)
lists the required response artifact, route metadata, settings, and training disclosure. Accepted
results enter a new versioned release; the published release is never edited in place.

## API and training

The Space exposes four named endpoints:

| Endpoint | Use |
|---|---|
| `/score_completion` | Score one completion on one official task |
| `/score_submission` | Score a complete JSON or JSONL artifact supplied as text |
| `/training_reward` | Query one of 342 anchor-disjoint train/validation reward maps |
| `/score_uploaded_submission` | Score an uploaded artifact and return a report |

Use **Use via API** in the running Space for generated Python, JavaScript, and curl clients. For
high-throughput RL, use the local deterministic reward function. The linked dataset includes
ready-to-load SFT, DPO, and GRPO views plus runnable LoRA recipes for Hugging Face Jobs.

The [prospective reward-transfer protocol](https://github.com/josefchen/flavourbench/blob/main/docs/reward-transfer-study.md)
defines the 84-task transfer split, seeds, six confirmatory contrasts, and multiplicity control.

[Dataset and lab kit](https://huggingface.co/datasets/josefchen/flavourbench)&nbsp;&nbsp;&nbsp;
[Paper](https://arxiv.org/abs/2608.20574)&nbsp;&nbsp;&nbsp;
[Source](https://github.com/josefchen/flavourbench)

Josef Chen, Independent Researcher<br>
Erim Hayretci, Imperial College London

```bibtex
@article{chen2026flavourbench,
  title  = {FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth},
  author = {Chen, Josef and Hayretci, Erim},
  journal = {arXiv preprint arXiv:2608.20574},
  year   = {2026},
  eprint = {2608.20574},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2608.20574}
}
```
