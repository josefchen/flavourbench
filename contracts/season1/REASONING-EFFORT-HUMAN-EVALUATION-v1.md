# Blinded human evaluation for the reasoning-effort study

Status: frozen design, development-only. The freeze command below performs no
provider, Epicure, or reviewer-service calls. It creates no human judgments.
Generation and review remain separate later operations.

## Study boundary

The protocol is cross-bound to the 24-task reasoning-effort executor plan. It
does not select a second task set. The shared contract fixes six licensed,
real-human-authored development questions in each of four culinary families.
The current four-task quarantine, specialist-scope holds, and text-dependency
holds are excluded. None of the 24 tasks is confirmatory or rank eligible.

The primary contrast is explicit low versus explicit high effort for each
endpoint and each Epicure condition:

| Endpoint | Fixed provider endpoint | Epicure off | Epicure on |
|---|---|---:|---:|
| Claude Sonnet 5 | Anthropic | low–high | low–high |
| Gemini 3.6 Flash | Google AI Studio Flex | low–high | low–high |
| DeepSeek V4 Flash | DeepInfra FP4 | low–high | low–high |

Gemini's omitted-parameter provider default, documented as medium effort, is a
secondary dose point. Low–default and default–high are analyzed separately.
There are no cross-model quality comparisons in this sensitivity study.

The shared immutable identities are:

- task-selection artifact:
  `ebb62d92fc0589d1b8473b72682e8c17f6f90d7d2b0b2fd23e9064d63958044a`;
- selected task set:
  `825526a211edd98a242f6eb5706114f2dc9c9921cc051c85b83d133a1bcbd682`;
- ordered task waves:
  `ec47d354d3794a8ad69533b740336911f99c310210e662e2a5f58a40d13bf76a`.

The final protocol additionally binds the exact executor-plan digest. Its
cross-verifier recomputes all 336 response-arm coordinates and all 240 human
comparison cells from that plan.

## Generation and presentation units

Each of the 24 task waves contains seven matched Epicure pairs: low and high
for Sonnet, Gemini, and DeepSeek, plus Gemini default. That is 14 real response
arms per task and 336 arms in total. Operational admission uses six atomic
four-task blocks, one task from each family per block. A block contains 28
pairs and 56 arms. Partial block admission is forbidden.

The human graph contains 144 primary low–high cells and 96 secondary Gemini
cells. Each cell has three planned judgments in each of two non-pooled cohorts:
qualified culinary experts and general-public reviewers. The resulting 1,440
original presentations are partitioned into 72 assignment blocks. Each block
has 20 distinct tasks and two exact position-swapped repeats, for 1,584 total
presentations.

The executor's side assignment is the anchor. Replication slot two mirrors
slot one. In slot three, the expert cohort retains the executor orientation and
the public cohort receives its mirror. Each cell is therefore balanced exactly
three-to-three across the six planned judgments. A reviewer may complete only
one assignment block and may belong to only one cohort. Repeats occur at
positions 15 and 22, after source presentations at positions 1 and 8; at least
13 presentations intervene. Repeat ballots are quality-control records and
never enter an effect estimate.

Reviewer delivery must omit endpoint, model, provider, effort, Epicure
condition, side orientation, and repeat-source fields. Identity is revealed
only after the ballot is sealed. The allocation itself is not released to
reviewers and is published only after all planned primary ballots have been
sealed or administratively withdrawn.

## Ballot

Before answers appear, the reviewer seals task validity and six context checks.
A valid task then requires:

- one preference: left, right, tie, or both bad;
- 1–5 scores for task completion, constraint compliance, coherence, sensory
  promise, cookability, clarity, originality, evidence use, and calibration;
- arm-specific failure tags, confidence, verification basis, an identity-leak
  flag, and a comparative rationale.

Originality and evidence use may be marked not applicable only with a reason.
Complete ballots are append-only; a correction is a content-addressed
superseding ballot. Research exports use a season-specific reviewer HMAC, not a
name or raw identity.

## Failures and missing outcomes

A technical generation failure, route mismatch, or invalid source arm prevents
delivery of that comparison. It is retained in reliability accounting; no
preference is solicited and no replacement is generated after output
inspection. A reviewer who marks the task revise or exclude does not see the
answers. Incomplete ballots are not imputed. Both-bad ballots remain observed
quality failures but have no directional preference value.

Every result table must show missingness by endpoint, condition, variant,
family, cohort, and reason. The complete-case estimate requires two valid real
arms and all three independent, valid directional-or-tie ballots for the
task-cell in that cohort. In parallel, worst- and best-case bounds use all
frozen slots: every missing directional outcome receives respectively zero or
one point for the higher-effort arm. Both bad is bounded in the same way.

## Inference

Preference points for the upper effort are 1 for an upper-effort win, 0.5 for a
tie, and 0 for a lower-effort win. Weighting is equal by family, then task, then
ballot. The primary interval uses 20,000 family-stratified task-cluster
resamples, drawing six tasks with replacement within each family and moving all
ballots and reused response arms for a selected task together. An interval is
suppressed if any family has fewer than four complete task clusters. A crossed
task-by-reviewer pigeonhole bootstrap is reported as a dependence sensitivity.
Individual ballot rows are never treated as independent observations.

The six endpoint-by-condition primary tests are Holm-adjusted within each
cohort. The four Gemini adjacent-dose tests form a separate Holm family. Expert
and public cohorts are never pooled, models are never pooled, and rubric
dimensions are reported separately rather than collapsed into a composite.

## Freeze and verify

Run from `flavourbench/` after the final executor plan has been frozen:

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.reasoning_effort_human_protocol \
  --repo-root .. \
  --task-dossier artifacts/season1/task-validity/development-v2/development-task-validity-v2-86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json \
  --executor-plan PATH_TO_FINAL_TASK_WAVE_PLAN \
  --protocol-schema contracts/season1/reasoning-effort-human-evaluation-protocol-v1.schema.json \
  --ballot-schema contracts/season1/reasoning-effort-human-ballot-v1.schema.json \
  --output-dir artifacts/season1/current-quality-run/reasoning-effort-human-protocol-v1
```

The human-side verifier checks the executor plan before writing. The executor
has an independent inverse verifier:

```bash
PYTHONPATH=src .venv/bin/python -m flavourbench.reasoning_effort_full_study_v1 \
  --repo-root .. verify-human-protocol \
  --plan PATH_TO_FINAL_TASK_WAVE_PLAN \
  --human-protocol PATH_PRINTED_BY_THE_FREEZE_COMMAND
```

Validate implementation and schemas with:

```bash
.venv/bin/ruff check src/flavourbench/reasoning_effort_human_protocol.py \
  tests/reasoning_effort_human_protocol_test.py

PYTHONPATH=src .venv/bin/pytest -q \
  tests/reasoning_effort_human_protocol_test.py
```

The superseded planning artifact
`4829a6832fa1aa71e2a65b979e5ef2a340ec416e442c12a9afc963562a8487c9`
used a different task-selection seed and only six reused wave labels. It is
retained as planning history and is not an execution or analysis authority.

## Claim boundary

This document precommits presentation and analysis choices. It contains no
model output and no human evidence. Even after execution, the study remains a
development sensitivity analysis: it cannot enter an official leaderboard,
establish a universal cross-model effort effect, or repair the independent
release and lineage requirements for Epicure.
