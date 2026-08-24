# FlavourBench task-contributor protocol

Protocol version: `flavourbench-human-task-contributor-v2`

## Purpose

This invitation-only pathway collects original culinary scenarios for the prospective FlavourBench
Season 1 task bank. Contributors write the prompt, constraints, unacceptable outcomes, and an
acceptable-solution outline. Model-written, model-rewritten, translated, expanded, or selected
tasks are not eligible. Contributors do not see or judge model answers through this pathway.

## Eligible content

Each task must belong to one general-track family:

1. ingredient substitution under explicit constraints;
2. multi-ingredient composition and bridge reasoning;
3. recipe design and practical cookability; or
4. interpretation of Epicure evidence with appropriate uncertainty.

Formal food-safety, allergen, nutrition, medical, legal, and cultural-authenticity evaluation is
outside the general track. A task that depends on specialist adjudication is rejected or routed to
a separately governed benchmark.

## Authorship and rights

The contributor attests that the scenario is their original human work and was not copied from a
recipe, benchmark, website, book, private document, or another person. Employer-owned material is
eligible only with authority to license it for research and public benchmark use.

The contributor grants Josef Chen, Independent Researcher, permission to store, edit, evaluate,
publish, and redistribute the task and its derived metadata as part of FlavourBench. No payment,
authorship, or publication credit is promised. A material revision creates a new immutable record.

## Person uniqueness and privacy

An administrator privately verifies one identity handle before issuing a contributor record. The
service derives a season-specific HMAC commitment and immediately discards the raw handle. This
commitment prevents one person from occupying multiple nominal accounts or more than one role on
the same task. It is not published and cannot be used across seasons without the season secret.

The benchmark stores the pseudonymous contributor record, invitation hash, person commitment,
task content, attestations, timestamps, and review events. It stores no raw identity handle or raw
network address. Contributors must not include names, contact details, private recipes,
credentials, health information, trade secrets, or other confidential material in a task.

Participation is voluntary. Before a task bank is frozen, the invitation holder may request
withdrawal using the candidate receipt hash. Withdrawal cannot be guaranteed after a task enters a
public release; later confirmed defects follow the correction procedure below.

## Independent admission

Submission never makes a task rank eligible. Each task requires six distinct verified people:

- one author;
- two source validators;
- one adjudicator;
- one executable-validator reviewer; and
- one contamination-review auditor.

Each source validator first solves and classifies the prompt without the author pack or model
output. The validator then reconciles that sealed record against the author pack. The adjudicator
reviews the two source records, resolves disagreements, and freezes the criterion pack shown to
response raters. The remaining reviewers independently reproduce the executable-validator and
contamination receipts. Any rejection stops admission; revisions use a new versioned candidate.

The bank import verifies authorship, qualification, six-person separation, event hashes, review
history, split and construct quotas, contamination calibration, and lifecycle ordering. No model
output may be visible during task admission.

## Lifecycle and correction

Authorship, sealing, first use, release, and retirement are append-only ordered events. A public
content-addressed challenge is considered by at least two qualified adjudicators who held no role
on the original task. A confirmed defect retires the item, excludes its battles from ranking, and
requires leaderboard snapshots to be recomputed. A replacement is a new versioned task and must
pass the complete admission process.

## Permitted methods statement

The following statement may be used only after the event ledger verifies every condition:

> Prospective tasks were contributed through an invitation-only pseudonymous pathway with original
> human-authorship and rights attestations. Six distinct verified people completed authorship,
> source validation, adjudication, validator review, and contamination review for each task. Raw
> identity handles were not retained.
