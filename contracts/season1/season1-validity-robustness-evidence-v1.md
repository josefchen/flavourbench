# Season 1 validity and robustness evidence contract

Status: prospective, required for a Season 1 benchmark-results release.

This contract defines the four completion artifacts required by study design v5. They are
empirical records, not model-ranking inputs. Each artifact is canonical JSON: UTF-8, object keys
sorted for hashing, and compact separators. `artifact_sha256` is SHA-256 over the object with that
field removed. Every artifact must declare `status: complete`, bind study-design artifact
`7a63cfd6117338a3af16a422d5ee3458298fdc0ff2fd0abfe45fe851a7e54506`, and declare
`synthetic_observations: 0`.

The release builder places the full artifacts under `validity-and-robustness/` and records their
four embedded digests in the signed research-archive manifest. An aggregate, screenshot, or
external link cannot satisfy this contract.

## Post-collection item audit

Schema: `flavourbench-season1-post-collection-item-audit-v1`.

`task_records` contains one row per unique audited task. Each row binds the task ID and content
hash, one or both controlled selection reasons (`random`, `anomaly`), and at least two distinct
season-specific auditor commitments. Original task-role holders are excluded. A no-defect row has
`resolution_status: no_material_defect`. A material-defect row must bind the public challenge,
retirement event, and snapshot-recomputation artifacts and use
`resolution_status: retired_and_snapshots_recomputed`.

The aggregate counts must reproduce exactly from the rows: at least 60 randomly selected tasks,
every anomaly-flagged task, at least two auditors per task, and zero unresolved material defects.
The random-sampling seed must have been committed before model results were available.

## Generation reliability panel

Schema: `flavourbench-season1-generation-reliability-panel-v1`.

`task_families` binds 20 tasks, five from each family. `cell_records` contains the complete
20-task by 16-endpoint by two-condition Cartesian product (640 unique cells). Every cell carries
three distinct response-arm IDs and a separate list of provider retry-attempt IDs; retries cannot
serve as repetitions. Exactly 1,920 panel arms must be present, of which 1,280 are incremental to
the primary schedule. A content hash binds the derived reliability metrics.

The panel reports valid-response pass-at-1, all-three-valid reliability, within-endpoint score
variance, pairwise-outcome consistency, and tool-trajectory consistency. These estimates remain
separate from the primary model and uplift point estimates.

## Prompt-sensitivity audit

Schema: `flavourbench-season1-prompt-sensitivity-audit-v1`.

`task_families` binds 20 development tasks, five per family. `arm_records` contains the complete
20-task by eight-endpoint by three-prompt Cartesian product. Each of the 480 rows binds one unique
arm, exact endpoint, exact prompt-variant hash, and `rank_eligible: false`. The endpoint subset and
prompt variants are frozen without quality outcomes; selection after results is prohibited. A
content hash binds score-shift, rank-correlation, endpoint-by-prompt interaction, and invalid-
response-shift results.

## Practical cookability execution

Schema: `flavourbench-season1-practical-cookability-execution-v1`.

`execution_records` contains 48 rows: one preregistered output for each of 24 cookability tasks,
executed by two distinct blinded cook commitments. Each row records a unique execution ID, task
and output IDs, blinding, completion, positive elapsed time, instruction deviations, yield
recording, and a blinded 1--5 acceptability score. The artifact binds the output-selection record
and the analysis relating rubric cookability to execution outcomes. It is construct-validity
evidence and cannot be pooled into the preference ranking.

## Fail-closed rules

- Missing, duplicate, malformed, under-sized, rehashed-but-underfilled, synthetic, post-selected,
  or wrong-design artifacts block release.
- Every row count and Cartesian-product claim is recomputed from row-level records.
- Rater and cook commitments are season-specific digests; raw identity is not released.
- Corrected artifacts supersede rather than overwrite prior files, and the signed archive binds
  the accepted versions.
