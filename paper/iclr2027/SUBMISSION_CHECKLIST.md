# ICLR 2027 submission control

Checked against the official ICLR pages on 26 August 2026. This file is operational guidance, not
part of the anonymous source or supplement.

## Venue decision

Submit this version to ICLR 2027. The paper's central contribution is a learning and evaluation
method: exhaustive reward maps turn an open-ended domain into paired, auditable language-model
evaluation and a prospective reinforcement-learning environment. That is a stronger ICLR framing
than presenting the work as a culinary leaderboard alone.

Do not submit the same manuscript concurrently elsewhere. If ICLR rejects it, the preferred next
route is TMLR, whose rolling review emphasizes technical correctness. NeurIPS 2027 Evaluations &
Datasets is the strongest later conference route if its future call resembles the 2026 track. The
present paper already includes the controlled reward-transfer experiment; a later version should
test additional model scales or optimization algorithms and follow the then-current code, hosting,
and machine-readable dataset-metadata rules. No official NeurIPS 2027 paper call was available when
this checklist was written.

Official sources:

- ICLR author guidelines: <https://iclr.cc/Conferences/2027/AuthorGuidelines>
- ICLR call for papers: <https://www.iclr.cc/Conferences/2027/CallForPapers>
- ICLR AI policy: <https://iclr.cc/Conferences/2027/AIPolicyForAuthors>
- ICLR dates: <https://iclr.cc/Conferences/2027/Dates>
- TMLR: <https://jmlr.org/tmlr/>
- NeurIPS 2026 Evaluations & Datasets precedent:
  <https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets>

## Hard deadlines

- [ ] Submit a genuine abstract by **18 September 2026, 23:59 AOE**.
- [ ] Submit the paper and supplement by **25 September 2026, AOE**.
- [ ] Set an internal deadline at least 24 hours earlier. ICLR says the deadlines are final.
- [ ] Include both authors by the abstract deadline. Authors cannot be added or removed afterward;
  only their order may change before the full-paper deadline.

## Technical state

- [x] The manuscript uses the unmodified official ICLR 2027 style files, checked by SHA-256.
- [x] The review switch is disabled and the author block says `Anonymous Authors`.
- [x] Main text ends on page 9, at the strict nine-page initial limit.
- [x] The mandatory AI-use statement begins on page 10 and sits outside the main text.
- [x] References and appendices follow the disclosure statements.
- [x] The 21-page PDF is US Letter, has anonymous metadata, and embeds every font.
- [x] LaTeX has no overfull boxes, unresolved citations, or unresolved references.
- [x] The anonymous source and supplement contain no author names, affiliations, public-paper link,
  local path, or credential-shaped value.
- [x] The supplement contains all 14,418 selected response records, 534 exhaustive task maps, the
  27-model route manifest, the analysis artifacts, training splits, and offline verification code.
- [x] A clean extraction reconstructs all published scores, ranks, and reward-transfer effects
  without provider access.
- [x] A clean extraction of the source archive rebuilds a PDF bitwise identical to the upload PDF.
- [x] `make ci PYTHON=.venv/bin/python PYTEST=.venv/bin/pytest RUFF=.venv/bin/ruff` passes.

The authoritative upload artifacts are:

- `build/flavourbench-iclr2027-anonymous.pdf`
- `build/flavourbench-iclr2027-anonymous-supplement.zip`
- `build/flavourbench-iclr2027-anonymous-source.tar.gz`, retained as the reproducible source bundle
- `build/ICLR2027-MANIFEST.sha256`, the authoritative checksums

## Scientific claim audit

- [x] The paper defines the estimand as agreement with a released culinary reward map, not universal
  taste or general model quality.
- [x] Every endpoint is evaluated on the same 534 tasks. Failures and filtered cells are not pooled
  into the ranked matrix.
- [x] Point ranks are separated from inferential claims. The paper reports that 101 of 351 paired
  contrasts are resolved and does not claim a unique best model.
- [x] Inference clusters by task anchor and controls the full pairwise family with Holm correction.
- [x] Panel replication, task-count stability, roster sensitivity, score-definition sensitivity,
  family-weight sensitivity, and public-reward-map sensitivity are reported.
- [x] Recipe1MSubs is described as label-independent convergent validation of public checkpoint
  substitution geometry, not validation of the unrecovered primary runtime or cooked outcomes.
- [x] The preregistered reward-transfer estimand compares Epicure SFT with a prompt-, format-, and
  label-matched control, rather than attributing parsing gains to reward learning.
- [x] The 13.30-point primary effect and 11.73-point public-map replication reconstruct from raw
  generations, manifests, and the sealed analysis plans.
- [x] Limitations explicitly cover construct validity, missing primary-runtime lineage, endpoint
  versioning, common-core selection, one 0.6B base checkpoint, three seeds, and SFT-only transfer.

The main reviewer risk is scientific, not statistical: the primary reward function is a fixed
reference environment whose original training seed and source revision were not recovered. The
second risk is transfer breadth: the controlled study uses one 0.6B checkpoint, three seeds, and
LoRA SFT, and its public-map result is task replication rather than independent retraining. Do not
soften either limitation in the submission.

## Author actions before the abstract deadline

- [ ] Josef Chen confirms the title, abstract, author order, and his `Independent Researcher`
  affiliation in OpenReview.
- [ ] Erim Hayretci confirms authorship, author order, and his current Imperial College London
  affiliation in OpenReview.
- [ ] Both OpenReview profiles contain current names, verified emails, affiliations, publication
  history, conflicts, and domains.
- [ ] Confirm whether either author satisfies ICLR's reciprocal-reviewer publication rule. If one
  does, ensure at least one qualified author registers to review at least three papers when invited.
  If neither does, document the stated exemption and one-submission quota.
- [ ] Both authors read and agree to the ICLR Code of Ethics and Code of Conduct.
- [ ] Both authors review every disclosure item in the AI-use statement against the ICLR task list.
  The current statement is intentionally broad; do not narrow it unless a listed use is factually
  wrong.
- [ ] Both authors approve the exact metadata in `OPENREVIEW_METADATA.md`.
- [ ] Confirm that no substantially similar manuscript is under review at another archival venue.

## Public arXiv version

The public record at <https://arxiv.org/abs/2608.20574> has the correct authors, Josef Chen and Erim
Hayretci, but remains version 1 with the older title and analysis. This does not block ICLR: its
guidelines permit related arXiv papers and nearly identical preprints, provided the anonymous
submission does not explicitly refer to the public record. Submit the prepared v2 source and
metadata from `../ARXIV_V2_METADATA.md` when ready, then verify the announced record. Do not cite or
link that record from the anonymous ICLR manuscript or supplement.

## Upload procedure

1. Run the full clean-room audit:

   ```bash
   make -C paper/iclr2027 package
   make ci PYTHON=.venv/bin/python PYTEST=.venv/bin/pytest RUFF=.venv/bin/ruff
   (cd paper/iclr2027/build && sha256sum --check ICLR2027-MANIFEST.sha256)
   ```

2. Submit the genuine title, abstract, and both authors from `OPENREVIEW_METADATA.md` before the
   abstract deadline.
3. Upload `flavourbench-iclr2027-anonymous.pdf` as the paper.
4. Upload `flavourbench-iclr2027-anonymous-supplement.zip` as supplementary material. The source
   tarball is a reproducibility receipt and need not replace the supplement.
5. Inspect the OpenReview-rendered first page. It must say `Anonymous Authors`, show the full title,
   and contain no clipped text.
6. Download the uploaded PDF and supplement from OpenReview and compare their hashes with
   `ICLR2027-MANIFEST.sha256`.
7. Check that OpenReview lists exactly Josef Chen and Erim Hayretci in the intended order and links
   each name to the correct profile.
8. Save the submission number, confirmation email, and downloaded upload receipts outside the
   public repository.

## Final stop conditions

Do not upload if any of these is true:

- an author has not approved the AI-use statement;
- an author profile or author list is wrong;
- `make ci` or the manifest check fails;
- the PDF author metadata is not `Anonymous Authors`;
- the OpenReview abstract materially differs from the manuscript abstract;
- a substantially similar manuscript is under concurrent archival review; or
- the upload cannot be downloaded and verified before the deadline.
