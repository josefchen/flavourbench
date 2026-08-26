# ICLR 2027 anonymous submission package

This directory builds the double-blind manuscript **FlavourBench: Executable Culinary Reward Maps
for Language Model Evaluation and Post-Training** with the unmodified official ICLR 2027 style
files.

```bash
make -C paper/iclr2027 verify
make -C paper/iclr2027 package
```

The build stages only the generated tables and figures referenced by `main.tex`. Verification checks
the nine-page main-text limit, anonymity, PDF metadata, embedded fonts, unresolved references,
credential patterns, official template hashes, the semantic integrity of the 27-model release, and
an independent reconstruction of both reward-transfer evaluations. If the repository virtual
environment is absent, install NumPy before running these targets or set `PYTHON` to an environment
that provides it.

The package target produces:

- `build/flavourbench-iclr2027-anonymous.pdf`
- `build/flavourbench-iclr2027-anonymous-source.tar.gz`
- `build/flavourbench-iclr2027-anonymous-supplement.zip`
- `build/ICLR2027-MANIFEST.sha256`

The public manuscript and public repository are deliberately not referenced in the review source.
Related prior work is cited in the third person, as required for double-blind review.

Official policy sources, checked on 26 August 2026:

- <https://iclr.cc/Conferences/2027/AuthorGuidelines>
- <https://iclr.cc/Conferences/2027/AIPolicyForAuthors>
- <https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>

The AI-use statement in the manuscript must be reviewed and approved by every author before upload.
No author may be added after the ICLR abstract deadline.

Before creating the OpenReview submission, use:

- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) for deadlines, author-only gates, upload
  checks, and venue fallback strategy; and
- [`OPENREVIEW_METADATA.md`](OPENREVIEW_METADATA.md) for the exact title, expanded abstract, author
  order, and suggested keywords.
