# FlavourBench paper

**FlavourBench: Ranking Frontier Language Models in an Executable Culinary Environment**

Josef Chen, Independent Researcher<br>
Erim Hayretci, Imperial College London

Canonical paper record: <https://arxiv.org/abs/2608.20574>

The live record remains version 1 until the revised source is submitted. Use
[`ARXIV_V2_METADATA.md`](ARXIV_V2_METADATA.md) with the source tarball below so its title, abstract,
comments, author list, and PDF stay synchronized.

The paper reports the final 27-model complete-common-core benchmark. Each model is scored on the
same 534 tasks, giving 14,418 valid model-task observations. The task set contains two independently
compiled panels and three equally weighted families. Epicure scores all 56 legal portfolios before
model execution; no human or language model judges the responses.

The primary analysis uses 50,000 ingredient-anchor cluster bootstraps, simultaneous 95% score
intervals, 100,000 cluster sign flips for all 351 model pairs, Holm correction, exact-chance tests,
and bootstrap rank intervals. The paper reports point ranks without claiming a statistically unique
winner when the leading intervals overlap.

A crossed-design precision supplement reports relative-decision generalizability of 0.936 and
5,000 balanced, score-blind subsamples at each of five smaller task counts. Its content-addressed
analysis, CSV tables, and vector figure are rebuilt with the paper assets.

A post-collection scorer-sensitivity analysis keeps all 534 tasks and 14,418 model choices fixed
while replacing the primary reward maps with the immutable public Epicure-Cooc, Epicure-Core, and
Epicure-Chem checkpoints. Model-rank correlations are 0.903 to 0.957, pair-order agreement is
86.9% to 91.7%, and all three point estimates retain the same leader. This checks reward-map
dependence conditional on the released candidate sets; it is not external culinary validation.

A separate, protocol-bound validation uses Recipe1MSubs labels extracted from recipe-user
comments. Exact matching yields 1,469 unique held-out directed pairs over 357 source ingredients.
The three public checkpoints place the observed target at equal-source, within-food-group rank
percentiles of 0.780 to 0.806; all source-clustered intervals exclude 0.5 after Holm correction.
The 594 pairs absent from Recipe1MSubs training remain at 0.718 to 0.754. This validates the public
checkpoints' substitution geometry, not the unrecovered primary runtime or cooked-food quality;
Recipe1M is also upstream of part of Epicure's recipe corpus.

## Build and verify

Run from the repository root after installing the development dependencies:

```bash
make -C paper -f Makefile.powered verify
make -C paper -f Makefile.powered arxiv
cd paper/build
sha256sum --check ARTIFACTS.sha256
```

`make ... verify` checks the compact statistical release, regenerates the tables and figures, runs
the focused statistical tests, and checks the Python sources. `make ... arxiv` performs a clean
LaTeX build, checks references and fonts, creates a deterministic source archive, extracts it into a
fresh directory, verifies its manifest, rebuilds the paper, and compares that PDF byte-for-byte with
the canonical PDF.

The first verification downloads three hash-pinned public Epicure checkpoints and one
commit-pinned ingredient metadata file if they are absent from the local cache. It also retrieves
the hash-pinned Recipe1MSubs train and test files from Meta's public host for aggregate validation;
the raw files are cached locally and never packaged. For an offline build, provide `EPICURE_TAGS`,
`PUBLIC_SCORER_CHECKPOINT_ROOT`, and `RECIPE1MSUBS_DIRECTORY`, then set
`PUBLIC_SCORER_NETWORK=` and `EXTERNAL_VALIDATION_NETWORK=`.

## Canonical outputs

- `build/flavourbench.pdf`
- `build/flavourbench-arxiv-source.tar.gz`
- `build/ARTIFACTS.sha256`
- `generated/complete-core/flavourbench-complete-core-release-0a20655c97aa1363c2266e247f3dd03b759d0f80bca9154c6619c5549b2fac99.json`
- `generated/complete-core/flavourbench-complete-core-leaderboard-fd751b51bfb4819adccf780b0c1ea69eaace5c1f1ea0c65ef5c7f0ff16387ce8.csv`
- `generated/complete-core/flavourbench-complete-core-pairwise-510c784aa04b2a1491481f611142d58adbaa4139b17769acfc011bbd102a2da4.csv`

The figures are in `figures/complete-core/`; generated TeX tables and macros are in
`generated/complete-core/`.

## Compact release verification

The release verifier does not read provider credentials or call a model endpoint:

```bash
python3 -I paper/verify_complete_core_release.py \
  --release paper/generated/complete-core/flavourbench-complete-core-release-0a20655c97aa1363c2266e247f3dd03b759d0f80bca9154c6619c5549b2fac99.json
```

The larger response layer is published separately on the Hugging Face dataset. The repository keeps
the analysis code, plans, compact release, tables, figures, PDF, and arXiv source package.

## Scope

FlavourBench ranks constrained culinary selections against the released Epicure reward surface. It
does not measure general intelligence, food safety, sensory preference, or complete recipe writing.
The primary runtime's exact training run and seed were not recovered. Endpoint results bind the
routes and collection period recorded in the release.
