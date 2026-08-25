# arXiv v2 metadata

The live arXiv record is still version 1. Use this metadata with
`build/flavourbench-arxiv-source.tar.gz` when replacing it. Confirm that the arXiv preview and
metadata show exactly two authors before finalizing the replacement.

## Title

FlavourBench: Ranking Frontier Language Models in an Executable Culinary Environment

## Authors

1. Josef Chen, Independent Researcher
2. Erim Hayretci, Imperial College London

## Abstract

Open-ended language-model benchmarks usually inherit a judge: a human preference panel, another
model, or a brittle exact-match key. We introduce FlavourBench, an automated benchmark built from
a versioned culinary environment. Each task presents eight ingredients and asks for a
three-ingredient portfolio. Before any model is called, Epicure assigns a score to all 56 possible
portfolios. We evaluate 27 frontier endpoints on the same 534 tasks spanning substitution, pairing,
and constrained composition, for 14,418 complete model-task cells. Anchor-cluster bootstraps give
simultaneous 95% score bands; paired cluster sign flips test all 351 model contrasts with Holm
control. The two independently compiled panels correlate at r = 0.89 (rank rho = 0.80), and
crossed-design relative-decision generalizability is 0.936. The largest leave-one-endpoint change
to the shared core yields rank rho = 0.955. Post-hoc rescoring of all model decisions with three
public, paper-linked Epicure checkpoints gives model-rank correlations of 0.903 to 0.957 and
retains the same point leader in each case. Against 1,469 unique held-out Recipe1MSubs pairs, those
checkpoints rank the human-observed substitute at an equal-source, within-food-group percentile of
0.780 to 0.806; all source-clustered intervals exclude chance after Holm correction. Grok 4.6 has
the largest primary point estimate at 65.1 (simultaneous 95% CI 61.0 to 69.2), but the corrected
comparisons do not establish a unique best endpoint. The release includes every prompt, portfolio
score map, raw response, route, content hash, and an offline verifier that reconstructs the
results.

## Comments

14 pages, 7 figures. Evaluation of 27 frontier language-model endpoints on 534 identical tasks per
model, comprising 14,418 scored model-task cells. Adds reward-map sensitivity, selection and metric
robustness, and held-out Recipe1MSubs substitution validation. Code, dataset, and interactive
leaderboard links remain unchanged.

## Subjects

Keep the current subjects unless arXiv moderation advises otherwise: Artificial Intelligence
(cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), and Software Engineering (cs.SE).

## Upload checks

1. Run `make -C paper -f Makefile.powered arxiv`.
2. Run `(cd paper/build && sha256sum --check ARTIFACTS.sha256)`.
3. Upload `paper/build/flavourbench-arxiv-source.tar.gz` as a replacement, not as a new paper.
4. Replace the title, abstract, and comments with the text above.
5. Confirm the preview is 14 pages and lists Josef Chen and Erim Hayretci in that order.
6. After announcement, confirm the live record exposes version 2 and that its PDF title, abstract,
   author list, code link, dataset link, and leaderboard link agree with this repository.
