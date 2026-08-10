# Anonymous NeurIPS Evaluations & Datasets package

This directory contains the anonymous venue version of **FlavourBench: Epicure as an Executable
Oracle for Ranking Culinary Agents**. The canonical paper lives one directory above; run
`sync_from_arxiv.py` to regenerate the anonymous body.

The submission presents one automated 20-model leaderboard, a 32-task executable-oracle task set,
640 matched Epicure-off/on pairs, and a deterministic training-reward interface. It contains no
human-rating study. Both Cohere Command A routes are direct, complete 64-arm members of the public
panel and are enforced by the offline verifier.

Build and verify with:

```bash
make sync
make verify
```

Build the anonymous supplementary package with:

```bash
make package
```

The paper includes the benchmark architecture, complete ranking, route table, score and uplift
plots, family heatmaps, all 640 paired outcomes, latency analysis, and real prompt/tool-call cases.
The release also generates a 16:9 results card for research sharing. The supplement carries the
full public release and dependency-free replay script.

The generated venue PDF uses nine main-content pages. Full prompt transcripts and the densest
diagnostic figures remain in the optional technical appendix, followed by references and the
required checklist.
