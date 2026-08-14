---
title: FlavourBench
emoji: 🍲
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: other
datasets:
- josefchen/flavourbench
---

# FlavourBench

**Josef Chen** — Independent Researcher · **Jakub Radzikowski** — Independent Researcher · **Erim Hayretci** — Independent Researcher

An interactive explorer for the powered FlavourBench release: 26 frontier endpoints, 640
executable culinary decisions, 16,640 primary responses, 1,664 label-permuted repeats, and all 325
paired statistical comparisons.

The Space reads one content-addressed bundle and makes no provider calls. Use it to inspect:

- the FlavourBench Score, simultaneous 95% intervals, and statistical rank groups;
- family-level model profiles and label-permutation repeatability;
- exact prompts, candidate sets, frozen 56-portfolio score maps, and real model responses; and
- any paired model contrast with its bootstrap interval and Holm-adjusted result.

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) |
[Dataset](https://huggingface.co/datasets/josefchen/flavourbench) |
[Source](https://github.com/josefchen/flavourbench)

The dataset contains full response records. The Space carries compact excerpts for fast browsing.
See the repository [rights boundary](https://github.com/josefchen/flavourbench/blob/main/LICENSES.md)
for component-level licensing.

## Citation

```bibtex
@article{chen2026flavourbench,
  title   = {FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth},
  author  = {Chen, Josef and Radzikowski, Jakub and Hayretci, Erim},
  year    = {2026}
}
```
