---
title: FlavourBench
emoji: 🥘
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: other
short_description: The executable culinary benchmark for frontier LLMs.
---

# FlavourBench: Executable Culinary Evaluation of Frontier Language Models

The executable culinary benchmark and evidence explorer for 20 frontier language-model endpoints,
scored against answer keys computed before evaluation.

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) ·
[Dataset](https://huggingface.co/datasets/josefchen/flavourbench) ·
[Source and reproduction](https://github.com/josefchen/flavourbench)

The Space presents the complete 20-model, 32-task public release. Its primary table contains only
FlavourBench Score, correct count, Wilson 95% interval, and parsed-answer count. Equal scores share
a score rank. Epicure-assisted results remain available in the model and pair views as a secondary
execution diagnostic. The Space does not call models or Epicure at runtime; the checked-in release
JSON is the sole data source.

## Local launch

```bash
pip install -r requirements.txt
python app.py
```

For a Hugging Face deployment, upload the contents of this directory as the Space repository. The
release file under `data/` keeps the first launch self-contained. A later revision can load the
same content-addressed table configs from the companion dataset repository.

See the repository [rights boundary](https://github.com/josefchen/flavourbench/blob/main/LICENSES.md)
before publishing or mirroring records.
