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
short_description: Inspect how Epicure changes frontier model decisions.
---

# FlavourBench Explorer

An evidence-first explorer for the FlavourBench Epicure-native automated pilot.

[Paper](https://github.com/josefchen/flavourbench/blob/main/paper/build/flavourbench.pdf) ·
[Dataset](https://huggingface.co/datasets/josefchen/flavourbench) ·
[Source and reproduction](https://github.com/josefchen/flavourbench)

The Space presents the complete 20-model, 32-task public release and makes every paired result
inspectable. It does not call models or Epicure at runtime. The checked-in release JSON is the sole
data source.

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
