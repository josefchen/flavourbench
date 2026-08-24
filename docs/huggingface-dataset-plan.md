# Hugging Face dataset publication plan

## Repository

Proposed dataset repository: `josefchen/flavourbench`

The source of truth is the content-addressed release at
`paper/generated/epicure-native/epicure-native-release.json`. The exporter in
`hf/dataset/build_dataset.py` validates its semantic hash, derives the five table configs, and
writes a checksum manifest.

## Publication contract

- Preserve all assigned arms, not only successful responses.
- Keep Model only and Model + Epicure observations separate and joinable.
- Preserve route, model, prompt, result, response, and release hashes.
- Never upload provider credentials, participant data, private databases, or unrestricted Epicure
  payloads.
- Pin the Space to an immutable dataset commit.
- Publish changed benchmark semantics as a new dataset version, not an in-place reinterpretation.

## Proposed commands

After creating the Hugging Face dataset repository and authenticating locally:

```bash
python hf/dataset/build_dataset.py
npx @huggingface/hub upload josefchen/flavourbench \
  hf/dataset . \
  --repo-type dataset \
  --commit-message "Publish FlavourBench executable benchmark release"
```

After creating the Space repository:

```bash
npx @huggingface/hub upload josefchen/flavourbench-explorer \
  hf/space . \
  --repo-type space \
  --commit-message "Launch FlavourBench evidence explorer"
```

These are external publication operations and should run only after the software and dataset
license choices are explicit.

## Quality checks

```bash
python hf/dataset/build_dataset.py --check
python -m py_compile hf/dataset/build_dataset.py hf/space/app.py
python -I paper/reproduce_epicure_native.py \
  --release paper/generated/epicure-native/epicure-native-release.json
```

Before upload, scan the full commit for secrets, absolute local paths, private artifact names, and
unexpected large files. After upload, verify every row count in the Hugging Face Data Viewer and
compare its dataset commit SHA with the revision pinned by the Space.
