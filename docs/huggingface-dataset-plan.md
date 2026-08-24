# Hugging Face dataset publication contract

## Repository

Dataset: `josefchen/flavourbench`

The dataset is the public, content-addressed record behind the Space. It contains the complete
common-core evaluation release and a separate development track for training. It contains no API
keys, private model weights, participant data, or unrestricted Epicure source data.

## Evaluation release

| Config | Rows | Unit |
|---|---:|---|
| `models` | 27 | Model identity, score, uncertainty, route, and family profile |
| `tasks` | 534 | Prompt, choices, anchor, family, and all 56 released scores |
| `primary_observations` | 14,418 | One complete response for every model-task cell |
| `leaderboard` | 27 | Point rank, statistical group, score band, and rank interval |
| `pairwise_comparisons` | 351 | Paired effect, sign-flip result, Holm result, and effect size |

The release is a complete 27 by 534 matrix. Each model receives the same 178 substitution, 178
pairing, and 178 constraint tasks. The 534 ingredient anchors are the resampling units.

## Lab track

| Config | Split or rows | Purpose |
|---|---:|---|
| `lab_tasks` | 342 train, 84 validation | Anchor-disjoint development reward maps |
| `sft` | 342 train, 84 validation | Optimal demonstrations |
| `dpo` | 1,368 train, 336 validation | Deterministic chosen and rejected pairs |
| `grpo` | 342 train, 84 validation | Dense 56-choice rewards |
| `supplemental_cultural_composition` | 283 development | Non-ranked regional composition evidence |

No lab-track anchor appears in the official 534-task release. The package includes runnable LoRA
recipes for SFT, DPO, and GRPO. GRPO evaluates reward locally and does not require one Space call
per rollout.

## Integrity

`data-complete-core/DATA_MANIFEST.json` binds the evaluation tables by byte size and SHA-256.
`data-lab/DATA_MANIFEST.json` independently binds the development views. The Space bundle is built
from those verified directories and records both manifest identities.

```bash
python hf/dataset/build_lab_dataset.py --check
python3 -I hf/dataset/verify_complete_core_dataset.py \
  --dataset-directory hf/dataset/data-complete-core
python hf/space/build_complete_core_space_bundle.py --check
```

These checks make no provider calls.

## Versioning

- Never replace a response or reinterpret a score inside a published release.
- Publish new model results, task changes, or scoring changes as a new versioned release.
- Keep raw responses and route metadata next to every derived score.
- Pin any external reproduction by commit, not by a moving branch.
- Keep official test maps separate from reward maps used for optimization.

## Visibility gate

The dataset and Space remain private until the arXiv submission is approved. The public switch is
one release operation: verify both Hub revisions, make the dataset public, make the Space public,
replace paper links with the arXiv URL, and recheck the Data Viewer and named Space endpoints.
