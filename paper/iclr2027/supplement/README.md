# Anonymous FlavourBench supplementary artifact

This archive supports the double-blind ICLR 2027 submission. It contains the complete selected
27-by-534 response matrix, frozen tasks and reward maps, endpoint and route records, statistical
outputs, public-checkpoint sensitivity data, aggregate external validation, and disjoint lab-track
training views. Provider credentials, private communications, author identities, and unselected
historical runs are not included.

## Verify the release

Python 3.11 or newer is sufficient for the integrity checks and summary reconstruction:

```bash
python code/verify_dataset.py --dataset-directory data/complete-core
python code/rebuild_summary.py \
  --dataset-directory data/complete-core \
  --output-directory reconstructed
```

The first command checks the semantic hash of the dataset manifest, the physical hash and row
count of every file, all response-level semantic hashes, the complete 27-by-534 model--task grid,
and the 351-row pairwise family. The second independently recomputes the primary and per-family
means from the selected response records, checks them against the released leaderboard, and emits
compact CSV tables.

## Contents

- `data/complete-core/`: tasks, all 14,418 selected raw response wrappers, routes, leaderboard,
  pairwise comparisons, release object, and analysis plan.
- `data/analysis/`: task-count stability, variance decomposition, score-definition and
  task-selection sensitivity, public-checkpoint score maps, and aggregate Recipe1MSubs validation.
- `data/lab/`: disjoint SFT, DPO, GRPO, validation, and predeclared evaluation views.
- `protocols/`: the protocol fixed before external substitution scores were inspected.
- `code/verify_dataset.py`: dependency-free release verifier.
- `code/rebuild_summary.py`: dependency-free score and task-diagnostic reconstruction.

The raw Recipe1MSubs pickle files are not redistributed. The aggregate analysis records their
official source URLs and immutable hashes; the paper's public-checkpoint validation builder can
retrieve and verify those files separately.

No provider calls are required to verify or rescore the selected release. A new endpoint evaluation
does require access to that endpoint and is outside the anonymous artifact.
