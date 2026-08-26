# Anonymous FlavourBench supplementary artifact

This archive supports the double-blind ICLR 2027 submission. It contains the complete selected
27-by-534 response matrix, frozen tasks and reward maps, endpoint and route records, statistical
outputs, public-checkpoint sensitivity data, aggregate external validation, and disjoint lab-track
training views. It also contains the complete controlled reward-transfer evidence: the prospective
protocol, treatment and format-control data, six training manifests, raw held-out generations,
sealed analyses, and reconstruction code. Provider credentials, private communications, author
identities, and unselected historical runs are not included.

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

The reward-transfer verifier needs NumPy but does not need model weights, a GPU, or network access:

```bash
python -m pip install numpy==2.3.3
PYTHONPATH=src python experiments/reward_transfer/verify_release.py
```

It verifies the plan, gate, six training manifests, two evaluation masters, fourteen run manifests,
and every released file hash. It reparses and rescores all 4,326 held-out generations, reconstructs
the seven runs on each split, and reruns the 50,000-draw crossed bootstrap and 100,000-draw
sign-flip test for both reported treatment effects.

To reproduce the six adapters, install `requirements-reward-transfer.txt` and run
`experiments/reward_transfer/train_sft.py` for the two conditions and three frozen seeds. The
protocol fixes the base revision, data, optimizer, LoRA configuration, checkpoint rule, and decode
contract. Adapter weights are not duplicated in this archive; their exact file hashes and run
manifests are included.

## Contents

- `data/complete-core/`: tasks, all 14,418 selected raw response wrappers, routes, leaderboard,
  pairwise comparisons, release object, and analysis plan.
- `data/analysis/`: task-count stability, variance decomposition, score-definition and
  task-selection sensitivity, public-checkpoint score maps, and aggregate Recipe1MSubs validation.
- `data/lab/`: disjoint SFT, DPO, GRPO, validation, and predeclared evaluation views.
- `hf/dataset/`: the same lab and evaluation-task paths used by the executable transfer scripts.
- `contracts/reward-transfer/`: the protocol frozen before transfer outcomes were generated.
- `experiments/reward_transfer/`: training, gating, evaluation, analysis, release, and verification
  programs.
- `src/flavourbench/`: the parser, scorer, task validator, and reward-transfer inference module.
- `protocols/`: the protocol fixed before external substitution scores were inspected.
- `code/verify_dataset.py`: dependency-free release verifier.
- `code/rebuild_summary.py`: dependency-free score and task-diagnostic reconstruction.

The raw Recipe1MSubs pickle files are not redistributed. The aggregate analysis records their
official source URLs and immutable hashes; the paper's public-checkpoint validation builder can
retrieve and verify those files separately.

No provider calls are required to verify either study. A new frontier endpoint evaluation requires
access to that endpoint and is outside the anonymous artifact.

For double-blind review, the packaged `src/flavourbench/lab.py` changes only the public dataset
repository default to `anonymous/flavourbench`. The transfer analysis imports its fixed family
tuple; no reported result depends on the replaced default.
