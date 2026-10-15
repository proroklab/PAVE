# PAVE: Policy Adaptation via Value Exchange

Code for **Leave It Better Than Found: Value-Guided Cooperation in Sequential Tasks**
(CoRL 2026). Mario Gomez Andreu, Michael Amir, Antonio Marino, Tianxu An,
Marco Hutter, Amanda Prorok.

Project page: https://proroklab.github.io/PAVE/

A leader robot plans online (CEM) to maximize its own return plus the follower's
shared value function at the handoff state, down-weighted by a kNN out-of-distribution
score. No joint training is needed.

## Install

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13, pinned in `uv.lock`):

    uv sync

## Paper experiments

`bash reproduce.sh --from ood` runs the paper's evaluations from the included checkpoints.

## Layout

    src/critic_sharing/   environments, PPO training, CEM planner, CLI (`critic-sharing`)
    configs/              all hyperparameters, per scenario
    evaluations/paper/    scripts behind the paper's tables and figures
    evaluations/pool/     trained leader/follower checkpoints
    reproduce.sh          the full evaluation pipeline
    website/              project page

## Citation

```bibtex
@inproceedings{gomezandreu2026pave,
  title     = {Leave It Better Than Found: Value-Guided Cooperation in Sequential Tasks},
  author    = {Gomez Andreu, Mario and Amir, Michael and Marino, Antonio and An, Tianxu
               and Hutter, Marco and Prorok, Amanda},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

MIT
