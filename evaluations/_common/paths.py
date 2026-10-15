"""Where everything lives. Scripts take these as defaults; never hardcode.

    evaluations/pool/      policy checkpoints and their training logs: the
                           inputs nothing else can recompute (external release)
    evaluations/results/   intermediates computed from the pool: selection,
                           OOD payloads, eval CSVs and logs
    evaluations/reports/   final tables, figures and HTML reports
    archive/               (repo root, local only) history: pre-guard runs,
                           old sweeps, superseded results, retired code;
                           never needed to reproduce the paper
"""

from __future__ import annotations

from pathlib import Path

EVALUATIONS = Path(__file__).resolve().parents[1]
REPO = EVALUATIONS.parent

POOL = EVALUATIONS / "pool"
RESULTS = EVALUATIONS / "results"
REPORTS = EVALUATIONS / "reports"
ARCHIVE = REPO / "archive"

# The cross-scenario paper sweep.
CS_POOL = POOL / "cross_scenario"
CS_CANDIDATES = CS_POOL / "candidates"   # <scenario>/seed<i>/{leader,follower}.zip
CS_FINETUNE = CS_POOL / "finetune"       # <cell>/leader_finetune.zip
CS_RESULTS = RESULTS / "cross_scenario"
CS_SELECTION = CS_RESULTS / "selection.json"
CS_OOD = CS_RESULTS / "ood"                          # <cell>/follower_ood.pt
CS_EVAL = CS_RESULTS / "eval"                        # <cell>/compare_<scenario>.csv
CS_EVAL_FINETUNE = CS_RESULTS / "eval_finetune"      # <cell>/compare_<scenario>.csv
CS_EVAL_LEADER_ONLY = CS_RESULTS / "eval_leader_only"  # <cell>/compare_<scenario>.csv
CS_ARCHIVE = ARCHIVE / "cross_scenario"

# The alpha sweep: PAVE at several alphas on a subset of the paper cells.
AS_RESULTS = RESULTS / "alpha_sweep"            # alpha_<a>/<cell>/compare_<scenario>.csv

TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
APPENDIX = REPORTS / "appendix"
