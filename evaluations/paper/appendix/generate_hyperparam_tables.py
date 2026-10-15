"""Generate LaTeX hyperparameter tables for the paper appendix.

Extracts the training, CEM-planner, and joint-utility-cost hyperparameters
straight from the resolved config files (via the same resolver the
training/eval code uses) and writes booktabs LaTeX tables. Run::

    python -m evaluations.paper.appendix.generate_hyperparam_tables

Outputs (written to ``evaluations/reports/appendix/``)::

    training_budget.tex   total timesteps + parallel envs per scenario/role
    ppo_defaults.tex      the PPO algorithm hyperparameters actually used
    cem_planner.tex       CEM planner settings per scenario
    cost_utility.tex      joint-utility-cost settings per scenario

IMPORTANT — PPO algorithm hyperparameters. The trainers construct
``PPO("MlpPolicy", env, seed=seed)`` and override **no** algorithm
hyperparameters, so learning rate / n_steps / batch_size / etc. are the
Stable-Baselines3 defaults — NOT the (vestigial) ``learning_rate`` /
``n_steps`` / ``batch_size`` keys in ``common/training.yaml``, which are
never passed to PPO. Only ``total_timesteps``, ``n_envs`` and ``seed`` are
config-driven and actually used. ``ppo_defaults.tex`` therefore reports the
real SB3 defaults (introspected from the installed version), and
``training_budget.tex`` reports only the config-driven budget.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from evaluations._common.latex import row, write_table
from evaluations._common.paths import APPENDIX

# The six paper scenarios (cross_scenario sweep order). pickup-blocked and
# pickup-unseen carry their own planner/cost/training YAMLs.
SCENARIOS: tuple[str, ...] = (
    "mobile_arm",
    "obstacle_arena",
    "warehouse",
    "warehouse_pickup_blocked",
    "warehouse_pickup_unseen",
    "household",
)

# Roles per scenario (follower first). The resolver maps these to the right
# train_*.yaml internally (e.g. obstacle_arena follower -> train_nav.yaml).
ROLES = ("follower", "leader")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _scenario(name: str) -> str:
    """``\\scenario{...}`` cell with LaTeX-escaped underscores."""
    return f"\\scenario{{{name.replace('_', r'\_')}}}"


def _num(x: Any) -> str:
    """Compact numeric / bool / string cell."""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else f"{x:.3g}"
    return str(x)


def _steps(x: Any) -> str:
    """Format a training-step count as e.g. 1.5\\,M / 400\\,k."""
    n = int(x)
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}\\,M"
    if n >= 1_000:
        return f"{n / 1_000:g}\\,k"
    return str(n)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _training_budget_table(out_dir: Path) -> None:
    from critic_sharing.config.resolver import load_train_config

    rows: list[str] = []
    seeds: set[int] = set()
    for scenario in SCENARIOS:
        cells = [_scenario(scenario)]
        for role in ROLES:
            cfg = load_train_config(scenario, role)
            cells.append(_steps(cfg["total_timesteps"]))
            cells.append(_num(int(cfg["n_envs"])))
            seeds.add(int(cfg.get("seed", 42)))
        rows.append(row(cells))

    seed_note = (
        f"seed~$=${next(iter(seeds))}" if len(seeds) == 1
        else "seeds vary; see config"
    )
    write_table(
        out_dir / "training_budget.tex",
        column_spec="l" + "cc" * len(ROLES),
        header=row([
            "Scenario",
            r"Foll.\ steps", r"Foll.\ $n_{\text{env}}$",
            r"Lead.\ steps", r"Lead.\ $n_{\text{env}}$",
        ]),
        rows=rows,
        caption=(
            "Per-agent training budget (config-driven). Follower / leader "
            "total environment steps and number of parallel rollout "
            f"environments; all runs use {seed_note}. Algorithm "
            "hyperparameters are the Stable-Baselines3 PPO defaults "
            "(Table~\\ref{tab:ppo_defaults})."
        ),
        label="tab:training_budget",
    )
    print(f"  wrote {out_dir / 'training_budget.tex'}")


def _ppo_defaults_table(out_dir: Path) -> None:
    """The PPO algorithm hyperparameters actually used = SB3 defaults,
    introspected from the installed Stable-Baselines3 so the table matches
    the version the experiments ran on."""
    from stable_baselines3 import PPO

    sig = inspect.signature(PPO.__init__)
    spec = [
        ("learning_rate", r"Learning rate"),
        ("n_steps", r"Rollout length $n_{\text{steps}}$ (per env)"),
        ("batch_size", r"Minibatch size"),
        ("n_epochs", r"Epochs per update"),
        ("gamma", r"Discount $\gamma$"),
        ("gae_lambda", r"GAE $\lambda$"),
        ("clip_range", r"PPO clip range $\epsilon$"),
        ("ent_coef", r"Entropy coef.\ $c_{\text{ent}}$"),
        ("vf_coef", r"Value-function coef.\ $c_{vf}$"),
        ("max_grad_norm", r"Max gradient norm"),
    ]
    rows: list[str] = []
    for key, label in spec:
        default = sig.parameters[key].default
        rows.append(row([label, _num(default)]))

    write_table(
        out_dir / "ppo_defaults.tex",
        column_spec="lc",
        header=row(["Hyperparameter", "Value"]),
        rows=rows,
        caption=(
            "PPO (Stable-Baselines3, \\texttt{MlpPolicy}) algorithm "
            "hyperparameters. The trainers override none of these; only the "
            "per-run budget (Table~\\ref{tab:training_budget}) is set. Network "
            "is the SB3 default two-layer (64, 64) tanh MLP. Values "
            "introspected from the installed SB3 version."
        ),
        label="tab:ppo_defaults",
        use_resizebox=False,
    )
    print(f"  wrote {out_dir / 'ppo_defaults.tex'}")


def _cem_planner_table(out_dir: Path) -> None:
    from critic_sharing.config.resolver import load_planner_config

    rows: list[str] = []
    for scenario in SCENARIOS:
        p = load_planner_config(scenario)
        rows.append(row([
            _scenario(scenario),
            _num(int(p.horizon)),
            _num(int(p.num_samples)),
            _num(int(p.num_iters)),
            _num(float(p.elite_frac)),
            _num(float(p.noise_sigma)),
            _num(float(p.min_std)),
            _num(int(p.replan_interval)),
            str(p.distribution),
        ]))

    write_table(
        out_dir / "cem_planner.tex",
        column_spec="l" + "c" * 8,
        header=row([
            "Scenario", "Horizon", "Samples", "Iters", "Elite frac.",
            r"$\sigma_{\text{noise}}$", r"Min.\ std", "Replan", "Dist.",
        ]),
        rows=rows,
        caption=(
            "Cross-entropy-method (CEM) planner hyperparameters per scenario. "
            "Horizon = planning steps; Samples / Iters / Elite frac.\\ = CEM "
            "population, refinement iterations, and elite fraction; "
            "$\\sigma_{\\text{noise}}$ / Min.\\ std = initial / floor action "
            "std; Replan = re-plan interval (steps); Dist.\\ = sampling "
            "distribution."
        ),
        label="tab:cem_planner",
    )
    print(f"  wrote {out_dir / 'cem_planner.tex'}")


def _cost_utility_table(out_dir: Path) -> None:
    from critic_sharing.config.resolver import load_cost_config

    rows: list[str] = []
    for scenario in SCENARIOS:
        c = load_cost_config(scenario)
        rows.append(row([
            _scenario(scenario),
            _num(float(c.alpha)),
            _num(float(c.leader_gamma)),
            _num(bool(c.normalize_returns)),
            _num(bool(c.clip_normalized)),
            _num(int(c.calibration_episodes)),
            _num(bool(c.calibrate_on_success_only)),
        ]))

    write_table(
        out_dir / "cost_utility.tex",
        column_spec="l" + "c" * 6,
        header=row([
            "Scenario", r"$\alpha$", r"$\gamma_L$",
            "Normalize", "Clip", r"Calib.\ eps.", r"Succ.-only",
        ]),
        rows=rows,
        caption=(
            "Joint-utility-cost hyperparameters per scenario. "
            "$\\alpha$ = V$_F$ weight in "
            "$r_L^{\\text{norm}} + \\alpha\\,\\mathrm{trust}\\,r_F^{\\text{norm}}$ "
            "(unified to 4 across scenarios; per-scenario sensitivity is in the "
            "$\\alpha$ sweep); $\\gamma_L$ = leader discount; Normalize / Clip = "
            "whether returns are range-normalized / clipped to $[0,1]$; "
            "Calib.\\ eps.\\ = naive episodes used to estimate the normalization "
            "range; Succ.-only = calibrate on successful naive episodes only."
        ),
        label="tab:cost_utility",
    )
    print(f"  wrote {out_dir / 'cost_utility.tex'}")


def main() -> int:
    out_dir = APPENDIX
    print(f"Writing appendix hyperparameter tables to: {out_dir}")
    _training_budget_table(out_dir)
    _ppo_defaults_table(out_dir)
    _cem_planner_table(out_dir)
    _cost_utility_table(out_dir)
    print("Done. Remember: \\scenario{} and booktabs (\\toprule etc.) must be "
          "defined/loaded in the paper preamble.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
