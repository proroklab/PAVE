"""``critic-sharing train`` subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import typer

from critic_sharing.cli._common import Role, Scenario, resolve_model

app = typer.Typer(no_args_is_help=True)


@app.command("leader")
def train_leader(
    scenario: Scenario = typer.Option(..., help="Which scenario to train on."),
    config: Optional[Path] = typer.Option(None, help="Path to YAML config override."),
    total_timesteps: Optional[int] = typer.Option(
        None, help="Override total training timesteps."
    ),
    n_envs: Optional[int] = typer.Option(
        None, help="Override number of parallel envs."
    ),
    save_path: Optional[Path] = typer.Option(
        None, help="Where to save the trained model."
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed."),
    wandb_project: Optional[str] = typer.Option(None, help="W&B project name."),
    wandb_name: Optional[str] = typer.Option(None, help="W&B run name."),
) -> None:
    """Train the leader agent for a given scenario."""
    _run_training(
        role=Role.leader,
        scenario=scenario,
        config=config,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        save_path=save_path,
        seed=seed,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


@app.command("follower")
def train_follower(
    scenario: Scenario = typer.Option(..., help="Which scenario to train on."),
    config: Optional[Path] = typer.Option(None, help="Path to YAML config override."),
    total_timesteps: Optional[int] = typer.Option(
        None, help="Override total training timesteps."
    ),
    n_envs: Optional[int] = typer.Option(
        None, help="Override number of parallel envs."
    ),
    save_path: Optional[Path] = typer.Option(
        None, help="Where to save the trained model."
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed."),
    wandb_project: Optional[str] = typer.Option(None, help="W&B project name."),
    wandb_name: Optional[str] = typer.Option(None, help="W&B run name."),
) -> None:
    """Train the follower agent for a given scenario."""
    _run_training(
        role=Role.follower,
        scenario=scenario,
        config=config,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        save_path=save_path,
        seed=seed,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


def _run_training(
    role: Role,
    scenario: Scenario,
    config: Optional[Path],
    total_timesteps: Optional[int],
    n_envs: Optional[int],
    save_path: Optional[Path],
    seed: Optional[int],
    wandb_project: Optional[str],
    wandb_name: Optional[str],
) -> None:
    """Dispatch training to the appropriate scenario module.

    Resolution order (later wins):
      1. common/training.yaml defaults
      2. scenario-specific training YAML
      3. ``--config`` YAML file
      4. Explicit CLI flags
    """
    from critic_sharing.config.resolver import load_env_config, load_train_config

    typer.echo(
        "Training {role} for scenario: {scenario}".format(
            role=role.value, scenario=scenario.value
        )
    )

    # Build CLI overrides dict (only non-None values)
    cli_overrides: dict = {}
    if total_timesteps is not None:
        cli_overrides["total_timesteps"] = total_timesteps
    if n_envs is not None:
        cli_overrides["n_envs"] = n_envs
    if save_path is not None:
        cli_overrides["save_path"] = str(save_path)
    if seed is not None:
        cli_overrides["seed"] = seed
    if wandb_project is not None:
        cli_overrides["wandb_project"] = wandb_project
    if wandb_name is not None:
        cli_overrides["wandb_name"] = wandb_name

    # Resolve: common YAML → scenario YAML → --config → CLI flags
    resolved = load_train_config(
        scenario=scenario.value,
        role=role.value,
        config_override_path=config,
        cli_overrides=cli_overrides,
    )

    # Load env config from scenario YAML (single source of truth)
    env_config = load_env_config(scenario=scenario.value, role=role.value)

    # Apply training-config env overrides + CLI overrides.
    env_config = _apply_env_overrides(env_config, resolved.pop("env_overrides", None))
    if scenario == Scenario.obstacle_arena:
        if role == Role.follower:
            from critic_sharing.training.nav import train_agent
        else:
            from critic_sharing.training.push import train_agent
    elif scenario in (
        Scenario.warehouse,
        Scenario.warehouse_pickup_blocked,
        Scenario.warehouse_pickup_unseen,
    ):
        if role == Role.follower:
            from critic_sharing.training.warehouse_follower import train_agent
        else:
            from critic_sharing.training.warehouse_leader import train_agent
    elif scenario == Scenario.mobile_arm:
        if role == Role.follower:
            from critic_sharing.training.mobile_arm_follower import train_agent
        else:
            from critic_sharing.training.mobile_arm_leader import train_agent
    elif scenario == Scenario.household:
        if role == Role.follower:
            from critic_sharing.training.household_nav import train_agent
        else:
            from critic_sharing.training.household_clean import train_agent
    else:
        typer.echo("Unknown scenario: {s}".format(s=scenario.value), err=True)
        raise typer.Exit(code=1)

    # Filter resolved dict to only keys the training function accepts,
    # and flatten nested sections (e.g. vae.latent_dim → vae_latent_dim for nav).
    kwargs = _prepare_train_kwargs(resolved, train_agent)
    kwargs["env_config"] = env_config
    # Route artifact path by (scenario, role) — distinguishes
    # warehouse_pickup_blocked_follower from warehouse_follower etc. Fall back
    # to the training script's module-level default when the param
    # isn't accepted (older training scripts that don't take ``task``).
    import inspect
    sig = inspect.signature(train_agent)
    if "task" in sig.parameters:
        from ._common import TASK as _TASK_MAP
        kwargs["task"] = _TASK_MAP.get(
            (scenario.value, role.value), None,
        )
    train_agent(**kwargs)


@app.command("ood")
def train_ood_cli(
    scenario: Scenario = typer.Option(..., help="Scenario whose follower's V_F will be modeled."),
    follower_model: Optional[Path] = typer.Option(
        None,
        help=(
            "Follower model checkpoint (.zip). Used only for the default "
            "output path (``<follower_dir>/follower_ood.pt``); the policy "
            "itself isn't consulted because the detector trains on "
            "env-reset obs only. Defaults to the latest artifact."
        ),
    ),
    detector: str = typer.Option(
        "knn",
        help=(
            "OOD detector to fit. ``knn`` (default) builds only the "
            "kNN-density tree from env-reset features — fast, no neural "
            "network training. ``vae`` additionally fits a VAE on the "
            "same features for the reconstruction-loss fallback path."
        ),
    ),
    n_samples: int = typer.Option(
        1000,
        help=(
            "Number of ``env.reset`` seeds whose initial obs feed the "
            "detector. Env-natural starting states only (no policy "
            "rollout) — that's the distribution the planner queries V_F at."
        ),
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Output checkpoint. Default: <follower_model_dir>/follower_ood.pt.",
    ),
    latent_dim: int = typer.Option(
        8, help="VAE latent dimension (used only when ``--detector vae``)."
    ),
    epochs: int = typer.Option(100, help="VAE training epochs (vae only)."),
    batch_size: int = typer.Option(64, help="VAE batch size (vae only)."),
    learning_rate: float = typer.Option(
        1e-3, help="VAE Adam learning rate (vae only)."
    ),
    ood_set: Optional[Path] = typer.Option(
        None,
        help=(
            "Optional .npz with key 'samples' of pre-computed OOD samples "
            "(post-obs_transform shape). When supplied the VAE threshold "
            "is calibrated via roc_threshold instead of percentiles. "
            "No effect when ``--detector knn`` (kNN uses its own quantile "
            "calibration on the train-features cKDTree)."
        ),
    ),
    ood_percentile: float = typer.Option(
        95.0,
        help="Percentile of training losses used as the VAE OOD threshold (vae only).",
    ),
    max_loss_percentile: float = typer.Option(
        99.9,
        help="Percentile of training losses used as VAE max_loss (vae only).",
    ),
    seed: int = typer.Option(0, help="Master seed for sampling + (optional) VAE training."),
) -> None:
    """Fit the OOD detector on the follower env's natural starting-state distribution.

    Two detectors are supported. Both ingest the same env-reset features
    (post-scenario-specific ``obs_transform``), so a checkpoint produced
    by ``--detector vae`` is a strict superset of one produced by
    ``--detector knn``:

    * **knn** (default) — builds a ``scipy.spatial.cKDTree`` over the
      training features and calibrates a 95th-percentile-of-intra-kNN
      threshold. Fast: no neural network is trained.
    * **vae** — additionally fits an MLP VAE on the same features and
      calibrates a reconstruction-loss threshold. Used as the fallback
      detector when the planner-side closure can't construct the kNN
      tree (e.g., legacy checkpoints).

    The detector provides the OOD signal that ``JointUtilityCost`` uses
    to trust-weight ``V_F``: at planning time we score the leader's
    CEM-terminal obs (post-transform) and produce a value in ``[0, 1]``
    where 1 = far from V_F's training distribution. The planner then
    multiplies V_F's contribution by ``(1 - score)``.

    Training data comes from ``env.reset`` outputs only — no policy
    rollout. This matches the planner's V_F query distribution: at
    deployment the leader hands off to the follower with some
    leader-induced terminal obs, which the follower env interprets as
    its initial state. Modelling exactly that distribution gives the
    sharpest OOD signal at the planner's query manifold.
    """
    from critic_sharing.config.resolver import load_env_config, load_train_config
    from critic_sharing.training.vae_from_rollouts import train_vae_from_rollouts

    detector = detector.lower()
    if detector not in ("knn", "vae"):
        raise typer.BadParameter(
            f"--detector must be 'knn' or 'vae' (got {detector!r})"
        )

    follower_path = resolve_model(follower_model, scenario.value, role="follower")
    typer.echo("Follower checkpoint (output dir source): {p}".format(p=follower_path))

    # Build follower env from scenario YAML. We don't load the PPO policy —
    # only env.reset is invoked.
    #
    # Apply the *same* env_overrides block that ``train follower`` uses,
    # so the VAE's training distribution matches V_F's by construction.
    # Without this, a cone-restricted spawn during follower training would
    # be modelled by a full-circle VAE here, and the OOD detector would
    # under-fire at out-of-cone leader queries.
    follower_cfg = load_env_config(scenario.value, "follower")
    train_resolved = load_train_config(
        scenario=scenario.value, role="follower",
        config_override_path=None, cli_overrides={},
    )
    follower_cfg = _apply_env_overrides(
        follower_cfg, train_resolved.get("env_overrides"),
    )
    env = _build_follower_env(scenario, follower_cfg)

    # Pull per-scenario vae_obs_transform if the scenario registers one.
    obs_transform, transform_name = _resolve_vae_obs_transform(scenario)

    # Optional OOD sample set for ROC calibration.
    ood_samples = None
    if ood_set is not None:
        import numpy as np

        with np.load(str(ood_set)) as f:
            if "samples" not in f:
                raise typer.BadParameter(
                    "--ood-set archive must contain key 'samples'.",
                )
            ood_samples = np.asarray(f["samples"], dtype=np.float32)
        typer.echo(
            "Loaded {n} OOD samples from {p} (shape={s})".format(
                n=int(ood_samples.shape[0]),
                p=ood_set,
                s=tuple(ood_samples.shape),
            )
        )

    vae, calibration, metadata = train_vae_from_rollouts(
        env=env,
        n_samples=int(n_samples),
        latent_dim=int(latent_dim),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        obs_transform=obs_transform,
        transform_name=transform_name,
        ood_samples=ood_samples,
        threshold_pct=float(ood_percentile),
        max_loss_pct=float(max_loss_percentile),
        seed=int(seed),
        fit_vae=(detector == "vae"),
    )

    # Default checkpoint name: ``follower_ood.pt`` (the new generic
    # name). ``_resolve_vae`` in cli/eval/_common.py also accepts the
    # legacy ``follower_vae.pt`` for back-compat with existing
    # artefacts on disk / on Drive.
    if output is None:
        output = Path(follower_path).parent / "follower_ood.pt"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch as th

    # Checkpoint contents differ by detector:
    # * Both detectors store the kNN payload — that's what the
    #   ``build_*_ood_metric`` closures consume on the planning side.
    # * Only ``--detector vae`` additionally stores the VAE state_dict
    #   + reconstruction-loss thresholds (used by the VAE-fallback
    #   branch when the kNN payload is missing).
    checkpoint: dict = {
        "detector": detector,
        "scenario": scenario.value,
        "transform_name": metadata["transform_name"],
        "feature_dim": int(metadata["feature_dim"]),
        "follower_model_path": str(follower_path),
        "n_samples": int(metadata["n_samples"]),
        "dataset_size": int(metadata["dataset_size"]),
        "train_samples": metadata["train_samples"],
        "knn_k": int(metadata["knn_k"]),
        "knn_threshold": float(metadata["knn_threshold"]),
        "knn_max_dist": float(metadata["knn_max_dist"]),
    }
    if detector == "vae":
        assert vae is not None and calibration is not None
        checkpoint.update({
            "state_dict": vae.state_dict(),
            "latent_dim": int(latent_dim),
            "threshold": float(calibration.threshold),
            "max_loss": float(calibration.max_loss),
            "calibration_method": calibration.method,
            "auroc": calibration.auroc,
            "loss_history": metadata["loss_history"],
        })
    th.save(checkpoint, str(output))
    typer.echo("{detector} OOD checkpoint saved to {p}".format(
        detector=detector, p=output,
    ))

    env.close()


@app.command("vae", hidden=True)
def train_vae_cli_legacy() -> None:
    """[deprecated] Use ``critic-sharing train ood --detector vae`` instead.

    The legacy ``train vae`` subcommand has been renamed to ``train
    ood`` with a ``--detector knn|vae`` switch (default kNN). Most
    users do not need the VAE branch; the default kNN-density
    detector is faster to fit and is what production evals use.
    """
    raise typer.BadParameter(
        "``critic-sharing train vae`` was renamed. Use:\n"
        "  critic-sharing train ood --scenario <s> --detector vae    "
        "# explicit VAE fallback path\n"
        "  critic-sharing train ood --scenario <s>                   "
        "# default knn-only (recommended)"
    )


def _build_follower_env(scenario: Scenario, env_config):
    """Instantiate the follower env for a scenario from its config."""
    if scenario == Scenario.obstacle_arena:
        from critic_sharing.scenarios.obstacle_arena.nav_env import OneObstacleReachEnv

        return OneObstacleReachEnv(env_config)
    if scenario in (
        Scenario.warehouse,
        Scenario.warehouse_pickup_blocked,
        Scenario.warehouse_pickup_unseen,
    ):
        from critic_sharing.scenarios.warehouse import WarehouseFollowerEnv

        return WarehouseFollowerEnv(env_config)
    if scenario == Scenario.mobile_arm:
        from critic_sharing.scenarios.mobile_arm import MobileArmFollowerEnv

        return MobileArmFollowerEnv(env_config)
    if scenario == Scenario.household:
        from critic_sharing.scenarios.household import HouseholdNavigatorEnv

        return HouseholdNavigatorEnv(env_config)
    raise typer.BadParameter("Unknown scenario: {s}".format(s=scenario.value))


def _resolve_vae_obs_transform(
    scenario: Scenario,
) -> tuple[Optional[Callable], str]:
    """Return ``(transform, transform_name)`` for a scenario, or ``(None, "identity")``.

    The transform is applied to follower-obs samples before fitting the
    VAE. ``None`` means identity (use full obs vector).
    """
    if scenario == Scenario.obstacle_arena:
        from critic_sharing.scenarios.obstacle_arena.ood_query import (
            VAE_OBS_TRANSFORM,
        )

        return VAE_OBS_TRANSFORM, "obstacle_arena_obstacle_pos_2d"
    if scenario in (
        Scenario.warehouse,
        Scenario.warehouse_pickup_blocked,
        Scenario.warehouse_pickup_unseen,
    ):
        from critic_sharing.scenarios.warehouse.ood_query import VAE_OBS_TRANSFORM

        return VAE_OBS_TRANSFORM, "warehouse_object_pos_2d"
    if scenario == Scenario.mobile_arm:
        from critic_sharing.scenarios.mobile_arm.ood_query import VAE_OBS_TRANSFORM

        return VAE_OBS_TRANSFORM, "mobile_arm_object_pos_base_2d"
    if scenario == Scenario.household:
        from critic_sharing.scenarios.household.ood_query import VAE_OBS_TRANSFORM

        return VAE_OBS_TRANSFORM, "household_table_pose_4d"
    return None, "identity"


def _apply_env_overrides(env_config, overrides):
    """:func:`critic_sharing.config.resolver.apply_env_overrides`, with a
    bad block reported as a CLI parameter error. Also used by ``train ood``
    so the detector sees V_F's training distribution."""
    from critic_sharing.config.resolver import apply_env_overrides

    try:
        return apply_env_overrides(env_config, overrides)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _prepare_train_kwargs(resolved: dict, train_fn) -> dict:
    """Filter resolved config to match the training function's signature.

    Handles:
      - Dropping keys the function doesn't accept
      - Flattening nested ``vae`` section into ``vae_*`` prefixed args (for nav)
    """
    import inspect

    sig = inspect.signature(train_fn)
    valid_params = set(sig.parameters.keys())
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    kwargs: dict = {}

    # Flatten nested vae section for nav's vae_* params
    vae_section = resolved.pop("vae", None)
    if isinstance(vae_section, dict):
        for vae_key, vae_val in vae_section.items():
            prefixed = "vae_{k}".format(k=vae_key)
            if prefixed in valid_params or has_var_keyword:
                kwargs[prefixed] = vae_val

    for key, value in resolved.items():
        if isinstance(value, dict):
            # Skip remaining nested sections (paths, wandb, run, etc.)
            continue
        if key in valid_params or has_var_keyword:
            kwargs[key] = value

    return kwargs
