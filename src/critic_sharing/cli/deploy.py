"""``critic-sharing deploy`` subcommands.

ROS2-backed runtimes that load a trained policy and drive a real
robot. ``rclpy`` is imported lazily inside the command so that
``critic-sharing --help`` works on machines without a ROS2 install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from critic_sharing.cli._common import Role, resolve_model

app = typer.Typer(no_args_is_help=True)


@app.command("household")
def deploy_household(
    config: Optional[Path] = typer.Option(
        None, help="YAML override on top of configs/deployment/household.yaml."
    ),
    model_path: Optional[Path] = typer.Option(
        None, help="PPO checkpoint (.zip). If omitted, the latest run for `role` is used."
    ),
    role: Optional[str] = typer.Option(
        None, help="'cleaner' or 'follower'. Overrides the YAML role."
    ),
    parent_frame: Optional[str] = typer.Option(None, help="Real-world parent TF frame."),
    cleaner_frame: Optional[str] = typer.Option(None, help="Cleaner child TF frame."),
    follower_frame: Optional[str] = typer.Option(None, help="Follower child TF frame."),
    table_frame: Optional[str] = typer.Option(None, help="Table (obstacle) child TF frame."),
    dirt_frame: Optional[str] = typer.Option(
        None, help="Dirt child TF frame; empty → latch to table pose at startup."
    ),
    cmd_topic: Optional[str] = typer.Option(None, help="freyja ReferenceState publish topic."),
    rate_hz: Optional[float] = typer.Option(None, help="Control rate in Hz."),
    goto_origin: bool = typer.Option(
        False,
        "--goto-origin/--no-goto-origin",
        help="Debug: skip the policy and steer the robot toward sim (0, 0).",
    ),
    goto_dirt: bool = typer.Option(
        False,
        "--goto-dirt/--no-goto-dirt",
        help="Debug: skip the policy and steer the robot directly toward the dirt.",
    ),
    episode: bool = typer.Option(
        False, "--episode/--no-episode",
        help="Bounded episode mode: terminate on pickup/goal/arena/max-steps/Ctrl-C, "
             "with explicit ready phase + zero-velocity-on-exit.",
    ),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps",
        help="Tick cap in --episode mode. Defaults to sim per-role "
             "(cleaner=200, follower=250).",
    ),
    arena_margin: Optional[float] = typer.Option(
        None, "--arena-margin",
        help="Shrink arena bounds by this many meters for the out-of-arena "
             "predicate (default 0.1).",
    ),
    ready_timeout: Optional[float] = typer.Option(
        None, "--ready-timeout",
        help="Seconds to wait for first valid table + self TF in Phase 1 "
             "(default 5.0).",
    ),
    no_confirm: bool = typer.Option(
        False, "--no-confirm/--confirm",
        help="Skip the 'press Enter to begin' gate after the ready log.",
    ),
    viz: bool = typer.Option(
        True, "--viz/--no-viz",
        help="Show the live matplotlib visualization. Disable for "
             "latency-sensitive runs — drawing competes with the TF "
             "listener for executor time and can stall the loop.",
    ),
    record: bool = typer.Option(
        False, "--record/--no-record",
        help="Save MP4 + metadata.json under experiments/deployments/. "
             "Requires --viz (recorder reuses the figure).",
    ),
    record_dir: Optional[Path] = typer.Option(
        None, "--record-dir",
        help="Override the auto-timestamped output directory.",
    ),
    profile: bool = typer.Option(
        False, "--profile/--no-profile",
        help="Per-phase latency breakdown of the control loop printed on exit.",
    ),
    controller: Optional[str] = typer.Option(
        None, "--controller",
        help="'policy' (default: trained PPO each tick) or 'cem' "
             "(CEM planner with the cleaner PPO as warm-start prior and "
             "follower V_F in the joint utility). cleaner role only.",
    ),
    follower_model_path: Optional[Path] = typer.Option(
        None, "--follower-model-path",
        help="Follower PPO checkpoint — required for --controller cem.",
    ),
    vae: Optional[Path] = typer.Option(
        None, "--vae",
        help="VAE checkpoint — when set, CEM runs in value_and_ood mode "
             "(follower V_F multiplied by an OOD trust factor).",
    ),
    alpha: Optional[float] = typer.Option(
        None, "--alpha",
        help="OOD trust scaling for value_and_ood mode (default 1.0).",
    ),
) -> None:
    """Deploy a trained household policy on a Robomaster via ROS2."""

    from critic_sharing.config.resolver import load_household_deployment_config
    from critic_sharing.deployment.household_node import run

    cli_overrides: dict = {}
    if model_path is not None:
        cli_overrides["model_path"] = str(model_path)
    if role is not None:
        cli_overrides["role"] = role
    if parent_frame is not None:
        cli_overrides["parent_frame"] = parent_frame
    if cleaner_frame is not None:
        cli_overrides["cleaner_frame"] = cleaner_frame
    if follower_frame is not None:
        cli_overrides["follower_frame"] = follower_frame
    if table_frame is not None:
        cli_overrides["table_frame"] = table_frame
    if dirt_frame is not None:
        cli_overrides["dirt_frame"] = dirt_frame
    if cmd_topic is not None:
        cli_overrides["cmd_topic"] = cmd_topic
    if rate_hz is not None:
        cli_overrides["rate_hz"] = float(rate_hz)
    if goto_origin:
        cli_overrides["goto_origin"] = True
    if goto_dirt:
        cli_overrides["goto_dirt"] = True
    if episode:
        cli_overrides["episode"] = True
    if max_steps is not None:
        cli_overrides["max_steps_override"] = int(max_steps)
    if arena_margin is not None:
        cli_overrides["arena_margin"] = float(arena_margin)
    if ready_timeout is not None:
        cli_overrides["ready_timeout"] = float(ready_timeout)
    if no_confirm:
        cli_overrides["no_confirm"] = True
    if not viz:
        cli_overrides["viz"] = False
    if record:
        cli_overrides["record"] = True
    if record_dir is not None:
        cli_overrides["record_dir"] = str(record_dir)
    if profile:
        cli_overrides["profile"] = True
    if controller is not None:
        if controller not in ("policy", "cem"):
            raise typer.BadParameter(
                f"--controller must be 'policy' or 'cem' (got {controller!r})."
            )
        cli_overrides["controller"] = controller
    if follower_model_path is not None:
        cli_overrides["follower_model_path"] = str(follower_model_path)
    if vae is not None:
        cli_overrides["vae_path"] = str(vae)
    if alpha is not None:
        cli_overrides["alpha"] = float(alpha)

    cfg = load_household_deployment_config(
        config_override_path=config,
        cli_overrides=cli_overrides or None,
    )

    # Mutexes / required combinations (grill Q7a).
    if cfg.controller == "cem":
        if cfg.goto_origin or cfg.goto_dirt:
            raise typer.BadParameter(
                "--controller cem is mutually exclusive with --goto-origin "
                "and --goto-dirt; pick one."
            )
        if not cfg.episode:
            raise typer.BadParameter(
                "--controller cem requires --episode (CEM bookkeeping needs "
                "a bounded rollout for incurred-return tracking)."
            )
        if cfg.role != "cleaner":
            raise typer.BadParameter(
                "--controller cem is cleaner-only; got --role {!r}.".format(cfg.role)
            )
        # Auto-resolve the follower model the same way the cleaner
        # model is resolved below — pick the latest follower run.
        if not cfg.follower_model_path:
            resolved_follower = resolve_model(
                path=None, scenario="household", role=Role.follower.value,
            )
            from dataclasses import replace as _replace
            cfg = _replace(cfg, follower_model_path=str(resolved_follower))
            typer.echo(
                "Auto-resolved follower model: {p}".format(p=cfg.follower_model_path),
            )

    # Warn for flags that only have meaning in --episode mode.
    if not cfg.episode:
        spurious = []
        if max_steps is not None: spurious.append("--max-steps")
        if arena_margin is not None: spurious.append("--arena-margin")
        if ready_timeout is not None: spurious.append("--ready-timeout")
        if no_confirm: spurious.append("--no-confirm")
        if record_dir is not None: spurious.append("--record-dir")
        if spurious:
            typer.echo(
                "Warning: {flags} have no effect without --episode.".format(
                    flags=", ".join(spurious),
                ),
                err=True,
            )

    if not cfg.goto_origin and not cfg.goto_dirt and not cfg.model_path:
        scenario_role = Role.leader.value if cfg.role == "cleaner" else Role.follower.value
        resolved = resolve_model(path=None, scenario="household", role=scenario_role)
        from dataclasses import replace
        cfg = replace(cfg, model_path=str(resolved))

    if cfg.goto_origin:
        typer.echo(
            "Debug mode: steering robot toward sim (0, 0) on topic {topic}.".format(
                topic=cfg.cmd_topic,
            )
        )
    elif cfg.goto_dirt:
        typer.echo(
            "Debug mode: steering robot toward dirt (latched from {table}) on topic {topic}.".format(
                table=cfg.table_frame, topic=cfg.cmd_topic,
            )
        )
    else:
        typer.echo(
            "Deploying household/{role} from {path} on topic {topic}.".format(
                role=cfg.role, path=cfg.model_path, topic=cfg.cmd_topic,
            )
        )
    if cfg.episode:
        from critic_sharing.deployment.household_episode import run_episode
        run_episode(cfg)
    else:
        run(cfg)


@app.command("household-compare")
def deploy_household_compare(
    config: Optional[Path] = typer.Option(
        None, help="YAML override on top of configs/deployment/household.yaml."
    ),
    cleaner_model_path: Optional[Path] = typer.Option(
        None, "--cleaner-model-path",
        help="Cleaner PPO checkpoint (.zip). Defaults to the latest household leader run.",
    ),
    follower_model_path: Optional[Path] = typer.Option(
        None, "--follower-model-path",
        help="Follower PPO checkpoint (.zip). Defaults to the latest household follower run.",
    ),
    order: str = typer.Option(
        "naive,ours", "--order",
        help="Comma-separated method order. Either 'naive,ours' or 'ours,naive'.",
    ),
    repeats: int = typer.Option(
        1, "--repeats",
        help="Number of (place-objects → run-both-methods) pairs.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir",
        help="Session directory. Defaults to experiments/comparisons/<timestamp>/.",
    ),
    webcam_device: str = typer.Option(
        "OsmoAction5pro", "--webcam-device",
        help="Camera spec: a name substring (matched against the stable "
             "/dev/v4l/by-id names — survives reconnect), a numeric index, "
             "or a /dev path (ls /dev/v4l/by-id lists the options).",
    ),
    webcam_fps: int = typer.Option(
        30, "--webcam-fps",
        help="Webcam recording FPS.",
    ),
    webcam_preview: bool = typer.Option(
        True, "--webcam-preview/--no-webcam-preview",
        help="Show the live cv2.imshow window while recording.",
    ),
    bag: bool = typer.Option(
        True, "--bag/--no-bag",
        help="Spawn `ros2 bag record -a` per phase.",
    ),
    webcam: bool = typer.Option(
        True, "--webcam/--no-webcam",
        help="Spawn the OpenCV webcam recorder per phase.",
    ),
    # Standard household passthroughs (subset of deploy_household's flags
    # that make sense here — debug modes excluded).
    parent_frame: Optional[str] = typer.Option(None, help="Real-world parent TF frame."),
    cleaner_frame: Optional[str] = typer.Option(None, help="Cleaner child TF frame."),
    follower_frame: Optional[str] = typer.Option(None, help="Follower child TF frame."),
    table_frame: Optional[str] = typer.Option(None, help="Table child TF frame."),
    dirt_frame: Optional[str] = typer.Option(
        None, help="Dirt child TF frame; empty → latched to table pose at startup."
    ),
    # NOTE: --cmd-topic is intentionally omitted — each robot listens on
    # /<frame>/reference_state, so the comparison harness derives the
    # topic per-role from cleaner_frame / follower_frame. Override the
    # frames instead.
    rate_hz: Optional[float] = typer.Option(None, help="Control rate in Hz."),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps",
        help="Tick cap per rollout. Default = sim per-role (cleaner=200, follower=250).",
    ),
    arena_margin: Optional[float] = typer.Option(
        None, "--arena-margin",
        help="Shrink arena bounds by this many meters (default 0.1).",
    ),
    ready_timeout: Optional[float] = typer.Option(
        None, "--ready-timeout",
        help="Seconds to wait for first valid TF (default 5.0).",
    ),
    viz: bool = typer.Option(
        True, "--viz/--no-viz",
        help="Show the live matplotlib visualization per rollout.",
    ),
    profile: bool = typer.Option(
        False, "--profile/--no-profile",
        help="Per-phase latency breakdown per rollout.",
    ),
    vae: Optional[Path] = typer.Option(
        None, "--vae",
        help="VAE checkpoint for CEM's value_and_ood mode (passed to 'ours' cleaner).",
    ),
    alpha: Optional[float] = typer.Option(
        None, "--alpha",
        help="OOD trust scaling for value_and_ood mode (default 1.0).",
    ),
    cem_num_samples: Optional[int] = typer.Option(
        None, "--cem-num-samples",
        help="Override CEM num_samples for 'ours' cleaner (default: planner.yaml=100).",
    ),
    cem_horizon: Optional[int] = typer.Option(
        None, "--cem-horizon",
        help="Override CEM horizon (default: planner.yaml=20).",
    ),
    cem_num_iters: Optional[int] = typer.Option(
        None, "--cem-num-iters",
        help="Override CEM num_iters (default: planner.yaml=5).",
    ),
    cem_replan_interval: Optional[int] = typer.Option(
        None, "--cem-replan-interval",
        help="Override CEM replan_interval (default: planner.yaml=5). "
             "Set 1 to replan every control tick once a replan fits the period.",
    ),
) -> None:
    """A/B comparison harness: naive vs CEM, with per-phase rosbag + webcam capture.

    Per session: ``--repeats N`` pairs. Per pair: 4 rollouts
    (method-A cleaner, method-A follower, reset view, method-B cleaner,
    method-B follower). Each rollout is bracketed by ``ros2 bag record -a``
    and a USB webcam (OpenCV) recording.
    """
    from dataclasses import replace as _replace

    from critic_sharing.config.resolver import load_household_deployment_config
    from critic_sharing.deployment.household_compare import (
        HouseholdComparisonConfig, run_comparison,
    )

    # Parse --order
    parsed_order = tuple(s.strip() for s in order.split(",") if s.strip())
    if not parsed_order:
        raise typer.BadParameter("--order must not be empty.")
    for m in parsed_order:
        if m not in ("naive", "ours"):
            raise typer.BadParameter(
                f"--order entries must be 'naive' or 'ours' (got {m!r})."
            )

    # Build the base HouseholdDeploymentConfig (per-rollout settings).
    cli_overrides: dict = {}
    if cleaner_model_path is not None:
        cli_overrides["model_path"] = str(cleaner_model_path)
    if follower_model_path is not None:
        cli_overrides["follower_model_path"] = str(follower_model_path)
    if parent_frame is not None:
        cli_overrides["parent_frame"] = parent_frame
    if cleaner_frame is not None:
        cli_overrides["cleaner_frame"] = cleaner_frame
    if follower_frame is not None:
        cli_overrides["follower_frame"] = follower_frame
    if table_frame is not None:
        cli_overrides["table_frame"] = table_frame
    if dirt_frame is not None:
        cli_overrides["dirt_frame"] = dirt_frame
    if rate_hz is not None:
        cli_overrides["rate_hz"] = float(rate_hz)
    if max_steps is not None:
        cli_overrides["max_steps_override"] = int(max_steps)
    if arena_margin is not None:
        cli_overrides["arena_margin"] = float(arena_margin)
    if ready_timeout is not None:
        cli_overrides["ready_timeout"] = float(ready_timeout)
    if not viz:
        cli_overrides["viz"] = False
    if profile:
        cli_overrides["profile"] = True
    if vae is not None:
        cli_overrides["vae_path"] = str(vae)
    if alpha is not None:
        cli_overrides["alpha"] = float(alpha)
    if cem_num_samples is not None:
        cli_overrides["cem_num_samples"] = int(cem_num_samples)
    if cem_horizon is not None:
        cli_overrides["cem_horizon"] = int(cem_horizon)
    if cem_num_iters is not None:
        cli_overrides["cem_num_iters"] = int(cem_num_iters)
    if cem_replan_interval is not None:
        cli_overrides["cem_replan_interval"] = int(cem_replan_interval)
    # The comparison harness always runs in --episode mode and never
    # uses the debug shortcuts.
    cli_overrides["episode"] = True

    cfg = load_household_deployment_config(
        config_override_path=config,
        cli_overrides=cli_overrides or None,
    )

    # Auto-resolve both model paths upfront so each rollout uses a
    # deterministic checkpoint.
    if not cfg.model_path:
        resolved = resolve_model(path=None, scenario="household", role=Role.leader.value)
        cfg = _replace(cfg, model_path=str(resolved))
        typer.echo(f"Auto-resolved cleaner model: {cfg.model_path}")
    if not cfg.follower_model_path:
        resolved = resolve_model(
            path=None, scenario="household", role=Role.follower.value,
        )
        cfg = _replace(cfg, follower_model_path=str(resolved))
        typer.echo(f"Auto-resolved follower model: {cfg.follower_model_path}")

    compare_config = HouseholdComparisonConfig(
        repeats=int(repeats),
        order=parsed_order,
        output_dir=output_dir,
        webcam_device=webcam_device,
        webcam_fps=int(webcam_fps),
        webcam_preview=bool(webcam_preview),
        bag_enabled=bool(bag),
        webcam_enabled=bool(webcam),
    )

    typer.echo(
        "Starting household-compare:\n"
        f"  order={parsed_order}, repeats={repeats}\n"
        f"  cleaner_model={cfg.model_path}\n"
        f"  follower_model={cfg.follower_model_path}\n"
        f"  webcam_device={webcam_device}, bag={bag}, webcam={webcam}",
    )
    session_dir = run_comparison(base_config=cfg, compare_config=compare_config)
    typer.echo(f"Done. Session: {session_dir}")


@app.command("visualize-frames")
def visualize_frames(
    config: Optional[Path] = typer.Option(
        None, help="YAML override on top of configs/deployment/household.yaml."
    ),
    parent_frame: Optional[str] = typer.Option(None, help="Real-world parent TF frame."),
    cleaner_frame: Optional[str] = typer.Option(None, help="Cleaner child TF frame."),
    follower_frame: Optional[str] = typer.Option(None, help="Follower child TF frame."),
    table_frame: Optional[str] = typer.Option(None, help="Table (obstacle) child TF frame."),
    dirt_frame: Optional[str] = typer.Option(
        None, help="Dirt child TF frame; empty → latch to table pose at first lookup."
    ),
    rate_hz: Optional[float] = typer.Option(None, help="TF poll + redraw rate (Hz)."),
) -> None:
    """Live matplotlib visualization of cleaner/follower/table TF in sim coords.

    Useful for confirming ``sim_origin_in_world`` / ``parent_z_down``
    align the lab geometry with the simulator's arena before launching
    the closed-loop policy.
    """

    from critic_sharing.config.resolver import load_household_deployment_config
    from critic_sharing.deployment.household_viz_node import run_visualizer

    cli_overrides: dict = {}
    if parent_frame is not None:
        cli_overrides["parent_frame"] = parent_frame
    if cleaner_frame is not None:
        cli_overrides["cleaner_frame"] = cleaner_frame
    if follower_frame is not None:
        cli_overrides["follower_frame"] = follower_frame
    if table_frame is not None:
        cli_overrides["table_frame"] = table_frame
    if dirt_frame is not None:
        cli_overrides["dirt_frame"] = dirt_frame
    if rate_hz is not None:
        cli_overrides["rate_hz"] = float(rate_hz)

    cfg = load_household_deployment_config(
        config_override_path=config,
        cli_overrides=cli_overrides or None,
    )
    typer.echo(
        "Visualizing TF: parent={p}, cleaner={c}, follower={f}, table={t}, "
        "sim_origin_in_world={origin}, parent_z_down={zd}".format(
            p=cfg.parent_frame, c=cfg.cleaner_frame, f=cfg.follower_frame,
            t=cfg.table_frame, origin=tuple(cfg.sim_origin_in_world),
            zd=cfg.parent_z_down,
        )
    )
    run_visualizer(cfg, fps=cfg.rate_hz)


__all__ = ["app"]
