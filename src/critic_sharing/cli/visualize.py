"""``critic-sharing visualize`` — value function, OOD metric, and rollout plots."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from critic_sharing.cli._common import Scenario, resolve_model

app = typer.Typer(no_args_is_help=True)


class RolloutAgent(str, Enum):
    leader = "leader"
    follower = "follower"
    both = "both"


def _default_output(command: str, scenario: Scenario, suffix: str) -> Path:
    """Return a default output path in the current working directory."""
    return Path(
        "{command}_{scenario}{suffix}".format(
            command=command, scenario=scenario.value, suffix=suffix
        )
    )


# ---------------------------------------------------------------------------
# visualize rollout
# ---------------------------------------------------------------------------


@app.command("rollout")
def visualize_rollout(
    scenario: Scenario = typer.Option(..., help="Scenario."),
    agent: RolloutAgent = typer.Option(
        RolloutAgent.both,
        help="Which agent(s) to roll out: leader | follower | both.",
    ),
    leader_model: Optional[Path] = typer.Option(
        None, help="Path to leader model (.zip). Defaults to latest artifact."
    ),
    follower_model: Optional[Path] = typer.Option(
        None, help="Path to follower model (.zip). Defaults to latest artifact."
    ),
    seed: int = typer.Option(0, help="Episode seed."),
    output: Optional[Path] = typer.Option(
        None,
        help=("Output path. Both scenarios default to .mp4. Use .gif for a GIF instead of MP4."),
    ),
    fps: int = typer.Option(4, help="Frames per second (animation only)."),
    show: bool = typer.Option(False, help="Display interactively instead of saving."),
) -> None:
    """Roll out leader, follower, or both agents sequentially and save a visualisation."""

    need_leader = agent in {RolloutAgent.leader, RolloutAgent.both}
    need_follower = agent in {RolloutAgent.follower, RolloutAgent.both}

    if need_leader:
        leader_model = resolve_model(leader_model, scenario.value, "leader")
    if need_follower:
        follower_model = resolve_model(follower_model, scenario.value, "follower")

    typer.echo(
        "Rolling out: scenario={s} agent={a} seed={seed}".format(
            s=scenario.value, a=agent.value, seed=seed
        )
    )

    if scenario == Scenario.obstacle_arena:
        _rollout_obstacle_arena(
            agent=agent,
            leader_model=leader_model,
            follower_model=follower_model,
            seed=seed,
            output=output,
            fps=fps,
            show=show,
        )

    else:
        typer.echo("Unknown scenario: {s}".format(s=scenario.value), err=True)
        raise typer.Exit(code=1)


def _save_or_show_fig(fig, output: Optional[Path], show: bool, default: Path) -> None:
    import matplotlib.pyplot as plt

    if show:
        plt.show()
    else:
        out = output if output is not None else default
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        typer.echo("Saved to {p}".format(p=out))
        plt.close(fig)


def _rollout_obstacle_arena(
    *,
    agent: RolloutAgent,
    leader_model: Optional[Path],
    follower_model: Optional[Path],
    seed: int,
    output: Optional[Path],
    fps: int,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import animation
    from stable_baselines3 import PPO

    from critic_sharing.cli.eval._animation import _save_anim as _save_animation
    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.obstacle_arena.nav_env import OneObstacleReachEnv
    from critic_sharing.scenarios.obstacle_arena.push_env import PushObstacleEnv

    push_cfg = load_env_config("obstacle_arena", "leader")
    nav_cfg = load_env_config("obstacle_arena", "follower")

    # Each frame: agent_x, agent_y, obstacle_x, obstacle_y, phase string
    frames: list[dict] = []
    final_obstacle_pos: Optional[np.ndarray] = None
    goal_pos: Optional[np.ndarray] = None

    # --- leader phase ---
    if agent in {RolloutAgent.leader, RolloutAgent.both}:
        assert leader_model is not None
        push_model = PPO.load(str(leader_model))
        push_env = PushObstacleEnv(push_cfg)
        obs, _ = push_env.reset(seed=seed)
        done = False
        while not done:
            frames.append(
                {
                    "agent_x": float(obs[0]),
                    "agent_y": float(obs[1]),
                    "obstacle_x": float(obs[2]),
                    "obstacle_y": float(obs[3]),
                    "phase": "leader",
                }
            )
            action, _ = push_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = push_env.step(action)
            done = terminated or truncated
        # append final state
        frames.append(
            {
                "agent_x": float(obs[0]),
                "agent_y": float(obs[1]),
                "obstacle_x": float(obs[2]),
                "obstacle_y": float(obs[3]),
                "phase": "leader",
            }
        )
        final_obstacle_pos = obs[2:4].copy()

    # --- follower phase ---
    if agent in {RolloutAgent.follower, RolloutAgent.both}:
        assert follower_model is not None
        nav_model = PPO.load(str(follower_model))
        nav_env = OneObstacleReachEnv(nav_cfg)
        reset_opts = (
            {"obstacle_pos": final_obstacle_pos} if final_obstacle_pos is not None else None
        )
        obs, _ = nav_env.reset(seed=seed, options=reset_opts)
        goal_pos = nav_env._goal.copy()
        done = False
        while not done:
            frames.append(
                {
                    "agent_x": float(obs[0]),
                    "agent_y": float(obs[1]),
                    "obstacle_x": float(obs[2]),
                    "obstacle_y": float(obs[3]),
                    "phase": "follower",
                }
            )
            action, _ = nav_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = nav_env.step(action)
            done = terminated or truncated
        frames.append(
            {
                "agent_x": float(obs[0]),
                "agent_y": float(obs[1]),
                "obstacle_x": float(obs[2]),
                "obstacle_y": float(obs[3]),
                "phase": "follower",
            }
        )

    if not frames:
        typer.echo("No frames collected — nothing to animate.", err=True)
        raise typer.Exit(code=1)

    # --- build animation ---
    COLORS = {"leader": "#E65100", "follower": "#1565C0"}
    world_min = nav_cfg.world_min
    world_max = nav_cfg.world_max

    def _obs_radius(phase: str) -> float:
        return float(push_cfg.obstacle_radius if phase == "leader" else nav_cfg.obstacle_radius)

    from critic_sharing.scenarios.obstacle_arena.common import draw_wall

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(world_min, world_max)
    ax.set_ylim(world_min, world_max)
    ax.set_aspect("equal")
    ax.set_title("obstacle_arena — {a} (seed={s})".format(a=agent.value, s=seed))

    # Draw wall
    draw_wall(ax, nav_cfg)

    if goal_pos is not None:
        ax.scatter(
            float(goal_pos[0]),
            float(goal_pos[1]),
            marker="*",
            s=200,
            color="#43A047",
            zorder=4,
            label="goal",
        )
        ax.legend(loc="upper right", fontsize=8)

    obstacle_patch = plt.Circle(
        (0.0, 0.0), _obs_radius("leader"), color="#757575", alpha=0.5, zorder=3
    )
    ax.add_patch(obstacle_patch)
    (trail_line,) = ax.plot([], [], "-", linewidth=1.2, alpha=0.6, zorder=2)
    (agent_dot,) = ax.plot([], [], "o", markersize=8, zorder=5)
    phase_text = ax.text(
        world_min + 0.05 * (world_max - world_min),
        world_max - 0.05 * (world_max - world_min),
        "",
        fontsize=9,
        va="top",
    )

    def _update(i: int):
        f = frames[i]
        phase = f["phase"]
        color = COLORS[phase]

        obstacle_patch.center = (f["obstacle_x"], f["obstacle_y"])
        obstacle_patch.radius = _obs_radius(phase)

        # trail: all same-phase frames up to i
        same = [ff for ff in frames[: i + 1] if ff["phase"] == phase]
        trail_line.set_data([ff["agent_x"] for ff in same], [ff["agent_y"] for ff in same])
        trail_line.set_color(color)

        agent_dot.set_data([f["agent_x"]], [f["agent_y"]])
        agent_dot.set_color(color)

        phase_text.set_text(phase)
        phase_text.set_color(color)

        return obstacle_patch, trail_line, agent_dot, phase_text

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=max(1, 1000 // fps),
        blit=True,
    )

    if show:
        plt.show()
        plt.close(fig)
        return

    out_suffix = ".mp4"
    if output is not None and output.suffix.lower() in {".gif", ".mp4"}:
        out_suffix = output.suffix.lower()
    out = (
        output
        if output is not None
        else Path("rollout_obstacle_arena_{a}{ext}".format(a=agent.value, ext=out_suffix))
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_animation(anim, out, fps=fps)
    typer.echo("Animation saved to {p}".format(p=out))
    plt.close(fig)


# ---------------------------------------------------------------------------
# visualize value
# ---------------------------------------------------------------------------


@app.command("value")
def visualize_value(
    scenario: Scenario = typer.Option(..., help="Scenario."),
    model: Optional[Path] = typer.Option(
        None, help="Path to trained follower model (.zip). Defaults to latest artifact."
    ),
    distilled: Optional[Path] = typer.Option(
        None,
        help="Path to distilled value MLP (.pt). If omitted, computes live.",
    ),
    grid_size: int = typer.Option(50, help="Resolution of the value heatmap."),
    extent: float = typer.Option(
        5.0,
        help=(
            "mobile_arm only: half-side of the (x, y) base-frame square "
            "the heatmap covers. Default 5.0 = full world extent so OOD "
            "behaviour is visible far outside the training spawn range."
        ),
    ),
    by: str = typer.Option(
        "follower",
        help=(
            "household only: which axis to sweep. ``follower`` (default) "
            "fixes the table at gap-center and varies follower (x, y); "
            "``table`` fixes the follower at the upper-half centre and "
            "varies table (x, y)."
        ),
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Output path (default: value_{scenario}.png).",
    ),
    show: bool = typer.Option(False, help="Display interactively instead of saving."),
) -> None:
    """Plot the follower's value function over the shared state space."""
    import numpy as np

    model = resolve_model(model, scenario.value)
    typer.echo(
        "Visualising value function: scenario={s} model={m}".format(s=scenario.value, m=model)
    )

    if scenario == Scenario.obstacle_arena:
        import matplotlib.pyplot as plt
        from stable_baselines3 import PPO

        from critic_sharing.config.schema import NavEnvConfig as EnvConfig
        from critic_sharing.scenarios.obstacle_arena.nav_env import (
            evaluate_value_grid as _evaluate_value_grid,
        )

        nav_model = PPO.load(str(model))
        config = EnvConfig()

        obstacle_x_min, obstacle_x_max, obstacle_y_min, obstacle_y_max = config.obstacle_bounds()
        rng = np.random.default_rng(0)
        obstacle_positions = np.stack(
            [
                rng.uniform(obstacle_x_min, obstacle_x_max, size=4).astype(np.float32),
                rng.uniform(obstacle_y_min, obstacle_y_max, size=4).astype(np.float32),
            ],
            axis=1,
        )

        fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
        for idx, obstacle_pos in enumerate(obstacle_positions):
            ax = axes.ravel()[idx]
            grid_x, grid_y, values = _evaluate_value_grid(
                nav_model, config, obstacle_pos, grid_size
            )
            ax.imshow(
                values,
                extent=(
                    config.world_min,
                    config.world_max,
                    config.world_min,
                    config.world_max,
                ),
                origin="lower",
                cmap="viridis",
                aspect="equal",
            )
            ax.scatter(
                float(obstacle_pos[0]),
                float(obstacle_pos[1]),
                marker="o",
                s=80,
                color="red",
                label="obstacle",
            )
            ax.set_title(
                "obstacle=({:.2f},{:.2f})".format(float(obstacle_pos[0]), float(obstacle_pos[1]))
            )
        fig.suptitle("Nav follower value function")
        plt.tight_layout()

        if show:
            plt.show()
        else:
            out = output if output is not None else _default_output("value", scenario, ".png")
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=150)
            typer.echo("Saved to {p}".format(p=out))
            plt.close(fig)

    elif scenario == Scenario.mobile_arm:
        _visualize_value_mobile_arm(
            model_path=model,
            grid_size=int(grid_size),
            extent=float(extent),
            output=output,
            show=show,
        )

    elif scenario == Scenario.household:
        _visualize_value_household(
            model_path=model,
            grid_size=int(grid_size),
            by=str(by),
            output=output,
            show=show,
        )

    else:
        typer.echo("Unknown scenario: {s}".format(s=scenario.value), err=True)
        raise typer.Exit(code=1)


def _visualize_value_mobile_arm(
    *,
    model_path: Path,
    grid_size: int,
    extent: float,
    output: Optional[Path],
    show: bool,
) -> None:
    """2x2 V heatmap over (object_x_base, object_y_base) for 4 θ_1 values.

    Plot orientation: panel +Y axis = base +X (so the tray, mounted on the
    chassis front along base +x, always points up). Each panel uses the
    same colour scale so panels are visually comparable; the arm overlay
    rotates with θ_1, the base footprint and tray stay fixed.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import torch as th
    from stable_baselines3 import PPO

    from critic_sharing.config.resolver import load_env_config, load_run_config
    from critic_sharing.utils.experiment import init_experiment

    follower_cfg = load_env_config("mobile_arm", "follower")
    follower_model = PPO.load(str(model_path))

    # ---- Build experiment bundle for output ----
    if output is not None:
        out_dir = output.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        target_path = output
    else:
        resolved_cfg = load_run_config("mobile_arm")
        exp_paths = init_experiment(
            name="visualize-value-mobile_arm",
            resolved_config=resolved_cfg,
            artifact_paths={"follower_model": str(model_path)},
            extra_metadata={
                "grid_size": grid_size,
                "extent": extent,
                "theta1_values": [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469],
            },
        )
        target_path = exp_paths.outputs_dir / "vfunc.png"
        typer.echo("Experiment dir: {p}".format(p=exp_paths.experiment_dir))

    # ---- Grid + V queries ----
    xs = np.linspace(-extent, extent, grid_size, dtype=np.float32)  # base x range
    ys = np.linspace(-extent, extent, grid_size, dtype=np.float32)  # base y range
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")  # shape (grid_size, grid_size)

    theta1_values = [
        ("θ₁ = 0",      0.0),
        ("θ₁ = π/2",    float(np.pi / 2)),
        ("θ₁ = π",      float(np.pi)),
        ("θ₁ = 3π/2",   float(3 * np.pi / 2)),
    ]
    theta2 = float(follower_cfg.theta2_default)
    theta3 = float(follower_cfg.theta3_default)
    theta4 = float(follower_cfg.theta4_default)
    sin2, cos2 = float(np.sin(theta2)), float(np.cos(theta2))
    sin3, cos3 = float(np.sin(theta3)), float(np.cos(theta3))
    sin4, cos4 = float(np.sin(theta4)), float(np.cos(theta4))

    n_pts = grid_size * grid_size
    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)

    panel_values = []  # (label, theta1, values_grid)
    for label, theta1 in theta1_values:
        sin1, cos1 = float(np.sin(theta1)), float(np.cos(theta1))
        cf_obs = np.zeros((n_pts, 12), dtype=np.float32)
        cf_obs[:, 0] = sin1
        cf_obs[:, 1] = cos1
        cf_obs[:, 2] = sin2
        cf_obs[:, 3] = cos2
        cf_obs[:, 4] = sin3
        cf_obs[:, 5] = cos3
        cf_obs[:, 6] = sin4
        cf_obs[:, 7] = cos4
        cf_obs[:, 8] = flat_x
        cf_obs[:, 9] = flat_y
        cf_obs[:, 10] = 0.0  # object z
        cf_obs[:, 11] = 0.0  # carry
        obs_tensor, _ = follower_model.policy.obs_to_tensor(cf_obs)
        with th.no_grad():
            values = follower_model.policy.predict_values(obs_tensor)
        v = values.detach().cpu().numpy().reshape(grid_size, grid_size)
        panel_values.append((label, theta1, v))

    # ---- Shared color scale ----
    vmin = min(v.min() for _, _, v in panel_values)
    vmax = max(v.max() for _, _, v in panel_values)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    fig.suptitle(
        "Follower V(object_pos | θ_1) — defaults θ_2=π/2, θ_3=π/4, θ_4=π/4, carry=0",
        fontsize=11,
    )

    last_im = None
    for ax, (label, theta1, v) in zip(axes.ravel(), panel_values):
        # Plot orientation: panel-vertical = base x, panel-horizontal = base y.
        # imshow expects values[row, col] with row=panel-y, col=panel-x. We
        # have values indexed by (x_base, y_base). To get panel-y = base x
        # increasing upward, transpose so panel-y maps to first axis (x) and
        # panel-x maps to second axis (y), and origin="lower" so first row
        # is at the bottom.
        last_im = ax.imshow(
            v,
            extent=(-extent, extent, -extent, extent),
            origin="lower",
            cmap="viridis",
            aspect="equal",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("y_base")
        ax.set_ylabel("x_base")
        # Note: x_base is on the vertical axis. To make this work intuitively,
        # we plot v[i, j] where i indexes x_base, j indexes y_base; imshow
        # with origin='lower' renders row index as vertical position. So
        # vertical axis IS x_base. Horizontal axis IS y_base. OK.

        # ---- Overlays: base footprint, tray, arm at θ_1, in panel coords ----
        # Panel coords: (panel_x = base_y, panel_y = base_x).
        side = float(follower_cfg.base_size)
        # Base footprint: square centred at origin; in panel coords still a
        # square (rotation by 90° around origin maps a square to a square).
        ax.add_patch(
            mpatches.Rectangle(
                (-side / 2, -side / 2), side, side,
                facecolor="#90A4AE", edgecolor="#37474F",
                linewidth=1.0, alpha=0.5, zorder=4,
            )
        )
        # Tray: at base_x = tray_offset_x, base_y = 0 → panel_x=0, panel_y=tray_offset_x.
        tray_x_panel = float(follower_cfg.tray_offset_y)  # = 0
        tray_y_panel = float(follower_cfg.tray_offset_x)  # = 0.25
        ax.plot(
            tray_x_panel, tray_y_panel,
            marker="s", markersize=10, color="#FB8C00",
            markeredgecolor="#E65100", zorder=6,
        )
        ax.text(
            tray_x_panel, tray_y_panel + 0.15,
            "tray", color="#E65100", ha="center", fontsize=8, zorder=6,
        )

        # Arm at θ_1: direction in base frame = (cos(θ_1), sin(θ_1)).
        # In panel coords: (panel_x = base_y = sin(θ_1), panel_y = base_x = cos(θ_1)).
        # Project all 3 segments using the planar FK and rotate to panel.
        L1 = float(follower_cfg.L1)
        L2a = float(follower_cfg.L2a)
        L2b = float(follower_cfg.L2b)
        phi_upper = theta2
        phi_L2a = theta2 - theta3
        phi_L2b = theta2 - theta3 - theta4
        elbow_r = L1 * np.cos(phi_upper)
        wrist_r = elbow_r + L2a * np.cos(phi_L2a)
        ee_r = wrist_r + L2b * np.cos(phi_L2b)
        # In base frame xy: base_x = r·cos(θ_1), base_y = r·sin(θ_1).
        # Panel coords: panel_x = base_y, panel_y = base_x.
        c1, s1 = float(np.cos(theta1)), float(np.sin(theta1))
        shoulder = (0.0, 0.0)
        elbow = (elbow_r * s1, elbow_r * c1)
        wrist = (wrist_r * s1, wrist_r * c1)
        ee = (ee_r * s1, ee_r * c1)
        ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]],
                color="#1565C0", linewidth=2.5, zorder=7)
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]],
                color="#0D47A1", linewidth=2.5, zorder=7)
        ax.plot([wrist[0], ee[0]], [wrist[1], ee[1]],
                color="#0D47A1", linewidth=2.5, alpha=0.85, zorder=7)
        ax.plot(*ee, marker="*", markersize=10, color="#43A047",
                markeredgecolor="#1B5E20", zorder=8)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.025, 0.76])
    fig.colorbar(last_im, cax=cbar_ax, label="V_follower")

    if show:
        plt.show()
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(target_path), dpi=150)
        typer.echo("Saved to {p}".format(p=target_path))
        plt.close(fig)


def _visualize_value_household(
    *,
    model_path: Path,
    grid_size: int,
    by: str,
    output: Optional[Path],
    show: bool,
) -> None:
    """Two-panel V_follower heatmap.

    ``by="follower"`` (default): fix the table at the gap-center preset
    and sweep follower (x, y); panels = follower yaw 0° vs 90°.
    Reads as "what's the follower's expected return from (x, y) given
    the table is at the gap?"

    ``by="table"``: fix the follower at the middle of the upper half
    facing the goal and sweep table (x, y); panels = table yaw 0° vs
    90°. Reads as "where can the table be so the follower (parked in
    the upper half) still expects to reach the goal?"
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import torch as th
    from stable_baselines3 import PPO

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.household.initial_conditions import (
        fixed_table_position,
    )
    from critic_sharing.scenarios.household.render import (
        draw_arena, draw_dirt, draw_follower, draw_goal_line,
        draw_table,
    )

    cfg_nav = load_env_config("household", "follower")
    cfg_clean = load_env_config("household", "leader")
    model = PPO.load(str(model_path))

    xs = np.linspace(cfg_nav.world_x_min, cfg_nav.world_x_max,
                     grid_size, dtype=np.float32)
    ys = np.linspace(cfg_nav.world_y_min, cfg_nav.world_y_max,
                     grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")  # (G, G)
    n_pts = grid_size * grid_size
    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)

    obs = np.zeros((n_pts, 8), dtype=np.float32)

    if by == "follower":
        table_pos, _ = fixed_table_position("gap-center", cfg_clean, cfg_nav)
        table_theta = 0.0
        # Vary follower (x, y); panels sweep follower yaw.
        sweep = [
            (r"$\theta_f = 0°$ (long side along x)",  0.0),
            (r"$\theta_f = 90°$ (long side along y)", float(np.pi / 2)),
        ]
        suptitle = "V_follower over (f_x, f_y) — table fixed at gap-center"
        overlay_fixed = (table_pos, table_theta, "table")
    elif by == "table":
        # Fix the follower in the middle of the upper half, facing the goal.
        upper_mid_y = 0.5 * (max(0.0, cfg_nav.wall_y) + cfg_nav.world_y_max)
        follower_pos = np.array([0.0, float(upper_mid_y)], dtype=np.float32)
        follower_theta = float(-np.pi / 2)  # heading along −y (toward goal)
        # Vary table (x, y); panels sweep table yaw.
        sweep = [
            (r"$\theta_t = 0°$  (long side along x)",  0.0),
            (r"$\theta_t = 90°$ (long side along y)", float(np.pi / 2)),
        ]
        suptitle = (
            "V_follower over (t_x, t_y) — follower fixed at upper-half "
            "centre facing the goal"
        )
        overlay_fixed = (follower_pos, follower_theta, "follower")
    else:
        raise typer.BadParameter(
            f"--by must be 'follower' or 'table' (got {by!r})"
        )

    panels = []
    for label, yaw in sweep:
        sin_v, cos_v = float(np.sin(yaw)), float(np.cos(yaw))
        if by == "follower":
            # Follower varies; table is fixed.
            sin_t, cos_t = float(np.sin(table_theta)), float(np.cos(table_theta))
            obs[:, 0] = flat_x
            obs[:, 1] = flat_y
            obs[:, 2] = sin_v
            obs[:, 3] = cos_v
            obs[:, 4] = float(table_pos[0])
            obs[:, 5] = float(table_pos[1])
            obs[:, 6] = sin_t
            obs[:, 7] = cos_t
        else:
            # Table varies; follower is fixed.
            sin_f = float(np.sin(follower_theta))
            cos_f = float(np.cos(follower_theta))
            obs[:, 0] = float(follower_pos[0])
            obs[:, 1] = float(follower_pos[1])
            obs[:, 2] = sin_f
            obs[:, 3] = cos_f
            obs[:, 4] = flat_x
            obs[:, 5] = flat_y
            obs[:, 6] = sin_v
            obs[:, 7] = cos_v
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with th.no_grad():
            values = model.policy.predict_values(obs_tensor)
        v = values.detach().cpu().numpy().reshape(grid_size, grid_size)
        panels.append((label, v, yaw))

    vmin = min(v.min() for _, v, _ in panels)
    vmax = max(v.max() for _, v, _ in panels)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.0), sharex=True, sharey=True)
    last_im = None
    for ax, (label, v, panel_yaw) in zip(axes, panels):
        draw_arena(ax, cfg_nav)
        last_im = ax.imshow(
            v,
            extent=(cfg_nav.world_x_min, cfg_nav.world_x_max,
                    cfg_nav.world_y_min, cfg_nav.world_y_max),
            origin="lower", cmap="viridis", aspect="equal",
            vmin=float(vmin), vmax=float(vmax), zorder=0,
        )
        draw_goal_line(ax, cfg_nav)
        fixed_pos, fixed_yaw, fixed_role = overlay_fixed
        if by == "follower":
            # Table pose is fixed across panels (gap-center).
            draw_table(ax, fixed_pos, fixed_yaw,
                       tuple(cfg_clean.table_half_extents))
            draw_dirt(ax, fixed_pos)
        else:
            # Follower pose is fixed (upper-half centre); table varies
            # so the overlay is just an outline at the panel's yaw at
            # the arena centre to show what "θ_t" means visually.
            draw_follower(ax, fixed_pos, fixed_yaw,
                          tuple(cfg_nav.follower_half_extents))
        ax.set_title(label, fontsize=10)

    fig.suptitle(suptitle, fontsize=11)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.018, 0.60])
    fig.colorbar(last_im, cax=cbar_ax, label="V_follower")

    if output is None:
        output = Path("value_household_by_{by}.png".format(by=by))
    if show:
        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
        typer.echo("Saved to {p}".format(p=output))
        plt.close(fig)


# ---------------------------------------------------------------------------
# visualize ood
# ---------------------------------------------------------------------------


@app.command("ood")
def visualize_ood(
    scenario: Scenario = typer.Option(..., help="Scenario."),
    model: Optional[Path] = typer.Option(
        None, help="Path to trained follower model (.zip). Defaults to latest artifact."
    ),
    vae: Optional[Path] = typer.Option(
        None,
        help=(
            "Path to VAE checkpoint (.pt). Defaults to follower_vae.pt next "
            "to the follower model (or legacy nav_env_vae.pt for obstacle_arena)."
        ),
    ),
    grid_size: int = typer.Option(80, help="Resolution of the OOD heatmap."),
    extent: float = typer.Option(
        5.0,
        help="Half-width of the (object/obstacle) grid in world or base-frame units.",
    ),
    raw_loss: bool = typer.Option(
        False,
        "--raw-loss",
        help=(
            "Plot raw VAE reconstruction loss instead of the normalized "
            "[0, 1] OOD metric. Useful for VAE-quality debugging."
        ),
    ),
    with_value: bool = typer.Option(
        False,
        "--with-value",
        help=(
            "Plot V_F and OOD side-by-side on the same grid. Two panels "
            "(or 2×4 for mobile_arm)."
        ),
    ),
    output: Optional[Path] = typer.Option(
        None, help="Output path (default: ood_{scenario}.png).",
    ),
    show: bool = typer.Option(False, help="Display interactively instead of saving."),
) -> None:
    """Plot the OOD metric over the shared state space.

    Mirrors :func:`visualize_value`'s per-scenario layout — single panel
    for obstacle_arena and warehouse (2-D obstacle / object position
    space), 2×2 panels for mobile_arm with θ_1 varied across panels.

    The metric is queried via each scenario's ``ood_query.py`` module,
    which slices the leader-shape obs to whatever feature dimension the
    VAE was trained on (so this works for any scenario that has a
    ``ood_query.py``).
    """
    from critic_sharing.cli.eval._common import _resolve_vae

    follower_path = resolve_model(model, scenario.value)
    vae_path = _resolve_vae(vae, follower_path, scenario.value)
    if vae_path is None:
        typer.echo(
            "No VAE found. Run 'critic-sharing train vae --scenario {s}' "
            "first, or pass --vae <path>.".format(s=scenario.value),
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(
        "Visualising OOD: scenario={s} model={m} vae={v}".format(
            s=scenario.value, m=follower_path, v=vae_path,
        )
    )

    if scenario == Scenario.obstacle_arena:
        _visualize_ood_obstacle_arena(
            vae_path=vae_path, grid_size=grid_size, extent=extent,
            raw_loss=raw_loss, with_value=with_value,
            follower_path=follower_path, output=output, show=show,
        )
    elif scenario == Scenario.warehouse:
        _visualize_ood_warehouse(
            vae_path=vae_path, grid_size=grid_size, extent=extent,
            raw_loss=raw_loss, with_value=with_value,
            follower_path=follower_path, output=output, show=show,
        )
    elif scenario == Scenario.mobile_arm:
        _visualize_ood_mobile_arm(
            vae_path=vae_path, grid_size=grid_size, extent=extent,
            raw_loss=raw_loss, with_value=with_value,
            follower_path=follower_path, output=output, show=show,
        )
    else:
        typer.echo(
            "visualize ood for {s}: not yet implemented.".format(s=scenario.value),
            err=True,
        )
        raise typer.Exit(code=1)


def _build_ood_grid_2d(
    extent: float, grid_size: int,
) -> tuple:
    """Build a 2-D grid over [-extent, +extent]^2 — shared across scenarios.

    Returns ``(xs, ys, grid_x, grid_y, positions)`` where ``positions``
    is the flat ``(grid_size**2, 2)`` array.
    """
    import numpy as np

    xs = np.linspace(-float(extent), float(extent), int(grid_size), dtype=np.float32)
    ys = np.linspace(-float(extent), float(extent), int(grid_size), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return xs, ys, grid_x, grid_y, positions


def _save_or_show(fig, scenario, output, show):
    """Common save/show tail used by all three OOD visualisers."""
    import matplotlib.pyplot as plt

    if show:
        plt.show()
    else:
        out = output if output is not None else _default_output("ood", scenario, ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        typer.echo("Saved to {p}".format(p=out))
        plt.close(fig)


def _visualize_ood_obstacle_arena(
    vae_path: Path,
    grid_size: int,
    extent: float,
    raw_loss: bool,
    with_value: bool,
    follower_path: Path,
    output: Optional[Path],
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from critic_sharing.scenarios.obstacle_arena.ood_query import (
        LEADER_OBSTACLE_X_IDX,
        LEADER_OBSTACLE_Y_IDX,
        build_obstacle_arena_ood_metric,
    )

    ood_fn = build_obstacle_arena_ood_metric(str(vae_path))

    xs, ys, grid_x, grid_y, positions = _build_ood_grid_2d(extent, grid_size)
    # obstacle_arena leader obs is (agent_x, agent_y, obstacle_x, obstacle_y).
    # We fix agent at origin and sweep obstacle position.
    leader_obs = np.zeros((positions.shape[0], 4), dtype=np.float32)
    leader_obs[:, LEADER_OBSTACLE_X_IDX] = positions[:, 0]
    leader_obs[:, LEADER_OBSTACLE_Y_IDX] = positions[:, 1]

    scores = ood_fn(leader_obs).reshape(int(grid_size), int(grid_size))
    title_suffix = "OOD score" if not raw_loss else "raw recon loss"
    if raw_loss:
        scores = _raw_recon_losses(vae_path, positions).reshape(int(grid_size), int(grid_size))

    if with_value:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        ax_ood, ax_value = axes
    else:
        fig, ax_ood = plt.subplots(figsize=(7, 6))
        ax_value = None

    im = ax_ood.imshow(
        scores, extent=(-extent, extent, -extent, extent), origin="lower",
        cmap="magma", aspect="equal",
        vmin=None if raw_loss else 0.0, vmax=None if raw_loss else 1.0,
    )
    ax_ood.set_title("OOD: obstacle_arena ({s})".format(s=title_suffix))
    ax_ood.set_xlabel("obstacle_x"); ax_ood.set_ylabel("obstacle_y")
    fig.colorbar(im, ax=ax_ood, label=title_suffix)

    if ax_value is not None:
        _overlay_value_obstacle_arena(ax_value, follower_path, extent, grid_size)

    plt.tight_layout()
    _save_or_show(fig, Scenario.obstacle_arena, output, show)


def _visualize_ood_warehouse(
    vae_path: Path,
    grid_size: int,
    extent: float,
    raw_loss: bool,
    with_value: bool,
    follower_path: Path,
    output: Optional[Path],
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from critic_sharing.scenarios.warehouse.ood_query import (
        LEADER_OBJ_X_IDX, LEADER_OBJ_Y_IDX,
        build_warehouse_ood_metric,
    )

    ood_fn = build_warehouse_ood_metric(str(vae_path))

    xs, ys, grid_x, grid_y, positions = _build_ood_grid_2d(extent, grid_size)
    # warehouse leader obs: [agent_x, agent_y, sin, cos, obj_x, obj_y, carry].
    leader_obs = np.zeros((positions.shape[0], 7), dtype=np.float32)
    leader_obs[:, LEADER_OBJ_X_IDX] = positions[:, 0]
    leader_obs[:, LEADER_OBJ_Y_IDX] = positions[:, 1]

    scores = ood_fn(leader_obs).reshape(int(grid_size), int(grid_size))
    title_suffix = "OOD score" if not raw_loss else "raw recon loss"
    if raw_loss:
        scores = _raw_recon_losses(vae_path, positions).reshape(int(grid_size), int(grid_size))

    if with_value:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        ax_ood, ax_value = axes
    else:
        fig, ax_ood = plt.subplots(figsize=(7, 6))
        ax_value = None

    im = ax_ood.imshow(
        scores, extent=(-extent, extent, -extent, extent), origin="lower",
        cmap="magma", aspect="equal",
        vmin=None if raw_loss else 0.0, vmax=None if raw_loss else 1.0,
    )
    ax_ood.set_title("OOD: warehouse ({s})".format(s=title_suffix))
    ax_ood.set_xlabel("object_x"); ax_ood.set_ylabel("object_y")
    fig.colorbar(im, ax=ax_ood, label=title_suffix)

    if ax_value is not None:
        ax_value.set_title("V_F overlay not implemented for warehouse")
        ax_value.set_xticks([]); ax_value.set_yticks([])

    plt.tight_layout()
    _save_or_show(fig, Scenario.warehouse, output, show)


def _visualize_ood_mobile_arm(
    vae_path: Path,
    grid_size: int,
    extent: float,
    raw_loss: bool,
    with_value: bool,
    follower_path: Path,
    output: Optional[Path],
    show: bool,
) -> None:
    """OOD heatmap over (object_x_base, object_y_base) at θ_1 = 0 (arm forward).

    Mirrors :func:`_visualize_value_mobile_arm` exactly:

    * Single panel (no θ_1 sweep — the VAE only sees the 2-D
      object-position slice, so θ_1 doesn't change the score; rendering
      4 identical panels was a waste).
    * Plot orientation: panel-vertical = base x, panel-horizontal =
      base y, so the tray (mounted on base +x) points up.
    * Overlays: base footprint, tray marker, arm at θ_1 = 0.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.guidance.ood import (
        OODMetric, load_vae_checkpoint, load_vae_model,
        reconstruction_losses,
    )

    follower_cfg = load_env_config("mobile_arm", "follower")

    # ---- Load VAE + build metric ----
    checkpoint = load_vae_checkpoint(str(vae_path))
    threshold = float(checkpoint["threshold"])
    max_loss = float(checkpoint["max_loss"])
    vae = load_vae_model(str(vae_path))
    metric = OODMetric(vae, threshold=threshold, max_loss=max_loss)

    # ---- Build grid using the same indexing convention as V viz ----
    # ``indexing="ij"`` makes the first axis = base_x, second axis =
    # base_y. With imshow + origin="lower", row index becomes panel-y
    # and column index becomes panel-x — so panel-y = base_x and
    # panel-x = base_y. Tray points up. Same as visualize_value.
    xs = np.linspace(-extent, extent, grid_size, dtype=np.float32)  # base x range
    ys = np.linspace(-extent, extent, grid_size, dtype=np.float32)  # base y range
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    flat = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    if raw_loss:
        scores = reconstruction_losses(vae, flat).reshape(
            int(grid_size), int(grid_size),
        )
        cbar_label = "raw recon loss"
        vmin, vmax = None, None
    else:
        scores = np.asarray(metric.metric(flat), dtype=np.float32).reshape(
            int(grid_size), int(grid_size),
        )
        cbar_label = "OOD score"
        vmin, vmax = 0.0, 1.0

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.suptitle(
        "OOD: mobile_arm (object_pos in base frame, θ_1=0) — " + cbar_label,
        fontsize=11,
    )
    im = ax.imshow(
        scores,
        extent=(-extent, extent, -extent, extent),
        origin="lower",
        cmap="magma",
        aspect="equal",
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("y_base")
    ax.set_ylabel("x_base")

    # ---- Overlays: base footprint, tray, arm at θ_1=0 (matches V viz) ----
    side = float(follower_cfg.base_size)
    ax.add_patch(
        mpatches.Rectangle(
            (-side / 2, -side / 2), side, side,
            facecolor="#90A4AE", edgecolor="#37474F",
            linewidth=1.0, alpha=0.5, zorder=4,
        )
    )
    tray_x_panel = float(follower_cfg.tray_offset_y)  # = 0
    tray_y_panel = float(follower_cfg.tray_offset_x)  # = 0.25
    ax.plot(
        tray_x_panel, tray_y_panel,
        marker="s", markersize=10, color="#FB8C00",
        markeredgecolor="#E65100", zorder=6,
    )
    ax.text(
        tray_x_panel, tray_y_panel + 0.15,
        "tray", color="#E65100", ha="center", fontsize=8, zorder=6,
    )

    # Arm at θ_1=0: (cos0, sin0) = (1, 0) → straight up in panel coords.
    L1 = float(follower_cfg.L1)
    L2a = float(follower_cfg.L2a)
    L2b = float(follower_cfg.L2b)
    theta2 = float(follower_cfg.theta2_default)
    theta3 = float(follower_cfg.theta3_default)
    theta4 = float(follower_cfg.theta4_default)
    phi_upper = theta2
    phi_L2a = theta2 - theta3
    phi_L2b = theta2 - theta3 - theta4
    elbow_r = L1 * np.cos(phi_upper)
    wrist_r = elbow_r + L2a * np.cos(phi_L2a)
    ee_r = wrist_r + L2b * np.cos(phi_L2b)
    c1, s1 = 1.0, 0.0  # cos(0), sin(0)
    shoulder = (0.0, 0.0)
    elbow = (elbow_r * s1, elbow_r * c1)
    wrist = (wrist_r * s1, wrist_r * c1)
    ee = (ee_r * s1, ee_r * c1)
    ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]],
            color="#1565C0", linewidth=2.5, zorder=7)
    ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]],
            color="#0D47A1", linewidth=2.5, zorder=7)
    ax.plot([wrist[0], ee[0]], [wrist[1], ee[1]],
            color="#0D47A1", linewidth=2.5, alpha=0.85, zorder=7)
    ax.plot(*ee, marker="*", markersize=10, color="#43A047",
            markeredgecolor="#1B5E20", zorder=8)

    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)

    _save_or_show(fig, Scenario.mobile_arm, output, show)


def _raw_recon_losses(vae_path, positions):
    """Compute raw reconstruction losses on a flat (N, feature_dim) array."""
    from critic_sharing.guidance.ood import load_vae_model, reconstruction_losses

    vae = load_vae_model(str(vae_path))
    return reconstruction_losses(vae, positions)


def _overlay_value_obstacle_arena(ax_value, follower_path, extent, grid_size):
    """Side-panel V_F overlay for obstacle_arena `--with-value` mode.

    Reuses the existing ``visualize value`` machinery's call into
    `value_query.build_nav_value_approximator`. Falls back to a "not
    available" panel if the dependency chain isn't satisfied.
    """
    import numpy as np
    from stable_baselines3 import PPO

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.obstacle_arena.value_query import (
        build_nav_value_approximator, load_nav_value_approximator,
    )

    nav_cfg = load_env_config("obstacle_arena", "follower")
    # Try the pre-distilled approximator first; fall back to building one.
    candidate = follower_path.parent / "nav_value_approx.pt"
    if candidate.exists():
        value_fn = load_nav_value_approximator(str(candidate))
    else:
        nav_model = PPO.load(str(follower_path))
        from critic_sharing.config.resolver import load_guidance_config

        guidance = load_guidance_config("obstacle_arena")
        value_fn = build_nav_value_approximator(
            nav_model=nav_model, nav_config=nav_cfg,
            obstacle_samples=guidance.distillation.obstacle_samples,
            agent_grid_size=guidance.distillation.agent_grid_size,
            epochs=guidance.distillation.epochs,
            batch_size=guidance.distillation.batch_size,
            lr=guidance.distillation.lr,
        )

    xs, ys, _, _, positions = _build_ood_grid_2d(extent, grid_size)
    # Synthesize leader-shape obs (4-D) with agent at origin.
    obs = np.zeros((positions.shape[0], 4), dtype=np.float32)
    obs[:, 2] = positions[:, 0]; obs[:, 3] = positions[:, 1]
    values = np.asarray(value_fn(obs), dtype=np.float32).reshape(int(grid_size), int(grid_size))

    im = ax_value.imshow(
        values, extent=(-extent, extent, -extent, extent), origin="lower",
        cmap="viridis", aspect="equal",
    )
    ax_value.set_title("V_F over obstacle space")
    ax_value.set_xlabel("obstacle_x"); ax_value.set_ylabel("obstacle_y")
    import matplotlib.pyplot as plt

    plt.colorbar(im, ax=ax_value, label="V_F")


# ---------------------------------------------------------------------------
# visualize heatmap  (obstacle-position utility, averaged over nav positions)
# ---------------------------------------------------------------------------


@app.command("heatmap")
def visualize_heatmap(
    scenario: Scenario = typer.Option(..., help="Scenario (only obstacle_arena supported)."),
    model: Optional[Path] = typer.Option(
        None, help="Path to trained follower model (.zip). Defaults to latest artifact."
    ),
    grid_size: int = typer.Option(40, help="Resolution of the obstacle-position grid."),
    nav_samples: int = typer.Option(
        200, help="Number of navigator positions averaged over per obstacle-grid point."
    ),
    nav_seed: int = typer.Option(0, help="RNG seed for navigator-position samples."),
    output: Optional[Path] = typer.Option(
        None, help="Output path (default: heatmap_{scenario}.png)."
    ),
    show: bool = typer.Option(False, help="Display interactively instead of saving."),
) -> None:
    """Heatmap of follower value over *obstacle* position space.

    For each point on a grid over valid obstacle positions the follower value
    V(nav_pos, obstacle_pos) is averaged over a sample of navigator positions,
    producing a 2-D surface that shows which obstacle placements are most
    (or least) favourable for the follower.
    """
    if scenario != Scenario.obstacle_arena:
        typer.echo("visualize heatmap is only implemented for obstacle_arena.", err=True)
        raise typer.Exit(code=1)

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from stable_baselines3 import PPO


    model_path = resolve_model(model, scenario.value)
    typer.echo(
        "Building obstacle heatmap: scenario={s} model={m} grid={g}x{g} nav_samples={n}".format(
            s=scenario.value, m=model_path, g=grid_size, n=nav_samples
        )
    )

    from critic_sharing.config.resolver import load_env_config

    nav_model = PPO.load(str(model_path))
    config = load_env_config("obstacle_arena", "follower")

    obs_x_min, obs_x_max, obs_y_min, obs_y_max = config.obstacle_bounds()

    # --- grid over obstacle positions ---
    obs_xs = np.linspace(obs_x_min, obs_x_max, grid_size, dtype=np.float32)
    obs_ys = np.linspace(obs_y_min, obs_y_max, grid_size, dtype=np.float32)
    grid_ox, grid_oy = np.meshgrid(obs_xs, obs_ys)  # (G, G)
    obs_positions = np.stack([grid_ox.ravel(), grid_oy.ravel()], axis=1)  # (G*G, 2)

    # --- random navigator positions to average over ---
    rng = np.random.default_rng(nav_seed)
    nav_pos = rng.uniform(config.world_min, config.world_max, size=(nav_samples, 2)).astype(
        np.float32
    )  # (N, 2)

    # --- build full observation matrix: (G*G * N, 4) ---
    n_obs = obs_positions.shape[0]  # G*G
    # repeat each obstacle position N times, repeat nav positions G*G times
    obs_rep = np.repeat(obs_positions, nav_samples, axis=0)  # (G*G*N, 2)
    nav_rep = np.tile(nav_pos, (n_obs, 1))  # (G*G*N, 2)
    full_obs = np.concatenate([nav_rep, obs_rep], axis=1)  # (G*G*N, 4)

    # --- evaluate value in one batched forward pass ---
    with torch.no_grad():
        obs_tensor = torch.as_tensor(full_obs, device=nav_model.policy.device)
        values = nav_model.policy.predict_values(obs_tensor).cpu().numpy().ravel()

    # --- average over navigator positions ---
    values_grid = (
        values.reshape(n_obs, nav_samples).mean(axis=1).reshape(grid_size, grid_size)
    )  # (G, G)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        values_grid,
        extent=(obs_x_min, obs_x_max, obs_y_min, obs_y_max),
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Mean follower value")
    ax.set_title(
        "Follower value over obstacle space\n(averaged over {n} navigator positions)".format(
            n=nav_samples
        )
    )
    ax.set_xlabel("obstacle_x")
    ax.set_ylabel("obstacle_y")
    plt.tight_layout()

    _save_or_show_fig(
        fig,
        output,
        show,
        default=_default_output("heatmap", scenario, ".png"),
    )


# ---------------------------------------------------------------------------
# visualize distilled-value
# ---------------------------------------------------------------------------


@app.command("distilled-value")
def visualize_distilled_value(
    scenario: Scenario = typer.Option(..., help="Scenario (only obstacle_arena supported)."),
    model: Optional[Path] = typer.Option(
        None, help="Path to trained follower model (.zip). Defaults to latest artifact."
    ),
    distilled: Optional[Path] = typer.Option(
        None,
        help=(
            "Path to distilled value MLP (.pt). "
            "If omitted, looks for nav_value_approx.pt next to the follower model."
        ),
    ),
    grid_size: int = typer.Option(80, help="Resolution of the heatmap grid."),
    output: Optional[Path] = typer.Option(
        None, help="Output path (default: distilled_value_{scenario}.png)."
    ),
    show: bool = typer.Option(False, help="Display interactively instead of saving."),
) -> None:
    """Heatmap of the distilled follower value function over obstacle position space.

    Unlike 'heatmap' (which queries the full policy and averages over nav
    positions), this command loads the pre-distilled MLP that maps
    obstacle_pos -> averaged value directly.  Much faster and shows exactly
    what the CEM planner sees during online adaptation.
    """
    if scenario != Scenario.obstacle_arena:
        typer.echo(
            "visualize distilled-value is only implemented for obstacle_arena.",
            err=True,
        )
        raise typer.Exit(code=1)

    import matplotlib.pyplot as plt
    import numpy as np

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.obstacle_arena.common import draw_wall
    from critic_sharing.scenarios.obstacle_arena.value_query import load_nav_value_approximator

    # Resolve distilled path
    if distilled is None:
        model_path = resolve_model(model, scenario.value)
        candidate = model_path.parent / "nav_value_approx.pt"
        if candidate.exists():
            distilled = candidate
        else:
            typer.echo(
                "No nav_value_approx.pt found next to {m}. "
                "Pass --distilled explicitly or train with latest code to generate it.".format(
                    m=model_path
                ),
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        model_path = model  # may be None, that's fine

    typer.echo("Loading distilled value function from {p}".format(p=distilled))
    value_approx = load_nav_value_approximator(str(distilled))
    nav_cfg = load_env_config("obstacle_arena", "follower")

    # Build grid over obstacle position space
    obs_x_min, obs_x_max, obs_y_min, obs_y_max = nav_cfg.obstacle_bounds()
    xs = np.linspace(obs_x_min, obs_x_max, grid_size, dtype=np.float32)
    ys = np.linspace(obs_y_min, obs_y_max, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (G*G, 2)

    # Evaluate distilled value function
    values = value_approx(positions)  # (G*G,)
    values_grid = np.asarray(values, dtype=np.float32).reshape(grid_size, grid_size)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        values_grid,
        extent=(obs_x_min, obs_x_max, obs_y_min, obs_y_max),
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Distilled follower value")

    # Draw wall and goal
    draw_wall(ax, nav_cfg)
    goal_y = nav_cfg.world_min + nav_cfg.agent_radius
    ax.axhline(y=goal_y, color="#43A047", linewidth=2, alpha=0.5, linestyle="--")

    # Mark gap
    half_gap = nav_cfg.gap_width / 2.0
    ax.plot(
        [nav_cfg.gap_x - half_gap, nav_cfg.gap_x + half_gap],
        [nav_cfg.wall_y, nav_cfg.wall_y],
        color="#43A047",
        linewidth=3,
        alpha=0.7,
    )

    ax.set_title("Distilled follower value V(obstacle_pos)")
    ax.set_xlabel("obstacle_x")
    ax.set_ylabel("obstacle_y")
    plt.tight_layout()

    _save_or_show_fig(
        fig,
        output,
        show,
        default=_default_output("distilled_value", scenario, ".png"),
    )

