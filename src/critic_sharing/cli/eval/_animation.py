"""MP4 generation for eval episodes.

Each scenario gets a :class:`FigureWorker` subclass that owns one render
call (one episode → one ``.mp4``). The top-level ``_save_*_episode_mp4s``
function builds a list of :class:`FigureTask` payloads and hands them to
:func:`run_figures_parallel`, so multi-process rendering happens for free
when the user passes ``--workers > 1`` to ``eval``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import typer

from critic_sharing.utils.parallel import (
    FigureTask,
    FigureWorker,
    run_figures_parallel,
)

from ._common import Scenario


def _save_episode_mp4s(
    all_results: list[dict],
    output_dir: Path,
    scenario: Scenario,
    fps: int = 4,
    max_workers: int = 1,
) -> None:
    """Save multi-panel MP4 animations for each episode.

    Dispatches to scenario-specific renderer. ``max_workers`` controls the
    figure pool size (mirrors the rollout pool's ``--workers`` flag).
    """
    if scenario == Scenario.obstacle_arena:
        _save_oa_episode_mp4s(all_results, output_dir, fps=fps, max_workers=max_workers)
    elif scenario in (
        Scenario.warehouse,
        Scenario.warehouse_pickup_blocked,
        Scenario.warehouse_pickup_unseen,
    ):
        from critic_sharing.cli.eval.warehouse import _save_wh_episode_mp4s

        _save_wh_episode_mp4s(
            all_results, output_dir, fps=fps, max_workers=max_workers,
            scenario_name=scenario.value,
        )
    elif scenario == Scenario.mobile_arm:
        from critic_sharing.cli.eval.mobile_arm import _save_ma_episode_mp4s

        _save_ma_episode_mp4s(all_results, output_dir, fps=fps, max_workers=max_workers)
    elif scenario == Scenario.household:
        _save_hh_episode_mp4s(all_results, output_dir, fps=fps, max_workers=max_workers)


_OA_COLORS = {"leader": "#E65100", "follower": "#1565C0"}


def _save_anim(anim, path, fps):
    """Shared helper: dispatch on file extension to PillowWriter or FFMpegWriter."""
    from matplotlib import animation as mpl_animation

    suffix = path.suffix.lower()
    if suffix == ".gif":
        writer = mpl_animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        writer = mpl_animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError("Unsupported extension: {s}".format(s=suffix))
    anim.save(str(path), writer=writer)


class OAEpisodeFigureWorker(FigureWorker):
    """Renders one obstacle_arena multi-panel MP4 per task.

    The task payload is a dict with ``episode``, ``mode_results``, and a
    pre-built per-mode frame list (so the worker doesn't need to re-walk
    raw trajectories). Configs are loaded once per worker in :meth:`setup`.
    """

    def setup(self) -> None:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive, thread-safe Agg backend
        from critic_sharing.config.resolver import load_env_config

        self._fps = int(self._kwargs.get("fps", 4))
        self._push_cfg = load_env_config("obstacle_arena", "leader")
        self._nav_cfg = load_env_config("obstacle_arena", "follower")

    def render(self, task: FigureTask) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation

        from critic_sharing.scenarios.obstacle_arena.common import draw_wall

        payload: dict[str, Any] = task.payload  # type: ignore[assignment]
        modes: list[str] = payload["modes"]
        mode_leader_frames: dict[str, list[dict]] = payload["leader_frames"]
        mode_follower_frames: dict[str, list[dict]] = payload["follower_frames"]
        max_leader_len: int = payload["max_leader_len"]
        total_frames: int = payload["total_frames"]
        n_modes = len(modes)
        if total_frames == 0 or n_modes == 0:
            return

        push_cfg = self._push_cfg
        nav_cfg = self._nav_cfg
        world_min = nav_cfg.world_min
        world_max = nav_cfg.world_max

        fig_width = max(4.0 * n_modes, 6.0)
        fig, axes = plt.subplots(1, n_modes, figsize=(fig_width, 5))
        if n_modes == 1:
            axes = [axes]

        obstacle_patches = []
        trail_lines = []
        agent_dots = []
        phase_texts = []

        for ax, mode in zip(axes, modes):
            ax.set_xlim(world_min, world_max)
            ax.set_ylim(world_min, world_max)
            ax.set_aspect("equal")
            draw_wall(ax, nav_cfg)
            goal_y = nav_cfg.world_min + nav_cfg.agent_radius
            ax.axhline(y=goal_y, color="#43A047", linewidth=2, alpha=0.5, zorder=2)
            ax.axhspan(nav_cfg.world_min, goal_y, color="#43A047", alpha=0.08, zorder=1)

            op = plt.Circle(
                (0, 0), push_cfg.obstacle_radius,
                color="#757575", alpha=0.5, zorder=3,
            )
            ax.add_patch(op)
            obstacle_patches.append(op)

            (tl,) = ax.plot([], [], "-", linewidth=1.2, alpha=0.6, zorder=2)
            trail_lines.append(tl)

            agent_r = push_cfg.agent_radius
            ap = plt.Circle((0, 0), agent_r, color="#1E88E5", alpha=0.7, zorder=5)
            ax.add_patch(ap)
            agent_dots.append(ap)

            pt = ax.text(
                world_min + 0.05 * (world_max - world_min),
                world_max - 0.05 * (world_max - world_min),
                "", fontsize=8, va="top",
            )
            phase_texts.append(pt)

        fig.tight_layout()

        def _update(global_i):
            artists = []
            for j, mode in enumerate(modes):
                l_frames = mode_leader_frames[mode]
                f_frames = mode_follower_frames[mode]
                patch = obstacle_patches[j]
                trail = trail_lines[j]
                dot = agent_dots[j]
                ptxt = phase_texts[j]

                if global_i < max_leader_len:
                    phase = "leader"
                    idx = min(global_i, len(l_frames) - 1)
                    f = l_frames[idx]
                    trail_src = l_frames[: idx + 1]
                else:
                    phase = "follower"
                    f_idx = global_i - max_leader_len
                    idx = min(f_idx, len(f_frames) - 1) if f_frames else 0
                    f = f_frames[idx] if f_frames else l_frames[-1]
                    trail_src = f_frames[: idx + 1] if f_frames else []

                color = _OA_COLORS[phase]
                patch.center = (f["obstacle_x"], f["obstacle_y"])
                trail.set_data(
                    [ff["agent_x"] for ff in trail_src],
                    [ff["agent_y"] for ff in trail_src],
                )
                trail.set_color(color)
                dot.center = (f["agent_x"], f["agent_y"])
                dot.set_facecolor(color)
                dot.set_edgecolor(color)
                ptxt.set_text(phase)
                ptxt.set_color(color)
                artists.extend([patch, trail, dot, ptxt])
            return artists

        anim = mpl_animation.FuncAnimation(
            fig, _update, frames=total_frames,
            interval=max(1, 1000 // self._fps), blit=True,
        )
        _save_anim(anim, task.output_path, fps=self._fps)
        plt.close(fig)


def _build_oa_payload(ep: int, mode_results: dict[str, dict]) -> dict[str, Any]:
    """Pre-walk raw trajectories into the per-frame dict format the worker
    expects. Done in the main process so workers receive small, picklable
    payloads (the trajectory tuples carry numpy arrays whose shapes vary)."""
    modes = sorted(mode_results.keys())
    mode_leader_frames: dict[str, list[dict]] = {}
    mode_follower_frames: dict[str, list[dict]] = {}
    max_leader_len = 0
    max_follower_len = 0

    for mode in modes:
        r = mode_results[mode]
        l_frames = []
        for agent_pos, obstacle_pos in r["leader_traj"]:
            l_frames.append({
                "agent_x": float(agent_pos[0]),
                "agent_y": float(agent_pos[1]),
                "obstacle_x": float(obstacle_pos[0]),
                "obstacle_y": float(obstacle_pos[1]),
            })
        f_frames = []
        for agent_pos, obstacle_pos in r["follower_traj"]:
            f_frames.append({
                "agent_x": float(agent_pos[0]),
                "agent_y": float(agent_pos[1]),
                "obstacle_x": float(obstacle_pos[0]),
                "obstacle_y": float(obstacle_pos[1]),
            })
        mode_leader_frames[mode] = l_frames
        mode_follower_frames[mode] = f_frames
        max_leader_len = max(max_leader_len, len(l_frames))
        max_follower_len = max(max_follower_len, len(f_frames))

    return {
        "episode": ep,
        "modes": modes,
        "leader_frames": mode_leader_frames,
        "follower_frames": mode_follower_frames,
        "max_leader_len": max_leader_len,
        "max_follower_len": max_follower_len,
        "total_frames": max_leader_len + max_follower_len,
    }


def _save_oa_episode_mp4s(
    all_results: list[dict],
    output_dir: Path,
    fps: int = 4,
    max_workers: int = 1,
) -> None:
    """Save multi-panel MP4 animations for obstacle_arena episodes.

    Builds one :class:`FigureTask` per episode and dispatches them through
    :func:`run_figures_parallel`.
    """
    mp4_dir = output_dir / "episodes"
    mp4_dir.mkdir(parents=True, exist_ok=True)

    recorded = [r for r in all_results if "leader_traj" in r and "follower_traj" in r]
    if not recorded:
        return

    episodes: dict[int, dict[str, dict]] = {}
    for r in recorded:
        ep = r["episode"]
        mode = r["mode"]
        if ep not in episodes:
            episodes[ep] = {}
        episodes[ep][mode] = r

    tasks: list[FigureTask] = []
    for ep, mode_results in sorted(episodes.items()):
        if not mode_results:
            continue
        out_path = mp4_dir / "ep{ep:03d}.mp4".format(ep=ep)
        tasks.append(
            FigureTask(output_path=out_path, payload=_build_oa_payload(ep, mode_results)),
        )

    if not tasks:
        return

    typer.echo(
        "Saving {n} multi-panel episode MP4s to {d}/ ({w} workers)".format(
            n=len(tasks), d=mp4_dir, w=max(1, min(max_workers, len(tasks))),
        )
    )
    run_figures_parallel(
        tasks=tasks,
        worker_cls=OAEpisodeFigureWorker,
        worker_kwargs={"fps": int(fps)},
        max_workers=int(max_workers),
        progress_desc="OA episodes",
    )
    typer.echo("Episodes → {d}/".format(d=mp4_dir))


# ---------------------------------------------------------------------------
# Household MP4 renderer
# ---------------------------------------------------------------------------

class HHEpisodeFigureWorker(FigureWorker):
    """Renders one household multi-panel MP4 per task.

    Each panel is one mode. Phase splits cleaner-traj frames from
    follower-traj frames, mirroring the obstacle_arena worker. The
    cleaner is blue with a heading arrow; the table is brown with a
    long-axis stroke; the follower is green; the dirt is a red dot.
    """

    def setup(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from critic_sharing.config.resolver import load_env_config

        self._fps = int(self._kwargs.get("fps", 4))
        self._clean_cfg = load_env_config("household", "leader")
        self._nav_cfg = load_env_config("household", "follower")

    def render(self, task: FigureTask) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation

        from critic_sharing.scenarios.household.render import (
            draw_arena, draw_cleaner, draw_dirt, draw_follower,
            draw_goal_line, draw_table,
        )

        payload: dict[str, Any] = task.payload  # type: ignore[assignment]
        modes: list[str] = payload["modes"]
        mode_leader_frames: dict[str, list[dict]] = payload["leader_frames"]
        mode_follower_frames: dict[str, list[dict]] = payload["follower_frames"]
        mode_dirt: dict[str, list[float]] = payload["dirt_pos"]
        mode_final_table: dict[str, dict] = payload["final_table"]
        max_leader_len: int = payload["max_leader_len"]
        total_frames: int = payload["total_frames"]
        n_modes = len(modes)
        if total_frames == 0 or n_modes == 0:
            return

        clean_cfg = self._clean_cfg
        cleaner_he = tuple(clean_cfg.cleaner_half_extents)
        table_he = tuple(clean_cfg.table_half_extents)
        follower_he = tuple(clean_cfg.follower_half_extents)

        fig_width = max(4.5 * n_modes, 6.0)
        fig, axes = plt.subplots(1, n_modes, figsize=(fig_width, 5.5))
        if n_modes == 1:
            axes = [axes]

        def _update(global_i):
            artists: list = []
            for ax, mode in zip(axes, modes):
                ax.clear()
                draw_arena(ax, clean_cfg)
                draw_goal_line(ax, clean_cfg)

                l_frames = mode_leader_frames[mode]
                f_frames = mode_follower_frames[mode]
                dirt = mode_dirt[mode]
                final = mode_final_table[mode]

                if global_i < max_leader_len:
                    phase = "cleaner"
                    idx = min(global_i, len(l_frames) - 1)
                    fl = l_frames[idx]
                    draw_table(
                        ax,
                        np.asarray(fl["table_pos"]), float(fl["table_theta"]),
                        table_he,
                    )
                    draw_dirt(ax, np.asarray(dirt))
                    draw_cleaner(
                        ax,
                        np.asarray(fl["cleaner_pos"]), float(fl["cleaner_theta"]),
                        cleaner_he,
                    )
                else:
                    phase = "follower"
                    f_idx = global_i - max_leader_len
                    idx = min(f_idx, len(f_frames) - 1) if f_frames else 0
                    if f_frames:
                        ff = f_frames[idx]
                        draw_table(
                            ax,
                            np.asarray(ff["table_pos"]), float(ff["table_theta"]),
                            table_he,
                        )
                        draw_dirt(ax, np.asarray(dirt))
                        draw_follower(
                            ax,
                            np.asarray(ff["follower_pos"]), float(ff["follower_theta"]),
                            follower_he,
                        )
                    else:
                        # No follower trajectory recorded; freeze on final
                        # table pose.
                        draw_table(
                            ax,
                            np.asarray(final["table_pos"]), float(final["table_theta"]),
                            table_he,
                        )
                        draw_dirt(ax, np.asarray(dirt))
                ax.text(
                    clean_cfg.world_x_min + 0.05 * (clean_cfg.world_x_max - clean_cfg.world_x_min),
                    clean_cfg.world_y_max - 0.05 * (clean_cfg.world_y_max - clean_cfg.world_y_min),
                    phase, fontsize=8, va="top",
                    color="#0D47A1" if phase == "cleaner" else "#1B5E20",
                )
            return artists

        anim = mpl_animation.FuncAnimation(
            fig, _update, frames=total_frames,
            interval=max(1, 1000 // self._fps), blit=False,
        )
        _save_anim(anim, task.output_path, fps=self._fps)
        plt.close(fig)


def _build_hh_payload(ep: int, mode_results: dict[str, dict]) -> dict[str, Any]:
    """Flatten leader/follower trajectories into per-frame dicts.

    Each frame dict carries the cleaner / follower pose plus the table
    pose at that instant (so the renderer doesn't have to interpolate).
    """

    modes = sorted(mode_results.keys())
    mode_leader_frames: dict[str, list[dict]] = {}
    mode_follower_frames: dict[str, list[dict]] = {}
    mode_dirt: dict[str, list[float]] = {}
    mode_final_table: dict[str, dict] = {}
    max_leader_len = 0
    max_follower_len = 0

    for mode in modes:
        r = mode_results[mode]
        l_frames: list[dict] = []
        for frame in r.get("leader_traj", []):
            sin_c, cos_c = frame["cleaner_theta_sin_cos"]
            sin_t, cos_t = frame["table_theta_sin_cos"]
            l_frames.append({
                "cleaner_pos": [float(frame["cleaner_pos"][0]), float(frame["cleaner_pos"][1])],
                "cleaner_theta": float(np.arctan2(sin_c, cos_c)),
                "table_pos": [float(frame["table_pos"][0]), float(frame["table_pos"][1])],
                "table_theta": float(np.arctan2(sin_t, cos_t)),
            })
        f_frames: list[dict] = []
        for frame in r.get("follower_traj", []):
            sin_f, cos_f = frame["follower_theta_sin_cos"]
            sin_t, cos_t = frame["table_theta_sin_cos"]
            f_frames.append({
                "follower_pos": [float(frame["follower_pos"][0]), float(frame["follower_pos"][1])],
                "follower_theta": float(np.arctan2(sin_f, cos_f)),
                "table_pos": [float(frame["table_pos"][0]), float(frame["table_pos"][1])],
                "table_theta": float(np.arctan2(sin_t, cos_t)),
            })
        mode_leader_frames[mode] = l_frames
        mode_follower_frames[mode] = f_frames
        mode_dirt[mode] = (
            [float(l_frames[0]["table_pos"][0]), float(l_frames[0]["table_pos"][1])]
            if l_frames else [0.0, 0.0]
        )
        ft = r.get("final_table_pos")
        ft_th = r.get("final_table_theta", 0.0)
        mode_final_table[mode] = {
            "table_pos": [float(ft[0]), float(ft[1])] if ft is not None else [0.0, 0.0],
            "table_theta": float(ft_th),
        }
        max_leader_len = max(max_leader_len, len(l_frames))
        max_follower_len = max(max_follower_len, len(f_frames))

    return {
        "episode": ep,
        "modes": modes,
        "leader_frames": mode_leader_frames,
        "follower_frames": mode_follower_frames,
        "dirt_pos": mode_dirt,
        "final_table": mode_final_table,
        "max_leader_len": max_leader_len,
        "max_follower_len": max_follower_len,
        "total_frames": max_leader_len + max_follower_len,
    }


def _save_hh_episode_mp4s(
    all_results: list[dict],
    output_dir: Path,
    fps: int = 4,
    max_workers: int = 1,
) -> None:
    """Save multi-panel household episode MP4s."""

    mp4_dir = output_dir / "episodes"
    mp4_dir.mkdir(parents=True, exist_ok=True)

    recorded = [r for r in all_results if "leader_traj" in r and "follower_traj" in r]
    if not recorded:
        return

    episodes: dict[int, dict[str, dict]] = {}
    for r in recorded:
        ep = r["episode"]
        mode = r["mode"]
        episodes.setdefault(ep, {})[mode] = r

    tasks: list[FigureTask] = []
    for ep, mode_results in sorted(episodes.items()):
        if not mode_results:
            continue
        out_path = mp4_dir / "ep{ep:03d}.mp4".format(ep=ep)
        tasks.append(
            FigureTask(output_path=out_path, payload=_build_hh_payload(ep, mode_results)),
        )

    if not tasks:
        return

    typer.echo(
        "Saving {n} household episode MP4s to {d}/ ({w} workers)".format(
            n=len(tasks), d=mp4_dir, w=max(1, min(max_workers, len(tasks))),
        )
    )
    run_figures_parallel(
        tasks=tasks,
        worker_cls=HHEpisodeFigureWorker,
        worker_kwargs={"fps": int(fps)},
        max_workers=int(max_workers),
        progress_desc="HH episodes",
    )
    typer.echo("Episodes → {d}/".format(d=mp4_dir))
