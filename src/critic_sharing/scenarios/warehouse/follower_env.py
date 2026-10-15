"""Follower env for the warehouse scenario.

The follower operates in the upper half of the arena (``y > wall_y``) and
weaves between three vertical shelves. At episode start an object is
pre-placed inside one of the 3 gap rectangles (the leader's would-be
dropoff). The follower must navigate to the gap, pick the object up, route
back through the shelves, and drop the object within ``goal_tolerance`` of
the per-episode goal point on a shelf face.

**Pickup/drop are automatic**: at the end of each step, the env auto-picks
up if the object is inside the pickup cone and not already carried, and
auto-drops on the goal whenever the carrying agent's would-be drop point
lands within ``goal_tolerance`` of the goal. The agent only learns
navigation.

Reward (shaped):
  * ``-step_penalty`` per step.
  * ``progress_weight * (prev_d - d)`` shaping toward the object (when not
    carrying) or toward the goal (when carrying).
  * ``+pickup_reward`` when the object is picked up.
  * ``+goal_reward`` and terminate when auto-dropped within
    ``goal_tolerance`` of the goal.
  * ``-collision_penalty`` per step when the agent attempted to enter a
    shelf (was clamped out).

Observation (9-D):
  ``[agent_x, agent_y, sin(theta), cos(theta), object_x, object_y,
     carry_flag, goal_x, goal_y]``
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from critic_sharing.config.schema import WarehouseFollowerEnvConfig as EnvConfig
from critic_sharing.scenarios.warehouse.common import (
    agent_bounds,
    agent_inscribed_radius,
    clamp_drop_position,
    drop_position,
    enforce_agent_object_collision,
    enforce_shelves,
    enforce_wall,
    in_pickup_cone,
    sample_goal_on_shelf_face,
    shelf_rects,
)


GOAL_OFFSET_FROM_FACE = 0.1


class WarehouseFollowerEnv(gym.Env):
    # Per-step ``info`` key naming the bool that flips True on a successful
    # episode (see :class:`WarehouseLeaderEnv` for the pattern). The
    # follower terminates on `on_goal` so the existing flag is exactly the
    # success signal — no composition needed.
    SUCCESS_INFO_KEY = "on_goal"

    """Follower env for the warehouse scenario."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()

        big = np.array([
            self.config.world_max,
            self.config.world_max,
            1.0, 1.0,
            self.config.world_max,
            self.config.world_max,
            1.0,
            self.config.world_max,
            self.config.world_max,
        ], dtype=np.float32)
        small = np.array([
            self.config.world_min,
            self.config.world_min,
            -1.0, -1.0,
            self.config.world_min,
            self.config.world_min,
            0.0,
            self.config.world_min,
            self.config.world_min,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=small, high=big, dtype=np.float32)
        act_low = np.array([
            -self.config.max_speed,
            -self.config.max_speed,
            -self.config.max_yaw_rate,
        ], dtype=np.float32)
        act_high = np.array([
            self.config.max_speed,
            self.config.max_speed,
            self.config.max_yaw_rate,
        ], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        self._agent_pos = np.zeros(2, dtype=np.float32)
        self._agent_theta = 0.0
        self._object_pos = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._carry = False
        self._steps = 0
        # Live carry-step-penalty value. Defaults to the static config value
        # so non-training consumers (CEM, eval, tests) get production-time
        # behaviour. A training-time callback may override this via
        # :meth:`set_step_penalty_carrying`.
        self._current_step_penalty_carrying = float(self.config.step_penalty_carrying)

    def set_step_penalty_carrying(self, value: float) -> None:
        """Override the live carry-step penalty (training-time scheduling)."""
        self._current_step_penalty_carrying = float(value)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_object_pos_in_gap(self, gap_index: Optional[int]) -> np.ndarray:
        """Sample object position uniformly inside one of the eligible gaps.

        ``pickup_disabled_gap_indices`` removes gaps from the spawn pool: the
        geometry is unchanged but the follower's auto-pickup is disabled
        there (warehouse_pickup_blocked / warehouse_pickup_unseen at
        training time). The follower never sees objects in these gaps, so
        V_F is OOD there at deployment.
        """
        gap_xs_sorted = sorted(float(x) for x in self.config.gap_xs)
        excluded = set(
            int(i) for i in getattr(self.config, "pickup_disabled_gap_indices", ())
        )
        unblocked = [
            (i, gx) for i, gx in enumerate(gap_xs_sorted) if i not in excluded
        ]
        if not unblocked:
            raise RuntimeError(
                "pickup_disabled_gap_indices excludes every gap; nothing to "
                "spawn into."
            )
        if gap_index is None:
            chosen_pos = int(self.np_random.integers(0, len(unblocked)))
            gap_index, gx = unblocked[chosen_pos]
        else:
            if gap_index in excluded:
                raise ValueError(
                    "gap_index={i} is excluded (pickup_disabled={p})".format(
                        i=gap_index, p=tuple(sorted(excluded)),
                    )
                )
            gx = gap_xs_sorted[gap_index]
        half = float(self.config.gap_width) / 2.0
        x = float(self.np_random.uniform(gx - half, gx + half))
        y = float(self.config.wall_y)
        return np.array([x, y], dtype=np.float32)

    def _object_in_pickup_disabled_gap(self) -> bool:
        """Check if the object's current x-position falls inside one of the
        ``pickup_disabled_gap_indices`` gap bands. The follower's auto-
        pickup branch in ``step()`` consults this and silently skips when
        True. Geometry-aware: uses the same ``|x − gap_x| < gap_width/2``
        condition as ``_sample_object_pos_in_gap``.
        """
        disabled = getattr(self.config, "pickup_disabled_gap_indices", ())
        if not disabled:
            return False
        gap_xs_sorted = sorted(float(x) for x in self.config.gap_xs)
        half = float(self.config.gap_width) / 2.0
        for idx in disabled:
            i = int(idx)
            if 0 <= i < len(gap_xs_sorted):
                if abs(float(self._object_pos[0]) - gap_xs_sorted[i]) < half:
                    return True
        return False

    def _sample_agent_pos_in_upper_aisles(self) -> np.ndarray:
        """Reject-sample an agent position uniformly above the wall, outside shelves."""
        low_x, high_x = agent_bounds(self.config)
        r = agent_inscribed_radius(self.config)
        y_min = float(self.config.wall_y) + r + 1e-3
        y_max = float(self.config.world_max) - r
        rects = shelf_rects(self.config)
        for _ in range(1000):
            x = float(self.np_random.uniform(low_x, high_x))
            y = float(self.np_random.uniform(y_min, y_max))
            ok = True
            for xmin, xmax, ymin, ymax in rects:
                if (x > xmin - r) and (x < xmax + r) and (y > ymin - r) and (y < ymax + r):
                    ok = False
                    break
            if ok:
                return np.array([x, y], dtype=np.float32)
        # Fallback: agent at top corner.
        return np.array([low_x, y_max], dtype=np.float32)

    def _pose_near_goal(self) -> tuple[np.ndarray, float]:
        """Place the follower ``drop_distance`` away from the goal, facing it.

        Inferred direction: away from the nearest shelf (so the spawn is in
        the aisle the goal opens into). Heading points back at the goal so
        ``agent_pos + heading * drop_distance`` lands exactly on the goal —
        meaning auto-drop fires the moment the agent returns from the gap
        with the object.
        """
        gx, gy = float(self._goal[0]), float(self._goal[1])
        nearest_sx = min(self.config.shelf_xs, key=lambda sx: abs(float(sx) - gx))
        direction = 1.0 if gx > float(nearest_sx) else -1.0
        spawn = np.array(
            [gx + direction * float(self.config.drop_distance), gy],
            dtype=np.float32,
        )
        theta = float(np.pi) if direction > 0 else 0.0
        return spawn, theta

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self._carry:
            obj = self._agent_pos
        else:
            obj = self._object_pos
        return np.array(
            [
                self._agent_pos[0],
                self._agent_pos[1],
                float(np.sin(self._agent_theta)),
                float(np.cos(self._agent_theta)),
                obj[0],
                obj[1],
                1.0 if self._carry else 0.0,
                self._goal[0],
                self._goal[1],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _shaping_distance(self) -> float:
        if self._carry:
            return float(np.linalg.norm(self._agent_pos - self._goal))
        return float(np.linalg.norm(self._agent_pos - self._object_pos))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        self._carry = False
        options = options or {}

        # Object: explicit position OR explicit gap_index OR uniform-over-3-gaps.
        if "object_pos" in options and options["object_pos"] is not None:
            self._object_pos = np.asarray(options["object_pos"], dtype=np.float32)
        else:
            gap_index = options.get("gap_index")
            self._object_pos = self._sample_object_pos_in_gap(gap_index)

        # Goal: explicit OR sample uniformly along left/right shelf faces.
        if "goal" in options and options["goal"] is not None:
            self._goal = np.asarray(options["goal"], dtype=np.float32)
        else:
            self._goal = sample_goal_on_shelf_face(
                self.np_random, self.config, offset=GOAL_OFFSET_FROM_FACE
            )

        # Follower spawn:
        #   1. explicit ``options['agent_pos']`` always wins
        #   2. else if ``config.spawn_near_goal`` is True (testing only), place
        #      the follower next to the goal facing it
        #   3. else sample uniformly between shelves in the upper half.
        agent_pos = options.get("agent_pos")
        agent_theta = options.get("agent_theta")
        if agent_pos is None:
            if self.config.spawn_near_goal:
                near_pos, near_theta = self._pose_near_goal()
                agent_pos = near_pos
                if agent_theta is None:
                    agent_theta = near_theta
            else:
                agent_pos = self._sample_agent_pos_in_upper_aisles()
        self._agent_pos = np.asarray(agent_pos, dtype=np.float32)

        if agent_theta is None:
            agent_theta = float(self.np_random.uniform(-np.pi, np.pi))
        self._agent_theta = float(agent_theta)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        prev_pos = self._agent_pos.copy()
        prev_d = self._shaping_distance()
        prev_carry = self._carry

        # Translation. Optional separate speed cap when carrying lets a
        # scenario harden the follower's carry leg so the leader's gap
        # choice (which sets the carry distance to the goal) becomes a
        # real planning lever. ``max_speed_carrying=None`` (default)
        # preserves the symmetric pre-2026-05-09 behaviour.
        vel = action[:2]
        speed = float(np.linalg.norm(vel))
        speed_cap = float(self.config.max_speed)
        msc = getattr(self.config, "max_speed_carrying", None)
        if self._carry and msc is not None:
            speed_cap = float(msc)
        if speed > speed_cap:
            vel = vel / (speed + 1e-8) * speed_cap
        next_pos = self._agent_pos + vel * self.config.dt
        low, high = agent_bounds(self.config)
        next_pos = np.clip(next_pos, low, high).astype(np.float32)
        next_pos = enforce_wall(
            next_pos, prev_pos, agent_inscribed_radius(self.config), self.config
        )
        next_pos, shelf_collided = enforce_shelves(
            next_pos, agent_inscribed_radius(self.config), self.config
        )
        # Agent ↔ object collision (only when not carrying — the object
        # follows the agent while carried).
        if not self._carry:
            next_pos = enforce_agent_object_collision(
                next_pos, self._object_pos,
                agent_inscribed_radius(self.config), self.config,
            )
        self._agent_pos = next_pos.astype(np.float32)

        # Rotation
        omega = float(np.clip(action[2], -self.config.max_yaw_rate, self.config.max_yaw_rate))
        self._agent_theta = float(((self._agent_theta + omega * self.config.dt + np.pi) % (2.0 * np.pi)) - np.pi)

        # Auto-pickup: object in cone, not already carrying. Silently skip
        # when the object is in a ``pickup_disabled_gap_indices`` gap —
        # this is the warehouse_pickup_blocked mechanism.
        picked_up = False
        if (
            not self._carry
            and not self._object_in_pickup_disabled_gap()
            and in_pickup_cone(
                self._agent_pos, self._agent_theta, self._object_pos, self.config,
            )
        ):
            self._carry = True
            picked_up = True

        # Auto-drop on goal: carrying AND drop_position is within goal_tolerance of goal.
        dropped = False
        on_goal = False
        if self._carry:
            drop_pt = drop_position(self._agent_pos, self._agent_theta, self.config)
            if np.linalg.norm(drop_pt - self._goal) <= float(self.config.goal_tolerance):
                drop_pt = clamp_drop_position(drop_pt, leader_side="above", config=self.config)
                self._object_pos = drop_pt.astype(np.float32)
                self._carry = False
                dropped = True
                on_goal = True

        # Reward — step penalty depends on carry state at start of step.
        # The carrying penalty uses the *live* value (may be temporarily
        # lowered by the training-time schedule callback).
        step_pen = (
            self._current_step_penalty_carrying if prev_carry else self.config.step_penalty
        )
        reward = -float(step_pen)
        new_d = self._shaping_distance()
        reward += float(self.config.progress_weight) * float(prev_d - new_d)
        if shelf_collided:
            reward -= float(self.config.collision_penalty)
        if picked_up:
            reward += float(self.config.pickup_reward)
        terminated = False
        if dropped:
            reward += float(self.config.goal_reward)
            terminated = True

        truncated = self._steps >= int(self.config.max_steps)
        info = {
            "picked_up": bool(picked_up),
            "dropped": bool(dropped),
            "on_goal": bool(on_goal),
            "carry": bool(self._carry),
            "shelf_collision": bool(shelf_collided),
            "steps": int(self._steps),
            "distance_to_goal": float(np.linalg.norm(self._object_pos - self._goal)),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


__all__ = ["EnvConfig", "WarehouseFollowerEnv", "GOAL_OFFSET_FROM_FACE"]
