"""Config schemas for all critic-sharing experiments.

Scenario-specific env configs (NavEnvConfig, PushEnvConfig,
WarehouseLeaderEnvConfig, etc.) are kept as separate dataclasses so
they can be composed or used standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Scenario environment configs
# ---------------------------------------------------------------------------


@dataclass
class BaseEnvConfig:
    dt: float = 0.1
    max_speed: float = 1.5
    world_min: float = -5.0
    world_max: float = 5.0
    obstacle_radius: float = 1.0
    max_steps: int = 250
    goal_reward: float = 25.0
    agent_radius: float = 0.25

    # Wall with gap
    wall_y: float = 0.0
    gap_x: float = 0.0
    gap_width: float = 2.0


@dataclass
class NavEnvConfig(BaseEnvConfig):
    """Follower (nav agent) environment config for obstacle_arena."""

    obstacle_x_min: Optional[float] = None
    obstacle_x_max: Optional[float] = None
    obstacle_y_min: Optional[float] = None
    obstacle_y_max: Optional[float] = None
    collision_penalty: float = 300.0
    progress_weight: float = 1.0
    distance_weight: float = 0.1

    def obstacle_bounds(self) -> Tuple[float, float, float, float]:
        default_min = self.world_min + self.obstacle_radius
        default_max = self.world_max - self.obstacle_radius
        obstacle_x_min = default_min if self.obstacle_x_min is None else self.obstacle_x_min
        obstacle_x_max = default_max if self.obstacle_x_max is None else self.obstacle_x_max
        obstacle_y_min = default_min if self.obstacle_y_min is None else self.obstacle_y_min
        obstacle_y_max = default_max if self.obstacle_y_max is None else self.obstacle_y_max
        if obstacle_x_min > obstacle_x_max or obstacle_y_min > obstacle_y_max:
            raise ValueError("Obstacle bounds must satisfy min <= max on both axes.")
        return obstacle_x_min, obstacle_x_max, obstacle_y_min, obstacle_y_max


@dataclass
class PushEnvConfig(BaseEnvConfig):
    """Leader (pusher agent) environment config for obstacle_arena."""

    push_strength: float = 1.0
    obstacle_radius: float = 0.8
    max_steps: int = 300


@dataclass
class _HouseholdBaseEnvConfig:
    """Shared world + rectangle geometry for the household leader+follower envs.

    Household = obstacle_arena reshaped: leader is a "cleaner" that
    picks up a piece of dirt frozen under a rectangular table; the
    follower still navigates to the lower boundary. All three rigid
    bodies (cleaner, table, follower) are rotated rectangles with
    explicit yaw, holonomic+yaw control ``(vx, vy, ω)``, and full SAT
    contact resolution between every pair.

    The wall + single-gap layout from obstacle_arena is reused but
    gated by ``wall_enabled``; with the default ``False`` the world is
    a single open box, useful for the long-table variant where the
    table itself is the primary spatial puzzle.
    """

    # World — per-axis bounds (real-metric arena: 2 m × 2.5 m).
    world_x_min: float = -1.0
    world_x_max: float = 1.0
    world_y_min: float = -1.25
    world_y_max: float = 1.25
    dt: float = 0.1
    max_steps: int = 300

    # Translational + rotational dynamics (real-metric).
    max_speed: float = 0.5             # m/s
    max_yaw_rate: float = 3.141592653589793  # π rad/s

    # Rectangle half-extents (real-metric, hx = heading half-length).
    cleaner_half_extents: Tuple[float, float] = (0.165, 0.12)
    follower_half_extents: Tuple[float, float] = (0.165, 0.12)
    table_half_extents: Tuple[float, float] = (0.31, 0.24)  # 0.62 × 0.48 m

    # Push response (cleaner → table). Quasi-static table: each step we
    # solve the SAT-MTV overlap and translate / rotate the table along
    # the contact normal + lever arm by these strengths.
    push_strength_lin: float = 1.0
    push_strength_rot: float = 1.0

    # Wall — 0.1 m thick strip at ``wall_y`` (y-coordinate of the wall
    # *centre*), with a single ``gap_width``-wide gap centred at
    # ``gap_x``. Modelled as two static SAT rectangles (left + right
    # segments) when ``wall_enabled``.
    wall_enabled: bool = True
    wall_y: float = 0.0
    wall_thickness: float = 0.1
    gap_x: float = 0.0
    gap_width: float = 0.36

    # Shared reward defaults.
    step_penalty: float = 0.01
    progress_weight: float = 3.0       # compensates for 3× weaker per-step shaping
    wall_collision_penalty: float = 0.1
    pickup_reward: float = 25.0
    goal_reward: float = 25.0


@dataclass
class HouseholdCleanerEnvConfig(_HouseholdBaseEnvConfig):
    """Leader env config for the household scenario."""

    max_steps: int = 200
    # After the dirt is picked up, the cleaner keeps running for this many
    # extra steps with no positive reward (penalties still apply) so it can
    # reposition the table to aid the follower. The episode terminates at
    # ``pickup_step + post_pickup_steps`` (capped at ``max_steps``). 0 = the
    # legacy behaviour (terminate immediately on pickup).
    post_pickup_steps: int = 50


@dataclass
class HouseholdNavEnvConfig(_HouseholdBaseEnvConfig):
    """Follower env config for the household scenario.

    The navigator's goal is the centered half of the lower boundary
    (``|x| <= goal_x_half_extent``, y at the bottom). During training the
    table is static-per-episode. Spawn distribution is full-arena uniform;
    deployment-time restriction to the upper half is applied externally at
    joint-eval time (see CLI evaluator).
    """

    max_steps: int = 250
    collision_penalty: float = 300.0
    distance_weight: float = 0.1
    # Half-width of the goal region on the lower boundary, centered on the
    # arena mid-x. ``None`` => (world_x_max - world_x_min) / 4, i.e. the goal
    # spans the centered *half* of the boundary. Reaching the goal requires
    # ``|x - mid_x| <= goal_x_half_extent`` and y at the bottom.
    goal_x_half_extent: Optional[float] = None


@dataclass
class _WarehouseBaseEnvConfig:
    """Shared world geometry for warehouse leader and follower envs.

    The world is a square ``[world_min, world_max]^2``. The horizontal wall at
    ``y = wall_y`` is agent-impassable except at three gaps (``gap_xs`` / ``gap_widths``)
    that are wide enough for the object to pass through but too narrow for the
    inscribed-circle agent to fit. Three vertical shelves in the upper half are
    fully untraversable rectangles centred at ``shelf_xs`` of width ``shelf_width``
    and y-extent ``[shelf_y_min, shelf_y_max]``.
    """

    # World
    world_min: float = -5.0
    world_max: float = 5.0
    dt: float = 0.1
    max_steps: int = 300

    # Translational + rotational dynamics
    max_speed: float = 1.5
    max_yaw_rate: float = 3.141592653589793  # π rad/s — full turn in 2 s

    # Agent square (collision treated as inscribed circle of r = side/2)
    agent_side: float = 0.6

    # Object circle
    object_radius: float = 0.2

    # Wall + 3 gaps (gaps double as dropoff zones)
    wall_y: float = 0.0
    gap_xs: Tuple[float, float, float] = (-3.0, 0.0, 3.0)
    gap_width: float = 0.55

    # 3 vertical shelves (upper half only)
    shelf_xs: Tuple[float, float, float] = (-2.0, 0.0, 2.0)
    shelf_width: float = 0.8
    shelf_y_min: float = 0.8
    shelf_y_max: float = 4.5

    # Pickup / drop geometry
    pickup_cone_half_angle: float = 1.0471975511965976  # π/3 = 60°
    pickup_max_distance: float = 0.9  # from agent center
    drop_distance: float = 0.55  # from agent center, in front
    interact_threshold: float = 0.5

    # Reward defaults shared by both envs.
    # ``step_penalty`` applies when the agent is *not* carrying at the start of
    # the step; ``step_penalty_carrying`` applies when it is. Make carrying
    # more expensive so the agents are incentivised to drop quickly.
    step_penalty: float = 0.01
    step_penalty_carrying: float = 0.05
    pickup_reward: float = 1.0
    goal_reward: float = 25.0
    bad_drop_penalty: float = 5.0
    progress_weight: float = 1.0


@dataclass
class WarehouseLeaderEnvConfig(_WarehouseBaseEnvConfig):
    """Leader env config for the warehouse scenario."""

    max_steps: int = 300


@dataclass
class WarehouseFollowerEnvConfig(_WarehouseBaseEnvConfig):
    """Follower env config for the warehouse scenario."""

    max_steps: int = 400
    goal_tolerance: float = 0.3
    collision_penalty: float = 0.5  # per-step shelf brush

    # Optional separate speed cap when the follower is carrying the
    # object. ``None`` (default) means use ``max_speed`` for both
    # carrying and empty motion, preserving pre-2026-05-09 behaviour.
    # Setting a value below ``max_speed`` (e.g., 1.0 vs the typical
    # warehouse ``max_speed=2.5``) makes the carry leg the bottleneck
    # so the leader's gap choice — which determines the carry distance
    # to the goal — becomes a real planning lever instead of a
    # near-saturated null.
    max_speed_carrying: Optional[float] = None

    # Testing-only: when True, the follower spawns ``drop_distance`` into the
    # aisle from the goal, already facing it. The follower must still descend
    # to the gap, pick up, and return — but the transport-to-goal leg is
    # trivial. Used to verify pickup + auto-drop mechanics without depending
    # on full upper-half navigation. Has no effect when ``options['agent_pos']``
    # is set explicitly at reset time.
    spawn_near_goal: bool = False

    # Indices into ``gap_xs`` where the follower's auto-pickup is silently
    # skipped. Geometry is unchanged — the gap exists physically and the
    # follower can navigate to objects there — but the env's ``step()``
    # auto-pickup branch refuses when the object lies in one of these
    # gaps' x-bands. Used by:
    #   * ``warehouse_pickup_blocked`` (set at training AND deployment;
    #     the leader can drop at the disabled gap and the follower will
    #     try, fail, and timeout — V_F overestimates the gap because it
    #     never saw a pickup outcome there during training).
    #   * ``warehouse_pickup_unseen`` (set only at training time via
    #     train_follower.yaml::env_overrides; deployment leaves the
    #     field empty so pickup is enabled — the follower's policy
    #     never trained on pickup-at-gap-i, so V_F's prediction is
    #     uncalibrated even though the env mechanic itself is normal).
    # When the field is non-empty, ``_sample_object_pos_in_gap`` skips these
    # gap indices so the training follower never sees objects there.
    pickup_disabled_gap_indices: tuple[int, ...] = ()


@dataclass
class _MobileArmBaseEnvConfig:
    """Shared world + arm geometry for the mobile_arm leader and follower envs.

    Two-agent scenario: a 2D mobile base (leader) drives near an object,
    parks, and triggers ``terminate``. A 3-DOF arm mounted on the base
    (follower) then picks the object up and drops it in a tray on the
    front of the base.

    The follower lives in **base frame** coordinates: the base is a frozen
    reference once the leader has parked. The shared value function
    transforms ``(leader_pose, object_world)`` into the follower's
    base-frame ``object_relative`` for V queries.
    """

    # World (leader 2D arena)
    world_min: float = -5.0
    world_max: float = 5.0
    dt: float = 0.1

    # Base (mobile chassis)
    base_size: float = 0.6              # square footprint side
    base_height: float = 0.4            # arm shoulder above ground

    # Arm (4-DOF planar after yaw rotation).
    #
    # Joint sign convention: ``θ_i`` is a *positive bend forward/down*. With
    # ``θ_2 = π/2`` the upper arm points straight up; positive ``θ_3`` /
    # ``θ_4`` bends each subsequent segment forward of its parent. Cumulative
    # angles from horizontal compound subtractively:
    #     φ_upper = θ_2
    #     φ_L2a   = θ_2 − θ_3
    #     φ_L2b   = θ_2 − θ_3 − θ_4
    L1: float = 0.5                     # upper-arm length
    L2a: float = 0.6                    # first lower-arm segment
    L2b: float = 0.6                    # second lower-arm segment

    theta1_low: float = -3.141592653589793     # turn (yaw) joint
    theta1_high: float = 3.141592653589793
    theta2_low: float = 0.0                    # shoulder pitch (0 = horizontal forward, π/2 = vertical)
    theta2_high: float = 1.5707963267948966    # π/2
    theta3_low: float = 0.0                    # elbow pitch (positive = bend forward/down)
    theta3_high: float = 3.141592653589793     # π
    theta4_low: float = 0.0                    # wrist pitch (positive = bend forward/down)
    theta4_high: float = 3.141592653589793     # π

    theta1_default: float = 0.0                    # 0 — arm forward, pointing at the tray
    theta2_default: float = 1.5707963267948966     # π/2 — upper arm vertical
    theta3_default: float = 0.7853981633974483     # π/4 — elbow bent forward 45°
    theta4_default: float = 0.7853981633974483     # π/4 — wrist bent forward 45°

    # Tray (in base frame)
    tray_offset_x: float = 0.25         # forward of base centre
    tray_offset_y: float = 0.0          # centred laterally
    tray_offset_z: float = 0.05         # just above the base top

    # Object
    object_radius: float = 0.05

    # Pickup / drop epsilons (3D Euclidean)
    eps_pickup: float = 0.10
    eps_tray: float = 0.10

    # Common reward defaults
    step_penalty: float = 0.01
    progress_weight: float = 1.0
    pickup_reward: float = 1.0
    goal_reward: float = 25.0


@dataclass
class MobileArmLeaderEnvConfig(_MobileArmBaseEnvConfig):
    """Leader env config for the mobile_arm scenario.

    The leader controls the mobile base in 2D and ends the episode by
    firing ``terminate``. Reward = stand-off shaping + Gaussian terminate
    bonus peaked at ``desired_distance``.
    """

    max_steps: int = 300
    max_speed: float = 1.5
    max_yaw_rate: float = 3.141592653589793
    interact_threshold: float = 0.5             # terminate ≥ 0.5 ends episode
    desired_distance: float = 1.0
    gaussian_sigma: float = 0.3                 # Gaussian terminate-bonus width
    leader_object_min_separation: float = 1.5   # leader/object spawn separation


@dataclass
class MobileArmFollowerEnvConfig(_MobileArmBaseEnvConfig):
    """Follower env config for the mobile_arm scenario.

    The follower controls the 3-DOF arm in base-frame coordinates. The
    base is static; the arm picks an object off the ground and drops it
    in the tray on the front of the base.
    """

    max_steps: int = 200
    max_joint_rate: float = 3.141592653589793   # rad/s per joint

    # Object spawn distribution (base frame, polar).
    # ``r_max`` extends past total arm reach (1.7m) so a chunk of training
    # episodes are out-of-reach — this is what teaches the V function to
    # assign low value to bad park poses (where the leader cannot see).
    object_r_min: float = 0.4
    object_r_max: float = 2.4

    # Angular bounds for the object spawn cone in base frame (radians).
    # ``φ = 0`` points along base +x (toward the tray), so the cone
    # ``[object_theta_low, object_theta_high]`` selects which directions
    # in front of the base get object spawns. Defaults to the full
    # circle ``[-π, π]`` (back-compat). Typically tuned per-training to
    # match θ_1's joint-limit reach so the follower is trained only on
    # objects the arm can actually swing to. Independent from θ_1's
    # joint limits in the schema — set them in lockstep via
    # ``train_follower.yaml`` if you want them coupled.
    object_theta_low: float = -3.141592653589793
    object_theta_high: float = 3.141592653589793

    # Per-step penalty when an attempted joint update would put a joint
    # endpoint inside the floor (z<0) or inside the base box. The update
    # is hard-reverted and this penalty is added to the step reward.
    collision_penalty: float = 0.1


# ---------------------------------------------------------------------------
# Shared infrastructure configs
# ---------------------------------------------------------------------------


@dataclass
class WandbConfig:
    project: str = "critic_sharing"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class RunConfig:
    id: Optional[str] = None


@dataclass
class PathsConfig:
    artifacts_root: str = "artifacts"
    save_path: Optional[str] = None


@dataclass
class VaeConfig:
    latent_dim: int = 2
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3


@dataclass
class PlannerConfig:
    """CEM planner schema defaults — production values live in per-scenario planner.yaml."""

    # CEM fields
    distribution: str = "gaussian"  # the only supported distribution
    horizon: int = 15
    num_samples: int = 64
    num_iters: int = 3
    noise_sigma: float = 0.5
    elite_frac: float = 0.2
    min_std: float = 0.05
    replan_interval: int = 1

    gamma: float = 0.99
    # ``"value"`` bootstraps with V_leader at the horizon-H terminal (one
    # extra value-function call per CEM candidate); ``"rollout"`` extends
    # each CEM candidate trajectory with the prior policy until episode
    # termination (more accurate, much more expensive — see
    # ``prior_extension_threads`` for the parallelization knob). Both ride
    # on the same calibration: leader_range is full-episode discounted
    # returns of successful naive episodes.
    terminal_value: str = "value"  # "value" | "rollout"
    num_rollouts: int = 1

    # Per-worker torch thread budget when ``terminal_value="rollout"``.
    # Default 1 keeps the run_episodes_parallel pool's thread-pinning
    # contract; bump to 2-4 when each rollout-mode worker has spare CPU
    # to amortize the prior-extension forward passes (set so that
    # ``num_workers * prior_extension_threads <= cpu_count``).
    prior_extension_threads: int = 1


# Backwards-compatible alias
CEMPlannerConfig = PlannerConfig


@dataclass
class OnlineAdaptationCostConfig:
    """Schema defaults — production values live in configs/planner/cost_online_adaptation.yaml."""

    # Value factor — multiplies V_F's contribution to the joint utility
    # cost. Applied REGARDLESS of whether an OOD detector is wired in
    # (the OOD trust factor is a separate, multiplicative attenuation).
    # The cost is:
    #
    #   utility = r_leader_norm + alpha * trust * r_follower_norm
    #
    # where ``trust = 1 - ood_score`` when OOD is on, else ``1``. Alpha
    # controls V_F's overall weight; the OOD trust factor only attenuates
    # V_F at states V_F shouldn't be trusted on.
    alpha: float = 4.0
    utility_type: str = "sum"  # "sum" | "sabotage"
    leader_gamma: float = 0.99

    # When True (default), each ``eval`` invocation runs a small
    # naive-policy calibration pass against the leader and follower envs'
    # natural reset distributions, computes ``(min, max)`` of the
    # successful-episode discounted returns, and passes those as
    # ``leader_range`` / ``follower_range`` to :class:`JointUtilityCost`
    # so the leader's full-episode return and the follower's V are mapped
    # into a common ``[0, 1]`` band before utility combination. Set to
    # ``False`` to keep the cost on raw scales (debugging, or scenarios
    # where the two sides are already on-scale by construction).
    normalize_returns: bool = True

    # When True (default), the leader/follower normalization clips
    # ``(r - lo) / (hi - lo)`` to the ``[0, 1]`` band — values outside
    # the calibration's successful-naive range are pinned. Set to
    # ``False`` to keep the unclipped affine form so plans that beat
    # the calibration ceiling get rewarded proportionally above 1.0
    # (and crashing plans get penalized below 0.0). The
    # :class:`JointUtilityCost` and the display-side ``_norm``
    # helpers in ``_compare_*`` honor this flag in lockstep so
    # plotted/tabled numbers always match what CEM optimized. The OOD
    # metric and trust-factor clips stay regardless — those are
    # structural bounds of the [0, 1] OOD score, not a calibration
    # choice.
    clip_normalized: bool = True

    # Number of naive-policy episodes used to calibrate
    # ``leader_range`` / ``follower_range``. Higher → tighter sample
    # bounds (less run-to-run jitter in the calibration ceiling). 30
    # was the original default; 100 is a safer default when the success
    # rate is ~50% and you want at least ~50 samples in each band.
    calibration_episodes: int = 100

    # When True, only successful naive episodes contribute to
    # ``(lo, hi)``. The band then represents "what naive achieves when
    # it succeeds" — narrow but discriminative; failed plans normalize
    # to large negative values (with ``clip_normalized=False``).
    # When False, ALL naive episodes contribute. The band spans
    # "naive's worst failure" to "naive's best success", so failed
    # plans normalize near 0 and successful ones near 1 — wider, more
    # graceful, and matches the planner's success/failure mental
    # model. Recommended for noisy V_F or asymmetric reward shaping
    # (e.g. mobile_arm with goal_reward >> step_penalty).
    calibrate_on_success_only: bool = True


# ---------------------------------------------------------------------------
# Training / guidance configs
# ---------------------------------------------------------------------------


@dataclass
class AgentTrainingConfig:
    """Shared training hyperparameters for both leader and follower."""

    total_timesteps: int = 200_000
    n_envs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 256
    batch_size: int = 64
    seed: int = 42
    checkpoint_freq: int = 10_000
    device: str = "auto"


@dataclass
class DistillationConfig:
    """Config for fitting the follower value MLP."""

    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    # obstacle_arena specific: how many obstacle positions to sample for targets
    obstacle_samples: int = 500
    agent_grid_size: int = 20


@dataclass
class OODConfig:
    """Config for the VAE-based OOD metric."""

    enabled: bool = True
    latent_dim: int = 2
    grid_size: int = 50
    # Path to a pre-trained VAE checkpoint; if None the VAE is trained inline
    vae_path: Optional[str] = None


@dataclass
class GuidanceConfig:
    """Combines distillation and OOD into the follower-guidance pipeline."""

    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    ood: OODConfig = field(default_factory=OODConfig)


@dataclass
class HouseholdDeploymentConfig:
    """Config for the ROS2 node that deploys a household policy.

    ``sim_origin_in_world`` is ``(tx, ty, yaw)`` describing the pose of
    the simulation arena origin in the real-world ``parent_frame``.
    Robot positions/orientations are looked up via TF and transformed
    into the sim frame; the policy's action ``(vx, vy, ω)`` is rotated
    back into the parent frame for publication.
    """

    model_path: str = ""
    role: str = "follower"          # "cleaner" or "follower"
    parent_frame: str = "map"
    cleaner_frame: str = "robomaster_1"
    follower_frame: str = "robomaster_2"
    table_frame: str = "obstacle_1"
    # Frozen dirt position. If empty, the dirt is pinned to the table's
    # pose at node startup (matches the sim mechanic where the dirt is
    # frozen at the table's spawn centroid).
    dirt_frame: str = ""
    cmd_topic: str = "/reference_state"
    rate_hz: float = 10.0
    sim_origin_in_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_speed: float = 0.5
    max_yaw_rate: float = 3.141592653589793
    clamp_action: bool = True
    tf_timeout_sec: float = 0.2
    # Chirality of the parent frame. When True (e.g. NED/lab where z
    # points into the floor), positive yaw in TF is CW from above; the
    # sim's positive yaw is CCW from above. The node negates yaws on
    # the way in (TF→sim) and on the way out (sim→TF) so the policy
    # operates in its native convention.
    parent_z_down: bool = True
    # Debug modes — skip the policy and publish a hand-crafted setpoint.
    # ``goto_origin``: drive toward sim (0, 0); ``goto_dirt``: drive
    # toward the dirt position (table centroid).
    goto_origin: bool = False
    goto_dirt: bool = False

    # Episode mode — bounded rollout with a defined termination contract
    # (pickup or goal_reached, arena exit, max_steps, Ctrl-C). When
    # off, the node runs the existing unbounded controller.
    episode: bool = False
    # None → use the sim default per role (cleaner=200, follower=250).
    max_steps_override: Optional[int] = None
    # Shrink the arena bounds inward by this many meters for the
    # out-of-arena predicate. Cleaner/follower center outside the
    # shrunk box terminates the episode.
    arena_margin: float = 0.1
    # Seconds to wait for the first valid table + self TF before erroring.
    ready_timeout: float = 5.0
    # Skip the "press Enter to begin" gate after the ready log.
    no_confirm: bool = False
    # Show the live matplotlib visualization. On by default; disable
    # for headless / latency-sensitive runs. When False, recording is
    # also forced off (the recorder reuses the figure).
    viz: bool = True
    # Save MP4 + metadata.json of the rollout. Off by default.
    record: bool = False
    # Override the auto-timestamped output directory under
    # ``experiments/deployments/household_<role>/<UTC-ISO>/``.
    record_dir: str = ""
    # Profile the control loop (per-phase percentile breakdown printed
    # on exit, JSON saved into the record dir when --record).
    profile: bool = False

    # ---- Controller selection -----------------------------------------
    # "policy"  → run the trained PPO directly each tick (default).
    # "cem"     → CEM planner over the cleaner dynamics, using the
    #             cleaner PPO as warm-start prior and the follower
    #             V_F as the terminal-value component of the cost.
    #             Cleaner role only; mutually exclusive with the debug
    #             modes (--goto-origin/--goto-dirt).
    controller: str = "policy"
    # Path to the follower PPO checkpoint. Required for --controller cem
    # (the JointUtilityCost needs the follower V_F).
    follower_model_path: str = ""
    # Optional VAE checkpoint enabling the value_and_ood CEM mode —
    # follower V_F gets multiplied by an OOD trust factor before the
    # joint utility is computed. Empty → plain "value" mode.
    vae_path: str = ""
    # OOD trust scaling for value_and_ood mode (no effect when
    # vae_path is empty).
    alpha: float = 1.0
    # Deployment-scoped CEM planner overrides (None → use the scenario
    # planner.yaml value). Let the operator shrink the per-replan cost so
    # a replan fits inside the control period — e.g. set replan_interval=1
    # to replan every tick once the budget allows. These override only the
    # deployment planner, not eval/training (which read planner.yaml).
    cem_num_samples: Optional[int] = None
    cem_horizon: Optional[int] = None
    cem_num_iters: Optional[int] = None
    cem_replan_interval: Optional[int] = None


__all__ = [
    # Env configs
    "BaseEnvConfig",
    "NavEnvConfig",
    "PushEnvConfig",
    "HouseholdCleanerEnvConfig",
    "HouseholdDeploymentConfig",
    "HouseholdNavEnvConfig",
    "MobileArmFollowerEnvConfig",
    "MobileArmLeaderEnvConfig",
    "WarehouseFollowerEnvConfig",
    "WarehouseLeaderEnvConfig",
    # Training / guidance configs
    "AgentTrainingConfig",
    "DistillationConfig",
    "GuidanceConfig",
    "OODConfig",
    # Planner / cost configs
    "CEMPlannerConfig",
    "PlannerConfig",
    "OnlineAdaptationCostConfig",
    # Infrastructure configs
    "PathsConfig",
    "RunConfig",
    "VaeConfig",
    "WandbConfig",
]
