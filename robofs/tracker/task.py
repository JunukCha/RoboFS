# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""Shadow Hand landmark-tracking task (RoboFS-Tracker): scene, manager settings, PPO settings and registration."""

from __future__ import annotations

import gymnasium as gym

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.core.reorient.config.shadow_hand.agents.rsl_rl_ppo_cfg import ShadowHandPPORunnerCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_common import PhysicsCfg, ShadowHandRobotCfg

from isaaclab_assets.robots.shadow_hand import JOINT_NAMES, TENDON_NAMES, TENDON_POSITION_LIMITS

from robofs.config import section
from robofs.tracker.commands import HandMotionCommandCfg
from robofs.tracker.mdp import landmark_state, landmark_tracking, motion_finished, track_landmarks
from robofs.tracker.penetration import penetration_penalty

TASK_NAME = "Isaac-Shadow-Hand-Landmark-Tracking"
_TRACKER = section("tracker")
_REWARDS = _TRACKER["rewards"]


def spawn_hand_without_grasp_marker(prim_path, cfg, translation=None, orientation=None):
    """Spawn the hand while deactivating the asset's non-physical grasp-site guide."""
    from pxr import Usd

    from isaaclab.sim.spawners.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    markers = [child for child in Usd.PrimRange(prim) if child.GetName() == "grasp_site"]
    for marker in markers:
        # The Newton site importer can retain invisible guides; omit this unused site entirely.
        marker.SetActive(False)
    return prim


@configclass
class HandSceneCfg(InteractiveSceneCfg):
    """Fixed-base (fixed-wrist) hand with self-collision enabled by the asset config."""

    robot = ShadowHandRobotCfg()
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/rh_(ff|mf|rf|lf|th)(proximal|middle|distal)",
        history_length=4,
    )

    def __post_init__(self):
        for variant in ("newton_mjwarp", "isaacsim_physx", "physx", "ovphysx", "default"):
            robot = getattr(self.robot, variant)
            robot.spawn.func = spawn_hand_without_grasp_marker
            # Scalar-last quaternion: rotate the inherited horizontal mount -90 degrees about world X.
            robot.init_state.rot = (-0.5, -0.5, -0.5, 0.5)
            properties = robot.spawn.rigid_props
            for fragment in properties if isinstance(properties, list) else [properties]:
                if hasattr(fragment, "disable_gravity"):
                    fragment.disable_gravity = False


@configclass
class CommandsCfg:
    motion = HandMotionCommandCfg()


@configclass
class ActionsCfg:
    # Wrist motors are omitted: the target is expressed in a fixed palm-local frame.
    # alpha=1.0 disables EMA smoothing; the finger actuators already lag the command by 0.3-1 s.
    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=JOINT_NAMES[2:], alpha=1.0, rescale_to_limits=True
    )
    tendon_pos = mdp.FixedTendonPositionActionCfg(
        asset_name="robot",
        tendon_names=TENDON_NAMES,
        scale=0.5 * TENDON_POSITION_LIMITS[1],
        offset=0.5 * TENDON_POSITION_LIMITS[1],
        clip={".*": TENDON_POSITION_LIMITS},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """693 values: robot landmark state (315) and tracking inputs (189 + 63 per look-ahead time)."""

        body_state = ObsTerm(func=landmark_state)
        # Look-ahead targets let the policy start moving before the lagging fingers fall behind.
        tracking = ObsTerm(
            func=landmark_tracking, params={"future_offsets_s": tuple(_TRACKER["future_offsets_s"])}
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy = PolicyCfg()


@configclass
class RewardsCfg:
    landmarks = RewTerm(
        func=track_landmarks, weight=_REWARDS["landmarks"]["weight"], params={"sigma": _REWARDS["landmarks"]["sigma"]}
    )
    # Fingertip errors stay near 4 cm, where a 2.5 cm kernel gives almost no gradient.
    fingertips = RewTerm(
        func=track_landmarks,
        weight=_REWARDS["fingertips"]["weight"],
        params={"sigma": _REWARDS["fingertips"]["sigma"], "fingertips_only": True},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=_REWARDS["action_rate"])
    penetration = RewTerm(func=penetration_penalty, weight=_REWARDS["penetration"])


@configclass
class EventsCfg:
    reset_hand = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # The task ends with the clip, so the value is not bootstrapped past it.
    motion_end = DoneTerm(func=motion_finished, time_out=False)


@configclass
class HandTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Palm-local landmark tracking of one word clip per episode."""

    scene = HandSceneCfg(num_envs=_TRACKER["num_envs"], env_spacing=0.75)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    events = EventsCfg()
    terminations = TerminationsCfg()
    decimation = 4
    episode_length_s = 20.0

    def __post_init__(self):
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(0.62, -0.80, 0.85), lookat=(0.0, -0.28, 0.53), focal_length=35.0
        )


@configclass
class HandGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian exploration distribution with a trained standard deviation."""

    learn_std: bool = True
    """Whether the exploration standard deviation is trained (stored in the saved agent settings)."""


@configclass
class HandTrackingPPORunnerCfg(ShadowHandPPORunnerCfg):
    """Compact actor and critic with a short discount horizon for immediate landmark tracking."""

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="swish",
        obs_normalization=True,
        distribution_cfg=HandGaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="swish",
        obs_normalization=True,
    )
    algorithm = ShadowHandPPORunnerCfg().algorithm.replace(gamma=0.8)
    experiment_name = "shadow_hand_landmark_tracking"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 32
    max_iterations = _TRACKER["max_iterations"]
    save_interval = _TRACKER["save_interval"]
    clip_actions = 1.0
    init_at_random_ep_len = False


def register_task() -> None:
    """Register the task with Gymnasium (once per process)."""
    if TASK_NAME not in gym.registry:
        gym.register(
            id=TASK_NAME,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": HandTrackingEnvCfg,
                "rsl_rl_cfg_entry_point": HandTrackingPPORunnerCfg,
                "default_agent": "rsl_rl",
            },
        )
