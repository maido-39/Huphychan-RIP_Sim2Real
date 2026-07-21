"""Inverse motor sine-tracking reinforcement-learning task.

Drop-in replacement for the existing inverse_env_cfg.py.

The task keeps the original hardware-side constraints:
- MuJoCo physics: 200 Hz
- Policy/action update: 50 Hz
- Position action scale: 20 deg
- Target rate limit: 1500 deg/s
- Target acceleration limit: 15000 deg/s^2
- Actuator gains, force range, joint damping/friction/armature: loaded from assets/inverse.xml

Task-specific design:
- One reward term only: sine position/velocity tracking
- No balance, swing-up, upright, chatter, or rotation-limit rewards/terminations
- Only time_out remains to create PPO episode boundaries
- No domain randomization
- Deterministic zero-state reset
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import last_action, time_out
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# =============================================================================
# Model and entity configuration
# =============================================================================

_INVERSE_XML = Path(__file__).parent / "assets" / "inverse.xml"

_JOINTS_CFG = SceneEntityCfg(
    "inverse",
    joint_names=("Revolute 3", "Revolute 5"),
)
_CYLINDER_CFG = SceneEntityCfg(
    "inverse",
    joint_names=("Revolute 3",),
)


# =============================================================================
# Hardware/control constraints: kept identical to the balance task
# =============================================================================

_ACTION_DELAY_STEPS = 0

_PHYSICS_DT_S = 0.005
_DECIMATION = 4
_POLICY_DT_S = _PHYSICS_DT_S * _DECIMATION  # 0.02 s = 50 Hz

_CYLINDER_TARGET_RATE_LIMIT = math.radians(1500.0)
_CYLINDER_TARGET_DELTA_LIMIT = (
    _CYLINDER_TARGET_RATE_LIMIT * _POLICY_DT_S
)

_CYLINDER_TARGET_ACCEL_LIMIT = math.radians(15000.0)
_CYLINDER_TARGET_DELTA_CHANGE_LIMIT = (
    _CYLINDER_TARGET_ACCEL_LIMIT
    * _POLICY_DT_S
    * _POLICY_DT_S
)

_ACTION_SCALE_RAD = math.radians(20.0)
_VEL_OBS_SCALE = 30.0


# =============================================================================
# Sine-reference settings
# =============================================================================

# q_ref(t) = offset + amplitude * sin(2*pi*frequency*t + phase)
_SINE_OFFSET_RAD = math.radians(0.0)
_SINE_AMPLITUDE_RAD = math.radians(60.0)
_SINE_FREQUENCY_HZ = 0.25
_SINE_PHASE_RAD = 0.0

# One reward term uses both position and velocity tracking.
# Larger sigma makes the reward broader/easier; smaller sigma demands precision.
_SINE_POSITION_SIGMA_RAD = math.radians(15.0)
_SINE_VELOCITY_SIGMA_RAD_S = math.radians(120.0)

_EPISODE_LENGTH_S = 12.0


# =============================================================================
# MuJoCo entity
# =============================================================================

def _get_spec() -> mujoco.MjSpec:
    """Load the same MJCF used by the existing inverse task."""
    return mujoco.MjSpec.from_file(str(_INVERSE_XML))


_INVERSE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=("Revolute 3",),
            delay_min_lag=_ACTION_DELAY_STEPS,
            delay_max_lag=_ACTION_DELAY_STEPS,
        ),
    ),
)

_INVERSE_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "Revolute 3": 0.0,
        "Revolute 5": 0.0,
    },
    joint_vel={".*": 0.0},
)


def _get_inverse_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_INVERSE_ARTICULATION,
        init_state=_INVERSE_INIT,
    )


# =============================================================================
# Rate- and acceleration-limited action
# =============================================================================

@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action with target rate and acceleration limits."""

    max_delta: float = _CYLINDER_TARGET_DELTA_LIMIT
    max_delta_change: float = _CYLINDER_TARGET_DELTA_CHANGE_LIMIT

    def build(
        self,
        env: ManagerBasedRlEnv,
    ) -> "RateLimitedJointPositionAction":
        return RateLimitedJointPositionAction(self, env)


class RateLimitedJointPositionAction(JointPositionAction):
    """Limit q_target changes before writing them to the actuator."""

    cfg: RateLimitedJointPositionActionCfg

    def __init__(
        self,
        cfg: RateLimitedJointPositionActionCfg,
        env: ManagerBasedRlEnv,
    ):
        super().__init__(cfg=cfg, env=env)

        self._limited_target = self._entity.data.joint_pos[
            :, self._target_ids
        ].clone()
        self._target_delta = torch.zeros_like(self._limited_target)

    def process_actions(self, actions: torch.Tensor) -> None:
        # Parent class applies the original action scale.
        super().process_actions(actions)

        desired_target = self._processed_actions

        # Rate limit: constrain q_target change per policy step.
        desired_delta = torch.clamp(
            desired_target - self._limited_target,
            min=-float(self.cfg.max_delta),
            max=float(self.cfg.max_delta),
        )

        # Acceleration limit: constrain change of target delta.
        delta_step = torch.clamp(
            desired_delta - self._target_delta,
            min=-float(self.cfg.max_delta_change),
            max=float(self.cfg.max_delta_change),
        )

        self._target_delta = torch.clamp(
            self._target_delta + delta_step,
            min=-float(self.cfg.max_delta),
            max=float(self.cfg.max_delta),
        )

        self._limited_target = (
            self._limited_target + self._target_delta
        )
        self._processed_actions = self._limited_target.clone()

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        super().reset(env_ids)

        if env_ids is None:
            env_ids = slice(None)

        self._limited_target[env_ids] = (
            self._entity.data.joint_pos[env_ids].index_select(
                dim=1,
                index=self._target_ids,
            )
        )
        self._target_delta[env_ids] = 0.0


# =============================================================================
# Sine reference
# =============================================================================

def _episode_time_s(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Elapsed episode time for every parallel environment."""
    return env.episode_length_buf.to(dtype=torch.float32) * float(
        env.step_dt
    )


def _sine_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Current sine phase in radians."""
    time_s = _episode_time_s(env)
    omega = 2.0 * math.pi * _SINE_FREQUENCY_HZ
    return omega * time_s + _SINE_PHASE_RAD


def _sine_target_position(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reference motor position q_ref(t), shape [num_envs]."""
    phase = _sine_phase(env)
    return (
        _SINE_OFFSET_RAD
        + _SINE_AMPLITUDE_RAD * torch.sin(phase)
    )


def _sine_target_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reference motor velocity dq_ref(t), shape [num_envs]."""
    phase = _sine_phase(env)
    omega = 2.0 * math.pi * _SINE_FREQUENCY_HZ
    return _SINE_AMPLITUDE_RAD * omega * torch.cos(phase)


# =============================================================================
# Observations
# =============================================================================

def cylinder_angle_cos_sin(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
    """Periodic representation of the actuated motor angle."""
    asset: Entity = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]

    return torch.cat(
        [torch.cos(angle), torch.sin(angle)],
        dim=-1,
    )


def scaled_joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    scale: float,
) -> torch.Tensor:
    """Joint position divided by a fixed task scale."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] / scale


def scaled_joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = _VEL_OBS_SCALE,
) -> torch.Tensor:
    """Joint velocity divided by a fixed task scale."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] / scale


def sine_phase_observation(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Give the policy sin(phase), cos(phase) so the target is observable."""
    phase = _sine_phase(env)

    return torch.stack(
        [torch.sin(phase), torch.cos(phase)],
        dim=-1,
    )


def sine_target_observation(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Normalized target position and target velocity."""
    target_position = _sine_target_position(env)
    target_velocity = _sine_target_velocity(env)

    position_scale = max(abs(_SINE_AMPLITUDE_RAD), 1e-6)
    velocity_scale = max(
        abs(
            _SINE_AMPLITUDE_RAD
            * 2.0
            * math.pi
            * _SINE_FREQUENCY_HZ
        ),
        1e-6,
    )

    return torch.stack(
        [
            target_position / position_scale,
            target_velocity / velocity_scale,
        ],
        dim=-1,
    )


# =============================================================================
# The only reward
# =============================================================================

def sine_tracking_reward(
    env: ManagerBasedRlEnv,
    cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
    position_sigma_rad: float = _SINE_POSITION_SIGMA_RAD,
    velocity_sigma_rad_s: float = _SINE_VELOCITY_SIGMA_RAD_S,
) -> torch.Tensor:
    """Reward accurate tracking of the sine position and velocity reference.

    This remains a single RewardTermCfg. Position and velocity are combined
    in one exponential tracking score:

        reward = exp(
            -(position_error / position_sigma)^2
            -(velocity_error / velocity_sigma)^2
        )

    Position error is wrapped to [-pi, pi].
    """

    asset: Entity = env.scene[cylinder_cfg.name]

    motor_position = asset.data.joint_pos[
        :, cylinder_cfg.joint_ids
    ].squeeze(-1)
    motor_velocity = asset.data.joint_vel[
        :, cylinder_cfg.joint_ids
    ].squeeze(-1)

    target_position = _sine_target_position(env)
    target_velocity = _sine_target_velocity(env)

    position_error = torch.atan2(
        torch.sin(motor_position - target_position),
        torch.cos(motor_position - target_position),
    )
    velocity_error = motor_velocity - target_velocity

    normalized_position_error = (
        position_error / float(position_sigma_rad)
    )
    normalized_velocity_error = (
        velocity_error / float(velocity_sigma_rad_s)
    )

    return torch.exp(
        -normalized_position_error.square()
        -normalized_velocity_error.square()
    )


# =============================================================================
# Reset
# =============================================================================

def reset_sine_task(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _JOINTS_CFG,
) -> None:
    """Reset both joints to zero without curriculum or randomization."""

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
        )

    asset: Entity = env.scene[asset_cfg.name]

    joint_pos = torch.zeros(
        (len(env_ids), 2),
        device=env.device,
    )
    joint_vel = torch.zeros_like(joint_pos)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(
            joint_ids,
            device=env.device,
        )

    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )


# =============================================================================
# Environment configuration
# =============================================================================

def _make_env_cfg() -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "cylinder_angle": ObservationTermCfg(
            func=scaled_joint_pos,
            params={
                "asset_cfg": _CYLINDER_CFG,
                "scale": math.pi,
            },
            clip=(-2.0, 2.0),
        ),
        "cylinder_angle_periodic": ObservationTermCfg(
            func=cylinder_angle_cos_sin,
            params={"asset_cfg": _CYLINDER_CFG},
        ),
        "joint_velocity": ObservationTermCfg(
            func=scaled_joint_vel,
            params={
                "asset_cfg": _JOINTS_CFG,
                "scale": _VEL_OBS_SCALE,
            },
            clip=(-5.0, 5.0),
        ),
        "sine_phase": ObservationTermCfg(
            func=sine_phase_observation,
        ),
        "sine_target": ObservationTermCfg(
            func=sine_target_observation,
        ),
        "last_action": ObservationTermCfg(
            func=last_action,
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            actor_terms,
            enable_corruption=False,
            history_length=2,
        ),
        "critic": ObservationGroupCfg(
            {**actor_terms},
            enable_corruption=False,
            history_length=2,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "position": RateLimitedJointPositionActionCfg(
            entity_name="inverse",
            actuator_names=("Revolute 3",),
            scale=_ACTION_SCALE_RAD,
            max_delta=_CYLINDER_TARGET_DELTA_LIMIT,
            max_delta_change=_CYLINDER_TARGET_DELTA_CHANGE_LIMIT,
        ),
    }

    events = {
        "reset_sine_task": EventTermCfg(
            func=reset_sine_task,
            mode="reset",
            params={"asset_cfg": _JOINTS_CFG},
        ),
    }

    # Exactly one reward term.
    rewards = {
        "sine_tracking": RewardTermCfg(
            func=sine_tracking_reward,
            weight=1.0,
            params={
                "cylinder_cfg": _CYLINDER_CFG,
                "position_sigma_rad": _SINE_POSITION_SIGMA_RAD,
                "velocity_sigma_rad_s": _SINE_VELOCITY_SIGMA_RAD_S,
            },
        ),
    }

    # time_out is retained only to create episode boundaries for PPO.
    # There are no failure/safety/task-specific terminations.
    terminations = {
        "time_out": TerminationTermCfg(
            func=time_out,
            time_out=True,
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=None,
            entities={"inverse": _get_inverse_cfg()},
            num_envs=64,
            env_spacing=1.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        metrics={},
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.011, 0.0, -0.028),
            distance=0.35,
            elevation=-15.0,
            azimuth=135.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=_PHYSICS_DT_S,
                disableflags=("contact",),
            ),
        ),
        decimation=_DECIMATION,
        episode_length_s=_EPISODE_LENGTH_S,
    )


# Keep the original exported function name so the existing
# Mjlab-Inverse-Balance registration can load this file unchanged.
def inverse_sine_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = _make_env_cfg()

    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["critic"].enable_corruption = False

    return cfg


# =============================================================================
# PPO configuration
# =============================================================================

def inverse_sine_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.008,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=0.5,
        ),
        experiment_name="inverse_sine",
        save_interval=50,
        num_steps_per_env=64,
        max_iterations=5000,
    )