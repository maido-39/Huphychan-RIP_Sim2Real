"""Motor-only sine-tracking RL task.

The passive pendulum assembly is removed from the MuJoCo model. Training and
playback use only the RobStride rotor and Revolute 3.

Model:
- assets/inverse_motor_only.xml
- one joint: Revolute 3
- no Revolute 5, pole body, pole inertia, or pole encoder

Observation per frame:
1) cos(motor_angle)
2) sin(motor_angle)
3) motor_velocity / 30
4) normalized sine target

history_length=2, so the policy input is 8-dimensional.

Target:
- 90 deg * sin(2*pi*0.25*t)
- period: 4 seconds

Control:
- MuJoCo physics: 200 Hz
- policy update: 50 Hz
- position action scale: 20 deg
- target rate limit: 1500 deg/s
- target acceleration limit: 15000 deg/s^2

Reward:
- negative squared circular motor-position tracking error
- action-rate penalty to suppress rapid command jitter

Reset:
- motor angle uniform in [-pi, pi]
- motor velocity zero
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
from mjlab.envs.mdp import time_out
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

_MOTOR_ONLY_XML = Path(__file__).parent / "assets" / "inverse_motor_only.xml"

_MOTOR_CFG = SceneEntityCfg(
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
_CYLINDER_TARGET_DELTA_LIMIT = _CYLINDER_TARGET_RATE_LIMIT * _POLICY_DT_S

_CYLINDER_TARGET_ACCEL_LIMIT = math.radians(15000.0)
_CYLINDER_TARGET_DELTA_CHANGE_LIMIT = (
  _CYLINDER_TARGET_ACCEL_LIMIT * _POLICY_DT_S * _POLICY_DT_S
)

_ACTION_SCALE_RAD = math.radians(20.0)
_VEL_OBS_SCALE = 30.0


# =============================================================================
# Motor sine-reference settings
# =============================================================================

# Motor target:
#   target(t) = 90 deg * sin(2*pi*0.25*t)
# One full cycle takes 4 seconds:
#   0 s: 0 deg, 1 s: +90 deg, 2 s: 0 deg, 3 s: -90 deg
_MOTOR_SINE_AMPLITUDE_RAD = math.radians(90.0)
_MOTOR_SINE_FREQUENCY_HZ = 0.25

_EPISODE_LENGTH_S = 12.0
# MuJoCo entity
# =============================================================================


def _get_spec() -> mujoco.MjSpec:
  """Load the same MJCF used by the existing inverse task."""
  return mujoco.MjSpec.from_file(str(_MOTOR_ONLY_XML))


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

    self._limited_target = self._entity.data.joint_pos[:, self._target_ids].clone()
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

    self._limited_target = self._limited_target + self._target_delta
    self._processed_actions = self._limited_target.clone()

  def reset(
    self,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    super().reset(env_ids)

    if env_ids is None:
      env_ids = slice(None)

    self._limited_target[env_ids] = self._entity.data.joint_pos[env_ids].index_select(
      dim=1,
      index=self._target_ids,
    )
    self._target_delta[env_ids] = 0.0


# =============================================================================
# Motor sine reference
# =============================================================================


def _episode_time_s(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Elapsed episode time for every parallel environment."""
  return env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt)


def _motor_sine_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Current phase of the sinusoidal motor target."""
  return 2.0 * math.pi * _MOTOR_SINE_FREQUENCY_HZ * _episode_time_s(env)


def _motor_target_position(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Motor target position ranging from -90 to +90 degrees."""
  return _MOTOR_SINE_AMPLITUDE_RAD * torch.sin(_motor_sine_phase(env))


# =============================================================================
# Observations
# =============================================================================


def cylinder_angle_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _MOTOR_CFG,
) -> torch.Tensor:
  """Periodic representation of the actuated motor angle."""
  asset: Entity = env.scene[asset_cfg.name]
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids]

  return torch.cat(
    [torch.cos(angle), torch.sin(angle)],
    dim=-1,
  )


def scaled_joint_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  scale: float = _VEL_OBS_SCALE,
) -> torch.Tensor:
  """Joint velocity divided by a fixed task scale."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids] / scale


def motor_target_observation(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Normalized motor sine target in the range [-1, 1]."""
  target_position = _motor_target_position(env)
  return (target_position / _MOTOR_SINE_AMPLITUDE_RAD).unsqueeze(-1)


# =============================================================================
# The only reward
# =============================================================================


def motor_sine_tracking_reward(
  env: ManagerBasedRlEnv,
  cylinder_cfg: SceneEntityCfg = _MOTOR_CFG,
) -> torch.Tensor:
  """Minimize squared error between motor angle and sinusoidal target."""
  asset: Entity = env.scene[cylinder_cfg.name]

  motor_position = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)

  target_position = _motor_target_position(env)

  # Circular error in [-pi, pi].
  position_error = torch.atan2(
    torch.sin(motor_position - target_position),
    torch.cos(motor_position - target_position),
  )

  # PPO maximizes reward, so minimizing error^2 is expressed as -error^2.
  return -position_error.square()


class action_rate_l2:
  """Penalize rapid changes between consecutive raw policy actions."""

  def __init__(
    self,
    cfg: RewardTermCfg,
    env: ManagerBasedRlEnv,
  ):
    del cfg
    self._previous_action = torch.zeros(
      (env.num_envs, 1),
      device=env.device,
      dtype=torch.float32,
    )
    self._initialized = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.bool,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
  ) -> torch.Tensor:
    current_action = env.action_manager.action[:, :1]
    action_delta = current_action - self._previous_action
    penalty = action_delta.square().squeeze(-1)

    penalty = torch.where(
      self._initialized,
      penalty,
      torch.zeros_like(penalty),
    )

    self._previous_action = current_action.clone()
    self._initialized[:] = True
    return penalty

  def reset(
    self,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = slice(None)

    self._previous_action[env_ids] = 0.0
    self._initialized[env_ids] = False


# =============================================================================
# Reset
# =============================================================================


def reset_motor_random_360(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _MOTOR_CFG,
) -> None:
  """Reset only Revolute 3 over one full revolution."""
  if env_ids is None:
    env_ids = torch.arange(
      env.num_envs,
      device=env.device,
      dtype=torch.long,
    )
  elif isinstance(env_ids, slice):
    env_ids = torch.arange(
      env.num_envs,
      device=env.device,
      dtype=torch.long,
    )[env_ids]

  asset: Entity = env.scene[asset_cfg.name]
  num_resets = int(env_ids.numel())

  joint_pos = (
    2.0
    * math.pi
    * torch.rand(
      (num_resets, 1),
      device=env.device,
      dtype=torch.float32,
    )
    - math.pi
  )
  joint_vel = torch.zeros_like(joint_pos)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(
      joint_ids,
      device=env.device,
      dtype=torch.long,
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
  # One frame = 4 values:
  #   [cos(motor_angle), sin(motor_angle), motor_velocity / 30, target / 90deg]
  #
  # history_length=2 means the policy receives 8 values total.
  # The passive pole remains in the simulation dynamics, but its angle and
  # velocity are intentionally not available to the policy.
  actor_terms = {
    "motor_angle_periodic": ObservationTermCfg(
      func=cylinder_angle_cos_sin,
      params={"asset_cfg": _MOTOR_CFG},
    ),
    "motor_velocity": ObservationTermCfg(
      func=scaled_joint_vel,
      params={
        "asset_cfg": _MOTOR_CFG,
        "scale": _VEL_OBS_SCALE,
      },
      clip=(-5.0, 5.0),
    ),
    "motor_target": ObservationTermCfg(
      func=motor_target_observation,
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
    "reset_motor_random_360": EventTermCfg(
      func=reset_motor_random_360,
      mode="reset",
      params={"asset_cfg": _MOTOR_CFG},
    ),
  }

  # Stage 1 jitter suppression:
  # keep the original tracking objective and add only action-rate smoothing.
  rewards = {
    "motor_sine_tracking": RewardTermCfg(
      func=motor_sine_tracking_reward,
      weight=1.0,
      params={"cylinder_cfg": _MOTOR_CFG},
    ),
    "action_rate": RewardTermCfg(
      func=action_rate_l2,
      weight=-0.01,
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


# Exported as Mjlab-Inverse-Motor-Sine.
def motor_only_sine_env_cfg(
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


def motor_only_sine_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.3,
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
      entropy_coef=0.001,
      num_learning_epochs=3,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=0.5,
    ),
    experiment_name="motor_only_sine",
    save_interval=50,
    num_steps_per_env=64,
    max_iterations=5000,
  )
