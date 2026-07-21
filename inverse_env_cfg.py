"""Inverse robot balance task configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import (
  last_action,
  time_out,
)
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.metrics_manager import MetricsTermCfg
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
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_INVERSE_XML = Path(__file__).parent / "assets" / "inverse.xml"
_JOINTS_CFG = SceneEntityCfg(
  "inverse",
  joint_names=("Revolute 3", "Revolute 5"),
)
_CYLINDER_CFG = SceneEntityCfg("inverse", joint_names=("Revolute 3",))
_POLE_CFG = SceneEntityCfg("inverse", joint_names=("Revolute 5",))
_POLE_BODY_CFG = SceneEntityCfg("inverse", body_names=("BoldHolder_1",))
_POLE_INERTIA_BODY_CFG = SceneEntityCfg("inverse", body_names=("BoldHolder_1",))

_TARGET_POLE_ANGLE = math.pi
_MAX_CYLINDER_ROTATION = 3.0 * math.pi
_CYLINDER_TERMINATION_ROTATION = 4.0 * math.pi
_CYLINDER_TERMINATION_SPEED = math.radians(2000.0)
_CYLINDER_CHATTER_ANGLE_RANGE = math.radians(10.0)
_CYLINDER_CHATTER_REVERSALS_PER_SEC = 3
_CYLINDER_CHATTER_UPRIGHT_EXEMPTION = math.radians(30.0)
_VEL_OBS_SCALE = 30.0
_BALANCE_START_PROBABILITY = 1.0
_BALANCE_HOLD_THRESHOLD = math.radians(30.0)
_EXACT_UPRIGHT_BONUS_THRESHOLD = math.radians(30.0)
_UPPER_SWING_THRESHOLD = math.radians(70.0)
_UPPER_SPEED_PENALTY_THRESHOLD = math.radians(60.0)
_NEAR_UPRIGHT_SPEED_PENALTY_THRESHOLD = math.radians(20.0)
_UPRIGHT_REACHED_THRESHOLD = math.radians(10.0)
_POLE_ONE_WAY_ROTATION_LIMIT = 4.0 * math.pi
_LOWER_SWING_SPEED_TARGET = math.radians(220.0)
_NEAR_UPRIGHT_SPEED_TARGET = math.radians(80.0)
_ACTION_DELAY_STEPS = 0
_CYLINDER_TARGET_RATE_LIMIT = math.radians(1500.0)
_CYLINDER_TARGET_DELTA_LIMIT = _CYLINDER_TARGET_RATE_LIMIT * 0.02
_CYLINDER_TARGET_ACCEL_LIMIT = math.radians(15000.0)
_CYLINDER_TARGET_DELTA_CHANGE_LIMIT = _CYLINDER_TARGET_ACCEL_LIMIT * 0.02 * 0.02
_JOINT5_RANDOMIZATION_SCALE = (0.5, 2.0)
_CYLINDER_START_POSITION_RANGE = (-math.pi, math.pi)
_POLE_COM_VERTICAL_OFFSET_RANGE = (-0.003, 0.003)
_RANDOMIZED_PLAY_ENV_VAR = "MJLAB_INVERSE_PLAY_RANDOMIZED"


def _env_flag(name: str) -> bool:
  return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _get_spec() -> mujoco.MjSpec:
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
  joint_pos={"Revolute 3": 0.0, "Revolute 5": 0.0},
  joint_vel={".*": 0.0},
)


def _get_inverse_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_INVERSE_ARTICULATION,
    init_state=_INVERSE_INIT,
  )


@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action with a per-policy-step target slew-rate limit."""

  max_delta: float = _CYLINDER_TARGET_DELTA_LIMIT
  max_delta_change: float = _CYLINDER_TARGET_DELTA_CHANGE_LIMIT

  def build(self, env: ManagerBasedRlEnv) -> "RateLimitedJointPositionAction":
    return RateLimitedJointPositionAction(self, env)


class RateLimitedJointPositionAction(JointPositionAction):
  """Limit commanded joint-position target changes before writing to actuators."""

  cfg: RateLimitedJointPositionActionCfg

  def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._limited_target = self._entity.data.joint_pos[:, self._target_ids].clone()
    self._target_delta = torch.zeros_like(self._limited_target)

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    desired_target = self._processed_actions
    desired_delta = torch.clamp(
      desired_target - self._limited_target,
      min=-float(self.cfg.max_delta),
      max=float(self.cfg.max_delta),
    )
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

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._limited_target[env_ids] = self._entity.data.joint_pos[env_ids].index_select(
      dim=1, index=self._target_ids
    )
    self._target_delta[env_ids] = 0.0


def pole_angle_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _POLE_CFG,
) -> torch.Tensor:
  """Cosine and sine of the pole joint angle."""
  asset: Entity = env.scene[asset_cfg.name]
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


def cylinder_angle_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Cosine and sine of the actuated cylinder angle."""
  asset: Entity = env.scene[asset_cfg.name]
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


def scaled_joint_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  scale: float,
) -> torch.Tensor:
  """Joint position normalized by a task-specific scale."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids] / scale


def scaled_joint_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  scale: float = _VEL_OBS_SCALE,
) -> torch.Tensor:
  """Joint velocity normalized to the same rough range as the reference task."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids] / scale


def pole_upright_reward(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
) -> torch.Tensor:
  """Reward pole height, rising early while still maxing at upright."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  pole_lift = 0.5 * (torch.cos(pole_error) + 1.0)
  return torch.sqrt(torch.clamp(pole_lift, min=0.0))


def pole_lift_angle_deg_metric(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
) -> torch.Tensor:
  """Pole lift angle in degrees: 0 down, 180 upright."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_lift = 0.5 * (1.0 - torch.cos(pole_angle))
  return torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * pole_lift, -1.0, 1.0)))


def pole_upright_weighted_metric(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
) -> torch.Tensor:
  """Current weighted pole_upright value used for quick TensorBoard reading."""
  return 24.0 * pole_upright_reward(env, pole_cfg=pole_cfg)


def pole_hold_30deg_reward(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
  hold_threshold_rad: float = _BALANCE_HOLD_THRESHOLD,
) -> torch.Tensor:
  """Reward immediately when the pole enters the practical +/-30 degree balance zone."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  return (torch.abs(pole_error) < float(hold_threshold_rad)).float()


def exact_upright_bonus(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
  active_threshold_rad: float = _EXACT_UPRIGHT_BONUS_THRESHOLD,
) -> torch.Tensor:
  """Large bonus that ramps up near upright and peaks at 180 degrees."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  normalized_error = torch.abs(pole_error) / float(active_threshold_rad)
  return torch.clamp(1.0 - normalized_error, min=0.0, max=1.0)


def pole_slow_reward(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
) -> torch.Tensor:
  """Reward low pole angular velocity only when the pole is near upright."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  upright_gate = torch.exp(-6.0 * pole_error.square())
  return upright_gate * torch.exp(-0.05 * pole_vel.square())


def lower_swing_reward(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
) -> torch.Tensor:
  """Reward motion that actually increases pole height in the lower swing-up zone."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids].squeeze(-1)

  # 0 at hanging down, 1 near upright.
  pole_lift = 0.5 * (1.0 - torch.cos(pole_angle))
  lower_zone_gate = torch.sigmoid(14.0 * (0.45 - pole_lift))

  height_increase_rate = torch.relu(torch.sin(pole_angle) * pole_vel)
  pole_motion_reward = torch.tanh(height_increase_rate / _LOWER_SWING_SPEED_TARGET)

  return lower_zone_gate * pole_motion_reward


def coordinated_swing_reward(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
  cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Reward opposite cylinder/pole motion only in the lower swing region."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids].squeeze(-1)
  cylinder_vel = asset.data.joint_vel[:, cylinder_cfg.joint_ids].squeeze(-1)

  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  upper_region = (torch.abs(pole_error) < _UPPER_SWING_THRESHOLD).float()
  lower_region = 1.0 - upper_region
  opposite_direction = (pole_vel * cylinder_vel < 0.0).float()
  pole_motion = torch.tanh(torch.abs(pole_vel) / _LOWER_SWING_SPEED_TARGET)
  cylinder_motion = torch.tanh(torch.abs(cylinder_vel) / _LOWER_SWING_SPEED_TARGET)
  return lower_region * opposite_direction * pole_motion * cylinder_motion


def upper_region_pole_speed_penalty(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
  upper_threshold_rad: float = _UPPER_SPEED_PENALTY_THRESHOLD,
  speed_scale: float = _LOWER_SWING_SPEED_TARGET,
) -> torch.Tensor:
  """Penalize fast pole motion only in the 120~240 degree upper region."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  upper_region = (torch.abs(pole_error) < float(upper_threshold_rad)).float()
  normalized_speed = torch.tanh(torch.abs(pole_vel) / float(speed_scale))
  return upper_region * normalized_speed


def near_upright_pole_speed_penalty(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
  near_threshold_rad: float = _NEAR_UPRIGHT_SPEED_PENALTY_THRESHOLD,
  speed_scale: float = _NEAR_UPRIGHT_SPEED_TARGET,
) -> torch.Tensor:
  """Penalize fast pole motion strongly inside the 160~200 degree balance region."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  near_upright = (torch.abs(pole_error) < float(near_threshold_rad)).float()
  normalized_speed = torch.tanh(torch.abs(pole_vel) / float(speed_scale))
  return near_upright * normalized_speed


class post_upright_pole_speed_l2:
  """Penalize fast pole motion after the pole has reached upright once."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._has_reached_upright = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    pole_cfg: SceneEntityCfg = _POLE_CFG,
    reached_threshold_rad: float = _UPRIGHT_REACHED_THRESHOLD,
    speed_scale: float = _LOWER_SWING_SPEED_TARGET,
  ) -> torch.Tensor:
    asset: Entity = env.scene[pole_cfg.name]
    pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids].squeeze(-1)
    pole_error = torch.atan2(
      torch.sin(pole_angle - _TARGET_POLE_ANGLE),
      torch.cos(pole_angle - _TARGET_POLE_ANGLE),
    )
    self._has_reached_upright |= torch.abs(pole_error) < reached_threshold_rad
    normalized_speed = pole_vel / speed_scale
    return self._has_reached_upright.float() * normalized_speed.square()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._has_reached_upright[env_ids] = False


def cylinder_rotation_limit(
  env: ManagerBasedRlEnv,
  cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Terminate only after the cylinder goes well beyond the penalty region."""
  asset: Entity = env.scene[cylinder_cfg.name]
  cylinder_angle = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)
  return torch.abs(cylinder_angle) > _CYLINDER_TERMINATION_ROTATION


def cylinder_speed_limit(
  env: ManagerBasedRlEnv,
  cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Terminate when the actuated cylinder speed exceeds the safety limit."""
  asset: Entity = env.scene[cylinder_cfg.name]
  cylinder_vel = asset.data.joint_vel[:, cylinder_cfg.joint_ids].squeeze(-1)
  return torch.abs(cylinder_vel) > _CYLINDER_TERMINATION_SPEED


class cylinder_chatter_limit:
  """Terminate if the motor axis jitters in a tiny range away from upright."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._window_steps = max(1, int(round(1.0 / env.step_dt)))
    self._step_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._reversal_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._last_vel_sign = torch.zeros(env.num_envs, device=env.device)
    self._last_reversal_angle = torch.full(
      (env.num_envs,), float("nan"), device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
    pole_cfg: SceneEntityCfg = _POLE_CFG,
    angle_range_rad: float = _CYLINDER_CHATTER_ANGLE_RANGE,
    reversals_per_sec: int = _CYLINDER_CHATTER_REVERSALS_PER_SEC,
    upright_exemption_rad: float = _CYLINDER_CHATTER_UPRIGHT_EXEMPTION,
    min_speed_rad_s: float = math.radians(1.0),
  ) -> torch.Tensor:
    asset: Entity = env.scene[cylinder_cfg.name]
    cylinder_angle = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)
    cylinder_vel = asset.data.joint_vel[:, cylinder_cfg.joint_ids].squeeze(-1)
    pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)

    self._step_count += 1

    vel_sign = torch.sign(cylinder_vel)
    vel_sign = torch.where(
      torch.abs(cylinder_vel) > min_speed_rad_s,
      vel_sign,
      torch.zeros_like(vel_sign),
    )
    reversed_direction = (
      (vel_sign != 0.0)
      & (self._last_vel_sign != 0.0)
      & (vel_sign != self._last_vel_sign)
    )
    has_last_reversal = torch.isfinite(self._last_reversal_angle)
    reversal_angle_delta = torch.abs(cylinder_angle - self._last_reversal_angle)
    small_reversal = reversed_direction & (
      (~has_last_reversal) | (reversal_angle_delta < angle_range_rad)
    )
    self._reversal_count += small_reversal.long()
    self._last_reversal_angle = torch.where(
      reversed_direction,
      cylinder_angle,
      self._last_reversal_angle,
    )
    self._last_vel_sign = torch.where(vel_sign != 0.0, vel_sign, self._last_vel_sign)

    window_done = self._step_count >= self._window_steps
    pole_error = torch.atan2(
      torch.sin(pole_angle - _TARGET_POLE_ANGLE),
      torch.cos(pole_angle - _TARGET_POLE_ANGLE),
    )
    upright_exempted = torch.abs(pole_error) < float(upright_exemption_rad)
    terminate = (
      (self._reversal_count >= int(reversals_per_sec))
      & ~upright_exempted
    )

    self._reset_windows(window_done)
    return terminate

  def _reset_windows(self, env_ids: torch.Tensor) -> None:
    self._step_count[env_ids] = 0
    self._reversal_count[env_ids] = 0
    self._last_reversal_angle[env_ids] = float("nan")

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._step_count[env_ids] = 0
    self._reversal_count[env_ids] = 0
    self._last_vel_sign[env_ids] = 0.0
    self._last_reversal_angle[env_ids] = float("nan")


def reset_swingup_balance_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _JOINTS_CFG,
  balance_start_probability: float = _BALANCE_START_PROBABILITY,
  cylinder_position_range: tuple[float, float] = (-0.05, 0.05),
  cylinder_velocity_range: tuple[float, float] = (-0.01, 0.01),
  swingup_pole_position_range: tuple[float, float] = (-0.05, 0.05),
  swingup_pole_velocity_range: tuple[float, float] = (-0.05, 0.05),
  balance_pole_position_range: tuple[float, float] = (
    math.pi - math.radians(15.0),
    math.pi + math.radians(15.0),
  ),
  balance_pole_velocity_range: tuple[float, float] = (-0.2, 0.2),
) -> None:
  """Reset each env from either swing-up or balance initial conditions."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = torch.zeros((len(env_ids), 2), device=env.device)
  joint_vel = torch.zeros_like(joint_pos)

  joint_pos[:, 0] = sample_uniform(
    *cylinder_position_range, (len(env_ids),), device=env.device
  )
  joint_vel[:, 0] = sample_uniform(
    *cylinder_velocity_range, (len(env_ids),), device=env.device
  )

  balance_mask = torch.rand(len(env_ids), device=env.device) < balance_start_probability
  swingup_mask = ~balance_mask

  joint_pos[swingup_mask, 1] = sample_uniform(
    *swingup_pole_position_range, (int(swingup_mask.sum()),), device=env.device
  )
  joint_vel[swingup_mask, 1] = sample_uniform(
    *swingup_pole_velocity_range, (int(swingup_mask.sum()),), device=env.device
  )

  joint_pos[balance_mask, 1] = sample_uniform(
    *balance_pole_position_range, (int(balance_mask.sum()),), device=env.device
  )
  joint_vel[balance_mask, 1] = sample_uniform(
    *balance_pole_velocity_range, (int(balance_mask.sum()),), device=env.device
  )

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    joint_ids=joint_ids,
    env_ids=env_ids,
  )


class terminate_after_clear_drop:
  """Terminate after the pole reaches the upright band and then leaves it."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._has_reached_upright_zone = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    pole_cfg: SceneEntityCfg = _POLE_CFG,
    reached_threshold_deg: float = 60.0,
    drop_threshold_deg: float = 60.0,
  ) -> torch.Tensor:
    asset: Entity = env.scene[pole_cfg.name]
    pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
    pole_error = torch.atan2(
      torch.sin(pole_angle - _TARGET_POLE_ANGLE),
      torch.cos(pole_angle - _TARGET_POLE_ANGLE),
    )

    reached_threshold = math.radians(reached_threshold_deg)
    drop_threshold = math.radians(drop_threshold_deg)

    self._has_reached_upright_zone |= torch.abs(pole_error) < reached_threshold
    return self._has_reached_upright_zone & (torch.abs(pole_error) > drop_threshold)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._has_reached_upright_zone[env_ids] = False


class terminate_after_pole_one_way_rotation:
  """Terminate if the pole rotates 360 degrees without reversing direction."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._last_pole_angle = torch.full(
      (env.num_envs,), float("nan"), device=env.device
    )
    self._rotation_accum = torch.zeros(env.num_envs, device=env.device)
    self._rotation_direction = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    pole_cfg: SceneEntityCfg = _POLE_CFG,
    rotation_limit_rad: float = _POLE_ONE_WAY_ROTATION_LIMIT,
    min_delta_rad: float = math.radians(0.1),
  ) -> torch.Tensor:
    asset: Entity = env.scene[pole_cfg.name]
    pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
    has_last = torch.isfinite(self._last_pole_angle)
    delta = torch.atan2(
      torch.sin(pole_angle - self._last_pole_angle),
      torch.cos(pole_angle - self._last_pole_angle),
    )
    delta = torch.where(has_last, delta, torch.zeros_like(delta))
    moving = torch.abs(delta) > float(min_delta_rad)
    direction = torch.sign(delta)
    direction_changed = (
      moving
      & (self._rotation_direction != 0.0)
      & (direction != self._rotation_direction)
    )

    next_accum = torch.where(direction_changed, delta, self._rotation_accum + delta)
    self._rotation_accum = torch.where(moving, next_accum, self._rotation_accum)
    self._rotation_direction = torch.where(
      moving,
      direction,
      self._rotation_direction,
    )
    self._last_pole_angle = pole_angle
    return torch.abs(self._rotation_accum) >= float(rotation_limit_rad)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._last_pole_angle[env_ids] = float("nan")
    self._rotation_accum[env_ids] = 0.0
    self._rotation_direction[env_ids] = 0.0


def _make_env_cfg() -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "cylinder_angle": ObservationTermCfg(
      func=scaled_joint_pos,
      params={"asset_cfg": _CYLINDER_CFG, "scale": _MAX_CYLINDER_ROTATION},
      clip=(-1.5, 1.5),
    ),
    "cylinder_angle_periodic": ObservationTermCfg(
      func=cylinder_angle_cos_sin,
      params={"asset_cfg": _CYLINDER_CFG},
    ),
    "pole_angle": ObservationTermCfg(
      func=pole_angle_cos_sin,
      params={"asset_cfg": _POLE_CFG},
    ),
    "joint_vel": ObservationTermCfg(
      func=scaled_joint_vel,
      params={"asset_cfg": _JOINTS_CFG},
      clip=(-5.0, 5.0),
    ),
    "last_action": ObservationTermCfg(
      func=last_action,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      actor_terms,
      enable_corruption=True,
      history_length=2,
    ),
    "critic": ObservationGroupCfg({**actor_terms}, history_length=2),
  }

  actions: dict[str, ActionTermCfg] = {
    "position": RateLimitedJointPositionActionCfg(
      entity_name="inverse",
      actuator_names=("Revolute 3",),
      scale=math.radians(20.0),
      max_delta=_CYLINDER_TARGET_DELTA_LIMIT,
      max_delta_change=_CYLINDER_TARGET_DELTA_CHANGE_LIMIT,
    ),
  }

  events = {
    "reset_curriculum": EventTermCfg(
      func=reset_swingup_balance_curriculum,
      mode="reset",
      params={
        "asset_cfg": _JOINTS_CFG,
        "balance_start_probability": _BALANCE_START_PROBABILITY,
        "cylinder_position_range": _CYLINDER_START_POSITION_RANGE,
        "cylinder_velocity_range": (-0.5, 0.5),
        "swingup_pole_position_range": (-math.pi, math.pi),
        "swingup_pole_velocity_range": (-1.0, 1.0),
        "balance_pole_position_range": (
          math.pi - math.radians(20.0),
          math.pi + math.radians(20.0),
        ),
        "balance_pole_velocity_range": (-0.5, 0.5),
      },
    ),
    "pole_com_vertical_randomization": EventTermCfg(
      func=dr.body_com_offset,
      mode="reset",
      params={
        "asset_cfg": _POLE_INERTIA_BODY_CFG,
        "ranges": _POLE_COM_VERTICAL_OFFSET_RANGE,
        "operation": "add",
        "axes": [2],
      },
    ),
    "pole_inertia_disturbance": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="reset",
      params={
        "asset_cfg": _POLE_INERTIA_BODY_CFG,
        # 에피소드마다 BoldHolder_1의 질량/관성 특성을 다시 샘플링해서
        # 조립 오차, CAD 오차, 실제 부품 편차에 더 강하게 학습시킨다.
        "alpha_range": (-0.10, 0.10),
        "d_range": (-0.06, 0.06),
        "t1_range": (-0.002, 0.002),
        "t2_range": (-0.002, 0.002),
        "t3_range": (-0.006, 0.006),
      },
    ),
    "pole_joint_damping_randomization": EventTermCfg(
      func=dr.joint_damping,
      mode="reset",
      params={
        "asset_cfg": _POLE_CFG,
        "ranges": _JOINT5_RANDOMIZATION_SCALE,
        "operation": "scale",
      },
    ),
    "pole_joint_frictionloss_randomization": EventTermCfg(
      func=dr.dof_frictionloss,
      mode="reset",
      params={
        "asset_cfg": _POLE_CFG,
        "ranges": _JOINT5_RANDOMIZATION_SCALE,
        "operation": "scale",
      },
    ),
    "pole_joint_armature_randomization": EventTermCfg(
      func=dr.joint_armature,
      mode="reset",
      params={
        "asset_cfg": _POLE_CFG,
        "ranges": _JOINT5_RANDOMIZATION_SCALE,
        "operation": "scale",
      },
    ),
  }

  rewards = {
    "pole_upright": RewardTermCfg(
      func=pole_upright_reward,
      weight=9.0,
      params={"pole_cfg": _POLE_CFG},
    ),
    "coordinated_swing": RewardTermCfg(
      func=coordinated_swing_reward,
      weight=3.0,
      params={"pole_cfg": _POLE_CFG, "cylinder_cfg": _CYLINDER_CFG},
    ),
    "pole_hold_30deg": RewardTermCfg(
      func=pole_hold_30deg_reward,
      weight=0.0,
      params={
        "pole_cfg": _POLE_CFG,
        "hold_threshold_rad": _BALANCE_HOLD_THRESHOLD,
      },
    ),
    "exact_upright_bonus": RewardTermCfg(
      func=exact_upright_bonus,
      weight=80.0,
      params={
        "pole_cfg": _POLE_CFG,
        "active_threshold_rad": _EXACT_UPRIGHT_BONUS_THRESHOLD,
      },
    ),
    "upper_region_pole_speed": RewardTermCfg(
      func=upper_region_pole_speed_penalty,
      weight=-24.0,
      params={
        "pole_cfg": _POLE_CFG,
        "upper_threshold_rad": _UPPER_SPEED_PENALTY_THRESHOLD,
        "speed_scale": _LOWER_SWING_SPEED_TARGET,
      },
    ),
    "near_upright_pole_speed": RewardTermCfg(
      func=near_upright_pole_speed_penalty,
      weight=-36.0,
      params={
        "pole_cfg": _POLE_CFG,
        "near_threshold_rad": _NEAR_UPRIGHT_SPEED_PENALTY_THRESHOLD,
        "speed_scale": _NEAR_UPRIGHT_SPEED_TARGET,
      },
    ),
  }

  metrics = {
    "max_pole_lift_angle_deg": MetricsTermCfg(
      func=pole_lift_angle_deg_metric,
      params={"pole_cfg": _POLE_CFG},
      reduce="max",
    ),
    "max_pole_upright_raw": MetricsTermCfg(
      func=pole_upright_reward,
      params={"pole_cfg": _POLE_CFG},
      reduce="max",
    ),
    "max_pole_upright_weighted": MetricsTermCfg(
      func=pole_upright_weighted_metric,
      params={"pole_cfg": _POLE_CFG},
      reduce="max",
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "cylinder_rotation_limit": TerminationTermCfg(
      func=cylinder_rotation_limit,
      params={"cylinder_cfg": _CYLINDER_CFG},
    ),
    "cylinder_speed_limit": TerminationTermCfg(
      func=cylinder_speed_limit,
      params={"cylinder_cfg": _CYLINDER_CFG},
    ),
    "cylinder_chatter_limit": TerminationTermCfg(
      func=cylinder_chatter_limit,
      params={
        "cylinder_cfg": _CYLINDER_CFG,
        "pole_cfg": _POLE_CFG,
        "angle_range_rad": _CYLINDER_CHATTER_ANGLE_RANGE,
        "reversals_per_sec": _CYLINDER_CHATTER_REVERSALS_PER_SEC,
        "upright_exemption_rad": _CYLINDER_CHATTER_UPRIGHT_EXEMPTION,
      },
    ),
    "pole_one_way_rotation_limit": TerminationTermCfg(
      func=terminate_after_pole_one_way_rotation,
      params={
        "pole_cfg": _POLE_CFG,
        "rotation_limit_rad": _POLE_ONE_WAY_ROTATION_LIMIT,
      },
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
    metrics=metrics,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.011, 0.0, -0.028),
      distance=0.35,
      elevation=-15.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005, disableflags=("contact",)),
    ),
    decimation=4,
    episode_length_s=12.0,
  )


def inverse_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg()
  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_curriculum"].params["balance_start_probability"] = 1.0
    cfg.events["reset_curriculum"].params["cylinder_position_range"] = (0.0, 0.0)
    cfg.events["reset_curriculum"].params["cylinder_velocity_range"] = (0.0, 0.0)
    cfg.events["reset_curriculum"].params["balance_pole_position_range"] = (
      math.pi,
      math.pi,
    )
    cfg.events["reset_curriculum"].params["balance_pole_velocity_range"] = (0.0, 0.0)
    if not _env_flag(_RANDOMIZED_PLAY_ENV_VAR):
      for event_name in (
        "pole_com_vertical_randomization",
        "pole_inertia_disturbance",
        "pole_joint_damping_randomization",
        "pole_joint_frictionloss_randomization",
        "pole_joint_armature_randomization",
      ):
        cfg.events.pop(event_name, None)
  return cfg


def inverse_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    experiment_name="inverse_balance",
    save_interval=50,
    num_steps_per_env=64,
    max_iterations=2000,
  )
