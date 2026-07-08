"""Inverse robot balance task configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_acc_l2,
  action_rate_l2,
  apply_body_impulse,
  last_action,
  time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
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

_TARGET_POLE_ANGLE = math.pi
_MAX_CYLINDER_ROTATION = 3.0 * math.pi
_CYLINDER_TERMINATION_ROTATION = 4.0 * math.pi
_VEL_OBS_SCALE = 30.0
_BALANCE_START_PROBABILITY = 0.2
_BALANCE_HOLD_THRESHOLD = math.radians(30.0)


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_INVERSE_XML))


_INVERSE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(target_names_expr=("Revolute 3",)),
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
  cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Reference-style swing-up reward with a soft cylinder centering factor."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  cylinder_angle = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  pole_reward = 0.5 * (torch.cos(pole_error) + 1.0)

  # Similar in spirit to the reference theta_reward: keep useful swing-up motion
  # while discouraging the motor axis from drifting far from the origin.
  normalized = torch.clamp(cylinder_angle / _MAX_CYLINDER_ROTATION, -1.0, 1.0)
  cylinder_reward = 1.0 - normalized.square()
  return pole_reward * cylinder_reward


def pole_balance_bonus(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
) -> torch.Tensor:
  """Sharp bonus only near the upright balance region."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  return torch.exp(-12.0 * pole_error.square())


def pole_hold_30deg_reward(
  env: ManagerBasedRlEnv,
  pole_cfg: SceneEntityCfg = _POLE_CFG,
) -> torch.Tensor:
  """Per-step reward for staying inside the practical +/-30 degree balance zone."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  pole_error = torch.atan2(
    torch.sin(pole_angle - _TARGET_POLE_ANGLE),
    torch.cos(pole_angle - _TARGET_POLE_ANGLE),
  )
  return (torch.abs(pole_error) < _BALANCE_HOLD_THRESHOLD).float()


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


def cylinder_center_reward(
  env: ManagerBasedRlEnv,
  cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Very soft preference for staying near the starting angle."""
  asset: Entity = env.scene[cylinder_cfg.name]
  cylinder_angle = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)
  normalized = cylinder_angle / _MAX_CYLINDER_ROTATION
  return torch.exp(-0.25 * normalized.square())


def cylinder_over_rotation(
  env: ManagerBasedRlEnv,
  cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Penalty after the actuated cylinder exceeds 1.5 turns in either direction."""
  asset: Entity = env.scene[cylinder_cfg.name]
  cylinder_angle = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)
  excess = torch.clamp(torch.abs(cylinder_angle) - _MAX_CYLINDER_ROTATION, min=0.0)
  return excess.square()


def cylinder_speed_l2(
  env: ManagerBasedRlEnv,
  cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Squared speed of the actuated cylinder."""
  asset: Entity = env.scene[cylinder_cfg.name]
  cylinder_vel = asset.data.joint_vel[:, cylinder_cfg.joint_ids].squeeze(-1)
  return cylinder_vel.square()


def cylinder_rotation_limit(
  env: ManagerBasedRlEnv,
  cylinder_cfg: SceneEntityCfg = _CYLINDER_CFG,
) -> torch.Tensor:
  """Terminate only after the cylinder goes well beyond the penalty region."""
  asset: Entity = env.scene[cylinder_cfg.name]
  cylinder_angle = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)
  return torch.abs(cylinder_angle) > _CYLINDER_TERMINATION_ROTATION


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
    "position": JointPositionActionCfg(
      entity_name="inverse",
      actuator_names=("Revolute 3",),
      scale=0.5 * math.pi,
      clip={"Revolute 3": (-0.5 * math.pi, 0.5 * math.pi)},
    ),
  }

  events = {
    "reset_curriculum": EventTermCfg(
      func=reset_swingup_balance_curriculum,
      mode="reset",
      params={
        "asset_cfg": _JOINTS_CFG,
        "balance_start_probability": _BALANCE_START_PROBABILITY,
      },
    ),
    "pole_tip_disturbance": EventTermCfg(
      func=apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": _POLE_BODY_CFG,
        "force_range": (-0.05, 0.05),
        "torque_range": (0.0, 0.0),
        "duration_s": (0.01, 0.03),
        "cooldown_s": (0.20, 0.60),
        "body_point_offset": (-0.0504, 0.0, -0.0228),
      },
    ),
  }

  rewards = {
    "pole_upright": RewardTermCfg(
      func=pole_upright_reward,
      weight=8.0,
      params={"pole_cfg": _POLE_CFG, "cylinder_cfg": _CYLINDER_CFG},
    ),
    "pole_balance_bonus": RewardTermCfg(
      func=pole_balance_bonus,
      weight=8.0,
      params={"pole_cfg": _POLE_CFG},
    ),
    "pole_hold_30deg": RewardTermCfg(
      func=pole_hold_30deg_reward,
      weight=4.0,
      params={"pole_cfg": _POLE_CFG},
    ),
    "pole_slow": RewardTermCfg(
      func=pole_slow_reward,
      weight=1.0,
      params={"pole_cfg": _POLE_CFG},
    ),
    "cylinder_center": RewardTermCfg(
      func=cylinder_center_reward,
      weight=0.03,
      params={"cylinder_cfg": _CYLINDER_CFG},
    ),
    "cylinder_over_rotation": RewardTermCfg(
      func=cylinder_over_rotation,
      weight=-0.1,
      params={"cylinder_cfg": _CYLINDER_CFG},
    ),
    "cylinder_speed": RewardTermCfg(
      func=cylinder_speed_l2,
      weight=-0.001,
      params={"cylinder_cfg": _CYLINDER_CFG},
    ),
    "action_rate": RewardTermCfg(
      func=action_rate_l2,
      weight=-0.006,
    ),
    "action_acc": RewardTermCfg(
      func=action_acc_l2,
      weight=-0.0015,
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "cylinder_rotation_limit": TerminationTermCfg(
      func=cylinder_rotation_limit,
      params={"cylinder_cfg": _CYLINDER_CFG},
    ),
    "pole_clear_drop": TerminationTermCfg(
      func=terminate_after_clear_drop,
      params={
        "pole_cfg": _POLE_CFG,
        "reached_threshold_deg": 60.0,
        "drop_threshold_deg": 60.0,
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
    decimation=1,
    episode_length_s=8.0,
  )


def inverse_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg()
  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_curriculum"].params["balance_start_probability"] = 0.0
    cfg.events["reset_curriculum"].params["cylinder_position_range"] = (0.0, 0.0)
    cfg.events["reset_curriculum"].params["cylinder_velocity_range"] = (0.0, 0.0)
    cfg.events["reset_curriculum"].params["swingup_pole_position_range"] = (0.0, 0.0)
    cfg.events["reset_curriculum"].params["swingup_pole_velocity_range"] = (0.0, 0.0)
  return cfg


def inverse_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
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
      entropy_coef=0.005,
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
    max_iterations=1000,
  )
