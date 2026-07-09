"""Reference-style inverse pendulum task configuration."""

from __future__ import annotations

import math

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.inverse.inverse_env_cfg import (
  _CYLINDER_CFG,
  _POLE_CFG,
  _make_env_cfg,
)

_REFERENCE_REWARD_SCALE = 12.0


def reference_theta_reward(env, cylinder_cfg=_CYLINDER_CFG) -> torch.Tensor:
  """Reference-like reward that prefers keeping the motor angle near the origin."""
  asset: Entity = env.scene[cylinder_cfg.name]
  theta = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)
  theta_rew = 0.5 * (torch.cos(theta + math.pi) + 1.0)
  return 1.0 - theta_rew.square()


def reference_alpha_theta_reward(
  env,
  pole_cfg=_POLE_CFG,
  cylinder_cfg=_CYLINDER_CFG,
) -> torch.Tensor:
  """Minimal reference-style swing-up objective: raise pole while centering motor angle."""
  asset: Entity = env.scene[pole_cfg.name]
  pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)
  alpha_reward = 0.5 * (1.0 - torch.cos(pole_angle))
  theta_reward = reference_theta_reward(env, cylinder_cfg)
  return alpha_reward * theta_reward


def inverse_reference_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Reference reward task plus only the hardware-protection constraints."""
  cfg = _make_env_cfg()

  # Keep the reference task reset distribution: start from the hanging swing-up
  # region, without balance starts or sim2real disturbances.
  cfg.events["reset_curriculum"].params["balance_start_probability"] = 0.0
  cfg.events["reset_curriculum"].params["cylinder_position_range"] = (-0.01, 0.01)
  cfg.events["reset_curriculum"].params["cylinder_velocity_range"] = (-0.01, 0.01)
  cfg.events["reset_curriculum"].params["swingup_pole_position_range"] = (-0.01, 0.01)
  cfg.events["reset_curriculum"].params["swingup_pole_velocity_range"] = (-0.01, 0.01)

  cfg.events.pop("pole_inertia_disturbance", None)
  cfg.events.pop("pole_tip_disturbance", None)

  cfg.rewards = {
    "alpha_theta": RewardTermCfg(
      func=reference_alpha_theta_reward,
      weight=_REFERENCE_REWARD_SCALE,
      params={"pole_cfg": _POLE_CFG, "cylinder_cfg": _CYLINDER_CFG},
    ),
  }

  # Keep only physical safety termination from _make_env_cfg():
  # - time_out
  # - cylinder_rotation_limit
  # - cylinder_speed_limit
  cfg.terminations.pop("pole_clear_drop", None)

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


def inverse_reference_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    experiment_name="inverse_reference",
    save_interval=50,
    num_steps_per_env=64,
    max_iterations=2000,
  )
