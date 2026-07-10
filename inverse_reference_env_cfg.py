"""Plain Furuta-reference inverse pendulum task configuration.

This config intentionally keeps the task as close as possible to the
minimal Furuta sample reward:

    reward = alpha_reward * theta_reward

where:
    alpha_reward : raise the pendulum
    theta_reward : keep the motor arm near the origin

No balance curriculum, no sim2real disturbance, no hardware-protection
termination, no action penalty, no velocity penalty.
Only time_out is kept so PPO episodes can end normally.
"""

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


# Furuta sample-style reward scale.
# Reward itself is in [0, 1], so 10~12 정도면 PPO에서 보기 편함.
_FURUTA_REWARD_SCALE = 12.0


def furuta_alpha_reward(env, pole_cfg=_POLE_CFG) -> torch.Tensor:
  """Pendulum swing-up reward.

  Assumption:
    pole_angle = 0      -> hanging down
    pole_angle = pi     -> upright

  Reward:
    down    -> 0
    upright -> 1
  """
  asset: Entity = env.scene[pole_cfg.name]
  alpha = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)

  return 0.5 * (1.0 - torch.cos(alpha))


def furuta_theta_reward(env, cylinder_cfg=_CYLINDER_CFG) -> torch.Tensor:
  """Motor arm centering reward.

  This follows the Furuta sample-style theta reward.

  theta = 0      -> reward near 1
  theta = ±pi    -> reward near 0

  It discourages solving the task by spinning the motor arm far away
  from the origin.
  """
  asset: Entity = env.scene[cylinder_cfg.name]
  theta = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)

  theta_rew = 0.5 * (torch.cos(theta + math.pi) + 1.0)
  return 1.0 - theta_rew.square()


def furuta_alpha_theta_reward(
  env,
  pole_cfg=_POLE_CFG,
  cylinder_cfg=_CYLINDER_CFG,
) -> torch.Tensor:
  """Plain Furuta reference reward.

  reward = pole swing-up reward * motor centering reward
  """
  alpha_rew = furuta_alpha_reward(env, pole_cfg)
  theta_rew = furuta_theta_reward(env, cylinder_cfg)

  return alpha_rew * theta_rew


def inverse_reference_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Plain Furuta-reference task.

  Keeps:
    - base MuJoCo model
    - base action config
    - base observation config
    - reset_curriculum
    - time_out only

  Removes:
    - balance-start curriculum
    - sim2real disturbances
    - pole clear/drop termination
    - cylinder rotation limit
    - cylinder speed limit
    - cylinder chatter limit
    - all extra reward terms
  """
  cfg = _make_env_cfg()

  # ---------------------------------------------------------------------------
  # Reset: plain Furuta sample style
  # ---------------------------------------------------------------------------
  # Start near the hanging-down equilibrium.
  # No upright/balance starts.
  reset_event = cfg.events["reset_curriculum"]
  cfg.events = {
    "reset_curriculum": reset_event,
  }

  cfg.events["reset_curriculum"].params["balance_start_probability"] = 0.0
  cfg.events["reset_curriculum"].params["cylinder_position_range"] = (-0.01, 0.01)
  cfg.events["reset_curriculum"].params["cylinder_velocity_range"] = (-0.01, 0.01)
  cfg.events["reset_curriculum"].params["swingup_pole_position_range"] = (-0.01, 0.01)
  cfg.events["reset_curriculum"].params["swingup_pole_velocity_range"] = (-0.01, 0.01)

  # ---------------------------------------------------------------------------
  # Reward: only Furuta alpha-theta reward
  # ---------------------------------------------------------------------------
  cfg.rewards = {
    "furuta_alpha_theta": RewardTermCfg(
      func=furuta_alpha_theta_reward,
      weight=_FURUTA_REWARD_SCALE,
      params={
        "pole_cfg": _POLE_CFG,
        "cylinder_cfg": _CYLINDER_CFG,
      },
    ),
  }

  # ---------------------------------------------------------------------------
  # Termination: remove all hardware/safety constraints
  # ---------------------------------------------------------------------------
  # PPO still needs finite episodes, so keep only time_out.
  # This removes:
  #   - cylinder_rotation_limit
  #   - cylinder_speed_limit
  #   - cylinder_chatter_limit
  #   - pole_clear_drop
  #   - anything else inherited from _make_env_cfg()
  time_out_term = cfg.terminations.get("time_out", None)
  cfg.terminations = {}

  if time_out_term is not None:
    cfg.terminations["time_out"] = time_out_term

  # ---------------------------------------------------------------------------
  # Play mode
  # ---------------------------------------------------------------------------
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
  """Plain PPO runner for the Furuta-reference task."""
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