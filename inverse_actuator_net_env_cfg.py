"""Inverse balance task using the learned real-motor ActuatorNet."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.entity import EntityArticulationInfoCfg
from mjlab.tasks.inverse.actuator_net_actuator import InverseActuatorNetCfg
from mjlab.tasks.inverse.inverse_env_cfg import (
  inverse_balance_env_cfg,
  inverse_ppo_runner_cfg,
)

_ACTUATOR_NET_FILE = Path(__file__).parent / "actuator_net.pt"
_ACTUATOR_NET_META_FILE = Path(__file__).parent / "actuator_net_meta.json"


def inverse_actuator_net_env_cfg(play: bool = False):
  """Create the balance environment with ActuatorNet replacing XML position PD."""
  cfg = inverse_balance_env_cfg(play=play)
  inverse_entity_cfg = cfg.scene.entities["inverse"]
  inverse_entity_cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      InverseActuatorNetCfg(
        target_names_expr=("Revolute 3",),
        network_file=str(_ACTUATOR_NET_FILE),
        metadata_file=str(_ACTUATOR_NET_META_FILE),
        stiffness=20.0,
        damping=0.8,
        history_length=6,
        hidden_dim=64,
        velocity_scale=1.0 / 1000.0,
        effort_limit=17.0,
      ),
    ),
  )

  # This base-task event rewrites the XML position actuator's kv. ActuatorNet
  # predicts torque directly and must not receive that extra servo bias.
  cfg.events.pop("cylinder_joint_armature_randomization", None)
  return cfg


def inverse_actuator_net_ppo_runner_cfg():
  """Use the balance PPO settings with a separate experiment directory."""
  return replace(
    inverse_ppo_runner_cfg(),
    experiment_name="inverse_balance_actuator_net",
    logger="tensorboard",
  )
