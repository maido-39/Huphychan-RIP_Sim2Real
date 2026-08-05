"""ActuatorNet-backed motor actuator for the inverse balance task.

This adapter loads the trained weights and feature-normalization metadata and encodes
the remaining deployment assumptions:

- kp=20.0, kd=0.8
- 200 Hz actuator update rate
- six samples of history, oldest to newest
- commanded target velocity fixed at zero
- motor and target velocities scaled by 1/1000
- input standardized with metadata x_mean/x_std; direct output is already N*m
- history initialized by repeating the first sample
- output torque clipped to +/-17 N*m
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.tasks.inverse.actuator_net_model import ActuatorNet

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class InverseActuatorNetCfg(ActuatorCfg):
  """Configuration for the inverse task's learned real-motor torque model."""

  network_file: str
  metadata_file: str
  stiffness: float = 20.0
  damping: float = 0.8
  history_length: int = 6
  hidden_dim: int = 64
  velocity_scale: float = 1.0 / 1000.0
  effort_limit: float = 17.0

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.stiffness < 0.0:
      raise ValueError("stiffness must be non-negative")
    if self.damping < 0.0:
      raise ValueError("damping must be non-negative")
    if self.history_length <= 0:
      raise ValueError("history_length must be positive")
    if self.hidden_dim <= 0:
      raise ValueError("hidden_dim must be positive")
    if self.effort_limit <= 0.0:
      raise ValueError("effort_limit must be positive")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> InverseActuatorNet:
    return InverseActuatorNet(self, entity, target_ids, target_names)


class InverseActuatorNet(Actuator[InverseActuatorNetCfg]):
  """Predict real motor torque from command and state histories."""

  def __init__(
    self,
    cfg: InverseActuatorNetCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.network: ActuatorNet | None = None
    self._target_torque_history: torch.Tensor | None = None
    self._motor_velocity_history: torch.Tensor | None = None
    self._target_velocity_history: torch.Tensor | None = None
    self._initialized: torch.Tensor | None = None
    self._last_target_torque: torch.Tensor | None = None
    self._x_mean: torch.Tensor | None = None
    self._x_std: torch.Tensor | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    """Convert the existing XML position actuator to a direct-torque motor."""
    for target_name in target_names:
      matches = [
        actuator for actuator in spec.actuators if actuator.target == target_name
      ]
      if len(matches) != 1:
        raise ValueError(
          f"Expected exactly one XML actuator for '{target_name}', found {len(matches)}"
        )

      actuator = matches[0]
      actuator.set_to_motor()
      actuator.target = target_name
      actuator.ctrllimited = True
      actuator.ctrlrange[:] = np.array([-self.cfg.effort_limit, self.cfg.effort_limit])
      actuator.forcelimited = True
      actuator.forcerange[:] = np.array([-self.cfg.effort_limit, self.cfg.effort_limit])
      self._mjs_actuators.append(actuator)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    input_dim = 3 * self.cfg.history_length
    metadata = json.loads(Path(self.cfg.metadata_file).read_text())
    if metadata.get("architecture") != "direct":
      raise ValueError("ActuatorNet metadata architecture must be 'direct'")
    if int(metadata.get("history", -1)) != self.cfg.history_length:
      raise ValueError("ActuatorNet metadata history does not match configuration")
    if int(metadata.get("hidden", -1)) != self.cfg.hidden_dim:
      raise ValueError("ActuatorNet metadata hidden size does not match configuration")
    if int(metadata.get("in_dim", -1)) != input_dim:
      raise ValueError(
        "ActuatorNet metadata input dimension does not match configuration"
      )

    x_mean = torch.tensor(metadata["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.tensor(metadata["x_std"], dtype=torch.float32, device=device)
    if x_mean.shape != (input_dim,) or x_std.shape != (input_dim,):
      raise ValueError(f"ActuatorNet normalization must contain {input_dim} values")
    if torch.any(x_std <= 0.0):
      raise ValueError("ActuatorNet x_std values must be positive")
    self._x_mean = x_mean
    self._x_std = x_std

    network = ActuatorNet(in_dim=input_dim, hidden=self.cfg.hidden_dim)
    state_dict = torch.load(
      Path(self.cfg.network_file),
      map_location=device,
      weights_only=True,
    )
    network.load_state_dict(state_dict, strict=True)
    network.to(device)
    network.eval()
    for parameter in network.parameters():
      parameter.requires_grad_(False)
    self.network = network

    shape = (data.nworld, len(self._target_names), self.cfg.history_length)
    self._target_torque_history = torch.zeros(shape, device=device)
    self._motor_velocity_history = torch.zeros(shape, device=device)
    self._target_velocity_history = torch.zeros(shape, device=device)
    self._initialized = torch.zeros(shape[:2], dtype=torch.bool, device=device)
    self._last_target_torque = torch.zeros(shape[:2], device=device)

  @staticmethod
  def _append_history(history: torch.Tensor, value: torch.Tensor) -> None:
    history[..., :-1] = history[..., 1:].clone()
    history[..., -1] = value

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.network is not None
    assert self._target_torque_history is not None
    assert self._motor_velocity_history is not None
    assert self._target_velocity_history is not None
    assert self._initialized is not None
    assert self._last_target_torque is not None
    assert self._x_mean is not None
    assert self._x_std is not None

    target_velocity = torch.zeros_like(cmd.position_target)
    target_torque = (
      self.cfg.stiffness * (cmd.position_target - cmd.pos)
      + self.cfg.damping * (target_velocity - cmd.vel)
      + cmd.effort_target
    )
    self._last_target_torque.copy_(target_torque)

    motor_velocity_scaled = torch.rad2deg(cmd.vel) * self.cfg.velocity_scale
    target_velocity_scaled = torch.rad2deg(target_velocity) * self.cfg.velocity_scale

    first_sample = ~self._initialized
    for history, value in (
      (self._target_torque_history, target_torque),
      (self._motor_velocity_history, motor_velocity_scaled),
      (self._target_velocity_history, target_velocity_scaled),
    ):
      history[first_sample] = (
        value[first_sample].unsqueeze(-1).expand(-1, self.cfg.history_length)
      )
      self._append_history(history, value)

    self._initialized.fill_(True)

    features = torch.cat(
      (
        self._target_torque_history,
        self._motor_velocity_history,
        self._target_velocity_history,
      ),
      dim=-1,
    )
    flat_features = features.reshape(-1, features.shape[-1])
    normalized_features = (flat_features - self._x_mean) / self._x_std
    with torch.inference_mode():
      predicted_torque = self.network(normalized_features)
    predicted_torque = predicted_torque.reshape(cmd.pos.shape)
    return torch.clamp(
      predicted_torque,
      -float(self.cfg.effort_limit),
      float(self.cfg.effort_limit),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    assert self._initialized is not None

    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False
