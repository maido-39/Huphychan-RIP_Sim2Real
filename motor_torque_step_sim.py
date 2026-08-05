#!/usr/bin/env python3
"""Run a repeated positive/negative torque sequence on the motor-only model.

The actuator in ``inverse_motor_only.xml`` is a position actuator because the
RL task uses position commands.  This script converts that actuator to a motor
in the in-memory ``MjSpec`` only; the shared XML file is not modified.

Run from the repository root:

  uv run src/mjlab/tasks/inverse/motor_torque_step_sim.py
"""

from __future__ import annotations

import csv
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro

_DEFAULT_XML = Path(__file__).parent / "assets" / "inverse_motor_only.xml"
_DEFAULT_LOG_DIR = Path(__file__).parent / "torque_step_logs"
_ACTUATOR_NAME = "position_revolute_3"
_JOINT_NAME = "Revolute 3"
_PHYSICS_DT_S = 0.005


@dataclass(frozen=True)
class MotorTorqueStepConfig:
  """Configuration for the open-loop torque step."""

  torque_nm: float = 1.0
  """Step torque in N*m."""

  segment_duration_s: float = 1.0
  """Duration of each zero, positive, zero, and negative segment."""

  repeats: int = 2
  """Number of times to repeat the four-segment sequence."""

  xml_file: str = str(_DEFAULT_XML)
  log_dir: str = str(_DEFAULT_LOG_DIR)
  render: bool = True
  realtime: bool = True
  plot: bool = True


def _load_torque_model(xml_path: Path) -> mujoco.MjModel:
  """Load the model and replace its position actuator in memory with a motor."""
  spec = mujoco.MjSpec.from_file(str(xml_path))
  actuator = spec.actuator(_ACTUATOR_NAME)
  if actuator is None:
    raise ValueError(f"Actuator not found: {_ACTUATOR_NAME}")

  actuator.set_to_motor()
  actuator.target = "Revolute 3"
  actuator.ctrllimited = True
  actuator.ctrlrange[:] = np.array([-17.0, 17.0])
  actuator.forcelimited = True
  actuator.forcerange[:] = np.array([-17.0, 17.0])

  model = spec.compile()
  model.opt.timestep = _PHYSICS_DT_S
  return model


def _joint_addresses(model: mujoco.MjModel) -> tuple[int, int]:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _JOINT_NAME)
  if joint_id < 0:
    raise ValueError(f"Joint not found: {_JOINT_NAME}")
  return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _viewer_context(model: mujoco.MjModel, data: mujoco.MjData, enabled: bool) -> Any:
  if not enabled:
    return nullcontext(None)

  import mujoco.viewer

  return mujoco.viewer.launch_passive(model, data)


def _plot_log(csv_path: Path) -> Path:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  data = np.genfromtxt(csv_path, delimiter=",", names=True)
  figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  axes[0].step(data["time_s"], data["command_torque_nm"], where="post", label="command")
  axes[0].plot(data["time_s"], data["actuator_torque_nm"], label="actuator")
  axes[0].set_ylabel("torque (N m)")
  axes[0].legend()
  axes[0].grid(True)

  axes[1].plot(data["time_s"], data["motor_angle_deg"])
  axes[1].set_ylabel("angle (deg)")
  axes[1].grid(True)

  axes[2].plot(data["time_s"], data["motor_velocity_deg_s"])
  axes[2].set_ylabel("velocity (deg/s)")
  axes[2].set_xlabel("time (s)")
  axes[2].grid(True)

  figure.tight_layout()
  png_path = csv_path.with_suffix(".png")
  figure.savefig(png_path, dpi=160)
  plt.close(figure)
  return png_path


def run(cfg: MotorTorqueStepConfig) -> Path:
  if cfg.segment_duration_s <= 0.0:
    raise ValueError("segment_duration_s must be positive")
  if cfg.repeats <= 0:
    raise ValueError("repeats must be positive")
  if abs(cfg.torque_nm) > 17.0:
    raise ValueError("torque_nm must be within the model limit [-17, 17] N*m")

  xml_path = Path(cfg.xml_file)
  if not xml_path.exists():
    raise FileNotFoundError(f"XML not found: {xml_path}")

  model = _load_torque_model(xml_path)
  data = mujoco.MjData(model)
  actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, _ACTUATOR_NAME)
  if actuator_id < 0:
    raise ValueError(f"Actuator not found after compilation: {_ACTUATOR_NAME}")
  qpos_address, qvel_address = _joint_addresses(model)

  segment_steps = int(round(cfg.segment_duration_s / model.opt.timestep))
  num_steps = 4 * cfg.repeats * segment_steps
  total_duration_s = num_steps * model.opt.timestep
  segment_commands = (0.0, cfg.torque_nm, 0.0, -cfg.torque_nm)

  log_dir = Path(cfg.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = log_dir / f"torque_sequence_{stamp}.csv"

  wall_start_s = time.perf_counter()

  print(
    f"[INFO] sequence: 0 -> +{cfg.torque_nm:g} -> 0 -> "
    f"-{cfg.torque_nm:g} N*m, {cfg.segment_duration_s:g} s each, "
    f"{cfg.repeats} repeats ({total_duration_s:g} s total)"
  )

  with (
    csv_path.open("w", newline="") as log_file,
    _viewer_context(model, data, cfg.render) as viewer,
  ):
    writer = csv.writer(log_file)
    writer.writerow(
      (
        "time_s",
        "command_torque_nm",
        "actuator_torque_nm",
        "motor_angle_deg",
        "motor_velocity_deg_s",
      )
    )

    for step in range(num_steps):
      segment = (step // segment_steps) % len(segment_commands)
      command_nm = segment_commands[segment]
      data.ctrl[actuator_id] = command_nm

      if step % segment_steps == 0:
        print(f"[INFO] t={step * model.opt.timestep:.3f} s: {command_nm:+.3f} N*m")

      mujoco.mj_step(model, data)
      writer.writerow(
        (
          f"{data.time:.6f}",
          f"{command_nm:.6f}",
          f"{float(data.actuator_force[actuator_id]):.6f}",
          f"{math.degrees(float(data.qpos[qpos_address])):.6f}",
          f"{math.degrees(float(data.qvel[qvel_address])):.6f}",
        )
      )

      if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
          print("[INFO] viewer closed; stopping.")
          break

      if cfg.realtime:
        target_wall_s = wall_start_s + data.time
        sleep_s = target_wall_s - time.perf_counter()
        if sleep_s > 0.0:
          time.sleep(sleep_s)

  # Leave the final command explicitly at zero even if the viewer was closed early.
  data.ctrl[actuator_id] = 0.0
  print(f"[INFO] finished at t={data.time:.3f} s; final torque command is 0.000 N*m")
  print(f"[INFO] CSV: {csv_path}")

  if cfg.plot:
    png_path = _plot_log(csv_path)
    print(f"[INFO] plot: {png_path}")

  return csv_path


def main() -> None:
  run(tyro.cli(MotorTorqueStepConfig))


if __name__ == "__main__":
  main()
