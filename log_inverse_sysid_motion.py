#!/usr/bin/env python3
from __future__ import annotations

import csv
from contextlib import nullcontext
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro

from mjlab.tasks.inverse.inverse_sysid_profiles import available_profiles, build_profile


DEFAULT_XML = Path("src/mjlab/tasks/inverse/assets/inverse.xml")
DEFAULT_LOG_DIR = Path("src/mjlab/tasks/inverse/sysid_logs")


@dataclass(frozen=True)
class LogInverseSysidMotionConfig:
  profile_name: str = "all"
  xml_file: str = str(DEFAULT_XML)
  log_dir: str = str(DEFAULT_LOG_DIR)
  csv_tag: str = ""
  move_time_s: float = 1.0
  hold_time_s: float = 1.0
  start_hold_s: float = 0.5
  profile_mode: str = "step"
  waypoint_hold_s: float = 0.25
  use_360_waypoints: bool = False
  sine_amplitude_deg: float | None = None
  sine_frequency_hz: float | None = None
  sine_duration_s: float | None = None
  sine_phase_deg: float = 0.0
  duration_s: float | None = None
  initial_pole_deg: float = 0.0
  ctrl_limit_deg: float = 720.0
  actuator_kp: float | None = 5.0
  actuator_kv: float | None = 1.0
  actuator_force_limit_nm: float | None = 8.0
  print_every: int = 100
  plot: bool = True
  plot_dpi: int = 160
  render: bool = False
  realtime: bool = True


CSV_HEADER = [
  "time_s",
  "profile_name",
  "target_rad",
  "target_deg",
  "target_vel_rad_s",
  "target_vel_deg_s",
  "cylinder_q_rad",
  "cylinder_q_deg",
  "cylinder_qd_rad_s",
  "cylinder_qd_deg_s",
  "tracking_error_rad",
  "tracking_error_deg",
  "pole_q_rad",
  "pole_q_deg",
  "pole_lift_deg",
  "pole_qd_rad_s",
  "pole_qd_deg_s",
  "actuator_force_nm",
  "qfrc_actuator_nm",
]


def _deg(rad: float) -> float:
  return math.degrees(rad)


def _wrap_deg(deg: float) -> float:
  return (deg + 180.0) % 360.0 - 180.0


def _pole_lift_deg(pole_q_rad: float) -> float:
  return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(pole_q_rad)))))


def _joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if joint_id < 0:
    raise ValueError(f"Joint not found: {joint_name}")
  return int(model.jnt_qposadr[joint_id])


def _joint_qvel_addr(model: mujoco.MjModel, joint_name: str) -> int:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if joint_id < 0:
    raise ValueError(f"Joint not found: {joint_name}")
  return int(model.jnt_dofadr[joint_id])


def _actuator_id(model: mujoco.MjModel, actuator_name: str) -> int:
  actuator_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
  )
  if actuator_id < 0:
    raise ValueError(f"Actuator not found: {actuator_name}")
  return int(actuator_id)


def _override_position_actuator(
  model: mujoco.MjModel,
  actuator: int,
  *,
  kp: float | None,
  kv: float | None,
  force_limit_nm: float | None,
) -> None:
  if kp is not None:
    model.actuator_gainprm[actuator, 0] = float(kp)
    model.actuator_biasprm[actuator, 1] = -float(kp)
  if kv is not None:
    model.actuator_biasprm[actuator, 2] = -float(kv)
  if force_limit_nm is not None:
    limit = abs(float(force_limit_nm))
    model.actuator_forcelimited[actuator] = 1
    model.actuator_forcerange[actuator, 0] = -limit
    model.actuator_forcerange[actuator, 1] = limit


def _viewer_context(model: mujoco.MjModel, data: mujoco.MjData, enabled: bool) -> Any:
  if not enabled:
    return nullcontext(None)

  import mujoco.viewer

  return mujoco.viewer.launch_passive(model, data)


def _run_profile(cfg: LogInverseSysidMotionConfig, profile_name: str) -> Path:
  xml_path = Path(cfg.xml_file)
  if not xml_path.exists():
    raise FileNotFoundError(f"XML not found: {xml_path}")

  profile = build_profile(
    profile_name,
    move_time_s=float(cfg.move_time_s),
    hold_time_s=float(cfg.hold_time_s),
    start_hold_s=float(cfg.start_hold_s),
    mode=cfg.profile_mode,
    waypoint_hold_s=float(cfg.waypoint_hold_s),
    use_360_waypoints=bool(cfg.use_360_waypoints),
    sine_amplitude_deg=cfg.sine_amplitude_deg,
    sine_frequency_hz=cfg.sine_frequency_hz,
    sine_duration_s=cfg.sine_duration_s,
    sine_phase_deg=float(cfg.sine_phase_deg),
  )
  duration_s = float(cfg.duration_s) if cfg.duration_s is not None else profile.duration_s

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)

  cylinder_qpos = _joint_qpos_addr(model, "Revolute 3")
  cylinder_qvel = _joint_qvel_addr(model, "Revolute 3")
  pole_qpos = _joint_qpos_addr(model, "Revolute 5")
  pole_qvel = _joint_qvel_addr(model, "Revolute 5")
  actuator = _actuator_id(model, "position_revolute_3")

  _override_position_actuator(
    model,
    actuator,
    kp=cfg.actuator_kp,
    kv=cfg.actuator_kv,
    force_limit_nm=cfg.actuator_force_limit_nm,
  )

  ctrl_limit_rad = math.radians(float(cfg.ctrl_limit_deg))
  model.actuator_ctrlrange[actuator, 0] = -ctrl_limit_rad
  model.actuator_ctrlrange[actuator, 1] = ctrl_limit_rad

  data.qpos[cylinder_qpos] = 0.0
  data.qpos[pole_qpos] = math.radians(float(cfg.initial_pole_deg))
  data.qvel[cylinder_qvel] = 0.0
  data.qvel[pole_qvel] = 0.0
  mujoco.mj_forward(model, data)

  log_dir = Path(cfg.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  tag = f"_{cfg.csv_tag}" if cfg.csv_tag else ""
  csv_path = log_dir / f"inverse_sysid_{profile.name}{tag}_{stamp}.csv"

  dt = float(model.opt.timestep)
  steps = int(np.ceil(duration_s / dt))
  last_target_rad = profile.target_rad(0.0)

  print(f"[INFO] XML: {xml_path}")
  print(f"[INFO] CSV: {csv_path}")
  print(
    f"[INFO] profile={profile.name} mode={cfg.profile_mode} "
    f"duration={duration_s:.3f}s dt={dt:.6f}s steps={steps} "
    f"start_hold={cfg.start_hold_s:.3f}s hold={cfg.hold_time_s:.3f}s "
    f"move_time={cfg.move_time_s:.3f}s waypoint_hold={cfg.waypoint_hold_s:.3f}s "
    f"use_360_waypoints={cfg.use_360_waypoints} "
    f"sine_amp={cfg.sine_amplitude_deg}deg sine_freq={cfg.sine_frequency_hz}Hz "
    f"ctrl_limit=+/-{cfg.ctrl_limit_deg:.1f}deg "
    f"actuator_kp={model.actuator_gainprm[actuator, 0]:.3f} "
    f"actuator_kv={-model.actuator_biasprm[actuator, 2]:.3f} "
    f"force_range=[{model.actuator_forcerange[actuator, 0]:.3f}, "
    f"{model.actuator_forcerange[actuator, 1]:.3f}]Nm"
  )

  with csv_path.open("w", newline="") as f, _viewer_context(model, data, cfg.render) as viewer:
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)
    wall_start_s = time.perf_counter()

    for step in range(steps + 1):
      time_s = step * dt
      target_rad = profile.target_rad(time_s)
      target_vel_rad_s = (target_rad - last_target_rad) / dt if step > 0 else 0.0
      last_target_rad = target_rad

      data.ctrl[actuator] = target_rad
      mujoco.mj_step(model, data)
      if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
          print("[INFO] viewer closed; stopping this profile.")
          break
        if cfg.realtime:
          target_wall_s = wall_start_s + time_s
          sleep_s = target_wall_s - time.perf_counter()
          if sleep_s > 0.0:
            time.sleep(sleep_s)

      cylinder_q = float(data.qpos[cylinder_qpos])
      cylinder_qd = float(data.qvel[cylinder_qvel])
      pole_q = float(data.qpos[pole_qpos])
      pole_qd = float(data.qvel[pole_qvel])
      error_rad = target_rad - cylinder_q

      row = [
        f"{time_s:.6f}",
        profile.name,
        f"{target_rad:.10f}",
        f"{_deg(target_rad):.6f}",
        f"{target_vel_rad_s:.10f}",
        f"{_deg(target_vel_rad_s):.6f}",
        f"{cylinder_q:.10f}",
        f"{_deg(cylinder_q):.6f}",
        f"{cylinder_qd:.10f}",
        f"{_deg(cylinder_qd):.6f}",
        f"{error_rad:.10f}",
        f"{_deg(error_rad):.6f}",
        f"{pole_q:.10f}",
        f"{_deg(pole_q):.6f}",
        f"{_pole_lift_deg(pole_q):.6f}",
        f"{pole_qd:.10f}",
        f"{_deg(pole_qd):.6f}",
        f"{float(data.actuator_force[actuator]):.10f}",
        f"{float(data.qfrc_actuator[cylinder_qvel]):.10f}",
      ]
      writer.writerow(row)

      if step % max(1, int(cfg.print_every)) == 0:
        print(
          f"[{step:05d}] t={time_s:.3f}s "
          f"target={_deg(target_rad):+.2f}deg "
          f"q={_deg(cylinder_q):+.2f}deg "
          f"err={_deg(error_rad):+.2f}deg "
          f"pole={_wrap_deg(_deg(pole_q)):+.2f}deg "
          f"tau={float(data.qfrc_actuator[cylinder_qvel]):+.4f}Nm"
        )

  print(f"[INFO] CSV saved: {csv_path}")
  return csv_path


def run(cfg: LogInverseSysidMotionConfig) -> list[Path]:
  profile_names = available_profiles() if cfg.profile_name == "all" else (cfg.profile_name,)
  csv_paths = []
  for profile_name in profile_names:
    csv_paths.append(_run_profile(cfg, profile_name))
  print("[INFO] Saved CSV files:")
  for path in csv_paths:
    print(f"  {path}")
  if cfg.plot:
    from mjlab.tasks.inverse.plot_inverse_sysid_motion import (
      PlotInverseSysidMotionConfig,
      run as plot_sysid_motion,
    )

    plot_sysid_motion(
      PlotInverseSysidMotionConfig(
        csv_files=[str(path) for path in csv_paths],
        dpi=int(cfg.plot_dpi),
      )
    )
  return csv_paths


if __name__ == "__main__":
  run(tyro.cli(LogInverseSysidMotionConfig))
