#!/usr/bin/env python3
"""Open-loop sine-motion check for the inverse_sub motor, in simulation.

Drives Revolute 3 directly with a hand-computed sine trajectory (no RL
policy, no ManagerBasedRlEnv) and shows the passive pendulum's response
live in the MuJoCo viewer. Physics runs at 200 Hz and the target is
recomputed at 50 Hz (decimation=4), matching inverse_sine_env_cfg.py and
the real hardware runner, so what you see here matches what a deployed
policy would experience.

Run:
```bash
uv run python src/mjlab/tasks/inverse_sub/sine_motor_sim_test.py \
  --amplitude-deg 30 --frequency-hz 1.0 --duration-s 10
```
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

from mjlab.tasks.inverse_sub.sine_reference import (
  RateAccelLimiter,
  SineReferenceConfig,
  sine_target_rad,
)

_DEFAULT_XML = Path(__file__).parent / "assets" / "inverse.xml"
_DEFAULT_LOG_DIR = Path(__file__).parent / "sine_test_logs"

_PHYSICS_DT_S = 0.005
_DECIMATION = 4


@dataclass(frozen=True)
class SineMotorSimTestConfig:
  amplitude_deg: float = 30.0
  frequency_hz: float = 1.0
  phase_deg: float = 0.0
  duration_s: float = 10.0

  xml_file: str = str(_DEFAULT_XML)
  log_dir: str = str(_DEFAULT_LOG_DIR)

  # None keeps the assets/inverse.xml value; set these to try different
  # gains without editing the shared model file.
  actuator_kp: float | None = None
  actuator_kv: float | None = None
  # If set, actuator_kv is instead computed from actuator_kp and the
  # model's actual effective inertia at Revolute 3 so the closed-loop
  # response has this damping ratio (1.0 = critically damped). Requires
  # actuator_kp to also be set. Overrides actuator_kv.
  damping_ratio: float | None = None

  # Joint-level overrides for Revolute 3 (assets/inverse.xml currently has
  # armature=0.0017, damping=0.007073 -- both smaller than the "matched"
  # values sim2real_parameters.md documents: armature=0.0142, damping=0.05).
  joint_armature: float | None = None
  joint_damping: float | None = None

  # Detach the passive pendulum (BoldHolder_1 + Revolute 5) so Revolute 3
  # drives a bare rotor with no reflected pendulum dynamics -- isolates the
  # actuator's own tracking bandwidth from motor/pendulum coupling.
  no_pendulum: bool = False

  # inverse_sine_env_cfg.py's SimulationCfg sets disableflags=("contact",)
  # because the mesh collision geoms self-collide (rotor vs. housing) even
  # at rest. Default True to match the RL/real deployment setup; without
  # this, spurious contact forces of several Nm fight the actuator.
  disable_contact: bool = True

  render: bool = True
  realtime: bool = True
  print_every: int = 25
  plot: bool = True


_CSV_HEADER = [
  "time_s",
  "target_angle_deg",
  "limited_target_angle_deg",
  "motor_angle_deg",
  "motor_vel_deg_s",
  "pole_angle_deg",
  "pole_vel_deg_s",
  "actuator_force_nm",
]


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
  actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
  if actuator_id < 0:
    raise ValueError(f"Actuator not found: {actuator_name}")
  return int(actuator_id)


def _load_model(xml_path: Path, *, no_pendulum: bool) -> mujoco.MjModel:
  if not no_pendulum:
    return mujoco.MjModel.from_xml_path(str(xml_path))

  # Edit the spec tree (not the shared XML on disk): drop the pendulum
  # body so Revolute 3 drives a bare rotor with no reflected dynamics.
  spec = mujoco.MjSpec.from_file(str(xml_path))
  pendulum_body = spec.body("BoldHolder_1")
  spec.delete(pendulum_body)
  return spec.compile()


def _effective_dof_inertia(
  model: mujoco.MjModel, data: mujoco.MjData, dof_id: int
) -> float:
  """Diagonal entry of the full joint-space inertia matrix at data.qpos.

  Includes armature and any reflected inertia from child bodies (e.g. the
  pendulum), unlike naively reading a body's own diaginertia.
  """
  full_m = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, full_m)
  return float(full_m[dof_id, dof_id])


def _kv_for_damping_ratio(
  *, kp: float, inertia: float, damping_ratio: float, joint_damping: float
) -> float:
  """Position-actuator kv giving the requested closed-loop damping ratio.

  Treats the position actuator as a PD controller on a single DOF with
  inertia `inertia`: I*qdd + (kv + joint_damping)*qd + kp*q = kp*target,
  so omega_n = sqrt(kp/I) and kv = 2*damping_ratio*sqrt(kp*I) - joint_damping.
  """
  return max(2.0 * damping_ratio * math.sqrt(kp * inertia) - joint_damping, 0.0)


def _override_position_actuator(
  model: mujoco.MjModel,
  actuator: int,
  *,
  kp: float | None,
  kv: float | None,
) -> None:
  if kp is not None:
    model.actuator_gainprm[actuator, 0] = float(kp)
    model.actuator_biasprm[actuator, 1] = -float(kp)
  if kv is not None:
    model.actuator_biasprm[actuator, 2] = -float(kv)


def _viewer_context(model: mujoco.MjModel, data: mujoco.MjData, enabled: bool) -> Any:
  if not enabled:
    return nullcontext(None)

  import mujoco.viewer

  return mujoco.viewer.launch_passive(model, data)


def _plot_log(csv_path: Path) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib이 설치되어 있지 않아 그래프를 건너뜁니다. (pip install matplotlib)")
    return

  t_list, target_list, limited_list = [], [], []
  motor_list, motor_vel_list = [], []
  pole_list, pole_vel_list = [], []
  with csv_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      t_list.append(float(row["time_s"]))
      target_list.append(float(row["target_angle_deg"]))
      limited_list.append(float(row["limited_target_angle_deg"]))
      motor_list.append(float(row["motor_angle_deg"]))
      motor_vel_list.append(float(row["motor_vel_deg_s"]))
      pole_list.append(float(row["pole_angle_deg"]))
      pole_vel_list.append(float(row["pole_vel_deg_s"]))

  if not t_list:
    print("로그가 비어 있어 그래프를 건너뜁니다.")
    return

  motor_unwrapped = np.degrees(np.unwrap(np.radians(motor_list)))
  pole_unwrapped = np.degrees(np.unwrap(np.radians(pole_list)))

  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  axes[0].plot(t_list, target_list, label="target_angle_deg (sine)", linestyle="--")
  axes[0].plot(t_list, limited_list, label="action_deg (rate/accel-limited, sent to actuator)", linestyle=":")
  axes[0].plot(t_list, motor_unwrapped, label="current_angle_deg (measured motor)")
  axes[0].set_ylabel("motor angle (deg)")
  axes[0].legend()
  axes[0].grid(True)

  axes[1].plot(t_list, pole_unwrapped, color="tab:green")
  axes[1].set_ylabel("pendulum angle (deg)")
  axes[1].grid(True)

  axes[2].plot(t_list, motor_vel_list, color="tab:orange", label="motor_vel_deg_s")
  axes[2].plot(t_list, pole_vel_list, color="tab:purple", label="pole_vel_deg_s")
  axes[2].set_ylabel("velocity (deg/s)")
  axes[2].set_xlabel("time (s)")
  axes[2].legend()
  axes[2].grid(True)

  fig.tight_layout()
  png_path = csv_path.with_suffix(".png")
  fig.savefig(png_path, dpi=150)
  print(f"그래프 저장 완료: {png_path}")


def run(cfg: SineMotorSimTestConfig) -> Path:
  xml_path = Path(cfg.xml_file)
  if not xml_path.exists():
    raise FileNotFoundError(f"XML not found: {xml_path}")

  model = _load_model(xml_path, no_pendulum=cfg.no_pendulum)
  # assets/inverse.xml has no <option timestep=...>, so it otherwise
  # compiles with MuJoCo's default (0.002s), not the 200 Hz physics rate
  # inverse_sine_env_cfg.py sets via MujocoCfg(timestep=_PHYSICS_DT_S).
  model.opt.timestep = _PHYSICS_DT_S
  if cfg.disable_contact:
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  data = mujoco.MjData(model)

  cylinder_qpos = _joint_qpos_addr(model, "Revolute 3")
  cylinder_qvel = _joint_qvel_addr(model, "Revolute 3")
  if cfg.no_pendulum:
    pole_qpos, pole_qvel = None, None
  else:
    pole_qpos = _joint_qpos_addr(model, "Revolute 5")
    pole_qvel = _joint_qvel_addr(model, "Revolute 5")
  actuator = _actuator_id(model, "position_revolute_3")

  if cfg.joint_armature is not None:
    model.dof_armature[cylinder_qvel] = float(cfg.joint_armature)
  if cfg.joint_damping is not None:
    model.dof_damping[cylinder_qvel] = float(cfg.joint_damping)

  mujoco.mj_forward(model, data)

  actuator_kv = cfg.actuator_kv
  if cfg.damping_ratio is not None:
    if cfg.actuator_kp is None:
      raise ValueError("--damping-ratio requires --actuator-kp to also be set.")
    cylinder_inertia = _effective_dof_inertia(model, data, cylinder_qvel)
    joint_damping = float(model.dof_damping[cylinder_qvel])
    actuator_kv = _kv_for_damping_ratio(
      kp=cfg.actuator_kp,
      inertia=cylinder_inertia,
      damping_ratio=cfg.damping_ratio,
      joint_damping=joint_damping,
    )
    print(
      f"[INFO] critical-damping calc: I_eff={cylinder_inertia:.6e} kg*m^2 "
      f"joint_damping={joint_damping:.5f} damping_ratio={cfg.damping_ratio:.2f} "
      f"kp={cfg.actuator_kp:.2f} -> kv={actuator_kv:.4f}"
    )

  _override_position_actuator(model, actuator, kp=cfg.actuator_kp, kv=actuator_kv)

  mujoco.mj_forward(model, data)

  control_dt_s = _PHYSICS_DT_S * _DECIMATION
  sine_cfg = SineReferenceConfig(
    amplitude_deg=cfg.amplitude_deg,
    frequency_hz=cfg.frequency_hz,
    phase_deg=cfg.phase_deg,
  )
  limiter = RateAccelLimiter.from_control_dt(control_dt_s=control_dt_s)
  limiter.reset(float(data.qpos[cylinder_qpos]))

  log_dir = Path(cfg.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = log_dir / (
    f"sine_sim_amp{cfg.amplitude_deg:g}deg_freq{cfg.frequency_hz:g}hz_{stamp}.csv"
  )

  control_steps = int(math.ceil(cfg.duration_s / control_dt_s))

  print(f"[INFO] XML: {xml_path}")
  print(f"[INFO] CSV: {csv_path}")
  print(
    f"[INFO] amplitude={cfg.amplitude_deg:.2f}deg frequency={cfg.frequency_hz:.3f}Hz "
    f"phase={cfg.phase_deg:.2f}deg duration={cfg.duration_s:.2f}s "
    f"control_dt={control_dt_s:.4f}s ({1.0 / control_dt_s:.1f}Hz) "
    f"physics_dt={_PHYSICS_DT_S:.4f}s decimation={_DECIMATION} "
    f"model.opt.timestep={model.opt.timestep:.4f}s"
  )
  print(
    f"[INFO] actuator_kp={model.actuator_gainprm[actuator, 0]:.3f} "
    f"actuator_kv={-model.actuator_biasprm[actuator, 2]:.3f}"
  )
  print(
    f"[INFO] joint_armature={float(model.dof_armature[cylinder_qvel]):.5f} "
    f"joint_damping={float(model.dof_damping[cylinder_qvel]):.5f}"
  )
  print(f"[INFO] no_pendulum={cfg.no_pendulum} disable_contact={cfg.disable_contact}")

  force_range = model.actuator_forcerange[actuator]
  print(
    f"[INFO] actuator_forcerange=[{force_range[0]:.2f}, {force_range[1]:.2f}]Nm"
  )
  max_abs_force_nm = 0.0

  wall_start_s = time.perf_counter()
  with csv_path.open("w", newline="") as f, _viewer_context(
    model, data, cfg.render
  ) as viewer:
    writer = csv.writer(f)
    writer.writerow(_CSV_HEADER)

    for step in range(control_steps + 1):
      time_s = step * control_dt_s

      desired_target_rad = sine_target_rad(sine_cfg, time_s)
      limited_target_rad = limiter.step(desired_target_rad)
      data.ctrl[actuator] = limited_target_rad

      for _ in range(_DECIMATION):
        mujoco.mj_step(model, data)

      if viewer is not None:
        viewer.sync()
        if not viewer.is_running():
          print("[INFO] viewer closed; stopping.")
          break
        if cfg.realtime:
          target_wall_s = wall_start_s + time_s
          sleep_s = target_wall_s - time.perf_counter()
          if sleep_s > 0.0:
            time.sleep(sleep_s)

      motor_angle_deg = math.degrees(float(data.qpos[cylinder_qpos]))
      motor_vel_deg_s = math.degrees(float(data.qvel[cylinder_qvel]))
      if pole_qpos is not None and pole_qvel is not None:
        pole_angle_deg = math.degrees(float(data.qpos[pole_qpos]))
        pole_vel_deg_s = math.degrees(float(data.qvel[pole_qvel]))
      else:
        pole_angle_deg = 0.0
        pole_vel_deg_s = 0.0
      actuator_force_nm = float(data.actuator_force[actuator])
      max_abs_force_nm = max(max_abs_force_nm, abs(actuator_force_nm))

      writer.writerow(
        [
          f"{time_s:.4f}",
          f"{math.degrees(desired_target_rad):.4f}",
          f"{math.degrees(limited_target_rad):.4f}",
          f"{motor_angle_deg:.4f}",
          f"{motor_vel_deg_s:.4f}",
          f"{pole_angle_deg:.4f}",
          f"{pole_vel_deg_s:.4f}",
          f"{actuator_force_nm:.6f}",
        ]
      )

      if step % max(1, int(cfg.print_every)) == 0:
        print(
          f"[{step:05d}] t={time_s:6.3f}s "
          f"target={math.degrees(desired_target_rad):+7.2f}deg "
          f"motor={motor_angle_deg:+7.2f}deg "
          f"pole={pole_angle_deg:+7.2f}deg"
        )

  print(f"[INFO] CSV 저장 완료: {csv_path}")
  print(
    f"[INFO] max |actuator_force| = {max_abs_force_nm:.3f} Nm "
    f"(limit +/-{force_range[1]:.2f}Nm)"
  )
  if cfg.plot:
    _plot_log(csv_path)
  return csv_path


def main() -> None:
  run(tyro.cli(SineMotorSimTestConfig))


if __name__ == "__main__":
  main()
