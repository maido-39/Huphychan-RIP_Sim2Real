#!/usr/bin/env python3
"""Open-loop sine-motion check for the inverse_sub motor, on real hardware.

No RL checkpoint is involved. This sends a hand-computed sine trajectory
(amplitude * sin(2*pi*f*t)) straight to the real motor over MIT/CAN, through
the same rate/accel limiter used by inverse_sine_env_cfg.py and
real_policy_inference.py, so the response you observe here is what an
RL-deployed policy would also be subject to. Meant to validate motor
step/sine tracking and pendulum coupling before any RL training is run
(see sim2real_parameters.md, "Highest Priority To Match" #1).

Recommended order:
1. `commission_motor.py` to confirm MIT mode / motor ID.
2. `robot_state_reader.py` to confirm motor + pendulum state reads look sane.
3. This script, starting with a small `--amplitude-multiplier` (e.g. 0.2).

Example:
```bash
uv run python src/mjlab/tasks/inverse_sub/sine_motor_real_test.py \
  --motor-id 8 \
  --channel can0 \
  --encoder-port /dev/ttyACM0 \
  --amplitude-deg 30 \
  --frequency-hz 1.0 \
  --duration-s 10 \
  --amplitude-multiplier 0.2
```
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tyro

from mjlab.tasks.inverse_sub.commission_motor import (
  _float_to_uint,
  _matches_mit_reply,
  _open_bus,
  _shutdown_bus,
  _wait_reply,
  can,
  disable_mit,
  enable_mit,
  parse_mit_reply_position,
  set_zero_mit,
)
from mjlab.tasks.inverse_sub.robot_state_reader import RobotStateReader
from mjlab.tasks.inverse_sub.sine_reference import (
  RateAccelLimiter,
  SineReferenceConfig,
  sine_target_rad,
)


class PoleVelocityPll:
  """PLL-style angular velocity estimator for wrapped encoder angles.

  Same estimator as run_policy_motor.PoleVelocityPll, duplicated here so
  this script only depends on mjlab.tasks.inverse_sub modules.
  """

  def __init__(self, *, kp: float, ki: float):
    self.kp = float(kp)
    self.ki = float(ki)
    self._theta_hat_rad: float | None = None
    self._omega_hat_rad_s = 0.0
    self._last_timestamp: float | None = None

  def update(self, *, angle_deg: float, timestamp: float) -> float:
    theta_meas_rad = math.radians(float(angle_deg))

    if self._theta_hat_rad is None or self._last_timestamp is None:
      self._theta_hat_rad = theta_meas_rad
      self._last_timestamp = float(timestamp)
      self._omega_hat_rad_s = 0.0
      return 0.0

    dt = max(float(timestamp) - self._last_timestamp, 1e-4)
    phase_error = math.sin(theta_meas_rad - self._theta_hat_rad)

    self._omega_hat_rad_s += self.ki * phase_error * dt
    self._theta_hat_rad += (self._omega_hat_rad_s + self.kp * phase_error) * dt
    self._theta_hat_rad = math.atan2(
      math.sin(self._theta_hat_rad),
      math.cos(self._theta_hat_rad),
    )
    self._last_timestamp = float(timestamp)
    return math.degrees(self._omega_hat_rad_s)


def _send_mit_position_and_get_reply(
  bus,
  motor_id: int,
  position_deg: float,
  *,
  kp: float,
  kd: float,
  velocity_deg_s: float,
  torque_nm: float,
  pmax: float = 12.57,
  vmax: float = 33.0,
  tmax: float = 17.0,
  timeout_s: float = 0.03,
):
  """Send MIT position command and return the reply message if one arrives."""
  position_rad = math.radians(float(position_deg))
  velocity_rad_s = math.radians(float(velocity_deg_s))

  kp_uint = _float_to_uint(float(kp), 0.0, 500.0, 12)
  kd_uint = _float_to_uint(float(kd), 0.0, 5.0, 12)
  q_uint = _float_to_uint(position_rad, -float(pmax), float(pmax), 16)
  dq_uint = _float_to_uint(velocity_rad_s, -float(vmax), float(vmax), 12)
  tau_uint = _float_to_uint(float(torque_nm), -float(tmax), float(tmax), 12)

  data = [0] * 8
  data[0] = (q_uint >> 8) & 0xFF
  data[1] = q_uint & 0xFF
  data[2] = dq_uint >> 4
  data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
  data[4] = kp_uint & 0xFF
  data[5] = kd_uint >> 4
  data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
  data[7] = tau_uint & 0xFF

  msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_mit_reply(m, motor_id=int(motor_id)),
  )
  return reply


def _wait_for_initial_state(reader: RobotStateReader, timeout_s: float = 3.0) -> None:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    state = reader.get_state()
    if (
      state.motor_angle_deg is not None
      and state.motor_velocity_deg_s is not None
      and state.pendulum_angle_deg is not None
    ):
      return
    time.sleep(0.01)
  raise RuntimeError(
    "Timed out waiting for RobotStateReader initial state. "
    "Check CAN motor feedback and encoder serial connection."
  )


@dataclass(frozen=True)
class SineMotorRealTestConfig:
  motor_id: int
  interface: str = "socketcan"
  channel: str = "can0"
  encoder_port: str | None = None

  # 실물 축 부호/오프셋 보정 (real_policy_inference.py와 동일한 규약).
  cylinder_sign: float = 1.0
  cylinder_zero_deg: float = 0.0

  # Sine 명령 파라미터.
  amplitude_deg: float = 30.0
  frequency_hz: float = 1.0
  phase_deg: float = 0.0
  duration_s: float = 10.0
  # 처음에는 작게 시작: 실제 진폭 = amplitude_deg * amplitude_multiplier.
  amplitude_multiplier: float = 0.2

  # 제어 루프 설정.
  control_hz: float = 50.0
  state_read_hz: float = 200.0
  kp: float = 16.5
  kd: float = 1.0
  velocity_limit_deg_s: float = 0.0
  torque_limit_nm: float = 0.0
  encoder_baud: int = 115200
  pole_pll_kp: float = 80.0
  pole_pll_ki: float = 1200.0

  # 안전 설정.
  start_with_zero_set: bool = False
  require_enable_ack: bool = True

  # 로깅/플롯 설정.
  log_dir: str = "logs"
  plot: bool = True


_CSV_HEADER = [
  "time_s",
  "target_angle_deg",
  "motor_angle_deg",
  "motor_vel_deg_s",
  "pole_angle_deg",
  "pole_vel_deg_s",
]


def _deg_to_rad(value_deg: float) -> float:
  return value_deg * math.pi / 180.0


def _normalize_cylinder_rad(
  cylinder_angle_deg: float,
  *,
  cylinder_zero_deg: float,
  cylinder_sign: float,
) -> float:
  return _deg_to_rad((cylinder_angle_deg - cylinder_zero_deg) * cylinder_sign)


def _denormalize_cylinder_deg(
  cylinder_angle_rad: float,
  *,
  cylinder_zero_deg: float,
  cylinder_sign: float,
) -> float:
  return cylinder_angle_rad * 180.0 / math.pi / cylinder_sign + cylinder_zero_deg


def _plot_log(csv_path: str) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib이 설치되어 있지 않아 그래프를 건너뜁니다. (pip install matplotlib)")
    return

  t_list, target_list, motor_list, motor_vel_list = [], [], [], []
  pole_list, pole_vel_list = [], []
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      t_list.append(float(row["time_s"]))
      target_list.append(float(row["target_angle_deg"]))
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

  axes[0].plot(t_list, target_list, label="sine_target_deg", linestyle="--")
  axes[0].plot(t_list, motor_unwrapped, label="motor_angle_deg")
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
  png_path = os.path.splitext(csv_path)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  print(f"그래프 저장 완료: {png_path}")


def run(cfg: SineMotorRealTestConfig) -> None:
  effective_amplitude_deg = cfg.amplitude_deg * cfg.amplitude_multiplier
  sine_cfg = SineReferenceConfig(
    amplitude_deg=effective_amplitude_deg,
    frequency_hz=cfg.frequency_hz,
    phase_deg=cfg.phase_deg,
  )
  control_dt_s = 1.0 / float(cfg.control_hz)
  limiter = RateAccelLimiter.from_control_dt(control_dt_s=control_dt_s)

  bus = _open_bus(cfg.interface, cfg.channel)
  reader = RobotStateReader(
    motor_id=cfg.motor_id,
    encoder_port=cfg.encoder_port,
    can_interface=cfg.interface,
    can_channel=cfg.channel,
    motor_mode="passive",
    motor_rate_hz=cfg.state_read_hz,
    encoder_baud=cfg.encoder_baud,
  )

  os.makedirs(cfg.log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join(
    cfg.log_dir,
    f"sine_real_amp{effective_amplitude_deg:g}deg_freq{cfg.frequency_hz:g}hz_{ts}.csv",
  )
  csv_file = open(csv_path, "w", newline="")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(_CSV_HEADER)

  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  print(
    f"Sine: amplitude={cfg.amplitude_deg:.2f}deg x multiplier={cfg.amplitude_multiplier:.2f} "
    f"= {effective_amplitude_deg:.2f}deg, frequency={cfg.frequency_hz:.3f}Hz, "
    f"duration={cfg.duration_s:.2f}s"
  )
  print(f"CSV 로그: {csv_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(duration_s 도달)"

  try:
    print("Starting RobotStateReader...")
    reader.start()

    if cfg.start_with_zero_set:
      print("Setting motor zero...")
      if not set_zero_mit(bus, cfg.motor_id):
        raise RuntimeError("set_zero_mit was not acknowledged")
      time.sleep(0.1)

    print("Enabling motor...")
    enable_ok = enable_mit(bus, cfg.motor_id)
    if cfg.require_enable_ack and not enable_ok:
      raise RuntimeError("enable_mit was not acknowledged")
    time.sleep(0.1)

    _wait_for_initial_state(reader)

    initial_state = reader.get_state()
    if initial_state.motor_angle_deg is None:
      raise RuntimeError("Motor state is not available from RobotStateReader")
    initial_cylinder_rad = _normalize_cylinder_rad(
      float(initial_state.motor_angle_deg),
      cylinder_zero_deg=cfg.cylinder_zero_deg,
      cylinder_sign=cfg.cylinder_sign,
    )
    limiter.reset(initial_cylinder_rad)

    pll = PoleVelocityPll(kp=cfg.pole_pll_kp, ki=cfg.pole_pll_ki)
    prev_time = time.monotonic()
    period_s = control_dt_s
    deadline = prev_time + float(cfg.duration_s)

    print("Starting open-loop sine control loop.")
    while time.monotonic() < deadline:
      loop_start = time.monotonic()
      elapsed_t = loop_start - start_t

      desired_target_rad = sine_target_rad(sine_cfg, elapsed_t)
      limited_target_rad = limiter.step(desired_target_rad)
      target_angle_deg = _denormalize_cylinder_deg(
        limited_target_rad,
        cylinder_zero_deg=cfg.cylinder_zero_deg,
        cylinder_sign=cfg.cylinder_sign,
      )

      reply = _send_mit_position_and_get_reply(
        bus,
        cfg.motor_id,
        target_angle_deg,
        kp=cfg.kp,
        kd=cfg.kd,
        velocity_deg_s=cfg.velocity_limit_deg_s,
        torque_nm=cfg.torque_limit_nm,
      )
      if reply is None:
        raise RuntimeError("No MIT reply received from motor")
      _ = parse_mit_reply_position(reply)

      state = reader.get_state()
      now = time.monotonic()
      prev_time = now

      cylinder_angle_deg = state.motor_angle_deg
      cylinder_vel_deg_s = state.motor_velocity_deg_s
      pole_angle_deg = state.pendulum_angle_deg

      if cylinder_angle_deg is None or cylinder_vel_deg_s is None:
        raise RuntimeError("Motor state is not available from RobotStateReader")
      if pole_angle_deg is None:
        raise RuntimeError(
          "Pendulum angle is not available from RobotStateReader. "
          "Provide --encoder-port and check the serial encoder stream."
        )

      encoder_state = (
        reader.encoder_reader.latest if reader.encoder_reader is not None else None
      )
      if encoder_state is not None:
        pole_vel_deg_s = pll.update(
          angle_deg=encoder_state.angle_deg,
          timestamp=encoder_state.timestamp,
        )
      else:
        pole_vel_deg_s = 0.0

      csv_writer.writerow(
        [
          f"{elapsed_t:.4f}",
          f"{target_angle_deg:.3f}",
          f"{cylinder_angle_deg:.3f}",
          f"{cylinder_vel_deg_s:.3f}",
          f"{pole_angle_deg:.3f}",
          f"{pole_vel_deg_s:.3f}",
        ]
      )

      print(
        "target={:+8.3f} deg  motor={:+8.3f} deg  pole={:+8.3f} deg".format(
          target_angle_deg, cylinder_angle_deg, pole_angle_deg
        )
      )

      elapsed = time.monotonic() - loop_start
      sleep_s = period_s - elapsed
      if sleep_s > 0.0:
        time.sleep(sleep_s)

  except KeyboardInterrupt:
    stopped_reason = "사용자 강제종료(Ctrl+C)"
    print("\n" + stopped_reason)

  finally:
    print("Disabling motor...")
    try:
      disable_mit(bus, cfg.motor_id)
    finally:
      reader.stop()
      _shutdown_bus(bus)
      csv_file.close()
      print(f"CSV 로그 저장 완료: {csv_path}")

  print(f"제어 종료 ({stopped_reason}).")

  if cfg.plot:
    _plot_log(csv_path)


def main() -> None:
  cfg = tyro.cli(SineMotorRealTestConfig)
  run(cfg)


if __name__ == "__main__":
  main()
