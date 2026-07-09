#!/usr/bin/env python3
from __future__ import annotations

"""System-identification 여기 신호(open-loop) 수집 스크립트.

`run_policy_motor.py`와 달리 RL 정책은 전혀 쓰지 않는다. 실린더 모터에
알려진 스텝 또는 처프(chirp, 주파수 스윕) 목표각을 직접 명령으로 보내고,
`robot_state_reader.py`로 응답(실린더 각/속도/토크, 진자 각)을 로깅한다.
이렇게 모은 로그는 `sysid_fit.py`가 `assets/inverse.xml`의 관절
damping/armature/frictionloss, 액추에이터 kp, 실린더/진자 질량·무게중심·
관성을 실물에 맞게 추정하는 데 쓰인다.

`run_policy_motor.py`가 토크(`motor_torque_nm`)를 읽고도 버리는 것과 달리,
이 스크립트는 토크를 반드시 CSV에 기록한다 (`mujoco_sysid`의 회귀 기반
질량/관성 추정에 실측 토크가 필요하기 때문).

사용 순서
1. `commission_motor.py`로 모터가 MIT 모드이며 원하는 ID인지 먼저 확인한다.
2. `robot_state_reader.py` 단독 실행으로 상태 스트림이 정상인지 확인한다.
3. 이 스크립트를 낮은 진폭(amplitude_deg)으로 먼저 실행해 본다.

예시 실행
```bash
uv run python src/mjlab/tasks/inverse/sysid_collect.py \
  --mode chirp --motor-id 8 --channel can0 \
  --encoder-port /dev/ttyACM0 \
  --amplitude-deg 10 --chirp-f1-hz 3.0 --chirp-duration-s 20
```
"""

import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import tyro

from mjlab.tasks.inverse.commission_motor import (
  _open_bus,
  _shutdown_bus,
  disable_mit,
  enable_mit,
  mit_position_command,
  set_zero_mit,
)
from mjlab.tasks.inverse.robot_state_reader import RobotStateReader


@dataclass(frozen=True)
class SysidCollectConfig:
  mode: Literal["step", "chirp"] = "chirp"

  motor_id: int = 8
  interface: str = "socketcan"
  channel: str = "can0"
  encoder_port: str | None = None
  encoder_baud: int = 115200

  control_hz: float = 200.0
  state_read_hz: float = 200.0

  # 실물 모터에 보내는 MIT 레벨 kp/kd (sim 액추에이터의 kp와는 다른 값).
  kp: float = 10.0
  kd: float = 0.5

  # 여기 신호 공통 설정.
  center_deg: float = 0.0
  amplitude_deg: float = 15.0
  ramp_in_s: float = 0.5

  # mode="step"
  hold_duration_s: float = 1.0
  num_steps: int = 10
  step_pattern: Literal["alternating", "random", "staircase"] = "alternating"
  seed: int = 0

  # mode="chirp"
  chirp_f0_hz: float = 0.1
  chirp_f1_hz: float = 5.0
  chirp_duration_s: float = 20.0
  chirp_scale: Literal["linear", "log"] = "log"

  # 안전 설정.
  max_amplitude_deg: float = 45.0
  torque_limit_nm: float = 3.0
  abort_on_torque_nm: float = 5.0
  abort_on_pole_deg: float | None = None
  max_runtime_s: float = 60.0
  start_with_zero_set: bool = False
  require_enable_ack: bool = True

  # 로깅/플롯 설정.
  log_dir: str = "logs"
  plot: bool = True


def _ramp(t: float, ramp_in_s: float) -> float:
  if ramp_in_s <= 0.0:
    return 1.0
  return min(1.0, t / ramp_in_s)


def build_chirp_targets(
  cfg: SysidCollectConfig, sample_times: np.ndarray
) -> np.ndarray:
  """`sample_times`(초, 0부터 시작)에 대응하는 목표각(deg) 시퀀스를 만든다."""
  t = np.clip(sample_times, 0.0, cfg.chirp_duration_s)
  f0, f1, T = cfg.chirp_f0_hz, cfg.chirp_f1_hz, cfg.chirp_duration_s
  if cfg.chirp_scale == "log":
    k = (f1 / f0) ** (1.0 / T)
    phase = 2.0 * math.pi * f0 * (np.power(k, t) - 1.0) / math.log(k)
  else:
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) / T * t**2)
  ramp = np.clip(sample_times / cfg.ramp_in_s, 0.0, 1.0) if cfg.ramp_in_s > 0 else 1.0
  target = cfg.center_deg + cfg.amplitude_deg * ramp * np.sin(phase)
  return np.clip(target, -cfg.max_amplitude_deg, cfg.max_amplitude_deg)


def build_step_targets(cfg: SysidCollectConfig, sample_times: np.ndarray) -> np.ndarray:
  """`sample_times`(초, 0부터 시작)에 대응하는 목표각(deg) 시퀀스를 만든다."""
  rng = np.random.default_rng(cfg.seed)
  if cfg.step_pattern == "alternating":
    offsets = np.array(
      [
        cfg.amplitude_deg if i % 2 == 0 else -cfg.amplitude_deg
        for i in range(cfg.num_steps)
      ]
    )
  elif cfg.step_pattern == "staircase":
    offsets = np.linspace(-cfg.amplitude_deg, cfg.amplitude_deg, cfg.num_steps)
  else:  # random
    offsets = rng.uniform(-cfg.amplitude_deg, cfg.amplitude_deg, size=cfg.num_steps)

  held_angles = cfg.center_deg + offsets
  step_idx = np.clip(
    (sample_times // cfg.hold_duration_s).astype(int), 0, cfg.num_steps - 1
  )
  target = held_angles[step_idx]

  # 첫 hold 구간에만 ramp-in을 적용해 최초 스텝이 급격하지 않게 한다.
  ramp = np.clip(sample_times / cfg.ramp_in_s, 0.0, 1.0) if cfg.ramp_in_s > 0 else 1.0
  first_hold = sample_times < cfg.hold_duration_s
  target = np.where(
    first_hold, cfg.center_deg + (target - cfg.center_deg) * ramp, target
  )
  return np.clip(target, -cfg.max_amplitude_deg, cfg.max_amplitude_deg)


def build_targets(cfg: SysidCollectConfig, sample_times: np.ndarray) -> np.ndarray:
  if cfg.mode == "chirp":
    return build_chirp_targets(cfg, sample_times)
  return build_step_targets(cfg, sample_times)


def signal_duration_s(cfg: SysidCollectConfig) -> float:
  if cfg.mode == "chirp":
    return cfg.chirp_duration_s
  return cfg.hold_duration_s * cfg.num_steps


_CSV_HEADER = [
  "time_s",
  "target_angle_deg",
  "motor_angle_deg",
  "motor_vel_deg_s",
  "motor_torque_nm",
  "pole_angle_deg",
  "pole_vel_deg_s",
]


def _plot_log(csv_path: str) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print(
      "matplotlib이 설치되어 있지 않아 그래프를 건너뜁니다. (pip install matplotlib)"
    )
    return

  rows: dict[str, list[float]] = {name: [] for name in _CSV_HEADER}
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      for name in _CSV_HEADER:
        rows[name].append(float(row[name]))

  if not rows["time_s"]:
    print("로그가 비어 있어 그래프를 건너뜁니다.")
    return

  t = rows["time_s"]
  # CSV에는 원본 각도(예: 0~360 랩어라운드)를 그대로 저장하지만, 그래프에서는
  # 그 경계(359 -> 1 같은)에서 생기는 수직선을 없애기 위해 unwrap한 값만 쓴다.
  motor_angle_deg_unwrapped = np.degrees(np.unwrap(np.radians(rows["motor_angle_deg"])))
  pole_angle_deg_unwrapped = np.degrees(np.unwrap(np.radians(rows["pole_angle_deg"])))

  fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

  axes[0].plot(t, rows["target_angle_deg"], label="target_angle_deg", linestyle="--")
  axes[0].plot(t, motor_angle_deg_unwrapped, label="motor_angle_deg")
  axes[0].set_ylabel("motor angle (deg)")
  axes[0].legend()
  axes[0].grid(True)

  axes[1].plot(t, pole_angle_deg_unwrapped, color="tab:green")
  axes[1].set_ylabel("pendulum angle (deg)")
  axes[1].grid(True)

  axes[2].plot(t, rows["motor_vel_deg_s"], color="tab:orange", label="motor_vel_deg_s")
  axes[2].plot(t, rows["pole_vel_deg_s"], color="tab:purple", label="pole_vel_deg_s")
  axes[2].set_ylabel("velocity (deg/s)")
  axes[2].legend()
  axes[2].grid(True)

  axes[3].plot(t, rows["motor_torque_nm"], color="tab:red")
  axes[3].set_ylabel("motor torque (Nm)")
  axes[3].set_xlabel("time (s)")
  axes[3].grid(True)

  fig.tight_layout()
  png_path = os.path.splitext(csv_path)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  print(f"그래프 저장 완료: {png_path}")


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


def run(cfg: SysidCollectConfig) -> None:
  duration_s = signal_duration_s(cfg)
  n_samples = int(duration_s * cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / cfg.control_hz
  targets_deg = build_targets(cfg, sample_times)

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
  csv_path = os.path.join(cfg.log_dir, f"sysid_{cfg.mode}_{ts}.csv")
  csv_file = open(csv_path, "w", newline="")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(_CSV_HEADER)

  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  print(f"mode={cfg.mode}, duration={duration_s:.1f}s, samples={n_samples}")
  print(f"CSV 로그: {csv_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(여기 신호 완료)"
  previous_encoder_state = None

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

    period_s = 1.0 / cfg.control_hz
    deadline = start_t + min(duration_s, cfg.max_runtime_s)

    print("Starting excitation loop.")
    sample_idx = 0
    while time.monotonic() < deadline and sample_idx < n_samples:
      loop_start = time.monotonic()
      target_deg = float(targets_deg[sample_idx])

      ok = mit_position_command(
        bus,
        cfg.motor_id,
        target_deg,
        kp=cfg.kp,
        kd=cfg.kd,
        torque_nm=0.0,
      )
      if not ok:
        raise RuntimeError("No MIT reply received from motor")

      state = reader.get_state()
      now = time.monotonic()

      motor_angle_deg = state.motor_angle_deg
      motor_vel_deg_s = state.motor_velocity_deg_s
      motor_torque_nm = state.motor_torque_nm
      pole_angle_deg = state.pendulum_angle_deg

      if motor_angle_deg is None or motor_vel_deg_s is None or motor_torque_nm is None:
        raise RuntimeError("Motor state is not available from RobotStateReader")
      if pole_angle_deg is None:
        raise RuntimeError(
          "Pendulum angle is not available from RobotStateReader. "
          "Provide --encoder-port and check the serial encoder stream."
        )

      if cfg.abort_on_torque_nm and abs(motor_torque_nm) > cfg.abort_on_torque_nm:
        raise RuntimeError(
          f"Aborting: |motor_torque_nm|={abs(motor_torque_nm):.2f} "
          f"exceeded abort_on_torque_nm={cfg.abort_on_torque_nm}"
        )
      if (
        cfg.abort_on_pole_deg is not None
        and abs(pole_angle_deg) > cfg.abort_on_pole_deg
      ):
        raise RuntimeError(
          f"Aborting: |pole_angle_deg|={abs(pole_angle_deg):.1f} "
          f"exceeded abort_on_pole_deg={cfg.abort_on_pole_deg}"
        )

      pole_vel_deg_s = 0.0
      encoder_state = (
        reader.encoder_reader.latest if reader.encoder_reader is not None else None
      )
      if encoder_state is not None and previous_encoder_state is not None:
        enc_dt = max(encoder_state.timestamp - previous_encoder_state.timestamp, 1e-4)
        raw_delta = encoder_state.angle_deg - previous_encoder_state.angle_deg
        delta_deg = (raw_delta + 180) % 360 - 180
        pole_vel_deg_s = delta_deg / enc_dt
      previous_encoder_state = encoder_state

      elapsed_t = now - start_t
      csv_writer.writerow(
        [
          f"{elapsed_t:.4f}",
          f"{target_deg:.3f}",
          f"{motor_angle_deg:.3f}",
          f"{motor_vel_deg_s:.3f}",
          f"{motor_torque_nm:.4f}",
          f"{pole_angle_deg:.3f}",
          f"{pole_vel_deg_s:.3f}",
        ]
      )

      if sample_idx % 20 == 0:
        print(
          "t={:6.2f}s  target={:+8.3f} deg  motor={:+8.3f} deg  "
          "pole={:+8.3f} deg  torque={:+5.2f} Nm".format(
            elapsed_t, target_deg, motor_angle_deg, pole_angle_deg, motor_torque_nm
          )
        )

      sample_idx += 1
      elapsed = time.monotonic() - loop_start
      sleep_s = period_s - elapsed
      if sleep_s > 0.0:
        time.sleep(sleep_s)

  except KeyboardInterrupt:
    stopped_reason = "사용자 강제종료(Ctrl+C)"
    print("\n" + stopped_reason)
  except RuntimeError as exc:
    stopped_reason = f"에러로 종료: {exc}"
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

  print(f"여기 신호 종료 ({stopped_reason}).")

  if cfg.plot:
    _plot_log(csv_path)


def main() -> None:
  cfg = tyro.cli(SysidCollectConfig)
  run(cfg)


if __name__ == "__main__":
  main()
