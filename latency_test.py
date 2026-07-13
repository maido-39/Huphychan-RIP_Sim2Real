#!/usr/bin/env python3
from __future__ import annotations

"""제어 루프 / 상태 읽기 지연시간 측정 스크립트.

`sysid_collect.py`의 chirp(주파수를 점점 높이는 사인파) 신호를 그대로 재사용해
실린더 모터에 위치 명령을 보내면서, 아래 지연시간들을 직접 측정해 CSV/PNG로
남긴다:

  - loop_dt_ms          : 목표 주기(1/control_hz) 대비 실제 루프 반복 시간
  - can_rtt_ms          : MIT 위치 명령을 보내고 응답(reply)을 받기까지 걸린
                          시간 (제어 쪽 지연)
  - motor_state_age_ms  : 이번에 쓴 모터(CAN) 상태가 얼마나 오래된 데이터인지
                          (실제로 갱신된 시각 대비)
  - encoder_state_age_ms: 이번에 쓴 진자 인코더 상태가 얼마나 오래된 데이터인지

chirp는 시간이 지날수록 주파수가 올라가므로, 로그 앞부분(저주파 구간)과
뒷부분(고주파 구간)을 비교하면 고주파/저주파 제어에서 지연시간이 어떻게
달라지는지 볼 수 있다. 실행이 끝나면 두 구간의 평균/최대값을 요약해서
출력한다.

이 스크립트는 진단 전용이라 `sysid_collect.py`/`run_policy_motor.py` 등
기존 파일은 건드리지 않는다.

예시 실행
```bash
uv run python src/mjlab/tasks/inverse/latency_test.py \
  --motor-id 8 --channel can0 --encoder-port /dev/ttyACM0 \
  --amplitude-deg 10 --chirp-f1-hz 10 --chirp-duration-s 30
```
"""

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime

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
from mjlab.tasks.inverse.sysid_collect import (
  SysidCollectConfig,
  build_chirp_targets,
  signal_duration_s,
)


@dataclass(frozen=True)
class LatencyTestConfig:
  motor_id: int = 8
  interface: str = "socketcan"
  channel: str = "can0"
  encoder_port: str | None = None
  encoder_baud: int = 115200

  control_hz: float = 200.0
  kp: float = 10.0
  kd: float = 0.5

  center_deg: float = 0.0
  amplitude_deg: float = 15.0
  ramp_in_s: float = 0.5

  chirp_f0_hz: float = 0.1
  chirp_f1_hz: float = 5.0
  chirp_duration_s: float = 30.0
  chirp_scale: str = "log"  # "log" or "linear"

  max_amplitude_deg: float = 45.0
  torque_limit_nm: float = 3.0
  abort_on_torque_nm: float = 5.0
  max_runtime_s: float = 60.0
  start_with_zero_set: bool = False
  require_enable_ack: bool = True

  log_dir: str = "logs"
  plot: bool = True


def _to_sysid_cfg(cfg: LatencyTestConfig) -> SysidCollectConfig:
  """chirp 신호 생성만 재사용하기 위해 SysidCollectConfig로 변환한다."""
  return SysidCollectConfig(
    mode="chirp",
    motor_id=cfg.motor_id,
    interface=cfg.interface,
    channel=cfg.channel,
    encoder_port=cfg.encoder_port,
    encoder_baud=cfg.encoder_baud,
    control_hz=cfg.control_hz,
    kp=cfg.kp,
    kd=cfg.kd,
    center_deg=cfg.center_deg,
    amplitude_deg=cfg.amplitude_deg,
    ramp_in_s=cfg.ramp_in_s,
    chirp_f0_hz=cfg.chirp_f0_hz,
    chirp_f1_hz=cfg.chirp_f1_hz,
    chirp_duration_s=cfg.chirp_duration_s,
    chirp_scale=cfg.chirp_scale,  # type: ignore[arg-type]
    max_amplitude_deg=cfg.max_amplitude_deg,
    torque_limit_nm=cfg.torque_limit_nm,
    abort_on_torque_nm=cfg.abort_on_torque_nm,
    max_runtime_s=cfg.max_runtime_s,
    start_with_zero_set=cfg.start_with_zero_set,
    require_enable_ack=cfg.require_enable_ack,
    log_dir=cfg.log_dir,
    plot=False,
  )


_CSV_HEADER = [
  "time_s",
  "target_angle_deg",
  "motor_angle_deg",
  "pole_angle_deg",
  "loop_dt_ms",
  "target_loop_dt_ms",
  "can_rtt_ms",
  "motor_state_age_ms",
  "encoder_state_age_ms",
]


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
  fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

  axes[0].plot(t, rows["target_angle_deg"], label="target_angle_deg", linestyle="--")
  axes[0].plot(t, rows["motor_angle_deg"], label="motor_angle_deg")
  axes[0].set_ylabel("cylinder angle (deg)")
  axes[0].legend()
  axes[0].grid(True)

  axes[1].plot(t, rows["loop_dt_ms"], label="loop_dt_ms", color="tab:orange")
  axes[1].plot(
    t,
    rows["target_loop_dt_ms"],
    label="target_loop_dt_ms",
    color="tab:orange",
    linestyle="--",
  )
  axes[1].plot(t, rows["can_rtt_ms"], label="can_rtt_ms", color="tab:red")
  axes[1].set_ylabel("control latency (ms)")
  axes[1].legend()
  axes[1].grid(True)

  axes[2].plot(
    t, rows["motor_state_age_ms"], label="motor_state_age_ms", color="tab:blue"
  )
  axes[2].plot(
    t, rows["encoder_state_age_ms"], label="encoder_state_age_ms", color="tab:green"
  )
  axes[2].set_ylabel("state read latency (ms)")
  axes[2].set_xlabel("time (s)")
  axes[2].legend()
  axes[2].grid(True)

  fig.tight_layout()
  png_path = os.path.splitext(csv_path)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  print(f"그래프 저장 완료: {png_path}")


def _print_summary(csv_path: str) -> None:
  """chirp는 시간이 지날수록 고주파가 되므로, 앞/뒤 절반을 나눠 비교한다."""
  rows: list[dict[str, float]] = []
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      rows.append({k: float(v) for k, v in row.items()})

  if len(rows) < 2:
    return

  mid = len(rows) // 2
  low_freq, high_freq = rows[:mid], rows[mid:]

  metrics = ["loop_dt_ms", "can_rtt_ms", "motor_state_age_ms", "encoder_state_age_ms"]
  print("\n=== 저주파 구간(앞 절반) vs 고주파 구간(뒷 절반) 지연시간 비교 ===")
  header = f"{'metric':<22}{'low_freq mean':>15}{'low_freq max':>15}{'high_freq mean':>16}{'high_freq max':>16}"
  print(header)
  for name in metrics:
    low_vals = np.array([r[name] for r in low_freq])
    high_vals = np.array([r[name] for r in high_freq])
    print(
      f"{name:<22}{low_vals.mean():>15.3f}{low_vals.max():>15.3f}"
      f"{high_vals.mean():>16.3f}{high_vals.max():>16.3f}"
    )


def run(cfg: LatencyTestConfig) -> None:
  sysid_cfg = _to_sysid_cfg(cfg)
  duration_s = signal_duration_s(sysid_cfg)
  n_samples = int(duration_s * cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / cfg.control_hz
  targets_deg = build_chirp_targets(sysid_cfg, sample_times)

  bus = _open_bus(cfg.interface, cfg.channel)
  reader = RobotStateReader(
    motor_id=cfg.motor_id,
    encoder_port=cfg.encoder_port,
    can_interface=cfg.interface,
    can_channel=cfg.channel,
    motor_mode="passive",
    motor_rate_hz=cfg.control_hz,
    encoder_baud=cfg.encoder_baud,
  )

  os.makedirs(cfg.log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join(cfg.log_dir, f"latency_test_{ts}.csv")
  csv_file = open(csv_path, "w", newline="")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(_CSV_HEADER)

  target_period_ms = 1000.0 / cfg.control_hz
  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  print(f"chirp {cfg.chirp_f0_hz}Hz -> {cfg.chirp_f1_hz}Hz over {duration_s:.1f}s")
  print(f"target loop period: {target_period_ms:.2f} ms ({cfg.control_hz:.0f} Hz)")
  print(f"CSV 로그: {csv_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(여기 신호 완료)"
  prev_loop_start = start_t

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

    print("Starting latency test loop.")
    sample_idx = 0
    while time.monotonic() < deadline and sample_idx < n_samples:
      loop_start = time.monotonic()
      loop_dt_ms = (loop_start - prev_loop_start) * 1000.0
      prev_loop_start = loop_start

      target_deg = float(targets_deg[sample_idx])

      t_send = time.monotonic()
      ok = mit_position_command(
        bus,
        cfg.motor_id,
        target_deg,
        kp=cfg.kp,
        kd=cfg.kd,
        torque_nm=0.0,
      )
      can_rtt_ms = (time.monotonic() - t_send) * 1000.0
      if not ok:
        raise RuntimeError("No MIT reply received from motor")

      now = time.monotonic()
      motor_latest = reader.motor_reader.latest
      encoder_latest = (
        reader.encoder_reader.latest if reader.encoder_reader is not None else None
      )

      if motor_latest is None:
        raise RuntimeError("Motor state is not available from RobotStateReader")
      if encoder_latest is None:
        raise RuntimeError(
          "Pendulum angle is not available from RobotStateReader. "
          "Provide --encoder-port and check the serial encoder stream."
        )

      motor_angle_deg = motor_latest.angle_deg
      pole_angle_deg = encoder_latest.angle_deg
      motor_state_age_ms = (now - motor_latest.timestamp) * 1000.0
      encoder_state_age_ms = (now - encoder_latest.timestamp) * 1000.0

      if (
        cfg.abort_on_torque_nm and abs(motor_latest.torque_nm) > cfg.abort_on_torque_nm
      ):
        raise RuntimeError(
          f"Aborting: |motor_torque_nm|={abs(motor_latest.torque_nm):.2f} "
          f"exceeded abort_on_torque_nm={cfg.abort_on_torque_nm}"
        )

      elapsed_t = now - start_t
      csv_writer.writerow(
        [
          f"{elapsed_t:.4f}",
          f"{target_deg:.3f}",
          f"{motor_angle_deg:.3f}",
          f"{pole_angle_deg:.3f}",
          f"{loop_dt_ms:.3f}",
          f"{target_period_ms:.3f}",
          f"{can_rtt_ms:.3f}",
          f"{motor_state_age_ms:.3f}",
          f"{encoder_state_age_ms:.3f}",
        ]
      )

      if sample_idx % 50 == 0:
        print(
          "t={:6.2f}s  target={:+8.3f}deg  loop_dt={:6.2f}ms  can_rtt={:6.2f}ms  "
          "motor_age={:6.2f}ms  enc_age={:6.2f}ms".format(
            elapsed_t,
            target_deg,
            loop_dt_ms,
            can_rtt_ms,
            motor_state_age_ms,
            encoder_state_age_ms,
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

  print(f"측정 종료 ({stopped_reason}).")

  _print_summary(csv_path)

  if cfg.plot:
    _plot_log(csv_path)


def main() -> None:
  cfg = tyro.cli(LatencyTestConfig)
  run(cfg)


if __name__ == "__main__":
  main()
