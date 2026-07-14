#!/usr/bin/env python3
"""고정 게인(Kp=5, Kd=1)으로 0 -> 180 -> 360 -> 180 -> 0 스텝 시퀀스를 실행하고
CSV/PNG로 로그를 남기는 스크립트.

시작 후 SETTLE_DURATION_S(기본 1초)간 0도에서 대기하다가 첫 스텝(180도)으로
출발한다. 이후 각 스텝은 STEP_HOLD_DURATION_S(기본 5초)씩 유지한 뒤 다음
스텝(180 -> 360 -> 180 -> 0)으로 넘어가고, 마지막 스텝(0도)도 5초 유지한
뒤 종료한다.

목표속도(velocity_deg_s)는 항상 0, 피드포워드 토크(torque_ff)도 항상 0으로
고정한다 -- 순수 위치 PD(Kp/Kd)로만 움직인다.
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional

from commission_motor import (
  _open_bus,
  _shutdown_bus,
  disable_mit,
  enable_mit,
  mit_position_command,
)
from robot_state_reader import RobotStateReader

# ---- 고정 제어 게인 ----
KP = 5.0
KD = 1.0
VELOCITY_DEG_S = 0.0
TORQUE_FF_NM = 0.0

# ---- 스텝 시퀀스 ----
SETTLE_DURATION_S = 1.0  # 시작 직후 0도에서 대기하는 시간
STEP_HOLD_DURATION_S = 5.0  # 각 스텝을 유지하는 시간
STEP_TARGETS_DEG = [180.0, 0]#, 360.0, 180.0, 0.0]

CONTROL_HZ = 200.0

MOTOR_ID = 8
INTERFACE = "socketcan"
CHANNEL = "can0"
# pole 인코더가 연결돼 있으면 포트를 지정한다. 없으면 None으로 두면
# pole 컬럼만 빈 값으로 기록되고 모터 값은 정상 기록된다.
ENCODER_PORT: Optional[str] = "/dev/ttyACM0"

# 안전 설정: 측정 토크가 이 값을 넘으면 즉시 중단한다.
ABORT_ON_TORQUE_NM = 8.0

_LOG_HEADER = [
  "time_s",
  "target_angle_deg",
  "motor_angle_deg",
  "motor_vel_deg_s",
  "motor_torque_nm",
  "pole_angle_deg",
  "pole_vel_deg_s",
]


def build_schedule() -> tuple[list[tuple[float, float]], float]:
  """(이 목표가 시작되는 경과시간, 목표각) 리스트와 총 실행 시간을 만든다."""
  schedule = [(0.0, 0.0)]  # 시작 시 SETTLE_DURATION_S 동안 0도에서 대기
  t = SETTLE_DURATION_S
  for target_deg in STEP_TARGETS_DEG:
    schedule.append((t, target_deg))
    t += STEP_HOLD_DURATION_S
  return schedule, t


def target_at(schedule: list[tuple[float, float]], elapsed_t: float) -> float:
  target_deg = schedule[0][1]
  for start_t, value in schedule:
    if elapsed_t >= start_t:
      target_deg = value
    else:
      break
  return target_deg


def _wait_for_initial_state(reader: RobotStateReader, timeout_s: float = 3.0) -> None:
  require_pole = reader.encoder_reader is not None
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    state = reader.get_state()
    if (
      state.motor_angle_deg is not None
      and state.motor_velocity_deg_s is not None
      and (not require_pole or state.pendulum_angle_deg is not None)
    ):
      return
    time.sleep(0.01)
  raise RuntimeError(
    "RobotStateReader의 초기 상태를 기다리다 타임아웃했습니다. "
    "CAN 모터 피드백과 인코더 시리얼 연결을 확인하세요."
  )


def _plot_log(csv_path: str) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
  except ImportError:
    print(
      "matplotlib이 설치되어 있지 않아 그래프를 건너뜁니다. (pip install matplotlib)"
    )
    return

  rows: dict[str, list[float]] = {name: [] for name in _LOG_HEADER}
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      for name in _LOG_HEADER:
        value = row[name]
        rows[name].append(float(value) if value != "" else float("nan"))

  if not rows["time_s"]:
    print("로그가 비어 있어 그래프를 건너뜁니다.")
    return

  t = rows["time_s"]
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


def main() -> None:
  schedule, total_duration_s = build_schedule()
  print("스텝 스케줄:")
  for start_t, target_deg in schedule:
    print(f"  t={start_t:5.1f}s -> target={target_deg:+7.1f}deg")
  print(f"총 실행 시간: {total_duration_s:.1f}s (Kp={KP}, Kd={KD})")

  bus = _open_bus(INTERFACE, CHANNEL)
  reader = RobotStateReader(
    motor_id=MOTOR_ID,
    encoder_port=ENCODER_PORT,
    can_interface=INTERFACE,
    can_channel=CHANNEL,
    motor_mode="passive",
    motor_rate_hz=CONTROL_HZ,
  )

  os.makedirs("logs", exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join("logs", f"step_sequence_{ts}.csv")
  csv_file = open(csv_path, "w", newline="")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(_LOG_HEADER)
  print(f"CSV 로그: {csv_path}")

  period_s = 1.0 / CONTROL_HZ
  stopped_reason = "정상 종료(시퀀스 완료)"
  previous_encoder_state = None

  try:
    print("RobotStateReader 시작...")
    reader.start()

    print(f"모터 ID {MOTOR_ID} 활성화(Enable) 시도 중...")
    if not enable_mit(bus, MOTOR_ID):
      raise RuntimeError("모터 활성화(enable_mit)가 응답하지 않았습니다.")
    time.sleep(0.1)

    _wait_for_initial_state(reader)

    start_t = time.monotonic()
    print("스텝 시퀀스 시작.")
    sample_idx = 0
    while True:
      loop_start = time.monotonic()
      elapsed_t = loop_start - start_t
      if elapsed_t >= total_duration_s:
        break

      target_deg = target_at(schedule, elapsed_t)

      ok = mit_position_command(
        bus=bus,
        motor_id=MOTOR_ID,
        position_deg=target_deg,
        kp=KP,
        kd=KD,
        velocity_deg_s=VELOCITY_DEG_S,
        torque_nm=TORQUE_FF_NM,
        timeout_s=0.03,
      )
      if not ok:
        raise RuntimeError("모터로부터 MIT 응답을 받지 못했습니다.")

      state = reader.get_state()
      motor_angle_deg = state.motor_angle_deg
      motor_vel_deg_s = state.motor_velocity_deg_s
      motor_torque_nm = state.motor_torque_nm
      pole_angle_deg = state.pendulum_angle_deg

      if motor_angle_deg is None or motor_vel_deg_s is None or motor_torque_nm is None:
        raise RuntimeError("모터 상태를 RobotStateReader에서 받지 못했습니다.")

      if abs(motor_torque_nm) > ABORT_ON_TORQUE_NM:
        raise RuntimeError(
          f"중단: |motor_torque_nm|={abs(motor_torque_nm):.2f}가 "
          f"ABORT_ON_TORQUE_NM={ABORT_ON_TORQUE_NM}를 초과했습니다."
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

      csv_writer.writerow(
        [
          f"{elapsed_t:.4f}",
          f"{target_deg:.3f}",
          f"{motor_angle_deg:.3f}",
          f"{motor_vel_deg_s:.3f}",
          f"{motor_torque_nm:.4f}",
          "" if pole_angle_deg is None else f"{pole_angle_deg:.3f}",
          f"{pole_vel_deg_s:.3f}",
        ]
      )

      if sample_idx % 40 == 0:
        print(
          "t={:6.2f}s  target={:+7.1f}deg  motor={:+8.2f}deg  "
          "pole={}  torque={:+5.2f}Nm".format(
            elapsed_t,
            target_deg,
            motor_angle_deg,
            "N/A" if pole_angle_deg is None else f"{pole_angle_deg:+8.2f}deg",
            motor_torque_nm,
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
    print("모터 비활성화(Disable) 중...")
    try:
      disable_mit(bus, MOTOR_ID)
    finally:
      reader.stop()
      _shutdown_bus(bus)
      csv_file.close()
      print(f"CSV 로그 저장 완료: {csv_path}")

  print(f"스텝 시퀀스 종료 ({stopped_reason}).")
  _plot_log(csv_path)


if __name__ == "__main__":
  main()
