#!/usr/bin/env python3
"""학습된 Motor-Sine 정책을 실제 MIT/CAN 모터에서 실행한다.

이 실행 파일은 폴 엔코더를 사용하지 않는다.

입력
- 모터 내부 엔코더 각도
- 모터 내부 엔코더 속도
- 실행 시작 이후 경과시간으로 계산한 sine 목표
- 기본적으로 실행 시작 위치를 software zero(0도)로 사용

처리
- motor_sine_real_policy_inference.py에서 체크포인트 로드
- 50 Hz policy 추론
- 학습과 동일한 위치 action scale 및 목표 rate/acceleration limiter 적용
- MIT/CAN 위치 명령 전송

처음 실행할 때
- action_scale_multiplier=0.1~0.2
- max_runtime_s=5
- 모터를 무부하 또는 안전하게 고정한 상태
- 비상 정지와 전원 차단 수단 확보
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import tyro

from mjlab.tasks.inverse.commission_motor import (
  _float_to_uint,
  _matches_mit_reply,
  _open_bus,
  _shutdown_bus,
  _wait_reply,
  can,
  disable_mit,
  enable_mit,
  set_zero_mit,
)
from mjlab.tasks.inverse.motor_sine_real_policy_inference import (
  MotorMeasurement,
  MotorSineInferenceConfig,
  MotorSineRealPolicy,
)
from mjlab.tasks.inverse.robot_state_reader import RobotStateReader


@dataclass(frozen=True)
class RunMotorSinePolicyConfig:
  checkpoint_file: str
  motor_id: int

  interface: str = "socketcan"
  channel: str = "can0"
  device: str = "cpu"

  # 실물 모터 좌표계 보정.
  cylinder_sign: float = 1.0
  cylinder_zero_deg: float = 0.0

  # True이면 실행 시작 시 읽은 현재 모터 각도를 software zero로 사용한다.
  # 하드웨어 영점은 변경하지 않으며, 현재 자세를 시뮬레이터의 0도로 본다.
  use_initial_position_as_zero: bool = True

  # 제어 주기와 MIT gains.
  control_hz: float = 50.0
  state_read_hz: float = 200.0
  kp: float = 16.5
  kd: float = 1.0

  # MIT 명령의 feed-forward 항목. 기본은 0.
  command_velocity_deg_s: float = 0.0
  command_torque_nm: float = 0.0

  # 처음에는 0.1~0.2 권장.
  action_scale_multiplier: float = 0.2

  # 안전 제한.
  max_runtime_s: float = 5.0
  max_abs_motor_velocity_deg_s: float = 720.0
  max_abs_position_error_deg: float = 90.0
  start_with_zero_set: bool = False
  require_enable_ack: bool = True

  # 로그.
  log_dir: str = "logs"
  print_every_n_steps: int = 10


_CSV_HEADER = [
  "time_s",
  "reference_target_deg",
  "command_target_deg",
  "motor_angle_deg",
  "motor_velocity_deg_s",
  "raw_policy_action",
  "clipped_policy_action",
  "applied_policy_action",
  "software_zero_deg",
]


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
  """MIT position command를 보내고 해당 모터의 reply를 기다린다."""
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

  message = can.Message(
    arbitration_id=int(motor_id),
    data=data,
    is_extended_id=False,
  )
  bus.send(message)

  return _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda msg: _matches_mit_reply(
      msg,
      motor_id=int(motor_id),
    ),
  )


def _wait_for_motor_state(
  reader: RobotStateReader,
  timeout_s: float = 3.0,
) -> None:
  """폴 엔코더 없이 모터 각도와 속도만 기다린다."""
  deadline = time.monotonic() + timeout_s

  while time.monotonic() < deadline:
    state = reader.get_state()
    if state.motor_angle_deg is not None and state.motor_velocity_deg_s is not None:
      return
    time.sleep(0.01)

  raise RuntimeError(
    "모터 초기 상태를 받지 못했습니다. CAN 연결, motor ID, MIT mode를 확인하세요."
  )


def _make_policy(
  cfg: RunMotorSinePolicyConfig,
  *,
  cylinder_zero_deg: float,
) -> MotorSineRealPolicy:
  """체크포인트를 로드하고 실물 모터의 software zero를 적용한다."""
  return MotorSineRealPolicy(
    MotorSineInferenceConfig(
      checkpoint_file=cfg.checkpoint_file,
      device=cfg.device,
      cylinder_sign=cfg.cylinder_sign,
      cylinder_zero_deg=float(cylinder_zero_deg),
      action_scale_multiplier=cfg.action_scale_multiplier,
    )
  )


def _circular_error_deg(
  target_deg: float,
  current_deg: float,
) -> float:
  """두 각도 사이의 최단 오차를 [-180, 180) 범위로 반환한다."""
  return (float(target_deg) - float(current_deg) + 180.0) % 360.0 - 180.0


def _nearest_equivalent_angle_deg(
  target_deg: float,
  current_deg: float,
) -> float:
  """target과 동치인 각도 중 현재 모터 각도에 가장 가까운 값을 반환한다.

  예:
  - current=352 deg, target=-10 deg -> 350 deg
  - current=5 deg, target=350 deg -> -10 deg
  """
  return float(current_deg) + _circular_error_deg(
    target_deg=target_deg,
    current_deg=current_deg,
  )


def _check_safety(
  cfg: RunMotorSinePolicyConfig,
  *,
  motor_angle_deg: float,
  motor_velocity_deg_s: float,
  command_target_deg: float,
) -> None:
  if abs(motor_velocity_deg_s) > cfg.max_abs_motor_velocity_deg_s:
    raise RuntimeError(f"모터 속도 안전 제한 초과: {motor_velocity_deg_s:.3f} deg/s")

  # 0/360도 경계를 고려한 최단 각도 오차를 사용한다.
  position_error_deg = _circular_error_deg(
    target_deg=command_target_deg,
    current_deg=motor_angle_deg,
  )
  if abs(position_error_deg) > cfg.max_abs_position_error_deg:
    raise RuntimeError(
      f"명령-실측 각도 차이 안전 제한 초과: {position_error_deg:.3f} deg"
    )


def run(cfg: RunMotorSinePolicyConfig) -> None:
  if abs(cfg.control_hz - 50.0) > 1e-9:
    raise ValueError("학습 policy 주기가 50 Hz이므로 control_hz는 50으로 사용하세요.")

  bus = _open_bus(cfg.interface, cfg.channel)

  # 폴 엔코더를 사용하지 않으므로 encoder_port를 전달하지 않는다.
  reader = RobotStateReader(
    motor_id=cfg.motor_id,
    encoder_port=None,
    can_interface=cfg.interface,
    can_channel=cfg.channel,
    motor_mode="passive",
    motor_rate_hz=cfg.state_read_hz,
  )

  os.makedirs(cfg.log_dir, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join(
    cfg.log_dir,
    f"motor_sine_policy_real_{timestamp}.csv",
  )

  csv_file = open(csv_path, "w", newline="")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(_CSV_HEADER)

  stopped_reason = "정상 종료(max_runtime_s 도달)"

  try:
    print("Starting RobotStateReader...")
    reader.start()

    if cfg.start_with_zero_set:
      print("Setting current motor position as hardware zero...")
      if not set_zero_mit(bus, cfg.motor_id):
        raise RuntimeError("set_zero_mit 응답을 받지 못했습니다.")
      time.sleep(0.1)

    print("Enabling motor...")
    enable_ok = enable_mit(bus, cfg.motor_id)
    if cfg.require_enable_ack and not enable_ok:
      raise RuntimeError("enable_mit 응답을 받지 못했습니다.")
    time.sleep(0.1)

    _wait_for_motor_state(reader)
    initial_state = reader.get_state()

    if (
      initial_state.motor_angle_deg is None
      or initial_state.motor_velocity_deg_s is None
    ):
      raise RuntimeError("초기 모터 상태가 유효하지 않습니다.")

    initial_motor_angle_deg = float(initial_state.motor_angle_deg)

    # 실행 시작 자세를 시뮬레이터의 0도로 사용한다.
    # 예: 실제 엔코더가 346.9도를 가리켜도 내부 policy에는 0도로 전달한다.
    if cfg.use_initial_position_as_zero:
      software_zero_deg = initial_motor_angle_deg
    else:
      software_zero_deg = cfg.cylinder_zero_deg

    print(
      "Motor coordinate setup: "
      f"initial={initial_motor_angle_deg:.3f} deg, "
      f"software_zero={software_zero_deg:.3f} deg, "
      f"sign={cfg.cylinder_sign:+.1f}"
    )

    policy = _make_policy(
      cfg,
      cylinder_zero_deg=software_zero_deg,
    )
    policy.reset(initial_motor_angle_deg)

    # 첫 command는 현재 물리 위치로 시작해 갑작스러운 점프를 막는다.
    command_target_deg = initial_motor_angle_deg

    start_time = time.monotonic()
    deadline = start_time + cfg.max_runtime_s
    period_s = 1.0 / cfg.control_hz
    step = 0

    print("Starting Motor-Sine RL control loop.")
    print(f"CSV log: {csv_path}")

    while time.monotonic() < deadline:
      loop_start = time.monotonic()
      elapsed_s = loop_start - start_time

      state = reader.get_state()
      if state.motor_angle_deg is None or state.motor_velocity_deg_s is None:
        raise RuntimeError("실행 중 모터 상태가 끊겼습니다.")

      measurement = MotorMeasurement(
        angle_deg=float(state.motor_angle_deg),
        velocity_deg_s=float(state.motor_velocity_deg_s),
        torque_nm=None,
      )

      result = policy.infer(
        measurement=measurement,
        elapsed_s=elapsed_s,
      )
      # 0/360도 경계에서도 현재 위치에서 가장 가까운 동치 각도로 변환한다.
      command_target_deg = _nearest_equivalent_angle_deg(
        target_deg=result["target_angle_deg"],
        current_deg=measurement.angle_deg,
      )

      _check_safety(
        cfg,
        motor_angle_deg=measurement.angle_deg,
        motor_velocity_deg_s=measurement.velocity_deg_s,
        command_target_deg=command_target_deg,
      )

      reply = _send_mit_position_and_get_reply(
        bus,
        cfg.motor_id,
        command_target_deg,
        kp=cfg.kp,
        kd=cfg.kd,
        velocity_deg_s=cfg.command_velocity_deg_s,
        torque_nm=cfg.command_torque_nm,
      )
      if reply is None:
        raise RuntimeError("MIT reply를 받지 못했습니다.")

      csv_writer.writerow(
        [
          f"{elapsed_s:.6f}",
          f"{result['reference_target_deg']:.6f}",
          f"{command_target_deg:.6f}",
          f"{measurement.angle_deg:.6f}",
          f"{measurement.velocity_deg_s:.6f}",
          f"{result['raw_policy_action']:.6f}",
          f"{result['clipped_policy_action']:.6f}",
          f"{result['applied_policy_action']:.6f}",
          f"{software_zero_deg:.6f}",
        ]
      )

      if step % max(1, cfg.print_every_n_steps) == 0:
        print(
          "t={:6.2f} ref={:+8.3f} motor={:+8.3f} "
          "vel={:+9.3f} cmd={:+8.3f} action={:+6.3f}".format(
            elapsed_s,
            result["reference_target_deg"],
            measurement.angle_deg,
            measurement.velocity_deg_s,
            command_target_deg,
            result["applied_policy_action"],
          )
        )

      step += 1
      elapsed_loop_s = time.monotonic() - loop_start
      sleep_s = period_s - elapsed_loop_s
      if sleep_s > 0.0:
        time.sleep(sleep_s)

  except KeyboardInterrupt:
    stopped_reason = "사용자 강제 종료(Ctrl+C)"
    print("\n" + stopped_reason)

  finally:
    print("Disabling motor...")

    try:
      disable_mit(bus, cfg.motor_id)
    except Exception as exc:
      print(f"Motor disable warning: {exc}")

    try:
      reader.stop()
    except Exception as exc:
      print(f"RobotStateReader stop warning: {exc}")

    _shutdown_bus(bus)
    csv_file.close()

  print(f"제어 종료: {stopped_reason}")
  print(f"CSV 로그 저장 완료: {csv_path}")


def main() -> None:
  cfg = tyro.cli(RunMotorSinePolicyConfig)
  run(cfg)


if __name__ == "__main__":
  main()
