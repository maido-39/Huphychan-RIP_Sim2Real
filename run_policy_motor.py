#!/usr/bin/env python3
from __future__ import annotations

"""Run the trained inverse policy on the real motor.

이 파일은 `commission_motor.py`의 MIT 모터 제어 기능과
`real_policy_inference.py`의 policy 추론 기능을 이어서,
실제 제어 루프를 구성하는 실행 스크립트입니다.

중요:
- 이 파일은 `robot_state_reader.py`를 그대로 사용한다.
- 즉, 상태 읽기 샘플은 수정하지 않고, 여기서 가져다 쓰는 방식이다.
- 액추에이터 제어는 이 파일이 담당하고, 상태 읽기는 RobotStateReader가 담당한다.

로깅/플롯 (sin_position_test.py와 동일한 방식)
- 200Hz 루프 안에서는 CSV 로깅만 하고, 터미널 출력은 요약만 표시한다
  (실시간 그래프는 루프 타이밍을 해치므로 넣지 않음).
- 종료 시(정상 종료/Ctrl+C/에러 모두) CSV를 읽어 PNG 그래프를 저장한다.
- 제어 목표각(target)과 실측각(current)만 시간에 대해 보기 쉽게 정리한
  텍스트 파일(.txt)도 별도로 생성한다.

사용 순서
1. `commission_motor.py`로 모터가 MIT 모드이며 원하는 ID인지 먼저 확인한다.
2. `robot_state_reader.py` 단독 실행으로
   - motor_angle_deg
   - motor_velocity_deg_s
   - pendulum_angle_deg
   가 정상적으로 들어오는지 먼저 확인한다.
3. 그 다음 이 파일을 실행한다.

권장 첫 실행 방법
- 처음에는 반드시 약한 배율로 시작한다.
- `action_scale_multiplier=0.1 ~ 0.3` 정도로 시작해서
  실제 장비가 어느 정도로 움직이는지 본 뒤 점차 올린다.
- 처음부터 1.0으로 주면 학습 정책이 생각보다 크게 움직일 수 있다.

예시 실행
```bash
uv run python src/mjlab/tasks/inverse/run_policy_motor.py \
  --checkpoint-file logs/rsl_rl/inverse_balance/2026-07-08_12-13-31/model_950.pt \
  --motor-id 8 \
  --channel can0 \
  --encoder-port /dev/ttyACM0 \
  --action-scale-multiplier 0.2
```
"""

import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
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
  parse_mit_reply_position,
  set_zero_mit,
)
from mjlab.tasks.inverse.real_policy_inference import (
  InverseRealPolicy,
  RealInferenceConfig,
  RealMeasurement,
)
from mjlab.tasks.inverse.robot_state_reader import RobotStateReader


@dataclass(frozen=True)
class RunPolicyMotorConfig:
  checkpoint_file: str
  motor_id: int
  interface: str = "socketcan"
  channel: str = "can0"
  device: str = "cpu"
  encoder_port: str | None = None

  # 실물 축 부호/오프셋 보정
  cylinder_sign: float = 1.0
  pole_sign: float = 1.0
  cylinder_zero_deg: float = 0.0
  pole_zero_deg: float = 0.0

  # 제어 루프 설정
  control_hz: float = 200.0
  state_read_hz: float = 200.0
  kp: float = 3.0  # K_P 값
  kd: float = 0.0  # K_D 값
  velocity_limit_deg_s: float = 0.0
  torque_limit_nm: float = 0.0
  encoder_baud: int = 115200
  action_scale_multiplier: float = 0.2

  # 안전 설정
  max_runtime_s: float = 30.0
  start_with_zero_set: bool = False
  require_enable_ack: bool = True

  # 로깅/플롯 설정
  log_dir: str = "logs"
  plot: bool = True


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


def _make_policy(cfg: RunPolicyMotorConfig) -> InverseRealPolicy:
  policy_cfg = RealInferenceConfig(
    checkpoint_file=cfg.checkpoint_file,
    device=cfg.device,
    cylinder_sign=cfg.cylinder_sign,
    pole_sign=cfg.pole_sign,
    cylinder_zero_deg=cfg.cylinder_zero_deg,
    pole_zero_deg=cfg.pole_zero_deg,
  )
  return InverseRealPolicy(policy_cfg)


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


_CSV_HEADER = [
  "time_s",
  "target_angle_deg",
  "motor_angle_deg",
  "motor_vel_deg_s",
  "pole_angle_deg",
  "pole_vel_deg_s",
  "policy_action",
]


def _plot_log(csv_path: str) -> None:
  """sin_position_test.py와 동일하게, 종료 후 CSV를 한 번에 읽어 PNG로 저장한다."""
  try:
    import matplotlib

    matplotlib.use("Agg")  # GUI 없이 파일로만 저장 (서버/헤드리스 환경에서도 동작)
    import matplotlib.pyplot as plt
  except ImportError:
    print(
      "matplotlib이 설치되어 있지 않아 그래프를 건너뜁니다. (pip install matplotlib)"
    )
    return

  t_list, target_list, motor_list, motor_vel_list = [], [], [], []
  pole_list, pole_vel_list, action_list = [], [], []
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      t_list.append(float(row["time_s"]))
      target_list.append(float(row["target_angle_deg"]))
      motor_list.append(float(row["motor_angle_deg"]))
      motor_vel_list.append(float(row["motor_vel_deg_s"]))
      pole_list.append(float(row["pole_angle_deg"]))
      pole_vel_list.append(float(row["pole_vel_deg_s"]))
      action_list.append(float(row["policy_action"]))

  if not t_list:
    print("로그가 비어 있어 그래프를 건너뜁니다.")
    return

  # CSV에는 원본 각도(예: 0~360 랩어라운드)를 그대로 저장하지만, 그래프에서는
  # 그 경계(359 -> 1 같은)에서 생기는 수직선을 없애기 위해 unwrap한 값만 쓴다.
  motor_list_unwrapped = np.degrees(np.unwrap(np.radians(motor_list)))
  pole_list_unwrapped = np.degrees(np.unwrap(np.radians(pole_list)))

  fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

  axes[0].plot(t_list, target_list, label="target_angle_deg", linestyle="--")
  axes[0].plot(t_list, motor_list_unwrapped, label="motor_angle_deg")
  axes[0].set_ylabel("motor angle (deg)")
  axes[0].legend()
  axes[0].grid(True)

  axes[1].plot(t_list, pole_list_unwrapped, color="tab:green")
  axes[1].set_ylabel("pendulum angle (deg)")
  axes[1].grid(True)

  axes[2].plot(t_list, motor_vel_list, color="tab:orange", label="motor_vel_deg_s")
  axes[2].plot(t_list, pole_vel_list, color="tab:purple", label="pole_vel_deg_s")
  axes[2].set_ylabel("velocity (deg/s)")
  axes[2].legend()
  axes[2].grid(True)

  axes[3].plot(t_list, action_list, color="tab:red")
  axes[3].set_ylabel("policy_action")
  axes[3].set_xlabel("time (s)")
  axes[3].grid(True)

  fig.tight_layout()
  png_path = os.path.splitext(csv_path)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  print(f"그래프 저장 완료: {png_path}")


def run(cfg: RunPolicyMotorConfig) -> None:
  policy = _make_policy(cfg)
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

  # 로그 파일 준비 (sin_position_test.py와 동일한 방식: CSV + 종료 후 PNG plot)
  os.makedirs(cfg.log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join(cfg.log_dir, f"policy_run_{ts}.csv")
  txt_path = os.path.join(cfg.log_dir, f"policy_run_{ts}_target_vs_current.txt")

  csv_file = open(csv_path, "w", newline="")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(_CSV_HEADER)

  txt_file = open(txt_path, "w")
  txt_file.write(
    "{:>10}  {:>16}  {:>18}\n".format("time_s", "target_angle_deg", "current_angle_deg")
  )

  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  print(f"CSV 로그: {csv_path}")
  print(f"목표/현재각 텍스트 로그: {txt_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(max_runtime_s 도달)"

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

    _wait_for_initial_state(reader)  # <- enable 이후로 옮김

    desired_angle_deg = 0.0
    prev_time = time.monotonic()
    period_s = 1.0 / float(cfg.control_hz)
    deadline = prev_time + float(cfg.max_runtime_s)

    print("Starting RL control loop.")
    while time.monotonic() < deadline:
      loop_start = time.monotonic()

      # 이번 사이클에 실제로 모터에 전송하는 목표각 (로그용으로 미리 저장해 둔다)
      sent_target_deg = desired_angle_deg

      reply = _send_mit_position_and_get_reply(
        bus,
        cfg.motor_id,
        desired_angle_deg,
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
      pole_vel_deg_s = 0.0

      encoder_state = (
        reader.encoder_reader.latest if reader.encoder_reader is not None else None
      )
      previous_encoder_state = getattr(run, "_previous_encoder_state", None)
      if encoder_state is not None and previous_encoder_state is not None:
        enc_dt = max(encoder_state.timestamp - previous_encoder_state.timestamp, 1e-4)
        raw_delta = encoder_state.angle_deg - previous_encoder_state.angle_deg
        delta_deg = (raw_delta + 180) % 360 - 180  # wrap 경계 최단경로 보정
        pole_vel_deg_s = delta_deg / enc_dt
      setattr(run, "_previous_encoder_state", encoder_state)

      meas = RealMeasurement(
        cylinder_angle_deg=cylinder_angle_deg,
        pole_angle_deg=pole_angle_deg,
        cylinder_vel_deg_s=cylinder_vel_deg_s,
        pole_vel_deg_s=pole_vel_deg_s,
        torque_nm=None,
      )
      result = policy.infer(meas)
      desired_angle_deg = result["target_angle_deg"] * float(
        cfg.action_scale_multiplier
      )

      elapsed_t = now - start_t
      csv_writer.writerow(
        [
          f"{elapsed_t:.4f}",
          f"{sent_target_deg:.3f}",
          f"{cylinder_angle_deg:.3f}",
          f"{cylinder_vel_deg_s:.3f}",
          f"{pole_angle_deg:.3f}",
          f"{pole_vel_deg_s:.3f}",
          f"{result['policy_action']:.6f}",
        ]
      )
      txt_file.write(
        "{:10.4f}  {:16.3f}  {:18.3f}\n".format(
          elapsed_t, sent_target_deg, cylinder_angle_deg
        )
      )

      print(
        "motor={:+8.3f} deg  pole={:+8.3f} deg  "
        "cmd={:+8.3f} deg  act={:+6.3f}  scale={:.2f}".format(
          cylinder_angle_deg,
          pole_angle_deg,
          desired_angle_deg,
          result["policy_action"],
          cfg.action_scale_multiplier,
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
      txt_file.close()
      print(f"CSV 로그 저장 완료: {csv_path}")
      print(f"목표/현재각 텍스트 로그 저장 완료: {txt_path}")

  print(f"제어 종료 ({stopped_reason}).")

  if cfg.plot:
    _plot_log(csv_path)


def main() -> None:
  cfg = tyro.cli(RunPolicyMotorConfig)
  run(cfg)


if __name__ == "__main__":
  main()