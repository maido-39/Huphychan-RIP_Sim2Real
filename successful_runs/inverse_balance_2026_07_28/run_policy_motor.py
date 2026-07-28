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

정책 추론 주기(policy_hz)
- `inverse_env_cfg_new.py`로 학습한 정책은 시뮬레이션에서 `decimation=4`로
  학습되었다. 즉 정책은 매 물리 스텝(200Hz)마다가 아니라 4스텝에 한 번(50Hz)만
  새 액션을 내고, 그 사이에는 이전 목표각이 그대로 유지된 채 물리 스텝만 진행됐다.
- 이 리듬을 실물에서도 맞추기 위해, 모터로 명령은 여전히 `control_hz`(기본 200Hz)로
  매 루프마다 보내되(저수준 PD 트래킹/CAN 통신은 그대로 촘촘하게 유지), 정책
  추론(policy.infer)만 `policy_hz`(기본 50Hz)에 맞춰 몇 루프에 한 번씩 호출하고,
  그 사이 루프에서는 직전에 계산된 목표각을 그대로 재전송한다.
- `--policy-hz`를 `--control-hz`와 같은 값으로 주면 예전 방식(매 루프 새 액션)으로
  되돌아간다 (`decimation=1`로 학습한 예전 체크포인트를 쓸 때 사용).

예시 실행
```bash
uv run python -u run_policy_motor.py \
  --checkpoint-file successful_checkpoints/inverse_balance_action_rate_model_6000.pt \
  --motor-id 8 \
  --channel can1 \
  --encoder-port /dev/ttyACM1 \
  --action-scale-multiplier 1.0
```
"""

import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tyro

from .commission_motor import (
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
from .real_policy_inference import (
  InverseRealPolicy,
  RealInferenceConfig,
  RealMeasurement,
)
from .robot_state_reader import RobotStateReader


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
  kp: float = 10.0  # K_P 값
  kd: float = 0.45  # K_D 값
  velocity_limit_deg_s: float = 0.0
  torque_limit_nm: float = 0.0
  encoder_baud: int = 115200
  action_scale_multiplier: float = 0.2

  # 정책 추론 주기: control_hz보다 낮게 주면, 그 사이 루프에서는 policy를 다시
  # 부르지 않고 직전 목표각을 그대로 유지한다 (inverse_env_cfg_new.py의
  # decimation=4, 즉 200Hz/4=50Hz와 맞추기 위한 기본값).
  policy_hz: float = 50.0

  # 명령 지연(초): 시뮬레이션(inverse_env_cfg_new.py의 delay_min_lag=delay_max_lag=4
  # 물리스텝, timestep=0.005s이면 20ms)에는 있지만 실물은 거의 없는 지연을 인위적으로
  # 흉내내기 위한 값. 0.0이면 지연 없음(기존과 동일). 필요한 만큼 올려서 튜닝한다.
  command_delay_s: float = 0.0

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
  vmax: float = 44.0,
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
  "motor_torque_nm",
  "pole_angle_deg",
  "pole_vel_deg_s",
  "policy_action",
  "policy_updated",
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
  torque_list: list[float] = []
  pole_list, pole_vel_list, action_list, updated_list = [], [], [], []
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      t_list.append(float(row["time_s"]))
      target_list.append(float(row["target_angle_deg"]))
      motor_list.append(float(row["motor_angle_deg"]))
      motor_vel_list.append(float(row["motor_vel_deg_s"]))
      torque_list.append(float(row["motor_torque_nm"]))
      pole_list.append(float(row["pole_angle_deg"]))
      pole_vel_list.append(float(row["pole_vel_deg_s"]))
      action_list.append(float(row["policy_action"]))
      updated_list.append(int(row["policy_updated"]))

  if not t_list:
    print("로그가 비어 있어 그래프를 건너뜁니다.")
    return

  # CSV에는 원본 각도(예: 0~360 랩어라운드)를 그대로 저장하지만, 그래프에서는
  # 그 경계(359 -> 1 같은)에서 생기는 수직선을 없애기 위해 unwrap한 값만 쓴다.
  motor_list_unwrapped = np.degrees(np.unwrap(np.radians(motor_list)))
  pole_list_unwrapped = np.degrees(np.unwrap(np.radians(pole_list)))

  fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

  axes[0].plot(t_list, target_list, label="target_angle_deg", linestyle="--")
  axes[0].plot(t_list, motor_list_unwrapped, label="motor_angle_deg")
  update_t = [t for t, u in zip(t_list, updated_list, strict=True) if u]
  update_target = [v for v, u in zip(target_list, updated_list, strict=True) if u]
  # axes[0].scatter(
  #  update_t, update_target, s=10, color="black", zorder=3, label="policy_updated"
  # )
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

  axes[3].plot(t_list, torque_list, color="tab:brown")
  axes[3].set_ylabel("motor_torque (Nm)")
  axes[3].grid(True)

  axes[4].plot(t_list, action_list, color="tab:red")
  axes[4].set_ylabel("policy_action")
  axes[4].set_xlabel("time (s)")
  axes[4].grid(True)

  fig.tight_layout()
  png_path = os.path.splitext(csv_path)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  print(f"그래프 저장 완료: {png_path}")


def run(cfg: RunPolicyMotorConfig) -> None:
  print(f"정책 로딩 중... (체크포인트: {cfg.checkpoint_file})", flush=True)
  policy = _make_policy(cfg)
  print("정책 로딩 완료.", flush=True)
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

    policy_decimation = max(1, round(float(cfg.control_hz) / float(cfg.policy_hz)))
    print(
      f"control_hz={cfg.control_hz:.1f}  policy_hz={cfg.policy_hz:.1f}  "
      f"-> policy는 {policy_decimation}루프마다 한 번만 갱신, 그 사이엔 목표각 유지"
    )

    # 명령 지연 버퍼: 매 루프 최신 desired_angle_deg를 넣고, command_delay_steps
    # 루프 전에 넣었던(가장 오래된) 값을 꺼내 실제로 모터에 보낸다.
    command_delay_steps = max(
      0, round(float(cfg.command_delay_s) * float(cfg.control_hz))
    )
    command_delay_buffer: deque[float] = deque(
      [0.0] * (command_delay_steps + 1), maxlen=command_delay_steps + 1
    )
    print(
      f"command_delay_s={cfg.command_delay_s:.4f} -> "
      f"{command_delay_steps}루프({command_delay_steps / cfg.control_hz * 1000:.1f}ms) 명령 지연"
    )

    desired_angle_deg = 0.0
    last_policy_action = 0.0
    sample_idx = 0
    prev_time = time.monotonic()
    period_s = 1.0 / float(cfg.control_hz)
    deadline = prev_time + float(cfg.max_runtime_s)

    print("Starting RL control loop.")
    while time.monotonic() < deadline:
      loop_start = time.monotonic()

      # 최신 목표각을 지연 버퍼에 넣고, command_delay_steps 루프 전에 넣었던
      # (가장 오래된) 값을 꺼내 이번에 실제로 모터에 전송한다.
      command_delay_buffer.append(desired_angle_deg)
      sent_target_deg = command_delay_buffer[0]

      reply = _send_mit_position_and_get_reply(
        bus,
        cfg.motor_id,
        sent_target_deg,
        kp=cfg.kp,
        kd=cfg.kd,
        velocity_deg_s=cfg.velocity_limit_deg_s,
        torque_nm=cfg.torque_limit_nm,
      )

      if reply is None:
        raise RuntimeError("No MIT reply received from motor")

      state = reader.get_state()
      now = time.monotonic()
      prev_time = now

      cylinder_angle_deg = state.motor_angle_deg
      cylinder_vel_deg_s = state.motor_velocity_deg_s
      cylinder_torque_nm = state.motor_torque_nm
      pole_angle_deg = state.pendulum_angle_deg

      if (
        cylinder_angle_deg is None
        or cylinder_vel_deg_s is None
        or cylinder_torque_nm is None
      ):
        raise RuntimeError("Motor state is not available from RobotStateReader")
      if pole_angle_deg is None:
        raise RuntimeError(
          "Pendulum angle is not available from RobotStateReader. "
          "Provide --encoder-port and check the serial encoder stream."
        )
      pole_vel_deg_s = state.pendulum_velocity_deg_s

      if pole_vel_deg_s is None:
        raise RuntimeError(
          "Pendulum velocity is not available from RobotStateReader. "
          "Check robot_state_reader.py and encoder input."
        )

      meas = RealMeasurement(
        cylinder_angle_deg=cylinder_angle_deg,
        pole_angle_deg=pole_angle_deg,
        cylinder_vel_deg_s=cylinder_vel_deg_s,
        pole_vel_deg_s=pole_vel_deg_s,
        torque_nm=None,
      )

      policy_updated = sample_idx % policy_decimation == 0
      if policy_updated:
        result = policy.infer(meas)
        desired_angle_deg = result["target_angle_deg"] * float(
          cfg.action_scale_multiplier
        )
        last_policy_action = result["policy_action"]
      # else: policy를 다시 부르지 않고, 직전 desired_angle_deg/last_policy_action을
      # 그대로 유지한다 (학습 시 decimation과 맞추기 위함).

      elapsed_t = now - start_t
      csv_writer.writerow(
        [
          f"{elapsed_t:.4f}",
          f"{sent_target_deg:.3f}",
          f"{cylinder_angle_deg:.3f}",
          f"{cylinder_vel_deg_s:.3f}",
          f"{cylinder_torque_nm:.3f}",
          f"{pole_angle_deg:.3f}",
          f"{pole_vel_deg_s:.3f}",
          f"{last_policy_action:.6f}",
          f"{int(policy_updated)}",
        ]
      )
      txt_file.write(
        "{:10.4f}  {:16.3f}  {:18.3f}\n".format(
          elapsed_t, sent_target_deg, cylinder_angle_deg
        )
      )

      print(
        "motor={:+8.3f} deg  pole={:+8.3f} deg  torque={:+6.3f} Nm  "
        "cmd={:+8.3f} deg  act={:+6.3f}  scale={:.2f}  upd={}".format(
          cylinder_angle_deg,
          pole_angle_deg,
          cylinder_torque_nm,
          desired_angle_deg,
          last_policy_action,
          cfg.action_scale_multiplier,
          "Y" if policy_updated else ".",
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
