#!/usr/bin/env python3
from __future__ import annotations

"""Run a *computed* (learning-free) LQR balance controller on the real motor.

`run_policy_motor.py`가 학습된 RL 체크포인트(policy.infer)를 불러와 목표각을
계산했다면, 이 파일은 그 자리를 `rotary_inverted_pendulum_lqr.py`와 동일한
방식(라그랑지안 동역학 -> 평형점 선형화 -> LQR 게인)으로 얻은 **닫힌 형태의
제어 게인 K**로 대체한다. 학습이나 체크포인트가 전혀 필요 없다.

구성은 run_policy_motor.py와 최대한 동일하게 유지했다:
- 모터 제어/CAN 통신: `commission_motor.py` 그대로 재사용 (MIT 모드)
- 상태 읽기: `robot_state_reader.py`의 `RobotStateReader` 그대로 재사용
- 로깅(CSV) + 종료 후 PNG 그래프: sin_position_test.py / run_policy_motor.py와 동일한 방식

바뀐 부분은 딱 하나, "정책 추론" 대신 "LQR 게인 계산"을 쓴다는 것:
    policy.infer(meas)  ->  u = -K @ x   (x = [alpha, theta, alpha_dot, theta_dot])

제어 모드
- MIT 명령은 tau_cmd = kp*(q_des - q) + kd*(dq_des - dq) + tau_ff 형태이다.
- 여기서는 kp=kd=0으로 두고 tau_ff = u(LQR로 계산한 토크)만 보낸다.
  즉 "위치를 목표로 추종"하는 게 아니라 "매 루프 계산한 토크를 그대로 명령"하는
  순수 토크(계산) 제어이다.
- 시작 직후에는 토크 0으로 대기하다가, theta(진자각)가 engage_angle_deg 이내로
  들어오는 순간 제어를 시작한다 (스윙업은 포함하지 않음 - 손으로 세운 뒤
  놓는 것을 가정). 일단 제어가 시작되면 engage_angle_deg를 살짝 벗어나도
  힘을 끊지 않고, safety_theta_deg 이내인 동안은 계속 LQR 토크를 인가해
  균형을 유지한다.
- theta가 safety_theta_deg를 넘어가면(즉 균형을 잃고 완전히 쓰러지면) 그때
  자동으로 disable한다.

사용 순서는 run_policy_motor.py와 동일하다.
1. `commission_motor.py`로 모터가 MIT 모드/원하는 ID인지 확인
2. `robot_state_reader.py` 단독 실행으로 motor_angle_deg / motor_velocity_deg_s /
   pendulum_angle_deg 가 정상적으로 들어오는지 확인
3. 이 파일 실행. 처음에는 반드시 torque_limit_nm을 작게(예: 0.05~0.1 Nm) 주고
   시작해서 거동을 본 뒤 점차 올린다.

예시 실행
```bash
uv run python src/mjlab/tasks/inverse/run_lqr_motor.py \
  --motor-id 8 \
  --channel can0 \
  --encoder-port /dev/ttyACM0 \
  --torque-limit-nm 0.10 \
  --pole-zero-deg 180.0
```
(pole_zero_deg는 인코더가 읽는 "진자가 아래로 처져 있을 때"의 각도를 0으로
 잡았다면, 똑바로 선(upright) 위치가 180도인 경우의 예시. 실제 하드웨어의
 진자 각도 기준(0도가 어디인지)에 맞춰 반드시 확인/수정할 것.)
"""

import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import linalg
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
from mjlab.tasks.inverse.robot_state_reader import RobotStateReader


# ---------------------------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunLQRMotorConfig:
  motor_id: int
  interface: str = "socketcan"
  channel: str = "can0"
  encoder_port: str | None = None

  # 실물 축 부호/오프셋 보정
  cylinder_sign: float = 1.0
  pole_sign: float = 1.0
  cylinder_zero_deg: float = 0.0
  # pole_zero_deg: 인코더 raw 값 기준으로 "진자가 똑바로 선(upright) 상태"가
  # 몇 도인지. LQR은 theta=0(upright)을 기준으로 선형화했기 때문에 반드시
  # 실제 하드웨어에 맞게 맞춰야 한다.
  pole_zero_deg: float = 180.0

  # 제어 루프 설정 (MIT 명령은 매 루프 순수 토크(tau_ff)만 보냄)
  control_hz: float = 200.0
  state_read_hz: float = 200.0
  mit_kp: float = 0.0
  mit_kd: float = 0.0
  velocity_limit_deg_s: float = 0.0
  encoder_baud: int = 115200

  # -------------------------------------------------------------------
  # 로터리 역진자 물리 파라미터 (rotary_inverted_pendulum_lqr.py의
  # Params 데이터클래스와 동일한 의미). 실제 하드웨어 값으로 바꿀 것.
  # -------------------------------------------------------------------
  arm_mass_kg: float = 0.095       # Mr (아직 실측/CAD 값 아님 - 추정치)
  arm_length_m: float = 0.025      # Lr (pivot -> 진자 pivot, 아직 추정치)
  arm_inertia_kgm2: float = 5.72e-5  # Jr (아직 추정치)
  arm_damping: float = 1.0e-3      # Br [N*m*s/rad] (아직 추정치)

  # 진자(pendulum) 쪽은 MuJoCo CAD body("BoldHolder_1")에서 계산한 값.
  # 무게중심이 길이의 절반에 있지 않으므로(비대칭 형상) Lp/2 근사 대신
  # 피벗->COM 거리(pend_com_dist_m)와 피벗 기준 관성모멘트(pend_inertia_pivot_kgm2)를
  # 직접 받는다.
  pend_mass_kg: float = 0.0318688         # Mp (XML mass)
  # 피벗->COM 거리(회전축에 수직 성분만): sqrt(y^2+z^2) of inertial pos
  # = sqrt(2.219e-8^2 + 0.0368677^2) ~= 0.036868 m.
  # 말씀하신 실측치(30mm)와 ~7mm 차이가 있으니 실제로는 이쪽을 확인해서
  # 필요하면 0.030으로 바꿔 넣을 것.
  pend_com_dist_m: float = 0.036868       # lp: 피벗 -> 무게중심 거리
  # 피벗 기준 관성모멘트: diaginertia를 quat로 회전 후 평행축 정리
  # (I_com_x ~= 4.663e-5) + m*lp^2 (~=4.332e-5) + armature(1e-5) 합산.
  # 점질량 근사(Mp*lp^2 ~= 4.33e-5)보다 약 2배 크므로 반드시 이 값을 써야 함.
  pend_inertia_pivot_kgm2: float = 9.994e-5  # Jp (점질량 근사 X, CAD 실측값)
  pend_damping: float = 1.0e-3            # Bp [N*m*s/rad] (joint damping="0.001")
  # 참고: joint frictionloss=0.0005 (쿨롱 마찰)는 선형 LQR에 반영 불가.
  # 평형점 근처에서 작은 데드존/정지마찰로 나타날 수 있음.

  gravity: float = 9.81

  # LQR 가중치: state = [alpha, theta, alpha_dot, theta_dot]
  #
  # r_torque를 크게(=토크를 "비싸게") 잡아야 하는 이유: 이 하드웨어는 관성/
  # 질량이 매우 작아서, q_theta=40/r_torque=0.12 같은 값으로 풀면 theta=20deg
  # 근방에서도 이론상 필요한 토크가 ~19Nm로 나온다 (실제 torque_limit_nm의
  # 20배 이상). 그러면 게인은 사실상 항상 포화되어 릴레이(bang-bang) 제어가
  # 되고, 스위칭 타이밍이 진자 고유 진동과 맞물려 오히려 진폭을 계속 키우는
  # 문제가 생긴다(그네를 밀어주는 것과 같은 원리) - 실제로 관측된 증상.
  # 아래 기본값은 theta=20deg(=engage_angle_deg)에서 필요 토크가
  # torque_limit_nm(기본 0.8) 이내에 들어오도록 다시 계산한 값이다.
  #
  # q_theta_dot을 낮춘 이유: theta_dot은 느린 시리얼 인코더를 유한차분한
  # 값이라 노이즈(에일리어싱)가 크다. 게인이 크면 그 노이즈가 그대로 토크로
  # 나가 진동을 오히려 키운다. theta_dot_filter_alpha로 1차 저역통과 필터를
  # 같이 적용해 완화한다.
  q_alpha: float = 2.5
  q_theta: float = 1.0
  q_alpha_dot: float = 0.05
  q_theta_dot: float = 5.0
  r_torque: float = 45.0
  # 참고: 이 시스템은 폐루프 지배극이 대략 -4(rad/s) 근방에서 게인을 꽤
  # 바꿔도 잘 변하지 않는다 (torque_limit_nm에 의해 사실상 반응속도가
  # 정해짐). 더 빠른 반응이 필요하면 q_theta/r_torque를 더 조이기보다
  # torque_limit_nm을 하드웨어 허용 범위 내에서 올리는 쪽이 맞다.

  # theta_dot 1차 저역통과(지수이동평균) 필터 계수. 0~1 사이, 작을수록 더
  # 강하게 필터링(더 느리게 반응). theta_dot_filtered = a*raw + (1-a)*prev.
  theta_dot_filter_alpha: float = 0.2

  # 이 입력은 "토크(Nm) 직접 명령"을 기준으로 LQR을 설계하므로 B는
  # 전압/역기전력 모델이 아니라 tau_ff 입력에 대한 것이다.

  # 안전 설정
  torque_limit_nm: float = 0.10     # LQR 출력 토크 clip
  engage_angle_deg: float = 20.0    # 대기 상태에서 이 각도 이내로 들어와야 제어 시작(진입 문턱값, 진입 후엔 safety_theta_deg까지 계속 제어)
  safety_theta_deg: float = 60.0    # 이 각도를 넘으면 자동 disable

  # 스윙업이 없으므로, 시작 직후 진자가 아래로 처져 있는 것은 정상이다.
  # 제어(및 safety_theta_deg 감시)를 시작하기 전에, 사람이 손으로 진자를
  # engage_angle_deg 이내로 세울 때까지 토크 0으로 대기한다.
  wait_for_upright: bool = True
  wait_timeout_s: float = 60.0
  wait_print_interval_s: float = 1.0

  max_runtime_s: float = 30.0
  start_with_zero_set: bool = False
  require_enable_ack: bool = True

  # 로깅/플롯 설정
  log_dir: str = "logs"
  plot: bool = True


# ---------------------------------------------------------------------------
# 1. 선형화 + LQR 게인 계산
#    (rotary_inverted_pendulum_lqr.py의 linearize()/lqr()과 동일한 유도이나,
#     입력을 "모터 전압"이 아니라 "모터 토크(tau_ff) 직접 명령"으로 바꿨다.
#     CAN 모터가 전류/토크 제어 루프를 내부에서 이미 닫고 있으므로, 역기전력
#     항(kt*km/Rm) 없이 우리가 명령한 토크가 그대로 인가된다고 가정한다.)
# ---------------------------------------------------------------------------
def linearize_torque_input(cfg: RunLQRMotorConfig):
  """State x = [alpha, theta, alpha_dot, theta_dot], input u = arm torque [Nm].

  진자 무게중심이 Lp/2에 있지 않은(비대칭) 형상이므로, lp(피벗->COM 거리)와
  Jp(피벗 기준 관성모멘트)는 점질량 근사(Lp/2, Mp*lp^2)로 유도하지 않고
  cfg.pend_com_dist_m / cfg.pend_inertia_pivot_kgm2 값을 그대로 사용한다
  (CAD/실측에서 직접 얻은 값).
  """
  lp = cfg.pend_com_dist_m
  Mp, Lr, Jr = cfg.pend_mass_kg, cfg.arm_length_m, cfg.arm_inertia_kgm2

  J0 = Jr + Mp * Lr**2       # 유효 arm 관성
  Jp = cfg.pend_inertia_pivot_kgm2   # 진자 pivot 기준 관성 (CAD 실측, 점질량 근사 아님)

  M = np.array([[J0, -Mp * Lr * lp],
                [-Mp * Lr * lp, Jp]])
  Minv = np.linalg.inv(M)

  C_damp = np.array([[cfg.arm_damping, 0.0],
                      [0.0, cfg.pend_damping]])
  G = np.array([[0.0, 0.0],
                [0.0, Mp * cfg.gravity * lp]])   # theta=0(upright)에서 destabilizing
  Bv = np.array([1.0, 0.0])                       # 입력 = 토크(직접)

  lower_left = Minv @ G
  lower_right = -Minv @ C_damp
  B_lower = Minv @ Bv

  A = np.zeros((4, 4))
  A[0, 2] = 1.0
  A[1, 3] = 1.0
  A[2:4, 0:2] = lower_left
  A[2:4, 2:4] = lower_right

  B = np.zeros((4, 1))
  B[2:4, 0] = B_lower

  return A, B


def compute_lqr_gain(cfg: RunLQRMotorConfig):
  A, B = linearize_torque_input(cfg)
  Q = np.diag([cfg.q_alpha, cfg.q_theta, cfg.q_alpha_dot, cfg.q_theta_dot])
  R = np.array([[cfg.r_torque]])
  P = linalg.solve_continuous_are(A, B, Q, R)
  K = np.linalg.inv(R) @ B.T @ P   # shape (1,4)

  eig_ol = np.linalg.eigvals(A)
  eig_cl = np.linalg.eigvals(A - B @ K)
  print("Linearized A =\n", A)
  print("Linearized B =\n", B)
  print("Open-loop eigenvalues:", eig_ol)
  print("LQR gain K =", K)
  print("Closed-loop eigenvalues:", eig_cl)
  return K


def wrap_deg_180(angle_deg: float) -> float:
  """[-180, 180) 범위로 wrap."""
  return (angle_deg + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# 2. MIT 명령 전송 (run_policy_motor.py와 동일)
# ---------------------------------------------------------------------------
def _send_mit_command_and_get_reply(
  bus,
  motor_id: int,
  *,
  position_deg: float,
  velocity_deg_s: float,
  torque_nm: float,
  kp: float,
  kd: float,
  pmax: float = 12.57,
  vmax: float = 33.0,
  tmax: float = 17.0,
  timeout_s: float = 0.03,
):
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


def _read_pendulum_state(
  cfg: RunLQRMotorConfig,
  reader: RobotStateReader,
  prev_encoder_state,
  theta_dot_filtered_prev: float = 0.0,
):
  """공통 상태 읽기.

  Returns: (alpha_deg, theta_deg, alpha_dot_deg_s, theta_dot_deg_s_filtered,
            new_prev_encoder_state, new_theta_dot_filtered)

  theta_dot는 느린 시리얼 인코더를 유한차분해서 얻기 때문에 노이즈(에일리어싱)가
  크다. 그대로 쓰면 D게인(q_theta_dot)을 타고 노이즈가 토크로 증폭되어 진동을
  키우는 문제가 생겨서, 1차 저역통과(지수이동평균) 필터를 거친 값을 반환한다.
  """
  state = reader.get_state()
  if (
    state.motor_angle_deg is None
    or state.motor_velocity_deg_s is None
    or state.pendulum_angle_deg is None
  ):
    raise RuntimeError("Motor/pendulum state is not available from RobotStateReader")

  alpha_deg = (state.motor_angle_deg - cfg.cylinder_zero_deg) * cfg.cylinder_sign
  alpha_dot_deg_s = state.motor_velocity_deg_s * cfg.cylinder_sign
  theta_deg = wrap_deg_180((state.pendulum_angle_deg - cfg.pole_zero_deg) * cfg.pole_sign)

  theta_dot_raw_deg_s = 0.0
  encoder_state = (
    reader.encoder_reader.latest if reader.encoder_reader is not None else None
  )
  if encoder_state is not None and prev_encoder_state is not None:
    enc_dt = max(encoder_state.timestamp - prev_encoder_state.timestamp, 1e-4)
    raw_delta = encoder_state.angle_deg - prev_encoder_state.angle_deg
    delta_deg = (raw_delta + 180) % 360 - 180
    theta_dot_raw_deg_s = (delta_deg / enc_dt) * cfg.pole_sign
  prev_encoder_state = encoder_state

  a = cfg.theta_dot_filter_alpha
  theta_dot_filtered = a * theta_dot_raw_deg_s + (1.0 - a) * theta_dot_filtered_prev

  return alpha_deg, theta_deg, alpha_dot_deg_s, theta_dot_filtered, prev_encoder_state, theta_dot_filtered


def _wait_until_upright(
  cfg: RunLQRMotorConfig, bus, reader: RobotStateReader, prev_encoder_state
):
  """토크 0을 유지하며(안전) 사람이 손으로 진자를 세울 때까지 대기.

  스윙업 로직이 없으므로, 시작 직후 진자가 아래로 처져 있는 상태
  (theta ~= 180deg)는 정상이다. 이 대기 단계에서는 safety_theta_deg 감시를
  하지 않고, engage_angle_deg 이내로 들어올 때까지 계속 토크 0 명령만
  유지한다.

  Returns: (prev_encoder_state, alpha_ref_deg). alpha_ref_deg는 engage 시점의
  arm 절대각으로, 이후 메인 루프에서 이 값을 빼서 "제어 시작 시점을 0"으로 하는
  상대각을 LQR 상태에 넣는다. 이렇게 하지 않으면 모터가 우연히 서 있던 절대각
  (예: 360도 근처)이 라디안으로 커서 alpha 항 하나만으로 토크가 즉시 포화되고,
  그 힘이 팔을 계속 한쪽으로 밀어붙여 진자를 끌고 다니다 쓰러뜨리는 문제가
  생긴다 (실제로 보고된 증상).
  """
  period_s = 1.0 / float(cfg.control_hz)
  deadline = time.monotonic() + float(cfg.wait_timeout_s)
  last_print = 0.0

  print(
    f"진자를 손으로 세워주세요 (|theta| <= {cfg.engage_angle_deg} deg 이내가 되면 "
    f"자동으로 LQR 제어를 시작합니다. 최대 {cfg.wait_timeout_s:.0f}초 대기)"
  )

  while time.monotonic() < deadline:
    loop_start = time.monotonic()

    alpha_deg, theta_deg, _, _, prev_encoder_state, _ = _read_pendulum_state(
      cfg, reader, prev_encoder_state
    )

    reply = _send_mit_command_and_get_reply(
      bus,
      cfg.motor_id,
      position_deg=alpha_deg,
      velocity_deg_s=0.0,
      torque_nm=0.0,
      kp=0.0,
      kd=0.0,
    )
    if reply is None:
      raise RuntimeError("No MIT reply received from motor (standby)")

    if abs(theta_deg) <= cfg.engage_angle_deg:
      print(f"\ntheta={theta_deg:+.1f}deg -> engage 범위 진입, 제어 시작")
      # 이 시점의 alpha_deg를 arm 기준점으로 삼는다 (아래 alpha_ref_deg 설명 참고).
      return prev_encoder_state, alpha_deg

    now = time.monotonic()
    if now - last_print >= cfg.wait_print_interval_s:
      print(f"대기 중... theta={theta_deg:+7.2f} deg", end="\r")
      last_print = now

    elapsed = time.monotonic() - loop_start
    sleep_s = period_s - elapsed
    if sleep_s > 0.0:
      time.sleep(sleep_s)

  raise RuntimeError(
    f"wait_timeout_s({cfg.wait_timeout_s}s) 동안 진자가 engage_angle_deg 이내로 "
    "들어오지 않았습니다. 손으로 세운 뒤 다시 시도하세요."
  )


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
  "alpha_rel_deg",
  "theta_deg",
  "alpha_dot_deg_s",
  "theta_dot_deg_s",
  "control_torque_nm",
  "lqr_engaged",
]


def _plot_log(csv_path: str) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib이 설치되어 있지 않아 그래프를 건너뜁니다. (pip install matplotlib)")
    return

  t_list, alpha_list, theta_list = [], [], []
  alpha_dot_list, theta_dot_list, tau_list, engaged_list = [], [], [], []
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      t_list.append(float(row["time_s"]))
      alpha_list.append(float(row["alpha_rel_deg"]))
      theta_list.append(float(row["theta_deg"]))
      alpha_dot_list.append(float(row["alpha_dot_deg_s"]))
      theta_dot_list.append(float(row["theta_dot_deg_s"]))
      tau_list.append(float(row["control_torque_nm"]))
      engaged_list.append(int(row["lqr_engaged"]))

  if not t_list:
    print("로그가 비어 있어 그래프를 건너뜁니다.")
    return

  fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

  axes[0].plot(t_list, alpha_list, label="alpha (arm) [deg]")
  axes[0].plot(t_list, theta_list, label="theta (pendulum) [deg]")
  axes[0].axhline(0, color="k", lw=0.5)
  axes[0].set_ylabel("Angle [deg]")
  axes[0].legend()
  axes[0].grid(True)

  axes[1].plot(t_list, alpha_dot_list, label="alpha_dot [deg/s]")
  axes[1].plot(t_list, theta_dot_list, label="theta_dot [deg/s]")
  axes[1].set_ylabel("Angular vel. [deg/s]")
  axes[1].legend()
  axes[1].grid(True)

  axes[2].plot(t_list, tau_list, color="tab:red", label="control torque [Nm]")
  engaged_t = [t for t, e in zip(t_list, engaged_list, strict=True) if not e]
  for t in engaged_t:
    axes[2].axvspan(t, t, color="grey", alpha=0.02)
  axes[2].set_ylabel("Torque [Nm]")
  axes[2].set_xlabel("time (s)")
  axes[2].legend()
  axes[2].grid(True)

  fig.tight_layout()
  png_path = os.path.splitext(csv_path)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  print(f"그래프 저장 완료: {png_path}")


# ---------------------------------------------------------------------------
# 3. 메인 제어 루프
# ---------------------------------------------------------------------------
def run(cfg: RunLQRMotorConfig) -> None:
  K = compute_lqr_gain(cfg)  # shape (1, 4), 한 번만 계산 (학습 없음)

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
  csv_path = os.path.join(cfg.log_dir, f"lqr_run_{ts}.csv")

  csv_file = open(csv_path, "w", newline="")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(_CSV_HEADER)

  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  print(f"CSV 로그: {csv_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(max_runtime_s 도달)"
  prev_encoder_state = None

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

    if cfg.wait_for_upright:
      prev_encoder_state, alpha_ref_deg = _wait_until_upright(
        cfg, bus, reader, prev_encoder_state
      )
    else:
      # 대기 단계를 건너뛰는 경우에도, 지금 이 순간의 alpha를 기준점으로 잡는다.
      alpha_ref_deg, _, _, _, prev_encoder_state, _ = _read_pendulum_state(
        cfg, reader, prev_encoder_state
      )

    print(f"arm 기준각(alpha_ref_deg) = {alpha_ref_deg:.2f} deg (이 값 기준 상대각으로 제어)")

    period_s = 1.0 / float(cfg.control_hz)
    deadline = time.monotonic() + float(cfg.max_runtime_s)

    print(
      f"engage_angle_deg={cfg.engage_angle_deg}, "
      f"safety_theta_deg={cfg.safety_theta_deg}, "
      f"torque_limit_nm={cfg.torque_limit_nm}, "
      f"theta_dot_filter_alpha={cfg.theta_dot_filter_alpha}"
    )
    print("Starting computed LQR control loop (no learning, no checkpoint).")

    theta_dot_filtered = 0.0

    while time.monotonic() < deadline:
      loop_start = time.monotonic()

      (
        alpha_deg,
        theta_deg,
        alpha_dot_deg_s,
        theta_dot_deg_s,
        prev_encoder_state,
        theta_dot_filtered,
      ) = _read_pendulum_state(cfg, reader, prev_encoder_state, theta_dot_filtered)
      # alpha_rel_deg: 제어 시작 시점(alpha_ref_deg)을 0으로 하는 상대각.
      # LQR은 alpha=0을 "팔이 있어야 할 위치"로 보고 설계됐으므로, 절대 인코더
      # 값(예: 360도 근처)을 그대로 넣으면 그 값 자체가 커서 토크가 즉시
      # 포화된다. 반드시 상대각을 써야 한다.
      alpha_rel_deg = alpha_deg - alpha_ref_deg

      x = np.array([
        math.radians(alpha_rel_deg),
        math.radians(theta_deg),
        math.radians(alpha_dot_deg_s),
        math.radians(theta_dot_deg_s),
      ])

      # engage_angle_deg는 대기 단계(_wait_until_upright)에서 "제어를 시작할지"만
      # 판단하는 문턱값이다. 일단 여기(메인 루프)에 들어온 뒤에는 safety_theta_deg
      # 이내인 한 계속 LQR 토크를 인가해야 균형을 잡을 수 있다. engage_angle_deg로
      # 매 루프 토크를 껐다 켰다 하면, 살짝만 흔들려도 그 순간 힘이 빠져서
      # safety_theta_deg까지 쓰러진 뒤 종료되어 버린다 (실제로 보고된 증상).
      engaged = abs(theta_deg) < cfg.safety_theta_deg
      if not engaged:
        print(f"\ntheta={theta_deg:.1f}deg > safety_theta_deg -> 자동 정지")
        stopped_reason = "안전 각도 초과로 자동 정지"
        break

      u = float(-(K @ x)[0])
      u = float(np.clip(u, -cfg.torque_limit_nm, cfg.torque_limit_nm))

      reply = _send_mit_command_and_get_reply(
        bus,
        cfg.motor_id,
        position_deg=alpha_deg,
        velocity_deg_s=0.0,
        torque_nm=u,
        kp=cfg.mit_kp,
        kd=cfg.mit_kd,
      )
      if reply is None:
        raise RuntimeError("No MIT reply received from motor")

      elapsed_t = time.monotonic() - start_t
      csv_writer.writerow([
        f"{elapsed_t:.4f}",
        f"{alpha_rel_deg:.3f}",
        f"{theta_deg:.3f}",
        f"{alpha_dot_deg_s:.3f}",
        f"{theta_dot_deg_s:.3f}",
        f"{u:.4f}",
        f"{int(engaged)}",
      ])

      print(
        "alpha_rel={:+7.2f} deg  theta={:+7.2f} deg  tau={:+6.3f} Nm  engaged={}".format(
          alpha_rel_deg, theta_deg, u, "Y" if engaged else "."
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
  cfg = tyro.cli(RunLQRMotorConfig)
  run(cfg)


if __name__ == "__main__":
  main()