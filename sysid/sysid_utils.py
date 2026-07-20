#!/usr/bin/env python3
from __future__ import annotations

"""Chirp 토크 기반 액추에이터 System Identification 통합 파이프라인.

`sysid_collect.py`(여기 신호 수집)와 `sysid_fit.py`(파라미터 피팅)를 하나로
합치고, 위치/스텝/멀티사인 모드를 걷어내어 "torque-chirp 전용"으로 단순화한
스크립트다. 실행 순서는 사용자가 정리한 3단계와 그대로 대응한다.

  1. collect  : 실제 액추에이터에 chirp(주파수 스윕) 토크를 open-loop로
                가하고 응답(각도/속도/토크)을 CSV+PNG로 로깅한다.
                (`--simulate`를 주면 하드웨어 없이 MuJoCo로 같은 실험을
                모사한다 — 파이프라인 점검이나 사전 튜닝용.)
  2. fit      : 1에서 얻은 로그로 관절의 armature / damping(viscous) /
                frictionloss(Coulomb)를 추정한다. `mujoco-sysid` 저장소의
                데모(Levenberg–Marquardt nonlinear least squares, 즉
                Gauss-Newton 근사 Hessian J^T J를 쓰는 `mujoco.minimize.
                least_squares`)와 동일한 방식이다 — 회귀식을 직접 세우는
                대신, 후보 파라미터로 MuJoCo를 재생(replay)해서 실측
                궤적과의 오차를 최소화한다.
  3. compare  : 2에서 얻은 파라미터를 XML에 반영한 뒤, 1과 "완전히 같은"
                cmd_torque_nm 시퀀스를 다시 MuJoCo로 흘려서 실측 궤적과
                겹쳐 그리고 RMSE로 비교한다.

`--stage full`(기본값)은 1→2→3을 한 번에 실행한다. 각 단계는 `--stage
collect|fit|compare`로 개별 실행도 가능하다(예: 실물에서 collect만 먼저
해두고, fit/compare는 나중에 별도로 반복).

실물 하드웨어 연동(CAN/시리얼)은 기존 프로젝트의
`mjlab.tasks.inverse.commission_motor` / `robot_state_reader`를 그대로
재사용한다 — 해당 모듈이 없는 환경(예: 시뮬레이션 전용 점검)에서도
`--simulate`만 쓴다면 import 실패가 나지 않도록 지연 import로 처리했다.

예시
```bash
# 하드웨어 없이 전체 파이프라인 점검 (synthetic ground-truth로 fit 검증)
uv run python sysid_chirp_pipeline.py --stage full --simulate \
  --torque-amp-nm 0.45 --chirp-f0-hz 0.2 --chirp-f1-hz 15 --chirp-duration-s 15

# 실물 collect만
uv run python sysid_chirp_pipeline.py --stage collect \
  --motor-id 8 --channel can0 \
  --torque-amp-nm 0.45 --chirp-f1-hz 15

# 이미 모은 로그로 fit + compare만
uv run python sysid_chirp_pipeline.py --stage fit --csv-path logs/sysid_chirp_....csv
uv run python sysid_chirp_pipeline.py --stage compare --csv-path logs/sysid_chirp_....csv \
  --fitted-params-json logs/sysid_report.json

# 1단계: damping-test(스핀업+자유감속 반복)로 damping을 먼저 깨끗하게 분리
uv run python sysid_chirp_pipeline.py --stage full --no-simulate --damping-test \
  --motor-id 8 --channel can0

# 2단계: 같은 --output-report를 그대로 두고 chirp 실행 — 위에서 구한 damping을
# 자동으로 읽어와 고정하고, armature/frictionloss만 다시 fit한다
uv run python sysid_chirp_pipeline.py --stage full --no-simulate \
  --motor-id 8 --channel can0
```
"""

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple

import mujoco
import mujoco.minimize  # type: ignore[import-not-found]
import numpy as np
import tyro
from prettytable import PrettyTable

_DEFAULT_XML = Path(__file__).parent / "assets" / "inverse.xml"


# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChirpSysidConfig:
  stage: Literal["collect", "fit", "compare", "full"] = "full"

  # 대상 관절/액추에이터 (XML에 맞게 조정).
  joint_name: str = "Revolute 3"
  actuator_name: str = "position_revolute_3"
  xml_path: str = str(_DEFAULT_XML)

  # --- 여기 신호 종류 ---
  # False(기본): chirp(주파수 스윕 토크). armature 식별에는 좋지만(다양한
  #   가속도), damping(속도 비례)과 frictionloss(Coulomb, 속도부호만 의존)가
  #   서로 트레이드오프되며 실험마다 크게 흔들리는 게 실측으로 확인됐다.
  # True: spin_coast(스핀업+자유감속 반복). spin_duration_s 동안 일정
  #   토크로 스핀업했다가 coast_duration_s 동안 토크를 0으로 끊어
  #   자유감속시키는 걸 n_cycles번 반복한다(매 cycle마다 방향을 번갈아
  #   드리프트를 상쇄하고 frictionloss의 양쪽 부호를 모두 여기한다).
  #   토크=0 구간은 damping/frictionloss만 깔끔하게 분리해서 알려준다 —
  #   damping 전용 실험으로 쓴다. `--damping-test`로 이 값을 구한 뒤,
  #   다음 chirp 실행(damping_test=False)은 output_report에서 그 damping을
  #   자동으로 읽어와 고정하고 armature/frictionloss만 다시 fit한다(아래
  #   fit_gauss_newton의 자동 계층적 식별 참고).
  damping_test: bool = False

  # --- Chirp(주파수 스윕) 토크 여기 신호 ---
  # 실측(2026-07-15)으로 확인된 두 가지 문제 때문에 기본값을 보수적으로
  # 잡는다.
  # 1) 순수 사인 chirp은 저주파 구간에서 반주기가 길어(f0=0.2Hz면 약 2.5초)
  #    그 동안 토크 부호가 안 바뀌어 속도/위치가 한쪽으로 크게 누적된다.
  #    (원본 sysid_collect.py가 멀티사인에서 저주파 성분에만 cos을 쓴 것도
  #    이 문제를 피하기 위해서였다 — cos(wt)의 적분은 유계 진동하지만
  #    sin(wt)의 적분은 단조 누적된다.) f0를 너무 낮게 잡지 않는 게 안전하다.
  # 2) 실측 motor_torque_nm이 cmd_torque_nm=1.0Nm 근처에서 saturate되는
  #    것이 관찰됐다(약 ±0.5Nm에서 눌림) — 실제 모터 토크 한계보다 큰 값을
  #    명령하면 예측 못한 비대칭 응답/드리프트로 이어질 수 있으므로, 그
  #    한계보다 확실히 낮은 값으로 시작해서 점진적으로 올리는 걸 권장한다.
  torque_amp_nm: float = 0.4
  chirp_f0_hz: float = 1.0
  chirp_f1_hz: float = 15.0
  chirp_duration_s: float = 30.0
  chirp_scale: Literal["log", "linear"] = "log"
  ramp_in_s: float = 0.5
  control_hz: float = 200.0

  # --- spin_coast 여기 신호 ---
  spin_torque_nm: float = 0.4
  spin_duration_s: float = 3.0
  coast_duration_s: float = 4.0
  n_cycles: int = 6
  spin_ramp_in_s: float = 0.2

  # --- 하드웨어 collect (실물) ---
  simulate: bool = True  # True: MuJoCo로 모사, False: 실제 CAN/시리얼 사용
  motor_id: int = 8
  interface: str = "socketcan"
  channel: str = "can0"
  encoder_port: str | None = None
  encoder_baud: int = 115200
  state_read_hz: float = 200.0
  require_enable_ack: bool = True

  # 안전 한계 (실물 collect 전용, torque open-loop이므로 위치 복원력이 없다).
  # drift 제한은 기본으로 끔(inf) — spin_coast/chirp 모두 정상적으로 수십~
  # 수천 도씩 회전하는 게 흔해서 180deg 같은 낮은 값은 오히려 정상 실험을
  # 자꾸 중단시켰다. 정말 폭주를 막고 싶으면 CLI에서 직접 값을 주면 된다.
  torque_drift_limit_deg: float = float("inf")
  torque_velocity_limit_deg_s: float = 2000.0
  abort_on_torque_nm: float = 5.0
  max_runtime_s: float = 60.0

  # --- fit (Gauss-Newton / Levenberg-Marquardt via mujoco.minimize) ---
  csv_path: str = ""  # 비워두면 collect가 방금 만든 CSV를 그대로 사용
  damping_max: float = 5.0  # viscous(=dof_damping) 상한
  armature_max: float = 0.05
  frictionloss_max: float = 0.5  # Coulomb(=dof_frictionloss) 상한
  # Pure open-loop torque replay는 복원력이 없어 파라미터 오차가 자유적분으로
  #누적된다 — sysid_fit.py의 fit_torque와 동일한 이유로 짧은 재앵커링
  # 윈도우가 필요하다 (0.2s가 경험적으로 잘 맞았음).
  shooting_window_s: float = 0.2
  # 샘플 간격이 median의 이 배수를 넘는 구간(CAN/Python 루프 지연으로 생긴
  # 결측에 가까운 튐)이 포함된 shooting window는 통째로 fit/RMSE 계산에서
  # 제외한다 — 그런 outlier 하나가 최소자승 비용을 지배해서 armature/
  # frictionloss가 경계값(0)으로 밀려버리는 게 실측으로 확인됐다.
  max_dt_multiple: float = 3.0
  # 실측 motor_torque_nm에 전류센싱/리플 노이즈가 커서(실측: 속도는 매끈한
  # 등가속 삼각파인데 토크만 ±0.2~0.3Nm씩 떠는 게 확인됨 — 진짜 힘이
  # 그렇게 떨렸다면 가속도도 같이 떨려야 하므로 센서 노이즈로 판단) 이
  # 값을 그대로 qfrc_applied에 넣으면 fit이 그 노이즈를 설명하려다
  # armature를 비정상적으로 작은 값으로 밀어버린다. 이동평균으로 스무딩한
  # 뒤 사용한다. 1이면 스무딩 없음(원본 그대로).
  torque_smooth_samples: int = 9
  # fit/compare에서 물리 입력(qfrc_applied)으로 어느 채널을 쓸지. "measured"
  # (기본): 실측 motor_torque_nm(스무딩 적용). "command": cmd_torque_nm(명령값,
  # 노이즈 없음) — armature=0으로 붕괴하거나 fit-실측 괴리가 큰 경우, 실측
  # 토크 채널 자체(센서 스케일/지연/back-EMF 제한 등)에 문제가 있는지
  # 진단하려고 넣은 비교 실험용 스위치다.
  torque_input: Literal["measured", "command"] = "measured"

  # 다른 실험(예: spin_coast의 coast 구간)에서 이미 잘 분리해 얻은 파라미터를
  # 이 fit에서 고정하고 싶을 때 쓴다 — 예를 들어 damping을 coast-down에서
  # 구한 값으로 고정하고, 이 chirp 로그로는 armature(+frictionloss)만
  # 다시 맞추는 계층적(hierarchical) 식별에 쓴다. None이면 그 파라미터도
  # 평소처럼 자유롭게 fit한다.
  fix_damping: float | None = None
  fix_armature: float | None = None
  fix_frictionloss: float | None = None

  # --- compare (fit된 파라미터로 재시뮬레이션 후 실측과 비교) ---
  fitted_params_json: str = "logs/sysid_report.json"

  # --- 출력 ---
  log_dir: str = "logs"
  output_report: str = "logs/sysid_report.json"
  plot: bool = True


_CSV_HEADER = [
  "time_s",
  "cmd_torque_nm",
  "motor_angle_deg",
  "motor_vel_deg_s",
  "motor_torque_nm",
]


# --------------------------------------------------------------------------
# Chirp 신호 생성
# --------------------------------------------------------------------------


def build_chirp_torque(cfg: ChirpSysidConfig, sample_times: np.ndarray) -> np.ndarray:
  """`sample_times`(초, 0부터 시작)에 대응하는 chirp 토크(Nm) 시퀀스를 만든다.

  위치 chirp(build_chirp_targets in sysid_collect.py)와 같은 방식으로 순간
  주파수를 f0->f1로 스윕하되, 목표각이 아니라 토크 진폭에 곱한다 — 순수
  토크 open-loop 여기이므로 위치 PD를 전혀 거치지 않는다(질문에서 정리한
  대로, 위치 제어는 컨트롤러 게인이 섞여 순수 플랜트 파라미터 식별을
  방해하기 때문).
  """
  t = np.clip(sample_times, 0.0, cfg.chirp_duration_s)
  f0, f1, T = cfg.chirp_f0_hz, cfg.chirp_f1_hz, cfg.chirp_duration_s
  if cfg.chirp_scale == "log":
    k = (f1 / f0) ** (1.0 / T)
    phase = 2.0 * math.pi * f0 * (np.power(k, t) - 1.0) / math.log(k)
  else:
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) / T * t**2)
  ramp = np.clip(sample_times / cfg.ramp_in_s, 0.0, 1.0) if cfg.ramp_in_s > 0 else 1.0
  return cfg.torque_amp_nm * ramp * np.sin(phase)


def build_spin_coast_torque(cfg: ChirpSysidConfig, sample_times: np.ndarray) -> np.ndarray:
  """spin(일정 토크로 스핀업) -> coast(토크 0, 자유감속) 사이클을 n_cycles번
  반복하는 토크(Nm) 시퀀스를 만든다. 매 cycle마다 방향을 번갈아(+/-) 걸어서
  누적 드리프트를 상쇄하고, frictionloss(Coulomb)의 양쪽 부호를 모두
  여기한다.

  coast 구간(토크=0)에서는 0 = I_eff*qddot + damping*qdot +
  frictionloss*sign(qdot)만 남는 순수 감쇠 운동이라, 인가 토크의 대역폭/
  saturation 불확실성이 fit에 전혀 섞이지 않는다. spin 구간은 armature
  (관성)를 절대 스케일로 식별하는 데 필요한 토크-가속도 관계를 제공한다.
  """
  cycle_s = cfg.spin_duration_s + cfg.coast_duration_s
  t_in_cycle = np.mod(sample_times, cycle_s)
  cycle_idx = np.floor(sample_times / cycle_s).astype(int)
  sign = np.where(cycle_idx % 2 == 0, 1.0, -1.0)

  is_spin = t_in_cycle < cfg.spin_duration_s
  ramp = (
    np.clip(t_in_cycle / cfg.spin_ramp_in_s, 0.0, 1.0)
    if cfg.spin_ramp_in_s > 0
    else 1.0
  )
  torque = np.where(is_spin, sign * cfg.spin_torque_nm * ramp, 0.0)
  return torque


def excitation_duration_s(cfg: ChirpSysidConfig) -> float:
  if cfg.damping_test:
    return cfg.n_cycles * (cfg.spin_duration_s + cfg.coast_duration_s)
  return cfg.chirp_duration_s


def build_excitation_torque(cfg: ChirpSysidConfig, sample_times: np.ndarray) -> np.ndarray:
  if cfg.damping_test:
    return build_spin_coast_torque(cfg, sample_times)
  return build_chirp_torque(cfg, sample_times)


# --------------------------------------------------------------------------
# 로깅 데이터 구조 / 입출력
# --------------------------------------------------------------------------


class LoggedTrajectory(NamedTuple):
  time_s: np.ndarray
  cmd_torque_nm: np.ndarray
  angle_rad: np.ndarray
  vel_rad_s: np.ndarray
  torque_nm: np.ndarray


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
  """중심 이동평균. window<=1이면 그대로 반환. 가장자리는 짧아진 커널로
  정규화해서(=경계에서 0으로 패딩된 것처럼 값이 줄어드는 걸 방지) 편향 없이
  스무딩한다."""
  if window <= 1:
    return x
  kernel = np.ones(window)
  num = np.convolve(x, kernel, mode="same")
  den = np.convolve(np.ones_like(x), kernel, mode="same")
  return num / den


# 실측 motor_angle_deg는 CAN 모터(MIT 프로토콜)가 보고하는 값이라 완전한
# 멀티턴이 아니라 [-4π, 4π] rad(=[-720, 720]deg) 범위로 wrap된다 — 이 범위를
# 넘어가면 다시 반대쪽 경계로 튀어서 로그에 삽니다처럼 톱니 패턴이 찍힌다
# (실측 그래프로 확인됨). np.unwrap(..., period=...)로 이 wrap을 되돌린다.
# 이 period(720deg)의 절반인 360deg가 "샘플 간 실제 변화가 이보다 크면
# wrap으로 간주" 하는 임계값인데, 지금 최고 속도(~1300deg/s)와 150ms
# 샘플링을 곱해도 한 샘플당 최대 변화는 195deg로 그 임계값보다 한참
# 작다 — 그래서 진짜 빠른 움직임을 wrap으로 착각해 오작동할 위험 없이,
# 실제 720deg 경계에서만 정확히 되돌려진다(기본 np.unwrap의 period=2π,
# 임계값 180deg였을 때는 195deg 움직임이 그 임계값을 넘어버려 정상적인
# 빠른 회전을 wrap으로 오인했었다 — 이게 바로 이전에 본 "속도는 음수인데
# 각도가 갑자기 확 뛰는" 버그의 진짜 원인이었다). 시뮬레이션 로그(MuJoCo
# qpos, 원래 랩어라운드 없음)에 적용해도 동일한 이유로 안전하다 — 오작동
# 임계값(360deg)보다 훨씬 작은 폭으로만 움직이기 때문이다.
_ANGLE_WRAP_PERIOD_RAD = 4 * math.pi


def _unwrap_with_velocity(
  angle_rad: np.ndarray, vel_rad_s: np.ndarray, time_s: np.ndarray, period_rad: float
) -> np.ndarray:
  """속도 채널로 보정한 unwrap.

  일반 `np.unwrap(period=...)`는 "샘플 간 위치 차이가 period/2를 넘으면
  wrap"이라는 규칙만으로 판단한다 — 노이즈가 그 경계(여기서는 360deg) 근처에
  걸리는 샘플이 하나라도 있으면 오판할 수 있고, 한 번 오판하면 그 이후
  전체가 ±720deg씩 밀린다(실측으로 "첫 구간에서 unwrap이 틀어진다"는 형태로
  확인됨). 우리는 wrap과 무관한 독립 속도 채널(motor_vel_deg_s)을 갖고
  있으므로, 각 스텝마다 "속도로 기대되는 변화량(vel*dt)에 가장 가까워지는
  wrap 배수"를 직접 골라서 위치 차이만 볼 때보다 훨씬 견고하게 판단한다.
  """
  angle = angle_rad.copy()
  for k in range(1, len(angle)):
    dt = float(time_s[k] - time_s[k - 1])
    expected_diff = 0.5 * (vel_rad_s[k] + vel_rad_s[k - 1]) * dt
    raw_diff = angle[k] - angle[k - 1]
    n_wraps = round((expected_diff - raw_diff) / period_rad)
    if n_wraps != 0:
      # 진단용: 어느 시점에서 몇 배수 보정이 들어갔는지 남긴다 — 남은
      # 이상 구간이 실제로 여기서 나온 오판인지 다른 원인인지 이 로그로
      # 바로 알 수 있다.
      residual_deg = math.degrees(raw_diff + n_wraps * period_rad - expected_diff)
      print(
        f"[unwrap] t={time_s[k]:.3f}s(idx {k}): {n_wraps:+d}회 wrap 보정 "
        f"(raw_diff={math.degrees(raw_diff):.1f}deg, "
        f"expected(v*dt)={math.degrees(expected_diff):.1f}deg, "
        f"보정 후 잔차={residual_deg:.1f}deg)"
      )
    angle[k] = angle[k - 1] + raw_diff + n_wraps * period_rad
  return angle


def load_csv(
  csv_path: str, torque_smooth_samples: int = 1, angle_wrap_period_rad: float = _ANGLE_WRAP_PERIOD_RAD
) -> LoggedTrajectory:
  data = np.genfromtxt(csv_path, delimiter=",", names=True)
  dt = np.diff(data["time_s"])
  if dt.size and (np.max(dt) - np.min(dt)) > 0.2 * np.median(dt):
    print(
      f"[warn] sample spacing is not uniform (min={dt.min():.4f}s, "
      f"max={dt.max():.4f}s) — replay/comparison may be biased."
    )
  torque_nm = data["motor_torque_nm"]
  if torque_smooth_samples > 1:
    torque_nm = _smooth(torque_nm, torque_smooth_samples)
  angle_rad = np.radians(data["motor_angle_deg"])
  vel_rad_s = np.radians(data["motor_vel_deg_s"])
  if angle_wrap_period_rad:
    angle_rad = _unwrap_with_velocity(angle_rad, vel_rad_s, data["time_s"], angle_wrap_period_rad)
  return LoggedTrajectory(
    time_s=data["time_s"],
    cmd_torque_nm=data["cmd_torque_nm"],
    angle_rad=angle_rad,
    vel_rad_s=vel_rad_s,
    torque_nm=torque_nm,
  )


def _physics_input_torque(traj: LoggedTrajectory, cfg: ChirpSysidConfig) -> np.ndarray:
  """fit/compare가 qfrc_applied로 쓸 토크 채널을 cfg.torque_input에 따라
  고른다. "command"면 cmd_torque_nm(노이즈 없는 명령값)을 그대로 쓴다 —
  실측 torque_nm 채널(센서 노이즈/스케일/지연 등) 자체가 fit-실측 괴리의
  원인인지 진단하기 위한 A/B 비교용."""
  if cfg.torque_input == "command":
    return traj.cmd_torque_nm
  return traj.torque_nm


def _new_csv_writer(log_dir: str, tag: str) -> tuple[str, "csv._writer", object]:
  os.makedirs(log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join(log_dir, f"sysid_chirp_{tag}_{ts}.csv")
  f = open(csv_path, "w", newline="")
  writer = csv.writer(f)
  writer.writerow(_CSV_HEADER)
  return csv_path, writer, f


def _plot_log(csv_path: str, title: str, cfg: ChirpSysidConfig) -> None:
  """진단용 그래프. torque 서브플롯에는 항상 원본(raw) motor_torque_nm과
  cmd_torque_nm을 그리고, 초록선은 "지금 fit/compare가 qfrc_applied로 실제
  사용할 값"을 그대로 재현해서 보여준다 — cfg.torque_input이 "measured"면
  스무딩된 motor_torque_nm, "command"면 cmd_torque_nm 그 자체가 초록선이
  된다(둘 다 _physics_input_torque와 완전히 같은 로직).
  """
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib이 없어 그래프를 건너뜁니다. (pip install matplotlib)")
    return

  traj = load_csv(csv_path)  # raw, 스무딩 없음 (그래프의 파란/빨간 원본선용)
  # motor_angle_deg는 원래 멀티턴(랩어라운드 없이 계속 누적)으로 설계돼 있어
  # np.unwrap()이 필요 없다 — 오히려 해롭다: 한 샘플 구간(1/control_hz) 동안
  # 실제 각도 변화가 180도를 넘으면(예: 150ms 샘플링 + 최고속도
  # ~1300deg/s면 0.15*1300=195deg로 이미 초과) unwrap이 이걸 "wrap됐다"고
  # 착각해서 ±360도를 잘못 더해버리고, 그 이후 전체 샘플이 그 오프셋만큼
  # 밀린다 — 실측 그래프에서 "속도는 음수인데 각도가 갑자기 확 뛰는" 것처럼
  # 보이는 원인이었다. (load_csv에서 이미 속도로 보정된 unwrap이 적용된다.)
  angle_deg = np.degrees(traj.angle_rad)
  vel_deg_s = np.degrees(traj.vel_rad_s)

  # fit/compare가 실제로 qfrc_applied에 넣을 값과 정확히 같은 걸 그린다 —
  # torque_smooth_samples>1이고 torque_input="measured"이면 이동평균까지
  # 적용된 값, torque_input="command"이면 cmd_torque_nm 그 자체.
  traj_for_fit = load_csv(csv_path, torque_smooth_samples=cfg.torque_smooth_samples)
  physics_torque = _physics_input_torque(traj_for_fit, cfg)
  if cfg.torque_input == "command":
    physics_label = "physics input = cmd_torque_nm (torque-input=command)"
  elif cfg.torque_smooth_samples > 1:
    physics_label = f"physics input = motor_torque_nm (smoothed, n={cfg.torque_smooth_samples})"
  else:
    physics_label = "physics input = motor_torque_nm (raw, 스무딩 없음)"

  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  axes[0].plot(traj.time_s, angle_deg, color="tab:blue")
  axes[0].set_ylabel("angle (deg)")
  axes[0].set_title(title)
  axes[0].grid(True)

  axes[1].plot(traj.time_s, vel_deg_s, color="tab:orange")
  axes[1].set_ylabel("velocity (deg/s)")
  axes[1].grid(True)

  axes[2].plot(
    traj.time_s, traj.cmd_torque_nm, color="0.5", linestyle="--", label="cmd_torque_nm"
  )
  axes[2].plot(
    traj.time_s, traj.torque_nm, color="tab:red", alpha=0.4, label="motor_torque_nm (raw)"
  )
  axes[2].plot(
    traj.time_s,
    physics_torque,
    color="tab:green",
    linewidth=1.8,
    label=physics_label,
  )
  axes[2].set_ylabel("torque (Nm)")
  axes[2].set_xlabel("time (s)")
  axes[2].legend()
  axes[2].grid(True)

  fig.tight_layout()
  png_path = os.path.splitext(csv_path)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  plt.close(fig)
  print(f"그래프 저장 완료: {png_path}")


# --------------------------------------------------------------------------
# 1. Collect (실물 또는 --simulate)
# --------------------------------------------------------------------------


def collect_hardware(cfg: ChirpSysidConfig) -> str:
  """실제 CAN 모터에 chirp 토크를 open-loop로 가하며 CSV 로그를 만든다."""
  # 하드웨어 의존 모듈은 여기서만 import한다 — --simulate 전용 사용자는
  # 이 모듈들이 없어도 스크립트 전체 import가 깨지지 않게 하기 위함.
  from mjlab.tasks.inverse.commission_motor import (
    _open_bus,
    _shutdown_bus,
    disable_mit,
    enable_mit,
    mit_position_command,
  )
  from mjlab.tasks.inverse.robot_state_reader import RobotStateReader

  duration_s = excitation_duration_s(cfg)
  n_samples = int(duration_s * cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / cfg.control_hz
  cmd_torque_arr = build_excitation_torque(cfg, sample_times)

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

  csv_path, writer, f = _new_csv_writer(cfg.log_dir, "hw")
  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  if cfg.damping_test:
    print(
      f"spin_coast: {cfg.n_cycles} cycles x (spin {cfg.spin_duration_s}s @ "
      f"{cfg.spin_torque_nm}Nm + coast {cfg.coast_duration_s}s), "
      f"total {duration_s:.1f}s"
    )
  else:
    print(f"chirp {cfg.chirp_f0_hz}->{cfg.chirp_f1_hz} Hz over {duration_s:.1f}s")
  print(f"CSV 로그: {csv_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(여기 신호 완료)"
  initial_angle_deg: float | None = None
  # try 블록 밖(if cfg.plot: 검사)에서도 참조하므로 미리 0으로 초기화 —
  # enable_mit 등 초반 단계에서 실패하면 한 샘플도 못 쓰고 끝날 수 있다.
  sample_idx = 0

  try:
    reader.start()
    print("Enabling motor...")
    enable_ok = enable_mit(bus, cfg.motor_id)
    if cfg.require_enable_ack and not enable_ok:
      raise RuntimeError("enable_mit was not acknowledged")
    time.sleep(0.1)

    # 초기 상태 대기.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
      state = reader.get_state()
      if state.motor_angle_deg is not None and state.motor_velocity_deg_s is not None:
        initial_angle_deg = state.motor_angle_deg
        break
      time.sleep(0.01)
    if initial_angle_deg is None:
      raise RuntimeError("모터 초기 상태를 읽지 못했습니다 (CAN 피드백 확인).")

    period_s = 1.0 / cfg.control_hz
    deadline = start_t + min(duration_s, cfg.max_runtime_s)
    print("Starting chirp torque excitation.")
    while time.monotonic() < deadline and sample_idx < n_samples:
      loop_start = time.monotonic()
      cmd_torque_nm = float(cmd_torque_arr[sample_idx])

      ok = mit_position_command(
        bus, cfg.motor_id, 0.0, kp=0.0, kd=0.0, torque_nm=cmd_torque_nm
      )
      if not ok:
        raise RuntimeError("No MIT reply received from motor")

      state = reader.get_state()
      angle_deg = state.motor_angle_deg
      vel_deg_s = state.motor_velocity_deg_s
      torque_nm = state.motor_torque_nm
      if angle_deg is None or vel_deg_s is None or torque_nm is None:
        raise RuntimeError("Motor state unavailable from RobotStateReader")

      if abs(torque_nm) > cfg.abort_on_torque_nm:
        raise RuntimeError(
          f"Aborting: |motor_torque_nm|={abs(torque_nm):.2f} exceeded "
          f"abort_on_torque_nm={cfg.abort_on_torque_nm}"
        )
      # 주의: 실린더(액추에이터) 쪽 CAN 피드백은 멀티턴(랩어라운드 없음,
      # 계속 누적)이므로 % 360으로 감싸면 안 된다 — 감싸면 몇 바퀴를 돌아도
      # drift가 항상 -180~180 사이로 계산되어 이 안전장치가 사실상 무력화된다
      # (실측으로 실제 겪은 버그: 3000도 이상 벗어났는데도 통과됨).
      drift_deg = angle_deg - initial_angle_deg
      if abs(drift_deg) > cfg.torque_drift_limit_deg:
        raise RuntimeError(
          f"Aborting: drift={drift_deg:.1f}deg exceeded "
          f"torque_drift_limit_deg={cfg.torque_drift_limit_deg}"
        )
      if abs(vel_deg_s) > cfg.torque_velocity_limit_deg_s:
        raise RuntimeError(
          f"Aborting: |vel|={abs(vel_deg_s):.1f}deg/s exceeded "
          f"torque_velocity_limit_deg_s={cfg.torque_velocity_limit_deg_s}"
        )

      elapsed_t = time.monotonic() - start_t
      writer.writerow(
        [
          f"{elapsed_t:.4f}",
          f"{cmd_torque_nm:.4f}",
          f"{angle_deg:.3f}",
          f"{vel_deg_s:.3f}",
          f"{torque_nm:.4f}",
        ]
      )
      if sample_idx % 20 == 0:
        print(
          f"t={elapsed_t:6.2f}s cmd={cmd_torque_nm:+5.2f}Nm "
          f"angle={angle_deg:+8.3f}deg torque={torque_nm:+5.2f}Nm"
        )

      sample_idx += 1
      sleep_s = period_s - (time.monotonic() - loop_start)
      if sleep_s > 0:
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
      f.close()
      print(f"CSV 로그 저장 완료: {csv_path}")

  print(f"여기 신호 종료 ({stopped_reason}).")
  if cfg.plot and sample_idx > 0:
    _plot_log(
      csv_path,
      f"hardware {'damping_test' if cfg.damping_test else 'chirp'} torque",
      cfg,
    )
  elif cfg.plot:
    print("[plot] 수집된 샘플이 없어 그래프를 건너뜁니다 (여기 신호가 시작되기 전에 종료됨).")
  return csv_path


def collect_simulated(
  cfg: ChirpSysidConfig,
  tag: str = "sim",
  param_overrides: dict[str, float] | None = None,
  cmd_torque_arr: np.ndarray | None = None,
  initial_state: tuple[float, float] | None = None,
  sample_times: np.ndarray | None = None,
) -> str:
  """MuJoCo로 chirp 토크 여기를 모사한다 (하드웨어 불필요).

  `param_overrides`가 있으면 관절의 damping/armature/frictionloss를 그
  값으로 덮어쓴다 — step 3(fit된 파라미터로 재시뮬레이션 후 비교)에서
  재사용한다. `cmd_torque_arr`가 있으면 새로 chirp을 만드는 대신 그
  시퀀스를 그대로 흘린다 — compare 단계에서는 여기에 (명령값이 아니라)
  실측 torque_nm을 넣어 호출한다: "실제로 관절에 가해진 힘과 완전히 같은
  입력"으로 재생해야 파라미터 비교가 공정하기 때문이다. 인자 이름은
  collect(자체 chirp 생성) 용도를 기준으로 남겨뒀다.

  `initial_state`가 있으면 (angle_rad, vel_rad_s)로 초기 qpos/qvel을
  맞춘다 — 실측 로그는 torque-only open-loop라 원점(0)에서 시작한다는
  보장이 없다(이전 실험이 원위치로 안 돌아온 채 끝났을 수 있음). 이걸
  안 맞추고 항상 qpos=0에서 시작하면, 실측 시작각이 0이 아닌 경우 fit이
  아무리 정확해도 free-run 비교의 angle이 처음부터 크게 어긋난다(실측으로
  확인된 버그: 실측이 215deg에서 시작했는데 sim은 0에서 시작해 계속
  어긋남).

  `sample_times`가 있으면 균일 간격(1/control_hz) 대신 그 실제 타임스탬프
  간격으로 스텝을 밟는다 — compare 단계에서 실측 로그의 샘플 지터(최대
  2~3배 차이)를 그대로 반영해야 free-run 비교가 공정하다.
  """
  model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  # explicit Euler(기본값)는 damping/armature가 거의 0에 가까운 모델에서
  # 노이즈 섞인 실측 토크를 그대로 힘으로 가하면 몇 스텝 만에 QACC가
  # 발산할 수 있다(실측으로 확인) — implicitfast가 훨씬 안정적이다.
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  joint = model.joint(cfg.joint_name)
  dof = joint.dofadr[0]
  qadr = joint.qposadr[0]
  act_id = model.actuator(cfg.actuator_name).id
  # 위치 액추에이터의 힘 기여를 0으로 죽인다 — 순수 토크 open-loop 여기이므로
  # 내장 PD가 복원력을 만들면 안 된다 (sysid_fit.py의 fit_torque에서 확인된
  # 동일 버그: ctrl 클램프로 인한 스퓨리어스 forcerange 힘을 피하려면 gain/bias
  # 자체를 0으로 만들어야 한다).
  model.actuator_gainprm[act_id, 0] = 0.0
  model.actuator_biasprm[act_id, 1] = 0.0

  if param_overrides:
    if "cylinder_damping" in param_overrides:
      model.dof_damping[dof] = param_overrides["cylinder_damping"]
    if "cylinder_armature" in param_overrides:
      model.dof_armature[dof] = param_overrides["cylinder_armature"]
    if "cylinder_frictionloss" in param_overrides:
      model.dof_frictionloss[dof] = param_overrides["cylinder_frictionloss"]

  if cmd_torque_arr is None:
    n_samples = int(excitation_duration_s(cfg) * cfg.control_hz) + 1
    sample_times = np.arange(n_samples) / cfg.control_hz
    cmd_torque_arr = build_excitation_torque(cfg, sample_times)
  else:
    n_samples = len(cmd_torque_arr)
    if sample_times is None:
      sample_times = np.arange(n_samples) / cfg.control_hz

  dt_arr = np.diff(sample_times) if len(sample_times) > 1 else np.array([1.0 / cfg.control_hz])
  # 실측 sample_times를 그대로 쓰는 경우(compare 단계의 free-run 재생 등),
  # 하드웨어 루프 지연으로 생긴 비정상적으로 큰 dt가 섞여 있을 수 있다.
  # fit_gauss_newton/_windowed_replay_rmse는 그런 window를 통째로 건너뛰지만,
  # 여기(collect_simulated)는 CSV/그래프용으로 n_samples개를 빠짐없이 다
  # 만들어야 해서 건너뛸 수 없다 — 대신 dt 자체를 median의 max_dt_multiple
  # 배로 캡을 씌운다. 이 캡이 없으면 단 하나의 큰 dt만으로도 그 한 스텝이
  # 사실상 "몇 초짜리 초대형 스텝"이 되어 QACC가 발산한다(실측으로 확인:
  # dt 튐 구간에서 속도가 5e9deg/s까지 튀는 수치 불안정 발생).
  dt_median = float(np.median(dt_arr)) if len(dt_arr) else 1.0 / cfg.control_hz
  dt_cap = dt_median * cfg.max_dt_multiple
  n_capped = int(np.sum(dt_arr > dt_cap))
  if n_capped:
    print(
      f"[simulate] dt outlier {n_capped}개를 {dt_cap:.4f}s로 캡 처리합니다 "
      f"(median dt={dt_median:.4f}s, max_dt_multiple={cfg.max_dt_multiple})."
    )
    dt_arr = np.minimum(dt_arr, dt_cap)
  model.opt.timestep = float(dt_arr[0])
  data = mujoco.MjData(model)
  if initial_state is not None:
    data.qpos[qadr] = initial_state[0]
    data.qvel[dof] = initial_state[1]
  mujoco.mj_forward(model, data)

  csv_path, writer, f = _new_csv_writer(cfg.log_dir, tag)
  print(f"[simulate:{tag}] {cfg.xml_path} 로 chirp 토크 여기 모사")
  if param_overrides:
    print(f"  파라미터 override: {param_overrides}")

  for i in range(n_samples):
    tau = float(cmd_torque_arr[i])
    # 먼저 "현재" 상태(=sample_times[i] 시점)를 로그에 쓰고, 그 다음에
    # sample_times[i]->sample_times[i+1] 구간만큼 스텝을 밟는다. 예전에는
    # 순서가 반대였다(스텝을 먼저 밟고 그 결과를 sample_times[i]로 기록) —
    # 그러면 row i에 실제로는 sample_times[i+1] 시점의 상태가 sample_times[i]
    # 라벨로 기록되어, 전체 sim 궤적이 한 스텝씩 시간상 앞으로 밀려 저장된다.
    # 200Hz(5ms)에서는 오차가 작아 티가 안 났지만, 150ms 회귀식 스크립트처럼
    # 스텝이 커지면 "같은 시각"으로 나란히 그린 각도/속도가 서로 안 맞는
    # 것처럼 보이는(실측으로 확인된) 원인이 된다 — 실측 로그는 이런 오프셋이
    # 없으므로 실측과 비교할 때 특히 문제가 된다.
    angle_deg = math.degrees(data.qpos[qadr])
    vel_deg_s = math.degrees(data.qvel[dof])
    writer.writerow(
      [f"{sample_times[i]:.4f}", f"{tau:.4f}", f"{angle_deg:.3f}", f"{vel_deg_s:.3f}", f"{tau:.4f}"]
    )
    if i < len(dt_arr):
      pre_qpos = float(data.qpos[qadr])
      pre_qvel = float(data.qvel[dof])
      data.qfrc_applied[dof] = tau
      model.opt.timestep = float(dt_arr[i])
      mujoco.mj_step(model, data)
      post_qvel = float(data.qvel[dof])
      # dt 캡을 씌워도 QACC 발산이 재현되는 경우가 있었다(실측으로 확인) —
      # dt가 아니라 그 순간의 입력 토크(input_torque[i], 즉 tau) 자체가
      # 센서 튐 등으로 비정상적으로 크거나, armature가 매우 작아 그 토크
      # 하나만으로도 한 스텝 안에서 속도가 감당 못 할 만큼 튀는 경우다.
      # 이런 스텝은 통째로 취소(이전 상태로 되돌림)하고 원인을 로그로
      # 남긴다 — 발산이 전체 궤적으로 전파되어 free-run RMSE/그래프가
      # 통째로 못 쓰게 되는 것보다, 이 한 스텝만 버리는 게 훨씬 낫다.
      #
      # 임계값은 실물 하드웨어에도 이미 걸어둔 안전 한계
      # (cfg.torque_velocity_limit_deg_s, 기본 2000deg/s)의 5배로 잡는다 —
      # 예전엔 1e5rad/s(=약 573만deg/s)로 너무 느슨하게 잡았더니, 이미
      # -8982rad/s(약 -51만deg/s, 실측 최대치의 수백 배)까지 발산한 상태를
      # 못 잡고 넘어가서 그 오염된 상태가 CSV에 계속 쌓이고 그걸 다시 불러올
      # 때 unwrap이 수백 번씩 보정을 시도하는 2차 증상으로 번졌다(실측으로
      # 확인). 발산 "시작 시점"에서 바로 잡아야 이런 전파를 막을 수 있다.
      divergence_vel_rad_s = math.radians(cfg.torque_velocity_limit_deg_s) * 5.0
      if not np.isfinite(post_qvel) or abs(post_qvel) > divergence_vel_rad_s:
        print(
          f"[simulate] 불안정 스텝 취소: i={i} t={sample_times[i]:.3f}s "
          f"tau={tau:.4f}Nm dt={dt_arr[i]:.4f}s "
          f"(스텝 전 qpos={pre_qpos:.4f}rad, qvel={pre_qvel:.4f}rad/s -> "
          f"스텝 후 qvel={post_qvel:.4e}rad/s로 발산, 임계값="
          f"{divergence_vel_rad_s:.1f}rad/s) — 이전 상태로 되돌립니다."
        )
        data.qpos[qadr] = pre_qpos
        data.qvel[dof] = pre_qvel
        mujoco.mj_forward(model, data)

  f.close()
  print(f"CSV 로그 저장 완료: {csv_path}")
  if cfg.plot:
    _plot_log(
      csv_path,
      f"simulated chirp torque ({tag})",
      cfg,
    )
  return csv_path


# --------------------------------------------------------------------------
# 2. Fit (Gauss-Newton / Levenberg-Marquardt, mujoco.minimize.least_squares)
# --------------------------------------------------------------------------

_PARAM_NAMES = ["cylinder_damping", "cylinder_armature", "cylinder_frictionloss"]
# damping == MuJoCo dof_damping == 점성(viscous) 마찰계수.
# frictionloss == MuJoCo dof_frictionloss == Coulomb(건마찰) 토크.


def fit_gauss_newton(
  traj: LoggedTrajectory, cfg: ChirpSysidConfig
) -> tuple[dict[str, float], float]:
  """MuJoCo 재생(replay) 기반 비선형 최소자승 피팅.

  회귀식을 직접 세워 한 번에 푸는 대신(예: tau ~= I*qddot + b*qdot + ...),
  후보 파라미터로 실제 MuJoCo 스텝을 반복 실행해 실측 (angle, velocity)
  궤적과의 오차를 줄이는 방향으로 파라미터를 갱신한다.
  `mujoco.minimize.least_squares`는 Levenberg-Marquardt를 구현하며, 매 반복
  Jacobian J로 Hessian을 J^T J로 근사(Gauss-Newton 근사)해 스텝을 계산한다
  — mujoco-sysid 저장소의 데모 노트북이 쓰는 것과 동일한 최적화기다.

  Pure open-loop torque replay는 위치 복원력이 없어 초기 파라미터 오차가
  지수적으로 누적되므로(카오스는 아니지만 자유적분), `shooting_window_s`
  마다 실측 상태로 재앵커링하는 multiple-shooting을 쓴다.
  """
  base_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  base_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  joint = base_model.joint(cfg.joint_name)
  dof = joint.dofadr[0]
  qadr = joint.qposadr[0]
  act_id = base_model.actuator(cfg.actuator_name).id

  x0 = np.array(
    [
      base_model.dof_damping[dof],
      base_model.dof_armature[dof],
      base_model.dof_frictionloss[dof],
    ]
  )
  lower = np.array([0.0, 0.0, 0.0])
  upper = np.array([cfg.damping_max, cfg.armature_max, cfg.frictionloss_max])

  # 자동 계층적 식별: 이번 실행이 damping-test(spin_coast)가 아니고
  # fix_damping을 명시적으로 안 줬다면, output_report에 이전 damping-test
  # 결과가 있는지 확인해서 있으면 그 damping을 자동으로 고정값으로 쓴다.
  # 이렇게 하면 `--damping-test`로 한 번 돌리고, 그 다음 chirp 실행에서는
  # --fix-damping을 따로 안 줘도 armature/frictionloss만 다시 fit된다.
  effective_fix_damping = cfg.fix_damping
  if effective_fix_damping is None and not cfg.damping_test:
    prev_path = Path(cfg.output_report)
    if prev_path.exists():
      try:
        prev_report = json.loads(prev_path.read_text())
        if prev_report.get("damping_test") is True:
          effective_fix_damping = prev_report["params"]["fitted"]["cylinder_damping"]
          print(
            f"[fit] {prev_path}의 damping-test 결과에서 "
            f"damping={effective_fix_damping:.6f}을 자동으로 고정합니다."
          )
      except (json.JSONDecodeError, KeyError):
        pass

  # fix_*가 지정된 파라미터는 최적화 대상에서 아예 빼고 상수로 고정한다
  # (예: coast-down에서 구한 damping을 여기 chirp fit에서는 상수로 취급하고
  # armature/frictionloss만 다시 맞추는 계층적 식별). mujoco.minimize.
  # least_squares는 bounds[0] < bounds[1]을 "모든" 파라미터에 대해 엄격히
  # 요구해서(lower[i]==upper[i]가 하나라도 있으면 ValueError), 고정
  # 파라미터를 lower=upper로 넣는 방식은 쓸 수 없다 — 대신 free 파라미터만
  # 골라 최적화하고, 고정값은 residual 계산 시점에 다시 채워 넣는다.
  fixed = [effective_fix_damping, cfg.fix_armature, cfg.fix_frictionloss]
  for i, fv in enumerate(fixed):
    if fv is not None:
      x0[i] = fv
  free_idx = [i for i, fv in enumerate(fixed) if fv is None]

  input_torque = _physics_input_torque(traj, cfg)

  real = np.stack([traj.angle_rad, traj.vel_rad_s], axis=1)
  # 속도(유한차분/센서 노이즈)보다 위치 채널을 더 신뢰한다.
  weights = np.array([1.0, 0.1])

  dt = float(np.median(np.diff(traj.time_s)))
  # 실측 샘플 간격이 균일하지 않을 수 있다(CAN/Python 루프 지터로 실측
  # min/max dt가 2~3배 차이나는 게 흔하다). 모든 스텝에 median dt 하나를
  # 강제로 쓰면 재생된 궤적의 타이밍이 체계적으로 어긋나 fit이 편향된다 —
  # 스텝마다 실제 측정된 간격을 그대로 쓴다.
  dt_arr = np.diff(traj.time_s)
  dt_median = float(np.median(dt_arr))
  dt_bad_threshold = dt_median * cfg.max_dt_multiple
  window = max(1, int(round(cfg.shooting_window_s / dt)))
  n_samples = len(traj.time_s)

  def _residual_single(x_free: np.ndarray) -> np.ndarray:
    x = x0.copy()
    for j, i in enumerate(free_idx):
      x[i] = x_free[j]
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.dof_damping[dof] = x[0]
    model.dof_armature[dof] = x[1]
    model.dof_frictionloss[dof] = x[2]
    model.actuator_gainprm[act_id, 0] = 0.0
    model.actuator_biasprm[act_id, 1] = 0.0
    data = mujoco.MjData(model)

    chunks = []
    for start in range(0, n_samples, window):
      end = min(start + window, n_samples)
      # 이 window 안에 CAN/Python 루프 지연으로 생긴 비정상적으로 큰 dt
      # (median의 max_dt_multiple배 이상)가 있으면 통째로 건너뛴다 — 그런
      # outlier 하나가 최소자승 비용을 지배해서 armature/frictionloss가
      # 경계값(0)으로 밀려버리는 걸 실측으로 확인했다.
      window_dt = dt_arr[start : min(end, len(dt_arr))]
      if window_dt.size and np.max(window_dt) > dt_bad_threshold:
        continue
      data.qpos[qadr] = traj.angle_rad[start]
      data.qvel[dof] = traj.vel_rad_s[start]
      mujoco.mj_forward(model, data)
      sim = np.empty((end - start, 2))
      for k in range(start, end):
        # cmd_torque_nm(명령값)이 아니라 torque_nm(실측)을 입력으로 쓴다 —
        # 대역폭 부족이나 고속에서의 back-EMF 토크 제한 등으로 실제 전달된
        # 토크가 명령보다 작을 수 있는데, 그 차이를 damping/frictionloss가
        # 대신 흡수해버리면 파라미터가 과대추정된다(실측으로 확인: 속도가
        # 커질수록 motor_torque_nm이 cmd_torque_nm보다 작아지는 구간 존재).
        data.qfrc_applied[dof] = input_torque[k]
        model.opt.timestep = float(dt_arr[k]) if k < len(dt_arr) else float(dt_arr[-1])
        mujoco.mj_step(model, data)
        sim[k - start, 0] = data.qpos[qadr]
        sim[k - start, 1] = data.qvel[dof]
      chunks.append((sim - real[start:end]) * weights)
    if not chunks:
      raise RuntimeError(
        "모든 shooting window가 dt outlier로 제외됐습니다 — "
        "max_dt_multiple을 늘리거나 로그 품질을 확인하세요."
      )
    return np.concatenate(chunks).reshape(-1)

  def residual(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
      return _residual_single(x)
    return np.stack([_residual_single(x[:, k]) for k in range(x.shape[1])], axis=1)

  if not free_idx:
    # 세 파라미터 다 고정된 경우(드묾) — 최적화할 게 없으니 residual만 한 번
    # 평가해서 cost를 낸다.
    res = _residual_single(np.empty(0))
    cost = float(np.sum(res**2))
    print(f"[fit] 모든 파라미터가 고정되어 최적화를 건너뜁니다. cost = {cost:.6f}")
    return dict(zip(_PARAM_NAMES, x0.tolist(), strict=True)), cost

  x0_free = x0[free_idx]
  lower_free = lower[free_idx]
  upper_free = upper[free_idx]

  x_fit_free, trace = mujoco.minimize.least_squares(  # type: ignore[attr-defined]
    x0_free, residual, bounds=(lower_free, upper_free), x_scale="jac", verbose=0
  )
  cost = float(trace[-1].objective) if trace else float("nan")
  print(f"[fit] Gauss-Newton(LM) final cost = {cost:.6f}")
  # 진단용(임시): armature 하나만 free일 때 optimizer가 x0에서 실제로 얼마나
  # 움직였는지 확인하기 위해 반복 횟수/초기-최종 값을 같이 출력한다.
  print(
    f"[fit][debug] free_idx={free_idx}, LM iterations={len(trace)}, "
    f"x0_free={x0_free.tolist()}, x_fit_free={x_fit_free.tolist()}"
  )

  x_fit = x0.copy()
  for j, i in enumerate(free_idx):
    x_fit[i] = x_fit_free[j]
  return dict(zip(_PARAM_NAMES, x_fit.tolist(), strict=True)), cost


def _nominal_params(cfg: ChirpSysidConfig) -> dict[str, float]:
  model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  dof = model.joint(cfg.joint_name).dofadr[0]
  values = [model.dof_damping[dof], model.dof_armature[dof], model.dof_frictionloss[dof]]
  return dict(zip(_PARAM_NAMES, (float(v) for v in values), strict=True))


def write_report(
  cfg: ChirpSysidConfig,
  csv_path: str,
  fitted: dict[str, float],
  cost: float,
  compare_rmse: dict[str, dict[str, float]] | None = None,
) -> None:
  nominal = _nominal_params(cfg)
  table = PrettyTable()
  table.field_names = ["parameter", "nominal", "fitted"]
  for name in _PARAM_NAMES:
    table.add_row([name, f"{nominal[name]:.6f}", f"{fitted[name]:.6f}"])
  print(table)

  report = {
    "csv_path": csv_path,
    "xml_path": cfg.xml_path,
    "joint_name": cfg.joint_name,
    "method": "gauss_newton_torque_replay",  # mujoco.minimize.least_squares (LM)
    "damping_test": cfg.damping_test,
    "final_cost": cost,
    "params": {"nominal": nominal, "fitted": fitted},
  }
  if compare_rmse is not None:
    report["compare_rmse"] = compare_rmse

  out_path = Path(cfg.output_report)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(report, indent=2))
  print(f"\nReport written to {out_path}")
  print("Note: XML은 자동 반영되지 않습니다 — 리포트를 확인 후 수동 반영하세요.")


# --------------------------------------------------------------------------
# 3. Compare (fit된 파라미터로 재시뮬레이션 vs 실측)
# --------------------------------------------------------------------------


def _windowed_replay_rmse(
  cfg: ChirpSysidConfig, real_traj: LoggedTrajectory, fitted: dict[str, float]
) -> dict[str, float]:
  """fit이 실제로 최적화한 것과 같은 지표: shooting_window_s마다 실측 상태로
  재앵커링하면서 그 구간만 예측하고, 구간별 예측오차를 모아 RMSE를 낸다.

  자유 적분(re-anchor 없이 끝까지 이어 붙이는) RMSE는 회전형 역진자처럼
  약하게 카오스적인 결합계에서는 파라미터가 조금만 달라도 지수적으로
  벌어지므로, "짧은 구간 예측이 얼마나 정확한가"를 보는 이 지표가 fit
  품질을 훨씬 공정하게 반영한다.
  """
  model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  joint = model.joint(cfg.joint_name)
  dof = joint.dofadr[0]
  qadr = joint.qposadr[0]
  act_id = model.actuator(cfg.actuator_name).id
  model.actuator_gainprm[act_id, 0] = 0.0
  model.actuator_biasprm[act_id, 1] = 0.0
  model.dof_damping[dof] = fitted["cylinder_damping"]
  model.dof_armature[dof] = fitted["cylinder_armature"]
  model.dof_frictionloss[dof] = fitted["cylinder_frictionloss"]

  input_torque = _physics_input_torque(real_traj, cfg)

  dt = float(np.median(np.diff(real_traj.time_s)))
  # fit_gauss_newton과 동일한 이유로 median dt 하나로 고정하지 않고 스텝마다
  # 실측 간격을 그대로 쓴다 (샘플 지터가 커서 median으로 뭉개면 타이밍이
  # 체계적으로 어긋난다).
  dt_arr = np.diff(real_traj.time_s)
  dt_median = float(np.median(dt_arr))
  dt_bad_threshold = dt_median * cfg.max_dt_multiple
  data = mujoco.MjData(model)

  window = max(1, int(round(cfg.shooting_window_s / dt)))
  n_samples = len(real_traj.time_s)

  angle_err = []
  vel_err = []
  for start in range(0, n_samples, window):
    end = min(start + window, n_samples)
    # fit_gauss_newton과 동일하게 dt outlier가 낀 window는 제외한다 —
    # 안 그러면 이 RMSE도 그 outlier 하나 때문에 부풀려진다.
    window_dt = dt_arr[start : min(end, len(dt_arr))]
    if window_dt.size and np.max(window_dt) > dt_bad_threshold:
      continue
    data.qpos[qadr] = real_traj.angle_rad[start]
    data.qvel[dof] = real_traj.vel_rad_s[start]
    mujoco.mj_forward(model, data)
    for k in range(start, end):
      # fit_gauss_newton과 동일하게 cfg.torque_input이 고른 채널을 입력으로
      # 쓴다 (아래 compare_real_vs_sim 참고).
      data.qfrc_applied[dof] = input_torque[k]
      model.opt.timestep = float(dt_arr[k]) if k < len(dt_arr) else float(dt_arr[-1])
      mujoco.mj_step(model, data)
      angle_err.append(data.qpos[qadr] - real_traj.angle_rad[k])
      vel_err.append(data.qvel[dof] - real_traj.vel_rad_s[k])

  angle_err_deg = np.degrees(np.array(angle_err))
  vel_err_deg_s = np.degrees(np.array(vel_err))
  return {
    "angle_rmse_deg": float(np.sqrt(np.mean(angle_err_deg**2))),
    "vel_rmse_deg_s": float(np.sqrt(np.mean(vel_err_deg_s**2))),
  }


def compare_real_vs_sim(
  cfg: ChirpSysidConfig, real_csv_path: str, fitted: dict[str, float]
) -> dict[str, dict[str, float]]:
  """실측 CSV와 "같은 (실측) 토크 입력"으로 fit된 파라미터를 재생해 비교한다.

  fit_gauss_newton과 동일한 이유로 cmd_torque_nm이 아니라 torque_nm(실측)을
  입력으로 쓴다 — 안 그러면 "명령한 대로 토크가 정확히 나갔다"는 틀린
  가정이 fit 단계와 compare 단계에서 서로 다르게 작용해 비교 자체가
  무의미해진다.

  두 가지 지표를 함께 낸다.
  - windowed RMSE: fit이 최적화한 것과 같은 shooting-window 재앵커링 예측
    오차. "이 파라미터가 실제로 얼마나 잘 맞는지"를 보려면 이 값을 봐야 한다.
  - free-run RMSE: 재앵커링 없이 전체 구간을 한 번에 이어 붙인 오차. 결합계가
    파라미터 민감도(카오스성)가 얼마나 큰지를 보여주는 참고 지표일 뿐,
    이 값이 크다고 fit이 잘못됐다는 뜻은 아니다.
  """
  real_traj = load_csv(real_csv_path, torque_smooth_samples=cfg.torque_smooth_samples)
  input_torque = _physics_input_torque(real_traj, cfg)
  sim_csv_path = collect_simulated(
    cfg,
    tag="fitted_compare",
    param_overrides=fitted,
    cmd_torque_arr=input_torque,
    initial_state=(float(real_traj.angle_rad[0]), float(real_traj.vel_rad_s[0])),
    sample_times=real_traj.time_s,
  )
  sim_traj = load_csv(sim_csv_path)

  n = min(len(real_traj.time_s), len(sim_traj.time_s))
  angle_err_deg = np.degrees(real_traj.angle_rad[:n] - sim_traj.angle_rad[:n])
  vel_err_deg_s = np.degrees(real_traj.vel_rad_s[:n] - sim_traj.vel_rad_s[:n])
  free_run_rmse = {
    "angle_rmse_deg": float(np.sqrt(np.mean(angle_err_deg**2))),
    "vel_rmse_deg_s": float(np.sqrt(np.mean(vel_err_deg_s**2))),
  }
  windowed_rmse = _windowed_replay_rmse(cfg, real_traj, fitted)
  rmse = {"windowed": windowed_rmse, "free_run": free_run_rmse}

  print(
    f"[compare] windowed({cfg.shooting_window_s}s re-anchor) RMSE: "
    f"angle={windowed_rmse['angle_rmse_deg']:.3f} deg, "
    f"vel={windowed_rmse['vel_rmse_deg_s']:.3f} deg/s"
  )
  print(
    f"[compare] free-run(재앵커링 없음) RMSE: "
    f"angle={free_run_rmse['angle_rmse_deg']:.3f} deg, "
    f"vel={free_run_rmse['vel_rmse_deg_s']:.3f} deg/s "
    "(참고용 — 결합계 민감도를 보여줄 뿐 fit 품질 지표 아님)"
  )

  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = real_traj.time_s[:n]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # _plot_log와 동일한 이유로 unwrap을 쓰지 않는다 — 빠른 움직임 +
    # 굵은(150ms급) 샘플링에서 unwrap이 정상적인 큰 각도 변화를 wrap으로
    # 착각해 ±360도 스퓨리어스 오프셋을 넣는 걸 방지한다.
    axes[0].plot(t, np.degrees(real_traj.angle_rad[:n]), label="real")
    axes[0].plot(t, np.degrees(sim_traj.angle_rad[:n]), label="sim(fitted)", linestyle="--")
    axes[0].set_ylabel("angle (deg)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, np.degrees(real_traj.vel_rad_s[:n]), label="real")
    axes[1].plot(t, np.degrees(sim_traj.vel_rad_s[:n]), label="sim(fitted)", linestyle="--")
    axes[1].set_ylabel("velocity (deg/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    out_png = os.path.join(cfg.log_dir, "sysid_compare.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"비교 그래프 저장 완료: {out_png}")
  except ImportError:
    pass

  return rmse


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
  cfg = tyro.cli(ChirpSysidConfig)

  csv_path = cfg.csv_path
  fitted: dict[str, float] | None = None
  cost = float("nan")
  did_fit = False  # 이번 실행에서 실제로 fit_gauss_newton을 돌렸는지.

  if cfg.stage in ("collect", "full"):
    csv_path = collect_simulated(cfg, tag="hw_sim") if cfg.simulate else collect_hardware(cfg)

  if cfg.stage in ("fit", "full"):
    if not csv_path:
      raise SystemExit("--csv-path가 필요합니다 (또는 --stage full/collect로 먼저 수집).")
    traj = load_csv(csv_path, torque_smooth_samples=cfg.torque_smooth_samples)
    fitted, cost = fit_gauss_newton(traj, cfg)
    did_fit = True

  compare_rmse = None
  if cfg.stage in ("compare", "full"):
    if fitted is None:
      if not Path(cfg.fitted_params_json).exists():
        raise SystemExit(
          f"{cfg.fitted_params_json}이 없습니다 — 먼저 --stage fit을 실행하세요."
        )
      report = json.loads(Path(cfg.fitted_params_json).read_text())
      fitted = report["params"]["fitted"]
    if not csv_path:
      raise SystemExit("--csv-path가 필요합니다 (비교 대상 실측 로그).")
    compare_rmse = compare_real_vs_sim(cfg, csv_path, fitted)

  # `--stage compare`만 단독으로 돌린 경우(fitted를 새로 fit한 게 아니라
  # 기존 --fitted-params-json에서 그냥 읽어온 경우)에는 output_report를 다시
  # 쓰지 않는다. 예전에는 여기서 항상 write_report를 호출했는데, 그러면
  # cfg.damping_test(이 compare 실행에서는 보통 기본값 False)로 리포트가
  # 덮어써져서 "damping_test": true 마커가 지워졌다 — 그 결과 이후 chirp
  # fit이 자동 계층적 식별(damping 자동 고정)을 못 하게 되는 버그로
  # 이어졌다(실측으로 확인: 디버깅용으로 --stage compare만 여러 번 돌린
  # 뒤 chirp fit을 했더니 damping이 고정 안 되고 자유롭게 흘러가버림).
  if fitted is not None and did_fit:
    write_report(cfg, csv_path, fitted, cost, compare_rmse)
  elif fitted is not None:
    print(
      "[main] --stage compare만 단독 실행 — 기존 리포트를 덮어쓰지 않습니다 "
      "(damping_test 마커 등 기존 리포트의 출처 정보를 보존)."
    )


if __name__ == "__main__":
  main()