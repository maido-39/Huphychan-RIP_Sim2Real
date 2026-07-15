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
  --motor-id 8 --channel can0 --encoder-port /dev/ttyACM0 \
  --torque-amp-nm 0.45 --chirp-f1-hz 15

# 이미 모은 로그로 fit + compare만
uv run python sysid_chirp_pipeline.py --stage fit --csv-path logs/sysid_chirp_....csv
uv run python sysid_chirp_pipeline.py --stage compare --csv-path logs/sysid_chirp_....csv \
  --fitted-params-json logs/sysid_report.json

# 1단계: damping-test(스핀업+자유감속 반복)로 damping을 먼저 깨끗하게 분리
uv run python sysid_chirp_pipeline.py --stage full --no-simulate --damping-test \
  --motor-id 8 --channel can0 --encoder-port /dev/ttyACM0

# 2단계: 같은 --output-report를 그대로 두고 chirp 실행 — 위에서 구한 damping을
# 자동으로 읽어와 고정하고, armature/frictionloss만 다시 fit한다
uv run python sysid_chirp_pipeline.py --stage full --no-simulate \
  --motor-id 8 --channel can0 --encoder-port /dev/ttyACM0
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


def load_csv(csv_path: str) -> LoggedTrajectory:
  data = np.genfromtxt(csv_path, delimiter=",", names=True)
  dt = np.diff(data["time_s"])
  if dt.size and (np.max(dt) - np.min(dt)) > 0.2 * np.median(dt):
    print(
      f"[warn] sample spacing is not uniform (min={dt.min():.4f}s, "
      f"max={dt.max():.4f}s) — replay/comparison may be biased."
    )
  return LoggedTrajectory(
    time_s=data["time_s"],
    cmd_torque_nm=data["cmd_torque_nm"],
    angle_rad=np.radians(data["motor_angle_deg"]),
    vel_rad_s=np.radians(data["motor_vel_deg_s"]),
    torque_nm=data["motor_torque_nm"],
  )


def _new_csv_writer(log_dir: str, tag: str) -> tuple[str, "csv._writer", object]:
  os.makedirs(log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join(log_dir, f"sysid_chirp_{tag}_{ts}.csv")
  f = open(csv_path, "w", newline="")
  writer = csv.writer(f)
  writer.writerow(_CSV_HEADER)
  return csv_path, writer, f


def _plot_log(csv_path: str, title: str) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib이 없어 그래프를 건너뜁니다. (pip install matplotlib)")
    return

  traj = load_csv(csv_path)
  angle_deg = np.degrees(np.unwrap(traj.angle_rad))
  vel_deg_s = np.degrees(traj.vel_rad_s)

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
  axes[2].plot(traj.time_s, traj.torque_nm, color="tab:red", label="motor_torque_nm")
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
    sample_idx = 0
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
  if cfg.plot:
    _plot_log(csv_path, f"hardware {'damping_test' if cfg.damping_test else 'chirp'} torque")
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
    data.qfrc_applied[dof] = tau
    model.opt.timestep = float(dt_arr[i]) if i < len(dt_arr) else float(dt_arr[-1])
    mujoco.mj_step(model, data)
    angle_deg = math.degrees(data.qpos[qadr])
    vel_deg_s = math.degrees(data.qvel[dof])
    writer.writerow(
      [f"{sample_times[i]:.4f}", f"{tau:.4f}", f"{angle_deg:.3f}", f"{vel_deg_s:.3f}", f"{tau:.4f}"]
    )

  f.close()
  print(f"CSV 로그 저장 완료: {csv_path}")
  if cfg.plot:
    _plot_log(csv_path, f"simulated chirp torque ({tag})")
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

  # fix_*가 지정된 파라미터는 lower=upper=그 값으로 만들어 LM이 움직이지
  # 못하게 고정한다 (예: coast-down에서 구한 damping을 여기 chirp fit에서는
  # 상수로 취급하고 armature/frictionloss만 다시 맞추는 계층적 식별).
  fixed = [effective_fix_damping, cfg.fix_armature, cfg.fix_frictionloss]
  for i, fv in enumerate(fixed):
    if fv is not None:
      x0[i] = fv
      lower[i] = fv
      upper[i] = fv

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

  def _residual_single(x: np.ndarray) -> np.ndarray:
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
        data.qfrc_applied[dof] = traj.torque_nm[k]
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

  x_fit, trace = mujoco.minimize.least_squares(  # type: ignore[attr-defined]
    x0, residual, bounds=(lower, upper), x_scale="jac", verbose=0
  )
  cost = float(trace[-1].objective) if trace else float("nan")
  print(f"[fit] Gauss-Newton(LM) final cost = {cost:.6f}")
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
      # fit_gauss_newton과 동일하게 실측 torque_nm을 입력으로 쓴다 (아래
      # compare_real_vs_sim 참고).
      data.qfrc_applied[dof] = real_traj.torque_nm[k]
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
  real_traj = load_csv(real_csv_path)
  sim_csv_path = collect_simulated(
    cfg,
    tag="fitted_compare",
    param_overrides=fitted,
    cmd_torque_arr=real_traj.torque_nm,
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
    axes[0].plot(t, np.degrees(np.unwrap(real_traj.angle_rad[:n])), label="real")
    axes[0].plot(t, np.degrees(np.unwrap(sim_traj.angle_rad[:n])), label="sim(fitted)", linestyle="--")
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

  if cfg.stage in ("collect", "full"):
    csv_path = collect_simulated(cfg, tag="hw_sim") if cfg.simulate else collect_hardware(cfg)

  if cfg.stage in ("fit", "full"):
    if not csv_path:
      raise SystemExit("--csv-path가 필요합니다 (또는 --stage full/collect로 먼저 수집).")
    traj = load_csv(csv_path)
    fitted, cost = fit_gauss_newton(traj, cfg)

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

  if fitted is not None:
    write_report(cfg, csv_path, fitted, cost, compare_rmse)


if __name__ == "__main__":
  main()
