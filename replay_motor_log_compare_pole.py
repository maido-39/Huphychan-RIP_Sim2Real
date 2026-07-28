#!/usr/bin/env python3
#!/usr/bin/env python3
"""실제 로봇에서 기록한 모터 궤적을 MuJoCo에서 재생하고,
실제 폴 각도와 시뮬레이션 폴 각도를 비교하는 스크립트.

목적
----
이 스크립트는 역진자 장치의 sim-to-real 검증과 시스템 식별을 위해 사용한다.

실제 로봇에서 기록한 CSV 로그를 불러온 뒤,
CSV에 저장된 실제 모터 각도를 MuJoCo의 모터 관절에 그대로 재생한다.

이때 모터 관절은 실제 로그와 동일한 궤적을 강제로 따라가고,
폴 관절은 별도의 제어 없이 MuJoCo 물리엔진에 의해 수동으로 움직인다.

그 결과로 계산된 시뮬레이션 폴 각도와
실제 로봇에서 측정된 폴 각도를 비교한다.

이 과정을 통해 다음과 같은 물리 파라미터를 조정할 수 있다.

  - 폴 관절 damping
  - 폴 관절 armature
  - 폴 관절 frictionloss
  - 폴의 질량
  - 폴의 무게중심 위치
  - 폴의 관성모멘트

이 스크립트의 목적은 강화학습 정책 자체를 검증하는 것이 아니다.

실제 정책의 출력 결과는 이미 실제 모터 움직임에 반영되어 있으므로,
이 스크립트에서는 동일한 모터 궤적을 입력했을 때
실제 장치와 시뮬레이터의 기계적 응답이 얼마나 비슷한지 확인한다.

필수 CSV 열
-----------
다음 열은 반드시 포함되어야 한다.

  time_s
      로그 시작 시점으로부터 경과한 시간.
      단위는 초이다.

  motor_angle_deg
      실제 모터에서 측정한 모터 관절 각도.
      단위는 degree이다.

      이 값이 MuJoCo 모터 관절의 위치로 입력된다.

  pole_angle_deg
      실제 로봇에서 측정한 폴 각도.
      단위는 degree이다.

      이 값과 시뮬레이션에서 계산된 폴 각도를 비교한다.

선택 CSV 열
-----------
다음 열은 없어도 실행할 수 있다.

  pole_vel_deg_s
      실제 폴의 각속도.
      단위는 degree/s이다.

      이 열이 존재하면 첫 번째 값을
      시뮬레이션 폴의 초기 각속도로 사용한다.

시뮬레이션 동작 방식
--------------------
각 MuJoCo simulation step마다 다음 과정을 반복한다.

  1. 현재 시뮬레이션 시간에 맞는 모터 각도를 CSV에서 보간한다.
  2. 모터 각도 데이터로부터 모터 각속도를 계산한다.
  3. MuJoCo 모터 관절의 qpos와 qvel을 해당 값으로 덮어쓴다.
  4. 폴 관절의 위치와 속도는 강제로 수정하지 않는다.
  5. MuJoCo가 수동 폴의 동역학을 계산한다.
  6. 시뮬레이션 폴 각도와 각속도를 기록한다.

모터 관절의 실제 actuator는 replay 중 비활성화된다.

이는 MuJoCo actuator controller가 CSV에서 강제로 입력하는
모터 궤적과 충돌하는 것을 막기 위한 것이다.

입력 파일 예시
--------------
다음과 같은 실제 로봇 로그를 사용할 수 있다.

  /home/aril/Pygmalion/mjlab/logs/policy_run_20260727_141219.csv

라즈베리파이에서 학습 PC로 로그 복사
-----------------------------------
다음 명령은 학습 PC에서 실행한다.

먼저 입력 로그 폴더를 만든다.

  mkdir -p /home/aril/mjlab/src/mjlab/tasks/inverse/input_log

그다음 라즈베리파이에서 CSV를 가져온다.

  scp \
    aril@라즈베리파이_IP:/home/aril/Pygmalion/mjlab/logs/policy_run_20260727_141219.csv \
    /home/aril/mjlab/src/mjlab/tasks/inverse/input_log/

예를 들어 라즈베리파이 IP가 192.168.0.50이면 다음과 같다.

  scp \
    aril@192.168.0.18:/home/aril/Pygmalion/mjlab/logs/policy_run_20260727_141219.csv \
    /home/aril/mjlab/src/mjlab/tasks/inverse/input_log/

권장 폴더 구조
--------------
  inverse/
    assets/
      inverse.xml
    input_log/
      policy_run_20260727_141219.csv
    replay_results/
    replay_motor_log_compare_pole.py

출력 파일
---------
실행이 끝나면 replay_results 폴더에 다음 파일이 생성된다.

  <입력파일이름>_real_vs_sim.csv
      기존 실제 로그에 시뮬레이션 결과를 추가한 CSV 파일.

  <입력파일이름>_real_vs_sim.png
      실제값과 시뮬레이션값을 비교한 그래프 이미지.

그래프에는 다음 내용이 포함된다.

  1. 실제 모터 각도와 시뮬레이터에 입력된 모터 각도
  2. 실제 폴 각도와 시뮬레이션 폴 각도
  3. 실제 폴과 시뮬레이션 폴의 circular angle error

터미널에는 다음 값도 출력된다.

  - 폴 각도 RMSE
  - 폴 각도 MAE
  - 사용된 damping 값
  - 사용된 armature 값
  - 사용된 frictionloss 값

기본 실행 방법
--------------
input_log 폴더에서 가장 최근 CSV를 자동 선택한다.

  uv run python replay_motor_log_compare_pole.py

특정 CSV를 지정한다.

  uv run python replay_motor_log_compare_pole.py \
    --csv input_log/policy_run_20260727_141219.csv

MuJoCo viewer를 열어 실제 움직임을 확인한다.

  uv run python replay_motor_log_compare_pole.py \
    --csv input_log/policy_run_20260727_141219.csv \
    --viewer

절반 속도로 재생한다.

  uv run python replay_motor_log_compare_pole.py \
    --csv input_log/policy_run_20260727_141219.csv \
    --viewer \
    --playback-speed 0.5

폴 관절 파라미터를 임시로 변경한다.

  uv run python replay_motor_log_compare_pole.py \
    --csv input_log/policy_run_20260727_141219.csv \
    --pole-damping 0.00003 \
    --pole-armature 0.00001 \
    --pole-frictionloss 0.000002

각도 방향과 영점 보정
---------------------
실제 모터와 시뮬레이터의 회전 방향이 반대인 경우:

  --motor-angle-sign -1

실제 폴과 시뮬레이터의 회전 방향이 반대인 경우:

  --pole-angle-sign -1

실제 폴과 시뮬레이터의 영점이 180도 차이 나는 경우:

  --pole-angle-offset-deg 180

주의사항
--------
한 개의 로그만 잘 맞는다고 해서
시뮬레이터가 실제 장치를 완벽하게 재현한다고 볼 수는 없다.

서로 다른 조건에서 얻은 여러 로그에서도
동일한 물리 파라미터로 비슷한 결과가 나와야 한다.

검증에 사용할 수 있는 조건은 다음과 같다.

  - 서로 다른 초기 폴 각도
  - 작은 모터 진동
  - 큰 스윙 동작
  - 저속 운동
  - 고속 운동
  - 정방향 회전
  - 역방향 회전
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from contextlib import nullcontext

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd


DEFAULT_MOTOR_JOINT = "Revolute 3"
DEFAULT_POLE_JOINT = "Revolute 5"


@dataclass(frozen=True)
class JointAddress:
  joint_id: int
  qpos_adr: int
  dof_adr: int


def parse_args() -> argparse.Namespace:
  script_dir = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(
    description=(
      "Replay the measured motor angle in MuJoCo and compare the resulting "
      "simulated pole angle with the measured real pole angle."
    )
  )
  parser.add_argument(
    "--csv",
    type=Path,
    default=None,
    help=(
      "Input CSV path. When omitted, the newest *.csv file in "
      "<script_dir>/input_log is selected."
    ),
  )
  parser.add_argument(
    "--xml",
    type=Path,
    default=script_dir / "assets" / "inverse.xml",
    help="MuJoCo XML path.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=script_dir / "replay_results",
    help="Directory for the augmented CSV and comparison plot.",
  )
  parser.add_argument("--motor-joint", default=DEFAULT_MOTOR_JOINT)
  parser.add_argument("--pole-joint", default=DEFAULT_POLE_JOINT)

  parser.add_argument(
    "--pole-damping",
    type=float,
    default=None,
    help="Override pendulum-joint damping in N*m*s/rad.",
  )
  parser.add_argument(
    "--pole-armature",
    type=float,
    default=None,
    help="Override pendulum-joint armature in kg*m^2.",
  )
  parser.add_argument(
    "--pole-frictionloss",
    type=float,
    default=None,
    help="Override pendulum-joint Coulomb frictionloss in N*m.",
  )

  parser.add_argument(
    "--motor-angle-sign",
    type=float,
    choices=(-1.0, 1.0),
    default=1.0,
    help="Multiply logged motor angles by this sign before replay.",
  )
  parser.add_argument(
    "--motor-angle-offset-deg",
    type=float,
    default=0.0,
    help="Add this offset to logged motor angles before replay.",
  )
  parser.add_argument(
    "--pole-angle-sign",
    type=float,
    choices=(-1.0, 1.0),
    default=1.0,
    help="Multiply logged real pole angles by this sign for comparison.",
  )
  parser.add_argument(
    "--pole-angle-offset-deg",
    type=float,
    default=0.0,
    help="Add this offset to logged real pole angles for comparison.",
  )
  parser.add_argument(
    "--start-time",
    type=float,
    default=None,
    help="Optional start time in the CSV time coordinate.",
  )
  parser.add_argument(
    "--end-time",
    type=float,
    default=None,
    help="Optional end time in the CSV time coordinate.",
  )
  parser.add_argument(
    "--viewer",
    action="store_true",
    help="Open the MuJoCo viewer and replay the CSV motion in real time.",
  )
  parser.add_argument(
    "--playback-speed",
    type=float,
    default=1.0,
    help=(
      "Viewer playback speed. 1.0 is real time, 0.5 is half speed, "
      "and 2.0 is double speed."
    ),
  )
  parser.add_argument(
    "--loop-viewer",
    action="store_true",
    help="Keep replaying the CSV until the MuJoCo viewer is closed.",
  )
  parser.add_argument(
    "--show",
    action="store_true",
    help="Show the comparison plot after saving it.",
  )
  return parser.parse_args()


def newest_csv(input_dir: Path) -> Path:
  files = sorted(
    input_dir.glob("*.csv"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
  )
  if not files:
    raise FileNotFoundError(f"No CSV files found in: {input_dir}")
  return files[0]


def resolve_path(path: Path, script_dir: Path) -> Path:
  if path.is_absolute():
    return path
  candidate = (Path.cwd() / path).resolve()
  if candidate.exists():
    return candidate
  return (script_dir / path).resolve()


def get_joint_address(model: mujoco.MjModel, name: str) -> JointAddress:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if joint_id < 0:
    available = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
      for idx in range(model.njnt)
    ]
    raise ValueError(f"Joint {name!r} not found. Available joints: {available}")

  joint_type = int(model.jnt_type[joint_id])
  if joint_type not in (
    int(mujoco.mjtJoint.mjJNT_HINGE),
    int(mujoco.mjtJoint.mjJNT_SLIDE),
  ):
    raise ValueError(
      f"Joint {name!r} must have one qpos and one dof; got type={joint_type}."
    )

  return JointAddress(
    joint_id=joint_id,
    qpos_adr=int(model.jnt_qposadr[joint_id]),
    dof_adr=int(model.jnt_dofadr[joint_id]),
  )


def validate_and_prepare_log(
  csv_path: Path,
  start_time: float | None,
  end_time: float | None,
) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  required = {"time_s", "motor_angle_deg", "pole_angle_deg"}
  missing = sorted(required.difference(df.columns))
  if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

  df = df.copy()
  for column in required:
    df[column] = pd.to_numeric(df[column], errors="coerce")

  df = df.dropna(subset=list(required))
  df = df.sort_values("time_s")
  df = df.drop_duplicates(subset="time_s", keep="last")

  if start_time is not None:
    df = df[df["time_s"] >= start_time]
  if end_time is not None:
    df = df[df["time_s"] <= end_time]

  if len(df) < 3:
    raise ValueError("At least three valid time samples are required.")

  time = df["time_s"].to_numpy(dtype=float)
  if np.any(np.diff(time) <= 0.0):
    raise ValueError("time_s must be strictly increasing after cleanup.")

  # Start the simulation clock at zero while preserving the original time.
  df["source_time_s"] = df["time_s"]
  df["time_s"] = df["time_s"] - float(df["time_s"].iloc[0])
  return df.reset_index(drop=True)


def unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
  return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def wrap_deg_360(angle_deg: np.ndarray) -> np.ndarray:
  return np.mod(angle_deg, 360.0)


def circular_error_deg(sim_deg: np.ndarray, real_deg: np.ndarray) -> np.ndarray:
  return (sim_deg - real_deg + 180.0) % 360.0 - 180.0


def set_passive_joint_parameters(
  model: mujoco.MjModel,
  pole: JointAddress,
  damping: float | None,
  armature: float | None,
  frictionloss: float | None,
) -> None:
  if damping is not None:
    if damping < 0.0:
      raise ValueError("--pole-damping must be nonnegative.")
    model.dof_damping[pole.dof_adr] = damping

  if armature is not None:
    if armature < 0.0:
      raise ValueError("--pole-armature must be nonnegative.")
    model.dof_armature[pole.dof_adr] = armature

  if frictionloss is not None:
    if frictionloss < 0.0:
      raise ValueError("--pole-frictionloss must be nonnegative.")
    model.dof_frictionloss[pole.dof_adr] = frictionloss


def disable_motor_actuators(model: mujoco.MjModel, motor: JointAddress) -> None:
  """Disable actuators driving the prescribed motor joint.

  The motor qpos/qvel are overwritten from the CSV at every simulation step,
  so actuator forces on this joint must not fight the prescribed trajectory.
  """
  for actuator_id in range(model.nu):
    transmission_type = int(model.actuator_trntype[actuator_id])
    if transmission_type != int(mujoco.mjtTrn.mjTRN_JOINT):
      continue
    if int(model.actuator_trnid[actuator_id, 0]) != motor.joint_id:
      continue

    model.actuator_gainprm[actuator_id, :] = 0.0
    model.actuator_biasprm[actuator_id, :] = 0.0
    model.actuator_forcerange[actuator_id, :] = 0.0


def replay(
  model: mujoco.MjModel,
  df: pd.DataFrame,
  motor: JointAddress,
  pole: JointAddress,
  motor_angle_sign: float,
  motor_angle_offset_deg: float,
  pole_angle_sign: float,
  pole_angle_offset_deg: float,
  viewer_enabled: bool = False,
  playback_speed: float = 1.0,
  loop_viewer: bool = False,
) -> pd.DataFrame:
  data = mujoco.MjData(model)

  log_time = df["time_s"].to_numpy(dtype=float)

  motor_deg_raw = df["motor_angle_deg"].to_numpy(dtype=float)
  motor_deg = motor_angle_sign * motor_deg_raw + motor_angle_offset_deg
  motor_rad_unwrapped = np.unwrap(np.deg2rad(motor_deg))

  # Derive a continuous motor velocity from the measured position trajectory.
  motor_vel_rad_s = np.gradient(motor_rad_unwrapped, log_time)

  real_pole_deg_raw = df["pole_angle_deg"].to_numpy(dtype=float)
  real_pole_deg = pole_angle_sign * real_pole_deg_raw + pole_angle_offset_deg
  real_pole_rad_unwrapped = np.unwrap(np.deg2rad(real_pole_deg))

  # Start from the measured initial state so the comparison isolates dynamics.
  data.qpos[motor.qpos_adr] = motor_rad_unwrapped[0]
  data.qvel[motor.dof_adr] = motor_vel_rad_s[0]
  data.qpos[pole.qpos_adr] = real_pole_rad_unwrapped[0]

  if "pole_vel_deg_s" in df.columns:
    first_real_pole_vel = pd.to_numeric(df["pole_vel_deg_s"], errors="coerce").iloc[0]
    if np.isfinite(first_real_pole_vel):
      data.qvel[pole.dof_adr] = np.deg2rad(pole_angle_sign * float(first_real_pole_vel))
    else:
      data.qvel[pole.dof_adr] = np.gradient(real_pole_rad_unwrapped, log_time)[0]
  else:
    data.qvel[pole.dof_adr] = np.gradient(real_pole_rad_unwrapped, log_time)[0]

  if playback_speed <= 0.0:
    raise ValueError("--playback-speed must be greater than zero.")

  initial_qpos = data.qpos.copy()
  initial_qvel = data.qvel.copy()

  end_time = float(log_time[-1])
  dt = float(model.opt.timestep)
  if dt <= 0.0:
    raise ValueError(f"Invalid MuJoCo timestep: {dt}")

  sim_times: list[float] = []
  sim_pole_rad: list[float] = []
  sim_pole_vel_rad_s: list[float] = []
  imposed_motor_rad: list[float] = []

  viewer_context = (
    mujoco.viewer.launch_passive(model, data) if viewer_enabled else nullcontext(None)
  )

  with viewer_context as viewer:
    first_pass = True

    while True:
      data.qpos[:] = initial_qpos
      data.qvel[:] = initial_qvel
      data.time = 0.0
      if model.nu:
        data.ctrl[:] = 0.0
      data.qfrc_applied[:] = 0.0
      data.xfrc_applied[:] = 0.0
      mujoco.mj_forward(model, data)

      if first_pass:
        sim_times.append(0.0)
        sim_pole_rad.append(float(data.qpos[pole.qpos_adr]))
        sim_pole_vel_rad_s.append(float(data.qvel[pole.dof_adr]))
        imposed_motor_rad.append(float(data.qpos[motor.qpos_adr]))

      wall_start = time.perf_counter()

      # Prescribe the motor state at every MuJoCo step. The pole is never
      # overwritten and evolves from MuJoCo dynamics.
      while data.time < end_time - 0.5 * dt:
        if viewer is not None and not viewer.is_running():
          break

        current_time = float(data.time)
        motor_q = float(np.interp(current_time, log_time, motor_rad_unwrapped))
        motor_qd = float(np.interp(current_time, log_time, motor_vel_rad_s))

        data.qpos[motor.qpos_adr] = motor_q
        data.qvel[motor.dof_adr] = motor_qd

        if model.nu:
          data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        next_time = min(float(data.time), end_time)
        next_motor_q = float(np.interp(next_time, log_time, motor_rad_unwrapped))
        next_motor_qd = float(np.interp(next_time, log_time, motor_vel_rad_s))
        data.qpos[motor.qpos_adr] = next_motor_q
        data.qvel[motor.dof_adr] = next_motor_qd
        mujoco.mj_forward(model, data)

        if first_pass:
          sim_times.append(next_time)
          sim_pole_rad.append(float(data.qpos[pole.qpos_adr]))
          sim_pole_vel_rad_s.append(float(data.qvel[pole.dof_adr]))
          imposed_motor_rad.append(next_motor_q)

        if viewer is not None:
          viewer.sync()

          target_wall_elapsed = next_time / playback_speed
          remaining = target_wall_elapsed - (time.perf_counter() - wall_start)
          if remaining > 0.0:
            time.sleep(remaining)

      first_pass = False

      if viewer is None:
        break
      if not viewer.is_running():
        break
      if not loop_viewer:
        # Leave the final pose visible until the window is closed.
        while viewer.is_running():
          viewer.sync()
          time.sleep(0.02)
        break

  sim_times_arr = np.asarray(sim_times)
  sim_pole_unwrapped_deg = np.rad2deg(np.asarray(sim_pole_rad))
  sim_pole_vel_deg_s = np.rad2deg(np.asarray(sim_pole_vel_rad_s))
  imposed_motor_deg = np.rad2deg(np.asarray(imposed_motor_rad))

  # Resample simulation output onto the original CSV timestamps.
  output = df.copy()
  output["replay_motor_angle_deg"] = np.interp(
    log_time, sim_times_arr, imposed_motor_deg
  )
  output["real_pole_angle_unwrapped_deg"] = real_pole_rad_unwrapped * 180.0 / np.pi
  output["sim_pole_angle_unwrapped_deg"] = np.interp(
    log_time, sim_times_arr, sim_pole_unwrapped_deg
  )
  output["real_pole_angle_wrapped_deg"] = wrap_deg_360(
    output["real_pole_angle_unwrapped_deg"].to_numpy()
  )
  output["sim_pole_angle_wrapped_deg"] = wrap_deg_360(
    output["sim_pole_angle_unwrapped_deg"].to_numpy()
  )
  output["sim_pole_vel_deg_s"] = np.interp(log_time, sim_times_arr, sim_pole_vel_deg_s)
  output["pole_angle_error_deg"] = circular_error_deg(
    output["sim_pole_angle_wrapped_deg"].to_numpy(),
    output["real_pole_angle_wrapped_deg"].to_numpy(),
  )
  return output


def save_plot(output: pd.DataFrame, plot_path: Path) -> None:
  time = output["time_s"].to_numpy(dtype=float)

  fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

  axes[0].plot(
    time,
    output["motor_angle_deg"],
    label="Real motor angle from CSV",
    linewidth=1.3,
  )
  axes[0].plot(
    time,
    output["replay_motor_angle_deg"],
    label="Motor angle imposed in simulation",
    linewidth=1.0,
    linestyle="--",
  )
  axes[0].set_ylabel("Motor angle [deg]")
  axes[0].grid(True, alpha=0.3)
  axes[0].legend()

  axes[1].plot(
    time,
    output["real_pole_angle_unwrapped_deg"],
    label="Real pole angle",
    linewidth=1.3,
  )
  axes[1].plot(
    time,
    output["sim_pole_angle_unwrapped_deg"],
    label="Simulated pole angle",
    linewidth=1.3,
  )
  axes[1].set_ylabel("Pole angle [deg, unwrapped]")
  axes[1].grid(True, alpha=0.3)
  axes[1].legend()

  axes[2].plot(
    time,
    output["pole_angle_error_deg"],
    label="Circular angle error: sim - real",
    linewidth=1.0,
  )
  axes[2].axhline(0.0, linewidth=0.8)
  axes[2].set_xlabel("Time [s]")
  axes[2].set_ylabel("Error [deg]")
  axes[2].grid(True, alpha=0.3)
  axes[2].legend()

  fig.tight_layout()
  fig.savefig(plot_path, dpi=160)
  plt.close(fig)


def main() -> int:
  args = parse_args()
  script_dir = Path(__file__).resolve().parent

  csv_path = (
    newest_csv(script_dir / "input_log")
    if args.csv is None
    else resolve_path(args.csv, script_dir)
  )
  xml_path = resolve_path(args.xml, script_dir)
  output_dir = resolve_path(args.output_dir, script_dir)

  if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")
  if not xml_path.exists():
    raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

  output_dir.mkdir(parents=True, exist_ok=True)

  df = validate_and_prepare_log(csv_path, args.start_time, args.end_time)

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  motor = get_joint_address(model, args.motor_joint)
  pole = get_joint_address(model, args.pole_joint)

  set_passive_joint_parameters(
    model,
    pole,
    damping=args.pole_damping,
    armature=args.pole_armature,
    frictionloss=args.pole_frictionloss,
  )
  disable_motor_actuators(model, motor)

  output = replay(
    model=model,
    df=df,
    motor=motor,
    pole=pole,
    motor_angle_sign=args.motor_angle_sign,
    motor_angle_offset_deg=args.motor_angle_offset_deg,
    pole_angle_sign=args.pole_angle_sign,
    pole_angle_offset_deg=args.pole_angle_offset_deg,
    viewer_enabled=args.viewer,
    playback_speed=args.playback_speed,
    loop_viewer=args.loop_viewer,
  )

  stem = csv_path.stem
  output_csv = output_dir / f"{stem}_real_vs_sim.csv"
  output_plot = output_dir / f"{stem}_real_vs_sim.png"

  output.to_csv(output_csv, index=False)
  save_plot(output, output_plot)

  error = output["pole_angle_error_deg"].to_numpy(dtype=float)
  rmse = float(np.sqrt(np.mean(error**2)))
  mae = float(np.mean(np.abs(error)))

  print(f"Input CSV       : {csv_path}")
  print(f"MuJoCo XML      : {xml_path}")
  print(f"Output CSV      : {output_csv}")
  print(f"Output plot     : {output_plot}")
  print(f"Pole angle RMSE : {rmse:.3f} deg")
  print(f"Pole angle MAE  : {mae:.3f} deg")
  print(
    "Pole parameters : "
    f"damping={model.dof_damping[pole.dof_adr]:.8g}, "
    f"armature={model.dof_armature[pole.dof_adr]:.8g}, "
    f"frictionloss={model.dof_frictionloss[pole.dof_adr]:.8g}"
  )

  if args.show:
    image = plt.imread(output_plot)
    plt.figure(figsize=(13, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except Exception as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    raise
