#!/usr/bin/env python3
from __future__ import annotations

"""액추에이터 SYSID 통합 파이프라인 — 실험 두 가지를 하나의 코드로.

  --test 1 : 등속도 계단(velocity-step) TEST — damping / frictionloss 측정.
             여러 목표 각속도 레벨을 kd 폐루프로 유지하고, 정상상태(qddot=0)
             구간의 (v, tau) 점들로 tau=damping*v+frictionloss*sign(v)를
             선형 최소자승으로 푼다. 가속도 추정(미분)이 필요 없어 가장
             깨끗하게 damping/frictionloss를 분리해낸다.

  --test 2 : 위치기반 chirp TEST — armature 측정. damping/frictionloss를
             --fix-damping/--fix-frictionloss로 test 1 결과에 고정하고, 목표
             위치를 f0->f1 Hz로 스윕하는 사인파를 kp/kd 폐루프로 추종시켜
             (진폭이 갇혀 있어 안전) armature를 shooting(output-error) 방식
             (Gauss-Newton/LM, `mujoco.minimize.least_squares`)으로 구한다.

둘 다 `--n-repeats`로 반복 측정해서 재현성(값이 실행마다 얼마나 일치하는지)을
직접 확인할 수 있다. test 1은 한 번의 하드웨어 연결 안에서 레벨 사이클을
n_repeats번 반복하고, test 2는 chirp 여기 자체를 n_repeats번 독립적으로
반복 수집·fit해서 armature가 실행마다 일관되는지 표로 비교한다.

리포트는 `--output-report`(기본 logs/sysid_report.json) 하나를 공유하되,
test 1/2가 서로 다른 최상위 키("velocity_step" / "chirp_armature")에 쓰기
때문에 서로 덮어쓰지 않는다.

사용법:
```bash
# 1) 시뮬레이션으로 둘 다 점검
uv run python sysid.py --test 1 --simulate
uv run python sysid.py --test 2 --simulate --fix-damping 0.007 --fix-frictionloss 0.14

# 2) 실물: damping/frictionloss부터 (n_repeats로 재현성 확인)
uv run python sysid.py --test 1 --no-simulate --n-repeats 3 \
  --motor-id 8 --channel can0

# 3) 실물: test 1에서 구한 값을 고정하고 armature (n_repeats로 재현성 확인)
uv run python sysid.py --test 2 --no-simulate --n-repeats 3 \
  --fix-damping 0.007073 --fix-frictionloss 0.142747 \
  --motor-id 8 --channel can0
```
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro
from prettytable import PrettyTable

from sysid_utils import (
  ChirpSysidConfig,
  LoggedTrajectory,
  _new_csv_writer,
  _plot_log,
  fit_gauss_newton,
  load_csv,
)

_DEFAULT_XML = str(ChirpSysidConfig().xml_path)


@dataclass(frozen=True)
class SysidConfig:
  test: Literal[1, 2]
  stage: Literal["collect", "fit", "full"] = "full"

  # --- 대상 관절 ---
  joint_name: str = "Revolute 3"
  actuator_name: str = "position_revolute_3"
  xml_path: str = _DEFAULT_XML

  # --- 재현성 ---
  # test 1: 레벨 사이클을 이 횟수만큼 반복(한 번의 하드웨어 세션 안에서).
  # test 2: chirp 여기 자체를 이 횟수만큼 독립적으로 반복 수집·fit.
  n_repeats: int = 3

  # --- test 1: 등속도 계단 여기 신호 ---
  vel_levels_deg_s: tuple[float, ...] = (
    -800.0, -600.0, -400.0, -200.0, 200.0, 400.0, 600.0, 800.0,
  )
  level_duration_s: float = 6.0
  steady_fraction: float = 0.5  # 각 구간 뒤쪽 이 비율만 정상상태로 평균
  velocity_kd: float = 1.0
  damping_max: float = 5.0
  frictionloss_max: float = 0.5

  # --- test 2: 위치기반 chirp 여기 신호 ---
  amplitude_deg: float = 30.0
  chirp_f0_hz: float = 0.5
  chirp_f1_hz: float = 5.0
  chirp_duration_s: float = 30.0
  chirp_scale: Literal["log", "linear"] = "log"
  ramp_in_s: float = 1.0
  velocity_feedforward: bool = True
  kp: float = 20.0
  kd: float = 3.0
  fix_damping: float | None = None
  fix_frictionloss: float | None = None
  shooting_window_s: float = 0.05
  torque_smooth_samples: int = 9
  armature_max: float = 0.05

  # --- 공통 제어/하드웨어 ---
  control_hz: float = 200.0
  simulate: bool = True
  motor_id: int = 8
  interface: str = "socketcan"
  channel: str = "can0"
  encoder_port: str | None = None
  encoder_baud: int = 115200
  state_read_hz: float = 200.0
  require_enable_ack: bool = True

  # --- 안전 한계 ---
  abort_on_torque_nm: float = 10.0
  velocity_limit_deg_s: float = 0.0  # 0이면 자동 산정
  drift_limit_deg: float = float("inf")
  max_runtime_s: float = 180.0

  # --- fit-only 용 ---
  csv_path: str = ""  # 단일 fit
  csv_paths: tuple[str, ...] = ()  # n_repeats>1일 때 --stage fit에서 사용

  # --- 출력 ---
  log_dir: str = "logs"
  output_report: str = "logs/sysid_report.json"
  plot: bool = True


def _plot_cfg(cfg: SysidConfig) -> ChirpSysidConfig:
  return ChirpSysidConfig(
    xml_path=cfg.xml_path,
    joint_name=cfg.joint_name,
    actuator_name=cfg.actuator_name,
    torque_smooth_samples=cfg.torque_smooth_samples,
  )


def _nominal_params(cfg: SysidConfig) -> dict[str, float]:
  model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  dof = model.joint(cfg.joint_name).dofadr[0]
  return {
    "cylinder_damping": float(model.dof_damping[dof]),
    "cylinder_armature": float(model.dof_armature[dof]),
    "cylinder_frictionloss": float(model.dof_frictionloss[dof]),
  }


def _update_report(cfg: SysidConfig, key: str, payload: dict) -> None:
  out_path = Path(cfg.output_report)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  report = {}
  if out_path.exists():
    try:
      report = json.loads(out_path.read_text())
    except json.JSONDecodeError:
      report = {}
  report[key] = payload
  out_path.write_text(json.dumps(report, indent=2))
  print(f"\nReport updated: {out_path} (key='{key}')")


# ==========================================================================
# test 1: 등속도 계단 (damping / frictionloss)
# ==========================================================================


def _velstep_schedule(cfg: SysidConfig, sample_times: np.ndarray) -> np.ndarray:
  levels = np.tile(np.array(cfg.vel_levels_deg_s, dtype=float), cfg.n_repeats)
  level_idx = np.clip((sample_times / cfg.level_duration_s).astype(int), 0, len(levels) - 1)
  return levels[level_idx]


def _velstep_duration_s(cfg: SysidConfig) -> float:
  return len(cfg.vel_levels_deg_s) * cfg.n_repeats * cfg.level_duration_s


def _velstep_velocity_limit(cfg: SysidConfig) -> float:
  if cfg.velocity_limit_deg_s > 0:
    return cfg.velocity_limit_deg_s
  return max(abs(v) for v in cfg.vel_levels_deg_s) * 1.5


def collect_velstep_hardware(cfg: SysidConfig) -> str:
  from mjlab.tasks.inverse.commission_motor import (
    _open_bus,
    _shutdown_bus,
    disable_mit,
    enable_mit,
    mit_position_command,
  )
  from mjlab.tasks.inverse.robot_state_reader import RobotStateReader

  duration_s = _velstep_duration_s(cfg)
  n_samples = int(duration_s * cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / cfg.control_hz
  vel_cmd_deg_s = _velstep_schedule(cfg, sample_times)
  vel_limit = _velstep_velocity_limit(cfg)

  bus = _open_bus(cfg.interface, cfg.channel)
  reader = RobotStateReader(
    motor_id=cfg.motor_id, encoder_port=cfg.encoder_port, can_interface=cfg.interface,
    can_channel=cfg.channel, motor_mode="passive", motor_rate_hz=cfg.state_read_hz,
    encoder_baud=cfg.encoder_baud,
  )
  csv_path, writer, f = _new_csv_writer(cfg.log_dir, "velstep_hw")
  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  print(
    f"등속도 계단: {len(cfg.vel_levels_deg_s)}레벨 x {cfg.n_repeats}회 반복, "
    f"레벨당 {cfg.level_duration_s}s (총 {duration_s:.1f}s), kd={cfg.velocity_kd}"
  )
  print(f"CSV 로그: {csv_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(여기 신호 완료)"
  initial_angle_deg: float | None = None
  sample_idx = 0

  try:
    reader.start()
    print("Enabling motor...")
    if cfg.require_enable_ack and not enable_mit(bus, cfg.motor_id):
      raise RuntimeError("enable_mit was not acknowledged")
    time.sleep(0.1)

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
    print("Starting velocity-step closed-loop control.")
    while time.monotonic() < deadline and sample_idx < n_samples:
      loop_start = time.monotonic()
      ok = mit_position_command(
        bus, cfg.motor_id, 0.0, kp=0.0, kd=cfg.velocity_kd,
        velocity_deg_s=float(vel_cmd_deg_s[sample_idx]), torque_nm=0.0,
      )
      if not ok:
        raise RuntimeError("No MIT reply received from motor")

      state = reader.get_state()
      angle_deg, vel_deg_s, torque_nm = state.motor_angle_deg, state.motor_velocity_deg_s, state.motor_torque_nm
      if angle_deg is None or vel_deg_s is None or torque_nm is None:
        raise RuntimeError("Motor state unavailable from RobotStateReader")
      if abs(torque_nm) > cfg.abort_on_torque_nm:
        raise RuntimeError(f"Aborting: |torque|={abs(torque_nm):.2f} > abort_on_torque_nm={cfg.abort_on_torque_nm}")
      if abs(angle_deg - initial_angle_deg) > cfg.drift_limit_deg:
        raise RuntimeError(f"Aborting: drift exceeded drift_limit_deg={cfg.drift_limit_deg}")
      if abs(vel_deg_s) > vel_limit:
        raise RuntimeError(f"Aborting: |vel|={abs(vel_deg_s):.1f} > velocity_limit_deg_s={vel_limit:.1f}")

      elapsed_t = time.monotonic() - start_t
      writer.writerow([f"{elapsed_t:.4f}", f"{vel_cmd_deg_s[sample_idx]:.4f}", f"{angle_deg:.3f}", f"{vel_deg_s:.3f}", f"{torque_nm:.4f}"])
      if sample_idx % 20 == 0:
        print(f"t={elapsed_t:6.2f}s cmd_vel={vel_cmd_deg_s[sample_idx]:+7.1f}deg/s vel={vel_deg_s:+7.1f}deg/s torque={torque_nm:+5.2f}Nm")

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
    _plot_velstep_log(csv_path, "hardware velocity-step")
  elif cfg.plot:
    print("[plot] 수집된 샘플이 없어 그래프를 건너뜁니다.")
  return csv_path


def collect_velstep_simulated(cfg: SysidConfig, tag: str = "velstep_sim") -> str:
  model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  joint = model.joint(cfg.joint_name)
  dof, qadr = joint.dofadr[0], joint.qposadr[0]
  act_id = model.actuator(cfg.actuator_name).id
  model.actuator_gainprm[act_id, 0] = 0.0
  model.actuator_biasprm[act_id, 1] = 0.0

  duration_s = _velstep_duration_s(cfg)
  n_samples = int(duration_s * cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / cfg.control_hz
  vel_cmd_deg_s = _velstep_schedule(cfg, sample_times)

  model.opt.timestep = 1.0 / cfg.control_hz
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  csv_path, writer, f = _new_csv_writer(cfg.log_dir, tag)
  print(f"[simulate:{tag}] {cfg.xml_path} 로 등속도 계단 폐루프 제어 모사")
  for i in range(n_samples):
    angle_deg = math.degrees(data.qpos[qadr])
    vel_deg_s = math.degrees(data.qvel[dof])
    vel_des_rad_s = math.radians(float(vel_cmd_deg_s[i]))
    tau = cfg.velocity_kd * (vel_des_rad_s - data.qvel[dof])
    writer.writerow([f"{sample_times[i]:.4f}", f"{vel_cmd_deg_s[i]:.4f}", f"{angle_deg:.3f}", f"{vel_deg_s:.3f}", f"{tau:.4f}"])
    if i < n_samples - 1:
      data.qfrc_applied[dof] = tau
      mujoco.mj_step(model, data)

  f.close()
  print(f"CSV 로그 저장 완료: {csv_path}")
  if cfg.plot:
    _plot_velstep_log(csv_path, f"simulated velocity-step ({tag})")
  return csv_path


def _plot_velstep_log(csv_path: str, title: str) -> None:
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    return
  data = np.genfromtxt(csv_path, delimiter=",", names=True)
  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  axes[0].plot(data["time_s"], data["motor_angle_deg"], color="tab:blue")
  axes[0].set_ylabel("angle (deg)")
  axes[0].set_title(title)
  axes[0].grid(True)
  axes[1].plot(data["time_s"], data["cmd_torque_nm"], color="0.5", linestyle="--", label="cmd_vel_deg_s")
  axes[1].plot(data["time_s"], data["motor_vel_deg_s"], color="tab:orange", label="motor_vel_deg_s")
  axes[1].set_ylabel("velocity (deg/s)")
  axes[1].legend()
  axes[1].grid(True)
  axes[2].plot(data["time_s"], data["motor_torque_nm"], color="tab:red")
  axes[2].set_ylabel("torque (Nm)")
  axes[2].set_xlabel("time (s)")
  axes[2].grid(True)
  fig.tight_layout()
  png_path = csv_path.rsplit(".", 1)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  plt.close(fig)
  print(f"그래프 저장 완료: {png_path}")


def fit_velstep(csv_path: str, cfg: SysidConfig) -> tuple[dict[str, float], float, list[dict]]:
  data = np.genfromtxt(csv_path, delimiter=",", names=True)
  t, cmd_vel, vel_meas, tau_meas = data["time_s"], data["cmd_torque_nm"], data["motor_vel_deg_s"], data["motor_torque_nm"]

  n_levels_total = len(cfg.vel_levels_deg_s) * cfg.n_repeats
  cycle_s = cfg.level_duration_s
  steady_v, steady_tau, level_table = [], [], []
  for k in range(n_levels_total):
    seg_start, seg_end = k * cycle_s, (k + 1) * cycle_s
    mask = (t >= seg_start + cfg.steady_fraction * cycle_s) & (t < seg_end)
    if not np.any(mask):
      continue
    v_mean = float(np.mean(vel_meas[mask]))
    tau_mean = float(np.mean(tau_meas[mask]))
    target = cfg.vel_levels_deg_s[k % len(cfg.vel_levels_deg_s)]
    level_table.append({"target_vel_deg_s": target, "measured_vel_deg_s": v_mean, "measured_torque_nm": tau_mean})
    steady_v.append(math.radians(v_mean))
    steady_tau.append(tau_mean)

  if len(steady_v) < 2:
    raise RuntimeError("정상상태 구간이 2개 미만입니다.")

  v_arr, tau_arr = np.array(steady_v), np.array(steady_tau)
  Y = np.stack([v_arr, np.sign(v_arr)], axis=1)

  def residual(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    single = x.ndim == 1
    if single:
      x = x[:, np.newaxis]
    res = Y @ x - tau_arr[:, np.newaxis]
    return res[:, 0] if single else res

  import mujoco.minimize  # type: ignore[import-not-found]

  x0 = np.array([0.02, 0.05])
  lower, upper = np.array([0.0, 0.0]), np.array([cfg.damping_max, cfg.frictionloss_max])
  x_fit, trace = mujoco.minimize.least_squares(x0, residual, bounds=(lower, upper), x_scale="jac", verbose=0)
  cost = float(trace[-1].objective) if trace else float("nan")
  fitted = {"cylinder_damping": float(x_fit[0]), "cylinder_frictionloss": float(x_fit[1])}
  return fitted, cost, level_table


def run_test1(cfg: SysidConfig) -> None:
  csv_path = cfg.csv_path
  if cfg.stage in ("collect", "full"):
    csv_path = collect_velstep_simulated(cfg) if cfg.simulate else collect_velstep_hardware(cfg)

  if cfg.stage in ("fit", "full"):
    if not csv_path:
      raise SystemExit("--csv-path가 필요합니다.")
    fitted, cost, level_table = fit_velstep(csv_path, cfg)
    nominal = _nominal_params(cfg)
    table = PrettyTable()
    table.field_names = ["parameter", "nominal", "fitted"]
    for name in ("cylinder_damping", "cylinder_frictionloss"):
      table.add_row([name, f"{nominal[name]:.6f}", f"{fitted[name]:.6f}"])
    print(table)
    print(f"[fit] 등속도 계단 선형 fit, cost={cost:.6f}, 레벨 수={len(level_table)}")

    _update_report(
      cfg, "velocity_step",
      {
        "csv_path": csv_path, "method": "constant_velocity_linear_fit", "final_cost": cost,
        "params": {"nominal": nominal, "fitted": fitted}, "levels": level_table,
      },
    )
    print(
      f"\n다음 단계(test 2)에 넘길 값:\n"
      f"  --fix-damping {fitted['cylinder_damping']:.6f} --fix-frictionloss {fitted['cylinder_frictionloss']:.6f}"
    )


# ==========================================================================
# test 2: 위치기반 chirp (armature) — fit은 sysid_pipeline.py의 shooting method 재사용
# ==========================================================================


def _chirp_phase_and_freq(cfg: SysidConfig, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  t = np.clip(t, 0.0, cfg.chirp_duration_s)
  f0, f1, T = cfg.chirp_f0_hz, cfg.chirp_f1_hz, cfg.chirp_duration_s
  if cfg.chirp_scale == "log":
    k = (f1 / f0) ** (1.0 / T)
    phase = 2.0 * math.pi * f0 * (np.power(k, t) - 1.0) / math.log(k)
    inst_freq_hz = f0 * np.power(k, t)
  else:
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) / T * t**2)
    inst_freq_hz = f0 + (f1 - f0) * t / T
  return phase, inst_freq_hz


def build_position_schedule(cfg: SysidConfig, sample_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  phase, inst_freq_hz = _chirp_phase_and_freq(cfg, sample_times)
  ramp = np.clip(sample_times / cfg.ramp_in_s, 0.0, 1.0) if cfg.ramp_in_s > 0 else 1.0
  pos_rel_deg = cfg.amplitude_deg * ramp * np.sin(phase)
  omega_rad_s = 2.0 * math.pi * inst_freq_hz
  vel_rel_deg_s = cfg.amplitude_deg * ramp * omega_rad_s * np.cos(phase)
  return pos_rel_deg, vel_rel_deg_s


def _chirp_velocity_limit(cfg: SysidConfig) -> float:
  if cfg.velocity_limit_deg_s > 0:
    return cfg.velocity_limit_deg_s
  return cfg.amplitude_deg * 2.0 * math.pi * cfg.chirp_f1_hz * 1.5


def collect_chirp_hardware(cfg: SysidConfig, tag_suffix: str = "") -> str:
  from mjlab.tasks.inverse.commission_motor import (
    _open_bus,
    _shutdown_bus,
    disable_mit,
    enable_mit,
    mit_position_command,
  )
  from mjlab.tasks.inverse.robot_state_reader import RobotStateReader

  duration_s = cfg.chirp_duration_s
  n_samples = int(duration_s * cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / cfg.control_hz
  pos_rel_deg, vel_rel_deg_s = build_position_schedule(cfg, sample_times)
  vel_limit = _chirp_velocity_limit(cfg)

  bus = _open_bus(cfg.interface, cfg.channel)
  reader = RobotStateReader(
    motor_id=cfg.motor_id, encoder_port=cfg.encoder_port, can_interface=cfg.interface,
    can_channel=cfg.channel, motor_mode="passive", motor_rate_hz=cfg.state_read_hz,
    encoder_baud=cfg.encoder_baud,
  )
  csv_path, writer, f = _new_csv_writer(cfg.log_dir, f"chirp_hw{tag_suffix}")
  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  print(f"위치 chirp: 진폭={cfg.amplitude_deg}deg, {cfg.chirp_f0_hz}->{cfg.chirp_f1_hz}Hz, {duration_s:.1f}s, kp={cfg.kp}, kd={cfg.kd}")
  print(f"CSV 로그: {csv_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(여기 신호 완료)"
  initial_angle_deg: float | None = None
  sample_idx = 0

  try:
    reader.start()
    print("Enabling motor...")
    if cfg.require_enable_ack and not enable_mit(bus, cfg.motor_id):
      raise RuntimeError("enable_mit was not acknowledged")
    time.sleep(0.1)

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
    print("Starting position-chirp closed-loop control.")
    while time.monotonic() < deadline and sample_idx < n_samples:
      loop_start = time.monotonic()
      target_abs_deg = initial_angle_deg + float(pos_rel_deg[sample_idx])
      vel_ff_deg_s = float(vel_rel_deg_s[sample_idx]) if cfg.velocity_feedforward else 0.0
      ok = mit_position_command(
        bus, cfg.motor_id, target_abs_deg, kp=cfg.kp, kd=cfg.kd,
        velocity_deg_s=vel_ff_deg_s, torque_nm=0.0,
      )
      if not ok:
        raise RuntimeError("No MIT reply received from motor")

      state = reader.get_state()
      angle_deg, vel_deg_s, torque_nm = state.motor_angle_deg, state.motor_velocity_deg_s, state.motor_torque_nm
      if angle_deg is None or vel_deg_s is None or torque_nm is None:
        raise RuntimeError("Motor state unavailable from RobotStateReader")
      if abs(torque_nm) > cfg.abort_on_torque_nm:
        raise RuntimeError(f"Aborting: |torque|={abs(torque_nm):.2f} > abort_on_torque_nm={cfg.abort_on_torque_nm}")
      if abs(angle_deg - initial_angle_deg) > cfg.drift_limit_deg:
        raise RuntimeError(f"Aborting: drift exceeded drift_limit_deg={cfg.drift_limit_deg}")
      if abs(vel_deg_s) > vel_limit:
        raise RuntimeError(f"Aborting: |vel|={abs(vel_deg_s):.1f} > velocity_limit_deg_s={vel_limit:.1f}")

      elapsed_t = time.monotonic() - start_t
      writer.writerow([f"{elapsed_t:.4f}", "0.0000", f"{angle_deg:.3f}", f"{vel_deg_s:.3f}", f"{torque_nm:.4f}"])
      if sample_idx % 20 == 0:
        print(f"t={elapsed_t:6.2f}s target={target_abs_deg:+8.2f}deg angle={angle_deg:+8.2f}deg vel={vel_deg_s:+7.1f}deg/s torque={torque_nm:+5.2f}Nm")

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
    _plot_log(csv_path, "hardware position-chirp", _plot_cfg(cfg))
  elif cfg.plot:
    print("[plot] 수집된 샘플이 없어 그래프를 건너뜁니다.")
  return csv_path


def collect_chirp_simulated(cfg: SysidConfig, tag: str = "chirp_sim") -> str:
  model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  joint = model.joint(cfg.joint_name)
  dof, qadr = joint.dofadr[0], joint.qposadr[0]
  act_id = model.actuator(cfg.actuator_name).id
  model.actuator_gainprm[act_id, 0] = 0.0
  model.actuator_biasprm[act_id, 1] = 0.0

  duration_s = cfg.chirp_duration_s
  n_samples = int(duration_s * cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / cfg.control_hz
  pos_rel_deg, vel_rel_deg_s = build_position_schedule(cfg, sample_times)

  model.opt.timestep = 1.0 / cfg.control_hz
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  initial_angle_deg = math.degrees(float(data.qpos[qadr]))

  csv_path, writer, f = _new_csv_writer(cfg.log_dir, tag)
  print(f"[simulate:{tag}] {cfg.xml_path} 로 위치기반 chirp 폐루프 제어 모사")
  for i in range(n_samples):
    angle_deg = math.degrees(float(data.qpos[qadr]))
    vel_deg_s = math.degrees(float(data.qvel[dof]))
    target_abs_deg = initial_angle_deg + float(pos_rel_deg[i])
    vel_ff_deg_s = float(vel_rel_deg_s[i]) if cfg.velocity_feedforward else 0.0
    pos_err_rad = math.radians(target_abs_deg - angle_deg)
    vel_err_rad_s = math.radians(vel_ff_deg_s) - float(data.qvel[dof])
    tau = cfg.kp * pos_err_rad + cfg.kd * vel_err_rad_s
    writer.writerow([f"{sample_times[i]:.4f}", "0.0000", f"{angle_deg:.3f}", f"{vel_deg_s:.3f}", f"{tau:.4f}"])
    if i < n_samples - 1:
      data.qfrc_applied[dof] = tau
      mujoco.mj_step(model, data)

  f.close()
  print(f"CSV 로그 저장 완료: {csv_path}")
  if cfg.plot:
    _plot_log(csv_path, f"simulated position-chirp ({tag})", _plot_cfg(cfg))
  return csv_path


def fit_chirp_armature(csv_path: str, cfg: SysidConfig) -> tuple[float, float]:
  """sysid_pipeline.py의 shooting(output-error) 방식을 그대로 재사용한다 —
  damping/frictionloss는 test 1 값으로 고정하고 armature만 free로 둔다."""
  if cfg.fix_damping is None or cfg.fix_frictionloss is None:
    raise SystemExit("--fix-damping과 --fix-frictionloss가 필요합니다 (test 1 결과값).")

  pipeline_cfg = ChirpSysidConfig(
    xml_path=cfg.xml_path, joint_name=cfg.joint_name, actuator_name=cfg.actuator_name,
    fix_damping=cfg.fix_damping, fix_frictionloss=cfg.fix_frictionloss,
    armature_max=cfg.armature_max, shooting_window_s=cfg.shooting_window_s,
    torque_smooth_samples=cfg.torque_smooth_samples, torque_input="measured",
  )
  traj: LoggedTrajectory = load_csv(csv_path, torque_smooth_samples=cfg.torque_smooth_samples)
  fitted, cost = fit_gauss_newton(traj, pipeline_cfg)
  return float(fitted["cylinder_armature"]), cost


def run_test2(cfg: SysidConfig) -> None:
  if cfg.n_repeats <= 1:
    csv_path = cfg.csv_path
    if cfg.stage in ("collect", "full"):
      csv_path = collect_chirp_simulated(cfg) if cfg.simulate else collect_chirp_hardware(cfg)
    if cfg.stage in ("fit", "full"):
      if not csv_path:
        raise SystemExit("--csv-path가 필요합니다.")
      armature, cost = fit_chirp_armature(csv_path, cfg)
      _report_chirp_result(cfg, [{"csv_path": csv_path, "armature": armature, "cost": cost}])
    return

  # n_repeats > 1: chirp 여기 자체를 독립적으로 반복해서 재현성을 확인한다.
  results = []
  csv_paths = list(cfg.csv_paths) if cfg.stage == "fit" else []
  for i in range(cfg.n_repeats):
    print(f"\n===== test 2 반복 {i + 1}/{cfg.n_repeats} =====")
    if cfg.stage in ("collect", "full"):
      csv_path = collect_chirp_simulated(cfg, tag=f"chirp_sim_r{i}") if cfg.simulate else collect_chirp_hardware(cfg, tag_suffix=f"_r{i}")
    else:
      if i >= len(csv_paths):
        raise SystemExit(f"--csv-paths에 {cfg.n_repeats}개의 경로가 필요합니다 ({len(csv_paths)}개만 받음).")
      csv_path = csv_paths[i]

    if cfg.stage in ("fit", "full"):
      try:
        armature, cost = fit_chirp_armature(csv_path, cfg)
        results.append({"csv_path": csv_path, "armature": armature, "cost": cost})
      except RuntimeError as exc:
        print(f"[fit] 반복 {i + 1} 실패: {exc}")

  if cfg.stage in ("fit", "full"):
    _report_chirp_result(cfg, results)


def _report_chirp_result(cfg: SysidConfig, results: list[dict]) -> None:
  nominal = _nominal_params(cfg)["cylinder_armature"]
  table = PrettyTable()
  table.field_names = ["repeat", "armature", "cost", "csv"]
  for i, r in enumerate(results):
    table.add_row([i + 1, f"{r['armature']:.6f}", f"{r['cost']:.4f}", Path(r["csv_path"]).name])
  print(table)

  armatures = np.array([r["armature"] for r in results])
  mean_a = float(np.mean(armatures)) if len(armatures) else float("nan")
  std_a = float(np.std(armatures)) if len(armatures) else float("nan")
  print(f"\nnominal armature = {nominal:.6f}")
  print(f"평균 armature = {mean_a:.6f} (표준편차 {std_a:.6f}, n={len(armatures)})")
  if len(armatures) > 1 and mean_a > 0:
    print(f"변동폭(표준편차/평균) = {100 * std_a / mean_a:.1f}%")

  _update_report(
    cfg, "chirp_armature",
    {
      "method": "position_chirp_shooting_gauss_newton",
      "fix_damping": cfg.fix_damping, "fix_frictionloss": cfg.fix_frictionloss,
      "nominal_armature": nominal, "mean_armature": mean_a, "std_armature": std_a,
      "repeats": results,
    },
  )


def main() -> None:
  cfg = tyro.cli(SysidConfig)
  if cfg.test == 1:
    run_test1(cfg)
  else:
    run_test2(cfg)


if __name__ == "__main__":
  main()
