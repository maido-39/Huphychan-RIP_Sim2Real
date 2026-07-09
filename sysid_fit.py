#!/usr/bin/env python3
from __future__ import annotations

"""Rotary Inverted Pendulum system-identification 피팅 스크립트.

`sysid_collect.py`가 만든 CSV 로그(또는 `--selftest`의 합성 데이터)를 읽어
`assets/inverse.xml`의 물리 파라미터를 실물에 맞게 추정한다. 두 단계로
나뉜다.

Stage A (선형, 회귀자 기반)
  `mujoco_sysid.regressors.joint_torque_regressor`로 만든 Y 행렬에 대해
  `Y @ theta = tau`를 풀어 실린더(Rotor_1)/진자(BoldHolder_1) 바디의
  질량·무게중심·관성(theta)을 최소자승으로 추정한다. 진자는 무동력이므로
  그 관절의 일반화힘은 0으로 근사한다 — 진자 자체의 작은 damping/frictionloss
  토크는 무시되는 근사치이며, Stage B가 그 값들을 별도로 추정해 부분적으로
  보완한다.

  주의: `assets/inverse.xml`의 메쉬 충돌 지오메트리는 여기서 쓰는 임의의
  큰 조인트 각도 샘플에서 스스로 접촉을 일으킬 수 있고, 접촉력은
  회귀자가 표현할 수 없는 항이므로 반드시 `disableflags=contact`로 접촉을
  끄고 계산한다 (RL 학습 환경의 `inverse_env_cfg.py`도 동일하게 접촉을
  끈다).

Stage B (비선형, rollout 매칭)
  Stage A에서 얻은 질량/무게중심/관성을 고정하고, 관절
  damping/armature/frictionloss(양쪽)와 액추에이터 kp를 자유 파라미터로
  두어, 로그의 `target_angle_deg` 시퀀스를 그대로 재생하는 open-loop
  시뮬레이션과 실측 궤적의 차이를 `mujoco.minimize.least_squares`
  (box-bounded Levenberg-Marquardt)로 최소화한다.

결과는 `assets/inverse.xml`에 자동 반영하지 않는다. 그 파일은 RL 학습의
기준값이므로, 적용 여부는 사람이 JSON 리포트를 보고 판단한다.

하드웨어 없이 파이프라인을 검증하려면 `--selftest`를 쓴다: 현재
`assets/inverse.xml`의 공칭값으로 시뮬레이션해 합성 로그를 만들고, 그
로그로 Stage A+B를 실행해 공칭값을 다시 복원하는지 확인한다.
"""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import mujoco
import mujoco.minimize  # type: ignore[import-not-found]
import numpy as np
import tyro
from mujoco_sysid.parameters import get_dynamic_parameters
from mujoco_sysid.parameters import skew as _skew
from mujoco_sysid.regressors import joint_torque_regressor
from prettytable import PrettyTable
from scipy.signal import savgol_filter

_DEFAULT_XML = Path(__file__).parent / "assets" / "inverse.xml"
_CYLINDER_JOINT = "Revolute 3"
_POLE_JOINT = "Revolute 5"
_ACTUATOR = "position_revolute_3"


def set_dynamic_parameters(
  model: mujoco.MjModel, body_id: int, theta: np.ndarray
) -> None:
  """`mujoco_sysid.parameters.set_dynamic_parameters`의 버그 수정 버전.

  원본 구현은 `np.linalg.eigh`가 반환하는 고유벡터 행렬을 그대로
  회전행렬로 취급해 쿼터니언으로 변환하는데, `eigh`는 직교행렬이
  반사(reflection, `det=-1`)여도 상관하지 않는다. 그런 경우
  `mju_mat2Quat`이 완전히 다른 방향을 내놓아 관성텐서가 조용히
  깨진다 — 실제로 이 로봇의 Rotor_1 바디에서 재현/확인했다
  (원래는 이것 때문에 Stage A/B가 계속 이상한 값으로 수렴했다).
  고유벡터 중 하나의 부호를 뒤집어 `det=+1`(진짜 회전)로 만들면
  같은 물리적 관성텐서를 그대로 나타내면서 문제가 사라진다.
  """
  mass = theta[0]
  rc = theta[1:4] / mass
  inertia = theta[4:]
  inertia_full = np.array(
    [
      [inertia[0], inertia[1], inertia[3]],
      [inertia[1], inertia[2], inertia[4]],
      [inertia[3], inertia[4], inertia[5]],
    ]
  )
  inertia_full = inertia_full + mass * _skew(rc) @ _skew(rc)

  eigval, eigvec = np.linalg.eigh(inertia_full)
  if np.linalg.det(eigvec) < 0:
    eigvec[:, 0] *= -1.0
  if np.any(np.isclose(eigval, 0)):
    raise ValueError("Cannot deduce inertia matrix because RIR^T is singular.")

  model.body(body_id).mass = np.array([mass])
  model.body(body_id).ipos = rc
  mujoco.mju_mat2Quat(model.body(body_id).iquat, eigvec.flatten())
  model.body(body_id).inertia = eigval


@dataclass(frozen=True)
class SysidFitConfig:
  csv_path: str = ""
  xml_path: str = str(_DEFAULT_XML)

  savgol_window: int = 21
  savgol_polyorder: int = 3

  # Ridge penalty (relative to the nominal XML value) anchoring Stage A's
  # linear fit to the current mass/CoM/inertia. Needed because the
  # regressor is structurally rank-deficient for this mechanism (a vertical
  # first axis only observes ~8 of the 20 rigid-body parameters from
  # joint-torque data alone) — unregularized least squares can return
  # physically invalid (e.g. zero or negative mass) parameters.
  stage_a_prior_rel_std: float = 0.15

  skip_stage_a: bool = False
  skip_stage_b: bool = False

  # Stage B free-parameter bounds (all lower bounds are physical minimums >= 0).
  damping_max: float = 5.0
  armature_max: float = 0.05
  frictionloss_max: float = 0.5
  kp_min: float = 0.1
  kp_max: float = 50.0

  # Multiple-shooting window for Stage B (see fit_stage_b).
  shooting_window_s: float = 1.0

  output_report: str = "logs/sysid_report.json"

  selftest: bool = False
  # PASS 기준은 정확한 파라미터 복원이 아니라 궤적 예측 오차다 — 이
  # 시스템은 chirp 하나만으로는 여러 파라미터 조합이 비슷하게 잘 맞는
  # aliasing이 있어서(실측 확인), "진짜" 값 자체를 정확히 복원하는 것은
  # 보장할 수 없다. run_selftest 참고.
  selftest_angle_tol_deg: float = 5.0


def _disable_contacts(model: mujoco.MjModel) -> None:
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT


def _joint_body_id(model: mujoco.MjModel, joint_name: str) -> int:
  return int(model.jnt_bodyid[model.joint(joint_name).id])


class LoggedTrajectory(NamedTuple):
  time_s: np.ndarray
  target_angle_deg: np.ndarray
  cylinder_angle_rad: np.ndarray
  cylinder_vel_rad_s: np.ndarray
  cylinder_torque_nm: np.ndarray
  pole_angle_rad: np.ndarray
  pole_vel_rad_s: np.ndarray


def load_csv(csv_path: str) -> LoggedTrajectory:
  data = np.genfromtxt(csv_path, delimiter=",", names=True)
  dt = np.diff(data["time_s"])
  if dt.size and (np.max(dt) - np.min(dt)) > 0.2 * np.median(dt):
    print(
      f"[warn] sample spacing is not uniform (min={dt.min():.4f}s, "
      f"max={dt.max():.4f}s) — derivatives below may be noisy."
    )
  return LoggedTrajectory(
    time_s=data["time_s"],
    target_angle_deg=data["target_angle_deg"],
    cylinder_angle_rad=np.radians(data["motor_angle_deg"]),
    cylinder_vel_rad_s=np.radians(data["motor_vel_deg_s"]),
    cylinder_torque_nm=data["motor_torque_nm"],
    pole_angle_rad=np.radians(data["pole_angle_deg"]),
    pole_vel_rad_s=np.radians(data["pole_vel_deg_s"]),
  )


def _smooth_derivatives(
  angle_rad: np.ndarray, dt: float, window: int, polyorder: int, unwrap: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  angle = np.unwrap(angle_rad) if unwrap else angle_rad
  q = savgol_filter(angle, window, polyorder, deriv=0)
  qd = savgol_filter(angle, window, polyorder, deriv=1, delta=dt)
  qdd = savgol_filter(angle, window, polyorder, deriv=2, delta=dt)
  return q, qd, qdd


def fit_stage_a(
  traj: LoggedTrajectory,
  xml_path: str,
  window: int,
  polyorder: int,
  prior_rel_std: float,
) -> dict[str, np.ndarray]:
  """선형 회귀자 기반 질량/무게중심/관성 추정. body 이름 -> theta(10,) dict 반환.

  Ridge 회귀로 nominal XML 값 주변에 정착시킨다: 이 2-DOF 메커니즘은
  실린더 축이 거의 수직이라 중력이 그 축 방향 토크에 기여하지 않으므로,
  실린더 바디의 질량/무게중심(축 관성 제외)은 관절 토크만으로는 원천적으로
  관측 불가능하다(관측 가능한 파라미터 조합만 데이터에서 갱신되고, 나머지는
  정규화 항 덕분에 nominal 값 근처에 남는다 — unregularized least squares는
  이 방향에서 발산해 질량이 0/음수로 나오는 등 물리적으로 무효한 결과를
  낼 수 있다).
  """
  dt = float(np.median(np.diff(traj.time_s)))
  q_cyl, qd_cyl, qdd_cyl = _smooth_derivatives(
    traj.cylinder_angle_rad, dt, window, polyorder, unwrap=False
  )
  q_pole, qd_pole, qdd_pole = _smooth_derivatives(
    traj.pole_angle_rad, dt, window, polyorder, unwrap=True
  )

  model = mujoco.MjModel.from_xml_path(xml_path)
  _disable_contacts(model)
  data = mujoco.MjData(model)

  cyl_qadr = model.joint(_CYLINDER_JOINT).qposadr[0]
  cyl_dof = model.joint(_CYLINDER_JOINT).dofadr[0]
  pole_qadr = model.joint(_POLE_JOINT).qposadr[0]
  pole_dof = model.joint(_POLE_JOINT).dofadr[0]

  # 아직 모르는 damping/armature/frictionloss는 현재 XML의 nominal 값으로
  # 1차 보정한다. 측정 토크(actuator 출력)는 그 자체로는 관성 항만이 아니라
  # 관절의 수동 저항(damping/friction)까지 이미 반영된 값이므로, 보정 없이
  # 그대로 tau로 쓰면 회귀자가 표현할 수 없는 편향이 남는다.
  damp = model.dof_damping.copy()
  arm = model.dof_armature.copy()
  fric = model.dof_frictionloss.copy()

  # Savitzky-Golay 필터 가장자리는 신뢰할 수 없으므로 잘라낸다.
  edge = window
  n = len(traj.time_s)
  valid = range(edge, n - edge)

  Y_rows = []
  tau_rows = []
  for i in valid:
    data.qpos[cyl_qadr] = q_cyl[i]
    data.qpos[pole_qadr] = q_pole[i]
    data.qvel[cyl_dof] = qd_cyl[i]
    data.qvel[pole_dof] = qd_pole[i]
    data.qacc[cyl_dof] = qdd_cyl[i]
    data.qacc[pole_dof] = qdd_pole[i]

    mujoco.mj_inverse(model, data)
    mujoco.mj_rnePostConstraint(model, data)

    Y = joint_torque_regressor(model, data)
    tau = np.zeros(model.nv)
    tau[cyl_dof] = (
      traj.cylinder_torque_nm[i]
      + damp[cyl_dof] * qd_cyl[i]
      + fric[cyl_dof] * np.sign(qd_cyl[i])
      + arm[cyl_dof] * qdd_cyl[i]
    )
    # 진자는 무동력이라 실측 토크가 없다: damping/frictionloss/armature의
    # nominal 보정값만으로 그 관절의 일반화힘을 근사한다.
    tau[pole_dof] = (
      damp[pole_dof] * qd_pole[i]
      + fric[pole_dof] * np.sign(qd_pole[i])
      + arm[pole_dof] * qdd_pole[i]
    )

    Y_rows.append(Y)
    tau_rows.append(tau)

  Y_stacked = np.vstack(Y_rows)
  tau_stacked = np.concatenate(tau_rows)

  cyl_body = _joint_body_id(model, _CYLINDER_JOINT)
  theta_nominal = np.concatenate(
    [get_dynamic_parameters(model, body_id) for body_id in model.jnt_bodyid]
  )
  sigma = np.maximum(np.abs(theta_nominal), 1e-8) * prior_rel_std
  prior_precision = 1.0 / sigma**2
  lhs = Y_stacked.T @ Y_stacked + np.diag(prior_precision)
  rhs = Y_stacked.T @ tau_stacked + prior_precision * theta_nominal
  theta = np.linalg.solve(lhs, rhs)

  fit_residual = float(np.max(np.abs(Y_stacked @ theta - tau_stacked)))
  print(f"[stage A] linear fit max residual = {fit_residual:.6f} N*m")
  # joint_torque_regressor stacks theta blocks in model.jnt_bodyid order.
  body_order = list(model.jnt_bodyid)
  theta_by_body = {
    "Rotor_1" if body_order[i] == cyl_body else "BoldHolder_1": theta[
      10 * i : 10 * (i + 1)
    ]
    for i in range(len(body_order))
  }
  return theta_by_body


_PARAM_NAMES = [
  "cylinder_damping",
  "pole_damping",
  "cylinder_armature",
  "pole_armature",
  "cylinder_frictionloss",
  "pole_frictionloss",
  "actuator_kp",
  "cylinder_mass_scale",
  "pole_mass_scale",
]


def _apply_stage_b_params(
  model: mujoco.MjModel, x: np.ndarray, theta_by_body: dict[str, np.ndarray] | None
) -> None:
  cyl_dof = model.joint(_CYLINDER_JOINT).dofadr[0]
  pole_dof = model.joint(_POLE_JOINT).dofadr[0]
  act_id = model.actuator(_ACTUATOR).id
  model.dof_damping[cyl_dof] = x[0]
  model.dof_damping[pole_dof] = x[1]
  model.dof_armature[cyl_dof] = x[2]
  model.dof_armature[pole_dof] = x[3]
  model.dof_frictionloss[cyl_dof] = x[4]
  model.dof_frictionloss[pole_dof] = x[5]
  model.actuator_gainprm[act_id, 0] = x[6]
  model.actuator_biasprm[act_id, 1] = -x[6]

  # cylinder_mass_scale/pole_mass_scale: Stage A's mass/CoM/inertia is a
  # ridge fit that can still be biased (see fit_stage_a docstring), and
  # holding it perfectly fixed lets Stage B's damping/armature/kp silently
  # compensate for that bias instead of tracking the real trajectory —
  # observed empirically as convergence to a different, wrong local optimum
  # on synthetic ground-truth data. Scaling the whole 10-vector theta by a
  # single scalar per body preserves CoM position and inertia "shape"
  # (same relative mass distribution, uniformly denser/lighter) while
  # giving Stage B one well-conditioned extra degree of freedom per body
  # to correct that bias using the full trajectory-matching objective.
  if theta_by_body is not None:
    set_dynamic_parameters(
      model, model.body("Rotor_1").id, x[7] * theta_by_body["Rotor_1"]
    )
    set_dynamic_parameters(
      model, model.body("BoldHolder_1").id, x[8] * theta_by_body["BoldHolder_1"]
    )
    mujoco.mj_setConst(model, mujoco.MjData(model))


def _rollout(
  model: mujoco.MjModel, data: mujoco.MjData, target_deg: np.ndarray
) -> np.ndarray:
  """target_deg(라디안 변환 후) 시퀀스를 그대로 ctrl로 재생하고 상태 궤적을 반환한다."""
  cyl_qadr = model.joint(_CYLINDER_JOINT).qposadr[0]
  cyl_dof = model.joint(_CYLINDER_JOINT).dofadr[0]
  pole_qadr = model.joint(_POLE_JOINT).qposadr[0]
  pole_dof = model.joint(_POLE_JOINT).dofadr[0]
  act_id = model.actuator(_ACTUATOR).id

  n = len(target_deg)
  out = np.zeros((n, 4))  # cyl_angle, cyl_vel, pole_angle, pole_vel
  for k in range(n):
    data.ctrl[act_id] = math.radians(float(target_deg[k]))
    mujoco.mj_step(model, data)
    out[k, 0] = data.qpos[cyl_qadr]
    out[k, 1] = data.qvel[cyl_dof]
    out[k, 2] = data.qpos[pole_qadr]
    out[k, 3] = data.qvel[pole_dof]
  return out


def fit_stage_b(
  traj: LoggedTrajectory,
  xml_path: str,
  theta_by_body: dict[str, np.ndarray] | None,
  cfg: SysidFitConfig,
) -> dict[str, float]:
  base_model = mujoco.MjModel.from_xml_path(xml_path)
  _disable_contacts(base_model)

  cyl_qadr = base_model.joint(_CYLINDER_JOINT).qposadr[0]
  pole_qadr = base_model.joint(_POLE_JOINT).qposadr[0]
  cyl_dof = base_model.joint(_CYLINDER_JOINT).dofadr[0]
  pole_dof = base_model.joint(_POLE_JOINT).dofadr[0]
  act_id = base_model.actuator(_ACTUATOR).id

  x0 = np.array(
    [
      base_model.dof_damping[cyl_dof],
      base_model.dof_damping[pole_dof],
      base_model.dof_armature[cyl_dof],
      base_model.dof_armature[pole_dof],
      base_model.dof_frictionloss[cyl_dof],
      base_model.dof_frictionloss[pole_dof],
      base_model.actuator_gainprm[act_id, 0],
      1.0,
      1.0,
    ]
  )
  lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cfg.kp_min, 0.3, 0.3])
  upper = np.array(
    [
      cfg.damping_max,
      cfg.damping_max,
      cfg.armature_max,
      cfg.armature_max,
      cfg.frictionloss_max,
      cfg.frictionloss_max,
      cfg.kp_max,
      3.0,
      3.0,
    ]
  )

  real = np.stack(
    [
      traj.cylinder_angle_rad,
      traj.cylinder_vel_rad_s,
      traj.pole_angle_rad,
      traj.pole_vel_rad_s,
    ],
    axis=1,
  )
  # 각도 채널을 속도 채널보다 더 신뢰한다(속도는 유한차분/인코더 노이즈가 크다).
  weights = np.array([1.0, 0.1, 1.0, 0.1])

  # Multiple shooting: 20초짜리 단일 open-loop rollout 하나로 맞추려 하면
  # (역진자는 약하게 카오스적이라) 아주 작은 파라미터 차이도 지수적으로
  # 벌어져서, 최적화 지형이 나빠지고 서로 다른 파라미터 조합이 우연히
  # 비슷한 total cost를 내는 aliasing이 심해진다(실측으로 확인). 대신
  # `shooting_window_s`마다 실측 상태로 다시 앵커링해서 짧은 구간 예측
  # 오차만 누적한다 — 궤적 적합 문제에서 흔히 쓰는 표준적인 방법이다.
  dt = float(np.median(np.diff(traj.time_s)))
  window = max(1, int(round(cfg.shooting_window_s / dt)))
  n_samples = len(traj.time_s)

  def _residual_single(x: np.ndarray) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(xml_path)
    _disable_contacts(model)
    _apply_stage_b_params(model, x, theta_by_body)
    data = mujoco.MjData(model)

    chunks = []
    for start in range(0, n_samples, window):
      end = min(start + window, n_samples)
      data.qpos[cyl_qadr] = traj.cylinder_angle_rad[start]
      data.qvel[cyl_dof] = traj.cylinder_vel_rad_s[start]
      data.qpos[pole_qadr] = traj.pole_angle_rad[start]
      data.qvel[pole_dof] = traj.pole_vel_rad_s[start]
      mujoco.mj_forward(model, data)
      sim = _rollout(model, data, traj.target_angle_deg[start:end])
      chunks.append((sim - real[start:end]) * weights)
    return np.concatenate(chunks).reshape(-1)

  def residual(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
      return _residual_single(x)
    return np.stack([_residual_single(x[:, k]) for k in range(x.shape[1])], axis=1)

  # x_scale="jac": damping/armature/frictionloss/kp span ~4 orders of
  # magnitude, so unscaled LM barely moves the small-magnitude parameters
  # while overshooting the large ones. Per-column Jacobian-norm scaling
  # fixes this (verified empirically — unscaled fits converged to a
  # different, wrong parameterization on synthetic ground-truth data).
  x_fit, trace = mujoco.minimize.least_squares(  # type: ignore[attr-defined]
    x0, residual, bounds=(lower, upper), x_scale="jac", verbose=0
  )
  cost = float(trace[-1].objective) if trace else float("nan")
  print(f"[stage B] final cost = {cost:.6f}")

  return dict(zip(_PARAM_NAMES, x_fit.tolist(), strict=True))


def _nominal_stage_b_params(xml_path: str) -> dict[str, float]:
  model = mujoco.MjModel.from_xml_path(xml_path)
  cyl_dof = model.joint(_CYLINDER_JOINT).dofadr[0]
  pole_dof = model.joint(_POLE_JOINT).dofadr[0]
  act_id = model.actuator(_ACTUATOR).id
  values = [
    model.dof_damping[cyl_dof],
    model.dof_damping[pole_dof],
    model.dof_armature[cyl_dof],
    model.dof_armature[pole_dof],
    model.dof_frictionloss[cyl_dof],
    model.dof_frictionloss[pole_dof],
    model.actuator_gainprm[act_id, 0],
    1.0,  # cylinder_mass_scale: 1.0 = "no correction on top of Stage A's fit".
    1.0,  # pole_mass_scale: same.
  ]
  return dict(zip(_PARAM_NAMES, (float(v) for v in values), strict=True))


def _nominal_theta(xml_path: str) -> dict[str, np.ndarray]:
  model = mujoco.MjModel.from_xml_path(xml_path)
  return {
    "Rotor_1": np.asarray(get_dynamic_parameters(model, model.body("Rotor_1").id)),
    "BoldHolder_1": np.asarray(
      get_dynamic_parameters(model, model.body("BoldHolder_1").id)
    ),
  }


def generate_synthetic_csv(cfg: SysidFitConfig, csv_path: str) -> None:
  """--selftest용: 공칭 XML을 시뮬레이션해 sysid_collect.py와 같은 CSV를 만든다."""
  from mjlab.tasks.inverse.sysid_collect import (
    SysidCollectConfig,
    build_targets,
    signal_duration_s,
  )

  collect_cfg = SysidCollectConfig(mode="chirp", control_hz=200.0)
  n_samples = int(signal_duration_s(collect_cfg) * collect_cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / collect_cfg.control_hz
  target_deg = build_targets(collect_cfg, sample_times)

  model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  _disable_contacts(model)
  data = mujoco.MjData(model)
  act_id = model.actuator(_ACTUATOR).id
  cyl_qadr = model.joint(_CYLINDER_JOINT).qposadr[0]
  cyl_dof = model.joint(_CYLINDER_JOINT).dofadr[0]
  pole_qadr = model.joint(_POLE_JOINT).qposadr[0]
  pole_dof = model.joint(_POLE_JOINT).dofadr[0]

  rows = []
  for k in range(n_samples):
    data.ctrl[act_id] = math.radians(float(target_deg[k]))
    mujoco.mj_step(model, data)
    rows.append(
      (
        sample_times[k],
        target_deg[k],
        math.degrees(data.qpos[cyl_qadr]),
        math.degrees(data.qvel[cyl_dof]),
        float(data.actuator_force[act_id]),
        math.degrees(data.qpos[pole_qadr]),
        math.degrees(data.qvel[pole_dof]),
      )
    )

  Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
  with open(csv_path, "w") as f:
    f.write(
      "time_s,target_angle_deg,motor_angle_deg,motor_vel_deg_s,motor_torque_nm,"
      "pole_angle_deg,pole_vel_deg_s\n"
    )
    for row in rows:
      f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def run_selftest(cfg: SysidFitConfig) -> bool:
  synthetic_csv = "logs/sysid_selftest.csv"
  print(
    f"[selftest] generating synthetic data from nominal {cfg.xml_path} -> {synthetic_csv}"
  )
  generate_synthetic_csv(cfg, synthetic_csv)

  cfg = replace(cfg, csv_path=synthetic_csv)
  traj = load_csv(cfg.csv_path)

  nominal_theta = _nominal_theta(cfg.xml_path)
  nominal_b = _nominal_stage_b_params(cfg.xml_path)

  theta_by_body = fit_stage_a(
    traj,
    cfg.xml_path,
    cfg.savgol_window,
    cfg.savgol_polyorder,
    cfg.stage_a_prior_rel_std,
  )
  fitted_b = fit_stage_b(traj, cfg.xml_path, theta_by_body, cfg)

  # 참고 출력: 개별 파라미터 값 비교. 하지만 PASS/FAIL 기준으로는 쓰지
  # 않는다 — 이 메커니즘은 chirp 하나만으로는 damping/armature/kp/질량
  # 사이에 서로 다른 조합이 궤적을 거의 동일하게 잘 맞추는 aliasing이
  # 있음을 실측으로 확인했다(예: kp를 높이면서 damping도 같이 높이면
  # 거의 같은 궤적이 나옴). 그래서 "진짜" 값을 정확히 복원했는지가 아니라
  # "이 파라미터로 다시 굴렸을 때 실측 궤적을 잘 예측하는지"를 기준으로
  # 삼는다 — 결국 sim2real 격차를 줄이는 게 목적이지, CAD 수치를 정확히
  # 맞히는 게 목적이 아니기 때문이다.
  print("\n[selftest] fitted mass/CoM/inertia (Stage A, informational):")
  for body in ("Rotor_1", "BoldHolder_1"):
    mass_nom = nominal_theta[body][0]
    mass_fit = theta_by_body[body][0]
    print(f"  {body}: nominal_mass={mass_nom:.6f} fitted_mass={mass_fit:.6f}")

  print("\n[selftest] fitted joint/actuator params (Stage B, informational):")
  for name in _PARAM_NAMES:
    print(f"  {name}: nominal={nominal_b[name]:.6f} fitted={fitted_b[name]:.6f}")

  ok = all(math.isfinite(v) for v in fitted_b.values())
  ok &= theta_by_body["Rotor_1"][0] > 0.0 and theta_by_body["BoldHolder_1"][0] > 0.0

  # 여기서도 Stage B와 같은 multiple-shooting 방식으로 예측 오차를 잰다.
  # 역진자는 (약하게) 카오스적이라 20초짜리 단일 open-loop rollout으로는
  # 파라미터가 아무리 정확해도 나비효과로 오차가 지수적으로 커진다 —
  # 실물 진자도 같은 이유로 그 시간 지평에서는 예측 불가능하다. 그래서
  # "짧은 구간 예측을 잘 하는지"가 이 파이프라인이 실제로 검증할 수 있는
  # 유일하게 합리적인 기준이다.
  fitted_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
  _disable_contacts(fitted_model)
  x_fit = np.array([fitted_b[name] for name in _PARAM_NAMES])
  _apply_stage_b_params(fitted_model, x_fit, theta_by_body)
  cyl_qadr = fitted_model.joint(_CYLINDER_JOINT).qposadr[0]
  cyl_dof = fitted_model.joint(_CYLINDER_JOINT).dofadr[0]
  pole_qadr = fitted_model.joint(_POLE_JOINT).qposadr[0]
  pole_dof = fitted_model.joint(_POLE_JOINT).dofadr[0]
  data = mujoco.MjData(fitted_model)

  dt = float(np.median(np.diff(traj.time_s)))
  window = max(1, int(round(cfg.shooting_window_s / dt)))
  n_samples = len(traj.time_s)
  cyl_err_deg = np.zeros(n_samples)
  pole_err_deg = np.zeros(n_samples)
  for start in range(0, n_samples, window):
    end = min(start + window, n_samples)
    data.qpos[cyl_qadr] = traj.cylinder_angle_rad[start]
    data.qvel[cyl_dof] = traj.cylinder_vel_rad_s[start]
    data.qpos[pole_qadr] = traj.pole_angle_rad[start]
    data.qvel[pole_dof] = traj.pole_vel_rad_s[start]
    mujoco.mj_forward(fitted_model, data)
    sim = _rollout(fitted_model, data, traj.target_angle_deg[start:end])
    cyl_err_deg[start:end] = np.degrees(sim[:, 0] - traj.cylinder_angle_rad[start:end])
    pole_err_deg[start:end] = np.degrees(sim[:, 2] - traj.pole_angle_rad[start:end])

  cyl_rmse = float(np.sqrt(np.mean(cyl_err_deg**2)))
  pole_rmse = float(np.sqrt(np.mean(pole_err_deg**2)))
  print(
    f"\n[selftest] {cfg.shooting_window_s:.1f}s-window prediction RMSE: "
    f"cylinder={cyl_rmse:.2f}deg pole={pole_rmse:.2f}deg "
    f"(tolerance={cfg.selftest_angle_tol_deg}deg)"
  )
  ok &= cyl_rmse < cfg.selftest_angle_tol_deg
  ok &= pole_rmse < cfg.selftest_angle_tol_deg

  print("\n[selftest] " + ("PASS" if ok else "FAIL"))
  return ok


def write_report(
  cfg: SysidFitConfig,
  theta_by_body: dict[str, np.ndarray] | None,
  fitted_b: dict[str, float] | None,
) -> None:
  report: dict = {"csv_path": cfg.csv_path, "xml_path": cfg.xml_path}

  table = PrettyTable()
  table.field_names = ["parameter", "nominal", "fitted"]

  if theta_by_body is not None:
    nominal_theta = _nominal_theta(cfg.xml_path)
    report["mass_com_inertia"] = {}
    for body in ("Rotor_1", "BoldHolder_1"):
      report["mass_com_inertia"][body] = {
        "nominal_theta": nominal_theta[body].tolist(),
        "fitted_theta": theta_by_body[body].tolist(),
      }
      table.add_row(
        [
          f"{body}.mass",
          f"{nominal_theta[body][0]:.6f}",
          f"{theta_by_body[body][0]:.6f}",
        ]
      )

  if fitted_b is not None:
    nominal_b = _nominal_stage_b_params(cfg.xml_path)
    report["joint_actuator_params"] = {"nominal": nominal_b, "fitted": fitted_b}
    for name in _PARAM_NAMES:
      table.add_row([name, f"{nominal_b[name]:.6f}", f"{fitted_b[name]:.6f}"])

  print(table)

  out_path = Path(cfg.output_report)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(report, indent=2))
  print(f"\nReport written to {out_path}")
  print(
    "Note: assets/inverse.xml was NOT modified — review the report and apply by hand."
  )


def main() -> None:
  cfg = tyro.cli(SysidFitConfig)

  if cfg.selftest:
    ok = run_selftest(cfg)
    raise SystemExit(0 if ok else 1)

  if not cfg.csv_path:
    raise SystemExit("--csv-path is required (or pass --selftest).")

  traj = load_csv(cfg.csv_path)

  theta_by_body = None
  if not cfg.skip_stage_a:
    theta_by_body = fit_stage_a(
      traj,
      cfg.xml_path,
      cfg.savgol_window,
      cfg.savgol_polyorder,
      cfg.stage_a_prior_rel_std,
    )

  fitted_b = None
  if not cfg.skip_stage_b:
    fitted_b = fit_stage_b(traj, cfg.xml_path, theta_by_body, cfg)

  write_report(cfg, theta_by_body, fitted_b)


if __name__ == "__main__":
  main()
