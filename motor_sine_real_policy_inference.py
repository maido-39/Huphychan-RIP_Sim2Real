#!/usr/bin/env python3
"""Motor-Sine 정책 전용 실물 추론 모듈.

대상 학습 태스크
- Task ID: Mjlab-Inverse-Motor-Sine
- 학습 파일: motor_sine_env_cfg.py
- 정책 주기: 50 Hz
- 한 프레임 observation:
    1) cos(motor_angle_rad)
    2) sin(motor_angle_rad)
    3) motor_velocity_rad_s / 30
    4) sine_target_rad / 90deg
- history_length=2
- 최종 policy 입력: 8차원

중요
- 폴 엔코더, 폴 각도, 폴 속도는 사용하지 않는다.
- 이전 action도 observation에 넣지 않는다.
- 실물에서도 학습과 동일하게 0.25 Hz, 진폭 90도의 sine 목표를 만든다.
- 실물 모터의 영점과 회전 방향은 cylinder_zero_deg와 cylinder_sign으로 맞춘다.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import torch

import mjlab.tasks  # noqa: F401  # task registry 등록
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


TASK_ID = "Mjlab-Inverse-Motor-Only-Sine"

OBS_DIM = 4
HISTORY_LENGTH = 2
POLICY_INPUT_DIM = OBS_DIM * HISTORY_LENGTH

POLICY_DT_S = 0.02
ACTION_SCALE_RAD = math.radians(20.0)
VEL_OBS_SCALE = 30.0

MOTOR_SINE_AMPLITUDE_RAD = math.radians(90.0)
MOTOR_SINE_FREQUENCY_HZ = 0.25

TARGET_RATE_LIMIT_RAD_S = math.radians(1500.0)
TARGET_ACCEL_LIMIT_RAD_S2 = math.radians(15000.0)
TARGET_DELTA_LIMIT_RAD = TARGET_RATE_LIMIT_RAD_S * POLICY_DT_S
TARGET_DELTA_CHANGE_LIMIT_RAD = TARGET_ACCEL_LIMIT_RAD_S2 * POLICY_DT_S * POLICY_DT_S

# ObservationManager는 term별로 history를 쌓은 뒤 concat한다.
# 최종 순서:
# [cos_prev, sin_prev, cos_cur, sin_cur,
#  velocity_prev, velocity_cur,
#  target_prev, target_cur]
_OBS_TERM_SLICES = (
  slice(0, 2),  # motor_angle_periodic
  slice(2, 3),  # motor_velocity
  slice(3, 4),  # motor_target
)


@dataclass(frozen=True)
class MotorMeasurement:
  """실제 모터에서 읽은 한 시점의 상태."""

  angle_deg: float
  velocity_deg_s: float
  torque_nm: float | None = None


@dataclass(frozen=True)
class MotorSineInferenceConfig:
  checkpoint_file: str
  device: str = "cpu"

  # 실물 각도를 시뮬레이터 좌표계로 변환한다.
  # sim_angle = (real_angle - zero) * sign
  cylinder_sign: float = 1.0
  cylinder_zero_deg: float = 0.0

  # 안전한 첫 시험을 위한 action 축소 배율.
  # 1.0이면 학습 action을 그대로 사용한다.
  action_scale_multiplier: float = 0.2


class MotorSineRealPolicy:
  """Motor-Sine 체크포인트를 실제 모터 상태와 연결하는 추론 래퍼."""

  def __init__(self, cfg: MotorSineInferenceConfig):
    if cfg.cylinder_sign not in (-1.0, 1.0):
      raise ValueError("cylinder_sign은 1.0 또는 -1.0이어야 합니다.")
    if not 0.0 < cfg.action_scale_multiplier <= 1.0:
      raise ValueError("action_scale_multiplier는 0보다 크고 1 이하여야 합니다.")

    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self._policy = self._load_policy(cfg)
    self._history: deque[torch.Tensor] = deque(maxlen=HISTORY_LENGTH)

    self._limited_target_rad: float | None = None
    self._target_delta_rad = 0.0

  def _load_policy(self, cfg: MotorSineInferenceConfig):
    """학습 환경과 동일한 actor 구조를 만든 뒤 체크포인트를 로드한다."""
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    wrapped_env = RslRlVecEnvWrapper(env)

    try:
      agent_cfg = load_rl_cfg(TASK_ID)
      runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
      runner = runner_cls(wrapped_env, asdict(agent_cfg), device=cfg.device)
      runner.load(
        cfg.checkpoint_file,
        load_cfg={"actor": True},
        strict=True,
        map_location=cfg.device,
      )
      return runner.get_inference_policy(device=cfg.device)
    finally:
      wrapped_env.close()

  @staticmethod
  def _deg_to_rad(value_deg: float) -> float:
    return math.radians(float(value_deg))

  @staticmethod
  def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

  def reset(self, current_motor_angle_deg: float) -> None:
    """새 실물 실행을 시작할 때 history와 목표 limiter를 초기화한다."""
    current_rad = self.normalize_motor_angle(current_motor_angle_deg)
    self._history.clear()
    self._limited_target_rad = current_rad
    self._target_delta_rad = 0.0

  def normalize_motor_angle(self, motor_angle_deg: float) -> float:
    """실물 degree 각도를 시뮬레이터 기준 rad 각도로 변환한다."""
    return self._deg_to_rad(
      (float(motor_angle_deg) - self.cfg.cylinder_zero_deg) * self.cfg.cylinder_sign
    )

  def normalize_motor_velocity(self, motor_velocity_deg_s: float) -> float:
    """실물 degree/s 속도를 시뮬레이터 기준 rad/s로 변환한다."""
    return self._deg_to_rad(float(motor_velocity_deg_s) * self.cfg.cylinder_sign)

  def denormalize_motor_angle(self, motor_angle_rad: float) -> float:
    """시뮬레이터 기준 rad 목표를 실제 모터 degree 목표로 변환한다."""
    return (
      math.degrees(float(motor_angle_rad)) / self.cfg.cylinder_sign
      + self.cfg.cylinder_zero_deg
    )

  @staticmethod
  def sine_target_rad(elapsed_s: float) -> float:
    """학습과 동일한 90도, 0.25 Hz sine 목표."""
    phase = 2.0 * math.pi * MOTOR_SINE_FREQUENCY_HZ * float(elapsed_s)
    return MOTOR_SINE_AMPLITUDE_RAD * math.sin(phase)

  def build_single_observation(
    self,
    measurement: MotorMeasurement,
    elapsed_s: float,
  ) -> torch.Tensor:
    """한 시점의 4차원 observation을 학습 순서 그대로 만든다."""
    motor_angle_rad = self.normalize_motor_angle(measurement.angle_deg)
    motor_velocity_rad_s = self.normalize_motor_velocity(measurement.velocity_deg_s)
    target_rad = self.sine_target_rad(elapsed_s)

    obs = torch.tensor(
      [
        math.cos(motor_angle_rad),
        math.sin(motor_angle_rad),
        self._clamp(motor_velocity_rad_s / VEL_OBS_SCALE, -5.0, 5.0),
        target_rad / MOTOR_SINE_AMPLITUDE_RAD,
      ],
      dtype=torch.float32,
      device=self.device,
    )

    if obs.numel() != OBS_DIM:
      raise RuntimeError(f"잘못된 observation 크기: {obs.numel()} != {OBS_DIM}")
    return obs

  def build_policy_input(
    self,
    measurement: MotorMeasurement,
    elapsed_s: float,
  ) -> torch.Tensor:
    """term-wise history 순서를 재현한 8차원 policy 입력을 만든다."""
    current_obs = self.build_single_observation(measurement, elapsed_s)

    if not self._history:
      self._history.append(current_obs.clone())
      self._history.append(current_obs.clone())
    else:
      self._history.append(current_obs.clone())

    previous_obs, current_obs = self._history[0], self._history[1]
    stacked = torch.cat(
      [
        torch.cat([previous_obs[term_slice], current_obs[term_slice]])
        for term_slice in _OBS_TERM_SLICES
      ]
    )

    if stacked.numel() != POLICY_INPUT_DIM:
      raise RuntimeError(
        f"잘못된 policy 입력 크기: {stacked.numel()} != {POLICY_INPUT_DIM}"
      )

    return stacked.unsqueeze(0)

  def _apply_target_limiter(
    self,
    desired_target_rad: float,
    current_motor_rad: float,
  ) -> float:
    """학습 환경과 같은 목표 위치 rate/acceleration limiter."""
    if self._limited_target_rad is None:
      self._limited_target_rad = current_motor_rad
      self._target_delta_rad = 0.0

    desired_delta = self._clamp(
      desired_target_rad - self._limited_target_rad,
      -TARGET_DELTA_LIMIT_RAD,
      TARGET_DELTA_LIMIT_RAD,
    )

    delta_change = self._clamp(
      desired_delta - self._target_delta_rad,
      -TARGET_DELTA_CHANGE_LIMIT_RAD,
      TARGET_DELTA_CHANGE_LIMIT_RAD,
    )

    self._target_delta_rad = self._clamp(
      self._target_delta_rad + delta_change,
      -TARGET_DELTA_LIMIT_RAD,
      TARGET_DELTA_LIMIT_RAD,
    )

    self._limited_target_rad += self._target_delta_rad
    return self._limited_target_rad

  def infer(
    self,
    measurement: MotorMeasurement,
    elapsed_s: float,
  ) -> dict[str, float]:
    """실물 상태와 시간을 받아 policy 목표각을 계산한다."""
    current_motor_rad = self.normalize_motor_angle(measurement.angle_deg)
    target_reference_rad = self.sine_target_rad(elapsed_s)
    policy_input = self.build_policy_input(measurement, elapsed_s)

    with torch.inference_mode():
      action_tensor = self._policy(policy_input)

    raw_action = float(action_tensor.squeeze().item())
    clipped_action = self._clamp(raw_action, -1.0, 1.0)
    applied_action = clipped_action * self.cfg.action_scale_multiplier

    desired_target_rad = applied_action * ACTION_SCALE_RAD
    limited_target_rad = self._apply_target_limiter(
      desired_target_rad=desired_target_rad,
      current_motor_rad=current_motor_rad,
    )

    return {
      "raw_policy_action": raw_action,
      "clipped_policy_action": clipped_action,
      "applied_policy_action": applied_action,
      "reference_target_rad": target_reference_rad,
      "reference_target_deg": math.degrees(target_reference_rad),
      "desired_target_rad": desired_target_rad,
      "desired_target_deg": self.denormalize_motor_angle(desired_target_rad),
      "target_angle_rad": limited_target_rad,
      "target_angle_deg": self.denormalize_motor_angle(limited_target_rad),
      "target_delta_deg": math.degrees(self._target_delta_rad),
    }
