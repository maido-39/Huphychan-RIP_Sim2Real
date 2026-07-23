"""Open-loop sine reference for the inverse_sub motor (Revolute 3).

No RL policy is involved here. `sine_target_rad` is a hand-computed
target angle amplitude * sin(2*pi*f*t + phase), and `RateAccelLimiter`
reproduces the same target-shaping that a deployed RL policy would go
through (RateLimitedJointPositionAction in inverse_sine_env_cfg.py,
InverseRealPolicy._apply_target_limiter in real_policy_inference.py), so a
sine command run through this module experiences the same rate/accel
limits as a trained policy would before training is ever attempted.

Used by sine_motor_sim_test.py (simulation) and sine_motor_real_test.py
(real motor) to validate that the motor tracks a sine trajectory as
expected, and that the passive pendulum responds sensibly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TARGET_RATE_LIMIT_DEG_S = 1500.0
TARGET_ACCEL_LIMIT_DEG_S2 = 15000.0


@dataclass(frozen=True)
class SineReferenceConfig:
  amplitude_deg: float = 30.0
  frequency_hz: float = 1.0
  phase_deg: float = 0.0


def sine_target_rad(cfg: SineReferenceConfig, time_s: float) -> float:
  amplitude_rad = math.radians(cfg.amplitude_deg)
  phase_rad = math.radians(cfg.phase_deg)
  return amplitude_rad * math.sin(
    2.0 * math.pi * cfg.frequency_hz * time_s + phase_rad
  )


def _clamp(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, value))


class RateAccelLimiter:
  """Rate- and acceleration-limited target follower, operating in radians."""

  def __init__(self, *, max_delta_rad: float, max_delta_change_rad: float):
    self.max_delta_rad = float(max_delta_rad)
    self.max_delta_change_rad = float(max_delta_change_rad)
    self._limited_target_rad: float | None = None
    self._target_delta_rad = 0.0

  @classmethod
  def from_control_dt(cls, *, control_dt_s: float) -> "RateAccelLimiter":
    rate_limit_rad_s = math.radians(TARGET_RATE_LIMIT_DEG_S)
    accel_limit_rad_s2 = math.radians(TARGET_ACCEL_LIMIT_DEG_S2)
    return cls(
      max_delta_rad=rate_limit_rad_s * control_dt_s,
      max_delta_change_rad=accel_limit_rad_s2 * control_dt_s * control_dt_s,
    )

  def reset(self, current_rad: float) -> None:
    self._limited_target_rad = float(current_rad)
    self._target_delta_rad = 0.0

  def step(self, desired_target_rad: float) -> float:
    if self._limited_target_rad is None:
      self._limited_target_rad = float(desired_target_rad)

    desired_delta = _clamp(
      desired_target_rad - self._limited_target_rad,
      -self.max_delta_rad,
      self.max_delta_rad,
    )
    delta_step = _clamp(
      desired_delta - self._target_delta_rad,
      -self.max_delta_change_rad,
      self.max_delta_change_rad,
    )
    self._target_delta_rad = _clamp(
      self._target_delta_rad + delta_step,
      -self.max_delta_rad,
      self.max_delta_rad,
    )
    self._limited_target_rad += self._target_delta_rad
    return self._limited_target_rad
