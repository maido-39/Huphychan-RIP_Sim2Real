#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
  start_s: float
  end_s: float
  start_deg: float
  end_deg: float
  mode: str = "smooth"


@dataclass(frozen=True)
class MotionProfile:
  name: str
  segments: tuple[Segment, ...]

  @property
  def duration_s(self) -> float:
    return max(segment.end_s for segment in self.segments)

  def target_deg(self, time_s: float) -> float:
    if time_s <= self.segments[0].start_s:
      return self.segments[0].start_deg
    for segment in self.segments:
      if time_s <= segment.end_s:
        return _segment_target_deg(segment, time_s)
    return self.segments[-1].end_deg

  def target_rad(self, time_s: float) -> float:
    return math.radians(self.target_deg(time_s))


@dataclass(frozen=True)
class SineMotionProfile:
  name: str
  amplitude_deg: float
  frequency_hz: float
  duration_s: float
  start_hold_s: float = 0.0
  phase_deg: float = 0.0

  def target_deg(self, time_s: float) -> float:
    if time_s < self.start_hold_s:
      return 0.0
    phase_rad = math.radians(self.phase_deg)
    elapsed_s = time_s - self.start_hold_s
    return self.amplitude_deg * math.sin(
      2.0 * math.pi * self.frequency_hz * elapsed_s + phase_rad
    )

  def target_rad(self, time_s: float) -> float:
    return math.radians(self.target_deg(time_s))


def _smoothstep(x: float) -> float:
  x = min(max(x, 0.0), 1.0)
  return x * x * (3.0 - 2.0 * x)


def _segment_target_deg(segment: Segment, time_s: float) -> float:
  duration = max(segment.end_s - segment.start_s, 1e-9)
  u = (time_s - segment.start_s) / duration
  if segment.mode == "hold":
    alpha = 0.0
  elif segment.mode == "step":
    alpha = 1.0
  elif segment.mode == "linear":
    alpha = min(max(u, 0.0), 1.0)
  elif segment.mode == "smooth":
    alpha = _smoothstep(u)
  else:
    raise ValueError(f"Unknown segment mode: {segment.mode}")
  return segment.start_deg + alpha * (segment.end_deg - segment.start_deg)


def build_profile(
  profile_name: str,
  *,
  move_time_s: float = 1.0,
  hold_time_s: float = 1.0,
  start_hold_s: float = 0.5,
  mode: str = "step",
  waypoint_hold_s: float = 0.25,
  use_360_waypoints: bool = False,
  sine_amplitude_deg: float | None = None,
  sine_frequency_hz: float | None = None,
  sine_duration_s: float | None = None,
  sine_phase_deg: float = 0.0,
) -> MotionProfile | SineMotionProfile:
  """Build standard motor-axis sysid profiles.

  Profiles:
  - zero_to_180: 0 deg hold, move to 180 deg, hold.
  - zero_to_360: 0 deg hold, move to 360 deg, hold.
  - zero_180_zero: 0 deg hold, move to 180 deg, hold, move back to 0 deg, hold.
  - sine: amplitude/frequency sine command for motor response system ID.
  """
  if profile_name == "sine":
    if sine_amplitude_deg is None or sine_frequency_hz is None:
      raise ValueError("sine profile needs sine_amplitude_deg and sine_frequency_hz.")
    duration = float(sine_duration_s) if sine_duration_s is not None else hold_time_s
    if duration <= 0.0:
      raise ValueError("sine_duration_s must be positive.")
    if float(sine_frequency_hz) <= 0.0:
      raise ValueError("sine_frequency_hz must be positive.")
    amp = float(sine_amplitude_deg)
    freq = float(sine_frequency_hz)
    return SineMotionProfile(
      name=f"sine_{amp:g}deg_{freq:g}hz",
      amplitude_deg=amp,
      frequency_hz=freq,
      duration_s=float(start_hold_s) + duration,
      start_hold_s=float(start_hold_s),
      phase_deg=float(sine_phase_deg),
    )

  if move_time_s <= 0.0 and mode != "step":
    raise ValueError("move_time_s must be positive unless mode='step'.")
  if hold_time_s < 0.0 or start_hold_s < 0.0:
    raise ValueError("hold times must be non-negative.")

  effective_move_time_s = 0.0 if mode == "step" else move_time_s

  t0 = 0.0
  t1 = start_hold_s
  t2 = t1 + effective_move_time_s
  t3 = t2 + hold_time_s

  if profile_name == "zero_to_180":
    return MotionProfile(
      name=profile_name,
      segments=(
        Segment(t0, t1, 0.0, 0.0, "hold"),
        Segment(t1, t2, 0.0, 180.0, mode),
        Segment(t2, t3, 180.0, 180.0, "hold"),
      ),
    )

  if profile_name == "zero_to_360":
    if mode == "step" and use_360_waypoints:
      waypoint_hold_s = max(float(waypoint_hold_s), 0.0)
      t2 = t1 + waypoint_hold_s
      t3 = t2 + waypoint_hold_s
      t4 = t3 + waypoint_hold_s
      t5 = t4 + hold_time_s
      return MotionProfile(
        name=profile_name,
        segments=(
          Segment(t0, t1, 0.0, 0.0, "hold"),
          Segment(t1, t2, 0.0, 90.0, "step"),
          Segment(t2, t3, 90.0, 180.0, "step"),
          Segment(t3, t4, 180.0, 270.0, "step"),
          Segment(t4, t5, 270.0, 360.0, "step"),
        ),
      )
    return MotionProfile(
      name=profile_name,
      segments=(
        Segment(t0, t1, 0.0, 0.0, "hold"),
        Segment(t1, t2, 0.0, 360.0, mode),
        Segment(t2, t3, 360.0, 360.0, "hold"),
      ),
    )

  if profile_name == "zero_180_zero":
    t4 = t3 + effective_move_time_s
    t5 = t4 + hold_time_s
    return MotionProfile(
      name=profile_name,
      segments=(
        Segment(t0, t1, 0.0, 0.0, "hold"),
        Segment(t1, t2, 0.0, 180.0, mode),
        Segment(t2, t3, 180.0, 180.0, "hold"),
        Segment(t3, t4, 180.0, 0.0, mode),
        Segment(t4, t5, 0.0, 0.0, "hold"),
      ),
    )

  raise ValueError(
    f"Unknown profile_name={profile_name!r}. "
    "Use one of: zero_to_180, zero_to_360, zero_180_zero, sine, all."
  )


def available_profiles() -> tuple[str, ...]:
  return ("zero_to_180", "zero_to_360", "zero_180_zero")
