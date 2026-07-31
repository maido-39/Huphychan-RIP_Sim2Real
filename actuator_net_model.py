#!/usr/bin/env python3
"""Actuator net model definition + feature construction, shared by training, eval, and
(eventually) the sim deployment wrapper.

tau_actual = ActuatorNet(features), predicting the real motor's output torque directly
from a short window of what was commanded and how the motor actually responded.

Features per sample (see build_dataset for the exact ordering), all knowable at command
time -- no leakage from the label:
  - target_torque_nm history, last `history` samples
    (tau_target = kp*pos_err + kd*vel_err + torque_ff, computed the same way MIT mode
    computes it internally -- see run_policy_motor.py / sine_control_log.py)
  - motor_vel_deg_s history (measured), last `history` samples
  - target velocity history, last `history` samples -- reconstructed as
    d(target_angle_deg)/dt since it isn't logged directly everywhere. tau_target alone
    conflates "large torque from a big position error" with "large torque from a big
    commanded velocity", which can behave differently on the real motor, so the net
    needs to see them separately.
  - optionally, two EMA-of-tau_target^2 features (short/long time constants) as a proxy
    for a thermal/current-integrated state. Ablation showed these don't help once the
    training data actually covers sustained-low-velocity + high-torque conditions (see
    held_data_*.csv) -- default is EMA off (ema_short_s=ema_long_s=None).

Label: motor_torque_nm at the current sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _ema(values, tau_s: float, dt_s: float):
  alpha = 1.0 - pow(2.71828182845904523536, -dt_s / tau_s)
  out = [0.0] * len(values)
  acc = 0.0
  for i, v in enumerate(values):
    acc = acc + alpha * (v * v - acc)
    out[i] = acc
  return out


def build_dataset(
  run: dict, *, history: int, ema_short_s: float | None = None, ema_long_s: float | None = None
):
  """run: dict with list/array fields "t", "target_torque_nm", "motor_torque_nm",
  "motor_vel_deg_s", "target_angle_deg" (same keys _load_run in train_actuator_net.py
  produces). Returns (X, y, target_now) numpy arrays, or None if the run is too short.
  """
  import numpy as np

  t = run["t"]
  n = len(t)
  if n < history + 10:
    return None
  dt_s = float(np.median(np.diff(t))) if n > 1 else 0.005

  target = run["target_torque_nm"]
  vel = run["motor_vel_deg_s"]
  motor = run["motor_torque_nm"]
  target_angle = run["target_angle_deg"]

  use_ema = ema_short_s is not None and ema_long_s is not None
  if use_ema:
    ema_short = _ema(target, ema_short_s, dt_s)
    ema_long = _ema(target, ema_long_s, dt_s)

  # Target velocity isn't logged directly everywhere (Pendulum policy logs only log the
  # position target), so it's reconstructed the same way for every source: the time
  # derivative of the position target. For sine_control_log.py runs this exactly
  # reproduces the analytic feedforward velocity that was actually commanded.
  target_vel = np.gradient(np.asarray(target_angle, dtype=np.float64), np.asarray(t, dtype=np.float64))

  X = []
  y = []
  target_now = []
  for i in range(history - 1, n):
    tau_hist = target[i - history + 1 : i + 1]
    vel_hist = [v / 1000.0 for v in vel[i - history + 1 : i + 1]]  # deg/s -> ~O(1) scale
    target_vel_hist = [v / 1000.0 for v in target_vel[i - history + 1 : i + 1]]
    feat = list(tau_hist) + vel_hist + list(target_vel_hist)
    if use_ema:
      feat += [ema_short[i], ema_long[i]]
    X.append(feat)
    y.append(motor[i])
    target_now.append(target[i])

  return (
    np.asarray(X, dtype=np.float32),
    np.asarray(y, dtype=np.float32),
    np.asarray(target_now, dtype=np.float32),
  )


def feature_dim(history: int, *, use_ema: bool) -> int:
  return 3 * history + (2 if use_ema else 0)


class ActuatorNet(nn.Module):
  """tau_actual = f(tau_target history, velocity history, target-velocity history[,
  EMA-of-tau_target^2 features]) -- predicts the actual torque value directly.

  See module docstring for the exact feature layout (build_dataset) and
  actuator_net_meta.json for the trained normalization stats / hyperparameters.
  """

  def __init__(self, in_dim: int, hidden: int, **_unused):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, hidden), nn.Softsign(),
      nn.Linear(hidden, hidden), nn.Softsign(),
      nn.Linear(hidden, 1),
    )

  def forward(self, x):
    return self.net(x).squeeze(-1)
