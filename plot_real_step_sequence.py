#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro


@dataclass(frozen=True)
class PlotRealStepSequenceConfig:
  csv_files: list[str] = field(default_factory=list)
  dpi: int = 160
  unwrap_pole_angle: bool = True
  clean_motor: bool = True
  max_valid_motor_abs_deg: float = 500.0
  max_valid_motor_vel_deg_s: float = 2000.0
  max_valid_motor_step_deg: float = 120.0


def _read_csv(path: Path) -> dict[str, np.ndarray]:
  rows: dict[str, list[float]] = {}
  with path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      for key, value in row.items():
        rows.setdefault(key, [])
        rows[key].append(float(value) if value != "" else float("nan"))
  return {key: np.asarray(values, dtype=float) for key, values in rows.items()}


def _unwrap_deg(values: np.ndarray) -> np.ndarray:
  return np.degrees(np.unwrap(np.radians(values)))


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
  values = values.astype(float).copy()
  valid = np.isfinite(values)
  if valid.all() or not valid.any():
    return values
  x = np.arange(len(values))
  values[~valid] = np.interp(x[~valid], x[valid], values[valid])
  return values


def _clean_motor_angle(
  motor_angle: np.ndarray,
  motor_velocity: np.ndarray,
  *,
  max_abs_deg: float,
  max_vel_deg_s: float,
  max_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
  clean = motor_angle.astype(float).copy()
  invalid = (
    ~np.isfinite(clean)
    | ~np.isfinite(motor_velocity)
    | (np.abs(clean) > float(max_abs_deg))
    | (np.abs(motor_velocity) > float(max_vel_deg_s))
  )

  diffs = np.abs(np.diff(clean, prepend=clean[0]))
  invalid |= diffs > float(max_step_deg)
  clean[invalid] = np.nan
  return _interpolate_nans(clean), invalid


def _plot_one(
  csv_path: Path,
  *,
  dpi: int,
  unwrap_pole_angle: bool,
  clean_motor: bool,
  max_valid_motor_abs_deg: float,
  max_valid_motor_vel_deg_s: float,
  max_valid_motor_step_deg: float,
) -> Path:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
  except ImportError as exc:
    raise RuntimeError(
      "matplotlib이 필요합니다. uv add matplotlib 또는 pip install matplotlib"
    ) from exc

  data = _read_csv(csv_path)
  t = data["time_s"]
  png_path = csv_path.with_name(f"{csv_path.stem}_clean.png")

  raw_motor_angle = data["motor_angle_deg"]
  motor_angle = raw_motor_angle
  pole_angle = data["pole_angle_deg"]
  invalid_motor = np.zeros_like(raw_motor_angle, dtype=bool)
  if clean_motor:
    motor_angle, invalid_motor = _clean_motor_angle(
      raw_motor_angle,
      data["motor_vel_deg_s"],
      max_abs_deg=float(max_valid_motor_abs_deg),
      max_vel_deg_s=float(max_valid_motor_vel_deg_s),
      max_step_deg=float(max_valid_motor_step_deg),
    )
  if unwrap_pole_angle:
    pole_angle = _unwrap_deg(pole_angle)

  tracking_error = data["target_angle_deg"] - motor_angle

  fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)

  axes[0].plot(t, data["target_angle_deg"], label="target_angle_deg", linestyle="--")
  axes[0].plot(t, raw_motor_angle, label="raw_motor_angle_deg", alpha=0.18)
  axes[0].plot(t, motor_angle, label="clean_motor_angle_deg")
  if invalid_motor.any():
    axes[0].scatter(
      t[invalid_motor],
      raw_motor_angle[invalid_motor],
      s=8,
      label="masked samples",
      alpha=0.35,
    )
  axes[0].set_ylabel("angle [deg]")
  axes[0].set_title(f"Real motor target vs actual - {csv_path.stem}")
  axes[0].grid(True)
  axes[0].legend()

  motor_vel = np.gradient(motor_angle, t)
  axes[1].plot(t, data["motor_vel_deg_s"], label="raw_motor_vel_deg_s", alpha=0.28)
  axes[1].plot(t, motor_vel, label="clean_motor_vel_deg_s")
  axes[1].axhline(0.0, linewidth=1.0, color="black")
  axes[1].set_ylabel("velocity [deg/s]")
  axes[1].set_title("Real motor velocity")
  axes[1].grid(True)
  axes[1].legend()

  axes[2].plot(t, pole_angle, label="pole_angle_deg", color="tab:green")
  axes[2].axhline(0.0, linewidth=1.0, color="black")
  axes[2].set_ylabel("angle [deg]")
  axes[2].set_title("Real pendulum angle")
  axes[2].grid(True)
  axes[2].legend()

  axes[3].plot(t, data["pole_vel_deg_s"], label="pole_vel_deg_s", color="tab:purple")
  axes[3].axhline(0.0, linewidth=1.0, color="black")
  axes[3].set_ylabel("velocity [deg/s]")
  axes[3].set_title("Real pendulum velocity")
  axes[3].grid(True)
  axes[3].legend()

  axes[4].plot(t, tracking_error, label="target - motor")
  axes[4].plot(t, data["motor_torque_nm"], label="motor_torque_nm", alpha=0.8)
  axes[4].axhline(0.0, linewidth=1.0, color="black")
  axes[4].set_ylabel("deg / Nm")
  axes[4].set_xlabel("time [s]")
  axes[4].set_title("Real tracking error and motor torque")
  axes[4].grid(True)
  axes[4].legend()

  for ax in axes:
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="both", labelbottom=True)
    ax.grid(True, which="major", axis="x", alpha=0.55)
    ax.grid(True, which="minor", axis="x", alpha=0.18)

  fig.tight_layout()
  fig.savefig(png_path, dpi=int(dpi))
  plt.close(fig)
  print(f"PNG 저장 완료: {png_path}")
  return png_path


def run(cfg: PlotRealStepSequenceConfig) -> list[Path]:
  if not cfg.csv_files:
    raise ValueError("csv_files를 하나 이상 넣어주세요.")

  csv_paths = [Path(path) for path in cfg.csv_files]
  for path in csv_paths:
    if not path.exists():
      raise FileNotFoundError(f"CSV not found: {path}")

  return [
    _plot_one(
      path,
      dpi=int(cfg.dpi),
      unwrap_pole_angle=bool(cfg.unwrap_pole_angle),
      clean_motor=bool(cfg.clean_motor),
      max_valid_motor_abs_deg=float(cfg.max_valid_motor_abs_deg),
      max_valid_motor_vel_deg_s=float(cfg.max_valid_motor_vel_deg_s),
      max_valid_motor_step_deg=float(cfg.max_valid_motor_step_deg),
    )
    for path in csv_paths
  ]


if __name__ == "__main__":
  run(tyro.cli(PlotRealStepSequenceConfig))
