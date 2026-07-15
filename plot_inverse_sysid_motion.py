#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro


@dataclass(frozen=True)
class PlotInverseSysidMotionConfig:
  csv_files: list[str] = field(default_factory=list)
  dpi: int = 160
  wrap_pole_angle: bool = True


def _wrap180(deg: np.ndarray) -> np.ndarray:
  return (deg + 180.0) % 360.0 - 180.0


def _read_csv(path: Path) -> dict[str, np.ndarray]:
  numeric: dict[str, list[float]] = {}
  text: dict[str, list[str]] = {}
  with path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      for key, value in row.items():
        try:
          numeric.setdefault(key, []).append(float(value))
        except ValueError:
          text.setdefault(key, []).append(value)
  data = {key: np.asarray(values, dtype=float) for key, values in numeric.items()}
  for key, values in text.items():
    data[key] = np.asarray(values, dtype=object)
  return data


def _label(path: Path, data: dict[str, np.ndarray]) -> str:
  if "profile_name" in data and len(data["profile_name"]) > 0:
    return f"{path.stem} ({data['profile_name'][0]})"
  return path.stem


def run(cfg: PlotInverseSysidMotionConfig) -> list[Path]:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
  except ImportError as exc:
    raise RuntimeError(
      "matplotlib이 필요합니다. uv add matplotlib 또는 pip install matplotlib"
    ) from exc

  if not cfg.csv_files:
    raise ValueError("csv_files를 하나 이상 넣어주세요.")

  csv_paths = [Path(path) for path in cfg.csv_files]
  missing = [path for path in csv_paths if not path.exists()]
  if missing:
    raise FileNotFoundError(f"CSV not found: {missing[0]}")

  output_paths: list[Path] = []
  for csv_path in csv_paths:
    data = _read_csv(csv_path)
    t = data["time_s"]
    png_path = csv_path.with_suffix(".png")
    label = _label(csv_path, data)

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)

    axes[0].plot(t, data["target_deg"], label="target_deg", linestyle="--")
    axes[0].plot(t, data["cylinder_q_deg"], label="cylinder_q_deg")
    axes[0].set_ylabel("angle [deg]")
    axes[0].set_title(f"Revolute 3 target vs actual - {label}")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, data["target_vel_deg_s"], label="target_vel_deg_s", linestyle="--")
    axes[1].plot(t, data["cylinder_qd_deg_s"], label="cylinder_qd_deg_s")
    axes[1].axhline(0.0, linewidth=1.0, color="black")
    axes[1].set_ylabel("velocity [deg/s]")
    axes[1].set_title("Revolute 3 velocity")
    axes[1].grid(True)
    axes[1].legend()

    pole_angle = _wrap180(data["pole_q_deg"]) if cfg.wrap_pole_angle else data["pole_q_deg"]
    pole_label = "pole_q_deg_wrap180" if cfg.wrap_pole_angle else "pole_q_deg"
    axes[2].plot(t, pole_angle, label=pole_label)
    if not cfg.wrap_pole_angle:
      axes[2].plot(t, data["pole_lift_deg"], label="pole_lift_deg", alpha=0.7)
      axes[2].axhline(180.0, linewidth=1.0, linestyle=":")
    else:
      axes[2].set_ylim(-190.0, 190.0)
    axes[2].axhline(0.0, linewidth=1.0, color="black")
    axes[2].set_ylabel("angle [deg]")
    axes[2].set_title("Pendulum angle wrapped [-180, 180]" if cfg.wrap_pole_angle else "Pendulum angle")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t, data["pole_qd_deg_s"], label="pole_qd_deg_s")
    axes[3].axhline(0.0, linewidth=1.0, color="black")
    axes[3].set_ylabel("velocity [deg/s]")
    axes[3].set_title("Pendulum velocity")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(t, data["tracking_error_deg"], label="target - cylinder")
    axes[4].plot(t, data["qfrc_actuator_nm"], label="qfrc_actuator_nm", alpha=0.75)
    axes[4].plot(t, data["actuator_force_nm"], label="actuator_force_nm", alpha=0.55)
    axes[4].axhline(0.0, linewidth=1.0, color="black")
    axes[4].set_ylabel("deg / Nm")
    axes[4].set_xlabel("time [s]")
    axes[4].set_title("Tracking error and actuator torque")
    axes[4].grid(True)
    axes[4].legend()

    for ax in axes:
      ax.xaxis.set_major_locator(MultipleLocator(1.0))
      ax.xaxis.set_minor_locator(MultipleLocator(0.1))
      ax.tick_params(axis="x", which="both", labelbottom=True)
      ax.grid(True, which="major", axis="x", alpha=0.55)
      ax.grid(True, which="minor", axis="x", alpha=0.18)

    fig.tight_layout()
    fig.savefig(png_path, dpi=int(cfg.dpi))
    plt.close(fig)
    print(f"PNG 저장 완료: {png_path}")
    output_paths.append(png_path)

  return output_paths


if __name__ == "__main__":
  run(tyro.cli(PlotInverseSysidMotionConfig))
