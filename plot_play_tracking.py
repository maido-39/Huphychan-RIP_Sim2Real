#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass(frozen=True)
class PlotPlayTrackingConfig:
  csv_file: str
  dpi: int = 160


def _read_csv(path: Path) -> dict[str, np.ndarray]:
  rows: dict[str, list[float]] = {}
  with path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      for key, value in row.items():
        rows.setdefault(key, [])
        rows[key].append(float(value))
  return {key: np.asarray(values, dtype=float) for key, values in rows.items()}


def run(cfg: PlotPlayTrackingConfig) -> Path:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise RuntimeError(
      "matplotlib이 필요합니다. uv add matplotlib 또는 pip install matplotlib"
    ) from exc

  csv_path = Path(cfg.csv_file)
  data = _read_csv(csv_path)
  t = data["time_s"]
  png_path = csv_path.with_suffix(".png")

  fig, axes = plt.subplots(7, 1, figsize=(13, 18), sharex=True)

  axes[0].plot(t, data["joint_pos_target_deg"], label="q_target_deg", linestyle="--")
  axes[0].plot(t, data["cylinder_q_post_deg"], label="q_post_deg")
  axes[0].plot(t, data["cylinder_q_pre_deg"], label="q_pre_deg", alpha=0.45)
  axes[0].set_ylabel("angle [deg]")
  axes[0].set_title("Revolute 3 target vs actual")
  axes[0].grid(True)
  axes[0].legend()

  axes[1].plot(t, data["tracking_error_post_deg"], label="post-step error")
  axes[1].plot(t, data["tracking_error_pre_deg"], label="pre-step error", alpha=0.5)
  axes[1].axhline(0.0, linewidth=1.0, color="black")
  axes[1].set_ylabel("error [deg]")
  axes[1].set_title("Tracking error")
  axes[1].grid(True)
  axes[1].legend()

  axes[2].plot(t, data["raw_action"], label="raw_action")
  axes[2].plot(t, data["processed_action_deg"] / 90.0, label="processed/90deg", alpha=0.7)
  axes[2].axhline(1.0, linewidth=1.0, linestyle=":")
  axes[2].axhline(-1.0, linewidth=1.0, linestyle=":")
  axes[2].axhline(0.0, linewidth=1.0, color="black")
  axes[2].set_ylabel("normalized")
  axes[2].set_title("Policy command")
  axes[2].grid(True)
  axes[2].legend()

  axes[3].plot(t, data["cylinder_qd_post_deg_s"], label="cylinder_qd_post_deg_s")
  axes[3].plot(t, data["pole_qd_post_deg_s"], label="pole_qd_post_deg_s")
  axes[3].set_ylabel("vel [deg/s]")
  axes[3].set_title("Joint velocities")
  axes[3].grid(True)
  axes[3].legend()

  if "qfrc_actuator_post_nm" in data:
    axes[4].plot(t, data["qfrc_actuator_post_nm"], label="qfrc_actuator_post_nm")
    axes[4].plot(t, data["actuator_force_post_nm"], label="actuator_force_post_nm", alpha=0.7)
    axes[4].axhline(0.0, linewidth=1.0, color="black")
    axes[4].set_ylabel("torque [Nm]")
    axes[4].set_title("Revolute 3 actuator torque")
    axes[4].grid(True)
    axes[4].legend()
  else:
    axes[4].text(0.5, 0.5, "No torque columns in CSV", ha="center", va="center")
    axes[4].set_title("Revolute 3 actuator torque")
    axes[4].grid(True)

  axes[5].plot(t, data["pole_lift_post_deg"], label="pole_lift_post_deg")
  axes[5].plot(t, data["pole_q_post_deg"], label="pole_q_post_deg", alpha=0.7)
  axes[5].axhline(180.0, linewidth=1.0, linestyle="--")
  axes[5].set_ylabel("pole [deg]")
  axes[5].set_title("Pole state")
  axes[5].grid(True)
  axes[5].legend()

  axes[6].plot(t, data["reward"], label="reward")
  axes[6].plot(t, data["done"], label="done", alpha=0.7)
  axes[6].set_xlabel("simulation time [s]")
  axes[6].set_title("Reward / resets")
  axes[6].grid(True)
  axes[6].legend()

  fig.tight_layout()
  fig.savefig(png_path, dpi=int(cfg.dpi))
  print(f"PNG 저장 완료: {png_path}")
  return png_path


if __name__ == "__main__":
  run(tyro.cli(PlotPlayTrackingConfig))
