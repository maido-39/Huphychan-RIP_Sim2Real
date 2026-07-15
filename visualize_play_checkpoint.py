#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from mjlab.tasks.inverse.log_play_tracking import (
  LogPlayTrackingConfig,
  run as run_log_play_tracking,
)
from mjlab.tasks.inverse.plot_play_tracking import (
  PlotPlayTrackingConfig,
  run as run_plot_play_tracking,
)


@dataclass(frozen=True)
class VisualizePlayCheckpointConfig:
  checkpoint_file: str
  duration_s: float = 8.0
  device: str | None = None
  with_disturbance: bool = False
  fixed_start: bool = False
  no_terminations: bool = False
  log_dir: str = "src/mjlab/tasks/inverse/play_tracking_logs"
  print_every: int = 10
  dpi: int = 160


def run(cfg: VisualizePlayCheckpointConfig) -> tuple[Path, Path]:
  csv_path = run_log_play_tracking(
    LogPlayTrackingConfig(
      checkpoint_file=cfg.checkpoint_file,
      duration_s=cfg.duration_s,
      device=cfg.device,
      with_disturbance=cfg.with_disturbance,
      fixed_start=cfg.fixed_start,
      no_terminations=cfg.no_terminations,
      log_dir=cfg.log_dir,
      print_every=cfg.print_every,
    )
  )
  png_path = run_plot_play_tracking(
    PlotPlayTrackingConfig(csv_file=str(csv_path), dpi=cfg.dpi)
  )
  print(f"[INFO] CSV: {csv_path}")
  print(f"[INFO] PNG: {png_path}")
  return csv_path, png_path


if __name__ == "__main__":
  run(tyro.cli(VisualizePlayCheckpointConfig))
