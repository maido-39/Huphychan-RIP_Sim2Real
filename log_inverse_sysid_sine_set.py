#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from mjlab.tasks.inverse.log_inverse_sysid_motion import (
  DEFAULT_LOG_DIR,
  DEFAULT_XML,
  LogInverseSysidMotionConfig,
  run as run_sysid_motion,
)


@dataclass(frozen=True)
class LogInverseSysidSineSetConfig:
  """Run sine motor-response sysid cases matching real logs.

  Each case is encoded as:
    amplitude_deg,frequency_hz,kp,kv

  The default cases match files such as:
    step_sequence_30_2.0hz_5_1.csv
  """

  cases: list[str] = field(
    default_factory=lambda: [
      "30,2.0,5,1",
      "30,2.0,10,1",
      "45,1.5,5,1",
      "60,1.0,5,1",
    ]
  )
  xml_file: str = str(DEFAULT_XML)
  log_dir: str = str(DEFAULT_LOG_DIR)
  duration_s: float = 20.0
  start_hold_s: float = 0.0
  sine_phase_deg: float = 0.0
  initial_pole_deg: float = 0.0
  ctrl_limit_deg: float = 720.0
  actuator_force_limit_nm: float | None = 8.0
  print_every: int = 100
  plot: bool = True
  plot_dpi: int = 160
  render: bool = False
  realtime: bool = True


def _parse_case(case: str) -> tuple[float, float, float, float]:
  parts = [part.strip() for part in case.split(",")]
  if len(parts) != 4:
    raise ValueError(
      f"Invalid case {case!r}. Use amplitude_deg,frequency_hz,kp,kv"
    )
  amp_deg, freq_hz, kp, kv = (float(part) for part in parts)
  if amp_deg <= 0.0:
    raise ValueError(f"amplitude_deg must be positive: {case!r}")
  if freq_hz <= 0.0:
    raise ValueError(f"frequency_hz must be positive: {case!r}")
  return amp_deg, freq_hz, kp, kv


def _case_tag(amp_deg: float, freq_hz: float, kp: float, kv: float) -> str:
  amp = f"{amp_deg:g}".replace(".", "p")
  freq = f"{freq_hz:g}".replace(".", "p")
  kp_s = f"{kp:g}".replace(".", "p")
  kv_s = f"{kv:g}".replace(".", "p")
  return f"kp{kp_s}_kv{kv_s}_amp{amp}_freq{freq}hz"


def run(cfg: LogInverseSysidSineSetConfig) -> list[Path]:
  csv_paths: list[Path] = []
  for case in cfg.cases:
    amp_deg, freq_hz, kp, kv = _parse_case(case)
    print(
      "[INFO] sine case "
      f"amp={amp_deg:g}deg freq={freq_hz:g}Hz kp={kp:g} kv={kv:g}"
    )
    motion_cfg = LogInverseSysidMotionConfig(
      profile_name="sine",
      xml_file=cfg.xml_file,
      log_dir=cfg.log_dir,
      csv_tag=_case_tag(amp_deg, freq_hz, kp, kv),
      start_hold_s=float(cfg.start_hold_s),
      sine_amplitude_deg=float(amp_deg),
      sine_frequency_hz=float(freq_hz),
      sine_duration_s=float(cfg.duration_s),
      sine_phase_deg=float(cfg.sine_phase_deg),
      initial_pole_deg=float(cfg.initial_pole_deg),
      ctrl_limit_deg=float(cfg.ctrl_limit_deg),
      actuator_kp=float(kp),
      actuator_kv=float(kv),
      actuator_force_limit_nm=cfg.actuator_force_limit_nm,
      print_every=int(cfg.print_every),
      plot=bool(cfg.plot),
      plot_dpi=int(cfg.plot_dpi),
      render=bool(cfg.render),
      realtime=bool(cfg.realtime),
    )
    csv_paths.extend(run_sysid_motion(motion_cfg))
  return csv_paths


if __name__ == "__main__":
  run(tyro.cli(LogInverseSysidSineSetConfig))
