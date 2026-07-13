#!/usr/bin/env python3
from __future__ import annotations

"""스텝 크기별 추종 성능(지연/정상상태 오차) 측정 스크립트.

1도부터 시작해서 지정한 크기까지 스텝 각도를 점점 키워가며 실린더 모터에
위치 명령을 보낸다. 각 스텝은 `hold_duration_s` 동안 유지한 뒤 `center_deg`로
돌아와 `rest_duration_s` 동안 쉬고, 다음(더 큰) 스텝으로 넘어간다.

스텝마다 다음을 계산해서 표로 출력한다:
  - rise_time_ms      : 스텝이 시작된 시점부터 목표각 근처(`settle_tol_deg`
                        이내)에 처음 도달하기까지 걸린 시간
  - reached           : hold 구간 안에 목표각에 도달했는지 여부
  - final_error_deg   : hold 구간 마지막 20%의 평균 오차(목표 - 실측)
  - overshoot_deg     : 목표를 넘어선 최대 초과량(있다면)

이를 통해 "몇 도부터 못 따라가기 시작하는지"와 "그 지연이 얼마나 되는지"를
직접 확인할 수 있다. 참고로 통신/상태읽기 지연(can_rtt_ms 등)은
`latency_test.py`에서 이미 측정했고, 그 값들은 스텝 크기와 무관하게
서브 밀리초~수 밀리초 수준이었다. 이 스크립트에서 재는 rise_time_ms은
그보다 훨씬 큰, 제어 대역폭(모터 토크/게인) 한계에 의한 지연이다.

이 스크립트는 진단 전용이라 `sysid_collect.py`/`run_policy_motor.py`/
`latency_test.py` 등 기존 파일은 건드리지 않는다.

예시 실행
```bash
uv run python src/mjlab/tasks/inverse/step_response_test.py \
  --motor-id 8 --channel can0 --encoder-port /dev/ttyACM0 \
  --step-max-deg 30 --hold-duration-s 2.0 --rest-duration-s 3.0
```
"""

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tyro

from mjlab.tasks.inverse.commission_motor import (
  _open_bus,
  _shutdown_bus,
  disable_mit,
  enable_mit,
  mit_position_command,
  set_zero_mit,
)
from mjlab.tasks.inverse.robot_state_reader import RobotStateReader


@dataclass(frozen=True)
class StepResponseConfig:
  motor_id: int = 8
  interface: str = "socketcan"
  channel: str = "can0"
  encoder_port: str | None = None
  encoder_baud: int = 115200

  control_hz: float = 200.0
  kp: float = 10.0
  kd: float = 0.5

  center_deg: float = 0.0
  step_min_deg: float = 20.0
  step_max_deg: float = 100.0
  step_increment_deg: float = 5.0

  hold_duration_s: float = 2.0
  rest_duration_s: float = 3.0
  settle_tol_deg: float = 0.5

  max_amplitude_deg: float = 180.0
  abort_on_torque_nm: float = 17.0
  max_runtime_s: float = 900.0
  start_with_zero_set: bool = False
  require_enable_ack: bool = True

  log_dir: str = "logs"
  plot: bool = True


_CSV_HEADER = [
  "time_s",
  "step_size_deg",
  "target_angle_deg",
  "motor_angle_deg",
  "pole_angle_deg",
  "loop_dt_ms",
  "can_rtt_ms",
  "motor_state_age_ms",
  "encoder_state_age_ms",
]


def _build_schedule(
  cfg: StepResponseConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(sample_times, target_angle_deg, step_size_deg) 배열을 만든다.

  각 스텝 크기마다 [rest(center_deg) -> hold(center_deg+step)] 순서로
  구간을 이어붙인다. step_size_deg는 rest 구간에서는 0, hold 구간에서는
  해당 스텝 크기(양수)로 기록되어, 로그만 보고도 구간을 구분할 수 있다.
  """
  step_sizes = np.arange(
    cfg.step_min_deg, cfg.step_max_deg + 1e-9, cfg.step_increment_deg
  )
  if abs(cfg.center_deg) + step_sizes.max() > cfg.max_amplitude_deg:
    raise ValueError(
      f"center_deg + step_max_deg ({cfg.center_deg + step_sizes.max():.1f}) "
      f"exceeds max_amplitude_deg={cfg.max_amplitude_deg}. "
      "Lower step_max_deg or raise max_amplitude_deg."
    )

  segments: list[tuple[float, float, float]] = []
  for s in step_sizes:
    segments.append((cfg.rest_duration_s, cfg.center_deg, 0.0))
    segments.append((cfg.hold_duration_s, cfg.center_deg + float(s), float(s)))

  total_duration = sum(d for d, _, _ in segments)
  n_samples = int(total_duration * cfg.control_hz) + 1
  sample_times = np.arange(n_samples) / cfg.control_hz
  target_deg = np.full(n_samples, cfg.center_deg)
  step_size_deg = np.zeros(n_samples)

  seg_start = 0.0
  for duration, tgt, s in segments:
    mask = (sample_times >= seg_start) & (sample_times < seg_start + duration)
    target_deg[mask] = tgt
    step_size_deg[mask] = s
    seg_start += duration

  return sample_times, target_deg, step_size_deg


def _wait_for_initial_state(reader: RobotStateReader, timeout_s: float = 3.0) -> None:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    state = reader.get_state()
    if (
      state.motor_angle_deg is not None
      and state.motor_velocity_deg_s is not None
      and state.pendulum_angle_deg is not None
    ):
      return
    time.sleep(0.01)
  raise RuntimeError(
    "Timed out waiting for RobotStateReader initial state. "
    "Check CAN motor feedback and encoder serial connection."
  )


def _print_summary(csv_path: str, settle_tol_deg: float) -> None:
  rows: list[dict[str, float]] = []
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      rows.append({k: float(v) for k, v in row.items()})

  if not rows:
    return

  t = np.array([r["time_s"] for r in rows])
  step_size = np.array([r["step_size_deg"] for r in rows])
  target = np.array([r["target_angle_deg"] for r in rows])
  motor = np.array([r["motor_angle_deg"] for r in rows])

  # step_size_deg > 0 인 연속 구간(=하나의 hold)을 찾는다.
  is_hold = step_size > 0.0
  edges = np.flatnonzero(np.diff(is_hold.astype(int)))
  starts = edges[is_hold[edges + 1]] + 1 if edges.size else np.array([], dtype=int)
  if is_hold[0]:
    starts = np.concatenate(([0], starts))
  ends = edges[~is_hold[edges + 1]] if edges.size else np.array([], dtype=int)
  if is_hold[-1]:
    ends = np.concatenate((ends, [len(is_hold) - 1]))

  print("\n=== 스텝 크기별 추종 성능 ===")
  header = (
    f"{'step_deg':>10}{'rise_time_ms':>14}{'reached':>10}"
    f"{'final_err_deg':>15}{'overshoot_deg':>15}"
  )
  print(header)

  first_unreached: float | None = None
  for start, end in zip(starts, ends, strict=True):
    seg_t = t[start : end + 1]
    seg_target = target[start : end + 1]
    seg_motor = motor[start : end + 1]
    step_deg = step_size[start]

    t0 = seg_t[0]
    tgt = seg_target[0]
    err = np.abs(seg_motor - tgt)

    reached_idx = np.flatnonzero(err <= settle_tol_deg)
    if reached_idx.size:
      rise_time_ms = (seg_t[reached_idx[0]] - t0) * 1000.0
      reached = True
    else:
      rise_time_ms = float("nan")
      reached = False
      if first_unreached is None:
        first_unreached = step_deg

    tail_n = max(1, len(seg_motor) // 5)
    final_error_deg = float(np.mean(seg_target[-tail_n:] - seg_motor[-tail_n:]))

    if tgt >= 0:
      overshoot_deg = float(max(0.0, seg_motor.max() - tgt))
    else:
      overshoot_deg = float(max(0.0, tgt - seg_motor.min()))

    rise_str = f"{rise_time_ms:.1f}" if reached else "N/A"
    print(
      f"{step_deg:>10.1f}{rise_str:>14}{'Y' if reached else 'N':>10}"
      f"{final_error_deg:>15.3f}{overshoot_deg:>15.3f}"
    )

  if first_unreached is not None:
    print(
      f"\n-> {settle_tol_deg}deg 이내로 hold 구간 안에 도달하지 못한 "
      f"첫 스텝 크기: {first_unreached:.1f}deg"
    )
  else:
    print(f"\n-> 모든 스텝이 hold 구간 안에 {settle_tol_deg}deg 이내로 도달함.")


def _plot_log(csv_path: str, settle_tol_deg: float) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print(
      "matplotlib이 설치되어 있지 않아 그래프를 건너뜁니다. (pip install matplotlib)"
    )
    return

  rows: dict[str, list[float]] = {name: [] for name in _CSV_HEADER}
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      for name in _CSV_HEADER:
        rows[name].append(float(row[name]))

  if not rows["time_s"]:
    print("로그가 비어 있어 그래프를 건너뜁니다.")
    return

  t = np.array(rows["time_s"])
  step_size = np.array(rows["step_size_deg"])
  target = np.array(rows["target_angle_deg"])
  motor = np.array(rows["motor_angle_deg"])

  fig, axes = plt.subplots(2, 1, figsize=(12, 8))

  axes[0].plot(t, target, label="target_angle_deg", linestyle="--")
  axes[0].plot(t, motor, label="motor_angle_deg")
  axes[0].set_xlabel("time (s)")
  axes[0].set_ylabel("cylinder angle (deg)")
  axes[0].legend()
  axes[0].grid(True)

  is_hold = step_size > 0.0
  edges = np.flatnonzero(np.diff(is_hold.astype(int)))
  starts = edges[is_hold[edges + 1]] + 1 if edges.size else np.array([], dtype=int)
  if is_hold[0]:
    starts = np.concatenate(([0], starts))
  ends = edges[~is_hold[edges + 1]] if edges.size else np.array([], dtype=int)
  if is_hold[-1]:
    ends = np.concatenate((ends, [len(is_hold) - 1]))

  step_degs = []
  rise_times_ms = []
  final_errors = []
  for start, end in zip(starts, ends, strict=True):
    seg_t = t[start : end + 1]
    seg_target = target[start : end + 1]
    seg_motor = motor[start : end + 1]
    err = np.abs(seg_motor - seg_target[0])
    reached_idx = np.flatnonzero(err <= settle_tol_deg)
    step_degs.append(step_size[start])
    rise_times_ms.append(
      (seg_t[reached_idx[0]] - seg_t[0]) * 1000.0 if reached_idx.size else np.nan
    )
    tail_n = max(1, len(seg_motor) // 5)
    final_errors.append(float(np.mean(seg_target[-tail_n:] - seg_motor[-tail_n:])))

  ax2 = axes[1]
  ax2.plot(step_degs, rise_times_ms, "o-", color="tab:red", label="rise_time_ms")
  ax2.set_xlabel("step size (deg)")
  ax2.set_ylabel("rise_time_ms", color="tab:red")
  ax2.tick_params(axis="y", labelcolor="tab:red")
  ax2.grid(True)

  ax3 = ax2.twinx()
  ax3.plot(step_degs, final_errors, "s--", color="tab:blue", label="final_error_deg")
  ax3.set_ylabel("final_error_deg", color="tab:blue")
  ax3.tick_params(axis="y", labelcolor="tab:blue")

  fig.tight_layout()
  png_path = os.path.splitext(csv_path)[0] + ".png"
  fig.savefig(png_path, dpi=150)
  print(f"그래프 저장 완료: {png_path}")


def run(cfg: StepResponseConfig) -> None:
  sample_times, targets_deg, step_size_deg = _build_schedule(cfg)
  n_samples = len(sample_times)
  duration_s = sample_times[-1]

  bus = _open_bus(cfg.interface, cfg.channel)
  reader = RobotStateReader(
    motor_id=cfg.motor_id,
    encoder_port=cfg.encoder_port,
    can_interface=cfg.interface,
    can_channel=cfg.channel,
    motor_mode="passive",
    motor_rate_hz=cfg.control_hz,
    encoder_baud=cfg.encoder_baud,
  )

  os.makedirs(cfg.log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = os.path.join(cfg.log_dir, f"step_response_{ts}.csv")
  csv_file = open(csv_path, "w", newline="")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(_CSV_HEADER)

  target_period_ms = 1000.0 / cfg.control_hz
  n_steps = (
    int(round((cfg.step_max_deg - cfg.step_min_deg) / cfg.step_increment_deg)) + 1
  )
  print(f"Opening motor control on {cfg.channel}, motor_id={cfg.motor_id}")
  print(
    f"step sweep {cfg.step_min_deg}deg -> {cfg.step_max_deg}deg "
    f"({n_steps} steps, hold={cfg.hold_duration_s}s, rest={cfg.rest_duration_s}s) "
    f"over {duration_s:.1f}s"
  )
  print(f"target loop period: {target_period_ms:.2f} ms ({cfg.control_hz:.0f} Hz)")
  print(f"CSV 로그: {csv_path}")

  start_t = time.monotonic()
  stopped_reason = "정상 종료(전체 스텝 완료)"
  prev_loop_start = start_t

  try:
    print("Starting RobotStateReader...")
    reader.start()

    if cfg.start_with_zero_set:
      print("Setting motor zero...")
      if not set_zero_mit(bus, cfg.motor_id):
        raise RuntimeError("set_zero_mit was not acknowledged")
      time.sleep(0.1)

    print("Enabling motor...")
    enable_ok = enable_mit(bus, cfg.motor_id)
    if cfg.require_enable_ack and not enable_ok:
      raise RuntimeError("enable_mit was not acknowledged")
    time.sleep(0.1)

    _wait_for_initial_state(reader)

    period_s = 1.0 / cfg.control_hz
    deadline = start_t + min(duration_s, cfg.max_runtime_s)

    print("Starting step response test loop.")
    sample_idx = 0
    while time.monotonic() < deadline and sample_idx < n_samples:
      loop_start = time.monotonic()
      loop_dt_ms = (loop_start - prev_loop_start) * 1000.0
      prev_loop_start = loop_start

      target_deg = float(targets_deg[sample_idx])
      step_deg = float(step_size_deg[sample_idx])

      t_send = time.monotonic()
      ok = mit_position_command(
        bus,
        cfg.motor_id,
        target_deg,
        kp=cfg.kp,
        kd=cfg.kd,
        torque_nm=0.0,
      )
      can_rtt_ms = (time.monotonic() - t_send) * 1000.0
      if not ok:
        raise RuntimeError("No MIT reply received from motor")

      now = time.monotonic()
      motor_latest = reader.motor_reader.latest
      encoder_latest = (
        reader.encoder_reader.latest if reader.encoder_reader is not None else None
      )

      if motor_latest is None:
        raise RuntimeError("Motor state is not available from RobotStateReader")
      if encoder_latest is None:
        raise RuntimeError(
          "Pendulum angle is not available from RobotStateReader. "
          "Provide --encoder-port and check the serial encoder stream."
        )

      motor_angle_deg = motor_latest.angle_deg
      pole_angle_deg = encoder_latest.angle_deg
      motor_state_age_ms = (now - motor_latest.timestamp) * 1000.0
      encoder_state_age_ms = (now - encoder_latest.timestamp) * 1000.0

      if (
        cfg.abort_on_torque_nm and abs(motor_latest.torque_nm) > cfg.abort_on_torque_nm
      ):
        raise RuntimeError(
          f"Aborting: |motor_torque_nm|={abs(motor_latest.torque_nm):.2f} "
          f"exceeded abort_on_torque_nm={cfg.abort_on_torque_nm}"
        )

      elapsed_t = now - start_t
      csv_writer.writerow(
        [
          f"{elapsed_t:.4f}",
          f"{step_deg:.3f}",
          f"{target_deg:.3f}",
          f"{motor_angle_deg:.3f}",
          f"{pole_angle_deg:.3f}",
          f"{loop_dt_ms:.3f}",
          f"{can_rtt_ms:.3f}",
          f"{motor_state_age_ms:.3f}",
          f"{encoder_state_age_ms:.3f}",
        ]
      )

      if sample_idx % 50 == 0:
        print(
          "t={:6.2f}s  step={:5.1f}deg  target={:+8.3f}deg  motor={:+8.3f}deg  "
          "can_rtt={:6.2f}ms".format(
            elapsed_t, step_deg, target_deg, motor_angle_deg, can_rtt_ms
          )
        )

      sample_idx += 1
      elapsed = time.monotonic() - loop_start
      sleep_s = period_s - elapsed
      if sleep_s > 0.0:
        time.sleep(sleep_s)

  except KeyboardInterrupt:
    stopped_reason = "사용자 강제종료(Ctrl+C)"
    print("\n" + stopped_reason)
  except RuntimeError as exc:
    stopped_reason = f"에러로 종료: {exc}"
    print("\n" + stopped_reason)

  finally:
    print("Disabling motor...")
    try:
      disable_mit(bus, cfg.motor_id)
    finally:
      reader.stop()
      _shutdown_bus(bus)
      csv_file.close()
      print(f"CSV 로그 저장 완료: {csv_path}")

  print(f"측정 종료 ({stopped_reason}).")

  _print_summary(csv_path, cfg.settle_tol_deg)

  if cfg.plot:
    _plot_log(csv_path, cfg.settle_tol_deg)


def main() -> None:
  cfg = tyro.cli(StepResponseConfig)
  run(cfg)


if __name__ == "__main__":
  main()
