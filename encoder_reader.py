#!/usr/bin/env python3
"""역진자(pole) 각도 인코더 읽기 라이브러리.

피코(PIO 쿼드러처 인코더)가 시리얼로 계속 출력하는
"encoder_value: X  angle: Y" 포맷을 백그라운드 스레드에서 읽어 최신 값을
제공한다. angle_deg는 0~360도 범위로 오는 wrap-around 값이므로, 이 값을
그대로 미분해서 각속도를 구하면 0/360 경계에서 튄다 — 각속도가 필요하면
PoleVelocityEstimator를 같이 쓴다.

robot_state_reader.py의 RobotStateReader도 이 모듈의 EncoderStateReader를
그대로 가져다 쓴다 (진자 인코더 읽기 로직은 여기 한 곳에만 있다).

사용 예:
    from mjlab.tasks.inverse.encoder_reader import (
        EncoderStateReader,
        PoleVelocityEstimator,
    )

    reader = EncoderStateReader("/dev/ttyACM2")
    velocity = PoleVelocityEstimator()
    reader.start()
    try:
        while True:
            state = reader.latest
            if state is not None:
                vel_deg_s = velocity.update(state.angle_deg, state.timestamp)
                print(state.angle_deg, vel_deg_s)
            time.sleep(0.02)
    finally:
        reader.stop()
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import serial

ANGLE_RE = re.compile(r"angle:\s*(-?\d+(?:\.\d+)?)")
VALUE_RE = re.compile(r"encoder_value:\s*(-?\d+)")


@dataclass
class EncoderState:
  raw_count: int
  angle_deg: float  # 0~360도 범위 (wrap-around).
  timestamp: float


class EncoderStateReader:
  """피코가 시리얼로 계속 출력하는 진자각도를 백그라운드에서 읽는다."""

  def __init__(self, port: str, baudrate: int = 115200):
    self.port = port
    self.baudrate = baudrate
    self._ser: Optional[serial.Serial] = None
    self._running = False
    self._thread: Optional[threading.Thread] = None
    self._latest: Optional[EncoderState] = None
    self._lock = threading.Lock()

  @property
  def latest(self) -> Optional[EncoderState]:
    with self._lock:
      return self._latest

  def start(self) -> None:
    self._ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
    # RP2040 USB CDC(TinyUSB)가 DTR 미설정 시 출력을 버리는 경우가 있어 명시적으로 세팅
    self._ser.dtr = True
    self._ser.rts = True
    # 포트를 열기 전부터 커널 수신 버퍼에 쌓여있던 줄들을 비운다. 안 비우면
    # 시작 직후 이 백로그를 한꺼번에(거의 dt=0으로) 읽어버려서, 실제로는 정상
    # 간격이던 샘플들 사이 dt가 비정상적으로 작게 찍히고 속도가 크게 튄다.
    self._ser.reset_input_buffer()
    self._running = True
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._running = False
    if self._thread:
      self._thread.join(timeout=1.0)
    if self._ser is not None:
      try:
        self._ser.close()
      except Exception:
        pass

  def __enter__(self) -> "EncoderStateReader":
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.stop()

  def _loop(self) -> None:
    while self._running:
      try:
        line = self._ser.readline().decode(errors="ignore").strip()
      except Exception:
        continue
      if not line:
        continue
      m_angle = ANGLE_RE.search(line)
      if m_angle is None:
        continue
      m_value = VALUE_RE.search(line)
      raw = int(m_value.group(1)) if m_value else 0
      angle_deg = float(m_angle.group(1))
      state = EncoderState(
        raw_count=raw, angle_deg=angle_deg, timestamp=time.monotonic()
      )
      with self._lock:
        self._latest = state


class PoleVelocityEstimator:
  """0~360도 wrap-around 각도 샘플에서 각속도(deg/s)를 추정한다.

  단순 (angle[n] - angle[n-1]) / dt는 0/360 경계를 넘을 때 튀므로, 두 샘플
  사이 최단 각도차(circular diff)를 시간차로 나눈다.

  호출 측(제어 루프)이 백그라운드 스레드가 갱신하는 것보다 더 자주 폴링하면,
  같은 EncoderState를 두 번 이상 연속으로 넘길 수 있다(timestamp가 직전과
  동일). 이건 실제로 속도가 0이 된 게 아니라 "이번엔 새 샘플이 없었다"는
  뜻이므로, 그 경우엔 0을 반환하지 않고 마지막으로 계산한 속도를 그대로
  반환한다 (진짜 정지는 새 timestamp에서 angle이 그대로일 때 자연히 0으로
  계산된다).

  위치를 유한 차분해서 속도를 구하는 방식은 태생적으로 노이즈에 민감하다
  (엔코더 분해능 + 샘플 간격 흔들림이 그대로 증폭됨). ema_alpha로 지수이동평균
  저역통과 필터를 걸 수 있다: filtered[n] = ema_alpha * raw[n]
  + (1 - ema_alpha) * filtered[n-1]. 기본값 1.0은 필터 없음(기존 동작과 동일)이고,
  1보다 작을수록 부드러워지는 대신 반응이 느려진다(정책 입력으로 쓸 때는
  지연이 생기므로 너무 작게 주지 않는 게 좋다).
  """

  def __init__(self, ema_alpha: float = 1.0) -> None:
    if not (0.0 < ema_alpha <= 1.0):
      raise ValueError("ema_alpha must be in (0, 1]")
    self._ema_alpha = ema_alpha
    self._prev_angle_deg: Optional[float] = None
    self._prev_timestamp: Optional[float] = None
    self._filtered_velocity_deg_s: float = 0.0

  def update(self, angle_deg: float, timestamp: float) -> float:
    """새 각도 샘플을 반영하고 (필터링된) 각속도(deg/s)를 반환한다. 첫 샘플은 0.0."""
    if self._prev_angle_deg is None or self._prev_timestamp is None:
      self._prev_angle_deg = angle_deg
      self._prev_timestamp = timestamp
      return 0.0

    dt = timestamp - self._prev_timestamp
    if dt <= 0.0:
      # 같은 샘플을 다시 받은 것 (새 데이터 없음) - 마지막 속도를 유지한다.
      return self._filtered_velocity_deg_s

    delta_deg = (angle_deg - self._prev_angle_deg + 180.0) % 360.0 - 180.0
    raw_velocity_deg_s = delta_deg / dt
    self._filtered_velocity_deg_s = (
      self._ema_alpha * raw_velocity_deg_s
      + (1.0 - self._ema_alpha) * self._filtered_velocity_deg_s
    )

    self._prev_angle_deg = angle_deg
    self._prev_timestamp = timestamp
    return self._filtered_velocity_deg_s


_LOG_HEADER = [
  "time_s",
  "angle_deg",
  "velocity_raw_deg_s",
  "velocity_filtered_deg_s",
]


def _write_log_csv(
  csv_path: str, rows: list[tuple[float, float, float, float]]
) -> None:
  with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(_LOG_HEADER)
    for t, angle_deg, vel_raw_deg_s, vel_filtered_deg_s in rows:
      writer.writerow(
        [
          f"{t:.6f}",
          f"{angle_deg:.4f}",
          f"{vel_raw_deg_s:.4f}",
          f"{vel_filtered_deg_s:.4f}",
        ]
      )


def _plot_log(csv_path: str) -> str:
  """위치/속도 CSV 로그를 읽어 각도(unwrap)/속도(raw vs 필터) 그래프를 png로 저장한다."""
  import matplotlib
  import numpy as np

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  time_s: list[float] = []
  angle_deg: list[float] = []
  velocity_raw_deg_s: list[float] = []
  velocity_filtered_deg_s: list[float] = []
  with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      time_s.append(float(row["time_s"]))
      angle_deg.append(float(row["angle_deg"]))
      velocity_raw_deg_s.append(float(row["velocity_raw_deg_s"]))
      velocity_filtered_deg_s.append(float(row["velocity_filtered_deg_s"]))

  # 0/360도 랩어라운드 경계에서 생기는 수직선을 없애기 위해 unwrap해서 그린다.
  angle_unwrapped = np.degrees(np.unwrap(np.radians(angle_deg)))

  fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
  axes[0].plot(time_s, angle_unwrapped, color="tab:green")
  axes[0].set_ylabel("encoder angle (deg)")

  axes[1].plot(
    time_s,
    velocity_raw_deg_s,
    color="tab:gray",
    lw=0.8,
    alpha=0.6,
    label="raw (ema_alpha=1.0)",
  )
  axes[1].plot(
    time_s,
    velocity_filtered_deg_s,
    color="tab:purple",
    lw=1.4,
    label="filtered",
  )
  axes[1].set_ylabel("velocity (deg/s)")
  axes[1].set_xlabel("time (s)")
  axes[1].legend()

  for ax in axes:
    ax.grid(True, alpha=0.3)
  fig.suptitle(Path(csv_path).name)
  fig.tight_layout()

  png_path = str(Path(csv_path).with_suffix(".png"))
  fig.savefig(png_path, dpi=150)
  plt.close(fig)
  return png_path


def _demo() -> int:
  ap = argparse.ArgumentParser(description="EncoderStateReader 단독 실행 데모")
  ap.add_argument("--port", required=True, help="예: /dev/ttyACM2")
  ap.add_argument("--baud", type=int, default=115200)
  ap.add_argument("--print-hz", type=float, default=10.0)
  ap.add_argument(
    "--duration-s",
    type=float,
    default=None,
    help=(
      "지정하면 이 시간(초) 후 자동 종료하고, 그동안의 각도/속도를 "
      "logs/ 아래 CSV+PNG로 저장한다 (예: --duration-s 10). "
      "생략하면 예전처럼 Ctrl+C로 끌 때까지 콘솔 출력만 한다."
    ),
  )
  ap.add_argument("--log-dir", default="logs")
  ap.add_argument("--no-plot", action="store_true", help="PNG 그래프 생성을 건너뛴다.")
  ap.add_argument(
    "--ema-alpha",
    type=float,
    default=1.0,
    help=(
      "속도 EMA 저역통과 필터 계수, (0, 1]. 1.0(기본값)은 필터 없음. "
      "작을수록 부드러워지지만 반응이 느려진다 (예: --ema-alpha 0.3)."
    ),
  )
  args = ap.parse_args()

  reader = EncoderStateReader(args.port, args.baud)
  console_velocity = PoleVelocityEstimator(ema_alpha=args.ema_alpha)
  log_velocity_raw = PoleVelocityEstimator(ema_alpha=1.0)
  log_velocity_filtered = PoleVelocityEstimator(ema_alpha=args.ema_alpha)

  print(f"Opening {args.port} @ {args.baud} baud (Ctrl+C로 종료)...")
  reader.start()

  logging_enabled = args.duration_s is not None
  log_rows: list[tuple[float, float, float, float]] = []
  last_logged_timestamp: Optional[float] = None

  start_t = time.monotonic()
  deadline = start_t + args.duration_s if logging_enabled else None
  last_print_t = 0.0
  print_period_s = 1.0 / args.print_hz

  try:
    while deadline is None or time.monotonic() < deadline:
      state = reader.latest
      now = time.monotonic()

      if state is None:
        if now - last_print_t >= print_period_s:
          print("  (아직 데이터 없음 - 시리얼 연결/포트/보드레이트를 확인하세요)")
          last_print_t = now
      else:
        # 로그는 print-hz에 상관없이, 실제로 새 샘플이 들어올 때마다 기록한다
        # (콘솔 출력 주기로 스로틀하면 대부분의 실측 샘플을 놓친다).
        if logging_enabled and state.timestamp != last_logged_timestamp:
          vel_raw = log_velocity_raw.update(state.angle_deg, state.timestamp)
          vel_filtered = log_velocity_filtered.update(state.angle_deg, state.timestamp)
          log_rows.append(
            (state.timestamp - start_t, state.angle_deg, vel_raw, vel_filtered)
          )
          last_logged_timestamp = state.timestamp

        if now - last_print_t >= print_period_s:
          vel_deg_s = console_velocity.update(state.angle_deg, state.timestamp)
          print(
            f"raw={state.raw_count:>8}  angle={state.angle_deg:+8.2f}deg  "
            f"vel={vel_deg_s:+9.2f}deg/s"
          )
          last_print_t = now

      time.sleep(0.001)
  except KeyboardInterrupt:
    print("\n종료.")
  finally:
    reader.stop()

  if logging_enabled:
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.log_dir, f"encoder_log_{timestamp}.csv")
    _write_log_csv(csv_path, log_rows)
    print(f"CSV 로그 저장 완료: {csv_path} ({len(log_rows)}개 샘플)")

    if not args.no_plot:
      try:
        png_path = _plot_log(csv_path)
        print(f"Plot 저장 완료: {png_path}")
      except Exception as exc:
        print(f"Plot 생성 실패: {exc}")

  return 0


if __name__ == "__main__":
  raise SystemExit(_demo())
