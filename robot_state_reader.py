#!/usr/bin/env python3
"""역진자 상태 읽기 라이브러리.

monitor.py(RobStride MIT 모드 poll/passive 디코딩 로직)를 클래스로 감싸고,
피코(PIO 쿼드러처 인코더)의 진자각도 시리얼 읽기를 더해, 둘을 하나의
RobotStateReader로 통합했다.

구성:
  - MotorStateReader   : CAN(MIT 모드)으로 액추에이터 각도/속도/토크를 백그라운드에서 계속 읽음
  - EncoderStateReader : 피코가 시리얼로 계속 출력하는 진자각도를 백그라운드에서 계속 읽음
  - RobotStateReader   : 위 둘을 묶어서 하나의 스냅샷(get_state())으로 제공

사용 예:
    from robot_state_reader import RobotStateReader

    reader = RobotStateReader(
        motor_id=8,
        can_channel="can0",
        motor_mode="passive",     # 다른 프로세스가 이미 MIT 명령을 보내고 있을 때
        encoder_port="/dev/ttyACM0",
    )
    reader.start()
    try:
        while True:
            state = reader.get_state()
            print(state)
            time.sleep(0.02)
    finally:
        reader.stop()

    # 또는 context manager로:
    with RobotStateReader(motor_id=8, encoder_port="/dev/ttyACM0") as reader:
        state = reader.get_state()

단독 실행하면 monitor.py의 CLI와 비슷하게 터미널에 값을 계속 찍어서 확인할 수 있다:
    python3 robot_state_reader.py --channel can0 --motor-id 8 --encoder-port /dev/ttyACM0
"""
from __future__ import annotations

import argparse
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import can
import serial

from commission_motor import _open_bus, _shutdown_bus, _flush_bus, _matches_mit_reply, CAN_CMD_CLEAR_FAULT
from monitor import decode_mit_feedback

ANGLE_RE = re.compile(r"angle:\s*(-?\d+(?:\.\d+)?)")
VALUE_RE = re.compile(r"encoder_value:\s*(-?\d+)")


@dataclass
class MotorState:
    motor_id: int
    angle_rad: float
    angle_deg: float
    velocity_rad_s: float
    velocity_deg_s: float
    torque_nm: float
    temp_raw: int
    timestamp: float


@dataclass
class EncoderState:
    raw_count: int
    angle_deg: float
    timestamp: float


@dataclass
class RobotState:
    pendulum_angle_deg: Optional[float]
    motor_angle_deg: Optional[float]
    motor_velocity_deg_s: Optional[float]
    motor_torque_nm: Optional[float]
    timestamp: float


class MotorStateReader:
    """CAN(MIT 모드)으로 액추에이터 각도/속도/토크를 백그라운드 스레드에서 계속 읽는다.

    mode="poll"    : clear-fault ping을 주기적으로 보내 응답을 받는다.
                     단독으로 사용 가능하지만, 다른 프로세스가 이미 명령을 보내고
                     있는 상태에서 같이 쓰면 서로 방해될 수 있다.
    mode="passive" : 아무것도 보내지 않고 버스를 듣기만 한다. 다른 프로세스(제어
                     루프 등)가 이미 MIT 명령을 계속 보내고 있을 때 그 응답을
                     공짜로 관찰하는 용도. 이쪽이 기본 권장 모드.
    """

    def __init__(
        self,
        motor_id: int,
        interface: str = "socketcan",
        channel: str = "can0",
        mode: str = "passive",
        rate_hz: float = 50.0,
        pmax: float = 12.57,
        vmax: float = 50.0,
        tmax: float = 6.0,
    ):
        if mode not in ("poll", "passive"):
            raise ValueError("mode must be 'poll' or 'passive'")
        self.motor_id = motor_id
        self.interface = interface
        self.channel = channel
        self.mode = mode
        self.rate_hz = rate_hz
        self.pmax = pmax
        self.vmax = vmax
        self.tmax = tmax

        self._bus = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[MotorState] = None
        self._lock = threading.Lock()

    @property
    def latest(self) -> Optional[MotorState]:
        with self._lock:
            return self._latest

    def start(self) -> None:
        self._bus = _open_bus(self.interface, self.channel)
        _flush_bus(self._bus)
        self._running = True
        target = self._poll_loop if self.mode == "poll" else self._passive_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._bus is not None:
            try:
                _shutdown_bus(self._bus)
            except Exception:
                pass

    def _update(self, msg) -> None:
        fb = decode_mit_feedback(msg, pmax=self.pmax, vmax=self.vmax, tmax=self.tmax)
        if fb is None:
            return
        state = MotorState(
            motor_id=fb["motor_id"],
            angle_rad=fb["angle_rad"],
            angle_deg=fb["angle_deg"],
            velocity_rad_s=fb["velocity_rad_s"],
            velocity_deg_s=fb["velocity_deg_s"],
            torque_nm=fb["torque_nm"],
            temp_raw=fb["temp_raw"],
            timestamp=time.monotonic(),
        )
        with self._lock:
            self._latest = state

    def _poll_loop(self) -> None:
        period_s = 1.0 / self.rate_hz
        data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
        while self._running:
            t0 = time.monotonic()
            msg = can.Message(arbitration_id=int(self.motor_id), data=data, is_extended_id=False)
            try:
                self._bus.send(msg)
                reply = self._bus.recv(timeout=max(period_s * 0.9, 0.005))
            except Exception:
                continue
            if reply is not None and _matches_mit_reply(reply, motor_id=self.motor_id):
                self._update(reply)
            sleep_left = period_s - (time.monotonic() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)

    def _passive_loop(self) -> None:
        while self._running:
            try:
                msg = self._bus.recv(timeout=0.2)
            except Exception:
                continue
            if msg is None:
                continue
            if not _matches_mit_reply(msg, motor_id=self.motor_id):
                continue
            self._update(msg)


class EncoderStateReader:
    """피코(PIO 쿼드러처 인코더)가 시리얼로 계속 출력하는 진자각도를 백그라운드에서 읽는다.

    피코 쪽 출력 포맷은 "encoder_value: X  angle: Y" 를 그대로 기대한다
    (pendulum_encoder_pio.py의 출력 포맷과 호환).
    """

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
            state = EncoderState(raw_count=raw, angle_deg=angle_deg, timestamp=time.monotonic())
            with self._lock:
                self._latest = state


class RobotStateReader:
    """진자 엔코더 + 액추에이터(CAN MIT 모드) 상태를 하나로 묶어 제공하는 통합 클래스."""

    def __init__(
        self,
        motor_id: int,
        encoder_port: Optional[str] = None,
        can_interface: str = "socketcan",
        can_channel: str = "can0",
        motor_mode: str = "passive",
        motor_rate_hz: float = 50.0,
        pmax: float = 12.57,
        vmax: float = 50.0,
        tmax: float = 6.0,
        encoder_baud: int = 115200,
    ):
        self.motor_reader = MotorStateReader(
            motor_id=motor_id,
            interface=can_interface,
            channel=can_channel,
            mode=motor_mode,
            rate_hz=motor_rate_hz,
            pmax=pmax,
            vmax=vmax,
            tmax=tmax,
        )
        self.encoder_reader: Optional[EncoderStateReader] = (
            EncoderStateReader(encoder_port, encoder_baud) if encoder_port else None
        )

    def start(self) -> None:
        self.motor_reader.start()
        if self.encoder_reader is not None:
            self.encoder_reader.start()

    def stop(self) -> None:
        self.motor_reader.stop()
        if self.encoder_reader is not None:
            self.encoder_reader.stop()

    def __enter__(self) -> "RobotStateReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def get_state(self) -> RobotState:
        motor = self.motor_reader.latest
        enc = self.encoder_reader.latest if self.encoder_reader is not None else None
        return RobotState(
            pendulum_angle_deg=enc.angle_deg if enc is not None else None,
            motor_angle_deg=motor.angle_deg if motor is not None else None,
            motor_velocity_deg_s=motor.velocity_deg_s if motor is not None else None,
            motor_torque_nm=motor.torque_nm if motor is not None else None,
            timestamp=time.monotonic(),
        )


def _demo() -> int:
    ap = argparse.ArgumentParser(description="RobotStateReader 데모/CLI")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--motor-id", type=int, required=True)
    ap.add_argument("--motor-mode", choices=["poll", "passive"], default="passive")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--pmax", type=float, default=12.57)
    ap.add_argument("--vmax", type=float, default=50.0)
    ap.add_argument("--tmax", type=float, default=6.0)
    ap.add_argument("--encoder-port", default=None)
    ap.add_argument("--encoder-baud", type=int, default=115200)
    ap.add_argument("--print-hz", type=float, default=5.0)
    args = ap.parse_args()

    reader = RobotStateReader(
        motor_id=args.motor_id,
        encoder_port=args.encoder_port,
        can_interface=args.interface,
        can_channel=args.channel,
        motor_mode=args.motor_mode,
        motor_rate_hz=args.rate,
        pmax=args.pmax,
        vmax=args.vmax,
        tmax=args.tmax,
        encoder_baud=args.encoder_baud,
    )
    reader.start()

    if args.encoder_port is None:
        print("[알림] --encoder-port 미지정: 진자각도 없이 액추에이터 값만 출력합니다.")
    print(f"[알림] motor_mode={args.motor_mode}, motor_id={args.motor_id}, "
          f"channel={args.channel}, print_rate={args.print_hz}Hz")
    print("RobotStateReader 시작 (Ctrl+C로 종료)...\n")

    def _fmt(value, unit: str, width: int = 8) -> str:
        if value is None:
            return f"{'N/A':>{width}}"
        return f"{value:+{width}.2f}{unit}"

    period = 1.0 / args.print_hz
    last_motor_ts = None
    last_enc_ts = None
    try:
        while True:
            state = reader.get_state()
            motor = reader.motor_reader.latest
            enc = reader.encoder_reader.latest if reader.encoder_reader is not None else None

            # 값이 실제로 갱신되고 있는지(스테일 여부) 같이 보여준다
            motor_fresh = "" if motor is None else (" (stale!)" if motor.timestamp == last_motor_ts else "")
            enc_fresh = "" if enc is None else (" (stale!)" if enc.timestamp == last_enc_ts else "")
            last_motor_ts = motor.timestamp if motor is not None else last_motor_ts
            last_enc_ts = enc.timestamp if enc is not None else last_enc_ts

            print(
                f"pendulum={_fmt(state.pendulum_angle_deg, 'deg')}{enc_fresh}   "
                f"motor_angle={_fmt(state.motor_angle_deg, 'deg')}   "
                f"motor_vel={_fmt(state.motor_velocity_deg_s, 'deg/s')}   "
                f"motor_torque={_fmt(state.motor_torque_nm, 'Nm')}{motor_fresh}"
            )
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n종료.")
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
