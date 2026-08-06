"""
can_bus.py

CAN transport + Robstride MIT-protocol codec for one leg's motors on one
CAN channel. Knows motor ids, raw degrees, and CAN bytes -- nothing about
joints, sides, or command safety (that's robot_leg.py's job).
"""

from __future__ import annotations

import importlib.util
import math
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from motor_config import (
    CAN_CMD_CLEAR_FAULT,
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_ZERO,
    MotorInfo,
)

_can_available = importlib.util.find_spec("can") is not None

if TYPE_CHECKING or _can_available:
    import can
else:
    # Minimal fallback so RobstrideBus can still be instantiated with an
    # injected mock bus when python-can is absent.
    class _MockMessage:
        def __init__(self, *, arbitration_id: int, data: bytes, is_extended_id: bool = False):
            self.arbitration_id = int(arbitration_id)
            self.data = bytearray(data)
            self.is_extended_id = bool(is_extended_id)

    can = SimpleNamespace(Message=_MockMessage)

_POSITION_BITS = 16
_VELOCITY_BITS = 12
_TORQUE_BITS = 12


@dataclass
class MotorState:
    position_deg: float = 0.0
    velocity_deg_s: float = 0.0
    torque_nm: float = 0.0
    temp_c: float = 0.0
    stamp: float = 0.0  # time.time() of last update; 0.0 = never received

    @property
    def is_valid(self) -> bool:
        return self.stamp > 0.0


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(x_min, min(x_max, x))
    span = x_max - x_min
    return int(((x - x_min) / span) * ((1 << bits) - 1)) if span > 0 else 0


def _uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    return (float(x) / ((1 << bits) - 1)) * span + x_min if span > 0 else x_min


def _pack_command(
    motor: MotorInfo,
    *,
    position_deg: float,
    velocity_deg_s: float,
    kp: float,
    kd: float,
    torque_nm: float,
) -> bytes:
    """Encode one MIT-mode command frame (8 bytes)."""
    pos_rad = math.radians(position_deg)
    vel_rad = math.radians(velocity_deg_s)
    q = _float_to_uint(pos_rad, -motor.pmax_rad, motor.pmax_rad, _POSITION_BITS)
    dq = _float_to_uint(vel_rad, -motor.vmax_rad_s, motor.vmax_rad_s, _VELOCITY_BITS)
    kp_u = _float_to_uint(kp, 0.0, 500.0, 12)
    kd_u = _float_to_uint(kd, 0.0, 5.0, 12)
    tau_u = _float_to_uint(torque_nm, -motor.tmax_nm, motor.tmax_nm, _TORQUE_BITS)

    data = [0] * 8
    data[0] = (q >> 8) & 0xFF
    data[1] = q & 0xFF
    data[2] = dq >> 4
    data[3] = ((dq & 0xF) << 4) | ((kp_u >> 8) & 0xF)
    data[4] = kp_u & 0xFF
    data[5] = kd_u >> 4
    data[6] = ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF)
    data[7] = tau_u & 0xFF
    return bytes(data)


def _decode_state(motor: MotorInfo, data: bytes) -> MotorState:
    """Decode one MIT-mode feedback frame (8 bytes)."""
    q = (data[1] << 8) | data[2]
    dq = (data[3] << 4) | (data[4] >> 4)
    tau = ((data[4] & 0x0F) << 8) | data[5]
    t_raw = (data[6] << 8) | data[7]

    pos_rad = _uint_to_float(q, -motor.pmax_rad, motor.pmax_rad, _POSITION_BITS)
    vel_rad = _uint_to_float(dq, -motor.vmax_rad_s, motor.vmax_rad_s, _VELOCITY_BITS)
    tau_nm = _uint_to_float(tau, -motor.tmax_nm, motor.tmax_nm, _TORQUE_BITS)

    return MotorState(
        position_deg=math.degrees(pos_rad),
        velocity_deg_s=math.degrees(vel_rad),
        torque_nm=tau_nm,
        temp_c=t_raw / 10.0,
        stamp=time.time(),
    )


class RobstrideBus:
    """One CAN channel talking MIT protocol to a fixed set of motors.

    Pass `bus=` an already-constructed python-can-compatible object (real
    `can.interface.Bus(...)` or a mock with `.send()`/`.recv()`/`.shutdown()`)
    to avoid opening real hardware.
    """

    def __init__(
        self,
        motors: Dict[int, MotorInfo],
        *,
        channel: str = "can0",
        interface: str = "socketcan",
        bus: Optional[object] = None,
        recv_timeout_s: float = 0.001,
    ):
        self.motors = motors
        self.bus = bus if bus is not None else can.interface.Bus(interface=interface, channel=channel)
        self.recv_timeout_s = float(recv_timeout_s)
        self._states: Dict[int, MotorState] = {mid: MotorState() for mid in motors}
        self._lock = threading.Lock()

    def disconnect(self) -> None:
        try:
            self.bus.shutdown()
        except Exception:
            pass

    # -------------------------
    # Commands
    # -------------------------
    def enable(self, motor_id: int) -> bool:
        return self._send_cmd_frame(motor_id, CAN_CMD_ENABLE)

    def disable(self, motor_id: int) -> bool:
        # disable is only a well-defined transition out of ENABLED -- sent
        # bare (e.g. right after a run of request_state() clear-fault
        # polling) it can leave the motor stuck with damping still active,
        # confirmed on hardware. Force a known ENABLED state first, then
        # disable out of it; the brief torque engagement this causes is
        # expected and harmless.
        self._send_cmd_frame(motor_id, CAN_CMD_ENABLE)
        return self._send_cmd_frame(motor_id, CAN_CMD_DISABLE)

    def zero(self, motor_id: int) -> bool:
        return self._send_cmd_frame(motor_id, CAN_CMD_ZERO)

    def enable_all(self) -> Dict[int, bool]:
        return {mid: self.enable(mid) for mid in self.motors}

    def disable_all(self) -> Dict[int, bool]:
        return {mid: self.disable(mid) for mid in self.motors}

    def check_connectivity(self, *, attempts: int = 3) -> Dict[int, bool]:
        """Ping every configured motor individually (CLEAR_FAULT) and report
        whether each one replied. Does not enable torque."""
        ok: Dict[int, bool] = {}
        for mid in self.motors:
            replied = False
            for _ in range(max(1, attempts)):
                self._send(mid, bytes([0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]))
                if mid in self._drain_rx():
                    replied = True
                    break
            ok[mid] = replied
        return ok

    def send_command(
        self,
        motor_id: int,
        *,
        position_deg: float,
        velocity_deg_s: float = 0.0,
        kp: float,
        kd: float,
        torque_nm: float = 0.0,
    ) -> None:
        motor = self.motors[motor_id]
        payload = _pack_command(
            motor,
            position_deg=position_deg,
            velocity_deg_s=velocity_deg_s,
            kp=kp,
            kd=kd,
            torque_nm=torque_nm,
        )
        self._send(motor_id, payload)
        self._drain_rx()

    def request_state(self, motor_ids: Optional[Iterable[int]] = None) -> Dict[int, MotorState]:
        ids = list(motor_ids) if motor_ids is not None else list(self.motors)
        for mid in ids:
            self._send(mid, bytes([0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]))
        self._drain_rx()
        with self._lock:
            return {mid: self._states[mid] for mid in ids}

    def states(self) -> Dict[int, MotorState]:
        with self._lock:
            return dict(self._states)

    # -------------------------
    # Internals
    # -------------------------
    def _send_cmd_frame(self, motor_id: int, cmd_byte: int) -> bool:
        self._send(motor_id, bytes([0xFF] * 7 + [cmd_byte]))
        return len(self._drain_rx()) > 0

    def _send(self, motor_id: int, data: bytes) -> None:
        self.bus.send(can.Message(arbitration_id=motor_id, data=data, is_extended_id=False))

    def _drain_rx(self) -> List[int]:
        touched: List[int] = []
        while True:
            msg = self.bus.recv(timeout=self.recv_timeout_s)
            if msg is None:
                break
            mid = int(msg.data[0])
            motor = self.motors.get(mid)
            if motor is None:
                continue
            state = _decode_state(motor, bytes(msg.data))
            with self._lock:
                self._states[mid] = state
            touched.append(mid)
        return touched
