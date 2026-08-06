"""
mock_bus.py

Fake CAN transport that plays the role of the motors themselves: decodes
MIT command frames sent to it, updates internal per-motor state, and
replies with MIT feedback frames. Pass an instance as RobstrideBus(bus=...)
to exercise the whole stack without real hardware.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

from can_bus import _float_to_uint, _uint_to_float, can
from motor_config import CAN_CMD_ZERO, MotorInfo

_RAD2DEG = 180.0 / math.pi
_DEG2RAD = math.pi / 180.0


@dataclass
class _FakeMotorState:
    position_deg: float = 0.0
    velocity_deg_s: float = 0.0
    torque_nm: float = 0.0
    temp_c: float = 30.0


class MockCanBus:
    """python-can-compatible fake: only .send()/.recv()/.shutdown() are used."""

    def __init__(
        self,
        motors: Dict[int, MotorInfo],
        *,
        initial_position_deg: Optional[Dict[int, float]] = None,
        default_temp_c: float = 30.0,
    ):
        self._motors = motors
        self._rx: "deque" = deque()
        initial_position_deg = initial_position_deg or {}
        self._state: Dict[int, _FakeMotorState] = {
            mid: _FakeMotorState(
                position_deg=float(initial_position_deg.get(mid, 0.0)),
                temp_c=float(default_temp_c),
            )
            for mid in motors
        }

    def send(self, msg) -> None:
        data = bytes(msg.data)
        if len(data) != 8:
            return
        mid = int(msg.arbitration_id)
        motor = self._motors.get(mid)
        if motor is None:
            return

        if all(b == 0xFF for b in data[:7]):
            cmd = int(data[7])
            if cmd == int(CAN_CMD_ZERO):
                self._state[mid].position_deg = 0.0
            self._reply(mid, motor)
            return

        # MIT command frame layout (no id byte -- id comes from arbitration_id):
        # pos(16) | vel(12)+kp_hi(4) | kp_lo(8) | kd(12)+tau_hi(4) | tau_lo(8)
        q = (data[0] << 8) | data[1]
        dq = (data[2] << 4) | (data[3] >> 4)
        tau = ((data[6] & 0x0F) << 8) | data[7]

        pos_rad = _uint_to_float(q, -motor.pmax_rad, motor.pmax_rad, 16)
        vel_rad = _uint_to_float(dq, -motor.vmax_rad_s, motor.vmax_rad_s, 12)
        tau_nm = _uint_to_float(tau, -motor.tmax_nm, motor.tmax_nm, 12)

        st = self._state[mid]
        st.position_deg = pos_rad * _RAD2DEG
        st.velocity_deg_s = vel_rad * _RAD2DEG
        st.torque_nm = tau_nm
        self._reply(mid, motor)

    def recv(self, timeout: Optional[float] = None):
        if timeout is not None and float(timeout) <= 0.0:
            return self._rx.popleft() if self._rx else None
        deadline = None if timeout is None else time.perf_counter() + float(timeout)
        while deadline is None or time.perf_counter() < deadline:
            if self._rx:
                return self._rx.popleft()
            time.sleep(0.0001)
        return None

    def shutdown(self) -> None:
        self._rx.clear()

    def _reply(self, mid: int, motor: MotorInfo) -> None:
        # MIT feedback frame layout: id(8) | pos(16) | vel(12) | tau(12) | temp(16)
        st = self._state[mid]
        q = _float_to_uint(st.position_deg * _DEG2RAD, -motor.pmax_rad, motor.pmax_rad, 16)
        dq = _float_to_uint(st.velocity_deg_s * _DEG2RAD, -motor.vmax_rad_s, motor.vmax_rad_s, 12)
        tau = _float_to_uint(st.torque_nm, -motor.tmax_nm, motor.tmax_nm, 12)
        temp_tenths = int(max(0, min(65535, round(st.temp_c * 10.0))))

        payload = [0] * 8
        payload[0] = mid & 0xFF
        payload[1] = (q >> 8) & 0xFF
        payload[2] = q & 0xFF
        payload[3] = (dq >> 4) & 0xFF
        payload[4] = ((dq & 0x0F) << 4) | ((tau >> 8) & 0x0F)
        payload[5] = tau & 0xFF
        payload[6] = (temp_tenths >> 8) & 0xFF
        payload[7] = temp_tenths & 0xFF
        self._rx.append(can.Message(arbitration_id=mid, data=bytes(payload), is_extended_id=False))
