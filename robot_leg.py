"""
robot_leg.py

One physical leg -- joint-space API + safety, on top of RobstrideBus. This
is the only file that knows joint names like "knee" or "ankle_pitch"; it
delegates raw CAN work to RobstrideBus and ankle geometry to AnkleKinematics.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from ankle_kinematics import AnkleKinematics, AnkleUnreachableError
from can_bus import RobstrideBus
from motor_config import (
    COMMAND_MARGIN_DEG,
    DAMPING_KD,
    MOTORS,
    NEAR_STOP_MARGIN_DEG,
    SIDE,
    STATE_MARGIN_DEG,
)

JOINT_NAMES = ("hipy", "hipx", "hipz", "knee", "ankle_pitch", "ankle_roll")


class Leg:
    def __init__(
        self,
        bus: RobstrideBus,
        *,
        side: str = SIDE,
        control_hz: float = 100.0,
        max_cmd_delta_deg: float = 20.0,
    ):
        self.bus = bus
        self.side = side
        self.control_hz = float(control_hz)
        self.max_cmd_delta_deg = float(max_cmd_delta_deg)
        self.ankle = AnkleKinematics(side=side)
        self._id_by_joint: Dict[str, int] = {info.joint_name: mid for mid, info in MOTORS.items()}
        self._last_pitch_deg = 0.0
        self._last_roll_deg = 0.0

        # Target is kept in joint space so a later calibration change doesn't
        # silently shift what an already-set target means.
        self._target_joint: Dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self._target_lock = threading.Lock()

        # Last raw setpoint actually sent per motor -- the ramp in
        # send_action() advances THIS toward the target, not the measured
        # position (see send_action's comment for why).
        self._sent_raw: Dict[int, float] = {}

        self._estop = False
        self._estop_reason = ""
        self._damping_active = False

        self._running = False
        self._loop_thread: Optional[threading.Thread] = None

    # -------------------------
    # Lifecycle
    # -------------------------
    def connect(self) -> None:
        """Read current motor state once, latch it as the initial target,
        and seed the sent-setpoint ramp from it -- so the first control
        cycle doesn't command a jump."""
        self.bus.request_state()
        self.latch_target_from_state()
        self._sent_raw = {mid: st.position_deg for mid, st in self.bus.states().items() if st.is_valid}

    def latch_target_from_state(self) -> None:
        try:
            self.set_action(**self.get_observation())
        except Exception:
            pass

    def enable_all(self) -> Dict[int, bool]:
        return self.bus.enable_all()

    def disable_all(self) -> Dict[int, bool]:
        return self.bus.disable_all()

    def start(self) -> None:
        if self._loop_thread is not None:
            return
        self._running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True, name="leg-loop")
        self._loop_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
            self._loop_thread = None
        self.disable_all()

    # -------------------------
    # Joint-space <-> motor-space
    # -------------------------
    def _calibrated(self, motor_id: int, raw_deg: float) -> float:
        m = MOTORS[motor_id]
        return m.sign * raw_deg + m.offset_deg

    @staticmethod
    def _wrap_near_zero(deg: float) -> float:
        """Resolve to the 360*k branch nearest 0deg, for INTERPRETING an
        angle against limit_deg. Mechanical zero is calibrated to raw=0, so
        any valid pose is within one wrap of 0 -- e.g. a raw/IK angle of
        359.99deg is physically the same pose as -0.01deg. Without this, a
        value one wrap away from the valid range reads as wildly out of
        limit_deg."""
        return deg - 360.0 * round(deg / 360.0)

    @staticmethod
    def _resolve_raw_near_current(want_raw: float, cur_raw: float) -> float:
        """Resolve want_raw to the 360*k branch nearest cur_raw, for
        deciding what to actually COMMAND. This is deliberately NOT
        wrapped near 0: the motor must be sent the setpoint closest to
        where it physically is right now, or a target that's one wrap away
        (e.g. 359.99deg when the motor sits near 0deg) would get commanded
        as a near-full-turn move instead of a tiny one."""
        k = round((cur_raw - want_raw) / 360.0)
        return want_raw + k * 360.0

    def _raw_from_calibrated(self, motor_id: int, cal_deg: float) -> float:
        m = MOTORS[motor_id]
        return (cal_deg - m.offset_deg) / m.sign

    def motor_to_joint(self, raw: Dict[int, float]) -> Dict[str, float]:
        # A motor's raw readback can come back a full 360deg off near its
        # calibrated zero after a power cycle (multi-turn lap-count
        # reconstruction picks the wrong lap at boot; the fine angle itself
        # is still correct) even though the motor never physically moved --
        # confirmed on hardware (hipy: 349.04deg == -10.96deg - 360deg,
        # reproduced across two independent power cycles). Wrap back to the
        # branch nearest 0 before this raw value is used for anything, same
        # fix already applied on the command/limit-check side (see
        # _wrap_near_zero's docstring).
        out: Dict[str, float] = {}
        for name in ("hipy", "hipx", "hipz", "knee"):
            mid = self._id_by_joint[name]
            out[name] = self._wrap_near_zero(self._calibrated(mid, raw[mid]))

        a1_id = self._id_by_joint["ankle_a1"]
        a2_id = self._id_by_joint["ankle_a2"]
        a1_raw = self._wrap_near_zero(raw[a1_id])
        a2_raw = self._wrap_near_zero(raw[a2_id])
        pitch, roll = self.ankle.solve_fk(
            a1_raw, a2_raw,
            guess_pitch_deg=self._last_pitch_deg, guess_roll_deg=self._last_roll_deg,
        )
        self._last_pitch_deg, self._last_roll_deg = pitch, roll
        out["ankle_pitch"], out["ankle_roll"] = pitch, roll
        return out

    def joint_to_motor(self, q: Dict[str, float]) -> Dict[int, float]:
        """Raises AnkleUnreachableError if the ankle target has no valid
        motor solution."""
        out: Dict[int, float] = {}
        for name in ("hipz", "hipx", "hipy", "knee"):
            mid = self._id_by_joint[name]
            out[mid] = self._raw_from_calibrated(mid, q[name])

        a1, a2 = self.ankle.solve_ik(q["ankle_pitch"], q["ankle_roll"])
        out[self._id_by_joint["ankle_a1"]] = a1
        out[self._id_by_joint["ankle_a2"]] = a2
        return out

    # -------------------------
    # Action / observation
    # -------------------------
    def set_action(self, **joints: float) -> None:
        unknown = set(joints) - set(JOINT_NAMES)
        if unknown:
            raise ValueError(f"unknown joint(s): {sorted(unknown)}")
        with self._target_lock:
            for name, value in joints.items():
                if value is not None:
                    self._target_joint[name] = float(value)

    def target(self) -> Dict[str, float]:
        with self._target_lock:
            return dict(self._target_joint)

    def get_observation(self) -> Dict[str, float]:
        states = self.bus.states()
        raw = {mid: st.position_deg for mid, st in states.items()}
        try:
            return self.motor_to_joint(raw)
        except Exception:
            return {name: float("nan") for name in JOINT_NAMES}

    def send_action(self) -> Dict[int, float]:
        """Build a command from the current target and send it, applying
        limit/ramp/near-stop safety. Returns the raw values actually sent."""
        if self._estop:
            return {}

        states = self.bus.states()
        try:
            target_raw = self.joint_to_motor(self.target())
        except AnkleUnreachableError:
            # Target has no valid ankle-motor solution -- don't go silent,
            # hold the last measured position via damping instead.
            self._damping_active = True
            self._send_damping(states)
            return {}

        if self._near_any_limit(states):
            self._damping_active = True
            self._send_damping(states)
            return {}
        self._damping_active = False

        sent: Dict[int, float] = {}
        for mid, want_raw in target_raw.items():
            st = states.get(mid)
            if st is None or not st.is_valid:
                continue

            # Advance the setpoint toward want_raw by at most max_cmd_delta_deg
            # per cycle, ramping from the LAST SENT setpoint (not the measured
            # position). Re-anchoring from measured caps the PD tracking
            # error -- and therefore the torque, ~kp*max_cmd_delta_deg -- so a
            # joint fighting gravity/friction can stall just short of the
            # target and never arrive. Ramping the setpoint lets the error
            # (and torque) keep growing until the joint actually catches up.
            prev_sp = self._sent_raw.get(mid, st.position_deg)
            want_raw = self._resolve_raw_near_current(want_raw, prev_sp)
            step = max(-self.max_cmd_delta_deg, min(self.max_cmd_delta_deg, want_raw - prev_sp))
            send_raw = prev_sp + step

            cal = self._wrap_near_zero(self._calibrated(mid, send_raw))
            lo, hi = MOTORS[mid].limit_deg
            if not (lo + COMMAND_MARGIN_DEG <= cal <= hi - COMMAND_MARGIN_DEG):
                continue  # reject out-of-limit setpoint
            m = MOTORS[mid]
            self.bus.send_command(mid, position_deg=send_raw, kp=m.kp, kd=m.kd)
            self._sent_raw[mid] = send_raw
            sent[mid] = send_raw
        return sent

    # -------------------------
    # Safety
    # -------------------------
    @property
    def estop(self) -> bool:
        return self._estop

    @property
    def estop_reason(self) -> str:
        return self._estop_reason

    @property
    def damping(self) -> bool:
        return self._damping_active

    def clear_estop(self) -> None:
        self._estop = False
        self._estop_reason = ""

    def check_estop(self) -> bool:
        for mid, st in self.bus.states().items():
            if not st.is_valid:
                continue
            cal = self._wrap_near_zero(self._calibrated(mid, st.position_deg))
            lo, hi = MOTORS[mid].limit_deg
            if not (lo - STATE_MARGIN_DEG <= cal <= hi + STATE_MARGIN_DEG):
                self._estop = True
                self._estop_reason = f"m{mid} out of bounds: {cal:.2f}deg, limit={MOTORS[mid].limit_deg}"
                self.bus.disable_all()
                break
        return self._estop

    def _near_any_limit(self, states: Dict[int, "object"]) -> bool:
        for mid, st in states.items():
            if not st.is_valid:
                continue
            cal = self._wrap_near_zero(self._calibrated(mid, st.position_deg))
            lo, hi = MOTORS[mid].limit_deg
            if cal <= lo + NEAR_STOP_MARGIN_DEG or cal >= hi - NEAR_STOP_MARGIN_DEG:
                return True
        return False

    def _send_damping(self, states: Dict[int, "object"]) -> None:
        for mid, st in states.items():
            if not st.is_valid:
                continue
            self.bus.send_command(mid, position_deg=st.position_deg, kp=0.0, kd=DAMPING_KD)
            # Resync the ramp to wherever the leg actually settles during
            # damping, so normal control resumes from there instead of
            # lurching back toward a stale pre-damping setpoint.
            self._sent_raw[mid] = st.position_deg

    # -------------------------
    # Internal loop
    # -------------------------
    def _run_loop(self) -> None:
        period = 1.0 / max(1.0, self.control_hz)
        next_tick = time.perf_counter()
        while self._running:
            self.bus.request_state()
            if not self.check_estop():
                self.send_action()

            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()
