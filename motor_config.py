"""
motor_config.py

Per-motor info for ONE physical leg (6 Robstride actuators, one CAN bus,
MIT protocol). No left/right split, no CAN0/CAN1 -- this describes exactly
the hardware that exists right now.

*** IMPORTANT -- READ BEFORE ENABLING TORQUE ON REAL HARDWARE ***
Every value below is a PLACEHOLDER. Replace each "# TODO: calibrate" value
with a number measured on the physical leg before running with torque
enabled. Wrong sign/offset/limit can drive a joint into its hard-stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Flip to True only after every "# TODO: calibrate" value below has been
# measured/verified on the physical leg.
CALIBRATED = False

# Which physical leg this file describes -- feeds AnkleKinematics(side=...)
# for the correct geometry mirroring.
SIDE = "right"


@dataclass(frozen=True)
class MotorInfo:
    joint_name: str      # "hipy" | "hipx" | "hipz" | "knee" | "ankle_a1" | "ankle_a2"
    pmax_rad: float       # MIT-mode encoding range (vendor spec)
    vmax_rad_s: float
    tmax_nm: float
    sign: float           # calibrated_deg = sign * raw_deg + offset_deg
    offset_deg: float
    limit_deg: Tuple[float, float]  # mechanical range, in calibrated space
    kp: float             # default MIT-mode gains
    kd: float


# TODO: calibrate -- every MotorInfo field below except joint_name/pmax/vmax/tmax
# is a placeholder. ankle_a1/ankle_a2 keep sign=1.0/offset=0.0 on purpose: their
# raw motor angle IS the AnkleKinematics alpha1/alpha2 input directly (the
# rotation/offset signs inside AnkleKinematics already account for calibration).
MOTORS: Dict[int, MotorInfo] = {
    7: MotorInfo("hipy",     pmax_rad=12.57, vmax_rad_s=44.0, tmax_nm=17.0, sign=1.0, offset_deg=0.0, limit_deg=(-117.07, 21.07), kp=20.0, kd=3.0),
    8: MotorInfo("hipx",     pmax_rad=12.57, vmax_rad_s=44.0, tmax_nm=17.0, sign=1.0, offset_deg=0.0, limit_deg=(-16.00, 85.00), kp=20.0, kd=3.0),
    9: MotorInfo("hipz",     pmax_rad=12.57, vmax_rad_s=44.0, tmax_nm=17.0, sign=1.0, offset_deg=0.0, limit_deg=(-50.00, 38.00), kp=20.0, kd=3.0),
    10: MotorInfo("knee",     pmax_rad=12.57, vmax_rad_s=44.0, tmax_nm=17.0, sign=1.0, offset_deg=0.0, limit_deg=(-28.00, 85.00), kp=20.0, kd=3.0),
    11: MotorInfo("ankle_a1", pmax_rad=12.57, vmax_rad_s=33.0, tmax_nm=14.0, sign=1.0, offset_deg=0.0, limit_deg=(-57.00, 61.00), kp=20.0, kd=3.0),
    12: MotorInfo("ankle_a2", pmax_rad=12.57, vmax_rad_s=33.0, tmax_nm=14.0, sign=1.0, offset_deg=0.0, limit_deg=(-50.00, 73.50), kp=20.0, kd=3.0),
}
MOTOR_IDS: Tuple[int, ...] = tuple(MOTORS.keys())
ANKLE_MOTOR_IDS: Tuple[int, int] = (11, 12)

# -------------------------------------------------------------------------
# CAN protocol command bytes (MIT-mode actuator convention).
# TODO: verify against your actuator's datasheet -- these bytes vary by vendor.
# -------------------------------------------------------------------------
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD
CAN_CMD_ZERO = 0xFE
CAN_CMD_CLEAR_FAULT = 0xFB

# -------------------------------------------------------------------------
# Safety margins / behavior.
# -------------------------------------------------------------------------
COMMAND_MARGIN_DEG = 0.6     # keep commands this far inside limit_deg
STATE_MARGIN_DEG = 0.4       # trip E-STOP if measured state exceeds limits by more than this
NEAR_STOP_MARGIN_DEG = 1.0   # switch to damping-only once this close to a limit
DAMPING_KD = 3.0             # kd used for damping-only (kp=0) commands
