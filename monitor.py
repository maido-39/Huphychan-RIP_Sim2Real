#!/usr/bin/env python3
"""RobStride MIT-mode: real-time angle/velocity/torque monitor.

MIT protocol has no "active reporting" like private protocol does -- the
motor only replies when it receives a command. So this gives two modes:

  --mode poll     (default) periodically sends a harmless clear-fault
                  ping (0xFB) to request a fresh feedback frame. Works
                  standalone, doesn't disturb enable/position state.
  --mode passive  sends nothing; just listens on the bus and decodes any
                  MIT feedback frame for this motor id. Use this while
                  another process (e.g. feedforward_torque_control.py,
                  or the interactive test controller) is already sending
                  MIT commands -- you'll see its replies for free.

Usage:
  python3 monitor.py --channel can0 --motor-id 8
  python3 monitor.py --channel can0 --motor-id 8 --mode passive
  python3 monitor.py --channel can0 --motor-id 8 --rate 100
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Optional

import can

from mjlab.tasks.inverse.commission_motor import (
  _open_bus,
  _shutdown_bus,
  _flush_bus,
  _matches_mit_reply,
  CAN_CMD_CLEAR_FAULT,
)


def _uint_to_float(raw: int, vmin: float, vmax: float, bits: int) -> float:
  span = vmax - vmin
  return raw / float((1 << bits) - 1) * span + vmin


def decode_mit_feedback(
  msg: Any,
  *,
  pmax: float = 12.57,
  vmax: float = 33.0,
  tmax: float = 17.0,
) -> Optional[dict]:
  """Decode a MIT 'Response Command 1' (Data Feedback / Motor Status) frame.

  Byte0: motor CAN id
  Byte1-2: angle, 16-bit unsigned -> (-pmax .. pmax) rad
  Byte3 + high nibble of Byte4: velocity, 12-bit unsigned -> (-vmax .. vmax) rad/s
  low nibble of Byte4 + Byte5: torque, 12-bit unsigned -> (-tmax .. tmax) Nm
  Byte6-7: winding temperature (scaling not fully confirmed, shown raw)
  """
  if bool(getattr(msg, "is_extended_id", False)):
    return None
  data = bytes(msg.data)
  if len(data) < 8:
    return None

  motor_id = data[0]
  angle_raw = (data[1] << 8) | data[2]
  speed_raw = (data[3] << 4) | (data[4] >> 4)
  torque_raw = ((data[4] & 0x0F) << 8) | data[5]
  temp_raw = (data[6] << 8) | data[7]

  angle_rad = _uint_to_float(angle_raw, -pmax, pmax, 16)
  speed_rad_s = _uint_to_float(speed_raw, -vmax, vmax, 12)
  torque_val = _uint_to_float(torque_raw, -tmax, tmax, 12)

  return {
    "motor_id": motor_id,
    "angle_rad": angle_rad,
    "angle_deg": math.degrees(angle_rad),
    "velocity_rad_s": speed_rad_s,
    "velocity_deg_s": math.degrees(speed_rad_s),
    "torque_nm": torque_val,
    "temp_raw": temp_raw,
  }


def _print_row(fb: dict) -> None:
  print(
    f"id={fb['motor_id']:>3}  "
    f"angle={fb['angle_deg']:+8.2f}deg ({fb['angle_rad']:+6.3f}rad)  "
    f"vel={fb['velocity_deg_s']:+8.2f}deg/s  "
    f"torque={fb['torque_nm']:+6.2f}Nm"
  )


def _poll_mode(
  bus: Any, motor_id: int, *, pmax: float, vmax: float, tmax: float, rate_hz: float
) -> None:
  period_s = 1.0 / rate_hz
  print(
    f"Polling motor id={motor_id} at {rate_hz:.0f}Hz via clear-fault ping (Ctrl+C to stop)..."
  )
  data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
  try:
    while True:
      t0 = time.monotonic()
      msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
      bus.send(msg)
      reply = bus.recv(timeout=max(period_s * 0.9, 0.005))
      if reply is not None and _matches_mit_reply(reply, motor_id=motor_id):
        fb = decode_mit_feedback(reply, pmax=pmax, vmax=vmax, tmax=tmax)
        if fb is not None:
          _print_row(fb)
      elif reply is None:
        print("  (no reply)")
      sleep_left = period_s - (time.monotonic() - t0)
      if sleep_left > 0:
        time.sleep(sleep_left)
  except KeyboardInterrupt:
    print("\nStopped.")


def _passive_mode(
  bus: Any, motor_id: int, *, pmax: float, vmax: float, tmax: float, print_hz: float
) -> None:
  print(f"Listening passively for motor id={motor_id} (Ctrl+C to stop)...")
  print(
    "(nothing is being sent -- make sure another process is actively commanding this motor)"
  )
  min_interval = 1.0 / print_hz
  last_print = 0.0
  try:
    while True:
      msg = bus.recv(timeout=1.0)
      if msg is None:
        continue
      if not _matches_mit_reply(msg, motor_id=motor_id):
        continue
      fb = decode_mit_feedback(msg, pmax=pmax, vmax=vmax, tmax=tmax)
      if fb is None:
        continue
      now = time.monotonic()
      if now - last_print >= min_interval:
        _print_row(fb)
        last_print = now
  except KeyboardInterrupt:
    print("\nStopped.")


def main() -> int:
  ap = argparse.ArgumentParser(
    description="RobStride MIT-mode real-time angle/velocity/torque monitor"
  )
  ap.add_argument("--interface", default="socketcan")
  ap.add_argument("--channel", default="can0")
  ap.add_argument(
    "--motor-id", type=int, default="8", help="known motor id; None to auto-scan"
  )
  ap.add_argument("--mode", choices=["poll", "passive"], default="poll")
  ap.add_argument("--rate", type=float, default=50.0, help="poll-mode request rate, Hz")
  ap.add_argument(
    "--print-rate",
    type=float,
    default=10.0,
    help="passive-mode print rate, Hz (data may arrive faster)",
  )
  ap.add_argument(
    "--pmax",
    type=float,
    default=12.57,
    help="must match the pmax used to command this motor",
  )
  ap.add_argument(
    "--vmax",
    type=float,
    default=33.0,
    help="must match the vmax used to command this motor",
  )
  ap.add_argument(
    "--tmax",
    type=float,
    default=17.0,
    help="must match the tmax used to command this motor",
  )
  args = ap.parse_args()

  bus = _open_bus(args.interface, args.channel)
  try:
    _flush_bus(bus)
    if args.mode == "poll":
      _poll_mode(
        bus,
        args.motor_id,
        pmax=args.pmax,
        vmax=args.vmax,
        tmax=args.tmax,
        rate_hz=args.rate,
      )
    else:
      _passive_mode(
        bus,
        args.motor_id,
        pmax=args.pmax,
        vmax=args.vmax,
        tmax=args.tmax,
        print_hz=args.print_rate,
      )
    return 0
  finally:
    _shutdown_bus(bus)


if __name__ == "__main__":
  raise SystemExit(main())
