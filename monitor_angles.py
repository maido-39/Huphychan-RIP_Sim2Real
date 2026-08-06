#!/usr/bin/env python3
"""
monitor_angles.py

Passive-ish monitor: polls every configured motor's raw angle (and
velocity/torque/temp) over CAN and logs it to a timestamped CSV, printing
a live row per motor at the same rate.

Does NOT enable torque -- request_state() only sends the harmless
CLEAR_FAULT (0xFB) query byte per motor, same as check_connectivity() in
can_bus.py. Safe to run alongside another process that already has the
motors enabled (--mode passive-style usage isn't needed here since MIT
motors only reply when polled; this script IS the poller).

Usage:
  python3 monitor_angles.py --channel can1
  python3 monitor_angles.py --use-mock-bus --duration-s 5
  python3 monitor_angles.py --channel can1 --rate-hz 50 --log-dir logs
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

from motor_config import MOTORS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poll and log all motors' angles over CAN.")
    p.add_argument("--channel", type=str, default="can1")
    p.add_argument("--interface", type=str, default="socketcan")
    p.add_argument("--use-mock-bus", action="store_true", help="Use an in-process fake CAN bus instead of real hardware.")
    p.add_argument("--rate-hz", type=float, default=20.0, help="Poll/log rate.")
    p.add_argument("--duration-s", type=float, default=None, help="Stop after this many seconds (default: run until Ctrl+C).")
    p.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory to write the CSV log into.")
    return p.parse_args()


def build_bus(args: argparse.Namespace):
    from can_bus import RobstrideBus

    transport = None
    if args.use_mock_bus:
        from mock_bus import MockCanBus

        initial = {mid: 0 for mid in MOTORS.keys()}
        transport = MockCanBus(MOTORS, initial_position_deg=initial)
    return RobstrideBus(MOTORS, channel=args.channel, interface=args.interface, bus=transport)


def main() -> int:
    args = parse_args()
    bus = build_bus(args)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"angles_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["t_s", "motor_id", "joint_name", "position_deg", "velocity_deg_s", "torque_nm", "temp_c"])

    print(f"[monitor_angles] logging to {log_path}", flush=True)
    print(f"[monitor_angles] polling {len(MOTORS)} motors @ {args.rate_hz:.0f}Hz (Ctrl+C to stop)...", flush=True)

    period_s = 1.0 / max(1.0, args.rate_hz)
    t_start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t_start
            if args.duration_s is not None and t >= args.duration_s:
                break

            states = bus.request_state()
            row_parts = []
            for mid, info in MOTORS.items():
                st = states.get(mid)
                if st is None or not st.is_valid:
                    row_parts.append(f"{info.joint_name}=NO REPLY")
                    continue
                writer.writerow([f"{t:.4f}", mid, info.joint_name, f"{st.position_deg:.3f}", f"{st.velocity_deg_s:.3f}", f"{st.torque_nm:.3f}", f"{st.temp_c:.1f}"])
                row_parts.append(f"{info.joint_name}={st.position_deg:+7.2f}deg")
            log_file.flush()
            print("  " + "  ".join(row_parts), flush=True)

            sleep_left = period_s - (time.monotonic() - t_start - t)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\n[monitor_angles] stopping...", flush=True)
    finally:
        try:
            bus.disable_all()
            print("[monitor_angles] all motors disabled.", flush=True)
        except Exception as e:
            print(f"[monitor_angles] WARNING: disable_all on exit failed: {e}", flush=True)
        log_file.close()
        bus.disconnect()

    print(f"[monitor_angles] log saved: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
