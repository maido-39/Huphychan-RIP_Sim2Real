#!/usr/bin/env python3
"""
record_trajectory.py

Kinesthetic-teaching recorder: leaves all motors DISABLED (backdrivable) so
the leg can be hand-posed, polls Leg.get_observation() -- calibrated
joint-space angles, same JOINT_NAMES dict shape that Policy.act()/
Leg.set_action() use -- at --rate-hz, and logs each sample to a timestamped
CSV. Feed the resulting file to ReplayPolicy.from_csv() (replay_policy.py)
to reproduce the demonstrated motion open-loop.

Usage:
  python3 record_trajectory.py --channel can1
  python3 record_trajectory.py --use-mock-bus --duration-s 5
  python3 record_trajectory.py --channel can1 --rate-hz 50 --out-dir demos
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from motor_config import MOTORS, SIDE
from robot_leg import JOINT_NAMES, Leg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand-guide the leg and record the joint trajectory for later replay.")
    p.add_argument("--channel", type=str, default="can1")
    p.add_argument("--interface", type=str, default="socketcan")
    p.add_argument("--use-mock-bus", action="store_true", help="Use an in-process fake CAN bus instead of real hardware.")
    p.add_argument("--rate-hz", type=float, default=50.0, help="Sample rate.")
    p.add_argument("--duration-s", type=float, default=None, help="Stop after this many seconds (default: run until Ctrl+C).")
    p.add_argument("--out-dir", type=Path, default=Path("demos"), help="Directory to write the trajectory CSV into.")
    return p.parse_args()


def build_bus(args: argparse.Namespace):
    from can_bus import RobstrideBus

    transport = None
    if args.use_mock_bus:
        from mock_bus import MockCanBus

        transport = MockCanBus(MOTORS)
    return RobstrideBus(MOTORS, channel=args.channel, interface=args.interface, bus=transport)


def main() -> int:
    args = parse_args()
    bus = build_bus(args)
    leg = Leg(bus, side=SIDE)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"traj_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out_file = open(out_path, "w", newline="")
    writer = csv.writer(out_file)
    writer.writerow(["t_s", *JOINT_NAMES])

    print("[record_trajectory] motors stay DISABLED -- hand-pose the leg now.", flush=True)
    print(f"[record_trajectory] logging to {out_path}", flush=True)
    print(f"[record_trajectory] sampling {len(JOINT_NAMES)} joints @ {args.rate_hz:.0f}Hz (Ctrl+C to stop)...", flush=True)

    period_s = 1.0 / max(1.0, args.rate_hz)
    t_start = time.monotonic()
    try:
        bus.disable_all()  # backdrivable regardless of whatever state the bus was left in
        while True:
            t = time.monotonic() - t_start
            if args.duration_s is not None and t >= args.duration_s:
                break

            bus.request_state()
            obs = leg.get_observation()
            writer.writerow([f"{t:.4f}", *[f"{obs[name]:.3f}" for name in JOINT_NAMES]])
            out_file.flush()
            print("  " + "  ".join(f"{name}={obs[name]:+7.2f}deg" for name in JOINT_NAMES), flush=True)

            sleep_left = period_s - (time.monotonic() - t_start - t)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\n[record_trajectory] stopping...", flush=True)
    finally:
        out_file.close()
        bus.disconnect()

    n_samples = max(0, sum(1 for _ in open(out_path)) - 1)
    print(f"[record_trajectory] trajectory saved: {out_path} ({n_samples} samples)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
