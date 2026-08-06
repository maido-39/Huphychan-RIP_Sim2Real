#!/usr/bin/env python3
"""
main.py

Entry point for single-leg control:
  1. connect to the CAN bus
  2. check_connectivity() -- confirm every motor actually replies
  3. latch the current pose as the initial target (no jump on enable)
  4. enable torque and start the control loop
  5. run the policy loop (falls back to holding the latched pose if no
     policy is wired in yet)

Safety already lives in Leg (robot_leg.py):
  - limit_deg is enforced on every command (Leg.send_action) and on every
    measured state (Leg.check_estop -> disables torque on violation)
  - max_cmd_delta_deg caps how far the sent setpoint can move per control
    cycle -- a distant target is approached gradually over many cycles
    (an absolute setpoint ramp, not a re-anchor-from-measured step: see
    Leg.send_action's comment for why that distinction matters under load)
  - near a limit, the leg switches to damping-only instead of position
    control (Leg._near_any_limit / _send_damping)
This script only wires those pieces together and adds the pre-torque
connectivity check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from can_bus import RobstrideBus
from motor_config import MOTORS, SIDE
from policy import Policy
from robot_leg import Leg
from zero_policy import ZeroPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-leg policy control entry point.")
    p.add_argument("--channel", type=str, default="can0")
    p.add_argument("--interface", type=str, default="socketcan")
    p.add_argument("--use-mock-bus", action="store_true", help="Use an in-process fake CAN bus instead of real hardware.")
    p.add_argument("--control-hz", type=float, default=100.0, help="Rate of Leg's internal request_state/send_action loop.")
    p.add_argument(
        "--max-cmd-delta-deg",
        type=float,
        default=20.0,
        help="Max setpoint advance (deg) per control cycle; a distant target is approached gradually, not jumped to.",
    )
    p.add_argument(
        "--connectivity-attempts",
        type=int,
        default=3,
        help="Ping attempts per motor before declaring it unreachable.",
    )
    p.add_argument(
        "--skip-connectivity-check",
        action="store_true",
        help="Proceed even if some motors don't reply. Do not use on real hardware.",
    )
    p.add_argument(
        "--pause-before-enable",
        dest="pause_before_enable",
        action="store_true",
        default=True,
        help="Wait for Enter before enabling torque, so you can sanity-check the connectivity report first (default).",
    )
    p.add_argument(
        "--no-pause-before-enable",
        dest="pause_before_enable",
        action="store_false",
        help="Skip the pre-enable pause (e.g. for scripted --use-mock-bus runs).",
    )
    p.add_argument("--policy-dir", type=Path, default=None, help="If given, load a Policy from this directory.")
    p.add_argument(
        "--replay-csv",
        type=Path,
        default=None,
        help="If given, play back a trajectory recorded with record_trajectory.py instead of a --policy-dir model.",
    )
    p.add_argument("--replay-loop", action="store_true", help="With --replay-csv, loop the trajectory instead of holding its last pose.")
    p.add_argument("--policy-hz", type=float, default=50.0)
    p.add_argument("--status-period-s", type=float, default=1.0, help="How often to print the current joint observation.")
    return p.parse_args()


def build_bus(args: argparse.Namespace) -> RobstrideBus:
    transport = None
    if args.use_mock_bus:
        from mock_bus import MockCanBus

        # Start every motor at raw=0, matching this project's zero convention
        # (mechanical zero -> raw 0 -> ankle pitch/roll 0). A joint-limit
        # midpoint would be an arbitrary, sometimes IK-unreachable pose.
        transport = MockCanBus(MOTORS)
    return RobstrideBus(MOTORS, channel=args.channel, interface=args.interface, bus=transport)


def run_connectivity_check(bus: RobstrideBus, args: argparse.Namespace) -> bool:
    print("[main] checking motor communication...", flush=True)
    ok = bus.check_connectivity(attempts=args.connectivity_attempts)
    all_ok = True
    for mid, replied in sorted(ok.items()):
        joint = MOTORS[mid].joint_name
        print(f"  m{mid:02d} ({joint}): {'OK' if replied else 'NO REPLY'}", flush=True)
        all_ok = all_ok and replied

    if all_ok:
        print("[main] all motors responding.", flush=True)
        return True

    bad = [mid for mid, replied in ok.items() if not replied]
    if args.skip_connectivity_check:
        print(f"[main] motors not responding: {bad} -- continuing anyway (--skip-connectivity-check).", flush=True)
        return True

    print(f"[main] motors not responding: {bad} -- aborting before enabling torque.", flush=True)
    return False


def run_enable_all(leg: Leg) -> bool:
    print("[main] enabling torque...", flush=True)
    acked = leg.enable_all()
    all_ok = True
    for mid, ok in sorted(acked.items()):
        joint = MOTORS[mid].joint_name
        print(f"  m{mid:02d} ({joint}): {'ENABLED' if ok else 'NOT ACKED'}", flush=True)
        all_ok = all_ok and ok

    if all_ok:
        print("[main] all motors acknowledged enable.", flush=True)
    else:
        bad = [mid for mid, ok in acked.items() if not ok]
        print(
            f"[main] WARNING: motors did not acknowledge enable: {bad} -- "
            "position control may be a no-op (freewheel) on these joints.",
            flush=True,
        )
    return all_ok


def main() -> int:
    args = parse_args()
    bus = build_bus(args)
    leg: Optional[Leg] = None

    try:
        if not run_connectivity_check(bus, args):
            return 1

        leg = Leg(bus, side=SIDE, control_hz=args.control_hz, max_cmd_delta_deg=args.max_cmd_delta_deg)
        leg.connect()  # latches current measured pose as target -- avoids a jump on enable
        print(f"[main] latched target: { {k: round(v, 2) for k, v in leg.target().items()} }", flush=True)

        if args.pause_before_enable:
            input("[main] about to enable torque -- press Enter to continue...")

        run_enable_all(leg)
        leg.start()
        print(f"[main] control loop running @ {args.control_hz:.0f} Hz, max_cmd_delta={args.max_cmd_delta_deg:.1f} deg.", flush=True)

        policy = ZeroPolicy()
        if args.replay_csv is not None:
            from replay_policy import ReplayPolicy

            policy = ReplayPolicy.from_csv(args.replay_csv, loop=args.replay_loop)
            print(
                f"[main] replaying {args.replay_csv} (loop={args.replay_loop}) -- "
                "leg will ramp toward the trajectory's first pose at --max-cmd-delta-deg, "
                "so start it near the recorded starting pose.",
                flush=True,
            )
        elif args.policy_dir is not None:
            try:
                policy = Policy.from_files(args.policy_dir)
                print(f"[main] policy loaded from {args.policy_dir}", flush=True)
            except NotImplementedError as exc:
                print(f"[main] policy loading not available yet ({exc}); using ZeroPolicy instead.", flush=True)
        else:
            print(
                "[main] no --policy-dir/--replay-csv given; using ZeroPolicy "
                "(drives toward all-joints-0deg, rate-limited by --max-cmd-delta-deg).",
                flush=True,
            )
        policy.reset()

        period = 1.0 / max(1.0, args.policy_hz)
        last_status = 0.0
        while True:
            if leg.estop:
                print(f"[main] E-STOP tripped: {leg.estop_reason}", flush=True)
                break
            obs = leg.get_observation()
            action = policy.act(obs)
            leg.set_action(**action)

            now = time.monotonic()
            if now - last_status >= args.status_period_s:
                rounded = {k: round(v, 1) for k, v in obs.items()}
                print(f"[main] obs={rounded} damping={leg.damping}", flush=True)
                last_status = now
            time.sleep(period)
        return 1
    except KeyboardInterrupt:
        print("[main] stopping...", flush=True)
        return 0
    finally:
        if leg is not None:
            leg.stop()
        bus.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
