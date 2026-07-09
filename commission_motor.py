#!/usr/bin/env python3
from __future__ import annotations

"""Commission one RobStride motor before assembly.

This utility is self-contained (no robstride_toolkit import).

Wizard flow:
1) Detect current motor ID and protocol.
2) Switch motor to MIT protocol (with reboot checkpoints).
3) Set target ID (based on motor model map).
4) Reboot and verify final ID/protocol.
5) Motion check: set zero, enable, move to 90 deg for 1s, then back to 0 deg.
"""

import argparse
import math
import sys
import time
from enum import IntEnum
from typing import Any, Optional

try:
  import can  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
  can = None
  _CAN_IMPORT_ERROR = exc
else:
  _CAN_IMPORT_ERROR = None

PROTOCOLS = ("canopen", "private", "mit")
DEFAULT_PROTOCOL_ORDER = ("canopen", "private", "mit")

CAN_CMD_CLEAR_FAULT = 0xFB
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD
CAN_CMD_SET_ZERO = 0xFE

MOTOR_MODEL_ID_MAP: dict[str, tuple[int, ...]] = {
  "o0": (1, 7),
  "o2": (2, 8),
  "o3": (3, 4, 9, 10),
  "o5": (5, 6, 11, 12),
}


class CommMode(IntEnum):
  PRIVATE = 0
  CANOPEN = 1
  MIT = 2


def _require_can() -> None:
  if can is None:
    raise RuntimeError(
      "python-can is required for motor commissioning. "
      "Install it with: pip install python-can"
    ) from _CAN_IMPORT_ERROR


def _open_bus(interface: str, channel: str):
  _require_can()
  return can.interface.Bus(interface=interface, channel=channel)


def _shutdown_bus(bus: Any) -> None:
  try:
    bus.shutdown()
  except Exception:
    pass


def _flush_bus(bus: Any, max_msgs: int = 1000) -> int:
  count = 0
  while count < max_msgs:
    msg = bus.recv(0.0005)
    if msg is None:
      break
    count += 1
  return count


def make_ext_id(comm_type: int, host_id: int, target_id: int) -> int:
  return ((comm_type & 0x1F) << 24) | ((host_id & 0xFFFF) << 8) | (target_id & 0xFF)


def parse_reply_id(arbitration_id: int) -> tuple[int, int, int]:
  comm_type = (arbitration_id >> 24) & 0x1F
  host_field = (arbitration_id >> 8) & 0xFFFF
  target_field = arbitration_id & 0xFF
  return comm_type, host_field, target_field


def _wait_reply(
  bus: Any,
  *,
  timeout_s: float,
  predicate,
  poll_s: float = 0.003,
) -> Optional[Any]:
  deadline = time.monotonic() + float(timeout_s)
  while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      return None
    msg = bus.recv(timeout=min(max(poll_s, 0.0005), remaining))
    if msg is None:
      continue
    if predicate(msg):
      return msg


def _matches_private_reply(
  msg: Any, *, motor_id: int, expected_comm: Optional[int] = None
) -> bool:
  if not bool(getattr(msg, "is_extended_id", False)):
    return False
  try:
    comm, host, target = parse_reply_id(int(msg.arbitration_id))
  except Exception:
    return False

  if expected_comm is not None and comm != int(expected_comm):
    return False

  mid = int(motor_id) & 0xFF
  return host == mid or target == mid


def _matches_canopen_reply(msg: Any, *, motor_id: int) -> bool:
  if bool(getattr(msg, "is_extended_id", False)):
    return False
  return int(msg.arbitration_id) == (0x580 + int(motor_id))


def _matches_mit_reply(msg: Any, *, motor_id: int) -> bool:
  if bool(getattr(msg, "is_extended_id", False)):
    return False
  return int(msg.arbitration_id) in (int(motor_id), 0xFD)


def ping_private(bus: Any, motor_id: int, *, timeout_s: float = 0.1) -> bool:
  identifier = make_ext_id(0x01, int(motor_id), int(motor_id))
  msg = can.Message(arbitration_id=identifier, data=[0x00] * 8, is_extended_id=True)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_private_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def ping_canopen(bus: Any, motor_id: int, *, timeout_s: float = 0.05) -> bool:
  req_id = 0x600 + int(motor_id)
  data = [0x40, 0x00, 0x10, 0x00, 0, 0, 0, 0]
  msg = can.Message(arbitration_id=req_id, data=data, is_extended_id=False)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_canopen_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def ping_mit(bus: Any, motor_id: int, *, timeout_s: float = 0.1) -> bool:
  data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
  msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_mit_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def switch_canopen_to_private(
  bus: Any, motor_id: int, *, host_id: int = 0x01, timeout_s: float = 0.5
) -> bool:
  arb_id = make_ext_id(0x19, int(host_id), int(motor_id))
  msg = can.Message(arbitration_id=arb_id, data=[0x00] * 8, is_extended_id=True)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_private_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def switch_mit_to_private(bus: Any, motor_id: int, *, timeout_s: float = 0.5) -> bool:
  data = [0xFF] * 8
  data[6] = int(CommMode.PRIVATE)
  data[7] = 0xFD
  msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_mit_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def switch_private_to_mit(
  bus: Any, motor_id: int, *, host_id: int = 0x01, timeout_s: float = 0.5
) -> bool:
  arb_id = make_ext_id(0x19, int(host_id), int(motor_id))
  data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, int(CommMode.MIT), 0x00]
  msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_private_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def change_motor_id(
  bus: Any, old_motor_id: int, new_motor_id: int, *, timeout_s: float = 0.5
) -> bool:
  data = [0xFF] * 8
  data[6] = int(new_motor_id) & 0xFF
  data[7] = 0xFA
  msg = can.Message(arbitration_id=int(old_motor_id), data=data, is_extended_id=False)
  bus.send(msg)

  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_mit_reply(m, motor_id=int(old_motor_id))
    or _matches_mit_reply(m, motor_id=int(new_motor_id)),
  )
  return reply is not None


def set_zero_mit(bus: Any, motor_id: int, *, timeout_s: float = 0.2) -> bool:
  data = [0xFF] * 7 + [CAN_CMD_SET_ZERO]
  msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_mit_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def enable_mit(bus: Any, motor_id: int, *, timeout_s: float = 0.2) -> bool:
  data = [0xFF] * 7 + [CAN_CMD_ENABLE]
  msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_mit_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def disable_mit(bus: Any, motor_id: int, *, timeout_s: float = 0.2) -> bool:
  data = [0xFF] * 7 + [CAN_CMD_DISABLE]
  msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_mit_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
  x_clamped = max(float(x_min), min(float(x_max), float(x)))
  span = float(x_max - x_min)
  data_norm = (x_clamped - float(x_min)) / span
  return int(data_norm * ((1 << int(bits)) - 1))


def parse_mit_reply_position(msg, pmax: float = 12.57) -> float:
  """MIT 프로토콜 답장 패킷(can.Message)에서 현재 기계적 위치(rad)를 파싱합니다."""
  # 데이터 바이트에서 16비트 위치 uint 값 복원 (보통 data[1]과 data[2] 사용)
  # (모터 펌웨어 버전에 따라 data[0], data[1]인 경우도 있으나 일반적으로 아래와 같습니다)
  high = msg.data[1]
  low = msg.data[2]
  q_uint = (high << 8) | low

  # _float_to_uint의 역연산 (_uint_to_float)
  # 16비트(0 ~ 65535) 범위를 -pmax ~ pmax 범위의 float로 변환
  span = 2.0 * float(pmax)
  value = (float(q_uint) * span / 65535.0) - float(pmax)
  return value


def mit_position_command(
  bus: Any,
  motor_id: int,
  position_deg: float,
  *,
  kp: float = 10.0,
  kd: float = 0.5,
  velocity_deg_s: float = 0.0,
  torque_nm: float = 0.0,
  pmax: float = 12.57,
  vmax: float = 33.0,
  tmax: float = 17.0,
  timeout_s: float = 0.03,
) -> bool:
  position_rad = math.radians(float(position_deg))
  velocity_rad_s = math.radians(float(velocity_deg_s))

  kp_uint = _float_to_uint(float(kp), 0.0, 500.0, 12)
  kd_uint = _float_to_uint(float(kd), 0.0, 5.0, 12)
  q_uint = _float_to_uint(position_rad, -float(pmax), float(pmax), 16)
  dq_uint = _float_to_uint(velocity_rad_s, -float(vmax), float(vmax), 12)
  tau_uint = _float_to_uint(float(torque_nm), -float(tmax), float(tmax), 12)

  data = [0] * 8
  data[0] = (q_uint >> 8) & 0xFF
  data[1] = q_uint & 0xFF
  data[2] = dq_uint >> 4
  data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
  data[4] = kp_uint & 0xFF
  data[5] = kd_uint >> 4
  data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
  data[7] = tau_uint & 0xFF

  msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
  bus.send(msg)
  reply = _wait_reply(
    bus,
    timeout_s=timeout_s,
    predicate=lambda m: _matches_mit_reply(m, motor_id=int(motor_id)),
  )
  return reply is not None


def _run_motion_test(bus: Any, motor_id: int, *, duration_s: float = 1.0) -> bool:
  print("Running motion test: set zero -> enable -> 90 deg (1s) -> 0 deg -> disable")

  if not set_zero_mit(bus, motor_id):
    print("Motion test failed: set zero command not acknowledged.")
    return False
  time.sleep(0.1)

  if not enable_mit(bus, motor_id):
    print("Motion test failed: enable command not acknowledged.")
    return False
  time.sleep(0.1)

  t0 = time.monotonic()
  ok_90 = False
  while (time.monotonic() - t0) < float(duration_s):
    if mit_position_command(bus, motor_id, 90.0):
      ok_90 = True
    time.sleep(0.02)

  t1 = time.monotonic()
  ok_0 = False
  while (time.monotonic() - t1) < float(duration_s):
    if mit_position_command(bus, motor_id, 0.0):
      ok_0 = True
    time.sleep(0.02)

  disable_ok = disable_mit(bus, motor_id)
  if not disable_ok:
    print("Motion test failed: disable command not acknowledged.")
    return False

  if not (ok_90 and ok_0):
    print(
      "Motion test warning: one or more MIT position commands were not acknowledged."
    )
    return False

  print("Motion test complete.")
  return True


def _ping_protocol(bus: Any, protocol: str, motor_id: int) -> bool:
  if protocol == "canopen":
    return ping_canopen(bus, motor_id)
  if protocol == "private":
    return ping_private(bus, motor_id)
  if protocol == "mit":
    return ping_mit(bus, motor_id)
  raise ValueError(f"Unsupported protocol: {protocol}")


def _parse_protocol_order(raw: str) -> tuple[str, ...]:
  order: list[str] = []
  for token in raw.split(","):
    proto = token.strip().lower()
    if not proto:
      continue
    if proto not in PROTOCOLS:
      raise ValueError(f"Invalid protocol in --protocol-order: {proto}")
    order.append(proto)
  if not order:
    raise ValueError("--protocol-order cannot be empty")
  return tuple(order)


def _find_protocol_for_id(
  bus: Any, motor_id: int, protocol_order: tuple[str, ...]
) -> Optional[str]:
  for proto in protocol_order:
    if _ping_protocol(bus, proto, motor_id):
      return proto
  return None


def _scan_for_motors(
  bus: Any,
  start_id: int,
  end_id: int,
  protocol_order: tuple[str, ...],
) -> list[tuple[int, str]]:
  found: list[tuple[int, str]] = []
  for motor_id in range(int(start_id), int(end_id) + 1):
    proto = _find_protocol_for_id(bus, motor_id, protocol_order)
    if proto is not None:
      found.append((motor_id, proto))
  return found


def _confirm(
  prompt: str, *, assume_yes: bool = False, default_yes: bool = True
) -> bool:
  if assume_yes:
    print(f"{prompt} -> yes (--yes)")
    return True

  suffix = "[Y/n]" if default_yes else "[y/N]"
  raw = input(f"{prompt} {suffix} ").strip().lower()
  if not raw:
    return default_yes
  return raw in ("y", "yes")


def _wait_reboot(*, assume_yes: bool = False) -> None:
  if assume_yes:
    print("Waiting 1.0s for reboot (--yes mode).")
    time.sleep(1.0)
    return
  input("Reboot motor now, then press Enter to continue...")


def _parse_int(raw: str) -> int:
  return int(raw.strip(), 0)


def _ensure_protocol(
  bus: Any,
  motor_id: int,
  current_protocol: str,
  target_protocol: str,
  *,
  protocol_order: tuple[str, ...],
  assume_yes: bool,
) -> tuple[bool, Optional[str]]:
  protocol = str(current_protocol)
  safety_steps = 0

  while protocol != target_protocol:
    safety_steps += 1
    if safety_steps > 6:
      return False, protocol

    if target_protocol == "private":
      if protocol == "canopen":
        ok = switch_canopen_to_private(bus, motor_id)
      elif protocol == "mit":
        ok = switch_mit_to_private(bus, motor_id)
      else:
        return False, protocol

    elif target_protocol == "mit":
      if protocol == "private":
        ok = switch_private_to_mit(bus, motor_id)
      elif protocol == "canopen":
        ok = switch_canopen_to_private(bus, motor_id)
      else:
        return False, protocol

    else:
      return False, protocol

    if not ok:
      return False, protocol

    _wait_reboot(assume_yes=assume_yes)
    _flush_bus(bus)

    detected = _find_protocol_for_id(bus, motor_id, protocol_order)
    if detected is None:
      return False, None
    protocol = detected

  return True, protocol


def _prompt_model() -> str:
  allowed = ", ".join(MOTOR_MODEL_ID_MAP.keys())
  while True:
    raw = input(f"Motor model ({allowed}): ").strip().lower()
    if raw in MOTOR_MODEL_ID_MAP:
      return raw
    print("Invalid model.")


def _prompt_target_id(model: str) -> int:
  allowed = MOTOR_MODEL_ID_MAP[model]
  allowed_s = ", ".join(str(v) for v in allowed)
  while True:
    raw = input(f"Target ID for model {model} [{allowed_s}]: ").strip()
    try:
      value = _parse_int(raw)
    except Exception:
      print("Invalid integer.")
      continue
    if value in allowed:
      return value
    print(f"ID {value} not allowed for model {model}.")


def _cmd_scan(args: argparse.Namespace) -> int:
  protocol_order = _parse_protocol_order(args.protocol_order)
  bus = _open_bus(args.interface, args.channel)
  try:
    _flush_bus(bus)
    found = _scan_for_motors(bus, args.start_id, args.end_id, protocol_order)
  finally:
    _shutdown_bus(bus)

  if not found:
    print("No motor detected.")
    return 1

  print(f"Detected motors on {args.channel}:")
  for motor_id, protocol in found:
    print(f"  id={motor_id:>3} protocol={protocol}")
  return 0


def _cmd_wizard(args: argparse.Namespace) -> int:
  protocol_order = _parse_protocol_order(args.protocol_order)

  bus = _open_bus(args.interface, args.channel)
  try:
    _flush_bus(bus)

    motor_id = args.motor_id
    protocol: Optional[str]

    if motor_id is not None:
      protocol = _find_protocol_for_id(bus, motor_id, protocol_order)
      if protocol is None:
        print(f"No response from motor id={motor_id} using protocols {protocol_order}")
        return 1
    else:
      found = _scan_for_motors(bus, args.start_id, args.end_id, protocol_order)
      if not found:
        print("No motor detected for wizard.")
        return 1
      if len(found) > 1:
        print("Multiple motors detected. Connect one motor only or pass --motor-id.")
        for mid, proto in found:
          print(f"  id={mid:>3} protocol={proto}")
        return 1
      motor_id, protocol = found[0]

    assert protocol is not None
    print(f"Detected motor id={motor_id}, protocol={protocol}")

    # Mandatory: switch to MIT before selecting/changing final ID.
    ok, protocol = _ensure_protocol(
      bus,
      motor_id,
      protocol,
      "mit",
      protocol_order=protocol_order,
      assume_yes=args.yes,
    )
    if not ok or protocol != "mit":
      print(f"Failed to reach MIT protocol (got {protocol}).")
      return 1
    print("Motor is now in MIT protocol.")

    model = args.motor_model
    if model is None:
      if args.yes:
        print("--yes mode requires --motor-model (o0/o2/o3/o5).")
        return 2
      model = _prompt_model()

    if model not in MOTOR_MODEL_ID_MAP:
      print(f"Unsupported motor model: {model}")
      return 2

    target_id = args.new_id
    if target_id is None:
      if args.yes:
        print("--yes mode requires --new-id matching the model ID map.")
        return 2
      target_id = _prompt_target_id(model)

    if int(target_id) not in MOTOR_MODEL_ID_MAP[model]:
      allowed = ", ".join(str(v) for v in MOTOR_MODEL_ID_MAP[model])
      print(f"ID {target_id} invalid for model {model}. Allowed: {allowed}")
      return 2

    if int(target_id) != int(motor_id):
      print(f"Changing motor ID {motor_id} -> {target_id}...")
      ok = change_motor_id(bus, int(motor_id), int(target_id))
      if not ok:
        print("Failed changing motor ID.")
        return 1

      _wait_reboot(assume_yes=args.yes)
      _flush_bus(bus)

      motor_id = int(target_id)
      protocol = _find_protocol_for_id(bus, motor_id, protocol_order)
      if protocol is None:
        print(f"No response from new motor ID {motor_id} after reboot.")
        return 1

      ok, protocol = _ensure_protocol(
        bus,
        motor_id,
        protocol,
        "mit",
        protocol_order=protocol_order,
        assume_yes=args.yes,
      )
      if not ok or protocol != "mit":
        print(f"Failed to keep MIT protocol after ID change (got {protocol}).")
        return 1

    if not args.skip_motion_test:
      if _confirm("Run motor motion test now?", assume_yes=args.yes, default_yes=True):
        ok = _run_motion_test(bus, motor_id, duration_s=args.motion_duration_s)
        if not ok:
          return 1

    print("")
    print("Commissioning complete:")
    print(f"  motor_model={model}")
    print(f"  motor_id={motor_id}")
    print(f"  protocol=mit")
    print(f"  id_map_allowed={MOTOR_MODEL_ID_MAP[model]}")
    return 0

  finally:
    _shutdown_bus(bus)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="RobStride motor commissioning utility.")
  sub = parser.add_subparsers(dest="cmd", required=True)

  p_scan = sub.add_parser("scan", help="Scan CAN bus for motors and detect protocol")
  p_scan.add_argument("--interface", default="socketcan")
  p_scan.add_argument("--channel", default="can0")
  p_scan.add_argument("--start-id", type=int, default=0)
  p_scan.add_argument("--end-id", type=int, default=127)
  p_scan.add_argument("--protocol-order", default=",".join(DEFAULT_PROTOCOL_ORDER))

  p_wizard = sub.add_parser("wizard", help="Guided commissioning flow")
  p_wizard.add_argument("--interface", default="socketcan")
  p_wizard.add_argument("--channel", default="can0")
  p_wizard.add_argument(
    "--motor-id", type=int, default=None, help="Known motor ID; if omitted, auto-detect"
  )
  p_wizard.add_argument("--start-id", type=int, default=0)
  p_wizard.add_argument("--end-id", type=int, default=127)
  p_wizard.add_argument("--protocol-order", default=",".join(DEFAULT_PROTOCOL_ORDER))
  p_wizard.add_argument(
    "--motor-model", choices=tuple(MOTOR_MODEL_ID_MAP.keys()), default=None
  )
  p_wizard.add_argument(
    "--new-id", type=int, default=None, help="Target final motor ID"
  )
  p_wizard.add_argument("--skip-motion-test", action="store_true")
  p_wizard.add_argument("--motion-duration-s", type=float, default=1.0)
  p_wizard.add_argument("--yes", action="store_true", help="Auto-confirm prompts")

  return parser


def main() -> int:
  parser = _build_parser()
  args = parser.parse_args()

  try:
    if args.cmd == "scan":
      return _cmd_scan(args)
    if args.cmd == "wizard":
      return _cmd_wizard(args)

    print(f"unsupported command: {args.cmd}", file=sys.stderr)
    return 2
  except KeyboardInterrupt:
    print("Interrupted by user.")
    return 130
  except Exception as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
