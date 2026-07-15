#!/usr/bin/env python3
"""RobStride motor: minimal physical setup wizard.

Handles exactly the three things you actually need to get right on the
bench: which CAN ID the motor answers to, which protocol it's currently
speaking (private / MIT), and its mechanical zero point. Gain/limit
tuning is left at factory defaults on purpose.

Zero-setting and ID-change are performed via the MIT-protocol commands
in commission_motor.py (set_zero_mit / change_motor_id), which are the
ones already confirmed working. If the motor isn't in MIT protocol when
you ask for one of these, the script temporarily switches to MIT, does
the operation, then switches back to whatever protocol you were in
(unless you asked to end up in MIT anyway).

Usage:
  python3 robstride_setup.py --channel can0 --motor-id 8
  python3 robstride_setup.py --channel can0          # auto-scan
"""
from __future__ import annotations

import argparse
import time

from commission_motor import (
    _open_bus,
    _shutdown_bus,
    _flush_bus,
    _find_protocol_for_id,
    _scan_for_motors,
    _ensure_protocol,
    _wait_reboot,
    _confirm,
    _parse_int,
    change_motor_id,
    set_zero_mit,
    DEFAULT_PROTOCOL_ORDER,
)


def _detect(bus, args, protocol_order):
    if args.motor_id is not None:
        motor_id = args.motor_id
        protocol = _find_protocol_for_id(bus, motor_id, protocol_order)
        if protocol is None:
            print(f"No response from motor id={motor_id}.")
            return None, None
        return motor_id, protocol

    found = _scan_for_motors(bus, args.start_id, args.end_id, protocol_order)
    if not found:
        print("No motor detected.")
        return None, None
    if len(found) > 1:
        print("Multiple motors detected; connect one only or pass --motor-id:")
        for mid, proto in found:
            print(f"  id={mid:>3} protocol={proto}")
        return None, None
    return found[0]


def _print_state(motor_id: int, protocol: str) -> None:
    print(f"\n>>> motor_id={motor_id}  protocol={protocol}\n")


def _with_mit(bus, motor_id: int, protocol: str, protocol_order, assume_yes: bool, action):
    """Run `action(bus, motor_id)` while guaranteed to be in MIT protocol,
    then switch back to the original protocol. Returns (ok, new_motor_id, new_protocol)."""
    original_protocol = protocol
    cur_protocol = protocol

    if cur_protocol != "mit":
        print(f"Temporarily switching {cur_protocol} -> mit ...")
        ok, cur_protocol = _ensure_protocol(
            bus, motor_id, cur_protocol, "mit",
            protocol_order=protocol_order, assume_yes=assume_yes,
        )
        if not ok or cur_protocol != "mit":
            print(f"Failed to reach MIT protocol (got {cur_protocol}).")
            return False, motor_id, cur_protocol

    new_motor_id = action(bus, motor_id)
    if new_motor_id is None:
        return False, motor_id, cur_protocol
    motor_id = new_motor_id

    if original_protocol != "mit":
        print(f"Switching back mit -> {original_protocol} ...")
        ok, cur_protocol = _ensure_protocol(
            bus, motor_id, "mit", original_protocol,
            protocol_order=protocol_order, assume_yes=assume_yes,
        )
        if not ok:
            print(f"WARNING: failed to switch back to {original_protocol}; motor left in {cur_protocol}.")
            return True, motor_id, cur_protocol

    return True, motor_id, cur_protocol


def _do_set_zero(bus, motor_id: int):
    ok = set_zero_mit(bus, motor_id)
    if not ok:
        print("Set-zero command not acknowledged.")
        return None
    print("Zero point set.")
    return motor_id


def _do_change_id(bus, motor_id: int, new_id: int):
    print(f"Changing motor ID {motor_id} -> {new_id} ...")
    ok = change_motor_id(bus, motor_id, new_id)
    if not ok:
        print("ID-change command not acknowledged.")
        return None
    return new_id


def main() -> int:
    ap = argparse.ArgumentParser(description="RobStride minimal setup: CAN ID / protocol / zero")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--motor-id", type=int, default=None, help="known motor id; omit to auto-scan")
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--end-id", type=int, default=127)
    ap.add_argument("--protocol-order", default=",".join(DEFAULT_PROTOCOL_ORDER))
    ap.add_argument("--yes", action="store_true", help="auto-confirm reboot waits etc.")
    args = ap.parse_args()

    protocol_order = tuple(p.strip() for p in args.protocol_order.split(",") if p.strip())

    bus = _open_bus(args.interface, args.channel)
    try:
        _flush_bus(bus)
        motor_id, protocol = _detect(bus, args, protocol_order)
        if motor_id is None:
            return 1

        while True:
            _print_state(motor_id, protocol)
            print("1) Change CAN ID")
            print("2) Switch protocol (private <-> mit)")
            print("3) Set zero point")
            print("0) Exit")
            choice = input("Select: ").strip()

            if choice in ("0", ""):
                break

            elif choice == "1":
                raw = input("New CAN ID: ").strip()
                try:
                    new_id = _parse_int(raw)
                except Exception:
                    print("Invalid integer.")
                    continue
                ok, motor_id, protocol = _with_mit(
                    bus, motor_id, protocol, protocol_order, args.yes,
                    action=lambda b, mid: _do_change_id(b, mid, new_id),
                )
                if ok:
                    _wait_reboot(assume_yes=args.yes)
                    _flush_bus(bus)
                    detected = _find_protocol_for_id(bus, motor_id, protocol_order)
                    if detected is None:
                        print(f"WARNING: no response from new id={motor_id} after reboot.")
                    else:
                        protocol = detected

            elif choice == "2":
                target = input("Target protocol [private/mit]: ").strip().lower()
                if target not in ("private", "mit"):
                    print("Invalid protocol.")
                    continue
                ok, new_protocol = _ensure_protocol(
                    bus, motor_id, protocol, target,
                    protocol_order=protocol_order, assume_yes=args.yes,
                )
                if ok:
                    protocol = new_protocol
                else:
                    print(f"Failed to switch protocol (currently: {new_protocol}).")

            elif choice == "3":
                if not _confirm(
                    "Set zero at the CURRENT physical position?",
                    assume_yes=args.yes, default_yes=False,
                ):
                    continue
                
                # 1. 영점 설정 수행 (필요 시 임시 MIT 전환 포함)
                ok, motor_id, protocol = _with_mit(
                    bus, motor_id, protocol, protocol_order, args.yes,
                    action=_do_set_zero,
                )
                
                if not ok:
                    print("Zero-set failed.")
                else:
                    print("Waiting for motor to apply new zero coordinates...")
                    time.sleep(1.0)  # 좌표계 정렬 대기
                    _flush_bus(bus)
                    
                    # 2. 현재 위치를 확인하기 위해 commission_motor의 함수들 가져오기
                    print("Checking current mechanical position...")
                    try:
                        import math
                        import can
                        # 방금 확인하신 함수와 새로 만들 함수를 import 합니다.
                        from commission_motor import (
                            mit_position_command, 
                            _wait_reply, 
                            _matches_mit_reply,
                            parse_mit_reply_position  # 위 1번에서 추가한 함수
                        )
                        
                        # 모터에 "힘을 빼고(kp=0, kd=0) 현재 위치 정보를 답장해라"라는 빈 명령 전송
                        # mit_position_command 내부에 이미 _wait_reply 로직이 포함되어 있으므로
                        # reply 메시지를 직접 받아오기 위해 아래처럼 가볍게 다시 요청 과정을 거칩니다.
                        pmax = 12.57
                        data = [0] * 8  # 0으로 채워진 빈 명령 (Kp=0, Kd=0이므로 안전함)
                        msg = can.Message(arbitration_id=int(motor_id), data=data, is_extended_id=False)
                        bus.send(msg)
                        
                        reply = _wait_reply(
                            bus,
                            timeout_s=0.05,
                            predicate=lambda m: _matches_mit_reply(m, motor_id=int(motor_id)),
                        )
                        
                        if reply is not None:
                            # 3. 답장 메시지 파싱하여 현재 mechPos 계산
                            current_pos_rad = parse_mit_reply_position(reply, pmax=pmax)
                            current_pos_deg = math.degrees(current_pos_rad)
                            
                            print(f"\n[검증 결과] 현재 기계적 위치:")
                            print(f" -> {current_pos_rad:+.4f} rad")
                            print(f" -> {current_pos_deg:+.2f} deg")
                            
                            # 영점이 잘 잡혔다면 0.0 rad (0.0도)에 매우 가까워야 합니다.
                            if abs(current_pos_rad) < 0.05:
                                print("✅ 영점이 성공적으로 올바르게 잡혔습니다! (0.0 근처 정렬 완료)")
                            else:
                                print("⚠️ 경고: 영점 세팅 완료 판정은 떴으나, 현재 읽힌 위치가 0.0에서 벗어나 있습니다.")
                        else:
                            print("⚠️ 모터가 응답하지 않아 현재 위치(mechPos)를 검증할 수 없습니다.")
                            
                    except ImportError as e:
                        print(f"⚠️ 함수 수입(Import) 실패: {e}. 함수명을 다시 확인하세요.")
                    except Exception as e:
                        print(f"⚠️ 위치 확인 중 오류 발생: {e}")

            else:
                print("Invalid selection.")

        _print_state(motor_id, protocol)
        return 0

    finally:
        _shutdown_bus(bus)


if __name__ == "__main__":
    raise SystemExit(main())