#!/usr/bin/env python3
import csv
import os
import threading
import time
from datetime import datetime
from typing import Any, Optional

# 가정: 기존 함수들이 정의되어 있는 모듈에서 로드하거나, 본문에 그대로 위치시킵니다.
from commission_motor import (
  _open_bus,
  _shutdown_bus,
  disable_mit,
  enable_mit,
  mit_position_command,
)
from robot_state_reader import RobotStateReader

# set_zero_mit는 더 이상 매번 호출하지 않으므로 import에서 제거했습니다.
# (전원 사이클 이후에도 영점이 유지된다는 걸 확인했다면 이대로 두시고,
#  혹시 다시 영점을 잡아야 할 일이 생기면 아래처럼 필요할 때만 import해서 쓰세요:
#    from commission_motor import set_zero_mit

# pole 인코더가 연결돼 있으면 포트를 지정한다. 연결이 없으면 None으로 두면
# 로그에서 pole 각/속도만 빈 값으로 남고 나머지(모터 각/속도/토크)는 정상 기록된다.
ENCODER_PORT: Optional[str] = "/dev/ttyACM0"

_LOG_HEADER = [
  "time_s",
  "motor_angle_deg",
  "motor_vel_deg_s",
  "motor_torque_nm",
  "pole_angle_deg",
  "pole_vel_deg_s",
]


class MotorTestController:
  def __init__(
    self, bus: Any, motor_id: int, state_reader: Optional[RobotStateReader] = None
  ):
    self.bus = bus
    self.motor_id = motor_id
    self.state_reader = state_reader

    # 실시간으로 변경될 모터 명령 변수들 (초기값은 안전하게 모두 0)
    self.target_deg = 0.0
    self.kp = 0.0
    self.kd = 0.0
    self.velocity_deg_s = 0.0
    self.torque_nm = 0.0

    self.running = False
    self.tx_thread = None

    # 로깅 -- 기본은 꺼짐. "log"/"log on"/"log off" 명령으로 언제든 켰다 껐다
    # 할 수 있다. 켜질 때 로그 파일을 (한 번만) 새로 만들고, 이후 껐다 켜도
    # 같은 파일에 이어서 기록한다.
    self.logging_enabled = False
    self._log_file = None
    self._log_writer = None
    self._log_path: Optional[str] = None
    self._log_start_t: Optional[float] = None
    self._prev_encoder_state = None

  def _open_log(self) -> None:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("logs", f"test_log_{ts}.csv")
    self._log_file = open(path, "w", newline="")
    self._log_writer = csv.writer(self._log_file)
    self._log_writer.writerow(_LOG_HEADER)
    self._log_start_t = time.monotonic()
    self._log_path = path

  def set_logging(self, enabled: bool) -> None:
    if self.state_reader is None:
      print(
        "⚠️ state_reader가 없어서 로깅을 켤 수 없습니다 (RobotStateReader 초기화 확인)."
      )
      return
    if enabled == self.logging_enabled:
      return
    self.logging_enabled = enabled
    if enabled and self._log_writer is None:
      self._open_log()
    state = "켜짐" if enabled else "꺼짐"
    suffix = f" (파일: {self._log_path})" if enabled else ""
    print(f"📝 로깅 {state}{suffix}")

  def toggle_logging(self) -> None:
    self.set_logging(not self.logging_enabled)

  def _log_row(self) -> None:
    if (
      not self.logging_enabled or self._log_writer is None or self.state_reader is None
    ):
      return

    state = self.state_reader.get_state()
    motor_angle_deg = state.motor_angle_deg
    motor_vel_deg_s = state.motor_velocity_deg_s
    motor_torque_nm = state.motor_torque_nm
    if motor_angle_deg is None or motor_vel_deg_s is None or motor_torque_nm is None:
      return  # 아직 모터 상태를 못 받았으면 이번 샘플은 건너뜀

    pole_angle_deg = state.pendulum_angle_deg
    pole_vel_deg_s = 0.0
    encoder_state = (
      self.state_reader.encoder_reader.latest
      if self.state_reader.encoder_reader is not None
      else None
    )
    if encoder_state is not None and self._prev_encoder_state is not None:
      enc_dt = max(encoder_state.timestamp - self._prev_encoder_state.timestamp, 1e-4)
      raw_delta = encoder_state.angle_deg - self._prev_encoder_state.angle_deg
      delta_deg = (raw_delta + 180) % 360 - 180  # 0-360 랩어라운드 최단경로 보정
      pole_vel_deg_s = delta_deg / enc_dt
    self._prev_encoder_state = encoder_state

    elapsed_t = time.monotonic() - self._log_start_t
    self._log_writer.writerow(
      [
        f"{elapsed_t:.4f}",
        f"{motor_angle_deg:.3f}",
        f"{motor_vel_deg_s:.3f}",
        f"{motor_torque_nm:.4f}",
        "" if pole_angle_deg is None else f"{pole_angle_deg:.3f}",
        f"{pole_vel_deg_s:.3f}",
      ]
    )

  def close(self) -> None:
    """프로그램 종료 시 로그 파일을 닫는다."""
    if self._log_file is not None:
      self._log_file.close()
      self._log_file = None
      self._log_writer = None

  def _tx_loop(self):
    """백그라운드에서 20ms(50Hz) 주기로 최신 명령을 모터에 지속 전송하는 루프"""
    print("\n[백그라운드 송신 루프 시작]")
    while self.running:
      # 20ms(0.02초) 주기로 지속 송신하여 timeout(0.03초)을 방지합니다.
      mit_position_command(
        bus=self.bus,
        motor_id=self.motor_id,
        position_deg=self.target_deg,
        kp=self.kp,
        kd=self.kd,
        velocity_deg_s=self.velocity_deg_s,
        torque_nm=self.torque_nm,
        timeout_s=0.03,
      )
      self._log_row()
      time.sleep(0.02)
    print("[백그라운드 송신 루프 종료]")

  def start_control(self):
    """모터 전원을 켜고(Enable) 주기적 송신 스레드를 시작합니다."""
    print(f"\n[알림] 모터 ID {self.motor_id} 활성화(Enable) 시도 중...")
    if not enable_mit(self.bus, self.motor_id):
      print("❌ 모터 활성화에 실패했습니다.")
      return False

    self.running = True
    self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
    self.tx_thread.start()
    return True

  def stop_control(self):
    """송신 스레드를 중지하고 모터 전원을 끕니다(Disable)."""
    self.running = False
    if self.tx_thread:
      self.tx_thread.join(timeout=1.0)
      self.tx_thread = None

    print(f"\n[알림] 모터 ID {self.motor_id} 비활성화(Disable) 시도 중...")
    disable_mit(self.bus, self.motor_id)


def _handle_log_command(controller: MotorTestController, text: str) -> bool:
  """입력이 로깅 관련 명령("log"/"log on"/"log off")이면 처리하고 True를 반환한다."""
  lowered = text.strip().lower()
  if lowered == "log":
    controller.toggle_logging()
    return True
  if lowered == "log on":
    controller.set_logging(True)
    return True
  if lowered == "log off":
    controller.set_logging(False)
    return True
  return False


def interactive_test_main(
  bus: Any, motor_id: int, state_reader: Optional[RobotStateReader] = None
):
  controller = MotorTestController(bus, motor_id, state_reader=state_reader)

  # 영점 설정 단계는 제거했습니다 -- 이전에 설정해 둔 기계적 영점을 그대로 사용합니다.

  print("\n====================================================")
  print("                실시간 모터 시험 제어기                ")
  print("====================================================")
  print(" * 대기 상태에서 원하는 파라미터를 입력하면 구동이 시작됩니다.")
  print(" * 구동 중 '0'을 입력하면 안전하게 모터가 비활성화(Idle)됩니다.")
  if state_reader is not None:
    print(" * 'log' 입력 시 pole/모터 각도·속도·토크 로깅을 켰다 껐다 할 수 있습니다.")
  print("====================================================")

  in_control_mode = False

  try:
    while True:
      if not in_control_mode:
        print("\n[현재: 대기 상태] 파라미터를 입력하거나 0을 누르면 종료합니다.")
        user_input = input(
          "구동을 시작하려면 Enter를 누르세요 (로깅 켜기/끄기: log, 종료: exit): "
        ).strip()
        if user_input == "exit":
          print("테스트 프로그램을 종료합니다.")
          break
        if _handle_log_command(controller, user_input):
          continue

        # 파라미터 입력 받기
        try:
          print("\n--- 파라미터 설정 ---")
          kp = float(input("1. Kp 게인 입력 (추천: 5.0 ~ 15.0): ") or 0.0)
          kd = float(input("2. Kd 게인 입력 (추천: 0.1 ~ 0.5): ") or 0.0)
          target_deg = float(input("3. 목표 각도(deg) 입력 (예: 45.0): ") or 0.0)
          velocity_deg_s = float(input("4. 목표 속도(deg/s) 입력 (기본 0): ") or 0.0)
          torque_nm = float(input("5. 피드포워드 토크(Nm) 입력 (기본 0): ") or 0.0)
        except ValueError:
          print("⚠️ 올바른 숫자를 입력해 주세요. 대기 상태로 돌아갑니다.")
          continue

        # 설정값 적용 및 스레드 가동
        controller.kp = kp
        controller.kd = kd
        controller.target_deg = target_deg
        controller.velocity_deg_s = velocity_deg_s
        controller.torque_nm = torque_nm

        if controller.start_control():
          in_control_mode = True

      else:
        print(
          f"\n[현재: 구동 상태] 각도={controller.target_deg}°, Kp={controller.kp}, Kd={controller.kd}"
        )
        print(" -> 새로운 목표 각도를 입력하면 즉시 반영됩니다.")
        print(" -> '0'을 입력하면 안전하게 제어를 멈추고 대기 상태로 빠져나갑니다.")

        next_action = input(
          "목표 각도 입력 (정지: stop, 로깅 켜기/끄기: log): "
        ).strip()

        if _handle_log_command(controller, next_action):
          continue

        if next_action == "stop":
          # 구동 종료 및 대기 모드 진입
          controller.stop_control()
          in_control_mode = False
          print("✅ 제어가 중지되었습니다. 모터가 안전하게 이완(Idle)되었습니다.")
        else:
          try:
            # 구동 상태를 유지하면서 목표 각도만 실시간 변경
            new_angle = float(next_action)
            controller.target_deg = new_angle
            print(f"🔄 목표 각도가 {new_angle}° 로 변경되었습니다.")
          except ValueError:
            print("⚠️ 올바른 각도(숫자)를 입력하거나 0을 입력하세요.")

  except KeyboardInterrupt:
    print("\n⚠️ 강제 종료 감지! 모터를 안전하게 차단합니다.")
  finally:
    # 어떤 상황에서든 종료될 때 모터가 켜진 상태로 방치되지 않도록 안전장치
    controller.stop_control()
    controller.close()


if __name__ == "__main__":
  # 사용 예시 (실제 실행 환경의 interface와 channel, motor_id에 맞게 설정하세요)
  bus = _open_bus("socketcan", "can0")

  state_reader: Optional[RobotStateReader] = None
  try:
    state_reader = RobotStateReader(
      motor_id=8,
      encoder_port=ENCODER_PORT,
      can_interface="socketcan",
      can_channel="can0",
      motor_mode="passive",
      motor_rate_hz=50.0,
    )
    state_reader.start()
  except Exception as exc:
    print(f"⚠️ 상태 리더(RobotStateReader) 초기화 실패 -- 로깅 없이 진행합니다: {exc}")
    state_reader = None

  try:
    interactive_test_main(bus, motor_id=8, state_reader=state_reader)
  finally:
    if state_reader is not None:
      state_reader.stop()
    _shutdown_bus(bus)
