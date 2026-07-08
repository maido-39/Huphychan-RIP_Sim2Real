"""
실물 적용용 Inverse policy 추론 템플릿.

강화학습 설정 요약
- 태스크: Mjlab-Inverse-Balance
- 관측 순서(현재 시점 1프레임):
  1) cylinder_angle_rad / (3*pi)
  2) cos(cylinder_angle_rad)
  3) sin(cylinder_angle_rad)
  4) cos(pole_angle_rad)
  5) sin(pole_angle_rad)
  6) cylinder_vel_rad_s / 30
  7) pole_vel_rad_s / 30
  8) last_action
- 관측 history_length = 2
  -> policy 입력은 [이전 obs, 현재 obs] 순서로 이어붙인 16차원 벡터
- action 의미:
  -> 위치 명령용 normalized action in [-1, 1]
  -> target_angle_rad = action * (0.5 * pi)
- 현재 학습 설정 기준 주기:
  -> timestep = 0.005
  -> decimation = 1
  -> policy/action update = 200 Hz

실물 설정 요약
- 실물에서 들어오는 각도 단위가 degree 라면 반드시 rad로 변환
- 실물에서 들어오는 속도 단위가 degree/s 라면 반드시 rad/s로 변환
- 아래로 매달린 pole 상태를 0 rad로 맞추고,
  위로 선 pole 상태를 pi rad로 맞춰야 학습 정책과 일치
- 모터 회전 방향이 시뮬레이터와 반대면 반드시 부호를 뒤집어 맞춰야 함
- 토크(Nm)는 현재 policy 입력에 사용하지 않음
  -> 안전 제한, 과부하 감지, 로그 저장 용도로만 활용 권장

이 파일의 목적
- 학습된 .pt 체크포인트를 그대로 불러온다
- 실물 센서값(deg, deg/s)을 policy 입력으로 변환한다
- policy 출력을 목표 각도(rad, deg)로 변환한다
- 실제 드라이버 송신 코드는 비워두고, 연결 지점만 명확히 남긴다

CAN 실물 연결 요약
- Linux SocketCAN 기준으로 can0 인터페이스에 직접 붙을 수 있게 구성
- policy는 CAN을 모르고, CAN 드라이버가 아래 역할을 담당
  1) 모터/엔코더 CAN 상태 프레임 수신
  2) 현재 각도/속도/토크를 degree, degree/s, Nm로 복원
  3) policy 출력 목표각(deg)을 CAN 명령 프레임으로 송신
- 중요한 점:
  -> CAN 프레임의 ID와 payload 바이트 구조는 모터 제조사 프로토콜마다 다름
  -> 그래서 아래 코드에는 "실행 루프"와 "SocketCAN 송수신"은 넣고,
     실제 payload pack/unpack만 드라이버 함수로 분리해 두었다
"""

from __future__ import annotations

import math
import select
import socket
import struct
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum

import torch
import tyro

import mjlab.tasks  # noqa: F401  # registry 채우기
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.inverse import inverse_env_cfg as inv_cfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


TASK_ID = "Mjlab-Inverse-Balance"
OBS_DIM = 8
HISTORY_LENGTH = 2
POLICY_INPUT_DIM = OBS_DIM * HISTORY_LENGTH
ACTION_SCALE_RAD = 0.5 * math.pi
CAN_FRAME_FORMAT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FORMAT)


@dataclass(frozen=True)
class RealInferenceConfig:
  checkpoint_file: str
  device: str = "cpu"

  # 아래 두 부호는 실물 축 방향이 시뮬레이터와 다를 때 -1.0으로 뒤집어 사용.
  cylinder_sign: float = 1.0
  pole_sign: float = 1.0

  # 실물 zero-calibration 오프셋. degree 기준으로 넣고 내부에서 rad로 변환한다.
  cylinder_zero_deg: float = 0.0
  pole_zero_deg: float = 0.0


@dataclass(frozen=True)
class CanBusConfig:
  channel: str = "can0"
  read_timeout_s: float = 0.02


@dataclass(frozen=True)
class CanIds:
  cylinder_state: int
  pole_state: int
  cylinder_command: int
  enable_command: int | None = None
  disable_command: int | None = None


@dataclass(frozen=True)
class RealMeasurement:
  """실물에서 읽어온 센서값 1프레임."""

  cylinder_angle_deg: float
  pole_angle_deg: float
  cylinder_vel_deg_s: float
  pole_vel_deg_s: float
  torque_nm: float | None = None


@dataclass(frozen=True)
class CanFrame:
  can_id: int
  data: bytes


class RunMode(str, Enum):
  EXAMPLE = "example"
  CAN_LOOP = "can-loop"


@dataclass(frozen=True)
class MainConfig:
  checkpoint_file: str
  device: str = "cpu"
  mode: RunMode = RunMode.EXAMPLE
  can_channel: str = "can0"

  cylinder_sign: float = 1.0
  pole_sign: float = 1.0
  cylinder_zero_deg: float = 0.0
  pole_zero_deg: float = 0.0

  # 아래 값들은 사용 중인 드라이버 프로토콜에 맞게 맞춰야 한다.
  cylinder_state_id: int = 0x141
  pole_state_id: int = 0x142
  cylinder_command_id: int = 0x241
  enable_command_id: int | None = None
  disable_command_id: int | None = None


class InverseRealPolicy:
  """학습된 inverse policy를 실물 입력에 연결하기 위한 얇은 래퍼."""

  def __init__(self, cfg: RealInferenceConfig):
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self._policy = self._load_policy(cfg)
    self._last_action = 0.0
    self._history: deque[torch.Tensor] = deque(maxlen=HISTORY_LENGTH)

  def _load_policy(self, cfg: RealInferenceConfig):
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    wrapped_env = RslRlVecEnvWrapper(env)

    agent_cfg = load_rl_cfg(TASK_ID)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=cfg.device)
    runner.load(
      cfg.checkpoint_file,
      load_cfg={"actor": True},
      strict=True,
      map_location=cfg.device,
    )
    policy = runner.get_inference_policy(device=cfg.device)

    # 추론용으로만 쓰므로 env는 바로 닫아도 된다.
    wrapped_env.close()
    return policy

  @staticmethod
  def _deg_to_rad(value_deg: float) -> float:
    return value_deg * math.pi / 180.0

  def _normalize_angles(self, meas: RealMeasurement) -> tuple[float, float]:
    cylinder_angle_rad = self._deg_to_rad(
      (meas.cylinder_angle_deg - self.cfg.cylinder_zero_deg) * self.cfg.cylinder_sign
    )
    pole_angle_rad = self._deg_to_rad(
      (meas.pole_angle_deg - self.cfg.pole_zero_deg) * self.cfg.pole_sign
    )
    return cylinder_angle_rad, pole_angle_rad

  def _normalize_velocities(self, meas: RealMeasurement) -> tuple[float, float]:
    cylinder_vel_rad_s = self._deg_to_rad(
      meas.cylinder_vel_deg_s * self.cfg.cylinder_sign
    )
    pole_vel_rad_s = self._deg_to_rad(meas.pole_vel_deg_s * self.cfg.pole_sign)
    return cylinder_vel_rad_s, pole_vel_rad_s

  def build_single_observation(self, meas: RealMeasurement) -> torch.Tensor:
    """실물 센서값 1프레임을 학습 때의 8차원 observation으로 변환."""
    cylinder_angle_rad, pole_angle_rad = self._normalize_angles(meas)
    cylinder_vel_rad_s, pole_vel_rad_s = self._normalize_velocities(meas)

    obs = torch.tensor(
      [
        cylinder_angle_rad / inv_cfg._MAX_CYLINDER_ROTATION,
        math.cos(cylinder_angle_rad),
        math.sin(cylinder_angle_rad),
        math.cos(pole_angle_rad),
        math.sin(pole_angle_rad),
        cylinder_vel_rad_s / inv_cfg._VEL_OBS_SCALE,
        pole_vel_rad_s / inv_cfg._VEL_OBS_SCALE,
        self._last_action,
      ],
      dtype=torch.float32,
      device=self.device,
    )
    return obs

  def build_policy_input(self, meas: RealMeasurement) -> torch.Tensor:
    """history_length=2 규칙에 맞춰 [previous_obs, current_obs]를 만든다."""
    obs = self.build_single_observation(meas)

    if not self._history:
      # 첫 스텝은 이전값이 없으므로 동일한 obs로 history를 채운다.
      self._history.append(obs.clone())
      self._history.append(obs.clone())
    else:
      self._history.append(obs.clone())
      while len(self._history) < HISTORY_LENGTH:
        self._history.appendleft(obs.clone())

    stacked = torch.cat(list(self._history), dim=0)
    return stacked.unsqueeze(0)  # (1, 16)

  def infer(self, meas: RealMeasurement) -> dict[str, float]:
    """실물 센서값 -> policy 출력 -> 목표 모터 각도 변환."""
    policy_input = self.build_policy_input(meas)

    with torch.no_grad():
      action_tensor = self._policy(policy_input)

    action = float(action_tensor.squeeze().item())
    action = max(-1.0, min(1.0, action))
    self._last_action = action

    target_angle_rad = action * ACTION_SCALE_RAD
    target_angle_deg = target_angle_rad * 180.0 / math.pi

    return {
      "policy_action": action,
      "target_angle_rad": target_angle_rad,
      "target_angle_deg": target_angle_deg,
    }


class SocketCanTransport:
  """Linux SocketCAN raw socket thin wrapper."""

  def __init__(self, cfg: CanBusConfig):
    self.cfg = cfg
    self._sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    self._sock.bind((cfg.channel,))
    self._sock.setblocking(False)

  def close(self) -> None:
    self._sock.close()

  def send(self, frame: CanFrame) -> None:
    dlc = min(len(frame.data), 8)
    payload = frame.data[:dlc].ljust(8, b"\x00")
    packed = struct.pack(CAN_FRAME_FORMAT, frame.can_id, dlc, payload)
    self._sock.send(packed)

  def recv(self) -> CanFrame | None:
    ready, _, _ = select.select([self._sock], [], [], self.cfg.read_timeout_s)
    if not ready:
      return None
    raw = self._sock.recv(CAN_FRAME_SIZE)
    can_id, dlc, data = struct.unpack(CAN_FRAME_FORMAT, raw)
    return CanFrame(can_id=can_id, data=data[:dlc])


class InverseCanDriver:
  """실제 CAN 프로토콜을 policy 루프에 연결하는 드라이버 베이스."""

  def __init__(self, transport: SocketCanTransport, ids: CanIds):
    self.transport = transport
    self.ids = ids
    self._latest_cylinder: RealMeasurement | None = None
    self._latest_pole: RealMeasurement | None = None

  def close(self) -> None:
    self.transport.close()

  def enable(self) -> None:
    if self.ids.enable_command is None:
      return
    self.transport.send(CanFrame(can_id=self.ids.enable_command, data=self._encode_enable()))

  def disable(self) -> None:
    if self.ids.disable_command is None:
      return
    self.transport.send(
      CanFrame(can_id=self.ids.disable_command, data=self._encode_disable())
    )

  def read_measurement(self) -> RealMeasurement:
    """양쪽 상태 프레임을 모아 policy 입력용 measurement 1개를 만든다."""
    while True:
      frame = self.transport.recv()
      if frame is None:
        raise TimeoutError("CAN state frame timeout")

      if frame.can_id == self.ids.cylinder_state:
        self._latest_cylinder = self._decode_cylinder_state(frame)
      elif frame.can_id == self.ids.pole_state:
        self._latest_pole = self._decode_pole_state(frame)

      if self._latest_cylinder is not None and self._latest_pole is not None:
        cylinder = self._latest_cylinder
        pole = self._latest_pole
        return RealMeasurement(
          cylinder_angle_deg=cylinder.cylinder_angle_deg,
          pole_angle_deg=pole.pole_angle_deg,
          cylinder_vel_deg_s=cylinder.cylinder_vel_deg_s,
          pole_vel_deg_s=pole.pole_vel_deg_s,
          torque_nm=cylinder.torque_nm,
        )

  def send_target_angle_deg(self, target_angle_deg: float) -> None:
    payload = self._encode_position_command(target_angle_deg)
    self.transport.send(CanFrame(can_id=self.ids.cylinder_command, data=payload))

  def _encode_enable(self) -> bytes:
    return b"\x00" * 8

  def _encode_disable(self) -> bytes:
    return b"\x00" * 8

  def _decode_cylinder_state(self, frame: CanFrame) -> RealMeasurement:
    raise NotImplementedError

  def _decode_pole_state(self, frame: CanFrame) -> RealMeasurement:
    raise NotImplementedError

  def _encode_position_command(self, target_angle_deg: float) -> bytes:
    raise NotImplementedError


class ExamplePackedCanDriver(InverseCanDriver):
  """예시용 드라이버.

  가정한 payload 형식:
  - state frame: int16 angle_cdeg, int16 vel_cdeg_s, int16 torque_mNm, reserved2
  - command frame: int32 target_cdeg, reserved4

  즉 0.01 deg 단위와 0.001 Nm 단위를 쓰는 단순 예시다.
  실제 Robstride 프로토콜과 다를 수 있으니, 실사용 전 데이터시트에 맞게
  이 클래스의 pack/unpack만 수정하면 된다.
  """

  def _decode_state_common(self, frame: CanFrame) -> tuple[float, float, float]:
    if len(frame.data) < 6:
      raise ValueError(f"State frame too short: can_id=0x{frame.can_id:X}")
    angle_cdeg, vel_cdeg_s, torque_mnm = struct.unpack("<hhh", frame.data[:6])
    return angle_cdeg / 100.0, vel_cdeg_s / 100.0, torque_mnm / 1000.0

  def _decode_cylinder_state(self, frame: CanFrame) -> RealMeasurement:
    angle_deg, vel_deg_s, torque_nm = self._decode_state_common(frame)
    return RealMeasurement(
      cylinder_angle_deg=angle_deg,
      pole_angle_deg=0.0,
      cylinder_vel_deg_s=vel_deg_s,
      pole_vel_deg_s=0.0,
      torque_nm=torque_nm,
    )

  def _decode_pole_state(self, frame: CanFrame) -> RealMeasurement:
    angle_deg, vel_deg_s, torque_nm = self._decode_state_common(frame)
    return RealMeasurement(
      cylinder_angle_deg=0.0,
      pole_angle_deg=angle_deg,
      cylinder_vel_deg_s=0.0,
      pole_vel_deg_s=vel_deg_s,
      torque_nm=torque_nm,
    )

  def _encode_position_command(self, target_angle_deg: float) -> bytes:
    target_cdeg = int(round(target_angle_deg * 100.0))
    return struct.pack("<i4x", target_cdeg)


def _print_example(result: dict[str, float]) -> None:
  print("policy_action     :", f"{result['policy_action']:.6f}")
  print("target_angle_rad  :", f"{result['target_angle_rad']:.6f}")
  print("target_angle_deg  :", f"{result['target_angle_deg']:.6f}")


def _to_inference_cfg(cfg: MainConfig) -> RealInferenceConfig:
  return RealInferenceConfig(
    checkpoint_file=cfg.checkpoint_file,
    device=cfg.device,
    cylinder_sign=cfg.cylinder_sign,
    pole_sign=cfg.pole_sign,
    cylinder_zero_deg=cfg.cylinder_zero_deg,
    pole_zero_deg=cfg.pole_zero_deg,
  )


def _to_can_ids(cfg: MainConfig) -> CanIds:
  return CanIds(
    cylinder_state=cfg.cylinder_state_id,
    pole_state=cfg.pole_state_id,
    cylinder_command=cfg.cylinder_command_id,
    enable_command=cfg.enable_command_id,
    disable_command=cfg.disable_command_id,
  )


def _run_example(cfg: MainConfig) -> None:
  policy = InverseRealPolicy(_to_inference_cfg(cfg))

  # 예제 입력:
  # - 아래로 매달린 시작 자세(대략 0도)
  # - 속도 0
  # 실제 적용 시에는 이 부분을 실물 센서값 읽기 코드로 바꾸면 된다.
  example = RealMeasurement(
    cylinder_angle_deg=0.0,
    pole_angle_deg=0.0,
    cylinder_vel_deg_s=0.0,
    pole_vel_deg_s=0.0,
    torque_nm=0.0,
  )

  result = policy.infer(example)
  _print_example(result)

  print()
  print("mode=example 완료")


def _run_can_loop(cfg: MainConfig) -> None:
  policy = InverseRealPolicy(_to_inference_cfg(cfg))
  transport = SocketCanTransport(CanBusConfig(channel=cfg.can_channel))
  driver = ExamplePackedCanDriver(transport=transport, ids=_to_can_ids(cfg))

  print(f"CAN loop start on {cfg.can_channel}")
  print(
    "주의: 현재 CAN payload 형식은 ExamplePackedCanDriver 기준 예시이므로 "
    "실제 모터 프로토콜과 다르면 decode/encode 함수를 수정해야 합니다."
  )

  try:
    driver.enable()
    while True:
      meas = driver.read_measurement()
      result = policy.infer(meas)

      # 토크는 정책 입력이 아니라 안전/로그용으로만 사용.
      if meas.torque_nm is not None and abs(meas.torque_nm) > 6.0:
        driver.disable()
        raise RuntimeError("Safe torque limit exceeded")

      driver.send_target_angle_deg(result["target_angle_deg"])
      print(
        "cyl={:+8.3f} deg  pole={:+8.3f} deg  "
        "cmd={:+8.3f} deg  act={:+6.3f}".format(
          meas.cylinder_angle_deg,
          meas.pole_angle_deg,
          result["target_angle_deg"],
          result["policy_action"],
        )
      )
  finally:
    driver.disable()
    driver.close()


def main(cfg: MainConfig) -> None:
  if cfg.mode == RunMode.EXAMPLE:
    _run_example(cfg)
    return
  if cfg.mode == RunMode.CAN_LOOP:
    _run_can_loop(cfg)
    return
  raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
  main(tyro.cli(MainConfig))
