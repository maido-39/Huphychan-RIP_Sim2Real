# Inverse balance 성공 스냅샷

2026-07-28 16:08:00 학습 실행에서 실물 동작이 성공한
`model_104850.pt`와 당시 사용한 코드 및 물리 설정을 보존한 폴더다.
원본 파일을 옮긴 것이 아니라 복사했으며, 체크포인트 SHA-256은 다음과 같다.

```text
7c08c7296053f875ecef0964249c58d34087598ec28c0d1ab78b5b1bc9260362
```

## 보존 내용

- `model_104850.pt`: 성공 체크포인트
- `params/agent.yaml`, `params/env.yaml`: 학습 실행이 기록한 전체 설정
- `inverse_env_cfg.py`: 학습 환경과 관측, 보상, randomization 설정
- `assets/inverse.xml`, `assets/meshes/`: 현재 MuJoCo 물리 모델과 메시
- `commission_motor.py`: 모터 프로토콜 설정 및 MIT 모드 제어
- `encoder_reader.py`: 진자 엔코더 리더
- `robot_state_reader.py`: CAN 모터 상태와 진자 엔코더 통합 리더
- `real_policy_inference.py`: 체크포인트 로딩 및 실물 관측/행동 변환
- `run_policy_motor.py`: 실물 정책 제어 루프
- `monitor.py`: MIT 피드백 디코더


## 성공 당시 핵심 물리량

`assets/inverse.xml` 기준:

- MuJoCo timestep: `0.005 s` (200 Hz)
- policy decimation: `4` (정책 갱신 50 Hz)
- 모터 관절 `Revolute 3`
  - 축: `(0, 0, -1)`
  - armature: `0.003`
  - frictionloss: `0.125`
  - position actuator: `kp=20.0`, `kv=0.502`
  - 제어 범위: `[-pi, pi] rad`
  - 힘 제한: `[-17, 17] N·m`
- 진자 관절 `Revolute 5`
  - 축: `(-1, 0, 0)`
  - damping: `0.000015`
  - frictionloss: `0.0`
  - armature: `0.0000042`
- rotor 질량: `0.053175 kg`
- pole holder 질량: `0.0301 kg`
- 정책 action scale: `20 deg`
- 관측 history: 2 프레임, 총 16차원
- 목표각 rate limit: `1500 deg/s`
- 목표각 acceleration limit: `15000 deg/s²`

세부 randomization, 관측 노이즈, 보상 및 종료 조건은
`inverse_env_cfg.py`와 `params/env.yaml`이 원본 값이다.

## 장비 연결

성공 구성의 실행 예시는 motor ID `8`, SocketCAN `can1`, 진자 엔코더
`/dev/ttyACM1`, 엔코더 baudrate `115200`이다. 실제 장치명이 바뀌었으면
`ip link`와 `/dev/ttyACM*`를 먼저 확인한다.

CAN 인터페이스 설정 예시:

```bash
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
ip -details link show can1
```

bitrate `1000000`은 Robstride MIT 구성에서 사용한 값이다. 모터에 저장된
bitrate가 다르면 반드시 그 값에 맞춘다.

모터 검색 및 MIT 모드 설정:

```bash
uv run python -u -m mjlab.tasks.inverse.successful_runs.inverse_balance_2026_07_28.commission_motor scan \
  --interface socketcan --channel can1 --start-id 0 --end-id 127

uv run python -u -m mjlab.tasks.inverse.successful_runs.inverse_balance_2026_07_28.commission_motor wizard \
  --interface socketcan --channel can1
```

`wizard`는 모터의 영점, ID 또는 프로토콜을 변경할 수 있으므로 축을 안전하게
고정하고 비상 정지 가능한 상태에서만 진행한다.

상태 입력 확인:

```bash
uv run python -u -m mjlab.tasks.inverse.successful_runs.inverse_balance_2026_07_28.robot_state_reader \
  --interface socketcan \
  --channel can1 \
  --motor-id 8 \
  --encoder-port /dev/ttyACM1
```

아래 세 값이 안정적으로 들어오는 것을 확인한 뒤 정책을 실행한다.

- `motor_angle_deg`
- `motor_velocity_deg_s`
- `pendulum_angle_deg`

## 정책 실행

저장소 루트(`/home/aril/mjlab`)에서 모듈 방식으로 실행한다.

```bash
uv run python -u -m mjlab.tasks.inverse.successful_runs.inverse_balance_2026_07_28.run_policy_motor \
  --checkpoint-file src/mjlab/tasks/inverse/successful_runs/inverse_balance_2026_07_28/model_104850.pt \
  --motor-id 8 \
  --channel can1 \
  --encoder-port /dev/ttyACM1 \
  --control-hz 200 \
  --policy-hz 50 \
  --kp 20.0 \
  --kd 0.8 \
  --action-scale-multiplier 1.0
```

실물 정책 로더는 mjlab task registry를 사용하므로, 이 스냅샷은 같은 mjlab
저장소와 `uv` 환경 안에서 실행한다. 향후 기본 inverse task 설정이 바뀌면
이 폴더의 `inverse_env_cfg.py`, XML 및 기록된 YAML을 기준으로 복원해야 한다.

처음 다시 재현할 때는 `--action-scale-multiplier 0.1` 또는 `0.2`로 시작하고,
회전 방향과 영점이 일치하는지 확인한 뒤 성공값 `1.0`으로 올린다. 아래로
매달린 진자각은 `0 rad`, 위로 선 상태는 `pi rad`가 되어야 한다. 축 방향이
반대면 `--cylinder-sign -1` 또는 `--pole-sign -1`을 사용한다.
