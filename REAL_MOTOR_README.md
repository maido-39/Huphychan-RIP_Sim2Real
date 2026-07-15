# Inverse Real Motor README

이 문서는 `Mjlab-Inverse-Balance` 정책을 실물 모터에서 돌릴 때 필요한 절차를 정리한 문서입니다.

정리 범위:

1. CANable을 `can0`로 올리기
2. 모터 스캔 / 커미셔닝
3. 모터 상태 모니터링
4. 엔코더 + 모터 상태 통합 읽기
5. 학습된 체크포인트를 실물 모터에 재생하기

모든 명령은 프로젝트 루트에서 실행합니다.

```bash
cd /home/aril/mjlab
```

## 1. CAN 연결 열기

먼저 CAN 커널 모듈을 올립니다.

```bash
sudo modprobe can
sudo modprobe can_raw
sudo modprobe slcan
```

CANable 장치가 어떤 포트로 잡혔는지 확인합니다.

```bash
ls /dev/ttyACM*
```

예를 들어 `/dev/ttyACM0`로 잡혔다면, `can0`로 연결합니다.

```bash
sudo slcand -o -c -s8 /dev/ttyACM0 can0
sudo ip link set up can0
```

상태 확인:

```bash
ip link show can0
```

정상 예시:

```text
can0: <NOARP,UP,LOWER_UP>
```

CAN 프레임이 실제로 들어오는지 보고 싶으면:

```bash
candump can0
```

## 2. 모터 스캔 / 커미셔닝

모터가 어떤 ID와 프로토콜로 잡히는지 먼저 스캔합니다.

```bash
uv run python src/mjlab/tasks/inverse/commission_motor.py scan \
  --channel can0
```

가이드형 커미셔닝을 쓰려면:

```bash
uv run python src/mjlab/tasks/inverse/commission_motor.py wizard \
  --channel can0
```

모터 ID를 알고 있으면:

```bash
uv run python src/mjlab/tasks/inverse/commission_motor.py wizard \
  --channel can0 \
  --motor-id 8
```

이 스크립트는 MIT 모드 전환, ID 설정, zero, 간단한 모션 테스트까지 포함한 흐름입니다.

## 3. 모터 상태만 모니터링

`monitor.py`는 MIT 응답 프레임을 디코딩해서 모터 각도/속도/토크를 봅니다.

### poll 모드

직접 clear-fault ping을 보내면서 응답을 받습니다.

```bash
uv run python src/mjlab/tasks/inverse/monitor.py \
  --channel can0 \
  --motor-id 8
```

### passive 모드

아무것도 보내지 않고, 다른 프로세스가 보내는 명령의 응답만 수동으로 듣습니다.

```bash
uv run python src/mjlab/tasks/inverse/monitor.py \
  --channel can0 \
  --motor-id 8 \
  --mode passive
```

## 4. 엔코더 + 모터 상태 같이 보기

`robot_state_reader.py`는

- 모터 각도
- 모터 속도
- 모터 토크
- 진자 엔코더 각도

를 함께 읽기 위한 통합 리더입니다.

예시:

```bash
uv run python src/mjlab/tasks/inverse/robot_state_reader.py \
  --channel can0 \
  --motor-id 8 \
  --motor-mode passive \
  --encoder-port /dev/ttyACM0
```

엔코더 없이 모터값만 확인할 수도 있습니다.

```bash
uv run python src/mjlab/tasks/inverse/robot_state_reader.py \
  --channel can0 \
  --motor-id 8 \
  --motor-mode passive
```

## 5. 학습된 정책을 실물 모터에서 재생

`run_policy_motor.py`는 다음을 연결합니다.

- `real_policy_inference.py`
- `RobotStateReader`
- MIT 모터 제어 루프

즉, 체크포인트를 로드해서 실물에서 바로 정책을 재생하는 실행 스크립트입니다.

권장 순서:

1. `commission_motor.py`로 MIT 모드/ID 확인
2. `robot_state_reader.py`로 상태가 정상인지 확인
3. `run_policy_motor.py`를 약한 배율로 시작

### 첫 실행 권장 예시

처음에는 반드시 작은 배율로 시작합니다.

```bash
uv run python src/mjlab/tasks/inverse/run_policy_motor.py \
  --checkpoint-file logs/rsl_rl/inverse_balance/2026-07-15_14-39-22/model_900.pt \
  --motor-id 8 \
  --channel can0 \
  --encoder-port /dev/ttyACM0 \
  --control-hz 50 \
  --state-read-hz 200 \
  --action-scale-multiplier 0.2
```

배율을 더 크게 주고 싶으면:

```bash
uv run python src/mjlab/tasks/inverse/run_policy_motor.py \
  --checkpoint-file logs/rsl_rl/inverse_balance/2026-07-15_14-39-22/model_900.pt \
  --motor-id 8 \
  --channel can0 \
  --encoder-port /dev/ttyACM0 \
  --control-hz 50 \
  --state-read-hz 200 \
  --action-scale-multiplier 1.0
```

## 6. 실행 시 생성되는 로그

`run_policy_motor.py`는 종료 시 다음을 저장합니다.

- CSV 로그
- target/current angle 텍스트 로그
- PNG 그래프

기본 로그 폴더:

```text
logs/
```

## 7. 빠른 점검 체크리스트

실물 재생 전에 아래 순서로 확인하면 됩니다.

1. `ip link show can0` 에서 `UP`
2. `commission_motor.py scan` 으로 모터 발견
3. `monitor.py` 또는 `robot_state_reader.py`로 값이 갱신되는지 확인
4. `run_policy_motor.py`를 `action-scale-multiplier 0.2` 정도로 시작
5. 이상 없으면 점진적으로 배율 증가

## 8. 주의

- `monitor.py --mode poll` 와 실제 제어 루프를 동시에 돌리면 버스를 방해할 수 있습니다.
- 실시간 제어 중에는 `passive` 모드 모니터링이 더 안전합니다.
- 처음부터 `action-scale-multiplier 1.0`으로 주면 예상보다 크게 움직일 수 있습니다.
- 엔코더가 없으면 정책 입력이 완전하지 않아서 `run_policy_motor.py`는 정상 동작하지 않을 수 있습니다.
