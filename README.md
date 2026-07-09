# Inverse Task

이 폴더는 `Mjlab-Inverse-Balance` 태스크의 모델, 강화학습 설정, 실물 적용 코드를 모아두는 위치입니다.

## 핵심 파일

### `inverse_env_cfg.py`
- 강화학습 환경 설정 파일입니다.
- 이 파일에서 다음 내용을 정의합니다.
  - observation 구성
  - action 의미와 스케일
  - reward 함수
  - termination 조건
  - reset curriculum
  - 시뮬레이션 timestep / decimation
  - PPO 학습 파라미터
- 즉, "무엇을 보고", "어떻게 행동하고", "무엇을 잘했다고 보상할지"를 정하는 중심 파일입니다.

### `assets/inverse.xml`
- MuJoCo에서 직접 읽는 MJCF 모델 파일입니다.
- 링크, 조인트, 관성, 시각 메쉬, 액추에이터 설정이 들어 있습니다.
- 강화학습과 play 실행 시 실제 물리 모델은 이 파일을 기준으로 동작합니다.
- CAD/URDF/xacro에서 모델을 다시 내보내면, 최종적으로 이 파일에 반영되는 값을 확인해야 합니다.

### `real_policy_inference.py`
- 학습된 체크포인트(`.pt`)를 실물 입력에 연결하기 위한 추론 코드입니다.
- 역할:
  - 실물 센서값(deg, deg/s)을 policy 입력 형식으로 변환
  - history length를 맞춰 observation 구성
  - 학습된 actor policy 로드
  - policy 출력을 목표 각도 명령으로 변환
  - CAN 통신 루프에 연결 가능한 구조 제공
- 즉, 시뮬레이터에서 학습한 정책을 실물 하드웨어 쪽으로 넘기는 어댑터 역할을 합니다.

### `sysid_collect.py`
- 실물 실린더 모터에 알려진 open-loop 여기 신호(step 또는 chirp)를 직접
  명령으로 보내고, 응답(실린더 각/속도/토크, 진자 각)을 CSV로 로깅하는
  system-identification 데이터 수집 스크립트입니다.
- RL 정책과는 무관하며, `commission_motor.py`의 MIT 명령 함수와
  `robot_state_reader.py`의 `RobotStateReader`를 그대로 재사용합니다.
- 토크(`motor_torque_nm`)를 반드시 CSV에 기록합니다 — `sysid_fit.py`의
  회귀 기반 질량/관성 추정에 실측 토크가 필요하기 때문입니다.

### `sysid_fit.py`
- `sysid_collect.py`가 만든 로그를 읽어 `assets/inverse.xml`의 물리
  파라미터를 실물에 맞게 추정하는 오프라인(하드웨어 불필요) 스크립트입니다.
- 두 단계로 나뉩니다.
  - Stage A: `mujoco_sysid`의 관절 토크 회귀자로 실린더/진자 바디의
    질량·무게중심·관성을 선형 최소자승(nominal 값 주변 ridge 정규화)으로
    추정합니다.
  - Stage B: Stage A 결과를 기반으로, 로그의 목표각 시퀀스를 그대로
    재생하는 시뮬레이션과 실측 궤적의 차이를 `mujoco.minimize.least_squares`
    로 최소화해 관절 damping/armature/frictionloss와 액추에이터 kp를
    추정합니다.
- 결과는 `assets/inverse.xml`에 자동 반영하지 않고 JSON 리포트로만
  출력합니다 — 적용 여부는 사람이 직접 검토해서 결정합니다.
- `--selftest` 플래그로 하드웨어 없이 파이프라인 자체를 검증할 수
  있습니다 (nominal 모델로 합성 데이터를 만들고 다시 피팅해 궤적 예측
  오차를 확인).

## 보조 폴더

### `assets/`
- `inverse.xml`과 raw export 파일들을 보관합니다.

### `references/`
- 참고용 코드, 외부 레퍼런스, 메모를 보관합니다.
- 이 폴더 내용은 기본적으로 태스크 실행 코드로 직접 import되지 않습니다.

### `successful_checkpoints/`
- 실험 중 성능이 좋았던 체크포인트를 따로 복사해 두는 폴더입니다.
- GitHub 정리나 하드웨어 전달용 후보 모델을 모아두는 용도로 사용합니다.

## 작업 흐름

1. CAD / xacro / URDF 수정
2. MJCF(`assets/inverse.xml`) 반영
3. `sysid_collect.py`로 실물 여기 신호(step/chirp)를 인가해 로그를 모으고,
   `sysid_fit.py`로 damping/armature/frictionloss·액추에이터 kp·질량/관성을
   추정해 `assets/inverse.xml`에 반영할지 검토
4. `inverse_env_cfg.py`에서 학습 설정 조정
5. 학습 후 성공한 `.pt`를 `successful_checkpoints/`에 정리
6. `real_policy_inference.py`로 실물 적용 파이프라인 연결
