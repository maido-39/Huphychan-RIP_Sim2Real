# Inverse Sim2Real Parameters

강화학습/시뮬레이션 값과 실기기 값을 맞춰 보기 위한 작업표입니다.

- `RL/Sim` 칸은 현재 코드에서 사용하는 값입니다.
- `Measured` 칸은 실험으로 측정해서 채웁니다.
- `Manufacturer` 칸은 모터/센서/부품 데이터시트에서 확인해서 채웁니다.

## Quick Summary

| Category               | RL/Sim                    |
| ---------------------- | ------------------------- |
| Physics step           | `0.005 s` / `200 Hz`      |
| Policy update          | `decimation=4` / `50 Hz`  |
| Command delay          | `0 ms`                    |
| Real policy loop       | `50 Hz`                   |
| Real state read        | `200 Hz`                  |
| Action scale           | `20 deg`                  |
| Target rate limit      | `1500 deg/s`              |
| Target accel limit     | `15000 deg/s^2`           |
| Real action multiplier | default `0.2`, test `1.0` |
| Pole COM randomization | `z +/- 3 mm`              |

## Highest Priority To Match

| Priority | Item                          | Why It Matters                                                           | How To Fill Measured                |
| -------: | ----------------------------- | ------------------------------------------------------------------------ | ----------------------------------- |
|        1 | Motor position response       | Swing timing depends on how fast `Revolute 3` follows target position.   | Step/sine command test on real motor |
|        2 | Motor torque/current limit    | Swing-up needs enough energy injection.                                  | Motor log or driver limit           |
|        3 | Pole free-swing damping       | Decides how quickly the pendulum loses energy.                           | Release pole and fit decay          |
|        4 | Pole mass / COM / inertia     | Changes swing-up difficulty and upright balance.                         | Weigh parts / CAD / pendulum test   |
|        5 | Encoder latency/noise         | Policy observes velocity; noisy velocity can cause action chatter.        | Static and motion encoder logs      |

## Timing

| Parameter             | RL/Sim                    | Measured                        | Manufacturer          |
| --------------------- | ------------------------- | --------                        | ------------          |
| MuJoCo timestep       | `0.005 s`                 | N/A                             | N/A                   |
| Physics frequency     | `200 Hz`                  | N/A                             | N/A                   |
| Policy decimation     | `4`                       | N/A                             | N/A                   |
| Policy frequency      | `50 Hz`                   | `50 Hz target, measured ___ Hz` | N/A                   |
| Command delay         | `0 physics steps = 0 ms`  | `___ ms`                        | CAN/motor dependent   |
| Real control loop     | `50 Hz`                   | `___ Hz`                        | N/A                   |
| Real state read loop  | `200 Hz`                  | `___ Hz`                        | encoder/CAN dependent |

## Action And Command

정책은 MuJoCo position actuator와 action target limiter를 통과한 응답을 기준으로 학습된다.
실기기 배포에서는 Real MIT `kp/kd`와 target 변환 코드가 학습 당시 시뮬레이션 응답을 재현해야 한다.
실기기가 이 응답을 안전하게 재현할 수 없으면, 시뮬레이션 actuator 모델을 실기기 응답에 맞게 수정한 뒤 재학습 또는 파인튜닝한다.

| Parameter                | RL/Sim                                  | Measured                                                         | Manufacturer |
|---                       |---                                      |---                                                               |---           |
| Policy action range      | `[-1, 1]`                               | N/A                                                              | N/A          |
| Action scale             | `20 deg`                                | same in real policy inference                                    | N/A          |
| Target rate limit        | `1500 deg/s`                            | same in real policy inference                                    | N/A          |
| Target accel limit       | `15000 deg/s^2`                         | same in real policy inference                                    | N/A          |
| Real action multiplier   | default `0.2`, commonly tested as `1.0` | set by `run_policy_motor.py --action-scale-multiplier`           | N/A          |
| Real motor tracking gain | MIT `kp=16.5`, `kd=1.0`                 | real motor response reference; measure step/sine response        | N/A          |
| Sim motor tracking gain  | actuator `kp=10.0`, `kv=0.45`           | tune to match real rise time, overshoot, settling, and phase lag | N/A          |
| Sim motor force range    | `[-10, 10] Nm`                          | compare with observed real torque/current limit                  | motor dependent |


## Actuated Axis: Revolute 3
Joint damping은 모터축의 마찰/감쇠, Joint armature는 모터축의 추가 회전관성이다. 둘 다 실물 모터 응답과 시뮬레이션 응답을 맞추기 위한 MuJoCo 물리 튜닝값이다.

| Parameter                  | RL/Sim                                                    | Measured | Manufacturer |
| -------------------------- | --------------------------------------------------------- | -------- | ------------ |
| Joint axis                 | `(0, 0, -1)`                                              | matched  |    N/A       |
| Joint damping              | `0.05`                                                    |          |              |
| Joint armature             | `0.0142`                                                  |          |    0.0142    |
| Rotor body mass            | `0.053175 kg`                                             | matched  |    N/A       |
| Rotor diagonal inertia     | `(4.07858e-05, 3.80533e-05, 8.48022e-06)`                 | matched  |    N/A       |

## Passive Pole Axis: Revolute 5
Joint damping
= 펜듈럼이 흔들릴 때 속도에 비례해서 에너지가 줄어드는 정도
= free-swing decay 보고 튜닝

Joint frictionloss
= 아주 느린 속도에서 걸리는 마찰/정지마찰 성향
= 작은 진폭에서 얼마나 빨리 멈추는지 보고 튜닝

Joint armature
= 펜듈럼 축에 추가로 걸리는 등가 회전관성
= free-swing period가 실물과 비슷한지 확인

| Parameter                  | RL/Sim                                                    | Measured                | Manufacturer |
| -------------------------- | --------------------------------------------------------- | --------                | ------------ |
| Joint axis                 | `(-1, 0, 0)`                                              | matched                 |    N/A       |
| Joint damping              | `0.0001`                                                  |                         |              |
| Joint frictionloss         | `0.00007`                                                 |                         |              |
| Joint armature             | `0.00001`                                                 |                         |              |
| Pole assembly mass         | `0.0318688 kg`                                            | matched                 |    N/A       |
| Pole COM                   | `(0.004277, -2.21893e-08, -0.037)`                        | CAD/PD-tuned, semi-matched | N/A       |
| Pole COM z randomization   | `+/- 0.003 m`                                             | range to validate       |    N/A       |
| Pole diagonal inertia      | `(4.61015e-05, 4.60103e-05, 3.26571e-07)`                 | CAD/PD-tuned, semi-matched | N/A       |
| Damping randomization      | `0.5x ~ 2.0x`                                             | training reset randomization | N/A       |
| Frictionloss randomization | `0.5x ~ 2.0x`                                             | training reset randomization | N/A       |
| Armature randomization     | `0.5x ~ 2.0x`                                             | training reset randomization | N/A       |

## Policy Observation Contract

이 섹션은 측정 대상이 아니라 정책 입력 형식입니다. 실기기 입력 변환 코드가 이 형식과 맞아야 합니다.

| Observation                     | Definition                     |
| ------------------------------- | ------------------------------ |
| Cylinder angle normalization    | `/ (3*pi)`                     |
| Cylinder angle periodic obs     | `cos(theta), sin(theta)`       |
| Pole angle periodic obs         | `cos(alpha), sin(alpha)`       |
| Joint velocity normalization    | `/ 30.0`                       |
| Last action observation         | `enabled`                      |
| Observation history length      | `2`                            |
| Real pole velocity estimator    | `PLL kp=80.0, ki=1200.0`       |

## Reward Contract

이 섹션도 제조사 스펙과 비교하는 항목이 아니라, 현재 학습 objective의 정의입니다.

| Reward-related value         | Definition                |
| ---------------------------- | ------------------------- |
| Upright target               | `180 deg`                 |
| Balance hold band            | `180 deg +/- 30 deg`      |
| Hold reward minimum time     | `0.2 s`                   |
| Hold reward ramp time        | `0.8 s`                   |
| Upper swing region           | `110 deg ~ 250 deg`       |
| Swing speed normalization    | `220 deg/s`               |
| Balance-start probability    | `0.1`                     |

## Code Pointers

| Area                                  | File                                                |
| ------------------------------------- | --------------------------------------------------- |
| RL environment, rewards, terminations | `src/mjlab/tasks/inverse/inverse_env_cfg.py`        |
| MuJoCo body/joint/actuator parameters | `src/mjlab/tasks/inverse/assets/inverse.xml`        |
| Real motor policy runner              | `src/mjlab/tasks/inverse/run_policy_motor.py`       |
| Real policy observation construction  | `src/mjlab/tasks/inverse/real_policy_inference.py`  |
| Modified real policy inference copy   | `src/mjlab/tasks/inverse/real_policy_inference_modified.py` |

## Notes

- 이 문서에는 실측/제조사 스펙과 비교할 수 있는 동역학 및 신호 관련 값만 남깁니다.
- 회전량 termination, 속도 termination, chatter termination 같은 하드웨어 안전 제한은 비교표에서 제외했습니다.
- 먼저 맞출 값은 `Revolute 3` step response와 `Revolute 5` free-swing decay입니다.
- 모터 제조사 스펙만으로 부족하면 실제 로그를 기준으로 시뮬레이션 값을 맞추는 편이 좋습니다.
