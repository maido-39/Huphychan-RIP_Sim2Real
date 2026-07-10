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
| Command delay          | `20 ms`                   |
| Real policy loop       | `50 Hz`                   |
| Real state read        | `200 Hz`                  |
| Action scale           | `0.5*pi rad`              |
| Real action multiplier | `0.2`                     |

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
| Command delay         | `4 physics steps = 20 ms` | `___ ms`                        | CAN/motor dependent   |
| Real control loop     | `50 Hz`                   | `___ Hz`                        | N/A                   |
| Real state read loop  | `200 Hz`                  | `___ Hz`                        | encoder/CAN dependent |

## Action And Command

| Parameter                  | RL/Sim                                  | Measured | Manufacturer |
| -------------------------- | --------------------------------------- | -------- | ------------ |
| Real MIT `kp`              | `3.0`                                   |          |    N/A       |
| Real MIT `kd`              | `0.0`                                   |          |    N/A       |


## Actuated Axis: Revolute 3

| Parameter                  | RL/Sim                                                    | Measured | Manufacturer |
| -------------------------- | --------------------------------------------------------- | -------- | ------------ |
| Joint axis                 | `(0, 0, -1)`                                              | matched  |    N/A       |
| Joint damping              | `0.05`                                                    |          |              |
| Joint armature             | `0.0005`                                                  |          |              |
| Rotor body mass            | `0.053175 kg`                                             | matched  |    N/A       |
| Rotor diagonal inertia     | `(4.07858e-05, 3.80533e-05, 8.48022e-06)`                 | matched  |    N/A       |
| Sim actuator `kp`          | `2.2`                                                     |          |              |
| Sim actuator `kv`          | `0.6`                                                     |          |              |

## Passive Pole Axis: Revolute 5

| Parameter                  | RL/Sim                                                    | Measured                | Manufacturer |
| -------------------------- | --------------------------------------------------------- | --------                | ------------ |
| Joint axis                 | `(-1, 0, 0)`                                              | matched                 |    N/A       |
| Joint damping              | `0.0002`                                                  |                         |              |
| Joint frictionloss         | `0.0001`                                                  |                         |              |
| Joint armature             | `0.00001`                                                 |                         |              |
| Pole assembly mass         | `0.0318688 kg`                                            | matched                 |    N/A       |
| Pole COM                   | `(0.004277, -2.21893e-08, -0.0405385)`                    | CAD-based, semi-matched |    N/A       |
| Pole diagonal inertia      | `(4.75513e-05, 4.74611e-05, 3.25502e-07)`                 | CAD-based, semi-matched |    N/A       |

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

## Code Pointers

| Area                                  | File                                                |
| ------------------------------------- | --------------------------------------------------- |
| RL environment, rewards, terminations | `src/mjlab/tasks/inverse/inverse_env_cfg.py`        |
| MuJoCo body/joint/actuator parameters | `src/mjlab/tasks/inverse/assets/inverse.xml`        |
| Real motor policy runner              | `src/mjlab/tasks/inverse/run_policy_motor.py`       |
| Real policy observation construction  | `src/mjlab/tasks/inverse/real_policy_inference.py`  |

## Notes

- 이 문서에는 실측/제조사 스펙과 비교할 수 있는 동역학 및 신호 관련 값만 남깁니다.
- 회전량 termination, 속도 termination, chatter termination 같은 하드웨어 안전 제한은 비교표에서 제외했습니다.
- 먼저 맞출 값은 `Revolute 3` step response와 `Revolute 5` free-swing decay입니다.
- 모터 제조사 스펙만으로 부족하면 실제 로그를 기준으로 시뮬레이션 값을 맞추는 편이 좋습니다.
