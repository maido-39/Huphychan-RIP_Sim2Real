#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Inverse-Balance"
DEFAULT_LOG_DIR = Path("src/mjlab/tasks/inverse/play_tracking_logs")


@dataclass(frozen=True)
class LogPlayTrackingConfig:
  checkpoint_file: str
  duration_s: float = 8.0
  device: str | None = None
  with_disturbance: bool = False
  fixed_start: bool = True
  no_terminations: bool = False
  log_dir: str = str(DEFAULT_LOG_DIR)
  print_every: int = 10


CSV_HEADER = [
  "time_s",
  "policy_step",
  "substep",
  "physics_step",
  "raw_action",
  "processed_action_rad",
  "processed_action_deg",
  "joint_pos_target_rad",
  "joint_pos_target_deg",
  "joint_pos_target_vel_rad_s",
  "joint_pos_target_vel_deg_s",
  "actuator_force_pre_nm",
  "actuator_force_post_nm",
  "qfrc_actuator_pre_nm",
  "qfrc_actuator_post_nm",
  "cylinder_q_pre_rad",
  "cylinder_q_pre_deg",
  "cylinder_q_post_rad",
  "cylinder_q_post_deg",
  "cylinder_qd_pre_rad_s",
  "cylinder_qd_pre_deg_s",
  "cylinder_qd_post_rad_s",
  "cylinder_qd_post_deg_s",
  "tracking_error_pre_rad",
  "tracking_error_pre_deg",
  "tracking_error_post_rad",
  "tracking_error_post_deg",
  "pole_q_pre_rad",
  "pole_q_pre_deg",
  "pole_q_post_rad",
  "pole_q_post_deg",
  "pole_qd_pre_rad_s",
  "pole_qd_pre_deg_s",
  "pole_qd_post_rad_s",
  "pole_qd_post_deg_s",
  "pole_lift_pre_deg",
  "pole_lift_post_deg",
  "reward",
  "done",
]


def _to_float(x: Any) -> float:
  if isinstance(x, torch.Tensor):
    return float(x.detach().cpu().flatten()[0].item())
  return float(x)


def _deg(rad: float) -> float:
  return math.degrees(rad)


def _pole_lift_deg(pole_q_rad: float) -> float:
  # 0 rad = down, pi rad = upright. Fold to 0~180 deg lift.
  return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(pole_q_rad)))))


def _reset_vec_env(env: RslRlVecEnvWrapper):
  out = env.reset()
  if isinstance(out, tuple):
    return out[0]
  return out


def _build_env(cfg: LogPlayTrackingConfig, device: str) -> ManagerBasedRlEnv:
  env_cfg = load_env_cfg(TASK_ID, play=not cfg.with_disturbance)
  env_cfg.scene.num_envs = 1
  env_cfg.episode_length_s = max(float(cfg.duration_s), env_cfg.episode_length_s)
  env_cfg.observations["actor"].enable_corruption = False

  if cfg.no_terminations:
    env_cfg.terminations = {}

  if cfg.fixed_start and "reset_curriculum" in env_cfg.events:
    params = env_cfg.events["reset_curriculum"].params
    params["balance_start_probability"] = 0.0
    params["cylinder_position_range"] = (0.0, 0.0)
    params["cylinder_velocity_range"] = (0.0, 0.0)
    params["swingup_pole_position_range"] = (0.0, 0.0)
    params["swingup_pole_velocity_range"] = (0.0, 0.0)

  return ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)


def _get_action_term(unwrapped: ManagerBasedRlEnv):
  action_manager = unwrapped.action_manager
  terms = getattr(action_manager, "_terms", None)
  if isinstance(terms, dict):
    if "position" in terms:
      return terms["position"]
    if len(terms) == 1:
      return next(iter(terms.values()))
  raise RuntimeError("Could not find inverse position action term.")


def _get_joint_snapshot(env: ManagerBasedRlEnv) -> dict[str, float]:
  asset = env.scene["inverse"]
  q = asset.data.joint_pos
  qd = asset.data.joint_vel
  q_target = asset.data.joint_pos_target
  actuator_force = asset.data.actuator_force
  qfrc_actuator = asset.data.qfrc_actuator
  return {
    "cylinder_q_rad": _to_float(q[:, 0]),
    "pole_q_rad": _to_float(q[:, 1]),
    "cylinder_qd_rad_s": _to_float(qd[:, 0]),
    "pole_qd_rad_s": _to_float(qd[:, 1]),
    "joint_pos_target_rad": _to_float(q_target[:, 0]),
    "actuator_force_nm": _to_float(actuator_force[:, 0]),
    "qfrc_actuator_nm": _to_float(qfrc_actuator[:, 0]),
  }


def _get_processed_action(action_term: Any) -> torch.Tensor:
  processed = getattr(action_term, "processed_action", None)
  if processed is not None:
    return processed
  processed = getattr(action_term, "_processed_actions", None)
  if processed is not None:
    return processed
  raise RuntimeError("Could not read processed action from action term.")


def _finish_policy_step(raw_env: ManagerBasedRlEnv):
  raw_env.episode_length_buf += 1
  raw_env.common_step_counter += 1

  raw_env.reset_buf = raw_env.termination_manager.compute()
  raw_env.reset_terminated = raw_env.termination_manager.terminated
  raw_env.reset_time_outs = raw_env.termination_manager.time_outs

  raw_env.reward_buf = raw_env.reward_manager.compute(dt=raw_env.step_dt)
  raw_env.metrics_manager.compute()

  reset_env_ids = raw_env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
  if raw_env.cfg.auto_reset and len(reset_env_ids) > 0:
    raw_env.recorder_manager.record_pre_reset(reset_env_ids)
    raw_env._reset_idx(reset_env_ids)
    raw_env.scene.write_data_to_sim()

  raw_env.sim.forward()
  raw_env.command_manager.compute(dt=raw_env.step_dt)

  if "step" in raw_env.event_manager.available_modes:
    raw_env.event_manager.apply(mode="step", dt=raw_env.step_dt)
  if "interval" in raw_env.event_manager.available_modes:
    raw_env.event_manager.apply(mode="interval", dt=raw_env.step_dt)

  raw_env.sim.sense()
  raw_env.obs_buf = raw_env.observation_manager.compute(update_history=True)

  if raw_env.cfg.auto_reset and len(reset_env_ids) > 0:
    raw_env.recorder_manager.record_post_reset(reset_env_ids)
  elif len(reset_env_ids) > 0:
    raw_env._manual_reset_pending[reset_env_ids] = True

  raw_env.recorder_manager.record_post_step()

  obs = TensorDict(raw_env.obs_buf, batch_size=[raw_env.num_envs])
  done = raw_env.reset_terminated | raw_env.reset_time_outs
  return obs, raw_env.reward_buf, done


def run(cfg: LogPlayTrackingConfig) -> Path:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint_path = Path(cfg.checkpoint_file)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  raw_env = _build_env(cfg, device)
  agent_cfg = load_rl_cfg(TASK_ID)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  action_term = _get_action_term(env.unwrapped)

  obs = _reset_vec_env(env)

  log_dir = Path(cfg.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  disturb_tag = "disturbance_on" if cfg.with_disturbance else "disturbance_off"
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_path = log_dir / f"play_tracking_{disturb_tag}_{stamp}.csv"

  policy_dt = float(env.unwrapped.step_dt)
  physics_dt = float(env.unwrapped.physics_dt)
  decimation = int(env.unwrapped.cfg.decimation)
  max_policy_steps = int(round(float(cfg.duration_s) / policy_dt))

  print(f"[INFO] CSV: {csv_path}")
  print(
    f"[INFO] duration={cfg.duration_s}s policy_dt={policy_dt:.4f}s "
    f"physics_dt={physics_dt:.4f}s decimation={decimation} "
    f"policy_steps={max_policy_steps} disturbance={cfg.with_disturbance}"
  )

  with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)

    physics_step = 0
    last_policy_target_rad: float | None = None
    for policy_step in range(max_policy_steps):
      with torch.inference_mode():
        action = policy(obs)

      if env.clip_actions is not None:
        action = torch.clamp(action, -env.clip_actions, env.clip_actions)

      raw_env = env.unwrapped
      raw_env.extras["log"] = dict()
      raw_env.action_manager.process_action(action.to(raw_env.device))

      raw_action = _to_float(action_term.raw_action[:, 0])
      processed_action_rad = _to_float(_get_processed_action(action_term)[:, 0])
      if last_policy_target_rad is None:
        joint_pos_target_vel_rad_s = 0.0
      else:
        joint_pos_target_vel_rad_s = (processed_action_rad - last_policy_target_rad) / policy_dt
      last_policy_target_rad = processed_action_rad

      substep_rows = []
      for substep in range(decimation):
        pre = _get_joint_snapshot(raw_env)

        raw_env._sim_step_counter += 1
        raw_env.action_manager.apply_action()
        raw_env.scene.write_data_to_sim()
        raw_env.sim.step()
        raw_env.scene.update(dt=raw_env.physics_dt)
        raw_env.metrics_manager.compute_substep()

        post = _get_joint_snapshot(raw_env)
        joint_pos_target_rad = post["joint_pos_target_rad"]
        tracking_error_pre_rad = joint_pos_target_rad - pre["cylinder_q_rad"]
        tracking_error_post_rad = joint_pos_target_rad - post["cylinder_q_rad"]
        time_s = physics_step * physics_dt

        substep_rows.append(
          [
            f"{time_s:.6f}",
            policy_step,
            substep,
            physics_step,
            f"{raw_action:.8f}",
            f"{processed_action_rad:.8f}",
            f"{_deg(processed_action_rad):.6f}",
            f"{joint_pos_target_rad:.8f}",
            f"{_deg(joint_pos_target_rad):.6f}",
            f"{joint_pos_target_vel_rad_s:.8f}",
            f"{_deg(joint_pos_target_vel_rad_s):.6f}",
            f"{pre['actuator_force_nm']:.8f}",
            f"{post['actuator_force_nm']:.8f}",
            f"{pre['qfrc_actuator_nm']:.8f}",
            f"{post['qfrc_actuator_nm']:.8f}",
            f"{pre['cylinder_q_rad']:.8f}",
            f"{_deg(pre['cylinder_q_rad']):.6f}",
            f"{post['cylinder_q_rad']:.8f}",
            f"{_deg(post['cylinder_q_rad']):.6f}",
            f"{pre['cylinder_qd_rad_s']:.8f}",
            f"{_deg(pre['cylinder_qd_rad_s']):.6f}",
            f"{post['cylinder_qd_rad_s']:.8f}",
            f"{_deg(post['cylinder_qd_rad_s']):.6f}",
            f"{tracking_error_pre_rad:.8f}",
            f"{_deg(tracking_error_pre_rad):.6f}",
            f"{tracking_error_post_rad:.8f}",
            f"{_deg(tracking_error_post_rad):.6f}",
            f"{pre['pole_q_rad']:.8f}",
            f"{_deg(pre['pole_q_rad']):.6f}",
            f"{post['pole_q_rad']:.8f}",
            f"{_deg(post['pole_q_rad']):.6f}",
            f"{pre['pole_qd_rad_s']:.8f}",
            f"{_deg(pre['pole_qd_rad_s']):.6f}",
            f"{post['pole_qd_rad_s']:.8f}",
            f"{_deg(post['pole_qd_rad_s']):.6f}",
            f"{_pole_lift_deg(pre['pole_q_rad']):.6f}",
            f"{_pole_lift_deg(post['pole_q_rad']):.6f}",
            "0.00000000",
            0,
          ]
        )
        physics_step += 1

      obs, reward, done = _finish_policy_step(raw_env)
      reward_value = f"{_to_float(reward):.8f}"
      done_value = int(bool(done.flatten()[0].item()))
      for row in substep_rows:
        row[-2] = reward_value
        row[-1] = done_value
        writer.writerow(row)

      last_row = substep_rows[-1]
      if policy_step % max(1, int(cfg.print_every)) == 0:
        print(
          f"[{policy_step:04d}] t={float(last_row[0]):.3f}s "
          f"raw={raw_action:+.3f} "
          f"target={float(last_row[8]):+.2f}deg "
          f"q={float(last_row[18]):+.2f}deg "
          f"err={float(last_row[26]):+.2f}deg "
          f"tau={float(last_row[14]):+.3f}Nm "
          f"target_vel={float(last_row[10]):+.1f}deg/s"
        )

      if bool(done.any()):
        print(f"[INFO] done at policy_step={policy_step}, resetting env.")
        obs = _reset_vec_env(env)

  env.close()
  print(f"[INFO] CSV saved: {csv_path}")
  return csv_path


if __name__ == "__main__":
  run(tyro.cli(LogPlayTrackingConfig))
