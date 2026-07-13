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

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


DEFAULT_LOG_DIR = Path("src/mjlab/tasks/inverse/play_tracking_logs")


@dataclass(frozen=True)
class LogSingleJointTrackingConfig:
  task_id: str
  checkpoint_file: str
  entity_name: str = "robot"
  action_term_name: str = "joint_pos"
  joint_name: str | None = None
  duration_s: float = 8.0
  device: str | None = None
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
  "reward",
  "done",
]


def _to_float(x: Any) -> float:
  if isinstance(x, torch.Tensor):
    return float(x.detach().cpu().flatten()[0].item())
  return float(x)


def _deg(rad: float) -> float:
  return math.degrees(rad)


def _reset_vec_env(env: RslRlVecEnvWrapper):
  out = env.reset()
  if isinstance(out, tuple):
    return out[0]
  return out


def _get_action_term(unwrapped, name: str):
  action_manager = unwrapped.action_manager
  if hasattr(action_manager, "get_term"):
    try:
      return action_manager.get_term(name)
    except Exception:
      pass
  terms = getattr(action_manager, "_terms", None)
  if isinstance(terms, dict) and name in terms:
    return terms[name]
  raise RuntimeError(f"Could not find action term: {name}")


def _get_processed_action(action_term: Any) -> torch.Tensor:
  processed = getattr(action_term, "processed_action", None)
  if processed is not None:
    return processed
  processed = getattr(action_term, "_processed_actions", None)
  if processed is not None:
    return processed
  raise RuntimeError("Could not read processed action from action term.")


def _choose_joint(action_term: Any, requested_joint_name: str | None) -> tuple[int, int, str]:
  target_names = list(getattr(action_term, "target_names"))
  target_ids = getattr(action_term, "target_ids")
  if requested_joint_name is None:
    action_idx = 0
  else:
    matches = [i for i, name in enumerate(target_names) if name == requested_joint_name]
    if not matches:
      raise ValueError(
        f"Unknown joint_name={requested_joint_name!r}. Available examples: {target_names[:12]}"
      )
    action_idx = int(matches[0])
  joint_id = int(target_ids[action_idx].detach().cpu().item())
  return action_idx, joint_id, target_names[action_idx]


def _snapshot(asset: Any, joint_id: int) -> dict[str, float]:
  q = asset.data.joint_pos[:, joint_id]
  qd = asset.data.joint_vel[:, joint_id]
  q_target = asset.data.joint_pos_target[:, joint_id]
  qfrc = asset.data.qfrc_actuator[:, joint_id]
  return {
    "q_rad": _to_float(q),
    "qd_rad_s": _to_float(qd),
    "q_target_rad": _to_float(q_target),
    "qfrc_nm": _to_float(qfrc),
  }


def run(cfg: LogSingleJointTrackingConfig) -> Path:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint_path = Path(cfg.checkpoint_file)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  raw_env = load_env_cfg(cfg.task_id, play=True)
  raw_env.scene.num_envs = 1
  raw_env.episode_length_s = max(float(cfg.duration_s), raw_env.episode_length_s)
  if "actor" in raw_env.observations:
    raw_env.observations["actor"].enable_corruption = False

  from mjlab.envs import ManagerBasedRlEnv

  manager_env = ManagerBasedRlEnv(cfg=raw_env, device=device, render_mode=None)
  agent_cfg = load_rl_cfg(cfg.task_id)
  env = RslRlVecEnvWrapper(manager_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  action_term = _get_action_term(env.unwrapped, cfg.action_term_name)
  action_idx, joint_id, joint_name = _choose_joint(action_term, cfg.joint_name)
  asset = env.unwrapped.scene[cfg.entity_name]

  obs = _reset_vec_env(env)

  log_dir = Path(cfg.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  safe_task = cfg.task_id.replace("/", "_")
  safe_joint = joint_name.replace("/", "_").replace(" ", "_")
  csv_path = log_dir / f"single_joint_{safe_task}_{safe_joint}_{stamp}.csv"

  dt = float(env.unwrapped.step_dt)
  max_steps = int(round(float(cfg.duration_s) / dt))
  last_target_rad: float | None = None

  print(f"[INFO] CSV: {csv_path}")
  print(f"[INFO] task={cfg.task_id} joint={joint_name} joint_id={joint_id} action_idx={action_idx}")
  print(f"[INFO] duration={cfg.duration_s}s dt={dt:.4f}s steps={max_steps}")

  with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)

    for step in range(max_steps):
      pre = _snapshot(asset, joint_id)
      with torch.inference_mode():
        action = policy(obs)
      if env.clip_actions is not None:
        action = torch.clamp(action, -env.clip_actions, env.clip_actions)

      obs, reward, done, _ = env.step(action)
      post = _snapshot(asset, joint_id)

      processed = _get_processed_action(action_term)
      raw_action = _to_float(action[:, action_idx])
      processed_action_rad = _to_float(processed[:, action_idx])
      target_rad = post["q_target_rad"]
      if last_target_rad is None:
        target_vel_rad_s = 0.0
      else:
        target_vel_rad_s = (target_rad - last_target_rad) / dt
      last_target_rad = target_rad

      err_pre = target_rad - pre["q_rad"]
      err_post = target_rad - post["q_rad"]
      row = [
        f"{step * dt:.6f}",
        step,
        0,
        step,
        f"{raw_action:.8f}",
        f"{processed_action_rad:.8f}",
        f"{_deg(processed_action_rad):.6f}",
        f"{target_rad:.8f}",
        f"{_deg(target_rad):.6f}",
        f"{target_vel_rad_s:.8f}",
        f"{_deg(target_vel_rad_s):.6f}",
        f"{pre['qfrc_nm']:.8f}",
        f"{post['qfrc_nm']:.8f}",
        f"{pre['qfrc_nm']:.8f}",
        f"{post['qfrc_nm']:.8f}",
        f"{pre['q_rad']:.8f}",
        f"{_deg(pre['q_rad']):.6f}",
        f"{post['q_rad']:.8f}",
        f"{_deg(post['q_rad']):.6f}",
        f"{pre['qd_rad_s']:.8f}",
        f"{_deg(pre['qd_rad_s']):.6f}",
        f"{post['qd_rad_s']:.8f}",
        f"{_deg(post['qd_rad_s']):.6f}",
        f"{err_pre:.8f}",
        f"{_deg(err_pre):.6f}",
        f"{err_post:.8f}",
        f"{_deg(err_post):.6f}",
        f"{_to_float(reward):.8f}",
        int(bool(done.flatten()[0].item())),
      ]
      writer.writerow(row)

      if step % max(1, int(cfg.print_every)) == 0:
        print(
          f"[{step:04d}] t={step * dt:.3f}s raw={raw_action:+.3f} "
          f"target={_deg(target_rad):+.2f}deg q={_deg(post['q_rad']):+.2f}deg "
          f"err={_deg(err_post):+.2f}deg tau={post['qfrc_nm']:+.3f}Nm "
          f"target_vel={_deg(target_vel_rad_s):+.1f}deg/s"
        )

      if bool(done.any()):
        obs = _reset_vec_env(env)
        last_target_rad = None

  env.close()
  print(f"[INFO] CSV saved: {csv_path}")
  return csv_path


if __name__ == "__main__":
  run(tyro.cli(LogSingleJointTrackingConfig))
