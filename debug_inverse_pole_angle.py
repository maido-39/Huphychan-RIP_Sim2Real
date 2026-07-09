# /home/aril/mjlab/debug_inverse_pole_angle.py
from copy import deepcopy
import math

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.inverse.inverse_env_cfg import _CYLINDER_CFG, _JOINTS_CFG, _POLE_CFG


TASK_ID = "Mjlab-Inverse-Reference"


def main():
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
  env.reset()

  joints_cfg = deepcopy(_JOINTS_CFG)
  cylinder_cfg = deepcopy(_CYLINDER_CFG)
  pole_cfg = deepcopy(_POLE_CFG)
  joints_cfg.resolve(env.scene)
  cylinder_cfg.resolve(env.scene)
  pole_cfg.resolve(env.scene)

  asset = env.scene[pole_cfg.name]

  print("cylinder joint names:", cylinder_cfg.joint_names)
  print("cylinder joint ids:", cylinder_cfg.joint_ids)
  print("pole joint names:", pole_cfg.joint_names)
  print("pole joint ids:", pole_cfg.joint_ids)
  print("all joint names:", joints_cfg.joint_names)
  print("all joint ids:", joints_cfg.joint_ids)
  print()

  for pole_q in [0.0, math.pi, -math.pi, math.pi / 2, -math.pi / 2]:
    joint_pos = torch.tensor([[0.0, pole_q]], device=env.device)
    joint_vel = torch.zeros_like(joint_pos)

    asset.write_joint_state_to_sim(
      joint_pos,
      joint_vel,
      joint_ids=joints_cfg.joint_ids,
      env_ids=torch.tensor([0], device=env.device),
    )
    env.sim.forward()

    cylinder_angle = asset.data.joint_pos[:, cylinder_cfg.joint_ids].squeeze(-1)
    pole_angle = asset.data.joint_pos[:, pole_cfg.joint_ids].squeeze(-1)

    print(f"set pole_q = {pole_q:+.6f} rad / {math.degrees(pole_q):+7.2f} deg")
    print("  measured cylinder_angle rad:", cylinder_angle.detach().cpu().numpy())
    print("  measured pole_angle rad:", pole_angle.detach().cpu().numpy())
    print("  measured pole_angle deg:", torch.rad2deg(pole_angle).detach().cpu().numpy())

    # Reward convention check.
    alpha_minus_cos = 0.5 * (1.0 - torch.cos(pole_angle))
    alpha_plus_cos = 0.5 * (1.0 + torch.cos(pole_angle))
    print("  reward if upright=pi :", alpha_minus_cos.detach().cpu().numpy())
    print("  reward if upright=0  :", alpha_plus_cos.detach().cpu().numpy())
    print()

  env.close()


if __name__ == "__main__":
  main()
