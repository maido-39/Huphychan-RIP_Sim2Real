from mjlab.tasks.inverse.inverse_env_cfg import (
  inverse_balance_env_cfg,
  inverse_ppo_runner_cfg,
)
from mjlab.tasks.inverse.inverse_reference_env_cfg import (
  inverse_reference_env_cfg,
  inverse_reference_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Inverse-Balance",
  env_cfg=inverse_balance_env_cfg(),
  play_env_cfg=inverse_balance_env_cfg(play=True),
  rl_cfg=inverse_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Inverse-Reference",
  env_cfg=inverse_reference_env_cfg(),
  play_env_cfg=inverse_reference_env_cfg(play=True),
  rl_cfg=inverse_reference_ppo_runner_cfg(),
)
