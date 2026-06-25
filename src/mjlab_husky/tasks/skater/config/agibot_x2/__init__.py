from mjlab_husky.tasks.registry import register_mjlab_task
from mjlab_husky.tasks.skater.rl import SkaterOnPolicyRunner

from .env_cfgs import (
  agibot_x2_skater_env_cfg,
)
from .rl_cfg import agibot_x2_skater_ppo_runner_cfg


register_mjlab_task(
  task_id="Mjlab-Skater-Flat-Agibot-X2",
  env_cfg=agibot_x2_skater_env_cfg(),
  play_env_cfg=agibot_x2_skater_env_cfg(play=True),
  rl_cfg=agibot_x2_skater_ppo_runner_cfg(),
  runner_cls=SkaterOnPolicyRunner,
)
