"""RL configuration for AgiBot X2-Ultra skater task."""

from mjlab.rl import (
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)

from mjlab_husky.rl import (
  RslRlAMPOnPolicyRunnerCfg,
)


def agibot_x2_skater_ppo_runner_cfg() -> RslRlAMPOnPolicyRunnerCfg:
  """Create RL runner configuration for AgiBot X2-Ultra skater task."""
  return RslRlAMPOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMP_PPO",
    ),
    experiment_name="x2_skater",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50_000,
    # AMP: the X2 push reference is produced by retargeting the G1 push clips
    # (scripts/retarget_push_g1_to_x2.py -> dataset/skate_push_x2). AMP obs are
    # all 31 X2 joint positions. Set amp_enabled=False to train the push phase on
    # task rewards only (no clips required).
    amp_enabled=True,
    amp_num_obs=31,
    amp_motion_files="dataset/skate_push_x2",
    amp_obs_slices=((7, 38),),  # all 31 X2 joints (base 7 .. 38)
  )
