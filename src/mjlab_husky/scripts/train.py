"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

# MuJoCo selects its GL backend on first import; force headless EGL *before* the
# mjlab/mujoco imports below so `--video`'s offscreen renderer has a valid
# context (launch_training sets this too late). Respects an explicit MUJOCO_GL.
os.environ.setdefault("MUJOCO_GL", "egl")

import tyro  # noqa: E402
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv  # noqa: E402
from mjlab_husky.envs import G1SkaterManagerBasedRlEnvCfg, G1SkaterManagerBasedRlEnv
from mjlab_husky.rl import RslRlVecEnvWrapper
from mjlab_husky.rl import RslRlAMPOnPolicyRunnerCfg
from mjlab_husky.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder
# os.environ["WANDB_MODE"] = "offline"


@dataclass(frozen=True)
class TrainConfig:
  env: G1SkaterManagerBasedRlEnvCfg
  agent: RslRlAMPOnPolicyRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_every_iters: int = 500
  """Record a video clip every N training iterations (the "Learning iteration N"
  shown in the logs). The first clip starts at iteration 0."""
  video_seconds: float = 6.0
  """Length of each clip, in seconds of footage (at the control rate)."""
  video_height: int = 720
  video_width: int = 1280
  """Resolution of the recorded clips (the renderer default is a tiny 240x320)."""
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlAMPOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  # Populate the task registry in worker processes. Under multi-GPU, torchrunx
  # ships this function to workers via cloudpickle (by value), so this module's
  # top-level imports -- which normally trigger task registration through
  # mjlab_husky.tasks/__init__.py's import_packages() -- do not run in the
  # worker. Without this, load_runner_cls() below raises KeyError. Import envs
  # first (mirroring this module's top-level import order) so the task configs
  # can resolve G1SkaterManagerBasedRlEnvCfg without a circular import.
  import mjlab_husky.envs  # noqa: F401
  import mjlab_husky.tasks  # noqa: F401

  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None


  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  # Only rank 0 records video, so only rank 0 builds the offscreen GL renderer.
  # Constructing it on the other workers (which never record) segfaults under
  # multi-GPU torchrunx, which then stalls rank 0 at the gradient all-reduce.
  record_video = cfg.video and rank == 0
  if record_video:
    # Bump the offscreen render resolution above the tiny ViewerConfig default.
    cfg.env.viewer.height = cfg.video_height
    cfg.env.viewer.width = cfg.video_width

  env = G1SkaterManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if record_video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if record_video:
    # The recorder counts raw env steps and one frame == one env step. Convert the
    # user-facing iterations/seconds into those units: the rollout takes
    # num_steps_per_env env-steps per training iteration, and the control rate is
    # 1/step_dt Hz. (env is still the raw env here, so it exposes step_dt.)
    fps = round(1.0 / env.step_dt)
    interval_steps = max(1, cfg.video_every_iters * cfg.agent.num_steps_per_env)
    length_frames = max(1, round(cfg.video_seconds * fps))
    video_dir = Path(log_dir) / "videos" / "train"
    env = VideoRecorder(
      env,
      video_folder=video_dir,
      step_trigger=lambda step: step % interval_steps == 0,
      video_length=length_frames,
      disable_logger=True,
    )
    print(
      f"[INFO] Recording a {cfg.video_seconds:g}s clip every "
      f"{cfg.video_every_iters} training iterations -> {video_dir}"
    )

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)

  runner_kwargs = {}


  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Video on multi-GPU works -- only rank 0 renders (see run_train), and that
  # coexists with NCCL. But rendering makes rank 0 momentarily slower than the
  # other ranks, which wait for it at the gradient all-reduce; a very frequent
  # cadence (or heavy contention for another rank's GPU) can stretch that wait
  # past the NCCL timeout. Keep --video-every-iters high (the default is) and
  # fall back to one GPU if a run ever stalls.
  if num_gpus > 1 and args.video:
    print(
      "[INFO] Recording video with multiple GPUs: only rank 0 renders, which adds "
      "a small sync cost there. Keep --video-every-iters high (default is fine); a "
      "very frequent cadence can stall the gradient all-reduce -- use --gpu-ids 0 "
      "if a run hangs.",
      flush=True,
    )

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
