"""Export ONNX policy files for a set of .pt checkpoints (standalone, no wandb).

Mirrors SkaterOnPolicyRunner.save's ONNX path exactly: exports the actor +
obs-normalizer via _OnnxPolicyExporter and attaches the deploy metadata (joint
names, PD gains, default joint pose, action scale, observation names) read from the
env. One env build is reused for every checkpoint (the metadata is identical across
the lineage — same task cfg — only the policy weights differ).

The written .onnx sits next to each .pt with a matching name, e.g.
  checkpoints_labeled/x2_iter313047_DR-length-width-thickness.pt
  -> checkpoints_labeled/x2_iter313047_DR-length-width-thickness.onnx

Usage (from repo root):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python scripts/export_checkpoints_onnx.py \
      checkpoints_labeled/*.pt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoints", nargs="+", help=".pt checkpoints to export")
  ap.add_argument("--task", default="Mjlab-Skater-Flat-Agibot-X2")
  args = ap.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  import torch

  import mjlab.tasks  # noqa: F401
  import mjlab_husky.tasks  # noqa: F401
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
  from mjlab_husky.rl import RslRlVecEnvWrapper
  from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab_husky.tasks.skater.rl.exporter import (
    attach_onnx_metadata,
    export_skater_policy_as_onnx,
  )

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.eval_mode = False
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  raw = G1SkaterManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=device)

  for ck in args.checkpoints:
    ckpt = Path(ck).resolve()
    if not ckpt.exists():
      print(f"SKIP (missing): {ckpt}")
      continue
    runner.load(str(ckpt), map_location=device)
    policy = runner.alg.policy
    normalizer = (
      policy.actor_obs_normalizer
      if getattr(policy, "actor_obs_normalization", False)
      else None
    )
    out_dir = str(ckpt.parent) + os.sep
    fname = ckpt.stem + ".onnx"
    export_skater_policy_as_onnx(policy, path=out_dir, normalizer=normalizer, filename=fname)
    # run_path is just an identifier embedded in metadata; use the checkpoint name.
    attach_onnx_metadata(env.unwrapped, run_path=ckpt.stem, path=out_dir, filename=fname)
    print(f"exported {out_dir}{fname}")

  env.close()


if __name__ == "__main__":
  main()
