"""Record a single-robot video of a trained skater policy (no overlay / no markers).

mjlab's offscreen renderer draws env 0 then overlays up to 32 other envs at the same
world origin, plus debug-visualizer markers — which looks like one exploded robot.
This renders env 0 with a private MuJoCo renderer (num_envs=1, no overlay, no markers),
so you see exactly one robot. Also dumps a couple of high-res stills (full + feet) for
close inspection.

Usage (from repo root):
    uv run python scripts/record_single_robot.py Mjlab-Skater-Flat-Agibot-X2 \
        --checkpoint logs/rsl_rl/x2_skater/<run>/model_12000.pt --out video/x2_single.mp4
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("task")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--steps", type=int, default=450)
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--out", default="video/single_robot.mp4")
  ap.add_argument("--device", default=None)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--width", type=int, default=720)
  ap.add_argument("--cam-distance", type=float, default=2.6)
  ap.add_argument("--fps", type=int, default=50)
  ap.add_argument("--still-dir", default=None, help="if set, dump full+feet stills here")
  args = ap.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  import numpy as np
  import imageio.v2 as imageio
  import mujoco
  import torch

  import mjlab.tasks  # noqa: F401
  import mjlab_husky.tasks  # noqa: F401
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
  from mjlab_husky.rl import RslRlVecEnvWrapper
  from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.eval_mode = False
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(args.task)

  raw = G1SkaterManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, map_location=device)
  policy = runner.get_inference_policy(device=device)

  m = raw.sim.mj_model
  m.vis.global_.offheight = max(args.height, 1200)
  m.vis.global_.offwidth = max(args.width, 1200)
  rdata = mujoco.MjData(m)
  renderer = mujoco.Renderer(m, height=args.height, width=args.width)
  torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.elevation, cam.azimuth, cam.distance = -10.0, 210.0, args.cam_distance

  obs, _ = env.reset()
  frames = []
  with torch.inference_mode():
    for _ in range(args.steps):
      obs, _, _, _ = env.step(policy(obs))
      rdata.qpos[:] = raw.sim.data.qpos[0].detach().cpu().numpy()
      rdata.qvel[:] = raw.sim.data.qvel[0].detach().cpu().numpy()
      mujoco.mj_forward(m, rdata)
      cam.lookat[:] = rdata.xpos[torso]
      renderer.update_scene(rdata, camera=cam)
      frames.append(renderer.render().copy())

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(str(out), frames, fps=args.fps)
  print(f"wrote {out}  ({len(frames)} frames, {len(frames) / args.fps:.1f}s)")

  if args.still_dir:
    import PIL.Image as Image
    sd = Path(args.still_dir); sd.mkdir(parents=True, exist_ok=True)
    hr = mujoco.Renderer(m, height=1200, width=1200)
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/left_ankle_roll_link")
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
    for tag, body, dist, elev in (("body", pelvis, 1.8, -10), ("feet", foot, 0.6, -20)):
      cam.lookat[:] = rdata.xpos[body]; cam.distance = dist; cam.elevation = elev; cam.azimuth = 210
      hr.update_scene(rdata, camera=cam)
      Image.fromarray(hr.render()).save(sd / f"fixed_{tag}.png")
    print(f"wrote stills -> {sd}/fixed_body.png, fixed_feet.png")

  env.close()


if __name__ == "__main__":
  main()
