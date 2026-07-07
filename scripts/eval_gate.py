"""Eval gate: deterministic rollout of a skater checkpoint with pass/fail physics checks.

Measures, over a seeded rollout (play cfg, num_envs=1):
  * riding stability: pelvis height start / end / min, first fall-onset step
  * foot-vs-deck integrity: per-foot steps with any foot collision geom fully below
    the deck top inside its footprint (the deck-pinch / penetration signature), and
    the minimum foot-surface height in the deck frame
  * which foot is the back (tail-side) foot

Foot collision geoms are auto-detected (robot/(left|right)_foot<N>_collision, sphere
or capsule), so this works for both the G1 and the X2.

Usage (from repo root):
    uv run python scripts/eval_gate.py Mjlab-Skater-Flat-Agibot-X2 <ckpt.pt> [--steps 1000]
    # gate mode (exit 1 on failure — for scripted checkpoint bisection):
    uv run python scripts/eval_gate.py <task> <ckpt.pt> --gate

Gate criteria: zero below-deck foot steps AND pelvis never below --fall-z.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("task")
  ap.add_argument("checkpoint")
  ap.add_argument("--steps", type=int, default=1000)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--fall-z", type=float, default=0.55,
                  help="pelvis height below this = fallen (X2 rides at ~0.65)")
  ap.add_argument("--gate", action="store_true", help="exit 1 if the gate fails")
  ap.add_argument("--device", default=None)
  args = ap.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  import numpy as np
  import mujoco
  import torch
  from tqdm import tqdm

  import mjlab.tasks  # noqa: F401
  import mjlab_husky.tasks  # noqa: F401
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
  from mjlab_husky.rl import RslRlVecEnvWrapper
  from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.eval_mode = False
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(args.task)
  raw = G1SkaterManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, map_location=device)
  policy = runner.get_inference_policy(device=device)

  m = raw.sim.mj_model
  import re
  foot = []  # (geom_id, is_left, radius, half_len_or_0)
  for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if re.match(r"^robot/(left|right)_foot\d+_collision$", nm):
      is_cap = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
      foot.append((g, "left" in nm, float(m.geom_size[g][0]),
                   float(m.geom_size[g][1]) if is_cap else 0.0))
  if not foot:
    sys.exit("no robot/*_foot*_collision geoms found — wrong task/model?")
  deck_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "skateboard/skateboard_deck_collision")
  if deck_id < 0:
    sys.exit("skateboard deck geom not found")
  deck_half = m.geom_size[deck_id].copy()
  lankle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/left_ankle_roll_link")
  rankle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/right_ankle_roll_link")
  pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
  rdata = mujoco.MjData(m)
  NS = 9  # samples along each capsule axis

  N = args.steps
  below = np.zeros((N, 2), bool)
  min_z = np.full((N, 2), np.nan)
  ankle_x = np.zeros((N, 2))
  root_z = np.zeros(N)
  done_at = -1

  obs, _ = env.reset()
  with torch.inference_mode():
    for t in tqdm(range(N)):
      obs, _, dones, _ = env.step(policy(obs))
      rdata.qpos[:] = raw.sim.data.qpos[0].detach().cpu().numpy()
      mujoco.mj_forward(m, rdata)
      dpos = rdata.geom_xpos[deck_id]
      dR = rdata.geom_xmat[deck_id].reshape(3, 3)
      root_z[t] = rdata.xpos[pelvis][2]
      ankle_x[t, 0] = (dR.T @ (rdata.xpos[lankle] - dpos))[0]
      ankle_x[t, 1] = (dR.T @ (rdata.xpos[rankle] - dpos))[0]
      lo = [1e9, 1e9]
      bl = [False, False]
      for gid, is_left, r, hl in foot:
        c = rdata.geom_xpos[gid]
        if hl > 0:  # capsule: sample along its axis (local z maps to xmat col 2)
          ax = rdata.geom_xmat[gid].reshape(3, 3)[:, 2]
          ts = np.linspace(-hl, hl, NS)
          pts = c[None, :] + ts[:, None] * ax[None, :]
        else:  # sphere
          pts = c[None, :]
        loc = (dR.T @ (pts - dpos).T).T
        inside = (np.abs(loc[:, 0]) < deck_half[0]) & (np.abs(loc[:, 1]) < deck_half[1])
        if inside.any():
          zmin = float((loc[inside, 2] - r).min())
          f = 0 if is_left else 1
          lo[f] = min(lo[f], zmin)
          bl[f] = bl[f] or (zmin < -deck_half[2])
      for f in (0, 1):
        min_z[t, f] = lo[f] if lo[f] < 1e8 else np.nan
        below[t, f] = bl[f]
      if done_at < 0 and bool(dones[0].item()):
        done_at = t

  back = 0 if ankle_x[:, 0].mean() < ankle_x[:, 1].mean() else 1
  name = ["LEFT", "RIGHT"]
  fall = np.argmax(root_z < args.fall_z) if (root_z < args.fall_z).any() else -1
  print("\n================ EVAL GATE ================")
  print(f"checkpoint: {args.checkpoint}")
  print(f"steps={N} seed={args.seed}  back(tail) foot = {name[back]}")
  for f in (0, 1):
    mz = np.nanmin(min_z[:, f]) if not np.isnan(min_z[:, f]).all() else None
    mz_s = f"{mz * 1000:+.1f}mm" if mz is not None else "n/a (never over deck)"
    print(f"  {name[f]:5s} ({'BACK ' if f == back else 'front'}): below-deck steps "
          f"{int(below[:, f].sum())}/{N}   min foot-surface z (deck frame) "
          f"{mz_s}  (deck bottom {-deck_half[2] * 1000:.0f}mm)")
  print(f"  pelvis z: start {root_z[:10].mean():.3f}  end {root_z[-10:].mean():.3f}  "
        f"min {root_z.min():.3f}   fall onset (z<{args.fall_z}): "
        f"{'never' if fall < 0 else f'step {fall}'}   done at: {done_at}")
  ok = (below.sum() == 0) and fall < 0
  print(f"  GATE: {'PASS' if ok else 'FAIL'}  (zero below-deck steps AND no fall)")
  env.close()
  if args.gate and not ok:
    sys.exit(1)


if __name__ == "__main__":
  main()
