"""Instrumented board-DR sweep: does each policy's behavior across board dimensions
make sense? Sweeps length / width / thickness (INCLUDING out-of-training-range values)
for each lineage policy and records rich per-rollout metrics.

Two questions it answers:
  1. PLUMBING: is board DR actually reaching PHYSICS (not just the render)? Logs the
     deck half-size read back from the warp model, and probes far out of range — a
     non-DR policy should degrade at extreme sizes if (and only if) physics sees them.
  2. BEHAVIOR: skating speed, on-board fraction, height stability, penetration — do
     the patterns across configs and across the 309049/311048/313047 lineage make sense?

Usage:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_dr_sweep.py [--steps 400]
Writes video/board_dr_comparison/sweep_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent
TASK = "Mjlab-Skater-Flat-Agibot-X2"

POLICIES = {
  "309049_noDR": REPO / "model_309049.pt",
  "311048_LW": REPO / "model_311048.pt",
  "313047_LWT": REPO / "model_313047.pt",
}
# (axis, [scales]); the other two axes held at 1.0. Training range: L/W +-10%, T +-25%.
SWEEPS = {
  "length": [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4],
  "width": [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4],
  "thickness": [0.4, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0],
}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", type=int, default=400)
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  import numpy as np
  import mujoco
  import torch

  import mjlab.tasks  # noqa: F401
  import mjlab_husky.tasks  # noqa: F401
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
  from mjlab_husky.rl import RslRlVecEnvWrapper
  from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab_husky.tasks.skater.mdp import randomize_board_dims

  device = "cuda:0"
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.eval_mode = False
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(TASK)
  raw = G1SkaterManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(env, asdict(agent_cfg), device=device)
  to_expand = tuple(f for f in randomize_board_dims.model_fields
                    if getattr(raw.sim.wp_model, f).shape[0] != raw.num_envs)
  if to_expand:
    raw.sim.expand_model_fields(to_expand)

  m = raw.sim.mj_model
  wm = raw.sim.model
  dt = raw.step_dt
  gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
  DECK = gid("skateboard/skateboard_deck_collision")
  DECK_B = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "skateboard/skateboard_deck")
  PELVIS = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
  import re
  cap_ids = np.array([g for g in range(m.ngeom)
                      if re.match(r"^robot/(left|right)_foot\d+_collision$",
                                  mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")])
  cap_r = m.geom_size[cap_ids, 0].copy()
  cap_hl = m.geom_size[cap_ids, 1].copy()
  ts = np.linspace(-1.0, 1.0, 5)
  rdata = mujoco.MjData(m)

  def set_board(sl, sw, st):
    ids = torch.zeros(1, dtype=torch.long, device=device)
    randomize_board_dims(raw, ids, length_scale_range=(sl, sl),
                         width_scale_range=(sw, sw), thickness_scale_range=(st, st))
    m.geom_size[DECK] = wm.geom_size[0, DECK].detach().cpu().numpy()
    return wm.geom_size[0, DECK].detach().cpu().numpy().tolist()  # physics deck half-size

  def rollout(ckpt):
    runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)
    # Re-seed before EVERY rollout so the velocity command + reset noise are identical
    # across all (policy, config) rollouts -> the board is the ONLY variable. Without
    # this, SkateUniformVelocityCommandCfg draws a fresh command each reset and swamps
    # any board effect in the motion metrics.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs, _ = env.reset()
    cmd = raw.command_manager.get_command("skate")[0].detach().cpu().numpy()
    rz, pel_xy, board_x, onboard, below = [], [], [], 0, 0
    with torch.inference_mode():
      for _ in range(args.steps):
        obs, _, _, _ = env.step(policy(obs))
        rdata.qpos[:] = raw.sim.data.qpos[0].detach().cpu().numpy()
        mujoco.mj_forward(m, rdata)
        rz.append(float(rdata.xpos[PELVIS][2]))
        pel_xy.append(rdata.xpos[PELVIS][:2].copy())
        board_x.append(float(rdata.xpos[DECK_B][0]))
        dpos, dR = rdata.geom_xpos[DECK], rdata.geom_xmat[DECK].reshape(3, 3)
        dh = m.geom_size[DECK]
        c = rdata.geom_xpos[cap_ids]
        axz = rdata.geom_xmat[cap_ids].reshape(-1, 3, 3)[:, :, 2]
        pts = c[:, None, :] + (ts[None, :, None] * cap_hl[:, None, None]) * axz[:, None, :]
        loc = (pts - dpos) @ dR
        inside = (np.abs(loc[..., 0]) < dh[0]) & (np.abs(loc[..., 1]) < dh[1])
        sole_z = loc[..., 2] - cap_r[:, None]
        below += bool((inside & (sole_z < -dh[2])).any())
        onboard += bool((inside & (np.abs(sole_z - dh[2]) < 0.02)).any())  # sole near deck top
    rz = np.array(rz)
    pel = np.array(pel_xy)
    fell = int(np.argmax(rz < 0.5)) if (rz < 0.5).any() else -1
    horiz = np.linalg.norm(np.diff(pel, axis=0), axis=1) / dt if len(pel) > 1 else np.array([0.0])
    return {
      "fell_at": fell,
      "cmd": [round(float(x), 3) for x in cmd],
      "survived_frac": round((fell if fell >= 0 else args.steps) / args.steps, 3),
      "root_z_mean": round(float(rz.mean()), 3),
      "root_z_min": round(float(rz.min()), 3),
      "root_z_std": round(float(rz.std()), 4),
      "speed_mean_mps": round(float(horiz.mean()), 3),
      "board_travel_m": round(float(board_x[-1] - board_x[0]), 3),
      "onboard_frac": round(onboard / args.steps, 3),
      "below_deck": int(below),
    }

  results = {}
  for axis, scales in SWEEPS.items():
    results[axis] = {}
    for s in scales:
      sl, sw, st = (s, 1.0, 1.0) if axis == "length" else \
                   (1.0, s, 1.0) if axis == "width" else (1.0, 1.0, s)
      deck_half = set_board(sl, sw, st)
      row = {"scale": s, "physics_deck_half": [round(x, 4) for x in deck_half], "policies": {}}
      for pname, ckpt in POLICIES.items():
        row["policies"][pname] = rollout(ckpt)
      results[axis][f"{s:.2f}"] = row
      tag = "  (in-range)" if (axis != "thickness" and 0.9 <= s <= 1.1) or (axis == "thickness" and 0.75 <= s <= 1.25) else "  (OUT of range)"
      surv = " ".join(f"{p.split('_')[0]}={row['policies'][p]['survived_frac']}" for p in POLICIES)
      print(f"{axis:9s} x{s:.2f} deck_half={deck_half}{tag}  survived: {surv}", flush=True)

  out = REPO / "video" / "board_dr_comparison" / "sweep_metrics.json"
  out.write_text(json.dumps(results, indent=2))
  env.close()
  print(f"\nwrote {out}")


if __name__ == "__main__":
  main()
