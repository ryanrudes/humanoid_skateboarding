"""Render side-by-side board-dimension-DR comparison videos.

For each board configuration (a fixed length/width/thickness scale), rolls out three
policies from the lineage and stacks them into one labeled video:
  iter 309049 (no board DR) | iter 311048 (L/W DR) | iter 313047 (L/W+thickness DR)

The physics uses per-world (warp) model fields set by mdp.randomize_board_dims; the
offscreen renderer uses the CPU mj_model (nominal), so we SYNC the deck/marker geoms of
the render model to the per-world values read back from the warp model — otherwise every
video would show a nominal board regardless of the physics.

Outputs under video/board_dr_comparison/:
  compare_<config>.mp4   three policies side by side, labeled with the board dims
  solo_dr_<config>.mp4   just the full-DR policy (iter 313047)
  robustness.json        survival + below-deck stats per (policy, config)

Usage (from repo root):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python scripts/render_dr_comparison.py [--validate] [--configs a,b]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "video" / "board_dr_comparison"
TASK = "Mjlab-Skater-Flat-Agibot-X2"

# (name, length_scale, width_scale, thickness_scale)
CONFIGS = [
  ("nominal", 1.00, 1.00, 1.00),
  ("short", 0.90, 1.00, 1.00),
  ("long", 1.10, 1.00, 1.00),
  ("narrow", 1.00, 0.90, 1.00),
  ("wide", 1.00, 1.10, 1.00),
  ("thin", 1.00, 1.00, 0.75),
  ("thick", 1.00, 1.00, 1.25),
  ("smallest", 0.90, 0.90, 0.75),
  ("largest", 1.10, 1.10, 1.25),
]

# (label, sublabel, accent RGB, checkpoint)
POLICIES = [
  ("iter 309049", "no board DR", (219, 83, 63), REPO / "model_309049.pt"),
  ("iter 311048", "length+width DR", (232, 170, 42), REPO / "model_311048.pt"),
  ("iter 313047", "L+W+thickness DR", (72, 181, 95), REPO / "model_313047.pt"),
]

PANEL_W, PANEL_H = 600, 600
TOP_H, CAP_H = 66, 52
FPS = 50


_FONT_CACHE: dict = {}


def font(size: int, bold: bool = False):
  import glob

  from PIL import ImageFont

  if (size, bold) in _FONT_CACHE:
    return _FONT_CACHE[size, bold]
  stem = "DejaVuSans-Bold" if bold else "DejaVuSans"
  ubuntu = "Ubuntu-Bold" if bold else "Ubuntu-Regular"
  candidates = [
    f"/usr/share/fonts/truetype/dejavu/{stem}.ttf",
    *glob.glob(f"/snap/*/*/usr/share/fonts/truetype/dejavu/{stem}.ttf"),
    f"/usr/share/fonts/truetype/ubuntu/{ubuntu}.ttf",
  ]
  path = next((p for p in candidates if os.path.exists(p)), None)
  f = ImageFont.truetype(path, size) if path else ImageFont.load_default()
  _FONT_CACHE[size, bold] = f
  return f


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--validate", action="store_true",
                  help="render iter313047 on nominal+thick, dump one still each, exit")
  ap.add_argument("--configs", default="all", help="comma list of config names, or 'all'")
  ap.add_argument("--steps", type=int, default=450)
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  import numpy as np
  import mujoco
  import torch
  import imageio.v2 as imageio
  from PIL import Image, ImageDraw

  import mjlab.tasks  # noqa: F401
  import mjlab_husky.tasks  # noqa: F401
  from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
  from mjlab_husky.rl import RslRlVecEnvWrapper
  from mjlab_husky.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab_husky.tasks.skater.mdp import randomize_board_dims

  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "solo_dr").mkdir(exist_ok=True)
  device = "cuda:0"

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.eval_mode = False
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(TASK)
  raw = G1SkaterManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(env, asdict(agent_cfg), device=device)

  # expand only the fields not already per-world (play cfg keeps base_com/skate_com)
  to_expand = tuple(f for f in randomize_board_dims.model_fields
                    if getattr(raw.sim.wp_model, f).shape[0] != raw.num_envs)
  if to_expand:
    raw.sim.expand_model_fields(to_expand)

  m = raw.sim.mj_model
  m.vis.global_.offwidth = max(PANEL_W, 1280)
  m.vis.global_.offheight = max(PANEL_H, 1280)
  wm = raw.sim.model  # warp bridge (torch views)
  gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
  bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
  DECK, MARKER = gid("skateboard/skateboard_deck_collision"), gid("skateboard/skateboard_marker_collision")
  TORSO, PELVIS = bid("robot/torso_link"), bid("robot/pelvis")
  nom_deck = m.geom_size[DECK].copy()
  cap_ids = np.array([g for g in range(m.ngeom)
                      if re.match(r"^robot/(left|right)_foot\d+_collision$",
                                  mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")])
  cap_r = m.geom_size[cap_ids, 0].copy()
  cap_hl = m.geom_size[cap_ids, 1].copy()
  ts_samp = np.linspace(-1.0, 1.0, 5)

  rdata = mujoco.MjData(m)
  renderer = mujoco.Renderer(m, height=PANEL_H, width=PANEL_W)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.elevation, cam.azimuth, cam.distance = -10.0, 210.0, 2.9

  def set_board(sl: float, sw: float, st: float) -> None:
    ids = torch.zeros(1, dtype=torch.long, device=device)
    randomize_board_dims(raw, ids, length_scale_range=(sl, sl),
                         width_scale_range=(sw, sw), thickness_scale_range=(st, st))
    # sync the RENDER model's board geoms to the per-world physics values
    m.geom_size[DECK] = wm.geom_size[0, DECK].detach().cpu().numpy()
    m.geom_size[MARKER] = wm.geom_size[0, MARKER].detach().cpu().numpy()
    m.geom_pos[MARKER] = wm.geom_pos[0, MARKER].detach().cpu().numpy()

  def rollout(ckpt: Path):
    runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)
    # Re-seed so every panel (all policies, all configs) shares one velocity command
    # + reset noise -> a fair A/B where only the policy and board differ.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs, _ = env.reset()
    frames, root_z, below = [], [], 0
    with torch.inference_mode():
      for _ in range(args.steps):
        obs, _, _, _ = env.step(policy(obs))
        rdata.qpos[:] = raw.sim.data.qpos[0].detach().cpu().numpy()
        mujoco.mj_forward(m, rdata)
        root_z.append(float(rdata.xpos[PELVIS][2]))
        # below-deck check (deck frame), vectorized over cap geoms x axis samples
        dpos, dR = rdata.geom_xpos[DECK], rdata.geom_xmat[DECK].reshape(3, 3)
        dh = m.geom_size[DECK]
        c = rdata.geom_xpos[cap_ids]                                    # (K,3)
        axz = rdata.geom_xmat[cap_ids].reshape(-1, 3, 3)[:, :, 2]        # (K,3)
        pts = c[:, None, :] + (ts_samp[None, :, None] * cap_hl[:, None, None]) * axz[:, None, :]
        loc = (pts - dpos) @ dR                                          # dR.T @ v == v @ dR
        inside = (np.abs(loc[..., 0]) < dh[0]) & (np.abs(loc[..., 1]) < dh[1])
        below += bool((inside & ((loc[..., 2] - cap_r[:, None]) < -dh[2])).any())
        cam.lookat[:] = rdata.xpos[TORSO]
        renderer.update_scene(rdata, camera=cam)
        frames.append(renderer.render().copy())
    rz = np.array(root_z)
    fell = int(np.argmax(rz < 0.5)) if (rz < 0.5).any() else -1
    return frames, {"fell_at": fell, "min_root_z": float(rz.min()),
                    "end_root_z": float(rz[-10:].mean()), "below_deck": int(below)}

  if args.validate:
    for name, sl, sw, st in [c for c in CONFIGS if c[0] in ("nominal", "thick")]:
      set_board(sl, sw, st)
      fr, _ = rollout(POLICIES[-1][3])
      Image.fromarray(fr[220]).save(OUT / f"_validate_{name}.png")
      print(f"validate {name}: deck half-size (render model) = {m.geom_size[DECK]}")
    print("wrote validate stills")
    renderer.close()
    env.close()
    return

  wanted = [c for c in CONFIGS if args.configs == "all" or c[0] in args.configs.split(",")]
  stats = {}
  bf, sf, tf = font(19, True), font(17), font(26, True)
  for name, sl, sw, st in wanted:
    set_board(sl, sw, st)
    L, W, T = 2 * m.geom_size[DECK][0], 2 * m.geom_size[DECK][1], 200 * m.geom_size[DECK][2]
    title = f"DECK   length {L:.2f} m (x{sl:.2f})    width {W:.2f} m (x{sw:.2f})    thickness {T:.1f} cm (x{st:.2f})"
    panels, st_row = [], {}
    for lab, sub, color, ckpt in POLICIES:
      fr, s = rollout(ckpt)
      panels.append((lab, sub, color, fr))
      st_row[lab] = s
    stats[name] = {"scales": [sl, sw, st], "policies": st_row}

    n = len(panels)
    W_canvas, H_canvas = n * PANEL_W, TOP_H + CAP_H + PANEL_H
    writer = imageio.get_writer(OUT / f"compare_{name}.mp4", fps=FPS, macro_block_size=1)
    solo = imageio.get_writer(OUT / "solo_dr" / f"{name}.mp4", fps=FPS, macro_block_size=1)
    for i in range(args.steps):
      canvas = Image.new("RGB", (W_canvas, H_canvas), (17, 17, 20))
      d = ImageDraw.Draw(canvas)
      d.text((W_canvas // 2, TOP_H // 2), title, font=bf, fill=(235, 235, 235), anchor="mm")
      for j, (lab, sub, color, fr) in enumerate(panels):
        x0 = j * PANEL_W
        d.rectangle([x0, TOP_H, x0 + PANEL_W, TOP_H + CAP_H], fill=tuple(int(c * 0.35) for c in color))
        d.text((x0 + PANEL_W // 2, TOP_H + CAP_H // 2 - 9), lab, font=bf, fill=(245, 245, 245), anchor="mm")
        d.text((x0 + PANEL_W // 2, TOP_H + CAP_H // 2 + 11), sub, font=sf, fill=color, anchor="mm")
        canvas.paste(Image.fromarray(fr[i]), (x0, TOP_H + CAP_H))
        d.rectangle([x0, TOP_H + CAP_H, x0 + PANEL_W - 1, H_canvas - 1], outline=color, width=3)
      writer.append_data(np.asarray(canvas))
      # solo: full-DR panel only, with board title
      s_can = Image.new("RGB", (PANEL_W, TOP_H + PANEL_H), (17, 17, 20))
      sd = ImageDraw.Draw(s_can)
      sd.text((PANEL_W // 2, TOP_H // 2), f"{L:.2f}m x {W:.2f}m x {T:.1f}cm", font=tf, fill=(235, 235, 235), anchor="mm")
      s_can.paste(Image.fromarray(panels[-1][3][i]), (0, TOP_H))
      solo.append_data(np.asarray(s_can))
    writer.close()
    solo.close()
    print(f"wrote compare_{name}.mp4  |  survival " +
          " ".join(f"{lab.split()[-1]}={'RODE' if st_row[lab]['fell_at'] < 0 else 'fell@'+str(st_row[lab]['fell_at'])}"
                   for lab, *_ in POLICIES))

  (OUT / "robustness.json").write_text(json.dumps(stats, indent=2))
  renderer.close()
  env.close()
  print(f"\ndone -> {OUT}")


if __name__ == "__main__":
  main()
