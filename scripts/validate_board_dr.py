"""Validate skateboard-dimension domain randomization (the `board_dims` event).

Builds the TRAIN-cfg env with N worlds and asserts, per world:
  1. the event ran: env.board_dims sampled within range, per-world model fields expanded
  2. derived-quantity consistency: rbound == ||half_size||, aabb == (0, half_size),
     marker inside the deck footprint and flush with the (thickness-scaled) deck
     top, tilt-body mass/inertia/COM matching per-world recompiled ground truth
  2.5 spawn-settle: feet land on their per-world deck top after reset
  3. CONTACT ground truth (the stale-rbound trap detector): with an identical qpos in
     every world placing the front foot near the NOMINAL tail edge, each world's
     foot-deck contacts must match a vanilla-MuJoCo recomputation using THAT world's
     deck size — short decks lose the contact, long decks keep it.
  4. stability: 100 sim steps, no NaNs
  5. play cfg control: the event is absent and the board is nominal in play mode

Usage (from repo root):  MUJOCO_GL=egl uv run python scripts/validate_board_dr.py
Exit code 0 = all assertions pass.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import mjlab_husky.tasks  # noqa: F401
from mjlab_husky.envs import G1SkaterManagerBasedRlEnv
from mjlab_husky.tasks.registry import load_env_cfg

TASK = "Mjlab-Skater-Flat-Agibot-X2"
N = 8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

passed, failed = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
  (passed if ok else failed).append(name)
  print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> None:  # noqa: PLR0915
  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg.scene.num_envs = N
  env_cfg.seed = 7
  print(f"building TRAIN env, num_envs={N} (warp compile takes a few minutes)...")
  env = G1SkaterManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode=None)
  sim = env.sim
  mjm = sim.mj_model

  gid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
  bid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
  DECK = gid("skateboard/skateboard_deck_collision")
  MARKER = gid("skateboard/skateboard_marker_collision")
  TILT = bid("skateboard/board_tilt_body")

  print("\n== 1. event ran, fields expanded ==")
  dims = getattr(env, "board_dims", None)
  check("env.board_dims exists", dims is not None)
  assert dims is not None
  d_np = dims.cpu().numpy()
  check("board_dims shape", d_np.shape == (N, 3), str(d_np.shape))
  check("L/W scales within (0.9,1.1)", bool((d_np[:, :2] >= 0.9 - 1e-6).all() and (d_np[:, :2] <= 1.1 + 1e-6).all()),
        f"min={d_np[:, :2].min():.3f} max={d_np[:, :2].max():.3f}")
  check("thickness scale within (0.75,1.25)", bool((d_np[:, 2] >= 0.75 - 1e-6).all() and (d_np[:, 2] <= 1.25 + 1e-6).all()),
        f"min={d_np[:, 2].min():.3f} max={d_np[:, 2].max():.3f}")
  check("scales vary across envs", float(d_np.std()) > 0.01, f"std={d_np.std():.4f}")
  GS = sim.model.geom_size
  for f in ("geom_size", "geom_rbound", "geom_aabb", "geom_pos", "body_mass", "body_inertia", "body_ipos", "qpos0"):
    arr = getattr(sim.model, f)
    check(f"{f} expanded per-world", arr.shape[0] == N, str(tuple(arr.shape)))

  print("\n== 2. derived-quantity consistency ==")
  nom_deck = mjm.geom_size[DECK].copy()      # CPU model keeps nominal values
  nom_marker = mjm.geom_size[MARKER].copy()
  nom_mpos = mjm.geom_pos[MARKER].copy()
  nom_mass = float(mjm.body_mass[TILT])
  gs = GS.cpu().numpy()
  rb = sim.model.geom_rbound.cpu().numpy()
  ab = sim.model.geom_aabb.cpu().numpy()
  gp = sim.model.geom_pos.cpu().numpy()
  bm = sim.model.body_mass.cpu().numpy()
  exp_deck = nom_deck[None] * d_np
  check("deck geom_size == nominal*scales", np.allclose(gs[:, DECK], exp_deck, atol=1e-6),
        f"max err {np.abs(gs[:, DECK] - exp_deck).max():.2e}")
  check("deck rbound == ||half||", np.allclose(rb[:, DECK], np.linalg.norm(gs[:, DECK], axis=1), atol=1e-6))
  check("deck aabb == (0, half)", np.allclose(ab[:, DECK, 0], 0) and np.allclose(ab[:, DECK, 1], gs[:, DECK], atol=1e-6))
  check("marker rbound == ||half||", np.allclose(rb[:, MARKER], np.linalg.norm(gs[:, MARKER], axis=1), atol=1e-6))
  in_x = np.abs(gp[:, MARKER, 0]) + gs[:, MARKER, 0] <= gs[:, DECK, 0] + 1e-6
  in_y = gs[:, MARKER, 1] <= gs[:, DECK, 1] + 1e-6
  check("marker inside deck footprint", bool(in_x.all() and in_y.all()))
  flush = gp[:, MARKER, 2] + gs[:, MARKER, 2] - gs[:, DECK, 2]
  check("marker top flush with deck top (thickness-tracking)", np.allclose(flush, 0.0, atol=1e-7),
        f"max dev {np.abs(flush).max():.2e}")

  # inertial GROUND TRUTH: recompile the standalone skateboard XML per world with
  # that world's dims and compare the compiler's body_inertia/body_ipos (this is
  # what caught the principal-frame permutation bug — do not weaken it).
  bi = sim.model.body_inertia.cpu().numpy()[:, TILT]
  ip = sim.model.body_ipos.cpu().numpy()[:, TILT]
  from mjlab_husky.asset_zoo.robots.skateboard.g1_skater_constants import SKATEBOARD_XML
  max_ierr, max_perr, max_merr = 0.0, 0.0, 0.0
  for w in range(N):
    spec = mujoco.MjSpec.from_file(str(SKATEBOARD_XML))
    sl, sw, st = float(d_np[w, 0]), float(d_np[w, 1]), float(d_np[w, 2])
    for g in spec.geoms:
      if g.name == "skateboard_deck_collision":
        g.size = [nom_deck[0] * sl, nom_deck[1] * sw, nom_deck[2] * st]
      elif g.name == "skateboard_marker_collision":
        g.size = [nom_marker[0] * sl, nom_marker[1] * sw, nom_marker[2]]
        g.pos = [nom_mpos[0] * sl, nom_mpos[1], nom_deck[2] * st - nom_marker[2]]
    ref = spec.compile()
    rt = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_BODY, "board_tilt_body")
    max_ierr = max(max_ierr, float(np.abs(bi[w] / ref.body_inertia[rt] - 1).max()))
    max_perr = max(max_perr, float(np.abs(ip[w] - ref.body_ipos[rt]).max()))
    max_merr = max(max_merr, float(abs(bm[w, TILT] / ref.body_mass[rt] - 1)))
  check("tilt mass matches recompiled ground truth", max_merr < 1e-5, f"max rel err {max_merr:.2e}")
  check("tilt inertia matches recompiled ground truth", max_ierr < 1e-3, f"max rel err {max_ierr:.2e}")
  check("tilt COM matches recompiled ground truth", max_perr < 1e-6, f"max abs err {max_perr:.2e}")

  print("\n== 2.5 spawn settle at per-world deck tops (thickness DR) ==")
  # The robot spawn keyframe seats the on-deck foot on the NOMINAL deck top; with
  # thickness DR the real top is +-2.5mm away. Measure the foot-sole vs deck-top
  # gap per world right after reset and after 25 substeps of settling.
  env.reset()
  vd = mujoco.MjData(mjm)

  def ondeck_gap(w: int) -> float:
    vd.qpos[:] = sim.data.qpos[w].detach().cpu().numpy()
    mujoco.mj_forward(mjm, vd)
    dc = vd.geom_xpos[DECK]
    top = dc[2] + gs[w, DECK, 2]
    best = 1e9
    for g in range(mjm.ngeom):
      nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
      if "foot" in nm and nm.endswith("_collision") and nm.startswith("robot/"):
        p = vd.geom_xpos[g]
        if abs(p[0] - dc[0]) < gs[w, DECK, 0] and abs(p[1] - dc[1]) < gs[w, DECK, 1]:
          best = min(best, p[2] - mjm.geom_size[g, 0] - top)
    return best

  # spawn-z correction: per-world qpos0 root z must equal nominal + deck-top delta
  q0 = sim.model.qpos0.cpu().numpy()
  jadr_r = [int(mjm.jnt_qposadr[j]) for j in range(mjm.njnt)
            if mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("robot/")][0]
  exp_z = mjm.qpos0[jadr_r + 2] + nom_deck[2] * (d_np[:, 2] - 1.0)
  check("qpos0 robot spawn z shifted by deck-top delta", np.allclose(q0[:, jadr_r + 2], exp_z, atol=1e-6),
        f"max err {np.abs(q0[:, jadr_r + 2] - exp_z).max() * 1000:.3f}mm")

  gaps0 = np.array([ondeck_gap(w) for w in range(N)])
  for _ in range(25):
    sim.step()
  gaps1 = np.array([ondeck_gap(w) for w in range(N)])
  print(f"    gap at reset  (mm): {np.round(gaps0 * 1000, 2).tolist()}")
  print(f"    gap after 25 substeps (mm): {np.round(gaps1 * 1000, 2).tolist()}")
  # spawn-z correction removes the thickness mismatch; residual is reset joint
  # noise / foot pitch only
  check("spawn gap at reset within +-2.5mm of surface", bool((np.abs(gaps0) < 2.5e-3).all()),
        f"max |gap| {np.abs(gaps0).max() * 1000:.2f}mm")
  check("foot settles onto per-world deck top", bool((gaps1 > -6e-3).all() and (gaps1 < 2e-3).all()),
        f"range [{gaps1.min() * 1000:.2f}, {gaps1.max() * 1000:.2f}]mm")

  print("\n== 3. per-world contact ground truth (stale-rbound trap detector) ==")
  env.reset()
  qpos0 = sim.data.qpos[0].detach().cpu().numpy().astype(np.float64).copy()
  vd.qpos[:] = qpos0
  mujoco.mj_forward(mjm, vd)
  deck_R = vd.geom_xmat[DECK].reshape(3, 3).copy()
  deck_c = vd.geom_xpos[DECK].copy()
  # front (on-deck) foot geoms + its heel position in deck-local x
  foot = {}
  for g in range(mjm.ngeom):
    nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "foot" in nm and nm.endswith("_collision") and nm.startswith("robot/"):
      foot.setdefault("left" if "left" in nm else "right", []).append(g)
  sides = {}
  for s, gs_ in foot.items():
    xs, zs = [], []
    for g in gs_:
      p = vd.geom_xpos[g]
      R = vd.geom_xmat[g].reshape(3, 3)
      hl, r = mjm.geom_size[g, 1], mjm.geom_size[g, 0]
      for sgn in (-1, 1):
        e = p + sgn * hl * R[:, 2]
        xs.append((deck_R.T @ (e - deck_c))[0])
        zs.append(e[2] - r)
    sides[s] = (min(xs), max(xs), min(zs))
  front = max(sides, key=lambda s: sides[s][0] + sides[s][1])
  FRONT = foot[front]
  heel_x, _, sole_z = sides[front]
  jadr = mjm.jnt_qposadr[
    [j for j in range(mjm.njnt)
     if "floating_base_joint_skateboard" in (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or "")][0]
  ]
  # identical qpos in all worlds: heel at deck-local x so that per-world deck length
  # decides contact — target the NOMINAL edge minus a hair (nominal keeps contact,
  # -10% decks lose it, +10% keep it).
  target = nom_deck[0] * 0.985
  q = qpos0.copy()
  q[jadr:jadr + 3] += deck_R[:, 0] * (heel_x - target)
  q[jadr + 2] = sole_z + 0.003 - nom_deck[2]
  q[jadr + 3:jadr + 7] = [1, 0, 0, 0]
  sim.data.qpos[:] = torch.tensor(np.stack([q] * N), device=DEVICE, dtype=torch.float32)
  sim.data.qvel[:] = 0.0
  sim.forward()
  wd = sim.wp_data if hasattr(sim, "wp_data") else sim.data
  nacon = int(wd.nacon.numpy()[0])
  cg = wd.contact.geom.numpy()[:nacon]
  cw = wd.contact.worldid.numpy()[:nacon]
  cd = wd.contact.dist.numpy()[:nacon]
  warp_hits = np.zeros(N, int)
  for i in range(nacon):
    if DECK in cg[i] and (set(cg[i]) & set(FRONT)) and cd[i] < 0:
      warp_hits[cw[i]] += 1
  agree = 0
  for w in range(N):
    save = (mjm.geom_size[DECK].copy(), mjm.geom_rbound[DECK].copy(), mjm.geom_aabb[DECK].copy())
    mjm.geom_size[DECK] = gs[w, DECK]
    mjm.geom_rbound[DECK] = np.linalg.norm(gs[w, DECK])
    mjm.geom_aabb[DECK] = [0, 0, 0, *gs[w, DECK]]
    vd.qpos[:] = q
    vd.qvel[:] = 0
    mujoco.mj_forward(mjm, vd)
    v_hits = sum(
      1 for i in range(vd.ncon)
      if DECK in vd.contact.geom[i] and (set(vd.contact.geom[i]) & set(FRONT)) and vd.contact.dist[i] < 0
    )
    mjm.geom_size[DECK], mjm.geom_rbound[DECK], mjm.geom_aabb[DECK] = save
    match = (warp_hits[w] > 0) == (v_hits > 0)
    agree += match
    print(f"    world {w}: s_len={d_np[w, 0]:.3f}  warp contacts={warp_hits[w]}  vanilla={v_hits}  "
          f"{'ok' if match else 'MISMATCH'}")
  check("warp contact presence matches per-world vanilla truth", agree == N, f"{agree}/{N}")
  # a capsule end-sphere penetrating `pen` below the deck top still touches the top
  # edge with its center overhanging by up to sqrt(r^2-(r-pen)^2) — classify "short"
  # only beyond that geometric margin so the assertion cannot flake near the edge.
  r_foot = float(mjm.geom_size[FRONT[0], 0])
  overhang = float(np.sqrt(max(r_foot**2 - (r_foot - 0.003) ** 2, 0.0)))
  short = d_np[:, 0] < 0.985 - overhang / nom_deck[0]
  if short.any():
    check("short decks lose the edge contact", bool((warp_hits[short] == 0).all()))
  # discriminating power: the probe state must produce contact in at least one world
  # (all-zero everywhere would make the vanilla agreement check vacuous)
  check("probe state discriminates (some world contacts)", bool((warp_hits > 0).any()),
        f"hits={warp_hits.tolist()}")

  print("\n== 4. stability: 100 steps ==")
  for _ in range(100):
    sim.step()
  qp = sim.data.qpos.detach().cpu().numpy()
  check("no NaNs after 100 steps", not np.isnan(qp).any())
  env.close()

  print("\n== 5. play cfg control (nominal board, no event) ==")
  play_cfg = load_env_cfg(TASK, play=True)
  check("board_dims absent from play events", "board_dims" not in play_cfg.events)

  print(f"\n{'=' * 60}\n{len(passed)} passed, {len(failed)} failed" + (f": {failed}" if failed else ""))
  raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
  main()
