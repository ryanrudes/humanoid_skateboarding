"""Skater-specific event terms (skateboard-dimension domain randomization).

``geom_size`` is not covered by mjlab's generic ``randomize_field`` (no FIELD_SPECS
entry), and a size change must co-write several derived quantities atomically from
ONE sample per env, so board-dimension DR is a dedicated event rather than a stack
of independent ``randomize_field`` terms:

  * ``geom_rbound`` / ``geom_aabb`` — mujoco_warp's broadphase reads these per
    world and NOTHING recomputes them after a ``geom_size`` write; stale bounds
    silently cull true contacts (empirically: zero contacts at 13mm penetration).
  * ``geom_pos`` — the tail-marker box scales/moves with the deck so it stays
    inside the deck footprint and flush with the deck top (it is the sole geom
    behind the ``left_feet_board_contact`` sensor; burying or beaching it kills
    ``bad_feet_off_board`` semantics).
  * ``body_mass`` / ``body_inertia`` / ``body_ipos`` on ``board_tilt_body`` —
    its inertial is compiler-derived from the deck+marker boxes at default
    density, so it must rescale with the dims or the policy trains on
    mislabeled boards.

Trucks, wheels, and the marker SITES are untouched: the wheelbase constant in
``steer_tilt_guide`` and the board spawn height stay valid. Thickness DR moves
the deck TOP by at most +-2.5mm; the robot spawn z is shifted per env to match
(via per-world ``qpos0``, which is what episode resets restore, plus a
consistency write to ``default_root_state``). The board-frame reference-pose
targets are left nominal: a 2.5mm offset costs exp(-2.5mm^2/sigma^2) ~ 0.01%
of the tracking reward (sigma^2 = 0.05) — noise. See the board-dims DR
feasibility notes.

Accepted approximation: compile-time constraint-impedance scalars
(``body_invweight0`` / ``dof_invweight0`` / ``stat.meaninertia``) stay nominal.
They only shape the constraint regularizer, not the mass matrix (dynamics use
the updated mass/inertia via CRB), and stock mjlab mass/inertia DR accepts the
same staleness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.managers.event_manager import requires_model_fields

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _box_composite_diag(
  m_deck: torch.Tensor,
  half_deck: torch.Tensor,
  m_marker: torch.Tensor,
  half_marker: torch.Tensor,
  marker_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Diagonal inertia + COM of deck box (at origin) + marker box (offset).

  Solid-box inertia about its own COM with HALF-extents h: I_a = m/3 * (h_b^2 + h_c^2),
  then parallel-axis both boxes to the composite COM. Off-diagonal products from the
  marker's (x, z) offset are ~0.05% of the diagonal here and are absorbed by ratio-
  scaling against the compiler's nominal values in the caller.
  """
  m_tot = m_deck + m_marker
  com = marker_center * (m_marker / m_tot).unsqueeze(-1)

  def box_diag(m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    hx2, hy2, hz2 = h[..., 0] ** 2, h[..., 1] ** 2, h[..., 2] ** 2
    return torch.stack([(hy2 + hz2), (hx2 + hz2), (hx2 + hy2)], dim=-1) * (m / 3.0).unsqueeze(-1)

  def parallel_axis(m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    dx2, dy2, dz2 = d[..., 0] ** 2, d[..., 1] ** 2, d[..., 2] ** 2
    return torch.stack([(dy2 + dz2), (dx2 + dz2), (dx2 + dy2)], dim=-1) * m.unsqueeze(-1)

  inertia = (
    box_diag(m_deck, half_deck)
    + parallel_axis(m_deck, -com)
    + box_diag(m_marker, half_marker)
    + parallel_axis(m_marker, marker_center - com)
  )
  return inertia, com


@requires_model_fields(
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "geom_pos",
  "body_mass",
  "body_inertia",
  "body_ipos",
  "qpos0",
)
def randomize_board_dims(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  length_scale_range: tuple[float, float] = (0.9, 1.1),
  width_scale_range: tuple[float, float] = (0.9, 1.1),
  thickness_scale_range: tuple[float, float] = (0.75, 1.25),
) -> None:
  """Per-env skateboard deck length/width/thickness randomization (startup mode).

  Samples one (length, width, thickness) scale triple per env and consistently
  writes the deck + marker collision boxes (size, bounds, marker position — the
  marker rides the deck TOP, so its z tracks the thickness to stay flush) and the
  tilt body's mass/inertia/COM. The deck body z is fixed by the trucks, so
  thickness moves the top/bottom faces by half the delta each (+-2.5mm at the
  default range). The sampled scales are stored at ``env.board_dims``
  (num_envs, 3) for logging / future observation use.
  """
  device = env.device
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=device)

  # --- resolve + cache ids and nominal (compiled) values on first call ----------
  cache = getattr(env, "_board_dr_cache", None)
  if cache is None:
    mjm = env.sim.mj_model
    deck = mujoco.mj_name2id(
      mjm, mujoco.mjtObj.mjOBJ_GEOM, "skateboard/skateboard_deck_collision"
    )
    marker = mujoco.mj_name2id(
      mjm, mujoco.mjtObj.mjOBJ_GEOM, "skateboard/skateboard_marker_collision"
    )
    tilt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "skateboard/board_tilt_body")
    assert deck >= 0 and marker >= 0 and tilt >= 0, "skateboard geoms/body not found"
    # Robot free-joint qpos address: episode resets restore qpos from per-world
    # qpos0 (mjwarp.reset_data), so the spawn-z correction below is written there.
    free = [
      j for j in range(mjm.njnt)
      if mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
      and (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("robot/")
    ]
    assert len(free) == 1, f"expected one robot free joint, found {len(free)}"
    root_qadr = int(mjm.jnt_qposadr[free[0]])
    t = lambda x: torch.tensor(x, device=device, dtype=torch.float32)  # noqa: E731
    # body_inertia is stored in the PRINCIPAL frame (body_iquat); for this body the
    # compiler sorts moments descending, a ~90deg rotation of the geom frame. The
    # analytic ratio below is computed in the geom/body frame, so it must be
    # permuted into the principal order via squared direction cosines:
    # principal_i = sum_j R[j,i]^2 * body_j  (columns of R = principal axes).
    R = torch.zeros(9, dtype=torch.float64)
    mujoco.mju_quat2Mat(R.numpy(), mjm.body_iquat[tilt])
    cache = {
      "deck": deck,
      "marker": marker,
      "tilt": tilt,
      "deck_half": t(mjm.geom_size[deck]),
      "marker_half": t(mjm.geom_size[marker]),
      "marker_pos": t(mjm.geom_pos[marker]),
      "tilt_mass": float(mjm.body_mass[tilt]),
      "tilt_inertia": t(mjm.body_inertia[tilt]),
      "tilt_ipos": t(mjm.body_ipos[tilt]),
      "tilt_R2": (R.reshape(3, 3) ** 2).to(device=device, dtype=torch.float32),
      "root_qadr": root_qadr,
      "root_z_nom": float(mjm.qpos0[root_qadr + 2]),
      "drs_z_nom": float(env.scene["robot"].data.default_root_state[0, 2]),
    }
    env._board_dr_cache = cache
    env.board_dims = torch.ones(env.num_envs, 3, device=device)

  deck, marker, tilt = cache["deck"], cache["marker"], cache["tilt"]
  n = len(env_ids)

  # --- sample per-env scales -----------------------------------------------------
  def sample(rng: tuple[float, float]) -> torch.Tensor:
    return torch.rand(n, device=device) * (rng[1] - rng[0]) + rng[0]

  s_len = sample(length_scale_range)
  s_wid = sample(width_scale_range)
  s_thk = sample(thickness_scale_range)
  env.board_dims[env_ids, 0] = s_len
  env.board_dims[env_ids, 1] = s_wid
  env.board_dims[env_ids, 2] = s_thk

  model = env.sim.model  # torch views over the (expanded) warp model arrays

  # --- deck + marker boxes: size, bounding radius, aabb, marker position ---------
  # Deck scales in all three axes; the marker keeps its own thickness (it is the
  # grip-zone contact target riding the deck TOP, not part of the wood).
  deck_half_new = cache["deck_half"].unsqueeze(0) * torch.stack([s_len, s_wid, s_thk], -1)
  marker_half_new = cache["marker_half"].unsqueeze(0) * torch.stack(
    [s_len, s_wid, torch.ones_like(s_len)], -1
  )
  # Marker slides with the tail (x scales), stays centered in y, and its z tracks
  # the deck top so its top face stays exactly flush (the left_feet_board_contact
  # sensor matches ONLY this geom — burying or beaching it breaks feet_off_board).
  marker_pos = cache["marker_pos"].unsqueeze(0).repeat(n, 1)
  marker_pos[:, 0] *= s_len
  marker_pos[:, 2] = deck_half_new[:, 2] - marker_half_new[:, 2]
  # (aabb is (center, half-extents) in the GEOM frame -> center stays 0.)
  for gid, half in ((deck, deck_half_new), (marker, marker_half_new)):
    model.geom_size[env_ids, gid] = half
    model.geom_rbound[env_ids, gid] = torch.linalg.norm(half, dim=-1)
    model.geom_aabb[env_ids, gid, 0] = 0.0
    model.geom_aabb[env_ids, gid, 1] = half
  model.geom_pos[env_ids, marker] = marker_pos

  # --- tilt-body inertial: rescale like the compiler would -----------------------
  # Both boxes share one density, so mass scales by total volume and COM/inertia
  # follow the analytic 2-box composite (unit density — it cancels in the ratios;
  # the analytic masses carry the per-box volume scaling). Ratio-scaling the
  # compiled nominal preserves the compiler's principal-frame subtleties.
  vol = lambda h: 8.0 * h[..., 0] * h[..., 1] * h[..., 2]  # noqa: E731
  inertia_new, com_new = _box_composite_diag(
    vol(deck_half_new), deck_half_new,
    vol(marker_half_new), marker_half_new,
    marker_pos,
  )
  deck_half_nom = cache["deck_half"].unsqueeze(0)
  marker_half_nom = cache["marker_half"].unsqueeze(0)
  inertia_nom, com_nom = _box_composite_diag(
    vol(deck_half_nom), deck_half_nom,
    vol(marker_half_nom), marker_half_nom,
    cache["marker_pos"].unsqueeze(0),
  )
  vol_ratio = (vol(deck_half_new) + vol(marker_half_new)) / (
    vol(deck_half_nom) + vol(marker_half_nom)
  )
  model.body_mass[env_ids, tilt] = cache["tilt_mass"] * vol_ratio
  # ratio is body-frame (x,y,z); permute to the principal order body_inertia uses.
  ratio_principal = (inertia_new / inertia_nom) @ cache["tilt_R2"]
  model.body_inertia[env_ids, tilt] = cache["tilt_inertia"].unsqueeze(0) * ratio_principal
  # COM: the analytic composite reproduces the compiler to machine precision at
  # nominal, so apply its delta to the compiled nominal.
  model.body_ipos[env_ids, tilt] = cache["tilt_ipos"].unsqueeze(0) + (com_new - com_nom)

  # --- robot spawn-z correction ---------------------------------------------------
  # Shift each env's spawn by its deck-top delta so the keyframe's seating relative
  # to the deck top is preserved at every thickness (exact for the X2, whose
  # keyframe seats the on-board foot on the nominal top; run validate_board_dr.py
  # against the G1 task before G1 training). A rigid root shift cannot fix both
  # support surfaces: the DECK foot is anchored (it gates the feet_off_board
  # termination sensor and presses on the free-floating board) and the GROUND foot
  # inherits the +-2.5mm residual against static terrain, where it settles benignly.
  # Resets restore qpos from per-world qpos0 (absolute write => idempotent);
  # default_root_state is kept consistent for any future reset_root_state_* event.
  dz = cache["deck_half"][2] * (s_thk - 1.0)
  model.qpos0[env_ids, cache["root_qadr"] + 2] = cache["root_z_nom"] + dz
  env.scene["robot"].data.default_root_state[env_ids, 2] = cache["drs_z_nom"] + dz
