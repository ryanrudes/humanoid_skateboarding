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

Deck thickness, trucks, wheels, and the marker SITES are untouched (tier a):
spawn heights, reference poses, and the wheelbase constant in
``steer_tilt_guide`` all stay valid. See the board-dims DR feasibility notes.

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
)
def randomize_board_dims(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  length_scale_range: tuple[float, float] = (0.9, 1.1),
  width_scale_range: tuple[float, float] = (0.9, 1.1),
) -> None:
  """Per-env skateboard deck length/width randomization (startup mode).

  Samples one (length_scale, width_scale) pair per env and consistently writes the
  deck + marker collision boxes (size, bounds, marker position) and the tilt body's
  mass/inertia/COM. The sampled scales are stored at ``env.board_dims`` (num_envs, 2)
  for logging / future observation use.
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
    }
    env._board_dr_cache = cache
    env.board_dims = torch.ones(env.num_envs, 2, device=device)

  deck, marker, tilt = cache["deck"], cache["marker"], cache["tilt"]
  n = len(env_ids)

  # --- sample per-env scales -----------------------------------------------------
  s_len = (
    torch.rand(n, device=device) * (length_scale_range[1] - length_scale_range[0])
    + length_scale_range[0]
  )
  s_wid = (
    torch.rand(n, device=device) * (width_scale_range[1] - width_scale_range[0])
    + width_scale_range[0]
  )
  env.board_dims[env_ids, 0] = s_len
  env.board_dims[env_ids, 1] = s_wid
  scale_xy1 = torch.stack([s_len, s_wid, torch.ones_like(s_len)], dim=-1)  # z fixed

  model = env.sim.model  # torch views over the (expanded) warp model arrays

  # --- deck + marker boxes: size, bounding radius, aabb, marker position ---------
  # (aabb is (center, half-extents) in the GEOM frame -> center stays 0.)
  for gid, half0 in ((deck, cache["deck_half"]), (marker, cache["marker_half"])):
    half = half0.unsqueeze(0) * scale_xy1
    model.geom_size[env_ids, gid] = half
    model.geom_rbound[env_ids, gid] = torch.linalg.norm(half, dim=-1)
    model.geom_aabb[env_ids, gid, 0] = 0.0
    model.geom_aabb[env_ids, gid, 1] = half
  # Marker slides with the tail (x scales); y stays centered, z stays flush with
  # the deck top (thickness unchanged in tier a).
  marker_pos = cache["marker_pos"].unsqueeze(0).repeat(n, 1)
  marker_pos[:, 0] *= s_len
  model.geom_pos[env_ids, marker] = marker_pos

  # --- tilt-body inertial: rescale like the compiler would -----------------------
  # Both boxes share one density and thickness is fixed, so mass scales exactly by
  # s_len*s_wid and the COM x-offset scales exactly by s_len. For inertia, ratio-
  # scale the compiled nominal by the analytic composite (unit density — it cancels
  # in the ratio, and the analytic masses carry the s_len*s_wid factor), preserving
  # the compiler's principal-frame subtleties in the nominal.
  vol = lambda h: 8.0 * h[..., 0] * h[..., 1] * h[..., 2]  # noqa: E731
  deck_half_new = cache["deck_half"].unsqueeze(0) * scale_xy1
  marker_half_new = cache["marker_half"].unsqueeze(0) * scale_xy1
  inertia_new, _ = _box_composite_diag(
    vol(deck_half_new), deck_half_new,
    vol(marker_half_new), marker_half_new,
    marker_pos,
  )
  deck_half_nom = cache["deck_half"].unsqueeze(0)
  marker_half_nom = cache["marker_half"].unsqueeze(0)
  inertia_nom, _ = _box_composite_diag(
    vol(deck_half_nom), deck_half_nom,
    vol(marker_half_nom), marker_half_nom,
    cache["marker_pos"].unsqueeze(0),
  )
  model.body_mass[env_ids, tilt] = cache["tilt_mass"] * s_len * s_wid
  # ratio is body-frame (x,y,z); permute to the principal order body_inertia uses.
  ratio_principal = (inertia_new / inertia_nom) @ cache["tilt_R2"]
  model.body_inertia[env_ids, tilt] = cache["tilt_inertia"].unsqueeze(0) * ratio_principal
  ipos = cache["tilt_ipos"].unsqueeze(0).repeat(n, 1)
  ipos[:, 0] *= s_len
  model.body_ipos[env_ids, tilt] = ipos
