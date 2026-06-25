"""Robot-to-robot motion retargeting utilities for AMP push clips.

The HUSKY AMP discriminator only consumes *joint positions* (see
``env.get_amp_observations`` -> ``robot.data.joint_pos``), and humanoids share
joint naming (hip/knee/ankle/shoulder/elbow/wrist/waist). So the natural, robust
robot->robot retarget for this objective is a **name-matched joint-space remap**:
copy each destination joint's trajectory from the same-named source joint, apply
optional per-joint sign/offset corrections (limb conventions differ between
robots -- e.g. the G1 elbow flexes positive, the X2 elbow flexes negative), then
clamp to the destination joint limits so every output frame is a valid pose.

This is intentionally simple and **pluggable**: ``remap_joint_clip`` is the one
hook a researcher swaps out to drop in a Cartesian/IK retargeter (which would
additionally match body/end-effector frames) without touching the AMP loader or
the training code.

Clip layout (both source and destination): ``(T, 7 + n_dof)`` =
``[base_pos(3), base_quat(4, wxyz), dof(n_dof)]``.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np


def _match(name: str, patterns: Mapping[str, float]) -> float | None:
  """Return the value whose key (exact name or regex) matches ``name``."""
  for key, val in patterns.items():
    if name == key or re.fullmatch(key, name):
      return val
  return None


def remap_joint_clip(
  clip: np.ndarray,
  src_joint_names: Sequence[str],
  dst_joint_names: Sequence[str],
  *,
  sign: Mapping[str, float] | None = None,
  offset: Mapping[str, float] | None = None,
  default: float = 0.0,
  clip_ranges: np.ndarray | None = None,
) -> np.ndarray:
  """Retarget one motion clip from a source robot's joints onto a destination robot's.

  Args:
    clip: ``(T, 7 + len(src_joint_names))`` source clip ``[base_pos, base_quat, dof]``.
    src_joint_names: source dof order (matches clip columns ``7:``).
    dst_joint_names: destination dof order (output columns ``7:``).
    sign: optional ``{name_or_regex: +/-1}`` applied to matched destination joints.
    offset: optional ``{name_or_regex: radians}`` added after the sign.
    default: value for destination joints with no same-named source joint.
    clip_ranges: optional ``(len(dst_joint_names), 2)`` ``[lo, hi]`` limits; the
      output dof are clamped into these so every frame is a valid pose.

  Returns:
    ``(T, 7 + len(dst_joint_names))`` retargeted clip (base pose copied through).
  """
  clip = np.asarray(clip, dtype=np.float64)
  T = clip.shape[0]
  n_src = len(src_joint_names)
  assert clip.shape[1] == 7 + n_src, (
    f"clip width {clip.shape[1]} != 7 + {n_src} source joints"
  )
  src_idx = {name: 7 + i for i, name in enumerate(src_joint_names)}

  out = np.zeros((T, 7 + len(dst_joint_names)), dtype=np.float64)
  out[:, :7] = clip[:, :7]  # base pos + quat carried through (unused by AMP obs)

  for j, name in enumerate(dst_joint_names):
    col = 7 + j
    if name in src_idx:
      vals = clip[:, src_idx[name]].copy()
      s = _match(name, sign) if sign else None
      if s is not None:
        vals = vals * s
      o = _match(name, offset) if offset else None
      if o is not None:
        vals = vals + o
    else:
      vals = np.full(T, default, dtype=np.float64)
    if clip_ranges is not None:
      lo, hi = clip_ranges[j]
      if hi > lo:  # skip unlimited joints (lo == hi == 0)
        vals = np.clip(vals, lo, hi)
    out[:, col] = vals

  return out
