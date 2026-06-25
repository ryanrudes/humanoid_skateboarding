"""Retarget the G1 AMP push clips onto the AgiBot X2 for AMP training.

GMR only retargets *human* motion -> robot, and the HUSKY authors never published
the human skate-push source, so the only push reference we have is the
G1-retargeted clips in ``dataset/skate_push``. This script does a robot->robot
joint-name remap (``rsl_rl.utils.retarget.remap_joint_clip``) to produce X2 push
clips in ``dataset/skate_push_x2`` that the (generalized) AMP loader can consume.

The result is a coarse reference (limb proportions differ); it is the "let's try"
research artifact, gated behind ``amp_enabled`` -- set ``amp_enabled=False`` in the
X2 rl_cfg to skip AMP entirely and shape the push phase with task rewards only.

Usage (from repo root):
    uv run python scripts/retarget_push_g1_to_x2.py
"""

from __future__ import annotations

import glob
import os

import mjlab_husky.tasks  # noqa: F401  -- complete task discovery before importing constants
import mujoco
import numpy as np

from mjlab_husky.asset_zoo.robots.agibot_x2.x2_skater_constants import get_x2_spec
from rsl_rl.utils.retarget import remap_joint_clip

# Canonical Unitree G1 29-DoF joint order -- the order the mocap dof columns 7:36
# follow (mirrors test_scene/replay_dataset.py::G1_29DOF_JOINT_ORDER).
G1_29DOF_JOINT_ORDER = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

SRC_DIR = "dataset/skate_push"
DST_DIR = "dataset/skate_push_x2"

# Per-joint convention corrections (robot->robot). The G1 elbow flexes positive
# while the X2 elbow range is [-2.3556, 0], so flip its sign. Extend this map to
# refine the retarget for other joints whose conventions differ.
SIGN_OVERRIDES = {".*_elbow_joint": -1.0}


def _x2_joint_names_and_ranges():
  model = get_x2_spec().compile()
  names, ranges = [], []
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    lo, hi = model.jnt_range[j]
    # Apply the same soft limit factor the articulation uses, to stay in-range.
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.9
    ranges.append((mid - half, mid + half))
  return names, np.array(ranges, dtype=np.float64)


def main() -> None:
  dst_names, dst_ranges = _x2_joint_names_and_ranges()
  os.makedirs(DST_DIR, exist_ok=True)

  src_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.npy")))
  if not src_files:
    raise FileNotFoundError(f"no source clips in {SRC_DIR}")

  for src_path in src_files:
    clip = np.load(src_path, allow_pickle=True)
    # G1 clips are (T, 36) = base(7) + 29 dof; use the first 7+29 columns.
    clip = np.asarray(clip, dtype=np.float64)[:, : 7 + len(G1_29DOF_JOINT_ORDER)]
    out = remap_joint_clip(
      clip,
      src_joint_names=G1_29DOF_JOINT_ORDER,
      dst_joint_names=dst_names,
      sign=SIGN_OVERRIDES,
      clip_ranges=dst_ranges,
    )
    dst_path = os.path.join(DST_DIR, os.path.basename(src_path))
    np.save(dst_path, out)
    print(f"{src_path} {clip.shape} -> {dst_path} {out.shape}")


if __name__ == "__main__":
  main()
