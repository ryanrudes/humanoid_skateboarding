"""Retarget the G1 transition reference poses onto the AgiBot X2.

The skater env tracks, during the two transition phases, target whole-body poses
expressed in the skateboard's root frame:
``dataset/ref_pose/<robot>_{push,steer}_start_pose_b.npy``, shape
``(num_bodies, 7)`` = ``[pos | quat(wxyz)]`` per robot body (the transition reward
only reads the ``beizer_names`` bodies, which share names between G1 and X2).

This is a robot->robot **retarget of the published G1 references**, mirroring the
AMP-clip retarget (``retarget_push_g1_to_x2.py``): take the G1 reference's root
pose (the pelvis row — this is what carries the steer yaw and the on-board
centering) and the G1 joint stance, remap the joints onto the X2 by name (same
elbow-sign flip, clamp to X2 limits, head -> 0), and re-run forward kinematics on
the X2 so the limbs stay connected and the X2 inherits the G1's placement. The
X2's different limb lengths make the foot seating approximate, but the gross
placement (facing, centering) comes straight from the G1.

Usage (from repo root):
    uv run python scripts/gen_ref_pose.py
"""

from __future__ import annotations

import re

import mjlab_husky.tasks  # noqa: F401  -- complete task discovery before importing constants
import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import subtract_frame_transforms

from mjlab_husky.asset_zoo.robots.agibot_x2.x2_skater_constants import get_x2_spec
from mjlab_husky.asset_zoo.robots.skateboard.g1_skater_constants import (
  PUSH_INIT_KEYFRAME as G1_PUSH_KEYFRAME,
)
from mjlab_husky.asset_zoo.robots.skateboard.g1_skater_constants import (
  SKATEBOARD_INIT_KEYFRAME,
  get_g1_spec,
)
from mjlab_husky.tasks.registry import load_env_cfg

# Per-joint convention corrections G1 -> X2 (the X2 elbow range is [-2.3556, 0],
# opposite sign to the G1). Mirrors retarget_push_g1_to_x2.py::SIGN_OVERRIDES.
SIGN_OVERRIDES = {".*_elbow_joint": -1.0}


def _hinge_joint_names(model) -> list[str]:
  return [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(model.njnt)
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  ]


def _sign(name: str) -> float:
  for expr, s in SIGN_OVERRIDES.items():
    if re.fullmatch(expr, name):
      return s
  return 1.0


def _remap_joints(g1_joints: dict[str, float], x2_model) -> dict[str, float]:
  """Name-matched G1 -> X2 joint remap (sign-corrected, clamped to X2 limits)."""
  out: dict[str, float] = {}
  for j in range(x2_model.njnt):
    name = mujoco.mj_id2name(x2_model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name is None or x2_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    val = g1_joints.get(name, 0.0) * _sign(name)  # X2-only joints (head) -> 0
    if x2_model.jnt_limited[j]:
      lo, hi = x2_model.jnt_range[j]
      val = float(np.clip(val, lo, hi))
    out[name] = val
  return out


def _set_qpos(model, data, root_pos, root_quat, joint_pos: dict[str, float]) -> None:
  mujoco.mj_resetData(model, data)
  data.qpos[0:3] = root_pos
  data.qpos[3:7] = root_quat
  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name is not None and name in joint_pos:
      data.qpos[model.jnt_qposadr[j]] = joint_pos[name]
  mujoco.mj_forward(model, data)


def _lowest_foot_sphere_z(model, data, side: str) -> float:
  """World-z of the lowest point of `side`'s foot collision spheres (uses current data)."""
  lo = float("inf")
  for g in range(model.ngeom):
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if f"{side}_foot" in nm and nm.endswith("_collision"):
      lo = min(lo, float(data.geom_xpos[g, 2] - model.geom_size[g, 0]))
  return lo


def _level_quat(quat) -> np.ndarray:
  """Flatten a foot orientation onto the surface, keeping its toe heading.

  The X2 foot's collision spheres lie in the ankle frame's z = const (sole) plane, so a
  sole parallel to the (horizontal) deck/ground means the ankle z-axis points world-up.
  Build that: z -> world up, x -> the horizontal heading of the input's toe (+x) axis,
  y = z x x. Replaces inheriting the G1's tilted ankle, which left the X2 sole resting on
  an edge ~10 deg off horizontal.
  """
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=np.float64))
  fwd = R.reshape(3, 3)[:, 0].copy()  # toe (+x) direction in world
  fwd[2] = 0.0
  if np.linalg.norm(fwd) < 1e-6:
    fwd = np.array([1.0, 0.0, 0.0])  # toe pointing ~straight up/down: pick an arbitrary heading
  fwd /= np.linalg.norm(fwd)
  z = np.array([0.0, 0.0, 1.0])
  y = np.cross(z, fwd); y /= np.linalg.norm(y)
  x = np.cross(y, z)
  out = np.zeros(4)
  mujoco.mju_mat2Quat(out, np.column_stack([x, y, z]).reshape(9))
  return out


_LEGS = {
  "left": [f"left_{j}" for j in
           ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint",
            "ankle_pitch_joint", "ankle_roll_joint")],
  "right": [f"right_{j}" for j in
            ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint",
             "ankle_pitch_joint", "ankle_roll_joint")],
}
_WAIST = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]

# Per-axis IK weights [pos x,y,z, rot x,y,z]. A foot planted on the deck matches
# the G1 ankle pose evenly; the pushing foot prioritises planting flat on the
# ground (z + orientation) over the G1's far lateral reach the X2 can't make.
_W_PLANTED = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
_W_PUSHING = np.array([0.6, 0.6, 1.5, 1.0, 1.0, 1.0])
_W_ORIENT = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])


def _pose_ik(model, data, ee_id, target_pos, target_quat, joints, weights,
             iters=600, lam=0.05, tol=3e-4) -> float:
  """Weighted 6-DoF damped-least-squares IK: move `ee_id` toward the target pose
  using only `joints` (root and others fixed). `weights` is a length-6 vector over
  [pos x,y,z, rot x,y,z] -- e.g. drop the lateral weight on the pushing foot so it
  prioritises planting flat on the ground over matching the G1's far lateral reach,
  or zero the position weights for an orientation-only target (the torso lean).
  Returns the final (unweighted) position error."""
  w = np.asarray(weights, dtype=np.float64)
  qadr, dadr, ranges = [], [], []
  for name in joints:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr.append(model.jnt_qposadr[j])
    dadr.append(model.jnt_dofadr[j])
    ranges.append(tuple(model.jnt_range[j]) if model.jnt_limited[j] else (-1e9, 1e9))
  jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
  neg, qerr, rot_err = np.zeros(4), np.zeros(4), np.zeros(3)
  pos_err = target_pos - data.xpos[ee_id]
  for _ in range(iters):
    mujoco.mj_forward(model, data)
    pos_err = target_pos - data.xpos[ee_id]
    mujoco.mju_negQuat(neg, data.xquat[ee_id])
    mujoco.mju_mulQuat(qerr, target_quat, neg)  # world-frame orientation error
    mujoco.mju_quat2Vel(rot_err, qerr, 1.0)
    err = w * np.concatenate([pos_err, rot_err])
    if np.linalg.norm(err) < tol:
      break
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
    J = w[:, None] * np.vstack([jacp[:, dadr], jacr[:, dadr]])
    dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(len(w)), err)
    for k, qa in enumerate(qadr):
      data.qpos[qa] = np.clip(data.qpos[qa] + dq[k], ranges[k][0], ranges[k][1])
  return float(np.linalg.norm(pos_err))


def _body_poses_in_board_frame(model, data, board_pos, board_quat) -> np.ndarray:
  """Return (num_bodies, 7) per-body [pos|quat] in the board frame, excluding world."""
  nb = model.nbody - 1
  body_pos = torch.tensor(data.xpos[1:], dtype=torch.float32)
  body_quat = torch.tensor(data.xquat[1:], dtype=torch.float32)  # MuJoCo xquat is wxyz
  bp = torch.tensor(board_pos, dtype=torch.float32).repeat(nb, 1)
  bq = torch.tensor(board_quat, dtype=torch.float32).repeat(nb, 1)
  pos_b, quat_b = subtract_frame_transforms(bp, bq, body_pos, body_quat)
  return torch.cat([pos_b, quat_b], dim=-1).numpy().astype(np.float32)


def main() -> None:
  board_pos = np.array(SKATEBOARD_INIT_KEYFRAME.pos, dtype=np.float64)
  board_quat = np.array([1.0, 0.0, 0.0, 0.0])

  # --- Source: the published G1 references + the G1 joint stances. ---
  g1_model = get_g1_spec().compile()
  g1_names = _hinge_joint_names(g1_model)
  g1_push_joints = {n: float(G1_PUSH_KEYFRAME.joint_pos.get(n, 0.0)) for n in g1_names}
  g1_steer_init = list(load_env_cfg("Mjlab-Skater-Flat-Unitree-G1").steer_init_pos)
  assert len(g1_steer_init) == len(g1_names)
  g1_steer_joints = dict(zip(g1_names, g1_steer_init))

  g1_push_ref = np.load("dataset/ref_pose/push_start_pose_b.npy")
  g1_steer_ref = np.load("dataset/ref_pose/steer_start_pose_b.npy")

  # G1 body order -> the ankle rows whose board positions the X2 feet must reach.
  g1_bodies = [
    mujoco.mj_id2name(g1_model, mujoco.mjtObj.mjOBJ_BODY, b)
    for b in range(1, g1_model.nbody)
  ]
  g1_ankle_row = {s: g1_bodies.index(f"{s}_ankle_roll_link") for s in ("left", "right")}
  g1_torso_row = g1_bodies.index("torso_link")

  # --- Target: the X2. Retarget root (pelvis row 0) + remapped joints, then IK
  # the torso to the G1 ref's lean and each ankle onto the G1's ankle pose. ---
  x2_model = get_x2_spec().compile()
  data = mujoco.MjData(x2_model)
  x2_ankle_id = {
    s: mujoco.mj_name2id(x2_model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
    for s in ("left", "right")
  }
  x2_torso_id = mujoco.mj_name2id(x2_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

  # root_dz lowers the inherited (G1) pelvis height: the X2's legs are ~10 cm
  # shorter, so for the push it must sit lower to span deck-to-ground and plant
  # the pushing foot. Facing (yaw) and xy are kept from the G1.
  sources = {
    "dataset/ref_pose/x2_push_start_pose_b.npy": (g1_push_ref, g1_push_joints, -0.06),
    "dataset/ref_pose/x2_steer_start_pose_b.npy": (g1_steer_ref, g1_steer_joints, 0.0),
  }
  for path, (g1_ref, g1_joints, root_dz) in sources.items():
    root_b = g1_ref[0]  # G1 pelvis pose in the board frame (carries the yaw)
    root_pos = board_pos + root_b[:3] + np.array([0.0, 0.0, root_dz])
    root_quat = root_b[3:].astype(np.float64)
    root_quat = root_quat / np.linalg.norm(root_quat)
    x2_joints = _remap_joints(g1_joints, x2_model)
    _set_qpos(x2_model, data, root_pos, root_quat, x2_joints)

    # Torso: match the G1 ref torso orientation via the waist joints so the X2
    # inherits the G1's forward push-lean (the authored joints leave it upright).
    torso_quat = g1_ref[g1_torso_row, 3:7].astype(np.float64)
    torso_quat = torso_quat / np.linalg.norm(torso_quat)
    _pose_ik(x2_model, data, x2_torso_id, data.xpos[x2_torso_id].copy(), torso_quat,
             _WAIST, _W_ORIENT)

    # Surfaces the X2 soles must rest on: the on-deck foot on the deck top
    # (board height + the deck-collision box half-thickness, 0.01), the pushing foot
    # on the ground (z=0).
    DECK_TOP = float(board_pos[2]) + 0.01
    GROUND = 0.0

    ik_err = []
    for side in ("left", "right"):
      on_deck = g1_ref[g1_ankle_row[side], 2] > 0.02
      surf = DECK_TOP if on_deck else GROUND
      target_w = (board_pos + g1_ref[g1_ankle_row[side], :3]).astype(np.float64)  # board is identity orientation
      target_quat = g1_ref[g1_ankle_row[side], 3:7].astype(np.float64)
      target_quat = target_quat / np.linalg.norm(target_quat)
      target_quat = _level_quat(target_quat)  # flat sole (parallel to deck/ground), keep heading
      weights = _W_PLANTED if on_deck else _W_PUSHING
      # The X2's legs are ~10 cm shorter than the G1's, so the pushing foot can't
      # reach the G1's far+low position; iterate harder/stiffer to extend it as far
      # toward the ground as the leg allows.
      iters, lam = (600, 0.05) if on_deck else (1500, 0.02)
      err = _pose_ik(x2_model, data, x2_ankle_id[side], target_w, target_quat,
                     _LEGS[side], weights, iters=iters, lam=lam)
      # Seat the SOLE on the surface: the IK targets the ANKLE, but the X2 sole sits
      # ~7cm below it, so shift the ankle target vertically until the lowest foot
      # collision sphere rests at the surface (+1mm clearance), re-IKing each time.
      # Removes the ~1cm deck/ground clipping the fixed ankle target left.
      for _ in range(8):
        mujoco.mj_forward(x2_model, data)
        dz = (surf + 0.001) - _lowest_foot_sphere_z(x2_model, data, side)
        if abs(dz) < 3e-4:
          break
        target_w = target_w + np.array([0.0, 0.0, dz])
        err = _pose_ik(x2_model, data, x2_ankle_id[side], target_w, target_quat,
                       _LEGS[side], weights, iters=iters, lam=lam)
      ik_err.append(err)
    mujoco.mj_forward(x2_model, data)
    ref = _body_poses_in_board_frame(x2_model, data, board_pos, board_quat)
    np.save(path, ref)
    yaw = np.degrees(2.0 * np.arctan2(root_quat[3], root_quat[0]))
    print(f"wrote {path}  shape={ref.shape}  pelvis_yaw={yaw:.0f}deg "
          f"foot_ik_err(L,R)={np.round(ik_err, 4)}  min_body_z={ref[:, 2].min():.3f}")
    joints = data.qpos[7:7 + (x2_model.nq - 7)]
    if "steer" in path:
      # The retargeted steer joint config — keep cfg.steer_init_pos in sync with it
      # (it is the joint target of the steer-phase steer_joint_pos reward).
      print("  steer_init_pos =", [round(float(v), 3) for v in joints])
    if "push" in path:
      # The on-board push pose — use it as the X2 spawn (PUSH_INIT_KEYFRAME) so the
      # robot starts on the board instead of splayed beside it.
      print("  PUSH spawn pos =", [round(float(v), 4) for v in root_pos])
      print("  PUSH spawn rot =", [round(float(v), 4) for v in root_quat])
      print("  PUSH joints =", [round(float(v), 3) for v in joints])


if __name__ == "__main__":
  main()
