"""Visualize the HUSKY reference *poses* (dataset/ref_pose/*.npy) in MuJoCo.

Unlike the push clips (see ``replay_dataset.py``), these files are not motion
trajectories -- each is a single target whole-body pose used by the transition
rewards. Layout is ``(30, 7)``: one row per robot body (pelvis ... wrists, in
MJCF order), each ``[pos(3), quat wxyz(4)]`` expressed in the **skateboard
body frame** (the ``_b`` suffix; see ``G1SkaterManagerBasedRlEnv``).

Because we only have *body* poses (not joint angles) we can't drive the robot
through ``qpos``. Instead we write each body's world transform straight into
``mjData`` (``xpos``/``xquat`` plus the per-geom ``geom_xpos``/``geom_xmat``)
and render without calling forward kinematics -- so the actual G1 meshes are
drawn in the captured pose. The skateboard is shown at the frame origin for
context, since the poses are board-relative.

Output is an orbiting video per pose (the natural "video" of a static pose);
pass ``--still`` for a single PNG instead.

Run from the repo root::

    uv run python test_scene/view_ref_pose.py                 # both poses -> video/ref_pose/
    uv run python test_scene/view_ref_pose.py --still
    uv run python test_scene/view_ref_pose.py --data dataset/ref_pose/steer_start_pose_b.npy --view
"""

import argparse
import glob
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from tqdm import tqdm

from replay_dataset import DEFAULT_XML, ROBOT_PRESETS, build_model, write_video

BOARD_ROOT_BODY = "board_skateboard_deck"


def quat2mat(quat: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.ascontiguousarray(quat, dtype=np.float64))
    return m.reshape(3, 3)


def pose_robot(model, data, ref, board_pos, board_quat, body_ids):
    """Write the 30 board-frame body poses into mjData as world transforms.

    world = board_pose ∘ body_b  (inverse of the subtract_frame_transforms the
    env uses to express bodies in the board frame).
    """
    Rb = quat2mat(board_quat)
    for row, bid in enumerate(body_ids):
        pos_b = ref[row, :3].astype(np.float64)
        quat_b = ref[row, 3:].astype(np.float64)
        quat_b = quat_b / np.linalg.norm(quat_b)
        world_pos = board_pos + Rb @ pos_b
        world_quat = np.zeros(4)
        mujoco.mju_mulQuat(world_quat, board_quat, quat_b)
        data.xpos[bid] = world_pos
        data.xquat[bid] = world_quat
        data.xmat[bid] = quat2mat(world_quat).reshape(9)

    # The renderer reads geom_xpos/geom_xmat directly, so recompute them from
    # the bodies we just moved (leaving floor + skateboard geoms as forwarded).
    body_id_set = set(body_ids)
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if b not in body_id_set:
            continue
        R = data.xmat[b].reshape(3, 3)
        data.geom_xpos[g] = data.xpos[b] + R @ model.geom_pos[g]
        data.geom_xmat[g] = (R @ quat2mat(model.geom_quat[g])).reshape(9)


def render_orbit(model, data, args):
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, args.lookat_z]
    cam.distance = args.distance
    cam.elevation = args.elevation
    frames = []
    try:
        if args.still:
            cam.azimuth = args.azimuth
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        else:
            n = max(1, int(round(args.seconds * args.fps)))
            for k in tqdm(range(n), desc="orbit", leave=False):
                cam.azimuth = args.azimuth + 360.0 * k / n
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def view_interactive(model, data, args):
    import time
    import mujoco.viewer as mjv

    with mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, args.lookat_z]
        viewer.cam.distance = args.distance
        viewer.cam.elevation = args.elevation
        viewer.cam.azimuth = args.azimuth
        while viewer.is_running():
            viewer.cam.azimuth = (viewer.cam.azimuth + 0.2) % 360.0
            viewer.sync()
            time.sleep(1.0 / 60.0)


def resolve_files(data_arg: str) -> list[str]:
    if os.path.isdir(data_arg):
        return sorted(glob.glob(os.path.join(data_arg, "*.npy")))
    if any(ch in data_arg for ch in "*?["):
        return sorted(glob.glob(data_arg))
    return [data_arg]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=sorted(ROBOT_PRESETS), default=None,
                   help="preset selecting the robot MJCF and default pose glob (g1, x2). "
                        "Overridden by explicit --xml/--data.")
    p.add_argument("--data", default=None,
                   help="npy pose file, directory, or glob (default: per --robot, else dataset/ref_pose)")
    p.add_argument("--xml", default=None)
    p.add_argument("--out-dir", default="video/ref_pose")
    p.add_argument("--no-board", action="store_true", help="hide the skateboard")
    p.add_argument("--still", action="store_true", help="render one PNG instead of an orbit video")
    p.add_argument("--view", action="store_true", help="open an interactive auto-orbiting viewer")
    p.add_argument("--seconds", type=float, default=6.0, help="orbit duration (one full revolution)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--distance", type=float, default=2.4)
    p.add_argument("--azimuth", type=float, default=210.0, help="start azimuth (and the angle for --still)")
    p.add_argument("--elevation", type=float, default=-15.0)
    p.add_argument("--lookat-z", type=float, default=0.5, help="height the camera points at")
    args = p.parse_args()

    # Resolve robot preset. The G1 and X2 reference poses share dataset/ref_pose/
    # (X2 files are prefixed x2_), so default --data to a robot-specific glob.
    preset_xml = ROBOT_PRESETS.get(args.robot, (DEFAULT_XML, None))[0]
    if args.xml is None:
        args.xml = preset_xml
    if args.data is None:
        args.data = (
            "dataset/ref_pose/x2_*.npy" if args.robot == "x2" else "dataset/ref_pose"
        )

    if args.view:
        os.environ["MUJOCO_GL"] = "glfw"

    files = resolve_files(args.data)
    if not files:
        raise SystemExit(f"No .npy pose files found at: {args.data}")

    add_board = not args.no_board
    print(f"Building model from {args.xml}" + (" + skateboard" if add_board else ""))
    model = build_model(args.xml, add_board=add_board)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # seats the board at z=0.10 and fills floor/board geoms

    if add_board:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOARD_ROOT_BODY)
        board_pos = data.xpos[bid].copy()
        board_quat = data.xquat[bid].copy()
    else:
        board_pos = np.array([0.0, 0.0, 0.10])
        board_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # Reference-pose rows follow the robot entity's body order == MJCF declaration
    # order (excluding the world body and the attached skateboard), so derive the
    # body ids straight from the model -- works for any robot (G1: 30, X2: 34).
    body_ids = [
        b for b in range(1, model.nbody)
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("board_")
    ]

    for pose_path in files:
        ref = np.load(pose_path, allow_pickle=True)
        if ref.shape != (len(body_ids), 7):
            print(f"Skipping {pose_path}: expected {(len(body_ids), 7)}, got {ref.shape}")
            continue
        name = os.path.splitext(os.path.basename(pose_path))[0]
        pose_robot(model, data, ref, board_pos, board_quat, body_ids)

        if args.view:
            print(f"[{name}] interactive view (close window for next pose)")
            view_interactive(model, data, args)
            continue

        frames = render_orbit(model, data, args)
        os.makedirs(args.out_dir, exist_ok=True)
        if args.still:
            import imageio.v2 as imageio
            out = os.path.join(args.out_dir, f"{name}.png")
            imageio.imwrite(out, frames[0])
        else:
            out = os.path.join(args.out_dir, f"{name}.mp4")
            write_video(frames, out, args.fps)
        print(f"[{name}] wrote {out}")


if __name__ == "__main__":
    main()
