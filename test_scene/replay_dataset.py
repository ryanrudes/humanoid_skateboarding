"""Replay HUSKY motion-capture clips in plain MuJoCo and render them to video.

The push dataset (``dataset/skate_push/human_push_*.npy``) is the human
push mocap that AMP trains against (see ``rsl_rl/utils/motion_loader_g1.py``).
Each clip is ``(T, 36)`` float, sampled at 50 Hz, laid out as::

    [ 0:3 ]  base position        (world frame, metres)
    [ 3:7 ]  base orientation     (quaternion, wxyz  -> MuJoCo native)
    [ 7:36]  29 DoF joint angles  (full Unitree G1 order, radians)

The robot model shipped here (``g1.xml``) is the **23-DoF** G1 (no wrists), so
the 29 mocap columns are gathered down to the model's joints *by name* -- this
is the same wrist-drop the AMP loader does (`[0:19] + [22:26]`).

This is a purely kinematic replay: each frame we write ``qpos`` and run
``mj_forward`` (no physics / actuators), then render an off-screen frame.

Run from the repo root (paths are relative)::

    uv run python test_scene/replay_dataset.py                       # all clips -> video/
    uv run python test_scene/replay_dataset.py --data dataset/skate_push/human_push_1.npy
    uv run python test_scene/replay_dataset.py --view                # interactive, no file
"""

import argparse
import glob
import os

# MuJoCo picks its GL backend at import time; force headless EGL for off-screen
# rendering unless the caller already chose one (e.g. MUJOCO_GL=glfw for --view).
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSET_ROOT = os.path.join(_REPO_ROOT, "src", "mjlab_husky", "asset_zoo", "robots")

DEFAULT_XML = os.path.join(_ASSET_ROOT, "skateboard", "xmls", "g1.xml")

# Convenience presets for ``--robot`` so callers needn't type long MJCF paths.
# Each maps to (robot MJCF, default clip directory).
ROBOT_PRESETS = {
    "g1": (DEFAULT_XML, "dataset/skate_push"),
    "x2": (
        os.path.join(_ASSET_ROOT, "agibot_x2", "xmls", "x2_ultra.xml"),
        "dataset/skate_push_x2",
    ),
}

# Canonical Unitree G1 29-DoF joint order -- the order the mocap columns 7:36
# follow. The 23-DoF model's joints are a subset of these names.
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

DATA_FPS = 50  # clips are sampled at 50 Hz (frame_duration = 1/50 in the loader)

# The 30 robot bodies in MJCF declaration order -- the order the per-body
# reference poses in dataset/ref_pose/*.npy follow (row i <-> this body).
ROBOT_BODY_ORDER = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
]

SKATEBOARD_XML = os.path.join(os.path.dirname(DEFAULT_XML), "skateboard.xml")


def build_model(xml_path: str, add_board: bool = False) -> mujoco.MjModel:
    """Load the bare robot and add a ground plane, skybox and lighting.

    If ``add_board`` is set, the skateboard MJCF is attached too (its deck root
    sits at z=0.10 with the wheels resting on the floor at z=0).
    """
    spec = mujoco.MjSpec.from_file(xml_path)

    if add_board:
        board = mujoco.MjSpec.from_file(SKATEBOARD_XML)
        spec.attach(board, prefix="board_", frame=spec.worldbody.add_frame())

    # Gradient skybox so the background isn't black.
    sky = spec.add_texture()
    sky.name = "skybox"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.3, 0.5, 0.7]
    sky.rgb2 = [0.0, 0.0, 0.0]
    sky.width = sky.height = 512

    # Checker material for the floor.
    tex = spec.add_texture()
    tex.name = "grid"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1 = [0.2, 0.3, 0.4]
    tex.rgb2 = [0.1, 0.15, 0.2]
    tex.width = tex.height = 300

    mat = spec.add_material()
    mat.name = "grid"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [4, 4]
    mat.reflectance = 0.1

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.material = "grid"

    # Lift ambient a touch so shadowed side of the robot stays readable.
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

    # Enlarge the off-screen framebuffer so high-res renders are allowed.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    return spec.compile()


def make_dof_gather(model: mujoco.MjModel, n_data_dof: int) -> list[int]:
    """Indices into a data frame's DoF block that fill the model's qpos joints.

    Maps the clip's joint columns onto the model's hinge joints by name, so a
    23-DoF model correctly drops the 6 wrist columns of a 29-DoF clip.
    """
    hinge_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if n_data_dof == len(G1_29DOF_JOINT_ORDER):
        col = {name: i for i, name in enumerate(G1_29DOF_JOINT_ORDER)}
        missing = [n for n in hinge_names if n not in col]
        if missing:
            raise ValueError(f"Model joints absent from G1 29-DoF order: {missing}")
        return [col[n] for n in hinge_names]
    if n_data_dof == len(hinge_names):
        return list(range(n_data_dof))  # already matches the model
    raise ValueError(
        f"Clip has {n_data_dof} DoF; model has {len(hinge_names)} hinge joints "
        f"and the canonical order has {len(G1_29DOF_JOINT_ORDER)}. Cannot map."
    )


def frame_camera(positions: np.ndarray, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    """A fixed camera framed to contain the whole base trajectory."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = positions.mean(axis=0)
    span = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    cam.distance = max(3.0, span * 1.3 + 2.0)
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def replay_clip(clip: np.ndarray, model, gather, args):
    """Yield rendered RGB frames (or drive the passive viewer when --view)."""
    data = mujoco.MjData(model)
    n_dof = len(gather)

    def set_frame(f):
        data.qpos[0:3] = f[0:3]
        data.qpos[3:7] = f[3:7]  # wxyz, MuJoCo native
        data.qpos[7:7 + n_dof] = f[7:7 + clip.shape[1] - 7][gather]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    if args.view:
        import time
        import mujoco.viewer as mjv

        with mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.distance = args.distance
            viewer.cam.azimuth = args.azimuth
            viewer.cam.elevation = args.elevation
            dt = 1.0 / (DATA_FPS * args.speed)
            while viewer.is_running():
                for f in clip:
                    if not viewer.is_running():
                        break
                    t0 = time.time()
                    set_frame(f)
                    viewer.cam.lookat[:] = data.qpos[0:3]
                    viewer.sync()
                    sleep = dt - (time.time() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
                if not args.loop:
                    break
        return None

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    fixed_cam = None
    if args.camera == "fixed":
        fixed_cam = frame_camera(clip[:, 0:3], args.azimuth, args.elevation)
    frames = []
    try:
        for f in tqdm(clip, desc="rendering", leave=False):
            set_frame(f)
            if fixed_cam is not None:
                cam = fixed_cam
            else:  # follow: keep the base centred, world scrolls past
                cam = mujoco.MjvCamera()
                cam.lookat[:] = data.qpos[0:3]
                cam.distance = args.distance
                cam.azimuth = args.azimuth
                cam.elevation = args.elevation
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def write_video(frames, path, fps):
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", macro_block_size=16, pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def resolve_clips(data_arg: str) -> list[str]:
    if os.path.isdir(data_arg):
        return sorted(glob.glob(os.path.join(data_arg, "*.npy")))
    if any(ch in data_arg for ch in "*?["):
        return sorted(glob.glob(data_arg))
    return [data_arg]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=sorted(ROBOT_PRESETS), default=None,
                   help="convenience preset that selects the robot MJCF and default clip dir "
                        "(g1 -> dataset/skate_push, x2 -> dataset/skate_push_x2). "
                        "Overridden by explicit --xml/--data.")
    p.add_argument("--data", default=None,
                   help="npy file, directory, or glob of mocap clips (default: per --robot, else dataset/skate_push)")
    p.add_argument("--xml", default=None, help="robot MJCF (default: per --robot, else bundled g1.xml)")
    p.add_argument("--out-dir", default="video/dataset_replay", help="where to write mp4s")
    p.add_argument("--fps", type=int, default=DATA_FPS, help="output video fps (default: 50 = data rate)")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--camera", choices=["follow", "fixed"], default="follow",
                   help="follow: track the robot; fixed: frame the whole trajectory")
    p.add_argument("--distance", type=float, default=3.5, help="follow-camera distance")
    p.add_argument("--azimuth", type=float, default=210.0)
    p.add_argument("--elevation", type=float, default=-10.0)
    p.add_argument("--view", action="store_true",
                   help="open an interactive viewer instead of writing video (needs a display)")
    p.add_argument("--loop", action="store_true", help="loop playback in --view mode")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed in --view mode")
    args = p.parse_args()

    # Resolve robot preset / explicit overrides for the MJCF and clip directory.
    preset_xml, preset_data = ROBOT_PRESETS.get(args.robot, (DEFAULT_XML, "dataset/skate_push"))
    if args.xml is None:
        args.xml = preset_xml
    if args.data is None:
        args.data = preset_data

    if args.view:
        os.environ["MUJOCO_GL"] = "glfw"

    clips = resolve_clips(args.data)
    if not clips:
        raise SystemExit(f"No .npy clips found at: {args.data}")

    print(f"Building model from {args.xml}")
    model = build_model(args.xml)

    for clip_path in clips:
        clip = np.load(clip_path, allow_pickle=True).astype(np.float64)
        if clip.ndim != 2 or clip.shape[1] < 8:
            print(f"Skipping {clip_path}: unexpected shape {clip.shape}")
            continue
        n_dof = clip.shape[1] - 7
        gather = make_dof_gather(model, n_dof)
        name = os.path.splitext(os.path.basename(clip_path))[0]
        print(f"[{name}] {clip.shape[0]} frames @ {DATA_FPS} Hz "
              f"({clip.shape[0] / DATA_FPS:.1f}s), {n_dof} DoF -> {len(gather)} model joints")

        frames = replay_clip(clip, model, gather, args)
        if frames is None:  # --view
            continue
        out_path = os.path.join(args.out_dir, f"{name}.mp4")
        write_video(frames, out_path, args.fps)
        print(f"[{name}] wrote {out_path}  ({len(frames)} frames, {args.width}x{args.height})")


if __name__ == "__main__":
    main()
