"""Visualize the skater policy on flat ground WITHOUT a skateboard.

This mirrors ``test_scene/sim.py`` but removes the skateboard from the scene so
you can see what the policy does when the robot is simply standing on flat
ground. This works because the *actor* observation is purely proprioceptive
(velocity command, heading, base angular velocity, projected gravity, joint
pos/vel, last action, gait phase) and never references the board -- only the
critic sees the board, and the critic is not used at inference. The exported
ONNX policy therefore runs here unchanged.

The board-free MuJoCo model is built at runtime from the same
``mjlab_scene.xml`` used by ``sim.py``: the skateboard body, its actuators and
the truck/tilt coupling constraints are stripped out via ``mujoco.MjSpec``.
Removing them does not change the robot's qpos/qvel/ctrl ordering (the robot is
entirely ahead of the board in every layout), so the observation construction
and the action ``reindex_list`` are identical to ``sim.py``.

Two modes:

* Interactive (default): opens a MuJoCo passive viewer; needs a display.
  Controls: Up/Down speed, Left/Right heading, ``5`` reset commands,
  Enter reset robot, Space pause.

* Headless video (``--video out.mp4``): renders offscreen with no display and
  writes a high-resolution MP4. Uses EGL by default (override with the
  ``MUJOCO_GL`` env var, e.g. ``MUJOCO_GL=osmesa``). The robot follows a fixed
  velocity command (``--v0``/``--h0``) since there is no keyboard input.

Examples:
  # Interactive (needs a display)
  python test_scene/sim_no_board.py --policy ckpts/test.onnx

  # Headless 1080p video, robot standing then pushed forward at 0.8 m/s
  python test_scene/sim_no_board.py --policy ckpts/test.onnx \
      --video out.mp4 --duration 10 --width 1920 --height 1080 --v0 0.8
"""

import argparse
import math
import os
import sys
import threading
import time
from collections import deque

# MuJoCo selects its GL backend the first time it is used, based on MUJOCO_GL.
# For headless video (--video) we must pick EGL *before* importing mujoco;
# otherwise it falls back to GLFW and fails with no display. Interactive mode is
# left untouched so the windowed viewer keeps using the default (GLFW) backend.
# `setdefault` means an explicit MUJOCO_GL (e.g. osmesa) is still respected.
if any(a == "--video" or a.startswith("--video=") for a in sys.argv):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from rich import print
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


def build_robot_only_model(xml_file: str) -> mujoco.MjModel:
    """Compile ``xml_file`` with the skateboard removed, leaving just the G1.

    The robot keeps its original joint/actuator/sensor ordering, so the rest of
    this script can reuse the exact observation/action layout from ``sim.py``.
    """
    spec = mujoco.MjSpec.from_file(xml_file)

    # Keyframes encode the full (robot + board) qpos/ctrl; drop them since the
    # dimensions change once the board is gone (we initialize in Python anyway).
    for key in list(spec.keys):
        spec.delete(key)
    # Truck/tilt coupling equalities reference board joints -> remove them.
    for eq in list(spec.equalities):
        refs = f"{getattr(eq, 'name1', '')} {getattr(eq, 'name2', '')}"
        if "skateboard" in refs:
            spec.delete(eq)
    # Board actuators (the trailing 7) reference board joints -> remove them.
    for act in list(spec.actuators):
        if act.name.startswith("skateboard/"):
            spec.delete(act)
    # Finally the skateboard body subtree itself.
    board = spec.body("skateboard/skateboard_deck")
    if board is not None:
        spec.delete(board)

    return spec.compile()


# ---------------------------------------------------------------------------
# Interactive keyboard control (lazy: pynput is only imported when a display is
# available, so the module still imports for headless video rendering).
# ---------------------------------------------------------------------------
keyboard = None  # populated by start_keyboard_listener() in interactive mode.

reset_flag = False
pause_flag = False
V_MIN, V_MAX = 0.0, 1.5
H_MIN, H_MAX = -math.pi / 4, math.pi / 4
# Default to a standing command (no forward push) so we see whether the policy
# can simply hold a stand on flat ground. Press Up to command forward speed.
v = 0.0
h = 0.0


def wrap_to_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def on_press(key):
    global v, h, reset_flag, pause_flag
    try:
        if key == keyboard.Key.up:
            v = round(min(v + 0.1, V_MAX), 1)
            print("v =", v, "h =", round(h, 3), "(rad)")
        elif key == keyboard.Key.down:
            v = round(max(v - 0.1, V_MIN), 1)
            print("v =", v, "h =", round(h, 3), "(rad)")
        elif key == keyboard.Key.left:
            h = round(max(h + 0.1, H_MIN), 2)
            print("v =", v, "h =", round(h, 3), "(rad)")
        elif key == keyboard.Key.right:
            h = round(min(h - 0.1, H_MAX), 2)
            print("v =", v, "h =", round(h, 3), "(rad)")
        elif key == keyboard.Key.enter:
            reset_flag = True
            print("Reset flag set! Simulation will reset...")
        elif key == keyboard.Key.space:
            pause_flag = not pause_flag
            if pause_flag:
                print("Simulation PAUSED. Press SPACE to resume.")
            else:
                print("Simulation RESUMED.")
        elif hasattr(key, "char") and key.char == "5":
            v = 0.0
            h = 0.0
            print("Commands reset: v = 0.0, h = 0.0")
    except AttributeError:
        pass


def start_keyboard_listener():
    """Import pynput lazily and start the key listener thread (interactive only)."""
    global keyboard

    def _listen():
        global keyboard
        from pynput import keyboard as _kb
        keyboard = _kb
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    thread = threading.Thread(target=_listen, daemon=True)
    thread.start()
    return thread


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def quat_apply_np(quat, vec):
    quat = np.asarray(quat)
    vec = np.asarray(vec)
    orig_shape = vec.shape

    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)

    w = q[:, 0]
    qvec = q[:, 1:4]

    t = 2 * np.cross(qvec, v)
    v_rot = v + (w[:, None] * t) + np.cross(qvec, t)
    v_rot = v_rot.reshape(orig_shape)
    return v_rot


# Maps policy-ordered joint vectors into the MuJoCo actuator/ctrl order.
reindex_list = [15, 16, 17, 18, 19, 20, 21, 22, 0, 2, 6, 8, 12, 1, 3, 7, 9, 13, 14, 4, 5, 10, 11]

# Policy reference pose (the "skating stance"; zero action maps to this). Same
# as sim.py's robot_default_dof_pos. Observations are computed relative to it,
# so it stays fixed regardless of how the robot is initialized.
ROBOT_DEFAULT_DOF_POS = np.array([
    0.0, 0.0, 0.0, 0.23, -0.20, 0.0,
    -0.7, 0.0, 0.0, 1.17, -0.45, 0.0,
    0.0, 0.0, 0.0,
    -0.03, 0.45, -0.21, 1.32,
    -0.7, -0.845, 0.83, 1.19,
], dtype=np.float32)

# A symmetric two-foot standing pose (policy joint order: left leg, right leg,
# waist yaw/roll/pitch, left arm, right arm). Slight knee bend for stability.
ROBOT_STAND_DOF_POS = np.array([
    -0.15, 0.0, 0.0, 0.30, -0.15, 0.0,   # left leg
    -0.15, 0.0, 0.0, 0.30, -0.15, 0.0,   # right leg
    0.0, 0.0, 0.0,                        # waist yaw, roll, pitch
    0.2, 0.2, 0.0, 0.9,                   # left arm
    0.2, -0.2, 0.0, 0.9,                  # right arm
], dtype=np.float32)


class RealTimePolicyController:
    def __init__(self,
                 xml_file,
                 policy_path,
                 device='cuda',
                 policy_frequency=50,
                 init_pose='stand',
                 settle_time=0.4,
                 ):

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        # Build a board-free model from the same scene used by sim.py.
        self.model = build_robot_only_model(xml_file)
        self.model.opt.timestep = 0.005
        self.model.opt.iterations = 10
        self.model.opt.ls_iterations = 20
        self.model.opt.ccd_iterations = 50

        self.data = mujoco.MjData(self.model)
        self.viewer = None  # created lazily by run() (interactive mode only).

        self.num_actions = 23
        self.sim_dt = 0.005
        self.cycle_time = 6
        self.step_dt = 1 / policy_frequency
        self.sim_decimation = int(1 / (policy_frequency * self.sim_dt))
        self.settle_steps = int(round(settle_time / self.sim_dt))

        print(f"sim_decimation: {self.sim_decimation}")

        self.robot_default_dof_pos = ROBOT_DEFAULT_DOF_POS.copy()
        if init_pose == 'stand':
            self.robot_init_dof_pos = ROBOT_STAND_DOF_POS.copy()
        elif init_pose == 'default':
            self.robot_init_dof_pos = ROBOT_DEFAULT_DOF_POS.copy()
        else:
            raise ValueError(f"Unknown init_pose: {init_pose!r} (expected 'stand' or 'default')")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.action_scale = np.array([
            0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
            0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
            0.5475, 0.4386, 0.4386,
            0.4386, 0.4386, 0.4386, 0.4386,
            0.4386, 0.4386, 0.4386, 0.4386,
        ])

        self.n_obs_single = 3 + 3 + 3 + 3 * 23 + 1
        self.history_len = 5
        self.total_obs_size = self.n_obs_single * (self.history_len)

        self.obs_block_dims = [2, 1, 3, 3, 23, 23, 23, 1]
        self.obs_block_starts = np.cumsum([0] + self.obs_block_dims[:-1])

        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

        # Cache the IMU gyro sensor address (used for the base ang vel observation).
        gyro_id = self.model.sensor("robot/imu_ang_vel").id
        self._gyro_adr = self.model.sensor_adr[gyro_id]
        self._gyro_dim = self.model.sensor_dim[gyro_id]

        self._pelvis_body_id = self.model.body("robot/pelvis").id

        # Settle the chosen pose onto the ground once and cache the resulting
        # qpos so every reset restores an identical, stable standing state.
        self.init_qpos = self._compute_init_qpos()

    def _make_qpos(self, root_z: float) -> np.ndarray:
        """Robot-only qpos: root pos (3) + root quat (4) + 23 dof (policy order)."""
        return np.concatenate([
            np.array([0.0, 0.0, root_z], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            self.robot_init_dof_pos,
        ])

    def _lowest_collision_point_z(self) -> float:
        """World-z of the lowest collision-geom bound in the current state."""
        contype = self.model.geom_contype
        conaffinity = self.model.geom_conaffinity
        lowest = np.inf
        for g in range(self.model.ngeom):
            if contype[g] == 0 and conaffinity[g] == 0:
                continue
            z = self.data.geom_xpos[g, 2] - self.model.geom_rbound[g]
            lowest = min(lowest, z)
        return lowest

    def _compute_init_qpos(self) -> np.ndarray:
        """Place the pose on the ground, let it settle, return the settled qpos."""
        mujoco.mj_resetData(self.model, self.data)

        # Start at a safe height, then drop so the lowest collision point rests
        # just above ground (the bounding-sphere estimate is conservative; the
        # short settle below removes any residual gap).
        self.data.qpos[:] = self._make_qpos(root_z=0.9)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        drop = self._lowest_collision_point_z()
        self.data.qpos[:] = self._make_qpos(root_z=0.9 - drop + 0.01)
        self.data.qvel[:] = 0.0

        # Hold the init pose with the position servos while it settles.
        self.data.ctrl[:] = self.robot_init_dof_pos[reindex_list]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)

        settled = self.data.qpos.copy()
        settled[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # keep upright facing +x
        return settled

    def reset_sim(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """Reset robot to the cached standing pose."""
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.robot_init_dof_pos[reindex_list]
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.proprio_history_buf.clear()
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        n_robot_dof = self.num_actions

        robot_quat = self.data.qpos[3:7]
        robot_dof_pos = self.data.qpos[7:7 + n_robot_dof]
        robot_ang_vel = self.data.qvel[3:6]
        robot_dof_vel = self.data.qvel[6:6 + n_robot_dof]

        return robot_dof_pos, robot_dof_vel, robot_quat, robot_ang_vel

    def _build_obs(self, phase: float, v_cmd: float, h_cmd: float) -> torch.Tensor:
        """Construct the 395-dim proprioceptive observation for the policy."""
        robot_dof_pos, robot_dof_vel, robot_quat, _ = self.extract_data()

        gravity_orientation = get_gravity_orientation(robot_quat)
        sensor_ang_vel = self.data.sensordata[self._gyro_adr: self._gyro_adr + self._gyro_dim]
        forward_w = quat_apply_np(robot_quat, np.array([1, 0, 0]))
        heading = np.array([np.arctan2(forward_w[1], forward_w[0])])

        obs_proprio = np.concatenate([
            np.array([v_cmd, h_cmd], dtype=np.float32) * [2.0, 1.0],
            heading * 1.0 / math.pi,
            sensor_ang_vel * 0.25,
            gravity_orientation,
            (robot_dof_pos - self.robot_default_dof_pos),
            robot_dof_vel * 0.05,
            self.last_action,
            np.array([phase], dtype=np.float32),
        ])

        self.proprio_history_buf.append(obs_proprio)
        history_array = np.array(self.proprio_history_buf)

        obs_buf_parts = []
        for start, dim in zip(self.obs_block_starts, self.obs_block_dims):
            obs_buf_parts.append(history_array[:, start:start + dim].flatten())
        obs_buf = np.concatenate(obs_buf_parts)

        return torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)

    def _compute_ctrl(self, phase: float, v_cmd: float, h_cmd: float) -> np.ndarray:
        """Run the policy for one control step; return the ctrl vector (actuator order)."""
        obs_tensor = self._build_obs(phase, v_cmd, h_cmd)
        with torch.no_grad():
            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
        self.last_action = raw_action
        pd_target_robot = raw_action * self.action_scale + self.robot_default_dof_pos
        return pd_target_robot[reindex_list]

    def run(self, sim_duration: float = 30.0):
        """Interactive playback in a MuJoCo passive viewer (requires a display)."""
        global reset_flag, pause_flag, v, h
        import mujoco.viewer as mjv

        print("Starting Skater simulation (no skateboard)...")

        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = 210.0
        self.viewer.cam.elevation = -10.0

        self.reset_sim()
        self.reset()

        steps = int(sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating Skater (no board)...")

        phase_counter = 0

        try:
            for _ in pbar:
                if not self.viewer.is_running():
                    print("Viewer closed, stopping simulation.")
                    break
                if reset_flag:
                    self.reset()
                    reset_flag = False
                    phase_counter = 0
                    print("Simulation RESET!")
                if pause_flag:
                    time.sleep(0.01)
                    continue
                t_start = time.time()

                phase_counter += 1
                phase = (phase_counter * self.step_dt / self.cycle_time) % 1.0

                ctrl = self._compute_ctrl(phase, v, h)

                viewer_closed = False
                for _ in range(self.sim_decimation):
                    if not self.viewer.is_running():
                        viewer_closed = True
                        break
                    self.data.ctrl[:] = ctrl
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.cam.lookat = self.data.xpos[self._pelvis_body_id]
                    self.viewer.sync()
                if viewer_closed:
                    break

                dt = self.model.opt.timestep * self.sim_decimation
                sleep = dt - (time.time() - t_start)
                if sleep > 0:
                    time.sleep(sleep)

        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.viewer:
                self.viewer.close()
            print("Simulation finished.")

    def record(self,
               output_path: str,
               duration: float = 10.0,
               fps: int | None = None,
               width: int = 1920,
               height: int = 1080,
               v_cmd: float = 0.0,
               h_cmd: float = 0.0,
               cam_distance: float = 4.0,
               cam_azimuth: float = 210.0,
               cam_elevation: float = -10.0,
               camera_name: str | None = None):
        """Headless offscreen rendering to a high-resolution MP4 (no display)."""
        import imageio.v2 as imageio

        # Pick a headless GL backend unless the user already chose one.
        os.environ.setdefault("MUJOCO_GL", "egl")

        # The offscreen framebuffer must be at least the requested resolution.
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)

        if fps is None:
            fps = int(round(1.0 / self.step_dt))  # one frame per control step.

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        if camera_name:
            cam = camera_name
        else:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = cam_distance
            cam.azimuth = cam_azimuth
            cam.elevation = cam_elevation

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,  # honor the exact requested resolution.
            ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )

        print(f"Rendering {duration:.1f}s @ {width}x{height}, {fps} fps "
              f"(MUJOCO_GL={os.environ['MUJOCO_GL']}, v={v_cmd}, h={h_cmd}) -> {output_path}")

        self.reset_sim()
        self.reset()

        n_steps = int(round(duration / self.step_dt))
        try:
            for step in tqdm(range(n_steps), desc="Rendering (no board)..."):
                phase = ((step + 1) * self.step_dt / self.cycle_time) % 1.0
                ctrl = self._compute_ctrl(phase, v_cmd, h_cmd)
                for _ in range(self.sim_decimation):
                    self.data.ctrl[:] = ctrl
                    mujoco.mj_step(self.model, self.data)

                if isinstance(cam, mujoco.MjvCamera):
                    cam.lookat[:] = self.data.xpos[self._pelvis_body_id]
                renderer.update_scene(self.data, cam)
                writer.append_data(renderer.render())
        finally:
            writer.close()
            renderer.close()
        print(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run skater policy on flat ground without a skateboard')
    parser.add_argument('--xml', type=str, default='mjlab_scene.xml',
                        help='Path to the (board-containing) MuJoCo scene XML; the board is stripped at runtime')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to skater ONNX policy file')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument("--policy_frequency", help="Policy frequency", default=50, type=int)
    parser.add_argument('--init_pose', type=str, default='stand', choices=['stand', 'default'],
                        help="Initial pose: 'stand' (symmetric two-foot stance) or "
                             "'default' (the policy's reference skating stance)")
    parser.add_argument('--settle_time', type=float, default=0.4,
                        help='Seconds to let the initial pose settle onto the ground before the policy runs')
    parser.add_argument('--v0', type=float, default=None, help='Initial/forward speed command (overrides default 0.0)')
    parser.add_argument('--h0', type=float, default=None, help='Initial/heading command in radians (overrides default 0.0)')

    # Headless video recording.
    parser.add_argument('--video', type=str, default=None,
                        help='Render a high-res video to this path (headless, no display needed) instead of opening the viewer')
    parser.add_argument('--duration', type=float, default=10.0, help='Video duration in seconds (record mode)')
    parser.add_argument('--fps', type=int, default=None,
                        help='Video framerate (record mode); default = policy_frequency for real-time playback')
    parser.add_argument('--width', type=int, default=1920, help='Video width in pixels (record mode)')
    parser.add_argument('--height', type=int, default=1080, help='Video height in pixels (record mode)')
    parser.add_argument('--cam_distance', type=float, default=4.0, help='Follow-camera distance (record mode)')
    parser.add_argument('--cam_azimuth', type=float, default=210.0, help='Follow-camera azimuth deg (record mode)')
    parser.add_argument('--cam_elevation', type=float, default=-10.0, help='Follow-camera elevation deg (record mode)')
    parser.add_argument('--camera', type=str, default=None,
                        help='Use a named model camera (e.g. robot/tracking) instead of the free follow camera (record mode)')
    args = parser.parse_args()

    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return

    if not os.path.exists(args.xml):
        print(f"Error: XML file {args.xml} does not exist")
        return

    global v, h
    if args.v0 is not None:
        v = args.v0
    if args.h0 is not None:
        h = args.h0

    print(f"Starting skater (no-board) {'video' if args.video else 'interactive'} controller...")
    print(f"  XML file: {args.xml}")
    print(f"  Policy file: {args.policy}")
    print(f"  Device: {args.device}")
    print(f"  Init pose: {args.init_pose}")

    controller = RealTimePolicyController(
        xml_file=args.xml,
        policy_path=args.policy,
        device=args.device,
        policy_frequency=args.policy_frequency,
        init_pose=args.init_pose,
        settle_time=args.settle_time,
    )

    if args.video:
        controller.record(
            output_path=args.video,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            v_cmd=v,
            h_cmd=h,
            cam_distance=args.cam_distance,
            cam_azimuth=args.cam_azimuth,
            cam_elevation=args.cam_elevation,
            camera_name=args.camera,
        )
    else:
        start_keyboard_listener()
        controller.run()


if __name__ == "__main__":
    main()
