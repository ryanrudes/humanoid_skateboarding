# husky_x2_deploy — HUSKY skater → AgiBot X2 hardware adapter

Runs a trained HUSKY skateboarding policy on a real **AgiBot X2 Ultra** through
the vendored, metadata-driven deploy host
([`../agibot_x2_ultra_deploy`](../agibot_x2_ultra_deploy), a git submodule). The
host is policy-agnostic but can't, by itself, express the skater observation;
this package supplies the missing skater-specific seams and leaves everything else
(IMU/joint plumbing, soft-start, E-stop, teleop, grouped joint-command publishing)
to the host.

## Why an adapter is needed

The host's generic `ObsBuilder` assembles a **single** observation frame from an
`actor_obs_spec`. The skater policy instead consumes (see `skater_env_cfg.py` /
`test_scene/sim.py`):

| term | dim (X2) | scale | source |
|------|------|-------|--------|
| `command` | 2 | ×[2.0, 1.0] | `[v, h]` = push speed + heading target |
| `heading` | 1 | ×1/π | base world yaw |
| `base_ang_vel` | 3 | ×0.25 | pelvis IMU gyro |
| `projected_gravity` | 3 | ×1 | from IMU quaternion |
| `joint_pos` | 31 | ×1 | `q − default` (default = PUSH_INIT) |
| `joint_vel` | 31 | ×0.05 | joint velocity |
| `actions` | 31 | ×1 | last raw action |
| `phase` | 1 | ×1 | cyclic stroke clock |

…stacked over a **5-frame history**, flattened **term-major** (per term: all 5
frames, oldest→newest) → `obs[515]` (X2) / `obs[395]` (G1). The host can't do the
history, the `heading` term, the scalar `phase` clock, or the `[v, h]` command
semantics — this package adds exactly those:

- **`SkateCommandProvider`** (`skate_command.py`) — emits `command`=[v, h] and the
  cyclic `phase`; `set_velocity(vx, vy, wz)` maps the operator triplet to the
  skater command so the host's `/cmd_vel` + gamepad teleop drive it unchanged.
- **`SkateObsAssembler`** (`skate_obs.py`) — keeps the host's per-term assembly
  (it already does `joint_pos − default`, projected-gravity, scaling) and adds the
  5-frame term-major history + the `heading` term.
- **`export_skater_deploy.py`** — writes the deploy `.onnx.metadata.json` sidecar.
- **`skate_deploy_node.py`** — a ROS 2 node subclassing the host node, swapping the
  two pieces above in. Action mapping (`q = a·scale + default`, joint-name → SDK
  group split) is the host's, unchanged.

## Install (offline, no robot)

The deploy host (`agibot-deploy`) is the runtime; install it and this adapter
editable into the mjlab venv:

```bash
# from the repo root
uv pip install -e deploy/agibot_x2_ultra_deploy -e deploy/husky_x2_deploy
```

(only `numpy` + `onnxruntime` are needed for the export/validate path; both ship
with the mjlab env.)

## Export the deploy artifact

A skater ONNX exported by training (`SkaterOnPolicyRunner`) already embeds
joint_names / gains / `default_joint_pos` / `action_scale` / `observation_names`.
The exporter copies it into `deploy/models/` and writes the sidecar
(`actor_obs_spec`, skater `command_spec`, `input_contract`, `control_dt`):

```bash
husky-skate-export --onnx logs/rsl_rl/x2_skater/<run>/<model>.onnx --name x2_skater_deploy.onnx
# -> deploy/models/x2_skater_deploy.onnx{,.metadata.json}
```

Robot-agnostic: it also handles the Unitree G1 (`ckpts/test.onnx`, 23 joints,
obs 395).

> `deploy/models/` is git-ignored (the ONNX duplicates a `logs/` checkpoint).

## Validate before any hardware run (offline gate)

```bash
husky-skate-validate --onnx logs/rsl_rl/x2_skater/<run>/<model>.onnx --steps 96
```

Drives a synthetic sensor trajectory through the adapter **and** a standalone
re-implementation of `test_scene/sim.py`'s observation, and asserts they agree
bit-for-bit (`max_obs_error`/`max_action_error` ≈ 0), then exercises the full host
path (`build → PolicyRunner.step → ActionMapper`). This proves the deploy obs/action
matches the validated standalone eval (and therefore the trained policy). Expected:

```
max_obs_error=0.000e+00
max_action_error=0.000e+00
PARITY OK
```

## Run on the robot

Build the host ROS package, then launch with this adapter's entry point + config.
**Keep the robot gantry-suspended for the whole bring-up** and follow the host's
safety checklist first — see
[`../agibot_x2_ultra_deploy/docs/HARDWARE_BRINGUP.md`](../agibot_x2_ultra_deploy/docs/HARDWARE_BRINGUP.md)
and the power-on→stand→policy sequence in the host README. The skater path mirrors
the host's "Deploy a Holosoma Locomotion Policy" worked example, with this config
and `command_provider.kind: skate`:

```bash
source /path/to/agibot_sdk/install/setup.bash
cd deploy/agibot_x2_ultra_deploy/ros2 && colcon build --packages-select agibot_deploy_ros && source install/setup.bash
uv pip install -e ../../husky_x2_deploy    # SkateCommandProvider/SkateObsAssembler on the host PYTHONPATH

# robot up & suspended in Position-Control Standing (JOINT_DEFAULT); then ARM the node:
python -m husky_x2_deploy.skate_deploy_node \
  --ros-args -p config_path:=$(pwd)/../../husky_x2_deploy/husky_x2_deploy/config/x2_skater.yaml \
             -p model_path:=$(pwd)/../../models/x2_skater_deploy.onnx
# in a second shell: hand joints to direct control, then start:
aima em stop-app mc
ros2 service call /agibot_deploy/start std_srvs/srv/Trigger {}
```

Steer with `/cmd_vel` (`geometry_msgs/Twist`): `linear.x` = push speed `[0, 1.5]`
m/s, `angular.z` = steer `[-1, 1]` → heading target `±π/4`; or the gamepad
(left-stick-Y = speed, right-stick-X = steer). Start at zero command, suspended,
and ramp up only once stable.

## Sim2real seams (read these)

- **`heading` is open-loop.** Hardware has no absolute world yaw. The default
  `control.heading_reference: start` yaw-normalizes to the heading at policy start
  (`heading(0)=0`); IMU yaw then drifts — same class as the host's documented
  odometry gap. Keep runs short / re-trigger to re-zero.
- **The robot must start on the skateboard**, in the PUSH_INIT push stance — the
  policy assumes a board underfoot and a foot planted to push. Soft-start ramps to
  PUSH_INIT; place the board first.
- **Steering mapping `[vx,vy,wz] → [v,h]`** is a deliberate choice (forward speed +
  heading target); `vy` is unused. Re-map in `skate_command.py` if your operator UX
  differs.
- **Gains/`default_joint_pos` are quantized** to the 3-decimal CSV the training
  exporter embeds (vs. the full-precision scene used by `sim.py`); the difference
  is < 1e-3 rad, well inside training joint-pos noise.
- **The X2 skater policy is an untuned scaffold** — the adapter is
  correct-by-construction (bit-exact vs. `sim.py`), but the policy itself still
  needs training/tuning before it will skate. The ROS node is import-checked here
  but only runs with the AgiBot SDK + robot.
- **Optional CSV debug** (`AGIBOT_DEPLOY_LOG_DIR`): the input-CSV header lists the
  single-frame term names while the row is the full 5-frame history — cosmetic only.
