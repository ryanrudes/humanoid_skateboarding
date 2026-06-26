# deploy/ — real-hardware deployment of HUSKY skater policies

Two pieces:

- **`agibot_x2_ultra_deploy/`** — git **submodule** of the upstream
  [AgiBot Deploy Host](https://github.com/fxxie2/agibot_x2_ultra_deploy): a
  metadata-driven, policy-agnostic ONNX runtime (Python + C++ + ROS 2) that talks
  to the AgiBot X2 SDK HAL joint/IMU topics, with soft-start / E-stop / teleop /
  fault-hold safety built in. **Vendored read-only** — do not edit; bump the pin
  to take upstream changes.
- **`husky_x2_deploy/`** — **our adapter** (this repo). The skater policy's
  observation (5-frame history, a `heading` term, a scalar `phase` clock, and a
  `[v, h]` command) is not expressible by the host's generic obs builder, so this
  package supplies exactly those seams and reuses everything else from the host.

The submodule is read-only vendor code; all skater-specific logic lives in
`husky_x2_deploy/` and imports `agibot_deploy` as a library.

## Quickstart

```bash
# 1. fetch the submodule (first checkout only)
git submodule update --init deploy/agibot_x2_ultra_deploy

# 2. install host + adapter into the mjlab venv
uv pip install -e deploy/agibot_x2_ultra_deploy -e deploy/husky_x2_deploy

# 3. export a trained skater ONNX to a deploy artifact (ONNX + metadata sidecar)
husky-skate-export --onnx logs/rsl_rl/x2_skater/<run>/<model>.onnx --name x2_skater_deploy.onnx

# 4. offline correctness gate (no robot): deploy obs/action == test_scene/sim.py, bit-exact
husky-skate-validate --onnx logs/rsl_rl/x2_skater/<run>/<model>.onnx
```

On-robot bring-up, steering, and the sim2real seams are in
[`husky_x2_deploy/README.md`](husky_x2_deploy/README.md). Read the host's
[`docs/HARDWARE_BRINGUP.md`](agibot_x2_ultra_deploy/docs/HARDWARE_BRINGUP.md)
before any robot run.
