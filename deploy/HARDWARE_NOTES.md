# X2 skater — hardware deployment notes

Practical watch-outs for running the skater policy on the real AgiBot X2, and a
symptom → likely-cause guide. The obs/action math is offline-verified bit-exact vs
`test_scene/sim.py` (`husky-skate-validate`); everything below is about the parts that
can only be checked on the robot.

> Status reminder: the X2 skater **policy** is still the untuned scaffold — it does not
> skate yet. The adapter is correct-by-construction, but until the policy is trained,
> on-hardware bring-up is about *safety and plumbing*, not performance.

## Before you run (pre-flight)

1. **Export + validate, every time.** `python -m husky_x2_deploy.export_skater_deploy
   --onnx <run>.onnx` then `python -m husky_x2_deploy.validate_skater_parity --onnx
   <run>.onnx`. It must print `PARITY OK` with `max_obs_error=max_action_error=0`. If it
   doesn't, the deployed obs/action no longer match the policy — stop.
2. **Re-run the heading probe on the FINAL policy.** `python -m
   husky_x2_deploy.heading_probe --onnx <run>.onnx`. The current (undertrained) X2 policy
   is **highly heading-sensitive** (~17° joint-target shift for a realistic 15° start
   offset). See the heading seam below.
3. **Rates match.** `control.rate_hz` (50) must equal `1/control_dt` (0.02 s). The phase
   clock advances exactly once per tick; a rate mismatch desyncs the gait clock *and* the
   PD dynamics.
4. **Start conservative.** Keep `require_start_trigger: true` (arm; no output until
   `/agibot_deploy/start`). First bring-up: `stand_only: true` to hold the default pose
   and confirm gains/holding before letting the policy drive. Generous `soft_start_secs`.
5. **The robot must start ON the board, in the PUSH_INIT stance** — `default_joint_pos`
   is that stance, and soft-start ramps to it. Off-board, the stance is meaningless.
6. Keep a hardware E-stop in hand independent of the software one.

## Known seams (what to watch, and why)

- **Heading frame (the big one).** Hardware has no absolute world yaw, so
  `heading_reference: start` zeroes `heading` at policy start (then it drifts with IMU
  yaw). Training used *absolute* yaw. The heading probe shows the policy's output depends
  strongly on the heading value, so the robot effectively steers/holds heading **relative
  to wherever it started**, not an absolute compass. If you need absolute steering or the
  probe stays HIGH on the trained policy, wire a real heading source and set
  `heading_reference: absolute`.
- **IMU body mapping.** `topics.imu: /aima/hal/imu/torso/state`, but the config notes the
  HAL "torso" IMU is *physically the pelvis* (= sim `imu_0`), which is what feeds
  `projected_gravity` and `base_ang_vel`. If that topic is actually a different body (or
  the axes are rotated), the robot's sense of "up" and rotation is wrong → it will lean or
  fall. Verify on the bench: tilt the robot and confirm `projected_gravity` tracks gravity
  the same way `sim.py` does.
- **Gains.** `stand_stiffness/stand_damping` in `x2_skater.yaml` are placeholder values —
  **tune on a gantry**. Active-phase gains come from the ONNX (the X2 training gains).
- **Joint order / SDK groups.** The policy joint order is the ONNX `joint_names`; the host
  splits it into `{leg, waist, arm, head}`. `husky-skate-validate` prints the split
  (`leg:12, waist:3, arm:14, head:2`). If a joint is renamed or reordered, the wrong
  actuator moves.
- **Quantization.** Gains/scales embedded in the ONNX are 3-decimal CSV (<1e-3 vs
  training) — negligible, but it's why `default_joint_pos` isn't bit-identical to the env.

## Symptom → likely cause

| What you observe | Most likely cause | Check / fix |
|---|---|---|
| Collapses the instant the policy engages | gains too low, soft-start too short, or wrong default pose | `stand_only: true` first; raise `soft_start_secs` / `policy_fade_secs`; confirm `default_joint_pos` is PUSH_INIT and gains are sane |
| Holds a stand but **drifts / yaws / steers the wrong way** | heading-frame mismatch (start vs absolute), or inverted steer axis | re-run `heading_probe`; flip `teleop.axes.wz.scale`; try `heading_reference: absolute` with a real heading source |
| Leans/falls consistently toward one direction even when "standing" | IMU is the wrong body or its axes are rotated (bad `projected_gravity`/`base_ang_vel`) | verify `topics.imu` is the pelvis IMU; bench-test gravity/ang-vel sign vs `sim.py` |
| Uncoordinated — *wrong* limbs move for a given motion | joint-name → SDK-group mismatch | confirm ONNX `joint_names` is a permutation of the SDK joints; check the `husky-skate-validate` group split |
| Buzzing / high-freq oscillation / instability | control loop not at 50 Hz, or gains too high | confirm actual loop rate == `rate_hz`; lower stiffness; check for dropped ticks |
| No response to teleop / command stuck | command not wired, or deadman not held | `command_provider.kind: skate`; `/cmd_vel` up; hold the deadman button; check `teleop.joystick` device |
| Behaves differently than it did in `sim.py` | obs/action contract drift | `husky-skate-validate` MUST pass; re-export the sidecar; confirm obs term order/scales unchanged in `skater_env_cfg.py` |
| Freezes / repeatedly snaps to a held pose | sensor state stale (state timeout) | `safety.state_timeout_sec` (0.25 s); check IMU + per-group joint-state publish rates |
| Joint targets slam into limits / clipped hard | action over-range (often just an undertrained policy) | inspect `safety.joint_limits` vs `action_scale`; expect this until the policy is trained |
| E-stop doesn't hold pose | E-stop damping/stiffness config | E-stop forces stiffness→0, holds measured pose via `estop_damping_scale`; verify it's nonzero |
| NaN in obs or policy output | unnormalized/garbage IMU quaternion, or a sensor dropout | validate the IMU quat is unit-norm; guard sensor freshness |

## After a successful stand

Only after the robot reliably soft-starts, holds the stance, and E-stops cleanly:
bring `stand_only: false`, keep `policy_fade_secs` generous, and command tiny forward
speed first. Re-confirm the heading behaviour matches your steering intent before any
real push.
