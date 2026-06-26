#!/usr/bin/env python3
"""Heading-offset sensitivity probe for the skater deploy.

On hardware, ``SkateObsAssembler(heading_reference="start")`` yaw-normalizes the
``heading`` obs term to the value at policy start (``heading(0)=0``), because the
X2 has no absolute world yaw and IMU yaw drifts. Training (and ``test_scene/sim.py``)
used the **absolute** world yaw. This probe quantifies how much that frame choice
changes the *policy's actions*: over a synthetic but non-trivial sensor trajectory
it adds a constant offset ``delta`` to the ``heading`` term (everything else held
fixed — and ``projected_gravity`` is yaw-invariant, so a heading offset is a clean,
isolated perturbation) and measures the resulting change in the policy's output.

Interpretation:
  * Small divergence (e.g. << the joint action_scale) => the policy is roughly
    heading-offset-tolerant; ``heading_reference="start"`` is safe.
  * Large divergence => the hardware heading frame meaningfully changes behaviour;
    validate on a gantry, or wire an absolute heading source and use "absolute".

Usage:
    python -m husky_x2_deploy.heading_probe --onnx <skater.onnx> [--steps 128]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import onnxruntime as ort

from agibot_deploy import load_policy_metadata

from husky_x2_deploy.export_skater_deploy import export
from husky_x2_deploy.skate_command import SkateCommandProvider
from husky_x2_deploy.skate_obs import SkateObsAssembler

DEFAULT_ONNX = "logs/rsl_rl/x2_skater/2026-06-25_17-40-42/2026-06-25_17-40-42.onnx"
DELTAS = (0.1, 0.25, 0.5, 1.0, math.pi)  # heading offsets to probe, radians


def _rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
         cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy],
        dtype=np.float64,
    )


def _rollout(onnx_path: Path, md, sess, n, steps, head_delta, cmd_delta=0.0):
    """Run the policy for `steps` adding `head_delta` to the heading obs term and
    `cmd_delta` to the command heading `h`; return per-step raw actions (closed-loop).

    head_delta only      -> the obs heading is off by Δ vs the command (operator
                            commands an absolute heading): the worst case for "start".
    head_delta==cmd_delta -> a common Δ frame shift of both (operator steers relative
                            to start): tests heading-vs-command-error invariance.
    """
    asm = SkateObsAssembler(md, command_provider=SkateCommandProvider(md), heading_reference="absolute")
    asm.command_provider.set_skate(1.0, 0.2 + cmd_delta)
    default = np.asarray(md.default_joint_pos, dtype=np.float64)
    rng = np.random.default_rng(0)  # same seed => same state trajectory across deltas
    prev = np.zeros(n, dtype=np.float64)
    out = []
    for t in range(steps):
        base_heading = 0.4 * math.sin(0.04 * t)            # a varying absolute heading
        quat = _rpy_to_quat_xyzw(0.10 * math.sin(0.2 * t), 0.08 * math.cos(0.15 * t), 0.0)
        ang = 0.3 * np.array([math.sin(0.3 * t), math.cos(0.2 * t), math.sin(0.1 * t)])
        jp = default + 0.15 * rng.standard_normal(n)
        jv = 0.5 * rng.standard_normal(n)
        raw = {
            "joint_pos": jp, "joint_vel": jv, "base_quat_xyzw": quat,
            "base_ang_vel": ang, "prev_action": prev,
            "heading": np.array([base_heading + head_delta], dtype=np.float64),  # explicit override
        }
        obs = asm.build_actor_obs(raw)
        act = np.asarray(sess.run(None, {"obs": obs[None, :]})[0]).reshape(-1)
        out.append(act)
        prev = act
    return np.stack(out, axis=0)


def run(onnx_path: Path, steps: int = 128) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="husky_heading_probe_"))
    try:
        out_onnx, _ = export(onnx_path, tmp)
        md = load_policy_metadata(out_onnx)
        n = md.action_dim
        sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
        scale = np.asarray(md.action_scale, dtype=np.float64)

        smax = float(scale.max())
        base = _rollout(onnx_path, md, sess, n, steps, 0.0, 0.0)
        print(f"onnx={onnx_path}")
        print(f"joints={n}  steps={steps}  max|action_scale|={smax:.4f} rad/unit")
        print("worst joint-target shift vs no-offset, for a heading-frame offset `delta`:")
        print(f"  {'delta':>10} | {'heading-only (cmd fixed)':>24} | {'heading+command (common)':>24}")
        for d in DELTAS:
            sh = float(np.max(np.abs(_rollout(onnx_path, md, sess, n, steps, d, 0.0) - base))) * smax
            sc = float(np.max(np.abs(_rollout(onnx_path, md, sess, n, steps, d, d) - base))) * smax
            print(f"  {math.degrees(d):7.1f}deg | {math.degrees(sh):>20.1f}deg | {math.degrees(sc):>20.1f}deg")
        print("\nheading-only  = operator commands an ABSOLUTE heading (worst case for 'start' mode)")
        print("heading+command = operator steers RELATIVE to start (the natural teleop case)")
        ref = math.radians(15.0)  # ~ the PUSH_INIT spawn yaw, a realistic start offset
        sc_ref = float(np.max(np.abs(_rollout(onnx_path, md, sess, n, steps, ref, ref) - base))) * smax
        verdict = ("LOW: ~invariant to a common heading shift; 'start' mode is safe"
                   if sc_ref < 0.05 else
                   "MODERATE: a realistic start-offset shifts targets a few deg — verify on a gantry"
                   if sc_ref < 0.20 else
                   "HIGH: even a common heading shift moves the policy a lot — prefer an absolute heading source")
        print(f"=> at a realistic ~15deg start offset, common-shift divergence ~= "
              f"{math.degrees(sc_ref):.1f}deg -> {verdict}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=Path(DEFAULT_ONNX))
    ap.add_argument("--steps", type=int, default=128)
    args = ap.parse_args()
    if not args.onnx.exists():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    return run(args.onnx, steps=args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
