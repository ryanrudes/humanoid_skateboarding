#!/usr/bin/env python3
"""Offline parity gate: deploy adapter obs/action == test_scene/sim.py, exactly.

Runs entirely on CPU with no robot. It drives a synthetic but non-trivial sensor
trajectory (varied base orientation, joint pos/vel, gyro) through two independent
observation builders and asserts they agree to ~1e-6:

  A. **reference** - a faithful, standalone re-implementation of the 515/395-dim
     observation exactly as ``test_scene/sim.py`` builds it (the validated
     standalone eval): per-term scales, ``joint_pos - default``, sim.py's own
     ``get_gravity_orientation`` and ``heading = atan2(forward_w.y, forward_w.x)``
     formulas, and the term-major 5-frame history flatten.

  B. **adapter** - :class:`SkateObsAssembler` (heading_reference="absolute" to
     match sim.py's absolute yaw) + :class:`SkateCommandProvider`, fed the same
     raw sensors, assembled through the vendored host ``ObsBuilder``.

Because A ports sim.py's quaternion math while B uses the host's
``projected_gravity_from_quat_xyzw`` / yaw, an exact match also *empirically*
confirms the two conventions are identical. It then runs both observations
through the ONNX and asserts the actions match, and exercises the full host path
(``build`` -> ``PolicyRunner.step`` -> ``ActionMapper``) for a few steps.

Usage:
    python -m husky_x2_deploy.validate_skater_parity --onnx <skater.onnx> [--steps 64]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from agibot_deploy import ActionMapper, JointMap, PolicyRunner, load_policy_metadata

from husky_x2_deploy.export_skater_deploy import export
from husky_x2_deploy.skate_command import SkateCommandProvider
from husky_x2_deploy.skate_obs import SkateObsAssembler

DEFAULT_ONNX = "logs/rsl_rl/x2_skater/2026-06-25_17-40-42/2026-06-25_17-40-42.onnx"


# -- sim.py-faithful reference (ported from test_scene/sim.py) ----------------

def _sim_gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ], dtype=np.float64)


def _sim_quat_apply(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w = quat_wxyz[0]
    qvec = quat_wxyz[1:4]
    t = 2 * np.cross(qvec, vec)
    return vec + w * t + np.cross(qvec, t)


def _sim_heading(quat_wxyz: np.ndarray) -> float:
    fwd = _sim_quat_apply(quat_wxyz, np.array([1.0, 0.0, 0.0]))
    return math.atan2(fwd[1], fwd[0])


class SimReferenceObs:
    """Rebuild sim.py's flat observation from the deploy metadata (same scales)."""

    def __init__(self, metadata, *, cycle_time=6.0):
        self.n = metadata.action_dim
        self.default = np.asarray(metadata.default_joint_pos, dtype=np.float64)
        self.control_dt = float(metadata.control_dt or 0.02)
        self.cycle_time = float(cycle_time)
        self.block_dims = [2, 1, 3, 3, self.n, self.n, self.n, 1]
        self.frame_dim = sum(self.block_dims)
        contract = metadata.input_contract.get("obs")
        total = int(contract.dim) if contract and contract.dim else self.frame_dim
        self.history_len = total // self.frame_dim
        from collections import deque
        self.buf = deque(
            (np.zeros(self.frame_dim) for _ in range(self.history_len)),
            maxlen=self.history_len,
        )
        self.k = 0

    def step(self, *, v, h, quat_wxyz, ang_vel, joint_pos, joint_vel, prev_action):
        self.k += 1
        phase = (self.k * self.control_dt / self.cycle_time) % 1.0
        frame = np.concatenate([
            np.array([v, h]) * np.array([2.0, 1.0]),
            np.array([_sim_heading(quat_wxyz)]) * (1.0 / math.pi),
            np.asarray(ang_vel, dtype=np.float64) * 0.25,
            _sim_gravity_orientation(quat_wxyz),
            (np.asarray(joint_pos, dtype=np.float64) - self.default),
            np.asarray(joint_vel, dtype=np.float64) * 0.05,
            np.asarray(prev_action, dtype=np.float64),
            np.array([phase]),
        ])
        self.buf.append(frame)
        hist = np.stack(self.buf, axis=0)
        parts, start = [], 0
        for dim in self.block_dims:
            parts.append(hist[:, start:start + dim].reshape(-1))
            start += dim
        return np.concatenate(parts).astype(np.float32)


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _rpy_to_quat_wxyz(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def run(onnx_path: Path, steps: int = 64, tol: float = 1e-6) -> int:
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="husky_skate_parity_"))
    try:
        out_onnx, _ = export(onnx_path, tmp)
        md = load_policy_metadata(out_onnx)
        return _run_with_model(onnx_path, out_onnx, md, rng, steps, tol)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_with_model(onnx_path, out_onnx, md, rng, steps, tol) -> int:
    n = md.action_dim
    print(f"onnx={onnx_path}")
    print(f"joints={n} obs_dim={md.input_contract['obs'].dim} "
          f"actor_obs_terms={len(md.actor_obs_spec)} command_kind={md.command_spec.get('kind')}")
    assert md.command_spec.get("kind") == "skate", "sidecar did not select the skate command"
    assert len(md.actor_obs_spec) == 8, "expected 8 single-frame obs terms"

    provider = SkateCommandProvider(md)
    assembler = SkateObsAssembler(md, command_provider=provider, heading_reference="absolute")
    reference = SimReferenceObs(md, cycle_time=provider.cycle_time)
    runner = PolicyRunner(out_onnx, metadata=md)
    mapper = ActionMapper(md, joint_map=JointMap(md.joint_names))

    v, h = 1.0, 0.2
    provider.set_skate(v, h)
    default = np.asarray(md.default_joint_pos, dtype=np.float64)

    prev_action = np.zeros(n, dtype=np.float64)
    max_obs_err = 0.0
    max_act_err = 0.0
    for t in range(steps):
        quat_wxyz = _rpy_to_quat_wxyz(
            0.10 * math.sin(0.20 * t), 0.08 * math.cos(0.15 * t), 0.25 * math.sin(0.05 * t)
        )
        quat_xyzw = _wxyz_to_xyzw(quat_wxyz)
        ang_vel = 0.3 * np.array([math.sin(0.3 * t), math.cos(0.2 * t), math.sin(0.1 * t)])
        joint_pos = default + 0.15 * rng.standard_normal(n)
        joint_vel = 0.5 * rng.standard_normal(n)

        ref_obs = reference.step(
            v=v, h=h, quat_wxyz=quat_wxyz, ang_vel=ang_vel,
            joint_pos=joint_pos, joint_vel=joint_vel, prev_action=prev_action,
        )
        raw = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "base_quat_xyzw": quat_xyzw,
            "base_ang_vel": ang_vel,
            "prev_action": prev_action,
        }
        asm_obs = assembler.build_actor_obs(raw)

        obs_err = float(np.max(np.abs(ref_obs - asm_obs)))
        max_obs_err = max(max_obs_err, obs_err)

        ref_act = np.asarray(runner.session.run(None, {"obs": ref_obs[None, :]})[0]).reshape(-1)
        asm_act = np.asarray(runner.session.run(None, {"obs": asm_obs[None, :]})[0]).reshape(-1)
        max_act_err = max(max_act_err, float(np.max(np.abs(ref_act - asm_act))))
        prev_action = asm_act  # feed the policy's own output back, for both builders

    # Exercise the full host path (build -> step -> map) the ROS node uses.
    assembler.reset()
    provider.reset()
    provider.set_skate(v, h)
    raw = {
        "joint_pos": default, "joint_vel": np.zeros(n),
        "base_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
        "base_ang_vel": np.zeros(3), "prev_action": np.zeros(n),
    }
    obs_inputs = assembler.build(raw)
    out = runner.step(obs_inputs)
    action = runner.selected_action_output(out).reshape(-1)
    targets = mapper.output_to_targets(action, mode=runner.action_output_mode)
    groups = mapper.targets_to_group_commands(targets)

    print(f"steps={steps}")
    print(f"max_obs_error={max_obs_err:.3e}")
    print(f"max_action_error={max_act_err:.3e}")
    print(f"host_path: obs_keys={sorted(obs_inputs)} action_mode={runner.action_output_mode} "
          f"targets_finite={bool(np.isfinite(targets).all())} "
          f"sdk_groups={{{', '.join(f'{g}:{len(v)}' for g, v in groups.items())}}}")

    ok = max_obs_err < tol and max_act_err < tol and np.isfinite(targets).all()
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=Path(DEFAULT_ONNX), help="skater ONNX to validate")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()
    if not args.onnx.exists():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    return run(args.onnx, steps=args.steps, tol=args.tol)


if __name__ == "__main__":
    raise SystemExit(main())
