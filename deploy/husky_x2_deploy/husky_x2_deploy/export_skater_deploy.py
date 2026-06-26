#!/usr/bin/env python3
"""Write the AgiBot deploy-host metadata sidecar for a HUSKY skater ONNX.

A skater ONNX exported by ``SkaterOnPolicyRunner`` (mjlab) already **embeds**
``joint_names`` / ``joint_stiffness`` / ``joint_damping`` / ``default_joint_pos``
/ ``action_scale`` / ``observation_names`` / ``command_names`` as CSV strings,
which the deploy host's ``load_policy_metadata`` parses natively. The host's
``ObsBuilder``, however, has no ``actor_obs_spec`` (it cannot assemble the obs),
no skater ``command_spec``, and infers only a bare ``obs`` input. This script
fills those gaps: it copies the ONNX into a deploy ``models/`` dir and writes the
``<model>.onnx.metadata.json`` sidecar in the host contract so the metadata-driven
host (plus the ``husky_x2_deploy`` adapter) can run it.

Robot-agnostic: term dims scale with the policy's joint count, so the same script
handles the AgiBot X2 (31 joints, obs 515) and the Unitree G1 (23 joints, obs
395). The per-term **scales** are the skater env's, fixed across robots (see
``skater_env_cfg.py::make_g1_skater_env_cfg`` policy terms).

Usage:
    python -m husky_x2_deploy.export_skater_deploy --onnx <skater.onnx> [--out deploy/models] [--name NAME]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

from agibot_deploy.metadata import load_policy_metadata

# Per-term scales, from skater_env_cfg.py policy terms. Robot-independent.
TERM_SCALE: dict[str, float | list[float]] = {
    "command": [2.0, 1.0],
    "heading": 1.0 / math.pi,
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,
    "actions": 1.0,
    "phase": 1.0,
}
# Fixed (joint-count-independent) term dims; everything else is per-joint.
TERM_FIXED_DIM: dict[str, int] = {
    "command": 2,
    "heading": 1,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "phase": 1,
}
PER_JOINT_TERMS = {"joint_pos", "joint_vel", "actions"}

# Canonical skater obs order (fallback when the ONNX predates observation_names).
DEFAULT_OBS_ORDER = [
    "command", "heading", "base_ang_vel", "projected_gravity",
    "joint_pos", "joint_vel", "actions", "phase",
]


def _onnx_obs_input_dim(onnx_path: Path) -> int:
    import onnx

    model = onnx.load(str(onnx_path))
    inits = {i.name for i in model.graph.initializer}
    for vi in model.graph.input:
        if vi.name in inits:
            continue
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim if d.dim_value > 0]
        if dims:
            return int(dims[-1])
    raise ValueError(f"could not read an obs input dim from {onnx_path}")


def build_sidecar(onnx_path: Path) -> dict:
    md = load_policy_metadata(onnx_path)
    n = md.action_dim
    if n <= 0:
        raise ValueError(f"{onnx_path} has no joint_names embedded; cannot export")

    obs_names = list(md.raw.get("observation_names") or DEFAULT_OBS_ORDER)
    actor_obs_spec = []
    for name in obs_names:
        if name in TERM_FIXED_DIM:
            dim = TERM_FIXED_DIM[name]
        elif name in PER_JOINT_TERMS:
            dim = n
        else:
            raise ValueError(f"unknown skater obs term {name!r}; update TERM_SCALE/TERM_FIXED_DIM")
        actor_obs_spec.append({"name": name, "dim": dim, "scale": TERM_SCALE[name]})

    single_frame = sum(term["dim"] for term in actor_obs_spec)
    obs_dim = _onnx_obs_input_dim(onnx_path)
    if obs_dim % single_frame != 0:
        raise ValueError(
            f"obs input dim {obs_dim} is not a multiple of single-frame dim {single_frame}"
        )
    history_len = obs_dim // single_frame
    control_dt = float(md.control_dt or 0.02)

    return {
        "run_path": md.run_path,
        "source": "husky_skater",
        "capabilities": ["feedforward"],
        "joint_names": list(md.joint_names),
        "joint_stiffness": list(md.joint_stiffness),
        "joint_damping": list(md.joint_damping),
        "default_joint_pos": list(md.default_joint_pos),
        "action_scale": list(md.action_scale),
        "control_dt": control_dt,
        "history_length": history_len,
        "actor_obs_spec": actor_obs_spec,
        "command_spec": {
            "kind": "skate",
            "terms": [
                {"name": "command", "dim": 2},
                {"name": "phase", "dim": 1},
            ],
            "cycle_time": 6.0,
            "v_max": 1.5,
            "h_max": math.pi / 4.0,
            "control_dt": control_dt,
        },
        "input_contract": {
            "obs": {"role": "actor_obs", "dim": obs_dim, "carried": False},
        },
        "action_output": "actions",
        "action_output_mode": "raw_action",
        "previous_action_output": "actions",
    }


def export(onnx_path: Path, out_dir: Path, name: str | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / (name or onnx_path.name)
    if out_onnx.resolve() != onnx_path.resolve():
        shutil.copy2(onnx_path, out_onnx)
    sidecar = build_sidecar(onnx_path)
    sidecar_path = Path(str(out_onnx) + ".metadata.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return out_onnx, sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="trained skater ONNX (G1 or X2)")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[2] / "models",
        help="output dir for the deploy ONNX + sidecar (default: deploy/models)",
    )
    parser.add_argument("--name", type=str, default=None, help="output ONNX filename (default: same as input)")
    args = parser.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    out_onnx, sidecar_path = export(args.onnx, args.out, args.name)
    sidecar = json.loads(sidecar_path.read_text())
    print(f"wrote {out_onnx}")
    print(f"wrote {sidecar_path}")
    print(
        f"joints={len(sidecar['joint_names'])} obs_dim={sidecar['input_contract']['obs']['dim']} "
        f"history={sidecar['history_length']} control_dt={sidecar['control_dt']} "
        f"command_kind={sidecar['command_spec']['kind']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
