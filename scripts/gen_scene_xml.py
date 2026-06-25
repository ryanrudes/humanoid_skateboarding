"""Generate a standalone MuJoCo scene XML (robot + skateboard) for test_scene/sim.py.

mjlab composes the training scene by prefixing each entity (``robot/``,
``skateboard/``); test_scene/sim.py loads that flattened scene in *plain* MuJoCo.
The bundled ``test_scene/mjlab_scene.xml`` is the G1 one. This script regenerates
the equivalent for another robot via mjlab's ``Scene.spec.to_xml()``, dumping all
meshes (robot + board live in different source dirs) into a co-located assets dir
so vanilla MuJoCo can resolve them.

Usage (from repo root):
    uv run python scripts/gen_scene_xml.py --robot x2   # -> test_scene/mjlab_scene_x2.xml
"""

from __future__ import annotations

import argparse
import os

import mjlab_husky.tasks  # noqa: F401  -- task discovery before constants import
import mujoco
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainImporterCfg

from mjlab_husky.asset_zoo.robots.agibot_x2.x2_skater_constants import (
  get_skateboard_cfg,
  get_x2_robot_cfg,
)

TEST_SCENE_DIR = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_scene"
)

ROBOT_CFGS = {
  "x2": get_x2_robot_cfg,
}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robot", choices=sorted(ROBOT_CFGS), default="x2")
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  assets_dirname = f"{args.robot}_assets"
  xml_path = os.path.join(TEST_SCENE_DIR, f"mjlab_scene_{args.robot}.xml")
  assets_dir = os.path.join(TEST_SCENE_DIR, assets_dirname)

  cfg = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    entities={"robot": ROBOT_CFGS[args.robot](), "skateboard": get_skateboard_cfg()},
  )
  scene = Scene(cfg, device=args.device)
  spec = scene.spec

  # Point the written XML at a co-located assets dir and dump every mesh there
  # (robot + skateboard meshes originate from different source directories). The
  # in-memory asset keys are like "assets/robot/foo.STL" while the XML mesh
  # file= refs are "robot/foo.STL", so strip the leading "assets/" to align them
  # with the compiler meshdir we set below.
  spec.meshdir = assets_dirname
  for key, data in spec.assets.items():
    file_ref = key[len("assets/"):] if key.startswith("assets/") else key
    target = os.path.join(assets_dir, file_ref)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as f:
      f.write(data)

  xml = spec.to_xml()
  # mjlab's to_xml emits a bare nested `<default/>` (empty class name) that plain
  # MuJoCo rejects; comment it out, matching the bundled G1 mjlab_scene.xml.
  xml = xml.replace("<default/>", "<!-- <default/> -->")
  with open(xml_path, "w") as f:
    f.write(xml)

  # Verify the standalone XML compiles in plain MuJoCo.
  model = mujoco.MjModel.from_xml_path(xml_path)
  print(f"wrote {xml_path}")
  print(f"  assets: {len(spec.assets)} meshes -> {assets_dir}")
  print(f"  compiled OK: nbody={model.nbody} nu={model.nu} nq={model.nq} nsensor={model.nsensor}")


if __name__ == "__main__":
  main()
