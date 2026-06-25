"""AgiBot X2-Ultra constants (skater task).

Mirrors ``robots/skateboard/g1_skater_constants.py`` but for the AgiBot X2-Ultra
humanoid (31 actuated joints + floating base, 34 bodies). Uses the pinned-mjlab
actuator API (``BuiltinPositionActuatorCfg`` + ``target_names_expr``). The
skateboard entity is robot-independent and is reused from the G1 constants.

MJCF + meshes are vendored under ``xmls/`` (MIT, AgiBot ``x2_description``); the
``imu_ang_vel``/``imu_lin_vel`` sensors and 4-corner foot sites
(``left/right_foot_1..4``) were added there for the skater env, mirroring the
edits made to ``g1.xml``.
"""

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

# Reuse the robot-independent skateboard entity from the G1 constants.
from mjlab_husky.asset_zoo.robots.skateboard.g1_skater_constants import (
  get_skateboard_cfg as get_skateboard_cfg,
)
from mjlab_husky.asset_zoo.robots.skateboard.g1_skater_constants import (
  get_skateboard_spec as get_skateboard_spec,
)

##
# MJCF and assets.
##

X2_XML: Path = Path(os.path.join(os.path.dirname(__file__), "xmls", "x2_ultra.xml"))
assert X2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, X2_XML.parent / "assets", meshdir)
  return assets


def get_x2_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X2_XML))
  # Drop the reference keyframe; mjlab supplies the initial state via EntityCfg.
  while spec.keys:
    spec.delete(spec.keys[0])
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config (AIMDK PD gains; pinned-mjlab BuiltinPositionActuatorCfg API).
##

X2_ACTUATOR_HIP_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint"),
  stiffness=40.0,
  damping=4.0,
  effort_limit=120.0,
  armature=0.03,
)
X2_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=30.0,
  damping=3.0,
  effort_limit=120.0,
  armature=0.03,
)
X2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=80.0,
  damping=8.0,
  effort_limit=120.0,
  armature=0.03,
)
X2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=40.0,
  damping=4.0,
  effort_limit=36.0,
  armature=0.03,
)
X2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=20.0,
  damping=2.0,
  effort_limit=24.0,
  armature=0.03,
)
X2_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=20.0,
  damping=4.0,
  effort_limit=120.0,
  armature=0.03,
)
X2_ACTUATOR_WAIST_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=20.0,
  damping=4.0,
  effort_limit=48.0,
  armature=0.03,
)
X2_ACTUATOR_SHOULDER_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=36.0,
  armature=0.03,
)
X2_ACTUATOR_ARM_SMALL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=20.0,
  damping=2.0,
  effort_limit=24.0,
  armature=0.03,
)
X2_ACTUATOR_WRIST_PITCH_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=4.8,
  armature=0.03,
)
X2_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("head_yaw_joint", "head_pitch_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=2.6,
  armature=0.03,
)

X2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X2_ACTUATOR_HIP_PITCH_ROLL,
    X2_ACTUATOR_HIP_YAW,
    X2_ACTUATOR_KNEE,
    X2_ACTUATOR_ANKLE_PITCH,
    X2_ACTUATOR_ANKLE_ROLL,
    X2_ACTUATOR_WAIST_YAW,
    X2_ACTUATOR_WAIST_PITCH_ROLL,
    X2_ACTUATOR_SHOULDER_PITCH_ROLL,
    X2_ACTUATOR_ARM_SMALL,
    X2_ACTUATOR_WRIST_PITCH_ROLL,
    X2_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)

##
# Keyframe config.
##

# Push stance adapted from the G1 ``PUSH_INIT_KEYFRAME`` to X2 joint names and
# ranges. NOTE: the X2 elbow range is [-2.3556, 0] (opposite sign to the G1), so
# elbow targets are negative here. These values are a sensible starting stance
# and are expected to need visual tuning for good skateboarding behaviour.
PUSH_INIT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(-0.03, 0.1, 0.70),
  joint_pos={
    "left_knee_joint": 0.23,
    "left_ankle_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.7,
    "right_knee_joint": 1.17,
    "right_ankle_pitch_joint": -0.45,
    "left_shoulder_pitch_joint": -0.03,
    "left_shoulder_roll_joint": 0.45,
    "left_shoulder_yaw_joint": -0.21,
    "left_elbow_joint": -1.32,
    "right_shoulder_pitch_joint": -0.7,
    "right_shoulder_roll_joint": -0.845,
    "right_shoulder_yaw_joint": 0.83,
    "right_elbow_joint": -1.19,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

X2_FOOT_COLLISION_EXPR = r"^(left|right)_foot([1-9]|1[0-2])_collision$"
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={X2_FOOT_COLLISION_EXPR: 3, ".*_collision": 1},
  priority={X2_FOOT_COLLISION_EXPR: 1},
  friction={X2_FOOT_COLLISION_EXPR: (1,)},
)

##
# Final config.
##


def get_x2_robot_cfg() -> EntityCfg:
  """Get a fresh AgiBot X2-Ultra robot configuration instance."""
  return EntityCfg(
    init_state=PUSH_INIT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_x2_spec,
    articulation=X2_ARTICULATION,
  )


X2_ACTION_SCALE: dict[str, float] = {}
for a in X2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    X2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_x2_robot_cfg())
  viewer.launch(robot.spec.compile())
