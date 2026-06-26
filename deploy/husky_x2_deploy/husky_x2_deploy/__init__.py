"""HUSKY skater adapter for the AgiBot X2 Ultra deploy host.

Bridges a trained HUSKY skater ONNX policy (G1 or AgiBot X2) onto the vendored
``agibot_deploy`` runtime (../agibot_x2_ultra_deploy). The host is policy-agnostic
but cannot, on its own, express the skater observation (5-frame history, a
``heading`` term, a scalar ``phase`` clock, and a ``[v, h]`` command). This
package supplies exactly those seams:

- :class:`SkateCommandProvider` - the ``[v, h]`` command + cyclic ``phase`` clock.
- :class:`SkateObsAssembler`   - 5-frame term-major history + the heading term.
- ``export_skater_deploy``     - write the deploy metadata sidecar for a skater ONNX.
- ``skate_deploy_node``        - a ROS 2 node (subclasses the host node) wiring it together.
"""

from husky_x2_deploy.skate_command import SkateCommandProvider
from husky_x2_deploy.skate_obs import SkateObsAssembler, yaw_from_quat_xyzw

__all__ = [
    "SkateCommandProvider",
    "SkateObsAssembler",
    "yaw_from_quat_xyzw",
]
