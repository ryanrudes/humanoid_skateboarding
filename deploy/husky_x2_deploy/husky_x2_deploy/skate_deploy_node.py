"""ROS 2 deploy node for the HUSKY skater policy on AgiBot X2 hardware.

Thin specialization of the vendored ``agibot_deploy_ros`` node
(``AgibotDeployNode``): it reuses **all** of the host's hardware machinery
unchanged — joint-state/IMU subscriptions, soft-start ramp, policy fade-in,
E-stop, actuator-fault hold, grouped ``JointCommand`` publishing, teleop, and
shutdown safety — because the host ``_tick`` is written against the abstract
``obs_builder.build(raw) -> runner.step(...) -> mapper.output_to_targets(...)``
interface.

The skater only needs three swaps, applied after the base node is constructed:

1. ``command_provider`` -> :class:`SkateCommandProvider` (``[v, h]`` + phase clock).
2. ``obs_builder`` -> :class:`SkateObsAssembler` (5-frame history + heading term).
3. a ``/cmd_vel`` subscription (the base node only wires one for velocity
   providers; ``SkateCommandProvider.set_velocity`` maps the operator triplet to
   the skater command, so ``/cmd_vel`` and gamepad teleop both drive it).

On every (re)start the assembler history, phase clock, and previous-action are
reset so the policy begins from a clean frame.

Run it exactly like the host node but with this entry point and the skater config:

    ros2 launch agibot_deploy_ros deploy.launch.py \
        config_path:=deploy/husky_x2_deploy/husky_x2_deploy/config/x2_skater.yaml \
        model_path:=deploy/models/<skater>.onnx

or directly: ``python -m husky_x2_deploy.skate_deploy_node`` (needs the AgiBot SDK
+ ``agibot_deploy_ros`` sourced). See the package README for the full bring-up.
"""

from __future__ import annotations

from pathlib import Path
import sys

from husky_x2_deploy.skate_command import SkateCommandProvider
from husky_x2_deploy.skate_obs import SkateObsAssembler


def _import_host_node():
    """Import the vendored AgibotDeployNode + _nested helper.

    Works both when ``agibot_deploy_ros`` is colcon-built and sourced (normal
    import) and standalone, by adding the submodule's ROS package dir to the path.
    """
    try:
        from agibot_deploy_ros.deploy_node import AgibotDeployNode, _nested
    except ImportError:
        host_ros = (
            Path(__file__).resolve().parents[2]
            / "agibot_x2_ultra_deploy" / "ros2" / "agibot_deploy_ros"
        )
        if host_ros.is_dir() and str(host_ros) not in sys.path:
            sys.path.insert(0, str(host_ros))
        from agibot_deploy_ros.deploy_node import AgibotDeployNode, _nested
    return AgibotDeployNode, _nested


# Resolve the vendored host node (works colcon-sourced or standalone).
AgibotDeployNode, _nested = _import_host_node()


class SkateDeployNode(AgibotDeployNode):
    def __init__(self) -> None:
        super().__init__()

        heading_reference = str(
            _nested(self.config, "control", "heading_reference", default="start")
        )
        # Swap the host's generic command provider + obs builder for the skater's.
        self.command_provider = SkateCommandProvider(self.metadata)
        self.obs_builder = SkateObsAssembler(
            self.metadata,
            command_provider=self.command_provider,
            heading_reference=heading_reference,
        )
        self.node.get_logger().info(
            f"skater adapter active: command=[v,h]+phase, 5-frame history, "
            f"heading_reference={heading_reference}"
        )

        # The base node only subscribes /cmd_vel for velocity providers; wire it
        # here so /cmd_vel (and gamepad teleop) drive the skater command.
        self._wire_cmd_vel()

        # Start clean (the swapped provider/assembler have their own fresh state).
        self.command_provider.reset()
        self.obs_builder.reset()

    def _wire_cmd_vel(self) -> None:
        try:
            from geometry_msgs.msg import Twist
        except Exception:  # geometry_msgs absent -> velocity input disabled
            self.node.get_logger().warn("geometry_msgs unavailable; /cmd_vel steering disabled")
            return
        topics = _nested(self.config, "topics", default={}) or {}
        cmd_vel_topic = topics.get("cmd_vel", "/cmd_vel")
        self.node.create_subscription(Twist, cmd_vel_topic, self._cmd_vel_callback, 10)
        self.node.get_logger().info(
            f"skater steering input on {cmd_vel_topic} "
            "(Twist: linear.x=push speed [0,v_max] m/s, angular.z=steer [-1,1] -> heading)"
        )

    def _start_or_restart(self, source: str) -> str:
        message = super()._start_or_restart(source)
        if message != "already started":
            # Reset history/phase/previous-action so the policy starts fresh.
            self.obs_builder.reset()
            self.command_provider.reset()
            self.runner.reset(batch_size=1)
        return message


def main(args=None) -> None:
    import signal

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    # Mirror the host node's signal handling so the "publish last safe pose on
    # shutdown" path runs (see agibot_deploy_ros.deploy_node.main for rationale).
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    class _Stop:
        def __init__(self):
            self.requested = False

        def __call__(self, *_):
            self.requested = True

    stop = _Stop()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    deploy = None
    executor = None
    try:
        deploy = SkateDeployNode()
        executor = SingleThreadedExecutor()
        executor.add_node(deploy.node)
        while rclpy.ok() and not stop.requested and not deploy.quit_requested:
            executor.spin_once(timeout_sec=0.1)
    finally:
        if deploy is not None:
            deploy.shutdown()
        if executor is not None:
            executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
