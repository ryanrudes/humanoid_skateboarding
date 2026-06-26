"""Skater command + phase-clock provider for the AgiBot deploy host.

The vendored host ships ``none``/``velocity``/``locomotion_gait``/``motion_file``
command providers (``agibot_deploy.command_provider``). None of them match the
HUSKY skater policy, whose command is **not** ``[vx, vy, wz]`` but a 2-vector
``[v, h]`` (forward push speed + heading target) plus a scalar **phase clock**
that advances once per control tick over a fixed ``cycle_time`` (the skating
stroke period). This provider emits exactly those two command terms.

It subclasses the host ``VelocityCommandProvider`` so the deploy node's existing
``/cmd_vel`` subscription and gamepad teleop (both call ``set_velocity``) drive it
unchanged: the operator's ``[vx, vy, wz]`` triplet is mapped to the skater
``[v, h]`` (forward speed from ``vx``; steering/heading target from ``wz``).

Terms are returned **unscaled**; ``ObsBuilder`` applies the ``actor_obs_spec``
scales (``command`` x [2.0, 1.0], ``phase`` x 1.0) when it assembles the frame —
matching ``test_scene/sim.py`` (``[v, h] * [2.0, 1.0]``).
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from agibot_deploy.command_provider import VelocityCommandProvider
from agibot_deploy.metadata import PolicyMetadata


class SkateCommandProvider(VelocityCommandProvider):
    """Emit the skater ``command`` = [v, h] and the cyclic ``phase`` clock.

    ``phase = (k * control_dt / cycle_time) % 1`` with ``k`` the control-step
    counter incremented once per :meth:`command` call (the first observation
    corresponds to ``k = 1``, matching ``test_scene/sim.py``'s ``phase_counter``
    and mjlab's ``phase`` term at episode start).

    Config (read from ``metadata.command_spec``, with defaults that match the
    training env in ``skater_env_cfg.py`` / ``x2_skater_constants.py``):

    - ``cycle_time``: stroke period in seconds (default 6.0).
    - ``v_max``: forward command clip, m/s (default 1.5 = ``lin_vel_x`` upper).
    - ``h_max``: heading command clip, rad (default pi/4 = ``heading`` range).
    - ``control_dt``: phase step (default ``metadata.control_dt`` = 0.02).
    """

    def __init__(
        self,
        metadata: PolicyMetadata,
        initial: Mapping[str, object] | None = None,
        *,
        cycle_time: float | None = None,
        v_max: float | None = None,
        h_max: float | None = None,
        control_dt: float | None = None,
    ) -> None:
        super().__init__(metadata, initial=None)
        spec = metadata.command_spec
        self.cycle_time = float(cycle_time if cycle_time is not None else spec.get("cycle_time", 6.0))
        self.v_max = float(v_max if v_max is not None else spec.get("v_max", 1.5))
        self.h_max = float(h_max if h_max is not None else spec.get("h_max", math.pi / 4.0))
        self.control_dt = float(
            control_dt
            if control_dt is not None
            else (spec.get("control_dt") or metadata.control_dt or 0.02)
        )
        if self.cycle_time <= 0.0:
            raise ValueError("cycle_time must be positive")
        self._v = 0.0
        self._h = 0.0
        self._k = 0
        if initial:
            self.set_command(initial)

    def reset(self) -> None:
        """Reset the phase-clock counter (call on policy (re)start)."""
        self._k = 0

    # -- operator input -------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Map an operator ``[vx, vy, wz]`` triplet to the skater ``[v, h]``.

        ``vx`` is the forward push speed (clipped to ``[0, v_max]``); ``wz`` is a
        normalized steer in ``[-1, 1]`` mapped to the heading target
        ``h = wz * h_max``. ``vy`` is unused (the skater has no lateral command).
        """
        self.set_skate(float(vx), float(np.clip(wz, -1.0, 1.0)) * self.h_max)
        # Keep the base-class velocity buffer in sync (harmless; lets any base
        # behaviour or logging see a consistent [vx, vy, wz]).
        self._velocity[:] = [self._v, float(vy), float(wz)]

    def set_skate(self, v: float, h: float) -> None:
        """Set the skater command directly: forward speed ``v`` and heading ``h``."""
        self._v = float(np.clip(v, 0.0, self.v_max))
        self._h = float(np.clip(h, -self.h_max, self.h_max))

    def set_command(self, values: Mapping[str, object]) -> None:
        if "skate_command" in values:
            vec = np.asarray(values["skate_command"], dtype=np.float64).reshape(-1)
            if vec.shape[0] != 2:
                raise ValueError("skate_command must be [v, h]")
            self.set_skate(float(vec[0]), float(vec[1]))
            return
        if all(name in values for name in ("vx", "vy", "wz")):
            self.set_velocity(float(values["vx"]), float(values["vy"]), float(values["wz"]))

    # -- phase + command ------------------------------------------------------

    def _phase(self) -> float:
        self._k += 1  # first observation corresponds to k = 1
        return (self._k * self.control_dt / self.cycle_time) % 1.0

    def command(self, raw: Mapping[str, object] | None = None) -> dict[str, np.ndarray]:
        if raw:
            for key in ("velocity_command", "cmd_vel", "base_velocity_command"):
                if key in raw:
                    vel = np.asarray(raw[key], dtype=np.float64).reshape(-1)
                    self.set_velocity(float(vel[0]), float(vel[1]), float(vel[2]))
                    break
            if "skate_command" in raw:
                self.set_command(raw)

        phase = self._phase()
        out = self.zeros()
        for term in self.metadata.command_spec.get("terms", []):
            name = str(term.get("name"))
            dim = int(term.get("dim", 0))
            base = name.split(".")[-1].lower()
            if base == "command" and dim == 2:
                out[name] = np.array([self._v, self._h], dtype=np.float64)
            elif base == "phase":
                out[name] = np.full(dim, phase, dtype=np.float64)
        return out
