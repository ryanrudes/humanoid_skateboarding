"""Skater observation assembly: 5-frame term-major history + the heading term.

The vendored ``agibot_deploy.ObsBuilder`` assembles a **single** observation
frame by concatenating the ``actor_obs_spec`` terms. The HUSKY skater policy
instead consumes a **5-frame history** of an 8-term proprio frame, flattened
**term-major** (per term: all frames, oldest->newest), exactly as produced by
mjlab's ``ObservationGroupCfg(history_length=5, flatten_history_dim=True)`` and
reproduced in ``test_scene/sim.py``.

This subclass keeps the host's per-term assembly (it already does ``joint_pos``
default-relative, ``projected_gravity`` from the IMU quaternion, command lookup,
and per-term scaling) and adds only the two skater-specific pieces:

1. the ``heading`` term — the base world yaw (the host has no such term); on
   hardware there is no absolute world frame, so by default it is yaw-normalized
   to the heading captured at policy start (``heading(0) = 0``); and
2. the 5-frame **term-major** history ring buffer + flatten.

The result is the flat ``obs`` vector the skater ONNX expects (515 for X2, 395
for G1), returned under both ``"obs"`` and ``"actor_obs"`` so the host
``PolicyRunner`` feeds it to either input name.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

from agibot_deploy.command_provider import CommandProvider
from agibot_deploy.obs_builder import ObsBuilder
from agibot_deploy.metadata import PolicyMetadata


def yaw_from_quat_xyzw(quat_xyzw) -> float:
    """Yaw (heading) angle of an ``xyzw`` quaternion.

    Equivalent to ``atan2(forward_w.y, forward_w.x)`` with ``forward_w = R(q) ex``
    used by ``test_scene/sim.py``.
    """
    x, y, z, w = (float(v) for v in np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SkateObsAssembler(ObsBuilder):
    """Assemble the skater policy's flat history observation.

    Args:
      metadata: policy metadata; must carry a non-empty ``actor_obs_spec`` (the
        single-frame term layout) and an ``obs``/``actor_obs`` input contract
        whose dim is ``history_len * single_frame_dim``.
      command_provider: supplies the ``command`` and ``phase`` terms (a
        :class:`~husky_x2_deploy.skate_command.SkateCommandProvider`).
      heading_reference: ``"start"`` (default) yaw-normalizes ``heading`` to the
        value captured on the first frame after :meth:`reset` (heading(0) = 0;
        the safe open-loop choice on hardware). ``"absolute"`` uses the raw IMU
        yaw directly (matches ``test_scene/sim.py``, used for offline parity).
    """

    def __init__(
        self,
        metadata: PolicyMetadata,
        *,
        command_provider: CommandProvider | None = None,
        heading_reference: str = "start",
    ) -> None:
        super().__init__(metadata, command_provider=command_provider)
        if heading_reference not in {"start", "absolute"}:
            raise ValueError("heading_reference must be 'start' or 'absolute'")
        self.heading_reference = heading_reference

        self._term_dims = [int(term.dim) for term in metadata.actor_obs_spec]
        self._frame_dim = sum(self._term_dims)
        if self._frame_dim <= 0:
            raise ValueError("SkateObsAssembler requires a non-empty actor_obs_spec")

        contract = metadata.input_contract.get("obs")
        if contract is None:
            name = metadata.first_input_by_role("actor_obs")
            contract = metadata.input_contract.get(name) if name else None
        total_dim = int(contract.dim) if contract is not None and contract.dim else self._frame_dim
        if total_dim % self._frame_dim != 0:
            raise ValueError(
                f"obs input dim {total_dim} is not a multiple of the single-frame "
                f"dim {self._frame_dim}"
            )
        self.history_len = max(1, total_dim // self._frame_dim)

        self._yaw0: float | None = None
        self._buffer: deque[np.ndarray] = deque(maxlen=self.history_len)
        self.reset()

    def reset(self) -> None:
        """Clear the history and heading reference; reset the command provider.

        Call on policy (re)start so the first observation is one real frame
        backfilled by zeros (matching ``sim.py``'s zero-initialized deque and
        mjlab's ``CircularBuffer`` at episode start).
        """
        self._yaw0 = None
        self._buffer = deque(
            (np.zeros(self._frame_dim, dtype=np.float64) for _ in range(self.history_len)),
            maxlen=self.history_len,
        )
        reset = getattr(self.command_provider, "reset", None)
        if callable(reset):
            reset()

    def _heading(self, raw) -> float:
        if "heading" in raw:  # allow an explicit override from the caller
            return float(np.asarray(raw["heading"], dtype=np.float64).reshape(-1)[0])
        if "base_quat_xyzw" not in raw:
            raise KeyError("SkateObsAssembler needs 'heading' or 'base_quat_xyzw' to derive heading")
        yaw = yaw_from_quat_xyzw(raw["base_quat_xyzw"])
        if self.heading_reference == "absolute":
            return yaw
        if self._yaw0 is None:
            self._yaw0 = yaw
        return _wrap_to_pi(yaw - self._yaw0)

    def build_actor_obs(self, raw) -> np.ndarray:
        # 1) inject the skater-specific heading term, then let the host assemble
        #    the single proprio frame (command/phase come from the provider).
        raw_frame = dict(raw)
        raw_frame["heading"] = np.array([self._heading(raw)], dtype=np.float64)
        frame = super().build_actor_obs(raw_frame)
        if frame.shape[0] != self._frame_dim:
            raise ValueError(
                f"assembled frame dim {frame.shape[0]} != expected {self._frame_dim}"
            )

        # 2) push into the oldest->newest ring buffer and flatten term-major.
        self._buffer.append(np.asarray(frame, dtype=np.float64))
        history = np.stack(self._buffer, axis=0)  # (history_len, frame_dim)
        parts = []
        start = 0
        for dim in self._term_dims:
            parts.append(history[:, start:start + dim].reshape(-1))  # [f0..fH-1] per term
            start += dim
        return np.concatenate(parts).astype(np.float32, copy=False)
