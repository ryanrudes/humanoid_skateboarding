"""Minimal quaternion helpers — a drop-in for the handful of
``pybullet_utils.transformations`` functions this repo used, so the package no
longer depends on pybullet (a ~80 MB physics engine pulled in only for this math).

Quaternions use the same convention as the original library: ``[x, y, z, w]``
(vector part first, scalar last — the order MuJoCo and scipy also call "xyzw").
The implementations mirror ``transformations.py`` (BSD-3-Clause, Christoph Gohlke)
for the specific functions used here, so results are numerically identical to the
old ``pybullet_utils.transformations`` calls.

Only ``quaternion_slerp`` is on a live code path (``motion_loader_g1.py``); the
others back commented-out helpers in ``pose3d.py``/``motion_util.py`` and are kept
so those references resolve to this module rather than pybullet.
"""

import math

import numpy as np

# Matches transformations.py: numpy.finfo(float).eps * 4.0
_EPS = np.finfo(float).eps * 4.0


def vector_norm(data, axis=None):
    """Return the Euclidean norm of an ndarray (along ``axis`` if given)."""
    data = np.array(data, dtype=np.float64, copy=True)
    if axis is None:
        return math.sqrt(np.dot(data.ravel(), data.ravel()))
    return np.sqrt(np.sum(data * data, axis=axis))


def unit_vector(data, axis=None):
    """Return ndarray normalized by Euclidean length, along ``axis`` if given."""
    data = np.array(data, dtype=np.float64, copy=True)
    if axis is None and data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


def quaternion_about_axis(angle, axis):
    """Return the quaternion [x, y, z, w] for a rotation of ``angle`` about ``axis``."""
    quaternion = np.zeros((4,), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle / 2.0) / qlen
    quaternion[3] = math.cos(angle / 2.0)
    return quaternion


def quaternion_conjugate(quaternion):
    """Return the conjugate (-x, -y, -z, w) of a quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[:3], q[:3])
    return q


def quaternion_inverse(quaternion):
    """Return the inverse of a quaternion (conjugate / |q|^2; conjugate if unit)."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    return quaternion_conjugate(q) / np.dot(q, q)


def quaternion_multiply(quaternion1, quaternion0):
    """Return the Hamilton product ``quaternion1 * quaternion0`` (both [x, y, z, w])."""
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0), dtype=np.float64)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Spherical linear interpolation between two quaternions ([x, y, z, w])."""
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0
