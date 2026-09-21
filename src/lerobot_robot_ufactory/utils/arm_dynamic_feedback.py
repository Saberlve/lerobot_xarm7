"""Stage C dynamic external-torque estimation. Pure functions, no I/O.

C1 is an architecture placeholder with deliberately unchanged math:

    external_torque = measured_effort - baseline

position/velocity already enter the interface so the C2 terms
(gravity(q), friction(q_dot)) slot in without touching call sites.
"""

import time

from .arm_external_torque import BaselineExternalTorqueEstimator


class DynamicExternalTorqueEstimator:
    """Maps a synchronized (q, qd, measured_effort) snapshot to tau_external."""

    def __init__(self, baseline):
        self._estimator = BaselineExternalTorqueEstimator(baseline)

    def estimate(self, position, velocity, measured_effort):
        """Return estimated external torque, shape (7,), same units as effort."""
        return self._estimator.update(
            position,
            velocity,
            position,
            measured_effort,
            time.monotonic_ns(),
        ).external_torque
