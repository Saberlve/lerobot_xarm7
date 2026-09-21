"""Stage C dynamic external-torque estimation. Pure functions, no I/O.

C1 is an architecture placeholder with deliberately unchanged math:

    external_torque = measured_effort - baseline

position/velocity already enter the interface so the C2 terms
(gravity(q), friction(q_dot)) slot in without touching call sites.
"""

from .arm_feedback import vector7


class DynamicExternalTorqueEstimator:
    """Maps a synchronized (q, qd, measured_effort) snapshot to tau_external."""

    def __init__(self, baseline):
        self._baseline = vector7(baseline, "baseline")

    def estimate(self, position, velocity, measured_effort):
        """Return estimated external torque, shape (7,), same units as effort."""
        vector7(position, "joint_position")
        vector7(velocity, "joint_velocity")
        effort = vector7(measured_effort, "measured_effort")
        return effort - self._baseline
