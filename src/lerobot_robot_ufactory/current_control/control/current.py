"""Fixed motor currents and passive constant-magnitude damping; no gravity model."""
import numpy as np
from ..config import tuning_slew, vector


class CurrentController:
    def __init__(self, profile, *, running_slew_a_s=None):
        self.profile = profile
        self.previous = np.zeros(7)
        self.running_slew_a_s = None if running_slew_a_s is None else tuning_slew(running_slew_a_s)

    def compute(self, position, velocity, dt, elapsed):
        p = self.profile
        if not 0 < dt <= p.state_timeout_s or not np.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("Invalid or stale control interval")
        position, velocity = np.asarray(position), np.asarray(velocity)
        if position.shape != (len(p.all_ids),) or velocity.shape != position.shape:
            raise ValueError("Unexpected state shape")
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("Non-finite encoder state")
        q, _ = p.model_state(position, velocity)  # Only for the model viewer.
        ramp = min(1.0, elapsed / p.ramp_s)
        constant = vector(p.constant_current_a, "constant_current_a")
        damping = p.constant_damping_a
        direction = np.where(np.abs(velocity[:7]) > p.damping_deadband_rad_s,
                             np.sign(velocity[:7]), 0.0)
        constant_current = ramp * constant
        damping_current = -ramp * damping * direction
        requested = constant_current + damping_current
        if not np.isfinite(requested).all():
            raise ValueError("Non-finite requested current")
        limited = np.clip(requested, -p.limits, p.limits)
        slew = p.slew
        if self.running_slew_a_s is not None and elapsed >= p.ramp_s:
            slew = np.asarray(tuning_slew(self.running_slew_a_s))
        output = self.previous + np.clip(limited - self.previous, -slew * dt, slew * dt)
        # Clear old damping on stop/reversal instead of briefly assisting motion.
        passive = (constant == 0) & (damping > 0)
        output[passive & ((direction == 0) | (output * direction > 0))] = 0.0
        self.previous = output.copy()
        return output, {
            "q": q.tolist(),
            "constant_current_a": constant_current.tolist(),
            "damping_current_a": damping_current.tolist(),
            "requested_a": requested.tolist(), "limited_a": limited.tolist(),
            "target_a": output.tolist(), "current_slew_a_s": slew.tolist(),
            "running_slew_a_s": (p.slew if self.running_slew_a_s is None else np.asarray(self.running_slew_a_s)).tolist(),
            "slew_error_a": (limited-output).tolist(),
            "saturated": bool(np.any(np.abs(requested) > p.limits)),
        }
