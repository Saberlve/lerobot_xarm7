"""Static dynamics only: no follower offsets or acceleration feedforward."""

import numpy as np

from ..config import tuning_slew, vector


class GravityModel:
    def __init__(self, profile):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Install the gravity extra (the Pinocchio package is named 'pin')"
            ) from exc
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(profile.urdf))
        self.data = self.model.createData()
        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError("URDF must have seven scalar revolute joints and a fixed base")
        for inertia in self.model.inertias[1:]:
            if (
                not np.isfinite(inertia.mass)
                or inertia.mass <= 0
                or not np.isfinite(inertia.lever).all()
            ):
                raise ValueError("Every moving link needs finite positive mass and a finite COM")
            tensor = np.asarray(inertia.inertia)
            if not np.isfinite(tensor).all() or np.min(np.linalg.eigvalsh(tensor)) <= 0:
                raise ValueError("URDF inertia tensors must be finite and positive definite")
        names = list(self.model.names)[1:]
        if set(names) != set(profile.joint_names):
            raise ValueError("URDF joints and profile joint_names do not match")
        self.order = np.array([names.index(name) for name in profile.joint_names])
        self.model.gravity.linear = profile.gravity

    def gravity(self, q):
        full = np.empty(7)
        full[self.order] = q
        return np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, full))[
            self.order
        ].copy()

    def potential(self, q):
        full = np.empty(7)
        full[self.order] = q
        return float(self.pin.computePotentialEnergy(self.model, self.data, full))


class CurrentController:
    def __init__(self, profile, model, *, running_slew_a_s=None):
        self.profile = profile
        self.model = model
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
        q, dq = p.model_state(position, velocity)
        constant = getattr(p, "constant_current_a", [None] * 7)
        gravity = self.model.gravity(q) if any(v is None for v in constant) else np.zeros(7)
        ramp = min(1.0, elapsed / p.ramp_s)
        gains = np.full(7, p.gain, dtype=float)
        joint_gains = getattr(p, "joint_gains", None)
        if joint_gains is not None:
            gains = vector(joint_gains, "joint_gains", nonnegative=True).copy()
        for joint in (5, 6):
            joint_gain = getattr(p, f"j{joint}_gain", None)
            if joint_gain is not None:
                if not np.isfinite(joint_gain) or joint_gain < 0:
                    raise ValueError(f"J{joint} gain must be finite and nonnegative")
                gains[joint - 1] = joint_gain
        gravity_current = p.signs * ramp * gains * gravity / p.nm_per_amp
        damping_current = -p.signs * ramp * p.damping * dq / p.nm_per_amp
        constant = getattr(p, "constant_current_a", [None] * 7)
        constant_current = np.zeros(7)
        for i, value in enumerate(constant):
            if value is not None:
                # A constant target replaces BOTH model and damping terms.
                gravity_current[i] = damping_current[i] = 0.0
                constant_current[i] = ramp * value
        fixed_damping = np.asarray(getattr(p, "constant_damping_a", np.zeros(7)))
        deadband = getattr(p, "damping_deadband_rad_s", 0.05)
        direction = np.where(np.abs(velocity[:7]) > deadband, np.sign(velocity[:7]), 0.0)
        friction_current = -ramp * fixed_damping * direction
        damping_current += friction_current
        requested = gravity_current + damping_current + constant_current
        if not np.isfinite(requested).all():
            raise ValueError("Non-finite model torque")
        limited = np.clip(requested, -p.limits, p.limits)
        slew = p.slew
        # Continuous-mode running rates never bypass the original startup softening.
        if self.running_slew_a_s is not None and elapsed >= p.ramp_s:
            slew = np.asarray(tuning_slew(self.running_slew_a_s))
        output = self.previous + np.clip(limited - self.previous, -slew * dt, slew * dt)
        # Pure damping must not push along motion after reversal or at rest.
        # Unload old damping immediately, then slew into the new opposing sign.
        for i, value in enumerate(constant):
            if value == 0 and fixed_damping[i] > 0:
                if direction[i] == 0 or output[i] * direction[i] > 0:
                    output[i] = 0.0
        self.previous = output.copy()
        return output, {
            "q": q.tolist(),
            "gravity_nm": gravity.tolist(),
            "gravity_gains": gains.tolist(),
            "gravity_current_a": gravity_current.tolist(),
            "damping_current_a": damping_current.tolist(),
            "constant_current_a": constant_current.tolist(),
            "requested_a": requested.tolist(),
            "limited_a": limited.tolist(),
            "target_a": output.tolist(),
            "current_slew_a_s": slew.tolist(),
            "running_slew_a_s": (p.slew if self.running_slew_a_s is None else np.asarray(self.running_slew_a_s)).tolist(),
            "slew_error_a": (limited - output).tolist(),
            "saturated": bool(np.any(np.abs(requested) > p.limits)),
        }
