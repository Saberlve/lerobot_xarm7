# GELLO current control

`control/current.py` implements fixed motor currents and passive constant-magnitude damping.
`control/runtime.py` owns the serial connection, safety checks and gripper requests.
`web/` retains the URDF viewer and supervised current-control session.
`models/` and `official_model.py` retain offline geometry/model generation only.

Default profile: `config/current_control/gello_A_working.yaml`.
Current profile: J2 -40 mA; J4 +80 mA; J3/J7 2 mA opposing velocity outside a 0.05 rad/s deadband.
Other joints: zero current. Startup ramp and electrical safeguards remain enabled.

The web page supports per-joint signed fixed currents and nonnegative damping.
Apply changes to the current session while stopped or running; the serial owner
updates both targets together and output follows the existing slew limits.
Web Apply automatically writes current/damping and running slew settings (and
the descriptive note) to the loaded profile. Running slew edits also auto-save.
Page reloads and new teleoperation sessions load those values. Default teleop and
recording configs use shared profile defaults; an explicit teleop running-slew
override remains supported. Failed writes report unsaved settings and allow retry.
Stop web current control before teleoperation.

Model-based gravity compensation and the previous experiments live on branch
`feature/gello-gravity-compensation`, not in this main-branch controller.
