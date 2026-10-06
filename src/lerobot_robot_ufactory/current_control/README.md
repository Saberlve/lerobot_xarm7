# GELLO current control

`control/current.py` implements fixed motor currents and passive constant-magnitude damping.
`control/runtime.py` owns the serial connection, safety checks and gripper requests.
`web/` retains the URDF viewer and supervised current-control session.
`models/` and `official_model.py` retain offline geometry/model generation only.

Default profile: `config/current_control/gello_A_working.yaml`.
J2: -50 mA; J4: +80 mA; J3/J7: 2 mA opposing velocity outside a 0.05 rad/s deadband.
Other joints: zero current. Startup ramp and electrical safeguards remain enabled.

Model-based gravity compensation and the previous experiments live on branch
`feature/gello-gravity-compensation`, not in this main-branch controller.
