"""Check browser model export against the dynamics model and offline packaging."""

from pathlib import Path

import numpy as np
import pytest

from lerobot_robot_ufactory.gravity_compensation.config import DeviceProfile
from lerobot_robot_ufactory.gravity_compensation.control.model import GravityModel
from lerobot_robot_ufactory.gravity_compensation.web.model_web import export_model

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/gravity/gello_A_working.yaml"


@pytest.mark.parametrize(
    "q",
    [
        [0] * 7,
        [0.2, -0.4, 0.7, 0.3, -0.5, 0.6, -0.2],
        [-0.7, 0.3, -0.2, 0.9, 0.8, -0.4, 0.5],
    ],
)
def test_exported_kinematics_match_pinocchio(q):
    profile = DeviceProfile(PROFILE)
    exported = export_model(profile)
    gravity = GravityModel(profile)
    full = np.empty(7)
    full[gravity.order] = q
    gravity.pin.forwardKinematics(gravity.model, gravity.data, full)
    frames = {exported["root"]: np.eye(4)}
    for index, joint in enumerate(exported["joints"]):
        origin = np.asarray(joint["origin"]).reshape(4, 4).T
        x, y, z = joint["axis"]
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(4)
        rotation[:3, :3] += np.sin(q[index]) * skew + (1 - np.cos(q[index])) * (skew @ skew)
        world = frames[joint["parent"]] @ origin @ rotation
        expected = gravity.data.oMi[gravity.model.getJointId(joint["name"])]
        np.testing.assert_allclose(world[:3, 3], expected.translation, atol=1e-12)
        np.testing.assert_allclose(world[:3, :3], expected.rotation, atol=1e-12)
        frames[joint["child"]] = world


def test_unified_html_is_self_contained():
    from lerobot_robot_ufactory.gravity_compensation.web.tuning_web import tuning_page

    html = tuning_page(DeviceProfile(PROFILE), "test-only-token").decode()
    assert "/*__THREE_JS__*/" not in html
    assert "/*__MODEL_DATA__*/" not in html
    assert "<script src=" not in html
    assert "MIT License" in html
    assert "gelloViewer" in html
    assert all(f'"name":"joint{index}"' in html for index in range(1, 8))
