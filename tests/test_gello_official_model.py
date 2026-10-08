"""Analytical mass-property and offline model integration checks."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from lerobot_robot_ufactory.current_control.config import DeviceProfile
from lerobot_robot_ufactory.current_control.models.mesh_mass import (
    combine_properties,
    read_binary_stl,
    solid_properties,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "config/current_control/model/estimate.json"

# The offline builder is a repository tool, outside the installed runtime package.
_builder_spec = importlib.util.spec_from_file_location(
    "current_control.official_model",
    ROOT / "src/lerobot_robot_ufactory/current_control/official_model.py",
)
builder = importlib.util.module_from_spec(_builder_spec)
_builder_spec.loader.exec_module(builder)


@pytest.mark.parametrize("index,axis", [(0, 1), (2, 0), (5, 1)])
def test_corrected_motors_extend_outside_opposite_stl_face(index, axis):
    spec = json.loads(SPEC.read_text())
    item = spec["links"][index]
    mount = item["next_motor_mount"]
    back = np.asarray(mount["back_center_mm"]) * 0.001
    normal = np.asarray(mount["outward_normal"])
    triangles = read_binary_stl(SPEC.parent / "source_meshes" / (item["mesh"] + ".STL"))
    # Check against the actual STL surface, not just duplicated configuration values.
    on_face = np.all(np.abs(triangles[:, :, axis] - back[axis]) < 1e-8, axis=1)
    assert on_face.any()
    faces = triangles[on_face]
    oriented_area = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]).sum(axis=0)
    assert oriented_area @ normal > 0
    case, horn, output, _ = builder.motor_geometry(mount, spec["motor_mass_kg"])
    corners = np.array([[x, y, z] for x in [-.010, .010] for y in [-.017, .017] for z in [-.0115, .0115]])
    world_corners = corners @ case[:3, :3].T + case[:3, 3]
    assert np.min((world_corners - back) @ normal) >= -1e-12
    assert (horn[:3, 3] - back) @ normal > 0
    assert (output[:3, 3] - back) @ normal > 0
    assert spec["motor_side_corrections"]["shaft_direction_verified"] is False


def tetrahedron():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    return v[[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]


def test_tetrahedron_analytical_properties_and_translation():
    for offset in (np.zeros(3), np.array([7, -4, 12])):
        props = solid_properties(tetrahedron() + offset, density_kg_m3=6)
        assert props["mass_kg"] == pytest.approx(1)
        assert props["volume_m3"] == pytest.approx(1 / 6)
        np.testing.assert_allclose(props["com_xyz_m"], offset + 0.25)
        tensor = np.full((3, 3), 1 / 80)
        np.fill_diagonal(tensor, 3 / 40)
        np.testing.assert_allclose(props["inertia_kg_m2"], tensor, atol=1e-13)


def test_density_scaling_changes_mass_and_tensor_but_not_com():
    a = solid_properties(tetrahedron(), mass_scale=1)
    b = solid_properties(tetrahedron(), mass_scale=0.3)
    assert b["mass_kg"] == pytest.approx(a["mass_kg"] * 0.3)
    np.testing.assert_allclose(b["com_xyz_m"], a["com_xyz_m"])
    np.testing.assert_allclose(b["inertia_kg_m2"], np.array(a["inertia_kg_m2"]) * 0.3)


@pytest.mark.parametrize("bad", [tetrahedron()[:-1], tetrahedron()[:, ::-1]])
def test_open_or_inverted_mesh_rejected(bad):
    with pytest.raises(ValueError):
        solid_properties(bad)


def test_parallel_axis_theorem():
    m, c, inertia = combine_properties([(1, [-1, 0, 0], np.eye(3)), (1, [1, 0, 0], np.eye(3))])
    assert m == 2
    np.testing.assert_allclose(c, 0)
    np.testing.assert_allclose(inertia, np.diag([2, 4, 4]))


def test_frame_inverse_and_gimbal_rpy_roundtrip():
    for normal, x in [
        ([1, 0, 0], [0, 0, 1]),
        ([0, 0, 1], [1, 0, 0]),
        ([0, -0.732, 0.681], [1, 0, 0]),
    ]:
        t = builder.frame([1, 2, 3], normal, x)
        np.testing.assert_allclose(builder.inverse(t) @ t, np.eye(4), atol=1e-15)
        pin = pytest.importorskip("pinocchio")

        np.testing.assert_allclose(pin.rpy.rpyToMatrix(*builder.rpy(t[:3, :3])), t[:3, :3], atol=1e-15)




def test_changed_source_mesh_rejected(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec["mesh_directory"] = str(SPEC.parent / "source_meshes")
    spec["mesh_sha256"]["L1"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.build_estimate(path, tmp_path / "model")


def test_shipped_profiles_use_official_model_and_visual_meshes():
    pin = pytest.importorskip("pinocchio")

    for name in ["gello_A_working"]:
        p = DeviceProfile(ROOT / f"config/current_control/{name}.yaml")
        assert p.urdf.name == "xarm7_gello.urdf"
        p.validate_live()
        model = pin.buildModelFromUrdf(str(p.urdf))
        visuals = pin.buildGeomFromUrdf(
            model, str(p.urdf), pin.GeometryType.VISUAL, package_dirs=[str(p.urdf.parent)]
        )
        assert visuals.ngeoms == 25


def test_recording_config_retains_feedback_and_parses_gravity():
    import draccus
    from lerobot.teleoperators import TeleoperatorConfig

    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
        GelloTeleopConfig,
    )

    original = yaml.safe_load((ROOT / "config/gello/xarm7_gello_record_config.yaml").read_text())
    configured = yaml.safe_load(
        (ROOT / "config/gello/xarm7_gello_record_current_config.yaml").read_text()
    )
    parsed = draccus.decode(TeleoperatorConfig, configured["teleop"])
    assert isinstance(parsed, GelloTeleopConfig)
    assert parsed.current_control.enabled
    p = DeviceProfile(ROOT / parsed.current_control.profile_path)
    assert parsed.port == p.port
    assert tuple(parsed.joint_ids) == tuple(p.ids)
    configured["teleop"].pop("current_control")
    assert configured["robot"] == original["robot"]
    assert configured["dataset"] == original["dataset"]


def test_regenerated_working_model_matches_shipped_dynamics(tmp_path):
    out = tmp_path / "regenerated"
    builder.build_estimate(SPEC, out)
    regenerated = ET.parse(out / "xarm7_gello.urdf").getroot()
    working = DeviceProfile(ROOT / "config/current_control/gello_A_working.yaml")
    shipped = ET.parse(working.urdf).getroot()
    # Visual mesh relative paths depend on output location; dynamics do not.
    for tag in ("joint", "link"):
        before = {node.get("name"): node for node in shipped.findall(tag)}
        after = {node.get("name"): node for node in regenerated.findall(tag)}
        assert before.keys() == after.keys()
        for name in before:
            if tag == "joint":
                assert ET.tostring(before[name]) == ET.tostring(after[name])
            else:
                a, b = before[name].find("inertial"), after[name].find("inertial")
                assert (None if a is None else ET.tostring(a)) == (None if b is None else ET.tostring(b))


def test_default_teleop_configuration_parses_saved_gains_and_running_rates():
    import draccus
    from lerobot.teleoperators import TeleoperatorConfig

    from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig

    configured = yaml.safe_load((ROOT / "config/gello/xarm7_gello_teleop_current.yaml").read_text())
    parsed = draccus.decode(TeleoperatorConfig, configured["teleop"])
    assert isinstance(parsed, GelloTeleopConfig)
    loaded = parsed.current_control.load_profile()
    assert loaded.constant_current_a == loaded.data["constant_current_a"]
    assert parsed.current_control.running_current_slew_a_s is None
    assert loaded.running_current_slew_a_s == loaded.data["running_current_slew_a_s"]
    assert loaded.constant_damping_a.tolist() == loaded.data["constant_damping_a"]
