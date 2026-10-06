"""Build an explicitly estimated xArm7 GELLO from the official printed parts.

This is an offline builder. It does not import the serial transport. Mesh mass
properties are exact for the stated uniform-density approximation. Assembly
frames, motor COMs, fasteners and the fixed trigger pose remain estimates.
"""

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from lerobot_robot_ufactory.gravity_compensation.models.mesh_mass import (
    combine_properties, read_binary_stl, solid_properties,
)


def vector3(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("Expected a finite vector with three elements")
    return a


def frame(point_mm, normal, x_hint):
    z = vector3(normal)
    if np.linalg.norm(z) < 1e-10:
        raise ValueError("Zero frame normal")
    z /= np.linalg.norm(z)
    x = vector3(x_hint)
    x -= z * (z @ x)
    if np.linalg.norm(x) < 1e-10:
        raise ValueError("Frame x direction is parallel to its normal")
    x /= np.linalg.norm(x)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    result[:3, 3] = vector3(point_mm) * 0.001
    return result


def inverse(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def rotation_z(angle):
    result = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    result[:2, :2] = [[c, -s], [s, c]]
    return result


def rpy(rotation):
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:
        return [math.atan2(-rotation[1, 2], rotation[1, 1]), pitch, 0.0]
    return [
        math.atan2(rotation[2, 1], rotation[2, 2]),
        pitch,
        math.atan2(rotation[1, 0], rotation[0, 0]),
    ]


def transformed(properties, transform):
    rotation, offset = transform[:3, :3], transform[:3, 3]
    return (
        properties["mass_kg"],
        rotation @ np.asarray(properties["com_xyz_m"]) + offset,
        rotation @ np.asarray(properties["inertia_kg_m2"]) @ rotation.T,
    )


def motor_geometry(mount, mass):
    """Nominal 20 x 34 x 26 mm envelope; 23 mm case plus 3 mm horn."""
    direction = vector3(mount["long_axis"])
    direction /= np.linalg.norm(direction)
    normal = vector3(mount["outward_normal"])
    normal /= np.linalg.norm(normal)
    if abs(direction @ normal) > 1e-5:
        raise ValueError("Motor long axis must be perpendicular to shaft")
    back = frame(mount["back_center_mm"], normal, np.cross(direction, normal))
    case = back.copy()
    case[:3, 3] += back[:3, :3] @ [0, 0, 0.0115]
    horn = back.copy()
    horn[:3, 3] += back[:3, :3] @ [0, 0.0075, 0.0245]
    output = back.copy()
    output[:3, 3] += back[:3, :3] @ [0, 0.0075, 0.026]
    center = back[:3, 3] + back[:3, :3] @ [0, 0, 0.013]
    dims = np.array([0.020, 0.034, 0.026])
    inertia = np.diag(mass * (np.sum(dims**2) - dims**2) / 12)
    props = (mass, center, back[:3, :3] @ inertia @ back[:3, :3].T)
    return case, horn, output, props


def build_estimate(spec_path, output_dir):
    source = Path(spec_path).resolve()
    spec = json.loads(source.read_text())
    if spec.get("version") != 1 or len(spec.get("links", [])) != 7:
        raise ValueError("Expected an estimated seven-link model specification")
    density = spec["pla_density_kg_m3"]
    mass_scale = spec["printed_mass_scale"]
    motor_mass = float(spec["motor_mass_kg"])
    extra_mass = float(spec["fasteners_and_wire_kg_per_link"])
    if not np.isfinite([motor_mass, extra_mass]).all() or motor_mass <= 0 or extra_mass < 0:
        raise ValueError("Invalid hardware mass assumptions")
    meshes = (source.parent / spec["mesh_directory"]).resolve()
    out = Path(output_dir).resolve()
    mesh_relative_dir = Path(os.path.relpath(meshes, out)).as_posix()
    expected = {"base", "L1", "L2", "L3", "L4", "L5", "L6", "handle", "trigger"}
    if set(spec["mesh_sha256"]) != expected:
        raise ValueError("Expected the complete official nine-mesh inventory")
    properties = {}
    for name in sorted(expected):
        path = meshes / (name + ".STL")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec["mesh_sha256"][name]:
            raise ValueError(f"Upstream mesh hash mismatch: {name}")
        properties[name] = solid_properties(
            read_binary_stl(path),
            density,
            spec.get("part_mass_scales", {}).get(name, mass_scale),
            weld_grid_m=spec["weld_grid_m"],
        )
        properties[name]["sha256"] = digest
    root = ET.Element("robot", name="xarm7_gello_official_estimate")
    root.append(
        ET.Comment(
            " UNVERIFIED_GEOMETRY: reconstructed assembly and estimated masses; offline only "
        )
    )
    root.append(
        ET.Comment(
            " Printed mesh coordinates are millimeters. All URDF coordinates and inertias are SI. "
        )
    )
    report = {
        "status": "ESTIMATE_NOT_COMMISSIONED",
        "upstream_revision": spec["upstream_revision"],
        "spec_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "assumptions": spec["assumptions"],
        "pla_density_kg_m3": density,
        "printed_mass_scale": mass_scale,
        "parts_in_original_stl_frames": properties,
        "links": [],
        "joints": [],
        "visuals": [],
    }

    def numbers(values):
        return " ".join(f"{float(x):.12g}" for x in values)

    def origin(element, transform):
        ET.SubElement(
            element, "origin", xyz=numbers(transform[:3, 3]), rpy=numbers(rpy(transform[:3, :3]))
        )

    def visual(link, name, transform, kind, value):
        node = ET.SubElement(link, "visual", name=name)
        origin(node, transform)
        geom = ET.SubElement(node, "geometry")
        if kind == "mesh":
            ET.SubElement(
                geom, "mesh", filename=f"{mesh_relative_dir}/{value}.STL", scale="0.001 0.001 0.001"
            )
        elif kind == "box":
            ET.SubElement(geom, "box", size=numbers(value))
        else:
            ET.SubElement(geom, "cylinder", radius=str(value[0]), length=str(value[1]))
        material = ET.SubElement(node, "material", name="pla" if kind == "mesh" else "servo")
        ET.SubElement(
            material, "color", rgba="0.2 0.55 0.8 1" if kind == "mesh" else "0.25 0.25 0.28 1"
        )
        report["visuals"].append(
            {
                "link": link.attrib["name"],
                "name": name,
                "kind": kind,
                "value": value,
                "transform": transform.tolist(),
            }
        )

    def add_motor(link, mount, parent_transform, name):
        case, horn, output, (m, c, tensor) = motor_geometry(mount, motor_mass)
        visual(link, name + "_case", parent_transform @ case, "box", [0.020, 0.034, 0.023])
        visual(link, name + "_horn", parent_transform @ horn, "cylinder", [0.008, 0.003])
        rot = parent_transform[:3, :3]
        return parent_transform @ output, (
            m,
            rot @ c + parent_transform[:3, 3],
            rot @ tensor @ rot.T,
        )

    # Native base Y is up; world Z is up. The fixed motor1/base are not moving masses.
    base = ET.SubElement(root, "link", name="base")
    base_transform = frame([0, 0, 0], [0, -1, 0], [1, 0, 0])
    visual(base, "base_print", base_transform, "mesh", "base")
    next_joint, _ = add_motor(base, spec["base_motor_mount"], base_transform, "motor1")
    for index, item in enumerate(spec["links"], 1):
        joint_pose = next_joint @ rotation_z(math.radians(float(item["zero_twist_deg"])))
        joint = ET.SubElement(root, "joint", name=f"joint{index}", type="revolute")
        ET.SubElement(joint, "parent", link="base" if index == 1 else f"link{index - 1}")
        ET.SubElement(joint, "child", link=f"link{index}")
        origin(joint, joint_pose)
        ET.SubElement(joint, "axis", xyz="0 0 1")
        # Broad computational limits are not collision-safe hardware limits.
        ET.SubElement(
            joint,
            "limit",
            lower=str(-2 * math.pi),
            upper=str(2 * math.pi),
            effort="0.52",
            velocity="6",
        )
        report["joints"].append(
            {"name": f"joint{index}", "transform": joint_pose.tolist(), "axis": [0, 0, 1]}
        )
        link = ET.SubElement(root, "link", name=f"link{index}")
        incoming = frame(**item["incoming_frame"])
        mesh_transform = inverse(incoming)
        name = item["mesh"]
        visual(link, name + "_print", mesh_transform, "mesh", name)
        components = [transformed(properties[name], mesh_transform)]
        next_joint, motor = add_motor(
            link, item["next_motor_mount"], mesh_transform, f"motor{index + 1}"
        )
        components.append(motor)
        if index == 7:
            trigger_frame = frame(**spec["trigger_incoming_frame"])
            trigger_transform = (
                next_joint
                @ rotation_z(math.radians(spec["trigger_park_deg"]))
                @ inverse(trigger_frame)
            )
            visual(link, "trigger_print", trigger_transform, "mesh", "trigger")
            components.append(transformed(properties["trigger"], trigger_transform))
        m, c, tensor = combine_properties(components)
        if extra_mass:
            # Explicit lumped estimate at component COM, not an invented cable CAD model.
            components.append((extra_mass, c.copy(), np.eye(3) * extra_mass * 0.005**2 * 2 / 5))
            m, c, tensor = combine_properties(components)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(c), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{m:.12g}")
        ET.SubElement(
            inertial,
            "inertia",
            **{
                key: f"{tensor[a, b]:.12g}"
                for key, a, b in [
                    ("ixx", 0, 0),
                    ("iyy", 1, 1),
                    ("izz", 2, 2),
                    ("ixy", 0, 1),
                    ("ixz", 0, 2),
                    ("iyz", 1, 2),
                ]
            },
        )
        report["links"].append(
            {
                "name": f"link{index}",
                "mass_kg": m,
                "com_xyz_m": c.tolist(),
                "inertia_kg_m2": tensor.tolist(),
                "motor_id_on_parent_body": index + 1,
            }
        )
    report["moving_mass_kg"] = sum(link["mass_kg"] for link in report["links"])
    ET.indent(root, space="  ")
    out.mkdir(parents=True, exist_ok=False)
    urdf = out / "xarm7_gello.urdf"
    urdf.write_text(ET.tostring(root, encoding="unicode") + "\n")
    report["urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    (out / "mass_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory; never overwrites an existing model",
    )
    args = parser.parse_args()
    report = build_estimate(args.spec, args.output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "moving_mass_kg": report["moving_mass_kg"],
                "urdf_sha256": report["urdf_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
