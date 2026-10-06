"""Generate a fixed-base 7-axis URDF from explicit link-frame measurements."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from ..config import vector


def build_model(source, output):
    d = yaml.safe_load(Path(source).read_text())
    if d.get("version") != 1 or len(d.get("joints", [])) != 7:
        raise ValueError("Expected seven measured joints")
    if not isinstance(d.get("verified"), bool):
        raise ValueError("An explicit verified boolean is required")
    root = ET.Element("robot", name="xarm7_gello_leader")
    if not d["verified"]:
        root.append(
            ET.Comment(" UNVERIFIED_GEOMETRY: offline fixture, not an assembled-arm model ")
        )
    ET.SubElement(root, "link", name="base")

    def numbers(values):
        return " ".join(f"{x:.12g}" for x in values)

    for i, j in enumerate(d["joints"], 1):
        xyz = vector(j["origin_xyz_m"], "origin_xyz_m", 3)
        rpy = vector(j["origin_rpy_rad"], "origin_rpy_rad", 3)
        axis = vector(j["axis"], "axis", 3)
        if not np.isclose(np.linalg.norm(axis), 1):
            raise ValueError("Joint axes must be normalized")
        com = vector(j["com_xyz_m"], "com_xyz_m", 3)
        mass = vector([j["mass_kg"]], "mass_kg", 1, positive=True)[0]
        inertia = vector(j["inertia_diagonal_kg_m2"], "inertia_diagonal_kg_m2", 3, positive=True)
        if 2 * max(inertia) > sum(inertia) + 1e-12:
            raise ValueError("Inertia violates the triangle inequality")
        link = ET.SubElement(root, "link", name=f"link{i}")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(mass))
        ET.SubElement(
            inertial,
            "inertia",
            ixx=str(inertia[0]),
            iyy=str(inertia[1]),
            izz=str(inertia[2]),
            ixy="0",
            ixz="0",
            iyz="0",
        )
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=numbers(com))
        ET.SubElement(ET.SubElement(visual, "geometry"), "sphere", radius="0.015")
        joint = ET.SubElement(root, "joint", name=f"joint{i}", type="revolute")
        ET.SubElement(joint, "parent", link="base" if i == 1 else f"link{i - 1}")
        ET.SubElement(joint, "child", link=f"link{i}")
        ET.SubElement(joint, "origin", xyz=numbers(xyz), rpy=numbers(rpy))
        ET.SubElement(joint, "axis", xyz=numbers(axis))
        ET.SubElement(
            joint,
            "limit",
            lower="-6.28318530718",
            upper="6.28318530718",
            effort="0.52",
            velocity="6",
        )
    ET.indent(root, space="  ")
    with open(output, "x") as stream:
        stream.write(ET.tostring(root, encoding="unicode") + "\n")
