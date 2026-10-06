"""URDF geometry and HTML assembly for the unified GELLO tuning page."""

import base64
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..models.geometry import geometry_triangles, vectors


def origin_matrix(node):
    origin = node.find("origin")
    if origin is None:
        return np.eye(4)
    roll, pitch, yaw = vectors(origin.get("rpy"), "0 0 0")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = vectors(origin.get("xyz"), "0 0 0")
    return result


def export_model(profile):
    """Bake visuals into link coordinates; preserve URDF joint transforms/axes."""
    root = ET.parse(profile.urdf).getroot()
    links = []
    for link in root.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            transform = origin_matrix(visual)
            vertices = geometry_triangles(visual.find("geometry"), profile.urdf.parent)
            vertices = vertices.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
            visuals.append(
                {
                    "name": visual.get("name", "visual"),
                    "positions": base64.b64encode(vertices.astype("<f4").tobytes()).decode("ascii"),
                    "motor": visual.find("geometry/mesh") is None,
                }
            )
        links.append({"name": link.get("name"), "visuals": visuals})
    joints = []
    for name in profile.joint_names:
        joint = root.find(f"joint[@name='{name}']")
        axis = vectors(joint.find("axis").get("xyz"), "0 0 1")
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            raise ValueError(f"Zero axis for {name}")
        joints.append(
            {
                "name": name,
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "origin": origin_matrix(joint).T.flatten().tolist(),
                "axis": (axis / norm).tolist(),
            }
        )
    children = {joint["child"] for joint in joints}
    roots = [link["name"] for link in links if link["name"] not in children]
    if len(roots) != 1:
        raise ValueError("Expected one fixed root link")
    return {
        "profile": profile.name,
        "urdf": profile.urdf.name,
        "urdf_sha256": hashlib.sha256(profile.urdf.read_bytes()).hexdigest(),
        "root": roots[0],
        "links": links,
        "joints": joints,
    }


def viewer_data(profile):
    data = export_model(profile)
    data.update(
        initial_q=[0.0] * 7,
        live={
            "profile_path": str(profile.path),
            "mode": "offline",
            "usb_serial": profile.serial,
            "baudrate": profile.baudrate,
            "joint_ids": list(profile.ids),
            "gripper_id": profile.gripper_id,
            "encoder_zero_rad": profile.zeros.tolist(),
            "model_signs": profile.signs.tolist(),
        },
    )
    return data


def render_html(data):
    """Assemble the single page with offline, read-only and compensation controls."""
    assets = Path(__file__).with_name("viewer_assets")
    template = (assets / "viewer.html").read_text()
    library = (assets / "three.min.js").read_text()
    license_text = (assets / "THREE_LICENSE.txt").read_text()
    library = f"/* Three.js 0.160.0\n{license_text}\n*/\n{library}"
    html = template.replace("/*__THREE_JS__*/", library.replace("</script", "<\\/script"))
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    html = html.replace("/*__MODEL_DATA__*/", payload)
    script = (assets / "live.js").read_text() + "\n" + (assets / "tuning.js").read_text()
    return html.replace("/*__LIVE_VIEW__*/", script)
