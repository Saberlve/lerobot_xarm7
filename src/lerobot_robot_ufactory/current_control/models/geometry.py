"""URDF visual geometry for the browser model; no hardware or plotting imports."""

import numpy as np

from .mesh_mass import read_binary_stl


def vectors(text, default):
    return np.array([float(x) for x in (text or default).split()])


def geometry_triangles(node, directory):
    mesh = node.find("mesh")
    if mesh is not None:
        path = directory / mesh.attrib["filename"]
        return read_binary_stl(path, meters_per_unit=1) * vectors(mesh.get("scale"), "1 1 1")
    box = node.find("box")
    if box is not None:
        v = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ]
            )
            * vectors(box.get("size"), "1 1 1")
            / 2
        )
        return v[
            np.array(
                [
                    [0, 2, 1],
                    [0, 3, 2],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ]
            )
        ]
    cylinder = node.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.get("radius"))
        half = float(cylinder.get("length")) / 2
        angles = np.linspace(0, 2 * np.pi, 17)
        low = np.array([[radius * np.cos(a), radius * np.sin(a), -half] for a in angles])
        high = low + [0, 0, 2 * half]
        faces = []
        for i in range(16):
            faces.extend(
                [
                    [low[i], low[i + 1], high[i]],
                    [low[i + 1], high[i + 1], high[i]],
                    [[0, 0, -half], low[i + 1], low[i]],
                    [[0, 0, half], high[i], high[i + 1]],
                ]
            )
        return np.asarray(faces)
    raise ValueError("This viewer supports binary STL, box and cylinder visuals")
