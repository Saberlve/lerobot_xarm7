"""Mass properties of closed binary STL solids, with explicit length units.

Integrate signed tetrahedra, including the complete inertia tensor. A uniform
mass scale represents an assumed effective density, not a slicer infill ratio.
No slicing or claims about the actual print are made. Optional grid welding
is explicit and reported; missing surfaces are never filled automatically.
"""

import struct
from pathlib import Path

import numpy as np


def read_binary_stl(path, meters_per_unit=0.001):
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("STL length scale must be positive and finite")
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError("Truncated binary STL")
    count = struct.unpack_from("<I", raw, 80)[0]
    if count == 0 or len(raw) != 84 + 50 * count:
        raise ValueError("Expected an exact binary STL triangle payload")
    records = np.frombuffer(
        raw,
        dtype=np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]),
        offset=84,
        count=count,
    )
    triangles = records["vertices"].astype(float) * meters_per_unit
    if not np.isfinite(triangles).all():
        raise ValueError("Non-finite STL vertices")
    return triangles


def solid_properties(triangles, density_kg_m3=1240.0, mass_scale=1.0, weld_grid_m=0.0):
    """Return SI properties about COM; reject open/inconsistently wound meshes."""
    t = np.asarray(triangles, dtype=float)
    if t.ndim != 3 or t.shape[1:] != (3, 3) or not len(t) or not np.isfinite(t).all():
        raise ValueError("Expected finite triangles with shape (N, 3, 3)")
    if not np.isfinite(density_kg_m3) or density_kg_m3 <= 0:
        raise ValueError("Density must be positive and finite")
    if not np.isfinite(mass_scale) or not 0 < mass_scale <= 1:
        raise ValueError("Uniform mass scale must be in (0, 1]")
    if not np.isfinite(weld_grid_m) or not 0 <= weld_grid_m <= 1e-6:
        raise ValueError("Weld grid must be finite and at most one micrometer")
    original_count = len(t)
    maximum_shift = 0.0
    if weld_grid_m:
        snapped = np.round(t / weld_grid_m) * weld_grid_m
        maximum_shift = float(np.max(np.linalg.norm(snapped - t, axis=2)))
        t = snapped
        area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        t = t[area > 1e-18]
        if not len(t):
            raise ValueError("Welding collapsed the mesh")
    vertices, inverse = np.unique(t.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    sorted_edges = np.sort(edges, axis=1)
    _, edge_ids, counts = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(edge_ids, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(counts != 2) or np.any(orientation != 0):
        raise ValueError("STL is not a closed consistently wound two-manifold")
    origin = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    a, b, c = (t - origin).transpose(1, 0, 2)
    if np.any(np.linalg.norm(np.cross(b - a, c - a), axis=1) <= 1e-16):
        raise ValueError("Degenerate STL triangles")
    tetra = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    volume = tetra.sum()
    if not np.isfinite(volume) or volume <= 1e-15:
        raise ValueError("STL must have positive signed volume and outward winding")
    sums = a + b + c
    center = np.einsum("i,ij->j", tetra, sums) / (4 * volume)
    second = sum(np.einsum("i,ij,ik->jk", tetra, v, v) for v in (a, b, c, sums)) / 20
    covariance_integral = second - volume * np.outer(center, center)
    density = density_kg_m3 * mass_scale
    inertia = density * (np.trace(covariance_integral) * np.eye(3) - covariance_integral)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.min(eigenvalues) <= 0 or 2 * max(eigenvalues) > sum(eigenvalues) + 1e-15:
        raise ValueError("STL gives a nonphysical inertia tensor")
    return {
        "volume_m3": float(volume),
        "mass_kg": float(volume * density),
        "com_xyz_m": (center + origin).tolist(),
        "inertia_kg_m2": inertia.tolist(),
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "triangles": len(t),
        "weld_grid_m": weld_grid_m,
        "maximum_vertex_shift_m": maximum_shift,
        "removed_degenerate_triangles": original_count - len(t),
        "closed_consistently_wound": True,
    }


def combine_properties(parts):
    """Combine (mass, COM, inertia-at-COM) expressed in a common frame."""
    if not parts:
        raise ValueError("At least one rigid component is required")
    mass = sum(p[0] for p in parts)
    com = sum(m * np.asarray(c) for m, c, _ in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, tensor in parts:
        delta = np.asarray(c) - com
        inertia += np.asarray(tensor) + m * (delta @ delta * np.eye(3) - np.outer(delta, delta))
    return float(mass), com, inertia
