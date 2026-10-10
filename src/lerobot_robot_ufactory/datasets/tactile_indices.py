"""Action-row ranges address full tactile images and optional Mesh3DFlow."""

import json

import numpy as np


def tactile_range_key(camera_name):
    return f"observation.{camera_name}.tactile_range"


def tactile_range_fields(features):
    """Recognize old datasets without writing a second range for new recordings."""
    fields = {}
    cameras = set()
    for key in features:
        for suffix in (".tactile_range", ".mesh3dflow_range"):
            if key.startswith("observation.") and key.endswith(suffix):
                camera = key.removeprefix("observation.").removesuffix(suffix)
                if camera in cameras:
                    raise ValueError(f"Duplicate tactile range fields for {camera}")
                fields[key] = camera
                cameras.add(camera)
                break
    return fields


def apply_tactile_index_plan(features, plan):
    for name, item in plan.items():
        if not (item.get("tactile") or item.get("require_mesh")):
            continue
        if item.get("require_mesh"):
            features.pop(f"observation.{name}.mesh_motion_3d", None)
        features.pop(f"observation.{name}.mesh3dflow_range", None)
        features[tactile_range_key(name)] = {
            "dtype": "int64",
            "shape": (2,),
            "names": None,
        }
        for old in ("mesh_row_storage", "mesh_range_bounds", "mesh_index_source"):
            item.pop(old, None)
        item["tactile_row_storage"] = "shared_range_index_v1"
        item["tactile_range_key"] = tactile_range_key(name)
        item["tactile_range_bounds"] = "[start_index, end_index)"
        item["tactile_range_targets"] = [
            "video" if item.get("storage", "video") == "video" else "image"
        ]
        if item.get("require_mesh"):
            item["tactile_range_targets"].append("mesh3dflow")
        item["tactile_index_source"] = (
            "tactile_streams/<camera>/episode_<episode_index>/samples.parquet"
        )


def frame_tactile_ranges(features, synchronization):
    fields = tactile_range_fields(features)
    if not fields:
        return {}
    if synchronization is None or not synchronization.frames:
        raise RuntimeError("Tactile index rows require camera interval synchronization")
    intervals = json.loads(synchronization.frames[-1]["camera_intervals_json"])
    result = {}
    for key, name in fields.items():
        interval = intervals[name]
        start, end = interval["start_index"], interval["end_index"]
        if start < 0 or end < start:
            raise ValueError(f"Invalid tactile range for {name}: {start}, {end}")
        result[key] = np.array([start, end], dtype=np.int64)
    return result


# Preserve existing callers; all new rows use the shared tactile field name.
mesh_range_key = tactile_range_key
mesh_range_fields = tactile_range_fields
apply_mesh_index_plan = apply_tactile_index_plan
frame_mesh_ranges = frame_tactile_ranges
