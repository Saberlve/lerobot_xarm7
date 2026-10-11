"""Strict YAML configuration storage; no hardware is instantiated here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from uuid import uuid4

import yaml


class ConfigError(ValueError):
    pass


class Conflict(ConfigError):
    pass


class StrictLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def validate_text(text):
    if not isinstance(text, str) or len(text.encode()) > 1024 * 1024:
        raise ConfigError("Configuration must be UTF-8 text below 1 MiB")
    try:
        raw = yaml.load(text, Loader=StrictLoader)
        if not isinstance(raw, dict):
            raise ConfigError("Configuration must be a YAML mapping")
        import draccus
        from lerobot_robot_ufactory.scripts.uf_lerobot_record import (
            UFRecordConfig, register_third_party_plugins,
        )
        register_third_party_plugins()
        cfg = draccus.decode(UFRecordConfig, raw)
        if raw.get("teleop", {}).get("type") != "uf::gello_teleop":
            raise ConfigError("The web recorder currently supports GELLO configurations only")
        if cfg.robot.robot_dof != 7 or cfg.robot.control_space != "joint":
            raise ConfigError("Web GELLO recording requires an xArm7 in joint control space")
        if cfg.dataset.num_episodes <= 0 or cfg.dataset.fps <= 0 or cfg.dataset.episode_time_s <= 0:
            raise ConfigError("Episode count, FPS and duration must be positive")
        if cfg.policy is not None:
            raise ConfigError("Policy recording is not supported by this GELLO console")
        if cfg.dataset.push_to_hub:
            raise ConfigError("Disable push_to_hub for the web recorder")
        return raw, cfg
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(str(exc)) from exc


def revision(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dataset_status(project, raw):
    root = (Path(project) / raw["dataset"]["root"]).resolve()
    result = {"root": str(root), "exists": root.exists(), "episodes": 0,
              "resumable": False, "reason": None}
    if not root.exists():
        return result
    try:
        from lerobot_robot_ufactory.scripts.uf_lerobot_record import _missing_dataset_files
        from lerobot_robot_ufactory.datasets.raw_episodes import RawEpisodeStore
        info = json.loads((root / "meta/info.json").read_text())
        result["episodes"] = int(info["total_episodes"])
        missing = _missing_dataset_files(root)
        pending = [p for p in RawEpisodeStore.checkpoints(root)
                   if json.loads(p.read_text())["episode_index"] >= result["episodes"]]
        if missing:
            result["reason"] = "Incomplete dataset: " + ", ".join(missing)
        elif pending and raw.get("defer_processing", False):
            indices = [json.loads(p.read_text())["episode_index"] for p in pending]
            if indices != list(range(result["episodes"], max(indices) + 1)):
                result["reason"] = "Raw episode indexes are not contiguous"
            else:
                result["episodes"] = max(indices) + 1
                result["resumable"] = True
        elif pending:
            result["reason"] = "Raw episodes require --postprocess-only before resuming"
        else:
            result["resumable"] = True
    except Exception as exc:
        result["reason"] = str(exc)
    return result


def postprocess_status(project, raw):
    """Inspect committed manifests without reading images or opening devices."""
    from lerobot_robot_ufactory.datasets.raw_episodes import RawEpisodeStore

    root = (Path(project) / raw["dataset"]["root"]).resolve()
    result = {"root": str(root), "processed_episodes": 0, "pending_episodes": 0,
              "pending_frames": 0, "ready": False, "reason": None}
    if not root.exists():
        return result
    try:
        info_path = root / "meta/info.json"
        journal_path = root / ".raw_postprocessing_transaction.json"
        if journal_path.exists():
            # Report the pre-publication count; startup recovery restores it.
            journal = json.loads(journal_path.read_text())
            for item in journal["items"]:
                if item["destination"] == "meta":
                    backup = (root / item["backup"] / "info.json").resolve()
                    if not backup.is_relative_to(root):
                        raise ValueError("Invalid postprocessing recovery path")
                    if backup.is_file():
                        info_path = backup
        info = json.loads(info_path.read_text())
        processed = int(info["total_episodes"])
        result["processed_episodes"] = processed
        pending = [(path, json.loads(path.read_text())) for path in RawEpisodeStore.checkpoints(root)]
        pending = [(path, item) for path, item in pending if item["episode_index"] >= processed]
        result["pending_episodes"] = len(pending)
        result["pending_frames"] = sum(int(item["size"]) for _, item in pending)
        if not pending:
            return result
        indices = [item["episode_index"] for _, item in pending]
        if indices != list(range(processed, processed + len(pending))):
            raise ValueError("Raw episode indexes are not contiguous")
        if any(int(item["size"]) <= 0 for _, item in pending):
            raise ValueError("Raw episode has no frames")
        for path, item in pending:
            if item["version"] != 1 or item["fps"] != info["fps"]:
                raise ValueError(f"Incompatible raw episode: {path.parent.name}")
            if not (path.parent / "frames.parquet").is_file():
                raise ValueError(f"Missing raw frame data: {path.parent.name}")
        if info["fps"] != raw["dataset"]["fps"]:
            raise ValueError("Dataset FPS differs from the selected configuration")
        result["ready"] = True
    except Exception as exc:
        result["reason"] = str(exc)
    return result


def dataset_stamp(root):
    """Reject stale rebuild confirmations, including newly written episodes."""
    root = Path(root)
    if not root.exists():
        return None
    digest = hashlib.sha256(str(root.resolve()).encode())
    # Metadata and transaction manifests change on publication; avoid walking
    # hundreds of thousands of RGB images on each preflight request.
    paths = [root, *sorted(root.iterdir()), *sorted((root / "meta").rglob("*"))]
    paths += sorted((root / "raw_episodes").glob("episode_*/manifest.json"))
    for path in paths:
        stat = path.lstat()
        digest.update(f"{path.relative_to(root)}:{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_size}".encode())
        if path.is_file() and (path.parent == root / "meta" or path.name == "manifest.json"):
            digest.update(path.read_bytes())
    return digest.hexdigest()


class ConfigStore:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.root = (self.project / "config/gello").resolve()
        self.lock = threading.RLock()

    def path(self, name):
        if not isinstance(name, str) or not name or "\\" in name:
            raise ConfigError("Invalid configuration path")
        relative = Path(name)
        if relative.is_absolute() or any(p in ("..", ".") or p.startswith(".") for p in relative.parts):
            raise ConfigError("Configuration path must stay inside config/gello")
        target = self.root / relative
        if target.suffix.lower() not in (".yaml", ".yml"):
            raise ConfigError("Only .yaml and .yml files are supported")
        # Do not follow even an in-root symlink when editing configuration files.
        if any(p.is_symlink() for p in [target, *target.parents] if p != self.root.parent):
            raise ConfigError("Symlink configurations are not writable")
        if not target.resolve().is_relative_to(self.root):
            raise ConfigError("Configuration path escapes config/gello")
        return target

    def listing(self):
        items = []
        for path in sorted(self.root.rglob("*")):
            name = path.relative_to(self.root).as_posix()
            if path.is_file() and path.suffix.lower() in (".yaml", ".yml") and not any(
                part.startswith(".") for part in Path(name).parts
            ):
                try:
                    self.path(name)
                    text = path.read_text(encoding="utf-8")
                    raw = yaml.safe_load(text) or {}
                    items.append({"path": name, "revision": revision(text),
                                  "task": raw.get("dataset", {}).get("single_task", "")})
                except Exception as exc:
                    items.append({"path": name, "error": str(exc)})
        return items

    def read(self, name):
        with self.lock:
            text = self.path(name).read_text(encoding="utf-8")
            return {"path": name, "text": text, "revision": revision(text)}

    def save(self, name, text, expected):
        validate_text(text)
        with self.lock:
            path = self.path(name)
            actual = revision(path.read_text(encoding="utf-8")) if path.exists() else None
            if actual != expected:
                raise Conflict("Configuration changed externally; reload before saving")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                with temp.open("x", encoding="utf-8", newline="") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                if path.exists():
                    os.chmod(temp, path.stat().st_mode & 0o777)
                current = revision(path.read_text(encoding="utf-8")) if path.exists() else None
                if current != expected:
                    raise Conflict("Configuration changed while saving; reload")
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
            return self.read(name)

    def delete(self, name, expected):
        with self.lock:
            item = self.read(name)
            if item["revision"] != expected:
                raise Conflict("Configuration changed externally")
            key = uuid4().hex
            folder = self.root / ".web-trash" / key
            folder.mkdir(parents=True)
            (folder / "metadata.json").write_text(json.dumps({"path": name}), encoding="utf-8")
            self.path(name).replace(folder / "config.yaml")
            return {"trash_id": key, "path": name}

    def trash(self):
        return [{"trash_id": p.parent.name, **json.loads(p.read_text())}
                for p in sorted((self.root / ".web-trash").glob("*/metadata.json"))
                if (p.parent / "config.yaml").exists()]

    def restore(self, key):
        if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
            raise ConfigError("Invalid trash identifier")
        with self.lock:
            folder = self.root / ".web-trash" / key
            name = json.loads((folder / "metadata.json").read_text())["path"]
            path = self.path(name)
            if path.exists():
                raise Conflict("Restore destination already exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            (folder / "config.yaml").replace(path)
            return self.read(name)
