from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .recognition_profiles import RecognitionProfile
from .utils import ensure_dir, imwrite_unicode, timestamp


class ProfileSampleStore:
    """Store isolated labelled samples for one recognition profile."""

    def __init__(self, root: Path, profile: RecognitionProfile) -> None:
        self.root = root
        self.profile = profile
        self.manifest_path = profile.manifest_path(root)
        self.sample_dir = profile.sample_dir(root)

    def read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": 1, "profile": self.profile.key, "samples": []}
        with self.manifest_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"sample manifest must contain an object: {self.manifest_path}")
        manifest_profile = str(data.get("profile", self.profile.key))
        if manifest_profile != self.profile.key:
            raise ValueError(
                f"sample manifest profile is {manifest_profile!r}, expected {self.profile.key!r}: {self.manifest_path}"
            )
        samples = data.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"sample manifest 'samples' must be a list: {self.manifest_path}")
        data["profile"] = self.profile.key
        return data

    def counts(self) -> dict[str, int]:
        counts = {label: 0 for label in self.profile.labels}
        for item in self.read_manifest()["samples"]:
            if isinstance(item, dict) and item.get("map_id") in counts:
                counts[str(item["map_id"])] += 1
        return counts

    def save(self, label: str, image: np.ndarray, *, roi_version: str, note: str) -> dict[str, Any]:
        if label not in self.profile.labels:
            raise ValueError(f"label {label!r} does not belong to profile {self.profile.key!r}")
        ensure_dir(self.sample_dir)
        ensure_dir(self.manifest_path.parent)
        sample_id = f"{label}_{timestamp()}"
        image_path = self.sample_dir / f"{sample_id}.png"
        if not imwrite_unicode(image_path, image):
            raise RuntimeError(f"failed to save sample image: {image_path}")
        item = {
            "sample_id": sample_id,
            "map_id": label,
            "map_name": self.profile.label_name(label),
            "image": image_path.relative_to(self.root).as_posix(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
            "roi_version": roi_version,
            "profile": self.profile.key,
            "entry_floor": self.profile.entry_floor,
            "entry_door": self.profile.entry_door,
        }
        manifest = self.read_manifest()
        manifest["version"] = 1
        manifest["profile"] = self.profile.key
        manifest["labels"] = list(self.profile.labels)
        manifest["entry_floor"] = self.profile.entry_floor
        manifest["entry_door"] = self.profile.entry_door
        manifest["floors"] = list(self.profile.floors)
        manifest["samples"].append(item)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return item

