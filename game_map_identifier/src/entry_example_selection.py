from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class EntryExampleCandidate:
    label: str
    sample_id: str
    path: Path
    created_at: str = ""


class EntryExampleSelectionStore:
    """Loads labelled samples and publishes manually selected entry examples."""

    def __init__(
        self,
        root: Path,
        labels: Iterable[str],
        sample_manifest_path: Path,
        output_dir: Path,
        selection_lock_path: Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.labels = tuple(labels)
        self.sample_manifest_path = Path(sample_manifest_path)
        self.output_dir = Path(output_dir)
        self.selection_lock_path = Path(selection_lock_path) if selection_lock_path else None
        self.selection_manifest_path = self.output_dir / "selection_manifest.json"
        self.samples_by_label: dict[str, list[EntryExampleCandidate]] = {
            label: [] for label in self.labels
        }
        self.selected_filenames: dict[str, str] = {}
        self._known_paths: dict[str, set[Path]] = {label: set() for label in self.labels}
        self.reload()

    @property
    def is_locked(self) -> bool:
        return self.selection_lock_path is not None and self.selection_lock_path.is_file()

    def reload(self) -> None:
        self.samples_by_label = {label: [] for label in self.labels}
        self._known_paths = {label: set() for label in self.labels}
        if self.sample_manifest_path.exists():
            payload = self._read_json(self.sample_manifest_path)
            raw_samples = payload.get("samples", []) if isinstance(payload, dict) else []
            for item in raw_samples:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("map_id", ""))
                if label not in self.samples_by_label:
                    continue
                raw_path = Path(str(item.get("image", "")))
                path = raw_path if raw_path.is_absolute() else self.root / raw_path
                path = path.resolve()
                if not path.is_file():
                    continue
                candidate = EntryExampleCandidate(
                    label=label,
                    sample_id=str(item.get("sample_id", path.stem)),
                    path=path,
                    created_at=str(item.get("created_at", "")),
                )
                self.samples_by_label[label].append(candidate)
                self._known_paths[label].add(path)
        for candidates in self.samples_by_label.values():
            candidates.sort(key=lambda item: (item.created_at, item.sample_id))

        self.selected_filenames = {}
        if self.selection_manifest_path.exists():
            payload = self._read_json(self.selection_manifest_path)
            entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
            if isinstance(entries, dict):
                self.selected_filenames = {
                    str(label): str(filename)
                    for label, filename in entries.items()
                    if label in self.samples_by_label and filename
                }

    def selected_candidate(self, label: str) -> EntryExampleCandidate | None:
        filename = self.selected_filenames.get(label)
        if not filename:
            return None
        for candidate in self.samples_by_label.get(label, []):
            if candidate.path.name == filename:
                return candidate
        return None

    def save(self, selections: Mapping[str, EntryExampleCandidate | Path]) -> list[Path]:
        if self.is_locked:
            raise PermissionError(f"例图发布基线已锁定：{self.selection_lock_path}")
        normalized: dict[str, Path] = {}
        for label, value in selections.items():
            if label not in self.samples_by_label:
                raise ValueError(f"未知布局标签：{label}")
            source = (value.path if isinstance(value, EntryExampleCandidate) else Path(value)).resolve()
            if source not in self._known_paths[label]:
                raise ValueError(f"所选图片不属于 {label} 的样本：{source}")
            if not source.is_file():
                raise FileNotFoundError(source)
            normalized[label] = source
        if not normalized:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []
        try:
            for label, source in normalized.items():
                target = self.output_dir / f"{label}.png"
                temporary = self.output_dir / f".{label}.png.tmp"
                shutil.copyfile(source, temporary)
                staged.append((temporary, target))
            for temporary, target in staged:
                os.replace(temporary, target)
        finally:
            for temporary, _target in staged:
                temporary.unlink(missing_ok=True)

        payload: dict[str, object] = {}
        if self.selection_manifest_path.exists():
            existing = self._read_json(self.selection_manifest_path)
            if isinstance(existing, dict):
                payload.update(existing)
        entries = dict(self.selected_filenames)
        entries.update({label: source.name for label, source in normalized.items()})
        payload.update(
            {
                "profile": "nightmare",
                "selection_rule": "Manual selection for the second-floor side-door confirmation view.",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "entries": {label: entries[label] for label in self.labels if label in entries},
            }
        )
        temporary_manifest = self.output_dir / ".selection_manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, self.selection_manifest_path)
        self.selected_filenames = entries
        return [self.output_dir / f"{label}.png" for label in normalized]

    @staticmethod
    def _read_json(path: Path) -> object:
        with path.open("r", encoding="utf-8-sig") as stream:
            return json.load(stream)
