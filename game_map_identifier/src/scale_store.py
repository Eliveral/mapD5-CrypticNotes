from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SavedScale:
    kind: str
    map_id: str
    resolution: str
    scale: float
    detail: str = ""


def scale_profile_app_config(app_config: Mapping[str, Any], profile_key: str) -> dict[str, Any]:
    """Return a scale-storage view isolated for one recognition profile."""
    result = dict(app_config)
    if str(profile_key).strip().lower() != "nightmare":
        return result
    result.pop("semantic_cv_per_layout_scales", None)
    result.pop("semantic_cv_fixed_scale", None)
    result["semantic_cv_per_layout_scale_path"] = str(
        app_config.get(
            "nightmare_semantic_cv_per_layout_scale_path",
            "config/nightmare_per_layout_semantic_cv_scales.json",
        )
    )
    result["semantic_cv_confirmed_scale_path"] = str(
        app_config.get(
            "nightmare_semantic_cv_confirmed_scale_path",
            "config/nightmare_confirmed_semantic_cv_scales.json",
        )
    )
    result["semantic_cv_estimated_scale_ranges_path"] = str(
        app_config.get(
            "nightmare_semantic_cv_estimated_scale_ranges_path",
            "config/nightmare_estimated_semantic_cv_scale_ranges.json",
        )
    )
    result["semantic_cv_estimated_scale_ranges_required"] = bool(
        app_config.get("nightmare_semantic_cv_estimated_scale_ranges_required", True)
    )
    return result


def _config_path(root_dir: Path, app_config: Mapping[str, Any], key: str, default: str) -> Path:
    raw = str(app_config.get(key, default)).strip() or default
    path = Path(raw)
    return path if path.is_absolute() else Path(root_dir) / path


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return dict(default)
    return loaded if isinstance(loaded, dict) else dict(default)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _positive_scale(value: Any) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    return scale


def _reference_resolution(app_config: Mapping[str, Any]) -> str:
    value = app_config.get("alignment_reference_capture_size", (2560, 1440))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return f"{int(value[0])}x{int(value[1])}"
        except (TypeError, ValueError):
            pass
    return "2560x1440"


def _resolution_tuple(value: str) -> tuple[int, int]:
    text = str(value).strip().lower().replace(" ", "")
    parts = text.split("x")
    if len(parts) != 2:
        raise ValueError("ROI resolution must be written as widthxheight, for example 1106x999")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("ROI resolution must be positive")
    return width, height


def scale_client_resolution(app_config: Mapping[str, Any], record: SavedScale) -> str:
    """Return the game client resolution that owns this active scale record."""
    if record.kind == "baseline":
        return record.resolution
    if record.kind != "confirmed":
        return "unknown"
    try:
        roi_width, roi_height = _resolution_tuple(record.resolution)
        reference_roi = app_config.get("semantic_cv_reference_map_roi_size")
        if not isinstance(reference_roi, (list, tuple)) or len(reference_roi) < 2:
            reference_roi = app_config.get(str(app_config.get("alignment_map_roi_key", "alignment_map_roi")))
            reference_roi = reference_roi[2:4] if isinstance(reference_roi, (list, tuple)) and len(reference_roi) >= 4 else None
        capture_size = app_config.get("alignment_reference_capture_size", (2560, 1440))
        if not isinstance(reference_roi, (list, tuple)) or not isinstance(capture_size, (list, tuple)):
            return "unknown"
        reference_width, reference_height = float(reference_roi[0]), float(reference_roi[1])
        capture_width, capture_height = int(capture_size[0]), int(capture_size[1])
        if reference_width <= 0 or reference_height <= 0:
            return "unknown"
        # The ROI itself was rounded during capture scaling. Display an estimated,
        # even-pixel client size rather than claiming a false exact inverse.
        client_width = round((roi_width * capture_width / reference_width) / 2) * 2
        client_height = round((roi_height * capture_height / reference_height) / 2) * 2
        return f"{client_width}x{client_height}"
    except (TypeError, ValueError, IndexError):
        return "unknown"


def scale_resolution_display(app_config: Mapping[str, Any], record: SavedScale) -> str:
    """Show the source ROI and its estimated game client resolution."""
    if record.kind == "baseline":
        return f"client {record.resolution}"
    if record.kind == "confirmed":
        client_size = scale_client_resolution(app_config, record)
        suffix = f" / client about {client_size}" if client_size != "unknown" else ""
        return f"ROI {record.resolution}{suffix}"
    return record.resolution


def _map_sort_key(map_id: str) -> tuple[int, int, str]:
    normalized = str(map_id).strip().lower()
    match = re.fullmatch(r"(?:map|nightmare)_(\d+)_(b1|1st|2nd)", normalized)
    if match is None:
        return (10**9, 10**9, normalized)
    floor_order = {"b1": 0, "1st": 1, "2nd": 2}
    return (int(match.group(1)), floor_order[match.group(2)], normalized)


def _latest_active_records(
    app_config: Mapping[str, Any], records: list[SavedScale]
) -> list[SavedScale]:
    """Keep the newest record for each layout and effective client size."""
    active: dict[tuple[str, str], tuple[str, int, SavedScale]] = {}
    for index, record in enumerate(records):
        key = (record.map_id.strip().lower(), scale_client_resolution(app_config, record))
        # confirmed_at is ISO formatted, so lexical order is chronological for
        # the timestamps written by this application. File order breaks ties
        # and lets a later legacy record supersede an undated earlier one.
        recency = record.detail if record.kind == "confirmed" else ""
        candidate = (recency, index, record)
        current = active.get(key)
        if current is None or candidate[:2] > current[:2]:
            active[key] = candidate
    return [
        item[2]
        for item in sorted(
            active.values(),
            key=lambda item: (
                _map_sort_key(item[2].map_id),
                scale_client_resolution(app_config, item[2]),
                item[2].resolution,
            ),
        )
    ]


def saved_scale_records(root_dir: Path, app_config: Mapping[str, Any]) -> list[SavedScale]:
    """Read the baseline and resolution-specific scales used by current CV localization."""
    records: list[SavedScale] = []
    baseline_map_ids: set[str] = set()
    base_path = _config_path(
        root_dir, app_config, "semantic_cv_per_layout_scale_path", "config/per_layout_semantic_cv_scales.json"
    )
    base_data = _read_json(base_path, {"layouts": {}})
    layouts = base_data.get("layouts", {})
    if isinstance(layouts, dict):
        for map_id, item in layouts.items():
            value = item.get("scale") if isinstance(item, dict) else item
            try:
                scale = _positive_scale(value)
            except (TypeError, ValueError):
                continue
            detail = str(item.get("source", "")) if isinstance(item, dict) else ""
            baseline_map_ids.add(str(map_id).strip().lower())
            records.append(SavedScale("baseline", str(map_id), _reference_resolution(app_config), scale, detail))

    confirmed_path = _config_path(
        root_dir, app_config, "semantic_cv_confirmed_scale_path", "config/confirmed_semantic_cv_scales.json"
    )
    confirmed_data = _read_json(confirmed_path, {"layouts": {}})
    confirmed_layouts = confirmed_data.get("layouts", {})
    if isinstance(confirmed_layouts, dict):
        for map_id, by_resolution in confirmed_layouts.items():
            if not isinstance(by_resolution, dict):
                continue
            for resolution, item in by_resolution.items():
                value = item.get("scale") if isinstance(item, dict) else item
                try:
                    scale = _positive_scale(value)
                except (TypeError, ValueError):
                    continue
                detail = str(item.get("confirmed_at", "")) if isinstance(item, dict) else ""
                record = SavedScale("confirmed", str(map_id), str(resolution), scale, detail)
                # At the reference resolution, runtime localization reads the
                # baseline scale whenever one is available for this layout.
                # Keep the old reuse value on disk, but do not present it as
                # an active scale in the manager.
                if (
                    scale_client_resolution(app_config, record) == _reference_resolution(app_config)
                    and str(map_id).strip().lower() in baseline_map_ids
                ):
                    continue
                records.append(record)

    return _latest_active_records(app_config, records)


def update_saved_scale(root_dir: Path, app_config: Mapping[str, Any], record: SavedScale, scale: float) -> None:
    """Update one selected persistent scale without changing its identifier or metadata."""
    value = _positive_scale(scale)
    if record.kind == "baseline":
        path = _config_path(root_dir, app_config, "semantic_cv_per_layout_scale_path", "config/per_layout_semantic_cv_scales.json")
        data = _read_json(path, {"version": 1, "layouts": {}})
        layouts = data.setdefault("layouts", {})
        item = layouts.get(record.map_id)
        if isinstance(item, dict):
            item["scale"] = value
        else:
            layouts[record.map_id] = {"scale": value, "source": "manual_scale_manager"}
        _write_json(path, data)
        return
    if record.kind == "confirmed":
        path = _config_path(root_dir, app_config, "semantic_cv_confirmed_scale_path", "config/confirmed_semantic_cv_scales.json")
        data = _read_json(path, {"version": 1, "layouts": {}})
        layouts = data.setdefault("layouts", {})
        by_resolution = layouts.setdefault(record.map_id, {})
        item = by_resolution.get(record.resolution)
        if isinstance(item, dict):
            item["scale"] = value
        else:
            width, height = _resolution_tuple(record.resolution)
            by_resolution[record.resolution] = {
                "scale": value,
                "roi_size": [width, height],
                "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        _write_json(path, data)
        return
    if record.kind == "legacy":
        path = _config_path(root_dir, app_config, "alignment_scale_config_path", "config/screenshot_scale_calibration.json")
        data = _read_json(path, {"version": 1, "scales": {}})
        scales = data.setdefault("scales", {})
        item = scales.get(record.map_id)
        if isinstance(item, dict):
            item["scale"] = value
        else:
            scales[record.map_id] = {"scale": value}
        _write_json(path, data)
        return
    raise ValueError(f"unknown saved scale kind: {record.kind}")


def add_confirmed_scale(
    root_dir: Path, app_config: Mapping[str, Any], map_id: str, resolution: str, scale: float
) -> SavedScale:
    normalized_map_id = str(map_id).strip().lower()
    if not normalized_map_id:
        raise ValueError("layout id is required")
    width, height = _resolution_tuple(resolution)
    record = SavedScale("confirmed", normalized_map_id, f"{width}x{height}", _positive_scale(scale))
    update_saved_scale(root_dir, app_config, record, record.scale)
    return record


def delete_saved_scale(root_dir: Path, app_config: Mapping[str, Any], record: SavedScale) -> None:
    if record.kind == "baseline":
        path = _config_path(root_dir, app_config, "semantic_cv_per_layout_scale_path", "config/per_layout_semantic_cv_scales.json")
        data = _read_json(path, {"version": 1, "layouts": {}})
        data.setdefault("layouts", {}).pop(record.map_id, None)
    elif record.kind == "confirmed":
        path = _config_path(root_dir, app_config, "semantic_cv_confirmed_scale_path", "config/confirmed_semantic_cv_scales.json")
        data = _read_json(path, {"version": 1, "layouts": {}})
        layouts = data.setdefault("layouts", {})
        by_resolution = layouts.get(record.map_id)
        if isinstance(by_resolution, dict):
            by_resolution.pop(record.resolution, None)
            if not by_resolution:
                layouts.pop(record.map_id, None)
    elif record.kind == "legacy":
        path = _config_path(root_dir, app_config, "alignment_scale_config_path", "config/screenshot_scale_calibration.json")
        data = _read_json(path, {"version": 1, "scales": {}})
        data.setdefault("scales", {}).pop(record.map_id, None)
    else:
        raise ValueError(f"unknown saved scale kind: {record.kind}")
    _write_json(path, data)
